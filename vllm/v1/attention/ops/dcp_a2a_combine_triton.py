# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton implementation of the direct symmetric-memory DCP A2A LSE reduce.

``direct_dcp_a2a_lse_reduce_triton`` is an alternative to the CUDA op
``torch.ops._C.direct_dcp_a2a_lse_reduce``
(csrc/attention/dcp_utils/dcp_direct_a2a_lse_reduce.cu). It takes the same
arguments, runs the same checks and the same four steps on the same workspace
buffers with the same epoch/signal protocol, so the two implementations can
serve calls on one workspace interchangeably:

1. advance the epoch; its low bit selects one of two staging slots,
2. copy each destination rank's head slice and LSE into that rank's buffers
   (rows of empty local KV shards publish an LSE of -inf and no payload),
3. publish completion to every peer with a release store,
4. wait for every peer, form the LSE weights and combine the partials.

Step 4 handles several (token, head) rows per program and polls all peer
signals together. Its arithmetic is the CUDA kernel's, operation by operation
and in the same order, spelled with explicit PTX rounding instructions so the
compiler cannot contract, flush or approximate any of it. The combined output
is bit-identical to the CUDA op.
"""

import torch

from vllm.triton_utils import HAS_TRITON, tl, triton

# Spin bound of the wait loop, as in csrc/attention/dcp_utils/dcp_direct_common.cuh.
SPIN_LIMIT = 100_000_000
# CUDART_L2E_F rounded to float32 (exact in float32).
_LOG2E_F32 = tl.constexpr(1.4426950216293335) if HAS_TRITON else 1.4426950216293335

_PAYLOAD_DTYPES = (torch.float16, torch.bfloat16)
_LSE_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


# Memory ordering follows the CUDA op: a producer publishes its signal with
# st.release.sys after its payload stores, and a consumer reads the payload only
# after an ld.acquire.sys that observed the signal, followed by a CTA barrier
# (PTX ISA, memory consistency model).
@triton.jit
def _ld_acquire_sys_u32(address):
    return tl.inline_asm_elementwise(
        "ld.global.acquire.sys.u32 $0, [$1];",
        "=r,l",
        [address],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _st_release_sys_u32(address, value, enabled):
    return tl.inline_asm_elementwise(
        """{
        .reg .pred %store_enabled;
        setp.ne.u32 %store_enabled, $3, 0;
        @%store_enabled st.global.release.sys.u32 [$1], $2;
        mov.u32 $0, 0;
        }""",
        "=r,l,r,r",
        [address, value, enabled],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _trap():
    tl.inline_asm_elementwise(
        "trap;\nmov.u32 $0, 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
    )


# Float32 operations with an explicit rounding mode. With these, the combine
# reproduces the CUDA kernel's exp2f(), 1.0f / x and multiply-add bit for bit
# (tests/v1/attention/test_dcp_a2a_combine_triton.py).
@triton.jit
def _ex2_f32(x):
    return tl.inline_asm_elementwise(
        "ex2.approx.f32 $0, $1;", "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _rcp_rn_f32(x):
    return tl.inline_asm_elementwise(
        "rcp.rn.f32 $0, $1;", "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _mul_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _add_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _sub_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _max_f32(a, b):
    return tl.inline_asm_elementwise(
        "max.f32 $0, $1, $2;",
        "=f,f,f",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fma_rn_f32(a, b, c):
    return tl.inline_asm_elementwise(
        "fma.rn.f32 $0, $1, $2, $3;",
        "=f,f,f,f",
        [a, b, c],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _increment_epoch_kernel(epoch_ptr):
    tl.store(epoch_ptr, tl.load(epoch_ptr) + 1)


@triton.jit(
    do_not_specialize=[
        "rank",
        "num_tokens",
        "max_num_tokens",
        "num_seqs",
        "output_token_stride",
        "lse_token_stride",
    ]
)
def _dispatch_output_lse_kernel(
    partial_output,
    partial_lse,
    seq_lens,
    query_start_loc,
    peer_output_ptrs,
    peer_lse_ptrs,
    epoch_ptr,
    rank,
    num_tokens,
    max_num_tokens,
    num_seqs,
    output_token_stride,
    lse_token_stride,
    WORLD_SIZE: tl.constexpr,
    HEADS_PER_RANK: tl.constexpr,
    HEADS_PER_RANK_P2: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    HAS_SEQ_LENS: tl.constexpr,
    SEQ_BLOCK: tl.constexpr,
):
    # One program per (destination rank, token).
    item = tl.program_id(0)
    destination = item // num_tokens
    token = item - destination * num_tokens

    empty_kv = 0
    if HAS_SEQ_LENS:
        # Sequence of this token: the last query_start_loc[:num_seqs] entry that
        # is <= token (query_start_loc is non-decreasing).
        count = token * 0
        for start in range(0, num_seqs, SEQ_BLOCK):
            seq = start + tl.arange(0, SEQ_BLOCK)
            bound = tl.load(
                query_start_loc + seq, mask=seq < num_seqs, other=2147483647
            )
            count += tl.sum((bound <= token).to(tl.int32), axis=0)
        seq_idx = tl.maximum(count - 1, 0)
        empty_kv = (tl.load(seq_lens + seq_idx) == 0).to(tl.int32)

    epoch = tl.load(epoch_ptr).to(tl.uint32)
    parity = (epoch & 1).to(tl.int64)
    token = token.to(tl.int64)
    destination_row = (
        (parity * WORLD_SIZE + rank) * max_num_tokens + token
    ) * HEADS_PER_RANK
    source_head = destination * HEADS_PER_RANK

    if empty_kv == 0:
        peer_output = tl.load(peer_output_ptrs + destination).to(
            tl.pointer_type(partial_output.dtype.element_ty)
        )
        peer_output = tl.multiple_of(peer_output, 16)
        # The host checks output_token_stride % 8 == 0; spelling it out lets the
        # compiler use 16-byte accesses.
        source = (
            partial_output
            + token * ((output_token_stride // 8) * 8)
            + source_head * HEAD_DIM
        )
        destination_ptr = peer_output + destination_row * HEAD_DIM
        offsets = tl.arange(0, BLOCK)
        for block in tl.static_range(NUM_BLOCKS):
            values = tl.load(source + block * BLOCK + offsets)
            tl.store(destination_ptr + block * BLOCK + offsets, values)

    peer_lse = tl.load(peer_lse_ptrs + destination).to(tl.pointer_type(tl.float32))
    heads = tl.arange(0, HEADS_PER_RANK_P2)
    head_mask = heads < HEADS_PER_RANK
    lse = tl.load(
        partial_lse + token * lse_token_stride + source_head + heads,
        mask=head_mask,
        other=0.0,
    ).to(tl.float32)
    lse = tl.where(empty_kv != 0, float("-inf"), lse)
    tl.store(peer_lse + destination_row + heads, lse, mask=head_mask)


@triton.jit(do_not_specialize=["rank"])
def _signal_kernel(
    peer_signal_ptrs,
    epoch_ptr,
    rank,
    WORLD_SIZE: tl.constexpr,
    WORLD_SIZE_P2: tl.constexpr,
):
    epoch = tl.load(epoch_ptr).to(tl.uint32)
    parity = (epoch & 1).to(tl.int64)
    peer = tl.arange(0, WORLD_SIZE_P2)
    valid = peer < WORLD_SIZE
    signal_base = tl.load(peer_signal_ptrs + peer, mask=valid, other=0)
    signal_address = signal_base + (parity * WORLD_SIZE + rank) * 4
    _st_release_sys_u32(signal_address, epoch, valid.to(tl.uint32))


@triton.jit
def _source_lse(
    received_lse,
    parity,
    source,
    max_num_tokens,
    token,
    head,
    row_mask,
    WORLD_SIZE: tl.constexpr,
    HEADS_PER_RANK: tl.constexpr,
    LSE_BASE_E: tl.constexpr,
):
    row = ((parity * WORLD_SIZE + source) * max_num_tokens + token) * HEADS_PER_RANK
    value = tl.load(received_lse + row + head, mask=row_mask, other=0.0)
    value = tl.where((value != value) | (value == float("inf")), float("-inf"), value)
    if LSE_BASE_E:
        value = _mul_rn_f32(value, _LOG2E_F32)
    return value


@triton.jit(do_not_specialize=["num_rows", "max_num_tokens"])
def _wait_lse_combine_kernel(
    received_output,
    received_lse,
    received_signal,
    epoch_ptr,
    combined_output,
    num_rows,
    max_num_tokens,
    WORLD_SIZE: tl.constexpr,
    WORLD_SIZE_P2: tl.constexpr,
    HEADS_PER_RANK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_P2: tl.constexpr,
    ROWS: tl.constexpr,
    LSE_BASE_E: tl.constexpr,
    SPIN: tl.constexpr,
    ROUND_TO_PAYLOAD: tl.constexpr,
):
    epoch = tl.load(epoch_ptr).to(tl.uint32)
    parity = (epoch & 1).to(tl.int64)

    # Poll every source rank's signal for this epoch at once, then let the
    # barrier order all payload loads of the program after those acquires.
    source = tl.arange(0, WORLD_SIZE_P2)
    signal_address = (
        received_signal + parity * WORLD_SIZE + tl.minimum(source, WORLD_SIZE - 1)
    ).to(tl.int64)
    seen = _ld_acquire_sys_u32(signal_address)
    ready = tl.min((seen == epoch).to(tl.int32), axis=0)
    spins = ready * 0
    while (ready == 0) & (spins < SPIN):
        seen = _ld_acquire_sys_u32(signal_address)
        ready = tl.min((seen == epoch).to(tl.int32), axis=0)
        spins += 1
    if ready == 0:
        _trap()
    tl.debug_barrier()

    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)[:, None]
    row_mask = rows < num_rows
    rows = rows.to(tl.int64)
    token = rows // HEADS_PER_RANK
    head = rows - token * HEADS_PER_RANK
    dims = tl.arange(0, HEAD_DIM_P2)[None, :]
    dim_mask = dims < HEAD_DIM

    lse_max = tl.full([ROWS, 1], float("-inf"), tl.float32)
    for src in tl.static_range(WORLD_SIZE):
        value = _source_lse(
            received_lse,
            parity,
            src,
            max_num_tokens,
            token,
            head,
            row_mask,
            WORLD_SIZE,
            HEADS_PER_RANK,
            LSE_BASE_E,
        )
        lse_max = _max_f32(lse_max, value)
    lse_max = tl.where(lse_max == float("-inf"), 0.0, lse_max)

    lse_sum = tl.zeros([ROWS, 1], tl.float32)
    for src in tl.static_range(WORLD_SIZE):
        value = _source_lse(
            received_lse,
            parity,
            src,
            max_num_tokens,
            token,
            head,
            row_mask,
            WORLD_SIZE,
            HEADS_PER_RANK,
            LSE_BASE_E,
        )
        lse_sum = _add_rn_f32(lse_sum, _ex2_f32(_sub_rn_f32(value, lse_max)))
    inverse_sum = tl.where(lse_sum > 0.0, _rcp_rn_f32(lse_sum), 0.0)

    acc = tl.zeros([ROWS, HEAD_DIM_P2], tl.float32)
    for src in tl.static_range(WORLD_SIZE):
        value = _source_lse(
            received_lse,
            parity,
            src,
            max_num_tokens,
            token,
            head,
            row_mask,
            WORLD_SIZE,
            HEADS_PER_RANK,
            LSE_BASE_E,
        )
        weight = _mul_rn_f32(_ex2_f32(_sub_rn_f32(value, lse_max)), inverse_sum)
        # Empty shards leave stale payloads, so zero-weight sources are not read.
        use = row_mask & (weight != 0.0)
        row = (
            (parity * WORLD_SIZE + src) * max_num_tokens + token
        ) * HEADS_PER_RANK + head
        x = tl.load(
            received_output + row * HEAD_DIM + dims, mask=use & dim_mask, other=0.0
        ).to(tl.float32)
        acc = tl.where(use, _fma_rn_f32(x, weight, acc), acc)

    if ROUND_TO_PAYLOAD:
        # The output dtype differs from the exchanged one: round to the payload
        # dtype first, as the CUDA op's output would be, then convert.
        acc = acc.to(received_output.dtype.element_ty).to(tl.float32)
    tl.store(
        combined_output + rows * HEAD_DIM + dims,
        acc.to(combined_output.dtype.element_ty),
        mask=row_mask & dim_mask,
    )


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def combine_launch_config(head_dim: int) -> tuple[int, int]:
    """(rows per program, warps) of the combine kernel: a single warp covering
    512 payload elements per program. Tunable."""
    return max(1, 512 // triton.next_power_of_2(head_dim)), 1


def direct_dcp_a2a_lse_reduce_triton(
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    seq_lens: torch.Tensor | None,
    query_start_loc: torch.Tensor | None,
    peer_output_ptrs: torch.Tensor,
    peer_lse_ptrs: torch.Tensor,
    peer_signal_ptrs: torch.Tensor,
    received_output: torch.Tensor,
    received_lse: torch.Tensor,
    received_signal: torch.Tensor,
    epoch: torch.Tensor,
    combined_output: torch.Tensor,
    world_size: int,
    rank: int,
    max_num_tokens: int,
    is_lse_base_on_e: bool,
) -> None:
    """Same contract as ``torch.ops._C.direct_dcp_a2a_lse_reduce``.

    ``combined_output`` may also use the other 16-bit float dtype than the
    exchanged payload; the combined rows are then rounded to the payload dtype
    first, which equals the CUDA op followed by ``combined_output.copy_()``.
    """
    _check(
        partial_output.is_cuda and partial_lse.is_cuda,
        "partial output and LSE must be CUDA tensors",
    )
    _check(
        partial_output.dtype in _PAYLOAD_DTYPES,
        "direct DCP A2A only supports FP16 and BF16 output",
    )
    _check(partial_lse.dtype in _LSE_DTYPES, "partial LSE must be FP32, FP16, or BF16")
    _check(
        partial_output.dim() == 3 and partial_lse.dim() == 2,
        "expected output [T,H,D] and LSE [T,H]",
    )
    _check(world_size > 1, "world_size must be greater than 1")
    _check(0 <= rank < world_size, "invalid rank")
    num_tokens, total_heads, head_dim = partial_output.shape
    output_token_stride = partial_output.stride(0)
    lse_token_stride = partial_lse.stride(0)
    _check(
        partial_output.stride(2) == 1
        and partial_output.stride(1) == head_dim
        and output_token_stride >= total_heads * head_dim
        and output_token_stride % 8 == 0,
        "partial output must have packed heads and an aligned token stride",
    )
    _check(
        partial_lse.stride(1) == 1 and lse_token_stride >= total_heads,
        "partial LSE must have packed heads",
    )
    _check(
        0 < num_tokens <= max_num_tokens,
        "token count exceeds symmetric buffer capacity",
    )
    _check(
        total_heads % world_size == 0,
        "attention heads must divide evenly across DCP ranks",
    )
    _check(
        tuple(partial_lse.shape) == (num_tokens, total_heads),
        "LSE shape must match attention output",
    )
    _check(head_dim % 8 == 0, "head_dim must be divisible by 8 for 16-byte stores")
    heads_per_rank = total_heads // world_size
    _check(
        combined_output.dtype in _PAYLOAD_DTYPES
        and combined_output.is_contiguous()
        and combined_output.device == partial_output.device,
        "combined output must be a contiguous FP16/BF16 tensor on the input device",
    )
    _check(
        tuple(combined_output.shape) == (num_tokens, heads_per_rank, head_dim),
        "combined output has the wrong shape",
    )
    _check(
        received_output.dtype == partial_output.dtype
        and received_lse.dtype == torch.float32
        and received_signal.dtype == torch.int32,
        "invalid symmetric staging dtypes",
    )
    _check(
        peer_output_ptrs.dtype == torch.int64
        and peer_lse_ptrs.dtype == torch.int64
        and peer_signal_ptrs.dtype == torch.int64,
        "peer pointer tables must be int64",
    )
    # The kernels address the staging buffers with the workspace layout.
    _check(
        received_output.is_contiguous()
        and received_lse.is_contiguous()
        and received_signal.is_contiguous()
        and tuple(received_output.shape)
        == (2, world_size, max_num_tokens, heads_per_rank, head_dim)
        and tuple(received_lse.shape)
        == (2, world_size, max_num_tokens, heads_per_rank)
        and tuple(received_signal.shape) == (2, world_size)
        and epoch.dtype == torch.int64,
        "unexpected direct DCP A2A workspace layout",
    )
    _check(
        (seq_lens is None) == (query_start_loc is None),
        "seq_lens and query_start_loc must be provided together",
    )
    num_seqs = 0
    if seq_lens is not None:
        assert query_start_loc is not None
        _check(
            seq_lens.is_cuda
            and seq_lens.dim() == 1
            and seq_lens.stride(0) == 1
            and seq_lens.dtype == torch.int32,
            "seq_lens must be a contiguous 1-D int32 CUDA tensor",
        )
        _check(
            query_start_loc.is_cuda
            and query_start_loc.dim() == 1
            and query_start_loc.stride(0) == 1
            and query_start_loc.dtype == torch.int32,
            "query_start_loc must be a contiguous 1-D int32 CUDA tensor",
        )
        num_seqs = seq_lens.shape[0]
        _check(
            num_seqs > 0 and query_start_loc.shape[0] == num_seqs + 1,
            "query_start_loc must contain one boundary per sequence",
        )
    else:
        # Unused when HAS_SEQ_LENS is False.
        seq_lens = query_start_loc = epoch

    payload = heads_per_rank * head_dim
    block = min(4096, payload & -payload)
    rows, combine_warps = combine_launch_config(head_dim)
    num_rows = num_tokens * heads_per_rank

    _increment_epoch_kernel[(1,)](epoch, num_warps=1)
    _dispatch_output_lse_kernel[(world_size * num_tokens,)](
        partial_output,
        partial_lse,
        seq_lens,
        query_start_loc,
        peer_output_ptrs,
        peer_lse_ptrs,
        epoch,
        rank,
        num_tokens,
        max_num_tokens,
        num_seqs,
        output_token_stride,
        lse_token_stride,
        WORLD_SIZE=world_size,
        HEADS_PER_RANK=heads_per_rank,
        HEADS_PER_RANK_P2=triton.next_power_of_2(heads_per_rank),
        HEAD_DIM=head_dim,
        BLOCK=block,
        NUM_BLOCKS=payload // block,
        HAS_SEQ_LENS=num_seqs > 0,
        SEQ_BLOCK=128,
        num_warps=max(1, min(8, block // 256)),
    )
    _signal_kernel[(1,)](
        peer_signal_ptrs,
        epoch,
        rank,
        WORLD_SIZE=world_size,
        WORLD_SIZE_P2=triton.next_power_of_2(world_size),
        num_warps=1,
    )
    _wait_lse_combine_kernel[(triton.cdiv(num_rows, rows),)](
        received_output,
        received_lse,
        received_signal,
        epoch,
        combined_output,
        num_rows,
        max_num_tokens,
        WORLD_SIZE=world_size,
        WORLD_SIZE_P2=triton.next_power_of_2(world_size),
        HEADS_PER_RANK=heads_per_rank,
        HEAD_DIM=head_dim,
        HEAD_DIM_P2=triton.next_power_of_2(head_dim),
        ROWS=rows,
        LSE_BASE_E=bool(is_lse_base_on_e),
        SPIN=SPIN_LIMIT,
        ROUND_TO_PAYLOAD=combined_output.dtype != received_output.dtype,
        num_warps=combine_warps,
    )

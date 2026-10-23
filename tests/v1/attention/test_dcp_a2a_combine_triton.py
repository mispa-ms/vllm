# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Triton direct DCP A2A LSE reduce must be bit-identical to the CUDA op
``torch.ops._C.direct_dcp_a2a_lse_reduce`` (VLLM_USE_TRITON_DIRECT_DCP_A2A)."""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import vllm.v1.attention.ops.dcp as dcp
from vllm.platforms import current_platform

ENV = "VLLM_USE_TRITON_DIRECT_DCP_A2A"


def _has_cuda_op() -> bool:
    if not current_platform.is_cuda():
        return False
    try:
        import vllm._custom_ops  # noqa: F401
    except ImportError:
        return False
    return hasattr(torch.ops._C, "direct_dcp_a2a_lse_reduce")


requires_cuda_op = pytest.mark.skipif(
    not _has_cuda_op(), reason="needs CUDA and the direct DCP A2A CUDA op"
)


def _same_bits(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    return (
        actual.shape == expected.shape
        and actual.dtype == expected.dtype
        and torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    )


# --------------------------------------------------------------------------
# Selection (CPU)
# --------------------------------------------------------------------------


class _FakeGroup:
    def __init__(self, world_size: int, rank: int) -> None:
        self._world_size = world_size
        self._rank = rank

    def size(self) -> int:
        return self._world_size

    def rank(self) -> int:
        return self._rank


def _local_workspace(monkeypatch, world_size: int = 2) -> dcp.DirectDCPA2AWorkspace:
    def allocate(self, shape, dtype):
        return torch.zeros(shape, dtype=dtype), torch.zeros(
            (self.num_ubatches, self.world_size), dtype=torch.int64
        )

    monkeypatch.setattr(dcp.DirectCPWorkspace, "_allocate", allocate)
    return dcp.DirectDCPA2AWorkspace(
        _FakeGroup(world_size, 1),
        torch.device("cpu"),
        max_num_tokens=8,
        heads_per_rank=2,
        head_dim=16,
        dtype=torch.bfloat16,
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_workspace_selects_lse_reduce_implementation(monkeypatch, enabled):
    import vllm.v1.attention.ops.dcp_a2a_combine_triton as triton_impl

    calls: dict[str, tuple] = {}
    monkeypatch.setenv(ENV, "1" if enabled else "0")
    monkeypatch.setattr(
        triton_impl,
        "direct_dcp_a2a_lse_reduce_triton",
        lambda *args: calls.setdefault("triton", args),
    )
    monkeypatch.setattr(
        torch.ops._C,
        "direct_dcp_a2a_lse_reduce",
        lambda *args: calls.setdefault("cuda", args),
        raising=False,
    )
    monkeypatch.setattr(dcp, "dbo_current_ubatch_id", lambda: 0)
    workspace = _local_workspace(monkeypatch)

    partial_output = torch.zeros(3, 4, 16, dtype=torch.bfloat16)
    partial_lse = torch.zeros(3, 4)
    output = workspace.lse_reduce(partial_output, partial_lse, False)

    assert list(calls) == (["triton"] if enabled else ["cuda"])
    (args,) = calls.values()
    assert args[0] is partial_output and args[1] is partial_lse
    assert args[2] is None and args[3] is None
    assert args[11] is output
    assert output.shape == (3, 2, 16) and output.dtype == torch.bfloat16
    assert args[12:] == (2, 1, 8, False)


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def _make_rank_inputs(
    rank: int,
    world_size: int,
    num_tokens: int,
    heads_per_rank: int,
    head_dim: int,
    dtype: torch.dtype,
    lse_dtype: torch.dtype,
    seed: int,
    device: torch.device,
    with_seq_lens: bool,
    padding_rows: int = 3,
    finite_padding_lse: bool = False,
):
    """Partials of one rank with padded storage strides; with seq_lens, also
    empty local shards and CUDA-graph padding rows past query_start_loc[-1]
    whose partials (and optionally LSE) are NaN."""
    total_heads = heads_per_rank * world_size
    generator = torch.Generator(device=device).manual_seed(seed * 131 + rank)
    storage = torch.randn(
        num_tokens, total_heads + 8, head_dim, generator=generator, device=device
    ).to(dtype)
    partial_output = storage[:, :total_heads]
    lse_storage = torch.empty(num_tokens, total_heads + 4, device=device)
    lse_storage.normal_(generator=generator).mul_(4.0).add_(10.0)
    partial_lse = lse_storage.to(lse_dtype)[:, :total_heads]
    if not with_seq_lens:
        return partial_output, partial_lse, None, None

    num_real = num_tokens - padding_rows
    query_lens = [1, 3, 2] + [1] * (num_real - 6)
    num_seqs = len(query_lens)
    # Sequence 0 is empty everywhere, sequence 1 + r is empty on rank r only.
    seq_lens = torch.tensor(
        [0 if s == 0 or s == 1 + rank else 5 + s for s in range(num_seqs)],
        dtype=torch.int32,
        device=device,
    )
    lens = torch.tensor(query_lens, dtype=torch.int32, device=device)
    query_start_loc = torch.cat((lens.new_zeros(1), lens.cumsum(0).to(torch.int32)))
    empty_rows = torch.ones(num_tokens, dtype=torch.bool, device=device)
    empty_rows[:num_real] = torch.repeat_interleave(seq_lens == 0, lens)
    partial_output[empty_rows] = float("nan")
    partial_lse[empty_rows] = float("nan")
    if finite_padding_lse:
        partial_lse[num_real:] = 12.0
    return partial_output, partial_lse, seq_lens, query_start_loc


_BF16, _FP16, _FP32 = torch.bfloat16, torch.float16, torch.float32
_CASES = [
    # (payload dtype, head_dim, lse dtype, base e, output dtype)
    pytest.param(_BF16, 512, _FP32, False, _BF16, id="bf16-512"),
    pytest.param(_BF16, 128, _BF16, True, _BF16, id="bf16-128-base-e"),
    pytest.param(_FP16, 128, _FP32, False, _FP16, id="fp16-128"),
    pytest.param(_FP16, 128, _FP32, False, _BF16, id="fp16-128-to-bf16"),
]


# --------------------------------------------------------------------------
# Single GPU: emulated ranks on separate streams
# --------------------------------------------------------------------------


def _emulated_workspaces(world_size, max_num_tokens, heads_per_rank, head_dim, dtype):
    device = torch.device("cuda")
    ranks = [
        SimpleNamespace(
            received_output=torch.zeros(
                (2, world_size, max_num_tokens, heads_per_rank, head_dim),
                dtype=dtype,
                device=device,
            ),
            received_lse=torch.zeros(
                (2, world_size, max_num_tokens, heads_per_rank),
                dtype=torch.float32,
                device=device,
            ),
            received_signal=torch.zeros(
                (2, world_size), dtype=torch.int32, device=device
            ),
            epoch=torch.zeros(1, dtype=torch.int64, device=device),
        )
        for _ in range(world_size)
    ]

    def table(name):
        return torch.tensor(
            [getattr(r, name).data_ptr() for r in ranks],
            dtype=torch.int64,
            device=device,
        )

    tables = (
        table("received_output"),
        table("received_lse"),
        table("received_signal"),
    )
    return ranks, tables


def _run_emulated(op, inputs, ranks, tables, outputs, max_num_tokens, base_e):
    """Run one reduce on every emulated rank, each on its own stream, so that
    the waiting consumer of one rank overlaps the producers of the others."""
    world_size = len(ranks)
    streams = [torch.cuda.Stream() for _ in range(world_size)]
    torch.accelerator.synchronize()
    for rank in range(world_size):
        po, pl, sl, qsl = inputs[rank]
        ws = ranks[rank]
        with torch.cuda.stream(streams[rank]):
            op(
                po,
                pl,
                sl,
                qsl,
                *tables,
                ws.received_output,
                ws.received_lse,
                ws.received_signal,
                ws.epoch,
                outputs[rank],
                world_size,
                rank,
                max_num_tokens,
                base_e,
            )
    torch.accelerator.synchronize()


@requires_cuda_op
@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("with_seq_lens", [False, True])
@pytest.mark.parametrize(
    ("dtype", "head_dim", "lse_dtype", "base_e", "out_dtype"), _CASES
)
def test_emulated_ranks_match_cuda_op(
    world_size, with_seq_lens, dtype, head_dim, lse_dtype, base_e, out_dtype
):
    from vllm.v1.attention.ops.dcp_a2a_combine_triton import (
        direct_dcp_a2a_lse_reduce_triton,
    )

    heads_per_rank, max_num_tokens, num_tokens = 4, 16, 12
    device = torch.device("cuda")
    ranks, tables = _emulated_workspaces(
        world_size, max_num_tokens, heads_per_rank, head_dim, dtype
    )
    for iteration in range(3):  # both staging slots, then the first again
        inputs = [
            _make_rank_inputs(
                r,
                world_size,
                num_tokens,
                heads_per_rank,
                head_dim,
                dtype,
                lse_dtype,
                seed=iteration,
                device=device,
                with_seq_lens=with_seq_lens,
                finite_padding_lse=iteration == 1,
            )
            for r in range(world_size)
        ]
        shape = (num_tokens, heads_per_rank, head_dim)
        expected = [torch.empty(shape, dtype=dtype, device=device) for _ in ranks]
        actual = [torch.empty(shape, dtype=out_dtype, device=device) for _ in ranks]
        _run_emulated(
            torch.ops._C.direct_dcp_a2a_lse_reduce,
            inputs,
            ranks,
            tables,
            expected,
            max_num_tokens,
            base_e,
        )
        _run_emulated(
            direct_dcp_a2a_lse_reduce_triton,
            inputs,
            ranks,
            tables,
            actual,
            max_num_tokens,
            base_e,
        )
        for rank in range(world_size):
            assert _same_bits(actual[rank], expected[rank].to(out_dtype)), (
                f"rank {rank} iteration {iteration}"
            )
        if with_seq_lens:
            # Rows of sequence 0 are empty on every rank and combine to zero.
            assert torch.equal(
                actual[0][0].float(), torch.zeros_like(actual[0][0].float())
            )


@requires_cuda_op
def test_rejects_bad_inputs():
    from vllm.v1.attention.ops.dcp_a2a_combine_triton import (
        direct_dcp_a2a_lse_reduce_triton,
    )

    ranks, tables = _emulated_workspaces(2, 4, 2, 16, torch.bfloat16)
    ws = ranks[0]
    partial_output = torch.zeros(5, 4, 16, dtype=torch.bfloat16, device="cuda")
    partial_lse = torch.zeros(5, 4, device="cuda")
    out = torch.empty(5, 2, 16, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="exceeds symmetric buffer capacity"):
        direct_dcp_a2a_lse_reduce_triton(
            partial_output,
            partial_lse,
            None,
            None,
            *tables,
            ws.received_output,
            ws.received_lse,
            ws.received_signal,
            ws.epoch,
            out,
            2,
            0,
            4,
            False,
        )
    assert int(ws.epoch.item()) == 0


# --------------------------------------------------------------------------
# Multi GPU: real symmetric-memory workspace
# --------------------------------------------------------------------------


def _distributed_worker(
    local_rank: int, world_size: int, init_file: str, dtype_name: str
) -> None:
    os.environ[ENV] = "1"
    dtype = getattr(torch, dtype_name)
    device = torch.device("cuda", local_rank)
    torch.accelerator.set_device_index(local_rank)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{init_file}",
        rank=local_rank,
        world_size=world_size,
    )
    try:
        import vllm._custom_ops  # noqa: F401

        heads_per_rank, head_dim, max_num_tokens = 6, 512, 32
        if dtype == torch.float16:
            head_dim = 128
        active_ubatch = [0]
        dcp.dbo_current_ubatch_id = lambda: active_ubatch[0]
        workspace = dcp.DirectDCPA2AWorkspace(
            dist.group.WORLD,
            device,
            max_num_tokens,
            heads_per_rank,
            head_dim,
            dtype,
            num_ubatches=2,
        )
        assert workspace._triton_lse_reduce is not None

        def cuda_op(po, pl, sl, qsl, base_e):
            ub = active_ubatch[0]
            out = po.new_empty((po.shape[0], heads_per_rank, head_dim))
            torch.ops._C.direct_dcp_a2a_lse_reduce(
                po,
                pl,
                sl,
                qsl,
                workspace.peer_output_ptrs[ub],
                workspace.peer_lse_ptrs[ub],
                workspace.peer_signal_ptrs[ub],
                workspace.received_output[ub],
                workspace.received_lse[ub],
                workspace.received_signal[ub],
                workspace.epoch[ub : ub + 1],
                out,
                world_size,
                local_rank,
                max_num_tokens,
                base_e,
            )
            return out

        iteration = 0
        for num_tokens in (1, 17, max_num_tokens):
            for with_seq_lens in (False, True):
                if with_seq_lens and num_tokens < 3 + 6 + world_size:
                    continue
                for base_e in (False, True):
                    iteration += 1
                    active_ubatch[0] = iteration % 2
                    po, pl, sl, qsl = _make_rank_inputs(
                        local_rank,
                        world_size,
                        num_tokens,
                        heads_per_rank,
                        head_dim,
                        dtype,
                        torch.bfloat16,
                        seed=iteration,
                        device=device,
                        with_seq_lens=with_seq_lens,
                    )
                    kwargs = dict(seq_lens=sl, query_start_loc=qsl)
                    actual = workspace.lse_reduce(po, pl, base_e, **kwargs)
                    expected = cuda_op(po, pl, sl, qsl, base_e)
                    torch.accelerator.synchronize()
                    assert _same_bits(actual, expected), (num_tokens, base_e)

        # CUDA graph capture and replay of the Triton path.
        inputs = _make_rank_inputs(
            local_rank,
            world_size,
            max_num_tokens,
            heads_per_rank,
            head_dim,
            dtype,
            torch.float32,
            seed=100,
            device=device,
            with_seq_lens=True,
        )
        po, pl, sl, qsl = inputs
        active_ubatch[0] = 1
        workspace.lse_reduce(po, pl, False, sl, qsl)
        torch.accelerator.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_out = workspace.lse_reduce(po, pl, False, sl, qsl)
        for replay in range(3):
            fresh = _make_rank_inputs(
                local_rank,
                world_size,
                max_num_tokens,
                heads_per_rank,
                head_dim,
                dtype,
                torch.float32,
                seed=200 + replay,
                device=device,
                with_seq_lens=True,
                finite_padding_lse=replay == 1,
            )
            for dst, src in zip(inputs, fresh):
                dst.copy_(src)
            graph.replay()
            expected = cuda_op(po, pl, sl, qsl, False)
            torch.accelerator.synchronize()
            assert _same_bits(graph_out, expected), replay
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("dtype_name", ["bfloat16", "float16"])
@pytest.mark.parametrize(
    "world_size",
    [
        pytest.param(
            2,
            marks=pytest.mark.skipif(
                torch.accelerator.device_count() < 2, reason="Need at least 2 GPUs."
            ),
        ),
        pytest.param(
            4,
            marks=pytest.mark.skipif(
                torch.accelerator.device_count() < 4, reason="Need at least 4 GPUs."
            ),
        ),
    ],
)
@requires_cuda_op
def test_distributed_workspace_matches_cuda_op(tmp_path, world_size, dtype_name):
    mp.spawn(
        _distributed_worker,
        args=(world_size, str(tmp_path / "dist_init"), dtype_name),
        nprocs=world_size,
    )

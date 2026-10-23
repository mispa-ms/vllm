# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""fp8 DCP chunked-context path for Kimi-K3 MLA prefill
(``VLLM_KIMI_K3_MLA_FP8_DCP_CONTEXT``).

The path keeps the context workspace in fp8 and replaces ``reorg_kvcache``
with a row map plus one Triton kernel. It must hand the prefill attention
exactly the K/V bytes of the default path whenever ``k_scale`` is 1.0.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.attention import mla_context_fp8
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonImpl,
    align_mla_chunked_context_workspace_size,
    build_mla_chunked_context_metadata,
    reorg_kvcache,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens

FP8 = torch.float8_e4m3fn
KV_LORA_RANK = 512
ROPE_DIM = 64
NOPE_DIM = 128
V_DIM = 128
ROW = KV_LORA_RANK + ROPE_DIM

requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="fp8 DCP context path requires CUDA"
)


def _aligned_workspace_size(size, block_size, dcp, interleave):
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp,
            cp_kv_cache_interleave_size=interleave,
        ),
    )
    return align_mla_chunked_context_workspace_size(config, size)


def _build_metadata(
    context_lens,
    query_lens,
    workspace_size,
    *,
    dcp,
    interleave,
    block_size,
    workspace_dtype,
    device,
):
    query_start_loc = torch.zeros(len(query_lens) + 1, dtype=torch.int32)
    query_start_loc[1:] = torch.tensor(query_lens, dtype=torch.int32).cumsum(0)
    workspace = torch.empty(
        (workspace_size + workspace_size // dcp, ROW),
        dtype=workspace_dtype,
        device=device,
    )
    metadata = build_mla_chunked_context_metadata(
        context_lens_cpu=torch.tensor(context_lens, dtype=torch.int32),
        prefill_query_start_loc_cpu=query_start_loc,
        chunked_prefill_workspace=workspace,
        chunked_prefill_workspace_size=workspace_size,
        block_size=block_size,
        align_chunk_to_block=True,
        device=device,
        dcp_world_size=dcp,
        dcp_local_block_size=interleave,
        dcp_virtual_block_size=interleave * dcp,
    )
    assert metadata is not None
    return metadata


_rng = np.random.default_rng(0)
ROW_MAP_LAYOUTS = [
    # dcp, interleave, block_size, workspace rows, context lengths
    (2, 1, 16, 256, [200, 48, 32, 0]),
    (4, 16, 16, 256, [1000, 0, 77, 5, 64, 300]),
    (8, 8, 32, 512, [1, 7, 8, 9, 63, 64, 65, 4000]),
    (4, 64, 64, 512, [3000, 2500, 1]),
    (8, 16, 16, 1024, [int(x) for x in _rng.integers(0, 3000, 16)]),
]


@pytest.mark.parametrize("dcp,interleave,block_size,size,context_lens", ROW_MAP_LAYOUTS)
def test_context_row_maps_match_reorg_kvcache(
    dcp, interleave, block_size, size, context_lens
):
    """Every chunk's row map is the index sequence reorg_kvcache concatenates."""
    workspace_size = _aligned_workspace_size(size, block_size, dcp, interleave)
    query_lens = [3] * len(context_lens)
    common = dict(
        dcp=dcp,
        interleave=interleave,
        block_size=block_size,
        device=torch.device("cpu"),
    )
    metadata = _build_metadata(
        context_lens, query_lens, workspace_size, workspace_dtype=FP8, **common
    )
    assert len(metadata.chunks) > 1, "the layout must span several chunks"
    for chunk in metadata.chunks:
        toks = chunk.num_local_context_tokens
        # Gathered row i holds the value i, so reorg_kvcache returns indices.
        gathered = torch.arange(dcp * toks, dtype=torch.float64).view(-1, 1, 1)
        expected, _ = reorg_kvcache(
            gathered,
            gathered,
            padded_local_chunk_seq_lens_lst=chunk.padded_local_seq_lens,
            local_context_lens_allranks=chunk.local_context_lens_allranks,
            local_starts=chunk.local_starts,
            sum_seq_len=chunk.num_context_tokens,
            max_seq_len=chunk.max_seq_len,
            toks=toks,
        )
        assert chunk.context_row_map is not None
        assert chunk.context_row_map.dtype == torch.int32
        assert chunk.context_row_map.tolist() == expected.view(-1).long().tolist()

    # The default workspace dtype keeps the default metadata.
    default = _build_metadata(
        context_lens,
        query_lens,
        workspace_size,
        workspace_dtype=torch.bfloat16,
        **common,
    )
    assert all(chunk.context_row_map is None for chunk in default.chunks)


# --------------------------------------------------------------------------
# KV-cache group selection
# --------------------------------------------------------------------------
def _k3_layer(name, *, draft=False, k_scale=1.0, kv_b_dtype=torch.bfloat16):
    from vllm.models.kimi_k3.nvidia.mla import MultiHeadLatentAttention

    layer = MultiHeadLatentAttention.__new__(MultiHeadLatentAttention)
    torch.nn.Module.__init__(layer)
    kv_b_proj = SimpleNamespace(
        weight=torch.zeros((8, KV_LORA_RANK), dtype=kv_b_dtype),
        params_dtype=torch.bfloat16,
        quant_method=None,
    )
    layer.layer_name = name
    layer.kv_cache_dtype = "fp8"
    layer.dcp_world_size = 4
    layer.dcp_manager = object()
    layer.non_causal_multi_token_decode = draft
    layer.is_draft_layer = draft
    layer.kv_b_proj = kv_b_proj
    layer.register_buffer("_k_scale", torch.tensor([k_scale]), persistent=False)
    layer.impl = SimpleNamespace(
        kv_cache_dtype="fp8",
        kv_b_proj=kv_b_proj,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=ROPE_DIM,
        qk_nope_head_dim=NOPE_DIM,
        v_head_dim=V_DIM,
    )
    return layer


def _select(layers, *, builder=None, dcp=4, cache_dtype="fp8", q_dtype=FP8):
    from vllm.v1.attention.backends.mla.tokenspeed_mla import (
        TokenspeedMLAMetadataBuilder,
    )

    if builder is None:
        builder = TokenspeedMLAMetadataBuilder.__new__(TokenspeedMLAMetadataBuilder)
    builder.q_data_type = q_dtype
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            static_forward_context={layer.layer_name: layer for layer in layers}
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=dcp),
        cache_config=SimpleNamespace(cache_dtype=cache_dtype),
    )
    return mla_context_fp8.use_fp8_dcp_context(
        builder, [layer.layer_name for layer in layers], config
    )


def test_group_selection():
    targets = [_k3_layer(f"model.layers.{i}.self_attn") for i in range(3)]
    assert _select(targets)
    # Draft layers never run chunked-context prefill.
    assert not _select([_k3_layer(f"draft.{i}", draft=True) for i in range(2)])
    # Groups that would not be bitwise identical keep the default path.
    assert not _select(targets + [_k3_layer("scaled", k_scale=2.0)])
    assert not _select([_k3_layer("fp8_kv_b", kv_b_dtype=FP8)])
    # Other models are left alone.
    other = SimpleNamespace(layer_name="other")
    assert not _select([other])


@pytest.mark.parametrize(
    "case,match",
    [
        ("mixed_draft", "target and draft"),
        ("mixed_model", "mixes Kimi-K3 MLA layers"),
        ("no_dcp", "decode context parallelism"),
        ("other_builder", "TOKENSPEED_MLA"),
        ("ds_mla_cache", "plain fp8 KV cache"),
        ("bf16_query", "use_prefill_query_quantization"),
    ],
)
def test_group_selection_rejects_unsupported_deployments(case, match):
    layers = [_k3_layer("model.layers.0.self_attn")]
    kwargs = {}
    if case == "mixed_draft":
        layers.append(_k3_layer("draft.0", draft=True))
    elif case == "mixed_model":
        layers.append(SimpleNamespace(layer_name="other"))
    elif case == "no_dcp":
        kwargs["dcp"] = 1
    elif case == "other_builder":
        kwargs["builder"] = SimpleNamespace()
    elif case == "ds_mla_cache":
        kwargs["cache_dtype"] = "fp8_ds_mla"
    elif case == "bf16_query":
        kwargs["q_dtype"] = torch.bfloat16
    with pytest.raises(ValueError, match=match):
        _select(layers, **kwargs)


# --------------------------------------------------------------------------
# Row-gather kernel
# --------------------------------------------------------------------------
def _random_fp8(shape, generator, device):
    raw = torch.randint(
        0, 256, shape, dtype=torch.uint8, device=device, generator=generator
    )
    # No NaN encodings: a KV cache never stores them.
    raw[(raw & 0x7F) == 0x7F] = 0x3C
    return raw.view(FP8)


@requires_cuda
@pytest.mark.parametrize("num_rows", [0, 1, 31, 32, 33, 1000])
def test_gather_fp8_context_rows(num_rows):
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(num_rows)
    gathered = _random_fp8((1200, ROW), generator, device)
    # Extreme values: zeros of both signs, the largest finite and the smallest
    # subnormal magnitudes.
    gathered[:4, :4] = torch.tensor(
        [0x00, 0x80, 0x7E, 0x01], dtype=torch.uint8, device=device
    ).view(FP8)
    row_map = torch.randperm(1200, device=device, generator=generator)[:num_rows]
    row_map = row_map.to(torch.int32)

    latent, k_pe = mla_context_fp8.gather_fp8_context_rows(
        gathered, row_map, KV_LORA_RANK
    )

    rows = gathered[row_map.long()]
    assert latent.shape == (num_rows, 1, KV_LORA_RANK)
    assert latent.dtype == torch.bfloat16 and latent.is_contiguous()
    assert k_pe.shape == (num_rows, 1, ROPE_DIM)
    assert k_pe.dtype == FP8 and k_pe.is_contiguous()
    expected_latent = rows[:, :KV_LORA_RANK].to(torch.bfloat16).unsqueeze(1)
    assert torch.equal(latent.view(torch.int16), expected_latent.view(torch.int16))
    expected_k_pe = rows[:, KV_LORA_RANK:].unsqueeze(1).contiguous()
    assert torch.equal(k_pe.view(torch.uint8), expected_k_pe.view(torch.uint8))


# --------------------------------------------------------------------------
# Context loop: fp8 path vs default path, with every DCP rank emulated in one
# process (distinct cache contents per rank)
# --------------------------------------------------------------------------
_DCP = 4
_RANK = 1
_INTERLEAVE = 16
_BLOCK_SIZE = 16
_NUM_HEADS = 2
_CONTEXT_LENS = [700, 0, 77, 1, 300, 129]
_QUERY_LENS = [5, 4, 3, 7, 2, 6]


class _KVBProj(torch.nn.Module):
    """bf16 ``kv_b_proj`` stand-in returning an ``(out, bias)`` tuple."""

    def __init__(self, device):
        super().__init__()
        generator = torch.Generator(device=device).manual_seed(7)
        weight = torch.randn(
            (_NUM_HEADS * (NOPE_DIM + V_DIM), KV_LORA_RANK),
            device=device,
            generator=generator,
        )
        self.register_buffer("weight", (weight * 0.05).to(torch.bfloat16))

    def forward(self, x):
        assert x.dtype == torch.bfloat16
        return torch.nn.functional.linear(x, self.weight), None


class _RecordingBackend:
    def __init__(self):
        self.kv: list[tuple[torch.Tensor, torch.Tensor]] = []

    def run_prefill_context_chunk(self, *, chunk, q, k, v, out=None):
        self.kv.append((k.view(torch.uint8).clone(), v.view(torch.uint8).clone()))
        digest = k.float().sum() + 2 * v.float().sum()
        out = torch.empty(
            (q.shape[0], _NUM_HEADS, V_DIM), dtype=torch.bfloat16, device=q.device
        )
        out.copy_(q.float()[..., :V_DIM] + digest)
        lse = torch.full(
            (_NUM_HEADS, q.shape[0]),
            float(chunk.index) + 0.5,
            dtype=torch.float32,
            device=q.device,
        )
        return out, lse


def _gather_local(cache, workspace, chunk, block_table, k_scale):
    """The local context gather of the context loop, for one rank's cache."""
    toks = chunk.num_local_context_tokens
    if workspace.dtype == FP8:
        ops.cp_gather_cache(
            src_cache=cache,
            dst=workspace[:toks],
            block_table=block_table[chunk.request_slice],
            cu_seq_lens=chunk.padded_local_cu_seq_lens,
            batch_size=chunk.num_requests,
            seq_starts=chunk.starts,
        )
    else:
        ops.gather_and_maybe_dequant_cache(
            src_cache=cache,
            dst=workspace,
            block_table=block_table[chunk.request_slice],
            cu_seq_lens=chunk.padded_local_cu_seq_lens,
            token_to_seq=chunk.padded_local_token_to_seq,
            num_tokens=toks,
            kv_cache_dtype="fp8",
            scale=k_scale,
            seq_starts=chunk.starts,
        )


class _EmulatedKVGather:
    """``kv_gather`` of one DCP rank: every peer's slot is filled by running
    the same local gather on that peer's cache."""

    def __init__(self, caches, chunks, block_table, k_scale, workspace):
        self.caches = caches
        self.chunks = chunks
        self.block_table = block_table
        self.k_scale = k_scale
        self.scratch = torch.empty_like(workspace)
        self.calls = 0

    def kv_gather(self, gathered, local):
        chunk = self.chunks[self.calls]
        self.calls += 1
        toks = local.shape[0]
        for rank, cache in enumerate(self.caches):
            if rank == _RANK:
                gathered[rank * toks : (rank + 1) * toks].copy_(local)
                continue
            _gather_local(cache, self.scratch, chunk, self.block_table, self.k_scale)
            gathered[rank * toks : (rank + 1) * toks].copy_(self.scratch[:toks])


def _run_context_loop(workspace_dtype, fused_pack, k_scale, device):
    workspace_size = _aligned_workspace_size(256, _BLOCK_SIZE, _DCP, _INTERLEAVE)
    metadata = _build_metadata(
        _CONTEXT_LENS,
        _QUERY_LENS,
        workspace_size,
        dcp=_DCP,
        interleave=_INTERLEAVE,
        block_size=_BLOCK_SIZE,
        workspace_dtype=workspace_dtype,
        device=device,
    )
    assert len(metadata.chunks) > 2
    assert any(chunk.is_continuation for chunk in metadata.chunks)

    # Same block table on every rank; each rank's cache holds its own rows.
    context_lens = torch.tensor(_CONTEXT_LENS, dtype=torch.int32)
    padded_local = [
        math.ceil(c / (_DCP * _INTERLEAVE)) * _INTERLEAVE for c in _CONTEXT_LENS
    ]
    pages = [max(1, math.ceil(n / _BLOCK_SIZE)) for n in padded_local]
    num_blocks = sum(pages) + 3
    order = torch.randperm(num_blocks, generator=torch.Generator().manual_seed(3))
    block_table = torch.zeros((len(pages), max(pages)), dtype=torch.int32)
    start = 0
    for i, n in enumerate(pages):
        block_table[i, :n] = order[start : start + n]
        start += n
    block_table = block_table.to(device)
    caches = []
    for rank in range(_DCP):
        local = get_dcp_local_seq_lens(context_lens, _DCP, rank, _INTERLEAVE)
        assert all(int(n) <= p for n, p in zip(local, padded_local))
        generator = torch.Generator(device=device).manual_seed(100 + rank)
        caches.append(_random_fp8((num_blocks, _BLOCK_SIZE, ROW), generator, device))

    kv_b_proj = _KVBProj(device)
    impl = SimpleNamespace(
        kv_cache_dtype="fp8",
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=ROPE_DIM,
        qk_nope_head_dim=NOPE_DIM,
        v_head_dim=V_DIM,
        num_heads=_NUM_HEADS,
        kv_b_proj=kv_b_proj,
        _use_flashinfer_concat_mla_k=False,
    )
    impl._concat_k_nope_k_pe = MLACommonImpl._concat_k_nope_k_pe.__get__(impl)
    if workspace_dtype == FP8:
        mla_context_fp8.set_fp8_dcp_context_state(impl, k_scale)

    fused_fn = None
    if fused_pack:
        from vllm.models.kimi_k3.nvidia.mla import MultiHeadLatentAttention

        layer = SimpleNamespace(qk_nope_head_dim=NOPE_DIM, v_head_dim=V_DIM)
        fused_fn = MultiHeadLatentAttention._fused_mla_kv_concat.__get__(layer)

    gather = _EmulatedKVGather(
        caches, metadata.chunks, block_table, k_scale, metadata.workspace
    )
    metadata.dcp_manager = gather
    backend = _RecordingBackend()
    prefill = SimpleNamespace(
        prefill_backend=backend,
        chunked_context=metadata,
        q_data_type=FP8,
        block_table=block_table,
    )
    generator = torch.Generator(device=device).manual_seed(11)
    q = torch.randn(
        (sum(_QUERY_LENS), _NUM_HEADS, NOPE_DIM + ROPE_DIM),
        device=device,
        generator=generator,
    ).to(FP8)
    out, lse = MLACommonImpl._context_parallel_compute_prefill_context(
        impl,
        q,
        caches[_RANK],
        SimpleNamespace(prefill=prefill),
        k_scale=k_scale,
        dcp_world_size=_DCP,
        fused_mla_kv_concat_fn=fused_fn,
    )
    assert gather.calls == len(metadata.chunks)
    return backend.kv, out, lse


@requires_cuda
@pytest.mark.parametrize("fused_pack", [False, True], ids=["generic", "k3_pack"])
@torch.inference_mode()
def test_fp8_context_matches_default_path(fused_pack):
    device = torch.device("cuda")
    k_scale = torch.ones(1, dtype=torch.float32, device=device)
    ref_kv, ref_out, ref_lse = _run_context_loop(
        torch.bfloat16, fused_pack, k_scale, device
    )
    kv, out, lse = _run_context_loop(FP8, fused_pack, k_scale, device)
    assert len(kv) == len(ref_kv)
    for index, ((k, v), (ref_k, ref_v)) in enumerate(zip(kv, ref_kv)):
        assert torch.equal(k, ref_k), f"chunk {index}: K bytes differ"
        assert torch.equal(v, ref_v), f"chunk {index}: V bytes differ"
    assert torch.equal(out.view(torch.int16), ref_out.view(torch.int16))
    assert torch.equal(lse, ref_lse)


@requires_cuda
def test_fp8_context_checks_k_scale():
    device = torch.device("cuda")
    impl = SimpleNamespace()
    k_scale = torch.ones(1, dtype=torch.float32, device=device)
    with pytest.raises(RuntimeError, match="not set up"):
        mla_context_fp8.get_fp8_dcp_context_state(impl, k_scale)
    mla_context_fp8.set_fp8_dcp_context_state(impl, k_scale)
    mla_context_fp8.get_fp8_dcp_context_state(impl, k_scale)
    # An in-place write that keeps 1.0 is accepted.
    k_scale.fill_(1.0)
    mla_context_fp8.get_fp8_dcp_context_state(impl, k_scale)
    k_scale.fill_(2.0)
    with pytest.raises(RuntimeError, match="k_scale == 1.0"):
        mla_context_fp8.get_fp8_dcp_context_state(impl, k_scale)
    with pytest.raises(RuntimeError, match="k_scale == 1.0"):
        mla_context_fp8.get_fp8_dcp_context_state(
            impl, torch.full((1,), 0.5, device=device)
        )

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused kv_b_proj GEMM + fp8 K/V pack of the Kimi-K3 fp8 DCP context path
(``VLLM_KIMI_K3_MLA_FUSED_KV_B_PACK``).

The fused kernel must write the same K/V bytes as ``kv_b_proj`` (``F.linear``
with a bf16 weight) followed by ``fused_mla_kv_concat_quant_fp8``, saturated
values included.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.attention import mla_context_fp8
from vllm.platforms import current_platform

FP8 = torch.float8_e4m3fn
KV_LORA_RANK = 512
ROPE_DIM = 64
HEAD_DIM = 128

requires_sm100 = pytest.mark.skipif(
    not (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(100)
    ),
    reason="the fused kv_b_proj GEMM targets SM100-family GPUs",
)


def _inputs(num_rows, num_heads, weight_scale=1.0, seed=0):
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    # Latents come from the fp8 cache, so every value is an e4m3 value.
    latent = torch.randn(
        (num_rows, 1, KV_LORA_RANK), device=device, generator=generator
    )
    latent = (latent * 4.0).to(FP8).to(torch.bfloat16)
    weight = torch.randn(
        (num_heads * 2 * HEAD_DIM, KV_LORA_RANK), device=device, generator=generator
    )
    weight = (weight * weight_scale / KV_LORA_RANK**0.5).to(torch.bfloat16)
    raw = torch.randint(
        0,
        256,
        (num_rows, 1, ROPE_DIM),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    raw[(raw & 0x7F) == 0x7F] = 0x3C  # no NaN encodings
    return latent, weight, raw.view(FP8)


def _unfused(latent, weight, k_pe, num_heads):
    from vllm.models.kimi_k3.nvidia.ops.fused_mla_key_concat_kv_cache import (
        fused_mla_kv_concat_quant_fp8,
    )

    kv_nope = torch.nn.functional.linear(latent, weight)
    kv_nope = kv_nope.view(-1, num_heads, 2 * HEAD_DIM)
    k_nope, v = kv_nope.split([HEAD_DIM, HEAD_DIM], dim=-1)
    return fused_mla_kv_concat_quant_fp8(k_nope, k_pe, v)


def _assert_same_bytes(actual, expected):
    k, v = actual
    k_ref, v_ref = expected
    assert k.shape == k_ref.shape and v.shape == v_ref.shape
    assert k.dtype == FP8 and v.dtype == FP8
    assert k.is_contiguous() and v.is_contiguous()
    assert torch.equal(k.view(torch.uint8), k_ref.view(torch.uint8))
    assert torch.equal(v.view(torch.uint8), v_ref.view(torch.uint8))


@requires_sm100
@pytest.mark.parametrize("num_heads", [12, 24])
@pytest.mark.parametrize(
    "num_rows", [0, 1, 2, 63, 64, 65, 127, 128, 129, 777, 4096, 8193, 30001]
)
@torch.inference_mode()
def test_fused_kv_b_pack_matches_kv_b_proj_and_pack(num_rows, num_heads):
    latent, weight, k_pe = _inputs(num_rows, num_heads, seed=num_rows)
    _assert_same_bytes(
        mla_context_fp8.fused_kv_b_proj_fp8_pack(latent, weight, k_pe, num_heads),
        _unfused(latent, weight, k_pe, num_heads),
    )


@requires_sm100
@pytest.mark.parametrize("num_rows", [1, 129, 4097])
@torch.inference_mode()
def test_fused_kv_b_pack_saturates_like_the_pack(num_rows):
    num_heads = 12
    # A large weight drives many outputs past the largest fp8 value.
    latent, weight, k_pe = _inputs(num_rows, num_heads, weight_scale=256.0)
    k, v = mla_context_fp8.fused_kv_b_proj_fp8_pack(latent, weight, k_pe, num_heads)
    _assert_same_bytes((k, v), _unfused(latent, weight, k_pe, num_heads))
    largest_magnitude = 0x7E
    assert int(((v.view(torch.uint8) & 0x7F) == largest_magnitude).sum()) > 0


def test_check_rows_cover_size_boundaries():
    rows = mla_context_fp8.fused_kv_b_pack_check_rows(66304)
    assert rows == sorted(set(rows))
    assert rows[-2:] == [66303, 66304]
    for exp in range(13, 17):
        for r in (2**exp - 1, 2**exp, 2**exp + 1):
            assert r in rows
    assert 49152 in rows
    assert {1, 77, 128, 896, 1000, 7173} <= set(rows)

    small = mla_context_fp8.fused_kv_b_pack_check_rows(5000)
    assert small == [1, 77, 128, 896, 1000, 4999, 5000, 7173]


class _KVBProj(torch.nn.Module):
    def __init__(self, weight, weight_used=None):
        super().__init__()
        self.weight = weight
        self.weight_used = weight if weight_used is None else weight_used

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight_used), None


def _layer(num_heads, weight, weight_used=None):
    from vllm.models.kimi_k3.nvidia.mla import MultiHeadLatentAttention

    dims = SimpleNamespace(qk_nope_head_dim=HEAD_DIM, v_head_dim=HEAD_DIM)
    return SimpleNamespace(
        layer_name="model.layers.0.self_attn",
        kv_b_proj=_KVBProj(weight, weight_used),
        impl=SimpleNamespace(
            num_heads=num_heads,
            kv_lora_rank=KV_LORA_RANK,
            qk_rope_head_dim=ROPE_DIM,
            qk_nope_head_dim=HEAD_DIM,
            v_head_dim=HEAD_DIM,
        ),
        _fused_mla_kv_concat=MultiHeadLatentAttention._fused_mla_kv_concat.__get__(
            dims
        ),
    )


@requires_sm100
@torch.inference_mode()
def test_startup_check():
    num_heads = 12
    _, weight, _ = _inputs(1, num_heads)
    mla_context_fp8.check_fused_kv_b_pack(_layer(num_heads, weight), max_rows=300)

    # A kv_b_proj that does not compute F.linear with its weight is rejected.
    other = (weight.float() * 1.5).to(torch.bfloat16)
    with pytest.raises(RuntimeError, match="not bitwise identical"):
        mla_context_fp8.check_fused_kv_b_pack(
            _layer(num_heads, weight, weight_used=other), max_rows=300
        )


def test_fused_kv_b_pack_requires_fp8_context(monkeypatch):
    monkeypatch.setenv("VLLM_KIMI_K3_MLA_FUSED_KV_B_PACK", "1")
    monkeypatch.delenv("VLLM_KIMI_K3_MLA_FP8_DCP_CONTEXT", raising=False)
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context={})
    )
    with pytest.raises(ValueError, match="VLLM_KIMI_K3_MLA_FP8_DCP_CONTEXT=1"):
        mla_context_fp8.use_fp8_dcp_context(SimpleNamespace(), [], config)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-head ``W_UV`` projection of MLA decode partials.

Used by ``VLLM_KIMI_K3_MLA_WUV_BEFORE_DCP_COMBINE``: the latent partial
attention outputs of every gathered head are projected to the value dimension
before the decode-context-parallel combine instead of after it, so the
exchange carries ``v_head_dim`` instead of ``kv_lora_rank`` values per head.
The LSE combine is a per-head weighted sum of the partials, so it commutes
with the per-head projection.
"""

import math

import torch

from vllm.triton_utils import tl, triton

# The decode latent is read back from the fp8 KV cache; e4m3 rounding is up to
# 2**-4 relative for normal values, widened here to 7%.
_FP8_CACHE_ROUNDING = 1.07
# Required headroom of the projection bound below the fp16 maximum.
_FP16_HEADROOM = 8.0

# Program tile: BLOCK_T tokens x v_head_dim of one head, BLOCK_K latent
# channels per step. Tunable.
_BLOCK_T = 128
_BLOCK_K = 64
_NUM_STAGES = 3
_NUM_WARPS = 4


@triton.jit(do_not_specialize=["num_tokens"])
def _wuv_projection_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    num_tokens,
    stride_x_token,
    stride_x_head,
    stride_w_head,
    stride_out_token,
    stride_out_head,
    LATENT_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    # out[t, h, :] = x[t, h, :] @ w[h], accumulated in fp32.
    token_block = tl.program_id(0)
    head = tl.program_id(1).to(tl.int64)
    tokens = token_block * BLOCK_T + tl.arange(0, BLOCK_T)
    token_mask = tokens < num_tokens
    k = tl.arange(0, BLOCK_K)
    v = tl.arange(0, V_DIM)
    x_rows = (
        x_ptr + head * stride_x_head + tokens.to(tl.int64)[:, None] * stride_x_token
    )
    w_head = w_ptr + head * stride_w_head
    acc = tl.zeros((BLOCK_T, V_DIM), dtype=tl.float32)
    for k0 in tl.range(0, LATENT_DIM, BLOCK_K, num_stages=NUM_STAGES):
        x = tl.load(x_rows + (k0 + k)[None, :], mask=token_mask[:, None], other=0.0)
        w = tl.load(w_head + (k0 + k)[:, None] * V_DIM + v[None, :])
        acc = tl.dot(x, w, acc)
    out = (
        out_ptr
        + head * stride_out_head
        + tokens.to(tl.int64)[:, None] * stride_out_token
        + v[None, :]
    )
    tl.store(out, acc.to(out_ptr.dtype.element_ty), mask=token_mask[:, None])


def mla_wuv_projection(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor) -> None:
    """``out[t, h, :] = x[t, h, :] @ w[h]`` with fp32 accumulation.

    Args:
        x: latent partials [T, H, L] (unit stride on L).
        w: contiguous per-head ``W_UV`` [H, L, V].
        out: [T, H, V] (unit stride on V); written in its own dtype.
    """
    num_tokens, num_heads, latent_dim = x.shape
    v_dim = w.shape[2]
    assert w.is_contiguous() and tuple(w.shape[:2]) == (num_heads, latent_dim)
    assert tuple(out.shape) == (num_tokens, num_heads, v_dim)
    assert x.stride(2) == 1 and out.stride(2) == 1
    assert latent_dim % _BLOCK_K == 0
    grid = (triton.cdiv(num_tokens, _BLOCK_T), num_heads)
    _wuv_projection_kernel[grid](
        x,
        w,
        out,
        num_tokens,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        out.stride(0),
        out.stride(1),
        LATENT_DIM=latent_dim,
        V_DIM=v_dim,
        BLOCK_T=_BLOCK_T,
        BLOCK_K=_BLOCK_K,
        NUM_STAGES=_NUM_STAGES,
        num_warps=_NUM_WARPS,
    )


def wuv_projection_bound(w_uv: torch.Tensor, kv_norm_weight: torch.Tensor) -> float:
    """Upper bound of ``|latent @ W_UV|`` over all heads and value channels.

    The decode latent is a convex combination of cached ``kv_a_layernorm``
    outputs. Each RMSNorm output channel satisfies
    ``|y_l| <= sqrt(L) * |gamma_l|``, widened for the fp8 cache rounding, and
    the LSE combine is again a convex combination.

    Args:
        w_uv: [H, L, V] per-head ``W_UV``.
        kv_norm_weight: [L] RMSNorm weight ``gamma`` of the latent.
    """
    latent_dim = w_uv.shape[1]
    channel_bound = math.sqrt(latent_dim) * kv_norm_weight.detach().float().abs()
    per_output = (w_uv.detach().float().abs() * channel_bound[None, :, None]).sum(1)
    return _FP8_CACHE_ROUNDING * float(per_output.max())


def fits_fp16_exchange(bound: float) -> bool:
    """Whether a projection bound leaves the required headroom in fp16."""
    return bound * _FP16_HEADROOM < torch.finfo(torch.float16).max

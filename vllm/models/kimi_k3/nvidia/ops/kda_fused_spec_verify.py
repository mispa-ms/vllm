# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Kimi-K3 KDA speculative verify (VLLM_KIMI_K3_KDA_FUSED_SPEC_VERIFY).

In a pure speculative-verify batch the native KDA path of
``KimiK3DeltaAttention._forward`` runs three kernels per layer:
``_causal_conv1d_update_kernel`` -> ``fused_recurrent_kda_fwd_kernel`` (T=4
tokens with per-token state checkpoints) -> ``layer_norm_gated_fwd_kernel``
(in place). The Gluon kernel in kda_fused_spec_verify_kernel.py does all three
in one launch, one CTA per (request, head):

1. the request's initial state tile is copied to shared memory with cp.async,
   in RC-row chunks, while the small per-request inputs are loaded;
2. the causal conv1d update of the head's q/k/v channels (and the new conv
   window);
3. the per-token q/k l2 norms and gate decay;
4. the gated delta-rule recurrence, chunk by chunk (value rows are
   independent), storing the per-token state checkpoints;
5. the sigmoid-gated RMSNorm of the outputs.

Numerics: bitwise identical to the three stock kernels (outputs of every
request token, the recurrent-state pool and the conv-state pool). Every
expression whose rounding depends on contraction (conv chain, l2 norms, k
dot, state update, q dot, norm variance) is written as inline PTX with
explicit ``.rn`` rounding, which ptxas does not contract, in the operation
order the stock kernels execute; reductions keep the stock in-thread trees and
lane butterflies. The remaining element-wise ops are the same Triton lowerings
as in the stock kernels.

Known difference: token rows that belong to no request (CUDA-graph token
padding after the last request) are left as they are, while the stock norm
normalizes their contents in place. Nothing reads those rows.

This module holds the launcher and the eligibility checks and imports no
Triton; the kernel module is imported on the first launch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.models.kimi_k3.nvidia.kda import KimiK3DeltaAttention
    from vllm.models.kimi_k3.nvidia.kda_metadata import KimiK3KDAMetadata

# The validated shape: T = 4 tokens per request (3 speculative tokens), conv
# width 4, 12 local heads of dimension 128 (Kimi-K3 at TP8).
SPEC_TOKENS = 4
CONV_WIDTH = 4
NUM_HEADS = 12
HEAD_DIM = 128
CONV_STATE_LEN = SPEC_TOKENS + CONV_WIDTH - 2
NUM_WARPS = 4
ROW_CHUNK = 64


def fused_spec_verify(
    mixed_qkv: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    g1: torch.Tensor,
    beta: torch.Tensor,
    g2: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Run conv1d update + gated delta rule + gated RMSNorm in one launch.

    Shapes (N requests, T = 4 tokens, H heads of size D): mixed_qkv
    [tokens, 3*H*D]; conv_state [slots, 3*H*D, T + 2] (dim-first); conv_weight
    [3*H*D, 4] fp32; g1 [1, tokens, H, D]; beta [1, tokens, H]; g2 [tokens, H,
    D]; state [slots, H, D, D] bf16; state_indices [N, T]; query_start_loc
    [N + 1]; num_accepted_tokens [N]; out [1, tokens, H, D] (written in place).
    """
    from vllm.models.kimi_k3.nvidia.ops.kda_fused_spec_verify_kernel import (
        _kda_fused_spec_verify_kernel,
    )

    ntok, conv_dim = mixed_qkv.shape
    _, H, D, _ = state.shape
    num_reqs, T = state_indices.shape
    W = conv_weight.shape[-1]
    SL = conv_state.shape[-1]
    assert T == SPEC_TOKENS and W == CONV_WIDTH and D == HEAD_DIM
    assert conv_dim == 3 * H * D and mixed_qkv.stride(1) == 1
    assert conv_state.stride()[1:] == (SL, 1) and SL == T + W - 2
    assert conv_weight.stride() == (W, 1) and conv_weight.dtype == torch.float32
    assert state.stride()[1:] == (D * D, D, 1) and state.dtype == torch.bfloat16
    assert g1.stride()[2:] == (D, 1) and g2.stride()[1:] == (D, 1)
    assert beta.stride(2) == 1
    assert out.is_contiguous() and state_indices.stride(1) == 1
    _kda_fused_spec_verify_kernel[(num_reqs * H,)](
        mixed_qkv,
        mixed_qkv.stride(0),
        conv_state,
        conv_state.stride(0),
        conv_weight,
        g1,
        g1.stride(1),
        beta,
        beta.stride(1),
        g2,
        g2.stride(0),
        norm_weight,
        eps,
        A_log,
        dt_bias,
        lower_bound,
        state,
        state.stride(0),
        state_indices,
        state_indices.stride(0),
        query_start_loc,
        num_accepted_tokens,
        out,
        H=H,
        D=D,
        T=T,
        SL=SL,
        W=W,
        NW=NUM_WARPS,
        RC=ROW_CHUNK,
        SCALE=D**-0.5,
        num_warps=NUM_WARPS,
    )
    return out


# --------------------------------------------------------------------------
# Layer and batch eligibility, launch variants.
# --------------------------------------------------------------------------

# Launch variants (exact int arguments, dtypes and 16-byte alignment of the
# tensors) that have been compiled in this process. A call during CUDA-graph
# capture with a variant not in this set uses the stock kernels instead of
# compiling inside the capture.
_compiled_variants: set[tuple] = set()


def layer_unsupported_reason(layer: KimiK3DeltaAttention) -> str | None:
    """Why ``layer`` cannot use the fused verify (None: it can)."""
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated

    f32, bf16 = torch.float32, torch.bfloat16
    if layer.kda_spec_decode_backend != "native":
        return f"kda_spec_decode_backend is {layer.kda_spec_decode_backend!r}"
    if layer.use_recoverssm:
        return "RecoverSSM is enabled"
    if layer.num_spec != SPEC_TOKENS - 1:
        return f"{layer.num_spec} speculative tokens (supported: {SPEC_TOKENS - 1})"
    H, D = layer.local_num_heads, layer.head_dim
    if H != NUM_HEADS or D != HEAD_DIM:
        return f"{H} local heads of size {D} (supported: {NUM_HEADS} of {HEAD_DIM})"
    if layer.local_projection_size != H * D:
        return "local projection size is not heads x head_dim"
    w = layer.conv1d.weight
    if (
        w.dtype != f32
        or tuple(w.shape) != (3 * H * D, 1, CONV_WIDTH)
        or not w.is_contiguous()
    ):
        return "conv1d weight is not a contiguous fp32 [3*H*D, 1, 4] tensor"
    if getattr(layer.conv1d, "bias", None) is not None:
        return "conv1d has a bias"
    if layer.gate_lower_bound is None:
        return "no gate lower bound"
    a_log, dt_bias = layer.A_log, layer.dt_bias
    if a_log.dtype != f32 or tuple(a_log.shape) != (H,) or not a_log.is_contiguous():
        return "A_log is not a contiguous fp32 [H] tensor"
    if (
        dt_bias.dtype != f32
        or tuple(dt_bias.shape) != (H * D,)
        or not dt_bias.is_contiguous()
    ):
        return "dt_bias is not a contiguous fp32 [H*D] tensor"
    norm = layer.o_norm
    if not isinstance(norm, FusedRMSNormGated) or norm.activation != "sigmoid":
        return "o_norm is not a sigmoid-gated FusedRMSNormGated"
    forward_method = getattr(norm, "_forward_method", None)
    if getattr(forward_method, "__func__", None) is not FusedRMSNormGated.forward_cuda:
        return "o_norm does not use the Triton gated RMSNorm (forward_cuda)"
    if (
        norm.weight.dtype != bf16
        or tuple(norm.weight.shape) != (D,)
        or getattr(norm, "bias", None) is not None
    ):
        return "o_norm weight is not bf16 [head_dim] or has a bias"
    if not is_conv_state_dim_first():
        return "the conv state layout is not dim-first (VLLM_SSM_CONV_STATE_LAYOUT)"
    return None


def _variant(args: tuple) -> tuple:
    """What a launch with ``args`` is compiled for: the device, the dtype and
    16-byte alignment of every tensor and every integer stride (a superset of
    what Triton specializes on)."""
    x, conv_state, _, _, _, _, g1, beta, g2, _, _, state, idx = args[:13]
    tensors = [a for a in args if isinstance(a, torch.Tensor)]
    ints = (
        x.stride(0),
        conv_state.stride(0),
        g1.stride(1),
        beta.stride(1),
        g2.stride(0),
        state.stride(0),
        idx.stride(0),
    )
    return (
        x.device.index,
        tuple((t.dtype, t.data_ptr() % 16 == 0) for t in tensors),
        tuple(int(i) for i in ints),
    )


def _launch_args(
    layer: KimiK3DeltaAttention,
    m: KimiK3KDAMetadata,
    mixed_qkv: torch.Tensor,
    g1: torch.Tensor,
    g2: torch.Tensor,
    beta: torch.Tensor,
    core_attn_out: torch.Tensor,
) -> tuple | None:
    """The kernel arguments for a pure speculative-verify batch, or None."""
    n = m.num_spec_decodes
    if n <= 0 or m.num_prefills != 0 or m.num_decodes != 0:
        return None
    if m.spec_token_start is not None:
        return None
    idx = m.spec_state_indices_tensor
    qsl = m.spec_query_start_loc
    nacc = m.num_accepted_tokens
    if idx is None or qsl is None or nacc is None:
        return None
    i32, bf16 = torch.int32, torch.bfloat16
    if (
        idx.dtype != i32
        or idx.ndim != 2
        or idx.shape[1] != SPEC_TOKENS
        or idx.shape[0] < n
        or idx.stride(1) != 1
    ):
        return None
    if qsl.dtype != i32 or qsl.ndim != 1 or qsl.numel() < n + 1 or qsl.stride(0) != 1:
        return None
    if nacc.dtype != i32 or nacc.ndim != 1 or nacc.numel() < n or nacc.stride(0) != 1:
        return None
    n_act = m.num_actual_tokens
    conv_state, state = layer.kv_cache[0], layer.kv_cache[1]
    H, D = layer.local_num_heads, layer.head_dim
    HD = H * D
    SL = CONV_STATE_LEN
    x = mixed_qkv[:n_act]
    g1 = g1[:, :n_act]
    beta = beta[:, :n_act]
    g2 = g2[:n_act]
    out = core_attn_out[:, :n_act]
    if x.dtype != bf16 or x.ndim != 2 or x.shape != (n_act, 3 * HD) or x.stride(1) != 1:
        return None
    if (
        conv_state.dtype != bf16
        or conv_state.ndim != 3
        or tuple(conv_state.shape[1:]) != (3 * HD, SL)
        or conv_state.stride()[1:] != (SL, 1)
    ):
        return None
    if (
        state.dtype != bf16
        or state.ndim != 4
        or tuple(state.shape[1:]) != (H, D, D)
        or state.stride()[1:] != (D * D, D, 1)
    ):
        return None
    if (
        g1.dtype != bf16
        or g1.ndim != 4
        or tuple(g1.shape) != (1, n_act, H, D)
        or g1.stride()[2:] != (D, 1)
    ):
        return None
    if (
        beta.dtype != bf16
        or beta.ndim != 3
        or tuple(beta.shape) != (1, n_act, H)
        or beta.stride(2) != 1
    ):
        return None
    if (
        g2.dtype != bf16
        or g2.ndim != 3
        or tuple(g2.shape) != (n_act, H, D)
        or g2.stride()[1:] != (D, 1)
    ):
        return None
    if (
        out.dtype != bf16
        or out.ndim != 4
        or tuple(out.shape) != (1, n_act, H, D)
        or not out.is_contiguous()
    ):
        return None
    device = x.device
    if any(
        t.device != device
        for t in (conv_state, state, g1, beta, g2, out, idx, qsl, nacc)
    ):
        return None
    norm = layer.o_norm
    return (
        x,
        conv_state,
        layer.conv1d.weight.view(3 * HD, CONV_WIDTH),
        layer.A_log,
        layer.dt_bias,
        float(layer.gate_lower_bound),
        g1,
        beta,
        g2,
        norm.weight,
        float(norm.eps),
        state,
        idx[:n],
        qsl[: n + 1],
        nacc[:n],
        out,
    )


def maybe_fused_spec_verify(
    layer: KimiK3DeltaAttention,
    m: KimiK3KDAMetadata,
    mixed_qkv: torch.Tensor,
    g1: torch.Tensor,
    g2: torch.Tensor,
    beta: torch.Tensor,
    core_attn_out: torch.Tensor,
) -> bool:
    """Run the fused verify for this batch if it applies; return whether it ran.

    It applies to pure speculative-verify batches of a supported layer whose
    tensors have the layouts the kernel reads; every other call returns False
    and the caller runs the stock kernels. The layer check runs once per
    layer.
    """
    supported = layer.__dict__.get("_fused_spec_verify_supported")
    if supported is None:
        supported = layer_unsupported_reason(layer) is None
        layer.__dict__["_fused_spec_verify_supported"] = supported
    if not supported:
        return False
    args = _launch_args(layer, m, mixed_qkv, g1, g2, beta, core_attn_out)
    if args is None:
        return False
    variant = _variant(args)
    if torch.cuda.is_current_stream_capturing() and variant not in _compiled_variants:
        return False
    fused_spec_verify(*args)
    _compiled_variants.add(variant)
    return True


def precompile_capture_variants(
    layer: KimiK3DeltaAttention, proj_rows: int, overlap: tuple[int, int] | None
) -> list[tuple]:
    """Compile the launch variants CUDA-graph capture will use.

    Launches the kernel once per input layout on dummy activations laid out as
    the model's (``proj_rows``: row stride of the in_proj_qkvgfab output;
    ``overlap``: (qkvg rows, f_a/beta rows) of the low-latency projection
    overlap path, or None) with the layer's own KV cache and weights. Every
    request uses state index 0, so the launch reads and writes no cache slot.
    """
    H, D = layer.local_num_heads, layer.head_dim
    HD = H * D
    conv_state, state = layer.kv_cache[0], layer.kv_cache[1]
    device = conv_state.device
    ntok, bf16 = 2 * SPEC_TOKENS, torch.bfloat16
    layouts: list[tuple[int, int | None]] = [(proj_rows, None)]
    if overlap is not None:
        layouts.append(overlap)
    variants = []
    with torch.accelerator.device_index(device.index):
        for rows, fab_rows in layouts:
            proj = torch.zeros(ntok, rows, dtype=bf16, device=device)
            x = proj[:, : 3 * HD]
            g2 = proj[:, 3 * HD : 4 * HD].unflatten(-1, (H, D))
            if fab_rows is None:
                beta = proj[:, 4 * HD + D : 4 * HD + D + H].unsqueeze(0)
            else:
                fab = torch.zeros(ntok, fab_rows, dtype=bf16, device=device)
                beta = fab[:, D : D + H].unsqueeze(0)
            g1 = torch.zeros(ntok, HD, dtype=bf16, device=device).view(1, ntok, H, D)
            out = torch.zeros(1, ntok, H, D, dtype=bf16, device=device)
            idx = torch.zeros(2, SPEC_TOKENS, dtype=torch.int32, device=device)
            qsl = torch.tensor(
                [0, SPEC_TOKENS, 2 * SPEC_TOKENS], dtype=torch.int32, device=device
            )
            nacc = torch.ones(2, dtype=torch.int32, device=device)
            norm = layer.o_norm
            args = (
                x,
                conv_state,
                layer.conv1d.weight.view(3 * HD, CONV_WIDTH),
                layer.A_log,
                layer.dt_bias,
                float(layer.gate_lower_bound),
                g1,
                beta,
                g2,
                norm.weight,
                float(norm.eps),
                state,
                idx,
                qsl,
                nacc,
                out,
            )
            fused_spec_verify(*args)
            variant = _variant(args)
            _compiled_variants.add(variant)
            variants.append(variant)
        torch.accelerator.synchronize(device)
    return variants

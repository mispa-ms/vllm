# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up Kimi-K3 Triton kernels."""

from __future__ import annotations

import importlib
import sys
from collections import Counter
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.models.kimi_k3.nvidia.kda import KimiK3DeltaAttention
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


def _get_kda_layer(
    worker: Worker, module: str = "vllm.models.kimi_k3.nvidia.kda"
) -> KimiK3DeltaAttention | None:
    # Kimi model construction already imports kda. Avoid importing it here.
    kda = sys.modules.get(module)
    if kda is None:
        return None

    compilation_config = getattr(
        worker.model_runner,
        "compilation_config",
        None,
    )
    static_context = getattr(compilation_config, "static_forward_context", None)
    if not isinstance(static_context, dict):
        return None
    return next(
        (
            layer
            for layer in static_context.values()
            if isinstance(layer, kda.KimiK3DeltaAttention)
        ),
        None,
    )


def _warm_attn_res(worker: Worker) -> None:
    from vllm.models.kimi_k3.nvidia.ops.attn_res import (
        attn_res,
        get_attn_res_triton_warmup_profiles,
    )

    config = worker.model_config.hf_text_config
    block_size = getattr(config, "attn_res_block_size", None)
    if block_size is None:
        return

    hidden_size = int(config.hidden_size)
    max_blocks = (int(config.num_hidden_layers) + block_size - 1) // block_size
    if max_blocks < 2:
        return

    dtype = worker.model_config.dtype
    device = torch.device("cuda")
    eps = float(config.rms_norm_eps)
    prefix = torch.zeros((1, hidden_size), dtype=dtype, device=device)
    delta = torch.zeros_like(prefix)
    blocks = torch.zeros(
        (1, max_blocks, hidden_size),
        dtype=dtype,
        device=device,
    )
    norm_weight = torch.zeros(hidden_size, dtype=dtype, device=device)
    qk_weight = torch.zeros_like(norm_weight)
    output_norm_weight = torch.zeros_like(norm_weight)

    for (
        num_blocks,
        has_delta,
        block_write_idx,
        apply_output_norm,
    ) in get_attn_res_triton_warmup_profiles(max_blocks):
        attn_res(
            prefix,
            delta if has_delta else None,
            blocks,
            norm_weight,
            qk_weight,
            output_norm_weight if apply_output_norm else None,
            num_blocks=num_blocks,
            block_write_idx=block_write_idx,
            eps=eps,
            output_norm_eps=eps if apply_output_norm else 0.0,
        )


def _warm_recurrent_kda(
    layer: KimiK3DeltaAttention,
    input_dtype: torch.dtype,
) -> None:
    from vllm.models.kimi_k3.nvidia.ops.third_party.kda.fused_recurrent import (
        fused_recurrent_kda,
        get_fused_recurrent_kda_fwd_warmup_profiles,
    )

    num_speculative_tokens = int(layer.num_spec)
    # fused_recurrent_kda_fwd_kernel is only used by speculative decode.
    if num_speculative_tokens <= 0:
        return

    kv_cache = layer.kv_cache
    if not isinstance(kv_cache, (list, tuple)) or len(kv_cache) < 2:
        return
    state = kv_cache[1]
    if not isinstance(state, torch.Tensor) or not state.numel():
        return

    logger.info("Warming up Kimi-K3 speculative KDA kernels.")
    h = int(layer.local_num_heads)
    d = int(layer.head_dim)
    tokens_per_sequence = num_speculative_tokens + 1
    for num_sequences in get_fused_recurrent_kda_fwd_warmup_profiles(h):
        num_tokens = num_sequences * tokens_per_sequence
        packed_qkv = torch.empty(
            (num_tokens, 3 * h * d),
            dtype=input_dtype,
            device=state.device,
        )
        q, k, v = (
            tensor.view(1, num_tokens, h, d)
            for tensor in packed_qkv.split(h * d, dim=-1)
        )
        fused_recurrent_kda(
            q=q,
            k=k,
            v=v,
            raw_g=torch.empty(
                (1, num_tokens, h, d),
                dtype=input_dtype,
                device=state.device,
            ),
            raw_beta=torch.empty(
                (1, num_tokens, h),
                dtype=input_dtype,
                device=state.device,
            ),
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            lower_bound=layer.gate_lower_bound,
            initial_state=state[:1],
            cu_seqlens=torch.arange(
                0,
                num_tokens + 1,
                tokens_per_sequence,
                dtype=torch.int32,
                device=state.device,
            ),
            ssm_state_indices=torch.zeros(
                (num_sequences, tokens_per_sequence),
                dtype=torch.int32,
                device=state.device,
            ),
            num_accepted_tokens=torch.ones(
                num_sequences,
                dtype=torch.int32,
                device=state.device,
            ),
            out=torch.empty(
                (1, num_tokens, h, d),
                dtype=input_dtype,
                device=state.device,
            ),
        )


def _warm_gluon_chunk_kda(layer, input_dtype: torch.dtype) -> None:
    from vllm.models.kimi_k3.amd.ops.kda_prefill import (
        chunk_kda_prefill,
        get_gluon_chunk_kda_warmup_profiles,
    )

    if layer.kda_prefill_backend != "gluon":
        return
    kv_cache = layer.kv_cache
    if not isinstance(kv_cache, (list, tuple)) or len(kv_cache) < 2:
        return
    state = kv_cache[1]
    if not isinstance(state, torch.Tensor) or not state.numel():
        return

    logger.info("Warming up Kimi-K3 Gluon chunk KDA prefill kernels.")
    h, d = int(layer.local_num_heads), int(layer.head_dim)
    for num_seqs, seq_len in get_gluon_chunk_kda_warmup_profiles():
        num_tokens = num_seqs * seq_len
        shape = (1, num_tokens, h, d)
        chunk_kda_prefill(
            q=torch.zeros(shape, dtype=input_dtype, device=state.device),
            k=torch.zeros(shape, dtype=input_dtype, device=state.device),
            v=torch.zeros(shape, dtype=input_dtype, device=state.device),
            raw_g=torch.zeros(shape, dtype=input_dtype, device=state.device),
            raw_beta=torch.zeros(
                (1, num_tokens, h), dtype=input_dtype, device=state.device
            ),
            A_log=layer.A_log,
            g_bias=layer.dt_bias,
            lower_bound=layer.gate_lower_bound,
            state_cache=state,
            state_indices=torch.zeros(num_seqs, dtype=torch.int32, device=state.device),
            has_initial_state=torch.zeros(
                num_seqs, dtype=torch.bool, device=state.device
            ),
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=torch.arange(
                0, num_tokens + 1, seq_len, dtype=torch.int32, device=state.device
            ),
            backend="gluon",
            out=torch.empty(shape, dtype=input_dtype, device=state.device),
        )


@torch.inference_mode()
def kimi_k3_triton_warmup(worker: Worker) -> None:
    """Warm Kimi-K3 Triton kernels reachable by this server."""
    if current_platform.is_rocm():
        layer = _get_kda_layer(worker, "vllm.models.kimi_k3.amd.kda")
        if layer is not None:
            _warm_gluon_chunk_kda(layer, worker.model_config.dtype)
        return
    if not current_platform.is_cuda():
        return

    layer = _get_kda_layer(worker)
    if layer is None:
        return

    _warm_attn_res(worker)
    _warm_recurrent_kda(layer, worker.model_config.dtype)


def _kv_layout_supported(layer: KimiK3DeltaAttention) -> bool:
    from vllm.models.kimi_k3.nvidia.ops.kda_fused_spec_verify import (
        CONV_STATE_LEN,
    )

    kv_cache = layer.kv_cache
    if not isinstance(kv_cache, (list, tuple)) or len(kv_cache) < 2:
        return False
    conv_state, state = kv_cache[0], kv_cache[1]
    if not isinstance(conv_state, torch.Tensor) or not isinstance(state, torch.Tensor):
        return False
    H, D = layer.local_num_heads, layer.head_dim
    return (
        conv_state.dtype == torch.bfloat16
        and tuple(conv_state.shape[1:]) == (3 * H * D, CONV_STATE_LEN)
        and conv_state.stride()[1:] == (CONV_STATE_LEN, 1)
        and state.dtype == torch.bfloat16
        and tuple(state.shape[1:]) == (H, D, D)
        and state.stride()[1:] == (D * D, D, 1)
    )


@torch.inference_mode()
def kimi_k3_kda_fused_spec_verify_warmup(worker: Worker) -> None:
    """Compile the fused KDA speculative verify before CUDA-graph capture.

    With VLLM_KIMI_K3_KDA_FUSED_SPEC_VERIFY set, checks that at least one KDA
    layer supports it and compiles the kernel for the two input layouts the
    captured decode graphs use (the in_proj_qkvgfab output and the
    low-latency projection-overlap output), once per KV-cache layout. A call
    during capture with a layout that was not compiled here runs the stock
    kernels.
    """
    if not envs.VLLM_KIMI_K3_KDA_FUSED_SPEC_VERIFY:
        return
    kda = sys.modules.get("vllm.models.kimi_k3.nvidia.kda")
    static_context = getattr(
        getattr(worker.model_runner, "compilation_config", None),
        "static_forward_context",
        None,
    )
    if kda is None or not isinstance(static_context, dict):
        return
    layers = [
        layer
        for layer in static_context.values()
        if isinstance(layer, kda.KimiK3DeltaAttention)
    ]
    if not layers:
        return
    if not (current_platform.is_cuda() and current_platform.has_device_capability(100)):
        raise NotImplementedError(
            "VLLM_KIMI_K3_KDA_FUSED_SPEC_VERIFY requires an NVIDIA Blackwell "
            "(SM100 or newer) GPU."
        )
    from vllm.models.kimi_k3.nvidia.ops import kda_fused_spec_verify as fused

    unsupported: Counter[str] = Counter()
    supported = []
    for layer in layers:
        reason = fused.layer_unsupported_reason(layer)
        if reason is None and not _kv_layout_supported(layer):
            reason = "KV cache is not a bf16 dim-first conv state + bf16 state"
        if reason is None:
            supported.append(layer)
        else:
            unsupported[reason] += 1
    if not supported:
        raise ValueError(
            "VLLM_KIMI_K3_KDA_FUSED_SPEC_VERIFY is set, but no KDA layer supports "
            f"it: {dict(unsupported)}."
        )
    if unsupported:
        logger.warning(
            "Fused KDA speculative verify is off for %d of %d KDA layers: %s",
            sum(unsupported.values()),
            len(layers),
            dict(unsupported),
        )
    # KimiK3DeltaAttention.forward splits the in_proj_qkvgfab output as
    # [3*H*D, H*D, D, H, padding]; the projection-overlap path produces the
    # first 4*H*D columns and [D, H] separately.
    first = supported[0]
    proj_rows = (
        4 * first.local_projection_size
        + first.head_dim
        + first.local_num_heads
        + int(first.in_proj_padding)
    )
    overlap = None
    low_latency_gemm = importlib.import_module(
        "vllm.models.kimi_k3.nvidia.low_latency_gemm"
    )
    qkvg_rows = getattr(low_latency_gemm, "_KDA_QKVG_SIZE", None)
    if isinstance(qkvg_rows, int) and 0 < qkvg_rows < proj_rows:
        overlap = (qkvg_rows, proj_rows - qkvg_rows)
    by_kv_layout = {}
    for layer in supported:
        conv_state, state = layer.kv_cache[0], layer.kv_cache[1]
        key = (
            conv_state.stride(0),
            state.stride(0),
            conv_state.data_ptr() % 16 == 0,
            state.data_ptr() % 16 == 0,
        )
        by_kv_layout.setdefault(key, layer)
    variants = []
    for layer in by_kv_layout.values():
        variants += fused.precompile_capture_variants(layer, proj_rows, overlap)
    logger.info(
        "Fused KDA speculative verify on %d of %d KDA layers; compiled %d launch "
        "variants (int args %s).",
        len(supported),
        len(layers),
        len(variants),
        ", ".join(str(v[2]) for v in variants),
    )

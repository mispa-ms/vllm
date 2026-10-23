# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""fp8 chunked-context path for Kimi-K3 MLA prefill under decode context
parallelism (DCP), enabled with ``VLLM_KIMI_K3_MLA_FP8_DCP_CONTEXT=1``.

With a plain per-tensor fp8 KV cache and an fp8 prefill query, the default DCP
context loop (``MLACommonBaseImpl._context_parallel_compute_prefill_context``)
dequantizes each rank's context rows into a model-dtype workspace, all-gathers
them across the DCP group, reorganizes them with one slice per (request, rank)
plus ``torch.cat``, and casts ``k_pe`` back to fp8 when it packs K.

This module keeps the workspace of a Kimi-K3 target KV-cache group in fp8
instead:

* each rank copies its context rows as stored (no dequantization),
* the DCP all-gather moves the fp8 rows,
* one Triton kernel reorganizes the gathered rows through a per-chunk row map
  (built once per scheduler step and shared by all layers of the group),
  writing the latent as bf16 for ``kv_b_proj`` and ``k_pe`` as the stored fp8
  bytes.

A group only takes this path when every layer's ``k_scale`` is 1.0. The default
dequantization then yields exactly the fp8 value in bf16 and the later fp8 cast
of ``k_pe`` returns the stored bytes, so the K/V given to the prefill attention
are bitwise identical to the default path. Groups that do not qualify keep the
default path; unsupported deployments are rejected when the metadata builder is
created.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import async_tensor_h2d

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonMetadataBuilder,
        MLACommonPrefillMetadata,
    )

logger = init_logger(__name__)

FP8_DTYPE = torch.float8_e4m3fn
_PLAIN_FP8_CACHE_DTYPES = ("fp8", "fp8_e4m3")
# (kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim, v_head_dim) of the
# supported Kimi-K3 MLA layers.
_SUPPORTED_MLA_DIMS = (512, 64, 128, 128)
# Attribute set on the MLA impl of every layer of a group using this path.
_LAYER_STATE_ATTR = "fp8_dcp_context_state"

# Launch shape of the row-gather kernel; tunable.
_GATHER_BLOCK_ROWS = 32
_GATHER_NUM_WARPS = 4


@dataclass
class Fp8DCPContextLayerState:
    """Per-layer state of the fp8 DCP context path."""

    # The k_scale tensor checked to be 1.0, and its in-place version counter.
    k_scale: torch.Tensor
    k_scale_version: int | None


def _tensor_version(t: torch.Tensor) -> int | None:
    # Inference tensors have no version counter.
    try:
        return t._version
    except RuntimeError:
        return None


def _is_draft_layer(layer: torch.nn.Module) -> bool:
    return bool(
        getattr(layer, "is_draft_layer", False)
        or getattr(layer, "non_causal_multi_token_decode", False)
    )


def _group_name(layer_names: list[str]) -> str:
    shown = ", ".join(layer_names[:3])
    if len(layer_names) > 3:
        shown += f", ... ({len(layer_names)} layers)"
    return shown


def use_fp8_dcp_context(
    builder: "MLACommonMetadataBuilder",
    layer_names: list[str],
    vllm_config: "VllmConfig",
) -> bool:
    """Decide whether a KV-cache group uses the fp8 DCP context path.

    Only groups of Kimi-K3 target MLA layers are considered; every other group
    keeps the default path. A Kimi-K3 target group whose deployment cannot use
    the path raises ``ValueError``. A group whose layers have ``k_scale != 1``
    or a ``kv_b_proj`` that does not consume a bf16 latent keeps the default
    path, because the fp8 path would no longer be bitwise identical to it.
    """
    from vllm.model_executor.layers.attention.mla_attention import (
        _get_kv_b_proj_input_dtype,
    )
    from vllm.models.kimi_k3.nvidia.mla import MultiHeadLatentAttention
    from vllm.v1.attention.backends.mla.tokenspeed_mla import (
        TokenspeedMLAMetadataBuilder,
    )

    option = "VLLM_KIMI_K3_MLA_FP8_DCP_CONTEXT"
    forward_context = vllm_config.compilation_config.static_forward_context
    layers = [forward_context[name] for name in layer_names]
    is_k3 = [isinstance(layer, MultiHeadLatentAttention) for layer in layers]
    if not any(is_k3):
        return False
    group = _group_name(layer_names)
    if not all(is_k3):
        raise ValueError(
            f"{option}: KV-cache group {group} mixes Kimi-K3 MLA layers with "
            "other layers."
        )
    is_draft = [_is_draft_layer(layer) for layer in layers]
    if all(is_draft):
        # Draft layers only run non-causal multi-token decode, never the
        # chunked-context prefill.
        logger.info(
            "%s: draft MLA group %s keeps the default chunked-context path.",
            option,
            group,
        )
        return False
    if any(is_draft):
        raise ValueError(
            f"{option}: KV-cache group {group} mixes Kimi-K3 target and draft "
            "MLA layers."
        )

    dcp_size = vllm_config.parallel_config.decode_context_parallel_size
    if dcp_size <= 1:
        raise ValueError(
            f"{option} requires decode context parallelism "
            f"(decode_context_parallel_size > 1), got {dcp_size}."
        )
    if not isinstance(builder, TokenspeedMLAMetadataBuilder):
        raise ValueError(
            f"{option} is only supported with the TOKENSPEED_MLA attention "
            f"backend, got {type(builder).__name__}."
        )
    cache_dtype = vllm_config.cache_config.cache_dtype
    if cache_dtype not in _PLAIN_FP8_CACHE_DTYPES:
        raise ValueError(
            f"{option} requires a plain fp8 KV cache (--kv-cache-dtype fp8), "
            f"got {cache_dtype!r}."
        )
    if builder.q_data_type != FP8_DTYPE:
        raise ValueError(
            f"{option} requires the fp8 prefill query; enable it with "
            "--attention-config '{\"use_prefill_query_quantization\": true}'."
        )

    fallback_reasons: list[str] = []
    for name, layer in zip(layer_names, layers):
        impl = layer.impl
        if (
            layer.kv_cache_dtype not in _PLAIN_FP8_CACHE_DTYPES
            or getattr(impl, "kv_cache_dtype", None) not in _PLAIN_FP8_CACHE_DTYPES
        ):
            raise ValueError(
                f"{option}: layer {name} has kv_cache_dtype "
                f"{layer.kv_cache_dtype!r}, not a plain fp8 cache."
            )
        if layer.dcp_world_size != dcp_size or layer.dcp_manager is None:
            raise ValueError(
                f"{option}: layer {name} is not set up for decode context "
                f"parallelism of size {dcp_size}."
            )
        if getattr(impl, "kv_b_proj", None) is not layer.kv_b_proj:
            raise ValueError(
                f"{option}: the attention impl of layer {name} does not use "
                "the layer's kv_b_proj."
            )
        dims = (
            impl.kv_lora_rank,
            impl.qk_rope_head_dim,
            impl.qk_nope_head_dim,
            impl.v_head_dim,
        )
        if dims != _SUPPORTED_MLA_DIMS:
            raise ValueError(
                f"{option}: layer {name} has MLA dims (kv_lora_rank, "
                f"qk_rope_head_dim, qk_nope_head_dim, v_head_dim) = {dims}; "
                f"supported: {_SUPPORTED_MLA_DIMS}."
            )
        k_scale = getattr(layer, "_k_scale", None)
        if not isinstance(k_scale, torch.Tensor) or k_scale.numel() != 1:
            raise ValueError(f"{option}: layer {name} has no scalar k_scale.")
        k_scale_value = float(k_scale.item())
        if k_scale_value != 1.0:
            fallback_reasons.append(f"{name}: k_scale is {k_scale_value}")
            continue
        weight = getattr(layer.kv_b_proj, "weight", None)
        input_dtype = _get_kv_b_proj_input_dtype(layer.kv_b_proj, True)
        if (
            input_dtype != torch.bfloat16
            or weight is None
            or weight.dtype != torch.bfloat16
        ):
            fallback_reasons.append(
                f"{name}: kv_b_proj takes {input_dtype} input with "
                f"{None if weight is None else weight.dtype} weight, not bf16"
            )
    if fallback_reasons:
        logger.info(
            "%s: MLA group %s keeps the default chunked-context path (%s).",
            option,
            group,
            "; ".join(fallback_reasons[:3])
            + (
                f"; and {len(fallback_reasons) - 3} more"
                if len(fallback_reasons) > 3
                else ""
            ),
        )
        return False
    return True


def set_fp8_dcp_context_state(
    impl: object, k_scale: torch.Tensor
) -> Fp8DCPContextLayerState:
    """Mark an MLA impl as set up for the fp8 DCP context path."""
    state = Fp8DCPContextLayerState(
        k_scale=k_scale, k_scale_version=_tensor_version(k_scale)
    )
    setattr(impl, _LAYER_STATE_ATTR, state)
    return state


def prepare_fp8_dcp_context(
    builder: "MLACommonMetadataBuilder", layer_names: list[str]
) -> None:
    """Set up every layer of a group that uses the fp8 DCP context path, and
    compile its kernel before serving starts."""
    forward_context = builder.compilation_config.static_forward_context
    for name in layer_names:
        layer = forward_context[name]
        set_fp8_dcp_context_state(layer.impl, layer._k_scale)

    workspace = builder.chunked_prefill_workspace
    device = workspace.device
    rows = torch.zeros((8, workspace.shape[1]), dtype=FP8_DTYPE, device=device)
    row_map = torch.arange(8, dtype=torch.int32, device=device)
    gather_fp8_context_rows(rows, row_map, builder.mla_dims.kv_lora_rank)
    torch.accelerator.synchronize()
    logger.info_once(
        "Using the fp8 DCP chunked-context path for Kimi-K3 MLA prefill "
        "(%d layers in group %s).",
        len(layer_names),
        _group_name(layer_names),
    )


def get_fp8_dcp_context_state(
    impl: object, k_scale: torch.Tensor
) -> Fp8DCPContextLayerState:
    """State of a layer prepared by ``prepare_fp8_dcp_context``.

    The path copies the stored fp8 bytes, which equals the default path only
    while ``k_scale`` is 1.0. That was checked when the group was set up; if the
    layer is later handed another k_scale tensor, or the same tensor after an
    in-place write, the value is checked again once.
    """
    state = getattr(impl, _LAYER_STATE_ATTR, None)
    if state is None:
        raise RuntimeError(
            "An fp8 chunked-context workspace reached an MLA layer that was "
            "not set up for the fp8 DCP context path."
        )
    if (
        k_scale is not state.k_scale
        or _tensor_version(k_scale) != state.k_scale_version
    ):
        if k_scale.numel() != 1 or float(k_scale.item()) != 1.0:
            raise RuntimeError(
                "VLLM_KIMI_K3_MLA_FP8_DCP_CONTEXT requires k_scale == 1.0, but "
                "the layer's k_scale changed after the metadata builder was "
                "created."
            )
        state.k_scale = k_scale
        state.k_scale_version = _tensor_version(k_scale)
    return state


def _row_map_segments(
    chunk: "MLACommonPrefillMetadata.ContextChunk", dcp_world_size: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Row map of one DCP context chunk as (offset, count) segments.

    ``reorg_kvcache`` walks the chunk's requests in order and, within a
    request, the ranks in order; from rank ``r`` it takes
    ``min(max(0, local_context_len - local_start), padded_local_seq_len)``
    rows starting at ``r * num_local_context_tokens`` plus the request's
    offset in that rank's slot. Each non-empty (request, rank) run is one
    segment; output row ``i`` of a segment reads gathered row ``i + offset``.
    """
    assert chunk.padded_local_seq_lens is not None
    assert chunk.local_context_lens_allranks is not None
    assert chunk.local_starts is not None
    slot_rows = chunk.num_local_context_tokens
    padded = np.asarray(chunk.padded_local_seq_lens, dtype=np.int64)
    local_lens = np.asarray(chunk.local_context_lens_allranks, dtype=np.int64)
    local_starts = np.asarray(chunk.local_starts, dtype=np.int64)
    assert local_lens.shape == (padded.shape[0], dcp_world_size)
    assert local_starts.shape == padded.shape

    counts = np.minimum(
        np.maximum(local_lens - local_starts[:, None], 0), padded[:, None]
    )
    request_offsets = np.cumsum(padded) - padded
    first_rows = (
        np.arange(dcp_world_size, dtype=np.int64)[None, :] * slot_rows
        + request_offsets[:, None]
    )
    # Same invariants as reorg_kvcache.
    assert int(counts.sum()) == chunk.num_context_tokens
    assert int(counts.sum(axis=1).max(initial=0)) == chunk.max_seq_len

    counts = counts.reshape(-1)
    first_rows = first_rows.reshape(-1)
    keep = counts > 0
    counts = counts[keep]
    first_rows = first_rows[keep]
    total = int(counts.sum())
    if total:
        assert int(first_rows.min()) >= 0
        assert int((first_rows + counts).max()) <= dcp_world_size * slot_rows
    out_starts = np.cumsum(counts) - counts
    return first_rows - out_starts, counts, total


def build_context_row_maps(
    chunks: "list[MLACommonPrefillMetadata.ContextChunk]",
    dcp_world_size: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """int32 row map per context chunk: for every reorganized context row, the
    row of the chunk's all-gathered buffer it comes from (the index sequence
    ``reorg_kvcache`` concatenates).

    The per-chunk segment tables are copied to the device in one transfer and
    expanded there, so the host work is per (request, rank), not per row, and
    nothing synchronizes.
    """
    segments = [_row_map_segments(chunk, dcp_world_size) for chunk in chunks]
    if not segments:
        return []
    offsets = np.concatenate([seg[0] for seg in segments])
    counts = np.concatenate([seg[1] for seg in segments])
    table = async_tensor_h2d(
        np.concatenate([offsets, counts]), device=device, dtype=torch.int64
    )
    offsets_dev = table[: offsets.shape[0]]
    counts_dev = table[offsets.shape[0] :]
    row_maps: list[torch.Tensor] = []
    start = 0
    for seg_offsets, _, total in segments:
        end = start + seg_offsets.shape[0]
        if total == 0:
            row_maps.append(torch.empty(0, dtype=torch.int32, device=device))
        else:
            rows = torch.repeat_interleave(
                offsets_dev[start:end], counts_dev[start:end], output_size=total
            )
            rows += torch.arange(total, dtype=torch.int64, device=device)
            row_maps.append(rows.to(torch.int32))
        start = end
    return row_maps


@triton.jit(do_not_specialize=["num_rows"])
def _gather_fp8_context_rows_kernel(
    gathered_ptr,
    row_map_ptr,
    latent_ptr,
    k_pe_ptr,
    num_rows,
    LATENT_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    # Output row i takes gathered row row_map[i]: its first LATENT_DIM entries
    # become the bf16 latent, its last ROPE_DIM fp8 entries are copied as is.
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    mask = rows < num_rows
    src_rows = tl.load(row_map_ptr + rows, mask=mask, other=0).to(tl.int64)
    dst_rows = rows.to(tl.int64)
    src = gathered_ptr + src_rows[:, None] * (LATENT_DIM + ROPE_DIM)

    latent_cols = tl.arange(0, LATENT_DIM)
    latent = tl.load(src + latent_cols[None, :], mask=mask[:, None], other=0.0)
    tl.store(
        latent_ptr + dst_rows[:, None] * LATENT_DIM + latent_cols[None, :],
        latent.to(tl.bfloat16),
        mask=mask[:, None],
    )

    rope_cols = tl.arange(0, ROPE_DIM)
    k_pe = tl.load(src + LATENT_DIM + rope_cols[None, :], mask=mask[:, None], other=0.0)
    tl.store(
        k_pe_ptr + dst_rows[:, None] * ROPE_DIM + rope_cols[None, :],
        k_pe,
        mask=mask[:, None],
    )


def gather_fp8_context_rows(
    gathered: torch.Tensor, row_map: torch.Tensor, kv_lora_rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reorganize all-gathered fp8 context rows through ``row_map``.

    Args:
        gathered: Contiguous fp8 rows ``[N, kv_lora_rank + rope_dim]``.
        row_map: int32 ``[C]``, the source row of every output row.
        kv_lora_rank: Width of the latent part of a row.

    Returns:
        ``(latent, k_pe)``: contiguous bf16 ``[C, 1, kv_lora_rank]`` (exact,
        every e4m3 value is a bf16 value) and fp8 ``[C, 1, rope_dim]`` holding
        the gathered bytes.
    """
    assert gathered.dtype == FP8_DTYPE and gathered.dim() == 2
    assert gathered.is_contiguous()
    assert row_map.dtype == torch.int32 and row_map.dim() == 1
    assert row_map.device == gathered.device
    rope_dim = gathered.shape[1] - kv_lora_rank
    num_rows = row_map.shape[0]
    device = gathered.device
    latent = torch.empty(
        (num_rows, 1, kv_lora_rank), dtype=torch.bfloat16, device=device
    )
    k_pe = torch.empty((num_rows, 1, rope_dim), dtype=FP8_DTYPE, device=device)
    if num_rows:
        grid = (triton.cdiv(num_rows, _GATHER_BLOCK_ROWS),)
        _gather_fp8_context_rows_kernel[grid](
            gathered,
            row_map,
            latent,
            k_pe,
            num_rows,
            LATENT_DIM=kv_lora_rank,
            ROPE_DIM=rope_dim,
            BLOCK_ROWS=_GATHER_BLOCK_ROWS,
            num_warps=_GATHER_NUM_WARPS,
        )
    return latent, k_pe

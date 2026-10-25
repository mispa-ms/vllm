# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TokenSpeed CuTe DSL MLA decode backend (Blackwell, FP8 KV cache only)."""

import functools
import types
from typing import TYPE_CHECKING, ClassVar

import torch

import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    QueryLenSupport,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
    MultipleOf,
)
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

# Workspace upper bound for tokenspeed_mla_decode (per-device, lazy):
#   num_sms * num_heads * MAX_Q_LEN * (kv_lora_rank + 1) * sizeof(float32)
# Matches the kernel's `get_workspace_size` formula. MAX_Q_LEN=8 covers up to
# EAGLE3 / MTP-2 spec decoding query lengths; larger q_len fails the kernel's
# own buffer check.
_TOKENSPEED_MAX_Q_LEN = 8


def _get_workspace(
    device: torch.device,
    num_heads: int,
    kv_lora_rank: int,
    min_split_kv: int,
    max_decode_tokens: int,
) -> torch.Tensor:
    from tokenspeed_mla import get_num_sm

    # TokenSpeed reduction scratch: B * H * Q * split_kv * (D + 1) * 4 bytes,
    # using the effective query layout and FP32 partial outputs plus LSE.
    # Automatic splitting lowers split_kv as the batch grows; split_kv=1
    # needs no reduction scratch, so an SM-based bound covers all batch sizes.
    needed = (
        get_num_sm(device) * num_heads * _TOKENSPEED_MAX_Q_LEN * (kv_lora_rank + 1) * 4
    )
    if min_split_kv > 1:
        # A forced floor breaks that bound: reserve for the maximum batch.
        # Packed queries pad head rows to 128; reserve one tile per query token.
        needed = max(
            needed,
            max_decode_tokens
            * max(num_heads, 128)
            * min_split_kv
            * (kv_lora_rank + 1)
            * 4,
        )
        logger.info_once(
            "TokenSpeed MLA decode workspace: %.2f GiB", needed / (1 << 30)
        )
    return current_workspace_manager().get_simultaneous(((needed,), torch.int8))[0]


def _check_fp16_partials_supported() -> None:
    from tokenspeed_mla import mla_decode

    if not hasattr(mla_decode, "_FP16_PARTIALS"):
        raise NotImplementedError(
            "VLLM_TOKENSPEED_MLA_TARGET_FP16_PARTIALS needs a tokenspeed_mla "
            "release that selects fp16 split-KV partials through "
            "tokenspeed_mla.mla_decode._FP16_PARTIALS."
        )
    if mla_decode._FP16_PARTIALS:
        raise ValueError(
            "VLLM_TOKENSPEED_MLA_TARGET_FP16_PARTIALS keeps draft-model layers on "
            "fp32 split-KV partials, but TOKENSPEED_MLA_FP16_PARTIALS=1 already "
            "enables fp16 partials for every layer. Unset one of them."
        )


def _check_lse_no_zero_fill_supported() -> None:
    from tokenspeed_mla import mla_decode

    names = ("torch", "_get_split_kv_and_workspace_size", "tokenspeed_mla_decode")
    if any(name not in vars(mla_decode) for name in names) or (
        mla_decode.tokenspeed_mla_decode.__globals__ is not vars(mla_decode)
    ):
        raise NotImplementedError(
            "VLLM_TOKENSPEED_MLA_TARGET_LSE_NO_ZERO_FILL needs a tokenspeed_mla "
            "whose mla_decode allocates the LSE with its module-level torch and "
            "sizes the split-KV workspace with _get_split_kv_and_workspace_size."
        )


def _use_lse_no_zero_fill(fp16_partials_for_target: bool) -> bool:
    """Whether VLLM_TOKENSPEED_MLA_TARGET_LSE_NO_ZERO_FILL applies."""
    if not envs.VLLM_TOKENSPEED_MLA_TARGET_LSE_NO_ZERO_FILL:
        return False
    if not fp16_partials_for_target:
        raise ValueError(
            "VLLM_TOKENSPEED_MLA_TARGET_LSE_NO_ZERO_FILL applies to the "
            "fp16-partials target-layer path; it requires "
            "VLLM_TOKENSPEED_MLA_TARGET_FP16_PARTIALS=1."
        )
    _check_lse_no_zero_fill_supported()
    return True


class _LseNoZeroFillTorch(types.ModuleType):
    """``torch`` as seen by tokenspeed_mla.mla_decode during one decode call.

    Everything is the real torch module except ``zeros``: the call's float32
    [B, q_len, H] LSE buffer is allocated with ``torch.empty`` when the
    split-KV reduction kernel runs (non-zero workspace), because that kernel
    writes every LSE element. Without the reduction the single kernel may skip
    empty rows, so the buffer stays zero-filled.
    """

    def __init__(self, real_torch: types.ModuleType):
        super().__init__("torch")
        self.__dict__["_real_torch"] = real_torch
        self.__dict__["_reduction"] = None
        self.__dict__["_lse_allocated"] = False

    def __getattr__(self, name: str):
        return getattr(self.__dict__["_real_torch"], name)

    def record_workspace_size(self, workspace_size: int) -> None:
        self.__dict__["_reduction"] = workspace_size != 0

    def zeros(self, *args, **kwargs):
        real_torch = self.__dict__["_real_torch"]
        reduction = self.__dict__["_reduction"]
        if (
            reduction is not None
            and not self.__dict__["_lse_allocated"]
            and len(args) == 1
            and isinstance(args[0], tuple)
            and len(args[0]) == 3
            and set(kwargs) == {"dtype", "device"}
            and kwargs["dtype"] is real_torch.float32
        ):
            self.__dict__["_lse_allocated"] = True
            if reduction:
                return real_torch.empty(*args, **kwargs)
        return real_torch.zeros(*args, **kwargs)


def _tokenspeed_mla_decode_fp16_partials(lse_no_zero_fill: bool = False, **kwargs):
    """``tokenspeed_mla_decode`` with fp16 split-KV partial outputs.

    tokenspeed_mla selects the partial dtype with a module-level switch that
    ``tokenspeed_mla_decode`` reads at call time (part of its kernel cache
    key), so the switch is set for the duration of this call only.

    With ``lse_no_zero_fill`` the LSE buffer of a call that runs the split-KV
    reduction is not zero-filled (see _LseNoZeroFillTorch): mla_decode's
    ``torch`` global and its workspace sizing are swapped for this call only.
    """
    from tokenspeed_mla import mla_decode

    previous = mla_decode._FP16_PARTIALS
    mla_decode._FP16_PARTIALS = True
    if not lse_no_zero_fill:
        try:
            return mla_decode.tokenspeed_mla_decode(**kwargs)
        finally:
            mla_decode._FP16_PARTIALS = previous

    real_torch = mla_decode.torch
    real_split = mla_decode._get_split_kv_and_workspace_size
    proxy = _LseNoZeroFillTorch(real_torch)

    @functools.wraps(real_split)
    def split(*args, **split_kwargs):
        split_kv, workspace_size = real_split(*args, **split_kwargs)
        proxy.record_workspace_size(workspace_size)
        return split_kv, workspace_size

    mla_decode.torch = proxy
    mla_decode._get_split_kv_and_workspace_size = split
    try:
        return mla_decode.tokenspeed_mla_decode(**kwargs)
    finally:
        mla_decode._FP16_PARTIALS = previous
        mla_decode.torch = real_torch
        mla_decode._get_split_kv_and_workspace_size = real_split


class TokenspeedMLAMetadataBuilder(MLACommonMetadataBuilder[MLACommonMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.UNIFORM
    # The kernel accepts an explicit causal mask, so a non-causal DSpark
    # block can remain fused instead of being flattened to single tokens.
    supports_non_causal_multi_token_decode: ClassVar[bool] = True
    supports_non_causal_multi_token_dcp: ClassVar[bool] = True

    def __init__(
        self,
        kv_cache_spec: "AttentionSpec",
        layer_names: list[str],
        vllm_config: "VllmConfig",
        device: torch.device,
    ) -> None:
        super().__init__(
            kv_cache_spec,
            layer_names,
            vllm_config,
            device,
            MLACommonMetadata,
            supports_dcp_with_varlen=True,
        )


class TokenspeedMLABackend(MLACommonBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "fp8",
        "fp8_e4m3",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        return [32, 64]

    @staticmethod
    def get_name() -> str:
        return "TOKENSPEED_MLA"

    @staticmethod
    def get_impl_cls() -> type["TokenspeedMLAImpl"]:
        return TokenspeedMLAImpl

    @staticmethod
    def get_builder_cls() -> type["TokenspeedMLAMetadataBuilder"]:
        return TokenspeedMLAMetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 10

    @classmethod
    def supports_non_causal(cls) -> bool:
        return True

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        # Surface a clear install hint up front rather than letting a raw
        # ModuleNotFoundError fire deep inside `forward_mqa` at first request.
        try:
            import tokenspeed_mla  # noqa: F401
        except ImportError:
            return (
                "tokenspeed_mla package is not installed. "
                "Install it with: `uv pip install tokenspeed-mla`"
            )

        # tokenspeed_mla CuTe DSL kernel is shape-specialized for DeepSeek R1
        # MLA dimensions (qk_nope=128, qk_rope=64, v=128). Reject anything else.
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        if vllm_config.model_config is not None:
            hf_text_config = vllm_config.model_config.hf_text_config
            qk_nope_head_dim = getattr(hf_text_config, "qk_nope_head_dim", 0)
            qk_rope_head_dim = getattr(hf_text_config, "qk_rope_head_dim", 0)
            v_head_dim = getattr(hf_text_config, "v_head_dim", 0)
            if qk_nope_head_dim != 128 or qk_rope_head_dim != 64 or v_head_dim != 128:
                return (
                    "tokenspeed_mla requires DeepSeek R1 MLA dimensions "
                    "(qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128), "
                    f"got ({qk_nope_head_dim}, {qk_rope_head_dim}, {v_head_dim})"
                )
        return None


class TokenspeedMLAImpl(MLACommonImpl[MLACommonMetadata]):
    can_return_lse_for_decode: bool = True
    supports_dcp: bool = True
    supports_mtp_with_cp_non_trivial_interleave_size: bool = True
    # tokenspeed_mla_decode returns LSE in log2 units; its own DCP test merges
    # partial outputs with exp2(lse).
    lse_base_on_e: bool = False
    # VLLM_TOKENSPEED_MLA_TARGET_FP16_PARTIALS: fp16 split-KV partials on
    # target-model layers (draft-model layers keep fp32).
    fp16_partials_for_target: bool = False
    # VLLM_TOKENSPEED_MLA_TARGET_LSE_NO_ZERO_FILL: skip zero-filling the LSE
    # buffer of target-layer calls that run the split-KV reduction.
    lse_no_zero_fill_for_target: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        **mla_args,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **mla_args,
        )

        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "TokenspeedMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "TokenspeedMLAImpl"
            )

        if not is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "TokenspeedMLAImpl requires an FP8 KV cache "
                "(--kv-cache-dtype fp8 or fp8_e4m3); "
                f"got kv_cache_dtype={self.kv_cache_dtype!r}."
            )

        config = get_current_vllm_config()
        self._min_split_kv = config.attention_config.tokenspeed_mla_min_split_kv
        self._max_decode_tokens = 0
        if self._min_split_kv > 1:
            spec = config.speculative_config
            query_len = 1 + (spec.num_speculative_tokens if spec else 0)
            self._max_decode_tokens = max(
                min(
                    config.scheduler_config.max_num_batched_tokens,
                    config.scheduler_config.max_num_seqs * query_len,
                ),
                config.compilation_config.max_cudagraph_capture_size or 0,
            )

        if envs.VLLM_TOKENSPEED_MLA_TARGET_FP16_PARTIALS:
            _check_fp16_partials_supported()
            logger.info_once(
                "TokenSpeed MLA decode uses fp16 split-KV partials on target-model "
                "layers (VLLM_TOKENSPEED_MLA_TARGET_FP16_PARTIALS)."
            )
            self.fp16_partials_for_target = True
        if _use_lse_no_zero_fill(self.fp16_partials_for_target):
            self.lse_no_zero_fill_for_target = True

        self.softmax_scale: float | None = None
        self.output_scale: float | None = None
        # NIXL resolves interleaving after model construction; retain the config
        # rather than caching its initial interleave size.
        self._parallel_config = config.parallel_config

        # Pre-JIT BF16 and FP8 prefill kernels here too — decode impl always
        # runs when tokenspeed is selected, prefill backend may not (user can
        # pair with flash_attn / trtllm). Idempotent.
        from tokenspeed_mla import warmup_compile_prefill

        for q_dtype in (torch.bfloat16, torch.float8_e4m3fn):
            warmup_compile_prefill(
                q_dtype=q_dtype,
                d_qk=self.qk_nope_head_dim + self.qk_rope_head_dim,
                d_v=self.v_head_dim,
                enable_pdl=False,
            )

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        from tokenspeed_mla import tokenspeed_mla_decode

        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        if isinstance(q, tuple):
            q_nope, q_pe = q
            q = torch.cat([q_nope, q_pe], dim=-1)

        # supports_quant_query_input=True (set in MLACommonImpl) tells the
        # pipeline to concat+FP8-quantize Q upstream via _decode_concat_quant_fp8_op.
        # The kernel is shape-specialized for FP8 Q + FP8 KV, so anything else
        # here means the upstream quant didn't run and the kernel will produce
        # garbage.
        assert q.dtype == torch.float8_e4m3fn, (
            f"TokenspeedMLAImpl expected FP8 query (supports_quant_query_input=True), "
            f"got {q.dtype}. Pipeline isinstance(q, tuple)={isinstance(q, tuple)}, "
            f"q_scale={layer._q_scale_float}, k_scale={layer._k_scale_float}."
        )

        num_decodes = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens
        block_tables = attn_metadata.decode.block_table
        seq_lens = attn_metadata.decode.seq_lens
        causal_seqs = attn_metadata.decode.dcp_tot_seq_lens

        # tokenspeed_mla_decode expects query shape
        # (num_decodes, q_len_per_request, num_heads, head_dim).
        if num_decode_tokens % num_decodes != 0:
            logger.warning_once(
                """TokenspeedMLAImpl got a query of uneven length.
                This usually indicates an issue in batch reordering
                or incorrect setup in dummy_run."""
            )
            q = q.unsqueeze(1)
        else:
            q = q.view(num_decodes, -1, q.shape[-2], q.shape[-1])

        if self.softmax_scale is None:
            # FP8 KV cache is mandatory for this backend, so q_scale/k_scale
            # always apply. softmax_scale is bmm1; output_scale is bmm2 — both
            # required to recover the correct attention output from the FP8
            # KV cache (V is stored as V_real/k_scale).
            self.softmax_scale = (
                self.scale * layer._q_scale_float * layer._k_scale_float
            )
            self.output_scale = layer._k_scale_float

        workspace = _get_workspace(
            q.device,
            q.shape[-2],
            self.kv_lora_rank,
            self._min_split_kv,
            self._max_decode_tokens,
        )

        # vLLM kv_c_and_k_pe_cache is already (num_blocks, block_size, head_size).
        # tokenspeed_mla_decode wants 3D — pass as-is (no unsqueeze, unlike trtllm).
        return_lse = self.need_to_return_lse_for_decode
        decode = tokenspeed_mla_decode
        # The speculator marks draft-model layers once the drafter is loaded.
        if self.fp16_partials_for_target and not getattr(
            layer, "is_draft_layer", False
        ):
            decode = functools.partial(
                _tokenspeed_mla_decode_fp16_partials,
                lse_no_zero_fill=self.lse_no_zero_fill_for_target,
            )
        kernel_out = decode(
            query=q,
            kv_cache=kv_c_and_k_pe_cache,
            workspace_buffer=workspace,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=attn_metadata.max_seq_len,
            softmax_scale=self.softmax_scale,
            output_scale=self.output_scale,
            enable_pdl=False,
            return_lse=return_lse,
            causal_mask=attn_metadata.causal,
            causal_seqs=causal_seqs if self.dcp_world_size > 1 else None,
            cp_world=self.dcp_world_size,
            cp_rank=self.dcp_rank,
            cp_interleave_size=self._parallel_config.cp_kv_cache_interleave_size,
            min_split_kv=self._min_split_kv,
        )
        if return_lse:
            o, lse = kernel_out
            # DCP ranks with no local KV may return undefined outputs (e.g.,
            # NaNs). The downstream DCP combine path must mask empty
            # shards and ignore their outputs when merging partial attention.
            lse = lse.view(-1, lse.shape[-1])
        else:
            o, lse = kernel_out, None

        o = o.view(-1, o.shape[-2], o.shape[-1])
        return o, lse

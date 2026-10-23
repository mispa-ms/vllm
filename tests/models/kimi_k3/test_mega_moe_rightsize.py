# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MegaMoE symmetric buffer sized for one sequence-parallel shard
(``VLLM_MEGA_MOE_SP_SHARD_BUFFER``): the per-rank token bound, the Kimi K3
wiring, and bitwise-equal outputs across buffer sizes."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe import deep_gemm_mega_moe
from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
    MEGA_MOE_MXFP4,
    DeepGemmMegaMoEExperts,
    MegaMoEActivation,
)
from vllm.models.kimi_k3.nvidia import model as kimi_model
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_supported

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="DeepGEMM MegaMoE requires CUDA"
)

NUM_EXPERTS, TOP_K, HIDDEN, INTER = 8, 2, 512, 256
ACTIVATION = MegaMoEActivation("situ", alpha=2.0, beta=3.0)


@pytest.fixture
def sm100_layout(monkeypatch):
    """Build the SM100 kernel layout on any GPU; kernels are gated per test."""
    monkeypatch.setattr(
        deep_gemm_mega_moe, "check_deep_gemm_mega_moe_supported", lambda *args: None
    )


def _make_experts(max_num_batched_tokens: int, prefix: str, **kwargs):
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=max_num_batched_tokens),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    return DeepGemmMegaMoEExperts(
        vllm_config,
        num_experts=NUM_EXPERTS,
        num_local_experts=NUM_EXPERTS,
        experts_start_idx=0,
        top_k=TOP_K,
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        activation=ACTIVATION,
        weight_format=MEGA_MOE_MXFP4,
        prefix=prefix,
        **kwargs,
    )


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("is_sequence_parallel", [False, True])
@pytest.mark.parametrize(
    "max_num_batched_tokens,tp_size", [(16384, 8), (16385, 8), (32768, 8), (8192, 4)]
)
def test_token_bound(
    sm100_layout,
    monkeypatch,
    enabled,
    is_sequence_parallel,
    max_num_batched_tokens,
    tp_size,
):
    monkeypatch.setenv("VLLM_MEGA_MOE_SP_SHARD_BUFFER", "1" if enabled else "0")
    monkeypatch.setattr(
        deep_gemm_mega_moe, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    experts = _make_experts(
        max_num_batched_tokens,
        "model.layers.0.mlp.experts",
        is_sequence_parallel=is_sequence_parallel,
    )
    if enabled and is_sequence_parallel:
        expected = -(-max_num_batched_tokens // tp_size)
    else:
        expected = max_num_batched_tokens
    assert experts.max_num_tokens == expected


class _ExpertsBuilt(Exception):
    pass


@pytest.mark.parametrize("use_sequence_parallel", [False, True])
def test_kimi_moe_passes_sequence_parallel(
    dist_init, monkeypatch, use_sequence_parallel
):
    """KimiMoE tells its MegaMoE experts whether their input is an SP shard."""
    seen: dict = {}

    def record(vllm_config, **kwargs):
        seen.update(kwargs)
        raise _ExpertsBuilt

    monkeypatch.setattr(kimi_model, "DeepGemmMegaMoEExperts", record)
    monkeypatch.setattr(kimi_model, "get_tensor_model_parallel_world_size", lambda: 8)
    monkeypatch.setattr(
        kimi_model,
        "get_ep_group",
        lambda: SimpleNamespace(world_size=8, rank_in_group=0),
    )
    config = SimpleNamespace(
        hidden_size=256,
        moe_intermediate_size=256,
        num_experts=NUM_EXPERTS * 8,
        num_experts_per_token=TOP_K,
        moe_renormalize=True,
        routed_expert_hidden_size=128,
        latent_moe_use_norm=False,
        routed_scaling_factor=1.0,
        use_grouped_topk=True,
        num_expert_group=1,
        topk_group=1,
        moe_router_activation_func="sigmoid",
        num_shared_experts=None,
        hidden_act="situ",
        activation_situ_beta=2.0,
        activation_situ_linear_beta=3.0,
        rms_norm_eps=1e-6,
    )
    vllm_config = SimpleNamespace(
        kernel_config=SimpleNamespace(moe_backend="deep_gemm_mega_moe"),
        parallel_config=SimpleNamespace(enable_expert_parallel=True),
    )
    with pytest.raises(_ExpertsBuilt):
        kimi_model.KimiMoE(
            config,
            vllm_config,
            prefix="model.layers.1.block_sparse_moe",
            use_sequence_parallel=use_sequence_parallel,
        )
    assert seen["is_sequence_parallel"] is use_sequence_parallel


def _load_random_weights(experts: DeepGemmMegaMoEExperts, seed: int) -> None:
    from vllm.third_party.deep_gemm.utils.math import per_token_cast_to_fp4

    generator = torch.Generator(device="cuda").manual_seed(seed)

    def mxfp4(rows: int, cols: int):
        w = torch.randn(rows, cols, device="cuda", generator=generator) / cols**0.5
        q, sf = per_token_cast_to_fp4(w.bfloat16(), use_ue8m0=True, gran_k=32)
        # Checkpoints store the UE8M0 scale as its exponent byte.
        return q, (sf.view(torch.int32) >> 23).to(torch.uint8)

    for expert_id in range(NUM_EXPERTS):
        w13, s13 = mxfp4(2 * INTER, HIDDEN)
        w2, s2 = mxfp4(HIDDEN, INTER)
        shards = (
            ("w1", "w13", w13[:INTER], s13[:INTER]),
            ("w3", "w13", w13[INTER:], s13[INTER:]),
            ("w2", "w2", w2, s2),
        )
        for shard_id, param, weight, scale in shards:
            tensors = {f"{param}_weight": weight, f"{param}_weight_scale": scale}
            for name, tensor in tensors.items():
                assert experts.weight_loader(
                    getattr(experts, name),
                    tensor,
                    f"experts.{name}",
                    shard_id=shard_id,
                    expert_id=expert_id,
                    return_success=True,
                )


@pytest.mark.skipif(
    not (
        current_platform.is_device_capability_family(100) and is_deep_gemm_supported()
    ),
    reason="DeepGEMM MegaMoE kernels require SM100 and DeepGEMM",
)
def test_outputs_do_not_depend_on_buffer_size(sm100_layout, dist_init, monkeypatch):
    """The same inputs give bitwise-equal outputs from a buffer sized for the
    whole batch and from one sized for a sequence-parallel shard."""
    max_num_batched_tokens, tp_size = 4096, 8
    monkeypatch.setenv("VLLM_MEGA_MOE_SP_SHARD_BUFFER", "1")
    monkeypatch.setattr(
        deep_gemm_mega_moe, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    with torch.device("cuda"):
        full = _make_experts(max_num_batched_tokens, "model.layers.0.full.experts")
        shard = _make_experts(
            max_num_batched_tokens,
            "model.layers.0.shard.experts",
            is_sequence_parallel=True,
        )
    assert (full.max_num_tokens, shard.max_num_tokens) == (4096, 512)
    _load_random_weights(full, seed=0)
    _load_random_weights(shard, seed=0)
    assert full.get_symm_buffer() is not shard.get_symm_buffer()
    assert (
        full.get_symm_buffer().num_max_tokens_per_rank
        > shard.get_symm_buffer().num_max_tokens_per_rank
    )

    generator = torch.Generator(device="cuda").manual_seed(1)
    for num_tokens in (1, 7, 64, 255, 256, 511, 512):
        hidden_states = torch.randn(
            num_tokens, HIDDEN, device="cuda", generator=generator
        ).bfloat16()
        scores = torch.randn(
            num_tokens, NUM_EXPERTS, device="cuda", generator=generator
        )
        topk_weights, topk_ids = torch.topk(scores.softmax(-1), TOP_K, dim=-1)
        expected = full(hidden_states, topk_weights, topk_ids).clone()
        actual = shard(hidden_states, topk_weights, topk_ids).clone()
        assert torch.isfinite(expected.float()).all()
        assert torch.equal(actual, expected), num_tokens
        assert torch.equal(shard(hidden_states, topk_weights, topk_ids), actual)

    # The shard bound is enforced.
    with pytest.raises(ValueError, match="sized for 512"):
        shard(
            torch.zeros(513, HIDDEN, device="cuda", dtype=torch.bfloat16),
            torch.zeros(513, TOP_K, device="cuda"),
            torch.zeros(513, TOP_K, device="cuda", dtype=torch.int64),
        )

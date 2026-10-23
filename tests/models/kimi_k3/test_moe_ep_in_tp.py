# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for expert parallelism inside the TP group.

``make_parallel_config(..., ep_size_in_tp=N)`` splits a TP-only MoE layout into
N expert-parallel groups (enabled for Kimi-K3 by
VLLM_KIMI_K3_MOE_EP_SIZE_IN_TP). These tests pin the rank layout, the expert
placement and weight sharding it produces, and the all-reduce size the Kimi-K3
latent MoE tail uses with it. ``ep_size_in_tp=1`` must give the stock layout.
The KimiMoE tests pin which layers get the setting and which values and
configurations it rejects.
"""

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.fused_allreduce_gemma_rms_norm as fused_ar_norm
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.fused_moe import config as moe_config_module
from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
from vllm.model_executor.layers.fused_moe.expert_map_manager import ExpertMapManager
from vllm.model_executor.layers.fused_moe.layer import make_parallel_config
from vllm.models.kimi_k3.nvidia import latent_moe_runner
from vllm.models.kimi_k3.nvidia.latent_moe_runner import LatentMoERunner
from vllm.platforms import current_platform


@pytest.fixture
def set_tp_rank(monkeypatch: pytest.MonkeyPatch):
    """Pretend to be ``rank`` of the TP group (and of the DP/PCP groups)."""

    def _set(rank: int, group_rank: int = 0) -> None:
        group = SimpleNamespace(rank_in_group=group_rank)
        monkeypatch.setattr(
            moe_config_module, "get_tensor_model_parallel_rank", lambda: rank
        )
        monkeypatch.setattr(moe_config_module, "get_dp_group", lambda: group)
        monkeypatch.setattr(moe_config_module, "get_pcp_group", lambda: group)

    return _set


def _parallel_config(
    expert_parallel: bool = False,
    all2all_backend: str = "allgather_reducescatter",
    eplb: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        enable_expert_parallel=expert_parallel,
        all2all_backend=all2all_backend,
        enable_eplb=eplb,
    )


def _make(
    tp_size: int,
    ep_size_in_tp: int | None,
    dp_size: int = 1,
    pcp_size: int = 1,
    sequence_parallel: bool = False,
    **parallel_config,
) -> FusedMoEParallelConfig:
    kwargs = {} if ep_size_in_tp is None else {"ep_size_in_tp": ep_size_in_tp}
    return make_parallel_config(
        tp_size=tp_size,
        dp_size=dp_size,
        pcp_size=pcp_size,
        is_sequence_parallel=sequence_parallel,
        parallel_config=_parallel_config(**parallel_config),
        **kwargs,
    )


@pytest.mark.parametrize(
    ("tp_size", "ep_size"), [(8, 2), (8, 4), (8, 8), (4, 2), (16, 2)]
)
def test_rank_layout(set_tp_rank, tp_size: int, ep_size: int):
    group_size = tp_size // ep_size
    seen = set()
    for rank in range(tp_size):
        set_tp_rank(rank)
        config = _make(tp_size, ep_size)

        assert config.tp_size == group_size
        assert config.tp_rank == rank % group_size
        assert config.ep_size == ep_size
        assert config.ep_rank == rank // group_size
        assert config.use_ep
        assert (config.dp_size, config.dp_rank) == (1, 0)
        assert (config.pcp_size, config.pcp_rank) == (1, 0)
        assert config.sp_size == 1
        assert not config.enable_eplb
        assert config.all2all_backend == "allgather_reducescatter"
        # Tokens stay replicated: no dispatch/combine kernels.
        assert not config.use_all2all_kernels
        assert not config.use_passthrough_all2all
        seen.add((config.ep_rank, config.tp_rank))
    assert len(seen) == tp_size


@pytest.mark.parametrize("tp_size", [1, 2, 8])
@pytest.mark.parametrize("expert_parallel", [False, True])
def test_ep_size_one_is_stock(set_tp_rank, tp_size: int, expert_parallel: bool):
    for rank in range(tp_size):
        set_tp_rank(rank)
        stock = FusedMoEParallelConfig.make(
            tp_size_=tp_size,
            pcp_size_=1,
            dp_size_=1,
            sp_size_=1,
            vllm_parallel_config=_parallel_config(expert_parallel),
        )
        assert _make(tp_size, None, expert_parallel=expert_parallel) == stock
        assert _make(tp_size, 1, expert_parallel=expert_parallel) == stock


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (dict(expert_parallel=True), "expert parallel"),
        (dict(dp_size=2), "data parallel"),
        (dict(pcp_size=2), "prefill context parallel"),
        (dict(sequence_parallel=True), "sequence parallel"),
        (dict(eplb=True), "EPLB"),
        (dict(all2all_backend="passthrough"), "passthrough"),
    ],
)
def test_rejects_non_tp_layouts(set_tp_rank, kwargs: dict, match: str):
    set_tp_rank(1)
    with pytest.raises(ValueError, match=match):
        _make(8, 2, **kwargs)


@pytest.mark.parametrize(("tp_size", "ep_size"), [(6, 4), (8, 3), (4, 8)])
def test_rejects_ep_size_not_dividing_tp(set_tp_rank, tp_size: int, ep_size: int):
    set_tp_rank(0)
    with pytest.raises(ValueError, match="must divide"):
        _make(tp_size, ep_size)


@pytest.mark.parametrize("placement", ["linear", "round_robin"])
def test_contiguous_expert_placement(set_tp_rank, placement: str):
    num_experts, tp_size, ep_size = 896, 8, 2
    num_local = num_experts // ep_size
    for rank in range(tp_size):
        set_tp_rank(rank)
        config = _make(tp_size, ep_size)
        manager = ExpertMapManager(
            max_num_batched_tokens=64,
            top_k=16,
            global_num_experts=num_experts,
            num_redundant_experts=0,
            num_expert_group=1,
            moe_parallel_config=config,
            placement_strategy=placement,
            enable_eplb=False,
        )
        assert manager.placement_strategy == "linear"
        assert manager.local_num_experts == num_local
        start = config.ep_rank * num_local
        expected = torch.full((num_experts,), -1, dtype=torch.int32)
        expected[start : start + num_local] = torch.arange(num_local)
        assert torch.equal(manager.expert_map.cpu(), expected)


@pytest.mark.parametrize(
    ("config_kwargs", "expected"),
    [
        (dict(tp_size=8, ep_size_in_tp=None), 8),  # plain TP
        (dict(tp_size=8, ep_size_in_tp=None, expert_parallel=True), 1),  # EP
        (dict(tp_size=8, ep_size_in_tp=2), 8),
        (dict(tp_size=8, ep_size_in_tp=4), 8),
        (dict(tp_size=16, ep_size_in_tp=2), 16),
        (dict(tp_size=1, ep_size_in_tp=None), 1),
    ],
)
def test_latent_reduce_size(set_tp_rank, config_kwargs: dict, expected: int):
    set_tp_rank(0)
    runner = _tail_runner(_make(**config_kwargs), hidden_size=16, latent_size=8)
    assert runner._latent_reduce_size == expected


def _tail_runner(
    parallel_config: FusedMoEParallelConfig, hidden_size: int, latent_size: int
) -> LatentMoERunner:
    """A runner carrying only what the latent tail reads."""
    runner = object.__new__(LatentMoERunner)
    attrs = {
        "moe_config": SimpleNamespace(moe_parallel_config=parallel_config),
        "routed_output_transform": SimpleNamespace(
            norm=None,
            up_proj=SimpleNamespace(
                weight=torch.randn(hidden_size, latent_size, dtype=torch.float64)
            ),
        ),
        "_maybe_reduce_final_output": (
            lambda states, trunc_size, output_is_reduced=None: states
        ),
    }
    for name, value in attrs.items():
        object.__setattr__(runner, name, value)
    return runner


@pytest.mark.parametrize("rank", range(8))
@pytest.mark.parametrize("ep_size_in_tp", [None, 2, 4])
def test_column_parallel_tail_uses_whole_tp_group(
    monkeypatch: pytest.MonkeyPatch, set_tp_rank, rank: int, ep_size_in_tp
):
    tp_world, hidden_size, latent_size, num_tokens = 8, 64, 16, 3
    set_tp_rank(rank)
    runner = _tail_runner(_make(tp_world, ep_size_in_tp), hidden_size, latent_size)
    # The latent all-reduce is the identity here; only the column shard matters.
    monkeypatch.setattr(
        latent_moe_runner, "tensor_model_parallel_all_reduce", lambda x: x
    )
    monkeypatch.setattr(
        latent_moe_runner, "get_tensor_model_parallel_rank", lambda: rank
    )

    latent = torch.randn(num_tokens, latent_size, dtype=torch.float64)
    shared = torch.randn(num_tokens, hidden_size, dtype=torch.float64)
    before = shared.clone()
    out = runner._shard_up_proj_tail(latent, shared, None)

    shard = hidden_size // tp_world
    start, end = rank * shard, (rank + 1) * shard
    weight = runner.routed_output_transform.up_proj.weight
    torch.testing.assert_close(
        out[:, start:end], before[:, start:end] + latent @ weight[start:end].T
    )
    assert torch.equal(out[:, :start], before[:, :start])
    assert torch.equal(out[:, end:], before[:, end:])


@pytest.mark.parametrize(
    ("tp_size", "ep_size_in_tp", "expected"),
    [(8, None, 8), (8, 2, 8), (4, 2, 4), (16, 4, 16)],
)
def test_fused_allreduce_norm_world_size(
    monkeypatch: pytest.MonkeyPatch, set_tp_rank, tp_size, ep_size_in_tp, expected
):
    set_tp_rank(0)
    runner = _tail_runner(_make(tp_size, ep_size_in_tp), hidden_size=16, latent_size=8)
    seen: dict = {}

    def can_use(hidden_states, world_size):
        seen["can_use"] = world_size
        return True, 4

    def fused_allreduce_norm(**kwargs):
        seen["world_size"] = kwargs["world_size"]
        kwargs["norm_out"].copy_(kwargs["allreduce_in"])

    monkeypatch.setattr(fused_ar_norm, "_can_use_flashinfer", can_use)
    monkeypatch.setattr(
        fused_ar_norm, "flashinfer_trtllm_fused_allreduce_norm", fused_allreduce_norm
    )
    norm = SimpleNamespace(weight=torch.ones(8), variance_epsilon=1e-6)
    hidden_states = torch.randn(2, 8)

    out = runner.allreduce_norm_latent_out(hidden_states, norm)

    assert seen == {"can_use": expected, "world_size": expected}
    assert torch.equal(out, hidden_states)


def test_env_default_is_plain_tp(monkeypatch: pytest.MonkeyPatch):
    import vllm.envs as envs

    monkeypatch.delenv("VLLM_KIMI_K3_MOE_EP_SIZE_IN_TP", raising=False)
    assert envs.VLLM_KIMI_K3_MOE_EP_SIZE_IN_TP == 1
    monkeypatch.setenv("VLLM_KIMI_K3_MOE_EP_SIZE_IN_TP", "2")
    assert envs.VLLM_KIMI_K3_MOE_EP_SIZE_IN_TP == 2


def _build_experts(rank: int, set_tp_rank, **kwargs) -> torch.nn.Module:
    set_tp_rank(rank)
    vllm_config = VllmConfig()
    vllm_config.compilation_config.static_forward_context = {}
    args = dict(
        num_experts=8,
        top_k=2,
        hidden_size=128,
        intermediate_size=512,
        params_dtype=torch.bfloat16,
        tp_size=4,
        dp_size=1,
        pcp_size=1,
        prefix=f"model.layers.{rank}.mlp.experts",
        ep_size_in_tp=2,
    )
    args.update(kwargs)
    with set_current_vllm_config(vllm_config):
        return FusedMoEFactory(**args)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like platform"
)
@pytest.mark.parametrize("rank", range(4))
def test_fused_moe_factory_shards_experts(dist_init, set_tp_rank, rank: int):
    runner = _build_experts(rank, set_tp_rank)
    moe_config = runner.moe_config
    ep_rank, tp_rank = rank // 2, rank % 2
    assert (moe_config.tp_size, moe_config.tp_rank) == (2, tp_rank)
    assert (moe_config.ep_size, moe_config.ep_rank) == (2, ep_rank)
    assert moe_config.num_local_experts == 4
    assert moe_config.intermediate_size_per_partition == 256

    experts = runner.routed_experts
    torch.manual_seed(0)
    full_w1 = torch.randn(512, 128, dtype=torch.bfloat16)
    full_w2 = torch.randn(128, 512, dtype=torch.bfloat16)
    for expert_id in range(8):
        loaded = [
            experts.weight_loader(
                experts.w13_weight,
                full_w1,
                "experts.w13_weight",
                shard_id="w1",
                expert_id=expert_id,
                return_success=True,
            ),
            experts.weight_loader(
                experts.w2_weight,
                full_w2,
                "experts.w2_weight",
                shard_id="w2",
                expert_id=expert_id,
                return_success=True,
            ),
        ]
        local = expert_id // 4 == ep_rank
        assert loaded == [local, local]
        if local:
            local_id = expert_id % 4
            rows = slice(tp_rank * 256, (tp_rank + 1) * 256)
            assert torch.equal(experts.w13_weight[local_id, :256], full_w1[rows])
            assert torch.equal(experts.w2_weight[local_id], full_w2[:, rows])


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like platform"
)
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (dict(enable_eplb=True), "EPLB"),
        (dict(num_experts=7), "divisible"),
    ],
)
def test_fused_moe_factory_rejects(dist_init, set_tp_rank, kwargs, match):
    with pytest.raises(ValueError, match=match):
        _build_experts(0, set_tp_rank, **kwargs)


# KimiMoE wiring: which layers get VLLM_KIMI_K3_MOE_EP_SIZE_IN_TP and which
# values and configurations it rejects. The routed experts are not built: the
# arguments KimiMoE passes to FusedMoEFactory are recorded instead.

_NUM_TARGET_LAYERS = 4

requires_cuda_alike = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like platform"
)


class _ExpertsBuilt(Exception):
    pass


def _kimi_moe_config(**overrides) -> SimpleNamespace:
    config = dict(
        hidden_size=256,
        moe_intermediate_size=2048,
        num_experts=16,
        num_experts_per_token=2,
        moe_renormalize=True,
        routed_expert_hidden_size=128,
        latent_moe_use_norm=False,
        routed_scaling_factor=1.0,
        use_grouped_topk=True,
        num_expert_group=1,
        topk_group=1,
        moe_router_activation_func="sigmoid",
        num_shared_experts=None,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        num_hidden_layers=_NUM_TARGET_LAYERS,
    )
    config.update(overrides)
    return SimpleNamespace(**config)


def _kimi_moe_experts_args(
    monkeypatch: pytest.MonkeyPatch,
    layer_idx: int,
    decode_config=None,
    **config_overrides,
) -> dict:
    """Construct a KimiMoE on a TP=8 deployment without expert parallelism
    and return the keyword arguments it passes to FusedMoEFactory."""
    from vllm.models.kimi_k3.nvidia import model as kimi_model

    seen: dict = {}

    def record_experts(**kwargs):
        seen.update(kwargs)
        raise _ExpertsBuilt

    monkeypatch.setattr(kimi_model, "FusedMoEFactory", record_experts)
    monkeypatch.setattr(kimi_model, "get_tensor_model_parallel_world_size", lambda: 8)
    if decode_config is not None:
        monkeypatch.setattr(
            kimi_model, "specialized_decode_config", lambda _: decode_config
        )
    vllm_config = SimpleNamespace(
        kernel_config=SimpleNamespace(moe_backend="auto"),
        parallel_config=SimpleNamespace(enable_expert_parallel=False),
        additional_config={},
    )
    with pytest.raises(_ExpertsBuilt):
        kimi_model.KimiMoE(
            _kimi_moe_config(**config_overrides),
            vllm_config,
            prefix=f"model.layers.{layer_idx}.block_sparse_moe",
            layer_idx=layer_idx,
        )
    return seen


@requires_cuda_alike
@pytest.mark.parametrize("env_value", [None, "1"])
def test_kimi_moe_default_is_plain_tp(
    dist_init, monkeypatch: pytest.MonkeyPatch, env_value
):
    if env_value is None:
        monkeypatch.delenv("VLLM_KIMI_K3_MOE_EP_SIZE_IN_TP", raising=False)
    else:
        monkeypatch.setenv("VLLM_KIMI_K3_MOE_EP_SIZE_IN_TP", env_value)
    assert _kimi_moe_experts_args(monkeypatch, layer_idx=0)["ep_size_in_tp"] == 1


@requires_cuda_alike
@pytest.mark.parametrize("ep_size", [2, 4])
@pytest.mark.parametrize("layer_idx", [0, _NUM_TARGET_LAYERS - 1])
def test_kimi_moe_target_layers_get_ep_config(
    dist_init, monkeypatch: pytest.MonkeyPatch, set_tp_rank, ep_size, layer_idx
):
    monkeypatch.setenv("VLLM_KIMI_K3_MOE_EP_SIZE_IN_TP", str(ep_size))
    args = _kimi_moe_experts_args(monkeypatch, layer_idx)
    assert args["ep_size_in_tp"] == ep_size
    assert args["is_sequence_parallel"] is False
    # These arguments give every rank of the TP group the EP-in-TP layout.
    group_size = 8 // ep_size
    for rank in range(8):
        set_tp_rank(rank)
        config = _make(
            8,
            args["ep_size_in_tp"],
            sequence_parallel=args["is_sequence_parallel"],
        )
        assert (config.ep_size, config.ep_rank) == (ep_size, rank // group_size)
        assert (config.tp_size, config.tp_rank) == (group_size, rank % group_size)


@requires_cuda_alike
@pytest.mark.parametrize("layer_idx", [_NUM_TARGET_LAYERS, _NUM_TARGET_LAYERS + 1])
def test_kimi_moe_draft_layers_keep_plain_tp(
    dist_init, monkeypatch: pytest.MonkeyPatch, layer_idx
):
    """MTP draft layers come after the target layers and keep plain TP."""
    monkeypatch.setenv("VLLM_KIMI_K3_MOE_EP_SIZE_IN_TP", "2")
    assert _kimi_moe_experts_args(monkeypatch, layer_idx)["ep_size_in_tp"] == 1


@requires_cuda_alike
@pytest.mark.parametrize("env_value", ["0", "-2"])
@pytest.mark.parametrize("layer_idx", [0, _NUM_TARGET_LAYERS])
def test_kimi_moe_rejects_ep_size_below_one(
    dist_init, monkeypatch: pytest.MonkeyPatch, env_value, layer_idx
):
    monkeypatch.setenv("VLLM_KIMI_K3_MOE_EP_SIZE_IN_TP", env_value)
    with pytest.raises(ValueError, match="must be >= 1"):
        _kimi_moe_experts_args(monkeypatch, layer_idx)


@requires_cuda_alike
@pytest.mark.parametrize(
    ("config_overrides", "decode_config"),
    [
        (dict(routed_expert_hidden_size=None), None),  # no latent MoE
        (dict(moe_intermediate_size=1024), None),  # padded per-rank shard
        ({}, object()),  # specialized decode backend
    ],
)
def test_kimi_moe_ep_requirements(
    dist_init, monkeypatch: pytest.MonkeyPatch, config_overrides, decode_config
):
    monkeypatch.setenv("VLLM_KIMI_K3_MOE_EP_SIZE_IN_TP", "2")
    with pytest.raises(NotImplementedError, match="requires latent MoE"):
        _kimi_moe_experts_args(monkeypatch, 0, decode_config, **config_overrides)
    # Draft layers keep plain TP, so the requirements do not apply to them.
    args = _kimi_moe_experts_args(
        monkeypatch, _NUM_TARGET_LAYERS, decode_config, **config_overrides
    )
    assert args["ep_size_in_tp"] == 1

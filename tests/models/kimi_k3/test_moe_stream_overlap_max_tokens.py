# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for VLLM_KIMI_K3_MOE_STREAM_OVERLAP_MAX_TOKENS.

The option raises the token bound of the two aux-stream overlaps in front of
the Kimi-K3 latent MoE experts: the router gate || routed down projection
(KimiMoE) and the shared experts (SharedExperts). The latent tail keeps
selecting its tier with VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD. The
decision and validation tests need no GPU; the CUDA tests check that the
moved work gives bitwise-identical results on the aux stream, eagerly and in
a CUDA graph.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.runner import shared_experts as se_mod
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
    SharedExpertsOrder,
)
from vllm.models.kimi_k3.nvidia import model as kimi_model
from vllm.models.kimi_k3.nvidia.latent_moe_runner import (
    LatentMoERunner,
    LatentTailTier,
)
from vllm.platforms import current_platform

ENV_NAME = "VLLM_KIMI_K3_MOE_STREAM_OVERLAP_MAX_TOKENS"
STOCK_TOKENS = 256
MAX_TOKENS = 640
TOKEN_COUNTS = range(1, 705)

requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="requires a CUDA GPU"
)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch):
    for name in (
        ENV_NAME,
        "VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD",
        "VLLM_DISABLE_SHARED_EXPERTS_STREAM",
    ):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


@pytest.fixture
def cuda_like(monkeypatch: pytest.MonkeyPatch):
    """Let the shared-experts decision see a CUDA-like platform."""
    monkeypatch.setattr(
        se_mod, "current_platform", SimpleNamespace(is_cuda_alike=lambda: True)
    )


def _moe_config(**parallel) -> SimpleNamespace:
    config = dict(
        enable_eplb=False,
        all2all_backend="allgather_reducescatter",
        use_fi_nvl_two_sided_kernels=False,
    )
    config.update(parallel)
    return SimpleNamespace(
        tp_size=8,
        is_sequence_parallel=False,
        moe_parallel_config=SimpleNamespace(**config),
    )


_KEEP = object()


def _shared_experts(
    layer: torch.nn.Module | None = None,
    stream=_KEEP,
    mk_overlap: bool = False,
    moe_config: SimpleNamespace | None = None,
) -> SharedExperts:
    se = SharedExperts(
        layer if layer is not None else torch.nn.Identity(),
        moe_config=moe_config if moe_config is not None else _moe_config(),
        enable_dbo=False,
        mk_can_overlap_shared_experts=lambda: mk_overlap,
    )
    if stream is not _KEEP:
        se._stream = stream
    return se


def _order(se: SharedExperts, num_tokens: int) -> SharedExpertsOrder:
    return se._determine_shared_experts_order(torch.empty(num_tokens, 1))


def _expected_order(num_tokens: int, max_tokens: int) -> SharedExpertsOrder:
    if num_tokens <= max_tokens:
        return SharedExpertsOrder.MULTI_STREAM_OVERLAPPED
    return SharedExpertsOrder.NO_OVERLAP


# ---------------------------------------------------------------- decisions


def test_shared_experts_default_bound_is_stock(clean_env, cuda_like):
    se = _shared_experts(stream=object())
    assert se.aux_stream_max_tokens is None
    for m in TOKEN_COUNTS:
        assert _order(se, m) == _expected_order(m, STOCK_TOKENS), m


def test_shared_experts_default_follows_env_threshold(clean_env, cuda_like):
    clean_env.setenv("VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD", "128")
    se = _shared_experts(stream=object())
    for m in (1, 128, 129, 256):
        assert _order(se, m) == _expected_order(m, 128), m


def test_shared_experts_widened_bound(clean_env, cuda_like):
    se = _shared_experts(stream=object())
    se.aux_stream_max_tokens = MAX_TOKENS
    for m in TOKEN_COUNTS:
        assert _order(se, m) == _expected_order(m, MAX_TOKENS), m


@pytest.mark.parametrize(
    "case",
    ["no_stream", "not_cuda", "mk_overlap", "eplb_unsafe", "sharded_sp"],
)
def test_shared_experts_widened_bound_keeps_other_conditions(
    clean_env, monkeypatch: pytest.MonkeyPatch, case
):
    """A widened bound changes only the token comparison."""
    platform = SimpleNamespace(is_cuda_alike=lambda: case != "not_cuda")
    monkeypatch.setattr(se_mod, "current_platform", platform)
    layer = torch.nn.Identity()
    if case == "sharded_sp":
        layer.shard_sequence_parallel = True
    moe_config = _moe_config(
        enable_eplb=case == "eplb_unsafe", all2all_backend="naive"
    )
    se = _shared_experts(
        layer=layer,
        stream=None if case == "no_stream" else object(),
        mk_overlap=case == "mk_overlap",
        moe_config=moe_config,
    )
    stock = {m: _order(se, m) for m in (1, 256, 257, 640)}
    se.aux_stream_max_tokens = MAX_TOKENS
    for m, order in stock.items():
        assert _order(se, m) == order, (case, m)
        assert order != SharedExpertsOrder.MULTI_STREAM_OVERLAPPED


class _RecordParallel:
    """Stands in for maybe_execute_in_parallel and records the aux stream."""

    def __init__(self) -> None:
        self.streams: list = []

    def __call__(self, fn0, fn1, event0, event1, aux_stream=None):
        self.streams.append(aux_stream)
        return fn0(), fn1()


def _kimi_moe_stub(
    max_tokens: int | None = None,
    *,
    mega_moe: bool = False,
    sequence_parallel: bool = False,
    shared: bool = True,
    shared_stream=True,
    down_proj_stream=True,
) -> kimi_model.KimiMoE:
    """A KimiMoE with only the attributes the overlap code reads."""
    moe = object.__new__(kimi_model.KimiMoE)
    torch.nn.Module.__init__(moe)
    moe.use_mega_moe = mega_moe
    moe.use_sequence_parallel = sequence_parallel
    moe.use_latent_moe = True
    moe.gate = lambda x: (x.sum(-1, keepdim=True), None)
    moe.routed_expert_down_proj = lambda x: (x * 2, None)
    moe._down_proj_stream = object() if down_proj_stream else None
    moe._down_proj_events = (None, None)
    moe._down_proj_stream_max_tokens = STOCK_TOKENS
    runner = object.__new__(LatentMoERunner)
    torch.nn.Module.__init__(runner)
    runner._shared_experts = None
    if shared:
        runner._shared_experts = _shared_experts(
            stream=object() if shared_stream else None
        )
    moe.experts = runner
    if max_tokens is not None:
        moe._widen_stream_overlap(max_tokens)
    return moe


def _router_streams(
    monkeypatch: pytest.MonkeyPatch, moe: kimi_model.KimiMoE
) -> dict[int, bool]:
    record = _RecordParallel()
    monkeypatch.setattr(kimi_model, "maybe_execute_in_parallel", record)
    on_aux = {}
    for m in TOKEN_COUNTS:
        moe._maybe_overlap_router_and_down_proj(torch.ones(m, 4))
        on_aux[m] = record.streams[-1] is moe._down_proj_stream
    return on_aux


def test_router_overlap_default_bound_is_stock(clean_env):
    moe = _kimi_moe_stub()
    on_aux = _router_streams(clean_env, moe)
    assert on_aux == {m: m <= STOCK_TOKENS for m in TOKEN_COUNTS}


def test_widened_layer_moves_both_overlaps(clean_env, cuda_like):
    moe = _kimi_moe_stub(MAX_TOKENS)
    assert moe._down_proj_stream_max_tokens == MAX_TOKENS
    se = moe.experts.shared_experts
    assert se.aux_stream_max_tokens == MAX_TOKENS
    on_aux = _router_streams(clean_env, moe)
    assert on_aux == {m: m <= MAX_TOKENS for m in TOKEN_COUNTS}
    for m in TOKEN_COUNTS:
        assert _order(se, m) == _expected_order(m, MAX_TOKENS), m


def test_widened_layer_keeps_the_tail_tier(clean_env):
    """Tier 1 stays bounded by VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD."""
    moe = _kimi_moe_stub(MAX_TOKENS)
    runner = moe.experts
    runner.enable_k3_latent_moe_tail_fusion = False
    runner.routed_output_transform = SimpleNamespace(norm=None, up_proj=None)
    for m in (1, 256, 257, 512, 640, 704):
        out = torch.empty(m, 1)
        tier = runner._select_tail_tier(out, out)
        expected = (
            LatentTailTier.ALLREDUCE_OVERLAP
            if m <= STOCK_TOKENS
            else LatentTailTier.COLUMN_PARALLEL
        )
        assert tier == expected, m


# ---------------------------------------------------------------- validation


@pytest.mark.parametrize("value", [STOCK_TOKENS, 100, -1])
def test_rejects_values_not_above_stock(clean_env, value):
    with pytest.raises(ValueError, match="larger than the stock"):
        _kimi_moe_stub(value)


def test_rejects_values_not_above_env_threshold(clean_env):
    clean_env.setenv("VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD", "512")
    with pytest.raises(ValueError, match="larger than the stock aux-stream bound 512"):
        _kimi_moe_stub(500)
    moe = _kimi_moe_stub(MAX_TOKENS)
    assert moe.experts.shared_experts.aux_stream_max_tokens == MAX_TOKENS


def test_rejects_disabled_shared_experts_stream(clean_env):
    clean_env.setenv("VLLM_DISABLE_SHARED_EXPERTS_STREAM", "1")
    with pytest.raises(ValueError, match="VLLM_DISABLE_SHARED_EXPERTS_STREAM"):
        _kimi_moe_stub(MAX_TOKENS)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(mega_moe=True),
        dict(sequence_parallel=True),
        dict(shared=False),
        dict(shared_stream=False),
        dict(down_proj_stream=False),
    ],
)
def test_rejects_unsupported_layers(clean_env, kwargs):
    with pytest.raises(NotImplementedError, match="requires latent MoE"):
        _kimi_moe_stub(MAX_TOKENS, **kwargs)


def test_env_variable_default(clean_env):
    import vllm.envs as envs

    assert getattr(envs, ENV_NAME) == 0
    clean_env.setenv(ENV_NAME, str(MAX_TOKENS))
    assert getattr(envs, ENV_NAME) == MAX_TOKENS


class _MoEBuilt(Exception):
    pass


requires_cuda_alike = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like platform"
)


@requires_cuda_alike
@pytest.mark.parametrize("env_value", [None, str(MAX_TOKENS)])
def test_kimi_moe_reads_the_variable(
    dist_init, monkeypatch: pytest.MonkeyPatch, env_value
):
    """KimiMoE.__init__ widens the bounds only with the variable set."""
    monkeypatch.delenv("VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD", raising=False)
    monkeypatch.delenv("VLLM_DISABLE_SHARED_EXPERTS_STREAM", raising=False)
    if env_value is None:
        monkeypatch.delenv(ENV_NAME, raising=False)
    else:
        monkeypatch.setenv(ENV_NAME, env_value)

    def build_runner(**kwargs):
        assert kwargs["runner_cls"] is LatentMoERunner
        runner = object.__new__(LatentMoERunner)
        torch.nn.Module.__init__(runner)
        runner._shared_experts = _shared_experts(stream=object())
        return runner

    monkeypatch.setattr(kimi_model, "FusedMoEFactory", build_runner)
    monkeypatch.setattr(kimi_model, "get_tensor_model_parallel_world_size", lambda: 8)
    config = SimpleNamespace(
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
        num_hidden_layers=4,
    )
    vllm_config = SimpleNamespace(
        kernel_config=SimpleNamespace(moe_backend="auto"),
        parallel_config=SimpleNamespace(enable_expert_parallel=False),
        additional_config={},
    )
    moe = kimi_model.KimiMoE(config, vllm_config, prefix="model.layers.0.mlp")
    expected = STOCK_TOKENS if env_value is None else MAX_TOKENS
    assert moe._down_proj_stream_max_tokens == expected
    se = moe.experts.shared_experts
    assert se.aux_stream_max_tokens == (None if env_value is None else MAX_TOKENS)


# ---------------------------------------------------------------- bits (GPU)


class _SharedMLP(torch.nn.Module):
    def __init__(self, hidden: int, inter: int, gen: torch.Generator):
        super().__init__()
        self.w1 = torch.randn(2 * inter, hidden, generator=gen, device="cuda")
        self.w2 = torch.randn(hidden, inter, generator=gen, device="cuda")
        self.w1 = (self.w1 * hidden**-0.5).to(torch.bfloat16)
        self.w2 = (self.w2 * inter**-0.5).to(torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = torch.nn.functional.linear(x, self.w1).chunk(2, dim=-1)
        return torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, self.w2)


def _shared_output(se: SharedExperts, x: torch.Tensor, busy: torch.Tensor):
    """The MoERunner protocol: enqueue, run other work, wait, read."""
    if se.maybe_forward_async(x):
        other = busy @ busy
        se.wait()
        return se.output, other
    other = busy @ busy
    se(x, SharedExpertsOrder.NO_OVERLAP)
    return se.output, other


@requires_cuda
@pytest.mark.parametrize("num_tokens", [257, 384, 512, 640])
@pytest.mark.parametrize("graph", [False, True])
def test_shared_experts_on_aux_stream_bitwise(clean_env, num_tokens, graph):
    gen = torch.Generator(device="cuda").manual_seed(num_tokens)
    layer = _SharedMLP(512, 384, gen)
    x = torch.randn(num_tokens, 512, generator=gen, device="cuda").to(torch.bfloat16)
    busy = torch.randn(256, 256, generator=gen, device="cuda")
    serial = _shared_experts(layer)
    widened = _shared_experts(layer)
    assert serial._stream is not None
    widened.aux_stream_max_tokens = MAX_TOKENS
    assert _order(serial, num_tokens) == SharedExpertsOrder.NO_OVERLAP
    assert _order(widened, num_tokens) == SharedExpertsOrder.MULTI_STREAM_OVERLAPPED
    expected, _ = _shared_output(serial, x, busy)
    if not graph:
        actual, _ = _shared_output(widened, x, busy)
    else:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            _shared_output(widened, x, busy)
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            actual, _ = _shared_output(widened, x, busy)
        g.replay()
    torch.accelerator.synchronize()
    assert torch.equal(actual, expected)


@requires_cuda
@pytest.mark.parametrize("num_tokens", [257, 448, 640])
def test_router_and_down_proj_on_aux_stream_bitwise(clean_env, num_tokens):
    from vllm.utils.torch_utils import aux_stream

    gen = torch.Generator(device="cuda").manual_seed(num_tokens)
    x = torch.randn(num_tokens, 512, generator=gen, device="cuda").to(torch.bfloat16)
    w_gate = torch.randn(64, 512, generator=gen, device="cuda").to(torch.bfloat16)
    w_down = torch.randn(256, 512, generator=gen, device="cuda").to(torch.bfloat16)

    def run(max_tokens: int):
        moe = _kimi_moe_stub()
        moe.gate = lambda h: (torch.mm(h, w_gate.t(), out_dtype=torch.float32), None)
        moe.routed_expert_down_proj = lambda h: (
            torch.nn.functional.linear(h, w_down),
            None,
        )
        moe._down_proj_stream = aux_stream()
        moe._down_proj_events = (torch.cuda.Event(), torch.cuda.Event())
        moe._down_proj_stream_max_tokens = max_tokens
        latent, logits, _ = moe._maybe_overlap_router_and_down_proj(x)
        torch.accelerator.synchronize()
        return latent, logits

    latent_s, logits_s = run(STOCK_TOKENS)
    latent_w, logits_w = run(MAX_TOKENS)
    assert torch.equal(latent_w, latent_s)
    assert torch.equal(logits_w, logits_s)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for VLLM_KIMI_K3_KDA_FUSED_SPEC_VERIFY.

The fused kernel replaces conv1d update -> fused_recurrent_kda ->
gated RMSNorm for pure speculative-verify batches and must be bitwise
identical to them. The eligibility, dispatch and warm-up tests need no GPU.
The GPU tests (SM100+) run KimiK3DeltaAttention._forward with and without the
fused kernel on two copies of the cache pools and compare the outputs of every
request token and the whole conv and recurrent state pools, over random
batches of three chained steps, both input layouts, CUDA-graph replay and the
system ptxas as well as Triton's bundled one.
"""

import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from vllm.models.kimi_k3.nvidia.ops import kda_fused_spec_verify as fused
from vllm.platforms import current_platform

ENV_NAME = "VLLM_KIMI_K3_KDA_FUSED_SPEC_VERIFY"
H, D, W, T = 12, 128, 4, 4
HD = H * D
SL = T + W - 2
LOWER_BOUND = -5.0
PREFIX = "model.layers.0.self_attn"


requires_blackwell = pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.has_device_capability(100)),
    reason="requires an SM100+ GPU",
)


@pytest.fixture(autouse=True)
def dim_first_conv_state(monkeypatch: pytest.MonkeyPatch):
    from vllm.model_executor.layers.mamba import mamba_utils

    monkeypatch.setenv("VLLM_SSM_CONV_STATE_LAYOUT", "DS")
    mamba_utils.get_conv_state_layout.cache_clear()
    yield
    mamba_utils.get_conv_state_layout.cache_clear()


def _norm(device: str):
    from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated

    if device == "cpu":
        # Without a GPU platform no VllmConfig can be built: a norm with only
        # the attributes the checks and the launch arguments read.
        norm = object.__new__(FusedRMSNormGated)
        torch.nn.Module.__init__(norm)
        norm.weight = torch.nn.Parameter(torch.ones(D, dtype=torch.bfloat16))
        norm.bias = None
        norm.eps = 1e-5
        norm.activation = "sigmoid"
        norm._forward_method = norm.forward_cuda
        return norm

    from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
    from vllm.config.compilation import CompilationMode

    config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.NONE, custom_ops=["all"]
        )
    )
    with set_current_vllm_config(config):
        norm = FusedRMSNormGated(
            D, activation="sigmoid", device=device, dtype=torch.bfloat16
        )
    assert norm._forward_method.__func__ is FusedRMSNormGated.forward_cuda
    return norm


class _Pool:
    """Hybrid cache pages: the dim-first conv state, then the bf16 state."""

    PAGE = 896 * 576

    def __init__(self, slots: int, device: str):
        conv_bytes = 3 * HD * SL * 2
        self.raw = torch.zeros(slots * self.PAGE, dtype=torch.uint8, device=device)
        b16 = self.raw.view(torch.bfloat16)
        self.conv = b16.as_strided((slots, 3 * HD, SL), (self.PAGE // 2, SL, 1), 0)
        self.state = b16.as_strided(
            (slots, H, D, D), (self.PAGE // 2, D * D, D, 1), conv_bytes // 2
        )

    def fill(self, gen: torch.Generator, scale: float) -> None:
        dev = self.raw.device
        self.state.copy_(
            (torch.randn(self.state.shape, generator=gen, device=dev) * scale).to(
                torch.bfloat16
            )
        )
        self.conv.copy_(
            torch.randn(self.conv.shape, generator=gen, device=dev).to(torch.bfloat16)
        )
        self.state[0].zero_()
        self.conv[0].zero_()

    def clone(self) -> "_Pool":
        other = object.__new__(_Pool)
        other.raw = self.raw.clone()
        b16 = other.raw.view(torch.bfloat16)
        other.conv = b16.as_strided(self.conv.shape, self.conv.stride(), 0)
        other.state = b16.as_strided(
            self.state.shape, self.state.stride(), self.state.storage_offset()
        )
        return other


def _layer(pool: _Pool, params: dict, fused_on: bool, **overrides) -> SimpleNamespace:
    layer = SimpleNamespace(
        prefix=PREFIX,
        kv_cache=(pool.conv, pool.state),
        kda_spec_decode_backend="native",
        kda_decode_backend="triton",
        kda_prefill_backend="triton",
        decode_conv1d_weight=None,
        decode_norm_weight=None,
        gate_lower_bound=LOWER_BOUND,
        A_log=params["A_log"],
        dt_bias=params["dt_bias"],
        o_norm=params["norm"],
        conv1d=SimpleNamespace(weight=params["conv_weight"], bias=None),
        local_projection_size=HD,
        head_dim=D,
        local_num_heads=H,
        in_proj_padding=0,
        use_recoverssm=False,
        spec_query_len=T,
        num_spec=T - 1,
        _fused_spec_verify_fn=fused.maybe_fused_spec_verify if fused_on else None,
    )
    for name, value in overrides.items():
        setattr(layer, name, value)
    return layer


def _params(device: str, seed: int = 1) -> dict:
    gen = torch.Generator(device=device).manual_seed(seed)
    norm = _norm(device)
    with torch.no_grad():
        norm.weight.copy_(1.0 + 0.1 * torch.randn(D, generator=gen, device=device))
    return dict(
        conv_weight=(torch.randn(3 * HD, W, generator=gen, device=device) * 0.3)
        .float()
        .view(3 * HD, 1, W),
        A_log=(torch.randn(H, generator=gen, device=device) * 0.5).float(),
        dt_bias=(torch.randn(HD, generator=gen, device=device) * 0.5).float(),
        norm=norm,
    )


class _Batch:
    """One verify batch: lens[i] tokens of request i (+ padded requests)."""

    def __init__(self, gen, device, state_indices, lens, nacc, pad, layout):
        num_real = len(lens)
        num_reqs = num_real + pad
        self.ntok = num_reqs * T
        if layout == "in_proj":
            rows = 4 * HD + D + H
            rows += -rows % 16
            proj = torch.randn(self.ntok, rows, generator=gen, device=device)
            proj = proj.to(torch.bfloat16)
            self.beta = proj[:, 4 * HD + D : 4 * HD + D + H].unsqueeze(0)
        else:  # the low-latency projection-overlap layout
            proj = torch.randn(self.ntok, 4 * HD, generator=gen, device=device)
            proj = proj.to(torch.bfloat16)
            fab = torch.randn(self.ntok, D + H + 4, generator=gen, device=device)
            self.beta = fab.to(torch.bfloat16)[:, D : D + H].unsqueeze(0)
        self.mixed_qkv = proj[:, : 3 * HD]
        self.g2 = proj[:, 3 * HD : 4 * HD].unflatten(-1, (H, D))
        self.g1 = torch.randn(1, self.ntok, H, D, generator=gen, device=device).to(
            torch.bfloat16
        )
        qsl = [0]
        for n in lens:
            qsl.append(qsl[-1] + n)
        qsl += [qsl[-1]] * pad
        self.qsl = torch.tensor(qsl, dtype=torch.int32, device=device)
        idx = torch.zeros(num_reqs, T, dtype=torch.int32)
        idx[:num_real] = state_indices
        self.idx = idx.to(device)
        accepted = torch.ones(num_reqs, dtype=torch.int32)
        accepted[:num_real] = torch.tensor(nacc, dtype=torch.int32)
        self.nacc = accepted.to(device)
        self.num_reqs = num_reqs
        rows = [t for i, n in enumerate(lens) for t in range(qsl[i], qsl[i] + n)]
        self.rows = torch.tensor(rows, dtype=torch.long, device=device)

    def metadata(self, **overrides):
        from vllm.models.kimi_k3.nvidia.kda_metadata import KimiK3KDAMetadata

        m = object.__new__(KimiK3KDAMetadata)
        fields = dict(
            has_initial_state=None,
            non_spec_query_start_loc=None,
            non_spec_state_indices_tensor=None,
            spec_token_indx=None,
            non_spec_token_indx=None,
            spec_token_start=None,
            non_spec_token_start=None,
            spec_state_indices_tensor=self.idx,
            spec_query_start_loc=self.qsl,
            num_accepted_tokens=self.nacc,
            num_actual_tokens=self.ntok,
            checkpoint=None,
            num_spec_decodes=self.num_reqs,
            num_prefills=0,
            num_decodes=0,
            num_spec_decode_tokens=self.ntok,
            num_prefill_tokens=0,
            num_decode_tokens=0,
        )
        fields.update(overrides)
        for key, value in fields.items():
            object.__setattr__(m, key, value)
        return m


def _set_context(monkeypatch: pytest.MonkeyPatch, metadata) -> None:
    from vllm.models.kimi_k3.nvidia import kda

    context = SimpleNamespace(attn_metadata={PREFIX: metadata})
    monkeypatch.setattr(kda, "get_forward_context", lambda: context)


def _forward(layer, batch: _Batch, out: torch.Tensor) -> None:
    from vllm.models.kimi_k3.nvidia.kda import KimiK3DeltaAttention

    KimiK3DeltaAttention._forward(
        layer,
        mixed_qkv=batch.mixed_qkv,
        g1=batch.g1,
        g2=batch.g2,
        beta=batch.beta,
        core_attn_out=out,
    )


def _bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


# ---------------------------------------------------------------- CPU tests


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        (dict(kda_spec_decode_backend="flashinfer"), "kda_spec_decode_backend"),
        (dict(use_recoverssm=True), "RecoverSSM"),
        (dict(num_spec=4), "speculative tokens"),
        (dict(local_num_heads=8), "local heads"),
        (dict(gate_lower_bound=None), "gate lower bound"),
        (dict(A_log=torch.zeros(H, dtype=torch.bfloat16)), "A_log"),
        (dict(dt_bias=torch.zeros(HD, dtype=torch.float16)), "dt_bias"),
        (
            dict(conv1d=SimpleNamespace(weight=torch.zeros(3 * HD, 1, W).half())),
            "conv1d weight",
        ),
    ],
)
def test_layer_unsupported_reason(overrides, reason):
    params = _params("cpu")
    pool = _Pool(2, "cpu")
    assert fused.layer_unsupported_reason(_layer(pool, params, True)) is None
    got = fused.layer_unsupported_reason(_layer(pool, params, True, **overrides))
    assert got is not None and reason in got


def test_layer_unsupported_reason_conv_bias_and_layout(monkeypatch):
    from vllm.model_executor.layers.mamba import mamba_utils

    params = _params("cpu")
    pool = _Pool(2, "cpu")
    layer = _layer(pool, params, True)
    layer.conv1d = SimpleNamespace(
        weight=params["conv_weight"], bias=torch.zeros(3 * HD)
    )
    assert "bias" in fused.layer_unsupported_reason(layer)
    layer = _layer(pool, params, True)
    layer.o_norm = SimpleNamespace(activation="sigmoid")
    assert "FusedRMSNormGated" in fused.layer_unsupported_reason(layer)
    monkeypatch.setenv("VLLM_SSM_CONV_STATE_LAYOUT", "SD")
    mamba_utils.get_conv_state_layout.cache_clear()
    layer = _layer(pool, params, True)
    assert "dim-first" in fused.layer_unsupported_reason(layer)


def _cpu_batch(lens=(4, 3), pad=1, layout="in_proj"):
    gen = torch.Generator().manual_seed(0)
    idx = torch.arange(1, 1 + len(lens) * T, dtype=torch.int32).view(len(lens), T)
    return _Batch(gen, "cpu", idx, list(lens), [1] * len(lens), pad, layout)


@pytest.mark.parametrize("layout", ["in_proj", "overlap"])
def test_launch_args_for_a_verify_batch(layout):
    params = _params("cpu")
    pool = _Pool(16, "cpu")
    layer = _layer(pool, params, True)
    batch = _cpu_batch(layout=layout)
    out = torch.empty(1, batch.ntok, H, D, dtype=torch.bfloat16)
    args = fused._launch_args(
        layer, batch.metadata(), batch.mixed_qkv, batch.g1, batch.g2, batch.beta, out
    )
    assert args is not None
    assert args[0].shape == (batch.ntok, 3 * HD)
    assert args[2].shape == (3 * HD, W)
    assert args[12].shape == (batch.num_reqs, T)
    assert args[13].numel() == batch.num_reqs + 1


@pytest.mark.parametrize(
    "overrides",
    [
        dict(num_prefills=1),
        dict(num_decodes=1),
        dict(num_spec_decodes=0),
        dict(spec_token_start=0),
        dict(num_accepted_tokens=None),
    ],
)
def test_launch_args_rejects_other_batches(overrides):
    params = _params("cpu")
    pool = _Pool(16, "cpu")
    layer = _layer(pool, params, True)
    batch = _cpu_batch()
    out = torch.empty(1, batch.ntok, H, D, dtype=torch.bfloat16)
    args = fused._launch_args(
        layer,
        batch.metadata(**overrides),
        batch.mixed_qkv,
        batch.g1,
        batch.g2,
        batch.beta,
        out,
    )
    assert args is None


def test_launch_args_rejects_other_dtypes():
    params = _params("cpu")
    pool = _Pool(16, "cpu")
    layer = _layer(pool, params, True)
    batch = _cpu_batch()
    out = torch.empty(1, batch.ntok, H, D, dtype=torch.bfloat16)
    m = batch.metadata(spec_state_indices_tensor=batch.idx.long())
    args = fused._launch_args(
        layer, m, batch.mixed_qkv, batch.g1, batch.g2, batch.beta, out
    )
    assert args is None
    args = fused._launch_args(
        layer,
        batch.metadata(),
        batch.mixed_qkv.float(),
        batch.g1,
        batch.g2,
        batch.beta,
        out,
    )
    assert args is None


def test_dispatch_and_capture_guard(monkeypatch: pytest.MonkeyPatch):
    launches = []
    monkeypatch.setattr(fused, "fused_spec_verify", lambda *a: launches.append(a))
    monkeypatch.setattr(fused, "_compiled_variants", set())
    params = _params("cpu")
    pool = _Pool(16, "cpu")
    layer = _layer(pool, params, True)
    batch = _cpu_batch()
    out = torch.empty(1, batch.ntok, H, D, dtype=torch.bfloat16)
    call_args = (batch.mixed_qkv, batch.g1, batch.g2, batch.beta, out)
    # A launch variant seen for the first time during capture: stock kernels.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert not fused.maybe_fused_spec_verify(layer, batch.metadata(), *call_args)
    assert not launches
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    assert fused.maybe_fused_spec_verify(layer, batch.metadata(), *call_args)
    assert len(launches) == 1
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert fused.maybe_fused_spec_verify(layer, batch.metadata(), *call_args)
    assert len(launches) == 2
    # Mixed batches and unsupported layers never launch it.
    assert not fused.maybe_fused_spec_verify(
        layer, batch.metadata(num_decodes=1), *call_args
    )
    other = _layer(pool, params, True, use_recoverssm=True)
    assert not fused.maybe_fused_spec_verify(other, batch.metadata(), *call_args)
    assert len(launches) == 2


@pytest.mark.parametrize("fused_ran", [True, False])
def test_forward_dispatch(monkeypatch: pytest.MonkeyPatch, fused_ran):
    """The stock path runs exactly when the fused verify did not."""
    from vllm.models.kimi_k3.nvidia import kda

    stock_calls = []
    monkeypatch.setattr(
        kda,
        "flashinfer_packed_fused_kda_decode",
        lambda **kwargs: stock_calls.append(kwargs),
    )
    fused_calls = []

    def fake_fused(layer, m, *args):
        fused_calls.append(m)
        return fused_ran

    params = _params("cpu")
    pool = _Pool(16, "cpu")
    batch = _cpu_batch()
    layer = _layer(
        pool,
        params,
        False,
        kda_spec_decode_backend="flashinfer",
        decode_conv1d_weight=torch.zeros(1),
        decode_norm_weight=torch.zeros(1),
        _fused_spec_verify_fn=fake_fused,
    )
    _set_context(monkeypatch, batch.metadata())
    out = torch.empty(1, batch.ntok, H, D, dtype=torch.bfloat16)
    _forward(layer, batch, out)
    assert len(fused_calls) == 1
    assert len(stock_calls) == (0 if fused_ran else 1)
    # Without the option the fused verify is never consulted.
    stock_calls.clear()
    layer._fused_spec_verify_fn = None
    _forward(layer, batch, out)
    assert len(fused_calls) == 1 and len(stock_calls) == 1


def test_env_variable_default(monkeypatch: pytest.MonkeyPatch):
    import vllm.envs as envs

    monkeypatch.delenv(ENV_NAME, raising=False)
    assert getattr(envs, ENV_NAME) is False
    monkeypatch.setenv(ENV_NAME, "1")
    assert getattr(envs, ENV_NAME) is True


def _warmup_worker(layers) -> SimpleNamespace:
    context = {f"model.layers.{i}.self_attn": layer for i, layer in enumerate(layers)}
    return SimpleNamespace(
        model_runner=SimpleNamespace(
            compilation_config=SimpleNamespace(static_forward_context=context)
        )
    )


def test_warmup(monkeypatch: pytest.MonkeyPatch):
    from vllm.model_executor.warmup import kimi_k3_triton_warmup as warmup
    from vllm.models.kimi_k3.nvidia import kda

    class _Layer(kda.KimiK3DeltaAttention):
        def __init__(self, **attrs):  # skip the real constructor
            self.__dict__.update(attrs)

    params = _params("cpu")
    pool = _Pool(4, "cpu")
    good = _Layer(**vars(_layer(pool, params, True)))
    bad = _Layer(**vars(_layer(pool, params, True, use_recoverssm=True)))
    compiled = []
    monkeypatch.setattr(
        fused,
        "precompile_capture_variants",
        lambda layer, rows, overlap: compiled.append((layer, rows, overlap)) or [],
    )
    monkeypatch.setattr(
        warmup,
        "current_platform",
        SimpleNamespace(is_cuda=lambda: True, has_device_capability=lambda c: True),
    )
    monkeypatch.delenv(ENV_NAME, raising=False)
    warmup.kimi_k3_kda_fused_spec_verify_warmup(_warmup_worker([good]))
    assert not compiled
    monkeypatch.setenv(ENV_NAME, "1")
    with pytest.raises(ValueError, match="no KDA layer supports it"):
        warmup.kimi_k3_kda_fused_spec_verify_warmup(_warmup_worker([bad]))
    warmup.kimi_k3_kda_fused_spec_verify_warmup(_warmup_worker([good, bad]))
    rows = 4 * HD + D + H
    assert compiled == [(good, rows, (4 * HD, rows - 4 * HD))]
    monkeypatch.setattr(
        warmup,
        "current_platform",
        SimpleNamespace(is_cuda=lambda: True, has_device_capability=lambda c: False),
    )
    with pytest.raises(NotImplementedError, match="SM100"):
        warmup.kimi_k3_kda_fused_spec_verify_warmup(_warmup_worker([good]))


# ---------------------------------------------------------------- GPU tests

BATCH_SIZES = [1, 2, 3, 4, 8, 16, 17, 32, 48, 64, 96, 128]


def _random_case(seed: int):
    gen = torch.Generator().manual_seed(seed)

    def rand() -> float:
        return float(torch.rand(1, generator=gen))

    def randint(lo: int, hi: int) -> int:  # inclusive
        return int(torch.randint(lo, hi + 1, (1,), generator=gen))

    B = BATCH_SIZES[randint(0, len(BATCH_SIZES) - 1)]
    ragged = rand() < 0.6
    pad = [0, 0, 8, 16][randint(0, 3)]
    slots = B * T + 8
    idx = (torch.randperm(slots - 1, generator=gen)[: B * T] + 1).to(torch.int32)
    idx = idx.view(B, T)
    if rand() < 0.3:  # unset checkpoint slots (stores skipped / zero state)
        for n in range(B):
            if rand() < 0.15:
                idx[n, randint(1, T - 1)] = 0
    steps = []
    prev = [T] * B
    for _ in range(3):
        lens = [randint(1, T - 1) if ragged and rand() < 0.25 else T for _ in range(B)]
        steps.append((lens, [randint(1, prev[n]) for n in range(B)]))
        prev = lens
    return dict(B=B, pad=pad, slots=slots, idx=idx, steps=steps)


def _run_cases(monkeypatch, num_cases: int, seed: int, layout: str) -> list:
    device = "cuda"
    params = _params(device, seed=seed + 1)
    failures = []
    for i in range(num_cases):
        case = _random_case(seed * 100000 + i)
        gen = torch.Generator(device=device).manual_seed(seed * 7 + i)
        stock_pool = _Pool(case["slots"], device)
        stock_pool.fill(gen, scale=0.05)
        fused_pool = stock_pool.clone()
        stock_layer = _layer(stock_pool, params, False)
        fused_layer = _layer(fused_pool, params, True)
        for step, (lens, nacc) in enumerate(case["steps"]):
            batch = _Batch(gen, device, case["idx"], lens, nacc, case["pad"], layout)
            _set_context(monkeypatch, batch.metadata())
            out_s = torch.randn(1, batch.ntok, H, D, generator=gen, device=device)
            out_s = out_s.to(torch.bfloat16)
            out_f = out_s.clone()
            _forward(stock_layer, batch, out_s)
            _forward(fused_layer, batch, out_f)
            torch.accelerator.synchronize()
            ok = (
                _bits_equal(out_s[0, batch.rows], out_f[0, batch.rows]),
                _bits_equal(stock_pool.state, fused_pool.state),
                _bits_equal(stock_pool.conv, fused_pool.conv),
            )
            if not all(ok):
                failures.append((i, step, case["B"], ok))
                break
    return failures


@requires_blackwell
@pytest.mark.parametrize("layout", ["in_proj", "overlap"])
def test_fused_matches_stock(monkeypatch: pytest.MonkeyPatch, layout):
    num_cases = int(os.environ.get("KDA_FUSED_VERIFY_TEST_CASES", "60"))
    launches = []
    real = fused.fused_spec_verify
    monkeypatch.setattr(
        fused, "fused_spec_verify", lambda *a: launches.append(1) or real(*a)
    )
    assert _run_cases(monkeypatch, num_cases, seed=3, layout=layout) == []
    assert len(launches) == 3 * num_cases


@requires_blackwell
def test_detects_a_small_change(monkeypatch: pytest.MonkeyPatch):
    """Negative control: a 1e-3 change of dt_bias in the fused arm differs."""
    device = "cuda"
    params = _params(device, seed=11)
    nudged = dict(params, dt_bias=params["dt_bias"] + 1e-3)
    gen = torch.Generator(device=device).manual_seed(5)
    case = _random_case(42)
    stock_pool = _Pool(case["slots"], device)
    stock_pool.fill(gen, scale=0.05)
    fused_pool = stock_pool.clone()
    lens, nacc = case["steps"][0]
    batch = _Batch(gen, device, case["idx"], lens, nacc, 0, "in_proj")
    _set_context(monkeypatch, batch.metadata())
    out_s = torch.zeros(1, batch.ntok, H, D, dtype=torch.bfloat16, device=device)
    out_f = out_s.clone()
    _forward(_layer(stock_pool, params, False), batch, out_s)
    _forward(_layer(fused_pool, nudged, True), batch, out_f)
    torch.accelerator.synchronize()
    assert not _bits_equal(out_s[0, batch.rows], out_f[0, batch.rows])
    assert not _bits_equal(stock_pool.state, fused_pool.state)


@requires_blackwell
@pytest.mark.parametrize("layout", ["in_proj", "overlap"])
def test_cuda_graph_replay_matches_eager(monkeypatch: pytest.MonkeyPatch, layout):
    device = "cuda"
    params = _params(device, seed=21)
    gen = torch.Generator(device=device).manual_seed(9)
    case = _random_case(7)
    lens, nacc = case["steps"][0]
    eager_pool = _Pool(case["slots"], device)
    eager_pool.fill(gen, scale=0.05)
    graph_pool = eager_pool.clone()
    batch = _Batch(gen, device, case["idx"], lens, nacc, case["pad"], layout)
    _set_context(monkeypatch, batch.metadata())
    out_e = torch.zeros(1, batch.ntok, H, D, dtype=torch.bfloat16, device=device)
    out_g = out_e.clone()
    _forward(_layer(eager_pool, params, True), batch, out_e)
    graph_layer = _layer(graph_pool, params, True)
    snapshot = graph_pool.raw.clone()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _forward(graph_layer, batch, out_g)  # compiles this variant
    torch.cuda.current_stream().wait_stream(side)
    torch.accelerator.synchronize()
    captured = []
    real = fused.fused_spec_verify
    monkeypatch.setattr(
        fused, "fused_spec_verify", lambda *a: captured.append(1) or real(*a)
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _forward(graph_layer, batch, out_g)
    assert captured == [1]
    graph_pool.raw.copy_(snapshot)
    out_g.zero_()
    graph.replay()
    torch.accelerator.synchronize()
    assert _bits_equal(out_e[0, batch.rows], out_g[0, batch.rows])
    assert _bits_equal(eager_pool.raw, graph_pool.raw)


def _system_ptxas() -> str | None:
    path = shutil.which("ptxas") or "/usr/local/cuda/bin/ptxas"
    return path if os.path.exists(path) else None


@requires_blackwell
@pytest.mark.parametrize("ptxas", ["bundled", "system"])
def test_fused_matches_stock_under_ptxas(tmp_path, ptxas):
    """The kernel must stay bitwise with either ptxas compiling it (the
    engine may select the system one through TRITON_PTXAS_BLACKWELL_PATH)."""
    env = dict(os.environ, TRITON_CACHE_DIR=str(tmp_path / "triton"))
    env.pop("TRITON_PTXAS_BLACKWELL_PATH", None)
    if ptxas == "system":
        path = _system_ptxas()
        if path is None:
            pytest.skip("no system ptxas")
        env["TRITON_PTXAS_BLACKWELL_PATH"] = path
    env["KDA_FUSED_VERIFY_TEST_CASES"] = "40"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            f"{__file__}::test_fused_matches_stock",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    assert "2 passed" in result.stdout

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for VLLM_FLASHINFER_MOE_TACTIC_PIN.

The parsing, candidate filtering and verification tests run without a GPU
against stand-ins for the FlashInfer pieces the module uses. The GPU test
autotunes a real TRT-LLM FP4 MoE (SM100, FlashInfer) with a pin and checks
that only the pinned tactic is profiled at the pinned bucket, that serving
uses it and that the verification passes or catches a conflicting cache
entry.
"""

import sys
import threading
import types
from collections import namedtuple
from enum import IntEnum
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.warmup import flashinfer_moe_tactic_pin as tactic_pin
from vllm.platforms import current_platform

MOE_OP = tactic_pin.MOE_OP
HS_IDX = 0  # hidden_states position in the stand-in MoeRunnerInputs


# ---------------------------------------------------------------- parsing


def test_parse_tactic_pins():
    assert tactic_pin.parse_tactic_pins("256=16:391,512=32:455") == {
        256: (16, 391),
        512: (32, 455),
    }
    assert tactic_pin.parse_tactic_pins(" 256 = 16:391 , ") == {256: (16, 391)}


@pytest.mark.parametrize("raw", ["", ",", "256", "256=16", "a=16:1", "256=16:1:2"])
def test_parse_tactic_pins_rejects(raw):
    with pytest.raises(ValueError, match="VLLM_FLASHINFER_MOE_TACTIC_PIN"):
        tactic_pin.parse_tactic_pins(raw)


def test_check_buckets():
    buckets = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768)
    tactic_pin.check_buckets({256: (16, 1), 512: (32, 2)}, buckets)
    with pytest.raises(ValueError, match=r"\[500\] are not FlashInfer MoE tuning"):
        tactic_pin.check_buckets({500: (16, 1)}, buckets)


def test_env_variable_default(monkeypatch: pytest.MonkeyPatch):
    import vllm.envs as envs

    monkeypatch.delenv(tactic_pin.ENV_NAME, raising=False)
    assert envs.VLLM_FLASHINFER_MOE_TACTIC_PIN == ""
    monkeypatch.setenv(tactic_pin.ENV_NAME, "256=16:391")
    assert envs.VLLM_FLASHINFER_MOE_TACTIC_PIN == "256=16:391"


# ---------------------------------------------------------------- stand-ins


class _Dtype(IntEnum):
    E2m1 = 1
    MxE2m1 = 2
    E4m3 = 3


class _FakeRunner:
    VALID = [[8, 1], [16, 391], [16, 7], [32, 455], [64, 3]]

    def __init__(self, dtype_weights=_Dtype.MxE2m1):
        self.dtype_weights = dtype_weights

    def get_valid_tactics(self, inputs, profile):
        return [list(t) for t in self.VALID]


@pytest.fixture
def fake_flashinfer(monkeypatch: pytest.MonkeyPatch):
    def module(name, **attrs):
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    runner_module = module(tactic_pin._RUNNER_MODULE, MoERunner=_FakeRunner)
    inputs_cls = SimpleNamespace(idx=lambda name: HS_IDX)
    tuner = SimpleNamespace(
        _lock=threading.RLock(),
        profiling_cache={},
        _winner_partitions={},
        _file_configs={},
        stats=SimpleNamespace(failed_tactics={}),
    )
    module("flashinfer")
    module("flashinfer.fused_moe")
    module("flashinfer.fused_moe.backends")
    module("flashinfer.fused_moe.backends.trtllm")
    module("flashinfer.fused_moe.shared")
    module("flashinfer.fused_moe.shared.inputs", MoeRunnerInputs=inputs_cls)
    module("flashinfer.tllm_enums", DtypeTrtllmGen=_Dtype)
    module(
        "flashinfer.autotuner",
        AutoTuner=SimpleNamespace(get=lambda: tuner),
    )
    return SimpleNamespace(runner_module=runner_module, tuner=tuner)


def _inputs(num_tokens: int):
    return [torch.empty(num_tokens, 1)]


def test_pin_restricts_only_pinned_fp4_buckets(fake_flashinfer):
    original = _FakeRunner.get_valid_tactics
    fp4, other = _FakeRunner(), _FakeRunner(dtype_weights=_Dtype.E4m3)
    pins = {256: (16, 391), 512: (32, 455)}
    with tactic_pin.pin_moe_tactics(pins) as restricted:
        assert fp4.get_valid_tactics(_inputs(256), None) == [[16, 391]]
        assert fp4.get_valid_tactics(_inputs(512), None) == [[32, 455]]
        assert fp4.get_valid_tactics(_inputs(128), None) == _FakeRunner.VALID
        assert other.get_valid_tactics(_inputs(256), None) == _FakeRunner.VALID
    assert restricted == {256: "16:391 of 5", 512: "32:455 of 5"}
    assert _FakeRunner.get_valid_tactics is original


def test_pin_must_be_a_valid_tactic(fake_flashinfer):
    original = _FakeRunner.get_valid_tactics
    with (
        pytest.raises(ValueError, match="not one of FlashInfer's 5 valid tactics"),
        tactic_pin.pin_moe_tactics({256: (16, 999)}),
    ):
        _FakeRunner().get_valid_tactics(_inputs(256), None)
    assert _FakeRunner.get_valid_tactics is original


_Key = namedtuple("_Key", ["custom_op", "nearest_profile"])


def _key(bucket: int, op: str = MOE_OP) -> _Key:
    return _Key(op, ((bucket, 3584), (bucket,)))


def test_entries(fake_flashinfer):
    tuner = fake_flashinfer.tuner
    tuner.profiling_cache[_key(256)] = ([16, 391], None)
    tuner.profiling_cache[_key(256, "flashinfer::other")] = ([1, 1], None)
    tuner._winner_partitions["p"] = {_key(128): ([8, 1], None)}
    file_key = str((MOE_OP, "MoERunner", ((512, 3584), (512,)), ("x",)))
    tuner._file_configs[file_key] = ("MoERunner", [32, 455])
    tuner._file_configs["('flashinfer::other', 'R', ((512,),), ())"] = ("R", [9, 9])
    assert tactic_pin.moe_tactic_entries() == {
        256: {"tuned": {(16, 391)}},
        128: {"tuned": {(8, 1)}},
        512: {"file": {(32, 455)}},
    }


PINS = {256: (16, 391), 512: (32, 455)}


@pytest.mark.parametrize(
    ("restricted", "entries", "failed", "expected"),
    [
        (
            {256: "", 512: ""},
            {256: {"tuned": {(16, 391)}}, 512: {"tuned": {(32, 455)}}},
            set(),
            [],
        ),
        (
            {},
            {256: {"file": {(16, 391)}}, 512: {"file": {(32, 455)}}},
            set(),
            [],
        ),
        (
            {},
            {256: {"file": {(16, 391)}}},
            set(),
            ["bucket 512 has no entry with its pin 32:455"],
        ),
        (
            {},
            {256: {"file": {(16, 391)}}, 512: {"file": {(64, 3)}}},
            set(),
            ["bucket 512 has no entry", "bucket 512 (file) holds 64:3"],
        ),
        (
            {256: ""},
            {
                256: {"tuned": {(16, 391)}, "file": {(32, 1)}},
                512: {"file": {(32, 455)}},
            },
            set(),
            ["bucket 256 (file) holds 32:1"],
        ),
        (
            {256: ""},
            {256: {"file": {(16, 391)}}, 512: {"file": {(32, 455)}}},
            set(),
            ["bucket 256 was profiled with its pin but has no tuned winner"],
        ),
        (
            {},
            {256: {"file": {(16, 391)}}, 512: {"file": {(32, 455)}}},
            {(16, 391)},
            ["pinned tactics 16:391 failed to profile"],
        ),
    ],
)
def test_verification_problems(restricted, entries, failed, expected):
    problems = tactic_pin.verification_problems(PINS, restricted, entries, failed)
    assert len(problems) == len(expected)
    for problem, prefix in zip(problems, expected):
        assert problem.startswith(prefix), problems


def test_verify_raises_and_passes(fake_flashinfer):
    tuner = fake_flashinfer.tuner
    tuner.profiling_cache[_key(256)] = ([16, 391], None)
    with pytest.raises(RuntimeError, match="bucket 512 has no entry"):
        tactic_pin.verify_moe_tactic_pins(PINS, {256: ""})
    tuner.profiling_cache[_key(512)] = ([32, 455], None)
    tactic_pin.verify_moe_tactic_pins(PINS, {256: "", 512: ""})
    tuner.stats.failed_tactics[MOE_OP + "::MoERunner"] = {(32, 455)}
    with pytest.raises(RuntimeError, match="failed to profile"):
        tactic_pin.verify_moe_tactic_pins(PINS, {256: "", 512: ""})


# ---------------------------------------------------------------- GPU


def _has_trtllm_fp4_moe() -> bool:
    if not (current_platform.is_cuda() and current_platform.has_device_capability(100)):
        return False
    try:
        from flashinfer.fused_moe import trtllm_fp4_block_scale_moe  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(
    not _has_trtllm_fp4_moe(), reason="needs an SM100 GPU and FlashInfer MoE"
)
def test_pinned_autotune(monkeypatch: pytest.MonkeyPatch):
    from flashinfer import mxfp8_quantize
    from flashinfer.autotuner import AutoTuner, autotune
    from flashinfer.fused_moe import trtllm_fp4_block_scale_moe
    from flashinfer.fused_moe.backends.trtllm import sm100_runner
    from flashinfer.fused_moe.shared.inputs import MoeRunnerInputs
    from flashinfer.tllm_enums import ActivationType, RoutingMethodType

    E, TOPK, H, I = 896, 16, 3584, 384  # noqa: N806
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(0)

    def rand_u8(*shape, lo=0, hi=256):
        return torch.randint(
            lo, hi, shape, dtype=torch.uint8, device=dev, generator=gen
        )

    w13, w2 = rand_u8(E, 2 * I, H // 2), rand_u8(E, H, I // 2)
    w13s = rand_u8(E, 2 * I, H // 32, lo=118, hi=128).view(torch.float8_e4m3fn)
    w2s = rand_u8(E, H, I // 32, lo=118, hi=128).view(torch.float8_e4m3fn)
    alpha = torch.full((E,), 4.0, device=dev)
    beta = torch.full((E,), 25.0, device=dev)
    bias = torch.randn(E, device=dev, generator=gen) * 0.02

    def run(num_tokens: int) -> None:
        x = torch.randn(num_tokens, H, device=dev, generator=gen).to(torch.bfloat16)
        xq, xs = mxfp8_quantize(
            x, is_sf_swizzled_layout=False, alignment=256, backend="cute-dsl"
        )
        trtllm_fp4_block_scale_moe(
            routing_logits=torch.randn(num_tokens, E, device=dev, generator=gen),
            routing_bias=bias,
            hidden_states=xq,
            hidden_states_scale=xs.view(num_tokens, -1).view(torch.float8_e4m3fn),
            gemm1_weights=w13,
            gemm1_weights_scale=w13s,
            gemm1_bias=None,
            gemm1_alpha=alpha,
            gemm1_beta=beta,
            gemm1_clamp_limit=None,
            gemm2_weights=w2,
            gemm2_weights_scale=w2s,
            gemm2_bias=None,
            output1_scale_scalar=None,
            output1_scale_gate_scalar=None,
            output2_scale_scalar=None,
            num_experts=E,
            top_k=TOPK,
            n_group=1,
            topk_group=1,
            intermediate_size=I,
            local_expert_offset=0,
            local_num_experts=E,
            routed_scaling_factor=1.0,
            routing_method_type=int(RoutingMethodType.DeepSeekV3),
            do_finalize=True,
            activation_type=int(ActivationType.Situ),
            tune_max_num_tokens=16384,
            output=torch.empty(num_tokens, H, dtype=torch.bfloat16, device=dev),
        )
        torch.accelerator.synchronize()

    # Keep tuning short: profile only the first few valid tactics of a bucket.
    hs_idx = MoeRunnerInputs.idx("hidden_states")
    stock_valid = sm100_runner.MoERunner.get_valid_tactics
    seen: dict[int, list] = {}

    def first_few(self, inputs, profile):
        tactics = stock_valid(self, inputs, profile)[:4]
        seen[int(inputs[hs_idx].shape[0])] = [tuple(map(int, t)) for t in tactics]
        return tactics

    monkeypatch.setattr(sm100_runner.MoERunner, "get_valid_tactics", first_few)
    tuner = AutoTuner.get()
    tuner.clear_cache()
    with autotune(True, tuning_buckets=(128, 256), round_up=True):
        run(256)
    assert len(seen[256]) > 1
    pin = seen[256][-1]

    profiled: dict[int, int] = {}
    stock_profile = AutoTuner._profile_single_kernel

    def count_profiles(self, runner, inputs, tactic, *args, **kwargs):
        if type(runner).__name__ == "MoERunner":
            m = int(inputs[hs_idx].shape[0])
            profiled[m] = profiled.get(m, 0) + 1
        return stock_profile(self, runner, inputs, tactic, *args, **kwargs)

    monkeypatch.setattr(AutoTuner, "_profile_single_kernel", count_profiles)
    tuner.clear_cache()
    pins = {256: pin}
    with (
        autotune(True, tuning_buckets=(128, 256), round_up=True),
        tactic_pin.pin_moe_tactics(pins) as restricted,
    ):
        run(256)
    assert sm100_runner.MoERunner.get_valid_tactics is first_few
    assert profiled.get(256) == 1 and profiled.get(128, 0) > 1
    assert set(restricted) == {256}
    entries = tactic_pin.moe_tactic_entries()
    assert entries[256] == {"tuned": {pin}}
    tactic_pin.verify_moe_tactic_pins(pins, restricted)

    picks = []
    stock_choose = tuner.choose_one

    def record_choice(op, runners, config, inputs, **kwargs):
        runner, tactic = stock_choose(op, runners, config, inputs, **kwargs)
        if op == MOE_OP:
            picks.append(tuple(map(int, tactic)))
        return runner, tactic

    monkeypatch.setattr(tuner, "choose_one", record_choice)
    with autotune(False):
        run(200)
    assert picks == [pin]

    # A conflicting cache-file entry for a pinned bucket is reported.
    key = next(
        k
        for k in tuner.profiling_cache
        if k.custom_op == MOE_OP and k.nearest_profile[hs_idx][0] == 256
    )
    other = next(t for t in seen[256] if t != pin)
    monkeypatch.setitem(tuner._file_configs, key.file_key, ("MoERunner", list(other)))
    with pytest.raises(RuntimeError, match="bucket 256 \\(file\\) holds"):
        tactic_pin.verify_moe_tactic_pins(pins, restricted)
    tuner.clear_cache()

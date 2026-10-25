# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for VLLM_KIMI_K3_LAMPORT_COPY_CTAS.

The option widens the launch of the Lamport mailbox copy that ends the Kimi-K3
latent-MoE tail op (KimiK3LatentMoETailOp) from the stock 32 CTAs to 128, 256
or 896 CTAs for TP8 bf16 tail ops on compute capability 10.3 devices. The
value and scope tests need no GPU. The GPU tests build the tail op with the
real LamportCopyKernel (the two other tail kernels need a TP group and are
replaced by inert stand-ins) and check that the widened copy returns the same
bytes as the stock copy and leaves the mailbox reset, eagerly, in CUDA graphs
and while the producer is still writing.
"""

import contextlib
import importlib
import sys
import time
from types import ModuleType, SimpleNamespace

import pytest
import torch

import vllm.envs as envs
from vllm.models.kimi_k3.nvidia.ops import latent_moe_tail as lmt
from vllm.platforms import current_platform

ENV_NAME = "VLLM_KIMI_K3_LAMPORT_COPY_CTAS"
KERNELS_MODULE = "vllm.models.kimi_k3.nvidia.ops.cute_dsl.latent_moe_tail"
HIDDEN, LATENT, MAX_M = 7168, 3584, 128
STOCK_CTAS, THREADS = 32, 128
CHOICES = (128, 256, 896)
# Every 32-bit word of an idle mailbox holds this sentinel.
SENTINEL = -(1 << 31)


def _contract(**overrides) -> lmt.KimiK3LatentMoETailContract:
    fields = dict(
        tp_group_id=1,
        tp_size=8,
        device=torch.device("cuda", 0),
        dtype=torch.bfloat16,
        hidden_size=HIDDEN,
        latent_size=LATENT,
        max_num_tokens=MAX_M,
        rms_eps=1e-6,
        experts_per_token=0,
    )
    fields.update(overrides)
    return lmt.KimiK3LatentMoETailContract(**fields)


OUT_OF_SCOPE = {
    "tp16": (dict(tp_size=16), "tp_size=16"),
    "tp4": (dict(tp_size=4), "tp_size=4"),
    "top_k_7": (dict(experts_per_token=7), "experts_per_token=7"),
    "max_m_64": (dict(max_num_tokens=64), "max_num_tokens=64"),
    "fp16": (dict(dtype=torch.float16), "dtype=torch.float16"),
    "hidden": (dict(hidden_size=4096), "hidden_size=4096"),
    "cpu": (dict(device=torch.device("cpu")), "device=cpu"),
}


class _Logger:
    def __init__(self):
        self.lines: list[str] = []

    def info_once(self, msg, *args, **kwargs):
        line = msg % args
        if line not in self.lines:
            self.lines.append(line)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(ENV_NAME, raising=False)
    return monkeypatch


@pytest.fixture
def capability(monkeypatch: pytest.MonkeyPatch):
    """Report a fixed compute capability (default (10, 3)) to the tail op."""
    state = {"value": (10, 3)}
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: state["value"]
    )
    return state


@pytest.fixture
def fake_logger(monkeypatch: pytest.MonkeyPatch) -> _Logger:
    logger = _Logger()
    monkeypatch.setattr(lmt, "logger", logger)
    return logger


def test_env_default_and_parse(clean_env):
    assert envs.VLLM_KIMI_K3_LAMPORT_COPY_CTAS == 0
    clean_env.setenv(ENV_NAME, "896")
    assert envs.VLLM_KIMI_K3_LAMPORT_COPY_CTAS == 896
    clean_env.setenv(ENV_NAME, "true")
    with pytest.raises(ValueError):
        _ = envs.VLLM_KIMI_K3_LAMPORT_COPY_CTAS


def test_stock_constants_unchanged():
    assert lmt._LAMPORT_COPY_CTAS == STOCK_CTAS
    assert lmt._LAMPORT_COPY_THREADS == THREADS
    assert lmt._MAX_NUM_TOKENS == MAX_M
    assert lmt._LAMPORT_COPY_CTAS_CHOICES == CHOICES


@pytest.mark.parametrize("name", ["k0", "k16", *OUT_OF_SCOPE])
def test_unset_keeps_stock(clean_env, capability, fake_logger, name):
    overrides = OUT_OF_SCOPE[name][0] if name in OUT_OF_SCOPE else {}
    if name == "k16":
        overrides = dict(experts_per_token=16)
    assert lmt._lamport_copy_ctas(_contract(**overrides)) == STOCK_CTAS
    assert fake_logger.lines == []


@pytest.mark.parametrize("ctas", CHOICES)
@pytest.mark.parametrize("top_k", [0, 16])
def test_in_scope_uses_value(clean_env, capability, fake_logger, ctas, top_k):
    clean_env.setenv(ENV_NAME, str(ctas))
    contract = _contract(experts_per_token=top_k)
    assert lmt._lamport_copy_ctas_unsupported(contract) is None
    assert lmt._lamport_copy_ctas(contract) == ctas
    assert fake_logger.lines == [
        f"K3 latent-MoE tail: Lamport copy with {ctas} CTAs ({ENV_NAME})."
    ]


@pytest.mark.parametrize("ctas", CHOICES)
@pytest.mark.parametrize("name", list(OUT_OF_SCOPE))
def test_out_of_scope_keeps_stock(clean_env, capability, fake_logger, ctas, name):
    overrides, reason = OUT_OF_SCOPE[name]
    clean_env.setenv(ENV_NAME, str(ctas))
    contract = _contract(**overrides)
    assert reason in lmt._lamport_copy_ctas_unsupported(contract)
    for _ in range(3):
        assert lmt._lamport_copy_ctas(contract) == STOCK_CTAS
    assert len(fake_logger.lines) == 1
    assert reason in fake_logger.lines[0]
    assert "keeps the stock 32-CTA Lamport copy" in fake_logger.lines[0]


@pytest.mark.parametrize("cap", [(10, 0), (10, 1), (12, 0), (9, 0)])
def test_other_capability_keeps_stock(clean_env, capability, fake_logger, cap):
    capability["value"] = cap
    clean_env.setenv(ENV_NAME, "896")
    assert lmt._lamport_copy_ctas(_contract()) == STOCK_CTAS
    assert f"compute capability {cap}" in fake_logger.lines[0]


@pytest.mark.parametrize("value", ["1", "32", "64", "127", "512", "1024", "-1"])
@pytest.mark.parametrize("in_scope", [True, False])
def test_invalid_value_refused(clean_env, capability, fake_logger, value, in_scope):
    clean_env.setenv(ENV_NAME, value)
    contract = _contract() if in_scope else _contract(tp_size=16)
    with pytest.raises(ValueError, match=ENV_NAME):
        lmt._lamport_copy_ctas(contract)


# ---- KimiK3LatentMoETailOp.__init__ with stand-in kernels (no GPU needed) ----


class _Recorder:
    def __init__(self, built: list, kind: str):
        self.built, self.kind = built, kind

    def __call__(self, **kwargs):
        self.built.append((self.kind, kwargs))
        return SimpleNamespace(kind=self.kind, **kwargs)


@pytest.fixture
def stand_in_kernels(monkeypatch: pytest.MonkeyPatch, capability):
    built: list = []
    kernels = ModuleType(KERNELS_MODULE)
    kernels.CollectiveKernel = _Recorder(built, "collective")
    kernels.AdaptiveUpProjectionKernel = _Recorder(built, "up_projection")
    kernels.LamportCopyKernel = _Recorder(built, "lamport_copy")
    monkeypatch.setitem(sys.modules, KERNELS_MODULE, kernels)
    monkeypatch.setattr(lmt, "register_cutedsl_warmup_provider", lambda op: None)
    monkeypatch.setattr(lmt, "dist", SimpleNamespace(get_rank=lambda group: 0))
    monkeypatch.setattr(
        torch.accelerator, "device_index", lambda index: contextlib.nullcontext()
    )
    return built


@pytest.mark.parametrize(
    "value, overrides, expected",
    [
        (None, {}, STOCK_CTAS),
        ("0", {}, STOCK_CTAS),
        ("128", {}, 128),
        ("256", dict(experts_per_token=16), 256),
        ("896", {}, 896),
        ("896", dict(experts_per_token=16), 896),
        ("896", dict(tp_size=16), STOCK_CTAS),
        ("896", dict(experts_per_token=7), STOCK_CTAS),
    ],
)
def test_op_builds_lamport_copy(
    clean_env, stand_in_kernels, fake_logger, value, overrides, expected
):
    if value is not None:
        clean_env.setenv(ENV_NAME, value)
    op = lmt.KimiK3LatentMoETailOp(_contract(**overrides), group=None)
    kinds = [kind for kind, _ in stand_in_kernels]
    assert kinds == ["collective", "up_projection", "lamport_copy"]
    assert stand_in_kernels[-1][1] == dict(
        hidden_dim=HIDDEN, max_m=MAX_M, ctas=expected, threads=THREADS
    )
    assert op._lamport_copy.ctas == expected
    assert lmt._LAMPORT_COPY_CTAS == STOCK_CTAS


def test_op_refuses_invalid_value_before_building(
    clean_env, stand_in_kernels, fake_logger
):
    clean_env.setenv(ENV_NAME, "64")
    with pytest.raises(ValueError, match=ENV_NAME):
        lmt.KimiK3LatentMoETailOp(_contract(), group=None)
    assert stand_in_kernels == []


# ---- GPU: the real LamportCopyKernel through the tail op ----


def _on_cc_10_3() -> bool:
    return (
        current_platform.is_cuda()
        and torch.cuda.is_available()
        and tuple(torch.cuda.get_device_capability(0)) == (10, 3)
    )


requires_cc_10_3 = pytest.mark.skipif(
    not _on_cc_10_3(), reason="requires a compute capability 10.3 GPU"
)

GRAPH_M = (1, 2, 3, 5, 8, 16, 31, 32, 33, 63, 64, 65, 96, 100, 127, 128)
MIXED_M = (128, 1, 64, 127, 33, 96, 5, 128, 2, 100, 31, 32, 128, 127, 1, 8, 7)
LATE_M = (1, 37, 64, 128)
WATCHDOG_S = 10.0


def _build_op(monkeypatch: pytest.MonkeyPatch, value: str | None, **overrides):
    """A tail op with the real LamportCopyKernel and inert other kernels."""
    kernels = importlib.import_module(KERNELS_MODULE)

    class _Inert:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(kernels, "CollectiveKernel", _Inert)
    monkeypatch.setattr(kernels, "AdaptiveUpProjectionKernel", _Inert)
    monkeypatch.setattr(lmt, "register_cutedsl_warmup_provider", lambda op: None)
    monkeypatch.setattr(lmt, "dist", SimpleNamespace(get_rank=lambda group: 0))
    if value is None:
        monkeypatch.delenv(ENV_NAME, raising=False)
    else:
        monkeypatch.setenv(ENV_NAME, value)
    return lmt.KimiK3LatentMoETailOp(_contract(**overrides), group=None)


def _new_mailbox() -> torch.Tensor:
    mailbox = torch.empty((1, MAX_M, HIDDEN), dtype=torch.bfloat16, device="cuda")
    mailbox.view(torch.int32).fill_(SENTINEL)
    return mailbox


def _data(m: int, salt: int = 0) -> torch.Tensor:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(1000003 * m + salt)
    bits = torch.randint(
        -32768, 32768, (1, m, HIDDEN), generator=gen, dtype=torch.int32, device="cuda"
    ).to(torch.int16)
    # The producers never write bf16 -0 (0x8000), so no word equals the
    # sentinel.
    bits[bits == -32768] = 0
    return bits.view(torch.bfloat16)


def _same(out: torch.Tensor, data: torch.Tensor) -> bool:
    return out.shape == data.shape and torch.equal(
        out.view(torch.int16), data.view(torch.int16)
    )


def _clean(mailbox: torch.Tensor) -> bool:
    return bool((mailbox.view(torch.int32) == SENTINEL).all().item())


def _wait(event: torch.cuda.Event, what: str) -> None:
    start = time.monotonic()
    while not event.query():
        if time.monotonic() - start > WATCHDOG_S:
            pytest.fail(f"Lamport copy did not finish: {what}")
        time.sleep(0.001)


def _sync(what: str) -> None:
    event = torch.cuda.Event()
    event.record()
    _wait(event, what)


@pytest.fixture
def stock_and_wide(monkeypatch: pytest.MonkeyPatch, request):
    torch.accelerator.set_device_index(0)
    stock = _build_op(monkeypatch, None)
    wide = _build_op(monkeypatch, str(request.param))
    assert stock._lamport_copy.ctas == STOCK_CTAS
    assert wide._lamport_copy.ctas == request.param
    assert wide._lamport_copy.threads == THREADS
    return stock._lamport_copy, wide._lamport_copy


@requires_cc_10_3
@pytest.mark.parametrize("stock_and_wide", CHOICES, indirect=True)
def test_gpu_eager_bitwise_vs_stock(stock_and_wide):
    stock, wide = stock_and_wide
    for m in range(1, MAX_M + 1):
        data = _data(m)
        mailbox_s, mailbox_w = _new_mailbox(), _new_mailbox()
        mailbox_s[:, :m].copy_(data)
        mailbox_w[:, :m].copy_(data)
        out_s = stock(mailbox_s, m=m)
        out_w = wide(mailbox_w, m=m)
        _sync(f"eager M={m}")
        assert _same(out_w, data), m
        assert _same(out_s, data), m
        assert torch.equal(out_w.view(torch.int16), out_s.view(torch.int16)), m
        assert _clean(mailbox_w) and _clean(mailbox_s), m


@requires_cc_10_3
@pytest.mark.parametrize("stock_and_wide", CHOICES, indirect=True)
def test_gpu_cuda_graph_bitwise(stock_and_wide):
    stock, wide = stock_and_wide
    for m in GRAPH_M:
        data = _data(m, salt=7)
        outs = {}
        for name, kernel in (("stock", stock), ("wide", wide)):
            mailbox = _new_mailbox()
            torch.accelerator.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                mailbox[:, :m].copy_(data)
                out = kernel(mailbox, m=m)
            for replay in range(2):
                graph.replay()
                _sync(f"graph {name} M={m} replay {replay}")
                assert _same(out, data), (name, m)
                assert _clean(mailbox), (name, m)
            outs[name] = out.clone()
            del graph
        assert _same(outs["wide"], outs["stock"])


@requires_cc_10_3
@pytest.mark.parametrize("stock_and_wide", CHOICES, indirect=True)
def test_gpu_cuda_graph_mixed_m(stock_and_wide):
    _, wide = stock_and_wide
    mailbox = _new_mailbox()
    datas = [_data(m, salt=100 + i) for i, m in enumerate(MIXED_M)]
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    outs = []
    with torch.cuda.graph(graph):
        for m, data in zip(MIXED_M, datas):
            mailbox[:, :m].copy_(data)
            outs.append(wide(mailbox, m=m))
    for replay in range(20):
        graph.replay()
        _sync(f"mixed replay {replay}")
    assert all(_same(out, data) for out, data in zip(outs, datas))
    assert _clean(mailbox)


@requires_cc_10_3
@pytest.mark.parametrize("stock_and_wide", CHOICES, indirect=True)
def test_gpu_late_producer(stock_and_wide):
    """The copy is launched on an empty mailbox and polls until the data lands.

    The data arrives by DMA from pinned host memory in reversed row blocks. No
    other kernel is launched while the copy polls.
    """
    stock, wide = stock_and_wide
    side = torch.cuda.Stream()
    for m in LATE_M:
        for name, kernel in (("stock", stock), ("wide", wide)):
            data = _data(m, salt=23)
            data_host = data.cpu().pin_memory()
            mailbox = _new_mailbox()
            torch.accelerator.synchronize()
            done = torch.cuda.Event()
            out = kernel(mailbox, m=m)
            done.record()
            time.sleep(200e-6)
            step = max(1, m // 4)
            hi = m
            with torch.cuda.stream(side):
                while hi > 0:
                    lo = max(0, hi - step)
                    mailbox[:, lo:hi].copy_(data_host[:, lo:hi], non_blocking=True)
                    side.synchronize()
                    time.sleep(20e-6)
                    hi = lo
            _wait(done, f"late producer {name} M={m}")
            torch.accelerator.synchronize()
            assert _same(out, data), (name, m)
            assert _clean(mailbox), (name, m)


@requires_cc_10_3
def test_gpu_out_of_scope_ops_keep_stock(monkeypatch: pytest.MonkeyPatch):
    torch.accelerator.set_device_index(0)
    for overrides in (dict(tp_size=16), dict(experts_per_token=7)):
        op = _build_op(monkeypatch, "896", **overrides)
        assert op._lamport_copy.ctas == STOCK_CTAS
    op = _build_op(monkeypatch, "896", experts_per_token=16)
    assert op._lamport_copy.ctas == 896

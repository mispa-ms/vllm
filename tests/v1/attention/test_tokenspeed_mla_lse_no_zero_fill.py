# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_TOKENSPEED_MLA_TARGET_LSE_NO_ZERO_FILL: no zero-fill of the LSE buffer
of target-layer TokenSpeed MLA decode calls that run the split-KV reduction.

The allocation tests use a stand-in tokenspeed_mla; the GPU test compares the
real kernel's outputs and LSE with and without the option (SM100 family)."""

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.mla import tokenspeed_mla as tokenspeed_mla_module

ENV = "VLLM_TOKENSPEED_MLA_TARGET_LSE_NO_ZERO_FILL"
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
HEAD_SIZE = KV_LORA_RANK + QK_ROPE_HEAD_DIM


def _make_impl(num_heads: int, max_decode_tokens: int, min_split_kv: int = 1):
    impl = object.__new__(tokenspeed_mla_module.TokenspeedMLAImpl)
    impl.dcp_world_size = 1
    impl.dcp_rank = 0
    impl._parallel_config = SimpleNamespace(cp_kv_cache_interleave_size=1)
    impl.need_to_return_lse_for_decode = True
    impl.kv_lora_rank = KV_LORA_RANK
    impl.qk_rope_head_dim = QK_ROPE_HEAD_DIM
    impl.num_heads = num_heads
    impl.scale = HEAD_SIZE**-0.5
    impl.softmax_scale = None
    impl.output_scale = None
    impl._min_split_kv = min_split_kv
    impl._max_decode_tokens = max_decode_tokens
    return impl


def _layer(is_draft_layer: bool):
    return SimpleNamespace(
        is_draft_layer=is_draft_layer, _q_scale_float=1.0, _k_scale_float=1.0
    )


def _metadata(block_table, seq_lens, max_seq_len, tokens_per_decode):
    return SimpleNamespace(
        num_decodes=seq_lens.shape[0],
        num_decode_tokens=seq_lens.shape[0] * tokens_per_decode,
        max_seq_len=max_seq_len,
        causal=True,
        decode=SimpleNamespace(
            block_table=block_table,
            seq_lens=seq_lens,
            dcp_tot_seq_lens=seq_lens,
        ),
    )


# --------------------------------------------------------------------------
# Allocation handling (CPU, stand-in tokenspeed_mla)
# --------------------------------------------------------------------------


def _fake_tokenspeed(monkeypatch, workspace_size: int, fail: bool = False):
    """A mla_decode whose decode sizes the workspace, then allocates the LSE
    with its module-level torch (as tokenspeed_mla does)."""
    mla_decode = types.ModuleType("tokenspeed_mla.mla_decode")
    mla_decode._FP16_PARTIALS = False
    mla_decode.torch = torch
    mla_decode._get_split_kv_and_workspace_size = lambda *a: (4, workspace_size)
    calls: list[dict] = []

    def tokenspeed_mla_decode(**kwargs):
        q = kwargs["query"]
        _, workspace = _get_split_kv_and_workspace_size(q.shape[0])  # noqa: F821
        lse = torch.zeros(q.shape[:3], dtype=torch.float32, device=q.device)
        other = torch.zeros((2, 2, 2), dtype=torch.float32, device=q.device)
        calls.append(
            dict(
                fp16=_FP16_PARTIALS,  # noqa: F821
                torch_is_real=torch is sys.modules["torch"],
                workspace=workspace,
                lse=lse,
                other=other,
            )
        )
        if fail:
            raise RuntimeError("decode failed")
        out = torch.zeros(*q.shape[:3], KV_LORA_RANK, dtype=torch.bfloat16)
        return out, lse

    # Run the decode with mla_decode's globals, like a function defined there.
    tokenspeed_mla_decode = types.FunctionType(
        tokenspeed_mla_decode.__code__,
        vars(mla_decode),
        "tokenspeed_mla_decode",
        None,
        tokenspeed_mla_decode.__closure__,
    )
    vars(mla_decode).update(sys=sys, KV_LORA_RANK=KV_LORA_RANK)
    mla_decode.tokenspeed_mla_decode = tokenspeed_mla_decode
    package = types.ModuleType("tokenspeed_mla")
    package.tokenspeed_mla_decode = tokenspeed_mla_decode
    package.mla_decode = mla_decode
    monkeypatch.setitem(sys.modules, "tokenspeed_mla", package)
    monkeypatch.setitem(sys.modules, "tokenspeed_mla.mla_decode", mla_decode)
    monkeypatch.setattr(
        tokenspeed_mla_module,
        "_get_workspace",
        lambda *args: torch.empty(1, dtype=torch.int8),
    )
    return mla_decode, calls


def _forward_cpu(impl, is_draft_layer: bool):
    q = torch.zeros(2, 8, HEAD_SIZE, dtype=torch.float8_e4m3fn)
    cache = torch.zeros(1, 64, HEAD_SIZE, dtype=torch.float8_e4m3fn)
    metadata = _metadata(
        torch.zeros(2, 1, dtype=torch.int32),
        torch.tensor([3, 5], dtype=torch.int32),
        max_seq_len=5,
        tokens_per_decode=1,
    )
    return impl.forward_mqa(q, cache, metadata, _layer(is_draft_layer))


@pytest.fixture
def record_allocations(monkeypatch):
    """Record which torch allocator produced each LSE-shaped buffer."""
    made: list[str] = []
    real_zeros, real_empty = torch.zeros, torch.empty

    def lse_shaped(args) -> bool:
        return len(args) == 1 and isinstance(args[0], tuple) and len(args[0]) == 3

    def zeros(*args, **kwargs):
        if lse_shaped(args):
            made.append("zeros")
        return real_zeros(*args, **kwargs)

    def empty(*args, **kwargs):
        if lse_shaped(args):
            made.append("empty")
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", zeros)
    monkeypatch.setattr(torch, "empty", empty)
    return made


@pytest.mark.parametrize(
    ("enabled", "is_draft_layer", "workspace_size", "expected"),
    [
        (False, False, 1024, ["zeros", "zeros"]),
        (True, True, 1024, ["zeros", "zeros"]),  # draft layers: stock
        (True, False, 0, ["zeros", "zeros"]),  # no reduction: keep the fill
        (True, False, 1024, ["empty", "zeros"]),  # only the LSE buffer
    ],
)
def test_lse_allocation(
    monkeypatch, record_allocations, enabled, is_draft_layer, workspace_size, expected
):
    mla_decode, calls = _fake_tokenspeed(monkeypatch, workspace_size)
    real_split = mla_decode._get_split_kv_and_workspace_size
    impl = _make_impl(num_heads=8, max_decode_tokens=2)
    impl.fp16_partials_for_target = True
    impl.lse_no_zero_fill_for_target = enabled
    record_allocations.clear()
    _forward_cpu(impl, is_draft_layer)
    (call,) = calls
    assert record_allocations == expected
    assert call["fp16"] is (not is_draft_layer)
    assert call["torch_is_real"] is not (enabled and not is_draft_layer)
    # The module globals are restored after the call.
    assert mla_decode.torch is torch
    assert mla_decode._get_split_kv_and_workspace_size is real_split
    assert mla_decode._FP16_PARTIALS is False


def test_globals_restored_when_decode_raises(monkeypatch):
    mla_decode, calls = _fake_tokenspeed(monkeypatch, 1024, fail=True)
    real_split = mla_decode._get_split_kv_and_workspace_size
    impl = _make_impl(num_heads=8, max_decode_tokens=2)
    impl.fp16_partials_for_target = True
    impl.lse_no_zero_fill_for_target = True
    with pytest.raises(RuntimeError, match="decode failed"):
        _forward_cpu(impl, is_draft_layer=False)
    assert len(calls) == 1 and not calls[0]["torch_is_real"]
    assert mla_decode.torch is torch
    assert mla_decode._get_split_kv_and_workspace_size is real_split
    assert mla_decode._FP16_PARTIALS is False


def test_proxy_delegates_everything_but_the_lse_zeros():
    proxy = tokenspeed_mla_module._LseNoZeroFillTorch(torch)
    assert proxy.float32 is torch.float32 and proxy.empty is torch.empty
    # Before the workspace is sized, and for other shapes or dtypes: zeros.
    lse = proxy.zeros((2, 3, 4), dtype=torch.float32, device="cpu")
    assert torch.equal(lse, torch.zeros(2, 3, 4))
    proxy.record_workspace_size(64)
    flat = proxy.zeros((2, 3), dtype=torch.float32, device="cpu")
    assert torch.equal(flat, torch.zeros(2, 3))
    assert torch.equal(
        proxy.zeros((2, 3, 4), dtype=torch.bfloat16, device="cpu"),
        torch.zeros(2, 3, 4, dtype=torch.bfloat16),
    )


def test_default_is_off():
    assert tokenspeed_mla_module.TokenspeedMLAImpl.lse_no_zero_fill_for_target is False


def test_rejects_package_without_hooks(monkeypatch):
    mla_decode, _ = _fake_tokenspeed(monkeypatch, 1024)
    tokenspeed_mla_module._check_lse_no_zero_fill_supported()
    del mla_decode._get_split_kv_and_workspace_size
    with pytest.raises(NotImplementedError, match="_get_split_kv_and_workspace_size"):
        tokenspeed_mla_module._check_lse_no_zero_fill_supported()


def test_option_requires_fp16_partials(monkeypatch):
    _fake_tokenspeed(monkeypatch, 1024)
    monkeypatch.delenv(ENV, raising=False)
    assert tokenspeed_mla_module._use_lse_no_zero_fill(True) is False
    monkeypatch.setenv(ENV, "1")
    assert tokenspeed_mla_module._use_lse_no_zero_fill(True) is True
    with pytest.raises(ValueError, match="VLLM_TOKENSPEED_MLA_TARGET_FP16_PARTIALS"):
        tokenspeed_mla_module._use_lse_no_zero_fill(False)


def test_env_var(monkeypatch):
    import vllm.envs as envs

    monkeypatch.delenv(ENV, raising=False)
    assert envs.VLLM_TOKENSPEED_MLA_TARGET_LSE_NO_ZERO_FILL is False
    monkeypatch.setenv(ENV, "1")
    assert envs.VLLM_TOKENSPEED_MLA_TARGET_LSE_NO_ZERO_FILL is True


# --------------------------------------------------------------------------
# Real kernel (SM100-family GPU with tokenspeed_mla)
# --------------------------------------------------------------------------


def _tokenspeed_available() -> bool:
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        from tokenspeed_mla import mla_decode  # noqa: F401
    except ImportError:
        return False
    try:
        tokenspeed_mla_module._check_fp16_partials_supported()
        tokenspeed_mla_module._check_lse_no_zero_fill_supported()
    except (NotImplementedError, ValueError):
        return False
    return True


@pytest.mark.skipif(
    not _tokenspeed_available(),
    reason="needs an SM100-family GPU and tokenspeed_mla",
)
@pytest.mark.parametrize("tokens_per_decode", [1, 4])
@pytest.mark.parametrize("min_split_kv", [1, 8])
def test_lse_no_zero_fill_kernel_bitwise(monkeypatch, tokens_per_decode, min_split_kv):
    from tokenspeed_mla import mla_decode

    from vllm.v1.worker.workspace import WorkspaceManager

    torch.manual_seed(13)
    device = torch.device("cuda")
    workspace_manager = WorkspaceManager(device)
    monkeypatch.setattr(
        tokenspeed_mla_module, "current_workspace_manager", lambda: workspace_manager
    )
    num_heads, block_size, num_decodes = 128, 64, 3
    seq_lens = torch.tensor([7000, 3333, 129], dtype=torch.int32, device=device)
    pages = cdiv(int(seq_lens.max()), block_size)
    cache = (
        torch.randn(num_decodes * pages, block_size, HEAD_SIZE, device=device) * 0.5
    ).to(torch.float8_e4m3fn)
    block_table = torch.arange(
        num_decodes * pages, dtype=torch.int32, device=device
    ).view(num_decodes, pages)
    num_tokens = num_decodes * tokens_per_decode
    q = (torch.randn(num_tokens, num_heads, HEAD_SIZE, device=device) * 0.5).to(
        torch.float8_e4m3fn
    )
    metadata = _metadata(
        block_table, seq_lens, int(seq_lens.max()), tokens_per_decode
    )
    reductions: list[bool] = []
    real_split = mla_decode._get_split_kv_and_workspace_size

    def record_split(*args, **kwargs):
        split_kv, workspace_size = real_split(*args, **kwargs)
        reductions.append(workspace_size != 0)
        return split_kv, workspace_size

    def run(no_zero_fill: bool, garbage: bool):
        impl = _make_impl(num_heads, num_tokens, min_split_kv=min_split_kv)
        impl.fp16_partials_for_target = True
        impl.lse_no_zero_fill_for_target = no_zero_fill
        if garbage:
            # Make uninitialized memory visible: fill the caching allocator's
            # free blocks with NaN before the call.
            torch.full((64 << 20,), float("nan"), device=device)
        with monkeypatch.context() as m:
            m.setattr(mla_decode, "_get_split_kv_and_workspace_size", record_split)
            out, lse = impl.forward_mqa(q, cache, metadata, _layer(False))
        torch.accelerator.synchronize()
        return out.clone(), lse.clone()

    stock = run(False, garbage=False)
    for garbage in (False, True):
        no_fill = run(True, garbage=garbage)
        assert torch.equal(no_fill[0], stock[0])
        assert torch.equal(no_fill[1], stock[1])
    assert mla_decode.torch is torch
    if min_split_kv > 1:
        assert all(reductions)

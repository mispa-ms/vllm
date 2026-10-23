# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_TOKENSPEED_MLA_TARGET_FP16_PARTIALS: fp16 split-KV partials in the
TokenSpeed MLA decode for target-model layers only."""

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.mla import tokenspeed_mla as tokenspeed_mla_module

ENV = "VLLM_TOKENSPEED_MLA_TARGET_FP16_PARTIALS"
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
# Switch handling (CPU, fake tokenspeed_mla)
# --------------------------------------------------------------------------


def _fake_tokenspeed(monkeypatch, fp16_at_import: bool = False, fail: bool = False):
    mla_decode = types.ModuleType("tokenspeed_mla.mla_decode")
    mla_decode._FP16_PARTIALS = fp16_at_import
    seen: list[bool] = []

    def tokenspeed_mla_decode(**kwargs):
        seen.append(mla_decode._FP16_PARTIALS)
        if fail:
            raise RuntimeError("decode failed")
        q = kwargs["query"]
        out = torch.zeros(*q.shape[:3], KV_LORA_RANK, dtype=torch.bfloat16)
        return out, torch.zeros(q.shape[:3])

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
    return mla_decode, seen


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


@pytest.mark.parametrize(
    ("enabled", "is_draft_layer", "expected"),
    [
        (False, False, False),
        (False, True, False),
        (True, True, False),
        (True, False, True),
    ],
)
def test_fp16_partials_only_on_target_layers(
    monkeypatch, enabled, is_draft_layer, expected
):
    mla_decode, seen = _fake_tokenspeed(monkeypatch)
    impl = _make_impl(num_heads=8, max_decode_tokens=2)
    impl.fp16_partials_for_target = enabled
    out, lse = _forward_cpu(impl, is_draft_layer)
    assert seen == [expected]
    assert mla_decode._FP16_PARTIALS is False
    assert out.shape == (2, 8, KV_LORA_RANK) and lse.shape == (2, 8)


def test_default_is_off():
    assert tokenspeed_mla_module.TokenspeedMLAImpl.fp16_partials_for_target is False


def test_switch_restored_when_decode_raises(monkeypatch):
    mla_decode, seen = _fake_tokenspeed(monkeypatch, fail=True)
    impl = _make_impl(num_heads=8, max_decode_tokens=2)
    impl.fp16_partials_for_target = True
    with pytest.raises(RuntimeError, match="decode failed"):
        _forward_cpu(impl, is_draft_layer=False)
    assert seen == [True]
    assert mla_decode._FP16_PARTIALS is False


def test_rejects_package_wide_fp16_partials(monkeypatch):
    _fake_tokenspeed(monkeypatch, fp16_at_import=True)
    with pytest.raises(ValueError, match="TOKENSPEED_MLA_FP16_PARTIALS=1"):
        tokenspeed_mla_module._check_fp16_partials_supported()


def test_rejects_package_without_switch(monkeypatch):
    mla_decode, _ = _fake_tokenspeed(monkeypatch)
    del mla_decode._FP16_PARTIALS
    with pytest.raises(NotImplementedError, match="_FP16_PARTIALS"):
        tokenspeed_mla_module._check_fp16_partials_supported()


def test_env_var(monkeypatch):
    import vllm.envs as envs

    monkeypatch.delenv(ENV, raising=False)
    assert envs.VLLM_TOKENSPEED_MLA_TARGET_FP16_PARTIALS is False
    monkeypatch.setenv(ENV, "1")
    assert envs.VLLM_TOKENSPEED_MLA_TARGET_FP16_PARTIALS is True


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
    return hasattr(mla_decode, "_FP16_PARTIALS")


@pytest.mark.skipif(
    not _tokenspeed_available(),
    reason="needs an SM100-family GPU and tokenspeed_mla",
)
@pytest.mark.parametrize("tokens_per_decode", [1, 4])
def test_fp16_partials_kernel(monkeypatch, tokens_per_decode):
    from tokenspeed_mla import mla_decode

    from vllm.v1.worker.workspace import WorkspaceManager

    torch.manual_seed(11)
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

    def run(enabled: bool, is_draft_layer: bool):
        # min_split_kv > 1 makes sure the split-KV reduction runs.
        impl = _make_impl(num_heads, num_tokens, min_split_kv=8)
        impl.fp16_partials_for_target = enabled
        out, lse = impl.forward_mqa(q, cache, metadata, _layer(is_draft_layer))
        return out.clone(), lse.clone()

    stock = run(False, False)
    draft = run(True, True)
    target = run(True, False)
    assert mla_decode._FP16_PARTIALS is False
    with monkeypatch.context() as m:
        m.setattr(mla_decode, "_FP16_PARTIALS", True)
        package_wide = run(False, False)

    for a, b in zip(draft, stock):
        assert torch.equal(a, b)
    for a, b in zip(target, package_wide):
        assert torch.equal(a, b)
    # The fp16 partials are in use, and stay close to the fp32 ones.
    assert not torch.equal(target[0], stock[0])
    torch.testing.assert_close(target[0], stock[0], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(target[1], stock[1], atol=1e-2, rtol=1e-3)

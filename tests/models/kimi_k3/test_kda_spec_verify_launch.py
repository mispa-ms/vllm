# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for VLLM_KIMI_K3_KDA_SPEC_VERIFY_BV16.

The option only changes the value-block width (BV) of gated speculative-verify
launches of ``fused_recurrent_kda_fwd`` above 192 (heads x sequences). The
launch-parameter tests run without a GPU by recording the kernel launch; the
GPU test checks that outputs and recurrent states are bitwise identical with
the option on and off.
"""

import pytest
import torch

from vllm.models.kimi_k3.nvidia.ops.third_party.kda import fused_recurrent as fr
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv

ENV_NAME = "VLLM_KIMI_K3_KDA_SPEC_VERIFY_BV16"


@pytest.fixture
def set_bv16(monkeypatch: pytest.MonkeyPatch):
    def _set(enabled: bool) -> None:
        monkeypatch.setenv(ENV_NAME, "1" if enabled else "0")
        fr._use_spec_verify_bv16.cache_clear()

    yield _set
    fr._use_spec_verify_bv16.cache_clear()


class _RecordingKernel:
    """Stands in for the Triton kernel and records ``kernel[grid](**kwargs)``."""

    def __init__(self) -> None:
        self.launches: list[tuple[tuple[int, ...], dict]] = []

    def __getitem__(self, grid):
        def launch(**kwargs):
            self.launches.append((grid, kwargs))

        return launch


def _launch_inputs(
    num_heads: int, num_seqs: int, query_len: int, spec: bool, gated: bool
) -> dict:
    head_dim = 16
    num_tokens = num_seqs * query_len
    q = torch.zeros(1, num_tokens, num_heads, head_dim)
    state = torch.zeros(num_tokens + 1, num_heads, head_dim, head_dim)
    cu_seqlens = torch.arange(0, num_tokens + 1, query_len, dtype=torch.int32)
    if spec:
        state_indices = torch.arange(num_tokens, dtype=torch.int32).view(
            num_seqs, query_len
        )
        num_accepted_tokens = torch.ones(num_seqs, dtype=torch.int32)
    else:
        state_indices = torch.arange(num_seqs, dtype=torch.int32)
        num_accepted_tokens = None
    return dict(
        q=q,
        k=torch.zeros_like(q),
        v=torch.zeros_like(q),
        g=torch.zeros_like(q),
        beta=torch.zeros(1, num_tokens, num_heads),
        initial_state=state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=state_indices,
        num_accepted_tokens=num_accepted_tokens,
        A_log=torch.zeros(num_heads, dtype=torch.float32) if gated else None,
        use_gate_in_kernel=gated,
        use_beta_sigmoid_in_kernel=gated,
    )


def _expected_launch(
    head_sequences: int, spec: bool, gated: bool, bv16: bool
) -> tuple[int, int, int]:
    """(BV, num_warps, num_stages) of the default launch table."""
    if not gated:
        return 8, 1, 2
    if head_sequences <= 48:
        return 4, 1, 4
    if head_sequences <= 96:
        return 8, 1, 3
    if head_sequences <= 192 or (spec and bv16):
        return 16, 1, 3
    return 8, 1, 3


@pytest.mark.parametrize("bv16", [False, True])
@pytest.mark.parametrize("gated", [True, False])
@pytest.mark.parametrize("spec", [True, False])
@pytest.mark.parametrize(
    ("num_heads", "num_seqs"),
    [(12, 4), (12, 8), (12, 16), (12, 17), (12, 40), (8, 24), (8, 25), (4, 97)],
)
def test_launch_parameters(
    monkeypatch: pytest.MonkeyPatch,
    set_bv16,
    num_heads: int,
    num_seqs: int,
    spec: bool,
    gated: bool,
    bv16: bool,
):
    set_bv16(bv16)
    kernel = _RecordingKernel()
    monkeypatch.setattr(fr, "fused_recurrent_kda_fwd_kernel", kernel)
    query_len = 4 if spec else 1
    inputs = _launch_inputs(num_heads, num_seqs, query_len, spec, gated)

    fr.fused_recurrent_kda_fwd(**inputs)

    assert len(kernel.launches) == 1
    grid, kwargs = kernel.launches[0]
    bv, num_warps, num_stages = _expected_launch(
        num_heads * num_seqs, spec, gated, bv16
    )
    assert (kwargs["BV"], kwargs["num_warps"], kwargs["num_stages"]) == (
        bv,
        num_warps,
        num_stages,
    )
    head_dim = inputs["v"].shape[-1]
    assert grid == (cdiv(head_dim, bv) * num_seqs * num_heads,)
    assert kwargs["IS_SPEC_DECODING"] is spec
    assert kwargs["USE_GATE_IN_KERNEL"] is gated


def test_option_is_read_once(monkeypatch: pytest.MonkeyPatch, set_bv16):
    set_bv16(True)
    assert fr._use_spec_verify_bv16()
    # Changing the variable later has no effect until the cache is cleared.
    monkeypatch.setenv(ENV_NAME, "0")
    assert fr._use_spec_verify_bv16()
    fr._use_spec_verify_bv16.cache_clear()
    assert not fr._use_spec_verify_bv16()


def _spec_verify_case(num_heads: int, num_seqs: int, query_len: int, seed: int):
    device = "cuda"
    head_dim = 128
    num_tokens = num_seqs * query_len
    gen = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, generator=gen, device=device, dtype=dtype)

    qkv = randn(1, num_tokens, 3 * num_heads * head_dim)
    q, k, v = (
        x.view(1, num_tokens, num_heads, head_dim)
        for x in qkv.split(num_heads * head_dim, dim=-1)
    )
    raw_g = randn(1, num_tokens, num_heads, head_dim)
    raw_beta = randn(1, num_tokens, num_heads)
    A_log = 0.5 * randn(num_heads, dtype=torch.float32)
    dt_bias = 0.1 * randn(num_heads, head_dim, dtype=torch.float32)
    cu_seqlens = torch.arange(
        0, num_tokens + 1, query_len, dtype=torch.int32, device=device
    )
    # Every token has its own state slot; slot 0 stays unused.
    state_indices = torch.arange(
        1, num_tokens + 1, dtype=torch.int32, device=device
    ).view(num_seqs, query_len)
    num_accepted_tokens = (
        torch.arange(num_seqs, dtype=torch.int32, device=device) % query_len + 1
    )
    state = 0.01 * randn(
        num_tokens + 1, num_heads, head_dim, head_dim, dtype=torch.float32
    )
    return dict(
        q=q,
        k=k,
        v=v,
        raw_g=raw_g,
        raw_beta=raw_beta,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=-5.0,
        initial_state=state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=state_indices,
        num_accepted_tokens=num_accepted_tokens,
    )


def _run(case: dict, fuse_gate: bool) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = dict(case)
    inputs["initial_state"] = case["initial_state"].clone()
    out, state = fr.fused_recurrent_kda(**inputs, fuse_gate=fuse_gate)
    torch.accelerator.synchronize()
    return out, state


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA.")
@pytest.mark.parametrize("fuse_gate", [True, False])
@pytest.mark.parametrize(
    ("num_heads", "num_seqs", "query_len"),
    [(12, 16, 4), (12, 17, 4), (12, 40, 4), (12, 64, 3), (8, 25, 2)],
)
@torch.inference_mode()
def test_spec_verify_outputs_bitwise_equal(
    set_bv16, num_heads: int, num_seqs: int, query_len: int, fuse_gate: bool
):
    case = _spec_verify_case(num_heads, num_seqs, query_len, seed=num_seqs)

    set_bv16(False)
    out_default, state_default = _run(case, fuse_gate)
    set_bv16(True)
    out_bv16, state_bv16 = _run(case, fuse_gate)

    assert torch.equal(out_default, out_bv16)
    assert torch.equal(state_default, state_bv16)
    # The verify writes one state per token; the checks above cover them all.
    assert not torch.equal(state_default, case["initial_state"])

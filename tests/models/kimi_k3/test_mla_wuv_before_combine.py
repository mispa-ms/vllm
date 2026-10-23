# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_KIMI_K3_MLA_WUV_BEFORE_DCP_COMBINE: Kimi-K3 MLA decode applies W_UV to
the partials of all gathered heads before the DCP combine (fp16 exchange)."""

import math
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.models.kimi_k3.nvidia.mla import MultiHeadLatentAttention
from vllm.models.kimi_k3.nvidia.ops.mla_wuv_projection import (
    fits_fp16_exchange,
    mla_wuv_projection,
    wuv_projection_bound,
)
from vllm.v1.attention.ops.dcp import _lse_weighted_combine

LATENT_DIM, V_DIM = 512, 128


def _bare_layer(**attrs) -> MultiHeadLatentAttention:
    layer = MultiHeadLatentAttention.__new__(MultiHeadLatentAttention)
    torch.nn.Module.__init__(layer)
    defaults = dict(
        layer_name="model.layers.0.self_attn",
        rotary_emb=None,
        non_causal_multi_token_decode=False,
        dcp_manager=SimpleNamespace(use_a2a=True),
        attn_backend=SimpleNamespace(get_name=lambda: "TOKENSPEED_MLA"),
        _dcp_wuv=None,
        _dcp_wuv_workspace=None,
    )
    for name, value in {**defaults, **attrs}.items():
        setattr(layer, name, value)
    return layer


# --------------------------------------------------------------------------
# CPU
# --------------------------------------------------------------------------


def test_projection_commutes_with_lse_combine():
    generator = torch.Generator().manual_seed(0)
    world_size, num_tokens, num_heads = 4, 5, 3
    partials = torch.randn(
        world_size, num_tokens, num_heads, LATENT_DIM, generator=generator
    ).double()
    lses = 4 * torch.randn(world_size, num_tokens, num_heads, generator=generator)
    lses[1, 0] = float("-inf")  # an empty shard
    w_uv = torch.randn(num_heads, LATENT_DIM, V_DIM, generator=generator).double()

    def project(x):
        return torch.einsum("...hl,hlv->...hv", x, w_uv)

    combine_then_project = project(
        _lse_weighted_combine(partials, lses.double(), is_lse_base_on_e=False)
    )
    project_then_combine = _lse_weighted_combine(
        project(partials), lses.double(), is_lse_base_on_e=False
    )
    torch.testing.assert_close(
        project_then_combine, combine_then_project, rtol=1e-10, atol=1e-10
    )


def test_projection_bound():
    generator = torch.Generator().manual_seed(1)
    w_uv = torch.randn(4, LATENT_DIM, V_DIM, generator=generator) / LATENT_DIM**0.5
    gamma = torch.rand(LATENT_DIM, generator=generator) + 0.5
    bound = wuv_projection_bound(w_uv, gamma)
    # The largest RMSNorm-shaped latent aligned with one W_UV column.
    channel_bound = math.sqrt(LATENT_DIM) * gamma
    worst = (w_uv.abs() * channel_bound[None, :, None]).sum(1).max().item()
    assert worst <= bound <= 1.07 * worst + 1e-6
    assert fits_fp16_exchange(8000.0)
    assert not fits_fp16_exchange(8200.0)


def test_config_checks():
    config = SimpleNamespace(additional_config={})
    assert _bare_layer()._check_wuv_before_dcp_combine(config)
    # Draft-model style layers keep the stock decode tail.
    rope_layer = _bare_layer(rotary_emb=object())
    assert not rope_layer._check_wuv_before_dcp_combine(config)
    noncausal_layer = _bare_layer(non_causal_multi_token_decode=True)
    assert not noncausal_layer._check_wuv_before_dcp_combine(config)
    with pytest.raises(ValueError, match="decode context parallelism"):
        _bare_layer(dcp_manager=None)._check_wuv_before_dcp_combine(config)
    with pytest.raises(ValueError, match="dcp_comm_backend='a2a'"):
        _bare_layer(
            dcp_manager=SimpleNamespace(use_a2a=False)
        )._check_wuv_before_dcp_combine(config)
    with pytest.raises(NotImplementedError, match="TOKENSPEED_MLA"):
        _bare_layer(
            attn_backend=SimpleNamespace(get_name=lambda: "TRITON_MLA")
        )._check_wuv_before_dcp_combine(config)
    with pytest.raises(ValueError, match="decode_backend"):
        _bare_layer()._check_wuv_before_dcp_combine(
            SimpleNamespace(additional_config={"decode_backend": "latency_specialized"})
        )


def test_env_var(monkeypatch):
    import vllm.envs as envs

    monkeypatch.delenv("VLLM_KIMI_K3_MLA_WUV_BEFORE_DCP_COMBINE", raising=False)
    assert envs.VLLM_KIMI_K3_MLA_WUV_BEFORE_DCP_COMBINE is False
    monkeypatch.setenv("VLLM_KIMI_K3_MLA_WUV_BEFORE_DCP_COMBINE", "1")
    assert envs.VLLM_KIMI_K3_MLA_WUV_BEFORE_DCP_COMBINE is True


# --------------------------------------------------------------------------
# Projection kernel (GPU)
# --------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("num_tokens", [1, 7, 200])
def test_projection_kernel(num_tokens):
    generator = torch.Generator(device="cuda").manual_seed(num_tokens)
    num_heads = 24
    # Partials as a view with padded head storage.
    storage = torch.randn(
        num_tokens, num_heads + 4, LATENT_DIM, device="cuda", generator=generator
    ).to(torch.bfloat16)
    x = storage[:, :num_heads]
    w = (
        torch.randn(num_heads, LATENT_DIM, V_DIM, device="cuda", generator=generator)
        / LATENT_DIM**0.5
    ).to(torch.bfloat16)
    out = torch.empty(num_tokens, num_heads, V_DIM, device="cuda", dtype=torch.float16)
    mla_wuv_projection(x, w, out)
    reference = torch.einsum("thl,hlv->thv", x.double(), w.double())
    torch.testing.assert_close(out.double(), reference, rtol=2e-3, atol=2e-3)


# --------------------------------------------------------------------------
# Decode tail with real DCP groups (multi GPU)
# --------------------------------------------------------------------------


def _manager_config(max_num_tokens: int):
    config = MagicMock()
    config.parallel_config.num_ubatches = 1
    config.parallel_config.dcp_comm_backend = "a2a"
    config.scheduler_config.max_num_batched_tokens = max_num_tokens
    config.scheduler_config.max_num_seqs = max_num_tokens
    config.num_speculative_tokens = 0
    config.speculative_config = None
    config.compilation_config.max_cudagraph_capture_size = 0
    config.kernel_config.enable_jit_warmup = False
    return config


def _rank_partials(rank, world_size, num_tokens, total_heads, seed, device):
    """Decode partials of one rank: base-2 LSE, a few empty KV shards."""
    generator = torch.Generator(device=device).manual_seed(seed * 17 + rank)
    storage = torch.randn(
        num_tokens, total_heads + 8, LATENT_DIM, device=device, generator=generator
    ).to(torch.bfloat16)
    latent = storage[:, :total_heads]
    lse = 10 + 3 * torch.randn(
        num_tokens, total_heads, device=device, generator=generator
    )
    seq_lens = torch.full((num_tokens,), 9, dtype=torch.int32, device=device)
    seq_lens[0] = 0  # empty everywhere
    seq_lens[1 + rank % (num_tokens - 1)] = 0  # empty on this rank only
    empty = seq_lens == 0
    latent[empty] = float("nan")
    lse[empty] = float("nan")
    query_start_loc = torch.arange(num_tokens + 1, dtype=torch.int32, device=device)
    return latent, lse, seq_lens, query_start_loc


def _reference(latent, lse, seq_lens, w_uv_local, rank, heads_per_rank):
    """fp64 combine of the gathered partials, then this rank's W_UV."""
    world_size = dist.get_world_size()
    gathered = [torch.empty_like(latent.contiguous()) for _ in range(world_size)]
    gathered_lse = [torch.empty_like(lse) for _ in range(world_size)]
    gathered_lens = [torch.empty_like(seq_lens) for _ in range(world_size)]
    dist.all_gather(gathered, latent.contiguous())
    dist.all_gather(gathered_lse, lse)
    dist.all_gather(gathered_lens, seq_lens)
    heads = slice(rank * heads_per_rank, (rank + 1) * heads_per_rank)
    outputs = torch.stack([g[:, heads] for g in gathered]).double()
    lses = torch.stack([g[:, heads] for g in gathered_lse]).double()
    for src, lens in enumerate(gathered_lens):
        outputs[src][lens == 0] = 0.0
        lses[src][lens == 0] = float("-inf")
    combined = _lse_weighted_combine(outputs, lses, is_lse_base_on_e=False)
    return torch.einsum("thl,hlv->thv", combined, w_uv_local.double()).flatten(1)


def _distributed_worker(local_rank: int, world_size: int, init_file: str) -> None:
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        get_dcp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.v1.attention.ops import dcp

    torch.accelerator.set_device_index(local_rank)
    device = torch.device("cuda", local_rank)
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=world_size,
            rank=local_rank,
            local_rank=local_rank,
            distributed_init_method=f"file://{init_file}",
        )
        initialize_model_parallel(
            tensor_model_parallel_size=world_size,
            decode_context_model_parallel_size=world_size,
        )
        try:
            heads_per_rank, max_num_tokens = 12, 32
            total_heads = heads_per_rank * world_size
            manager = dcp.MLADCPManager(
                vllm_config=_manager_config(max_num_tokens),
                device=device,
                num_heads=heads_per_rank,
                query_head_dim=LATENT_DIM + 64,
                output_head_dim=LATENT_DIM,
                query_dtype=torch.bfloat16,
                output_dtype=torch.bfloat16,
                padded_num_heads=None,
                is_lse_base_on_e=False,
                use_pcp=False,
            )
            generator = torch.Generator(device=device).manual_seed(1234)
            w_uv_all = (
                torch.randn(
                    total_heads, LATENT_DIM, V_DIM, device=device, generator=generator
                )
                / LATENT_DIM**0.5
            ).to(torch.bfloat16)
            heads = slice(
                local_rank * heads_per_rank, (local_rank + 1) * heads_per_rank
            )
            layer = _bare_layer(
                num_local_heads=heads_per_rank,
                kv_lora_rank=LATENT_DIM,
                v_head_dim=V_DIM,
                dcp_world_size=world_size,
                dcp_manager=manager,
                impl=SimpleNamespace(lse_base_on_e=False),
                kv_a_layernorm=SimpleNamespace(
                    weight=torch.ones(LATENT_DIM, device=device)
                ),
                W_UV=w_uv_all[heads].clone(),
            )
            layer._init_wuv_before_dcp_combine(torch.bfloat16)
            assert torch.equal(layer._dcp_wuv, w_uv_all)
            assert layer._dcp_wuv_workspace is not None
            direct_workspace = layer._dcp_wuv_workspace

            os.environ["VLLM_USE_TRITON_DIRECT_DCP_A2A"] = "1"
            triton_workspace = dcp.DirectDCPA2AWorkspace(
                get_dcp_group().device_group,
                device,
                max_num_tokens,
                heads_per_rank,
                V_DIM,
                torch.float16,
            )

            def projected_tail(inputs, workspace):
                layer._dcp_wuv_workspace = workspace
                out = torch.empty(
                    inputs[0].shape[0],
                    heads_per_rank * V_DIM,
                    dtype=torch.bfloat16,
                    device=device,
                )
                layer._dcp_combine_projected(*inputs, out)
                return out

            for seed, num_tokens in enumerate((5, 17, max_num_tokens)):
                inputs = _rank_partials(
                    local_rank, world_size, num_tokens, total_heads, seed, device
                )
                latent, lse, seq_lens, query_start_loc = inputs
                stock = torch.empty(
                    num_tokens,
                    heads_per_rank * V_DIM,
                    dtype=torch.bfloat16,
                    device=device,
                )
                combined = manager.combine(
                    latent, lse, seq_lens=seq_lens, query_start_loc=query_start_loc
                )
                layer._v_up_proj(combined, stock)
                projected = projected_tail(inputs, direct_workspace)
                routed = projected_tail(inputs, triton_workspace)
                fallback = projected_tail(inputs, None)
                reference = _reference(
                    latent, lse, seq_lens, w_uv_all[heads], local_rank, heads_per_rank
                )
                torch.accelerator.synchronize()

                # The Triton combine writes the same bits as the CUDA op + copy.
                assert torch.equal(
                    routed.view(torch.int16), projected.view(torch.int16)
                )
                for out in (projected, fallback):
                    assert torch.isfinite(out).all()
                    error = (out.double() - reference).abs().max().item()
                    stock_error = (stock.double() - reference).abs().max().item()
                    assert error <= stock_error * 1.05 + 1e-6, (error, stock_error)
                    torch.testing.assert_close(
                        out.float(), stock.float(), rtol=2**-6, atol=1e-2
                    )
                # Rows empty on every rank combine to zero, as in the stock tail.
                assert torch.equal(projected[0], torch.zeros_like(projected[0]))

            # CUDA graph capture and replay of the projected tail.
            inputs = _rank_partials(
                local_rank, world_size, max_num_tokens, total_heads, 50, device
            )
            layer._dcp_wuv_workspace = triton_workspace
            graph_out = torch.empty(
                max_num_tokens,
                heads_per_rank * V_DIM,
                dtype=torch.bfloat16,
                device=device,
            )
            layer._dcp_combine_projected(*inputs, graph_out)
            torch.accelerator.synchronize()
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                layer._dcp_combine_projected(*inputs, graph_out)
            for replay in range(3):
                fresh = _rank_partials(
                    local_rank,
                    world_size,
                    max_num_tokens,
                    total_heads,
                    60 + replay,
                    device,
                )
                for dst, src in zip(inputs, fresh):
                    dst.copy_(src)
                graph.replay()
                eager = projected_tail(inputs, direct_workspace)
                torch.accelerator.synchronize()
                assert torch.equal(graph_out.view(torch.int16), eager.view(torch.int16))
            dist.barrier()
        finally:
            cleanup_dist_env_and_memory()


@pytest.mark.parametrize(
    "world_size",
    [
        pytest.param(
            2,
            marks=pytest.mark.skipif(
                torch.accelerator.device_count() < 2, reason="Need at least 2 GPUs."
            ),
        ),
        pytest.param(
            4,
            marks=pytest.mark.skipif(
                torch.accelerator.device_count() < 4, reason="Need at least 4 GPUs."
            ),
        ),
    ],
)
def test_distributed_projected_tail(tmp_path, world_size):
    mp.spawn(
        _distributed_worker,
        args=(world_size, str(tmp_path / "dist_init")),
        nprocs=world_size,
    )

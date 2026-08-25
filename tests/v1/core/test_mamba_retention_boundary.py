# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse-retention boundaries must be blocks align-mode mamba materialises.

`prefix_cache_retention_interval == 0` keeps one state per request, at the
replay boundary. `MambaManager.reachable_block_mask` rounds that boundary down
to `scheduler_block_size`, but align mode only ever holds physical blocks for
the running slot and the speculative window -- every earlier index is a null
placeholder. When the scheduler block spans several mamba blocks the rounded
boundary lands behind the window, so `cache_full_blocks` finds a null block and
registers nothing, and the group's only remaining entry is the prompt's partial
tail.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.request import Request

from .utils import create_requests

pytestmark = pytest.mark.cpu_test

# Mirrors the Kimi-K3 deployment where the collapse was measured: the scheduler
# block spans eight mamba blocks, so the rounded boundary sits up to eight
# blocks behind the tail.
ATTN_BLOCK_SIZE = 128
MAMBA_BLOCK_SIZE = 12 * ATTN_BLOCK_SIZE
SCHEDULER_BLOCK_SIZE = 8 * MAMBA_BLOCK_SIZE
NUM_SPEC = 4
# Just under a scheduler-block multiple, which puts the rounded boundary at its
# furthest from the tail -- the K3 shape (block 79 of 86).
PROMPT_LEN = 4 * SCHEDULER_BLOCK_SIZE - 200
MAMBA_GROUP_ID = 1


def _make_manager() -> KVCacheManager:
    config = KVCacheConfig(
        num_blocks=10000,
        kv_cache_tensors=[],
        prefix_cache_retention_interval=0,
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full_layer"],
                FullAttentionSpec(
                    block_size=ATTN_BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba_layer"],
                MambaSpec(
                    block_size=MAMBA_BLOCK_SIZE,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                    num_speculative_blocks=NUM_SPEC,
                ),
            ),
        ],
    )
    return KVCacheManager(
        config,
        max_model_len=262144,
        scheduler_block_size=SCHEDULER_BLOCK_SIZE,
        hash_block_size=ATTN_BLOCK_SIZE,
        enable_caching=True,
        use_eagle=True,
    )


def _split(request: Request, num_new_tokens: int) -> int:
    stub = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=MAMBA_BLOCK_SIZE),
        use_eagle=True,
        max_num_scheduled_tokens=16384,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        mamba_partial_cache_hit=False,
        hash_block_size=ATTN_BLOCK_SIZE,
    )
    return Scheduler._mamba_block_aligned_split(stub, request, num_new_tokens)


def _prefill(manager: KVCacheManager, request: Request) -> None:
    for _ in range(256):
        computed = request.num_computed_tokens
        if computed >= request.num_tokens:
            return
        num_new = _split(request, request.num_tokens - computed)
        if num_new == 0:
            continue
        assert (
            manager.allocate_slots(request, num_new, num_lookahead_tokens=NUM_SPEC)
            is not None
        )
        request.num_computed_tokens = computed + num_new
    raise AssertionError("prefill did not complete")


def test_retention_boundary_is_a_block_mamba_materialises() -> None:
    """The one block sparse retention keeps must end up in the cache.

    Measured on Kimi-K3 (GB300, 131k prefix replay): the mask marked block 79
    on all forty traced requests, twenty were stored and twenty were skipped as
    null, leaving only the partial tail at 130,944 -- which the EAGLE drop then
    steps below, taking the mamba hit to 0 and the hit rate from 49.1% to 22.7%.
    """
    manager = _make_manager()
    (request,) = create_requests(1, num_tokens=PROMPT_LEN, block_size=ATTN_BLOCK_SIZE)
    _prefill(manager, request)

    mamba_manager = manager.coordinator.single_type_managers[MAMBA_GROUP_ID]
    blocks = mamba_manager.req_to_blocks[request.request_id]
    cached = [
        idx
        for idx, block in enumerate(blocks)
        if not block.is_null and block.block_hash is not None
    ]
    assert cached, (
        "sparse retention registered no full block for this request, so a "
        "later request sharing the prefix has only the partial tail to match, "
        "and the EAGLE drop puts that out of reach"
    )


def test_a_caller_without_a_chunk_gets_dense() -> None:
    """No chunk to name a block with must mean "keep everything", not "nothing".

    `MooncakeStoreConnector`'s coordinator filters blocks it already holds, so
    it calls the mask with no `num_tokens`. An empty mask there stops the tier
    from retaining any mamba block at all -- silently, because nothing fails.
    """
    from vllm.v1.core.single_type_kv_cache_manager import MambaManager

    spec = MambaSpec(
        block_size=MAMBA_BLOCK_SIZE,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    common = dict(
        start_block=0,
        end_block=16,
        alignment_tokens=SCHEDULER_BLOCK_SIZE,
        kv_cache_spec=spec,
        use_eagle=True,
        retention_interval=0,
        reachable_boundaries=(131067,),
    )

    assert MambaManager.reachable_block_mask(**common) is None

    # Given a chunk end, it keeps the block that chunk filled and only that one.
    mask = MambaManager.reachable_block_mask(**common, num_tokens=16 * MAMBA_BLOCK_SIZE)
    assert mask is not None and [i for i, v in enumerate(mask) if v] == [15]

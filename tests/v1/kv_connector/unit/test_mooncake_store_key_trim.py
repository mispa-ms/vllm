# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MooncakeStore value trimming (``VLLM_MOONCAKE_STORE_KEY_TRIM``).

Covers the plan arithmetic, plans built from real KV cache views of a hybrid
attention + state cache with DCP, and put/get round trips through the Store
transfer threads with a byte-copying fake store.
"""

import bisect
import ctypes
import queue
import random
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import (
    worker as store_worker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    BoundaryPut,
    ChunkedTokenDatabase,
    KeyMetadata,
    LBHNCStoreLayout,
    LoadSpec,
    ReqMeta,
    TailKeyBoundary,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.key_trim import (
    AttentionKeyTrimPlan,
    StateKeyTrimPlan,
    build_key_trim_plans,
    dcp_rank_num_tokens,
    size_class_ceil,
    trim_lengths,
)
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    create_kv_cache_views,
    group_kernel_blocks,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

NUM_BLOCKS = 8
NUM_REGIONS = 5
TOKENS_PER_PAGE = 32
ROW_BYTES = 64
DCP_SIZE = 4
BLOCK_SIZE = TOKENS_PER_PAGE * DCP_SIZE
HASH_BLOCK_SIZE = 16
PAGE_BYTES = TOKENS_PER_PAGE * ROW_BYTES
STATE_SHAPES = ((2, 64), (4, 8, 8))
STATE_BYTES = 2 * 64 * 2 + 4 * 8 * 8 * 2
FULL_VALUE_BYTES = NUM_REGIONS * PAGE_BYTES

# Per-rank specs, as the KV cache is allocated.
ATTN_SPEC = MLAAttentionSpec(
    block_size=TOKENS_PER_PAGE,
    num_kv_heads=1,
    head_size=ROW_BYTES,
    dtype=torch.float8_e4m3fn,
)
STATE_SPEC = MambaSpec(
    block_size=BLOCK_SIZE,
    shapes=STATE_SHAPES,
    dtypes=(torch.bfloat16, torch.bfloat16),
    page_size_padded=PAGE_BYTES,
    mamba_cache_mode="align",
)
ATTN_LAYERS = [f"model.layers.{i}.self_attn" for i in range(NUM_REGIONS)]
# The first state group has a layer in every region, the second only in two.
STATE_LAYERS = (
    [f"model.layers.{10 + i}.linear_attn" for i in range(NUM_REGIONS)],
    [f"model.layers.{20 + i}.linear_attn" for i in range(2)],
)
# Groups as the Store worker sees them: attention blocks span all DCP ranks.
GROUPS = [
    KVCacheGroupSpec(
        ATTN_LAYERS,
        MLAAttentionSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=1,
            head_size=ROW_BYTES,
            dtype=torch.float8_e4m3fn,
        ),
    ),
    KVCacheGroupSpec(STATE_LAYERS[0], STATE_SPEC),
    KVCacheGroupSpec(STATE_LAYERS[1], STATE_SPEC),
]
SRC_BLOCKS = (1, 2, 3)
DST_BLOCKS = (4, 5, 6)


def _size_classes() -> list[int]:
    classes = set(range(1, 16))
    for shift in range(48):
        classes.update((8 + m) << shift for m in range(8))
    return sorted(classes)


def test_size_class_ceil_is_the_smallest_class():
    classes = _size_classes()
    rng = random.Random(0)
    values = list(range(1, 70000)) + [rng.randrange(1, 1 << 40) for _ in range(20000)]
    for n in values:
        assert size_class_ceil(n) == classes[bisect.bisect_left(classes, n)], n
    with pytest.raises(ValueError):
        size_class_ceil(0)


def test_trim_lengths_keeps_used_bytes_and_hits_a_size_class():
    rng = random.Random(1)
    for _ in range(20000):
        num_regions = rng.randint(1, 40)
        if rng.random() < 0.5:
            pages = [rng.choice([4096, 65536, 1 << 19, 1 << 20])] * num_regions
        else:
            pages = [rng.randint(1, 1 << 20) for _ in range(num_regions)]
        used = [rng.randint(0, p) if rng.random() < 0.7 else 0 for p in pages]
        target = size_class_ceil(max(sum(used), 4096))
        lengths = trim_lengths(pages, used)
        if lengths is None:
            assert target >= sum(pages)
            continue
        assert sum(lengths) == target < sum(pages)
        assert all(u <= n <= p for u, n, p in zip(used, lengths, pages))


def test_trim_lengths_pads_used_pages_first():
    # 1000 used bytes pad to 4096: the used page fills up first.
    assert trim_lengths([2048, 2048, 2048], [0, 1000, 0]) == [2048, 2048, 0]
    assert trim_lengths([1024, 4096, 4096], [10, 0, 0]) == [1024, 3072, 0]
    # A size class at least as large as the full value keeps full pages.
    assert trim_lengths([4096], [10]) is None
    with pytest.raises(ValueError):
        trim_lengths([4096], [4097])


def test_dcp_rank_num_tokens_matches_token_placement():
    tokens_per_page, dcp_size = 4, 3
    block_size = tokens_per_page * dcp_size
    for rank in range(dcp_size):
        for key_tokens in range(1, 4 * block_size + 1):
            block_start = (key_tokens - 1) // block_size * block_size
            covered = [
                t
                for t in range(block_start, key_tokens)
                if (t - block_start) // tokens_per_page == rank
            ]
            expected = None if key_tokens % block_size == 0 else len(covered)
            actual = dcp_rank_num_tokens(key_tokens, block_size, tokens_per_page, rank)
            assert actual == expected


def _hybrid_kv_caches(
    kernel_block_size: int | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """One int8 allocation with NUM_REGIONS regions; every group overlays it."""
    raw = torch.zeros(NUM_REGIONS * NUM_BLOCKS * PAGE_BYTES, dtype=torch.int8)
    kv_caches: dict[str, torch.Tensor] = {}
    for spec, layers in (
        (ATTN_SPEC, ATTN_LAYERS),
        (STATE_SPEC, STATE_LAYERS[0]),
        (STATE_SPEC, STATE_LAYERS[1]),
    ):
        tensor = KVCacheTensor(
            size=raw.numel(),
            layers=layers,
            layer_stride=NUM_BLOCKS * PAGE_BYTES,
            block_stride=PAGE_BYTES,
        )
        views = create_kv_cache_views(
            raw,
            spec,
            NUM_BLOCKS,
            KVCacheLayout.LBNHC,
            tensor,
            kernel_block_size=kernel_block_size if spec is ATTN_SPEC else None,
        )
        kv_caches.update(zip(layers, views))
    return raw, kv_caches


def _token_dbs(
    kv_caches: dict[str, torch.Tensor], dcp_rank: int
) -> list[ChunkedTokenDatabase]:
    caches = [group_kernel_blocks(cache, NUM_BLOCKS) for cache in kv_caches.values()]
    dbs = []
    for group_id, group in enumerate(GROUPS):
        db = ChunkedTokenDatabase(
            KeyMetadata("test-model", dcp_rank, 0, dcp_rank, 0, group_id=group_id),
            group.kv_cache_spec.block_size,
            hash_block_size=HASH_BLOCK_SIZE,
        )
        db.store_layout.register_kv_caches(caches, NUM_BLOCKS)
        dbs.append(db)
    return dbs


def _plans(kv_caches, dbs, dcp_rank: int, **kwargs):
    options = dict(
        pcp_size=1,
        dcp_size=DCP_SIZE,
        dcp_rank=dcp_rank,
        cp_kv_cache_interleave_size=TOKENS_PER_PAGE,
    )
    options.update(kwargs)
    return build_key_trim_plans(kv_caches, GROUPS, dbs, NUM_BLOCKS, **options)


@pytest.mark.parametrize("kernel_block_size", [None, 8])
def test_plans_from_hybrid_kv_cache_views(kernel_block_size):
    _, kv_caches = _hybrid_kv_caches(kernel_block_size)
    dbs = _token_dbs(kv_caches, dcp_rank=1)
    assert dbs[0].store_layout.block_len == [PAGE_BYTES] * NUM_REGIONS
    attn, state, sparse_state = _plans(kv_caches, dbs, dcp_rank=1)

    assert isinstance(attn, AttentionKeyTrimPlan)
    assert (attn.row_bytes, attn.tokens_per_page, attn.block_size) == (
        ROW_BYTES,
        TOKENS_PER_PAGE,
        BLOCK_SIZE,
    )
    assert isinstance(state, StateKeyTrimPlan)
    assert state.state_bytes == [STATE_BYTES] * NUM_REGIONS
    assert isinstance(sparse_state, StateKeyTrimPlan)
    assert sparse_state.state_bytes == [STATE_BYTES] * 2 + [0] * (NUM_REGIONS - 2)

    # State keys keep every layer's state, whatever the key boundary; the
    # 5 x 768 state bytes are padded to the 4096-byte minimum.
    assert state.lengths(HASH_BLOCK_SIZE) == [1024] + [STATE_BYTES] * 4
    unused = [0] * (NUM_REGIONS - 2)
    assert sparse_state.lengths(3 * HASH_BLOCK_SIZE) == [PAGE_BYTES] * 2 + unused
    # Rank 1 holds block tokens [32, 64): a key ending 40 tokens into a block
    # covers 8 of its rows; a key on a block boundary keeps full pages.
    rows = [8 * ROW_BYTES] * (NUM_REGIONS - 1)
    assert attn.lengths(BLOCK_SIZE + 40) == [PAGE_BYTES] + rows
    assert attn.lengths(2 * BLOCK_SIZE) is None
    # Covering almost the whole page does not reach a smaller size class.
    assert attn.lengths(BLOCK_SIZE - 1) is None


def test_plan_refuses_unsupported_layouts():
    _, kv_caches = _hybrid_kv_caches(None)
    dbs = _token_dbs(kv_caches, dcp_rank=0)
    with pytest.raises(NotImplementedError, match="interleave"):
        _plans(kv_caches, dbs, dcp_rank=0, cp_kv_cache_interleave_size=1)
    with pytest.raises(NotImplementedError, match="context parallel"):
        _plans(kv_caches, dbs, dcp_rank=0, pcp_size=2)

    sharded = LBHNCStoreLayout(
        dbs[0].metadata,
        BLOCK_SIZE,
        HASH_BLOCK_SIZE,
        local_tp_size=1,
        store_tp_size=2,
        tp_rank=0,
        num_kv_heads=2,
    )
    dbs[0] = ChunkedTokenDatabase(
        dbs[0].metadata, BLOCK_SIZE, HASH_BLOCK_SIZE, store_layout=sharded
    )
    with pytest.raises(NotImplementedError, match="rank-local"):
        _plans(kv_caches, dbs, dcp_rank=0)


class _FakeStore:
    """Copies the listed bytes on put; a get must request the stored size."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def batch_is_exist(self, keys):
        return [int(key in self.objects) for key in keys]

    def batch_put_from_multi_buffers(self, keys, addrs, sizes, config=None):
        for key, key_addrs, key_sizes in zip(keys, addrs, sizes):
            chunks = [ctypes.string_at(a, n) for a, n in zip(key_addrs, key_sizes)]
            self.objects.setdefault(key, b"".join(chunks))
        return [0] * len(keys)

    def batch_get_into_multi_buffers(self, keys, addrs, sizes):
        results = []
        for key, key_addrs, key_sizes in zip(keys, addrs, sizes):
            data = self.objects.get(key)
            if data is None or sum(key_sizes) != len(data):
                results.append(-1)
                continue
            offset = 0
            for addr, size in zip(key_addrs, key_sizes):
                ctypes.memmove(addr, data[offset : offset + size], size)
                offset += size
            results.append(len(data))
        return results


def _coordinator():
    # Partial hash hits: a hit may end inside a block, at a hash block.
    return store_worker.MooncakeStoreCoordinator(
        GROUPS,
        scheduler_block_size=BLOCK_SIZE,
        hash_block_size=HASH_BLOCK_SIZE,
        enable_partial_hash_hits=True,
    )


def _sending_thread(store, dbs, dcp_rank):
    thread = store_worker.KVCacheStoreSendingThread(
        store=store,
        coord=_coordinator(),
        token_databases=dbs,
        block_size=BLOCK_SIZE,
        tp_rank=dcp_rank,
        group_put_steps=[1] * len(dbs),
        kv_role="kv_producer",
        ready_event=threading.Event(),
    )
    thread.request_queue.task_done = MagicMock()
    return thread


def _recving_thread(store, dbs, dcp_rank):
    thread = store_worker.KVCacheStoreRecvingThread(
        store=store,
        coord=_coordinator(),
        token_databases=dbs,
        block_size=BLOCK_SIZE,
        tp_rank=dcp_rank,
        ready_event=threading.Event(),
        is_hma_required=True,
    )
    thread.request_queue.task_done = MagicMock()
    return thread


def _hashes(num_tokens: int, seed: int) -> list[bytes]:
    return [
        (seed * 1000003 + i).to_bytes(32, "little")
        for i in range(num_tokens // HASH_BLOCK_SIZE + 2)
    ]


@pytest.mark.parametrize("trim", [False, True])
@pytest.mark.parametrize("dcp_rank", range(DCP_SIZE))
@pytest.mark.parametrize("kernel_block_size", [None, 8])
def test_tail_keys_round_trip(trim, dcp_rank, kernel_block_size):
    """Partial-block attention keys and state keys are put through
    ``_boundary_puts`` and loaded by the receive thread into other blocks."""
    raw, kv_caches = _hybrid_kv_caches(kernel_block_size)
    dbs = _token_dbs(kv_caches, dcp_rank)
    plans = _plans(kv_caches, dbs, dcp_rank) if trim else [None] * len(dbs)
    for db, plan in zip(dbs, plans):
        db.key_trim_plan = plan
    pages = raw.view(NUM_REGIONS, NUM_BLOCKS, PAGE_BYTES)
    # Layer i of every group lives in region i.
    state_regions = [range(len(layers)) for layers in STATE_LAYERS]
    generator = torch.Generator().manual_seed(dcp_rank)

    for case, tail_tokens in enumerate((32, 48, 64, 80, 112, 128)):
        num_full_blocks = 1 + case % 2
        # The attention key ends ``tail_tokens`` into a block; the state
        # checkpoint one hash block earlier.
        key_end = num_full_blocks * BLOCK_SIZE + tail_tokens
        checkpoint = key_end - HASH_BLOCK_SIZE
        hashes = _hashes(key_end, seed=case)
        for src, dst in zip(SRC_BLOCKS, DST_BLOCKS):
            data = torch.randint(
                -128, 128, (NUM_REGIONS, PAGE_BYTES), generator=generator
            )
            pages[:, src].copy_(data.to(torch.int8))
            pages[:, dst].fill_(-1)

        store = _FakeStore()
        puts = _sending_thread(store, dbs, dcp_rank)._boundary_puts(
            SimpleNamespace(
                boundary_puts=[
                    BoundaryPut(0, SRC_BLOCKS[0], key_end),
                    BoundaryPut(1, SRC_BLOCKS[1], checkpoint),
                    BoundaryPut(2, SRC_BLOCKS[2], checkpoint),
                ],
                block_hashes=hashes,
            )
        )
        keys, addrs, sizes, _ = map(list, zip(*puts))
        assert [f"@group:{g}@" in key for g, key in enumerate(keys)] == [True] * 3
        store.batch_put_from_multi_buffers(keys, addrs, sizes)

        # A tail hit at the checkpoint: the attention tail block is keyed at
        # ``key_end``, the state blocks at the checkpoint.
        num_chunks = -(-checkpoint // BLOCK_SIZE)
        recv = _recving_thread(store, dbs, dcp_rank)
        recv._handle_request(
            ReqMeta(
                req_id=f"req-{case}",
                token_len_chunk=checkpoint,
                block_ids=tuple([0] * (num_chunks - 1) + [dst] for dst in DST_BLOCKS),
                block_hashes=hashes,
                load_spec=LoadSpec(
                    vllm_cached_tokens=(num_chunks - 1) * BLOCK_SIZE,
                    kvpool_cached_tokens=checkpoint,
                    can_load=True,
                    token_len=checkpoint,
                    tail_key_boundaries=(
                        TailKeyBoundary(0, key_end),
                        TailKeyBoundary(1, checkpoint),
                        TailKeyBoundary(2, checkpoint),
                    ),
                ),
            )
        )
        assert not recv.get_and_clear_failed_requests()

        rank_rows = dcp_rank_num_tokens(key_end, BLOCK_SIZE, TOKENS_PER_PAGE, dcp_rank)
        for group_id, (src, dst) in enumerate(zip(SRC_BLOCKS, DST_BLOCKS)):
            plan = plans[group_id]
            key_tokens = key_end if group_id == 0 else checkpoint
            lengths = plan.lengths(key_tokens) if plan is not None else None
            if lengths is None:
                lengths = [PAGE_BYTES] * NUM_REGIONS
            assert sizes[group_id] == [n for n in lengths if n]
            # Loads write exactly the stored bytes, at their page offsets.
            for region, kept in enumerate(lengths):
                src_page, dst_page = pages[region, src], pages[region, dst]
                assert torch.equal(dst_page[:kept], src_page[:kept])
                assert bool((dst_page[kept:] == -1).all())
            # Every byte a reader uses is among them.
            if group_id == 0:
                rows = TOKENS_PER_PAGE if rank_rows is None else rank_rows
                used = [(region, rows * ROW_BYTES) for region in range(NUM_REGIONS)]
            else:
                used = [(region, STATE_BYTES) for region in state_regions[group_id - 1]]
            for region, num_bytes in used:
                assert lengths[region] >= num_bytes
        if trim:
            # State values never keep their padding or the pages of regions
            # without a layer of the group.
            assert sum(sizes[1]) < FULL_VALUE_BYTES
            assert sum(sizes[2]) < FULL_VALUE_BYTES


def test_full_block_save_is_unchanged():
    """The positional save keys every chunk at a block boundary, so trimming
    leaves its values untouched."""
    _, kv_caches = _hybrid_kv_caches(None)
    put_calls = []
    for trim in (False, True):
        dbs = _token_dbs(kv_caches, dcp_rank=2)
        if trim:
            for db, plan in zip(dbs, _plans(kv_caches, dbs, dcp_rank=2)):
                db.key_trim_plan = plan
        store = _FakeStore()
        store.batch_put_from_multi_buffers = MagicMock(
            wraps=store.batch_put_from_multi_buffers
        )
        thread = _sending_thread(store, dbs, dcp_rank=2)
        thread.add_request(
            ReqMeta(
                req_id="save",
                token_len_chunk=2 * BLOCK_SIZE,
                block_ids=([1, 2], [3, 4], [5, 6]),
                block_hashes=_hashes(2 * BLOCK_SIZE, seed=7),
                can_save=True,
                store_job_id=1,
            )
        )
        thread._handle_request(thread.request_queue.get())
        put_calls.append(store.batch_put_from_multi_buffers.call_args_list)
    assert put_calls[0]
    assert put_calls[0] == put_calls[1]


def _bare_worker(kv_caches, dcp_rank: int) -> store_worker.MooncakeStoreWorker:
    """A worker with only what register_kv_caches reads (no Mooncake)."""
    w = object.__new__(store_worker.MooncakeStoreWorker)
    w._capacity_only = False
    w.cache_config = SimpleNamespace(num_gpu_blocks=NUM_BLOCKS)
    w._kv_cache_config = KVCacheConfig(
        num_blocks=NUM_BLOCKS, kv_cache_tensors=[], kv_cache_groups=GROUPS
    )
    w._kv_cache_groups = GROUPS
    w.store = MagicMock()
    w.store.register_buffer.return_value = 0
    w.token_dbs = [
        ChunkedTokenDatabase(
            KeyMetadata("test-model", dcp_rank, 0, dcp_rank, 0, group_id=g),
            group.kv_cache_spec.block_size,
            hash_block_size=HASH_BLOCK_SIZE,
        )
        for g, group in enumerate(GROUPS)
    ]
    w.can_put = True
    w.kv_role = "kv_both"
    w.coord = _coordinator()
    w.block_size = BLOCK_SIZE
    w.tp_rank = dcp_rank
    w._group_tp_replication_factors = (1,) * len(GROUPS)
    w.enable_kv_events = False
    w.store_replicate_config = SimpleNamespace()
    w.enable_group_semantics = False
    w._supports_group_ids = False
    w.num_recv_threads = 1
    w.disk_offload_buffer_budget_bytes = None
    w.recv_request_queue = queue.Queue()
    w._is_hma_required = True
    w.pcp_size = 1
    w.dcp_size = DCP_SIZE
    w.dcp_rank = dcp_rank
    w._key_trim_interleave_size = TOKENS_PER_PAGE
    return w


def _start_mocked_thread(*args, **kwargs):
    for value in (*args, *kwargs.values()):
        if isinstance(value, threading.Event):
            value.set()
    return MagicMock()


@pytest.mark.parametrize("trim", [False, True])
def test_register_kv_caches_attaches_plans_only_when_enabled(monkeypatch, trim):
    monkeypatch.setenv("VLLM_MOONCAKE_STORE_KEY_TRIM", "1" if trim else "0")
    _, kv_caches = _hybrid_kv_caches(None)
    w = _bare_worker(kv_caches, dcp_rank=3)
    prefix = f"{store_worker.__name__}."
    with (
        patch(prefix + "KVCacheStoreSendingThread", side_effect=_start_mocked_thread),
        patch(prefix + "KVCacheStoreRecvingThread", side_effect=_start_mocked_thread),
    ):
        w.register_kv_caches(kv_caches)
    plans = [db.key_trim_plan for db in w.token_dbs]
    if not trim:
        assert plans == [None] * len(GROUPS)
        return
    assert isinstance(plans[0], AttentionKeyTrimPlan) and plans[0].dcp_rank == 3
    assert [type(plan) for plan in plans[1:]] == [StateKeyTrimPlan] * 2

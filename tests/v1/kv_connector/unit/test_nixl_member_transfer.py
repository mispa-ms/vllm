# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for pure NIXL member-transfer planning."""

import msgspec
import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.member_transfer import (
    KVLayoutMismatchError,
    plan_member_transfer,
    validate_region_members,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlAgentMetadata,
)


def _metadata(
    region_members: list[list[str]],
    base_addresses: list[int],
    block_lens: list[int],
    *,
    packed_block_stride: int = 0,
    packed_member_layouts: dict[str, tuple[int, int]] | None = None,
) -> NixlAgentMetadata:
    return NixlAgentMetadata(
        engine_id="remote-engine",
        agent_metadata=b"agent",
        kv_caches_base_addr=base_addresses,
        device_id=7,
        num_blocks=2,
        block_lens=block_lens,
        kv_cache_layout="HND",
        block_size=16,
        ssm_sizes=(0, 0),
        attn_backend_name="FLASH_ATTN",
        physical_blocks_per_logical_kv_block=1,
        region_members=region_members,
        packed_block_stride=packed_block_stride,
        packed_member_layouts=packed_member_layouts or {},
    )


def test_member_metadata_round_trip():
    metadata = _metadata([["L0", "L1"]], [0x10000], [256])

    encoded = msgspec.msgpack.encode(metadata)
    assert msgspec.msgpack.Decoder(NixlAgentMetadata).decode(encoded) == metadata


def test_packed_member_metadata_round_trip():
    metadata = _metadata(
        [["L0", "L1"]],
        [0x10000],
        [256],
        packed_block_stride=256,
        packed_member_layouts={"L0": (0, 128), "L1": (128, 128)},
    )

    encoded = msgspec.msgpack.encode(metadata)
    assert msgspec.msgpack.Decoder(NixlAgentMetadata).decode(encoded) == metadata


def test_plan_member_transfer_expands_pooled_regions():
    metadata = _metadata(
        [["a", "a.swa"], ["b"]],
        [0xA000, 0xB000],
        [128, 128],
    )

    prepared, plan = plan_member_transfer(
        metadata,
        [["a", "a.swa"], ["b"]],
        {"a": 0, "a.swa": 1, "b": 0},
    )

    assert plan.member_names == ("a", "a.swa", "b")
    assert plan.local_regions == (0, 0, 1)
    assert plan.group_ids == (0, 1, 0)
    assert prepared.kv_caches_base_addr == [0xA000, 0xA000, 0xB000]
    assert prepared.block_lens == [128, 128, 128]
    assert prepared.region_members == []


def test_plan_member_transfer_filters_and_reorders_a_pp_stage():
    metadata = _metadata(
        [["l2"], ["l0"], ["l3"], ["l1"]],
        [0xC000, 0xA000, 0xD000, 0xB000],
        [65536, 65536, 32768, 32768],
    )

    prepared, plan = plan_member_transfer(
        metadata,
        [["l2"], ["l3"]],
        {"l2": 0, "l3": 1},
    )

    assert plan.local_regions == (0, 1)
    assert plan.group_ids == (0, 1)
    assert prepared.kv_caches_base_addr == [0xC000, 0xD000]
    assert prepared.block_lens == [65536, 32768]


def test_plan_member_transfer_rejects_missing_local_member():
    metadata = _metadata([["l0"]], [0xA000], [128])

    with pytest.raises(RuntimeError, match="missing locally owned"):
        plan_member_transfer(
            metadata,
            [["l0"], ["l1"]],
            {"l0": 0, "l1": 1},
        )


def test_plan_member_transfer_rejects_duplicate_remote_member():
    metadata = _metadata([["a"], ["a"]], [0xA000, 0xB000], [128, 128])

    with pytest.raises(RuntimeError, match="multiple NIXL regions"):
        plan_member_transfer(metadata, [["a"]], {"a": 0})


def test_plan_member_transfer_is_canonical_across_remote_orderings():
    rank0 = _metadata([["x"], ["y"]], [0x1000, 0x2000], [64, 128])
    rank1 = _metadata([["y"], ["x"]], [0x2000, 0x1000], [128, 64])
    local_members = [["x"], ["y"]]
    layer_to_group = {"x": 0, "y": 1}

    _, plan0 = plan_member_transfer(rank0, local_members, layer_to_group)
    prepared1, plan1 = plan_member_transfer(rank1, local_members, layer_to_group)

    assert plan0 == plan1
    assert prepared1.kv_caches_base_addr == [0x1000, 0x2000]
    assert prepared1.block_lens == [64, 128]


def test_validate_region_members_rejects_duplicate_local_member():
    with pytest.raises(RuntimeError, match="spans multiple NIXL regions"):
        validate_region_members([["a"], ["a"]])


def test_plan_packed_member_transfer_keeps_local_and_remote_strides():
    remote = _metadata(
        [["L0", "L1"]],
        [0x10000],
        [256],
        packed_block_stride=256,
        packed_member_layouts={"L0": (0, 128), "L1": (128, 128)},
    )

    prepared, plan = plan_member_transfer(
        remote,
        [["L1"]],
        {"L1": 0},
        local_packed_layouts={"L1": (0, 128)},
        local_block_stride=128,
    )

    assert prepared.kv_caches_base_addr == [0x10080]
    assert prepared.block_lens == [128]
    assert prepared.packed_block_stride == 0
    assert prepared.packed_member_layouts == {}
    assert plan.member_names == ("L1",)
    assert plan.local_layouts == ((0, 128),)
    assert plan.local_block_stride == 128
    assert plan.remote_block_stride == 256
    assert plan.is_packed


@pytest.mark.parametrize(
    ("local_packed", "remote_packed"),
    [(True, False), (False, True)],
)
def test_plan_member_transfer_rejects_mixed_packed_layouts(
    local_packed: bool,
    remote_packed: bool,
):
    remote = (
        _metadata(
            [["L0"]],
            [0x10000],
            [128],
            packed_block_stride=128,
            packed_member_layouts={"L0": (0, 128)},
        )
        if remote_packed
        else _metadata([["L0"]], [0x10000], [128])
    )

    with pytest.raises(RuntimeError, match="cannot be mixed"):
        plan_member_transfer(
            remote,
            [["L0"]],
            {"L0": 0},
            local_packed_layouts={"L0": (0, 128)} if local_packed else None,
            local_block_stride=128 if local_packed else 0,
        )


class TestNonOverlappingStagesAreNotPlannable:
    """Why the handshake filters stages instead of tolerating the miss.

    ``plan_member_transfer`` raises when the remote metadata does not name a
    layer we own, which is exactly the shape of a decode stage holding the other
    half of the model. There is no empty-plan path to fall through to, so a
    handshake that queries every stage is a startup crash rather than wasted
    work. These are the K3 proportions: 93 layers, split by ``get_pp_indices``.
    """

    LAYERS = [f"layer.{i}" for i in range(93)]

    @classmethod
    def _stage(cls, rank, size):
        from vllm.distributed.utils import get_pp_indices

        start, end = get_pp_indices(len(cls.LAYERS), rank, size)
        return cls.LAYERS[start:end]

    def _plan(self, local_layers, remote_layers):
        meta = NixlAgentMetadata(
            engine_id="remote",
            agent_metadata=b"",
            kv_caches_base_addr=[0] * len(remote_layers),
            device_id=0,
            num_blocks=1,
            block_lens=[128] * len(remote_layers),
            kv_cache_layout="HND",
            block_size=16,
            ssm_sizes=(0, 0),
            attn_backend_name="FLASH_ATTN",
            physical_blocks_per_logical_kv_block=1,
            region_members=[[name] for name in remote_layers],
        )
        return plan_member_transfer(
            meta,
            [[name] for name in local_layers],
            {name: 0 for name in local_layers},
        )

    def test_full_overlap_plans(self):
        stage = self._stage(0, 2)
        _meta, plan = self._plan(stage, stage)
        assert plan.member_names == tuple(stage)

    def test_the_raise_is_typed_so_callers_can_stop_retrying(self):
        """A layer set mismatch is static; the type says so.

        This is what a speculative-config on one side only looks like: the draft
        registers KV the peer never advertises. Retrying it per request cost an
        hour of an idle-reaper timeout with nothing in the CI log (63751281).
        """
        with pytest.raises(KVLayoutMismatchError) as exc:
            self._plan(self._stage(0, 2), self._stage(1, 2))
        assert "speculative-config" in str(exc.value)
        assert isinstance(exc.value, RuntimeError)

    def test_disjoint_stage_raises(self):
        with pytest.raises(RuntimeError, match="missing locally owned KV cache"):
            self._plan(self._stage(0, 2), self._stage(1, 2))

    def test_partial_overlap_raises(self):
        """P_PP2 s0 against D_PP4 s0: D holds a strict subset of P's window."""
        with pytest.raises(RuntimeError, match="missing locally owned KV cache"):
            self._plan(self._stage(0, 2), self._stage(0, 4))


class TestDraftLayerNumberingIsPPInvariant:
    """Draft KV layer names must not depend on how the model is PP-sharded.

    Member routing transfers by layer name, so a name is a contract between two
    engines that may be sharded differently. Numbering the draft from the
    PP-*local* layer count breaks it two ways at once, and this pins both.

    Not a test of the connector -- a test of the rule the connector depends on.
    The failure it guards against (AIB 63781036) was a prefiller at PP=2 asking
    a PP=1 decoder for 'model.layers.46.self_attn', a name only its own stage
    used; symmetric PP passed because both sides were wrong identically.
    """

    TOTAL = 93
    DRAFT = 5

    @staticmethod
    def _local_count(total, rank, size):
        from vllm.distributed.utils import get_pp_indices

        start, end = get_pp_indices(total, rank, size)
        return end - start

    @pytest.mark.parametrize("pp_size", [1, 2, 4])
    def test_total_based_numbering_never_collides_with_target_layers(self, pp_size):
        draft = set(range(self.TOTAL, self.TOTAL + self.DRAFT))
        assert draft.isdisjoint(range(self.TOTAL))
        # and every stage agrees on the same names
        assert all(
            set(range(self.TOTAL, self.TOTAL + self.DRAFT)) == draft
            for _ in range(pp_size)
        )

    def test_pp_local_numbering_collides_with_the_stages_own_layers(self):
        """The old expression, kept so the bug cannot come back unnoticed."""
        from vllm.distributed.utils import get_pp_indices

        start, end = get_pp_indices(self.TOTAL, 1, 2)
        local = self._local_count(self.TOTAL, 1, 2)
        draft = set(range(local, local + self.DRAFT))
        assert not draft.isdisjoint(range(start, end)), (
            "PP-local numbering is supposed to collide -- if this passes, the "
            "premise of the fix changed"
        )

    def test_pp_local_numbering_differs_between_a_sharded_and_unsharded_peer(self):
        sharded = self._local_count(self.TOTAL, 1, 2)
        unsharded = self._local_count(self.TOTAL, 0, 1)
        assert sharded != unsharded
        assert self.TOTAL == unsharded  # the value the fix uses, on both peers

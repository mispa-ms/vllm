# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Store rank-local MooncakeStore values without bytes no reader uses.

Opt-in with ``VLLM_MOONCAKE_STORE_KEY_TRIM=1``.

A rank-local Store value is one page per registered KV-cache region. Some keys
carry pages, or page tails, that no layer ever reads back:

* State groups (``MambaSpec``): a page may be padded past the layer state
  (``page_size_padded``), and when a group has fewer layers than there are
  regions, some regions hold no layer of the group at all.
* Attention groups, for a key whose token range ends inside a KV block (for
  example the prompt's partial tail): only the first rows of the rank's page
  are covered by the key. The remaining rows belong to tokens past the key's
  end, which a request that hits this key recomputes before attention reads
  them.

With the option on, such a value stores a prefix of each region's page: every
byte a reader can use, padded with bytes from the same pages up to a size
class of Mooncake's offset allocator (``(8 + m) << k`` with a 3-bit mantissa
``m``). This module also pads every trimmed value to at least
``_MIN_OBJECT_BYTES``; that lower bound is its own choice, not an allocator
limit. A value is only trimmed when the padded size is smaller than the full
value, so every kept byte is the byte the full value would hold at the same
offset.

Puts and gets derive the lengths from the same inputs (KV-cache group, the
token count the key's hash covers, DCP rank), so the stored size equals the
size every reader with the same setting requests. Every rank that reads the
store must use the same setting: a reader with a different setting requests a
different value size for these keys.
"""

from collections.abc import Mapping, Sequence

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    ChunkedTokenDatabase,
    RankLocalStoreLayout,
)
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import is_non_overlapping_and_dense
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    ChunkedLocalAttentionSpec,
    KVCacheGroupSpec,
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
    group_kernel_blocks,
)

# Lower bound on the size of a trimmed value.
_MIN_OBJECT_BYTES = 4096
# Windowed and chunked-local attention groups are not trimmed.
_WINDOW_SPECS = (SlidingWindowSpec, ChunkedLocalAttentionSpec)


def size_class_ceil(num_bytes: int) -> int:
    """Smallest ``(8 + m) << k`` (``0 <= m < 8``) that is ``>= num_bytes``.

    Sizes up to 16 bytes are their own class.
    """
    if num_bytes <= 0:
        raise ValueError(f"num_bytes must be positive, got {num_bytes}")
    shift = max(0, num_bytes.bit_length() - 4)
    return cdiv(num_bytes, 1 << shift) << shift


def trim_lengths(
    page_bytes: Sequence[int], used_bytes: Sequence[int]
) -> list[int] | None:
    """Per-region prefix lengths of one trimmed Store value.

    Args:
        page_bytes: Page size of every registered region.
        used_bytes: Bytes at the start of each region's page that a reader
            uses; 0 for a region the key does not use.

    Returns:
        ``lengths`` with ``used_bytes[i] <= lengths[i] <= page_bytes[i]`` and
        ``sum(lengths) == size_class_ceil(max(sum(used_bytes),
        _MIN_OBJECT_BYTES))``. The padding comes first from the tails of
        pages with used bytes, then from the starts of unused pages, in
        region order. ``None`` when that size is not smaller than the full
        value, which then keeps every page whole.
    """
    if len(page_bytes) != len(used_bytes) or not page_bytes:
        raise ValueError("page_bytes and used_bytes must be non-empty and match")
    for page, used in zip(page_bytes, used_bytes):
        if page <= 0 or not 0 <= used <= page:
            raise ValueError(f"invalid region: page {page} B, used {used} B")
    total_used = sum(used_bytes)
    target = size_class_ceil(max(total_used, _MIN_OBJECT_BYTES))
    if target >= sum(page_bytes):
        return None
    lengths = list(used_bytes)
    padding = target - total_used
    for pages_with_used_bytes in (True, False):
        for i, used in enumerate(used_bytes):
            if padding and (used > 0) == pages_with_used_bytes:
                extra = min(padding, page_bytes[i] - lengths[i])
                lengths[i] += extra
                padding -= extra
    return lengths if padding == 0 else None


def dcp_rank_num_tokens(
    key_tokens: int, block_size: int, tokens_per_page: int, dcp_rank: int
) -> int | None:
    """Rows of this rank's page covered by a key ending at ``key_tokens``.

    A block holds ``block_size == tokens_per_page * dcp_size`` tokens. With
    ``cp_kv_cache_interleave_size == tokens_per_page``, DCP rank ``r`` holds
    tokens ``[r * tokens_per_page, (r + 1) * tokens_per_page)`` of each block,
    in page rows ``[0, tokens_per_page)``. Returns ``None`` when the key ends
    on a block boundary and covers the whole page.
    """
    if key_tokens <= 0:
        raise ValueError(f"key_tokens must be positive, got {key_tokens}")
    tokens_in_block = key_tokens - (key_tokens - 1) // block_size * block_size
    if tokens_in_block >= block_size:
        return None
    return min(max(tokens_in_block - tokens_per_page * dcp_rank, 0), tokens_per_page)


class KeyTrimPlan:
    """Trim plan of one KV-cache group's rank-local Store values."""

    def __init__(self, page_bytes: Sequence[int]):
        self.page_bytes = list(page_bytes)

    def lengths(self, key_tokens: int) -> list[int] | None:
        """Per-region lengths of the value keyed by the hash of the first
        ``key_tokens`` tokens; ``None`` keeps full pages."""
        raise NotImplementedError

    def trim(
        self, addrs: list[int], sizes: list[int], key_tokens: int
    ) -> tuple[list[int], list[int]]:
        """Trim one value's buffer list; regions trimmed to 0 bytes are
        dropped. A value that is not exactly one full page per region is
        returned unchanged."""
        lengths = self.lengths(key_tokens)
        if (
            lengths is None
            or len(addrs) != len(self.page_bytes)
            or list(sizes) != self.page_bytes
        ):
            return addrs, sizes
        kept = [(addr, n) for addr, n in zip(addrs, lengths) if n]
        return [addr for addr, _ in kept], [n for _, n in kept]

    def trim_values(
        self,
        addrs: list[list[int]],
        sizes: list[list[int]],
        key_tokens: Sequence[int],
    ) -> None:
        """Trim the buffer lists of several values in place."""
        assert len(addrs) == len(sizes) == len(key_tokens)
        for i, tokens in enumerate(key_tokens):
            addrs[i], sizes[i] = self.trim(addrs[i], sizes[i], tokens)


class StateKeyTrimPlan(KeyTrimPlan):
    """Every key of a state group keeps the same bytes: each layer's state at
    the start of its region's page."""

    def __init__(self, page_bytes: Sequence[int], state_bytes: Sequence[int]):
        super().__init__(page_bytes)
        self.state_bytes = list(state_bytes)
        self._lengths = trim_lengths(self.page_bytes, self.state_bytes)

    def lengths(self, key_tokens: int) -> list[int] | None:
        return self._lengths


class AttentionKeyTrimPlan(KeyTrimPlan):
    """A key that ends inside a block keeps the rank's rows of the tokens it
    covers; a key ending on a block boundary keeps full pages."""

    def __init__(
        self,
        page_bytes: Sequence[int],
        row_bytes: int,
        tokens_per_page: int,
        block_size: int,
        dcp_rank: int,
    ):
        super().__init__(page_bytes)
        if not (
            row_bytes > 0
            and tokens_per_page > 0
            and all(page == row_bytes * tokens_per_page for page in self.page_bytes)
            and block_size % tokens_per_page == 0
            and 0 <= dcp_rank < block_size // tokens_per_page
        ):
            raise ValueError(
                f"Inconsistent attention trim plan: row {row_bytes} B, "
                f"{tokens_per_page} tokens per page, pages {set(self.page_bytes)} "
                f"B, block {block_size} tokens, DCP rank {dcp_rank}."
            )
        self.row_bytes = row_bytes
        self.tokens_per_page = tokens_per_page
        self.block_size = block_size
        self.dcp_rank = dcp_rank

    def lengths(self, key_tokens: int) -> list[int] | None:
        num_rows = dcp_rank_num_tokens(
            key_tokens, self.block_size, self.tokens_per_page, self.dcp_rank
        )
        if num_rows is None:
            return None
        used = [num_rows * self.row_bytes] * len(self.page_bytes)
        return trim_lengths(self.page_bytes, used)


def _region_index(
    cache: torch.Tensor,
    num_blocks: int,
    layout: RankLocalStoreLayout,
    layer_name: str,
) -> int:
    """Index of the region ``RankLocalStoreLayout.register_kv_caches``
    registered for this (kernel-block grouped) layer cache."""
    if not is_non_overlapping_and_dense(cache[0]):
        raise NotImplementedError(
            f"{layer_name}: Store value trimming does not support per-head "
            "KV cache regions."
        )
    storage = cache.untyped_storage()
    if cache.stride(0) * cache.element_size() * num_blocks == storage.nbytes():
        addr = storage.data_ptr()
    else:
        addr = cache.data_ptr()
    if addr not in layout.kv_caches_base_addr:
        raise ValueError(f"{layer_name}: KV cache region {addr:#x} is not registered.")
    return layout.kv_caches_base_addr.index(addr)


def _build_state_plan(
    group: KVCacheGroupSpec,
    spec: MambaSpec,
    layout: RankLocalStoreLayout,
    kv_caches: Mapping[str, torch.Tensor | list[torch.Tensor]],
    num_blocks: int,
) -> KeyTrimPlan:
    page_bytes = layout.block_len
    state_bytes = [0] * len(page_bytes)
    for name in group.layer_names:
        value = kv_caches[name]
        cache = group_kernel_blocks(
            value if isinstance(value, torch.Tensor) else value[0], num_blocks
        )
        region = _region_index(cache, num_blocks, layout, name)
        region_addr = layout.kv_caches_base_addr[region]
        if cache.stride(0) * cache.element_size() != page_bytes[region]:
            raise ValueError(
                f"{name}: block stride {cache.stride(0) * cache.element_size()} "
                f"B differs from the registered page {page_bytes[region]} B."
            )
        block = cache[0]
        last_element = sum(
            (size - 1) * stride for size, stride in zip(block.shape, block.stride())
        )
        extent = (last_element + 1) * block.element_size()
        end = block.data_ptr() - region_addr + extent
        if (
            block.data_ptr() != region_addr
            or extent != spec.state_content_size_bytes
            or end > page_bytes[region]
        ):
            raise ValueError(
                f"{name}: state view [{block.data_ptr() - region_addr}, {end}) is "
                f"not [0, {spec.state_content_size_bytes}) of its "
                f"{page_bytes[region]} B page."
            )
        state_bytes[region] = max(state_bytes[region], end)
    return StateKeyTrimPlan(page_bytes, state_bytes)


def _build_attention_plan(
    group: KVCacheGroupSpec,
    db: ChunkedTokenDatabase,
    kv_caches: Mapping[str, torch.Tensor | list[torch.Tensor]],
    num_blocks: int,
    dcp_size: int,
    dcp_rank: int,
    cp_kv_cache_interleave_size: int,
) -> KeyTrimPlan:
    layout = db.store_layout
    assert isinstance(layout, RankLocalStoreLayout)
    page_bytes = layout.block_len
    used = [False] * len(page_bytes)
    page_rows: tuple[int, int] | None = None
    for name in group.layer_names:
        value = kv_caches[name]
        assert isinstance(value, torch.Tensor)
        cache = group_kernel_blocks(value, num_blocks)
        region = _region_index(cache, num_blocks, layout, name)
        # [blocks, kernel blocks, heads, tokens, content]
        blocks = cache if cache.dim() == 5 else cache.unsqueeze(1)
        if blocks.dim() != 5 or cache.data_ptr() != layout.kv_caches_base_addr[region]:
            raise ValueError(f"{name}: unexpected KV cache view {tuple(cache.shape)}.")
        elem = blocks.element_size()
        block_stride, kernel_stride, head_stride, token_stride, content_stride = (
            stride * elem for stride in blocks.stride()
        )
        _, kernel_blocks, num_heads, tokens, content = blocks.shape
        row_bytes = num_heads * content * elem
        # Each token's row (all heads) must be contiguous and rows must be in
        # token order, so the first n rows of a page are its first n tokens.
        if not (
            content_stride == elem
            and (num_heads == 1 or head_stride == content * elem)
            and token_stride == row_bytes
            and (kernel_blocks == 1 or kernel_stride == tokens * row_bytes)
            and block_stride == page_bytes[region]
            and kernel_blocks * tokens * row_bytes == page_bytes[region]
        ):
            raise NotImplementedError(
                f"{name}: Store value trimming needs token-major pages, got "
                f"shape {tuple(blocks.shape)} with byte strides "
                f"{tuple(stride * elem for stride in blocks.stride())}."
            )
        if page_rows is None:
            page_rows = (row_bytes, kernel_blocks * tokens)
        elif page_rows != (row_bytes, kernel_blocks * tokens):
            raise ValueError(f"{name}: layers of one group disagree on the page rows.")
        used[region] = True
    if page_rows is None or not all(used):
        raise NotImplementedError(
            "Store value trimming needs every registered region to hold a "
            f"layer of each attention group ({sum(used)}/{len(used)} do)."
        )
    row_bytes, tokens_per_page = page_rows
    if dcp_size > 1:
        if (
            cp_kv_cache_interleave_size != tokens_per_page
            or db.block_size != tokens_per_page * dcp_size
        ):
            raise NotImplementedError(
                "Store value trimming with DCP needs cp_kv_cache_interleave_size "
                f"({cp_kv_cache_interleave_size}) equal to the tokens per page "
                f"({tokens_per_page}) and a block of tokens per page x DCP "
                f"({db.block_size} vs {tokens_per_page} x {dcp_size})."
            )
    elif db.block_size != tokens_per_page:
        raise ValueError(
            f"Block size {db.block_size} differs from the {tokens_per_page} "
            "tokens per page."
        )
    return AttentionKeyTrimPlan(
        page_bytes, row_bytes, tokens_per_page, db.block_size, dcp_rank
    )


def build_key_trim_plans(
    kv_caches: Mapping[str, torch.Tensor | list[torch.Tensor]],
    kv_cache_groups: Sequence[KVCacheGroupSpec],
    token_dbs: Sequence[ChunkedTokenDatabase],
    num_blocks: int,
    *,
    pcp_size: int,
    dcp_size: int,
    dcp_rank: int,
    cp_kv_cache_interleave_size: int,
) -> list[KeyTrimPlan | None]:
    """One plan per KV-cache group, from the registered KV cache views.

    Groups that do not take part in prefix caching get ``None``. Raises when
    a layout is not one whose trimmed bytes are known to be unused.
    """
    if pcp_size != 1:
        raise NotImplementedError(
            "Store value trimming does not support prefill context parallelism."
        )
    plans: list[KeyTrimPlan | None] = []
    for group, db in zip(kv_cache_groups, token_dbs, strict=True):
        layout = db.store_layout
        if type(layout) is not RankLocalStoreLayout:
            raise NotImplementedError(
                "Store value trimming needs the rank-local Store layout, got "
                f"{type(layout).__name__}."
            )
        bases = layout.kv_caches_base_addr
        if not bases or len(layout.block_len) != len(bases):
            raise ValueError("Every Store region needs exactly one page size.")
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            raise NotImplementedError(
                "Store value trimming does not support UniformTypeKVCacheSpecs."
            )
        plan: KeyTrimPlan | None
        if not spec.prefix_cacheable:
            plan = None
        elif isinstance(spec, MambaSpec):
            plan = _build_state_plan(group, spec, layout, kv_caches, num_blocks)
        elif isinstance(spec, AttentionSpec) and not isinstance(spec, _WINDOW_SPECS):
            plan = _build_attention_plan(
                group,
                db,
                kv_caches,
                num_blocks,
                dcp_size,
                dcp_rank,
                cp_kv_cache_interleave_size,
            )
        else:
            raise NotImplementedError(
                f"Store value trimming does not support {type(spec).__name__}."
            )
        plans.append(plan)
    return plans

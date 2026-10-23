# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Copy-on-write block copies on KV caches whose cache groups overlay pages.

In hybrid models the pages of different KV cache groups start at the same
address, e.g. a Mamba state page and an attention page. A block copy must
cover the whole page, whichever group's view comes first in the runner's list.
"""

import pytest
import torch

from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheLayout,
    KVCacheSpec,
    KVCacheTensor,
    MambaSpec,
)
from vllm.v1.worker.utils import allocate_kv_cache, copy_kv_cache_blocks_inplace

BLOCK_SIZE = 16
NUM_BLOCKS = 6
COPIES = [
    KVCacheBlockCopy(src_block_id=1, dst_block_id=4),
    KVCacheBlockCopy(src_block_id=3, dst_block_id=0),
]
DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA"),
    ),
]


def _attention_spec(page_size_padded: int | None = None) -> FullAttentionSpec:
    # 2 heads x 16 tokens x (8 + 8) bf16 values: a 1024-byte page.
    return FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.bfloat16,
        page_size_padded=page_size_padded,
    )


def _mamba_spec(conv_width: int, page_size_padded: int | None = None) -> MambaSpec:
    return MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((3, conv_width), (2, 8, 8)),
        dtypes=(torch.bfloat16, torch.float32),
        page_size_padded=page_size_padded,
    )


def _random_bytes(num_bytes: int, device: str) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    data = torch.randint(0, 256, (num_bytes,), dtype=torch.uint8, generator=generator)
    return data.to(device)


def _backing_bytes(cache: torch.Tensor) -> torch.Tensor:
    raw = torch.empty(0, dtype=torch.uint8, device=cache.device)
    return raw.set_(cache.untyped_storage())


def _allocate(
    groups: list[tuple[list[str], KVCacheSpec]],
    page: int,
    layout: KVCacheLayout,
    num_layers: int,
    kernel_block_sizes: list[int],
    device: str,
) -> tuple[dict[str, torch.Tensor], int, int]:
    """Allocate one KV cache tensor per group, all overlaid on one backing."""
    if layout.is_layer_compact:
        layer_stride, block_stride = NUM_BLOCKS * page, page
    else:
        layer_stride, block_stride = page, num_layers * page
    config = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[
            KVCacheTensor(
                size=NUM_BLOCKS * num_layers * page,
                layers=names,
                layer_stride=layer_stride,
                block_stride=block_stride,
            )
            for names, _ in groups
        ],
        kv_cache_groups=[KVCacheGroupSpec(names, spec) for names, spec in groups],
    )
    caches = allocate_kv_cache(config, torch.device(device), layout, kernel_block_sizes)
    return caches, layer_stride, block_stride


def _expected_after_copies(
    before: torch.Tensor,
    copies: list[KVCacheBlockCopy],
    num_layers: int,
    layer_stride: int,
    block_stride: int,
    num_bytes: int,
) -> torch.Tensor:
    """``before`` with the first ``num_bytes`` of each copied page replaced."""
    expected = before.clone()
    for layer in range(num_layers):
        for copy in copies:
            dst = layer * layer_stride + copy.dst_block_id * block_stride
            src = layer * layer_stride + copy.src_block_id * block_stride
            expected[dst : dst + num_bytes] = before[src : src + num_bytes]
    return expected


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("attention_first", [False, True])
@pytest.mark.parametrize(
    "layout,num_layers,mamba_is_larger,attention_kernel_block_size",
    [
        (KVCacheLayout.LBHNC, 2, False, BLOCK_SIZE),
        (KVCacheLayout.LBHNC, 2, False, BLOCK_SIZE // 2),
        (KVCacheLayout.LBHNC, 2, True, BLOCK_SIZE),
        (KVCacheLayout.LBHNC, 1, False, BLOCK_SIZE),
        (KVCacheLayout.BLHNC, 2, False, BLOCK_SIZE),
        (KVCacheLayout.BLHNC, 2, True, BLOCK_SIZE),
    ],
)
def test_copy_overlaid_hybrid_pages_in_full(
    layout: KVCacheLayout,
    num_layers: int,
    mamba_is_larger: bool,
    attention_kernel_block_size: int,
    attention_first: bool,
    device: str,
):
    """Every byte of a copied page is copied, whichever overlaid view is
    listed first and whichever group's content is longer."""
    if mamba_is_larger:
        mamba = _mamba_spec(conv_width=100)
        attention = _attention_spec(page_size_padded=mamba.page_size_bytes)
        assert mamba.state_content_size_bytes > attention.unpadded_page_size_bytes
    else:
        attention = _attention_spec()
        mamba = _mamba_spec(conv_width=20, page_size_padded=attention.page_size_bytes)
        assert mamba.state_content_size_bytes < attention.unpadded_page_size_bytes
    page = attention.page_size_bytes
    assert mamba.page_size_bytes == page

    mamba_layers = [f"layers.{i}.mamba" for i in range(num_layers)]
    attention_layers = [f"layers.{i}.attn" for i in range(num_layers)]
    caches, layer_stride, block_stride = _allocate(
        [(mamba_layers, mamba), (attention_layers, attention)],
        page,
        layout,
        num_layers,
        [BLOCK_SIZE, attention_kernel_block_size],
        device,
    )
    for mamba_layer, attention_layer in zip(mamba_layers, attention_layers):
        assert caches[mamba_layer].data_ptr() == caches[attention_layer].data_ptr()
    raw = _backing_bytes(caches[mamba_layers[0]])
    raw.copy_(_random_bytes(raw.numel(), device))
    before = raw.clone()

    first, second = (
        (attention_layers, mamba_layers)
        if attention_first
        else (mamba_layers, attention_layers)
    )
    kv_caches = [caches[name] for name in first + second]
    copy_kv_cache_blocks_inplace(kv_caches, NUM_BLOCKS, COPIES)

    expected = _expected_after_copies(
        before, COPIES, num_layers, layer_stride, block_stride, page
    )
    assert torch.equal(raw, expected)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "copies",
    [
        [KVCacheBlockCopy(src_block_id=1, dst_block_id=4)],
        # Block 2 is both a destination and a source: it is read first.
        [
            KVCacheBlockCopy(src_block_id=0, dst_block_id=2),
            KVCacheBlockCopy(src_block_id=2, dst_block_id=5),
        ],
    ],
)
def test_copy_overlaid_views_without_covering_view(
    copies: list[KVCacheBlockCopy], device: str
):
    """If no overlaid view covers the others, each is copied, all as one copy,
    and bytes outside every view are left alone."""
    page = 16
    # A tail past the last block keeps the copy off the whole-storage path.
    raw = _random_bytes(NUM_BLOCKS * page + 8, device)
    before = raw.clone()
    # Bytes 0-3 of each page and bytes {0, 1, 8, 9}: neither covers the other.
    head = raw.as_strided((NUM_BLOCKS, 4), (page, 1))
    split = raw.as_strided((NUM_BLOCKS, 2, 2), (page, 8, 1))

    copy_kv_cache_blocks_inplace([head, split], NUM_BLOCKS, copies)

    expected = before.clone()
    offsets = torch.tensor([0, 1, 2, 3, 8, 9], device=device)
    for copy in copies:
        expected[copy.dst_block_id * page + offsets] = before[
            copy.src_block_id * page + offsets
        ]
    assert torch.equal(raw, expected)


@pytest.mark.parametrize("device", DEVICES)
def test_copy_skips_view_with_other_block_stride(device: str):
    """A view at a page's address that does not index scheduler blocks with
    the page's block stride (here the raw backing tensor) is not a page."""
    num_layers = 2
    attention = _attention_spec()
    page = attention.page_size_bytes
    layers = [f"layers.{i}.attn" for i in range(num_layers)]
    caches, layer_stride, block_stride = _allocate(
        [(layers, attention)],
        page,
        KVCacheLayout.LBHNC,
        num_layers,
        [BLOCK_SIZE],
        device,
    )
    raw = _backing_bytes(caches[layers[0]])
    raw.copy_(_random_bytes(raw.numel(), device))
    before = raw.clone()

    kv_caches = [caches[name] for name in layers] + [raw]
    copy_kv_cache_blocks_inplace(kv_caches, NUM_BLOCKS, COPIES)

    expected = _expected_after_copies(
        before, COPIES, num_layers, layer_stride, block_stride, page
    )
    assert torch.equal(raw, expected)


@pytest.mark.parametrize("device", DEVICES)
def test_copy_single_view_leaves_page_padding(device: str):
    """An address with a single view is copied through that view, so padding
    at the end of a page is not copied."""
    num_layers = 2
    attention = _attention_spec(page_size_padded=1024 + 64)
    page = attention.page_size_bytes
    layers = [f"layers.{i}.attn" for i in range(num_layers)]
    caches, layer_stride, block_stride = _allocate(
        [(layers, attention)],
        page,
        KVCacheLayout.LBHNC,
        num_layers,
        [BLOCK_SIZE],
        device,
    )
    raw = _backing_bytes(caches[layers[0]])
    raw.copy_(_random_bytes(raw.numel(), device))
    before = raw.clone()

    copy_kv_cache_blocks_inplace([caches[name] for name in layers], NUM_BLOCKS, COPIES)

    expected = _expected_after_copies(
        before,
        COPIES,
        num_layers,
        layer_stride,
        block_stride,
        attention.unpadded_page_size_bytes,
    )
    assert torch.equal(raw, expected)

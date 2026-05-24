"""Tests for sv_pgs.ld_blocks."""

from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.ld_blocks import (
    assign_ld_blocks,
    block_partition,
    load_ld_blocks,
    normalize_chromosome,
)


def test_load_returns_1703_blocks() -> None:
    blocks = load_ld_blocks(build="hg38", ancestry="EUR")
    assert blocks.shape == (1703, 3)
    assert blocks.dtype == np.int64
    # Coverage: autosomes 1..22, no sex chromosomes in LDetect EUR.
    chroms = np.unique(blocks[:, 0])
    assert chroms.min() == 1
    assert chroms.max() == 22
    assert set(chroms.tolist()) == set(range(1, 23))


def test_load_returns_sorted_non_overlapping() -> None:
    blocks = load_ld_blocks()
    # Sorted by (chrom, start)
    chrom_then_start = np.lexsort((blocks[:, 1], blocks[:, 0]))
    assert np.array_equal(chrom_then_start, np.arange(blocks.shape[0]))
    # End > start
    assert np.all(blocks[:, 2] > blocks[:, 1])


def test_load_is_cached_returns_same_array() -> None:
    a = load_ld_blocks()
    b = load_ld_blocks()
    assert a is b


def test_normalize_chromosome_variants() -> None:
    assert normalize_chromosome("chr1") == 1
    assert normalize_chromosome("1") == 1
    assert normalize_chromosome("CHR1") == 1
    assert normalize_chromosome("Chr1") == 1
    assert normalize_chromosome(" chr1 ") == 1
    assert normalize_chromosome("X") == 23
    assert normalize_chromosome("chrX") == 23
    assert normalize_chromosome("Y") == 24
    assert normalize_chromosome("MT") == 25
    assert normalize_chromosome("chrM") == 25
    assert normalize_chromosome("M") == 25
    with pytest.raises(ValueError):
        normalize_chromosome("chr0")
    with pytest.raises(ValueError):
        normalize_chromosome("foo")


def test_assign_synthetic_variants_inside_blocks() -> None:
    blocks = load_ld_blocks()
    # Pick first three EUR blocks on chr1; use midpoints.
    chr1_blocks = blocks[blocks[:, 0] == 1][:3]
    mids = (chr1_blocks[:, 1] + chr1_blocks[:, 2]) // 2
    chroms = np.array(["chr1", "1", "CHR1"])
    block_ids = assign_ld_blocks(chroms, mids)
    # Block IDs must be the absolute indices of those three blocks in the
    # sorted block table.
    n_blocks = blocks.shape[0]
    expected = np.flatnonzero(blocks[:, 0] == 1)[:3]
    assert np.array_equal(block_ids, expected)
    # All within the canonical range.
    assert np.all(block_ids < n_blocks)


def test_assign_chromosome_label_normalization_equivalence() -> None:
    blocks = load_ld_blocks()
    chr1_first = blocks[blocks[:, 0] == 1][0]
    pos = (chr1_first[1] + chr1_first[2]) // 2
    for label in ["chr1", "1", "CHR1", "Chr1"]:
        ids = assign_ld_blocks(np.array([label]), np.array([pos]))
        assert ids[0] == np.flatnonzero(blocks[:, 0] == 1)[0]


def test_assign_unmapped_variants_get_singleton_ids() -> None:
    blocks = load_ld_blocks()
    n_blocks = blocks.shape[0]
    # Pick positions outside any block: position 1 on chr1 (LDetect EUR
    # starts at 10583), and chr X which is not in EUR LDetect.
    chroms = np.array(["chr1", "chrX", "chrY"])
    positions = np.array([1, 1_000_000, 1_000_000])
    ids = assign_ld_blocks(chroms, positions)
    # All three are unmapped and get unique singleton IDs >= n_blocks.
    assert np.all(ids >= n_blocks)
    assert len(set(ids.tolist())) == 3


def test_assign_mixed_mapped_and_unmapped() -> None:
    blocks = load_ld_blocks()
    n_blocks = blocks.shape[0]
    chr1_first = blocks[blocks[:, 0] == 1][0]
    in_pos = (chr1_first[1] + chr1_first[2]) // 2
    chroms = np.array(["chr1", "chrX", "chr1"])
    positions = np.array([in_pos, 50_000, 1])  # mapped, unmapped, unmapped
    ids = assign_ld_blocks(chroms, positions)
    assert ids[0] < n_blocks
    assert ids[1] >= n_blocks
    assert ids[2] >= n_blocks
    assert ids[1] != ids[2]


def test_block_partition_round_trip() -> None:
    blocks = load_ld_blocks()
    # Construct a small synthetic variant set spanning a few blocks.
    chr1_blocks = blocks[blocks[:, 0] == 1][:5]
    chr2_blocks = blocks[blocks[:, 0] == 2][:3]
    chrom_labels = []
    positions = []
    for blk in chr1_blocks:
        chrom_labels += ["chr1", "1"]
        positions += [int(blk[1]) + 100, int(blk[2]) - 100]
    for blk in chr2_blocks:
        chrom_labels += ["chr2"]
        positions += [(int(blk[1]) + int(blk[2])) // 2]
    chroms = np.asarray(chrom_labels)
    pos = np.asarray(positions, dtype=np.int64)
    ids = assign_ld_blocks(chroms, pos)
    partition = block_partition(ids)
    # Reconstruct ids from the partition.
    reconstructed = np.empty_like(ids)
    for bid, idx in partition.items():
        reconstructed[idx] = bid
    assert np.array_equal(reconstructed, ids)
    # Every variant assigned exactly once.
    seen = np.concatenate(list(partition.values()))
    assert seen.size == ids.size
    assert set(seen.tolist()) == set(range(ids.size))


def test_block_partition_empty() -> None:
    assert block_partition(np.array([], dtype=np.int64)) == {}


def test_assign_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        assign_ld_blocks(np.array(["chr1", "chr2"]), np.array([1]))


def test_load_unknown_build_raises() -> None:
    with pytest.raises(NotImplementedError):
        load_ld_blocks(build="hg19")
    with pytest.raises(NotImplementedError):
        load_ld_blocks(ancestry="AFR")

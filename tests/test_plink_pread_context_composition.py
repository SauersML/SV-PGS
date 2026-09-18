"""The direct PLINK pread path must read the BED columns/samples the wrappers select.

``_gpu_plink_pread_transpose_matmul_direct`` preads raw BED records and decodes
them on the GPU. The AoU loader wraps a PLINK source in an
``IndexedRawGenotypeMatrix`` when it drops within-source duplicates (and in a
``RowSubsetRawGenotypeMatrix`` for the sample split), so the pread context has
to translate the caller's column/row coordinates to BED positions. Reading
dataset column ``j`` as BED variant ``j`` silently decodes a neighbouring
variant. No GPU required: the GPU kernel is bit-identical to
``_decode_packed_bytes_reference`` (see test_plink_gpu_decode_equivalence.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from sv_pgs.genotype import (
    ConcatenatedRawGenotypeMatrix,
    IndexedRawGenotypeMatrix,
    PlinkRawGenotypeMatrix,
    RowSubsetRawGenotypeMatrix,
    _decode_packed_bytes_reference,
    _resolve_pure_single_plink_pread_context,
)
from sv_pgs.plink import PLINK_MISSING_INT8, to_bed


def _write_bed(path: Path, dosage: np.ndarray) -> None:
    sample_count, variant_count = dosage.shape
    sample_ids = [f"s{index}" for index in range(sample_count)]
    to_bed(
        path,
        dosage,
        properties={
            "fid": sample_ids,
            "iid": sample_ids,
            "sid": [f"v{index}" for index in range(variant_count)],
            "chromosome": ["1"] * variant_count,
            "bp_position": list(range(1, variant_count + 1)),
        },
    )


def _plink_leaf(tmp_path: Path) -> PlinkRawGenotypeMatrix:
    rng = np.random.default_rng(7)
    dosage = rng.integers(0, 3, size=(11, 7)).astype(np.float32)
    dosage[rng.random(dosage.shape) < 0.1] = np.nan
    bed_path = tmp_path / "cohort.bed"
    _write_bed(bed_path, dosage)
    return PlinkRawGenotypeMatrix(
        bed_path=bed_path,
        sample_indices=np.array([0, 2, 3, 5, 7, 8, 10]),
        variant_count=dosage.shape[1],
        total_sample_count=dosage.shape[0],
    )


def _pread_decode(raw, selected: np.ndarray) -> np.ndarray:
    context = _resolve_pure_single_plink_pread_context(raw)
    assert context is not None
    bytes_per_variant = (int(context.iid_count) + 3) // 4
    payload = context.reader._pread_indexed_variant_payload(
        context.bed_variant_positions(selected),
        bytes_per_variant=bytes_per_variant,
    )
    return _decode_packed_bytes_reference(
        payload,
        bytes_per_variant=bytes_per_variant,
        sample_indices=context.bed_sample_indices,
        n_variants=int(selected.shape[0]),
    )


def _as_int8_with_missing(values: np.ndarray) -> np.ndarray:
    return np.where(np.isnan(values), PLINK_MISSING_INT8, values).astype(np.int8)


def test_pread_context_composes_indexed_and_row_subset_wrappers(tmp_path: Path) -> None:
    leaf = _plink_leaf(tmp_path)
    raw = RowSubsetRawGenotypeMatrix(
        child=ConcatenatedRawGenotypeMatrix(
            children=(
                RowSubsetRawGenotypeMatrix(
                    child=IndexedRawGenotypeMatrix(
                        child=IndexedRawGenotypeMatrix(child=leaf, selected_columns=np.array([6, 5, 3, 2, 1, 0])),
                        selected_columns=np.array([0, 2, 3, 5]),
                    ),
                    row_indices=np.array([6, 4, 2, 1, 0]),
                ),
            )
        ),
        row_indices=np.array([3, 0, 4]),
    )
    selected = np.array([3, 0, 2])

    decoded = _pread_decode(raw, selected)

    expected = _as_int8_with_missing(raw.materialize(selected))
    assert decoded.shape == expected.shape == (3, 3)
    np.testing.assert_array_equal(decoded, expected)


def test_pread_context_without_wrappers_reads_bed_positions_directly(tmp_path: Path) -> None:
    leaf = _plink_leaf(tmp_path)
    selected = np.array([6, 1, 4])

    decoded = _pread_decode(leaf, selected)

    np.testing.assert_array_equal(decoded, _as_int8_with_missing(leaf.materialize(selected)))


def test_pread_context_declines_multi_source_concatenation(tmp_path: Path) -> None:
    leaf = _plink_leaf(tmp_path)
    mixed = ConcatenatedRawGenotypeMatrix(
        children=(leaf, IndexedRawGenotypeMatrix(child=leaf, selected_columns=np.array([0, 1]))),
    )

    assert _resolve_pure_single_plink_pread_context(mixed) is None

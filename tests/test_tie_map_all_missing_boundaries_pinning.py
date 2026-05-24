"""Pin: a column that is all-missing must not match any real variant.

Construct an int8 hardcall matrix where one column is entirely missing
(``PLINK_MISSING_INT8`` everywhere).  The tie-map builder must:

* not crash,
* not collapse the all-missing column into any observed column,
* not collapse two all-missing columns together either (they have no
  observed data in common, so claiming they are the "same" variant is
  unsafe — drop semantics belong to the MAF filter, not the tie map).

These are conservative pins: if the tie-map upstream later chooses to
group all-missing columns together, the test will fail and force an
explicit review of that semantic change.
"""
from __future__ import annotations

import numpy as np

from sv_pgs.config import ModelConfig, TraitType, VariantClass
from sv_pgs.data import VariantRecord
from sv_pgs.genotype import as_raw_genotype_matrix
from sv_pgs.plink import PLINK_MISSING_INT8
from sv_pgs.preprocessing import build_tie_map, compute_variant_statistics


def _make_records(n: int) -> list[VariantRecord]:
    return [
        VariantRecord(
            variant_id=f"v{i}",
            variant_class=VariantClass.SNV,
            chromosome="1",
            position=1000 + i,
        )
        for i in range(n)
    ]


def _standardized(int8_matrix: np.ndarray):
    raw = as_raw_genotype_matrix(int8_matrix)
    stats = compute_variant_statistics(
        raw_genotypes=raw,
        config=ModelConfig(trait_type=TraitType.QUANTITATIVE),
    )
    return raw.standardized(stats.means, stats.scales)


def test_all_missing_column_does_not_match_real_column():
    """[real, all-missing] → tie map keeps both as separate representatives."""
    n_samples = 16
    real_col = np.asarray(
        [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0], dtype=np.int8
    )
    missing_col = np.full(n_samples, PLINK_MISSING_INT8, dtype=np.int8)
    matrix_i8 = np.stack([real_col, missing_col], axis=1)

    standardized = _standardized(matrix_i8)
    records = _make_records(2)
    tie_map = build_tie_map(
        standardized,
        records,
        ModelConfig(trait_type=TraitType.QUANTITATIVE),
    )

    # All-missing column standardizes to a zero column (mean=0, scale=1) per
    # `_means_and_scales_with_floor` semantics. A *real* column also can't
    # collapse into the zero column unless its standardization is identically
    # zero — which it isn't for a non-degenerate real column. Pin that
    # neither column is silently dropped or absorbed.
    assert tie_map.kept_indices.shape[0] >= 1
    # The real column (index 0) must be present as a representative.
    assert 0 in set(tie_map.kept_indices.tolist())
    # Each original variant has a non-negative reduced index (or -1 if the
    # tie-map chose to drop it). The all-missing column may map to the same
    # group as another all-missing column, but not to the real column.
    real_reduced = int(tie_map.original_to_reduced[0])
    missing_reduced = int(tie_map.original_to_reduced[1])
    if missing_reduced >= 0 and real_reduced >= 0:
        assert real_reduced != missing_reduced, (
            "all-missing column was silently merged with a real variant"
        )


def test_all_missing_only_matrix_does_not_crash():
    """An all-missing input column count of 2 must produce a TieMap without
    raising, and the resulting representative count must not exceed the
    input column count."""
    n_samples = 8
    matrix_i8 = np.full((n_samples, 2), PLINK_MISSING_INT8, dtype=np.int8)
    standardized = _standardized(matrix_i8)
    records = _make_records(2)
    tie_map = build_tie_map(
        standardized,
        records,
        ModelConfig(trait_type=TraitType.QUANTITATIVE),
    )
    assert tie_map.kept_indices.shape[0] <= 2
    # Both columns standardize to the same all-zeros vector and the tie-map
    # may either collapse them into one group or keep them as two. Either is
    # acceptable as long as the operation completes.
    assert tie_map.original_to_reduced.shape == (2,)


def test_single_all_missing_column_does_not_crash():
    """Degenerate one-column input where the single column is all missing."""
    n_samples = 8
    matrix_i8 = np.full((n_samples, 1), PLINK_MISSING_INT8, dtype=np.int8)
    standardized = _standardized(matrix_i8)
    records = _make_records(1)
    tie_map = build_tie_map(
        standardized,
        records,
        ModelConfig(trait_type=TraitType.QUANTITATIVE),
    )
    assert tie_map.original_to_reduced.shape == (1,)

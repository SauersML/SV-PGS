"""Pin: per-variant variance with a single sample (n=1).

Preprocessing uses ddof=0 — for n=1 every column has centered_sum_squares=0
which yields scale=0. The low-variance floor kicks in and the scale is
mapped to 1.0 (the documented sentinel) so that downstream divisions do
not blow up. This file pins:

* n=1 with one variant value: ``compute_variant_statistics`` returns a
  finite scale (the 1.0 floor), not zero or NaN.
* n=1 also returns a finite mean (the single sample value).
* The standardized column under the floor sentinel is exactly zero per
  element (``(c - c) / 1.0 == 0``).
"""
from __future__ import annotations

import numpy as np

from sv_pgs.config import ModelConfig, TraitType
from sv_pgs.genotype import as_raw_genotype_matrix
from sv_pgs.preprocessing import compute_variant_statistics


def test_single_sample_yields_floored_scale_and_finite_mean():
    raw = as_raw_genotype_matrix(np.asarray([[1.0]], dtype=np.float32))
    stats = compute_variant_statistics(
        raw_genotypes=raw,
        config=ModelConfig(trait_type=TraitType.QUANTITATIVE),
    )
    assert stats.means.shape == (1,)
    assert stats.scales.shape == (1,)
    assert np.isfinite(float(stats.means[0]))
    assert float(stats.means[0]) == 1.0
    # Var=0 → low-variance branch → scale floored to 1.0.
    assert float(stats.scales[0]) == 1.0


def test_single_sample_standardized_column_is_exactly_zero():
    raw = as_raw_genotype_matrix(np.asarray([[2.0]], dtype=np.float32))
    stats = compute_variant_statistics(
        raw_genotypes=raw,
        config=ModelConfig(trait_type=TraitType.QUANTITATIVE),
    )
    standardized = raw.standardized(stats.means, stats.scales)
    materialized = np.asarray(standardized.materialize(), dtype=np.float64)
    np.testing.assert_array_equal(materialized, np.zeros_like(materialized))


def test_constant_column_multi_sample_also_yields_floored_scale():
    """Constant column with n>1 hits the same low-variance branch."""
    raw = as_raw_genotype_matrix(np.full((8, 1), 1.0, dtype=np.float32))
    stats = compute_variant_statistics(
        raw_genotypes=raw,
        config=ModelConfig(trait_type=TraitType.QUANTITATIVE),
    )
    assert float(stats.scales[0]) == 1.0
    assert float(stats.means[0]) == 1.0

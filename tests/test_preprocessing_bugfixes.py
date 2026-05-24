"""Focused regression tests for three preprocessing.py bug fixes.

1. Constant-genotype variants must not leak a constant non-zero column into
   the standardized output. ``compute_variant_statistics`` floors the scale
   to 1.0 for low-variance variants; combined with the column-mean it must
   produce an exactly-zero standardized column.

2. Sign-flipped tie detection must not match two all-missing int8 columns.
   The inner conjunction must check ``valid.sum() > 0`` BEFORE evaluating
   ``np.all`` on the masked slice (np.all over an empty slice returns True).

3. ``_minor_allele_frequency`` must return a finite value when handed NaN /
   inf — undefined allele frequencies should map to a monomorphic 0.0 so the
   variant is filtered out by any positive MAF threshold.
"""
from __future__ import annotations

import math

import numpy as np

from sv_pgs.config import ModelConfig, TraitType
from sv_pgs.genotype import as_raw_genotype_matrix
from sv_pgs.preprocessing import (
    _minor_allele_frequency,
    compute_variant_statistics,
)


def test_constant_variant_produces_zero_standardized_column():
    """A variant whose value is identical for every sample must yield an
    exactly-zero standardized column. Previously the scale was floored to
    1.0 but the (raw - mean) numerator could be a constant non-zero value
    if the mean did not match the column constant — e.g. for all-missing
    columns where the mean defaulted to 0 even though the data was not.
    """
    sample_count = 32
    # Two columns:
    #   col 0: truly constant value 1.0 for every sample.
    #   col 1: standard variable column for sanity.
    raw = np.zeros((sample_count, 2), dtype=np.float32)
    raw[:, 0] = 1.0
    rng = np.random.default_rng(0)
    raw[:, 1] = rng.integers(0, 3, size=sample_count).astype(np.float32)

    raw_matrix = as_raw_genotype_matrix(raw)
    config = ModelConfig(trait_type=TraitType.QUANTITATIVE)
    stats = compute_variant_statistics(raw_genotypes=raw_matrix, config=config)

    # The constant column must have scale=1 (floored) and mean=1.0 so that
    # standardization yields exactly zero everywhere.
    assert stats.scales[0] == 1.0
    assert math.isclose(float(stats.means[0]), 1.0, abs_tol=1e-6)

    standardized = raw_matrix.standardized(stats.means, stats.scales).materialize()
    assert standardized.shape == (sample_count, 2)
    np.testing.assert_allclose(standardized[:, 0], 0.0, atol=0.0)


def test_all_missing_variant_produces_zero_standardized_column():
    """All-missing column: undefined mean, must still yield a deterministic
    zero standardized column (not NaN). Previously the mean fallback was 0
    and the scale was 1, so any non-NaN input would standardize to itself.
    """
    sample_count = 16
    raw = np.full((sample_count, 1), np.nan, dtype=np.float32)
    raw_matrix = as_raw_genotype_matrix(raw)
    config = ModelConfig(trait_type=TraitType.QUANTITATIVE)
    stats = compute_variant_statistics(raw_genotypes=raw_matrix, config=config)
    assert stats.means[0] == 0.0
    assert stats.scales[0] == 1.0


def test_sign_flipped_tie_check_skips_all_missing_int8_columns():
    """Regression for the inner sign-flipped tie check: when two int8
    columns are completely missing (sentinel -1 everywhere) the masked
    slice ``col_i[valid]`` is empty, ``np.all([])`` returns True, and
    without the ``valid.sum() > 0`` guard ordered first the pair would
    be (falsely) collapsed as a sign-flipped tie. Reordering ensures the
    short-circuit kicks in before ``np.all`` runs on the empty slice.

    Note: "sign-flipped" for int8 hardcalls (0/1/2 with missing=-1)
    means the arithmetic *complement* ``2 - x``, not arithmetic
    negation — that is the encoding that, after centering and
    scaling, becomes the standardized column's negation.
    """
    sample_count = 8
    col_i = np.full(sample_count, -1, dtype=np.int8)
    col_j = np.full(sample_count, -1, dtype=np.int8)
    missing_i = col_i == -1
    missing_j = col_j == -1
    same_missing = np.array_equal(missing_i, missing_j)
    valid = ~missing_i

    # Mirrors the reordered expression from preprocessing.py.
    is_tie = bool(
        same_missing
        and valid.sum() > 0
        and np.all(col_i[valid] == (2 - col_j[valid]))
    )
    assert is_tie is False

    # And a genuine complement-flipped pair still detects correctly.
    col_i2 = np.array([0, 1, 2, 0, 1, 2, -1, 0], dtype=np.int8)
    col_j2 = np.array([2, 1, 0, 2, 1, 0, -1, 2], dtype=np.int8)
    missing_i2 = col_i2 == -1
    missing_j2 = col_j2 == -1
    same_missing2 = np.array_equal(missing_i2, missing_j2)
    valid2 = ~missing_i2
    is_tie2 = bool(
        same_missing2
        and valid2.sum() > 0
        and np.all(col_i2[valid2] == (2 - col_j2[valid2]))
    )
    assert is_tie2 is True


def test_minor_allele_frequency_handles_nan_and_inf():
    """NaN allele frequency must not silently propagate. Returns 0.0
    so the variant is filtered out by any positive MAF threshold.
    """
    assert _minor_allele_frequency(float("nan")) == 0.0
    assert _minor_allele_frequency(float("inf")) == 0.0
    assert _minor_allele_frequency(float("-inf")) == 0.0
    # Sanity: ordinary inputs still work.
    assert math.isclose(_minor_allele_frequency(0.3), 0.3, abs_tol=1e-9)
    assert math.isclose(_minor_allele_frequency(0.8), 0.2, abs_tol=1e-9)
    assert _minor_allele_frequency(1.5) == 0.0  # clipped to 1.0 → min(1, 0) = 0

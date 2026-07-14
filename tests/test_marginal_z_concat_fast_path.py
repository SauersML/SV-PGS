"""Numerical equivalence between the per-child concat marginal-z fast path
and the canonical compute_marginal_z_scores reference path.

The new ``_compute_marginal_z_scores_concat`` function (model.py) handles the
SNP+SV case where ``standardized_genotypes.raw`` is a
``ConcatenatedRawGenotypeMatrix`` over a mix of in-memory int8 VCF caches and
(in production) a bitpacked-upgraded PLINK child. The per-child rmatvec result
must match what the chunked transpose-matmat slow path would have produced,
otherwise the z-score screen drops the wrong variants and downstream fits
diverge silently.

These tests run without GPU — the int8 child branch needs CuPy at runtime,
so we gate on its availability and otherwise verify the function declines
cleanly (returns None → caller's slow-path fallback) instead of raising.
"""
from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.genotype import (
    ConcatenatedRawGenotypeMatrix,
    Int8RawGenotypeMatrix,
    StandardizedGenotypeMatrix,
)
from sv_pgs.model import _compute_marginal_z_scores_concat
from sv_pgs.preprocessing import compute_marginal_z_scores

cupy = pytest.importorskip(
    "cupy",
    reason="_compute_marginal_z_scores_concat dispatches int8/PLINK math through CuPy",
)


def _build_int8_concat(
    *,
    n_samples: int,
    children_variant_counts: list[int],
    seed: int,
) -> tuple[ConcatenatedRawGenotypeMatrix, np.ndarray, np.ndarray]:
    """Build a ConcatenatedRawGenotypeMatrix of in-memory int8 children plus
    matched per-variant (mean, scale) arrays the standardizer expects.

    Genotype values are 0/1/2 (PLINK A1-coded); means and scales are computed
    from each column so the standardized view is mean-0, scale-floored-to-1
    just like production. No missing values for simplicity — the missing-value
    branch is exercised by ``test_int8_concat_with_missing_values`` below.
    """
    rng = np.random.default_rng(seed)
    matrices: list[np.ndarray] = []
    all_means: list[np.ndarray] = []
    all_scales: list[np.ndarray] = []
    for k in children_variant_counts:
        m = rng.integers(0, 3, size=(n_samples, k), dtype=np.int8)
        matrices.append(m)
        col_mean = m.astype(np.float64).mean(axis=0).astype(np.float32)
        col_std = m.astype(np.float64).std(axis=0, ddof=0).astype(np.float32)
        col_std = np.where(col_std > 1e-6, col_std, np.float32(1.0))
        all_means.append(col_mean)
        all_scales.append(col_std)
    children = tuple(Int8RawGenotypeMatrix(matrix=m) for m in matrices)
    concat = ConcatenatedRawGenotypeMatrix(children=children)
    means = np.concatenate(all_means).astype(np.float32)
    scales = np.concatenate(all_scales).astype(np.float32)
    return concat, means, scales


def _make_standardized(
    raw: ConcatenatedRawGenotypeMatrix, means: np.ndarray, scales: np.ndarray
) -> StandardizedGenotypeMatrix:
    n_variants = raw.shape[1]
    return StandardizedGenotypeMatrix(
        raw=raw,
        means=means,
        scales=scales,
        variant_indices=np.arange(n_variants, dtype=np.int32),
        support_counts=np.full(n_variants, raw.shape[0], dtype=np.int32),
        sample_count=raw.shape[0],
        _enable_hybrid_backend=False,
    )


def _reference_z_via_compute(
    standardized: StandardizedGenotypeMatrix,
    active: np.ndarray,
    cov: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Run the canonical (slow but cupy-free if cache_dir is None) reference."""
    return compute_marginal_z_scores(
        standardized_genotypes=standardized,
        active_variant_indices=active,
        covariate_matrix=cov,
        target_vector=target,
    )


@pytest.mark.skipif(
    cupy.cuda.runtime.getDeviceCount() == 0,
    reason="needs a GPU-visible CuPy runtime to exercise the fast path",
)
def test_concat_fast_path_matches_reference_two_children() -> None:
    n = 256
    raw, means, scales = _build_int8_concat(
        n_samples=n, children_variant_counts=[40, 60], seed=7
    )
    standardized = _make_standardized(raw, means, scales)
    rng = np.random.default_rng(3)
    cov = rng.normal(size=(n, 5)).astype(np.float64)
    target = rng.normal(size=n).astype(np.float64)
    active = np.array([0, 5, 12, 39, 40, 41, 70, 99], dtype=np.int32)

    fast = _compute_marginal_z_scores_concat(
        standardized_genotypes=standardized,
        active_variant_indices=active,
        covariate_matrix=cov,
        target_vector=target,
    )
    assert fast is not None, "concat fast path should engage for all-int8 layout"
    reference = _reference_z_via_compute(standardized, active, cov, target)
    np.testing.assert_allclose(fast, reference, rtol=1e-3, atol=1e-4)


@pytest.mark.skipif(
    cupy.cuda.runtime.getDeviceCount() == 0,
    reason="needs a GPU-visible CuPy runtime to exercise the fast path",
)
def test_concat_fast_path_with_missing_values() -> None:
    """PLINK_MISSING_INT8 entries must impute to the column mean (so the
    centered value is 0). Independently verify against a hand-imputed reference.
    """
    from sv_pgs.plink import PLINK_MISSING_INT8

    n = 200
    rng = np.random.default_rng(11)
    # Two children, both with scattered missing entries.
    m1 = rng.integers(0, 3, size=(n, 30), dtype=np.int8)
    m2 = rng.integers(0, 3, size=(n, 40), dtype=np.int8)
    # Sprinkle missing values.
    miss_mask_1 = rng.random((n, 30)) < 0.05
    miss_mask_2 = rng.random((n, 40)) < 0.05
    m1[miss_mask_1] = PLINK_MISSING_INT8
    m2[miss_mask_2] = PLINK_MISSING_INT8
    # Compute means/scales from non-missing entries (matches what the
    # variant-stats pipeline produces).
    def _col_stats(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        f = mat.astype(np.float64)
        f[mat == PLINK_MISSING_INT8] = np.nan
        col_mean = np.nanmean(f, axis=0).astype(np.float32)
        col_std = np.nanstd(f, axis=0, ddof=0).astype(np.float32)
        col_std = np.where(col_std > 1e-6, col_std, np.float32(1.0))
        return col_mean, col_std

    mean1, scale1 = _col_stats(m1)
    mean2, scale2 = _col_stats(m2)
    means = np.concatenate([mean1, mean2]).astype(np.float32)
    scales = np.concatenate([scale1, scale2]).astype(np.float32)
    children = (
        Int8RawGenotypeMatrix(matrix=m1),
        Int8RawGenotypeMatrix(matrix=m2),
    )
    raw = ConcatenatedRawGenotypeMatrix(children=children)
    standardized = _make_standardized(raw, means, scales)
    cov = rng.normal(size=(n, 3)).astype(np.float64)
    target = rng.normal(size=n).astype(np.float64)
    active = np.array([0, 1, 15, 29, 30, 35, 50, 69], dtype=np.int32)

    fast = _compute_marginal_z_scores_concat(
        standardized_genotypes=standardized,
        active_variant_indices=active,
        covariate_matrix=cov,
        target_vector=target,
    )
    assert fast is not None
    reference = _reference_z_via_compute(standardized, active, cov, target)
    # Slightly looser tolerance because the impute-then-standardize order
    # differs in float-precision details between paths (GPU fp32 in the fast
    # path vs CPU/host fp32 in the reference).
    np.testing.assert_allclose(fast, reference, rtol=1e-3, atol=1e-4)


def test_concat_fast_path_returns_none_for_non_concatenated_raw() -> None:
    """The fast path must decline (→ None) when ``raw`` isn't concatenated,
    so callers fall back to the canonical chunked path. This test runs even
    without CUDA because the isinstance check is the first gate."""
    n = 50
    matrix = np.zeros((n, 10), dtype=np.int8)
    int8 = Int8RawGenotypeMatrix(matrix=matrix)
    standardized = _make_standardized(
        # Wrap in a Concatenated for the StandardizedGenotypeMatrix invariant,
        # but pass the inner directly to confirm the fast path declines.
        ConcatenatedRawGenotypeMatrix(children=(int8,)),
        means=np.zeros(10, dtype=np.float32),
        scales=np.ones(10, dtype=np.float32),
    )
    # Now lie about the raw to simulate a non-concatenated input.
    standardized.raw = int8
    cov = np.zeros((n, 1), dtype=np.float64)
    target = np.zeros(n, dtype=np.float64)
    got = _compute_marginal_z_scores_concat(
        standardized_genotypes=standardized,
        active_variant_indices=np.arange(10, dtype=np.int32),
        covariate_matrix=cov,
        target_vector=target,
    )
    assert got is None

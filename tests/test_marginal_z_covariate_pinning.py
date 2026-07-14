"""Pin: ``compute_marginal_z_scores`` actually residualizes on covariates.

Without the per-variant denominator correction
    Var(x_j^T y_resid) = sigma2 * x_j^T (I - C(C'C)^{-1}C') x_j
the z-score is inflated for variants in the column-span of the covariates.

Adversarial test:
* Variant X = covariate C (perfect collinearity).
* True target = C only — variant has zero residual effect.
* The denominator must subtract the C-projection, leaving xpx_diag ≈ 0
  for that variant, and the function returns z = 0 (no signal beyond C).
* The naïve denominator sqrt(n*sigma2) would still produce a non-zero z.
"""
from __future__ import annotations

import numpy as np

from sv_pgs.genotype import as_raw_genotype_matrix
from sv_pgs.preprocessing import compute_marginal_z_scores, compute_variant_statistics
from sv_pgs.config import ModelConfig, TraitType


def _build_standardized(raw_array: np.ndarray):
    raw = as_raw_genotype_matrix(raw_array.astype(np.float32))
    stats = compute_variant_statistics(
        raw_genotypes=raw, config=ModelConfig(trait_type=TraitType.QUANTITATIVE),
    )
    return raw.standardized(stats.means, stats.scales)


def test_variant_in_covariate_span_has_near_zero_z():
    """A variant column that lies exactly in the covariate span must give
    z ≈ 0 after residualization — pinning the projection-aware denominator."""
    n = 64
    rng = np.random.default_rng(123)
    # Covariate: a continuous column.
    cov = rng.normal(size=(n, 1)).astype(np.float64)
    # Variant identical to the covariate (perfectly collinear) plus a
    # second, independent variant for sanity comparison.
    independent_variant = rng.normal(size=n).astype(np.float32)
    raw = np.stack([cov[:, 0].astype(np.float32), independent_variant], axis=1)
    target = (cov[:, 0] + 0.01 * rng.normal(size=n)).astype(np.float64)

    standardized = _build_standardized(raw)
    active = np.asarray([0, 1], dtype=np.int32)
    z = compute_marginal_z_scores(
        standardized_genotypes=standardized,
        active_variant_indices=active,
        covariate_matrix=cov.astype(np.float32),
        target_vector=target.astype(np.float32),
    )
    # Collinear variant: residualized signal must be effectively zero.
    assert abs(z[0]) < 1e-3, (
        f"variant in covariate span should give z≈0 (got {z[0]:.6f}); "
        "denominator is not subtracting the covariate projection."
    )
    # Independent variant still uses the standard denominator: finite, real.
    assert np.isfinite(z[1])


def test_zero_covariate_path_yields_standard_marginal_z():
    """When covariate_matrix has zero columns, the function falls back to
    the no-projection denominator (sqrt(n * sigma2)). Pin that branch."""
    n = 64
    rng = np.random.default_rng(7)
    variant = rng.normal(size=n).astype(np.float32)
    target = (0.5 * variant + 0.5 * rng.normal(size=n)).astype(np.float32)
    raw = variant[:, None]
    standardized = _build_standardized(raw)
    active = np.asarray([0], dtype=np.int32)
    empty_cov = np.zeros((n, 0), dtype=np.float32)
    z = compute_marginal_z_scores(
        standardized_genotypes=standardized,
        active_variant_indices=active,
        covariate_matrix=empty_cov,
        target_vector=target,
    )
    assert np.isfinite(z[0])
    # With a true effect of 0.5 (signal-to-noise ~ 1), |z| should be well
    # above the small-N noise floor.
    assert abs(z[0]) > 1.5


def test_empty_active_returns_empty_array():
    """Zero active variants → empty z array (no crash)."""
    n = 16
    cov = np.ones((n, 1), dtype=np.float32)
    standardized = _build_standardized(np.ones((n, 1), dtype=np.float32))
    z = compute_marginal_z_scores(
        standardized_genotypes=standardized,
        active_variant_indices=np.asarray([], dtype=np.int32),
        covariate_matrix=cov,
        target_vector=np.zeros(n, dtype=np.float32),
    )
    assert z.shape == (0,)

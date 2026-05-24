"""Boundary pinning for ``compute_marginal_z_scores`` degenerate inputs.

Cases:

* Rank-deficient covariate matrix (a constant column twice): the helper
  uses ``np.linalg.pinv`` so this must not crash and must still produce
  finite z.
* Single covariate IS the variant: residualization wipes the signal,
  pinning a near-zero z (the projection-aware denominator must subtract
  the C-projection).  This complements the existing similar test but
  uses the variant-as-its-own-covariate framing to catch a different
  failure mode (no separate independent covariate present).
* ``n=2`` samples: too small for a meaningful z, but must not crash and
  must return a finite or zero value.
"""
from __future__ import annotations

import numpy as np

from sv_pgs.config import ModelConfig, TraitType
from sv_pgs.genotype import as_raw_genotype_matrix
from sv_pgs.preprocessing import compute_marginal_z_scores, compute_variant_statistics


def _standardized(raw_array: np.ndarray):
    raw = as_raw_genotype_matrix(raw_array.astype(np.float32))
    stats = compute_variant_statistics(
        raw_genotypes=raw,
        config=ModelConfig(trait_type=TraitType.QUANTITATIVE),
    )
    return raw.standardized(stats.means, stats.scales)


def test_rank_deficient_covariate_does_not_crash():
    """Two identical (constant) covariate columns — C'C is singular.
    The helper uses pinv internally; result must be finite."""
    n = 64
    rng = np.random.default_rng(0)
    variant = rng.normal(size=n).astype(np.float32)
    target = (0.5 * variant + 0.5 * rng.normal(size=n)).astype(np.float32)
    # Duplicate constant column: rank-1 in a 2-column matrix.
    cov = np.ones((n, 2), dtype=np.float32)
    standardized = _standardized(variant[:, None])
    z = compute_marginal_z_scores(
        standardized_genotypes=standardized,
        active_variant_indices=np.asarray([0], dtype=np.int32),
        covariate_matrix=cov,
        target_vector=target,
    )
    assert z.shape == (1,)
    assert np.isfinite(z[0]), z


def test_variant_is_sole_covariate_yields_near_zero_z():
    """When the only covariate IS (a copy of) the variant, residualizing
    on covariates removes the signal and z must be near zero."""
    n = 64
    rng = np.random.default_rng(1)
    variant = rng.normal(size=n).astype(np.float32)
    target = (variant + 0.01 * rng.normal(size=n)).astype(np.float32)
    cov = variant[:, None].astype(np.float32)
    standardized = _standardized(variant[:, None])
    z = compute_marginal_z_scores(
        standardized_genotypes=standardized,
        active_variant_indices=np.asarray([0], dtype=np.int32),
        covariate_matrix=cov,
        target_vector=target,
    )
    assert np.isfinite(z[0])
    assert abs(z[0]) < 1e-2, (
        f"variant equal to sole covariate should give z≈0, got {z[0]:.4f}"
    )


def test_n_equals_two_samples_does_not_crash():
    """Tiny n is unrealistic but must not segfault, raise, or NaN."""
    n = 2
    variant = np.asarray([0.0, 2.0], dtype=np.float32)
    target = np.asarray([0.0, 1.0], dtype=np.float32)
    cov = np.ones((n, 1), dtype=np.float32)
    standardized = _standardized(variant[:, None])
    z = compute_marginal_z_scores(
        standardized_genotypes=standardized,
        active_variant_indices=np.asarray([0], dtype=np.int32),
        covariate_matrix=cov,
        target_vector=target,
    )
    assert z.shape == (1,)
    # Either finite (well-defined edge value) or 0 (sigma2_resid <= 0 fallback);
    # never NaN/inf.
    assert np.isfinite(z[0]), z

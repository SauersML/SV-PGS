"""Tests for the exact ELBO stationary-point σ_e² update in
`_quantitative_posterior_state`.

The previous implementation used a leverage proxy
    posterior_fit_uncertainty = σ_e² · Σ_j (τ²_j − Σ_β_jj) / τ²_j
which is only correct at the true posterior.  The new implementation uses the
closed-form ELBO stationary point
    σ_e²_new = (RSS + tr(X Σ_β Xᵀ)) / n
            = (RSS + n · Σ_j Σ_β_jj) / n         (standardized columns).
"""

from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.genotype import as_raw_genotype_matrix
from sv_pgs.mixture_inference import _quantitative_posterior_state


def _make_standardized(genotype_values: np.ndarray):
    raw = as_raw_genotype_matrix(genotype_values.astype(np.float32))
    variant_count = genotype_values.shape[1]
    standardized = raw.standardized(
        means=np.zeros(variant_count, dtype=np.float32),
        scales=np.ones(variant_count, dtype=np.float32),
    )
    return standardized


def test_sigma_e2_matches_exact_elbo_stationary_point():
    """σ_e²_new = (RSS + n · Σ_j Σ_β_jj) / n on a small standardized problem."""
    rng = np.random.default_rng(0)
    sample_count, variant_count = 200, 10
    sigma_e_true = 2.0

    # Generate already-standardized genotypes (zero mean, unit variance per column).
    raw_genotypes = rng.standard_normal((sample_count, variant_count)).astype(np.float64)
    raw_genotypes -= raw_genotypes.mean(axis=0, keepdims=True)
    column_norms = np.linalg.norm(raw_genotypes, axis=0)
    raw_genotypes *= np.sqrt(sample_count) / column_norms  # ‖X[:,j]‖² == n exactly
    genotype_matrix = raw_genotypes.astype(np.float32)

    standardized = _make_standardized(genotype_matrix)
    standardized._dense_cache = standardized.materialize()

    true_beta = rng.standard_normal(variant_count) * 0.3
    targets = (
        genotype_matrix.astype(np.float64) @ true_beta
        + rng.standard_normal(sample_count) * sigma_e_true
    ).astype(np.float32)

    # Intercept-only "covariates" so the GLS problem is well-conditioned.
    covariate_matrix = np.ones((sample_count, 1), dtype=np.float32)
    prior_variances = np.full(variant_count, 0.5, dtype=np.float64)
    sigma_error2 = float(sigma_e_true ** 2)

    (
        _alpha,
        beta,
        beta_variance,
        linear_predictor,
        _objective,
        sigma_error2_new,
    ) = _quantitative_posterior_state(
        genotype_matrix=standardized,
        covariate_matrix=covariate_matrix,
        targets=targets,
        prior_variances=prior_variances,
        sigma_error2=sigma_error2,
        sigma_error_floor=1e-8,
    )

    residual = np.asarray(targets, dtype=np.float64) - np.asarray(linear_predictor, dtype=np.float64)
    rss = float(np.dot(residual, residual))
    # tr(X Σ_β Xᵀ) = Σ_j (Σ_β)_jj · ‖X[:,j]‖² = n · Σ_j Σ_β_jj  for standardized X.
    trace_term = float(sample_count) * float(np.sum(np.maximum(beta_variance, 0.0)))
    expected = (rss + trace_term) / sample_count

    assert sigma_error2_new == pytest.approx(expected, rel=1e-10, abs=1e-12)


def test_sigma_e2_increases_when_beta_perturbed_away_from_posterior_mean():
    """Perturbing the linear predictor away from the posterior mean increases RSS,
    so a recomputation of σ_e² from the formula must be non-decreasing in residual
    magnitude.  We exercise this by computing the formula directly with a
    perturbed predictor (the function itself doesn't accept an external β,
    so we verify the formula behavior, which is what the implementation uses).
    """
    rng = np.random.default_rng(1)
    sample_count, variant_count = 200, 10

    raw_genotypes = rng.standard_normal((sample_count, variant_count)).astype(np.float64)
    raw_genotypes -= raw_genotypes.mean(axis=0, keepdims=True)
    column_norms = np.linalg.norm(raw_genotypes, axis=0)
    raw_genotypes *= np.sqrt(sample_count) / column_norms
    genotype_matrix = raw_genotypes.astype(np.float32)

    standardized = _make_standardized(genotype_matrix)
    standardized._dense_cache = standardized.materialize()

    true_beta = rng.standard_normal(variant_count) * 0.3
    targets = (
        genotype_matrix.astype(np.float64) @ true_beta
        + rng.standard_normal(sample_count) * 2.0
    ).astype(np.float32)
    covariate_matrix = np.ones((sample_count, 1), dtype=np.float32)
    prior_variances = np.full(variant_count, 0.5, dtype=np.float64)
    sigma_error2 = 4.0

    (
        _alpha,
        beta,
        beta_variance,
        linear_predictor,
        _objective,
        sigma_error2_new,
    ) = _quantitative_posterior_state(
        genotype_matrix=standardized,
        covariate_matrix=covariate_matrix,
        targets=targets,
        prior_variances=prior_variances,
        sigma_error2=sigma_error2,
        sigma_error_floor=1e-8,
    )

    targets64 = np.asarray(targets, dtype=np.float64)
    predictor_base = np.asarray(linear_predictor, dtype=np.float64)
    trace_term = float(sample_count) * float(np.sum(np.maximum(beta_variance, 0.0)))

    rss_base = float(np.dot(targets64 - predictor_base, targets64 - predictor_base))
    sigma_base = (rss_base + trace_term) / sample_count

    # The function's returned σ_e²_new should match the base formula exactly.
    assert sigma_error2_new == pytest.approx(sigma_base, rel=1e-10, abs=1e-12)

    # Perturb the linear predictor along the residual direction (moving the
    # predictor *away* from the targets).  This monotonically increases RSS,
    # so the σ_e² formula must be monotonically non-decreasing in the
    # perturbation scale.  This verifies "non-decreasing in fit quality
    # degradation" -- i.e., worse fit → larger σ_e².
    residual_base = targets64 - predictor_base
    previous_sigma = sigma_base
    for scale in (0.0, 0.1, 0.25, 0.5, 1.0):
        # predictor_base - scale · residual_base moves the predictor further
        # from the targets along the same direction; ‖residual‖² scales as
        # (1 + scale)².
        predictor_perturbed = predictor_base - scale * residual_base
        rss_perturbed = float(
            np.dot(targets64 - predictor_perturbed, targets64 - predictor_perturbed)
        )
        sigma_perturbed = (rss_perturbed + trace_term) / sample_count
        assert sigma_perturbed >= previous_sigma - 1e-12
        previous_sigma = sigma_perturbed
    # Final perturbation must be strictly larger than baseline.
    assert previous_sigma > sigma_base

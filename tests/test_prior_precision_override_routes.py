"""Every restricted-solve route must honour ``prior_precision_override``.

From the second CAVI iteration on (and in the final posterior) the E-step
passes the CAVI prior precision E[1/lambda] / (sigma_g s_j)^2, which differs
from 1 / reduced_prior_variances. The exact variant-space route used it, but
the sample-space CG route built D + X diag(prior_variances) X^T and the
working-set route solved its subsets with 1 / prior_variances, so the
posterior mean depended on which route the problem size selected.
"""
from __future__ import annotations

import numpy as np

from sv_pgs import mixture_inference
from sv_pgs.genotype import as_raw_genotype_matrix


def _problem():
    rng = np.random.default_rng(21)
    sample_count, variant_count = 24, 40
    genotype_values = rng.standard_normal((sample_count, variant_count)).astype(np.float32)
    standardized = as_raw_genotype_matrix(genotype_values).standardized(
        means=np.zeros(variant_count, dtype=np.float32),
        scales=np.ones(variant_count, dtype=np.float32),
    )
    standardized._dense_cache = standardized.materialize()
    covariate_matrix = np.column_stack([np.ones(sample_count), rng.standard_normal(sample_count)])
    targets = rng.standard_normal(sample_count)
    prior_variances = rng.uniform(0.2, 1.5, size=variant_count)
    prior_precision_override = rng.uniform(2.0, 5.0, size=variant_count) / prior_variances
    diagonal_noise = rng.uniform(0.6, 1.4, size=sample_count)
    return standardized, covariate_matrix, targets, prior_variances, prior_precision_override, diagonal_noise


def _closed_form_beta(standardized, covariate_matrix, targets, prior_precision, diagonal_noise) -> np.ndarray:
    genotypes = np.asarray(standardized.materialize(), dtype=np.float64)
    inverse_noise = 1.0 / diagonal_noise
    weighted_covariates = inverse_noise[:, None] * covariate_matrix
    projector = np.diag(inverse_noise) - weighted_covariates @ np.linalg.solve(
        covariate_matrix.T @ weighted_covariates, weighted_covariates.T
    )
    precision = genotypes.T @ projector @ genotypes + np.diag(prior_precision)
    return np.linalg.solve(precision, genotypes.T @ projector @ targets)


def _mean_only_beta(route_arguments: dict) -> np.ndarray:
    standardized, covariate_matrix, targets, prior_variances, prior_precision_override, diagonal_noise = _problem()
    _alpha, beta, _projected, _predictor, _quadratic = mixture_inference._solve_restricted_mean_only(
        genotype_matrix=standardized,
        covariate_matrix=covariate_matrix,
        targets=targets,
        prior_variances=prior_variances,
        diagonal_noise=diagonal_noise,
        solver_tolerance=1e-6,
        maximum_linear_solver_iterations=2000,
        posterior_variance_batch_size=8,
        random_seed=0,
        prior_precision_override=prior_precision_override,
        **route_arguments,
    )
    return np.asarray(beta, dtype=np.float64)


def test_restricted_mean_routes_agree_under_prior_precision_override() -> None:
    standardized, covariate_matrix, targets, _prior_variances, prior_precision_override, diagonal_noise = _problem()
    reference = _closed_form_beta(standardized, covariate_matrix, targets, prior_precision_override, diagonal_noise)

    exact_variant_beta = _mean_only_beta({"exact_solver_matrix_limit": 100})
    sample_space_beta = _mean_only_beta({"exact_solver_matrix_limit": 2, "sample_space_preconditioner_rank": 0})
    working_set_beta = _mean_only_beta(
        {
            "exact_solver_matrix_limit": 2,
            "sample_space_preconditioner_rank": 0,
            "posterior_working_set_min_variants": 1,
            "posterior_working_set_initial_size": 8,
            "posterior_working_set_growth": 8,
            "posterior_working_set_max_passes": 10,
            "posterior_working_set_coefficient_tolerance": 0.0,
        }
    )

    scale = float(np.max(np.abs(reference)))
    np.testing.assert_allclose(exact_variant_beta, reference, rtol=0.0, atol=1e-4 * scale)
    np.testing.assert_allclose(sample_space_beta, reference, rtol=0.0, atol=1e-4 * scale)
    np.testing.assert_allclose(working_set_beta, reference, rtol=0.0, atol=1e-4 * scale)


def test_restricted_full_sample_space_route_uses_override_precision() -> None:
    standardized, covariate_matrix, targets, prior_variances, prior_precision_override, diagonal_noise = _problem()
    reference = _closed_form_beta(standardized, covariate_matrix, targets, prior_precision_override, diagonal_noise)

    _alpha, beta, _variance, _projected, _predictor, _quadratic, _logdet, _logdet_gls = mixture_inference._solve_restricted_full(
        genotype_matrix=standardized,
        covariate_matrix=covariate_matrix,
        targets=targets,
        prior_variances=prior_variances,
        diagonal_noise=diagonal_noise,
        solver_tolerance=1e-6,
        maximum_linear_solver_iterations=2000,
        logdet_probe_count=4,
        logdet_lanczos_steps=8,
        exact_solver_matrix_limit=2,
        posterior_variance_batch_size=8,
        posterior_variance_probe_count=4,
        random_seed=0,
        compute_logdet=False,
        compute_beta_variance=True,
        sample_space_preconditioner_rank=0,
        prior_precision_override=prior_precision_override,
    )

    scale = float(np.max(np.abs(reference)))
    np.testing.assert_allclose(np.asarray(beta, dtype=np.float64), reference, rtol=0.0, atol=1e-4 * scale)

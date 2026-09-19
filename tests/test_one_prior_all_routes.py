"""One model on every solver route: the beta step uses the prior the M-steps fit.

The EM fits tau_j^2 = (sigma_g s_j)^2 E[lambda_j] (the scale and TPB M-steps
use E[beta^2] / E[lambda] and log E[lambda]), so the beta step must be the
Gaussian posterior under N(0, tau^2). Solving it instead under E[1/lambda] /
(sigma_g s_j)^2 mixed two parameterizations (E[1/lambda] >= 1/E[lambda]), and
that mismatch collapsed the global scale.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from sv_pgs import mixture_inference
from sv_pgs.config import ModelConfig, TraitType
from sv_pgs.genotype import as_raw_genotype_matrix
from sv_pgs.mixture_inference import fit_variational_em
from sv_pgs.preprocessing import build_tie_map

from tests.conftest import make_variant_records


def _standardized(genotype_values: np.ndarray):
    variant_count = genotype_values.shape[1]
    standardized = as_raw_genotype_matrix(genotype_values).standardized(
        means=np.zeros(variant_count, dtype=np.float32),
        scales=np.ones(variant_count, dtype=np.float32),
    )
    standardized._dense_cache = standardized.materialize()
    return standardized


def _closed_form_posterior_mean(genotypes, covariate_matrix, targets, prior_variances, diagonal_noise) -> np.ndarray:
    inverse_noise = 1.0 / diagonal_noise
    weighted_covariates = inverse_noise[:, None] * covariate_matrix
    projector = np.diag(inverse_noise) - weighted_covariates @ np.linalg.solve(
        covariate_matrix.T @ weighted_covariates, weighted_covariates.T
    )
    precision = genotypes.T @ projector @ genotypes + np.diag(1.0 / prior_variances)
    return np.linalg.solve(precision, genotypes.T @ projector @ targets)


@pytest.mark.parametrize(("sample_count", "variant_count"), [(24, 40), (60, 30)], ids=["p_gt_n", "p_lt_n"])
def test_restricted_mean_routes_agree_with_the_closed_form_posterior(sample_count: int, variant_count: int) -> None:
    random_generator = np.random.default_rng(21)
    genotype_values = random_generator.standard_normal((sample_count, variant_count)).astype(np.float32)
    covariate_matrix = np.column_stack([np.ones(sample_count), random_generator.standard_normal(sample_count)])
    targets = random_generator.standard_normal(sample_count)
    prior_variances = random_generator.uniform(0.2, 1.5, size=variant_count)
    diagonal_noise = random_generator.uniform(0.6, 1.4, size=sample_count)
    reference = _closed_form_posterior_mean(
        genotype_values.astype(np.float64), covariate_matrix, targets, prior_variances, diagonal_noise
    )

    def route_beta(**route_arguments: Any) -> np.ndarray:
        _alpha, beta, _projected, _predictor, _quadratic = mixture_inference._solve_restricted_mean_only(
            genotype_matrix=_standardized(genotype_values),
            covariate_matrix=covariate_matrix,
            targets=targets,
            prior_variances=prior_variances,
            diagonal_noise=diagonal_noise,
            solver_tolerance=1e-6,
            maximum_linear_solver_iterations=4000,
            posterior_variance_batch_size=8,
            random_seed=0,
            **route_arguments,
        )
        return np.asarray(beta, dtype=np.float64)

    exact_beta = route_beta(exact_solver_matrix_limit=100)
    iterative_beta = route_beta(exact_solver_matrix_limit=2, sample_space_preconditioner_rank=0)
    working_set_beta = route_beta(
        exact_solver_matrix_limit=2,
        sample_space_preconditioner_rank=0,
        posterior_working_set_min_variants=1,
        posterior_working_set_initial_size=8,
        posterior_working_set_growth=8,
        posterior_working_set_max_passes=10,
        posterior_working_set_coefficient_tolerance=0.0,
    )
    scale = float(np.max(np.abs(reference)))
    for route_result in (exact_beta, iterative_beta, working_set_beta):
        np.testing.assert_allclose(route_result, reference, rtol=0.0, atol=1e-4 * scale)


def _ld_problem(random_generator: np.random.Generator, sample_count: int, variant_count: int):
    latent = random_generator.standard_normal((sample_count, variant_count))
    for column_index in range(1, variant_count):
        if column_index % 10:
            latent[:, column_index] = 0.9 * latent[:, column_index - 1] + np.sqrt(1 - 0.81) * latent[:, column_index]
    genotype_values = ((latent - latent.mean(0)) / latent.std(0)).astype(np.float32)
    covariate_matrix = np.column_stack([np.ones(sample_count), random_generator.standard_normal(sample_count)])
    true_beta = np.zeros(variant_count)
    true_beta[random_generator.choice(variant_count, 5, replace=False)] = 0.5
    targets = genotype_values.astype(np.float64) @ true_beta + random_generator.standard_normal(sample_count)
    return genotype_values, covariate_matrix.astype(np.float32), targets.astype(np.float32)


@pytest.mark.parametrize(("sample_count", "variant_count"), [(80, 50), (50, 80)], ids=["p_lt_n", "p_gt_n"])
def test_fit_reaches_one_fixed_point_on_every_route(sample_count: int, variant_count: int) -> None:
    genotype_values, covariate_matrix, targets = _ld_problem(np.random.default_rng(5), sample_count, variant_count)
    records = make_variant_records(variant_count)

    def fit_on_route(**route_arguments: Any):
        config = ModelConfig(
            trait_type=TraitType.QUANTITATIVE,
            max_outer_iterations=4,
            linear_solver_tolerance=1e-6,
            maximum_linear_solver_iterations=4000,
            # Orthogonal probes spanning the whole space make the iterative
            # routes' posterior-variance diagonal exact, so any remaining
            # difference between routes would be a difference in the model.
            posterior_variance_probe_count=80,
            # Refresh the variance every iteration on every route: the default
            # schedule itself depends on exact_solver_matrix_limit.
            beta_variance_update_interval=1,
            stochastic_variational_updates=False,
            random_seed=0,
            **route_arguments,
        )
        return fit_variational_em(
            genotypes=genotype_values,
            covariates=covariate_matrix,
            targets=targets,
            records=records,
            config=config,
            tie_map=build_tie_map(genotype_values, records, config),
        )

    exact_fit = fit_on_route(exact_solver_matrix_limit=256)
    iterative_fit = fit_on_route(exact_solver_matrix_limit=2)
    working_set_fit = fit_on_route(
        exact_solver_matrix_limit=2,
        posterior_working_set_min_variants=1,
        posterior_working_set_initial_size=8,
        posterior_working_set_growth=8,
        posterior_working_set_max_passes=20,
        posterior_working_set_coefficient_tolerance=0.0,
    )
    # The iterative routes run float32 CG at a 1e-6 tolerance, so they agree
    # with the exact route to a few percent. A route-dependent prior moved the
    # global scale by 2.5x on problems like this one.
    exact_beta = np.asarray(exact_fit.beta_reduced, dtype=np.float64)
    scale = float(np.max(np.abs(exact_beta)))
    for route_fit in (iterative_fit, working_set_fit):
        np.testing.assert_allclose(np.asarray(route_fit.beta_reduced, dtype=np.float64), exact_beta, rtol=0.0, atol=5e-2 * scale)
        assert route_fit.sigma_error2 == pytest.approx(exact_fit.sigma_error2, rel=2e-2)
        assert route_fit.global_scale == pytest.approx(exact_fit.global_scale, rel=2e-2)
        np.testing.assert_allclose(route_fit.prior_scales, exact_fit.prior_scales, rtol=5e-2)


def test_final_posterior_is_the_gaussian_posterior_under_the_reported_prior(monkeypatch: pytest.MonkeyPatch) -> None:
    genotype_values, covariate_matrix, targets = _ld_problem(np.random.default_rng(9), 120, 60)
    records = make_variant_records(60)
    config = ModelConfig(trait_type=TraitType.QUANTITATIVE, max_outer_iterations=6, random_seed=0)
    captured_solve_inputs: list[dict[str, Any]] = []
    quantitative_posterior_state = mixture_inference._quantitative_posterior_state

    def capturing_quantitative_posterior_state(**kwargs: Any):
        captured_solve_inputs.append(kwargs)
        return quantitative_posterior_state(**kwargs)

    monkeypatch.setattr(mixture_inference, "_quantitative_posterior_state", capturing_quantitative_posterior_state)
    result = fit_variational_em(
        genotypes=genotype_values,
        covariates=covariate_matrix,
        targets=targets,
        records=records,
        config=config,
        tie_map=build_tie_map(genotype_values, records, config),
    )

    final_solve = captured_solve_inputs[-1]
    np.testing.assert_allclose(final_solve["prior_variances"], result.prior_scales, rtol=1e-12)
    reference = _closed_form_posterior_mean(
        genotype_values.astype(np.float64),
        covariate_matrix.astype(np.float64),
        targets.astype(np.float64),
        np.asarray(result.prior_scales, dtype=np.float64),
        np.full(targets.shape[0], float(final_solve["sigma_error2"])),
    )
    np.testing.assert_allclose(
        np.asarray(result.beta_reduced, dtype=np.float64),
        reference,
        rtol=0.0,
        atol=1e-4 * float(np.max(np.abs(reference))),
    )


def test_deterministic_fit_keeps_its_global_scale_when_p_exceeds_n() -> None:
    """Accuracy reproducer. Under the mixed E[1/lambda] beta step the global
    scale collapsed (about 10x) and held-out accuracy fell well below the
    stochastic path's plug-in fit on the same data; one model keeps both paths
    at one fixed point."""
    random_generator = np.random.default_rng(0)
    sample_count, test_count, variant_count = 300, 1000, 700
    latent = random_generator.standard_normal((sample_count + test_count, variant_count))
    for column_index in range(1, variant_count):
        if column_index % 20:
            latent[:, column_index] = 0.9 * latent[:, column_index - 1] + np.sqrt(1 - 0.81) * latent[:, column_index]
    genotypes = (latent - latent[:sample_count].mean(0)) / latent[:sample_count].std(0)
    true_beta = np.zeros(variant_count)
    true_beta[random_generator.choice(variant_count, 20, replace=False)] = random_generator.standard_normal(20)
    true_beta += random_generator.standard_normal(variant_count) * 0.1
    genetic_value = genotypes @ true_beta
    genetic_value *= np.sqrt(0.4 / np.var(genetic_value[:sample_count]))
    targets = (genetic_value[:sample_count] + random_generator.standard_normal(sample_count) * np.sqrt(0.6)).astype(np.float32)
    covariate_matrix = np.ones((sample_count, 1), dtype=np.float32)
    genotype_values = genotypes[:sample_count].astype(np.float32)
    records = make_variant_records(variant_count)

    def held_out_accuracy(stochastic: bool) -> tuple[float, float]:
        config = ModelConfig(
            trait_type=TraitType.QUANTITATIVE,
            max_outer_iterations=25,
            random_seed=0,
            stochastic_variational_updates=stochastic,
            stochastic_min_variant_count=0,
            stochastic_variant_batch_size=175,
        )
        tie_map = build_tie_map(genotype_values, records, config)
        result = fit_variational_em(
            genotypes=genotype_values,
            covariates=covariate_matrix,
            targets=targets,
            records=records,
            config=config,
            tie_map=tie_map,
        )
        beta = np.zeros(variant_count)
        beta[tie_map.kept_indices] = result.beta_reduced
        prediction = genotypes[sample_count:] @ beta
        return float(np.corrcoef(prediction, genetic_value[sample_count:])[0, 1] ** 2), float(result.global_scale)

    deterministic_accuracy, deterministic_scale = held_out_accuracy(stochastic=False)
    stochastic_accuracy, stochastic_scale = held_out_accuracy(stochastic=True)
    assert deterministic_scale > 0.5 * stochastic_scale
    assert deterministic_accuracy > stochastic_accuracy - 0.03

"""Stage 2 full-data Gaussian E-step against dense references written independently of the block code.

References: the restricted precision X' P_W X + diag(tau) built densely, its
explicit inverse and solves, and Newton's method on the dense penalized
logistic likelihood in (alpha, beta) jointly.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit

from sv_pgs.config import TraitType
from sv_pgs.exact_polish import (
    DenseGenotypeBlockSource,
    FullDataGaussian,
    GaussianModel,
    _BlockJacobi,
    _conjugate_gradient,
    _Device,
    _Operator,
    _Reads,
    _SampleSystem,
)


def _simulate(*, sample_count: int, variant_count: int, seed: int):
    """Standardized genotypes with AR(1) LD across every block boundary, two masks, both trait types."""
    generator = np.random.default_rng(seed)
    latent = generator.standard_normal((sample_count, variant_count))
    for column in range(1, variant_count):
        latent[:, column] = 0.7 * latent[:, column - 1] + np.sqrt(1.0 - 0.49) * latent[:, column]
    dosage = (latent > 0.6).astype(np.float64) + (latent > 1.4).astype(np.float64)
    dosage += 0.05 * generator.standard_normal(dosage.shape)
    genotypes = (dosage - dosage.mean(axis=0)) / dosage.std(axis=0)
    covariates = np.column_stack([np.ones(sample_count), generator.standard_normal(sample_count)])
    effects = np.zeros(variant_count)
    causal = generator.choice(variant_count, size=max(variant_count // 10, 3), replace=False)
    effects[causal] = generator.standard_normal(causal.size) * 0.3
    genetic = genotypes @ effects
    genetic *= 0.6 / np.std(genetic)
    quantitative = 1.5 + 0.3 * covariates[:, 1] + genetic + generator.standard_normal(sample_count)
    binary = (generator.random(sample_count) < expit(-0.8 + 0.3 * covariates[:, 1] + genetic)).astype(np.float64)
    masks = np.ones((2, sample_count))
    masks[1, generator.choice(sample_count, size=sample_count // 5, replace=False)] = 0.0
    return genotypes, covariates, quantitative, binary, masks


def _blocks(variant_count: int, block_size: int) -> list[np.ndarray]:
    return [np.arange(start, min(start + block_size, variant_count)) for start in range(0, variant_count, block_size)]


def _sites(variant_count: int, model_count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    precision = np.exp(generator.uniform(np.log(5.0), np.log(500.0), size=(variant_count, model_count)))
    precision[generator.random((variant_count, model_count)) < 0.1] = 0.0
    shift = generator.standard_normal((variant_count, model_count)) * 0.2
    return precision, shift


def _projector(covariates: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """P_W = W - W C (C'WC + ridge)^-1 C' W densely, with the module's relative ridge."""
    weighted = covariates.T @ (weights[:, None] * covariates)
    ridge = 1e-10 * np.trace(weighted) / covariates.shape[1]
    inverse = np.linalg.inv(weighted + ridge * np.eye(covariates.shape[1]))
    diagonal = np.diag(weights)
    return diagonal - (weights[:, None] * covariates) @ inverse @ (covariates.T * weights[None, :])


def _dense_precision(genotypes, covariates, weights, precision) -> np.ndarray:
    return genotypes.T @ _projector(covariates, weights) @ genotypes + np.diag(precision)


def _exact(gaussian: FullDataGaussian) -> np.ndarray:
    """Exact-curvature preconditioning for every binary model."""
    return gaussian.system.is_binary.copy()


def _models(quantitative, binary) -> list[GaussianModel]:
    sample_count = quantitative.shape[0]
    return [
        GaussianModel(TraitType.QUANTITATIVE, quantitative, 0, np.zeros(sample_count)),
        GaussianModel(TraitType.BINARY, binary, 1, np.zeros(sample_count)),
        GaussianModel(TraitType.QUANTITATIVE, quantitative, 1, np.full(sample_count, 0.1)),
    ]


def test_operator_and_conjugate_gradient_match_the_dense_restricted_precision():
    genotypes, covariates, quantitative, binary, masks = _simulate(sample_count=300, variant_count=90, seed=1)
    models = _models(quantitative, binary)
    source = DenseGenotypeBlockSource(genotypes, _blocks(90, 20))
    device = _Device(np)
    system = _SampleSystem(device=device, models=models, covariates=covariates, sample_masks=masks)
    generator = np.random.default_rng(2)
    linear_predictor = generator.standard_normal((300, 3)) * 0.5
    noise_variance = np.array([0.8, 1.0, 1.3])
    curvature = system.curvature(linear_predictor, noise_variance)
    inverses = system.covariate_inverses(curvature)
    precision, _shift = _sites(90, 3, seed=3)
    precision[precision == 0.0] = 1e-3
    reads = _Reads(source)
    operator = _Operator(reads=reads, system=system, curvature=curvature, covariate_inverses=inverses, site_precision=precision)
    column_models = np.array([0, 1, 2, 1, 0, 2, 2])
    directions = generator.standard_normal((90, column_models.size))
    applied, image = operator.apply(directions, column_models)
    assert reads.count == 2
    dense = [_dense_precision(genotypes, covariates, curvature[:, model_index], precision[:, model_index]) for model_index in range(3)]
    for column, model_index in enumerate(column_models):
        np.testing.assert_allclose(applied[:, column], dense[model_index] @ directions[:, column], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(image[:, column], genotypes @ directions[:, column], rtol=1e-10, atol=1e-10)
    preconditioner = _BlockJacobi(
        reads=reads, system=system, curvature=curvature, site_precision=precision, exact_curvature=np.array([False, True, False])
    )
    right_hand_side = generator.standard_normal((90, column_models.size))
    solution, solution_image, residual = _conjugate_gradient(
        operator=operator,
        preconditioner=preconditioner,
        right_hand_side=right_hand_side,
        column_models=column_models,
        tolerance=1e-13,
    )
    assert np.all(residual <= 1e-13)
    for column, model_index in enumerate(column_models):
        np.testing.assert_allclose(solution[:, column], np.linalg.solve(dense[model_index], right_hand_side[:, column]), rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(solution_image, genotypes @ solution, rtol=1e-9, atol=1e-10)


def test_one_quantitative_iteration_gives_the_exact_restricted_posterior_mean():
    genotypes, covariates, quantitative, binary, masks = _simulate(sample_count=250, variant_count=120, seed=4)
    models = [GaussianModel(TraitType.QUANTITATIVE, quantitative, 1, np.full(250, 0.2))]
    source = DenseGenotypeBlockSource(genotypes, _blocks(120, 25))
    precision, shift = _sites(120, 1, seed=5)
    noise_variance = np.array([0.7])
    gaussian = FullDataGaussian(
        source=source, models=models, covariates=covariates, sample_masks=masks, initial_mean=np.zeros((120, 1)), probe_count=4, seed=6
    )
    gaussian.iterate(site_precision=precision, site_shift=shift, noise_variance=noise_variance, tolerance=1e-13, refactor=True, exact_curvature=_exact(gaussian))
    weights = masks[1] / noise_variance[0]
    dense = _dense_precision(genotypes, covariates, weights, precision[:, 0])
    expected = np.linalg.solve(dense, genotypes.T @ _projector(covariates, weights) @ (quantitative - 0.2) + shift[:, 0])
    np.testing.assert_allclose(gaussian.mean[:, 0], expected, rtol=1e-8, atol=1e-10)
    covariate_gram = covariates.T @ (weights[:, None] * covariates)
    expected_alpha = np.linalg.solve(covariate_gram, covariates.T @ (weights * (quantitative - 0.2 - genotypes @ expected)))
    np.testing.assert_allclose(gaussian.alpha[:, 0], expected_alpha, rtol=1e-8, atol=1e-10)
    certificate = gaussian.iterate(site_precision=precision, site_shift=shift, noise_variance=noise_variance, tolerance=1e-13, refactor=False, exact_curvature=_exact(gaussian))
    assert certificate.gradient_relative_norm[0] <= 1e-10
    assert certificate.covariate_gradient_relative_norm[0] <= 1e-10
    assert certificate.probe_residual[0] <= 1e-10


def _dense_penalized_logistic_mode(genotypes, covariates, targets, mask, precision, shift, offset):
    """Newton on sum_i mask_i (y eta - log(1 + e^eta)) - tau beta^2/2 + nu beta over (alpha, beta)."""
    design = np.column_stack([covariates, genotypes])
    penalty = np.concatenate([np.zeros(covariates.shape[1]), precision])
    linear_shift = np.concatenate([np.zeros(covariates.shape[1]), shift])
    coefficients = np.zeros(design.shape[1])
    for _iteration in range(100):
        probability = expit(offset + design @ coefficients)
        gradient = design.T @ (mask * (targets - probability)) - penalty * coefficients + linear_shift
        hessian = design.T @ ((mask * probability * (1.0 - probability))[:, None] * design) + np.diag(penalty)
        step = np.linalg.solve(hessian, gradient)
        coefficients += step
        if np.max(np.abs(step)) <= 1e-14:
            break
    return coefficients[: covariates.shape[1]], coefficients[covariates.shape[1] :]


def test_binary_newton_iterations_reach_the_dense_penalized_logistic_mode():
    genotypes, covariates, quantitative, binary, masks = _simulate(sample_count=400, variant_count=60, seed=7)
    models = [GaussianModel(TraitType.BINARY, binary, 1, np.full(400, -0.1))]
    source = DenseGenotypeBlockSource(genotypes, _blocks(60, 15))
    precision, shift = _sites(60, 1, seed=8)
    precision[precision == 0.0] = 0.5
    gaussian = FullDataGaussian(
        source=source, models=models, covariates=covariates, sample_masks=masks, initial_mean=np.zeros((60, 1)), probe_count=4, seed=9
    )
    for _iteration in range(30):
        certificate = gaussian.iterate(
            site_precision=precision, site_shift=shift, noise_variance=np.ones(1), tolerance=1e-12, refactor=True, exact_curvature=_exact(gaussian)
        )
        if certificate.gradient_relative_norm[0] <= 1e-11:
            break
    assert certificate.gradient_relative_norm[0] <= 1e-11
    expected_alpha, expected_beta = _dense_penalized_logistic_mode(
        genotypes, covariates, binary, masks[1], precision[:, 0], shift[:, 0], -0.1
    )
    np.testing.assert_allclose(gaussian.mean[:, 0], expected_beta, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(gaussian.alpha[:, 0], expected_alpha, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("block_size", [80, 16])
def test_marginal_variances_are_exact_with_one_block_and_unbiased_with_many(block_size):
    genotypes, covariates, quantitative, binary, masks = _simulate(sample_count=300, variant_count=80, seed=10)
    models = [GaussianModel(TraitType.QUANTITATIVE, quantitative, 0, np.zeros(300)), GaussianModel(TraitType.BINARY, binary, 1, np.zeros(300))]
    source = DenseGenotypeBlockSource(genotypes, _blocks(80, block_size))
    precision, shift = _sites(80, 2, seed=11)
    probe_count = 400
    gaussian = FullDataGaussian(
        source=source, models=models, covariates=covariates, sample_masks=masks, initial_mean=np.zeros((80, 2)), probe_count=probe_count, seed=12
    )
    noise_variance = np.array([0.9, 1.0])
    gaussian.iterate(site_precision=precision, site_shift=shift, noise_variance=noise_variance, tolerance=1e-13, refactor=True, exact_curvature=_exact(gaussian))
    # The probes are solved at the curvature of the iteration's start state.
    variances = gaussian.marginal_variances(precision)
    for model_index in range(2):
        start_curvature = gaussian.probe_curvature[:, model_index]
        exact = np.diag(np.linalg.inv(_dense_precision(genotypes, covariates, start_curvature, precision[:, model_index])))
        estimate = variances.variance[:, model_index]
        if block_size == 80:
            np.testing.assert_allclose(estimate, exact, rtol=1e-9)
            assert variances.relative_standard_error[model_index] <= 1e-9
        else:
            assert np.abs(np.sum(estimate * precision[:, model_index]) - np.sum(exact * precision[:, model_index])) <= 4.0 * (
                variances.relative_standard_error[model_index] * np.sum(exact * precision[:, model_index])
            ) + 1e-12
            assert np.median(np.abs(estimate / exact - 1.0)) <= 0.05
        upper = np.where(precision[:, model_index] > 0.0, 1.0 / np.maximum(precision[:, model_index], 1e-300), np.inf)
        assert np.all(estimate <= upper * (1.0 + 1e-12))
        assert np.all(estimate > 0.0)


def test_draws_have_the_posterior_mean_and_covariance():
    genotypes, covariates, quantitative, binary, masks = _simulate(sample_count=200, variant_count=24, seed=13)
    models = [GaussianModel(TraitType.QUANTITATIVE, quantitative, 1, np.zeros(200))]
    source = DenseGenotypeBlockSource(genotypes, _blocks(24, 8))
    precision, shift = _sites(24, 1, seed=14)
    precision[precision == 0.0] = 2.0
    gaussian = FullDataGaussian(
        source=source, models=models, covariates=covariates, sample_masks=masks, initial_mean=np.zeros((24, 1)), probe_count=2, seed=15
    )
    noise_variance = np.array([1.2])
    gaussian.iterate(site_precision=precision, site_shift=shift, noise_variance=noise_variance, tolerance=1e-13, refactor=True, exact_curvature=_exact(gaussian))
    draws = gaussian.draws(site_precision=precision, draw_count=20000, tolerance=1e-12)[:, 0, :]
    covariance = np.linalg.inv(_dense_precision(genotypes, covariates, masks[1] / noise_variance[0], precision[:, 0]))
    standard_deviation = np.sqrt(np.diag(covariance))
    assert np.max(np.abs(draws.mean(axis=1) - gaussian.mean[:, 0]) / standard_deviation) <= 5.0 / np.sqrt(20000)
    sample_correlation = np.corrcoef(draws)
    np.testing.assert_allclose(np.sqrt(np.var(draws, axis=1)) / standard_deviation, 1.0, atol=0.04)
    np.testing.assert_allclose(sample_correlation, covariance / np.outer(standard_deviation, standard_deviation), atol=0.04)


def test_reads_per_iteration():
    genotypes, covariates, quantitative, binary, masks = _simulate(sample_count=120, variant_count=30, seed=16)
    models = [GaussianModel(TraitType.QUANTITATIVE, quantitative, 0, np.zeros(120))]
    source = DenseGenotypeBlockSource(genotypes, _blocks(30, 10))
    precision, shift = _sites(30, 1, seed=17)
    gaussian = FullDataGaussian(
        source=source, models=models, covariates=covariates, sample_masks=masks, initial_mean=np.zeros((30, 1)), probe_count=2, seed=18
    )
    assert gaussian.reads.count == 1
    gaussian.iterate(site_precision=precision, site_shift=shift, noise_variance=np.ones(1), tolerance=1e-10, refactor=True, exact_curvature=_exact(gaussian))
    after_first = gaussian.reads.count
    assert (after_first - 1 - 1 - 1) % 2 == 0
    gaussian.iterate(site_precision=precision, site_shift=shift, noise_variance=np.ones(1), tolerance=1e-10, refactor=False, exact_curvature=_exact(gaussian))
    assert (gaussian.reads.count - after_first - 1) % 2 == 0

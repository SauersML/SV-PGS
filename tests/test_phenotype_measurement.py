"""phenotype_measurement: the per-occasion EP-EB model's exact limits, its evidence and its gross errors.

The limits are exact: with one occasion a person's EP site is the exact tilted moment, so q(T_i) is the
exact posterior; with a one-point noise density every site is exact, so q(T_i) is the dense Henderson
solution and the evidence is the Gaussian marginal likelihood. The simulations check the algebra (sim-only).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate
from scipy.stats import multivariate_normal, norm

from sv_pgs.phenotype_measurement import (
    LEVEL_MOVE_TOLERANCE,
    Occasions,
    _prior_on,
    box_cox,
    ep_fixed_point,
    fit_at_exponent,
    fit_occasion_model,
)
from sv_pgs.scale_mixture_ep import MixtureHyperparameters, ScaleMixturePrior, class_log_density, derived_lattice
from tests.phenotype_bounds import rounding_gamma, sampling_bound, variance_component_standard_errors

WORKING_BYTES = 1 << 28
# Below every node's own share: exp of this is eps^2, so an absent node moves no moment at double precision.
_ABSENT = -2.0 * np.log(1.0 / np.finfo(np.float64).eps)


def _lattice(residuals: np.ndarray, level_variance: float) -> ScaleMixturePrior:
    precision = np.full(residuals.shape[0], 1.0 / level_variance)
    nodes, floor, top = derived_lattice(precision, residuals * precision, np.zeros(residuals.shape[0]), 0.5)
    return _prior_on(nodes, floor, top, residuals.shape[0])


def _density(prior: ScaleMixturePrior, masses: dict[int, float]) -> MixtureHyperparameters:
    """Hyperparameters whose lattice masses are ``masses`` (node -> mass) and absent elsewhere."""
    log_density = np.full(prior.grid_size, _ABSENT)
    for node, mass in masses.items():
        log_density[node] = np.log(mass)
    coefficients = np.linalg.lstsq(prior.coefficient_map, log_density - log_density.mean(), rcond=None)[0]
    return MixtureHyperparameters(coefficients=coefficients, log_smoothing=np.zeros(len(prior.smoothing_blocks)))


def _kernel_node(prior: ScaleMixturePrior, log_variance: float) -> int:
    nodes = prior.log_variance_grid
    kernel = np.flatnonzero(nodes >= prior.kernel_floor)
    return int(kernel[np.argmin(np.abs(nodes[kernel] - log_variance))])


def _simulated(persons: int, level_variance: float, noise: float, seed: int, gross_share: float = 0.0):
    generator = np.random.default_rng(seed)
    counts = generator.integers(1, 6, persons)
    person_index = np.repeat(np.arange(persons), counts)
    ages = generator.uniform(20.0, 80.0, person_index.shape[0])
    levels = generator.normal(0.0, np.sqrt(level_variance), persons)
    values = 50.0 + 0.05 * (ages - 50.0) + levels[person_index] + generator.normal(0.0, np.sqrt(noise), person_index.shape[0])
    gross = generator.random(values.shape[0]) < gross_share
    design = np.column_stack([np.ones_like(ages), ages - ages.mean()])
    return person_index, values, np.where(gross, 10.0 * values, values), design, gross, levels


def test_box_cox_is_increasing_with_an_exact_log_jacobian():
    values = np.geomspace(0.01, 1000.0, 200)
    for exponent in (-0.5, 0.0, 0.5, 1.0, 2.0):
        transformed, log_jacobian = box_cox(values, exponent)
        assert np.all(np.diff(transformed) > 0.0)
        # h'(y) = y^(lambda - 1): one log, one product and the power's own two roundings.
        np.testing.assert_allclose(log_jacobian, np.log(np.power(values, exponent - 1.0)), rtol=0.0,
                                   atol=2.0 * rounding_gamma(4) * np.max(np.abs(log_jacobian)))
    np.testing.assert_array_equal(box_cox(values, 0.0)[0], np.log(values))


def test_one_occasion_people_get_the_exact_posterior_of_their_level():
    generator = np.random.default_rng(1)
    persons, level_variance = 200, 4.0
    values = 50.0 + generator.normal(0.0, 2.0, persons) + generator.standard_t(3, persons)
    design = np.ones((persons, 1))
    occasions = Occasions(person_index=np.arange(persons), values=values, design=design)
    fixed_effects = np.array([np.mean(values) - 1.0])
    residuals = values - 1.0 - fixed_effects[0]
    prior = _lattice(residuals, level_variance)
    small, large = _kernel_node(prior, 0.0), _kernel_node(prior, np.log(100.0))
    hyperparameters = _density(prior, {small: 0.9, large: 0.1})
    fit = ep_fixed_point(occasions, 1.0, fixed_effects, level_variance, prior, hyperparameters, WORKING_BYTES)

    masses = np.exp(class_log_density(prior, hyperparameters.coefficients)[0])
    variances = np.exp(prior.log_variance_grid)
    # T | r is the mixture over nodes k of N(r tau^2 / (tau^2 + s_k), tau^2 s_k / (tau^2 + s_k)) with weights
    # proportional to pi_k N(r; 0, tau^2 + s_k).
    log_weights = np.log(masses)[None, :] + norm.logpdf(residuals[:, None], scale=np.sqrt(level_variance + variances)[None, :])
    weights = np.exp(log_weights - log_weights.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    component_mean = residuals[:, None] * level_variance / (level_variance + variances)[None, :]
    component_variance = (level_variance * variances / (level_variance + variances))[None, :]
    mean = np.sum(weights * component_mean, axis=1)
    variance = np.sum(weights * (component_variance + np.square(component_mean)), axis=1) - np.square(mean)
    # Both sides sum K weighted terms after exponentials; the engine's tilted moment then subtracts from r.
    operations = 4 * prior.grid_size + 16
    np.testing.assert_allclose(fit.level_mean, mean, rtol=0.0, atol=2.0 * rounding_gamma(operations) * (np.abs(residuals) + np.abs(mean)))
    np.testing.assert_allclose(fit.level_posterior_variance, variance, rtol=0.0,
                               atol=2.0 * rounding_gamma(operations) * (np.square(residuals) + level_variance))


def test_a_one_point_noise_density_gives_the_dense_henderson_solution_and_evidence():
    person_index, values, _gross_values, design, _gross, _levels = _simulated(40, 4.0, 1.0, seed=2)
    occasions = Occasions(person_index=person_index, values=values, design=design)
    level_variance = 4.0
    fixed_effects = np.linalg.lstsq(design, values - 1.0, rcond=None)[0]
    residuals = values - 1.0 - design @ fixed_effects
    prior = _lattice(residuals, level_variance)
    node = _kernel_node(prior, 0.0)
    noise = float(np.exp(prior.log_variance_grid[node]))
    fit = ep_fixed_point(occasions, 1.0, fixed_effects, level_variance, prior, _density(prior, {node: 1.0}), WORKING_BYTES)

    counts = np.bincount(person_index).astype(np.float64)
    precision = 1.0 / level_variance + counts / noise
    mean = np.bincount(person_index, weights=residuals) / noise / precision
    # Exact Gaussian sites after one pass; each person's mean is a sum over its k occasions.
    operations = 4 * prior.grid_size + 8 * int(counts.max()) + 16
    np.testing.assert_allclose(fit.level_mean, mean, rtol=0.0, atol=2.0 * rounding_gamma(operations) * (np.max(np.abs(residuals)) + np.abs(mean)))
    np.testing.assert_allclose(fit.level_posterior_variance, 1.0 / precision, rtol=2.0 * rounding_gamma(operations))
    evidence = sum(
        multivariate_normal(mean=np.zeros(int(count)), cov=noise * np.eye(int(count)) + level_variance).logpdf(residuals[person_index == person])
        for person, count in enumerate(counts)
    )
    assert fit.log_evidence == pytest.approx(evidence, rel=2.0 * rounding_gamma(operations) * values.shape[0])


def test_the_evidence_is_a_density_over_the_reading_at_the_log_transform():
    level_variance, fixed_effects = 0.25, np.array([np.log(80.0)])
    design = np.ones((1, 1))
    lattice_residuals = np.linspace(-1.5, 1.5, 7)
    prior = _lattice(lattice_residuals, level_variance)
    hyperparameters = _density(prior, {_kernel_node(prior, np.log(0.04)): 0.8, _kernel_node(prior, np.log(1.0)): 0.2})
    one_prior = _prior_on(prior.log_variance_grid, prior.kernel_floor, prior.kernel_top, 1)

    def density(log_reading: float) -> float:
        occasion = Occasions(person_index=np.zeros(1, dtype=np.int64), values=np.array([np.exp(log_reading)]), design=design)
        fit = ep_fixed_point(occasion, 0.0, fixed_effects, level_variance, one_prior, hyperparameters, WORKING_BYTES)
        return float(np.exp(fit.log_evidence + log_reading))  # dy = y d(log y)

    total, error = integrate.quad(density, -np.inf, np.inf)
    assert abs(total - 1.0) <= error + 2.0 * rounding_gamma(4 * prior.grid_size + 16)


def test_the_fit_is_an_ep_fixed_point_and_recovers_gaussian_data_within_sampling_error():
    level_variance, noise = 4.0, 1.0
    person_index, values, _gross_values, design, _gross, levels = _simulated(400, level_variance, noise, seed=3)
    occasions = Occasions(person_index=person_index, values=values, design=design)
    fit = fit_at_exponent(occasions, 1.0, WORKING_BYTES)
    again = ep_fixed_point(occasions, 1.0, fit.fixed_effects, fit.level_variance, fit.prior, fit.hyperparameters, WORKING_BYTES)
    assert float(np.sum(np.square(again.level_mean - fit.level_mean) / again.level_posterior_variance)) < LEVEL_MOVE_TOLERANCE
    # Maximum likelihood is efficient, so the Henderson III moment estimators' standard errors bound its own.
    between_error, within_error = variance_component_standard_errors(np.bincount(person_index), level_variance, noise)
    assert abs(fit.level_variance - level_variance) < sampling_bound(between_error)
    assert abs(fit.noise_second_moment - noise) < sampling_bound(within_error)


def test_gross_errors_are_downweighted_by_the_learned_density():
    # At one transform, so the levels of the two fits share a scale.
    person_index, values, gross_values, design, gross, _levels = _simulated(400, 4.0, 1.0, seed=4, gross_share=0.03)
    clean = fit_at_exponent(Occasions(person_index=person_index, values=values, design=design), 1.0, WORKING_BYTES)
    contaminated_occasions = Occasions(person_index=person_index, values=gross_values, design=design)
    contaminated = fit_at_exponent(contaminated_occasions, 1.0, WORKING_BYTES)
    affected = np.unique(person_index[gross])
    assert affected.shape[0] > 0
    sd = np.sqrt(contaminated.level_posterior_variance[affected])
    # The learned density moves an affected person's level by less than that person's posterior sd ...
    assert np.all(np.abs(contaminated.level_mean[affected] - clean.level_mean[affected]) < sd)
    # ... where Gaussian noise of the same second moment moves every one of them by more.
    prior = contaminated.prior
    gaussian = ep_fixed_point(
        contaminated_occasions, 1.0, contaminated.fixed_effects, contaminated.level_variance, prior,
        _density(prior, {_kernel_node(prior, np.log(contaminated.noise_second_moment)): 1.0}), WORKING_BYTES,
    )
    assert np.all(np.abs(gaussian.level_mean[affected] - clean.level_mean[affected]) > sd)

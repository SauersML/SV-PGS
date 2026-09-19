"""Shared pleiotropy scale: exact densities, leave-one-trait-out algebra, data-derived range, and density recovery."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import multivariate_normal, norm

from sv_pgs.pleiotropy_layer import (
    PleiotropyInputs,
    density_statistics,
    initial_density,
    log_multiplier_range,
    maximize_coefficients,
    node_log_densities,
    node_posteriors,
    penalized_log_likelihood,
    prior_weights,
    recentred,
    update_smoothing,
)


def _correlation(trait_count: int, value: float) -> np.ndarray:
    return np.full((trait_count, trait_count), value) + (1.0 - value) * np.eye(trait_count)


def _simulated(trait_count, column_count, log_multipliers, correlation, seed, class_count=1,
               scale_range=(2.0, 4.0), noise_range=(0.5, 1.0)):
    rng = np.random.default_rng(seed)
    class_index = rng.integers(0, class_count, size=column_count)
    cavity_variances = rng.uniform(*noise_range, size=(trait_count, column_count))
    prior_scale_moments = rng.uniform(*scale_range, size=(trait_count, column_count))
    noise_correlation = _correlation(trait_count, correlation)
    multipliers = np.exp(log_multipliers)
    cavity_means = np.empty((trait_count, column_count))
    for column in range(column_count):
        noise_sd = np.sqrt(cavity_variances[:, column])
        covariance = noise_sd[:, None] * noise_correlation * noise_sd[None, :]
        covariance += np.diag(multipliers[column] * prior_scale_moments[:, column])
        cavity_means[:, column] = rng.multivariate_normal(np.zeros(trait_count), covariance)
    return PleiotropyInputs(cavity_means=cavity_means, cavity_variances=cavity_variances,
                            prior_scale_moments=prior_scale_moments, noise_correlation=noise_correlation,
                            class_index=class_index)


def _small(trait_count=3, column_count=9, correlation=0.4, seed=3):
    rng = np.random.default_rng(seed + 100)
    return _simulated(trait_count, column_count, rng.normal(0.0, 1.0, column_count), correlation, seed, class_count=2)


def _density_for(inputs, class_count=2):
    lower, upper = log_multiplier_range(inputs)
    density = initial_density(lower, upper, class_count)
    rng = np.random.default_rng(7)
    return replace(density, coefficients=density.coefficients + rng.normal(0.0, 0.3, density.coefficients.shape))


def _fit(inputs, rounds, class_count=1):
    lower, upper = log_multiplier_range(inputs)
    density = initial_density(lower, upper, class_count)
    for _round in range(rounds):
        statistics = density_statistics(inputs, density)
        density = update_smoothing(maximize_coefficients(density, statistics), statistics)
        density, shift = recentred(density)
        scale = np.exp(shift[inputs.class_index])[None, :]
        inputs = replace(inputs, prior_scale_moments=inputs.prior_scale_moments * scale)
    return inputs, density


def _log_s_moments(density, class_value=0):
    weights = np.exp(density.log_weights()[class_value])
    nodes = density.log_multipliers[class_value]
    mean = float(weights @ nodes)
    return mean, float(np.sqrt(weights @ (nodes - mean) ** 2))


def test_node_densities_match_dense_multivariate_normal():
    inputs = _small()
    density = _density_for(inputs)
    joint, _conditional = node_log_densities(inputs, density)
    multipliers = np.exp(density.log_multipliers[inputs.class_index])
    for column in range(inputs.cavity_means.shape[1]):
        noise_sd = np.sqrt(inputs.cavity_variances[:, column])
        noise = noise_sd[:, None] * inputs.noise_correlation * noise_sd[None, :]
        for node in (0, 20, 63):
            covariance = noise + np.diag(multipliers[column, node] * inputs.prior_scale_moments[:, column])
            reference = multivariate_normal(cov=covariance).logpdf(inputs.cavity_means[:, column])
            assert joint[column, node] == pytest.approx(reference, rel=1e-10, abs=1e-10)


def test_independent_noise_splits_into_per_trait_densities():
    inputs = _small(correlation=0.0)
    density = _density_for(inputs)
    joint, _conditional = node_log_densities(inputs, density)
    multipliers = np.exp(density.log_multipliers[inputs.class_index])
    per_trait = norm.logpdf(inputs.cavity_means[:, :, None],
                            scale=np.sqrt(inputs.cavity_variances[:, :, None]
                                          + multipliers[None, :, :] * inputs.prior_scale_moments[:, :, None]))
    np.testing.assert_allclose(joint, per_trait.sum(axis=0), rtol=1e-10, atol=1e-10)


def test_leave_one_out_weights_equal_posteriors_from_the_other_traits():
    inputs = _small(trait_count=4, correlation=0.3)
    density = _density_for(inputs)
    weights, _multipliers = prior_weights(inputs, density)
    for trait in range(inputs.cavity_means.shape[0]):
        others = [index for index in range(inputs.cavity_means.shape[0]) if index != trait]
        reduced = PleiotropyInputs(cavity_means=inputs.cavity_means[others],
                                   cavity_variances=inputs.cavity_variances[others],
                                   prior_scale_moments=inputs.prior_scale_moments[others],
                                   noise_correlation=inputs.noise_correlation[np.ix_(others, others)],
                                   class_index=inputs.class_index)
        np.testing.assert_allclose(weights[trait], node_posteriors(reduced, density), rtol=1e-8, atol=1e-12)


def test_single_trait_gets_its_class_density_as_weights():
    inputs = _small(trait_count=1, correlation=0.0)
    density = _density_for(inputs)
    weights, _multipliers = prior_weights(inputs, density)
    np.testing.assert_allclose(weights[0], np.exp(density.log_weights()[inputs.class_index]), atol=1e-12)


def test_range_ends_hold_their_derived_bounds():
    inputs = _small(trait_count=3, column_count=40, correlation=0.3)
    lower, upper = log_multiplier_range(inputs)
    assert lower <= 0.0 <= upper
    for column in range(inputs.cavity_means.shape[1]):
        noise_sd = np.sqrt(inputs.cavity_variances[:, column])
        noise = noise_sd[:, None] * inputs.noise_correlation * noise_sd[None, :]
        scale = np.diag(inputs.prior_scale_moments[:, column])
        values = inputs.cavity_means[:, column]

        def log_density(multiplier):
            return multivariate_normal(cov=noise + multiplier * scale).logpdf(values)

        above = np.exp(upper)
        assert log_density(above * 1.01) < log_density(above)
        assert abs(log_density(np.exp(lower)) - log_density(0.0)) <= np.sqrt(np.finfo(np.float64).eps)


def test_recentring_sets_unit_mean_and_keeps_the_likelihood():
    inputs = _small()
    density = _density_for(inputs)
    before = density_statistics(inputs, density).log_likelihood
    centred, shift = recentred(density)
    mean_multiplier = np.sum(np.exp(centred.log_weights() + centred.log_multipliers), axis=1)
    np.testing.assert_allclose(mean_multiplier, 1.0, rtol=1e-12)
    rescaled = replace(inputs, prior_scale_moments=inputs.prior_scale_moments * np.exp(shift[inputs.class_index])[None, :])
    assert density_statistics(rescaled, centred).log_likelihood == pytest.approx(before, rel=1e-10)


def test_coefficient_step_never_decreases_the_penalized_likelihood():
    inputs = _simulated(4, 1500, np.random.default_rng(1).normal(-0.3, 0.8, 1500), 0.3, seed=2)
    lower, upper = log_multiplier_range(inputs)
    density = initial_density(lower, upper, 1)
    statistics = density_statistics(inputs, density)
    objective = penalized_log_likelihood(density, statistics)
    for _step in range(8):
        density = maximize_coefficients(density, statistics)
        statistics = density_statistics(inputs, density)
        new_objective = penalized_log_likelihood(density, statistics)
        assert new_objective >= objective - 1e-7
        objective = new_objective


def test_recovers_a_continuous_multiplier_density():
    spread = 0.7
    log_multipliers = np.random.default_rng(21).normal(-0.5 * spread ** 2, spread, 6000)
    inputs = _simulated(6, 6000, log_multipliers, 0.2, seed=22)
    _inputs, density = _fit(inputs, rounds=40)
    mean, deviation = _log_s_moments(density)
    assert mean == pytest.approx(-0.5 * spread ** 2, abs=0.25)
    assert deviation == pytest.approx(spread, abs=0.25)


def test_no_pleiotropy_concentrates_the_density_at_one():
    inputs = _simulated(6, 6000, np.zeros(6000), 0.2, seed=31)
    _inputs, density = _fit(inputs, rounds=40)
    mean, deviation = _log_s_moments(density)
    assert abs(mean) < 0.15
    assert deviation < 0.3

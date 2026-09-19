"""Shared pleiotropy layer: nesting, leave-one-trait-out algebra, and hyperparameter recovery."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import multivariate_normal, norm

from sv_pgs.pleiotropy_layer import (
    PleiotropyInputs,
    PleiotropyState,
    evidence,
    penalized_log_likelihood,
    prior_weights,
    update_state,
)


def _correlation(trait_count: int, value: float) -> np.ndarray:
    return np.full((trait_count, trait_count), value) + (1.0 - value) * np.eye(trait_count)


def _simulated(trait_count, column_count, rates, multiplier, correlation, seed):
    rng = np.random.default_rng(seed)
    class_index = rng.integers(0, rates.size, size=column_count)
    indicator = rng.random(column_count) < rates[class_index]
    cavity_variances = rng.uniform(0.5, 2.0, size=(trait_count, column_count))
    prior_scale_moments = rng.uniform(0.2, 1.0, size=(trait_count, column_count))
    noise_correlation = _correlation(trait_count, correlation)
    cavity_means = np.empty((trait_count, column_count))
    for column in range(column_count):
        noise_sd = np.sqrt(cavity_variances[:, column])
        covariance = noise_sd[:, None] * noise_correlation * noise_sd[None, :]
        covariance += np.diag(prior_scale_moments[:, column] * (multiplier if indicator[column] else 1.0))
        cavity_means[:, column] = rng.multivariate_normal(np.zeros(trait_count), covariance)
    inputs = PleiotropyInputs(cavity_means=cavity_means, cavity_variances=cavity_variances,
                              prior_scale_moments=prior_scale_moments, noise_correlation=noise_correlation,
                              class_index=class_index)
    return inputs


def _small_inputs(trait_count=3, column_count=7, correlation=0.4, seed=3):
    return _simulated(trait_count, column_count, np.array([0.3, 0.6]), 5.0, correlation, seed)


def test_unit_multiplier_is_exact_nesting():
    inputs = _small_inputs()
    joint, leave_one_out = evidence(inputs, 1.0)
    np.testing.assert_allclose(joint, 0.0, atol=1e-12)
    np.testing.assert_allclose(leave_one_out, 0.0, atol=1e-12)
    rates = np.array([0.2, 0.7])
    weights = prior_weights(inputs, PleiotropyState(rates=rates, scale_multiplier=1.0))
    np.testing.assert_allclose(weights, np.broadcast_to(rates[inputs.class_index], weights.shape), atol=1e-12)


def test_joint_evidence_matches_dense_multivariate_normal():
    inputs = _small_inputs()
    multiplier = 3.7
    joint, _leave_one_out = evidence(inputs, multiplier)
    for column in range(inputs.cavity_means.shape[1]):
        noise_sd = np.sqrt(inputs.cavity_variances[:, column])
        noise = noise_sd[:, None] * inputs.noise_correlation * noise_sd[None, :]
        scale = np.diag(inputs.prior_scale_moments[:, column])
        values = inputs.cavity_means[:, column]
        reference = (multivariate_normal(cov=noise + multiplier * scale).logpdf(values)
                     - multivariate_normal(cov=noise + scale).logpdf(values))
        assert joint[column] == pytest.approx(reference, rel=1e-10, abs=1e-10)


def test_leave_one_out_equals_evidence_of_the_other_traits_alone():
    inputs = _small_inputs(trait_count=4, correlation=0.3)
    multiplier = 2.5
    _joint, leave_one_out = evidence(inputs, multiplier)
    trait_count = inputs.cavity_means.shape[0]
    for trait in range(trait_count):
        others = [index for index in range(trait_count) if index != trait]
        reduced = PleiotropyInputs(cavity_means=inputs.cavity_means[others],
                                   cavity_variances=inputs.cavity_variances[others],
                                   prior_scale_moments=inputs.prior_scale_moments[others],
                                   noise_correlation=inputs.noise_correlation[np.ix_(others, others)],
                                   class_index=inputs.class_index)
        reduced_joint, _unused = evidence(reduced, multiplier)
        np.testing.assert_allclose(leave_one_out[trait], reduced_joint, rtol=1e-10, atol=1e-10)


def test_independent_noise_splits_into_per_trait_evidence():
    inputs = _small_inputs(correlation=0.0)
    multiplier = 4.0
    joint, _leave_one_out = evidence(inputs, multiplier)
    noise_var = inputs.cavity_variances
    scale = inputs.prior_scale_moments
    per_trait = (norm.logpdf(inputs.cavity_means, scale=np.sqrt(noise_var + multiplier * scale))
                 - norm.logpdf(inputs.cavity_means, scale=np.sqrt(noise_var + scale)))
    np.testing.assert_allclose(joint, per_trait.sum(axis=0), rtol=1e-10, atol=1e-10)


def test_single_trait_gets_the_rate_as_its_weight():
    inputs = _small_inputs(trait_count=1, correlation=0.0)
    rates = np.array([0.1, 0.4])
    weights = prior_weights(inputs, PleiotropyState(rates=rates, scale_multiplier=6.0))
    np.testing.assert_allclose(weights[0], rates[inputs.class_index], atol=1e-12)


def test_em_recovers_rates_and_multiplier_and_never_decreases_objective():
    true_rates = np.array([0.05, 0.25])
    inputs = _simulated(4, 20000, true_rates, 6.0, 0.3, seed=11)
    state = PleiotropyState(rates=np.array([0.5, 0.5]), scale_multiplier=2.0)
    objective = penalized_log_likelihood(inputs, state)
    for _step in range(40):
        state = update_state(inputs, state)
        new_objective = penalized_log_likelihood(inputs, state)
        assert new_objective >= objective - 1e-6
        objective = new_objective
    np.testing.assert_allclose(state.rates, true_rates, atol=0.03)
    assert state.scale_multiplier == pytest.approx(6.0, rel=0.15)


def test_no_pleiotropy_drives_multiplier_to_one():
    inputs = _simulated(4, 20000, np.array([0.2, 0.2]), 1.0, 0.3, seed=5)
    state = PleiotropyState(rates=np.array([0.3, 0.3]), scale_multiplier=3.0)
    for _step in range(40):
        state = update_state(inputs, state)
    assert state.scale_multiplier < 1.1

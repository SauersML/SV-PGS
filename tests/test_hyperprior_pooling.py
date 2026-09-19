import numpy as np
import pytest
from scipy.optimize import minimize

from sv_pgs.hyperprior_pooling import pooled_hyperprior_step


def _traits(seed, trait_count=12):
    generator = np.random.default_rng(seed)
    pooled = np.array([0.5, -0.3, 0.2])
    spread = np.array([0.3, 0.5, 0.2])
    modes, informations = [], []
    for _ in range(trait_count):
        basis = generator.normal(size=(3, 3))
        information = basis @ basis.T + np.diag(generator.uniform(10.0, 60.0, 3))
        own = pooled + spread * generator.normal(size=3)
        modes.append(own + np.linalg.solve(np.linalg.cholesky(information).T, generator.normal(size=3)))
        informations.append(information)
    return modes, informations


def _map_estimates(modes, informations, mean, variance):
    """Each trait's MAP of its quadratic objective under the prior N(mean, diag(variance))."""
    return [np.linalg.solve(information + np.diag(1.0 / variance), information @ mode + mean / variance)
            for mode, information in zip(modes, informations)]


def _log_marginal(modes, informations, variance, penalty):
    """Exact log marginal likelihood of the modes with theta_bar integrated (flat when penalty is None)."""
    precisions = [np.linalg.inv(np.diag(variance) + np.linalg.inv(information)) for information in informations]
    combined = np.sum(precisions, axis=0) + (0.0 if penalty is None else penalty)
    linear = np.sum([precision @ mode for precision, mode in zip(precisions, modes)], axis=0)
    value = sum(0.5 * np.linalg.slogdet(precision)[1] - 0.5 * mode @ precision @ mode
                for precision, mode in zip(precisions, modes))
    value += -0.5 * np.linalg.slogdet(combined)[1] + 0.5 * linear @ np.linalg.solve(combined, linear)
    if penalty is not None:
        value += 0.5 * np.linalg.slogdet(penalty)[1]
    return value


def _iterate(modes, informations, penalties, weights, steps):
    mean, variance = np.zeros(3), np.ones(3)
    weights = np.asarray(weights, dtype=float)
    for _ in range(steps):
        estimates = _map_estimates(modes, informations, mean, variance)
        step = pooled_hyperprior_step(estimates, informations, mean, variance, penalties, weights)
        mean, variance, weights = step.mean, step.variance, step.penalty_weights
    return mean, variance, weights


def test_flat_pooling_reaches_the_exact_reml_maximum():
    modes, informations = _traits(1)
    mean, variance, _ = _iterate(modes, informations, [], [], 3_000)
    fitted = minimize(lambda log_variance: -_log_marginal(modes, informations, np.exp(log_variance), None),
                      np.zeros(3), method="Nelder-Mead", options=dict(xatol=1e-10, fatol=1e-12, maxiter=20_000))
    np.testing.assert_allclose(variance, np.exp(fitted.x), rtol=1e-4)
    precisions = [np.linalg.inv(np.diag(variance) + np.linalg.inv(information)) for information in informations]
    expected_mean = np.linalg.solve(np.sum(precisions, axis=0),
                                    np.sum([precision @ mode for precision, mode in zip(precisions, modes)], axis=0))
    np.testing.assert_allclose(mean, expected_mean, rtol=1e-8, atol=1e-10)


def test_ridge_weight_on_the_pooled_map_is_learned_with_the_variances():
    modes, informations = _traits(2)
    mean, variance, weights = _iterate(modes, informations, [np.eye(3)], [1.0], 5_000)

    def objective(parameters):
        return -_log_marginal(modes, informations, np.exp(parameters[:3]), np.exp(parameters[3]) * np.eye(3))

    fitted = minimize(objective, np.zeros(4), method="Nelder-Mead",
                      options=dict(xatol=1e-10, fatol=1e-12, maxiter=40_000))
    np.testing.assert_allclose(variance, np.exp(fitted.x[:3]), rtol=1e-3)
    np.testing.assert_allclose(weights[0], np.exp(fitted.x[3]), rtol=1e-3)


def test_shifted_estimates_are_the_maps_under_the_new_mean():
    modes, informations = _traits(3)
    mean, variance = np.array([0.1, 0.0, -0.1]), np.array([0.2, 0.4, 0.1])
    estimates = _map_estimates(modes, informations, mean, variance)
    step = pooled_hyperprior_step(estimates, informations, mean, variance, [], np.array([]))
    for shifted, expected in zip(step.shifted_estimates, _map_estimates(modes, informations, step.mean, variance)):
        np.testing.assert_allclose(shifted, expected, rtol=1e-10, atol=1e-12)


def test_pooling_rejects_mismatched_inputs():
    modes, informations = _traits(4, trait_count=3)
    with pytest.raises(ValueError, match="one estimate"):
        pooled_hyperprior_step(modes[:2], informations, np.zeros(3), np.ones(3), [], np.array([]))
    with pytest.raises(ValueError, match="one weight"):
        pooled_hyperprior_step(modes, informations, np.zeros(3), np.ones(3), [np.eye(3)], np.array([]))

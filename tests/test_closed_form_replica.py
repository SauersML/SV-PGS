"""Math checks for benchmarks/closed_form (synthetic data only)."""
import numpy as np

from benchmarks.closed_form import replica
from benchmarks.closed_form.vamp import vamp


def test_scalar_mmse_is_exact_for_a_gaussian_prior_and_converged_in_order():
    for gamma in (0.1, 1.0, 37.0):
        assert np.isclose(replica.scalar_mmse(gamma, [2.0], [1.0]), 1.0 / (gamma + 0.5), rtol=1e-12)
    variances, weights = np.array([0.0, 0.01, 4.0]), np.array([0.9, 0.08, 0.02])
    grid = np.linspace(-60.0, 60.0, 600_001)
    density = sum(weight * np.exp(-0.5 * grid ** 2 / (variance + 1 / 3.0)) / np.sqrt(2 * np.pi * (variance + 1 / 3.0)) for variance, weight in zip(variances, weights))
    _, posterior_variance = replica._posterior_moments(grid, 3.0, variances, weights)
    brute = float(np.sum(density * posterior_variance) * (grid[1] - grid[0]))
    assert np.isclose(replica.scalar_mmse(3.0, variances, weights), brute, rtol=1e-6)


def test_scalar_mmse_matches_monte_carlo_for_a_sparse_mixture():
    generator = np.random.default_rng(3)
    variances, weights = np.array([0.0, 0.05, 3.0]), np.array([0.85, 0.12, 0.03])
    gamma = 2.0
    component = generator.choice(3, size=400_000, p=weights)
    beta = generator.standard_normal(component.size) * np.sqrt(variances[component])
    r = beta + generator.standard_normal(beta.size) / np.sqrt(gamma)
    mean, _ = replica._posterior_moments(r, gamma, variances, weights)
    empirical = float(np.mean((beta - mean) ** 2))
    assert abs(replica.scalar_mmse(gamma, variances, weights) - empirical) < 5 * empirical / np.sqrt(beta.size) * 3


def test_gaussian_prior_fixed_point_is_ridge_and_risk_is_the_posterior_trace():
    generator = np.random.default_rng(1)
    train = generator.standard_normal((60, 150))
    test = generator.standard_normal((40, 150))
    variance, noise = 0.004, 0.5
    eigenvalues = np.linalg.eigvalsh(train @ train.T)
    gamma2, mmse = replica.fixed_point(eigenvalues, 150, noise, [variance], [1.0])
    assert np.isclose(gamma2, 1.0 / variance, rtol=1e-8)
    posterior = np.linalg.inv(train.T @ train / noise + np.eye(150) / variance)
    assert np.isclose(mmse, np.trace(posterior) / 150, rtol=1e-8)
    risk, _ = replica.excess_risk(train, test, noise, gamma2)
    assert np.isclose(risk, np.trace(test.T @ test / 40 @ posterior), rtol=1e-9)


def test_sparse_fixed_point_matches_vamp_on_an_iid_design():
    generator = np.random.default_rng(5)
    count, dimension = 300, 600
    variances, weights = np.array([0.0, 1.0]), np.array([0.95, 0.05])
    noise = 0.3
    design = generator.standard_normal((count, dimension)) / np.sqrt(count)
    eigenvalues = np.linalg.eigvalsh(design @ design.T)
    _, predicted = replica.fixed_point(eigenvalues, dimension, noise, variances, weights)
    errors = []
    for _ in range(20):
        beta = generator.standard_normal(dimension) * (generator.random(dimension) < weights[1])
        response = design @ beta + generator.standard_normal(count) * np.sqrt(noise)
        estimate, _ = vamp(design, response, noise, variances, weights, damping=0.3, iterations=200)
        errors.append(np.mean((estimate - beta) ** 2))
    assert abs(np.mean(errors) - predicted) < 0.15 * predicted


def test_expected_sample_r2_limits():
    assert np.isclose(replica.expected_sample_r2(0.0, 101), 1.0 / 100, rtol=1e-9)
    assert np.isclose(replica.expected_sample_r2(1.0, 50), 1.0)


def test_polygenic_limit_matches_daetwyler_small_signal():
    assert np.isclose(replica.polygenic_r2(100, 0.5, 1e6), 0.5 ** 2 * 100 / 1e6, rtol=1e-3)

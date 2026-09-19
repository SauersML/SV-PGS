"""The TPB shape M-step: exact expectations and the exact optimum of its objective.

With lambda | delta ~ Gamma(a, rate delta) and delta ~ Gamma(b, 1), the M-step
maximizes E_q[log p(lambda, delta | a, b)], which needs E_q[log lambda] and
E_q[log delta], not log E[lambda] and log E[delta]:
  E[log lambda] = 0.5 log(chi / psi) + d/dnu log K_nu(sqrt(chi psi)) at nu = a - 1/2,
  E[log delta] = digamma(a + b) - log(1 + E[lambda]).
For a = 1 the first is elementary: 0.5 log(chi / psi) + exp(2z) E_1(2z).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate, optimize
from scipy.special import digamma, exp1, gammaln, kve

from sv_pgs.config import ModelConfig
from sv_pgs.mixture_inference import (
    _gig_log_moment,
    _local_scale_log_expectations,
    _update_local_scales,
    _update_tpb_shape_vectors,
)


def _quadrature_gig_log_moment(p_parameter: float, chi: float, psi: float) -> float:
    # Integrate in u = log x around the mode of the u-density, e^u = (p + s) / psi,
    # s = sqrt(p^2 + chi psi), written as chi / (s - p) when p < 0.
    root = np.sqrt(p_parameter * p_parameter + chi * psi)
    mode = (p_parameter + root) / psi if p_parameter >= 0.0 else chi / (root - p_parameter)
    center = float(np.log(mode))
    z_value = np.sqrt(chi * psi)
    log_normaliser = 0.5 * p_parameter * np.log(psi / chi) - np.log(2.0 * kve(p_parameter, z_value)) + z_value

    def density(u_value: float) -> float:
        x_value = np.exp(u_value)
        return float(np.exp(log_normaliser + p_parameter * u_value - 0.5 * (chi / x_value + psi * x_value)))

    first, _ = integrate.quad(lambda u: u * density(u), center - 60.0, center + 60.0, points=[center], limit=400)
    return float(first)


@pytest.mark.parametrize("p_parameter", [-0.4, -0.1, 0.0, 0.1, 0.5, 2.3, 9.5])
def test_gig_log_moment_matches_quadrature(p_parameter: float) -> None:
    chi = np.array([1e-12, 1e-6, 1e-2, 1.0, 1e3])
    psi = np.array([2e-8, 0.3, 3.0, 40.0, 2.0])
    reference = np.array([_quadrature_gig_log_moment(p_parameter, c, s) for c, s in zip(chi, psi)])
    moment = _gig_log_moment(p_parameter=np.full(chi.shape, p_parameter), chi=chi, psi=psi)
    np.testing.assert_allclose(moment, reference, rtol=0.0, atol=1e-7)


def test_gig_log_moment_matches_the_exponential_integral_form_at_shape_one() -> None:
    chi = np.array([1e-10, 1e-4, 0.3, 5.0, 1e3])
    psi = np.array([2.0, 0.5, 1.0, 3.0, 0.01])
    two_z = 2.0 * np.sqrt(chi * psi)
    closed_form = 0.5 * np.log(chi / psi) + np.exp(two_z) * exp1(two_z)
    moment = _gig_log_moment(p_parameter=np.full(chi.shape, 0.5), chi=chi, psi=psi)
    np.testing.assert_allclose(moment, closed_form, rtol=0.0, atol=1e-7)


def test_log_expectations_come_from_the_factors_the_local_update_formed() -> None:
    config = ModelConfig()
    second_moment = np.array([4.0, 0.25, 16.0])
    baseline = np.array([1.0, 0.5, 4.0])
    shape_a = np.array([1.0, 2.0, 0.75])
    shape_b = np.array([0.5, 1.0, 0.5])
    delta = np.array([0.7, 1.2, 0.4])
    local_scale, _ = _update_local_scales(second_moment, baseline, shape_a, shape_b, delta, config)
    log_scale, log_delta = _local_scale_log_expectations(
        coefficient_second_moment=second_moment,
        baseline_prior_variances=baseline,
        local_shape_a=shape_a,
        local_shape_b=shape_b,
        auxiliary_delta=delta,
        local_scale=local_scale,
        config=config,
    )
    chi, psi = second_moment / baseline, 2.0 * delta
    expected_log_scale = [_quadrature_gig_log_moment(a - 0.5, c, s) for a, c, s in zip(shape_a, chi, psi)]
    np.testing.assert_allclose(log_scale, expected_log_scale, atol=1e-6)
    np.testing.assert_allclose(log_delta, digamma(shape_a + shape_b) - np.log1p(local_scale), rtol=1e-12)
    # Jensen: the plug-in logs overstate both expectations.
    assert np.all(np.log(local_scale) > log_scale)
    assert np.all(np.log((shape_a + shape_b) / (1.0 + local_scale)) > log_delta)


def _hard_membership(sizes: list[int]) -> np.ndarray:
    membership = np.zeros((sum(sizes), len(sizes)))
    start = 0
    for class_index, size in enumerate(sizes):
        membership[start:start + size, class_index] = 1.0
        start += size
    return membership


def test_shape_m_step_without_the_pull_is_the_gamma_shape_mle() -> None:
    rng = np.random.default_rng(4)
    membership = _hard_membership([3000, 2000, 1000])
    log_scale = rng.normal(-1.0, 1.5, size=membership.shape[0])
    log_delta = rng.normal(-0.3, 0.6, size=membership.shape[0])
    config = ModelConfig(tpb_hierarchical_prior_variance=1e12)
    shape_a, shape_b = _update_tpb_shape_vectors(
        class_membership_matrix=membership,
        current_shape_a_vector=np.ones(3),
        current_shape_b_vector=np.full(3, 0.5),
        expected_log_local_scale=log_scale,
        expected_log_auxiliary_delta=log_delta,
        config=config,
    )
    labels = np.argmax(membership, axis=1)
    for class_index in range(3):
        members = labels == class_index
        target_a = float(np.mean(log_delta[members] + log_scale[members]))
        target_b = float(np.mean(log_delta[members]))
        exact_a = optimize.brentq(lambda value: digamma(value) - target_a, 1e-3, 1e3, xtol=1e-14)
        exact_b = optimize.brentq(lambda value: digamma(value) - target_b, 1e-3, 1e3, xtol=1e-14)
        assert shape_a[class_index] == pytest.approx(np.clip(exact_a, 0.1, 10.0), rel=1e-9)
        assert shape_b[class_index] == pytest.approx(np.clip(exact_b, 0.1, 10.0), rel=1e-9)


def _shape_objective(log_shape: np.ndarray, membership: np.ndarray, statistic: np.ndarray, pull: float) -> float:
    local_shape = membership @ np.exp(log_shape)
    centered = log_shape - np.mean(log_shape)
    return float(np.sum(local_shape * statistic - gammaln(local_shape)) - 0.5 * pull * np.dot(centered, centered))


def test_shape_m_step_is_the_optimum_with_the_pull_and_mixed_memberships() -> None:
    rng = np.random.default_rng(9)
    membership = np.vstack([
        np.tile([1.0, 0.0, 0.0], (400, 1)),
        np.tile([0.0, 1.0, 0.0], (300, 1)),
        np.tile([0.5, 0.5, 0.0], (100, 1)),
        np.tile([0.0, 0.0, 1.0], (50, 1)),
    ])
    log_scale = rng.normal(-2.0, 1.0, size=membership.shape[0])
    log_delta = rng.normal(-0.5, 0.4, size=membership.shape[0])
    config = ModelConfig(maximum_tpb_shape_iterations=50)
    shape_a, shape_b = _update_tpb_shape_vectors(
        class_membership_matrix=membership,
        current_shape_a_vector=np.ones(3),
        current_shape_b_vector=np.full(3, 0.5),
        expected_log_local_scale=log_scale,
        expected_log_auxiliary_delta=log_delta,
        config=config,
    )
    bounds = [(np.log(config.minimum_tpb_shape), np.log(config.maximum_tpb_shape))] * 3
    pull = 1.0 / config.tpb_hierarchical_prior_variance
    for shape, statistic in [(shape_a, log_delta + log_scale), (shape_b, log_delta)]:
        reference = optimize.minimize(
            lambda value: -_shape_objective(value, membership, statistic, pull),
            np.zeros(3),
            method="L-BFGS-B",
            bounds=bounds,
            options=dict(ftol=1e-15, gtol=1e-12, maxiter=10_000),
        )
        assert _shape_objective(np.log(shape), membership, statistic, pull) >= -reference.fun - 1e-9
        np.testing.assert_allclose(np.log(shape), reference.x, atol=1e-5)


def test_shape_m_step_stops_at_the_configured_bound() -> None:
    membership = _hard_membership([500])
    config = ModelConfig()
    shape_a, shape_b = _update_tpb_shape_vectors(
        class_membership_matrix=membership,
        current_shape_a_vector=np.ones(1),
        current_shape_b_vector=np.full(1, 0.5),
        expected_log_local_scale=np.full(500, 15.0),
        expected_log_auxiliary_delta=np.full(500, -12.0),
        config=config,
    )
    assert shape_a[0] == pytest.approx(config.maximum_tpb_shape)
    assert shape_b[0] == pytest.approx(config.minimum_tpb_shape)


def test_exact_expectations_keep_the_shapes_off_the_bounds() -> None:
    # Orthogonal design, beta_hat_j = beta_j + N(0, s^2), beta ~ TPB(a=1, b=1/2),
    # iterated with the plug-in beta and scale steps the EM loop uses. The plug-in
    # log E[.] statistics drive both shapes to the upper bound; the exact
    # expectations stay near the truth.
    rng = np.random.default_rng(1)
    variant_count, noise_sd, true_scale = 4000, 1.0 / np.sqrt(1e5), 1e-3
    lam = rng.gamma(1.0, 1.0 / rng.gamma(0.5, 1.0, size=variant_count))
    beta_hat = rng.standard_normal(variant_count) * true_scale * np.sqrt(lam)
    beta_hat += rng.standard_normal(variant_count) * noise_sd
    config = ModelConfig()
    membership = np.ones((variant_count, 1))
    results = {}
    for exact in (False, True):
        shape_a, shape_b = np.ones(1), np.full(1, 0.5)
        scale2 = max(float(np.mean(beta_hat**2)) - noise_sd**2, 1e-12)
        local_scale, delta = np.ones(variant_count), np.full(variant_count, 0.5)
        for _ in range(150):
            variance = 1.0 / (1.0 / noise_sd**2 + 1.0 / (scale2 * local_scale))
            second_moment = (variance * beta_hat / noise_sd**2) ** 2 + variance
            baseline = np.full(variant_count, scale2)
            local_a = membership @ shape_a
            local_b = membership @ shape_b
            new_scale, new_delta = _update_local_scales(second_moment, baseline, local_a, local_b, delta, config)
            scale2 = float(np.mean(second_moment / new_scale))
            if exact:
                log_scale, log_delta = _local_scale_log_expectations(
                    coefficient_second_moment=second_moment,
                    baseline_prior_variances=baseline,
                    local_shape_a=local_a,
                    local_shape_b=local_b,
                    auxiliary_delta=delta,
                    local_scale=new_scale,
                    config=config,
                )
            else:
                log_scale, log_delta = np.log(new_scale), np.log(new_delta)
            shape_a, shape_b = _update_tpb_shape_vectors(membership, shape_a, shape_b, log_scale, log_delta, config)
            local_scale = new_scale
            delta = (membership @ shape_a + membership @ shape_b) / (1.0 + local_scale)
        results[exact] = (float(shape_a[0]), float(shape_b[0]))
    assert results[False][0] == pytest.approx(config.maximum_tpb_shape)
    assert results[False][1] == pytest.approx(config.maximum_tpb_shape)
    assert results[True][0] < 4.0
    assert results[True][1] < 2.0

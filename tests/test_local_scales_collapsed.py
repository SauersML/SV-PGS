"""Tests for the collapsed (single CAVI step) local-scale update.

The inner lambda <-> delta fixed-point loop has been collapsed to a single
closed-form GIG moment evaluation.  These tests verify:

  1. The function returns the analytic GIG mean E[lambda] in one call.
  2. The fixed point is reached when the function is re-applied with the
     returned delta (i.e., the inner iteration the old code performed is
     now handled by the outer EM loop in 1-2 calls).
"""

from __future__ import annotations

import numpy as np
from scipy.special import kve

from sv_pgs.mixture_inference import ModelConfig, _update_local_scales


def _analytic_gig_mean(
    coefficient_second_moment: np.ndarray,
    baseline_prior_variances: np.ndarray,
    local_shape_a: np.ndarray,
    auxiliary_delta: np.ndarray,
) -> np.ndarray:
    chi = coefficient_second_moment / baseline_prior_variances
    psi = 2.0 * auxiliary_delta
    p_parameter = local_shape_a - 0.5
    z_value = np.sqrt(chi * psi)
    return np.sqrt(chi / psi) * (
        kve(np.abs(p_parameter + 1.0), z_value) / kve(np.abs(p_parameter), z_value)
    )


def test_single_call_returns_gig_moment() -> None:
    config = ModelConfig()
    coefficient_second_moment = np.array([4.0, 0.25, 16.0], dtype=np.float64)
    baseline_prior_variances = np.array([1.0, 0.5, 4.0], dtype=np.float64)
    local_shape_a = np.array([1.5, 2.0, 0.75], dtype=np.float64)
    local_shape_b = np.array([0.5, 1.0, 0.5], dtype=np.float64)
    auxiliary_delta = np.array([0.7, 1.2, 0.4], dtype=np.float64)

    updated_local_scale, updated_auxiliary_delta = _update_local_scales(
        coefficient_second_moment=coefficient_second_moment,
        baseline_prior_variances=baseline_prior_variances,
        local_shape_a=local_shape_a,
        local_shape_b=local_shape_b,
        auxiliary_delta=auxiliary_delta,
        config=config,
    )

    expected_local_scale = _analytic_gig_mean(
        coefficient_second_moment=coefficient_second_moment,
        baseline_prior_variances=baseline_prior_variances,
        local_shape_a=local_shape_a,
        auxiliary_delta=auxiliary_delta,
    )
    expected_auxiliary_delta = (local_shape_a + local_shape_b) / (1.0 + expected_local_scale)

    np.testing.assert_allclose(updated_local_scale, expected_local_scale, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(updated_auxiliary_delta, expected_auxiliary_delta, atol=1e-10, rtol=1e-10)


def test_idempotent_at_fixed_point() -> None:
    """Calling the update with the returned delta should be near a fixed point."""
    config = ModelConfig()
    coefficient_second_moment = np.array([4.0, 0.25, 16.0], dtype=np.float64)
    baseline_prior_variances = np.array([1.0, 0.5, 4.0], dtype=np.float64)
    local_shape_a = np.array([1.5, 2.0, 0.75], dtype=np.float64)
    local_shape_b = np.array([0.5, 1.0, 0.5], dtype=np.float64)
    auxiliary_delta = np.array([0.7, 1.2, 0.4], dtype=np.float64)

    # Iterate the function the way the outer EM would: feed the returned
    # delta back in.  After a handful of calls the joint (lambda, delta) is
    # at the fixed point that the old inner loop reached.
    # Geometric contraction rate of the inner (lambda, delta) fixed-point
    # depends on the problem; ~50 calls is enough to land within 1e-8 for
    # these inputs.
    current_delta = auxiliary_delta
    previous_local_scale = None
    for _ in range(60):
        updated_local_scale, current_delta = _update_local_scales(
            coefficient_second_moment=coefficient_second_moment,
            baseline_prior_variances=baseline_prior_variances,
            local_shape_a=local_shape_a,
            local_shape_b=local_shape_b,
            auxiliary_delta=current_delta,
            config=config,
        )
        previous_local_scale = updated_local_scale

    # One more call should reproduce the same (lambda, delta) to ~1e-8.
    refixed_local_scale, refixed_delta = _update_local_scales(
        coefficient_second_moment=coefficient_second_moment,
        baseline_prior_variances=baseline_prior_variances,
        local_shape_a=local_shape_a,
        local_shape_b=local_shape_b,
        auxiliary_delta=current_delta,
        config=config,
    )
    np.testing.assert_allclose(refixed_local_scale, previous_local_scale, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(refixed_delta, current_delta, atol=1e-8, rtol=1e-8)

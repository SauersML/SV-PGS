"""Tests for sv_pgs.optimizer_helpers."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from sv_pgs.optimizer_helpers import (
    closed_form_global_scale,
    gig_inverse_first_moment,
    pack_em_hyperparameters,
    unpack_em_hyperparameters,
)


def test_closed_form_global_scale_recovers_known_sigma_g():
    """Generate beta^2 consistent with a known sigma_g* and verify recovery."""
    rng = np.random.default_rng(0)
    p = 5000
    sigma_g_true = 0.37
    s_j = np.exp(rng.normal(0.0, 0.3, size=p))
    lam = rng.gamma(shape=2.0, scale=0.5, size=p) + 1e-3
    # Draw beta_j ~ N(0, (sigma_g * s_j)^2 * lam_j); use squared draws as E[beta^2].
    tau2 = (sigma_g_true * s_j) ** 2 * lam
    beta = rng.normal(0.0, np.sqrt(tau2))
    beta_sq = beta ** 2  # noisy estimate of E[beta^2]; with p=5000 mean concentrates

    sigma_g_hat = closed_form_global_scale(
        coefficient_second_moment=beta_sq,
        metadata_baseline_scales=s_j,
        local_scale=lam,
        prior_shape=0.0,
        prior_rate=0.0,
    )
    rel_err = abs(sigma_g_hat - sigma_g_true) / sigma_g_true
    assert rel_err < 0.05, f"sigma_g recovery off: got {sigma_g_hat}, true {sigma_g_true}, rel_err={rel_err}"


def test_closed_form_global_scale_clips_to_bounds():
    p = 10
    out_lo = closed_form_global_scale(
        coefficient_second_moment=np.zeros(p),
        metadata_baseline_scales=np.ones(p),
        local_scale=np.ones(p),
        floor=1e-4,
        ceiling=1.0,
    )
    assert out_lo == pytest.approx(1e-4)

    out_hi = closed_form_global_scale(
        coefficient_second_moment=np.full(p, 1e20),
        metadata_baseline_scales=np.ones(p),
        local_scale=np.ones(p),
        floor=1e-8,
        ceiling=10.0,
    )
    assert out_hi == pytest.approx(10.0)


def test_gig_inverse_first_moment_matches_inverse_gamma_limit():
    """GIG(p, chi, psi) with psi -> 0 and p < 0 reduces to InvGamma(-p, chi/2).

    For X ~ InvGamma(alpha, beta): E[1/X] = alpha / beta.
    So with p = -alpha, chi = 2*beta, psi -> 0: E[1/X] = alpha / beta = -2 p / chi.
    """
    alpha = 3.5
    beta = 2.0
    p_param = np.array([-alpha])
    chi = np.array([2.0 * beta])
    psi = np.array([1e-12])  # near-zero limit

    e_inv = gig_inverse_first_moment(p_parameter=p_param, chi=chi, psi=psi)
    expected = alpha / beta

    # Cross-check via scipy.stats.invgamma sampling.
    rv = stats.invgamma(a=alpha, scale=beta)
    sample_e_inv = np.mean(1.0 / rv.rvs(size=200000, random_state=0))

    assert e_inv[0] == pytest.approx(expected, rel=1e-3), (
        f"GIG E[1/X] in InvGamma limit: got {e_inv[0]}, expected {expected}"
    )
    assert e_inv[0] == pytest.approx(sample_e_inv, rel=5e-2)


def test_gig_inverse_first_moment_vectorized():
    """Vectorization smoke test: independent inputs give independent outputs."""
    p_param = np.array([-2.0, -3.0, -4.0])
    chi = np.array([2.0, 4.0, 6.0])
    psi = np.full(3, 1e-12)
    out = gig_inverse_first_moment(p_parameter=p_param, chi=chi, psi=psi)
    # Each is InvGamma(-p, chi/2), so E[1/X] = (-p) / (chi/2) = -2p/chi.
    expected = -2.0 * p_param / chi
    np.testing.assert_allclose(out, expected, rtol=1e-3)


def test_pack_unpack_em_hyperparameters_roundtrip():
    rng = np.random.default_rng(42)
    log_sg = -0.42
    scale_coefs = rng.normal(size=7)
    log_a = rng.normal(size=3)
    log_b = rng.normal(size=3)
    packed = pack_em_hyperparameters(
        log_global_scale=log_sg,
        scale_model_coefficients=scale_coefs,
        log_tpb_shape_a_vector=log_a,
        log_tpb_shape_b_vector=log_b,
    )
    assert packed.shape == (1 + 7 + 3 + 3,)
    sg2, sc2, la2, lb2 = unpack_em_hyperparameters(
        packed, scale_model_dim=7, tpb_class_count=3
    )
    assert sg2 == pytest.approx(log_sg)
    np.testing.assert_array_equal(sc2, scale_coefs)
    np.testing.assert_array_equal(la2, log_a)
    np.testing.assert_array_equal(lb2, log_b)

    repacked = pack_em_hyperparameters(
        log_global_scale=sg2,
        scale_model_coefficients=sc2,
        log_tpb_shape_a_vector=la2,
        log_tpb_shape_b_vector=lb2,
    )
    np.testing.assert_array_equal(repacked, packed)

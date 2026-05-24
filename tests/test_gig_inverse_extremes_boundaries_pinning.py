"""Boundary-condition pinning for ``gig_inverse_first_moment``.

Pins behavior at numerically extreme inputs that the optimizer can hit on
real data:

* ``chi`` at its floor (1e-300) — the helper clamps to ``np.finfo.tiny``
  and must still return a finite positive value rather than +inf or NaN.
* ``psi`` very large (1e300) — the ratio formula must not produce NaN.
* ``p = 0`` — neither the inverse-gamma nor the ``p > 1`` branch applies;
  must still be finite.
* ``p = -0.5`` (the GIG prior on a positive scale) at ``chi/psi`` extremes.
* Very negative ``p`` (alpha-stable regime, e.g. ``p = -50``) — inverse-
  gamma analytic limit should activate when ``psi -> 0``.

These complement ``test_gig_edge_cases_pinning.py`` which already pins
large-z and broadcast behavior; this file pins the under-flow and zero-p
boundaries.
"""
from __future__ import annotations

import numpy as np

from sv_pgs.optimizer_helpers import gig_inverse_first_moment


def _call(p, chi, psi):
    return gig_inverse_first_moment(
        p_parameter=np.asarray(p, dtype=np.float64),
        chi=np.asarray(chi, dtype=np.float64),
        psi=np.asarray(psi, dtype=np.float64),
    )


def test_chi_at_floor_returns_finite_positive():
    """``chi = 1e-300`` (effective zero) must not produce inf or NaN."""
    result = _call([-0.5], [1e-300], [1.0])
    assert np.all(np.isfinite(result)), result
    assert np.all(result > 0.0), result


def test_psi_huge_returns_finite_positive():
    """``psi = 1e300`` must not feed +inf to the Bessel ratio."""
    result = _call([-0.5], [1.0], [1e300])
    assert np.all(np.isfinite(result)), result
    assert np.all(result > 0.0), result


def test_p_exactly_zero_returns_finite_positive():
    """p=0 lies between the negative-p inverse-gamma branch and the p>1
    branch; result must remain finite and positive (no division by zero
    in any branch)."""
    result = _call([0.0], [1.0], [1.0])
    assert np.all(np.isfinite(result)), result
    assert np.all(result > 0.0), result


def test_p_negative_half_at_tiny_chi_and_psi():
    """The local-scale GIG prior uses ``p = -0.5``; both psi and chi may
    underflow during early optimization iterations.  Must not blow up."""
    result = _call([-0.5], [1e-300], [1e-300])
    assert np.all(np.isfinite(result)), result
    assert np.all(result >= np.finfo(np.float64).tiny)


def test_very_negative_p_with_psi_to_zero_matches_inverse_gamma():
    """For ``p = -50`` and ``psi -> 0`` the analytic limit is ``-2p/chi``.
    Pin the inverse-gamma short-circuit at the alpha-stable regime."""
    chi = np.asarray([2.0], dtype=np.float64)
    psi = np.asarray([1e-300], dtype=np.float64)
    p = np.asarray([-50.0], dtype=np.float64)
    result = gig_inverse_first_moment(p_parameter=p, chi=chi, psi=psi)
    # -2 * (-50) / 2 == 50.0
    np.testing.assert_allclose(result, 50.0, rtol=1e-6)


def test_vector_inputs_all_finite():
    """Sweep a vector of (p, chi, psi) tuples that span the boundary
    regimes — every entry must be finite and positive."""
    p = np.asarray([-2.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype=np.float64)
    chi = np.full(p.shape, 1e-200, dtype=np.float64)
    psi = np.full(p.shape, 1e200, dtype=np.float64)
    result = gig_inverse_first_moment(p_parameter=p, chi=chi, psi=psi)
    assert result.shape == p.shape
    assert np.all(np.isfinite(result)), result
    assert np.all(result > 0.0), result

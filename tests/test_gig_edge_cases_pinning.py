"""Pin numerical stability of ``gig_inverse_first_moment`` at extremes.

The optimizer hammers this helper across many local-scale updates per
iteration. Two failure modes are easy to introduce by accident:

* ``chi*psi`` overflows to ``inf`` for large inputs, producing ``nan`` in the
  ratio K_{p-1}(z)/K_p(z) and corrupting the prior precision silently.
* The tiny-z branch must never propagate ``-inf`` when ``log(0)`` is hit
  on a near-zero ``z``.

The asymptotic behaviour we pin: for large z, K_{nu}(z) ~ K_{nu+0}(z), so
sqrt(psi/chi) * K_{p-1}/K_p approaches sqrt(psi/chi). For chi == psi (any
large value), the result must approach 1.
"""
from __future__ import annotations

import numpy as np

from sv_pgs.optimizer_helpers import gig_inverse_first_moment


def test_large_equal_chi_psi_does_not_overflow_to_nan():
    """chi = psi = 1e200 → chi*psi == 1e400 (overflows float64). The
    implementation must guard against feeding +inf to scipy.special.kve."""
    chi = np.asarray([1e200], dtype=np.float64)
    psi = np.asarray([1e200], dtype=np.float64)
    p = np.asarray([-0.5], dtype=np.float64)
    result = gig_inverse_first_moment(p_parameter=p, chi=chi, psi=psi)
    assert np.all(np.isfinite(result)), f"result must be finite; got {result}"
    # Asymptotically sqrt(psi/chi) == 1 and the Bessel ratio -> 1.
    assert result[0] > 0.0


def test_large_psi_small_chi_returns_finite_positive():
    """psi >> chi but neither extreme — must return a finite positive value."""
    result = gig_inverse_first_moment(
        p_parameter=np.asarray([-0.5], dtype=np.float64),
        chi=np.asarray([1e-6], dtype=np.float64),
        psi=np.asarray([1e10], dtype=np.float64),
    )
    assert np.all(np.isfinite(result))
    assert result[0] > 0.0


def test_tiny_z_with_negative_p_uses_inverse_gamma_limit():
    """When psi -> 0 and p < 0, the analytic limit is -2p/chi. Pin the
    formula path."""
    chi = np.asarray([4.0], dtype=np.float64)
    psi = np.asarray([1e-300], dtype=np.float64)
    p = np.asarray([-1.0], dtype=np.float64)
    result = gig_inverse_first_moment(p_parameter=p, chi=chi, psi=psi)
    # -2 * (-1.0) / 4.0 == 0.5
    np.testing.assert_allclose(result, 0.5, rtol=1e-6)


def test_broadcasted_inputs_return_correct_shape():
    """Inputs may be broadcast (scalar p, vector chi/psi). The output shape
    must match the broadcast shape — no silent reshaping."""
    chi = np.full(5, 1.0, dtype=np.float64)
    psi = np.full(5, 1.0, dtype=np.float64)
    p = np.asarray(-0.5, dtype=np.float64)
    result = gig_inverse_first_moment(p_parameter=p, chi=chi, psi=psi)
    assert result.shape == (5,)
    assert np.all(np.isfinite(result))
    assert np.all(result > 0.0)


def test_floor_prevents_zero_output():
    """The returned value is floored at np.finfo(float64).tiny so downstream
    1/E[lambda] never divides by zero."""
    chi = np.asarray([1e-300], dtype=np.float64)
    psi = np.asarray([1e-300], dtype=np.float64)
    p = np.asarray([0.5], dtype=np.float64)  # not in the psi->0 branch
    result = gig_inverse_first_moment(p_parameter=p, chi=chi, psi=psi)
    assert np.all(result >= np.finfo(np.float64).tiny)
    assert np.all(np.isfinite(result))

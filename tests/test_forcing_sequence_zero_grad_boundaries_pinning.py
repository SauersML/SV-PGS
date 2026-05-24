"""Pin: ``forcing_tolerance`` at gradient_norm=0 returns finite η bounded
by ``eta_max``.

At iteration 0 of an inner solver the outer gradient can be exactly zero
(trivially converged starting point).  ``g^q`` with ``g=0`` is 0; the
implementation clips to ``[eta_min, eta_max]``, so the returned value
must be:

* finite,
* >= ``eta_min`` (after the sqrt(eps) safeguard if enabled),
* <= ``eta_max``,
* never NaN or negative.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from sv_pgs.forcing_sequence import forcing_tolerance, relaxed_iteration_cap


def test_zero_gradient_returns_finite_and_bounded():
    eta = forcing_tolerance(gradient_norm=0.0, eta_max=0.5, eta_min=1e-12)
    assert np.isfinite(eta)
    assert 0.0 < eta <= 0.5


def test_zero_gradient_safeguard_clips_above_sqrt_eps():
    """With ``safeguard_against_oversolve=True`` (default), the floor is
    at least ``sqrt(eps_float64) ≈ 1.5e-8``."""
    eta = forcing_tolerance(
        gradient_norm=0.0,
        eta_max=0.5,
        eta_min=1e-12,
        safeguard_against_oversolve=True,
    )
    assert eta >= math.sqrt(np.finfo(np.float64).eps)


def test_zero_gradient_no_safeguard_respects_explicit_floor():
    """Disable the safeguard: η must just be eta_min."""
    eta = forcing_tolerance(
        gradient_norm=0.0,
        eta_max=0.5,
        eta_min=1e-10,
        safeguard_against_oversolve=False,
    )
    np.testing.assert_allclose(eta, 1e-10, rtol=0.0, atol=0.0)


def test_negative_gradient_raises():
    with pytest.raises(ValueError):
        forcing_tolerance(gradient_norm=-1.0)


def test_non_finite_gradient_raises():
    with pytest.raises(ValueError):
        forcing_tolerance(gradient_norm=float("inf"))
    with pytest.raises(ValueError):
        forcing_tolerance(gradient_norm=float("nan"))


def test_relaxed_iteration_cap_at_eta_close_to_one_returns_minimum_cap():
    cap = relaxed_iteration_cap(forcing_tolerance_value=0.9, base_cap=128, minimum_cap=16)
    assert cap == 16


def test_relaxed_iteration_cap_at_eta_below_floor_saturates_at_base():
    cap = relaxed_iteration_cap(
        forcing_tolerance_value=1e-12,
        base_cap=128,
        minimum_cap=16,
        log_tol_floor=-12.0,
    )
    assert cap == 128

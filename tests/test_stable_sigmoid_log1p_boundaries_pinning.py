"""Boundary pinning for stable sigmoid, log1p(exp(x)), and softplus.

These three functions appear in the binary-trait likelihood, the ELBO
log-sigmoid bound, and the TR-Newton objective.  Each must:

* never produce NaN at the extremes (±inf, ±700, ±100, 0),
* preserve monotonicity (σ is non-decreasing, log1p(exp(x)) is strictly
  increasing for finite x),
* return values in the documented range (σ ∈ [0, 1], log1p(exp(x)) ≥ 0).

A regression here would silently corrupt every binary-trait fit.
"""
from __future__ import annotations

import numpy as np

from sv_pgs.numeric import stable_sigmoid
from sv_pgs.tr_newton import _log1p_exp


def _np_sigmoid(values: np.ndarray) -> np.ndarray:
    return np.asarray(stable_sigmoid(values), dtype=np.float64)


def test_stable_sigmoid_at_extremes_is_finite_and_in_unit_interval():
    inputs = np.asarray(
        [-np.inf, -700.0, -100.0, -1.0, 0.0, 1.0, 100.0, 700.0, np.inf],
        dtype=np.float64,
    )
    out = _np_sigmoid(inputs)
    assert not np.any(np.isnan(out)), out
    # σ(-inf) -> 0, σ(+inf) -> 1; intermediate values must stay in [0, 1].
    assert np.all(out >= 0.0) and np.all(out <= 1.0), out


def test_stable_sigmoid_at_zero_equals_half():
    out = _np_sigmoid(np.asarray([0.0], dtype=np.float64))
    np.testing.assert_allclose(out, 0.5, atol=1e-7)


def test_stable_sigmoid_monotonic_on_dense_grid():
    grid = np.linspace(-50.0, 50.0, 1001, dtype=np.float64)
    out = _np_sigmoid(grid)
    diffs = np.diff(out)
    # Non-decreasing under floating-point: small negative slack only.
    assert np.all(diffs >= -1e-12), float(diffs.min())


def test_stable_sigmoid_saturates_correctly_at_large_magnitude():
    # σ(-700) underflows naively but the stable branch keeps it at 0 exactly.
    # σ(+700) saturates at 1.0 (or just under, by 1 ulp).
    out_neg = _np_sigmoid(np.asarray([-700.0], dtype=np.float64))
    out_pos = _np_sigmoid(np.asarray([+700.0], dtype=np.float64))
    assert 0.0 <= float(out_neg[0]) <= 1e-300
    assert 1.0 - 1e-12 <= float(out_pos[0]) <= 1.0


def test_log1p_exp_at_extremes_is_finite_and_nonnegative():
    inputs = np.asarray(
        [-700.0, -100.0, -1.0, 0.0, 1.0, 100.0, 700.0],
        dtype=np.float64,
    )
    out = _log1p_exp(inputs)
    assert np.all(np.isfinite(out)), out
    # log(1 + e^x) >= 0 for all real x.
    assert np.all(out >= 0.0), out


def test_log1p_exp_handles_positive_infinity_without_nan():
    """numpy's logaddexp(0, inf) returns +inf cleanly (not NaN).  Pin
    that — the inner-loop objective relies on this branch behavior."""
    out = _log1p_exp(np.asarray([np.inf], dtype=np.float64))
    assert np.isposinf(out[0])


def test_log1p_exp_large_positive_equals_input_to_machine_precision():
    """For very large x, log(1 + e^x) ≈ x.  At x=700 the difference is
    well below 1e-12 in float64."""
    x = 700.0
    out = float(_log1p_exp(np.asarray([x], dtype=np.float64))[0])
    assert abs(out - x) < 1e-12


def test_log1p_exp_monotonic_on_dense_grid():
    grid = np.linspace(-50.0, 50.0, 1001, dtype=np.float64)
    out = _log1p_exp(grid)
    diffs = np.diff(out)
    assert np.all(diffs >= -1e-12), float(diffs.min())


def test_softplus_log_sigmoid_identity():
    """``_log_sigmoid(x) = -softplus(-x)`` — pin the identity at the
    extremes used by the binary ELBO."""
    from sv_pgs.elbo import _log_sigmoid

    x = np.asarray([-700.0, -1.0, 0.0, 1.0, 700.0], dtype=np.float64)
    out = _log_sigmoid(x)
    expected = -np.logaddexp(0.0, -x)
    np.testing.assert_allclose(out, expected, atol=1e-15)
    # log σ(x) is always <= 0.
    assert np.all(out <= 1e-15), out

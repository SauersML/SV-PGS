"""Pin: Anderson(m=1) acceleration reduces to plain fixed-point iteration.

With ``memory_depth=1`` the residual history is length 1 → the first call
to ``anderson_step`` returns the plain map value (no acceleration is
possible).  On subsequent calls, Anderson(1) can still extrapolate from
a single residual difference, but with the strict safeguard the accepted
iterate must equal the plain step whenever the accelerated proposal
either degenerates or fails the safeguard.

Goal: pin that ``safeguarded_anderson(memory_depth=1)`` on a strict
contraction converges to the same fixed point as plain iteration and
does not raise.
"""
from __future__ import annotations

import numpy as np

from sv_pgs.anderson import AndersonState, anderson_step, safeguarded_anderson


def test_anderson_step_depth_one_first_call_returns_map_value():
    """With empty history, ``anderson_step`` must return the plain map
    value (no acceleration available)."""
    state = AndersonState(memory_depth=1)
    x = np.zeros(3, dtype=np.float64)
    t = np.asarray([0.5, -0.25, 1.0], dtype=np.float64)
    out = anderson_step(state, x_current=x, map_value=t)
    np.testing.assert_array_equal(out, t)


def test_anderson_depth_one_converges_to_fixed_point():
    """A strict contraction must still converge under Anderson(1)."""
    contraction = 0.5
    fixed_point_value = 2.0  # x* satisfies x = 0.5 x + 1 → x = 2

    def fixed_point_map(vector: np.ndarray) -> np.ndarray:
        return contraction * vector + 1.0

    def objective(vector: np.ndarray) -> float:
        # Maximised at the fixed point (negative residual).
        return -float(np.linalg.norm(fixed_point_map(vector) - vector))

    result, history, converged = safeguarded_anderson(
        initial_iterate=np.zeros(1, dtype=np.float64),
        fixed_point_map=fixed_point_map,
        objective=objective,
        memory_depth=1,
        max_iterations=200,
        tolerance=1e-10,
        safeguard_slack=1e-6,
    )
    assert converged, history[-3:]
    np.testing.assert_allclose(result, fixed_point_value, atol=1e-6)


def test_anderson_depth_one_does_not_raise_on_repeated_calls():
    """Drive ``anderson_step`` past the first call with depth=1 — the
    delta_residuals matrix has 1 column, so the SVD condition check and
    the small linear solve must both handle that shape without raising."""
    state = AndersonState(memory_depth=1)
    rng = np.random.default_rng(0)
    x = rng.normal(size=4).astype(np.float64)
    for _ in range(5):
        t = 0.7 * x + 0.3  # contraction toward 1.0
        out = anderson_step(state, x_current=x, map_value=t)
        assert np.all(np.isfinite(out))
        x = out

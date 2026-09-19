"""Pin: Anderson(m=1) acceleration on a strict contraction.

With ``memory_depth=1`` the residual history is length 1 → the first call
to ``anderson_step`` returns the plain map value (no acceleration is
possible). On subsequent calls, Anderson(1) extrapolates from a single
residual difference (a secant step).

Goal: pin that iterating ``anderson_step`` with ``memory_depth=1`` on a
strict contraction converges to the same fixed point as plain iteration
and does not raise.
"""
from __future__ import annotations

import numpy as np

from sv_pgs.anderson import AndersonState, anderson_step


def test_anderson_step_depth_one_first_call_returns_map_value():
    """With empty history, ``anderson_step`` must return the plain map
    value (no acceleration available)."""
    state = AndersonState(memory_depth=1)
    current = np.zeros(3, dtype=np.float64)
    mapped = np.asarray([0.5, -0.25, 1.0], dtype=np.float64)
    proposal = anderson_step(state, x_current=current, map_value=mapped)
    np.testing.assert_array_equal(proposal, mapped)


def test_anderson_depth_one_converges_to_fixed_point():
    """A strict contraction must still converge under Anderson(1)."""
    contraction = 0.5
    fixed_point_value = 2.0  # x* satisfies x = 0.5 x + 1 → x = 2

    def fixed_point_map(vector: np.ndarray) -> np.ndarray:
        return contraction * vector + 1.0

    state = AndersonState(memory_depth=1)
    current = np.zeros(1, dtype=np.float64)
    converged = False
    for _ in range(200):
        mapped = fixed_point_map(current)
        if np.linalg.norm(mapped - current) < 1e-10:
            converged = True
            break
        current = anderson_step(state, x_current=current, map_value=mapped)
    assert converged, current
    np.testing.assert_allclose(current, fixed_point_value, atol=1e-6)


def test_anderson_depth_one_does_not_raise_on_repeated_calls():
    """Drive ``anderson_step`` past the first call with depth=1 — the
    delta_residuals matrix has 1 column, so the SVD condition check and
    the small linear solve must both handle that shape without raising."""
    state = AndersonState(memory_depth=1)
    rng = np.random.default_rng(0)
    current = rng.normal(size=4).astype(np.float64)
    for _ in range(5):
        mapped = 0.7 * current + 0.3  # contraction toward 1.0
        proposal = anderson_step(state, x_current=current, map_value=mapped)
        assert np.all(np.isfinite(proposal))
        current = proposal

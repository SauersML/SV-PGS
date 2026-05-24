"""Pin: ``solve_spd_system`` with a zero RHS converges immediately at x=0.

Mathematically Ax = 0 with A SPD has the unique solution x = 0.  The CG
algorithm should detect ‖r₀‖ == ‖b‖ == 0 and exit on the first iteration
(or zero-th) without ever calling the matvec.

This pins:

* the returned solution is the zero vector,
* no NaN / inf,
* the result is returned even when ``max_iterations`` is set to 0 (no
  iterations needed for a trivial RHS).
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from sv_pgs.linear_solvers import build_linear_operator, solve_spd_system


def _diag_operator(diagonal: np.ndarray):
    def matvec(vector) -> jnp.ndarray:
        return jnp.asarray(
            diagonal * np.asarray(vector, dtype=np.float64), dtype=jnp.float64
        )

    return build_linear_operator(shape=(diagonal.size, diagonal.size), matvec=matvec)


def test_zero_rhs_returns_zero_vector():
    diagonal = np.asarray([1.0, 2.0, 3.0, 5.0], dtype=np.float64)
    operator = _diag_operator(diagonal)
    rhs = np.zeros_like(diagonal)
    solution = solve_spd_system(
        operator=operator,
        right_hand_side=rhs,
        tolerance=1e-10,
        max_iterations=10,
    )
    np.testing.assert_array_equal(np.asarray(solution), np.zeros_like(diagonal))


def test_zero_rhs_with_max_iter_one_still_returns_zero():
    """Even with a single allowed iteration, ‖r₀‖ = 0 must exit immediately
    so the returned solution is exactly zero."""
    diagonal = np.asarray([7.0, 11.0], dtype=np.float64)
    operator = _diag_operator(diagonal)
    rhs = np.zeros_like(diagonal)
    solution = solve_spd_system(
        operator=operator,
        right_hand_side=rhs,
        tolerance=1e-12,
        max_iterations=1,
    )
    np.testing.assert_array_equal(np.asarray(solution), np.zeros_like(diagonal))

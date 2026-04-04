from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from sv_pgs.linear_solvers import build_linear_operator, solve_spd_system


def test_solve_spd_system_accepts_relative_convergence_for_float32_operator(random_generator):
    dimension = 64
    orthonormal_basis, _ = np.linalg.qr(random_generator.normal(size=(dimension, dimension)))
    eigenvalues = np.geomspace(1.0, 1e4, dimension)
    dense_operator = (orthonormal_basis * eigenvalues) @ orthonormal_basis.T
    dense_operator = ((dense_operator + dense_operator.T) * 0.5).astype(np.float64)
    float32_operator = dense_operator.astype(np.float32)

    def matvec(vector) -> jnp.ndarray:
        vector32 = np.asarray(vector, dtype=np.float32)
        return jnp.asarray(float32_operator @ vector32, dtype=jnp.float64)

    operator = build_linear_operator(shape=dense_operator.shape, matvec=matvec)
    expected_solution = random_generator.normal(size=dimension).astype(np.float64)
    right_hand_side = dense_operator @ expected_solution
    right_hand_side *= 1e2

    solution = solve_spd_system(
        operator=operator,
        right_hand_side=right_hand_side,
        tolerance=1e-6,
        max_iterations=256,
    )

    residual_norm = np.linalg.norm(dense_operator @ solution - right_hand_side)
    rhs_norm = np.linalg.norm(right_hand_side)
    assert residual_norm <= 1e-6 * rhs_norm

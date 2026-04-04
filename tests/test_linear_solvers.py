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


def test_solve_spd_system_uses_preconditioner_as_initial_guess():
    diagonal = np.array([2.0, 5.0, 11.0], dtype=np.float64)
    operator = build_linear_operator(
        shape=(3, 3),
        matvec=lambda vector: jnp.asarray(diagonal * np.asarray(vector, dtype=np.float64), dtype=jnp.float64),
    )
    expected_solution = np.array([1.5, -2.0, 0.25], dtype=np.float64)
    right_hand_side = diagonal * expected_solution
    inverse_diagonal = 1.0 / diagonal

    solution = solve_spd_system(
        operator=operator,
        right_hand_side=right_hand_side,
        tolerance=1e-12,
        max_iterations=1,
        preconditioner=lambda vector: jnp.asarray(inverse_diagonal * np.asarray(vector, dtype=np.float64), dtype=jnp.float64),
    )

    np.testing.assert_allclose(solution, expected_solution, atol=1e-12)


def test_solve_spd_system_rejects_vector_initial_guess_for_matrix_rhs():
    operator = build_linear_operator(
        shape=(2, 2),
        matvec=lambda vector: jnp.asarray(np.asarray(vector, dtype=np.float64), dtype=jnp.float64),
    )
    right_hand_side = np.eye(2, dtype=np.float64)

    with np.testing.assert_raises_regex(
        ValueError,
        "matrix right_hand_side requires initial_guess with matching column dimension",
    ):
        solve_spd_system(
            operator=operator,
            right_hand_side=right_hand_side,
            tolerance=1e-12,
            max_iterations=1,
            initial_guess=np.ones(2, dtype=np.float64),
        )

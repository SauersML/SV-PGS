from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import sv_pgs.linear_solvers as linear_solvers
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


def test_solve_spd_system_batched_matrix_rhs_matches_dense_solution(random_generator):
    dimension = 12
    rhs_count = 4
    basis, _ = np.linalg.qr(random_generator.normal(size=(dimension, dimension)))
    eigenvalues = np.geomspace(1.0, 30.0, dimension)
    dense_operator = (basis * eigenvalues) @ basis.T
    dense_operator = ((dense_operator + dense_operator.T) * 0.5).astype(np.float64)
    rhs = random_generator.normal(size=(dimension, rhs_count)).astype(np.float64)
    diagonal_preconditioner = np.diag(dense_operator).astype(np.float64)
    operator = build_linear_operator(
        shape=dense_operator.shape,
        matvec=lambda vector: jnp.asarray(dense_operator @ np.asarray(vector, dtype=np.float64), dtype=jnp.float64),
        matmat=lambda matrix: jnp.asarray(dense_operator @ np.asarray(matrix, dtype=np.float64), dtype=jnp.float64),
    )

    solution = solve_spd_system(
        operator=operator,
        right_hand_side=rhs,
        tolerance=1e-8,
        max_iterations=128,
        preconditioner=diagonal_preconditioner,
    )

    np.testing.assert_allclose(solution, np.linalg.solve(dense_operator, rhs), rtol=1e-7, atol=1e-7)


def test_solve_spd_system_batched_matrix_rhs_accepts_vector_only_callable_preconditioner():
    diagonal = np.array([2.0, 5.0, 11.0], dtype=np.float64)
    rhs = np.column_stack(
        [
            np.array([3.0, -1.0, 2.0], dtype=np.float64),
            np.array([-2.0, 4.0, 1.0], dtype=np.float64),
        ]
    )
    operator = build_linear_operator(
        shape=(3, 3),
        matvec=lambda vector: jnp.asarray(diagonal * np.asarray(vector, dtype=np.float64), dtype=jnp.float64),
        matmat=lambda matrix: jnp.asarray(diagonal[:, None] * np.asarray(matrix, dtype=np.float64), dtype=jnp.float64),
    )
    inverse_diagonal = 1.0 / diagonal

    solution = solve_spd_system(
        operator=operator,
        right_hand_side=rhs,
        tolerance=1e-12,
        max_iterations=4,
        preconditioner=lambda vector: jnp.asarray(inverse_diagonal * np.asarray(vector, dtype=np.float64), dtype=jnp.float64),
    )

    np.testing.assert_allclose(solution, inverse_diagonal[:, None] * rhs, atol=1e-12)


def test_stochastic_logdet_uses_numpy_eigh(monkeypatch):
    monkeypatch.setattr(
        linear_solvers.jnp.linalg,
        "eigh",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected JAX eigh")),
    )
    operator = build_linear_operator(
        shape=(2, 2),
        matvec=lambda vector: jnp.asarray(
            np.array([[3.0, 0.5], [0.5, 2.0]], dtype=np.float64) @ np.asarray(vector, dtype=np.float64),
            dtype=jnp.float64,
        ),
    )

    estimate = linear_solvers.stochastic_logdet(
        operator=operator,
        dimension=2,
        probe_count=2,
        lanczos_steps=2,
        random_seed=0,
    )

    assert np.isfinite(estimate)


def test_small_symmetric_eigh_rejects_non_finite_input():
    with np.testing.assert_raises_regex(RuntimeError, "non-finite values"):
        linear_solvers._small_symmetric_eigh(np.array([[1.0, np.nan], [np.nan, 2.0]], dtype=np.float64))


def test_small_symmetric_eigh_retries_with_jitter(monkeypatch):
    calls = {"count": 0}
    original_eigh = np.linalg.eigh

    def flaky_eigh(matrix):
        calls["count"] += 1
        if calls["count"] == 1:
            raise np.linalg.LinAlgError("transient failure")
        return original_eigh(matrix)

    monkeypatch.setattr(linear_solvers.np.linalg, "eigh", flaky_eigh)
    eigenvalues, eigenvectors = linear_solvers._small_symmetric_eigh(np.array([[2.0, 1e-14], [0.0, 3.0]], dtype=np.float64))

    assert calls["count"] >= 2
    assert eigenvalues.shape == (2,)
    assert eigenvectors.shape == (2, 2)


def test_solve_spd_system_uses_jax_cg_for_jax_compatible_operator(monkeypatch):
    calls: list[dict[str, object]] = []

    def fake_cg(A, b, x0=None, *, tol=1e-5, atol=0.0, maxiter=None, M=None):
        calls.append(
            {
                "shape": tuple(np.asarray(b).shape),
                "tol": tol,
                "atol": atol,
                "maxiter": maxiter,
                "has_preconditioner": M is not None,
                "x0": None if x0 is None else np.asarray(x0, dtype=np.float64),
            }
        )
        matrix = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
        solution = np.linalg.solve(matrix, np.asarray(b, dtype=np.float64))
        return jnp.asarray(solution, dtype=jnp.float64), None

    monkeypatch.setattr(linear_solvers.jax_sparse.linalg, "cg", fake_cg)

    operator = build_linear_operator(
        shape=(2, 2),
        matvec=lambda vector: jnp.asarray(
            np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64) @ np.asarray(vector, dtype=np.float64),
            dtype=jnp.float64,
        ),
        dtype=jnp.float64,
        jax_compatible=True,
    )
    right_hand_side = np.array([1.0, 2.0], dtype=np.float64)
    diagonal_preconditioner = np.array([4.0, 3.0], dtype=np.float64)

    solution = solve_spd_system(
        operator=operator,
        right_hand_side=right_hand_side,
        tolerance=1e-8,
        max_iterations=32,
        preconditioner=diagonal_preconditioner,
    )

    np.testing.assert_allclose(
        solution,
        np.linalg.solve(np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64), right_hand_side),
    )
    assert len(calls) == 1
    assert calls[0]["has_preconditioner"] is True
    np.testing.assert_allclose(calls[0]["x0"], right_hand_side / diagonal_preconditioner)

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import sv_pgs._jax  # noqa: F401
import jax.numpy as jnp
import numpy as np


@dataclass(slots=True)
class LinearOperator:
    shape: tuple[int, int]
    matvec: Callable[[jnp.ndarray], jnp.ndarray]
    matmat: Callable[[jnp.ndarray], jnp.ndarray] | None = None


def build_linear_operator(
    shape: tuple[int, int],
    matvec: Callable[[jnp.ndarray], jnp.ndarray],
    matmat: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> LinearOperator:
    return LinearOperator(shape=shape, matvec=matvec, matmat=matmat)


def solve_spd_system(
    operator: LinearOperator | np.ndarray | jnp.ndarray,
    right_hand_side: np.ndarray | jnp.ndarray,
    tolerance: float,
    max_iterations: int,
    initial_guess: np.ndarray | jnp.ndarray | None = None,
) -> np.ndarray:
    linear_operator = _as_linear_operator(operator)
    rhs_array = jnp.asarray(right_hand_side, dtype=jnp.float64)
    if rhs_array.ndim == 1:
        solution = _solve_single_rhs(
            linear_operator=linear_operator,
            rhs=rhs_array,
            tolerance=tolerance,
            max_iterations=max_iterations,
            initial_guess=None if initial_guess is None else jnp.asarray(initial_guess, dtype=jnp.float64),
        )
        return np.asarray(solution, dtype=np.float64)

    solution_columns: list[np.ndarray] = []
    initial_matrix = None if initial_guess is None else jnp.asarray(initial_guess, dtype=jnp.float64)
    for column_index in range(rhs_array.shape[1]):
        solution_columns.append(
            np.asarray(
                _solve_single_rhs(
                    linear_operator=linear_operator,
                    rhs=rhs_array[:, column_index],
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                    initial_guess=None if initial_matrix is None else initial_matrix[:, column_index],
                ),
                dtype=np.float64,
            )
        )
    return np.column_stack(solution_columns).astype(np.float64, copy=False)


def stochastic_logdet(
    operator: LinearOperator | np.ndarray | jnp.ndarray,
    dimension: int,
    probe_count: int,
    lanczos_steps: int,
    random_seed: int,
) -> float:
    linear_operator = _as_linear_operator(operator)
    step_count = min(max(lanczos_steps, 2), dimension)
    random_generator = np.random.default_rng(random_seed)
    estimates: list[float] = []
    for _probe_index in range(probe_count):
        probe_vector = jnp.asarray(
            random_generator.choice((-1.0, 1.0), size=dimension).astype(np.float64),
            dtype=jnp.float64,
        )
        probe_vector /= jnp.maximum(jnp.linalg.norm(probe_vector), 1e-12)
        tridiagonal = _lanczos_tridiagonal(
            linear_operator=linear_operator,
            start_vector=probe_vector,
            step_count=step_count,
        )
        eigenvalues, eigenvectors = jnp.linalg.eigh(tridiagonal)
        clipped_eigenvalues = jnp.maximum(eigenvalues, 1e-12)
        estimates.append(float(jnp.sum((eigenvectors[0, :] ** 2) * jnp.log(clipped_eigenvalues))))
    return float(dimension * np.mean(estimates))


def _as_linear_operator(operator: LinearOperator | np.ndarray | jnp.ndarray) -> LinearOperator:
    if isinstance(operator, LinearOperator):
        return operator
    matrix = jnp.asarray(operator, dtype=jnp.float64)
    return LinearOperator(
        shape=tuple(matrix.shape),
        matvec=lambda vector: matrix @ vector,
        matmat=lambda block: matrix @ block,
    )


def _solve_single_rhs(
    linear_operator: LinearOperator,
    rhs: jnp.ndarray,
    tolerance: float,
    max_iterations: int,
    initial_guess: jnp.ndarray | None,
) -> jnp.ndarray:
    solution = jnp.zeros_like(rhs) if initial_guess is None else initial_guess
    residual = rhs - linear_operator.matvec(solution)
    search_direction = residual
    residual_dot = float(jnp.vdot(residual, residual))
    if residual_dot <= tolerance * tolerance:
        return jnp.asarray(solution, dtype=jnp.float64)
    for _iteration_index in range(max_iterations):
        operator_search_direction = linear_operator.matvec(search_direction)
        step_denominator = float(jnp.vdot(search_direction, operator_search_direction))
        if step_denominator <= 0.0:
            raise RuntimeError("Conjugate-gradient operator is not positive definite.")
        step_size = residual_dot / step_denominator
        solution = solution + step_size * search_direction
        residual = residual - step_size * operator_search_direction
        updated_residual_dot = float(jnp.vdot(residual, residual))
        if updated_residual_dot <= tolerance * tolerance:
            return jnp.asarray(solution, dtype=jnp.float64)
        beta = updated_residual_dot / residual_dot
        search_direction = residual + beta * search_direction
        residual_dot = updated_residual_dot
    raise RuntimeError("Conjugate-gradient solve failed to converge.")


def _lanczos_tridiagonal(
    linear_operator: LinearOperator,
    start_vector: jnp.ndarray,
    step_count: int,
) -> jnp.ndarray:
    normalized_start = jnp.asarray(start_vector, dtype=jnp.float64)
    normalized_start /= jnp.maximum(jnp.linalg.norm(normalized_start), 1e-12)
    basis_vectors: list[jnp.ndarray] = []
    alpha_values: list[float] = []
    beta_values: list[float] = []
    current_vector = normalized_start
    previous_vector = jnp.zeros_like(current_vector)
    previous_beta = 0.0
    for _step_index in range(step_count):
        projected = jnp.asarray(linear_operator.matvec(current_vector), dtype=jnp.float64)
        if basis_vectors:
            projected = projected - previous_beta * previous_vector
        alpha_value = float(jnp.dot(current_vector, projected))
        projected = projected - alpha_value * current_vector
        for basis_vector in basis_vectors:
            projected = projected - basis_vector * jnp.dot(basis_vector, projected)
        beta_value = float(jnp.linalg.norm(projected))
        basis_vectors.append(current_vector)
        alpha_values.append(alpha_value)
        if beta_value < 1e-10:
            break
        beta_values.append(beta_value)
        previous_vector = current_vector
        previous_beta = beta_value
        current_vector = projected / beta_value

    tridiagonal = jnp.diag(jnp.asarray(alpha_values, dtype=jnp.float64))
    if beta_values:
        off_diagonal = jnp.asarray(beta_values[: max(len(alpha_values) - 1, 0)], dtype=jnp.float64)
        tridiagonal = tridiagonal.at[jnp.arange(off_diagonal.shape[0]), jnp.arange(1, off_diagonal.shape[0] + 1)].set(off_diagonal)
        tridiagonal = tridiagonal.at[jnp.arange(1, off_diagonal.shape[0] + 1), jnp.arange(off_diagonal.shape[0])].set(off_diagonal)
    return tridiagonal

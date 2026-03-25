from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.sparse.linalg import LinearOperator, aslinearoperator, cg


def build_linear_operator(
    shape: tuple[int, int],
    matvec: Callable[[np.ndarray], np.ndarray],
    matmat: Callable[[np.ndarray], np.ndarray] | None = None,
    dtype: np.dtype | type = np.float64,
) -> LinearOperator:
    return LinearOperator(shape=shape, matvec=matvec, matmat=matmat, dtype=np.dtype(dtype))


def solve_spd_system(
    operator: LinearOperator | np.ndarray,
    right_hand_side: np.ndarray,
    tolerance: float,
    max_iterations: int,
    initial_guess: np.ndarray | None = None,
) -> np.ndarray:
    linear_operator = aslinearoperator(operator)
    rhs_array = np.asarray(right_hand_side, dtype=np.float64)
    if rhs_array.ndim == 1:
        return _solve_single_rhs(
            linear_operator=linear_operator,
            rhs=rhs_array,
            tolerance=tolerance,
            max_iterations=max_iterations,
            initial_guess=None if initial_guess is None else np.asarray(initial_guess, dtype=np.float64),
        )

    solution = np.zeros_like(rhs_array, dtype=np.float64)
    initial_matrix = None if initial_guess is None else np.asarray(initial_guess, dtype=np.float64)
    for column_index in range(rhs_array.shape[1]):
        solution[:, column_index] = _solve_single_rhs(
            linear_operator=linear_operator,
            rhs=rhs_array[:, column_index],
            tolerance=tolerance,
            max_iterations=max_iterations,
            initial_guess=None if initial_matrix is None else initial_matrix[:, column_index],
        )
    return solution


def stochastic_logdet(
    operator: LinearOperator | np.ndarray,
    dimension: int,
    probe_count: int,
    lanczos_steps: int,
    random_seed: int,
) -> float:
    linear_operator = aslinearoperator(operator)
    step_count = min(max(lanczos_steps, 2), dimension)
    random_generator = np.random.default_rng(random_seed)
    estimates: list[float] = []
    for _probe_index in range(probe_count):
        probe_vector = random_generator.choice((-1.0, 1.0), size=dimension).astype(np.float64)
        probe_vector /= np.linalg.norm(probe_vector)
        tridiagonal = _lanczos_tridiagonal(
            linear_operator=linear_operator,
            start_vector=probe_vector,
            step_count=step_count,
        )
        eigenvalues, eigenvectors = np.linalg.eigh(tridiagonal)
        clipped_eigenvalues = np.maximum(eigenvalues, 1e-12)
        estimates.append(float(np.sum((eigenvectors[0, :] ** 2) * np.log(clipped_eigenvalues))))
    return float(dimension * np.mean(estimates))


def _solve_single_rhs(
    linear_operator: LinearOperator,
    rhs: np.ndarray,
    tolerance: float,
    max_iterations: int,
    initial_guess: np.ndarray | None,
) -> np.ndarray:
    solution, info = cg(
        linear_operator,
        rhs,
        x0=initial_guess,
        rtol=tolerance,
        atol=0.0,
        maxiter=max_iterations,
    )
    if info != 0:
        raise RuntimeError(f"Conjugate-gradient solve failed to converge (info={info}).")
    return np.asarray(solution, dtype=np.float64)


def _lanczos_tridiagonal(
    linear_operator: LinearOperator,
    start_vector: np.ndarray,
    step_count: int,
) -> np.ndarray:
    normalized_start = np.asarray(start_vector, dtype=np.float64)
    normalized_start /= float(max(float(np.linalg.norm(normalized_start)), 1e-12))
    basis_vectors: list[np.ndarray] = []
    alpha_values: list[float] = []
    beta_values: list[float] = []
    current_vector = normalized_start
    previous_vector = np.zeros_like(current_vector)
    previous_beta = 0.0
    for _step_index in range(step_count):
        projected = np.asarray(linear_operator.matvec(current_vector), dtype=np.float64)
        if basis_vectors:
            projected -= previous_beta * previous_vector
        alpha_value = float(np.dot(current_vector, projected))
        projected -= alpha_value * current_vector
        for basis_vector in basis_vectors:
            projected -= basis_vector * np.dot(basis_vector, projected)
        beta_value = float(np.linalg.norm(projected))
        basis_vectors.append(current_vector.copy())
        alpha_values.append(alpha_value)
        if beta_value < 1e-10:
            break
        beta_values.append(beta_value)
        previous_vector = current_vector
        previous_beta = beta_value
        current_vector = projected / beta_value

    tridiagonal = np.diag(alpha_values).astype(np.float64)
    if beta_values:
        off_diagonal = np.asarray(beta_values[: max(len(alpha_values) - 1, 0)], dtype=np.float64)
        tridiagonal[np.arange(off_diagonal.shape[0]), np.arange(1, off_diagonal.shape[0] + 1)] = off_diagonal
        tridiagonal[np.arange(1, off_diagonal.shape[0] + 1), np.arange(off_diagonal.shape[0])] = off_diagonal
    return tridiagonal

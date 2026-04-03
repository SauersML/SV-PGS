"""Iterative linear algebra solvers for large-scale Bayesian inference.

These solvers avoid building huge matrices (n_samples x n_samples) by
working with "operators" — objects that know how to multiply by a vector
but never store the full matrix.  This is essential when the sample count
is in the hundreds of thousands.

Two key algorithms:
  1. Conjugate Gradient (CG): iteratively solves Ax = b for symmetric
     positive-definite A.  Think of it as finding the bottom of a bowl
     by sliding downhill, with each step orthogonal to the previous one.

  2. Stochastic Lanczos Quadrature: estimates log(det(A)) — a measure of
     the "volume" of the covariance matrix — without computing eigenvalues.
     Uses random probe vectors and a tridiagonal approximation.  This is
     needed for the restricted log-likelihood (model quality score).
"""
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


# Solve the linear system A @ x = b where A is symmetric positive-definite.
# Uses conjugate gradient (CG) — an iterative method that converges in at
# most n steps for an n x n matrix, but usually much fewer with a good
# preconditioner or well-conditioned system.  Handles both single vectors
# and matrices (solving column by column).
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


# Estimate log(determinant(A)) without computing the full eigendecomposition.
#
# Why we need this: the restricted log-likelihood includes a log-det term
# that measures model complexity.  For a 447k x 447k matrix, computing
# eigenvalues directly is infeasible.
#
# How it works: shoot random probe vectors through the matrix, build a
# small tridiagonal approximation (Lanczos), compute eigenvalues of that
# small matrix, and average.  Each probe gives a noisy estimate of
# log(det); averaging over several probes reduces variance.
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


# Conjugate Gradient (CG) for a single right-hand side vector.
#
# Starting from an initial guess (or zero), iteratively refine the
# solution by:
#   1. Compute the residual r = b - A @ x (how far off we are)
#   2. Pick a search direction (conjugate to all previous directions)
#   3. Take an optimal step along that direction
#   4. Repeat until ||residual|| < tolerance
#
# Each iteration requires one matrix-vector product (A @ direction).
# Convergence is guaranteed for positive-definite A.
def _solve_single_rhs(
    linear_operator: LinearOperator,
    rhs: jnp.ndarray,
    tolerance: float,
    max_iterations: int,
    initial_guess: jnp.ndarray | None,
) -> jnp.ndarray:
    tol_sq = tolerance * tolerance
    solution = jnp.zeros_like(rhs) if initial_guess is None else initial_guess
    residual = rhs - linear_operator.matvec(solution)
    search_direction = residual
    # Keep residual_dot as a JAX scalar to avoid GPU sync every iteration.
    # Only call float() when we actually need to check convergence.
    residual_dot_jax = jnp.vdot(residual, residual)
    if float(residual_dot_jax) <= tol_sq:
        return jnp.asarray(solution, dtype=jnp.float64)
    for _iteration_index in range(max_iterations):
        operator_search_direction = linear_operator.matvec(search_direction)
        step_denom_jax = jnp.vdot(search_direction, operator_search_direction)
        # Compute step_size as JAX scalar — no sync
        step_size = residual_dot_jax / step_denom_jax
        solution = solution + step_size * search_direction
        residual = residual - step_size * operator_search_direction
        updated_residual_dot_jax = jnp.vdot(residual, residual)
        # Only sync to check convergence (every iteration, but just one float())
        updated_residual_dot = float(updated_residual_dot_jax)
        if updated_residual_dot <= tol_sq:
            return jnp.asarray(solution, dtype=jnp.float64)
        if float(step_denom_jax) <= 0.0:
            raise RuntimeError("Conjugate-gradient operator is not positive definite.")
        beta = updated_residual_dot_jax / residual_dot_jax
        search_direction = residual + beta * search_direction
        residual_dot_jax = updated_residual_dot_jax
    raise RuntimeError("Conjugate-gradient solve failed to converge.")


# Build a small tridiagonal matrix that approximates the large operator A.
#
# Lanczos algorithm: starting from a random vector, repeatedly multiply
# by A and orthogonalize.  After k steps we have a k x k tridiagonal
# matrix T whose eigenvalues approximate those of A — like reading the
# "cliff notes" version of a huge matrix.  The eigenvalues of T are used
# to estimate log(det(A)) in the stochastic_logdet function above.
#
# The reorthogonalization loop (inner for-loop) prevents numerical
# instability where basis vectors slowly lose orthogonality.
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
        if basis_vectors:
            basis_matrix = jnp.stack(basis_vectors)
            coeffs = basis_matrix @ projected
            projected = projected - coeffs @ basis_matrix
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

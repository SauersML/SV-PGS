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
    dtype: jnp.dtype | None = None


def build_linear_operator(
    shape: tuple[int, int],
    matvec: Callable[[jnp.ndarray], jnp.ndarray],
    matmat: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    dtype: jnp.dtype | None = None,
) -> LinearOperator:
    return LinearOperator(shape=shape, matvec=matvec, matmat=matmat, dtype=dtype)


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
    preconditioner: Callable[[jnp.ndarray], jnp.ndarray] | np.ndarray | jnp.ndarray | None = None,
) -> np.ndarray:
    linear_operator = _as_linear_operator(operator)
    rhs_array = jnp.asarray(right_hand_side)
    operator_dtype = linear_operator.dtype or rhs_array.dtype
    solver_dtype = jnp.result_type(operator_dtype, rhs_array.dtype)
    rhs_array = rhs_array.astype(solver_dtype)
    apply_preconditioner = _as_preconditioner(preconditioner, solver_dtype)
    output_dtype = np.asarray(jnp.zeros((), dtype=solver_dtype)).dtype
    if rhs_array.ndim == 1:
        solution = _solve_single_rhs(
            linear_operator=linear_operator,
            rhs=rhs_array,
            solver_dtype=solver_dtype,
            tolerance=tolerance,
            max_iterations=max_iterations,
            initial_guess=None if initial_guess is None else jnp.asarray(initial_guess, dtype=solver_dtype),
            preconditioner=apply_preconditioner,
        )
        return np.asarray(solution, dtype=output_dtype)

    from sv_pgs.progress import log
    n_cols = rhs_array.shape[1]
    solution_columns: list[np.ndarray] = []
    initial_matrix = None if initial_guess is None else jnp.asarray(initial_guess, dtype=solver_dtype)
    for column_index in range(n_cols):
        if n_cols > 1:
            log(f"    CG solve: column {column_index+1}/{n_cols}")
        solution_columns.append(
            np.asarray(
                _solve_single_rhs(
                    linear_operator=linear_operator,
                    rhs=rhs_array[:, column_index],
                    solver_dtype=solver_dtype,
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                    initial_guess=None if initial_matrix is None else initial_matrix[:, column_index],
                    preconditioner=apply_preconditioner,
                ),
                dtype=output_dtype,
            )
        )
    return np.column_stack(solution_columns).astype(output_dtype, copy=False)


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
    operator_dtype = linear_operator.dtype or jnp.float32
    step_count = min(max(lanczos_steps, 2), dimension)
    random_generator = np.random.default_rng(random_seed)
    from sv_pgs.progress import log
    estimates: list[float] = []
    for _probe_index in range(probe_count):
        log(f"      logdet probe {_probe_index+1}/{probe_count} ({step_count} Lanczos steps)")
        probe_vector = jnp.asarray(
            random_generator.choice((-1.0, 1.0), size=dimension).astype(np.asarray(jnp.zeros((), dtype=operator_dtype)).dtype),
            dtype=operator_dtype,
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
    matrix = jnp.asarray(operator)
    matrix_dtype = jnp.result_type(matrix.dtype, jnp.float32)
    matrix = matrix.astype(matrix_dtype)
    return LinearOperator(
        shape=tuple(matrix.shape),
        matvec=lambda vector: matrix @ vector,
        matmat=lambda block: matrix @ block,
        dtype=matrix_dtype,
    )


def _as_preconditioner(
    preconditioner: Callable[[jnp.ndarray], jnp.ndarray] | np.ndarray | jnp.ndarray | None,
    solver_dtype: jnp.dtype,
) -> Callable[[jnp.ndarray], jnp.ndarray] | None:
    if preconditioner is None:
        return None
    if callable(preconditioner):
        return lambda vector: jnp.asarray(preconditioner(vector), dtype=solver_dtype)
    diagonal = jnp.asarray(preconditioner, dtype=solver_dtype)
    if diagonal.ndim != 1:
        raise ValueError("array preconditioner must be a diagonal vector.")
    return lambda vector: vector / jnp.maximum(diagonal, 1e-12)


# Conjugate Gradient (CG) for a single right-hand side vector.
#
# Starting from an initial guess (or zero), iteratively refine the
# solution by:
#   1. Compute the residual r = b - A @ x (how far off we are)
#   2. Pick a search direction (conjugate to all previous directions)
#   3. Take an optimal step along that direction
#   4. Repeat until the residual is below a mixed absolute/relative tolerance
#
# Each iteration requires one matrix-vector product (A @ direction).
# Convergence is guaranteed for positive-definite A.
def _solve_single_rhs(
    linear_operator: LinearOperator,
    rhs: jnp.ndarray,
    solver_dtype: jnp.dtype,
    tolerance: float,
    max_iterations: int,
    initial_guess: jnp.ndarray | None,
    preconditioner: Callable[[jnp.ndarray], jnp.ndarray] | None,
) -> jnp.ndarray:
    import time
    import math
    from sv_pgs.progress import log
    residual_refresh_interval = 32
    tol_sq = tolerance * tolerance
    if initial_guess is None:
        if preconditioner is None:
            solution = jnp.zeros_like(rhs)
        else:
            # Jacobi-like preconditioners are often a good first approximation.
            solution = jnp.asarray(preconditioner(rhs), dtype=solver_dtype)
    else:
        solution = initial_guess
    residual = rhs - jnp.asarray(linear_operator.matvec(solution), dtype=solver_dtype)
    residual_norm_sq_jax = jnp.vdot(residual, residual)
    initial_residual = float(residual_norm_sq_jax)
    rhs_norm_sq = float(jnp.vdot(rhs, rhs))
    convergence_threshold_sq = max(
        tol_sq,
        tol_sq * max(initial_residual, rhs_norm_sq),
    )
    if initial_residual <= convergence_threshold_sq:
        return jnp.asarray(solution, dtype=solver_dtype)
    preconditioned_residual = residual if preconditioner is None else preconditioner(residual)
    search_direction = preconditioned_residual
    residual_dot_jax = jnp.vdot(residual, preconditioned_residual)
    best_residual_dot = float(residual_norm_sq_jax)
    # Progress is measured in log-space: how far the residual has dropped
    # from initial toward the tolerance.  E.g. if initial=1e0, tol=1e-6,
    # and current=1e-3, we're 50% of the way (3 out of 6 orders of magnitude).
    log_initial = math.log10(max(initial_residual, 1e-30))
    log_target = math.log10(max(convergence_threshold_sq, 1e-30))
    log_range = max(log_initial - log_target, 1e-6)
    t_start = time.monotonic()
    last_log = t_start
    for _iteration_index in range(max_iterations):
        operator_search_direction = jnp.asarray(linear_operator.matvec(search_direction), dtype=solver_dtype)
        step_denom_jax = jnp.vdot(search_direction, operator_search_direction)
        step_denom = float(step_denom_jax)
        if not np.isfinite(step_denom) or step_denom <= 0.0:
            residual = rhs - jnp.asarray(linear_operator.matvec(solution), dtype=solver_dtype)
            preconditioned_residual = residual if preconditioner is None else preconditioner(residual)
            search_direction = preconditioned_residual
            residual_dot_jax = jnp.vdot(residual, preconditioned_residual)
            operator_search_direction = jnp.asarray(linear_operator.matvec(search_direction), dtype=solver_dtype)
            step_denom_jax = jnp.vdot(search_direction, operator_search_direction)
            step_denom = float(step_denom_jax)
            if not np.isfinite(step_denom) or step_denom <= 0.0:
                raise RuntimeError("Conjugate-gradient operator is not positive definite.")
        step_size = residual_dot_jax / step_denom_jax
        solution = solution + step_size * search_direction
        residual = residual - step_size * operator_search_direction
        if (_iteration_index + 1) % residual_refresh_interval == 0:
            residual = rhs - jnp.asarray(linear_operator.matvec(solution), dtype=solver_dtype)
        updated_residual_norm_sq_jax = jnp.vdot(residual, residual)
        updated_residual_dot = float(updated_residual_norm_sq_jax)
        best_residual_dot = min(best_residual_dot, updated_residual_dot)
        now = time.monotonic()
        if now - last_log >= 5.0 or updated_residual_dot <= convergence_threshold_sq:
            log_current = math.log10(max(updated_residual_dot, 1e-30))
            pct = min(100.0, max(0.0, 100.0 * (log_initial - log_current) / log_range))
            log(f"      CG iter {_iteration_index+1}/{max_iterations}: {pct:.0f}% converged  residual={updated_residual_dot:.2e}  ({now - t_start:.1f}s)")
            last_log = now
        if updated_residual_dot <= convergence_threshold_sq:
            return jnp.asarray(solution, dtype=solver_dtype)
        updated_preconditioned_residual = residual if preconditioner is None else preconditioner(residual)
        updated_residual_dot_jax = jnp.vdot(residual, updated_preconditioned_residual)
        beta = updated_residual_dot_jax / residual_dot_jax
        beta_value = float(beta)
        if not np.isfinite(beta_value) or beta_value < 0.0:
            residual = rhs - jnp.asarray(linear_operator.matvec(solution), dtype=solver_dtype)
            search_direction = updated_preconditioned_residual
            updated_preconditioned_residual = residual if preconditioner is None else preconditioner(residual)
            updated_residual_dot_jax = jnp.vdot(residual, updated_preconditioned_residual)
            updated_residual_dot = float(jnp.vdot(residual, residual))
            best_residual_dot = min(best_residual_dot, updated_residual_dot)
        else:
            search_direction = updated_preconditioned_residual + beta * search_direction
        residual_dot_jax = updated_residual_dot_jax
    final_residual = float(jnp.vdot(residual, residual))
    raise RuntimeError(
        "Conjugate-gradient solve failed to converge: "
        + f"residual={final_residual:.2e} threshold={convergence_threshold_sq:.2e} "
        + f"iterations={max_iterations}"
    )


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
    operator_dtype = linear_operator.dtype or jnp.result_type(start_vector.dtype, jnp.float32)
    normalized_start = jnp.asarray(start_vector, dtype=operator_dtype)
    normalized_start /= jnp.maximum(jnp.linalg.norm(normalized_start), 1e-12)
    basis_vectors: list[jnp.ndarray] = []
    alpha_values: list[float] = []
    beta_values: list[float] = []
    current_vector = normalized_start
    previous_vector = jnp.zeros_like(current_vector)
    previous_beta = 0.0
    for _step_index in range(step_count):
        projected = jnp.asarray(linear_operator.matvec(current_vector), dtype=operator_dtype)
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

    tridiagonal = jnp.diag(jnp.asarray(alpha_values, dtype=operator_dtype))
    if beta_values:
        off_diagonal = jnp.asarray(beta_values[: max(len(alpha_values) - 1, 0)], dtype=operator_dtype)
        tridiagonal = tridiagonal.at[jnp.arange(off_diagonal.shape[0]), jnp.arange(1, off_diagonal.shape[0] + 1)].set(off_diagonal)
        tridiagonal = tridiagonal.at[jnp.arange(1, off_diagonal.shape[0] + 1), jnp.arange(off_diagonal.shape[0])].set(off_diagonal)
    return tridiagonal

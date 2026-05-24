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

import importlib
import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import numpy as np

from sv_pgs._typing import F64Array, JaxArray, NDArray

# sv_pgs._jax MUST be imported before any direct import of jax so that the
# XLA env-var setup runs first. We deliberately import jax via importlib so
# the ordering survives static-analysis sorting passes.
importlib.import_module("sv_pgs._jax")
jnp = importlib.import_module("jax.numpy")
jax_sparse = importlib.import_module("jax.scipy.sparse")

JaxDType = Any


class LinearSolveStatus(Enum):
    CONVERGED = "converged"
    MAX_ITER = "max_iter"
    TIME_BUDGET = "time_budget"
    NUMERICAL_FAILURE = "numerical_failure"


@dataclass(slots=True)
class CGProgress:
    iteration: int
    max_iterations: int
    elapsed_seconds: float
    residual_norm: float
    matvec_count: int
    active_rhs: int | None = None
    status: LinearSolveStatus | None = None


@dataclass(slots=True)
class LinearSolveResult:
    solution: NDArray
    status: LinearSolveStatus
    iterations: int
    residual_norm: float
    matvec_count: int
    elapsed_seconds: float


CGProgressCallback = Callable[[CGProgress], None]


@dataclass(slots=True)
class LinearOperator:
    shape: tuple[int, int]
    matvec: Callable[[JaxArray], JaxArray]
    matmat: Callable[[JaxArray], JaxArray] | None = None
    dtype: JaxDType | None = None
    jax_compatible: bool = False
    gpu_resident: bool | None = None
    streaming_warning_emitted: bool = False


def build_linear_operator(
    shape: tuple[int, int],
    matvec: Callable[[JaxArray], JaxArray],
    matmat: Callable[[JaxArray], JaxArray] | None = None,
    dtype: JaxDType | None = None,
    jax_compatible: bool = False,
    gpu_resident: bool | None = None,
) -> LinearOperator:
    return LinearOperator(
        shape=shape,
        matvec=matvec,
        matmat=matmat,
        dtype=dtype,
        jax_compatible=jax_compatible,
        gpu_resident=gpu_resident,
    )


def _default_cg_progress_callback(progress: CGProgress) -> None:
    from sv_pgs.progress import log

    status_suffix = "" if progress.status is None else f" status={progress.status.name}"
    active_suffix = "" if progress.active_rhs is None else f" active={progress.active_rhs}"
    log(
        f"      CG iter {progress.iteration}/{progress.max_iterations}: "
        + f"elapsed={progress.elapsed_seconds:.1f}s residual_norm={progress.residual_norm:.2e} "
        + f"matvecs={progress.matvec_count}{active_suffix}{status_suffix}"
    )


def _emit_cg_progress(
    *,
    callback: CGProgressCallback | None,
    iteration: int,
    max_iterations: int,
    started_at: float,
    residual_norm: float,
    matvec_count: int,
    active_rhs: int | None = None,
    status: LinearSolveStatus | None = None,
) -> None:
    resolved_callback = _default_cg_progress_callback if callback is None else callback
    resolved_callback(
        CGProgress(
            iteration=iteration,
            max_iterations=max_iterations,
            elapsed_seconds=time.monotonic() - started_at,
            residual_norm=residual_norm,
            matvec_count=matvec_count,
            active_rhs=active_rhs,
            status=status,
        )
    )


def _maybe_emit_cg_progress(
    *,
    callback: CGProgressCallback | None,
    progress_interval: int,
    iteration: int,
    max_iterations: int,
    started_at: float,
    residual_norm: float,
    matvec_count: int,
    active_rhs: int | None = None,
    status: LinearSolveStatus | None = None,
) -> None:
    if progress_interval <= 0:
        return
    if iteration % progress_interval != 0 and status is None:
        return
    _emit_cg_progress(
        callback=callback,
        iteration=iteration,
        max_iterations=max_iterations,
        started_at=started_at,
        residual_norm=residual_norm,
        matvec_count=matvec_count,
        active_rhs=active_rhs,
        status=status,
    )


def _linear_solve_deadline(time_budget_seconds: float | None) -> float | None:
    if time_budget_seconds is None:
        return None
    resolved_budget = float(time_budget_seconds)
    if not np.isfinite(resolved_budget) or resolved_budget <= 0.0:
        raise ValueError("time_budget_seconds must be positive when provided.")
    return time.monotonic() + resolved_budget


def _time_budget_exceeded(deadline: float | None) -> bool:
    return deadline is not None and time.monotonic() >= deadline


def _jax_backend_name() -> str:
    try:
        jax = importlib.import_module("jax")
        return str(jax.default_backend())
    except RuntimeError as error:
        return f"unavailable:{error}"


def _closure_values(function: Callable[..., Any]) -> tuple[Any, ...]:
    closure = getattr(function, "__closure__", None)
    if closure is None:
        return ()
    values: list[Any] = []
    for cell in closure:
        try:
            values.append(cell.cell_contents)
        except ValueError:
            continue
    return tuple(values)


def _infer_gpu_resident(linear_operator: LinearOperator) -> bool | None:
    if linear_operator.gpu_resident is not None:
        return bool(linear_operator.gpu_resident)
    if linear_operator.jax_compatible:
        return True
    for function in (linear_operator.matvec, linear_operator.matmat):
        if function is None:
            continue
        for value in _closure_values(function):
            if isinstance(value, LinearOperator):
                nested = _infer_gpu_resident(value)
                if nested is not None:
                    return nested
            if hasattr(value, "_cupy_cache"):
                return getattr(value, "_cupy_cache") is not None
    return None


def _warn_if_streaming_operator(linear_operator: LinearOperator) -> None:
    if linear_operator.streaming_warning_emitted:
        return
    gpu_resident = _infer_gpu_resident(linear_operator)
    if gpu_resident is True:
        return
    linear_operator.streaming_warning_emitted = True
    from sv_pgs.progress import log

    log(
        "WARNING: CG matvec is using a non-GPU-resident linear operator "
        + f"shape={linear_operator.shape} backend={_jax_backend_name()} "
        + "source=streaming-or-host"
    )


def _return_linear_solve(
    result: LinearSolveResult,
    *,
    return_status: bool,
) -> NDArray | LinearSolveResult:
    if return_status:
        return result
    if result.status == LinearSolveStatus.CONVERGED:
        return result.solution
    if result.status == LinearSolveStatus.TIME_BUDGET:
        from sv_pgs.progress import log

        log(
            "      CG time budget exceeded; returning best-so-far "
            + f"residual_norm={result.residual_norm:.2e} matvecs={result.matvec_count}"
        )
        return result.solution
    if result.status == LinearSolveStatus.NUMERICAL_FAILURE:
        raise RuntimeError("Conjugate-gradient operator is not positive definite.")
    raise RuntimeError(
        "Conjugate-gradient solve failed to converge: "
        + f"residual={result.residual_norm:.2e} iterations={result.iterations}"
    )


def _combine_status(current: LinearSolveStatus, new: LinearSolveStatus) -> LinearSolveStatus:
    priority = {
        LinearSolveStatus.CONVERGED: 0,
        LinearSolveStatus.MAX_ITER: 1,
        LinearSolveStatus.TIME_BUDGET: 2,
        LinearSolveStatus.NUMERICAL_FAILURE: 3,
    }
    return new if priority[new] > priority[current] else current


# Solve the linear system A @ x = b where A is symmetric positive-definite.
# Uses conjugate gradient (CG) — an iterative method that converges in at
# most n steps for an n x n matrix, but usually much fewer with a good
# preconditioner or well-conditioned system.  Handles both single vectors
# and matrices (solving column by column).
def solve_spd_system(
    operator: LinearOperator | NDArray | JaxArray,
    right_hand_side: NDArray | JaxArray,
    tolerance: float,
    max_iterations: int,
    initial_guess: NDArray | JaxArray | None = None,
    preconditioner: Callable[[JaxArray], JaxArray] | NDArray | JaxArray | None = None,
    progress_callback: CGProgressCallback | None = None,
    progress_interval: int = 5,
    time_budget_seconds: float | None = None,
    return_status: bool = False,
) -> NDArray | LinearSolveResult:
    linear_operator = _as_linear_operator(operator)
    rhs_array = jnp.asarray(right_hand_side)
    operator_dtype = linear_operator.dtype or rhs_array.dtype
    solver_dtype = jnp.result_type(operator_dtype, rhs_array.dtype)
    rhs_array = rhs_array.astype(solver_dtype)
    deadline = _linear_solve_deadline(time_budget_seconds)
    if (
        deadline is None
        and progress_callback is None
        and linear_operator.jax_compatible
        and (preconditioner is None or not callable(preconditioner))
    ):
        return _return_linear_solve(
            _solve_spd_system_with_jax_cg(
                linear_operator=linear_operator,
                rhs=rhs_array,
                solver_dtype=solver_dtype,
                tolerance=tolerance,
                max_iterations=max_iterations,
                initial_guess=None if initial_guess is None else jnp.asarray(initial_guess, dtype=solver_dtype),
                preconditioner=preconditioner,
            ),
            return_status=return_status,
        )
    apply_preconditioner = _as_preconditioner(preconditioner, solver_dtype)
    output_dtype = np.asarray(jnp.zeros((), dtype=solver_dtype)).dtype
    if rhs_array.ndim == 1:
        result = _solve_single_rhs(
            linear_operator=linear_operator,
            rhs=rhs_array,
            solver_dtype=solver_dtype,
            tolerance=tolerance,
            max_iterations=max_iterations,
            initial_guess=None if initial_guess is None else jnp.asarray(initial_guess, dtype=solver_dtype),
            preconditioner=apply_preconditioner,
            progress_callback=progress_callback,
            progress_interval=progress_interval,
            deadline=deadline,
        )
        return _return_linear_solve(
            LinearSolveResult(
                solution=np.asarray(result.solution, dtype=output_dtype),
                status=result.status,
                iterations=result.iterations,
                residual_norm=result.residual_norm,
                matvec_count=result.matvec_count,
                elapsed_seconds=result.elapsed_seconds,
            ),
            return_status=return_status,
        )

    from sv_pgs.progress import log
    n_cols = rhs_array.shape[1]
    initial_matrix = None if initial_guess is None else jnp.asarray(initial_guess, dtype=solver_dtype)
    if initial_matrix is not None and initial_matrix.ndim != 2:
        raise ValueError("matrix right_hand_side requires initial_guess with matching column dimension.")
    if initial_matrix is not None and initial_matrix.shape != rhs_array.shape:
        raise ValueError("matrix initial_guess must have the same shape as right_hand_side.")
    if linear_operator.matmat is not None:
        solution_matrix = _solve_multiple_rhs(
            linear_operator=linear_operator,
            rhs=rhs_array,
            solver_dtype=solver_dtype,
            tolerance=tolerance,
            max_iterations=max_iterations,
            initial_guess=initial_matrix,
            preconditioner=apply_preconditioner,
            progress_callback=progress_callback,
            progress_interval=progress_interval,
            deadline=deadline,
        )
        return _return_linear_solve(
            LinearSolveResult(
                solution=np.asarray(solution_matrix.solution, dtype=output_dtype),
                status=solution_matrix.status,
                iterations=solution_matrix.iterations,
                residual_norm=solution_matrix.residual_norm,
                matvec_count=solution_matrix.matvec_count,
                elapsed_seconds=solution_matrix.elapsed_seconds,
            ),
            return_status=return_status,
        )

    solution_columns: list[NDArray] = []
    combined_status = LinearSolveStatus.CONVERGED
    combined_iterations = 0
    combined_matvecs = 0
    combined_residual_norm = 0.0
    column_solve_started_at = time.monotonic()
    for column_index in range(n_cols):
        if n_cols > 1:
            log(f"    CG solve: column {column_index+1}/{n_cols}")
        column_result = _solve_single_rhs(
            linear_operator=linear_operator,
            rhs=rhs_array[:, column_index],
            solver_dtype=solver_dtype,
            tolerance=tolerance,
            max_iterations=max_iterations,
            initial_guess=None if initial_matrix is None else initial_matrix[:, column_index],
            preconditioner=apply_preconditioner,
            progress_callback=progress_callback,
            progress_interval=progress_interval,
            deadline=deadline,
        )
        solution_columns.append(np.asarray(column_result.solution, dtype=output_dtype))
        combined_status = _combine_status(combined_status, column_result.status)
        combined_iterations += column_result.iterations
        combined_matvecs += column_result.matvec_count
        combined_residual_norm = max(combined_residual_norm, column_result.residual_norm)
        if column_result.status == LinearSolveStatus.TIME_BUDGET:
            for remaining_column in range(column_index + 1, n_cols):
                if initial_matrix is None:
                    solution_columns.append(np.zeros(rhs_array.shape[0], dtype=output_dtype))
                else:
                    solution_columns.append(np.asarray(initial_matrix[:, remaining_column], dtype=output_dtype))
            break
    combined_solution = np.column_stack(solution_columns).astype(output_dtype, copy=False)
    return _return_linear_solve(
        LinearSolveResult(
            solution=combined_solution,
            status=combined_status,
            iterations=combined_iterations,
            residual_norm=combined_residual_norm,
            matvec_count=combined_matvecs,
            elapsed_seconds=time.monotonic() - column_solve_started_at,
        ),
        return_status=return_status,
    )


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
    operator: LinearOperator | NDArray | JaxArray,
    dimension: int,
    probe_count: int,
    lanczos_steps: int,
    random_seed: int,
    minimum_probe_count: int = 4,
    relative_error_tolerance: float = 0.05,
    absolute_error_tolerance: float = 0.0,
    control_variate_diagonal: NDArray | JaxArray | None = None,
    treat_as_rank_deficient: bool = True,
) -> float:
    """Estimate log-determinant via stochastic Lanczos quadrature.

    The ``treat_as_rank_deficient`` flag controls how near-zero Ritz
    eigenvalues are handled and is the key knob for the two semantics
    this estimator can take on:

    * ``True`` (default, pseudo-determinant semantics): drops Ritz
      eigenvalues below ``max(matrix_norm * 1e-12, 1e-14)`` and
      renormalizes the remaining quadrature weights.  This is the right
      choice for SPSD/rank-deficient operators where exact-zero
      eigenvalues should not contribute ``log(roundoff)`` noise to the
      estimate (i.e. you actually want a pseudo-determinant over the
      column-space).

    * ``False`` (strict SPD semantics): includes every Ritz eigenvalue
      that is strictly positive without renormalizing the weights.
      This is the right choice when the operator is SPD by construction
      (e.g. ``prior_scale_floor + diagonal_noise``) and a genuinely tiny
      eigenvalue carries real ``log(eigenvalue)`` signal that must not
      be dropped.
    """
    linear_operator = _as_linear_operator(operator)
    baseline_logdet = 0.0
    if control_variate_diagonal is not None:
        base_operator = linear_operator
        diagonal = np.asarray(control_variate_diagonal, dtype=np.float64).reshape(-1)
        if diagonal.shape != (dimension,):
            raise ValueError("control_variate_diagonal must have one entry per operator dimension.")
        if np.any(~np.isfinite(diagonal)) or np.any(diagonal <= 0.0):
            raise ValueError("control_variate_diagonal must be finite and strictly positive.")
        inverse_sqrt_diagonal = np.asarray(1.0 / np.sqrt(diagonal), dtype=np.float64)
        baseline_logdet = float(np.sum(np.log(diagonal)))
        transformed_dtype = linear_operator.dtype or jnp.float64

        def transformed_matvec(vector: JaxArray) -> JaxArray:
            scaled_vector = inverse_sqrt_diagonal * np.asarray(vector, dtype=np.float64)
            applied = np.asarray(base_operator.matvec(jnp.asarray(scaled_vector, dtype=transformed_dtype)), dtype=np.float64)
            return jnp.asarray(inverse_sqrt_diagonal * applied, dtype=transformed_dtype)

        def transformed_matmat(matrix: JaxArray) -> JaxArray:
            matrix_array = np.asarray(matrix, dtype=np.float64)
            scaled_matrix = inverse_sqrt_diagonal[:, None] * matrix_array
            if base_operator.matmat is not None:
                applied = np.asarray(base_operator.matmat(jnp.asarray(scaled_matrix, dtype=transformed_dtype)), dtype=np.float64)
            else:
                applied = np.column_stack(
                    [
                        np.asarray(base_operator.matvec(jnp.asarray(scaled_matrix[:, column_index], dtype=transformed_dtype)), dtype=np.float64)
                        for column_index in range(scaled_matrix.shape[1])
                    ]
                )
            return jnp.asarray(inverse_sqrt_diagonal[:, None] * applied, dtype=transformed_dtype)

        linear_operator = build_linear_operator(
            shape=linear_operator.shape,
            matvec=transformed_matvec,
            matmat=transformed_matmat,
            dtype=transformed_dtype,
            jax_compatible=False,
        )
    operator_dtype = linear_operator.dtype or jnp.float32
    step_count = min(max(lanczos_steps, 2), dimension)
    random_generator = np.random.default_rng(random_seed)
    from sv_pgs.progress import log
    estimates: list[float] = []
    probe_dtype = np.asarray(jnp.zeros((), dtype=operator_dtype)).dtype
    probe_block_size = min(max(1, min(8, dimension)), probe_count)
    minimum_required_probes = min(max(int(minimum_probe_count), 1), max(int(probe_count), 1))
    resolved_relative_error_tolerance = max(float(relative_error_tolerance), 0.0)
    resolved_absolute_error_tolerance = max(float(absolute_error_tolerance), 0.0)
    probes_completed = 0
    while probes_completed < probe_count:
        current_block_size = min(probe_block_size, probe_count - probes_completed)
        block_start = probes_completed + 1
        block_stop = probes_completed + current_block_size
        log(f"      logdet probe block {block_start}-{block_stop}/{probe_count} ({step_count} Lanczos steps)")
        probe_block = _normalized_rademacher_probe_block(
            dimension=dimension,
            probe_count=current_block_size,
            random_generator=random_generator,
            dtype=probe_dtype,
        )
        tridiagonal_blocks = _lanczos_tridiagonal_block(
            linear_operator=linear_operator,
            start_matrix=jnp.asarray(probe_block, dtype=operator_dtype),
            step_count=step_count,
        )
        for tridiagonal in tridiagonal_blocks:
            eigenvalues, eigenvectors = _small_symmetric_eigh(tridiagonal)
            if treat_as_rank_deficient:
                # Pseudo-determinant semantics: drop eigenvalues below a
                # noise floor based on matrix norm.  Clipping numerical-noise
                # (near-zero / negative) eigenvalues to a tiny positive
                # constant would inflate log-det by ~27 nats each, so we
                # exclude them entirely and renormalize the weights over the
                # retained subspace.
                matrix_norm = float(np.linalg.norm(eigenvalues, ord=np.inf))
                noise_floor = max(matrix_norm * 1e-12, 1e-14)
                positive_mask = eigenvalues > noise_floor
                if not np.any(positive_mask):
                    continue
                filtered_eigenvalues = eigenvalues[positive_mask]
                filtered_first_components = eigenvectors[0, positive_mask]
                # Renormalize squared components to sum to 1 over the retained set.
                sum_sq = float(np.sum(filtered_first_components ** 2))
                if sum_sq <= 1e-12:
                    continue
                weights = (filtered_first_components ** 2) / sum_sq
                estimates.append(float(np.sum(weights * np.log(filtered_eigenvalues))))
            else:
                # Strict SPD semantics: include every strictly positive
                # eigenvalue without renormalizing.  Genuinely small
                # eigenvalues (e.g. 1e-13) contribute their real
                # log(eigenvalue) to the estimate.  We still skip exactly
                # zero / negative Ritz values (mathematically these cannot
                # occur for SPD operators; if they do appear they are pure
                # numerical noise and ``log`` would be undefined).
                positive_mask = eigenvalues > 0.0
                if not np.any(positive_mask):
                    continue
                filtered_eigenvalues = eigenvalues[positive_mask]
                filtered_first_components = eigenvectors[0, positive_mask]
                weights = filtered_first_components ** 2
                estimates.append(float(np.sum(weights * np.log(filtered_eigenvalues))))
        probes_completed += current_block_size
        if probes_completed < minimum_required_probes or len(estimates) < 2:
            continue
        block_estimate = float(dimension * np.mean(estimates))
        block_standard_error = float(
            dimension * np.std(np.asarray(estimates, dtype=np.float64), ddof=1) / np.sqrt(len(estimates))
        )
        error_threshold = max(
            resolved_absolute_error_tolerance,
            resolved_relative_error_tolerance * max(abs(baseline_logdet + block_estimate), 1.0),
        )
        if block_standard_error <= error_threshold:
            log(
                "      logdet adaptive stop: "
                + f"used {probes_completed}/{probe_count} probes  stderr={block_standard_error:.2e} "
                + f"threshold={error_threshold:.2e}"
            )
            break
    if not estimates:
        # No probe contributed (matrix appears effectively zero in the probed subspace).
        return float(baseline_logdet)
    return float(baseline_logdet + dimension * np.mean(estimates))


def _normalized_rademacher_probe_block(
    dimension: int,
    probe_count: int,
    random_generator: np.random.Generator,
    dtype: np.dtype[Any],
) -> NDArray:
    probe_block = random_generator.choice((-1.0, 1.0), size=(dimension, probe_count)).astype(dtype, copy=False)
    probe_norms = np.linalg.norm(probe_block, axis=0, keepdims=True)
    probe_norms = np.maximum(probe_norms, 1e-12)
    result: NDArray = probe_block / probe_norms
    return result


def _apply_operator_block(
    linear_operator: LinearOperator,
    block: JaxArray,
    operator_dtype: JaxDType,
) -> JaxArray:
    if block.ndim != 2:
        raise ValueError("operator block application expects a matrix input.")
    if block.shape[1] == 1:
        return jnp.asarray(linear_operator.matvec(block[:, 0]), dtype=operator_dtype)[:, None]
    if linear_operator.matmat is not None:
        return jnp.asarray(linear_operator.matmat(block), dtype=operator_dtype)
    return jnp.column_stack(
        [
            jnp.asarray(linear_operator.matvec(block[:, column_index]), dtype=operator_dtype)
            for column_index in range(block.shape[1])
        ]
    )


def _small_symmetric_eigh(matrix: NDArray | JaxArray) -> tuple[F64Array, F64Array]:
    # Tiny Lanczos tridiagonals should stay off GPU solver libraries entirely.
    matrix_np = np.asarray(matrix, dtype=np.float64)
    if matrix_np.ndim != 2 or matrix_np.shape[0] != matrix_np.shape[1]:
        raise ValueError("small symmetric eigendecomposition requires a square matrix.")
    if not np.all(np.isfinite(matrix_np)):
        raise RuntimeError("small symmetric eigendecomposition received non-finite values.")
    symmetric_matrix = 0.5 * (matrix_np + matrix_np.T)
    diagonal_scale = max(float(np.max(np.abs(np.diag(symmetric_matrix)))), 1.0)
    for jitter_scale in (0.0, 1e-12, 1e-10, 1e-8):
        try:
            if jitter_scale == 0.0:
                return np.linalg.eigh(symmetric_matrix)
            jitter = np.eye(symmetric_matrix.shape[0], dtype=np.float64) * (jitter_scale * diagonal_scale)
            return np.linalg.eigh(symmetric_matrix + jitter)
        except np.linalg.LinAlgError:
            continue
    raise RuntimeError("small symmetric eigendecomposition failed after CPU retries.")


def _as_linear_operator(operator: LinearOperator | NDArray | JaxArray) -> LinearOperator:
    if isinstance(operator, LinearOperator):
        return operator
    matrix = jnp.asarray(operator)
    matrix_dtype = jnp.result_type(matrix.dtype, jnp.float32)
    matrix = matrix.astype(matrix_dtype)
    return LinearOperator(
        shape=(int(matrix.shape[0]), int(matrix.shape[1])),
        matvec=lambda vector: matrix @ vector,
        matmat=lambda block: matrix @ block,
        dtype=matrix_dtype,
        jax_compatible=True,
    )


def _solve_spd_system_with_jax_cg(
    linear_operator: LinearOperator,
    rhs: JaxArray,
    solver_dtype: JaxDType,
    tolerance: float,
    max_iterations: int,
    initial_guess: JaxArray | None,
    preconditioner: NDArray | JaxArray | None,
) -> LinearSolveResult:
    started_at = time.monotonic()
    output_dtype = np.asarray(jnp.zeros((), dtype=solver_dtype)).dtype
    diagonal_preconditioner = None if preconditioner is None else jnp.asarray(preconditioner, dtype=solver_dtype)
    if diagonal_preconditioner is not None and diagonal_preconditioner.ndim != 1:
        raise ValueError("array preconditioner must be a diagonal vector.")
    safe_diagonal = None if diagonal_preconditioner is None else jnp.maximum(diagonal_preconditioner, 1e-12)

    def apply_operator(vector: JaxArray) -> JaxArray:
        return jnp.asarray(linear_operator.matvec(vector), dtype=solver_dtype)

    def apply_preconditioner(vector: JaxArray) -> JaxArray:
        if safe_diagonal is None:
            return vector
        return vector / safe_diagonal

    # Convention: `tolerance` is a RELATIVE tolerance applied to ||r|| / ||b||.
    # The absolute residual threshold is tolerance * max(||b||, 1.0); using
    # max(.., 1.0) avoids requiring subnormal residuals for tiny right-hand sides.
    rhs_norm_for_threshold = float(jnp.linalg.norm(rhs)) if rhs.ndim == 1 else 0.0
    absolute_threshold = float(tolerance) * max(rhs_norm_for_threshold, 1.0)

    def solve_one(rhs_vector: JaxArray, x0_vector: JaxArray | None) -> JaxArray:
        local_rhs_norm = float(jnp.linalg.norm(rhs_vector))
        local_threshold = float(tolerance) * max(local_rhs_norm, 1.0)
        solution, _ = jax_sparse.linalg.cg(
            apply_operator,
            rhs_vector,
            x0=x0_vector,
            tol=0.0,
            atol=local_threshold,
            maxiter=max_iterations,
            M=None if safe_diagonal is None else apply_preconditioner,
        )
        return jnp.asarray(solution, dtype=solver_dtype)

    if rhs.ndim == 1:
        initial_vector = (
            initial_guess
            if initial_guess is not None
            else (None if safe_diagonal is None else apply_preconditioner(rhs))
        )
        solution = solve_one(rhs, initial_vector)
        residual = rhs - apply_operator(solution)
        residual_norm = float(jnp.linalg.norm(residual))
        threshold = absolute_threshold
        status = LinearSolveStatus.CONVERGED
        if not np.isfinite(residual_norm):
            status = LinearSolveStatus.NUMERICAL_FAILURE
        elif residual_norm > threshold:
            status = LinearSolveStatus.MAX_ITER
        return LinearSolveResult(
            solution=np.asarray(solution, dtype=output_dtype),
            status=status,
            iterations=max_iterations if status != LinearSolveStatus.CONVERGED else 0,
            residual_norm=residual_norm,
            matvec_count=1,
            elapsed_seconds=time.monotonic() - started_at,
        )

    if rhs.ndim != 2:
        raise ValueError("right_hand_side must be a vector or matrix.")
    if initial_guess is not None and initial_guess.shape != rhs.shape:
        raise ValueError("matrix initial_guess must have the same shape as right_hand_side.")
    solutions: list[NDArray] = []
    combined_status = LinearSolveStatus.CONVERGED
    combined_residual_norm = 0.0
    combined_matvecs = 0
    for column_index in range(rhs.shape[1]):
        initial_vector = (
            None
            if initial_guess is None
            else initial_guess[:, column_index]
        )
        if initial_vector is None and safe_diagonal is not None:
            initial_vector = apply_preconditioner(rhs[:, column_index])
        column_result = _solve_spd_system_with_jax_cg(
            linear_operator=linear_operator,
            rhs=rhs[:, column_index],
            solver_dtype=solver_dtype,
            tolerance=tolerance,
            max_iterations=max_iterations,
            initial_guess=initial_vector,
            preconditioner=None if safe_diagonal is None else np.asarray(safe_diagonal),
        )
        solutions.append(column_result.solution)
        combined_status = _combine_status(combined_status, column_result.status)
        combined_residual_norm = max(combined_residual_norm, column_result.residual_norm)
        combined_matvecs += column_result.matvec_count
    return LinearSolveResult(
        solution=np.column_stack(solutions).astype(output_dtype, copy=False),
        status=combined_status,
        iterations=max_iterations if combined_status != LinearSolveStatus.CONVERGED else 0,
        residual_norm=combined_residual_norm,
        matvec_count=combined_matvecs,
        elapsed_seconds=time.monotonic() - started_at,
    )


def _as_preconditioner(
    preconditioner: Callable[[JaxArray], JaxArray] | NDArray | JaxArray | None,
    solver_dtype: JaxDType,
) -> Callable[[JaxArray], JaxArray] | None:
    if preconditioner is None:
        return None
    if callable(preconditioner):
        def apply_callable(vector: JaxArray) -> JaxArray:
            array = jnp.asarray(vector, dtype=solver_dtype)
            if array.ndim == 1:
                result = jnp.asarray(preconditioner(array), dtype=solver_dtype)
                if result.shape != array.shape:
                    raise ValueError("callable preconditioner must preserve vector shape.")
                return result
            if array.ndim != 2:
                raise ValueError("preconditioner input must be a vector or matrix.")
            try:
                result = jnp.asarray(preconditioner(array), dtype=solver_dtype)
            except (TypeError, ValueError):
                result = jnp.column_stack(
                    [
                        jnp.asarray(preconditioner(array[:, column_index]), dtype=solver_dtype)
                        for column_index in range(array.shape[1])
                    ]
                )
            if result.shape != array.shape:
                raise ValueError(
                    f"callable preconditioner returned shape {result.shape} for input shape {array.shape}."
                )
            return result

        return apply_callable
    diagonal = jnp.asarray(preconditioner, dtype=solver_dtype)
    if diagonal.ndim != 1:
        raise ValueError("array preconditioner must be a diagonal vector.")
    safe_diagonal = jnp.maximum(diagonal, 1e-12)

    def apply_diagonal(vector: JaxArray) -> JaxArray:
        array = jnp.asarray(vector, dtype=solver_dtype)
        if array.ndim == 1:
            return array / safe_diagonal
        if array.ndim == 2:
            return array / safe_diagonal[:, None]
        raise ValueError("preconditioner input must be a vector or matrix.")

    return apply_diagonal


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
    rhs: JaxArray,
    solver_dtype: JaxDType,
    tolerance: float,
    max_iterations: int,
    initial_guess: JaxArray | None,
    preconditioner: Callable[[JaxArray], JaxArray] | None,
    progress_callback: CGProgressCallback | None,
    progress_interval: int,
    deadline: float | None,
) -> LinearSolveResult:
    residual_refresh_interval = 32
    t_start = time.monotonic()
    matvec_count = 0

    def apply_operator(vector: JaxArray) -> JaxArray:
        nonlocal matvec_count
        _warn_if_streaming_operator(linear_operator)
        matvec_count += 1
        return jnp.asarray(linear_operator.matvec(vector), dtype=solver_dtype)

    # Convention: `tolerance` is RELATIVE: we require ||r|| <= tolerance * max(||b||, 1).
    # We work with squared quantities throughout for efficiency: the equivalent
    # squared check is ||r||^2 <= tolerance^2 * max(||b||^2, 1).
    tol_sq = tolerance * tolerance
    if initial_guess is None:
        if preconditioner is None:
            solution = jnp.zeros_like(rhs)
        else:
            # Jacobi-like preconditioners are often a good first approximation.
            solution = jnp.asarray(preconditioner(rhs), dtype=solver_dtype)
    else:
        solution = initial_guess
    best_solution = jnp.asarray(solution, dtype=solver_dtype)
    residual = rhs - apply_operator(solution)
    residual_norm_sq_jax = jnp.vdot(residual, residual)
    initial_residual = float(residual_norm_sq_jax)
    rhs_norm_sq = float(jnp.vdot(rhs, rhs))
    convergence_threshold_sq = tol_sq * max(rhs_norm_sq, 1.0)
    if initial_residual <= convergence_threshold_sq:
        return LinearSolveResult(
            solution=np.asarray(solution, dtype=np.asarray(jnp.zeros((), dtype=solver_dtype)).dtype),
            status=LinearSolveStatus.CONVERGED,
            iterations=0,
            residual_norm=math.sqrt(max(initial_residual, 0.0)),
            matvec_count=matvec_count,
            elapsed_seconds=time.monotonic() - t_start,
        )
    preconditioned_residual = residual if preconditioner is None else preconditioner(residual)
    search_direction = preconditioned_residual
    residual_dot_jax = jnp.vdot(residual, preconditioned_residual)
    best_residual_dot = float(residual_norm_sq_jax)
    for _iteration_index in range(max_iterations):
        iteration = _iteration_index + 1
        if _time_budget_exceeded(deadline):
            residual_norm = math.sqrt(max(best_residual_dot, 0.0))
            _maybe_emit_cg_progress(
                callback=progress_callback,
                progress_interval=progress_interval,
                iteration=_iteration_index,
                max_iterations=max_iterations,
                started_at=t_start,
                residual_norm=residual_norm,
                matvec_count=matvec_count,
                status=LinearSolveStatus.TIME_BUDGET,
            )
            return LinearSolveResult(
                solution=np.asarray(best_solution, dtype=np.asarray(jnp.zeros((), dtype=solver_dtype)).dtype),
                status=LinearSolveStatus.TIME_BUDGET,
                iterations=_iteration_index,
                residual_norm=residual_norm,
                matvec_count=matvec_count,
                elapsed_seconds=time.monotonic() - t_start,
            )
        operator_search_direction = apply_operator(search_direction)
        step_denom_jax = jnp.vdot(search_direction, operator_search_direction)
        step_denom = float(step_denom_jax)
        if not np.isfinite(step_denom) or step_denom <= 0.0:
            residual = rhs - apply_operator(solution)
            preconditioned_residual = residual if preconditioner is None else preconditioner(residual)
            search_direction = preconditioned_residual
            residual_dot_jax = jnp.vdot(residual, preconditioned_residual)
            operator_search_direction = apply_operator(search_direction)
            step_denom_jax = jnp.vdot(search_direction, operator_search_direction)
            step_denom = float(step_denom_jax)
            if not np.isfinite(step_denom) or step_denom <= 0.0:
                residual_norm = math.sqrt(max(best_residual_dot, 0.0))
                _maybe_emit_cg_progress(
                    callback=progress_callback,
                    progress_interval=progress_interval,
                    iteration=iteration,
                    max_iterations=max_iterations,
                    started_at=t_start,
                    residual_norm=residual_norm,
                    matvec_count=matvec_count,
                    status=LinearSolveStatus.NUMERICAL_FAILURE,
                )
                return LinearSolveResult(
                    solution=np.asarray(best_solution, dtype=np.asarray(jnp.zeros((), dtype=solver_dtype)).dtype),
                    status=LinearSolveStatus.NUMERICAL_FAILURE,
                    iterations=iteration,
                    residual_norm=residual_norm,
                    matvec_count=matvec_count,
                    elapsed_seconds=time.monotonic() - t_start,
                )
        step_size = residual_dot_jax / step_denom_jax
        solution = solution + step_size * search_direction
        residual = residual - step_size * operator_search_direction
        if (_iteration_index + 1) % residual_refresh_interval == 0:
            residual = rhs - apply_operator(solution)
        updated_residual_norm_sq_jax = jnp.vdot(residual, residual)
        updated_residual_dot = float(updated_residual_norm_sq_jax)
        if updated_residual_dot < best_residual_dot:
            best_residual_dot = updated_residual_dot
            best_solution = jnp.asarray(solution, dtype=solver_dtype)
        residual_norm = math.sqrt(max(updated_residual_dot, 0.0))
        _maybe_emit_cg_progress(
            callback=progress_callback,
            progress_interval=progress_interval,
            iteration=iteration,
            max_iterations=max_iterations,
            started_at=t_start,
            residual_norm=residual_norm,
            matvec_count=matvec_count,
            status=LinearSolveStatus.CONVERGED if updated_residual_dot <= convergence_threshold_sq else None,
        )
        if updated_residual_dot <= convergence_threshold_sq:
            return LinearSolveResult(
                solution=np.asarray(solution, dtype=np.asarray(jnp.zeros((), dtype=solver_dtype)).dtype),
                status=LinearSolveStatus.CONVERGED,
                iterations=iteration,
                residual_norm=residual_norm,
                matvec_count=matvec_count,
                elapsed_seconds=time.monotonic() - t_start,
            )
        updated_preconditioned_residual = residual if preconditioner is None else preconditioner(residual)
        updated_residual_dot_jax = jnp.vdot(residual, updated_preconditioned_residual)
        beta = updated_residual_dot_jax / residual_dot_jax
        beta_value = float(beta)
        if not np.isfinite(beta_value) or beta_value < 0.0:
            residual = rhs - apply_operator(solution)
            search_direction = updated_preconditioned_residual
            updated_preconditioned_residual = residual if preconditioner is None else preconditioner(residual)
            updated_residual_dot_jax = jnp.vdot(residual, updated_preconditioned_residual)
            updated_residual_dot = float(jnp.vdot(residual, residual))
            if updated_residual_dot < best_residual_dot:
                best_residual_dot = updated_residual_dot
                best_solution = jnp.asarray(solution, dtype=solver_dtype)
        else:
            search_direction = updated_preconditioned_residual + beta * search_direction
        residual_dot_jax = updated_residual_dot_jax
    final_residual = float(jnp.vdot(residual, residual))
    residual_norm = math.sqrt(max(min(final_residual, best_residual_dot), 0.0))
    _maybe_emit_cg_progress(
        callback=progress_callback,
        progress_interval=progress_interval,
        iteration=max_iterations,
        max_iterations=max_iterations,
        started_at=t_start,
        residual_norm=residual_norm,
        matvec_count=matvec_count,
        status=LinearSolveStatus.MAX_ITER,
    )
    return LinearSolveResult(
        solution=np.asarray(best_solution, dtype=np.asarray(jnp.zeros((), dtype=solver_dtype)).dtype),
        status=LinearSolveStatus.MAX_ITER,
        iterations=max_iterations,
        residual_norm=residual_norm,
        matvec_count=matvec_count,
        elapsed_seconds=time.monotonic() - t_start,
    )


def _solve_multiple_rhs(
    linear_operator: LinearOperator,
    rhs: JaxArray,
    solver_dtype: JaxDType,
    tolerance: float,
    max_iterations: int,
    initial_guess: JaxArray | None,
    preconditioner: Callable[[JaxArray], JaxArray] | None,
    progress_callback: CGProgressCallback | None,
    progress_interval: int,
    deadline: float | None,
) -> LinearSolveResult:
    residual_refresh_interval = 32
    t_start = time.monotonic()
    output_dtype = np.asarray(jnp.zeros((), dtype=solver_dtype)).dtype
    matvec_count = 0
    # Convention: `tolerance` is RELATIVE per column: require
    # ||r_i|| <= tolerance * max(||b_i||, 1), equivalently
    # ||r_i||^2 <= tolerance^2 * max(||b_i||^2, 1).  We work in squared space.
    tol_sq = tolerance * tolerance
    if initial_guess is None:
        if preconditioner is None:
            solution = jnp.zeros_like(rhs)
        else:
            solution = jnp.asarray(preconditioner(rhs), dtype=solver_dtype)
    else:
        solution = initial_guess
    operator_matmat = linear_operator.matmat
    if operator_matmat is None:
        raise ValueError("matrix solve requires an operator matmat implementation.")

    def apply_operator_active(matrix: JaxArray, active_columns: NDArray | None = None) -> JaxArray:
        nonlocal matvec_count
        _warn_if_streaming_operator(linear_operator)
        matvec_count += 1
        if active_columns is None:
            return jnp.asarray(operator_matmat(matrix), dtype=solver_dtype)
        if active_columns.size == 0:
            return jnp.zeros_like(matrix)
        active_index_array = jnp.asarray(active_columns, dtype=jnp.int32)
        active_matrix = matrix[:, active_index_array]
        active_result = jnp.asarray(operator_matmat(active_matrix), dtype=solver_dtype)
        return jnp.zeros_like(matrix).at[:, active_index_array].set(active_result)

    residual = rhs - apply_operator_active(solution)
    residual_norm_sq = np.asarray(jnp.sum(residual * residual, axis=0), dtype=np.float64)
    rhs_norm_sq = np.asarray(jnp.sum(rhs * rhs, axis=0), dtype=np.float64)
    convergence_threshold_sq = tol_sq * np.maximum(rhs_norm_sq, 1.0)
    converged = residual_norm_sq <= convergence_threshold_sq
    best_solution = jnp.asarray(solution, dtype=solver_dtype)
    best_residual_norm_sq = float(np.max(residual_norm_sq))
    if np.all(converged):
        return LinearSolveResult(
            solution=np.asarray(solution, dtype=output_dtype),
            status=LinearSolveStatus.CONVERGED,
            iterations=0,
            residual_norm=math.sqrt(max(best_residual_norm_sq, 0.0)),
            matvec_count=matvec_count,
            elapsed_seconds=time.monotonic() - t_start,
        )

    preconditioned_residual = residual if preconditioner is None else preconditioner(residual)
    search_direction = preconditioned_residual
    residual_dot = np.asarray(jnp.sum(residual * preconditioned_residual, axis=0), dtype=np.float64)
    for iteration_index in range(max_iterations):
        iteration = iteration_index + 1
        if _time_budget_exceeded(deadline):
            residual_norm = math.sqrt(max(best_residual_norm_sq, 0.0))
            _maybe_emit_cg_progress(
                callback=progress_callback,
                progress_interval=progress_interval,
                iteration=iteration_index,
                max_iterations=max_iterations,
                started_at=t_start,
                residual_norm=residual_norm,
                matvec_count=matvec_count,
                active_rhs=int(np.sum(~converged)),
                status=LinearSolveStatus.TIME_BUDGET,
            )
            return LinearSolveResult(
                solution=np.asarray(best_solution, dtype=output_dtype),
                status=LinearSolveStatus.TIME_BUDGET,
                iterations=iteration_index,
                residual_norm=residual_norm,
                matvec_count=matvec_count,
                elapsed_seconds=time.monotonic() - t_start,
            )
        active_columns = np.flatnonzero(~converged).astype(np.int32, copy=False)
        active_mask = jnp.asarray((~converged).astype(np.asarray(jnp.zeros((), dtype=solver_dtype)).dtype), dtype=solver_dtype)
        masked_search = search_direction * active_mask[None, :]
        operator_search = apply_operator_active(masked_search, active_columns) * active_mask[None, :]
        step_denom = np.asarray(jnp.sum(masked_search * operator_search, axis=0), dtype=np.float64)
        invalid = (~converged) & (~np.isfinite(step_denom) | (step_denom <= 0.0))
        if np.any(invalid):
            invalid_mask = jnp.asarray(invalid.astype(np.asarray(jnp.zeros((), dtype=solver_dtype)).dtype), dtype=solver_dtype)
            residual = rhs - apply_operator_active(solution, active_columns)
            refreshed_preconditioned = residual if preconditioner is None else preconditioner(residual)
            search_direction = search_direction * (1.0 - invalid_mask[None, :]) + refreshed_preconditioned * invalid_mask[None, :]
            residual_dot = np.where(
                invalid,
                np.asarray(jnp.sum(residual * refreshed_preconditioned, axis=0), dtype=np.float64),
                residual_dot,
            )
            masked_search = search_direction * active_mask[None, :]
            operator_search = apply_operator_active(masked_search, active_columns) * active_mask[None, :]
            step_denom = np.asarray(jnp.sum(masked_search * operator_search, axis=0), dtype=np.float64)
            if np.any((~converged) & (~np.isfinite(step_denom) | (step_denom <= 0.0))):
                residual_norm = math.sqrt(max(best_residual_norm_sq, 0.0))
                _maybe_emit_cg_progress(
                    callback=progress_callback,
                    progress_interval=progress_interval,
                    iteration=iteration,
                    max_iterations=max_iterations,
                    started_at=t_start,
                    residual_norm=residual_norm,
                    matvec_count=matvec_count,
                    active_rhs=int(np.sum(~converged)),
                    status=LinearSolveStatus.NUMERICAL_FAILURE,
                )
                return LinearSolveResult(
                    solution=np.asarray(best_solution, dtype=output_dtype),
                    status=LinearSolveStatus.NUMERICAL_FAILURE,
                    iterations=iteration,
                    residual_norm=residual_norm,
                    matvec_count=matvec_count,
                    elapsed_seconds=time.monotonic() - t_start,
                )
        step_scale = np.zeros_like(step_denom)
        active_indices = ~converged
        step_scale[active_indices] = residual_dot[active_indices] / step_denom[active_indices]
        step_scale_jax = jnp.asarray(step_scale, dtype=solver_dtype)
        solution = solution + masked_search * step_scale_jax[None, :]
        residual = residual - operator_search * step_scale_jax[None, :]
        if (iteration_index + 1) % residual_refresh_interval == 0:
            residual = rhs - apply_operator_active(solution, active_columns)
        residual_norm_sq = np.asarray(jnp.sum(residual * residual, axis=0), dtype=np.float64)
        converged = residual_norm_sq <= convergence_threshold_sq
        new_active_count = int(np.sum(~converged))
        residual_summary = float(np.max(residual_norm_sq))
        if residual_summary < best_residual_norm_sq:
            best_residual_norm_sq = residual_summary
            best_solution = jnp.asarray(solution, dtype=solver_dtype)
        _maybe_emit_cg_progress(
            callback=progress_callback,
            progress_interval=progress_interval,
            iteration=iteration,
            max_iterations=max_iterations,
            started_at=t_start,
            residual_norm=math.sqrt(max(residual_summary, 0.0)),
            matvec_count=matvec_count,
            active_rhs=new_active_count,
            status=LinearSolveStatus.CONVERGED if new_active_count == 0 else None,
        )
        if new_active_count == 0:
            return LinearSolveResult(
                solution=np.asarray(solution, dtype=output_dtype),
                status=LinearSolveStatus.CONVERGED,
                iterations=iteration,
                residual_norm=math.sqrt(max(residual_summary, 0.0)),
                matvec_count=matvec_count,
                elapsed_seconds=time.monotonic() - t_start,
            )
        updated_preconditioned = residual if preconditioner is None else preconditioner(residual)
        updated_residual_dot = np.asarray(jnp.sum(residual * updated_preconditioned, axis=0), dtype=np.float64)
        beta = np.zeros_like(updated_residual_dot)
        beta[~converged] = updated_residual_dot[~converged] / np.maximum(residual_dot[~converged], 1e-30)
        invalid_beta = (~converged) & (~np.isfinite(beta) | (beta < 0.0))
        if np.any(invalid_beta):
            invalid_beta_mask = jnp.asarray(invalid_beta.astype(np.asarray(jnp.zeros((), dtype=solver_dtype)).dtype), dtype=solver_dtype)
            residual = rhs - apply_operator_active(solution, active_columns)
            updated_preconditioned = residual if preconditioner is None else preconditioner(residual)
            updated_residual_dot = np.asarray(jnp.sum(residual * updated_preconditioned, axis=0), dtype=np.float64)
            search_direction = search_direction * (1.0 - invalid_beta_mask[None, :]) + updated_preconditioned * invalid_beta_mask[None, :]
            beta[invalid_beta] = 0.0
            refreshed_summary = float(np.max(np.asarray(jnp.sum(residual * residual, axis=0), dtype=np.float64)))
            if refreshed_summary < best_residual_norm_sq:
                best_residual_norm_sq = refreshed_summary
                best_solution = jnp.asarray(solution, dtype=solver_dtype)
        beta_jax = jnp.asarray(beta, dtype=solver_dtype)
        search_direction = updated_preconditioned + search_direction * beta_jax[None, :]
        search_direction = search_direction * jnp.asarray((~converged).astype(np.asarray(jnp.zeros((), dtype=solver_dtype)).dtype), dtype=solver_dtype)[None, :]
        residual_dot = updated_residual_dot
    final_residual = float(np.max(np.asarray(jnp.sum(residual * residual, axis=0), dtype=np.float64)))
    residual_norm = math.sqrt(max(min(final_residual, best_residual_norm_sq), 0.0))
    _maybe_emit_cg_progress(
        callback=progress_callback,
        progress_interval=progress_interval,
        iteration=max_iterations,
        max_iterations=max_iterations,
        started_at=t_start,
        residual_norm=residual_norm,
        matvec_count=matvec_count,
        active_rhs=int(np.sum(~converged)),
        status=LinearSolveStatus.MAX_ITER,
    )
    return LinearSolveResult(
        solution=np.asarray(best_solution, dtype=output_dtype),
        status=LinearSolveStatus.MAX_ITER,
        iterations=max_iterations,
        residual_norm=residual_norm,
        matvec_count=matvec_count,
        elapsed_seconds=time.monotonic() - t_start,
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
    start_vector: JaxArray,
    step_count: int,
) -> JaxArray:
    operator_dtype = linear_operator.dtype or jnp.result_type(start_vector.dtype, jnp.float32)
    normalized_start = jnp.asarray(start_vector, dtype=operator_dtype)
    normalized_start /= jnp.maximum(jnp.linalg.norm(normalized_start), 1e-12)
    basis_vectors: list[JaxArray] = []
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
    required_off_diag_len = max(len(alpha_values) - 1, 0)
    if required_off_diag_len > 0:
        # Pad with zeros if Lanczos terminated early (beta < 1e-10): a zero
        # off-diagonal block-decouples the tridiagonal at that point, which
        # is the mathematically correct lock-out for the remaining subspace.
        if len(beta_values) < required_off_diag_len:
            padded = list(beta_values) + [0.0] * (required_off_diag_len - len(beta_values))
            off_diagonal = jnp.asarray(padded, dtype=operator_dtype)
        else:
            off_diagonal = jnp.asarray(beta_values[:required_off_diag_len], dtype=operator_dtype)
        tridiagonal = tridiagonal.at[jnp.arange(off_diagonal.shape[0]), jnp.arange(1, off_diagonal.shape[0] + 1)].set(off_diagonal)
        tridiagonal = tridiagonal.at[jnp.arange(1, off_diagonal.shape[0] + 1), jnp.arange(off_diagonal.shape[0])].set(off_diagonal)
    return tridiagonal


def _lanczos_tridiagonal_block(
    linear_operator: LinearOperator,
    start_matrix: JaxArray,
    step_count: int,
) -> tuple[F64Array, ...]:
    if start_matrix.ndim != 2:
        raise ValueError("Lanczos block requires a matrix of start vectors.")
    if start_matrix.shape[1] == 1:
        return (np.asarray(_lanczos_tridiagonal(linear_operator, start_matrix[:, 0], step_count), dtype=np.float64),)
    operator_dtype = linear_operator.dtype or jnp.result_type(start_matrix.dtype, jnp.float32)
    current_block = jnp.asarray(start_matrix, dtype=operator_dtype)
    current_norms = jnp.maximum(jnp.linalg.norm(current_block, axis=0, keepdims=True), 1e-12)
    current_block = current_block / current_norms
    basis_blocks: list[JaxArray] = []
    alpha_rows: list[F64Array] = []
    beta_rows: list[F64Array] = []
    previous_block = jnp.zeros_like(current_block)
    previous_beta = jnp.zeros(current_block.shape[1], dtype=operator_dtype)
    active = np.ones(current_block.shape[1], dtype=bool)
    step_lengths = np.full(current_block.shape[1], step_count, dtype=np.int32)
    steps_completed = 0
    for step_index in range(step_count):
        projected_block = _apply_operator_block(linear_operator, current_block, operator_dtype)
        if basis_blocks:
            projected_block = projected_block - previous_block * previous_beta[None, :]
        alpha_row = np.asarray(jnp.sum(current_block * projected_block, axis=0), dtype=np.float64)
        projected_block = projected_block - current_block * jnp.asarray(alpha_row, dtype=operator_dtype)[None, :]
        if basis_blocks:
            for basis_block in basis_blocks:
                coefficients = jnp.sum(basis_block * projected_block, axis=0)
                projected_block = projected_block - basis_block * coefficients[None, :]
        beta_row = np.asarray(jnp.linalg.norm(projected_block, axis=0), dtype=np.float64)
        basis_blocks.append(current_block)
        alpha_rows.append(alpha_row)
        steps_completed = step_index + 1
        if step_index + 1 >= step_count:
            break
        beta_rows.append(beta_row)
        newly_converged = active & (beta_row < 1e-10)
        step_lengths[newly_converged] = step_index + 1
        active = active & ~newly_converged
        if not np.any(active):
            break
        beta_safe = np.where(active, beta_row, 1.0)
        normalized_next = np.asarray(projected_block, dtype=np.float64) / beta_safe[None, :]
        normalized_next[:, ~active] = 0.0
        previous_block = current_block
        previous_beta = jnp.asarray(np.where(active, beta_row, 0.0), dtype=operator_dtype)
        current_block = jnp.asarray(normalized_next, dtype=operator_dtype)
    tridiagonal_blocks: list[F64Array] = []
    alpha_matrix = np.asarray(alpha_rows, dtype=np.float64)
    beta_matrix = np.asarray(beta_rows, dtype=np.float64) if beta_rows else np.zeros((0, current_block.shape[1]), dtype=np.float64)
    for column_index in range(current_block.shape[1]):
        current_steps = int(step_lengths[column_index]) if not active[column_index] else steps_completed
        diagonal = alpha_matrix[:current_steps, column_index]
        tridiagonal = np.diag(diagonal)
        if current_steps > 1:
            off_diagonal = beta_matrix[: current_steps - 1, column_index]
            tridiagonal[np.arange(current_steps - 1), np.arange(1, current_steps)] = off_diagonal
            tridiagonal[np.arange(1, current_steps), np.arange(current_steps - 1)] = off_diagonal
        tridiagonal_blocks.append(tridiagonal.astype(np.float64, copy=False))
    return tuple(tridiagonal_blocks)

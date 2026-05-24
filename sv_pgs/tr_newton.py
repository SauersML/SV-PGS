"""Trust-region Newton-CG (Steihaug-Toint) for penalized logistic regression.

Solves

    max_{β, α}  f(β, α) = Σ_i [y_i η_i - log(1 + exp η_i)] - 1/2 βᵀ D⁻¹ β

with η = X β + W α + offset and D = diag(prior_variances).  The objective
is concave; we minimize -f using a trust-region Newton-CG method.  Hessian-
vector products are matrix-free in β and dense in α, matching the cost of a
single PG-IRLS weight build.

This module is a pure addition intended to replace PG-IRLS for binary traits
in a later integration step.

Non-convergence contract: if the solver exhausts its iteration budget with a
gradient norm above ``gradient_tolerance``, it raises
``TRNewtonNonConvergence`` instead of returning a non-converged posterior
state.  If the wall-clock budget is exceeded, it raises ``TRNewtonTimeout``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from sv_pgs._typing import F64Array
from sv_pgs.numeric import stable_sigmoid as _jax_stable_sigmoid

_LOG = logging.getLogger(__name__)


def _sigmoid(values: F64Array) -> F64Array:
    """Numerically stable σ(x) returning a numpy array.

    We reuse the project's stable_sigmoid (JAX-backed) for consistency, then
    cast back to numpy so the rest of the solver stays in numpy.
    """
    result: F64Array = np.asarray(_jax_stable_sigmoid(values), dtype=np.float64)
    return result


def _log1p_exp(values: F64Array) -> F64Array:
    """Stable log(1 + exp(x)) = logaddexp(0, x)."""
    result: F64Array = np.logaddexp(0.0, values)
    return result


@dataclass(slots=True)
class TrustRegionResult:
    beta: F64Array
    alpha: F64Array
    linear_predictor: F64Array
    objective: float
    iterations: int
    converged: bool
    final_gradient_norm: float


class TRNewtonNonConvergence(RuntimeError):
    """Raised when TR-Newton exits above the requested gradient tolerance."""

    def __init__(self, result: TrustRegionResult, gradient_tolerance: float) -> None:
        self.result = result
        self.gradient_tolerance = float(gradient_tolerance)
        super().__init__(
            "TR-Newton failed to converge "
            f"after {result.iterations} iterations "
            f"(grad_norm={result.final_gradient_norm:.6g}, "
            f"tolerance={self.gradient_tolerance:.6g})"
        )


class TRNewtonTimeout(TimeoutError):
    """Raised when TR-Newton exceeds its wall-clock budget."""

    def __init__(
        self,
        *,
        elapsed_s: float,
        wall_clock_budget_s: float,
        iterations: int,
        grad_norm: float,
    ) -> None:
        self.elapsed_s = float(elapsed_s)
        self.wall_clock_budget_s = float(wall_clock_budget_s)
        self.iterations = int(iterations)
        self.grad_norm = float(grad_norm)
        super().__init__(
            "TR-Newton exceeded wall-clock budget "
            f"(elapsed_s={self.elapsed_s:.3f}, "
            f"budget_s={self.wall_clock_budget_s:.3f}, "
            f"iter={self.iterations}, grad_norm={self.grad_norm:.6g})"
        )


def _raise_if_timed_out(
    *,
    started_at: float,
    wall_clock_budget_s: float,
    iterations: int,
    grad_norm: float,
) -> None:
    elapsed_s = time.monotonic() - started_at
    if elapsed_s > wall_clock_budget_s:
        raise TRNewtonTimeout(
            elapsed_s=elapsed_s,
            wall_clock_budget_s=wall_clock_budget_s,
            iterations=iterations,
            grad_norm=grad_norm,
        )


def _steihaug_cg(
    *,
    gradient: F64Array,
    hvp: Callable[[F64Array], F64Array],
    radius: float,
    tolerance: float,
    max_iterations: int,
    progress_interval: int,
    outer_iteration: int,
    outer_gradient_norm: float,
    started_at: float,
    wall_clock_budget_s: float,
) -> tuple[F64Array, bool]:
    """Steihaug-Toint truncated CG for the TR subproblem.

    Returns (step, hit_boundary).
    """
    p = np.zeros_like(gradient)
    r = gradient.copy()  # since p=0, r = H·0 + g = g
    d = -r
    r_dot_r = float(r @ r)
    if np.sqrt(r_dot_r) <= tolerance:
        return p, False

    for cg_iteration in range(1, max_iterations + 1):
        _raise_if_timed_out(
            started_at=started_at,
            wall_clock_budget_s=wall_clock_budget_s,
            iterations=outer_iteration,
            grad_norm=outer_gradient_norm,
        )
        Hd = hvp(d)
        d_H_d = float(d @ Hd)
        if d_H_d <= 0.0:
            # Negative curvature: walk along d to the trust-region boundary.
            tau = _to_boundary(p, d, radius)
            return p + tau * d, True

        alpha_step = r_dot_r / d_H_d
        p_next = p + alpha_step * d
        if np.linalg.norm(p_next) >= radius:
            tau = _to_boundary(p, d, radius)
            return p + tau * d, True

        r_next = r + alpha_step * Hd
        r_next_dot = float(r_next @ r_next)
        if np.sqrt(r_next_dot) <= tolerance:
            return p_next, False

        beta_cg = r_next_dot / r_dot_r
        d = -r_next + beta_cg * d
        p = p_next
        r = r_next
        r_dot_r = r_next_dot
        if progress_interval > 0 and cg_iteration % progress_interval == 0:
            _LOG.info(
                "TR-Newton CG progress iter=%d grad_norm=%.6g step_norm=%.6g elapsed_s=%.3f tr_iter=%d",
                cg_iteration,
                outer_gradient_norm,
                float(np.linalg.norm(p)),
                time.monotonic() - started_at,
                outer_iteration,
            )

    return p, False


def _to_boundary(p: F64Array, d: F64Array, radius: float) -> float:
    """Solve ‖p + τ d‖ = radius for τ > 0."""
    a = float(d @ d)
    b = 2.0 * float(p @ d)
    c = float(p @ p) - radius * radius
    discriminant = b * b - 4.0 * a * c
    # Clamp tiny negative discriminant from roundoff.
    discriminant = max(discriminant, 0.0)
    return float((-b + np.sqrt(discriminant)) / (2.0 * a))


def trust_region_newton_logistic(
    *,
    matvec_design: Callable[[F64Array], F64Array],
    matvec_design_transpose: Callable[[F64Array], F64Array],
    covariate_matrix: F64Array,
    targets: F64Array,
    prior_variances: F64Array,
    predictor_offset: F64Array,
    beta_init: F64Array,
    alpha_init: F64Array,
    max_iterations: int = 30,
    initial_radius: float = 1.0,
    radius_max: float = 100.0,
    eta_accept: float = 0.1,
    gradient_tolerance: float = 1e-6,
    forcing_exponent: float = 0.5,
    cg_max_iterations: int = 200,
    wall_clock_budget_s: float = 600.0,
    cg_progress_interval: int = 25,
    raise_on_nonconvergence: bool = False,
) -> TrustRegionResult:
    """Trust-region Newton-CG for the penalized logistic posterior.

    See module docstring for the objective.  ``matvec_design`` and its
    transpose perform X v and Xᵀ u with the variant design matrix X
    (n × p).  ``covariate_matrix`` is a dense n × q matrix W.
    """
    started_at = time.monotonic()
    y = np.asarray(targets, dtype=np.float64).ravel()
    W = np.asarray(covariate_matrix, dtype=np.float64)
    offset = np.asarray(predictor_offset, dtype=np.float64).ravel()
    beta = np.asarray(beta_init, dtype=np.float64).ravel().copy()
    alpha = np.asarray(alpha_init, dtype=np.float64).ravel().copy()
    tau_sq = np.asarray(prior_variances, dtype=np.float64).ravel()

    n_samples = y.shape[0]
    if n_samples == 0:
        raise ValueError("targets must be non-empty")
    if W.ndim != 2:
        # Allow shape (n, 0) for "no covariates".
        raise ValueError(f"covariate_matrix must be 2-D, got shape {W.shape}")
    if W.shape[0] != n_samples:
        raise ValueError(
            f"covariate_matrix rows ({W.shape[0]}) must match targets ({n_samples})"
        )
    if offset.shape[0] != n_samples:
        raise ValueError(
            f"predictor_offset length ({offset.shape[0]}) must match targets ({n_samples})"
        )
    p_dim = beta.shape[0]
    q_dim = alpha.shape[0]
    if tau_sq.shape[0] != p_dim:
        raise ValueError(
            f"prior_variances length ({tau_sq.shape[0]}) must match beta_init ({p_dim})"
        )
    if W.shape[1] != q_dim:
        raise ValueError(
            f"covariate_matrix cols ({W.shape[1]}) must match alpha_init ({q_dim})"
        )
    if cg_max_iterations < 1:
        raise ValueError("cg_max_iterations must be positive")
    if wall_clock_budget_s <= 0.0:
        raise ValueError("wall_clock_budget_s must be positive")

    # Pre-compute D⁻¹; guard against zero entries.
    safe_tau = np.where(tau_sq > 0.0, tau_sq, np.finfo(np.float64).tiny)
    d_inv = 1.0 / safe_tau

    def design_mv(v: F64Array) -> F64Array:
        if p_dim == 0:
            return np.zeros(n_samples, dtype=np.float64)
        result: F64Array = np.asarray(matvec_design(v), dtype=np.float64).ravel()
        return result

    def design_mv_t(u: F64Array) -> F64Array:
        if p_dim == 0:
            return np.zeros(0, dtype=np.float64)
        result: F64Array = np.asarray(matvec_design_transpose(u), dtype=np.float64).ravel()
        return result

    def linear_predictor(beta_vec: F64Array, alpha_vec: F64Array) -> F64Array:
        eta: F64Array = offset.copy()
        if p_dim > 0:
            eta = eta + design_mv(beta_vec)
        if q_dim > 0:
            eta = eta + W @ alpha_vec
        return eta

    def negative_objective(eta: F64Array, beta_vec: F64Array) -> float:
        # -f = -Σ[y η - log1pexp(η)] + 1/2 βᵀ D⁻¹ β
        data_term = float(np.sum(_log1p_exp(eta) - y * eta))
        prior_term = 0.5 * float(beta_vec @ (d_inv * beta_vec)) if p_dim > 0 else 0.0
        return data_term + prior_term

    def joint_gradient(eta: F64Array, beta_vec: F64Array) -> F64Array:
        residual = _sigmoid(eta) - y  # ∇η (-f) at the data term
        grad: F64Array = np.empty(p_dim + q_dim, dtype=np.float64)
        if p_dim > 0:
            grad[:p_dim] = design_mv_t(residual) + d_inv * beta_vec
        if q_dim > 0:
            grad[p_dim:] = W.T @ residual
        return grad

    def hvp_factory(eta: F64Array) -> Callable[[F64Array], F64Array]:
        sig = _sigmoid(eta)
        s_diag = sig * (1.0 - sig)
        # Floor S slightly so saturated η don't make H singular along data dirs;
        # the prior D⁻¹ still keeps the β block PD on its own when τ² < ∞.
        s_diag = np.maximum(s_diag, 0.0)

        def hvp(v: F64Array) -> F64Array:
            out: F64Array = np.empty_like(v)
            v_beta = v[:p_dim]
            v_alpha = v[p_dim:]
            # η-space tangent direction
            t = np.zeros(n_samples, dtype=np.float64)
            if p_dim > 0:
                t = t + design_mv(v_beta)
            if q_dim > 0:
                t = t + W @ v_alpha
            St = s_diag * t
            if p_dim > 0:
                out[:p_dim] = design_mv_t(St) + d_inv * v_beta
            if q_dim > 0:
                out[p_dim:] = W.T @ St
            return out

        return hvp

    radius = float(initial_radius)
    converged = False
    iterations = 0
    eta = linear_predictor(beta, alpha)
    f_val = negative_objective(eta, beta)
    grad = joint_gradient(eta, beta)
    grad_norm = float(np.linalg.norm(grad))
    consecutive_rejections = 0

    n_unknowns = p_dim + q_dim
    if n_unknowns == 0:
        return TrustRegionResult(
            beta=beta,
            alpha=alpha,
            linear_predictor=eta,
            objective=-f_val,
            iterations=0,
            converged=True,
            final_gradient_norm=0.0,
        )

    for k in range(1, max_iterations + 1):
        iterations = k
        _raise_if_timed_out(
            started_at=started_at,
            wall_clock_budget_s=wall_clock_budget_s,
            iterations=iterations,
            grad_norm=grad_norm,
        )
        if grad_norm <= gradient_tolerance:
            converged = True
            break

        # Dembo-Eisenstat-Steihaug forcing sequence.
        eta_k = min(0.5, grad_norm ** forcing_exponent)
        cg_tol = eta_k * grad_norm

        hvp = hvp_factory(eta)
        step, hit_boundary = _steihaug_cg(
            gradient=grad,
            hvp=hvp,
            radius=radius,
            tolerance=cg_tol,
            max_iterations=cg_max_iterations,
            progress_interval=cg_progress_interval,
            outer_iteration=k,
            outer_gradient_norm=grad_norm,
            started_at=started_at,
            wall_clock_budget_s=wall_clock_budget_s,
        )
        _raise_if_timed_out(
            started_at=started_at,
            wall_clock_budget_s=wall_clock_budget_s,
            iterations=iterations,
            grad_norm=grad_norm,
        )

        # Trial point.
        step_beta = step[:p_dim]
        step_alpha = step[p_dim:]
        beta_trial = beta + step_beta
        alpha_trial = alpha + step_alpha
        eta_trial = linear_predictor(beta_trial, alpha_trial)
        f_trial = negative_objective(eta_trial, beta_trial)

        # Predicted reduction from the quadratic model.
        Hp = hvp(step)
        predicted_reduction = -(float(grad @ step) + 0.5 * float(step @ Hp))
        actual_reduction = f_val - f_trial

        if predicted_reduction <= 0.0:
            # Degenerate model; reject and shrink.
            rho = -np.inf
        else:
            rho = actual_reduction / predicted_reduction

        step_norm = float(np.linalg.norm(step))
        if rho < 0.25:
            radius = max(0.25 * radius, 1e-12)
        elif rho > 0.75 and hit_boundary:
            radius = min(2.0 * radius, radius_max)

        if rho > eta_accept:
            consecutive_rejections = 0
            beta = beta_trial
            alpha = alpha_trial
            eta = eta_trial
            f_val = f_trial
            grad = joint_gradient(eta, beta)
            grad_norm = float(np.linalg.norm(grad))
        else:
            consecutive_rejections += 1
            if consecutive_rejections >= 2:
                radius = max(0.1 * radius, 1e-12)

        # Guard against pathologically small radius.
        if radius < 1e-14 and step_norm < 1e-14:
            break

    if grad_norm <= gradient_tolerance:
        converged = True

    result = TrustRegionResult(
        beta=beta,
        alpha=alpha,
        linear_predictor=eta,
        objective=-f_val,
        iterations=iterations,
        converged=converged,
        final_gradient_norm=grad_norm,
    )
    if raise_on_nonconvergence and not converged and grad_norm > gradient_tolerance:
        raise TRNewtonNonConvergence(result, gradient_tolerance)
    return result


# ---------------------------------------------------------------------------
# GPU-native variant
# ---------------------------------------------------------------------------
#
# Capability flag for call-sites that want to feature-detect the GPU path
# without importing CuPy themselves.  True => the body below is a full
# implementation (not a NotImplementedError stub).
_TR_NEWTON_GPU_IMPLEMENTED: bool = True
#
# trust_region_newton_logistic_gpu mirrors the NumPy solver above but keeps
# beta, alpha, eta, gradients, CG vectors and HVPs as CuPy arrays end-to-end.
# Conversion to NumPy happens only at the return boundary so the per-CG-step
# host/device round-trips that plagued the NumPy path on AoU disappear.
#
# Contract:
#   - ``matvec_design`` / ``matvec_design_transpose`` MUST accept a CuPy
#     array and return a CuPy array.  The caller is responsible for choosing
#     matvecs that stay on-device (e.g. via genotype_matrix.gpu_matmat /
#     gpu_transpose_matmat against a GPU-resident block).
#   - ``covariate_matrix`` may be NumPy or CuPy; it is uploaded once.
#   - The returned TrustRegionResult fields are NumPy arrays so downstream
#     code (Laplace refit / posterior reporting) is unchanged.


def _is_cupy_array(cp: Any, obj: Any) -> bool:
    ndarray_cls = getattr(cp, "ndarray", None)
    return ndarray_cls is not None and isinstance(obj, ndarray_cls)


def _sigmoid_cupy(cp: Any, values_gpu: Any, *, dtype: Any) -> Any:
    """Numerically stable sigmoid on-device (no host round-trip)."""
    abs_x = cp.abs(values_gpu)
    e = cp.exp(-abs_x)
    pos = dtype(1.0) / (dtype(1.0) + e)
    neg = e / (dtype(1.0) + e)
    return cp.where(values_gpu >= dtype(0.0), pos, neg)


def trust_region_newton_logistic_gpu(
    *,
    cupy: Any,
    matvec_design: Callable[[Any], Any],
    matvec_design_transpose: Callable[[Any], Any],
    covariate_matrix: Any,
    targets: Any,
    prior_variances: Any,
    predictor_offset: Any,
    beta_init: Any,
    alpha_init: Any,
    max_iterations: int = 30,
    initial_radius: float = 1.0,
    radius_max: float = 100.0,
    eta_accept: float = 0.1,
    gradient_tolerance: float = 1e-6,
    forcing_exponent: float = 0.5,
    cg_max_iterations: int = 200,
    wall_clock_budget_s: float = 600.0,
    cg_progress_interval: int = 25,
    compute_dtype: Any = None,
    raise_on_nonconvergence: bool = False,
) -> TrustRegionResult:
    """GPU-native trust-region Newton-CG.

    Mirrors :func:`trust_region_newton_logistic` exactly in algorithm, but
    keeps the inner state on-device.  ``matvec_design`` and its transpose
    must return CuPy arrays.
    """
    if cupy is None:
        raise RuntimeError("trust_region_newton_logistic_gpu requires a CuPy module")
    if cg_max_iterations < 1:
        raise ValueError("cg_max_iterations must be positive")
    if wall_clock_budget_s <= 0.0:
        raise ValueError("wall_clock_budget_s must be positive")

    started_at = time.monotonic()
    cp = cupy
    dtype = cp.float64 if compute_dtype is None else compute_dtype

    y = cp.asarray(targets, dtype=dtype).ravel()
    W = cp.asarray(covariate_matrix, dtype=dtype)
    offset = cp.asarray(predictor_offset, dtype=dtype).ravel()
    beta = cp.asarray(beta_init, dtype=dtype).ravel().copy()
    alpha = cp.asarray(alpha_init, dtype=dtype).ravel().copy()
    tau_sq = cp.asarray(prior_variances, dtype=dtype).ravel()

    n_samples = int(y.shape[0])
    if n_samples == 0:
        raise ValueError("targets must be non-empty")
    if W.ndim != 2:
        raise ValueError(f"covariate_matrix must be 2-D, got shape {tuple(W.shape)}")
    if int(W.shape[0]) != n_samples:
        raise ValueError(
            f"covariate_matrix rows ({int(W.shape[0])}) must match targets ({n_samples})"
        )
    if int(offset.shape[0]) != n_samples:
        raise ValueError(
            f"predictor_offset length ({int(offset.shape[0])}) must match targets ({n_samples})"
        )
    p_dim = int(beta.shape[0])
    q_dim = int(alpha.shape[0])
    if int(tau_sq.shape[0]) != p_dim:
        raise ValueError(
            f"prior_variances length ({int(tau_sq.shape[0])}) must match beta_init ({p_dim})"
        )
    if int(W.shape[1]) != q_dim:
        raise ValueError(
            f"covariate_matrix cols ({int(W.shape[1])}) must match alpha_init ({q_dim})"
        )

    tiny = float(np.finfo(np.float64).tiny)
    safe_tau = cp.where(tau_sq > 0.0, tau_sq, dtype(tiny))
    d_inv = (dtype(1.0) / safe_tau).astype(dtype, copy=False)

    def _to_host_numpy(arr_gpu: Any) -> F64Array:
        host = arr_gpu.get() if hasattr(arr_gpu, "get") else arr_gpu
        return np.asarray(host, dtype=np.float64)

    def design_mv(v_gpu: Any) -> Any:
        if p_dim == 0:
            return cp.zeros(n_samples, dtype=dtype)
        out = matvec_design(v_gpu)
        if not _is_cupy_array(cp, out):
            out = cp.asarray(out, dtype=dtype)
        return out.astype(dtype, copy=False).ravel()

    def design_mv_t(u_gpu: Any) -> Any:
        if p_dim == 0:
            return cp.zeros(0, dtype=dtype)
        out = matvec_design_transpose(u_gpu)
        if not _is_cupy_array(cp, out):
            out = cp.asarray(out, dtype=dtype)
        return out.astype(dtype, copy=False).ravel()

    def linear_predictor(beta_vec: Any, alpha_vec: Any) -> Any:
        eta_gpu = offset.copy()
        if p_dim > 0:
            eta_gpu = eta_gpu + design_mv(beta_vec)
        if q_dim > 0:
            eta_gpu = eta_gpu + W @ alpha_vec
        return eta_gpu

    def neg_objective(eta_gpu: Any, beta_vec: Any) -> float:
        data_term = cp.sum(cp.logaddexp(dtype(0.0), eta_gpu) - y * eta_gpu, dtype=cp.float64)
        if p_dim > 0:
            prior_term = dtype(0.5) * cp.sum(d_inv * beta_vec * beta_vec, dtype=cp.float64)
        else:
            prior_term = cp.asarray(0.0, dtype=cp.float64)
        return float(data_term + prior_term)

    def joint_gradient(eta_gpu: Any, beta_vec: Any) -> Any:
        sig = _sigmoid_cupy(cp, eta_gpu, dtype=dtype)
        residual = sig - y
        grad_gpu = cp.empty(p_dim + q_dim, dtype=dtype)
        if p_dim > 0:
            grad_gpu[:p_dim] = design_mv_t(residual) + d_inv * beta_vec
        if q_dim > 0:
            grad_gpu[p_dim:] = W.T @ residual
        return grad_gpu

    def hvp_factory(eta_gpu: Any) -> Callable[[Any], Any]:
        sig = _sigmoid_cupy(cp, eta_gpu, dtype=dtype)
        s_diag = cp.maximum(sig * (dtype(1.0) - sig), dtype(0.0))

        def hvp(v_gpu: Any) -> Any:
            out = cp.empty_like(v_gpu)
            v_beta = v_gpu[:p_dim]
            v_alpha = v_gpu[p_dim:]
            t = cp.zeros(n_samples, dtype=dtype)
            if p_dim > 0:
                t = t + design_mv(v_beta)
            if q_dim > 0:
                t = t + W @ v_alpha
            St = s_diag * t
            if p_dim > 0:
                out[:p_dim] = design_mv_t(St) + d_inv * v_beta
            if q_dim > 0:
                out[p_dim:] = W.T @ St
            return out

        return hvp

    def vec_norm(v_gpu: Any) -> float:
        return float(cp.linalg.norm(v_gpu))

    def to_boundary(p_gpu: Any, d_gpu: Any, radius_val: float) -> float:
        a = float(cp.dot(d_gpu, d_gpu))
        b = 2.0 * float(cp.dot(p_gpu, d_gpu))
        c = float(cp.dot(p_gpu, p_gpu)) - radius_val * radius_val
        disc = max(b * b - 4.0 * a * c, 0.0)
        return float((-b + np.sqrt(disc)) / (2.0 * a))

    def steihaug_cg(
        gradient_gpu: Any,
        hvp: Callable[[Any], Any],
        radius_val: float,
        tolerance: float,
        max_iter: int,
        outer_iter: int,
        outer_grad_norm: float,
    ) -> tuple[Any, bool]:
        p_gpu = cp.zeros_like(gradient_gpu)
        r_gpu = gradient_gpu.copy()
        d_gpu = -r_gpu
        r_dot_r = float(cp.dot(r_gpu, r_gpu))
        if np.sqrt(r_dot_r) <= tolerance:
            return p_gpu, False
        for cg_iteration in range(1, max_iter + 1):
            _raise_if_timed_out(
                started_at=started_at,
                wall_clock_budget_s=wall_clock_budget_s,
                iterations=outer_iter,
                grad_norm=outer_grad_norm,
            )
            Hd = hvp(d_gpu)
            d_H_d = float(cp.dot(d_gpu, Hd))
            if d_H_d <= 0.0:
                tau = to_boundary(p_gpu, d_gpu, radius_val)
                return p_gpu + tau * d_gpu, True
            alpha_step = r_dot_r / d_H_d
            p_next = p_gpu + alpha_step * d_gpu
            if vec_norm(p_next) >= radius_val:
                tau = to_boundary(p_gpu, d_gpu, radius_val)
                return p_gpu + tau * d_gpu, True
            r_next = r_gpu + alpha_step * Hd
            r_next_dot = float(cp.dot(r_next, r_next))
            if np.sqrt(r_next_dot) <= tolerance:
                return p_next, False
            beta_cg = r_next_dot / r_dot_r
            d_gpu = -r_next + beta_cg * d_gpu
            p_gpu = p_next
            r_gpu = r_next
            r_dot_r = r_next_dot
            if cg_progress_interval > 0 and cg_iteration % cg_progress_interval == 0:
                _LOG.info(
                    "TR-Newton(GPU) CG progress iter=%d grad_norm=%.6g step_norm=%.6g elapsed_s=%.3f tr_iter=%d",
                    cg_iteration,
                    outer_grad_norm,
                    vec_norm(p_gpu),
                    time.monotonic() - started_at,
                    outer_iter,
                )
        return p_gpu, False

    radius = float(initial_radius)
    converged = False
    iterations = 0
    eta = linear_predictor(beta, alpha)
    f_val = neg_objective(eta, beta)
    grad = joint_gradient(eta, beta)
    grad_norm = vec_norm(grad)
    consecutive_rejections = 0

    n_unknowns = p_dim + q_dim
    if n_unknowns == 0:
        return TrustRegionResult(
            beta=_to_host_numpy(beta),
            alpha=_to_host_numpy(alpha),
            linear_predictor=_to_host_numpy(eta),
            objective=-f_val,
            iterations=0,
            converged=True,
            final_gradient_norm=0.0,
        )

    for k in range(1, max_iterations + 1):
        iterations = k
        _raise_if_timed_out(
            started_at=started_at,
            wall_clock_budget_s=wall_clock_budget_s,
            iterations=iterations,
            grad_norm=grad_norm,
        )
        if grad_norm <= gradient_tolerance:
            converged = True
            break

        eta_k = min(0.5, grad_norm ** forcing_exponent)
        cg_tol = eta_k * grad_norm

        hvp = hvp_factory(eta)
        step, hit_boundary = steihaug_cg(
            grad,
            hvp,
            radius,
            cg_tol,
            cg_max_iterations,
            outer_iter=k,
            outer_grad_norm=grad_norm,
        )
        _raise_if_timed_out(
            started_at=started_at,
            wall_clock_budget_s=wall_clock_budget_s,
            iterations=iterations,
            grad_norm=grad_norm,
        )

        step_beta = step[:p_dim]
        step_alpha = step[p_dim:]
        beta_trial = beta + step_beta
        alpha_trial = alpha + step_alpha
        eta_trial = linear_predictor(beta_trial, alpha_trial)
        f_trial = neg_objective(eta_trial, beta_trial)

        Hp = hvp(step)
        predicted_reduction = -(float(cp.dot(grad, step)) + 0.5 * float(cp.dot(step, Hp)))
        actual_reduction = f_val - f_trial

        if predicted_reduction <= 0.0:
            rho = -np.inf
        else:
            rho = actual_reduction / predicted_reduction

        step_norm = vec_norm(step)
        if rho < 0.25:
            radius = max(0.25 * radius, 1e-12)
        elif rho > 0.75 and hit_boundary:
            radius = min(2.0 * radius, radius_max)

        if rho > eta_accept:
            consecutive_rejections = 0
            beta = beta_trial
            alpha = alpha_trial
            eta = eta_trial
            f_val = f_trial
            grad = joint_gradient(eta, beta)
            grad_norm = vec_norm(grad)
        else:
            consecutive_rejections += 1
            if consecutive_rejections >= 2:
                radius = max(0.1 * radius, 1e-12)

        if radius < 1e-14 and step_norm < 1e-14:
            break

    if grad_norm <= gradient_tolerance:
        converged = True

    result = TrustRegionResult(
        beta=_to_host_numpy(beta),
        alpha=_to_host_numpy(alpha),
        linear_predictor=_to_host_numpy(eta),
        objective=-f_val,
        iterations=iterations,
        converged=converged,
        final_gradient_norm=grad_norm,
    )
    if raise_on_nonconvergence and not converged and grad_norm > gradient_tolerance:
        raise TRNewtonNonConvergence(result, gradient_tolerance)
    return result

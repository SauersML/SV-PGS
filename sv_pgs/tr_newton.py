"""Trust-region Newton-CG (Steihaug-Toint) for penalized logistic regression.

Solves

    max_{β, α}  f(β, α) = Σ_i [y_i η_i - log(1 + exp η_i)] - 1/2 βᵀ D⁻¹ β

with η = X β + W α + offset and D = diag(prior_variances).  The objective
is concave; we minimize -f using a trust-region Newton-CG method.  Hessian-
vector products are matrix-free in β and dense in α, matching the cost of a
single PG-IRLS weight build.

This module is a pure addition intended to replace PG-IRLS for binary traits
in a later integration step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from sv_pgs._typing import F64Array
from sv_pgs.numeric import stable_sigmoid as _jax_stable_sigmoid


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


def _steihaug_cg(
    *,
    gradient: F64Array,
    hvp: Callable[[F64Array], F64Array],
    radius: float,
    tolerance: float,
    max_iterations: int,
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

    for _ in range(max_iterations):
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
) -> TrustRegionResult:
    """Trust-region Newton-CG for the penalized logistic posterior.

    See module docstring for the objective.  ``matvec_design`` and its
    transpose perform X v and Xᵀ u with the variant design matrix X
    (n × p).  ``covariate_matrix`` is a dense n × q matrix W.
    """
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
            beta = beta_trial
            alpha = alpha_trial
            eta = eta_trial
            f_val = f_trial
            grad = joint_gradient(eta, beta)
            grad_norm = float(np.linalg.norm(grad))

        # Guard against pathologically small radius.
        if radius < 1e-14 and step_norm < 1e-14:
            break

    if grad_norm <= gradient_tolerance:
        converged = True

    return TrustRegionResult(
        beta=beta,
        alpha=alpha,
        linear_predictor=eta,
        objective=-f_val,
        iterations=iterations,
        converged=converged,
        final_gradient_norm=grad_norm,
    )

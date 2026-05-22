"""True variational ELBO for the polygenic-score model.

For each block update CAVI performs, this ELBO is provably non-decreasing.
Monitor it for monotonicity to catch numerical bugs in the inner solvers.
Use its relative change as a scale-invariant convergence criterion.
"""

from __future__ import annotations

import numpy as np

from sv_pgs._typing import F64Array, NDArray
from sv_pgs.config import TraitType


_LOG_2PI = float(np.log(2.0 * np.pi))
_LOG_2PI_E = float(np.log(2.0 * np.pi) + 1.0)


def _jj_lambda(xi: F64Array) -> F64Array:
    """Jaakkola-Jordan λ(ξ) = tanh(ξ/2) / (4 ξ), with safe limit 1/8 at ξ → 0."""
    xi = np.asarray(xi, dtype=np.float64)
    out = np.empty_like(xi)
    small = np.abs(xi) < 1e-6
    # Taylor: tanh(x/2)/(4x) = 1/8 - x²/96 + O(x⁴)
    out[small] = 0.125 - (xi[small] ** 2) / 96.0
    big = ~small
    out[big] = np.tanh(xi[big] / 2.0) / (4.0 * xi[big])
    return out


def _log_sigmoid(x: F64Array) -> F64Array:
    """Numerically stable log σ(x) = -softplus(-x)."""
    x = np.asarray(x, dtype=np.float64)
    # log σ(x) = -log(1 + exp(-x)); use -softplus(-x) trick
    return -np.logaddexp(0.0, -x)


def compute_elbo(
    *,
    trait_type: TraitType,
    targets: NDArray,
    covariate_matrix: NDArray,
    alpha: NDArray,
    beta: NDArray,
    beta_variance: NDArray,
    linear_predictor: NDArray,
    reduced_prior_variances: NDArray,
    sigma_error2: float,
    column_norms_sq: NDArray | None = None,
    predictor_variance: NDArray | None = None,
    local_scale_prior_objective: float = 0.0,
    scale_penalty_objective: float = 0.0,
) -> float:
    """Compute the variational ELBO for the Bayesian PGS model.

    The ELBO is a sum of: expected log-likelihood under q(β) (Gaussian for
    quantitative, Jaakkola-Jordan lower bound for binary), the Gaussian effect
    prior term, the entropy of q(β), and the point-mass prior contributions
    for λ and θ which the caller passes in as scalars (already evaluated at
    the current point estimates).

    Parameters
    ----------
    trait_type
        QUANTITATIVE or BINARY.
    targets
        Length-n response vector.
    covariate_matrix
        n × k design matrix for the fixed covariates W. May be empty (k=0).
    alpha
        Length-k covariate effects α̂.
    beta
        Length-p variant effects β̂ (variational posterior mean).
    beta_variance
        Length-p diagonal of the variational posterior covariance Σ_β.
    linear_predictor
        Length-n vector η = (any offset) + W α̂ + X β̂; we re-use the caller's
        copy rather than recomputing it.
    reduced_prior_variances
        Length-p vector τ_j² = (σ_g s_j)² λ_j.
    sigma_error2
        Quantitative residual variance σ_e²; ignored for binary.
    column_norms_sq
        Length-p vector ‖X[:,j]‖². If None we assume standardized columns
        with ‖X[:,j]‖² = n (matches the project's preprocessing convention).
    predictor_variance
        Length-n vector Var_q[η_i] = Σ_j X_ij² · Σ_β,jj. Required for
        binary traits; for quantitative only the trace is needed.
    local_scale_prior_objective
        log p(λ | a, b) evaluated at the current point estimates.
    scale_penalty_objective
        log p(θ) (ridge penalty) at the current θ̂.
    """
    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    n = y.shape[0]
    beta = np.asarray(beta, dtype=np.float64).reshape(-1)
    beta_var = np.asarray(beta_variance, dtype=np.float64).reshape(-1)
    tau2 = np.asarray(reduced_prior_variances, dtype=np.float64).reshape(-1)
    eta = np.asarray(linear_predictor, dtype=np.float64).reshape(-1)
    p = beta.shape[0]

    # covariate_matrix and alpha are part of the API (their effect is already
    # baked into linear_predictor by the caller); validate shapes as a no-op
    # consistency check rather than recomputing W α here.
    _cov = np.asarray(covariate_matrix, dtype=np.float64)
    _alpha = np.asarray(alpha, dtype=np.float64).reshape(-1)
    if _cov.ndim != 2 or _cov.shape[0] != n or _cov.shape[1] != _alpha.shape[0]:
        raise ValueError(
            "covariate_matrix shape must be (n, k) and match alpha length k"
        )

    if column_norms_sq is None:
        col_sq = np.full(p, float(n), dtype=np.float64)
    else:
        col_sq = np.asarray(column_norms_sq, dtype=np.float64).reshape(-1)

    # Guard against zero/negative variances which would blow up log/divide.
    tau2_safe = np.maximum(tau2, 1e-300)
    bvar_safe = np.maximum(beta_var, 1e-300)

    # ----- E_q[log p(β | τ²)] -----
    log_prior_beta = (
        -0.5 * p * _LOG_2PI
        - 0.5 * np.sum(np.log(tau2_safe))
        - 0.5 * np.sum((beta * beta + beta_var) / tau2_safe)
    )

    # ----- H[q(β)] -----
    entropy_beta = 0.5 * np.sum(np.log(bvar_safe)) + 0.5 * p * _LOG_2PI_E

    # ----- E_q[log p(y | ·)] -----
    if trait_type == TraitType.QUANTITATIVE:
        sigma2 = float(sigma_error2)
        residual = y - eta
        # tr(X Σ_β Xᵀ) = Σ_j ‖X[:,j]‖² · Σ_β,jj
        trace_term = float(np.sum(col_sq * beta_var))
        expected_loglik = (
            -0.5 * n * (_LOG_2PI + np.log(max(sigma2, 1e-300)))
            - 0.5 / max(sigma2, 1e-300) * (float(residual @ residual) + trace_term)
        )
    else:
        assert trait_type == TraitType.BINARY
        if predictor_variance is None:
            raise ValueError(
                "predictor_variance (Σ_j X_ij² · Σ_β,jj) is required for binary traits"
            )
        pvar = np.asarray(predictor_variance, dtype=np.float64).reshape(-1)
        if pvar.shape[0] != n:
            raise ValueError("predictor_variance must have length n")
        mu = eta
        # Tightest JJ bound at ξ_i = μ_i; the (μ² - ξ²) cancels, leaving -λ(ξ)·Var[η].
        xi = mu
        lam = _jj_lambda(xi)
        kappa = y - 0.5
        expected_loglik = float(
            np.sum(_log_sigmoid(xi) + kappa * mu - xi / 2.0 - lam * pvar)
        )

    elbo = (
        expected_loglik
        + log_prior_beta
        + float(local_scale_prior_objective)
        + float(scale_penalty_objective)
        + entropy_beta
    )
    return float(elbo)

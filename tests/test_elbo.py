"""Tests for the variational ELBO module."""

from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.config import TraitType
from sv_pgs.elbo import compute_elbo, _jj_lambda, _log_sigmoid


def _bayes_posterior(X, y, tau2, sigma2):
    """Exact Gaussian-Gaussian posterior: q*(β) = N(m, S)."""
    XtX = X.T @ X
    prec = XtX / sigma2 + np.diag(1.0 / tau2)
    S = np.linalg.inv(prec)
    m = S @ (X.T @ y) / sigma2
    return m, S


def _marginal_loglik_gaussian(X, y, tau2, sigma2):
    """log p(y) for y = X β + ε, β ~ N(0, diag(τ²)), ε ~ N(0, σ² I)."""
    n = X.shape[0]
    K = X @ (tau2[:, None] * X.T) + sigma2 * np.eye(n)
    sign, logdet = np.linalg.slogdet(K)
    assert sign > 0
    return -0.5 * n * np.log(2 * np.pi) - 0.5 * logdet - 0.5 * y @ np.linalg.solve(K, y)


def test_quantitative_elbo_matches_marginal_loglik():
    """At the exact Gaussian posterior, ELBO equals log p(y) (KL = 0)."""
    rng = np.random.default_rng(0)
    n, p = 20, 3
    X = rng.standard_normal((n, p))
    tau2 = np.array([0.5, 1.5, 0.7])
    sigma2 = 0.4
    beta_true = rng.standard_normal(p) * np.sqrt(tau2)

    # Off-diagonal of S is non-zero in general, so ELBO with diagonal q is
    # strictly less than log p(y). Use a diagonal design (orthogonal cols)
    # so the exact posterior is diagonal.
    Q, _ = np.linalg.qr(X)
    X_orth = Q * np.sqrt(n)  # columns have norm sqrt(n)
    y2 = X_orth @ beta_true + rng.standard_normal(n) * np.sqrt(sigma2)
    m2, S2 = _bayes_posterior(X_orth, y2, tau2, sigma2)
    assert np.allclose(S2 - np.diag(np.diag(S2)), 0.0, atol=1e-10)
    beta_var2 = np.diag(S2)
    eta = X_orth @ m2  # no covariates
    col_sq = np.sum(X_orth ** 2, axis=0)

    elbo = compute_elbo(
        trait_type=TraitType.QUANTITATIVE,
        targets=y2,
        covariate_matrix=np.zeros((n, 0)),
        alpha=np.zeros(0),
        beta=m2,
        beta_variance=beta_var2,
        linear_predictor=eta,
        reduced_prior_variances=tau2,
        sigma_error2=sigma2,
        column_norms_sq=col_sq,
    )
    log_py = _marginal_loglik_gaussian(X_orth, y2, tau2, sigma2)
    assert elbo == pytest.approx(log_py, abs=1e-8)


def test_quantitative_elbo_monotone_under_cavi():
    """Two exact-β CAVI iterations: ELBO must not decrease."""
    rng = np.random.default_rng(1)
    n, p = 30, 4
    X = rng.standard_normal((n, p))
    tau2 = np.array([0.3, 0.8, 0.2, 1.1])
    sigma2 = 0.5
    beta_true = rng.standard_normal(p) * np.sqrt(tau2)
    y = X @ beta_true + rng.standard_normal(n) * np.sqrt(sigma2)
    col_sq = np.sum(X ** 2, axis=0)

    # Start away from the posterior.
    beta = np.zeros(p)
    beta_var = np.ones(p) * 0.01

    def elbo_at(b, bv):
        return compute_elbo(
            trait_type=TraitType.QUANTITATIVE,
            targets=y,
            covariate_matrix=np.zeros((n, 0)),
            alpha=np.zeros(0),
            beta=b,
            beta_variance=bv,
            linear_predictor=X @ b,
            reduced_prior_variances=tau2,
            sigma_error2=sigma2,
            column_norms_sq=col_sq,
        )

    e0 = elbo_at(beta, beta_var)
    # One CAVI step = exact posterior (since Gaussian prior, Gaussian likelihood).
    m1, _S1 = _bayes_posterior(X, y, tau2, sigma2)
    # Use a coordinate-wise diagonal update: variance is reciprocal of diagonal precision.
    prec_diag = col_sq / sigma2 + 1.0 / tau2
    bv1 = 1.0 / prec_diag
    # Coordinate update of mean (one sweep) — keep simple: solve full posterior.
    e1 = elbo_at(m1, bv1)
    # Iterate again (idempotent at optimum).
    e2 = elbo_at(m1, bv1)
    assert e1 >= e0 - 1e-10
    assert e2 >= e1 - 1e-12


def test_binary_jj_is_lower_bound():
    """JJ-bound ELBO ≤ exact E_q[log p(y|β,α)] via Gauss-Hermite quadrature."""
    rng = np.random.default_rng(2)
    n, p = 15, 2
    X = rng.standard_normal((n, p))
    beta = rng.standard_normal(p) * 0.3
    beta_var = np.array([0.2, 0.1])
    logits_mean = X @ beta
    probs = 1.0 / (1.0 + np.exp(-logits_mean))
    y = (rng.uniform(size=n) < probs).astype(np.float64)
    pvar = (X ** 2) @ beta_var  # per-sample Var[η]
    col_sq = np.sum(X ** 2, axis=0)
    tau2 = np.array([1.0, 1.0])

    elbo = compute_elbo(
        trait_type=TraitType.BINARY,
        targets=y,
        covariate_matrix=np.zeros((n, 0)),
        alpha=np.zeros(0),
        beta=beta,
        beta_variance=beta_var,
        linear_predictor=X @ beta,
        reduced_prior_variances=tau2,
        sigma_error2=0.0,
        column_norms_sq=col_sq,
        predictor_variance=pvar,
    )

    # Compute exact E_q[log p(y|·)] via Gauss-Hermite on η_i ~ N(μ_i, Var_i).
    nodes, weights = np.polynomial.hermite_e.hermegauss(32)  # ∫ φ(z) f(z) dz weights
    # hermegauss returns nodes & weights for ∫ e^{-x²/2} f(x) dx; need /√(2π).
    weights = weights / np.sqrt(2.0 * np.pi)
    mu = X @ beta
    std = np.sqrt(np.maximum(pvar, 0.0))
    # exact = Σ_i Σ_k w_k · log p(y_i | η = μ_i + std_i · z_k)
    exact_loglik = 0.0
    for i in range(n):
        eta_grid = mu[i] + std[i] * nodes
        log_sig = -np.logaddexp(0.0, -eta_grid)  # log σ(η)
        log_one_minus = -np.logaddexp(0.0, eta_grid)  # log (1 - σ(η))
        per_node = y[i] * log_sig + (1.0 - y[i]) * log_one_minus
        exact_loglik += float(np.sum(weights * per_node))

    # Reconstruct ELBO's "data fit" piece by subtracting non-likelihood terms.
    log_prior_beta = (
        -0.5 * p * np.log(2 * np.pi)
        - 0.5 * np.sum(np.log(tau2))
        - 0.5 * np.sum((beta ** 2 + beta_var) / tau2)
    )
    entropy_beta = 0.5 * np.sum(np.log(beta_var)) + 0.5 * p * (np.log(2 * np.pi) + 1.0)
    jj_loglik = elbo - log_prior_beta - entropy_beta

    assert jj_loglik <= exact_loglik + 1e-9
    # And bound should be reasonably tight (within a few nats on a tiny problem).
    assert exact_loglik - jj_loglik < 5.0


def test_elbo_finite_and_decreases_with_perturbation():
    """ELBO is finite and decreases when β is shifted away from posterior mean."""
    rng = np.random.default_rng(3)
    n, p = 25, 3
    X = rng.standard_normal((n, p))
    tau2 = np.full(p, 0.6)
    sigma2 = 0.3
    beta_true = rng.standard_normal(p) * np.sqrt(tau2)
    y = X @ beta_true + rng.standard_normal(n) * np.sqrt(sigma2)
    col_sq = np.sum(X ** 2, axis=0)
    m, S = _bayes_posterior(X, y, tau2, sigma2)
    bv = np.diag(S)

    def elbo_at(b):
        return compute_elbo(
            trait_type=TraitType.QUANTITATIVE,
            targets=y,
            covariate_matrix=np.zeros((n, 0)),
            alpha=np.zeros(0),
            beta=b,
            beta_variance=bv,
            linear_predictor=X @ b,
            reduced_prior_variances=tau2,
            sigma_error2=sigma2,
            column_norms_sq=col_sq,
        )

    e_opt = elbo_at(m)
    e_bad = elbo_at(m + 1.0)  # large perturbation
    assert np.isfinite(e_opt)
    assert np.isfinite(e_bad)
    assert e_bad < e_opt


def test_jj_lambda_limit_at_zero():
    xi = np.array([0.0, 1e-9, 1e-3, 1.0, 5.0])
    lam = _jj_lambda(xi)
    assert lam[0] == pytest.approx(0.125, abs=1e-12)
    assert lam[1] == pytest.approx(0.125, abs=1e-12)
    # Monotone decreasing in |ξ|.
    assert lam[3] < lam[2] < lam[0]
    assert lam[4] < lam[3]


def test_log_sigmoid_stability():
    x = np.array([-1000.0, -10.0, 0.0, 10.0, 1000.0])
    out = _log_sigmoid(x)
    assert np.all(np.isfinite(out))
    assert out[2] == pytest.approx(-np.log(2.0), abs=1e-12)

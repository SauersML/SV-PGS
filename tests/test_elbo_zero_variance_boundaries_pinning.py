"""Pin: ``compute_elbo`` with all-zero ``beta_variance`` does not NaN.

A point estimate of β is the limit ``Var_q[β] → 0`` — the Gaussian
entropy term ``0.5 Σ log(Var_q[β_j])`` diverges to ``-inf`` in the limit,
but the implementation floors ``beta_var`` at ``1e-300`` so the result
stays finite (a deterministic very-negative entropy contribution).  This
pins that the function:

* returns a finite float (no NaN),
* tolerates ``predictor_variance=0`` for binary traits (the JJ bound at
  ξ² = μ² + 0 = μ² is well-defined),
* tolerates a near-zero ``tau²`` (prior variance) in tandem.
"""
from __future__ import annotations

import numpy as np

from sv_pgs.config import TraitType
from sv_pgs.elbo import compute_elbo


def test_quantitative_elbo_with_zero_beta_variance_is_finite():
    n, p = 8, 3
    rng = np.random.default_rng(0)
    targets = rng.normal(size=n).astype(np.float64)
    cov = rng.normal(size=(n, 2)).astype(np.float64)
    alpha = np.zeros(2, dtype=np.float64)
    beta = np.zeros(p, dtype=np.float64)
    beta_variance = np.zeros(p, dtype=np.float64)  # Dirac
    eta = cov @ alpha  # design contribution is zero
    tau2 = np.full(p, 1.0, dtype=np.float64)
    elbo = compute_elbo(
        trait_type=TraitType.QUANTITATIVE,
        targets=targets,
        covariate_matrix=cov,
        alpha=alpha,
        beta=beta,
        beta_variance=beta_variance,
        linear_predictor=eta,
        reduced_prior_variances=tau2,
        sigma_error2=1.0,
    )
    assert np.isfinite(elbo), elbo


def test_binary_elbo_with_zero_beta_variance_is_finite():
    n, p = 8, 3
    rng = np.random.default_rng(1)
    targets = (rng.uniform(size=n) > 0.5).astype(np.float64)
    cov = rng.normal(size=(n, 1)).astype(np.float64)
    alpha = np.zeros(1, dtype=np.float64)
    beta = np.zeros(p, dtype=np.float64)
    beta_variance = np.zeros(p, dtype=np.float64)  # Dirac
    eta = cov @ alpha
    pvar = np.zeros(n, dtype=np.float64)  # zero predictor variance
    tau2 = np.full(p, 0.1, dtype=np.float64)
    elbo = compute_elbo(
        trait_type=TraitType.BINARY,
        targets=targets,
        covariate_matrix=cov,
        alpha=alpha,
        beta=beta,
        beta_variance=beta_variance,
        linear_predictor=eta,
        reduced_prior_variances=tau2,
        sigma_error2=0.0,
        predictor_variance=pvar,
    )
    assert np.isfinite(elbo), elbo


def test_quantitative_elbo_with_zero_prior_variance_is_finite():
    """``tau2 = 0`` is the degenerate "infinitely tight prior" — floored
    internally so the result must be finite (very-negative ELBO ok)."""
    n, p = 4, 2
    targets = np.asarray([1.0, -1.0, 0.5, -0.5], dtype=np.float64)
    cov = np.zeros((n, 0), dtype=np.float64)
    elbo = compute_elbo(
        trait_type=TraitType.QUANTITATIVE,
        targets=targets,
        covariate_matrix=cov,
        alpha=np.zeros(0, dtype=np.float64),
        beta=np.zeros(p, dtype=np.float64),
        beta_variance=np.ones(p, dtype=np.float64),
        linear_predictor=np.zeros(n, dtype=np.float64),
        reduced_prior_variances=np.zeros(p, dtype=np.float64),
        sigma_error2=1.0,
    )
    assert np.isfinite(elbo), elbo

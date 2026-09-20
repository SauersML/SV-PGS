"""Closed-form (replica / VAMP state-evolution) predictions of out-of-sample prediction accuracy.

Model: y = X beta + e with standardized training genotypes X (n x p), e ~ N(0, noise), and beta_j i.i.d. from a
Gaussian scale mixture sum_k w_k N(0, v_k) (v_k may be 0: a point mass at zero). Derivation in
/Users/user/svpgs-team/closed_form/REPLICA.md. Three pieces:

- `scalar_mmse`: the Bayes-optimal scalar denoiser's error on r = beta + N(0, 1/gamma).
- `fixed_point`: the replica-symmetric fixed point for right-rotationally-invariant designs (VAMP state evolution),
  which depends on the design only through the nonzero eigenvalues of X^T X. For a Gaussian prior it is exact
  (gamma2 = 1/v, ridge).
- `excess_risk`: tr(Sigma_test C2), the test population's excess prediction risk under the fixed point's Gaussian
  posterior C2 = (X^T X / noise + gamma2 I)^{-1}, computed in n x n form (Woodbury).

`predicted_r2` combines them into the Bayes-optimal expected squared correlation in the test population. It uses
Nishimori's identity (cov(yhat, y) = var(yhat) for the posterior mean).
`expected_sample_r2` is the exact finite-sample expectation of a squared sample correlation for bivariate normal
data, so that predictions can be compared with realized per-group r^2.
"""
import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy import special

# Gauss-Hermite order for the scalar channel. Each mixture component is integrated under its own marginal scale, so
# the integrand (a ratio of Gaussian mixtures times r^2) is smooth. The order is checked for convergence in the tests
# by doubling it.
QUADRATURE_ORDER = 121


def _posterior_moments(r, gamma, variances, weights):
    """Posterior mean and variance of beta given r = beta + N(0, 1/gamma) under the scale mixture."""
    noise = 1.0 / gamma
    total = variances[None, :] + noise
    log_likelihood = np.log(weights)[None, :] - 0.5 * np.log(2 * np.pi * total) - 0.5 * r[:, None] ** 2 / total
    log_likelihood -= log_likelihood.max(axis=1, keepdims=True)
    responsibility = np.exp(log_likelihood)
    responsibility /= responsibility.sum(axis=1, keepdims=True)
    shrink = variances / total
    component_mean = shrink[None, :] * r[:, None]
    component_variance = variances * noise / total
    mean = np.sum(responsibility * component_mean, axis=1)
    second = np.sum(responsibility * (component_variance[None, :] + component_mean ** 2), axis=1)
    return mean, second - mean ** 2


def scalar_mmse(gamma, variances, weights, order=QUADRATURE_ORDER):
    """E[(beta - E[beta | r])^2] for r = beta + N(0, 1/gamma), beta ~ sum_k w_k N(0, v_k)."""
    variances, weights = np.asarray(variances, dtype=np.float64), np.asarray(weights, dtype=np.float64)
    nodes, node_weights = hermegauss(order)
    node_weights = node_weights / node_weights.sum()
    second_moment = float(np.sum(weights * variances))
    explained = 0.0
    for variance, weight in zip(variances, weights):
        r = np.sqrt(variance + 1.0 / gamma) * nodes
        mean, _ = _posterior_moments(r, gamma, variances, weights)
        explained += weight * float(np.sum(node_weights * mean ** 2))
    return max(second_moment - explained, 0.0)


def lmmse_error(gamma2, eigenvalues, dimension, noise):
    """(1/p) tr (X^T X / noise + gamma2 I)^{-1} from the nonzero eigenvalues of X^T X."""
    scaled = np.asarray(eigenvalues, dtype=np.float64) / noise
    return (np.sum(1.0 / (scaled + gamma2)) + (dimension - scaled.size) / gamma2) / dimension


def fixed_point(eigenvalues, dimension, noise, variances, weights, damping=None):
    """Replica-symmetric (VAMP state-evolution) Bayes-optimal fixed point.

    Returns (gamma2, mmse): the extrinsic prior precision reaching the linear stage, and the per-coordinate MMSE.
    The iteration is the matched VAMP state evolution. Convergence is declared when the two stages' errors agree to
    the float64 resolution of their mean, which is exactly the fixed-point condition E1 = E2.
    """
    variances, weights = np.asarray(variances, dtype=np.float64), np.asarray(weights, dtype=np.float64)
    second_moment = float(np.sum(weights * variances))
    gamma1 = 1.0 / second_moment
    tolerance = np.sqrt(np.finfo(np.float64).eps)
    previous = None
    for _ in range(100_000):
        error1 = scalar_mmse(gamma1, variances, weights)
        eta1 = 1.0 / error1
        gamma2 = max(eta1 - gamma1, np.finfo(np.float64).tiny)
        error2 = lmmse_error(gamma2, eigenvalues, dimension, noise)
        eta2 = 1.0 / error2
        proposal = max(eta2 - gamma2, np.finfo(np.float64).tiny)
        step = proposal if damping is None else np.exp((1 - damping) * np.log(proposal) + damping * np.log(gamma1))
        if previous is not None and abs(error1 - error2) <= tolerance * 0.5 * (error1 + error2):
            return gamma2, 0.5 * (error1 + error2)
        previous = gamma1
        gamma1 = step
    raise RuntimeError("state evolution did not converge")


def excess_risk(train, test, noise, gamma2):
    """tr(Sigma_test (X^T X / noise + gamma2 I)^{-1}) with Sigma_test = X_t^T X_t / n_t, in n x n form."""
    train, test = np.asarray(train, dtype=np.float64), np.asarray(test, dtype=np.float64)
    count, test_count = train.shape[0], test.shape[0]
    kernel = train @ train.T
    cross = train @ test.T
    solved = np.linalg.solve(noise * gamma2 * np.eye(count) + kernel, cross)
    trace_sigma = float(np.sum(test * test)) / test_count
    return (trace_sigma - float(np.sum(cross * solved)) / test_count) / gamma2, trace_sigma


def predicted_r2(train, test, heritability, variances, weights):
    """Bayes-optimal expected population r^2 in the test sample's genotype distribution.

    `variances`/`weights` give the mixing law of beta_j in units of the per-variant second moment. They are rescaled
    so that p * E[beta^2] = heritability (phenotype variance 1), and noise = 1 - heritability.
    """
    dimension = train.shape[1]
    variances = np.asarray(variances, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    scale = heritability / dimension / float(np.sum(weights * variances))
    variances = variances * scale
    noise = 1.0 - heritability
    eigenvalues = np.linalg.eigvalsh(np.asarray(train, dtype=np.float64) @ np.asarray(train, dtype=np.float64).T)
    eigenvalues = eigenvalues[eigenvalues > eigenvalues.max() * dimension * np.finfo(np.float64).eps]
    gamma2, mmse = fixed_point(eigenvalues, dimension, noise, variances, weights)
    risk, trace_sigma = excess_risk(train, test, noise, gamma2)
    genetic = float(np.sum(weights * variances)) * trace_sigma
    return max(genetic - risk, 0.0) / (genetic + noise), {"gamma2": gamma2, "mmse": mmse, "excess_risk": risk, "genetic_variance": genetic}


def expected_sample_r2(rho2, count):
    """Exact E[r^2] of a sample correlation from `count` bivariate-normal pairs with population rho^2."""
    return 1.0 - (count - 2.0) / (count - 1.0) * (1.0 - rho2) * special.hyp2f1(1.0, 1.0, (count + 1.0) / 2.0, rho2)


def polygenic_r2(count, heritability, effective_segments):
    """Daetwyler et al. (2008) r^2 = h^2 s / (1 + s), s = n h^2 / M_e.

    For n << M_e, `predicted_r2` with a Gaussian prior reduces to h^4 n / M_e when X X^T ~ p I and
    X Sigma X^T ~ (p^2 / M_e) I (REPLICA.md section 3), which is this formula's small-s limit.
    """
    signal = count * heritability / effective_segments
    return heritability * signal / (1.0 + signal)


def detection_variance(dimension, count, causal=1):
    """Smallest variance share of one of k equal effects that a Bayes-optimal sparse predictor recovers.

    The all-or-nothing threshold for k-sparse regression with k / p -> 0 (Reeves, Xu & Zadik 2019): recovery once
    n log(1 / (1 - v)) >= 2 log(p / k) for each effect's share v. Below it, the MMSE equals the prior variance.
    """
    return 1.0 - np.exp(-2.0 * np.log(dimension / causal) / count)

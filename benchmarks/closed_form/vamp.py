"""VAMP (Rangan, Schniter & Fletcher 2019) for y = X beta + e with a Gaussian scale-mixture prior.

This is the algorithm whose fixed point `replica.fixed_point` predicts. It runs on a concrete design so that the
prediction can be checked on real LD, where right-rotational invariance holds only approximately. It uses the SVD of
X, so the linear stage is exact.
"""
import numpy as np

from benchmarks.closed_form.replica import _posterior_moments


def vamp(design, response, noise, variances, weights, damping, iterations):
    """Posterior-mean estimate of beta. Returns (beta_hat, gamma2) at the last iteration."""
    design = np.asarray(design, dtype=np.float64)
    count, dimension = design.shape
    left, singular, right_t = np.linalg.svd(design, full_matrices=False)
    scaled = singular ** 2 / noise
    projected = (right_t @ (design.T @ response)) / noise
    second_moment = float(np.sum(weights * variances))
    gamma1, r1 = 1.0 / second_moment, np.zeros(dimension)
    estimate = np.zeros(dimension)
    for _ in range(iterations):
        mean, variance = _posterior_moments(r1, gamma1, variances, weights)
        alpha1 = max(gamma1 * float(np.mean(variance)), np.finfo(np.float64).tiny)
        eta1 = gamma1 / alpha1
        gamma2 = max(eta1 - gamma1, np.finfo(np.float64).tiny)
        r2 = (eta1 * mean - gamma1 * r1) / gamma2
        # Linear stage: (X^T X / noise + gamma2 I)^{-1} (X^T y / noise + gamma2 r2), split on span(V) and its complement.
        in_span = right_t @ r2
        x2 = r2 + right_t.T @ ((projected + gamma2 * in_span) / (scaled + gamma2) - in_span)
        alpha2 = gamma2 * (np.sum(1.0 / (scaled + gamma2)) + (dimension - scaled.size) / gamma2) / dimension
        eta2 = gamma2 / alpha2
        proposal = max(eta2 - gamma2, np.finfo(np.float64).tiny)
        r1_new = (eta2 * x2 - gamma2 * r2) / proposal
        gamma1 = np.exp(damping * np.log(gamma1) + (1 - damping) * np.log(proposal))
        r1 = damping * r1 + (1 - damping) * r1_new
        estimate = x2
    return estimate, gamma2

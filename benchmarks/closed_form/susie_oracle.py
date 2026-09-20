"""Oracle SuSiE (Wang et al. 2020, IBSS) with the true number of effects, slab variance and noise.

A near-Bayes-optimal reference for k-sparse architectures on a real design, used to tell a wrong replica prediction
apart from an estimator that falls short of the Bayes limit. Columns must be standardized so x_j^T x_j = n.
"""
import numpy as np


def single_effect(design, residual, slab, noise):
    """Posterior (alpha, mean, second moment) of one effect with a uniform prior over columns and N(0, slab)."""
    count = design.shape[0]
    estimate = design.T @ residual / count
    sampling = noise / count
    log_factor = 0.5 * np.log(sampling / (sampling + slab)) + 0.5 * estimate ** 2 / sampling * slab / (sampling + slab)
    alpha = np.exp(log_factor - log_factor.max())
    alpha /= alpha.sum()
    mean = slab / (slab + sampling) * estimate
    return alpha, mean, mean ** 2 + slab * sampling / (slab + sampling)


def susie(design, response, effects, slab, noise):
    """Posterior mean of beta from IBSS, iterated until the fitted values stop changing at float64 resolution."""
    design = np.asarray(design, dtype=np.float64)
    count, dimension = design.shape
    coefficients = np.zeros((effects, dimension))
    fitted = np.zeros(count)
    tolerance = np.sqrt(np.finfo(np.float64).eps)
    for _ in range(100_000):
        previous = fitted.copy()
        for effect in range(effects):
            own = design @ coefficients[effect]
            alpha, mean, _ = single_effect(design, response - fitted + own, slab, noise)
            coefficients[effect] = alpha * mean
            fitted += design @ coefficients[effect] - own
        if np.linalg.norm(fitted - previous) <= tolerance * max(np.linalg.norm(fitted), np.finfo(np.float64).tiny):
            return coefficients.sum(axis=0)
    raise RuntimeError("IBSS did not converge")

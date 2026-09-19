"""Cross-trait pooling of the prior's annotation coefficients.

Each trait's fit learns the coefficients theta_t that map the annotation design
(SV context, length, repeat status, external evidence; continuous annotations
as spline bases) to log prior variance. One trait identifies them weakly; the
traits of a fold share one genome, so the map is pooled hierarchically, still
one model, with every level learned by empirical Bayes:

    theta_t ~ N(theta_bar, Omega),  Omega = diag(omega^2),
    theta_bar ~ N(0, (sum_k nu_k S_k)^+)   (smoothness / ridge penalties on the pooled map).

Omega -> 0 is full pooling and Omega -> infinity gives back the per-trait fits.
S_k are fixed by the basis (a spline's integrated squared second derivative, or
a coefficient's identity row); the weights nu_k are learned. With no penalties
theta_bar is flat and the objective is ordinary REML.

Each trait enters through its current MAP coefficients m_t under the prior and
the data-only information H_t of theta_t there (the local quadratic
approximation of its objective). With S_t = (H_t + Omega^-1)^-1,
W_t = (Omega + H_t^-1)^-1 = Omega^-1 - Omega^-1 S_t Omega^-1, P = sum_k nu_k S_k
and A = sum_t W_t + P, one step is exact for the quadratic model:

    theta_bar' = theta_bar + A^-1 [Omega^-1 sum_t (m_t - theta_bar) - P theta_bar],
    m_t'       = m_t + S_t Omega^-1 (theta_bar' - theta_bar),
    omega_f^2 <- sum_t (m'_tf - theta_bar'_f)^2 / (sum_t gamma_tf - omega_f^2 [sum_t W_t A^-1 W_t]_ff),
    gamma_tf   = 1 - (S_t)_ff / omega_f^2,
    nu_k      <- (rank S_k - nu_k tr(A^-1 S_k)) / (theta_bar'^T S_k theta_bar'),

the variance and weight lines being the marginal-likelihood score equations in
fixed-point (Fellner-Schall / MacKay) form. The caller refits the traits under
the new prior N(theta_bar', Omega') and repeats until nothing moves.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from sv_pgs._typing import F64Array, NDArray

# Numerical guards only: a variance may shrink by at most this factor per step
# (full pooling is approached, never divided by), and a weight's quadratic form
# is floored at this fraction of its numerator (an exactly flat direction).
_VARIANCE_SHRINK_LIMIT = 1e-6
_QUADRATIC_FLOOR = 1e-12


@dataclass(frozen=True)
class PooledHyperprior:
    """One pooling step: the new hyperprior and each trait's MAP moved to its new mean."""

    mean: F64Array
    variance: F64Array
    penalty_weights: F64Array
    mean_covariance: F64Array
    shifted_estimates: tuple[F64Array, ...]


def pooled_hyperprior_step(
    estimates: Sequence[NDArray],
    informations: Sequence[NDArray],
    mean: NDArray,
    variance: NDArray,
    penalties: Sequence[NDArray],
    penalty_weights: NDArray,
) -> PooledHyperprior:
    """One exact marginal-likelihood fixed-point step for the cross-trait hyperprior.

    ``estimates`` are the traits' MAP coefficients under N(mean, diag(variance));
    ``informations`` their data-only information matrices at those points;
    ``penalties`` the fixed positive semi-definite penalty matrices S_k on the
    pooled map and ``penalty_weights`` their current weights nu_k.
    """
    if len(estimates) != len(informations) or not estimates:
        raise ValueError("pooling needs one estimate and one information matrix per trait.")
    if len(penalties) != len(penalty_weights):
        raise ValueError("pooling needs one weight per penalty.")
    mean = np.asarray(mean, dtype=np.float64)
    variance = np.asarray(variance, dtype=np.float64)
    weights = np.asarray(penalty_weights, dtype=np.float64)
    dimension = mean.shape[0]
    penalty_matrices = [np.asarray(penalty, dtype=np.float64) for penalty in penalties]
    penalty = np.zeros((dimension, dimension))
    for weight, matrix in zip(weights, penalty_matrices):
        penalty += weight * matrix
    inverse_variance = 1.0 / variance
    posterior_covariances, marginal_precisions = [], []
    for information in informations:
        covariance = np.linalg.inv(np.asarray(information, dtype=np.float64) + np.diag(inverse_variance))
        covariance = 0.5 * (covariance + covariance.T)
        posterior_covariances.append(covariance)
        marginal_precisions.append(np.diag(inverse_variance) - inverse_variance[:, None] * covariance * inverse_variance[None, :])
    combined_inverse = np.linalg.inv(np.sum(marginal_precisions, axis=0) + penalty)
    combined_inverse = 0.5 * (combined_inverse + combined_inverse.T)
    deviation_sum = np.sum([np.asarray(estimate, dtype=np.float64) - mean for estimate in estimates], axis=0)
    new_mean = mean + combined_inverse @ (inverse_variance * deviation_sum - penalty @ mean)
    shifted = tuple(
        np.asarray(estimate, dtype=np.float64) + covariance @ (inverse_variance * (new_mean - mean))
        for estimate, covariance in zip(estimates, posterior_covariances)
    )
    numerator = np.sum([np.square(estimate - new_mean) for estimate in shifted], axis=0)
    effective = np.sum([1.0 - np.diag(covariance) / variance for covariance in posterior_covariances], axis=0)
    correction = variance * np.sum(
        [np.diag(precision @ combined_inverse @ precision) for precision in marginal_precisions], axis=0
    )
    denominator = effective - correction
    if np.any(denominator <= 0):
        raise ValueError("the marginal-likelihood step's effective trait count is not positive; Omega is unidentified.")
    new_variance = np.maximum(numerator / denominator, _VARIANCE_SHRINK_LIMIT * variance)
    new_weights = np.empty_like(weights)
    for index, (weight, matrix) in enumerate(zip(weights, penalty_matrices)):
        freedom = int(np.linalg.matrix_rank(matrix, hermitian=True)) - weight * float(np.trace(combined_inverse @ matrix))
        if freedom <= 0:
            raise ValueError("a penalty's effective rank is not positive; its weight is unidentified.")
        quadratic = float(new_mean @ matrix @ new_mean)
        new_weights[index] = freedom / max(quadratic, _QUADRATIC_FLOOR * freedom)
    return PooledHyperprior(
        mean=new_mean,
        variance=new_variance,
        penalty_weights=new_weights,
        mean_covariance=combined_inverse,
        shifted_estimates=shifted,
    )

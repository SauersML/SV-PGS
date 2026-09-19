"""Shared pleiotropy indicator across the traits of one fold, as one factor node of the EP-EB model.

Each reduced column j carries one latent indicator g_j ~ Bernoulli(rate_k), where k is the column's variant
class, and g_j is shared by every trait of the fold. When g_j = 1 the prior variance of beta_jt is multiplied by
one shared scale multiplier kappa >= 1 in every trait. With rate -> 0 or kappa = 1 the model is exactly the
independent single-trait fits (the layer then returns weights equal to the rate and changes nothing).

The layer reads only by-products of each trait's EP sweep, so it needs no genotype pass:

* the cavity mean m_jt and variance v_jt of every column (the site's own information removed);
* the prior second-moment scale u_jt = v0_jt * E_tilted[lambda] of the trait's current TPB prior;
* the correlation C of the traits' noise, which couples their cavities when the traits share people.

Under g_j = 0 the cavity vector m_j (one entry per trait) is approximately N(0, U_j + V_j), with U_j = diag(u_j) and
V_j = diag(sqrt v_j) C diag(sqrt v_j); under g_j = 1 it is N(0, kappa U_j + V_j). The evidence for g_j = 1 is the log
ratio of the two densities. Trait t's own site update must not see its own data twice, so the weight it receives
uses the leave-one-trait-out evidence: the ratio of the two conditional-on-others densities of the other traits'
cavities, obtained from the same T x T factorization through log N(m) = log N(m_(-t)) + log N(m_t | m_(-t)).

Callers pass one chunk of columns at a time; every array's column axis is the chunk.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

from sv_pgs._typing import F64Array, I64Array

LOG_TWO_PI = float(np.log(2.0 * np.pi))


@dataclass(frozen=True, slots=True)
class PleiotropyInputs:
    """Per-fold inputs for one chunk of reduced columns."""

    cavity_means: F64Array
    cavity_variances: F64Array
    prior_scale_moments: F64Array
    noise_correlation: F64Array
    class_index: I64Array

    def __post_init__(self) -> None:
        trait_count, column_count = self.cavity_means.shape
        if self.cavity_variances.shape != (trait_count, column_count):
            raise ValueError("cavity_variances must have the shape of cavity_means (traits, columns).")
        if self.prior_scale_moments.shape != (trait_count, column_count):
            raise ValueError("prior_scale_moments must have the shape of cavity_means (traits, columns).")
        if self.noise_correlation.shape != (trait_count, trait_count):
            raise ValueError("noise_correlation must be traits x traits.")
        if self.class_index.shape != (column_count,):
            raise ValueError("class_index needs one entry per column.")
        if np.any(self.cavity_variances <= 0.0) or np.any(self.prior_scale_moments <= 0.0):
            raise ValueError("cavity variances and prior scale moments must be positive.")


@dataclass(frozen=True, slots=True)
class PleiotropyState:
    """Hyperparameters of the layer: one rate per variant class and the shared scale multiplier."""

    rates: F64Array
    scale_multiplier: float


def _covariances(inputs: PleiotropyInputs, scale_multiplier: float) -> tuple[F64Array, F64Array]:
    """(columns, traits, traits) covariances of the cavity vector under g = 0 and under g = 1."""
    noise_sd = np.sqrt(inputs.cavity_variances.T)
    noise = noise_sd[:, :, None] * inputs.noise_correlation[None, :, :] * noise_sd[:, None, :]
    identity = np.eye(inputs.noise_correlation.shape[0])[None, :, :]
    scale = inputs.prior_scale_moments.T[:, :, None]
    return noise + identity * scale, noise + identity * (scale_multiplier * scale)


def _log_densities(covariance: F64Array, values: F64Array) -> tuple[F64Array, F64Array]:
    """Joint log N(values; 0, covariance) per column, and log N(value_t | the other traits' values) per (column, trait)."""
    trait_count = values.shape[1]
    factor = np.linalg.cholesky(covariance)
    log_determinant = 2.0 * np.sum(np.log(np.einsum("jtt->jt", factor)), axis=1)
    precision = np.linalg.inv(covariance)
    precision_times_values = np.einsum("jts,js->jt", precision, values)
    joint = -0.5 * (trait_count * LOG_TWO_PI + log_determinant + np.einsum("jt,jt->j", values, precision_times_values))
    precision_diagonal = np.einsum("jtt->jt", precision)
    conditional = -0.5 * LOG_TWO_PI + 0.5 * np.log(precision_diagonal) - 0.5 * precision_times_values ** 2 / precision_diagonal
    return joint, conditional


def evidence(inputs: PleiotropyInputs, scale_multiplier: float) -> tuple[F64Array, F64Array]:
    """Log evidence for g_j = 1 per column, and its leave-one-trait-out version per (trait, column)."""
    values = inputs.cavity_means.T
    null_covariance, alternative_covariance = _covariances(inputs, scale_multiplier)
    null_joint, null_conditional = _log_densities(null_covariance, values)
    alternative_joint, alternative_conditional = _log_densities(alternative_covariance, values)
    joint_evidence = alternative_joint - null_joint
    leave_one_out = (alternative_joint[:, None] - alternative_conditional) - (null_joint[:, None] - null_conditional)
    return joint_evidence, leave_one_out.T


def prior_weights(inputs: PleiotropyInputs, state: PleiotropyState) -> F64Array:
    """Weight w_jt of the kappa-scaled prior component in trait t's next site update, per (trait, column)."""
    _joint, leave_one_out = evidence(inputs, state.scale_multiplier)
    column_rates = state.rates[inputs.class_index]
    return expit(logit(column_rates)[None, :] + leave_one_out)


def log_marginal_likelihood(inputs: PleiotropyInputs, state: PleiotropyState) -> float:
    """The layer's type-II objective: sum over columns of log[rate e^{E_j} + 1 - rate] (up to a state-free
    constant). update_state never decreases it."""
    joint_evidence, _leave_one_out = evidence(inputs, state.scale_multiplier)
    column_rates = state.rates[inputs.class_index]
    return float(np.sum(np.logaddexp(np.log(column_rates) + joint_evidence, np.log1p(-column_rates))))


def largest_useful_multiplier(inputs: PleiotropyInputs) -> float:
    """Upper end of the kappa search, derived from the data.

    Each column's density N(m_j; 0, kappa U_j + V_j) decreases in kappa once kappa u_jt exceeds m_jt^2 in every
    coordinate, so no weighted sum of column evidences can increase past the largest m_jt^2 / u_jt.
    """
    return float(max(1.0, np.max(inputs.cavity_means ** 2 / inputs.prior_scale_moments)))


def update_state(inputs: PleiotropyInputs, state: PleiotropyState) -> PleiotropyState:
    """One EM step on the layer's hyperparameters, both learned by type-II maximum likelihood.

    Responsibilities r_j = P(g_j = 1 | cavities); each class rate becomes the mean responsibility of its columns;
    kappa maximizes the expected complete-data log likelihood
    sum_j r_j log N(m_j; 0, kappa U_j + V_j) + (1 - r_j) log N(m_j; 0, U_j + V_j) over [1, largest_useful_multiplier].
    log_marginal_likelihood never decreases.
    """
    joint_evidence, _leave_one_out = evidence(inputs, state.scale_multiplier)
    column_rates = state.rates[inputs.class_index]
    responsibilities = expit(logit(column_rates) + joint_evidence)
    class_count = state.rates.size
    responsibility_sums = np.bincount(inputs.class_index, weights=responsibilities, minlength=class_count)
    column_counts = np.bincount(inputs.class_index, minlength=class_count)
    rates = np.where(column_counts > 0, responsibility_sums / np.maximum(column_counts, 1), state.rates)

    def negative_expected(log_multiplier: float) -> float:
        multiplier_evidence, _unused = evidence(inputs, float(np.exp(log_multiplier)))
        return -float(np.dot(responsibilities, multiplier_evidence))

    upper = float(np.log(max(largest_useful_multiplier(inputs), state.scale_multiplier)))
    current = float(np.log(state.scale_multiplier))
    search = minimize_scalar(negative_expected, bounds=(0.0, upper), method="bounded", options={"xatol": 1.0e-6})
    best = float(search.x) if search.fun < negative_expected(current) else current
    return PleiotropyState(rates=rates, scale_multiplier=float(np.exp(best)))

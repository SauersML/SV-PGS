"""Shared pleiotropy scale across the traits of one fold, as one factor node of the EP-EB model.

Each reduced column j carries one latent multiplier s_j > 0 shared by every trait of the fold: trait t's prior
variance of beta_jt is s_j times its own. The multiplier is a continuous quantity, so its prior is a continuous
density g_c on log s for the column's variant class c, learned nonparametrically by empirical Bayes: log g_c is a
cubic B-spline in log s whose exact integrated squared second derivative is penalized, with the penalty weight
learned by the Fellner-Schall update. g_c is carried on equally spaced quadrature nodes over a log-s range derived
from the data (log_multiplier_range). A density concentrated at s = 1 is the independent single-trait fits.

The layer reads only by-products of each trait's EP sweep, so it needs no genotype pass:

* the cavity mean m_jt and variance v_jt of every column (the site's own information removed);
* the prior second-moment scale u_jt = v0_jt * E_tilted[lambda] of the trait's current prior;
* the correlation C of the traits' noise, which couples their cavities when the traits share people.

Given s_j, the cavity vector m_j (one entry per trait) is approximately N(0, s_j U_j + V_j), with U_j = diag(u_j) and
V_j = diag(sqrt v_j) C diag(sqrt v_j). Trait t's own site update must not see its own data twice, so the node
weights it receives come from the other traits only, via log N(m) = log N(m_(-t)) + log N(m_t | m_(-t)) at each node.

The multiplier and the traits' class levels are identified only as a product, so after each update every class
density is recentred to E_g[s] = 1 and the shift is returned for the traits to add to their class level. That
reparametrization leaves the joint likelihood unchanged.

Across chunks of columns the layer is fitted by generalized EM: density_statistics is summed over chunks, then
maximize_coefficients and update_smoothing act on the sums.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.interpolate import BSpline
from scipy.special import logsumexp, softmax

from sv_pgs._typing import F64Array, I64Array

LOG_TWO_PI = float(np.log(2.0 * np.pi))
# Numerical resolution of the quadrature of each class density on log s, and of the spline that carries log g_c.
# Neither is a prior parameter: the density's shape and smoothness are learned, and these only set how finely a
# learned continuous density is represented.
QUADRATURE_NODES = 64
SPLINE_INTERVALS = 12
SPLINE_DEGREE = 3
# Relative change of the spline coefficients at which the inner Newton iteration of maximize_coefficients stops.
COEFFICIENT_TOLERANCE = 1.0e-9


def _spline_basis() -> tuple[F64Array, F64Array]:
    """Cubic B-spline basis at the quadrature nodes (nodes x basis), and the exact Gram of its second derivatives.

    Positions run over [0, 1] in node units. The second derivative of a cubic spline is piecewise linear, so two
    Gauss-Legendre points per knot interval integrate its square exactly.
    """
    interior = np.linspace(0.0, 1.0, SPLINE_INTERVALS + 1)
    knots = np.concatenate([np.zeros(SPLINE_DEGREE), interior, np.ones(SPLINE_DEGREE)])
    basis_count = knots.size - SPLINE_DEGREE - 1
    basis = BSpline.design_matrix(np.linspace(0.0, 1.0, QUADRATURE_NODES), knots, SPLINE_DEGREE).toarray()
    points, weights = np.polynomial.legendre.leggauss(2)
    penalty = np.zeros((basis_count, basis_count))
    for left, right in zip(interior[:-1], interior[1:]):
        abscissae = 0.5 * (left + right) + 0.5 * (right - left) * points
        second = np.column_stack([BSpline(knots, np.eye(basis_count)[index], SPLINE_DEGREE).derivative(2)(abscissae)
                                  for index in range(basis_count)])
        penalty += second.T @ (0.5 * (right - left) * weights[:, None] * second)
    return basis, penalty


BASIS, PENALTY = _spline_basis()
PENALTY_RANK = int(np.linalg.matrix_rank(PENALTY))


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
class SharedScaleDensity:
    """Per-class density of log s: quadrature nodes, spline coefficients of log g_c, and the learned penalty weight."""

    log_multipliers: F64Array
    coefficients: F64Array
    smoothing: F64Array

    def log_weights(self) -> F64Array:
        """log of the normalized node weights of every class density, (classes, nodes)."""
        logits = self.coefficients @ BASIS.T
        return logits - logsumexp(logits, axis=1, keepdims=True)


@dataclass(frozen=True, slots=True)
class DensityStatistics:
    """Sufficient statistics of one E-step, summable over chunks of columns."""

    node_counts: F64Array
    score_products: F64Array
    log_likelihood: float

    def __add__(self, other: DensityStatistics) -> DensityStatistics:
        return DensityStatistics(node_counts=self.node_counts + other.node_counts,
                                 score_products=self.score_products + other.score_products,
                                 log_likelihood=self.log_likelihood + other.log_likelihood)


def _noise_covariances(inputs: PleiotropyInputs) -> F64Array:
    noise_sd = np.sqrt(inputs.cavity_variances.T)
    return noise_sd[:, :, None] * inputs.noise_correlation[None, :, :] * noise_sd[:, None, :]


def log_multiplier_range(inputs: PleiotropyInputs) -> tuple[float, float]:
    """Range of log s that carries all of one chunk's information about the multiplier; chunks combine by min / max.

    Upper end: with A = sum_t m_t^2 / u_t and rho the largest eigenvalue of U^(-1/2) V U^(-1/2), the derivative in s
    of log N(m; 0, sU + V) is below (A / s^2 - T / (s + rho)) / 2, negative for s above the larger root of
    T s^2 - A s - A rho. Lower end: log N(m; 0, sU + V) - log N(m; 0, V) is bounded in magnitude by
    s (tr(V^-1 U) + m' V^-1 U V^-1 m) / 2, so below the returned value every column's density is within
    sqrt(machine epsilon) of its s -> 0 limit. Both ends include log s = 0.
    """
    trait_count = inputs.cavity_means.shape[0]
    values = inputs.cavity_means.T
    scales = inputs.prior_scale_moments.T
    noise = _noise_covariances(inputs)
    whitened = noise / np.sqrt(scales[:, :, None] * scales[:, None, :])
    largest = np.linalg.eigvalsh(whitened)[:, -1]
    signal = np.sum(values ** 2 / scales, axis=1)
    upper_roots = (signal + np.sqrt(signal ** 2 + 4.0 * trait_count * signal * largest)) / (2.0 * trait_count)
    noise_inverse = np.linalg.inv(noise)
    whitened_values = np.einsum("jts,js->jt", noise_inverse, values)
    bound = 0.5 * (np.einsum("jtt,jt->j", noise_inverse, scales) + np.sum(scales * whitened_values ** 2, axis=1))
    lower = float(np.log(np.sqrt(np.finfo(np.float64).eps) / np.max(bound)))
    return min(lower, 0.0), max(float(np.log(np.max(upper_roots))), 0.0)


def initial_density(lower: float, upper: float, class_count: int) -> SharedScaleDensity:
    """Every class density concentrated at s = 1 (the independent single-trait fits), on nodes over [lower, upper].

    The starting shape is log g = -(log s)^2 / (2 h^2) with h the node spacing, a quadratic and so exact in the cubic
    spline space; the starting penalty weight is 1. Both only start the iteration, whose fixed point they do not define.
    """
    nodes = np.linspace(lower, upper, QUADRATURE_NODES)
    spacing = nodes[1] - nodes[0]
    target = -0.5 * (nodes / spacing) ** 2
    coefficients = np.linalg.lstsq(BASIS, target - target.max(), rcond=None)[0]
    return SharedScaleDensity(log_multipliers=np.tile(nodes, (class_count, 1)),
                              coefficients=np.tile(coefficients, (class_count, 1)),
                              smoothing=np.ones(class_count))


def _log_densities(covariance: F64Array, values: F64Array) -> tuple[F64Array, F64Array]:
    """Joint log N(values; 0, covariance) per column, and log N(value_t | the other traits' values) per (column, trait)."""
    trait_count = values.shape[1]
    factor = np.linalg.cholesky(covariance)
    log_determinant = 2.0 * np.sum(np.log(np.einsum("jtt->jt", factor)), axis=1)
    precision = np.linalg.inv(covariance)
    precision_times_values = np.einsum("jts,js->jt", precision, values)
    joint = -0.5 * (trait_count * LOG_TWO_PI + log_determinant + np.einsum("jt,jt->j", values, precision_times_values))
    precision_diagonal = np.einsum("jtt->jt", precision)
    conditional = (-0.5 * LOG_TWO_PI + 0.5 * np.log(precision_diagonal)
                   - 0.5 * precision_times_values ** 2 / precision_diagonal)
    return joint, conditional


def node_log_densities(inputs: PleiotropyInputs, density: SharedScaleDensity) -> tuple[F64Array, F64Array]:
    """log N(m_j; 0, s_k U_j + V_j) at every node, (columns, nodes), and log N(m_jt | m_j,-t), (traits, columns, nodes)."""
    values = inputs.cavity_means.T
    noise = _noise_covariances(inputs)
    scales = inputs.prior_scale_moments.T
    identity = np.eye(values.shape[1])[None, :, :]
    multipliers = np.exp(density.log_multipliers[inputs.class_index])
    joint = np.empty((values.shape[0], QUADRATURE_NODES))
    conditional = np.empty((values.shape[1], values.shape[0], QUADRATURE_NODES))
    for node in range(QUADRATURE_NODES):
        covariance = noise + identity * (multipliers[:, node, None] * scales)[:, :, None]
        joint[:, node], node_conditional = _log_densities(covariance, values)
        conditional[:, :, node] = node_conditional.T
    return joint, conditional


def node_posteriors(inputs: PleiotropyInputs, density: SharedScaleDensity) -> F64Array:
    """Posterior weight of every node given all traits' cavities, (columns, nodes)."""
    joint, _conditional = node_log_densities(inputs, density)
    return softmax(density.log_weights()[inputs.class_index] + joint, axis=1)


def prior_weights(inputs: PleiotropyInputs, density: SharedScaleDensity) -> tuple[F64Array, F64Array]:
    """Node weights for trait t's next site update from the other traits only, (traits, columns, nodes), and the
    multiplier s at every (column, node). Trait t's prior on beta_jt is then sum_k w_jtk * prior(s_jk * v_jt)."""
    joint, conditional = node_log_densities(inputs, density)
    others = joint[None, :, :] - conditional
    weights = softmax(density.log_weights()[inputs.class_index][None, :, :] + others, axis=2)
    return weights, np.exp(density.log_multipliers[inputs.class_index])


def density_statistics(inputs: PleiotropyInputs, density: SharedScaleDensity) -> DensityStatistics:
    """Expected node counts per class, the outer products of the per-column scores in the spline coefficients, and
    the chunk's log marginal likelihood sum_j log sum_k g_ck N(m_j; 0, s_k U_j + V_j)."""
    joint, _conditional = node_log_densities(inputs, density)
    class_count = density.coefficients.shape[0]
    log_prior = density.log_weights()[inputs.class_index]
    log_posterior = log_prior + joint
    marginal = logsumexp(log_posterior, axis=1)
    posterior = np.exp(log_posterior - marginal[:, None])
    scores = (posterior - np.exp(log_prior)) @ BASIS
    counts = np.zeros((class_count, QUADRATURE_NODES))
    np.add.at(counts, inputs.class_index, posterior)
    products = np.zeros((class_count, BASIS.shape[1], BASIS.shape[1]))
    for class_value in range(class_count):
        selected = scores[inputs.class_index == class_value]
        products[class_value] = selected.T @ selected
    return DensityStatistics(node_counts=counts, score_products=products, log_likelihood=float(marginal.sum()))


def penalized_log_likelihood(density: SharedScaleDensity, statistics: DensityStatistics) -> float:
    """statistics.log_likelihood (computed at density) minus the smoothness penalties of every class."""
    penalty = np.einsum("cd,de,ce,c->", density.coefficients, PENALTY, density.coefficients, density.smoothing)
    return statistics.log_likelihood - 0.5 * float(penalty)


def _class_objective(coefficients: F64Array, counts: F64Array, smoothing: float) -> float:
    logits = BASIS @ coefficients
    return float(counts @ (logits - logsumexp(logits))) - 0.5 * smoothing * float(coefficients @ PENALTY @ coefficients)


def maximize_coefficients(density: SharedScaleDensity, statistics: DensityStatistics) -> SharedScaleDensity:
    """M-step for the spline coefficients at fixed penalty weights.

    Maximizes sum_k N_ck log g_ck - (nu_c / 2) theta_c' S theta_c by damped Newton. The objective is concave (a
    multinomial log likelihood in theta plus a concave penalty), so the step never decreases the penalized
    marginal likelihood. Constants in theta leave g unchanged, so steps are minimum-norm.
    """
    coefficients = density.coefficients.copy()
    for class_value, counts in enumerate(statistics.node_counts):
        total = counts.sum()
        if total == 0.0:
            continue
        smoothing = float(density.smoothing[class_value])
        theta = coefficients[class_value]
        current = _class_objective(theta, counts, smoothing)
        while True:
            weights = softmax(BASIS @ theta)
            mean_basis = BASIS.T @ weights
            gradient = BASIS.T @ counts - total * mean_basis - smoothing * (PENALTY @ theta)
            information = (total * (BASIS.T @ (weights[:, None] * BASIS) - np.outer(mean_basis, mean_basis))
                           + smoothing * PENALTY)
            step = np.linalg.lstsq(information, gradient, rcond=None)[0]
            length = 1.0
            candidate = theta + step
            candidate_value = _class_objective(candidate, counts, smoothing)
            while candidate_value < current and length > np.finfo(np.float64).eps:
                length *= 0.5
                candidate = theta + length * step
                candidate_value = _class_objective(candidate, counts, smoothing)
            if candidate_value < current:
                break
            change = np.max(np.abs(candidate - theta)) / max(1.0, np.max(np.abs(theta)))
            theta, current = candidate, candidate_value
            if change < COEFFICIENT_TOLERANCE:
                break
        coefficients[class_value] = theta
    return replace(density, coefficients=coefficients)


def update_smoothing(density: SharedScaleDensity, statistics: DensityStatistics) -> SharedScaleDensity:
    """Fellner-Schall update of every class's penalty weight: nu <- (rank S - nu tr(H^+ S)) / (theta' S theta).

    H = I_obs + nu S, with I_obs the outer products of the per-column scores (the observed information of the
    marginal likelihood in theta, not the complete-data one). A class whose data carry no curvature keeps its weight.
    """
    smoothing = density.smoothing.copy()
    for class_value in range(smoothing.size):
        theta = density.coefficients[class_value]
        curvature = float(theta @ PENALTY @ theta)
        hessian = statistics.score_products[class_value] + smoothing[class_value] * PENALTY
        numerator = PENALTY_RANK - smoothing[class_value] * float(np.trace(np.linalg.pinv(hessian) @ PENALTY))
        if curvature > 0.0 and numerator > 0.0:
            smoothing[class_value] = numerator / curvature
    return replace(density, smoothing=smoothing)


def recentred(density: SharedScaleDensity) -> tuple[SharedScaleDensity, F64Array]:
    """Translate each class's nodes so that E_g[s] = 1, and return log E_g[s] per class for the traits to add to
    their class level. The joint likelihood is unchanged."""
    shift = logsumexp(density.log_weights() + density.log_multipliers, axis=1)
    return replace(density, log_multipliers=density.log_multipliers - shift[:, None]), shift

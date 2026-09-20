"""Sample-side EP for binary traits: the logistic likelihood's Gaussian sites, certified.

A binary trait's likelihood is prod_i sigmoid(sign_i eta_i), with sign_i = 2 y_i - 1 and
eta_i = c_i' alpha + x_i' beta: the covariates alpha under a flat prior, the effects beta under
the variant-side EP of ``scale_mixture_ep``. EP replaces each factor by a Gaussian site in eta_i,

    t_i(eta) = exp(normalizer_i - precision_i eta^2 / 2 + shift_i eta),

so q is the Gaussian regression with per-sample likelihood weight precision_i and working response
shift_i / precision_i. That is exactly ``dual_solve.DualModels`` with those weights, one column per
trait and no per-trait Gram (``working_model``).

Log-concavity. log sigmoid is concave, so each tilted distribution is a log-concave reweighting of
its Gaussian cavity. By Brascamp-Lieb its variance is at most the cavity's, so every site precision
is >= 0, and the split for non-positive sites is never needed on the sample side.

Tilted moments (``tilted_moments``). With the cavity N(mean, variance) and eta = mean + sqrt(variance) z,
the moments are ratios of I_k = E[z^k sigmoid(sign (mean + sqrt(variance) z))], k = 0, 1, 2, which the
trapezoid rule in z evaluates to a relative error of eps with a step and a truncation derived a
priori (``_trapezoid_rule``). The sums are taken against the reference sigmoid(sign mean), so a
cavity far on the wrong side of its label keeps its relative accuracy.

Sample marginals. A cavity needs each sample's marginal of eta under q. ``dense_marginals`` computes
them exactly from the joint precision over (alpha, beta); it is the small-problem path and the
reference. On the full data the same marginals are diag of the n x n dual operator, the sample-side
analogue of ``marginal_variances``, and they belong to the Stage 2 driver.

Evidence. log Z_EP = sum_i normalizer_i + log of the Gaussian integral of the unnormalized sites
exp(-precision_i eta_i^2 / 2 + shift_i eta_i) against the prior. That second term is the weighted
Gaussian evidence the Gaussian path already computes, so ``site_normalizers`` is the binary trait's
whole addition to it.

Separation. With alpha flat, the posterior is improper when the covariates alone (quasi-)separate
the labels (for example a sex indicator on a sex-limited disease), and EP would drift without
bound. ``assert_no_separation`` detects it exactly with a linear program before any fit.

A continuous liability target, where one exists, is a quantitative trait and takes the Gaussian
path instead.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy import linalg
from scipy.optimize import linprog
from scipy.special import log_expit, ndtr

from sv_pgs._typing import F64Array, I64Array

_EPSILON = float(np.finfo(np.float64).eps)
_LOG_SIGMOID_CURVATURE_BOUND = 0.25
"""max over w of -(log sigmoid)''(w) = sigmoid(w) sigmoid(-w), attained at w = 0."""
_LARGEST_EXPONENT = float(np.log(np.finfo(np.float64).max))
_INVERSE_ROOT_TWO_PI = 1.0 / math.sqrt(2.0 * math.pi)


class ImproperCavity(ValueError):
    """A sample's marginal precision under q does not exceed its own site's precision."""


class SeparatedLabels(ValueError):
    """The covariates alone (quasi-)separate the labels, so the flat-prior posterior is improper."""


class NotConverged(RuntimeError):
    """A sweep of sequential EP failed to shrink the largest site change: no contraction."""


@dataclass(frozen=True)
class TiltedMoments:
    """Moments of sigmoid(sign eta) N(eta; cavity), with their certified error.

    ``log_normalizer`` is log of the tilted integral, ``mean`` and ``variance`` those of eta, and
    ``standardized_mean`` and ``standardized_variance`` those of z = (eta - cavity mean) / cavity sd.
    ``relative_error`` bounds |Z_hat - Z| / Z; the standardized moments E[z], E[z^2] carry at most
    ``moment_error`` absolute error each.
    """

    log_normalizer: F64Array
    mean: F64Array
    variance: F64Array
    relative_error: F64Array
    moment_error: F64Array
    standardized_mean: F64Array
    standardized_variance: F64Array

    @property
    def variance_error(self) -> F64Array:
        """A bound on |standardized_variance - Var_t z|.

        Var_t z = E z^2 - (E z)^2. With e = moment_error on each of E z and E z^2,
        |m_hat^2 - m^2| <= e (2 |m_hat| + e), so the moments contribute e (1 + 2 |m_hat| + e). Forming
        it rounds four times (the two ratios, the square and the difference), each by at most eps / 2
        of its result, and the square doubles the rounding of m_hat: at most eps (E z^2 + 2 m_hat^2).
        """
        error = self.moment_error
        square = self.standardized_mean * self.standardized_mean
        second_moment = self.standardized_variance + square
        return error * (1.0 + 2.0 * np.abs(self.standardized_mean) + error) + _EPSILON * (second_moment + 2.0 * square)


@dataclass(frozen=True)
class SampleSites:
    """One trait's sample sites: precision_i >= 0 and shift_i per sample."""

    precision: F64Array
    shift: F64Array


@dataclass(frozen=True)
class DenseFit:
    """A dense sample-side EP fixed point: the sites, q's joint moments and the EP evidence."""

    sites: SampleSites
    mean: F64Array
    covariance: F64Array
    log_evidence: float
    sweeps: int
    largest_change: float


def _signs(labels: F64Array) -> F64Array:
    outcome = np.asarray(labels, dtype=np.float64)
    if not np.all((outcome == 0.0) | (outcome == 1.0)):
        raise ValueError("labels must be 0/1.")
    return 2.0 * outcome - 1.0


def _trapezoid_rule(variance: F64Array) -> tuple[F64Array, I64Array]:
    """Step h and half-width K (nodes k h, |k| <= K) of the trapezoid rule for I_0, I_1, I_2.

    With c = sqrt(variance) and w = sign (mean + c z), the integrand z^k sigmoid(w) phi(z) is analytic
    in |Im z| < pi / c. On |Im z| <= a <= pi / (2 c), |Im w| <= pi / 2, so Re e^-w >= 0 and
    |1 + e^-w|^2 >= 1 + e^(-2 Re w), which gives |sigmoid(w)| <= sqrt(2) sigmoid(Re w). With
    |z^k phi(z)| <= (|x| + a)^k phi(x) e^(a^2 / 2), the line integrals are at most
    sqrt(2) e^(a^2 / 2) E0[(|z| + a)^k sigmoid(Re w)], where E0 is the standard normal expectation.

    Relative to Z = I_0 this is sqrt(2) e^(a^2 / 2) E_t[(|z| + a)^k], with E_t over the tilted law.
    That law is a log-concave reweighting of the standard normal whose potential has slope at most
    c. So |E_t z| <= c (Stein) and Var_t z <= 1 (Brascamp-Lieb), hence E_t[(|z| + a)^k] <=
    (sqrt(1 + c^2) + a)^k <= (sqrt(1 + c^2) + a)^2 for k <= 2. The infinite trapezoid sum then errs
    by at most 2 M / (e^(2 pi a / h) - 1) relative (Trefethen and Weideman 2014, Theorem 5.1), which is
    eps at h = 2 pi a / ln(1 + 2 M / eps). Any a in (0, pi / (2 c)] certifies; the one used nearly
    maximizes the step.

    Truncation. log sigmoid has slope at most 1 and curvature at least -1/4. So
    sigmoid(w) <= sigmoid(sign mean) e^(c |z|), and Z >= sigmoid(sign mean) e^(-c^2 / 8) (Jensen on the
    quadratic lower bound). The nodes beyond |z| = T therefore omit at most
    2 e^(c^2 / 8) integral_T^inf z^k e^(c z) phi(z) dz relative to Z. That is
    2 e^(c^2 / 8 + c^2 / 2) g_k(T - c), with g_0 = Q, g_1 = phi + c Q and g_2 = (T + c) phi + (1 + c^2) Q
    (Q the normal tail). T is the least value at which the k = 2 bound, the largest, is eps.
    """
    spread = np.sqrt(np.asarray(variance, dtype=np.float64))
    growth = 0.5 * spread * spread * (1.0 + _LOG_SIGMOID_CURVATURE_BOUND)
    if np.any(growth >= _LARGEST_EXPONENT):
        raise ValueError("a cavity variance this large leaves no finite tail bound.")
    moment_scale = np.sqrt(1.0 + spread * spread)
    widest = np.sqrt(2.0 * np.log(2.0 * math.sqrt(2.0) * moment_scale * moment_scale / _EPSILON))
    with np.errstate(divide="ignore"):
        strip = np.minimum(np.pi / (2.0 * spread), widest)
    majorant = math.sqrt(2.0) * np.exp(0.5 * strip * strip) * (moment_scale + strip) ** 2
    step = 2.0 * np.pi * strip / np.log1p(2.0 * majorant / _EPSILON)

    def omitted(truncation: F64Array) -> F64Array:
        tail_start = truncation - spread
        tail = ndtr(-tail_start)
        density = _INVERSE_ROOT_TWO_PI * np.exp(-0.5 * tail_start * tail_start)
        second = (truncation + spread) * density + (1.0 + spread * spread) * tail
        return 2.0 * np.exp(growth) * second

    lower = spread.copy()
    upper = spread + 1.0
    while np.any(unfinished := omitted(upper) > _EPSILON):
        upper = np.where(unfinished, spread + 2.0 * (upper - spread), upper)
    while np.any(open_interval := (upper - lower) > _EPSILON * upper):
        middle = 0.5 * (lower + upper)
        enough = omitted(middle) <= _EPSILON
        upper = np.where(open_interval & enough, middle, upper)
        lower = np.where(open_interval & ~enough, middle, lower)
    return step, np.ceil(upper / step).astype(np.int64)


def tilted_moments(cavity_mean: F64Array, cavity_variance: F64Array, labels: F64Array) -> TiltedMoments:
    """The tilted moments of every sample's logistic factor against its Gaussian cavity.

    Entries are processed in order of their node count, so the work is the total node count and the
    memory is linear in the entries.
    """
    mean = np.asarray(cavity_mean, dtype=np.float64)
    variance = np.asarray(cavity_variance, dtype=np.float64)
    sign = _signs(labels)
    if not (mean.shape == variance.shape == sign.shape):
        raise ValueError("cavity_mean, cavity_variance and labels must have the same shape.")
    if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(variance)) and np.all(variance > 0.0)):
        raise ValueError("cavities must have finite means and finite positive variances.")
    step, half_width = _trapezoid_rule(variance.ravel())
    order = np.argsort(half_width, kind="stable")
    sorted_half_width = half_width[order]
    sorted_mean = mean.ravel()[order]
    sorted_sign = sign.ravel()[order]
    sorted_spread = np.sqrt(variance.ravel())[order]
    sorted_step = step[order]
    reference = log_expit(sorted_sign * sorted_mean)
    zeroth = np.ones_like(sorted_mean)
    first = np.zeros_like(sorted_mean)
    second = np.zeros_like(sorted_mean)
    largest_node = int(sorted_half_width[-1]) if sorted_half_width.size else 0
    for node in range(1, largest_node + 1):
        active = slice(int(np.searchsorted(sorted_half_width, node)), None)
        offset = node * sorted_step[active]
        base = -reference[active] - 0.5 * offset * offset
        upper = np.exp(log_expit(sorted_sign[active] * (sorted_mean[active] + sorted_spread[active] * offset)) + base)
        lower = np.exp(log_expit(sorted_sign[active] * (sorted_mean[active] - sorted_spread[active] * offset)) + base)
        zeroth[active] += upper + lower
        first[active] += offset * (upper - lower)
        second[active] += offset * offset * (upper + lower)
    standardized_mean = first / zeroth
    standardized_variance = second / zeroth - standardized_mean * standardized_mean
    node_count = 2.0 * sorted_half_width + 1.0
    rounding = node_count * _EPSILON * (1.0 + sorted_half_width * sorted_step) ** 2
    relative_error = 2.0 * _EPSILON + rounding
    moment_error = (relative_error * (1.0 + np.abs(standardized_mean) + second / zeroth)) / (1.0 - relative_error)
    log_normalizer = np.log(sorted_step) + math.log(_INVERSE_ROOT_TWO_PI) + reference + np.log(zeroth)

    def unsorted(values: F64Array) -> F64Array:
        result = np.empty_like(values)
        result[order] = values
        return result.reshape(mean.shape)

    return TiltedMoments(
        log_normalizer=unsorted(log_normalizer),
        mean=unsorted(sorted_mean + sorted_spread * standardized_mean),
        variance=unsorted(sorted_spread * sorted_spread * standardized_variance),
        relative_error=unsorted(relative_error),
        moment_error=unsorted(moment_error),
        standardized_mean=unsorted(standardized_mean),
        standardized_variance=unsorted(standardized_variance),
    )


def cavities(marginal_mean: F64Array, marginal_variance: F64Array, sites: SampleSites) -> tuple[F64Array, F64Array]:
    """Each sample's cavity from q's marginal of eta and its own site."""
    cavity_precision = 1.0 / np.asarray(marginal_variance, dtype=np.float64) - sites.precision
    if not np.all(cavity_precision > 0.0):
        raise ImproperCavity("a sample's marginal precision under q does not exceed its site's precision.")
    cavity_variance = 1.0 / cavity_precision
    cavity_mean = cavity_variance * (np.asarray(marginal_mean, dtype=np.float64) / marginal_variance - sites.shift)
    return cavity_mean, cavity_variance


def site_update(cavity_mean: F64Array, cavity_variance: F64Array, moments: TiltedMoments) -> SampleSites:
    """The sites whose product with each cavity has the tilted moments.

    precision = (1 / Var_t z - 1) / cavity variance is >= 0 by log-concavity, since Var_t z <= 1. A
    computed Var_t z above one by at most its certified error (``TiltedMoments.variance_error``) is
    projected onto the true value's known sign, precision zero; beyond that it raises.
    """
    if np.any(moments.standardized_variance - 1.0 > moments.variance_error):
        raise ArithmeticError("a tilted variance exceeds its cavity's beyond the certified error: log-concavity is violated.")
    precision = np.maximum(1.0 / moments.standardized_variance - 1.0, 0.0) / cavity_variance
    shift = moments.mean / moments.variance - cavity_mean / cavity_variance
    return SampleSites(precision=precision, shift=shift)


def site_normalizers(cavity_mean: F64Array, cavity_variance: F64Array, sites: SampleSites, moments: TiltedMoments) -> F64Array:
    """normalizer_i with integral t_i(eta) N(eta; cavity_i) d eta = Z_i, in the precision form (no 1 / precision)."""
    scaled = 1.0 + sites.precision * cavity_variance
    completed = (sites.shift * cavity_variance + cavity_mean) ** 2 / (cavity_variance * scaled) - cavity_mean * cavity_mean / cavity_variance
    return moments.log_normalizer + 0.5 * np.log(scaled) - 0.5 * completed


def working_model(sites: SampleSites) -> tuple[F64Array, F64Array]:
    """dual_solve.DualModels' weights and working response for one trait: (precision, shift / precision).

    A zero-precision site is flat. Its root weight is zero, so its working response is never read,
    and it is set to zero. A non-zero shift on a zero-precision site is not a Gaussian and raises.
    """
    flat = sites.precision == 0.0
    if np.any(flat & (sites.shift != 0.0)):
        raise ValueError("a site with zero precision must have zero shift.")
    response = np.divide(sites.shift, sites.precision, out=np.zeros_like(sites.shift), where=~flat)
    return sites.precision.copy(), response


def assert_no_separation(covariates: F64Array, labels: F64Array) -> None:
    """Raise SeparatedLabels if some direction alpha != 0 has sign_i c_i' alpha >= 0 for every sample.

    A linear program maximizes sum_i sign_i c_i' alpha over those alpha in the unit box. The
    objective is homogeneous, so the box only normalizes it. The maximum is zero exactly when no
    (quasi-)separating direction exists; the test allows the program's rounding, n eps ||C||.
    """
    design = np.asarray(covariates, dtype=np.float64)
    signed = _signs(labels)[:, None] * design
    if signed.shape[1] == 0:
        return
    result = linprog(
        c=-signed.sum(axis=0),
        A_ub=-signed,
        b_ub=np.zeros(signed.shape[0]),
        bounds=[(-1.0, 1.0)] * signed.shape[1],
        method="highs",
    )
    if result.status != 0:
        raise ArithmeticError(f"the separation linear program did not solve: {result.message}")
    rounding = signed.shape[0] * _EPSILON * float(np.abs(signed).sum())
    if -result.fun > rounding:
        raise SeparatedLabels(
            f"the covariates (quasi-)separate the labels along direction {result.x.tolist()};"
            " restrict the cohort to the at-risk group or drop the separating covariate."
        )


def _joint_precision(design: F64Array, covariates: F64Array, prior_precision: F64Array, sites: SampleSites) -> tuple[F64Array, F64Array]:
    features = np.hstack([covariates, design])
    precision = features.T @ (sites.precision[:, None] * features)
    covariate_count = covariates.shape[1]
    precision[covariate_count:, covariate_count:] += np.diag(prior_precision)
    return features, precision


def dense_marginals(
    design: F64Array,
    covariates: F64Array,
    prior_precision: F64Array,
    prior_shift: F64Array,
    sites: SampleSites,
) -> tuple[F64Array, F64Array, F64Array, F64Array]:
    """q's joint mean and covariance over (alpha, beta), and every sample's marginal of eta, exactly.

    q is the flat-prior covariates plus the Gaussian effect sites (prior_precision, prior_shift) times
    the sample sites. Its joint precision must be positive definite; if it is not, the Cholesky
    factorization raises.
    """
    features, precision = _joint_precision(design, covariates, prior_precision, sites)
    linear = features.T @ sites.shift
    linear[covariates.shape[1]:] += prior_shift
    factor = linalg.cho_factor(precision, lower=True)
    mean = linalg.cho_solve(factor, linear)
    covariance = linalg.cho_solve(factor, np.eye(precision.shape[0]))
    marginal_mean = features @ mean
    marginal_variance = np.einsum("ij,jk,ik->i", features, covariance, features)
    return mean, covariance, marginal_mean, marginal_variance


def dense_log_evidence(
    design: F64Array,
    covariates: F64Array,
    prior_precision: F64Array,
    prior_shift: F64Array,
    sites: SampleSites,
    normalizers: F64Array,
) -> float:
    """log Z_EP of the dense model, up to the flat covariate prior's constant.

    It is sum_i normalizer_i plus the log Gaussian integral of the unnormalized sites against the
    effect prior, with the flat alpha integrated out: 0.5 (b' Q^-1 b - log |Q| + log |P| - m' P m),
    with Q the joint precision, b its linear term, P = diag(prior_precision) and m = P^-1 prior_shift.
    """
    features, precision = _joint_precision(design, covariates, prior_precision, sites)
    linear = features.T @ sites.shift
    linear[covariates.shape[1]:] += prior_shift
    factor = linalg.cho_factor(precision, lower=True)
    quadratic = float(linear @ linalg.cho_solve(factor, linear))
    log_determinant = 2.0 * float(np.sum(np.log(np.diag(factor[0]))))
    prior_quadratic = float(prior_shift @ (prior_shift / prior_precision))
    prior_log_determinant = float(np.sum(np.log(prior_precision)))
    return float(np.sum(normalizers)) + 0.5 * (quadratic - log_determinant + prior_log_determinant - prior_quadratic)


def dense_fit(
    design: F64Array,
    covariates: F64Array,
    labels: F64Array,
    prior_precision: F64Array,
    prior_shift: F64Array,
    tolerance: float,
) -> DenseFit:
    """Sequential EP on the sample sites of a small binary trait, with exact dense marginals.

    Each site in turn is updated from its cavity, and q follows by a rank-one update. A sweep ends
    when every site has been visited once. The fit stops when a sweep's largest change in any site's
    (precision, shift), relative to the site's own cavity scale, is at most ``tolerance``. It raises
    NotConverged when a sweep fails to shrink that change, since then the updates are not
    contracting. Separation of the covariates is refused first.
    """
    covariate_matrix = np.asarray(covariates, dtype=np.float64)
    features = np.hstack([covariate_matrix, np.asarray(design, dtype=np.float64)])
    outcome = np.asarray(labels, dtype=np.float64)
    assert_no_separation(covariate_matrix, outcome)
    sample_count = features.shape[0]
    covariate_count = covariate_matrix.shape[1]
    # A flat alpha with all-zero sites is improper, so the first visit uses a covariate-only proper
    # start: every site at the logistic curvature at eta = 0, 1/4, with zero shift, the Taylor site.
    sites = SampleSites(precision=np.full(sample_count, _LOG_SIGMOID_CURVATURE_BOUND), shift=np.zeros(sample_count))
    mean, covariance, _, _ = dense_marginals(design, covariate_matrix, prior_precision, prior_shift, sites)
    linear = features.T @ sites.shift
    linear[covariate_count:] += prior_shift
    previous_change = math.inf
    sweeps = 0
    while True:
        largest_change = 0.0
        for sample in range(sample_count):
            feature = features[sample]
            projected = covariance @ feature
            marginal_variance = float(feature @ projected)
            marginal_mean = float(feature @ mean)
            current = SampleSites(precision=sites.precision[sample : sample + 1], shift=sites.shift[sample : sample + 1])
            cavity_mean, cavity_variance = cavities(np.array([marginal_mean]), np.array([marginal_variance]), current)
            moments = tilted_moments(cavity_mean, cavity_variance, outcome[sample : sample + 1])
            updated = site_update(cavity_mean, cavity_variance, moments)
            precision_change = float(updated.precision[0] - sites.precision[sample])
            shift_change = float(updated.shift[0] - sites.shift[sample])
            scale = math.sqrt(float(cavity_variance[0]))
            largest_change = max(largest_change, abs(precision_change) * scale * scale, abs(shift_change) * scale)
            sites.precision[sample] = updated.precision[0]
            sites.shift[sample] = updated.shift[0]
            denominator = 1.0 + precision_change * marginal_variance
            covariance -= np.outer(projected, projected) * (precision_change / denominator)
            linear += shift_change * feature
            mean = covariance @ linear
        sweeps += 1
        if largest_change <= tolerance:
            break
        if largest_change >= previous_change:
            raise NotConverged(f"sweep {sweeps} changed a site by {largest_change:.3e}, no less than the sweep before ({previous_change:.3e}).")
        previous_change = largest_change
    mean, covariance, marginal_mean, marginal_variance = dense_marginals(design, covariate_matrix, prior_precision, prior_shift, sites)
    cavity_mean, cavity_variance = cavities(marginal_mean, marginal_variance, sites)
    moments = tilted_moments(cavity_mean, cavity_variance, outcome)
    normalizers = site_normalizers(cavity_mean, cavity_variance, sites, moments)
    log_evidence = dense_log_evidence(design, covariate_matrix, prior_precision, prior_shift, sites, normalizers)
    return DenseFit(sites=sites, mean=mean, covariance=covariance, log_evidence=log_evidence, sweeps=sweeps, largest_change=largest_change)

"""The measurement model of a quantitative trait's occasions (docs/design/math/novel-pheno.md §3).

Occasion j of person i is a positive reading y_ij in the trait's canonical unit, and

    z_ij = h(y_ij) = d_ij' gamma + T_i + e_ij,   T_i ~ N(0, tau^2),   e_ij ~ sum_k pi_k N(0, s_k).

- h is the Box-Cox transform (y^lambda - 1) / lambda (log y at lambda = 0), whose log Jacobian
  (lambda - 1) log y is part of the likelihood; lambda maximizes the profile evidence (the null space of
  §3.1's smooth transform, whose smooth deviation comes later).
- T_i is the person's long-run level. The target is E[T_i | occasions], with reliability
  1 - Var(T_i | occasions) / tau^2 (§1, the Tweedie identities).
- pi is the lattice form of the noise's continuous mixing density g over t = log s (SPEC 131b205): log g is
  a smooth function of t whose third-difference roughness is penalized with a learned weight, on the
  engine's own one-class prior (``scale_mixture_ep``). A gross error (a typo, a unit confusion) is
  downweighted by the density, not cut at a hand-set range.

The lattice starts at the recording resolution's variance. Readings are rounded to a resolution delta (the
smallest gap between distinct readings, §3.2), so the rounded-data likelihood convolves the noise with the
rounding error, of variance (y^(lambda - 1) delta)^2 / 12 on the transformed scale: no smaller noise variance
is resolvable. The lattice reaches the largest squared within-person deviation, past it by its own width,
and its spacing is halved until its quadrature in t meets the evidence tolerance. PENDING (lead ruling): the
extent past the largest deviation is generous, not derived; math-density's R2 rule (the fitted mass beyond the
last node, mixing_density.md section 2) replaces it.

Inference is exact in T_i, which is one-dimensional (the lead's ruling): EM over the complete data
(T_i, each occasion's lattice component), with the density's step on the exact likelihood.
- E-step: p(T_i | z_i) by the trapezoid rule in T. Each person's step is certified by the Trefethen-Weideman
  bound with the mass-weighted strip modulus, bounded in closed form (its strip width searched per person),
  and the ends by a tail bound (``level_posterior``; math-density, mixing_density.md §11). Duplicated
  readings, whose integrand spikes, need no separate treatment: their modulus bound is large, so their step
  is small. At every node come each occasion's component responsibilities.
- M-steps (ECME, Liu and Rubin 1994): gamma and tau^2 by PX-EM (the levels' prior mean expanded and absorbed
  into the intercept, Liu, Rubin and Wu 1998), then the density's coefficients by Newton on the exact penalized
  marginal likelihood, whose curvature is the observed information by Louis' identity. Where that curvature is
  not negative definite (far from the maximum, or at lambda = 0, where components' masses vanish) EM's own
  updates are accelerated by SQUAREM (Varadhan and Roland 2008). Speculative points whose E-step does not exist
  are rejected, never approximated.
- The penalty weight maximizes the Laplace evidence of the exact marginal likelihood (the engine's B + S
  form: the penalty's log pseudo-determinant, the null space profiled), whose curvature is the observed
  information by Louis' identity. At the ends of its resolvable range it moves to 0 or infinity exactly
  (the engine's edge rule). A maximum that is not certified (no positive definite -H, by Cholesky, or a
  Newton decrement above FIT_TOLERANCE) counts as the lowest evidence.

Tolerances hold numerical error below statistical error. Every stop resolves EVIDENCE_TOLERANCE, 1/2 nat of
total evidence (within one posterior standard deviation of the parameters it settles), and each person's
quadrature has relative error 1 - e^(-EVIDENCE_TOLERANCE / n), so the n persons together stay within it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.linalg
from scipy.special import digamma, log_ndtr, logsumexp, polygamma
from scipy.stats import chi2

from sv_pgs._typing import F64Array, I64Array
from sv_pgs.scale_mixture_ep import (
    ROUGHNESS_ORDER,
    MixtureHyperparameters,
    ScaleMixturePrior,
    class_log_density,
    halved_lattice,
    relattice,
    scale_mixture_prior,
    spacing_bound,
)

# One posterior standard deviation of the parameters, in nats (module docstring).
EVIDENCE_TOLERANCE = 0.5
# Each fit's remaining gain: the searches compare spans of evidence to EVIDENCE_TOLERANCE, which fits short of their
# maxima by at most this much move by at most twice it, half the tolerance.
FIT_TOLERANCE = EVIDENCE_TOLERANCE / 4
_EPSILON = float(np.finfo(np.float64).eps)
_HALF_PRECISION = _EPSILON**0.5
_LOG_TWO_PI = float(np.log(2.0 * np.pi))
_GOLDEN = 0.5 * (3.0 - np.sqrt(5.0))
# The E-step's float64 arrays per (person, node, occasion, component) entry live at once: the log components and,
# at most, the modulus bound's terms with their logsumexp temporary, or the responsibilities and their weighted
# product, or one logsumexp temporary; and one more while a grown grid's components replace the last.
_FLOAT64_BYTES = 8
_ENTRY_BYTES = 4 * _FLOAT64_BYTES
# Rounded operations behind each (node, occasion, component) term of the split information: the product with the
# responsibility and the division by s (R1, R2), a square, two scalings, a subtraction and the accumulation.
_SPLIT_OPERATIONS_PER_TERM = 8


def _logsumexp(values: F64Array, axis: int) -> F64Array:
    """log sum exp over ``axis``, shifted by the largest term (-inf where every term is -inf)."""
    largest = np.max(values, axis=axis, keepdims=True)
    shift = np.where(np.isfinite(largest), largest, 0.0)
    shifted = values - shift
    np.exp(shifted, out=shifted)
    with np.errstate(divide="ignore"):
        return np.squeeze(shift, axis=axis) + np.log(np.sum(shifted, axis=axis))


def box_cox(values: F64Array, exponent: float) -> tuple[F64Array, F64Array]:
    """(h(y), log h'(y)) for positive y."""
    log_values = np.log(values)
    if exponent == 0.0:
        return log_values, -log_values
    return np.expm1(exponent * log_values) / exponent, (exponent - 1.0) * log_values


@dataclass(frozen=True)
class Occasions:
    """Every occasion's person (0..m-1, each present), positive reading and fixed-effect design row."""

    person_index: I64Array
    values: F64Array
    design: F64Array

    def __post_init__(self) -> None:
        if not np.all(self.values > 0.0):
            raise ValueError("readings must be positive: the Box-Cox transform is defined on y > 0")
        if np.any(np.bincount(self.person_index) == 0):
            raise ValueError("every person index below the largest must have an occasion")
        if self.design.shape[0] != self.values.shape[0]:
            raise ValueError("the design needs one row per occasion")
        if np.linalg.matrix_rank(self.design) < self.design.shape[1]:
            raise ValueError("the fixed-effect design must have full column rank")
        if np.unique(self.values).shape[0] < 2:
            raise ValueError("the readings need two distinct values to show their recording resolution")

    @property
    def person_count(self) -> int:
        return int(self.person_index.max()) + 1

    @property
    def resolution(self) -> float:
        """The recording resolution: the smallest gap between distinct readings (§3.2)."""
        return float(np.min(np.diff(np.unique(self.values))))


@dataclass(frozen=True)
class OccasionModelFit:
    """The fitted model and every person's exact level posterior."""

    exponent: float
    fixed_effects: F64Array
    level_variance: float
    prior: ScaleMixturePrior
    hyperparameters: MixtureHyperparameters
    level_mean: F64Array
    level_posterior_variance: F64Array
    log_likelihood: float
    log_evidence: float
    # Louis' observed information for (log tau^2, a), a scaling every noise variance by e^a, the magnitudes of
    # its terms, and how many rounded operations accumulated them (``split_identified``).
    split_information: F64Array
    split_magnitude: F64Array
    split_operations: int

    @property
    def reliability(self) -> F64Array:
        return 1.0 - self.level_posterior_variance / self.level_variance

    @property
    def split_identified(self) -> bool:
        """Whether the data separate the level variance from the noise (the lead's ruling on traits with few
        within-person replicates): the observed information for (log tau^2, a) is positive definite beyond its
        rounding. Each entry, a sum of m rounded terms, errs by at most gamma_m times the sum of their magnitudes
        (Higham 2002, section 4.2), so by Weyl's inequality the matrix is certified positive definite when its
        smallest eigenvalue exceeds the Frobenius norm of those bounds. Without replicates only the sum
        tau^2 + E[s] is identified and the smallest eigenvalue is zero up to rounding."""
        unit_roundoff = 0.5 * _EPSILON
        if self.split_operations * unit_roundoff >= 1.0:
            return False
        gamma = self.split_operations * unit_roundoff / (1.0 - self.split_operations * unit_roundoff)
        if _log_determinant(self.split_information) is None:
            return False
        return bool(np.linalg.eigvalsh(0.5 * (self.split_information + self.split_information.T))[0] > gamma * np.linalg.norm(self.split_magnitude))

    @property
    def noise_second_moment(self) -> float:
        """E[e^2] under the fitted noise density."""
        masses = np.exp(class_log_density(self.prior, self.hyperparameters.coefficients)[0])
        return float(masses @ np.exp(self.prior.log_variance_grid))


# ------------------------------------------------------------------ the exact level posterior


def person_tolerance(person_count: int) -> float:
    """Each person's relative error in L_i: 1 - e^(-tol / n), so the n log-likelihoods together move by at most
    EVIDENCE_TOLERANCE."""
    return float(-np.expm1(-EVIDENCE_TOLERANCE / person_count))


def strip_half_width(level_variance: float, occasion_count: int, occasion_variance: float, relative_tolerance: float) -> float:
    """The strip |Im T| < b maximizing the certified step for J Gaussian occasions of variance s, whose modulus
    bound is exactly M(b) = e^(b^2 c / 2) L with c = 1 / tau^2 + J / s: b = sqrt(2 ln(2 / eps) / c)."""
    curvature = 1.0 / level_variance + occasion_count / occasion_variance
    return float(np.sqrt(2.0 * np.log(2.0 / relative_tolerance) / curvature))


def gaussian_step(level_variance: float, occasion_count: int, occasion_variance: float, relative_tolerance: float) -> float:
    """The trapezoid step in T with relative error at most ``relative_tolerance`` for J Gaussian occasions of
    variance s, and for any J occasions whose variances are all at least s.

    On the strip |Im T| < b every Gaussian factor N(x + i y; mu, v) has modulus N(x; mu, v) e^(y^2 / (2 v)), so
    the integrand's absolute integral along the strip is at most M(b) <= e^(b^2 c / 2) L, with
    c = 1 / tau^2 + J / s. The trapezoid rule's error is at most 2 M(b) / (e^(2 pi b / h) - 1) (Trefethen and
    Weideman 2014, Theorem 5.1), so h = 2 pi b / ln(1 + 2 e^(b^2 c / 2) / eps) at b = ``strip_half_width``.
    """
    curvature = 1.0 / level_variance + occasion_count / occasion_variance
    half_width = strip_half_width(level_variance, occasion_count, occasion_variance, relative_tolerance)
    return float(2.0 * np.pi * half_width / np.log1p(2.0 * np.exp(0.5 * half_width * half_width * curvature) / relative_tolerance))


@dataclass(frozen=True)
class LevelPosterior:
    """The exact E-step for a set of persons: level moments, log-likelihoods, the sufficient statistics of the
    M-step (counts per component, each occasion's E[1/s] and E[T/s]) and of Louis' identity, and each person's
    admissible trapezoid step at these parameters (the start of the next E-step)."""

    log_likelihood: F64Array
    level_mean: F64Array
    level_second_moment: F64Array
    counts: F64Array
    occasion_precision: F64Array
    occasion_shift: F64Array
    missing_information: F64Array
    split_information: F64Array
    split_magnitude: F64Array
    split_operations: int
    admissible_step: F64Array
    grid: LevelGrid


def _log_components(residuals: F64Array, levels: F64Array, log_masses: F64Array, variances: F64Array) -> F64Array:
    """log pi_k N(r_j - T_n; 0, s_k): persons x nodes x occasions x components."""
    deviations = residuals[:, None, :] - levels[:, :, None]
    # One full-size array, filled in place (the E-step's memory accounting counts it once).
    components = np.square(deviations)[..., None] * (-0.5 / variances)
    components += log_masses - 0.5 * (_LOG_TWO_PI + np.log(variances))
    return components


def _log_prior(levels: F64Array, level_variance: float) -> F64Array:
    return -0.5 * (_LOG_TWO_PI + np.log(level_variance)) - 0.5 * np.square(levels) / level_variance


def _log_occasion_density(residuals: F64Array, points: F64Array, log_masses: F64Array, variances: F64Array) -> F64Array:
    """log f(r_j - T_p) at one point T_p per person: persons x occasions."""
    return _logsumexp(_log_components(residuals, points[:, None], log_masses, variances)[:, 0], axis=2)


# The level integrals certified: of F, |T| F and T^2 F (L_i and the two moments' absolute integrals).
def _log_prior_tail_moments(distance: F64Array, level_variance: float) -> F64Array:
    """log of the integral of |T|^m N(T; 0, tau^2) over T > distance, for m = 0, 1, 2 (persons x 3). For
    a = distance / tau >= 0 these are Phi(-a), tau phi(a) and tau^2 (a phi(a) + Phi(-a)); below 0 they are the
    whole moments 1, tau sqrt(2 / pi) and tau^2 less the tails beyond -distance."""
    scale = np.sqrt(level_variance)
    standardized = np.abs(distance) / scale
    log_survival = log_ndtr(-standardized)
    log_density = -0.5 * (_LOG_TWO_PI + np.square(standardized))
    with np.errstate(divide="ignore"):  # a = 0: the first term of the second moment's tail is 0
        second = np.logaddexp(np.log(standardized) + log_density, log_survival)
    beyond = np.stack([log_survival, np.log(scale) + log_density, 2.0 * np.log(scale) + second], axis=1)
    whole = np.array([0.0, np.log(scale) + 0.5 * (np.log(2.0) - np.log(np.pi)), 2.0 * np.log(scale)])
    within = whole + np.log(-np.expm1(beyond - whole))
    return np.where((distance >= 0.0)[:, None], beyond, within)


def _log_prior_sum_bound(distance: F64Array, level_variance: float, steps: F64Array) -> F64Array:
    """log of a bound on h sum over the nodes beyond ``distance`` of |T|^m N(T; 0, tau^2), m = 0, 1, 2.

    On each interval where a function is monotone, its trapezoid terms sum to at most its integral plus h times
    its largest value there. |T|^m N(T; 0, tau^2) has two monotone pieces for m = 0 and four for m > 0 (modes
    at +-sqrt(m) tau), and its largest value past ``distance`` is at max(distance, sqrt(m) tau).
    """
    powers = np.arange(3.0)
    pieces = np.where(powers > 0.0, 4.0, 2.0)
    peak = np.maximum(distance[:, None], np.sqrt(powers * level_variance)[None, :])
    # |T|^0 = 1, also at the m = 0 peak T = 0.
    log_power = powers * np.log(np.where(powers > 0.0, np.abs(peak), 1.0))
    log_peak = log_power - 0.5 * (_LOG_TWO_PI + np.log(level_variance) + np.square(peak) / level_variance)
    return np.logaddexp(_log_prior_tail_moments(distance, level_variance), np.log(pieces * steps[:, None]) + log_peak)


def _log_tail(
    residuals: F64Array, edge: F64Array, level_variance: float, log_masses: F64Array, variances: F64Array, steps: F64Array,
    upper: bool,
) -> F64Array:
    """log of a bound on h sum over the nodes beyond ``edge`` (above it when ``upper``) of |T|^m F(T), m = 0, 1, 2.

    Past the edge each f(r_j - T) is at most f(r_j - edge) when r_j lies on the edge's inner side (it decreases
    away from r_j) and at most f(0) otherwise, and the prior's terms are bounded by ``_log_prior_sum_bound``
    (the prior is symmetric, so the lower tail is the upper one beyond -edge).
    """
    inner = residuals <= edge[:, None] if upper else residuals >= edge[:, None]
    at_edge = _log_occasion_density(residuals, edge, log_masses, variances)
    at_peak = float(logsumexp(log_masses - 0.5 * (_LOG_TWO_PI + np.log(variances))))
    occasions = np.sum(np.where(inner, at_edge, at_peak), axis=1)
    return occasions[:, None] + _log_prior_sum_bound(edge if upper else -edge, level_variance, steps)


def _log_widened_tail(
    residuals: F64Array, edge: F64Array, level_variance: float, log_weights: F64Array, variances: F64Array, half_width: F64Array,
    upper: bool,
) -> F64Array:
    """log of a bound on the integral beyond ``edge`` (above it when ``upper``) of the strip modulus's majorant
    (|x| + b)^m N(x; 0, tau^2) e^(b^2 / (2 tau^2)) prod_j W_j(x), m = 0, 1, 2 (``_log_modulus_bound``).

    Past the edge each W_j(x) is at most W_j at the edge when r_j lies on the edge's inner side and at most
    W_j(r_j) otherwise, as in ``_log_tail``; (|x| + b)^m expands into the prior's tail moments."""
    log_normalizers = -0.5 * (_LOG_TWO_PI + np.log(variances))
    deviations = residuals - edge[:, None]
    at_edge = _logsumexp(log_weights[:, None, :] + log_normalizers - 0.5 * np.square(deviations)[..., None] / variances, axis=2)
    at_peak = _logsumexp(log_weights + log_normalizers, axis=1)
    inner = residuals <= edge[:, None] if upper else residuals >= edge[:, None]
    occasions = np.sum(np.where(inner, at_edge, at_peak[:, None]), axis=1)
    tails = _log_prior_tail_moments(edge if upper else -edge, level_variance)
    log_width = np.log(half_width)
    moments = np.stack([
        tails[:, 0],
        np.logaddexp(tails[:, 1], log_width + tails[:, 0]),
        _logsumexp(np.stack([tails[:, 2], np.log(2.0) + log_width + tails[:, 1], 2.0 * log_width + tails[:, 0]], axis=1), axis=1),
    ], axis=1)
    return occasions[:, None] + moments + (0.5 * np.square(half_width) / level_variance)[:, None]


def _log_modulus_bound(
    residuals: F64Array, levels: F64Array, valid: np.ndarray, steps: F64Array, level_variance: float, log_masses: F64Array,
    variances: F64Array, half_width: F64Array,
) -> F64Array:
    """log of a bound on the strip modulus M_m(b), m = 0, 1, 2, at each person's b (persons x 3).

    On |Im T| = y < b, |N(e + i y; 0, s)| = N(e; 0, s) e^(y^2 / (2 s)) and |T^m| <= (|x| + b)^m, so with
    w_k = pi_k e^(b^2 / (2 s_k)) and W_j(x) = sum_k w_k N(r_j - x; 0, s_k), M_m(b) is at most the integral of
    g(x) = (|x| + b)^m N(x; 0, tau^2) e^(b^2 / (2 tau^2)) prod_j W_j(x) (the mass-weighted modulus,
    mixing_density.md section 11). Between two neighbouring nodes each Gaussian factor is largest at the point
    of the interval nearest its centre and (|x| + b)^m at the end farthest from 0, so h times their product
    bounds g's integral there (an upper Riemann sum on the trapezoid's own nodes); beyond the end nodes,
    ``_log_widened_tail``. The bound tightens as h does and weights every component by its own mass, so a narrow
    component of little mass costs little.
    """
    log_weights = log_masses[None, :] + (0.5 * np.square(half_width))[:, None] / variances[None, :]
    log_normalizers = -0.5 * (_LOG_TWO_PI + np.log(variances))
    left, right = levels[:, :-1], levels[:, 1:]
    intervals = valid[:, 1:]
    nearest = np.maximum(np.maximum(left[:, :, None] - residuals[:, None, :], residuals[:, None, :] - right[:, :, None]), 0.0)
    terms = np.square(nearest)[..., None] * (-0.5 / variances)
    terms += (log_weights + log_normalizers)[:, None, None, :]
    occasions = _logsumexp(terms, axis=3).sum(axis=2)
    del terms
    to_zero = np.maximum(np.maximum(left, -right), 0.0)
    farthest = np.maximum(np.abs(left), np.abs(right)) + half_width[:, None]
    log_prior = _log_prior(to_zero, level_variance) + (0.5 * np.square(half_width) / level_variance)[:, None]
    log_interior = np.where(intervals, log_prior + occasions, -np.inf)
    interior = _log_moment_sums(log_interior, farthest, steps, 0.0)
    counts = valid.sum(axis=1)
    first, last = levels[:, 0], levels[np.arange(levels.shape[0]), counts - 1]
    below = _log_widened_tail(residuals, first, level_variance, log_weights, variances, half_width, upper=False)
    above = _log_widened_tail(residuals, last, level_variance, log_weights, variances, half_width, upper=True)
    return _logsumexp(np.stack([interior, below, above], axis=2), axis=2)


class LatticeTooLarge(MemoryError):
    """A noise-density lattice whose K x K matrices exceed the memory budget."""


# The lattice's K x K float64 matrices live at once, at most: in the engine's prior (the sum-to-zero basis, the
# roughness factor, the coefficient map, the penalty and its eigenvectors) and, beside it, a least-squares refit
# of its coefficients (the matrix's copy and its two singular-vector factors).
_LATTICE_MATRICES = 8


def _check_lattice(node_count: int, working_bytes: int) -> None:
    if node_count * node_count * _LATTICE_MATRICES * _FLOAT64_BYTES > working_bytes:
        raise LatticeTooLarge(f"a noise-density lattice of {node_count} nodes exceeds the {working_bytes} byte budget")


class PieceTooLarge(MemoryError):
    """An E-step piece whose persons x nodes x J x K working arrays exceed the memory budget."""


def _log_moment_sums(log_integrand: F64Array, levels: F64Array, steps: F64Array, offset: F64Array | float) -> F64Array:
    """log of h sum_n (|T_n| + offset)^m exp(log_integrand) for m = 0, 1, 2 (persons x 3; offset per person)."""
    with np.errstate(divide="ignore"):  # the node at T = 0 has |T|^m = 0 for m > 0
        log_magnitude = np.log(np.abs(levels) + np.reshape(offset, (-1, 1)))
    log_powers = np.stack([np.zeros_like(log_magnitude), log_magnitude, 2.0 * log_magnitude], axis=2)
    return _logsumexp(log_integrand[:, :, None] + log_powers, axis=1) + np.log(steps)[:, None]


@dataclass(frozen=True)
class LevelGrid:
    """Each person's trapezoid rule in T, the start of their next E-step (NaN: none yet): the step, the centre,
    the reach below and above it, and the strip half-width b of the step's certificate."""

    step: F64Array
    centre: F64Array
    lower_reach: F64Array
    upper_reach: F64Array
    half_width: F64Array

    @staticmethod
    def unset(persons: int) -> LevelGrid:
        empty = np.full(persons, np.nan)
        return LevelGrid(empty, empty.copy(), empty.copy(), empty.copy(), empty.copy())

    def take(self, rows: I64Array | slice) -> LevelGrid:
        """A copy of the rows' entries."""
        return LevelGrid(*(np.array(getattr(self, name)[rows]) for name in ("step", "centre", "lower_reach", "upper_reach", "half_width")))


def _level_grid(
    residuals: F64Array, level_variance: float, log_masses: F64Array, variances: F64Array, steps: F64Array,
    centres: F64Array, lower_reach: F64Array, upper_reach: F64Array, relative_tolerance: float, working_bytes: int,
) -> tuple[F64Array, np.ndarray, F64Array, F64Array, F64Array, F64Array, F64Array, F64Array]:
    """Trapezoid nodes on each person's grid c_p + n h_p (persons x nodes, a validity mask), the log components
    at them (``_log_components``), each occasion's log density and the log integrand there, the log integrals
    of |T|^m F, m = 0, 1, 2 (persons x 3), and the reaches below and above the centre to start the next grid at.

    Each side starts at its reach and doubles until every one of its tail bounds (``_log_tail``, which bounds
    the omitted nodes' terms) is at most a quarter of relative_tolerance times its integral so far. The reach
    returned for the next grid is the accepted one halved while its tail bound still holds. The centre and
    reaches set only the cost.
    """
    lower = -np.maximum(np.ceil(lower_reach / steps), 1.0)
    upper = np.maximum(np.ceil(upper_reach / steps), 1.0)
    while True:
        # Checked in floating point before any persons x nodes array exists (or an integer count overflows): the
        # grid's own index, mask and nodes are smaller than the components per entry the budget counts.
        widest = float(np.max(upper - lower)) + 1.0
        if not residuals.shape[0] * widest * residuals.shape[1] * variances.shape[0] * _ENTRY_BYTES <= working_bytes:
            raise PieceTooLarge(f"{residuals.shape[0]} persons need {widest:.3g} level nodes each")
        counts = (upper - lower).astype(np.int64) + 1
        width = int(counts.max())
        offsets = np.arange(width)[None, :]
        valid = offsets < counts[:, None]
        levels = centres[:, None] + (lower[:, None] + np.minimum(offsets, counts[:, None] - 1)) * steps[:, None]
        log_components = _log_components(residuals, levels, log_masses, variances)
        per_occasion = _logsumexp(log_components, axis=3)
        log_integrand = np.where(valid, _log_prior(levels, level_variance) + per_occasion.sum(axis=2), -np.inf)
        log_totals = _log_moment_sums(log_integrand, levels, steps, 0.0)
        if not np.all(np.isfinite(log_totals[:, 0])):
            raise FloatingPointError("a person's level integral is not finite")
        bound = np.log(0.25 * relative_tolerance) + log_totals

        def holds(offset: F64Array, side_upper: bool) -> np.ndarray:
            tail = _log_tail(residuals, centres + offset, level_variance, log_masses, variances, steps, upper=side_upper)
            return np.all(tail <= bound, axis=1)

        grow_lower, grow_upper = ~holds(lower * steps, False), ~holds(upper * steps, True)
        if not (np.any(grow_lower) or np.any(grow_upper)):
            break
        lower = np.where(grow_lower, 2.0 * lower, lower)
        upper = np.where(grow_upper, 2.0 * upper, upper)
    reaches = []
    for extent, side_upper in ((-lower * steps, False), (upper * steps, True)):
        tightening = np.ones(extent.shape[0], dtype=bool)
        while np.any(tightening):
            half = np.where(tightening, 0.5 * extent, extent)
            tightening &= (half >= steps) & holds(half if side_upper else -half, side_upper)
            extent = np.where(tightening, half, extent)
        reaches.append(extent)
    return levels, valid, log_components, per_occasion, log_integrand, log_totals, reaches[0], reaches[1]


def level_posterior(
    residuals: F64Array, level_variance: float, log_masses: F64Array, variances: F64Array, relative_tolerance: float,
    start: LevelGrid | None, louis: bool, working_bytes: int,
) -> LevelPosterior:
    """The exact E-step for persons with J occasions each (``residuals`` is persons x J).

    L_i and the absolute moment integrals I_m of |T| F and T^2 F each have relative error at most
    ``relative_tolerance``: half from the trapezoid rule, a quarter from each truncated tail. The trapezoid
    rule's error is at most 2 M_m(b) / (e^(2 pi b / h) - 1) for any strip |Im T| < b (Trefethen and Weideman
    2014, Theorem 5.1), with M_m(b) bounded by ``_log_modulus_bound``, so a person's step h is certified when
    h <= min_m 2 pi b / ln(1 + 2 M_m(b) / (eps I_m)), I_m at least its lattice sum over 1 + relative_tolerance.

    Every b is valid, so b only sets the cost: a person keeps the b of their ``start``, and one without it, or
    whose step it does not certify, moves b by factors of 2 while the certified step grows. ``start`` (None, or
    NaN entries) otherwise begins at the Gaussian posterior of occasions at the density's harmonic-mean variance:
    its mean as the centre, its standard deviation as the reaches, its ``gaussian_step`` and strip. Steps shrink
    to the certified step until every person's is.
    """
    persons, occasion_count = residuals.shape
    trapezoid_share = 0.5 * relative_tolerance
    harmonic = 1.0 / float(np.exp(_logsumexp(log_masses - np.log(variances), axis=0)))
    precision = 1.0 / level_variance + occasion_count / harmonic
    start = LevelGrid.unset(persons) if start is None else start

    def filled(values: F64Array, default: F64Array | float) -> F64Array:
        return np.where(np.isnan(values), default, values)

    steps = filled(start.step, gaussian_step(level_variance, occasion_count, harmonic, trapezoid_share))
    centres = filled(start.centre, residuals.sum(axis=1) / harmonic / precision)
    lower_reach = filled(start.lower_reach, 1.0 / np.sqrt(precision))
    upper_reach = filled(start.upper_reach, 1.0 / np.sqrt(precision))
    half_widths = filled(start.half_width, strip_half_width(level_variance, occasion_count, harmonic, trapezoid_share))
    searched = np.isnan(start.half_width)
    grid_size = variances.shape[0]
    log_likelihood, level_mean, level_second = np.empty(persons), np.empty(persons), np.empty(persons)
    occasion_precision, occasion_shift = np.empty((persons, occasion_count)), np.empty((persons, occasion_count))
    admissible = np.empty(persons)
    counts, missing = np.zeros(grid_size), np.zeros((grid_size, grid_size))
    split, magnitude, operations = np.zeros((2, 2)), np.zeros((2, 2)), 0
    # Only the persons whose step is not yet certified are integrated again.
    pending = np.arange(persons)
    while pending.size:
        levels, valid, log_components, per_occasion, log_integrand, log_totals, lower_next, upper_next = _level_grid(
            residuals[pending], level_variance, log_masses, variances, steps[pending], centres[pending], lower_reach[pending],
            upper_reach[pending], relative_tolerance, working_bytes,
        )
        log_lower = log_totals - np.log1p(relative_tolerance)

        def certified_at(rows: I64Array, width: F64Array) -> F64Array:
            log_modulus = _log_modulus_bound(
                residuals[pending[rows]], levels[rows], valid[rows], steps[pending[rows]], level_variance, log_masses, variances, width
            )
            ratio = np.log(2.0) + log_modulus - log_lower[rows] - np.log(trapezoid_share)
            return np.min(2.0 * np.pi * width[:, None] / np.logaddexp(0.0, ratio), axis=1)

        everyone = np.arange(pending.size)
        widths = half_widths[pending]
        certified = certified_at(everyone, widths)
        search = np.flatnonzero(searched[pending] | (steps[pending] > certified))
        for factor in (0.5, 2.0):
            moving = search
            while moving.size:
                trial = widths[moving] * factor
                trial_certified = certified_at(moving, trial)
                better = trial_certified > certified[moving]
                widths[moving[better]], certified[moving[better]] = trial[better], trial_certified[better]
                moving = moving[better]
        searched[pending] = False
        half_widths[pending] = widths
        lower_reach[pending], upper_reach[pending] = lower_next, upper_next
        admissible[pending] = certified
        done = steps[pending] <= certified
        rows = pending[done]
        # A view, not a copy, when every person is certified (the usual case).
        finished = slice(None) if np.all(done) else np.flatnonzero(done)
        log_likelihood[rows] = log_totals[finished, 0]
        weights = np.exp(log_integrand[finished] - _logsumexp(log_integrand[finished], axis=1)[:, None])
        node_levels = levels[finished]
        level_mean[rows] = np.sum(weights * node_levels, axis=1)
        level_second[rows] = np.sum(weights * np.square(node_levels), axis=1)
        responsibility = log_components[finished] - per_occasion[finished][..., None]
        del log_components
        np.exp(responsibility, out=responsibility)
        weighted = weights[:, :, None, None] * responsibility
        # The sums over persons, nodes and occasions are matrix products over their flattened entries (BLAS).
        flat_weighted = weighted.reshape(-1, grid_size)
        piece_counts = flat_weighted.sum(axis=0)
        counts += piece_counts
        if louis:
            # Var(C_i | z_i) = E_T[sum_j (diag rho_j - rho_j rho_j')] + Var_T(sum_j rho_j(T)).
            node_sums = responsibility.sum(axis=2)
            mean_sums = np.matmul(weights[:, None, :], node_sums)[:, 0, :]
            flat_sums = node_sums.reshape(-1, grid_size)
            missing += (
                np.diag(piece_counts)
                - flat_weighted.T @ responsibility.reshape(-1, grid_size)
                + (flat_sums * weights.reshape(-1, 1)).T @ flat_sums
                - mean_sums.T @ mean_sums
            )
            # The level/noise split: Louis' identity for (log tau^2, a), a scaling every noise variance by e^a.
            # The complete-data scores are t = T^2 / (2 tau^2) - 1/2 and sum_j (e_j^2 / (2 s) - 1/2), e_j = r_j - T;
            # given T each occasion's component is independent, so E and Var of the noise score at a node are
            # sums over occasions of e^2 R1 / 2 - 1/2 and e^4 (R2 - R1^2) / 4, R_p = sum_k rho_k / s_k^p.
            squares = np.square(residuals[rows][:, None, :] - node_levels[:, :, None])
            first = responsibility @ (1.0 / variances)
            second = responsibility @ (1.0 / np.square(variances))
            noise_score = np.sum(0.5 * squares * first - 0.5, axis=2)
            noise_variance = 0.25 * np.sum(np.square(squares) * (second - np.square(first)), axis=2)
            level_score = 0.5 * np.square(node_levels) / level_variance - 0.5
            complete = np.stack([
                np.sum(weights * (level_score + 0.5), axis=1),
                np.sum(weights * 0.5 * np.sum(squares * first, axis=2), axis=1),
            ])
            means = np.stack([np.sum(weights * level_score, axis=1), np.sum(weights * noise_score, axis=1)])
            raw = np.stack([
                np.sum(weights * np.square(level_score), axis=1),
                np.sum(weights * (noise_variance + np.square(noise_score)), axis=1),
            ])
            cross = np.sum(weights * level_score * noise_score, axis=1)
            split += np.array([
                [np.sum(complete[0] - raw[0] + np.square(means[0])), -np.sum(cross - means[0] * means[1])],
                [-np.sum(cross - means[0] * means[1]), np.sum(complete[1] - raw[1] + np.square(means[1]))],
            ])
            off_diagonal = np.sum(np.sum(weights * np.abs(level_score * noise_score), axis=1) + np.abs(means[0] * means[1]))
            magnitude += np.array([
                [np.sum(complete[0] + raw[0] + np.square(means[0])), off_diagonal],
                [off_diagonal, np.sum(complete[1] + raw[1] + np.square(means[1]))],
            ])
            # Per (node, occasion, component) term: a product with rho, a square, two scalings and the sums.
            operations += responsibility.size * _SPLIT_OPERATIONS_PER_TERM
        node_precision = weighted @ (1.0 / variances)
        occasion_precision[rows] = node_precision.sum(axis=1)
        occasion_shift[rows] = np.sum(node_precision * node_levels[:, :, None], axis=1)
        # A lattice sum moves by at most a factor 1 +- relative_tolerance between certified grids, which moves the
        # certified step by less than a factor 1 + relative_tolerance: shrinking by that much more leaves room
        # below the next certificate, where shrinking to it exactly creeps toward it from above.
        steps[pending[~done]] = certified[~done] / (1.0 + relative_tolerance)
        pending = pending[~done]
    return LevelPosterior(
        log_likelihood=log_likelihood,
        level_mean=level_mean,
        level_second_moment=level_second,
        counts=counts,
        occasion_precision=occasion_precision,
        occasion_shift=occasion_shift,
        missing_information=missing,
        split_information=split,
        split_magnitude=magnitude,
        split_operations=operations,
        admissible_step=admissible,
        # The next step starts halfway in log between the certified one and the largest admissible here, so it
        # grows while the certificate allows and falls back by one re-integration when it does not.
        grid=LevelGrid(np.sqrt(steps * np.maximum(admissible, steps)), level_mean.copy(), lower_reach, upper_reach, half_widths),
    )


# ------------------------------------------------------------------ the density's M-step and evidence


def _penalty_matrix(prior: ScaleMixturePrior, log_smoothing: F64Array) -> F64Array:
    penalty = np.zeros((prior.coefficient_size, prior.coefficient_size))
    for block, log_weight in zip(prior.smoothing_blocks, log_smoothing, strict=True):
        if np.isfinite(log_weight):
            penalty[np.ix_(block.coordinates, block.coordinates)] += np.exp(log_weight) * block.matrix
    return penalty


def _allowed(prior: ScaleMixturePrior, log_smoothing: F64Array) -> F64Array:
    """A basis of the coefficients the weights allow: the penalty's null space where a weight is infinite."""
    if np.all(log_smoothing < np.inf):
        return np.eye(prior.coefficient_size)
    return prior.null_basis


def maximize_count_density(prior: ScaleMixturePrior, log_smoothing: F64Array, counts: F64Array, start: F64Array) -> F64Array:
    """x maximizing sum_k c_k log pi_k(x) - 1/2 x' S x over the allowed coefficients.

    The log-softmax of non-negative counts is concave, so Newton with a halving line search reaches the
    maximum; it stops when the Newton step's predicted gain is at the objective's rounding level.
    """
    allowed = _allowed(prior, log_smoothing)
    mapping = prior.coefficient_map[: prior.grid_size] @ allowed
    penalty = allowed.T @ _penalty_matrix(prior, log_smoothing) @ allowed
    total = float(counts.sum())

    def objective(reduced: F64Array) -> float:
        return float(counts @ class_log_density(prior, allowed @ reduced)[0]) - 0.5 * float(reduced @ penalty @ reduced)

    reduced = allowed.T @ start
    value = objective(reduced)
    while True:
        masses = np.exp(class_log_density(prior, allowed @ reduced)[0])
        gradient = mapping.T @ (counts - total * masses) - penalty @ reduced
        curvature = mapping.T @ (total * (np.diag(masses) - np.outer(masses, masses))) @ mapping + penalty
        step = np.linalg.lstsq(curvature, gradient, rcond=None)[0]
        if 0.5 * float(gradient @ step) <= _EPSILON * (abs(value) + total):
            return allowed @ reduced
        length = 1.0
        while True:
            candidate = reduced + length * step
            candidate_value = objective(candidate)
            if candidate_value > value:
                reduced, value = candidate, candidate_value
                break
            length *= 0.5
            if length * float(np.linalg.norm(step)) <= _HALF_PRECISION * (1.0 + float(np.linalg.norm(reduced))):
                return allowed @ reduced


def _log_determinant(matrix: F64Array) -> float | None:
    """log det of a symmetric positive definite matrix by its Cholesky factor, or None when it is not positive
    definite (a determinant's sign cannot tell: an even number of negative eigenvalues gives a positive one)."""
    try:
        factor = scipy.linalg.cholesky(0.5 * (matrix + matrix.T), lower=True)
    except np.linalg.LinAlgError:
        return None
    return 2.0 * float(np.sum(np.log(np.diag(factor))))


def laplace_evidence(
    prior: ScaleMixturePrior, log_smoothing: F64Array, coefficients: F64Array, log_likelihood: float, observed_information: F64Array
) -> float:
    """V = l(x) - 1/2 x' S x + 1/2 log|S|_+ - 1/2 log|-H| + 1/2 log|N'(-H)N| (the engine's B + S form).

    -H = M' I M + S in x, with I the observed information in the density's log values (Louis) and N the
    profiled null space. At an infinite weight x lies in the null space; at an infinite or zero weight every
    allowed direction is profiled, so V is the log-likelihood there.

    The inner problem (the penalized marginal likelihood in x) is not concave in general: the observed
    information of a scale mixture can be indefinite. So a strict maximum is required, and the Laplace V is not
    corrected toward the exact integral here (the engine's Tierney-Kadane certification is not applied). PENDING:
    the engine's rounding test (Demmel's componentwise bound, computed on its profiled factor in the [N, C]
    basis) once that factor is public; the same bound on the full -H here is too conservative to use.
    """
    if not np.all(np.isfinite(log_smoothing)):
        return log_likelihood
    mapping = prior.coefficient_map[: prior.grid_size]
    penalty = _penalty_matrix(prior, log_smoothing)
    eigenvalues = np.linalg.eigvalsh(penalty)
    kept = eigenvalues > _EPSILON * penalty.shape[0] * max(float(eigenvalues[-1]), np.finfo(np.float64).tiny)
    negative_hessian = mapping.T @ observed_information @ mapping + penalty
    null = prior.null_basis
    full, profiled = _log_determinant(negative_hessian), _log_determinant(null.T @ negative_hessian @ null)
    if full is None or profiled is None:
        # Not a strict maximum: the Laplace evidence is undefined, and the search counts it as the lowest value.
        return -np.inf
    return (
        log_likelihood
        - 0.5 * float(coefficients @ penalty @ coefficients)
        + 0.5 * float(np.sum(np.log(eigenvalues[kept])))
        - 0.5 * full
        + 0.5 * profiled
    )


# ------------------------------------------------------------------ the EM


@dataclass(frozen=True)
class _State:
    fixed_effects: F64Array
    level_variance: float
    prior: ScaleMixturePrior
    hyperparameters: MixtureHyperparameters


@dataclass(frozen=True)
class _Expectation:
    log_likelihood: float
    level_mean: F64Array
    level_second_moment: F64Array
    counts: F64Array
    occasion_precision: F64Array
    occasion_shift: F64Array
    missing_information: F64Array
    split_information: F64Array
    split_magnitude: F64Array
    split_operations: int
    # Every person's trapezoid rule at this E-step, the start of the next.
    grid: LevelGrid


def _density_prior(nodes: F64Array, top: float, occasion_count: int) -> ScaleMixturePrior:
    """The noise density's lattice as the engine's one-class prior; every node is a kernel node (no mass below
    the resolution variance, the first node)."""
    return scale_mixture_prior(
        class_index=np.zeros(occasion_count, dtype=np.int64),
        log_variance_offset=np.zeros(occasion_count),
        annotation_design=np.zeros((occasion_count, 0)),
        annotation_groups=(),
        nodes=nodes,
        floor=float(nodes[0]),
        top=min(top, float(nodes[-1])),
    )


def _moment_hyperparameters(prior: ScaleMixturePrior, squared_deviations: F64Array, spacing: float) -> MixtureHyperparameters:
    """A start density in the roughness penalty's null space (a normal log-density in t = log s) matching the
    moments of log e^2 over the within-person squared deviations e^2 (each scaled by J / (J - 1) to be unbiased
    for s under Gaussian noise).

    With e^2 = s chi^2_1 and t ~ N(mu, sigma^2), E log e^2 = mu + psi(1/2) + log 2 and
    Var log e^2 = sigma^2 + psi'(1/2). A deviation below the resolution variance is not resolvable, so it counts
    at that floor (the lattice's first node), and sigma is at least the lattice spacing, the narrowest density the
    lattice resolves. The start sets only the EM's cost.
    """
    log_squares = np.log(np.maximum(squared_deviations, float(np.exp(prior.log_variance_grid[0]))))
    mean = float(np.mean(log_squares)) - float(digamma(0.5) + np.log(2.0))
    variance = max(float(np.var(log_squares)) - float(polygamma(1, 0.5)), spacing * spacing)
    log_density = -0.5 * np.square(prior.log_variance_grid - mean) / variance
    mapping = prior.coefficient_map[: prior.grid_size]
    coefficients = np.linalg.lstsq(mapping, log_density - log_density.mean(), rcond=None)[0]
    # The weight starts at unit: this model's own bracketed search below covers its whole resolvable range, so the
    # start sets only the EM's cost (the engine's start at the lambda = infinity edge is not a bracket point).
    return MixtureHyperparameters(coefficients=coefficients, log_smoothing=np.zeros(len(prior.smoothing_blocks)))


class _Model:
    """The EM at one Box-Cox exponent."""

    def __init__(self, occasions: Occasions, exponent: float, working_bytes: int) -> None:
        self.occasions = occasions
        self.exponent = exponent
        self.working_bytes = working_bytes
        self.transformed, log_jacobian = box_cox(occasions.values, exponent)
        self.log_jacobian = float(log_jacobian.sum())
        self.resolution_variance = float(np.min(np.square(np.exp(log_jacobian) * occasions.resolution))) / 12.0
        counts = np.bincount(occasions.person_index)
        order = np.argsort(occasions.person_index, kind="stable")
        sorted_persons = occasions.person_index[order]
        self.groups = [order[counts[sorted_persons] == count].reshape(-1, count) for count in np.unique(counts)]
        self.relative_tolerance = person_tolerance(occasions.person_count)
        # Each person's last trapezoid rule, the start of their next E-step.
        self.grid = LevelGrid.unset(occasions.person_count)
        # The coefficients that give a constant 1 when the design's columns span it (for PX-EM), else None.
        ones = np.ones(occasions.values.shape[0])
        constant = np.linalg.lstsq(occasions.design, ones, rcond=None)[0]
        spanned = np.max(np.abs(occasions.design @ constant - ones)) <= _HALF_PRECISION
        self.constant = constant if spanned else None

    def start(self) -> _State:
        """Least-squares fixed effects, a robust level variance, and a moment-matched start density on the lattice
        from the resolution variance to the largest squared within-person deviation (past it by its width), with
        at least the engine's minimum of ROUGHNESS_ORDER + 2 nodes (``scale_mixture_prior``)."""
        occasions = self.occasions
        fixed_effects = np.linalg.lstsq(occasions.design, self.transformed, rcond=None)[0]
        residuals = self.transformed - occasions.design @ fixed_effects
        counts = np.bincount(occasions.person_index).astype(np.float64)
        person_mean = np.bincount(occasions.person_index, weights=residuals) / counts
        deviations = residuals - person_mean[occasions.person_index]
        repeated = counts[occasions.person_index] > 1
        if not np.any(repeated):
            # Only tau^2 + E[s] is identified: the target is then the transformed reading (``marginal_transform``).
            raise ValueError("no person has a repeated occasion, so the level and the noise are not separable")
        largest = float(np.max(np.square(deviations[repeated])))
        # The person means' variance by their median absolute deviation, which gross errors barely move: it
        # overstates tau^2 by the noise of a mean, which the EM removes.
        level_variance = float(np.median(np.square(person_mean - np.median(person_mean)))) / float(chi2.median(1))
        if not level_variance > 0.0:
            raise ValueError("no between-person variance: most persons' mean readings are equal")
        floor = float(np.log(self.resolution_variance))
        top = float(np.log(max(largest, self.resolution_variance)))
        spacing = spacing_bound(float(occasions.values.shape[0]), EVIDENCE_TOLERANCE)
        extent = max(top + (top - floor), floor + (ROUGHNESS_ORDER + 1) * spacing)
        nodes = np.arange(floor, extent + spacing, spacing)
        _check_lattice(nodes.shape[0], self.working_bytes)
        prior = _density_prior(nodes, top, occasions.values.shape[0])
        # J / (J - 1) over the repeated occasions only: a singleton's J - 1 is 0, and it has no deviation to scale.
        repeated_counts = counts[occasions.person_index[repeated]]
        scaled = np.square(deviations[repeated]) * repeated_counts / (repeated_counts - 1.0)
        return _State(fixed_effects, level_variance, prior, _moment_hyperparameters(prior, scaled, spacing))

    def residuals(self, state: _State) -> F64Array:
        return self.transformed - self.occasions.design @ state.fixed_effects

    def expectation(self, state: _State, louis: bool, start: LevelGrid | None = None, adopt: bool = True) -> _Expectation:
        """The exact E-step over every person, in pieces whose working arrays fit ``working_bytes``, from the grids
        ``start`` (default: the model's current grids), which it adopts as current when ``adopt``."""
        occasions = self.occasions
        residuals = self.residuals(state)
        log_masses = class_log_density(state.prior, state.hyperparameters.coefficients)[0]
        variances = np.exp(state.prior.log_variance_grid)
        level_mean, level_second = np.empty(occasions.person_count), np.empty(occasions.person_count)
        precision, shift = np.empty(residuals.shape[0]), np.empty(residuals.shape[0])
        counts = np.zeros(variances.shape[0])
        missing = np.zeros((variances.shape[0], variances.shape[0]))
        split, magnitude, operations = np.zeros((2, 2)), np.zeros((2, 2)), 0
        log_likelihood = self.log_jacobian
        # A piece's arrays are as wide as its widest grid, so persons are pieced by their last node count to within
        # a factor of 2 (none yet: one piece per occasion count).
        grid = (self.grid if start is None else start).take(slice(None))
        with np.errstate(invalid="ignore"):
            node_counts = np.nan_to_num((grid.lower_reach + grid.upper_reach) / grid.step, nan=0.0)
        pending = []
        for rows in self.groups:
            size_class = np.floor(np.log2(1.0 + node_counts[occasions.person_index[rows[:, 0]]]))
            pending += [rows[size_class == value] for value in np.unique(size_class)]
        while pending:
            piece = pending.pop()
            persons = occasions.person_index[piece[:, 0]]
            try:
                posterior = level_posterior(
                    residuals[piece], state.level_variance, log_masses, variances, self.relative_tolerance,
                    grid.take(persons), louis, self.working_bytes,
                )
            except PieceTooLarge:
                if piece.shape[0] == 1:
                    raise
                pending += [piece[: piece.shape[0] // 2], piece[piece.shape[0] // 2 :]]
                continue
            log_likelihood += float(posterior.log_likelihood.sum())
            level_mean[persons] = posterior.level_mean
            level_second[persons] = posterior.level_second_moment
            precision[piece] = posterior.occasion_precision
            shift[piece] = posterior.occasion_shift
            counts += posterior.counts
            missing += posterior.missing_information
            split += posterior.split_information
            magnitude += posterior.split_magnitude
            operations += posterior.split_operations
            for name in ("step", "centre", "lower_reach", "upper_reach", "half_width"):
                getattr(grid, name)[persons] = getattr(posterior.grid, name)
        if adopt:
            self.grid = grid
        return _Expectation(log_likelihood, level_mean, level_second, counts, precision, shift, missing, split, magnitude, operations, grid)

    def trial(self, state: _State, louis: bool, start: LevelGrid) -> _Expectation | None:
        """The E-step at a speculative point (an extrapolation or a Newton step), or None where its level integrals
        are not finite or a person's certified rule needs more nodes than memory allows: the point is then
        rejected, never approximated. It leaves the model's grids unchanged."""
        try:
            return self.expectation(state, louis, start, adopt=False)
        except (PieceTooLarge, FloatingPointError):
            return None

    def level_step(self, state: _State, expectation: _Expectation) -> _State:
        """The PX-EM step (Liu, Rubin and Wu 1998) of gamma and tau^2 at a fixed density.

        EM runs in the model expanded by a prior mean a, T_i ~ N(a, tau^2): gamma by weighted least squares
        (weights E[1/s], shifts E[T/s]), a = mean E[T_i] and tau^2 = mean E[T_i^2] - a^2. Reduced back (a moves
        into the design's constant, when its columns span one), it is an EM step of the original model that
        moves the levels' common offset at once; plain EM moves it only as fast as tau^2 shrinks.
        """
        design = self.occasions.design
        weighted = design * expectation.occasion_precision[:, None]
        fixed_effects = np.linalg.solve(design.T @ weighted, weighted.T @ self.transformed - design.T @ expectation.occasion_shift)
        offset = float(np.mean(expectation.level_mean)) if self.constant is not None else 0.0
        if self.constant is not None:
            fixed_effects = fixed_effects + offset * self.constant
        level_variance = float(np.mean(expectation.level_second_moment)) - offset * offset
        return _State(fixed_effects, level_variance, state.prior, state.hyperparameters)

    def newton_direction(self, state: _State, expectation: _Expectation, log_smoothing: F64Array) -> tuple[F64Array, float] | None:
        """The Newton direction of the density's coefficients on the exact penalized marginal log-likelihood and
        its decrement (the quadratic model's gain, g' d / 2), or None where its curvature is not negative definite.

        Its gradient is M'(C - N pi) - S x (Fisher's identity, C the expected counts) and its negative curvature
        M' I M + S, with I Louis' observed information (``expectation`` carries both).
        """
        prior = state.prior
        allowed = _allowed(prior, log_smoothing)
        mapping = prior.coefficient_map[: prior.grid_size] @ allowed
        penalty = allowed.T @ _penalty_matrix(prior, log_smoothing) @ allowed
        reduced = allowed.T @ state.hyperparameters.coefficients
        masses = np.exp(class_log_density(prior, state.hyperparameters.coefficients)[0])
        total = float(expectation.counts.sum())
        gradient = mapping.T @ (expectation.counts - total * masses) - penalty @ reduced
        curvature = mapping.T @ (total * (np.diag(masses) - np.outer(masses, masses)) - expectation.missing_information) @ mapping + penalty
        try:
            reduced_direction = scipy.linalg.cho_solve(scipy.linalg.cho_factor(0.5 * (curvature + curvature.T)), gradient)
        except np.linalg.LinAlgError:
            return None
        return allowed @ reduced_direction, 0.5 * float(gradient @ reduced_direction)

    def em_map(self, state: _State, expectation: _Expectation, log_smoothing: F64Array) -> _State:
        """EM's own update from one E-step: gamma and tau^2 (``level_step``) and the density maximizing the expected
        counts' penalized multinomial log-likelihood (``maximize_count_density``). It never lowers the penalized
        log-likelihood."""
        levels = self.level_step(state, expectation)
        coefficients = maximize_count_density(state.prior, log_smoothing, expectation.counts, state.hyperparameters.coefficients)
        return self._with_density(levels, coefficients, log_smoothing)

    def squarem_step(self, state: _State, expectation: _Expectation, log_smoothing: F64Array) -> tuple[_State, _Expectation]:
        """Two EM updates extrapolated by SQUAREM (Varadhan and Roland 2008, scheme S3): with r = F(x) - x and
        v = F(F(x)) - 2 F(x) + x, x' = x - 2 a r + a^2 v at a = -|r| / |v| (at most -1, where x' = F(F(x))),
        kept when its penalized log-likelihood is at least that of F(F(x)) (and its E-step exists, ``trial``). The
        parameters are gamma, log tau^2 and the density's allowed coefficients. The model's grids are not changed;
        the caller adopts the returned E-step's."""
        allowed = _allowed(state.prior, log_smoothing)

        def vector(point: _State) -> F64Array:
            return np.concatenate([point.fixed_effects, [np.log(point.level_variance)], allowed.T @ point.hyperparameters.coefficients])

        def point(values: F64Array) -> _State:
            size = state.fixed_effects.shape[0]
            levels = _State(values[:size], float(np.exp(values[size])), state.prior, state.hyperparameters)
            return self._with_density(levels, allowed @ values[size + 1 :], log_smoothing)

        first = self.em_map(state, expectation, log_smoothing)
        first_expectation = self.expectation(first, louis=False, start=expectation.grid, adopt=False)
        second = self.em_map(first, first_expectation, log_smoothing)
        second_expectation = self.expectation(second, louis=True, start=first_expectation.grid, adopt=False)
        origin, once, twice = vector(state), vector(first), vector(second)
        change, curvature = once - origin, twice - 2.0 * once + origin
        curvature_norm = float(np.linalg.norm(curvature))
        if curvature_norm == 0.0:
            return second, second_expectation
        steplength = min(-float(np.linalg.norm(change)) / curvature_norm, -1.0)
        if steplength == -1.0:
            return second, second_expectation
        extrapolated = point(origin - 2.0 * steplength * change + steplength * steplength * curvature)
        extrapolated_expectation = self.trial(extrapolated, True, second_expectation.grid)
        if extrapolated_expectation is not None and self.penalized(extrapolated, extrapolated_expectation) >= self.penalized(second, second_expectation):
            return extrapolated, extrapolated_expectation
        return second, second_expectation

    @staticmethod
    def _with_density(state: _State, coefficients: F64Array, log_smoothing: F64Array) -> _State:
        return _State(state.fixed_effects, state.level_variance, state.prior, MixtureHyperparameters(coefficients, log_smoothing.copy()))

    def penalized(self, state: _State, expectation: _Expectation) -> float:
        coefficients = state.hyperparameters.coefficients
        return expectation.log_likelihood - 0.5 * float(coefficients @ _penalty_matrix(state.prior, state.hyperparameters.log_smoothing) @ coefficients)

    def converge(self, state: _State, log_smoothing: F64Array) -> tuple[_State, _Expectation, bool]:
        """ECME (Liu and Rubin 1994) to the penalized maximum at one penalty weight; returns the state and its
        E-step with Louis' information.

        Each iteration takes gamma and tau^2 by their EM step (``level_step``) together with the density's Newton
        step on the exact marginal likelihood (``newton_direction``), from one E-step. Where that curvature is not
        negative definite (far from the maximum, or with components whose mass vanishes, as at lambda = 0) or the
        step does not rise, it takes SQUAREM-accelerated EM updates instead (``squarem_step``), which rise. Near
        the maximum the ascent is linear at a rate r, so after an increment d the remaining gain is
        d r / (1 - r). It stops when that bound, with r the ratio of the last two increments, the increment itself
        and (where there is one) the density's Newton decrement are below FIT_TOLERANCE, with the lattice's
        quadrature bound met, or when an increment is not positive. It also says whether the maximum is certified:
        at a finite weight, where the Laplace evidence needs a strict maximum, only a Newton decrement below
        FIT_TOLERANCE certifies it; at an edge weight, where the evidence is the log-likelihood, convergence does.
        """
        allowed = _allowed(state.prior, log_smoothing)
        state = self._with_density(state, allowed @ (allowed.T @ state.hyperparameters.coefficients), log_smoothing)
        expectation = self.expectation(state, louis=True)
        previous = np.inf
        while True:
            refined = self._refined(state, expectation)
            if refined.prior.grid_size != state.prior.grid_size:
                state, expectation, previous = refined, self.expectation(refined, louis=True), np.inf
            base = self.penalized(state, expectation)
            candidate = candidate_expectation = None
            newton = self.newton_direction(state, expectation, log_smoothing)
            decrement = np.inf if newton is None else newton[1]
            if newton is not None:
                trial = self._with_density(self.level_step(state, expectation), state.hyperparameters.coefficients + newton[0], log_smoothing)
                trial_expectation = self.trial(trial, True, expectation.grid)
                if trial_expectation is not None and self.penalized(trial, trial_expectation) > base:
                    candidate, candidate_expectation = trial, trial_expectation
            if candidate is None:
                candidate, candidate_expectation = self.squarem_step(state, expectation, log_smoothing)
            increment = self.penalized(candidate, candidate_expectation) - base
            state, expectation = candidate, candidate_expectation
            self.grid = expectation.grid
            edge = not np.all(np.isfinite(log_smoothing))
            newton_certified = newton is not None and decrement < FIT_TOLERANCE
            if increment <= 0.0:
                # Every step rises in exact arithmetic, so the quadrature no longer resolves the gain.
                return state, expectation, edge or newton_certified
            if previous < np.inf:
                rate = increment / previous
                remaining = increment * rate / (1.0 - rate) if 0.0 <= rate < 1.0 else np.inf
                # The density's own remaining gain is its Newton decrement where the curvature allows one.
                if increment < FIT_TOLERANCE and remaining < FIT_TOLERANCE and (newton is None or decrement < FIT_TOLERANCE):
                    return state, expectation, edge or newton_certified
            previous = increment

    def _refined(self, state: _State, expectation: _Expectation) -> _State:
        """Halve the lattice's spacing until its trapezoid in t meets the engine's spacing bound for these
        occasions, h <= pi^2 / ln(1 + 2 sum_j M_j / Z_j / tol) (``spacing_bound``), taken in logs: an occasion far
        outside the density's mass has a majorant ratio past double precision's range.

        At t + i pi/2 a node's kernel N(e; 0, i e^t) has modulus (2 pi e^t)^(-1/2), so each occasion's majorant
        ratio is sum_k pi_k (2 pi s_k)^(-1/2) / f(e), taken at its posterior mean level. A lattice whose matrices
        exceed the memory budget raises ``LatticeTooLarge``.
        """
        residuals = self.residuals(state) - expectation.level_mean[self.occasions.person_index]
        while True:
            log_masses = class_log_density(state.prior, state.hyperparameters.coefficients)[0]
            log_heights = log_masses - 0.5 * (_LOG_TWO_PI + state.prior.log_variance_grid)
            variances = np.exp(state.prior.log_variance_grid)
            peak = float(_logsumexp(log_heights, axis=0))
            # The occasions x nodes kernel values, a chunk of occasions at a time within the budget.
            chunk = max(1, self.working_bytes // (_ENTRY_BYTES * variances.shape[0]))
            log_ratios = np.concatenate([
                np.maximum(peak - _logsumexp(log_heights[None, :] - 0.5 * np.square(part)[:, None] / variances[None, :], axis=1), 0.0)
                for part in np.array_split(residuals, -(-residuals.shape[0] // chunk))
            ])
            log_ratio_sum = float(_logsumexp(log_ratios, axis=0))
            bound = np.pi**2 / np.logaddexp(0.0, np.log(2.0) + log_ratio_sum - np.log(EVIDENCE_TOLERANCE))
            nodes = state.prior.log_variance_grid
            if nodes[1] - nodes[0] <= bound:
                return state
            _check_lattice(2 * nodes.shape[0] - 1, self.working_bytes)
            prior, hyperparameters = halved_lattice(state.prior, state.hyperparameters)
            state = _State(state.fixed_effects, state.level_variance, prior, hyperparameters)

    def evidence(self, state: _State, expectation: _Expectation) -> float:
        masses = np.exp(class_log_density(state.prior, state.hyperparameters.coefficients)[0])
        total = float(expectation.counts.sum())
        observed = total * (np.diag(masses) - np.outer(masses, masses)) - expectation.missing_information
        return laplace_evidence(state.prior, state.hyperparameters.log_smoothing, state.hyperparameters.coefficients, expectation.log_likelihood, observed)

    def smoothing_range(self, state: _State, expectation: _Expectation) -> tuple[float, float]:
        """The log weights double precision resolves (``scale_mixture_ep._smoothing_bounds``): below
        lambda s = sqrt(eps) ||D|| the weakest penalized direction is lost in D's rounding, and above
        lambda s = ||D|| / sqrt(eps) every penalized coefficient is frozen to half of double precision."""
        masses = np.exp(class_log_density(state.prior, state.hyperparameters.coefficients)[0])
        total = float(expectation.counts.sum())
        mapping = state.prior.coefficient_map[: state.prior.grid_size]
        data = mapping.T @ (total * (np.diag(masses) - np.outer(masses, masses)) - expectation.missing_information) @ mapping
        block = state.prior.smoothing_blocks[0]
        data_norm = max(float(np.max(np.abs(np.linalg.eigvalsh(0.5 * (data + data.T))))), np.finfo(np.float64).tiny)
        penalty_eigenvalues = np.linalg.eigvalsh(block.matrix)
        smallest = float(np.min(penalty_eigenvalues[penalty_eigenvalues > _EPSILON * block.matrix.shape[0] * float(penalty_eigenvalues[-1])]))
        return float(np.log(_HALF_PRECISION * data_norm / smallest)), float(np.log(data_norm / (_HALF_PRECISION * smallest)))

    def fit(self, start: _State) -> tuple[_State, _Expectation, float]:
        """The penalty weight maximizing the evidence (``bracketed_maximum`` over its resolvable range, each EM
        warm-started from the nearest weight fitted), moved to 0 or infinity exactly when the ascent reaches an
        end of that range (the engine's edge rule, ``scale_mixture_ep._maximize_evidence``)."""
        if len(start.prior.smoothing_blocks) != 1:
            raise ValueError("the occasion noise density has one class and one roughness penalty")
        fits: dict[float, tuple[_State, _Expectation, float]] = {}
        unevaluated: set[float] = set()

        def at(log_weight: float) -> float:
            if log_weight in unevaluated:
                return -np.inf
            if log_weight not in fits:
                nearest = min(fits, key=lambda known: abs(known - log_weight)) if fits else None
                origin = start if nearest is None else fits[nearest][0]
                try:
                    state, expectation, certified = self.converge(origin, np.array([log_weight]))
                except (MemoryError, FloatingPointError):
                    # Not evaluable within the memory budget, or with finite integrals: never approximated, it
                    # counts as an uncertified maximum.
                    unevaluated.add(log_weight)
                    return -np.inf
                # An uncertified maximum counts as the lowest evidence (the engine's rule).
                fits[log_weight] = (state, expectation, self.evidence(state, expectation) if certified else -np.inf)
            return fits[log_weight][2]

        initial = float(start.hyperparameters.log_smoothing[0])
        at(initial)
        if initial not in fits:
            raise FloatingPointError("the start weight's fit is not evaluable within the memory budget")
        lower, upper = self.smoothing_range(*fits[initial][:2])
        bracketed_maximum(at, initial, lower, upper)
        # The best weight the search evaluated (an uncertified fit anywhere breaks unimodality).
        best = max((weight for weight in fits if np.isfinite(weight)), key=lambda weight: fits[weight][2])
        if fits[best][2] == -np.inf:
            raise FloatingPointError("no penalty weight in the resolvable range reaches a certified maximum")
        if best in (lower, upper):
            # The ascent reached an end of the resolvable range: the weight moves to that edge exactly. The
            # engine (engine-tk) also compares V(infinity) with an interior maximum; that needs the global
            # null-space fit of mixing_density.md B4 (a grid over the log-normal's location and width, then a
            # polish), which this model does not have yet: a local EM there is neither global nor cheap.
            edge = np.inf if best == upper else -np.inf
            at(edge)
            if edge in fits:
                best = edge
        return fits[best]


def bracketed_maximum(function: Callable[[float], float], start: float, lower: float, upper: float) -> float:
    """A maximizer of ``function`` over [lower, upper], for a function unimodal there.

    From ``start`` (clipped into the range) the ascent steps outward, one unit first and growing by the golden
    ratio (the steps set only the search's cost), until the function falls, which brackets the maximum, or the
    range ends while it still rises, which returns that end. The bracket then narrows by golden-section search
    until its certified values span less than EVIDENCE_TOLERANCE or it is half of double precision wide. Values
    of -inf (uncertified maxima) count as the lowest.
    """
    growth = 1.0 / _GOLDEN - 1.0
    origin = float(np.clip(start, lower, upper))
    right, left = min(origin + 1.0, upper), max(origin - 1.0, lower)
    if right > origin and function(right) > function(origin):
        previous, current = origin, right
    elif left < origin and function(left) > function(origin):
        previous, current = origin, left
    else:
        previous = current = origin
    if current != origin:
        while True:
            beyond = float(np.clip(current + growth * (current - previous), lower, upper))
            if beyond == current:
                return current
            if function(beyond) <= function(current):
                break
            previous, current = current, beyond
        low, high = min(previous, beyond), max(previous, beyond)
    else:
        low, high = left, right
    # Golden-section search on [low, high], keeping two interior points.
    inner_low, inner_high = high - (1.0 - _GOLDEN) * (high - low), low + (1.0 - _GOLDEN) * (high - low)
    while high - low > _HALF_PRECISION * (1.0 + abs(high) + abs(low)):
        values = [function(point) for point in (low, inner_low, inner_high, high)]
        certified = [value for value in values if np.isfinite(value)]
        if len(certified) > 1 and max(certified) - min(certified) < EVIDENCE_TOLERANCE:
            break
        if values[1] >= values[2]:
            high, inner_high = inner_high, inner_low
            inner_low = high - (1.0 - _GOLDEN) * (high - low)
        else:
            low, inner_low = inner_low, inner_high
            inner_high = low + (1.0 - _GOLDEN) * (high - low)
    candidates = (low, inner_low, inner_high, high)
    return max(candidates, key=function)


def _result(model: _Model, state: _State, expectation: _Expectation, evidence: float) -> OccasionModelFit:
    return OccasionModelFit(
        exponent=model.exponent,
        fixed_effects=state.fixed_effects,
        level_variance=state.level_variance,
        prior=state.prior,
        hyperparameters=state.hyperparameters,
        level_mean=expectation.level_mean,
        level_posterior_variance=expectation.level_second_moment - np.square(expectation.level_mean),
        log_likelihood=expectation.log_likelihood,
        log_evidence=evidence,
        split_information=expectation.split_information,
        split_magnitude=expectation.split_magnitude,
        split_operations=expectation.split_operations,
    )


def _carried(start: _State, model: _Model, previous: _Model) -> _State:
    """The previous exponent's density carried to this one: its lattice shifted by the log ratio of the two
    transforms' within-person variances, then re-latticed onto this exponent's lattice."""
    fresh = model.start()
    shift = float(np.log(np.var(model.transformed) / np.var(previous.transformed)))
    moved = _density_prior(start.prior.log_variance_grid + shift, start.prior.kernel_top + shift, model.occasions.values.shape[0])
    prior, hyperparameters = relattice(
        moved, start.hyperparameters, fresh.prior.log_variance_grid, fresh.prior.kernel_floor, fresh.prior.kernel_top
    )
    return _State(fresh.fixed_effects, fresh.level_variance, prior, hyperparameters)


def fit_at_exponent(occasions: Occasions, exponent: float, working_bytes: int) -> OccasionModelFit:
    model = _Model(occasions, exponent, working_bytes)
    return _result(model, *model.fit(model.start()))


def fit_occasion_model(occasions: Occasions, working_bytes: int) -> OccasionModelFit:
    """The fit at the Box-Cox exponent maximizing the profile evidence (``bracketed_maximum`` from the identity,
    one unit to the log), each exponent's EM starting from the nearest one already fitted."""
    fits: dict[float, tuple[_Model, tuple[_State, _Expectation, float]]] = {}

    def evidence(exponent: float) -> float:
        if exponent not in fits:
            model = _Model(occasions, exponent, working_bytes)
            fitted = [known for known in fits if fits[known][1] is not None]
            try:
                if fitted:
                    nearest = min(fitted, key=lambda known: abs(known - exponent))
                    previous_model, (previous_state, _expectation, _evidence) = fits[nearest]
                    start = _carried(previous_state, model, previous_model)
                else:
                    start = model.start()
                fits[exponent] = (model, model.fit(start))
            except (FloatingPointError, MemoryError):
                # No certified maximum at this exponent, or none evaluable within the memory budget: it counts as
                # the lowest evidence.
                fits[exponent] = (model, None)
        fit = fits[exponent][1]
        return -np.inf if fit is None else fit[2]

    bracketed_maximum(evidence, 1.0, -np.inf, np.inf)
    middle = max(fits, key=evidence)
    if evidence(middle) == -np.inf:
        raise FloatingPointError("no Box-Cox exponent reaches a certified maximum")
    model, (state, expectation, evidence_value) = fits[middle]
    return _result(model, state, expectation, evidence_value)


def marginal_transform(values: F64Array, design: F64Array) -> tuple[float, F64Array]:
    """The Box-Cox exponent learned from the readings' marginal, and the transformed readings: the target of a
    trait whose level and noise are not separable (no within-person replicates, or a split the data do not
    certify; the lead's ruling), with reliability 1. The noise then stays in the genetic model's residual, where
    it belongs when it cannot be separated.

    The exponent maximizes the Gaussian marginal profile log-likelihood of the readings given the design,
    -(n / 2) log(RSS / n) + (lambda - 1) sum log y (Box and Cox 1964), by ``bracketed_maximum`` from the identity.
    """
    log_values = np.log(values)

    def profile(exponent: float) -> float:
        transformed, _log_jacobian = box_cox(values, exponent)
        coefficients = np.linalg.lstsq(design, transformed, rcond=None)[0]
        residual_sum = float(np.sum(np.square(transformed - design @ coefficients)))
        return -0.5 * values.shape[0] * np.log(residual_sum / values.shape[0]) + (exponent - 1.0) * float(log_values.sum())

    exponent = bracketed_maximum(profile, 1.0, -np.inf, np.inf)
    return exponent, box_cox(values, exponent)[0]


def level_posterior_at(
    occasions: Occasions, exponent: float, fixed_effects: F64Array, level_variance: float, prior: ScaleMixturePrior,
    hyperparameters: MixtureHyperparameters, working_bytes: int,
) -> OccasionModelFit:
    """Every person's exact level posterior at given model parameters, learning nothing: the levels of people
    outside the fit, and the exactness checks. ``prior`` supplies the lattice of ``hyperparameters``."""
    model = _Model(occasions, exponent, working_bytes)
    state = _State(np.asarray(fixed_effects, dtype=np.float64), float(level_variance), prior, hyperparameters)
    expectation = model.expectation(state, louis=False)
    return _result(model, state, expectation, expectation.log_likelihood)

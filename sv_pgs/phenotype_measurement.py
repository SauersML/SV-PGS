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
and its spacing is halved until its quadrature in t meets the evidence tolerance.

Inference is exact in T_i, which is one-dimensional (the lead's ruling): EM over the complete data
(T_i, each occasion's lattice component).
- E-step: p(T_i | z_i) by the trapezoid rule in T. Each person's step is certified by the Trefethen-Weideman
  bound with the mass-weighted modulus on a strip where the integrand is analytic (its width searched per
  person), and the ends by a tail bound (``level_posterior``; math-density, mixing_density.md §11). Duplicated readings, whose integrand spikes,
  need no separate treatment: their modulus bound is large, so their step is small. At every node come each
  occasion's component responsibilities.
- M-step: gamma by weighted least squares (weights E[1/s]), tau^2 = mean E[T_i^2], and the density's
  coefficients maximizing the expected counts' multinomial log-likelihood minus the roughness penalty
  (concave, by Newton).
- The penalty weight maximizes the Laplace evidence of the exact marginal likelihood (the engine's B + S
  form: the penalty's log pseudo-determinant, the null space profiled), whose curvature is the observed
  information by Louis' identity. At the ends of its resolvable range it moves to 0 or infinity exactly
  (the engine's edge rule).

Tolerances hold numerical error below statistical error. Every stop resolves EVIDENCE_TOLERANCE, 1/2 nat of
total evidence (within one posterior standard deviation of the parameters it settles), and each person's
quadrature has relative error 1 - e^(-EVIDENCE_TOLERANCE / n), so the n persons together stay within it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.special import log_ndtr, logsumexp
from scipy.stats import chi2

from sv_pgs._typing import F64Array, I64Array
from sv_pgs.scale_mixture_ep import (
    ROUGHNESS_ORDER,
    MixtureHyperparameters,
    ScaleMixturePrior,
    class_log_density,
    halved_lattice,
    initial_hyperparameters,
    relattice,
    scale_mixture_prior,
    spacing_bound,
)

# One posterior standard deviation of the parameters, in nats (module docstring).
EVIDENCE_TOLERANCE = 0.5
_EPSILON = float(np.finfo(np.float64).eps)
_HALF_PRECISION = _EPSILON**0.5
_LOG_TWO_PI = float(np.log(2.0 * np.pi))
_GOLDEN = 0.5 * (3.0 - np.sqrt(5.0))
# The E-step's float64 arrays per (person, node, occasion, component) entry live at once: the log components, the
# responsibilities, their weighted product and one temporary (logsumexp's, or an einsum's).
_FLOAT64_BYTES = 8
_ENTRY_BYTES = 4 * _FLOAT64_BYTES


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

    @property
    def reliability(self) -> F64Array:
        return 1.0 - self.level_posterior_variance / self.level_variance

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
    admissible_step: F64Array


def _log_components(residuals: F64Array, levels: F64Array, log_masses: F64Array, variances: F64Array) -> F64Array:
    """log pi_k N(r_j - T_n; 0, s_k): persons x nodes x occasions x components."""
    deviations = residuals[:, None, :] - levels[:, :, None]
    return (
        log_masses[None, None, None, :]
        - 0.5 * (_LOG_TWO_PI + np.log(variances))[None, None, None, :]
        - 0.5 * np.square(deviations)[..., None] / variances[None, None, None, :]
    )


def _log_prior(levels: F64Array, level_variance: float) -> F64Array:
    return -0.5 * (_LOG_TWO_PI + np.log(level_variance)) - 0.5 * np.square(levels) / level_variance


def _log_occasion_density(residuals: F64Array, points: F64Array, log_masses: F64Array, variances: F64Array) -> F64Array:
    """log f(r_j - T_p) at one point T_p per person: persons x occasions."""
    return logsumexp(_log_components(residuals, points[:, None], log_masses, variances)[:, 0], axis=2)


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
    with np.errstate(divide="ignore"):  # the m = 0 peak at T = 0 has |T|^0 = 1
        log_peak = np.where(powers > 0.0, powers * np.log(np.abs(peak)), 0.0) - 0.5 * (_LOG_TWO_PI + np.log(level_variance) + np.square(peak) / level_variance)
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


def _log_modulus_bound(
    residuals: F64Array, level_variance: float, log_masses: F64Array, variances: F64Array, half_width: F64Array
) -> F64Array:
    """log of a closed-form bound on the strip modulus M_m(b), m = 0, 1, 2, at each person's b (persons x 3).

    On |Im T| = y < b, |N(e + i y; 0, s)| = N(e; 0, s) e^(y^2 / (2 s)) and |T^m| <= (|x| + b)^m, so with
    w_k = pi_k e^(b^2 / (2 s_k)), A = sum_k w_k, W_j(x) = sum_k w_k N(r_j - x; 0, s_k) and
    q(x) = (|x| + b)^m N(x; 0, tau^2) e^(b^2 / (2 tau^2)), M_m(b) <= integral q prod_j W_j (the mass-weighted
    modulus, mixing_density.md section 11). Hoelder's inequality over the J occasions bounds that by
    prod_j (integral q W_j^J)^(1 / J), and the power-mean inequality inside each mixture by
    W_j^J <= A^(J - 1) sum_k w_k N(r_j - x; 0, s_k)^J. Each term is closed-form:
    N(e; 0, s)^J = (2 pi s)^(-(J - 1) / 2) J^(-1/2) N(e; 0, s / J), and
    N(x; 0, tau^2) N(r - x; 0, s / J) = N(r; 0, tau^2 + s / J) N(x; mu, v), under which E(|X| + b)^m is a
    folded-normal moment. The bound is exact at J = 1, tight when the occasions agree (duplicated readings), and
    needs no quadrature, so it cannot miss a narrow component the nodes step over.
    """
    occasion_count = residuals.shape[1]
    widening = 0.5 * np.square(half_width)
    log_weights = log_masses[None, :] + widening[:, None] / variances[None, :]
    log_total = logsumexp(log_weights, axis=1)
    narrowed = variances / occasion_count
    marginal = level_variance + narrowed
    log_marginal = -0.5 * (_LOG_TWO_PI + np.log(marginal) + np.square(residuals[:, :, None]) / marginal)
    mean = residuals[:, :, None] * (level_variance / marginal)
    variance = level_variance * narrowed / marginal
    scale = np.sqrt(variance)
    folded = scale * np.sqrt(2.0 / np.pi) * np.exp(-0.5 * np.square(mean) / variance) + mean * -np.expm1(np.log(2.0) + log_ndtr(-mean / scale))
    offset = half_width[:, None, None]
    log_moments = np.log(np.stack(
        [np.ones_like(mean), folded + offset, np.square(mean) + variance + 2.0 * offset * folded + np.square(offset)], axis=3
    ))
    log_power_terms = (
        log_weights[:, None, :, None]
        - 0.5 * (occasion_count - 1) * (_LOG_TWO_PI + np.log(variances))[None, None, :, None]
        - 0.5 * np.log(occasion_count)
        + log_marginal[..., None]
        + log_moments
    )
    log_powers = (occasion_count - 1) * log_total[:, None, None] + logsumexp(log_power_terms, axis=2)
    return (widening / level_variance)[:, None] + log_powers.sum(axis=1) / occasion_count


class PieceTooLarge(MemoryError):
    """An E-step piece whose persons x nodes x J x K working arrays exceed the memory budget."""


def _log_moment_sums(log_integrand: F64Array, levels: F64Array, steps: F64Array, offset: F64Array | float) -> F64Array:
    """log of h sum_n (|T_n| + offset)^m exp(log_integrand) for m = 0, 1, 2 (persons x 3; offset per person)."""
    with np.errstate(divide="ignore"):  # the node at T = 0 has |T|^m = 0 for m > 0
        log_magnitude = np.log(np.abs(levels) + np.reshape(offset, (-1, 1)))
    log_powers = np.stack([np.zeros_like(log_magnitude), log_magnitude, 2.0 * log_magnitude], axis=2)
    return logsumexp(log_integrand[:, :, None] + log_powers, axis=1) + np.log(steps)[:, None]


def _level_grid(
    residuals: F64Array, level_variance: float, log_masses: F64Array, variances: F64Array, steps: F64Array,
    centres: F64Array, reach: F64Array, relative_tolerance: float, working_bytes: int,
) -> tuple[F64Array, np.ndarray, F64Array, F64Array]:
    """Trapezoid nodes on each person's grid c_p + n h_p (persons x nodes, a validity mask), the log components
    at them (``_log_components``) and the log integrals of |T|^m F, m = 0, 1, 2 (persons x 3).

    Each side starts ``reach`` past the centre and doubles until every one of its tail bounds (``_log_tail``,
    which bounds the omitted nodes' terms) is at most a quarter of relative_tolerance times its integral so far.
    The centre and reach set only the cost.
    """
    upper = np.maximum(np.ceil(reach / steps), 1.0)
    lower = -upper
    while True:
        counts = (upper - lower).astype(np.int64) + 1
        offsets = np.arange(int(counts.max()))[None, :]
        valid = offsets < counts[:, None]
        levels = centres[:, None] + (lower[:, None] + np.minimum(offsets, counts[:, None] - 1)) * steps[:, None]
        if levels.size * residuals.shape[1] * variances.shape[0] * _ENTRY_BYTES > working_bytes:
            raise PieceTooLarge(f"{levels.shape[0]} persons need {levels.shape[1]} level nodes each")
        log_components = _log_components(residuals, levels, log_masses, variances)
        per_occasion = logsumexp(log_components, axis=3)
        log_integrand = np.where(valid, _log_prior(levels, level_variance) + per_occasion.sum(axis=2), -np.inf)
        log_totals = _log_moment_sums(log_integrand, levels, steps, 0.0)
        if not np.all(np.isfinite(log_totals[:, 0])):
            raise FloatingPointError("a person's level integral is not finite")
        bound = np.log(0.25 * relative_tolerance) + log_totals
        low_edge, high_edge = centres + lower * steps, centres + upper * steps
        grow_lower = np.any(_log_tail(residuals, low_edge, level_variance, log_masses, variances, steps, upper=False) > bound, axis=1)
        grow_upper = np.any(_log_tail(residuals, high_edge, level_variance, log_masses, variances, steps, upper=True) > bound, axis=1)
        if not (np.any(grow_lower) or np.any(grow_upper)):
            return levels, valid, log_components, log_totals
        lower = np.where(grow_lower, 2.0 * lower, lower)
        upper = np.where(grow_upper, 2.0 * upper, upper)


def level_posterior(
    residuals: F64Array, level_variance: float, log_masses: F64Array, variances: F64Array, relative_tolerance: float,
    steps: F64Array | None, centres: F64Array | None, louis: bool, working_bytes: int,
) -> LevelPosterior:
    """The exact E-step for persons with J occasions each (``residuals`` is persons x J).

    L_i and the absolute moment integrals I_m of |T| F and T^2 F each have relative error at most
    ``relative_tolerance``: half from the trapezoid rule, a quarter from each truncated tail. The trapezoid
    rule's error is at most 2 M_m(b) / (e^(2 pi b / h) - 1) for any strip |Im T| < b (Trefethen and Weideman
    2014, Theorem 5.1), with M_m(b) bounded in closed form (``_log_modulus_bound``), so a person's step h is
    certified when h <= max_b min_m 2 pi b / ln(1 + 2 M_m(b) / (eps I_m)), I_m at least its lattice sum over
    1 + relative_tolerance.

    Every b is valid, so b only sets the cost. It is searched on a ladder of halvings between the strips that are
    optimal for Gaussian occasions of the largest and of the smallest variance (``strip_half_width``). ``steps``
    (None or NaN: ``gaussian_step`` at the density's harmonic-mean variance) shrink to the certified step until
    every person's is. The grid is centred at ``centres`` (None or NaN: the Gaussian posterior mean at that
    variance) and starts one Gaussian posterior standard deviation to either side; both set only the cost.
    """
    persons, occasion_count = residuals.shape
    trapezoid_share = 0.5 * relative_tolerance
    narrowest = strip_half_width(level_variance, occasion_count, float(variances.min()), trapezoid_share)
    widest = strip_half_width(level_variance, occasion_count, float(variances.max()), trapezoid_share)
    harmonic = 1.0 / float(np.exp(logsumexp(log_masses - np.log(variances))))
    start = gaussian_step(level_variance, occasion_count, harmonic, trapezoid_share)
    steps = np.full(persons, start) if steps is None else np.where(np.isnan(steps), start, steps)
    precision = 1.0 / level_variance + occasion_count / harmonic
    gaussian_mean = residuals.sum(axis=1) / harmonic / precision
    centres = gaussian_mean if centres is None else np.where(np.isnan(centres), gaussian_mean, centres)
    reach = np.full(persons, 1.0 / np.sqrt(precision))
    ladder = np.maximum(widest * np.exp2(-np.arange(int(np.ceil(np.log2(widest / narrowest))) + 1)), narrowest)
    log_moduli = [_log_modulus_bound(residuals, level_variance, log_masses, variances, np.full(persons, width)) for width in ladder]
    while True:
        levels, valid, log_components, log_totals = _level_grid(
            residuals, level_variance, log_masses, variances, steps, centres, reach, relative_tolerance, working_bytes
        )
        log_lower = log_totals - np.log1p(relative_tolerance)
        admissible = np.max([
            np.min(2.0 * np.pi * width / np.logaddexp(0.0, np.log(2.0) + log_modulus - log_lower - np.log(trapezoid_share)), axis=1)
            for width, log_modulus in zip(ladder, log_moduli)
        ], axis=0)
        uncertified = steps > admissible
        if not np.any(uncertified):
            break
        steps = np.where(uncertified, admissible, steps)
    per_occasion = logsumexp(log_components, axis=3)
    log_integrand = np.where(valid, _log_prior(levels, level_variance) + per_occasion.sum(axis=2), -np.inf)
    log_sum = logsumexp(log_integrand, axis=1)
    weights = np.exp(log_integrand - log_sum[:, None])
    responsibility = np.exp(log_components - per_occasion[..., None])
    weighted = weights[:, :, None, None] * responsibility
    grid_size = variances.shape[0]
    # The sums over persons, nodes and occasions are matrix products over their flattened entries (BLAS).
    flat_weighted = weighted.reshape(-1, grid_size)
    counts = flat_weighted.sum(axis=0)
    missing = np.zeros((grid_size, grid_size))
    if louis:
        # Var(C_i | z_i) = E_T[sum_j (diag rho_j - rho_j rho_j')] + Var_T(sum_j rho_j(T)).
        node_sums = responsibility.sum(axis=2)
        mean_sums = np.matmul(weights[:, None, :], node_sums)[:, 0, :]
        flat_sums = node_sums.reshape(-1, grid_size)
        missing = (
            np.diag(counts)
            - flat_weighted.T @ responsibility.reshape(-1, grid_size)
            + (flat_sums * weights.reshape(-1, 1)).T @ flat_sums
            - mean_sums.T @ mean_sums
        )
    node_precision = weighted @ (1.0 / variances)
    return LevelPosterior(
        log_likelihood=log_totals[:, 0],
        level_mean=np.sum(weights * levels, axis=1),
        level_second_moment=np.sum(weights * np.square(levels), axis=1),
        counts=counts,
        occasion_precision=node_precision.sum(axis=1),
        occasion_shift=np.sum(node_precision * levels[:, :, None], axis=1),
        missing_information=missing,
        admissible_step=admissible,
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


def laplace_evidence(
    prior: ScaleMixturePrior, log_smoothing: F64Array, coefficients: F64Array, log_likelihood: float, observed_information: F64Array
) -> float:
    """V = l(x) - 1/2 x' S x + 1/2 log|S|_+ - 1/2 log|-H| + 1/2 log|N'(-H)N| (the engine's B + S form).

    -H = M' I M + S in x, with I the observed information in the density's log values (Louis) and N the
    profiled null space. At an infinite weight x lies in the null space; at an infinite or zero weight every
    allowed direction is profiled, so V is the log-likelihood there.
    """
    if not np.all(np.isfinite(log_smoothing)):
        return log_likelihood
    mapping = prior.coefficient_map[: prior.grid_size]
    penalty = _penalty_matrix(prior, log_smoothing)
    eigenvalues = np.linalg.eigvalsh(penalty)
    kept = eigenvalues > _EPSILON * penalty.shape[0] * max(float(eigenvalues[-1]), np.finfo(np.float64).tiny)
    negative_hessian = mapping.T @ observed_information @ mapping + penalty
    sign, log_determinant = np.linalg.slogdet(negative_hessian)
    null = prior.null_basis
    null_sign, null_log_determinant = np.linalg.slogdet(null.T @ negative_hessian @ null)
    if sign <= 0.0 or null_sign <= 0.0:
        # Not a strict maximum: the Laplace evidence is undefined, and the search counts it as the lowest value.
        return -np.inf
    return (
        log_likelihood
        - 0.5 * float(coefficients @ penalty @ coefficients)
        + 0.5 * float(np.sum(np.log(eigenvalues[kept])))
        - 0.5 * float(log_determinant)
        + 0.5 * float(null_log_determinant)
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
        # Each person's last admissible trapezoid step and level mean, the start of their next E-step (NaN: none yet).
        self.steps = np.full(occasions.person_count, np.nan)
        self.centres = np.full(occasions.person_count, np.nan)

    def start(self) -> _State:
        """Least-squares fixed effects, a robust level variance, and the engine's start density on the lattice
        from the resolution variance to the largest squared within-person deviation (past it by its width), with
        at least the engine's minimum of ROUGHNESS_ORDER + 2 nodes (``scale_mixture_prior``)."""
        occasions = self.occasions
        fixed_effects = np.linalg.lstsq(occasions.design, self.transformed, rcond=None)[0]
        residuals = self.transformed - occasions.design @ fixed_effects
        counts = np.bincount(occasions.person_index).astype(np.float64)
        person_mean = np.bincount(occasions.person_index, weights=residuals) / counts
        deviations = residuals - person_mean[occasions.person_index]
        repeated = counts[occasions.person_index] > 1
        largest = float(np.max(np.square(deviations[repeated]))) if np.any(repeated) else float(np.max(np.square(residuals)))
        # The person means' variance by their median absolute deviation, which gross errors barely move: it
        # overstates tau^2 by the noise of a mean, which the EM removes.
        level_variance = float(np.median(np.square(person_mean - np.median(person_mean)))) / float(chi2.median(1))
        if not level_variance > 0.0:
            raise ValueError("no between-person variance: most persons' mean readings are equal")
        floor = float(np.log(self.resolution_variance))
        top = float(np.log(max(largest, self.resolution_variance)))
        spacing = spacing_bound(float(occasions.values.shape[0]), EVIDENCE_TOLERANCE)
        extent = max(top + (top - floor), floor + (ROUGHNESS_ORDER + 1) * spacing)
        prior = _density_prior(np.arange(floor, extent + spacing, spacing), top, occasions.values.shape[0])
        return _State(fixed_effects, level_variance, prior, initial_hyperparameters(prior))

    def residuals(self, state: _State) -> F64Array:
        return self.transformed - self.occasions.design @ state.fixed_effects

    def expectation(self, state: _State, louis: bool) -> _Expectation:
        """The exact E-step over every person, in pieces whose working arrays fit ``working_bytes``."""
        occasions = self.occasions
        residuals = self.residuals(state)
        log_masses = class_log_density(state.prior, state.hyperparameters.coefficients)[0]
        variances = np.exp(state.prior.log_variance_grid)
        level_mean, level_second = np.empty(occasions.person_count), np.empty(occasions.person_count)
        precision, shift = np.empty(residuals.shape[0]), np.empty(residuals.shape[0])
        counts = np.zeros(variances.shape[0])
        missing = np.zeros((variances.shape[0], variances.shape[0]))
        log_likelihood = self.log_jacobian
        pending = [rows for rows in self.groups]
        while pending:
            piece = pending.pop()
            persons = occasions.person_index[piece[:, 0]]
            try:
                posterior = level_posterior(
                    residuals[piece], state.level_variance, log_masses, variances, self.relative_tolerance,
                    self.steps[persons], self.centres[persons], louis, self.working_bytes,
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
            self.steps[persons] = posterior.admissible_step
            self.centres[persons] = posterior.level_mean
        return _Expectation(log_likelihood, level_mean, level_second, counts, precision, shift, missing)

    def maximization(self, state: _State, expectation: _Expectation, log_smoothing: F64Array) -> _State:
        """gamma by weighted least squares (weights E[1/s]), tau^2 = mean E[T^2], and the density."""
        design = self.occasions.design
        weighted = design * expectation.occasion_precision[:, None]
        fixed_effects = np.linalg.solve(design.T @ weighted, weighted.T @ self.transformed - design.T @ expectation.occasion_shift)
        coefficients = maximize_count_density(state.prior, log_smoothing, expectation.counts, state.hyperparameters.coefficients)
        return _State(
            fixed_effects, float(np.mean(expectation.level_second_moment)), state.prior, MixtureHyperparameters(coefficients, log_smoothing.copy())
        )

    def penalized(self, state: _State, expectation: _Expectation) -> float:
        coefficients = state.hyperparameters.coefficients
        return expectation.log_likelihood - 0.5 * float(coefficients @ _penalty_matrix(state.prior, state.hyperparameters.log_smoothing) @ coefficients)

    def converge(self, state: _State, log_smoothing: F64Array) -> tuple[_State, _Expectation]:
        """EM to its fixed point at one penalty weight; returns the state and its E-step with Louis' information.

        Near its fixed point EM ascends at a linear rate r, so after an increment d the remaining gain is
        d r / (1 - r). It stops when that bound, with r the ratio of the last two increments, and the increment
        itself are below EVIDENCE_TOLERANCE, with the lattice's quadrature bound met.
        """
        allowed = _allowed(state.prior, log_smoothing)
        state = _State(
            state.fixed_effects, state.level_variance, state.prior,
            MixtureHyperparameters(allowed @ (allowed.T @ state.hyperparameters.coefficients), log_smoothing.copy()),
        )
        expectation = self.expectation(state, louis=False)
        previous = np.inf
        while True:
            refined = self._refined(state, expectation)
            if refined.prior.grid_size != state.prior.grid_size:
                state, expectation, previous = refined, self.expectation(refined, louis=False), np.inf
            candidate = self.maximization(state, expectation, log_smoothing)
            candidate_expectation = self.expectation(candidate, louis=False)
            increment = self.penalized(candidate, candidate_expectation) - self.penalized(state, expectation)
            state, expectation = candidate, candidate_expectation
            rate = increment / previous if 0.0 < previous < np.inf else 0.0
            if increment < EVIDENCE_TOLERANCE and 0.0 <= rate < 1.0 and increment * rate / (1.0 - rate) < EVIDENCE_TOLERANCE:
                return state, self.expectation(state, louis=True)
            previous = increment

    def _refined(self, state: _State, expectation: _Expectation) -> _State:
        """Halve the lattice's spacing until its trapezoid in t meets ``spacing_bound`` for these occasions.

        At t + i pi/2 a node's kernel N(e; 0, i e^t) has modulus (2 pi e^t)^(-1/2), so each occasion's majorant
        ratio is sum_k pi_k (2 pi s_k)^(-1/2) / f(e), taken at its posterior mean level.
        """
        while True:
            log_masses = class_log_density(state.prior, state.hyperparameters.coefficients)[0]
            log_heights = log_masses - 0.5 * (_LOG_TWO_PI + state.prior.log_variance_grid)
            variances = np.exp(state.prior.log_variance_grid)
            residuals = self.residuals(state) - expectation.level_mean[self.occasions.person_index]
            log_density = logsumexp(log_heights[None, :] - 0.5 * np.square(residuals)[:, None] / variances[None, :], axis=1)
            ratio_sum = float(np.sum(np.exp(np.maximum(float(logsumexp(log_heights)) - log_density, 0.0))))
            nodes = state.prior.log_variance_grid
            if nodes[1] - nodes[0] <= spacing_bound(ratio_sum, EVIDENCE_TOLERANCE):
                return state
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
        warm-started from the nearest weight fitted), moved to 0 or infinity exactly when the maximum sits at an
        end of that range (the engine's edge rule)."""
        if len(start.prior.smoothing_blocks) != 1:
            raise ValueError("the occasion noise density has one class and one roughness penalty")
        fits: dict[float, tuple[_State, _Expectation, float]] = {}

        def at(log_weight: float) -> float:
            if log_weight not in fits:
                nearest = min(fits, key=lambda known: abs(known - log_weight)) if fits else None
                origin = start if nearest is None else fits[nearest][0]
                state, expectation = self.converge(origin, np.array([log_weight]))
                fits[log_weight] = (state, expectation, self.evidence(state, expectation))
            return fits[log_weight][2]

        initial = float(start.hyperparameters.log_smoothing[0])
        at(initial)
        lower, upper = self.smoothing_range(*fits[initial][:2])
        best = bracketed_maximum(at, float(np.clip(initial, lower, upper)), lower, upper)
        if best in (lower, upper):
            edge = np.inf if best == upper else -np.inf
            at(edge)
            if fits[edge][2] >= fits[best][2]:
                best = edge
        return fits[best]


def bracketed_maximum(function: Callable[[float], float], start: float, lower: float, upper: float) -> float:
    """A maximizer of ``function`` over [lower, upper]: a bracket grown from ``start`` by golden-ratio steps,
    starting one unit out (the step sets only the search's cost), then golden-section search until the bracket's
    values span less than EVIDENCE_TOLERANCE, or it is half of double precision wide. An end of the range is
    returned when the function still rises there. Values of -inf (uncertified maxima) count as the lowest."""
    growth = 1.0 / _GOLDEN - 1.0
    forward, backward = float(min(start + 1.0, upper)), float(max(start - 1.0, lower))
    centre = function(start)
    if forward > start and function(forward) >= centre:
        previous, middle = start, forward
    elif backward < start and function(backward) > centre:
        previous, middle = start, backward
    else:
        previous = middle = None
        low, high = backward, forward
    if middle is not None:
        while True:
            beyond = float(np.clip(middle + growth * (middle - previous), lower, upper))
            if beyond == middle:
                return middle
            if function(beyond) < function(middle):
                low, high = min(previous, beyond), max(previous, beyond)
                break
            previous, middle = middle, beyond
    else:
        middle = start
    while high - low > _HALF_PRECISION * (1.0 + abs(high) + abs(low)):
        if max(function(low), function(middle), function(high)) - min(function(low), function(high)) < EVIDENCE_TOLERANCE:
            break
        if high - middle > middle - low:
            trial = middle + _GOLDEN * (high - middle)
            if function(trial) > function(middle):
                low, middle = middle, trial
            else:
                high = trial
        else:
            trial = middle - _GOLDEN * (middle - low)
            if function(trial) > function(middle):
                high, middle = middle, trial
            else:
                low = trial
    return middle


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
            if fits:
                nearest = min(fits, key=lambda known: abs(known - exponent))
                previous_model, (previous_state, _expectation, _evidence) = fits[nearest]
                start = _carried(previous_state, model, previous_model)
            else:
                start = model.start()
            fits[exponent] = (model, model.fit(start))
        return fits[exponent][1][2]

    middle = bracketed_maximum(evidence, 1.0, -np.inf, np.inf)
    model, (state, expectation, evidence_value) = fits[middle]
    return _result(model, state, expectation, evidence_value)


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

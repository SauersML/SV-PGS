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
- E-step: p(T_i | z_i) by the trapezoid rule in T. The step comes from the Trefethen-Weideman bound on the
  strip where the integrand is analytic (``trapezoid_step``), and the ends from the integrand's monotone
  tails. At every node come each occasion's component responsibilities.
- M-step: gamma by weighted least squares (weights E[1/s]), tau^2 = mean E[T_i^2], and the density's
  coefficients maximizing the expected counts' multinomial log-likelihood minus the roughness penalty
  (concave, by Newton).
- The penalty weight maximizes the Laplace evidence of the exact marginal likelihood (the engine's B + S
  form: the penalty's log pseudo-determinant, the null space profiled), whose curvature is the observed
  information by Louis' identity. At the ends of its resolvable range it moves to 0 or infinity exactly
  (the engine's edge rule).

Tolerances hold numerical error below statistical error. Every stop resolves EVIDENCE_TOLERANCE, 1/2 nat of
total evidence (within one posterior standard deviation of the parameters it settles), and each person's
quadrature has relative error EVIDENCE_TOLERANCE / n, so the n persons together stay within it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from sv_pgs._typing import F64Array, I64Array
from sv_pgs.scale_mixture_ep import (
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
# The E-step's working arrays per (person, node, occasion, component) entry: components, responsibilities and
# their weighted product, in float64.
_ENTRY_BYTES = 3 * 8


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


def trapezoid_step(level_variance: float, occasion_count: int, smallest_variance: float, relative_tolerance: float) -> float:
    """The trapezoid step in T with relative error at most ``relative_tolerance`` for a person with J occasions.

    On the strip |Im T| < b every Gaussian factor N(x + i y; mu, v) has modulus N(x; mu, v) e^(y^2 / (2 v)), so
    the integrand's absolute integral along the strip is at most e^(b^2 c / 2) times its real integral, with
    c = 1 / tau^2 + J / s_min. The trapezoid rule's relative error is then at most
    2 e^(b^2 c / 2) / (e^(2 pi b / h) - 1) (Trefethen and Weideman 2014, Theorem 5.1). At
    b = sqrt(2 ln(2 / eps) / c), the b that maximizes the admissible h, the step is
    h = 2 pi b / ln(1 + 2 e^(b^2 c / 2) / eps).
    """
    curvature = 1.0 / level_variance + occasion_count / smallest_variance
    half_width = np.sqrt(2.0 * np.log(2.0 / relative_tolerance) / curvature)
    return float(2.0 * np.pi * half_width / np.log1p(2.0 * np.exp(0.5 * half_width * half_width * curvature) / relative_tolerance))


@dataclass(frozen=True)
class LevelPosterior:
    """The exact E-step for a set of persons: level moments, log-likelihoods, and the sufficient statistics of
    the M-step (counts per component, each occasion's E[1/s] and E[T/s]) and of Louis' identity."""

    log_likelihood: F64Array
    level_mean: F64Array
    level_second_moment: F64Array
    counts: F64Array
    occasion_precision: F64Array
    occasion_shift: F64Array
    missing_information: F64Array


def _log_components(residuals: F64Array, levels: F64Array, log_masses: F64Array, variances: F64Array) -> F64Array:
    """log pi_k N(r_j - T_n; 0, s_k): persons x nodes x occasions x components."""
    deviations = residuals[:, None, :] - levels[:, :, None]
    return (
        log_masses[None, None, None, :]
        - 0.5 * (_LOG_TWO_PI + np.log(variances))[None, None, None, :]
        - 0.5 * np.square(deviations)[..., None] / variances[None, None, None, :]
    )


def _log_integrand(residuals: F64Array, levels: F64Array, level_variance: float, log_masses: F64Array, variances: F64Array) -> F64Array:
    """log N(T_n; 0, tau^2) prod_j f(r_j - T_n): persons x nodes."""
    per_occasion = logsumexp(_log_components(residuals, levels, log_masses, variances), axis=3)
    prior = -0.5 * (_LOG_TWO_PI + np.log(level_variance)) - 0.5 * np.square(levels) / level_variance
    return prior + per_occasion.sum(axis=2)


def _grid(lower: F64Array, upper: F64Array, step: float) -> tuple[F64Array, np.ndarray]:
    counts = (upper - lower).astype(np.int64) + 1
    offsets = np.arange(int(counts.max()))[None, :]
    valid = offsets < counts[:, None]
    return (lower[:, None] + np.minimum(offsets, counts[:, None] - 1)) * step, valid


def _level_nodes(
    residuals: F64Array, level_variance: float, log_masses: F64Array, variances: F64Array, step: float, relative_tolerance: float
) -> tuple[F64Array, np.ndarray]:
    """Trapezoid nodes on the grid n h (persons x nodes, with a validity mask), past each person's tails.

    Beyond R >= max(0, max_j r_j) every factor of the integrand decreases, so its integral from R on is at most
    F(R) tau^2 / R (the Gaussian Mills ratio); likewise below min(0, min_j r_j). Each side doubles until that
    bound is below a quarter of relative_tolerance times the integral so far.
    """
    lower = np.floor(np.minimum(residuals.min(axis=1), 0.0) / step) - 1.0
    upper = np.ceil(np.maximum(residuals.max(axis=1), 0.0) / step) + 1.0
    while True:
        levels, valid = _grid(lower, upper, step)
        log_integrand = np.where(valid, _log_integrand(residuals, levels, level_variance, log_masses, variances), -np.inf)
        log_total = logsumexp(log_integrand, axis=1) + np.log(step)
        if not np.all(np.isfinite(log_total)):
            raise FloatingPointError("a person's level integral is not finite")
        last = valid.sum(axis=1) - 1
        rows = np.arange(levels.shape[0])
        bound = np.log(0.25 * relative_tolerance) + log_total
        grow_lower = log_integrand[:, 0] + np.log(level_variance) - np.log(-levels[:, 0]) > bound
        grow_upper = log_integrand[rows, last] + np.log(level_variance) - np.log(levels[rows, last]) > bound
        if not (np.any(grow_lower) or np.any(grow_upper)):
            return levels, valid
        lower = np.where(grow_lower, 2.0 * lower, lower)
        upper = np.where(grow_upper, 2.0 * upper, upper)


def level_posterior(
    residuals: F64Array, level_variance: float, log_masses: F64Array, variances: F64Array, relative_tolerance: float, louis: bool
) -> LevelPosterior:
    """The exact E-step for persons with J occasions each (``residuals`` is persons x J); its relative error is
    at most ``relative_tolerance``: half from the trapezoid rule, a quarter from each truncated tail."""
    step = trapezoid_step(level_variance, residuals.shape[1], float(variances.min()), 0.5 * relative_tolerance)
    levels, valid = _level_nodes(residuals, level_variance, log_masses, variances, step, relative_tolerance)
    log_components = _log_components(residuals, levels, log_masses, variances)
    per_occasion = logsumexp(log_components, axis=3)
    prior = -0.5 * (_LOG_TWO_PI + np.log(level_variance)) - 0.5 * np.square(levels) / level_variance
    log_integrand = np.where(valid, prior + per_occasion.sum(axis=2), -np.inf)
    log_total = logsumexp(log_integrand, axis=1)
    weights = np.exp(log_integrand - log_total[:, None])
    responsibility = np.exp(log_components - per_occasion[..., None])
    weighted = weights[:, :, None, None] * responsibility
    inverse_variances = 1.0 / variances
    counts = weighted.sum(axis=(0, 1, 2))
    grid_size = variances.shape[0]
    missing = np.zeros((grid_size, grid_size))
    if louis:
        # Var(C_i | z_i) = E_T[sum_j (diag rho_j - rho_j rho_j')] + Var_T(sum_j rho_j(T)).
        node_sums = responsibility.sum(axis=2)
        mean_sums = np.einsum("pn,pnk->pk", weights, node_sums)
        missing = (
            np.diag(counts)
            - np.einsum("pnjk,pnjl->kl", weighted, responsibility)
            + np.einsum("pn,pnk,pnl->kl", weights, node_sums, node_sums)
            - mean_sums.T @ mean_sums
        )
    return LevelPosterior(
        log_likelihood=log_total + np.log(step),
        level_mean=np.sum(weights * levels, axis=1),
        level_second_moment=np.sum(weights * np.square(levels), axis=1),
        counts=counts,
        occasion_precision=np.einsum("pnjk,k->pj", weighted, inverse_variances),
        occasion_shift=np.einsum("pnjk,k,pn->pj", weighted, inverse_variances, levels),
        missing_information=missing,
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
        raise FloatingPointError("the penalized maximum is not strict: -H is not positive definite")
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
        self.relative_tolerance = EVIDENCE_TOLERANCE / occasions.person_count

    def start(self) -> _State:
        """Least-squares fixed effects, the moment level variance, and the engine's start density on the lattice
        from the resolution variance to the largest squared within-person deviation (past it by its width)."""
        occasions = self.occasions
        fixed_effects = np.linalg.lstsq(occasions.design, self.transformed, rcond=None)[0]
        residuals = self.transformed - occasions.design @ fixed_effects
        counts = np.bincount(occasions.person_index).astype(np.float64)
        person_mean = np.bincount(occasions.person_index, weights=residuals) / counts
        deviations = residuals - person_mean[occasions.person_index]
        repeated = counts[occasions.person_index] > 1
        largest = float(np.max(np.square(deviations[repeated]))) if np.any(repeated) else float(np.max(np.square(residuals)))
        within = float(np.sum(np.square(deviations))) / max(float(counts.sum() - counts.shape[0]), 1.0)
        level_variance = float(np.var(person_mean)) - within * float(np.mean(1.0 / counts))
        if not level_variance > 0.0:
            raise ValueError("no between-person variance: person means carry no signal beyond occasion noise")
        floor = float(np.log(self.resolution_variance))
        top = float(np.log(max(largest, self.resolution_variance)))
        spacing = spacing_bound(float(occasions.values.shape[0]), EVIDENCE_TOLERANCE)
        extent = max(top + (top - floor), floor + 4.0 * spacing)
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
        for rows in self.groups:
            for piece in self._pieces(rows, residuals, state.level_variance, variances):
                posterior = level_posterior(residuals[piece], state.level_variance, log_masses, variances, self.relative_tolerance, louis)
                persons = occasions.person_index[piece[:, 0]]
                log_likelihood += float(posterior.log_likelihood.sum())
                level_mean[persons] = posterior.level_mean
                level_second[persons] = posterior.level_second_moment
                precision[piece] = posterior.occasion_precision
                shift[piece] = posterior.occasion_shift
                counts += posterior.counts
                missing += posterior.missing_information
        return _Expectation(log_likelihood, level_mean, level_second, counts, precision, shift, missing)

    def _pieces(self, rows: I64Array, residuals: F64Array, level_variance: float, variances: F64Array) -> Iterator[I64Array]:
        """Pieces of one occasion-count group whose persons x nodes x J x K entries fit ``working_bytes``, the node
        count taken from each person's residual range and step (the tails add a bounded number of doublings)."""
        step = trapezoid_step(level_variance, rows.shape[1], float(variances.min()), 0.5 * self.relative_tolerance)
        spans = (np.maximum(residuals[rows].max(axis=1), 0.0) - np.minimum(residuals[rows].min(axis=1), 0.0)) / step
        entries = 4.0 * (spans.max() + 3.0) * rows.shape[1] * variances.shape[0] * _ENTRY_BYTES
        if entries > self.working_bytes:
            raise MemoryError(
                f"a person's certified level quadrature needs {entries:.3g} bytes, beyond the {self.working_bytes} available"
            )
        size = max(1, int(self.working_bytes // entries))
        for start in range(0, rows.shape[0], size):
            yield rows[start : start + size]

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
        """The penalty weight maximizing the evidence, by golden-section search over its resolvable range with
        warm-started EM, moving to 0 or infinity exactly when the maximum sits at an end of the range."""
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

        initial = start.hyperparameters.log_smoothing[0]
        at(initial)
        lower, upper = self.smoothing_range(*fits[initial][:2])
        best = _golden_maximum(at, lower, upper)
        if best in (lower, upper):
            at(np.inf if best == upper else -np.inf)
            best = np.inf if best == upper else -np.inf
        return fits[best]


def _golden_maximum(function: Callable[[float], float], lower: float, upper: float) -> float:
    """The argmax of ``function`` over [lower, upper] by golden-section search, down to a bracket whose values
    span less than EVIDENCE_TOLERANCE; an end of the range is returned when the maximum sits there."""
    left = lower + _GOLDEN * (upper - lower)
    right = upper - _GOLDEN * (upper - lower)
    while True:
        values = {point: function(point) for point in (lower, left, right, upper)}
        if max(values.values()) - min(values.values()) < EVIDENCE_TOLERANCE:
            return max(values, key=values.__getitem__)
        if values[left] >= values[right]:
            upper, right = right, left
            left = lower + _GOLDEN * (upper - lower)
        else:
            lower, left = left, right
            right = upper - _GOLDEN * (upper - lower)
        if upper - lower <= _HALF_PRECISION * (1.0 + abs(upper) + abs(lower)):
            return max(values, key=values.__getitem__)


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
    """The fit at the Box-Cox exponent maximizing the profile evidence, by golden-section search from the log (0)
    and the identity (1): the bracket grows until its middle point is the best of three, then narrows until its
    evidence spans less than EVIDENCE_TOLERANCE. Each exponent's EM starts from the nearest one already fitted."""
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

    low, middle = 0.0, 1.0
    if evidence(middle) < evidence(low):
        low, middle = middle, low
    high = middle + (middle - low) * (1.0 - _GOLDEN) / _GOLDEN
    while evidence(high) > evidence(middle):
        low, middle, high = middle, high, high + (high - middle) * (1.0 - _GOLDEN) / _GOLDEN
    while max(evidence(low), evidence(middle), evidence(high)) - min(evidence(low), evidence(high)) >= EVIDENCE_TOLERANCE:
        if abs(high - middle) > abs(middle - low):
            trial = middle + _GOLDEN * (high - middle)
            if evidence(trial) > evidence(middle):
                low, middle = middle, trial
            else:
                high = trial
        else:
            trial = middle - _GOLDEN * (middle - low)
            if evidence(trial) > evidence(middle):
                high, middle = middle, trial
            else:
                low = trial
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

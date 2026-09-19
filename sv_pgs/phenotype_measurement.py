"""The measurement model of a quantitative trait's occasions (docs/design/math/novel-pheno.md §3).

Occasion j of person i is a positive reading y_ij in the trait's canonical unit, and

    z_ij = h(y_ij) = d_ij' gamma + T_i + e_ij,   T_i ~ N(0, tau^2),   e_ij ~ integral N(0, e^t) g(t) dt.

- h is the Box-Cox transform (y^lambda - 1) / lambda (log y at lambda = 0). Its log Jacobian
  (lambda - 1) log y is part of the likelihood, and lambda maximizes the profile marginal likelihood:
  the transform is the null space of §3.1's smooth transform, whose smooth deviation comes later.
- T_i is the person's long-run level. The target is E[T_i | occasions], with reliability
  1 - Var(T_i | occasions) / tau^2 (§1, the Tweedie identities).
- g is the occasion noise's continuous mixing density, learned by the effect prior's own machinery
  (``scale_mixture_ep``, SPEC 131b205): the occasions are its "effects", in one class with no offset and
  no annotations. A gross error (a typo, a unit confusion) is downweighted by the density, not cut at a
  hand-set range.

Inference is EP-EB.
- Each occasion's noise has a Gaussian site exp(-a e^2 / 2 + b e). Given the sites, q(T_i) is Gaussian with
  precision A_i = 1 / tau^2 + sum_j a_ij and mean mu_i = B_i / A_i, where B_i = sum_j (a_ij r_ij - b_ij) and
  r_ij = z_ij - d_ij' gamma. Since e_ij = r_ij - T_i, its marginal is N(r_ij - mu_i, 1 / A_i), and its
  cavity comes in closed form.
- A pass updates the sites occasion by occasion within each person (every person's k-th occasion at once)
  from the engine's exact tilted moments: sequential EP within a person.
- At fixed sites, gamma maximizes the EP evidence in closed form and tau^2 by a scalar root. The density's
  coefficients and smoothing weights are the engine's hyper step.
- The density's lattice is re-derived from the current cavities, and the density carried onto it, when the
  cavities leave its kernel range; its spacing is halved when the quadrature bound asks for it.

Tolerances hold numerical error below statistical error.
- A pass has converged when the levels' move sum_i A_i (delta mu_i)^2 is below one person's posterior
  variance in total: on average 1/n of each person's.
- The hyper step, the lattice's quadrature and the profile over lambda resolve 1/2 nat: a Newton decrement of
  1/2 leaves the hyperparameters within one posterior standard deviation, and a profile bracket spanning
  1/2 nat lies within lambda's one-sd interval.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from sv_pgs._typing import F64Array, I64Array
from sv_pgs.scale_mixture_ep import (
    Cavity,
    MixtureHyperparameters,
    ScaleMixturePrior,
    cavities,
    class_log_density,
    derived_lattice,
    halved_lattice,
    hyper_step,
    initial_hyperparameters,
    kernel_floor,
    kernel_top,
    moment_matched_prior_sites,
    prior_second_moment,
    quadrature_majorant_ratio,
    scale_mixture_prior,
    site_targets,
    spacing_bound,
    tilted_moments,
)

# One person's posterior variance in total over a pass (module docstring).
LEVEL_MOVE_TOLERANCE = 1.0
# One posterior standard deviation of the hyperparameters, in nats (module docstring).
EVIDENCE_TOLERANCE = 0.5
_GOLDEN = 0.5 * (3.0 - np.sqrt(5.0))


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
        counts = np.bincount(self.person_index)
        if np.any(counts == 0):
            raise ValueError("every person index below the largest must have an occasion")
        if self.design.shape[0] != self.values.shape[0]:
            raise ValueError("the design needs one row per occasion")
        if np.linalg.matrix_rank(self.design) < self.design.shape[1]:
            raise ValueError("the fixed-effect design must have full column rank")

    @property
    def person_count(self) -> int:
        return int(self.person_index.max()) + 1


@dataclass(frozen=True)
class OccasionModelFit:
    """The fitted model and every person's level posterior q(T_i)."""

    exponent: float
    fixed_effects: F64Array
    level_variance: float
    prior: ScaleMixturePrior
    hyperparameters: MixtureHyperparameters
    site_precision: F64Array
    site_shift: F64Array
    level_mean: F64Array
    level_posterior_variance: F64Array
    log_evidence: float

    @property
    def reliability(self) -> F64Array:
        return 1.0 - self.level_posterior_variance / self.level_variance

    @property
    def noise_second_moment(self) -> float:
        """E[e^2] under the fitted noise density."""
        return float(prior_second_moment(self.prior, self.hyperparameters)[0])


def _level_posterior(
    occasions: Occasions, residuals: F64Array, site_precision: F64Array, site_shift: F64Array, level_variance: float
) -> tuple[F64Array, F64Array]:
    """(mu_i, A_i): q(T_i)'s mean and precision."""
    count = occasions.person_count
    precision = 1.0 / level_variance + np.bincount(occasions.person_index, weights=site_precision, minlength=count)
    linear = np.bincount(occasions.person_index, weights=site_precision * residuals - site_shift, minlength=count)
    return linear / precision, precision


def _occasion_cavity(
    rows: I64Array, occasions: Occasions, residuals: F64Array, site_precision: F64Array, site_shift: F64Array,
    level_mean: F64Array, level_precision: F64Array,
) -> Cavity:
    persons = occasions.person_index[rows]
    cavity = cavities(
        residuals[rows] - level_mean[persons], 1.0 / level_precision[persons], site_precision[rows], site_shift[rows]
    )
    if not np.all(cavity.precision > 0.0):
        raise FloatingPointError(
            "an occasion's cavity is improper: the other sites of its person have negative total precision"
        )
    return cavity


def _prior_on(nodes: F64Array, floor: float, top: float, count: int) -> ScaleMixturePrior:
    """The noise density's prior over ``count`` occasions: one class, no offset, no annotations."""
    return scale_mixture_prior(
        class_index=np.zeros(count, dtype=np.int64),
        log_variance_offset=np.zeros(count),
        annotation_design=np.zeros((count, 0)),
        annotation_groups=(),
        nodes=nodes,
        floor=floor,
        top=top,
    )


def _moment_level_variance(occasions: Occasions, residuals: F64Array) -> float:
    """The start of tau^2: the between-person moment of the residuals net of the within-person one."""
    counts = np.bincount(occasions.person_index).astype(np.float64)
    person_mean = np.bincount(occasions.person_index, weights=residuals) / counts
    repeated = counts.sum() - counts.shape[0]
    within = float(np.sum(np.square(residuals - person_mean[occasions.person_index]))) / repeated if repeated else 0.0
    between = float(np.var(person_mean)) - within * float(np.mean(1.0 / counts))
    if not between > 0.0:
        raise ValueError("no between-person variance: person means carry no signal beyond occasion noise")
    return between


def _carried(
    prior: ScaleMixturePrior, hyperparameters: MixtureHyperparameters, nodes: F64Array, floor: float, top: float, count: int
) -> tuple[ScaleMixturePrior, MixtureHyperparameters]:
    """The density on a re-derived lattice of the same spacing: the cubic through the old nodes inside their
    range and the end slopes beyond it, so the log-tails stay linear (D3 leaves linear tails free)."""
    old = prior.log_variance_grid
    spline = CubicSpline(old, class_log_density(prior, hyperparameters.coefficients)[0])
    values = spline(np.clip(nodes, old[0], old[-1]))
    below, above = nodes < old[0], nodes > old[-1]
    values[below] += spline(old[0], 1) * (nodes[below] - old[0])
    values[above] += spline(old[-1], 1) * (nodes[above] - old[-1])
    carried = _prior_on(nodes, floor, top, count)
    coefficients = np.linalg.lstsq(carried.coefficient_map, values - values.mean(), rcond=None)[0]
    return carried, MixtureHyperparameters(coefficients=coefficients, log_smoothing=hyperparameters.log_smoothing.copy())


class _Fit:
    """EP-EB at one Box-Cox exponent."""

    def __init__(
        self, occasions: Occasions, exponent: float, working_bytes: int, fixed_effects: F64Array, level_variance: float,
        prior: ScaleMixturePrior, hyperparameters: MixtureHyperparameters,
    ) -> None:
        self.occasions = occasions
        self.working_bytes = working_bytes
        self.transformed, log_jacobian = box_cox(occasions.values, exponent)
        self.log_jacobian = float(log_jacobian.sum())
        order = np.argsort(occasions.person_index, kind="stable")
        first = np.searchsorted(occasions.person_index[order], np.arange(occasions.person_count))
        rank = np.empty(order.shape[0], dtype=np.int64)
        rank[order] = np.arange(order.shape[0]) - first[occasions.person_index[order]]
        self.levels = [np.flatnonzero(rank == position) for position in range(int(rank.max()) + 1)]
        self.fixed_effects = np.asarray(fixed_effects, dtype=np.float64).copy()
        self.level_variance = float(level_variance)
        self._set_prior(_prior_on(prior.log_variance_grid, prior.kernel_floor, prior.kernel_top, occasions.values.shape[0]))
        self.hyperparameters = hyperparameters
        self.site_precision = np.zeros(occasions.values.shape[0])
        self.site_shift = np.zeros(occasions.values.shape[0])

    @classmethod
    def start(cls, occasions: Occasions, exponent: float, working_bytes: int) -> _Fit:
        """The derived start: least-squares fixed effects, the moment level variance, the lattice from the cavities
        of T's prior alone, the engine's start density and sites at that density's own variance."""
        transformed, _log_jacobian = box_cox(occasions.values, exponent)
        fixed_effects = np.linalg.lstsq(occasions.design, transformed, rcond=None)[0]
        residuals = transformed - occasions.design @ fixed_effects
        level_variance = _moment_level_variance(occasions, residuals)
        precision = np.full(residuals.shape[0], 1.0 / level_variance)
        nodes, floor, top = derived_lattice(precision, residuals * precision, np.zeros(residuals.shape[0]), EVIDENCE_TOLERANCE)
        prior = _prior_on(nodes, floor, top, residuals.shape[0])
        fit = cls(occasions, exponent, working_bytes, fixed_effects, level_variance, prior, initial_hyperparameters(prior))
        fit.site_precision, fit.site_shift = moment_matched_prior_sites(fit.prior, fit.hyperparameters)
        return fit

    def _set_prior(self, prior: ScaleMixturePrior) -> None:
        self.prior = prior
        self.level_priors = [_prior_on(prior.log_variance_grid, prior.kernel_floor, prior.kernel_top, rows.shape[0]) for rows in self.levels]

    def residuals(self) -> F64Array:
        return self.transformed - self.occasions.design @ self.fixed_effects

    def level_posterior(self) -> tuple[F64Array, F64Array]:
        return _level_posterior(self.occasions, self.residuals(), self.site_precision, self.site_shift, self.level_variance)

    def cavity(self, rows: I64Array) -> Cavity:
        level_mean, level_precision = self.level_posterior()
        return _occasion_cavity(
            rows, self.occasions, self.residuals(), self.site_precision, self.site_shift, level_mean, level_precision
        )

    def ep_pass(self) -> float:
        """One sequential pass; returns the levels' move sum_i A_i (delta mu_i)^2."""
        start_mean, start_precision = self.level_posterior()
        for rows, level_prior in zip(self.levels, self.level_priors, strict=True):
            cavity = self.cavity(rows)
            moments = tilted_moments(level_prior, self.hyperparameters, cavity, self.working_bytes)
            self.site_precision[rows], self.site_shift[rows] = site_targets(moments, cavity)
        level_mean, _precision = self.level_posterior()
        return float(np.sum(start_precision * np.square(level_mean - start_mean)))

    def converge_sites(self) -> int:
        passes = 1
        while self.ep_pass() >= LEVEL_MOVE_TOLERANCE:
            passes += 1
        return passes

    def update_fixed_effects(self) -> None:
        """gamma maximizing the EP evidence at fixed sites: sum_i D_i' M_i D_i gamma = sum_i D_i' (M_i z_i - c_i),
        M_i = W_i - a_i a_i' / A_i and c_i = b_i - a_i (1' b_i) / A_i."""
        occasions, a, b, z = self.occasions, self.site_precision, self.site_shift, self.transformed
        count = occasions.person_count
        _mean, level_precision = self.level_posterior()
        design = occasions.design
        weighted = design * a[:, None]
        person_weighted = np.stack([np.bincount(occasions.person_index, weights=column, minlength=count) for column in weighted.T], axis=1)
        normal = design.T @ weighted - person_weighted.T @ (person_weighted / level_precision[:, None])
        person_signal = np.bincount(occasions.person_index, weights=a * z - b, minlength=count)
        right = design.T @ (a * z - b) - person_weighted.T @ (person_signal / level_precision)
        if np.linalg.eigvalsh(normal)[0] <= 0.0:
            raise FloatingPointError("the EP evidence is not concave in the fixed effects at these sites")
        self.fixed_effects = np.linalg.solve(normal, right)

    def update_level_variance(self) -> None:
        """tau^2 at the stationary point of the EP evidence at fixed sites: n tau^2 = sum_i (1 / A_i + mu_i^2)."""
        occasions = self.occasions
        count = occasions.person_count
        site_total = np.bincount(occasions.person_index, weights=self.site_precision, minlength=count)
        linear = np.bincount(occasions.person_index, weights=self.site_precision * self.residuals() - self.site_shift, minlength=count)
        if not float(np.sum(np.square(linear))) > float(np.sum(site_total)):
            raise ValueError("no between-person variance: person means carry no signal beyond occasion noise")

        def stationarity(log_precision: float) -> float:
            precision = np.exp(log_precision) + site_total
            return count * np.exp(-log_precision) - float(np.sum(1.0 / precision + np.square(linear / precision)))

        start = -np.log(self.level_variance)
        low, high = start, start
        while stationarity(low) <= 0.0:
            low -= np.log(2.0)
        while stationarity(high) >= 0.0:
            high += np.log(2.0)
        self.level_variance = float(np.exp(-brentq(stationarity, low, high, xtol=np.finfo(np.float64).eps)))

    def fit_lattice(self) -> None:
        """Re-derive the kernel range from the current cavities and refine the spacing to the quadrature bound."""
        cavity = self.cavity(np.arange(self.occasions.values.shape[0]))
        scales = np.zeros(cavity.precision.shape[0])
        floor = min(self.prior.kernel_floor, kernel_floor(cavity.precision, cavity.shift, scales, EVIDENCE_TOLERANCE))
        top = max(self.prior.kernel_top, kernel_top(cavity.precision, cavity.shift, scales, floor))
        if floor < self.prior.kernel_floor or top > self.prior.kernel_top:
            nodes = self.prior.log_variance_grid
            spacing = float(nodes[1] - nodes[0])
            width = max(top - floor, spacing)
            extended = np.arange(floor - width, top + width + spacing, spacing)
            prior, self.hyperparameters = _carried(self.prior, self.hyperparameters, extended, floor, top, cavity.precision.shape[0])
            self._set_prior(prior)
        while True:
            nodes = self.prior.log_variance_grid
            bound = spacing_bound(quadrature_majorant_ratio(self.prior, self.hyperparameters, cavity, self.working_bytes), EVIDENCE_TOLERANCE)
            if nodes[1] - nodes[0] <= bound:
                return
            prior, self.hyperparameters = halved_lattice(self.prior, self.hyperparameters)
            self._set_prior(prior)

    def log_evidence(self) -> float:
        """The EP marginal likelihood of the readings: sum_i G_i + sum_ij log C_ij + the Jacobian.

        G_i = -log(tau^2 A_i) / 2 + B_i^2 / (2 A_i) + sum_j (-a r^2 / 2 + b r) integrates q's Gaussian part, and
        log C_ij = log Z_ij - log(2 pi / A_i) / 2 - A_i m_ij^2 / 2 normalizes each site to its tilted mass, with
        m_ij = r_ij - mu_i the marginal mean of e_ij (the cavity's P + a is A_i and its h + b is A_i m).
        """
        occasions, a, b = self.occasions, self.site_precision, self.site_shift
        residuals = self.residuals()
        level_mean, level_precision = self.level_posterior()
        linear = level_mean * level_precision
        gaussian = float(np.sum(-0.5 * np.log(self.level_variance * level_precision) + 0.5 * np.square(linear) / level_precision))
        gaussian += float(np.sum(-0.5 * a * np.square(residuals) + b * residuals))
        rows = np.arange(occasions.values.shape[0])
        cavity = self.cavity(rows)
        tilted = tilted_moments(self.prior, self.hyperparameters, cavity, self.working_bytes)
        precision = level_precision[occasions.person_index]
        marginal_mean = residuals - level_mean[occasions.person_index]
        normalizers = tilted.log_normalizer - 0.5 * np.log(2.0 * np.pi / precision) - 0.5 * precision * np.square(marginal_mean)
        return gaussian + float(np.sum(normalizers)) + self.log_jacobian

    def run(self) -> float:
        """EP-EB to its fixed point; returns the log evidence."""
        previous = -np.inf
        while True:
            passes = self.converge_sites()
            self.update_fixed_effects()
            self.update_level_variance()
            self.fit_lattice()
            step = hyper_step(self.prior, self.hyperparameters, self.cavity(np.arange(self.occasions.values.shape[0])), self.working_bytes, EVIDENCE_TOLERANCE)
            self.hyperparameters = step.hyperparameters
            current = self.log_evidence()
            if passes == 1 and step.newton_decrement <= EVIDENCE_TOLERANCE and abs(current - previous) < EVIDENCE_TOLERANCE:
                return current
            previous = current

    def result(self, exponent: float, evidence: float) -> OccasionModelFit:
        level_mean, level_precision = self.level_posterior()
        return OccasionModelFit(
            exponent=exponent,
            fixed_effects=self.fixed_effects.copy(),
            level_variance=self.level_variance,
            prior=self.prior,
            hyperparameters=self.hyperparameters,
            site_precision=self.site_precision.copy(),
            site_shift=self.site_shift.copy(),
            level_mean=level_mean,
            level_posterior_variance=1.0 / level_precision,
            log_evidence=evidence,
        )


def fit_at_exponent(occasions: Occasions, exponent: float, working_bytes: int) -> OccasionModelFit:
    fit = _Fit.start(occasions, exponent, working_bytes)
    return fit.result(exponent, fit.run())


def ep_fixed_point(
    occasions: Occasions, exponent: float, fixed_effects: F64Array, level_variance: float, prior: ScaleMixturePrior,
    hyperparameters: MixtureHyperparameters, working_bytes: int,
) -> OccasionModelFit:
    """q(T_i) at the EP fixed point for given model parameters, learning nothing: the levels of people outside
    the fit, and the exactness checks. ``prior`` supplies the lattice (nodes, floor, top) of ``hyperparameters``."""
    fit = _Fit(occasions, exponent, working_bytes, fixed_effects, level_variance, prior, hyperparameters)
    fit.converge_sites()
    return fit.result(exponent, fit.log_evidence())


def fit_occasion_model(occasions: Occasions, working_bytes: int) -> OccasionModelFit:
    """The fit at the Box-Cox exponent maximizing the profile evidence, by golden-section search.

    The search starts from the log (0) and the identity (1), grows the bracket until its middle point is the
    best of the three, and narrows it until the bracket's evidence spans less than EVIDENCE_TOLERANCE.
    """
    fits: dict[float, OccasionModelFit] = {}

    def evidence(exponent: float) -> float:
        if exponent not in fits:
            fits[exponent] = fit_at_exponent(occasions, exponent, working_bytes)
        return fits[exponent].log_evidence

    low, middle = 0.0, 1.0
    if evidence(middle) < evidence(low):
        low, middle = middle, low
    high = middle + (middle - low) / _GOLDEN * (1.0 - _GOLDEN)
    while evidence(high) > evidence(middle):
        low, middle, high = middle, high, high + (high - middle) / _GOLDEN * (1.0 - _GOLDEN)
    while max(evidence(low), evidence(high), evidence(middle)) - min(evidence(low), evidence(high)) >= EVIDENCE_TOLERANCE:
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
    return fits[middle]

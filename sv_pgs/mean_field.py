"""The small-n route's mean-field inference: q(beta) = prod_j q_j, each q_j the prior's mixture tilted by its
pseudo-likelihood, by coordinate ascent on the evidence lower bound (ELBO), as ``scale_mixture_ep.FixedPoints``.

The definition of done (TEAM_RULES) chooses the small-n inference by measurement: whichever of {mr.ash's coordinate
VB, EP} predicts better under the same prior on bench-real. On gene 1 [real] they tied under mr.ash's prior once
tie members kept their own effects (svpgs-profiler's 2 x 2), while EP cost 500x the CPU and refused most of the 20
chr22 genes on memory. This oracle is the VB side of that measurement on SV-PGS's own prior, so the outer loop
(``fit_hyperparameters``: the certified empirical Bayes of the mixing density, its smoothness and the scale terms)
is the same as EP's; only the fixed point differs.

**Model and bound.** With Xp = (I - H_C) X the covariate-projected design over the members (``small_n._Design``),
y_P = (I - H_C) y and noise sigma^2, the likelihood on the covariates' complement is
N(y_P; Xp beta, sigma^2 I) on n - k dimensions, and the prior of member j is the class mixture over the lattice,
beta_j ~ sum_k pi_{c(j) k} N(0, u_j e^{t_k}) (``ScaleMixturePrior``). For a product q = prod_j q_j,

    ELBO(q, x, sigma^2) = E_q log p(y_P | beta) - sum_j KL(q_j || p_j(. | x)),
    E_q log p(y_P | beta) = -(n - k)/2 log(2 pi sigma^2) - (||r||^2 + sum_j ||x_j||^2 Var_q beta_j) / (2 sigma^2),

with r = y_P - Xp E_q beta (E||y_P - Xp beta||^2 splits so under independence). The ELBO's maximizer over one q_j
with the rest held is the prior tilted by the pseudo-likelihood l_j(beta) = exp(h_j beta - omega_j beta^2 / 2),
omega_j = ||x_j||^2 / sigma^2 and h_j = x_j'(r + x_j m_j) / sigma^2 (the residual without j's own term): a mixture
over the nodes with responsibilities and moments given by ``scale_mixture_ep._components`` at the cavity
(omega_j, h_j), and KL(q_j || p_j) = h_j m_j - omega_j (m_j^2 + v_j) / 2 - log Z_j(omega_j, h_j) with (m_j, v_j)
q_j's mean and variance and log Z_j the tilted normalizer. So a sweep is coordinate ascent (monotone in the ELBO)
and its accumulated terms give the ELBO exactly at the sweep's end, each q_j against the pseudo-likelihood it was
built from.

**Noise.** At fixed q the ELBO's maximizer is sigma^2 = (||r||^2 + sum_j ||x_j||^2 v_j) / (n - k)
(``scale_mixture_ep.noise_variance``'s form with the mean-field variances), and its gain is
``scale_mixture_ep.noise_gain``.

**The fixed point.** Sweeps end when the gain they can still find is at most the fit's resolution, 1/(2K) nats
for a scorer with K posterior draws: the remaining gain is bounded by the last sweep's gain g_t times
rho / (1 - rho), rho = g_t / g_(t-1) the measured contraction (the same extrapolation the EP oracles use for their
distance to the fixed point), so a call sweeps at least until two gains are measured; a gain below the ELBO's own
rounding (eps times the sum of its terms' sizes) is resolved. Nothing is clipped, damped or capped.

**The outer loop's view.** The pseudo-likelihoods are the ``Cavity`` the hyper step maximizes over: at the fixed
point q_j = t_j(x), so ELBO = sum_j log Z_j(x; omega, h) + terms free of x, and sum_j log Z_j is exactly what
``hyper_step`` maximizes at a cavity. The total curvature's linear response comes from the mean-field fixed point's
own: m_j = f_j(h_j(m_-j); x) gives dm/dx = (I - J)^-1 df/dx with J_ji = v_j (-x_j' x_i / sigma^2) for i != j, i.e.
(I - J) = diag(v) [diag(1 / v_j - omega_j) + Xp'Xp / sigma^2], the Gaussian posterior with sites
tau_j = 1 / v_j - omega_j (the tilted precision less the pseudo-likelihood's). That is ``small_n._Kernel`` at
sigma^2 tau, so ``small_n._DensePosterior`` answers the outer loop's solves and variance products exactly, as for EP.
Posterior draws are q's own: each member's node from its responsibilities, then its conditional normal.
"""

from __future__ import annotations

import time
from typing import Sequence

import numba
import numpy as np

from sv_pgs._typing import F64Array, I64Array
from sv_pgs.scale_mixture_ep import (
    Cavity,
    FixedPoint,
    MixtureHyperparameters,
    ScaleMixturePrior,
    _components,
    class_log_density,
    log_scale,
    noise_gain,
)
from sv_pgs.small_n import _LIVE_FIXED_POINTS, DenseStatistics, _DensePosterior, _Kernel, _new_profile

_EPSILON = float(np.finfo(np.float64).eps)


@numba.njit(cache=True)
def _sweep(design, squares, members, class_index, log_density, log_scale_rows, grid, noise, mean, residual, variance, shift):
    """One coordinate-ascent sweep over the members in order, in place: ``mean`` and ``variance`` (each q_j's
    moments), ``residual`` (r = y_P - Xp mean) and ``shift`` (the h_j each q_j was built from). Returns
    (sum_j KL(q_j || p_j), sum_j ||x_j||^2 v_j, ||r||^2) at the sweep's end, so the ELBO is exact there.

    ``design`` is Xp over the groups (n x groups, Fortran order), ``squares`` their ||x_g||^2, ``members[j]`` member
    j's group; ``log_density`` is (classes x nodes), ``log_scale_rows`` per member, ``grid`` the nodes' log
    variances. The node terms are ``scale_mixture_ep._kernel_terms``' own, with the same overflow limits: a node
    whose variance overflows contributes conditional variance 1 / omega and weight 0."""
    sample_count = design.shape[0]
    member_count = members.shape[0]
    node_count = grid.shape[0]
    divergence = 0.0
    weighted_variance = 0.0
    log_weights = np.empty(node_count)
    conditional = np.empty(node_count)
    for member in range(member_count):
        group = members[member]
        omega = squares[group] / noise
        projection = 0.0
        for sample in range(sample_count):
            projection += design[sample, group] * residual[sample]
        old_mean = mean[member]
        h = (projection + squares[group] * old_mean) / noise
        row = class_index[member]
        peak = -np.inf
        for node in range(node_count):
            log_variance = log_scale_rows[member] + grid[node]
            variance_node = np.exp(log_variance)
            ratio = variance_node * omega
            if ratio == np.inf:
                conditional[node] = 1.0 / omega
                log_weights[node] = -np.inf
            else:
                conditional[node] = 1.0 / (1.0 / variance_node + omega) if variance_node > 0.0 else 0.0
                log_weights[node] = log_density[row, node] - 0.5 * np.log1p(ratio) + 0.5 * h * h * conditional[node]
            if log_weights[node] > peak:
                peak = log_weights[node]
        total = 0.0
        for node in range(node_count):
            log_weights[node] = np.exp(log_weights[node] - peak)
            total += log_weights[node]
        log_normalizer = peak + np.log(total)
        new_mean = 0.0
        for node in range(node_count):
            new_mean += log_weights[node] / total * h * conditional[node]
        # Var = E_w[c] + Var_w(h c): both terms non-negative, so nothing cancels.
        new_variance = 0.0
        for node in range(node_count):
            offset = h * conditional[node] - new_mean
            new_variance += log_weights[node] / total * (conditional[node] + offset * offset)
        step = new_mean - old_mean
        if step != 0.0:
            for sample in range(sample_count):
                residual[sample] -= design[sample, group] * step
        mean[member] = new_mean
        variance[member] = new_variance
        shift[member] = h
        divergence += h * new_mean - 0.5 * omega * (new_mean * new_mean + new_variance) - log_normalizer
        weighted_variance += squares[group] * new_variance
    residual_square = 0.0
    for sample in range(sample_count):
        residual_square += residual[sample] * residual[sample]
    return divergence, weighted_variance, residual_square


class MeanFieldFixedPoints:
    """``scale_mixture_ep.FixedPoints`` for one model on a dense training matrix by mean-field coordinate ascent
    (module docstring). The state (q's moments, the residual, the noise) is warm across calls; a call that ends
    without a certified fixed point restores it and records the refusal."""

    def __init__(
        self, statistics: DenseStatistics, prior: ScaleMixturePrior, start_noise: float, draw_count: int, working_bytes: int
    ) -> None:
        self.statistics = statistics
        self.prior = prior
        self.draw_count = int(draw_count)
        self.working_bytes = int(working_bytes)
        self.design = statistics.design
        self.sample_count = statistics.sample_count
        self.covariate_count = int(statistics.covariates.shape[1])
        self.residual_dimension = self.sample_count - self.covariate_count
        # Xp over the groups, dense, once: every sweep is a pass over it.
        self.projected = np.asfortranarray(self.design.group_columns(np.arange(self.design.group_count)))
        self.group_squares = np.einsum("ij,ij->j", self.projected, self.projected)
        self.members = np.asarray(self.design.members, dtype=np.int64)
        self.member_squares = self.group_squares[self.members]
        self.class_index = np.asarray(prior.class_index, dtype=np.int64)
        self.noise = float(start_noise)
        self.mean = np.zeros(prior.variant_count)
        self.variance = np.zeros(prior.variant_count)
        self.shift = np.zeros(prior.variant_count)
        self.residual = np.array(statistics.projected_target, dtype=np.float64, copy=True)
        self.site_precision = np.zeros(prior.variant_count)
        self.effective = float(prior.variant_count)
        self.mean_move = np.inf
        self.noise_gain = np.inf
        self.refusals: list[str] = []
        self.profile = _new_profile() | {"sweeps": 0, "sweep_seconds": 0.0, "elbo": -np.inf}
        self.kernel: _Kernel | None = None

    # the ELBO and its pieces

    def _elbo(self, divergence: float, weighted_variance: float, residual_square: float) -> float:
        noise = self.noise
        return -0.5 * self.residual_dimension * float(np.log(2.0 * np.pi * noise)) - (residual_square + weighted_variance) / (2.0 * noise) - divergence

    def _sweep(self, hyperparameters: MixtureHyperparameters) -> tuple[float, float, float]:
        started = time.perf_counter()
        prior = self.prior
        values = _sweep(
            self.projected, self.group_squares, self.members, self.class_index,
            np.ascontiguousarray(class_log_density(prior, hyperparameters.coefficients)),
            np.ascontiguousarray(log_scale(prior, hyperparameters.coefficients)), np.ascontiguousarray(prior.log_variance_grid),
            self.noise, self.mean, self.residual, self.variance, self.shift,
        )
        self.profile["sweeps"] += 1
        self.profile["passes"] += 1
        self.profile["sweep_seconds"] += time.perf_counter() - started
        return values

    def _snapshot(self) -> dict:
        return {
            "mean": self.mean.copy(), "variance": self.variance.copy(), "shift": self.shift.copy(), "residual": self.residual.copy(),
            "noise": self.noise, "site_precision": self.site_precision.copy(), "effective": self.effective, "kernel": self.kernel,
        }

    def _restore(self, snapshot: dict) -> None:
        self.mean, self.variance, self.shift, self.residual = (snapshot[name].copy() for name in ("mean", "variance", "shift", "residual"))
        self.noise, self.site_precision, self.effective, self.kernel = snapshot["noise"], snapshot["site_precision"].copy(), snapshot["effective"], snapshot["kernel"]

    def __call__(self, hyperparameters: Sequence[MixtureHyperparameters]) -> list[FixedPoint | None]:
        (model_hyperparameters,) = hyperparameters
        self.profile["fixed_point_calls"] += 1
        snapshot = self._snapshot()
        try:
            return [self._solve(model_hyperparameters)]
        except (FloatingPointError, np.linalg.LinAlgError) as error:
            self._restore(snapshot)
            self.refusals.append(str(error))
            return [None]

    def _solve(self, hyperparameters: MixtureHyperparameters) -> FixedPoint:
        """Sweeps at the current noise until the remaining gain (module docstring) is within the tolerance. The noise
        moves to its stationary value only between sweeps, so the returned state (q, sigma^2) is the one the last
        sweep built: its pseudo-likelihoods are the cavity, exactly. The noise's pending gain counts as remaining."""
        tolerance = 0.5 / self.draw_count
        previous_gain: float | None = None
        # The hyperparameters changed since the last call, so the state's ELBO is unknown until a sweep measures it:
        # the first sweep's gain is not a gain, and the bound starts at the second.
        elbo: float | None = None
        pending_noise: float | None = None
        while True:
            if pending_noise is not None:
                elbo = elbo + self.noise_gain if elbo is not None else None
                self.noise = pending_noise
            divergence, weighted_variance, residual_square = self._sweep(hyperparameters)
            if not (np.isfinite(divergence) and np.isfinite(weighted_variance) and np.isfinite(residual_square)):
                raise FloatingPointError("a mean-field sweep is not finite")
            value = self._elbo(divergence, weighted_variance, residual_square)
            # The noise's stationary value at this q, and the gain it would bring (both exact); applied before the
            # next sweep, if there is one.
            pending_noise = (residual_square + weighted_variance) / self.residual_dimension
            self.noise_gain = noise_gain(pending_noise, self.noise, self.sample_count, self.covariate_count)
            # The ELBO's rounding: the sum of its terms' sizes times eps. A gain below it is not resolved.
            magnitude = 0.5 * self.residual_dimension * abs(float(np.log(2.0 * np.pi * self.noise))) + (residual_square + weighted_variance) / (2.0 * self.noise) + abs(divergence)
            rounding = _EPSILON * magnitude
            gain = (value - elbo) if elbo is not None else None
            elbo = value
            self.profile["elbo"] = value
            if gain is None:
                continue
            if gain < -rounding:
                raise FloatingPointError(f"a mean-field sweep lowered the ELBO by {-gain:.3g} nats: the bound's ascent is broken")
            gain = max(float(gain), 0.0)
            if gain <= rounding:
                remaining = 0.0
            elif previous_gain is not None and previous_gain > 0.0:
                rate = gain / previous_gain
                remaining = gain * rate / (1.0 - rate) if rate < 1.0 else np.inf
            else:
                remaining = np.inf
            previous_gain = gain
            self.mean_move = remaining
            if remaining + self.noise_gain <= tolerance:
                return self._fixed_point(hyperparameters)

    def _fixed_point(self, hyperparameters: MixtureHyperparameters) -> FixedPoint:
        """The certified state as the outer loop's fixed point: the pseudo-likelihoods as the cavity, the linear
        response's Gaussian (sites 1 / v_j - omega_j) as the posterior."""
        omega = self.member_squares / self.noise
        with np.errstate(divide="ignore"):
            tau = np.where(self.variance > 0.0, 1.0 / self.variance, np.inf) - omega
        if not np.all(np.isfinite(tau)):
            raise FloatingPointError("a member's mean-field variance is 0: its linear response has no finite site")
        started = time.perf_counter()
        kernel = _Kernel(self.design, self.noise * tau)
        self.profile["factorizations"] += 1
        self.profile["factor_seconds"] += time.perf_counter() - started
        variances, removed, _cavity_precision = kernel.cavity()
        self.kernel = kernel
        self.site_precision = tau
        self.effective = max(float(np.sum(removed)), _EPSILON * tau.shape[0])
        self.profile["refreshes"] += 1
        posterior = _DensePosterior(kernel, self.noise, self.working_bytes // _LIVE_FIXED_POINTS, self.profile)
        design, noise, precision = self.design, self.noise, tau.copy()

        def norm(direction: F64Array) -> float:
            values = np.asarray(direction, dtype=np.float64)
            image = design.image(values)
            return float(image @ image) / noise + float(np.sum(precision * values * values))

        return FixedPoint(
            cavity=Cavity(precision=omega, shift=self.shift.copy()), posterior=posterior.gaussian_posterior(), mean=self.mean.copy(),
            precision_norm=norm, effective_effects=float(self.effective),
        )

    def draws(self, hyperparameters: MixtureHyperparameters, generator: np.random.Generator, draw_count: int) -> F64Array:
        """(p x K) draws from q itself: each member's node from its responsibilities at its pseudo-likelihood, then its
        conditional normal N(h c_k, c_k) (``scale_mixture_ep._components``)."""
        prior = self.prior
        log_density = class_log_density(prior, hyperparameters.coefficients)
        scales = log_scale(prior, hyperparameters.coefficients)
        omega = self.member_squares / self.noise
        draws = np.empty((prior.variant_count, int(draw_count)))
        for class_position in range(log_density.shape[0]):
            rows = np.flatnonzero(self.class_index == class_position)
            if rows.size == 0:
                continue
            terms = _components(log_density[class_position], scales[rows], prior.log_variance_grid, omega[rows], self.shift[rows])
            cumulative = np.cumsum(terms.responsibility, axis=1)
            uniform = generator.random((rows.shape[0], int(draw_count)))
            nodes = np.minimum(np.sum(cumulative[:, :, None] < uniform[:, None, :], axis=1), prior.grid_size - 1)
            conditional = np.take_along_axis(terms.conditional_variance, nodes, axis=1)
            draws[rows] = self.shift[rows][:, None] * conditional + np.sqrt(conditional) * generator.standard_normal(conditional.shape)
        return draws

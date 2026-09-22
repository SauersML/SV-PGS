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
rounding (``_elbo``: a forward-error bound from the pieces' sizes) is resolved. Nothing is clipped, damped or capped.

**The outer loop's view.** The pseudo-likelihoods are the ``Cavity`` the hyper step maximizes over: at the fixed
point q_j = t_j(x), so by the envelope theorem d ELBO*/dx = sum_j d log Z_j(x; omega_j, h_j)/dx at fixed cavities,
exactly the fixed-cavity gradient EP's outer loop uses. The total curvature B = -d2 ELBO*/dx2 needs the cavity's
response to x: only h moves (omega is fixed), through the means, dh = -(Xp'Xp - diag ||x_j||^2) dm / sigma^2 with
dm = f_h dh + f_x E, f_h = v (the tilted variance) and f_x E = m_x E the fixed-cavity mean change, so
(diag(1 / v - omega) + Xp'Xp / sigma^2) dm = diag(1 / v) m_x E. ``GaussianPosterior.cavity_response`` hands (dh, 0)
to ``_total_curvature_columns``, which forms B from it as it does from EP's response; the solve is ``_Response``
(the matrix is symmetric, not always positive definite). The prediction check moves q's means in q's own metric,
the mixture's component-wise Gaussian KL upper bound, and p_eff = sum_j omega_j v_j (tr(Xp Sigma_q Xp') / sigma^2 for the product q). Posterior draws
are q's own: each member's node from its responsibilities, then its conditional normal.
"""

from __future__ import annotations

import time
from typing import Sequence

import numba
import numpy as np
from scipy import linalg

from sv_pgs._typing import F64Array, I64Array
from sv_pgs.scale_mixture_ep import (
    Cavity,
    FixedPoint,
    GaussianPosterior,
    MixtureHyperparameters,
    ScaleMixturePrior,
    _components,
    _class_terms,
    class_log_density,
    log_scale,
    noise_gain,
)
from sv_pgs.small_n import DenseStatistics, _Design, _new_profile

_EPSILON = float(np.finfo(np.float64).eps)


@numba.njit(cache=True)
def _sweep(design, squares, members, class_index, log_density, log_scale_rows, grid, noise, mean, residual, variance, shift):
    """One coordinate-ascent sweep over the members in order, in place: ``mean`` and ``variance`` (each q_j's
    moments), ``residual`` (r = y_P - Xp mean) and ``shift`` (the h_j each q_j was built from). Returns
    (sum_j KL(q_j || p_j), sum_j ||x_j||^2 v_j, ||r||^2, the sizes of the KL terms' pieces) at the sweep's end, so the ELBO
    and its rounding bound are exact there.

    ``design`` is Xp over the groups (n x groups, Fortran order), ``squares`` their ||x_g||^2, ``members[j]`` member
    j's group; ``log_density`` is (classes x nodes), ``log_scale_rows`` per member, ``grid`` the nodes' log
    variances. The node terms are ``scale_mixture_ep._kernel_terms``' own, with the same overflow limits: a node
    whose variance overflows contributes conditional variance 1 / omega and weight 0."""
    sample_count = design.shape[0]
    member_count = members.shape[0]
    node_count = grid.shape[0]
    divergence = 0.0
    weighted_variance = 0.0
    sizes = 0.0
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
        pull = h * new_mean
        shrink = 0.5 * omega * (new_mean * new_mean + new_variance)
        divergence += pull - shrink - log_normalizer
        sizes += abs(pull) + shrink + abs(log_normalizer)
        weighted_variance += squares[group] * new_variance
    residual_square = 0.0
    for sample in range(sample_count):
        residual_square += residual[sample] * residual[sample]
    return divergence, weighted_variance, residual_square, sizes


class _Response:
    """(Xp'Xp + diag t)^-1 applied to columns, for sites t of either sign: the mean-field fixed point's response
    matrix diag(tau) + Xp'Xp / sigma^2 (t = sigma^2 tau, module docstring) is symmetric but need not be positive
    definite. A member whose tilted variance exceeds its pseudo-likelihood's, v_j > 1 / omega_j, has tau_j < 0, and
    two tied members with such sites make it indefinite along their difference; it is still nonsingular wherever the
    fixed point has a linear response.

    The bulk P (t_j >= ||x_j||^2, where Woodbury keeps every digit: ``small_n._Kernel``) enters through the kernel
    K = I + Xp_P T_P^-1 Xp_P' (positive definite), and the rest N exactly through its Schur complement
    S = T_N + Xp_N' K^-1 Xp_N (|N| x |N|, symmetric, any sign), factored by LU. A dead row (a point-mass tilted law,
    t = infinity) has no response: it is a bulk row with T^-1 = 0."""

    def __init__(self, design: _Design, scaled_sites: F64Array, live: np.ndarray) -> None:
        self.design = design
        squares = design.squares
        bulk = ~live | (scaled_sites >= squares)
        with np.errstate(divide="ignore"):
            self.bulk_inverse = np.where(bulk & live, 1.0 / np.where(bulk & live, scaled_sites, 1.0), 0.0)
        kernel = design.weighted_gram(self.bulk_inverse)
        kernel[np.diag_indices_from(kernel)] += 1.0
        self.upper = linalg.cholesky(kernel, lower=False, check_finite=False, overwrite_a=True)
        self.rest = np.flatnonzero(~bulk)
        if self.rest.size:
            self.rest_columns = design.columns(self.rest)
            whitened = linalg.solve_triangular(self.upper, self.rest_columns, trans="T", lower=False, check_finite=False)
            schur = whitened.T @ whitened
            schur[np.diag_indices_from(schur)] += scaled_sites[self.rest]
            self.schur = linalg.lu_factor(schur, check_finite=False)
            if not np.all(np.isfinite(self.schur[0])) or np.any(np.diag(self.schur[0]) == 0.0):
                raise np.linalg.LinAlgError("the mean-field fixed point's response matrix is singular on its Schur block")

    def _bulk_solve(self, right: F64Array) -> F64Array:
        """(T_P + Xp_P'Xp_P)^-1 on the bulk rows of ``right`` (zero elsewhere), by Woodbury."""
        scaled = self.bulk_inverse[:, None] * right
        image = self.design.image(scaled)
        back = self.design.back(linalg.cho_solve((self.upper, False), image, check_finite=False))
        return scaled - self.bulk_inverse[:, None] * back

    def solve(self, right: F64Array) -> F64Array:
        """(Xp'Xp + diag t)^-1 right for right (p x r)."""
        values = np.asarray(right, dtype=np.float64)
        solution = self._bulk_solve(values)
        if not self.rest.size:
            return solution
        rest_right = values[self.rest] - self.rest_columns.T @ self.design.image(solution)
        rest_solution = linalg.lu_solve(self.schur, rest_right, check_finite=False)
        solution = solution - self._bulk_solve(self.design.back(self.rest_columns @ rest_solution))
        solution[self.rest] = rest_solution
        return solution


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
        self.covariate_count = statistics.covariate_rank
        self.residual_dimension = self.sample_count - self.covariate_count
        # Xp over the groups, dense, once: every sweep is a pass over it.
        self.projected = np.asfortranarray(self.design.group_columns(np.arange(self.design.group_count)))
        self.members = np.asarray(self.design.members, dtype=np.int64)
        # ||x_g||^2 as the design defines it (``_Design.squares``: the same numbers the response and the tests use).
        self.member_squares = np.asarray(self.design.squares, dtype=np.float64)
        self.group_squares = np.zeros(self.design.group_count)
        self.group_squares[self.members] = self.member_squares
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

    # the ELBO and its pieces

    def _elbo(self, divergence: float, weighted_variance: float, residual_square: float, sizes: float) -> tuple[float, float]:
        """(the ELBO at this q and noise, a bound on its rounding). The bound: each of the N = 2p + n summands (p KL
        terms, p weighted variances, n residual squares) is formed from pieces whose sizes sum to S, with at most
        K + 1 rounded operations on each (its log-sum-exp over the K nodes and its combination), and recursive
        summation of N terms adds at most N eps of their sizes (Higham, Accuracy and Stability of Numerical
        Algorithms, 2nd ed., Lemma 3.1 with gamma_N <= N eps at N eps << 1): |rounding| <= (K + 1 + N) eps S."""
        noise = self.noise
        residual_term = 0.5 * self.residual_dimension * float(np.log(2.0 * np.pi * noise))
        fit_term = (residual_square + weighted_variance) / (2.0 * noise)
        value = -residual_term - fit_term - divergence
        summands = 2 * self.prior.variant_count + self.sample_count
        return value, (self.prior.grid_size + 1 + summands) * _EPSILON * (abs(residual_term) + fit_term + sizes)

    def _sweep(self, hyperparameters: MixtureHyperparameters) -> tuple[float, float, float, float]:
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
            "noise": self.noise, "site_precision": self.site_precision.copy(), "effective": self.effective,
        }

    def _restore(self, snapshot: dict) -> None:
        self.mean, self.variance, self.shift, self.residual = (snapshot[name].copy() for name in ("mean", "variance", "shift", "residual"))
        self.noise, self.site_precision, self.effective = snapshot["noise"], snapshot["site_precision"].copy(), snapshot["effective"]

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
            divergence, weighted_variance, residual_square, sizes = self._sweep(hyperparameters)
            if not (np.isfinite(divergence) and np.isfinite(weighted_variance) and np.isfinite(residual_square)):
                raise FloatingPointError("a mean-field sweep is not finite")
            value, rounding = self._elbo(divergence, weighted_variance, residual_square, sizes)
            # The noise's stationary value at this q, and the gain it would bring (both exact); applied before the
            # next sweep, if there is one.
            pending_noise = (residual_square + weighted_variance) / self.residual_dimension
            self.noise_gain = noise_gain(pending_noise, self.noise, self.sample_count, self.covariate_count)
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
            # The remainder in the certificate's units: KL(q || q') = move / 2, so the move is twice the remaining gain.
            self.mean_move = 2.0 * remaining
            if remaining + self.noise_gain <= tolerance:
                return self._fixed_point(hyperparameters)

    def _fixed_point(self, hyperparameters: MixtureHyperparameters) -> FixedPoint:
        """The certified state as the outer loop's fixed point: the pseudo-likelihoods as the cavity, q's own metric
        for the prediction check, p_eff = sum_j omega_j v_j, and the linear response through ``_Response``."""
        omega = self.member_squares / self.noise
        live = self.variance > 0.0
        with np.errstate(divide="ignore"):
            tau = np.where(live, 1.0 / np.where(live, self.variance, 1.0), np.inf) - omega
        self.site_precision = tau
        self.effective = max(float(np.sum(omega * self.variance)), _EPSILON * tau.shape[0])
        started = time.perf_counter()
        response = _Response(self.design, self.noise * tau, live)
        self.profile["factorizations"] += 1
        self.profile["factor_seconds"] += time.perf_counter() - started
        self.profile["refreshes"] += 1
        design, noise, squares, variance = self.design, self.noise, self.member_squares, self.variance.copy()
        location_precision = np.empty_like(variance)
        for _, rows, terms in _class_terms(self.prior, hyperparameters.coefficients, Cavity(omega, self.shift), self.working_bytes):
            with np.errstate(divide="ignore", invalid="ignore"):
                precision = np.divide(terms.responsibility, terms.conditional_variance, out=np.zeros_like(terms.responsibility), where=terms.responsibility > 0.0)
            location_precision[rows] = precision.sum(axis=1)

        def cavity_response(mean_by_z: F64Array, _variance_by_z: F64Array) -> tuple[F64Array, F64Array]:
            # dm = (diag(tau) + Xp'Xp / sigma^2)^-1 diag(1 / v) (m_x E) on the live rows, 0 on the rest; then
            # dh = -(Xp'Xp - diag(||x_j||^2)) dm / sigma^2 and dP = 0 (module docstring).
            started = time.perf_counter()
            scaled = np.where(live[:, None], mean_by_z / np.where(live, variance, 1.0)[:, None], 0.0)
            mean_step = noise * response.solve(scaled)
            shift_step = -(design.back(design.image(mean_step)) - squares[:, None] * mean_step) / noise
            self.profile["response_seconds"] += time.perf_counter() - started
            self.profile["responses"] += 1
            return shift_step, np.zeros_like(shift_step)

        def norm(direction: F64Array) -> float:
            # Keeping the latent component gives KL = d² E[1/c] / 2. Marginalizing
            # that component can only decrease KL, so this bounds a location shift
            # even for non-Gaussian mixtures; 1 / Var(q) does not.
            values = np.asarray(direction, dtype=np.float64)
            moving = values != 0.0
            if np.any(moving & ~live):
                return np.inf
            return float(np.sum(np.square(values[moving]) * location_precision[moving]))

        posterior = GaussianPosterior(cavity_response=cavity_response, exact=True)
        return FixedPoint(
            cavity=Cavity(precision=omega, shift=self.shift.copy()), posterior=posterior, mean=self.mean.copy(),
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

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
for a scorer with K posterior draws: the remaining gain is estimated as the last sweep's gain g_t times
rho / (1 - rho), rho = g_t / g_(t-1) the measured contraction (the same extrapolation the EP oracles use for their
distance to the fixed point), so a call sweeps at least until two gains are measured; a gain below the ELBO's own
rounding (``_elbo``: a forward-error bound from the pieces' sizes) is resolved. The extrapolation is a geometric
estimate from two gains, not a bound: a slow mode that has not yet shown its rate is missed by it (the audit's
M12), which is why the Newton corrections through the response and their fresh decrement, not the extrapolation,
decide the certificate wherever a response exists. Nothing is clipped, damped or capped.

**The outer loop's view.** The pseudo-likelihoods are the ``Cavity`` the hyper step maximizes over: at the fixed
point q_j = t_j(x), so by the envelope theorem d ELBO*/dx = sum_j d log Z_j(x; omega_j, h_j)/dx at fixed cavities,
exactly the fixed-cavity gradient EP's outer loop uses. The total curvature B = -d2 ELBO*/dx2 needs the cavity's
response to x, with the noise re-solved as the fixed point re-solves it (sigma^2 is profiled, so B is the
profile's curvature): dh = -(Xp'Xp - diag ||x_j||^2) dm / sigma^2 - h dsigma^2 / sigma^2 and
domega = -omega dsigma^2 / sigma^2, with dm = f_h dh + f_omega domega + f_x E, f_h = v (the tilted variance),
f_omega = -(mu_3 + 2 m v) / 2 (the tilted mean's response to its precision, mu_3 the third central moment) and
f_x E = m_x E the fixed-cavity mean change, so

    (diag(1 / v - omega) + Xp'Xp / sigma^2) dm = diag(1 / v) m_x E - (dsigma^2 / sigma^2) c,
    c = h + f_omega omega / v,

dm = dm_0 - (dsigma^2 / sigma^2) dm_1 with one solve for dm_1 = R^-1 c per fixed point; and the noise's
stationarity (n - k) sigma^2 = ||r||^2 + sum_j ||x_j||^2 v_j gives the scalar, with dr = -Xp dm and
dv = v_x E + g dh + kappa domega (g = mu_3, kappa = -(mu_4 - v^2) / 2 - m mu_3, the tilted variance's responses),

    dsigma^2 [(n - k) - (2 r'Xp dm_1 + sum_j ||x_j||^2 (g_j dh_1j - kappa_j omega_j)) / sigma^2]
        = -2 r'Xp dm_0 + sum_j ||x_j||^2 (v_x E_j + g_j dh_0j),

dh_0 = -(Xp'Xp - diag ||x_j||^2) dm_0 / sigma^2 and dh_1 = (Xp'Xp - diag ||x_j||^2) dm_1 / sigma^2 - h.
``GaussianPosterior.cavity_response`` hands (dh, domega) to ``_total_curvature_columns``, which forms B from it as it
does from EP's response; the solve is ``_Response`` (the matrix is symmetric, not always positive definite). With
the noise's response left out (as it was), the outer loop's Newton steps in x converged linearly (a rate of 0.1 on
the test problem and 0.37 on gene 1 [real]: sixteen extra outer states), the signature of a model curvature that
misses the fixed point's own motion. The prediction check moves q's means in q's own local metric, the tilted
family's Fisher information in its mean coordinate, sum_j d_j^2 / v_j (``_fixed_point``'s ``norm``: half of it is
KL(q || q moved)'s leading term, exact only where q_j is Gaussian, not a finite-step KL),
and p_eff = sum_j omega_j v_j (tr(Xp Sigma_q Xp') / sigma^2 for the product q). The draws are q's own, so they are
conditional variational draws of the product approximation at the fitted hyperparameters and not of the posterior
it approximates (``draws``): each member's node from its responsibilities, then its conditional normal.
"""

from __future__ import annotations

import time
from typing import Sequence

import numba
import numpy as np
from scipy import linalg

from sv_pgs._typing import F64Array, I64Array
from sv_pgs import engine_kernels
from sv_pgs.scale_mixture_ep import (
    _DEVICE,
    Cavity,
    FixedPoint,
    GaussianPosterior,
    MixtureHyperparameters,
    ScaleMixturePrior,
    _components,
    _data_value,
    _row_chunks,
    class_log_density,
    log_scale,
    noise_gain,
    tilted_cumulants,
)
from sv_pgs.small_n import DenseStatistics, _Design, _new_profile

_EPSILON = float(np.finfo(np.float64).eps)


@numba.njit(cache=True, fastmath=True)
def _sweep(design, squares, members, class_index, log_density, node_variance, log_node_variance, noise, mean, residual, variance, shift, third, fourth):
    """One coordinate-ascent sweep over the members in order, in place: ``mean`` and ``variance`` (each q_j's
    moments), ``residual`` (r = y_P - Xp mean) and ``shift`` (the h_j each q_j was built from). Returns
    (sum_j KL(q_j || p_j), sum_j ||x_j||^2 v_j, ||r||^2, the sizes of the KL terms' pieces) at the sweep's end, so the ELBO
    and its rounding bound are exact there.

    ``design`` is Xp over the groups (n x groups, Fortran order), ``squares`` their ||x_g||^2, ``members[j]`` member
    j's group; ``log_density`` is (classes x nodes), ``node_variance`` (members x nodes) each member's variance at
    each node, u_j e^{t_k}, and ``log_node_variance`` its log (read only where the variance overflowed), formed once per hyperparameters (``MeanFieldFixedPoints._node_variance``: the sweep's
    exponentials are its cost, and this one does not move between sweeps). The node terms are
    ``scale_mixture_ep._kernel_terms``' own, with the same overflow limits: a node whose variance overflows
    contributes conditional variance 1 / omega and weight 0. ``third`` and ``fourth`` receive each q_j's third and
    fourth central moments (the noise's response, ``MeanFieldFixedPoints._fixed_point``): with d_k = h c_k - m,
    mu_3 = sum_k w_k (d_k^3 + 3 c_k d_k) and mu_4 = sum_k w_k (d_k^4 + 6 c_k d_k^2 + 3 c_k^2).

    Compiled with fastmath: the node sums and the sample dot products may be reassociated and vectorized, which the
    ELBO's rounding bound (``_elbo``: N eps of the pieces' sizes, for any summation order) already covers; on
    ENSG00000254709.8 [real, 37,106 members, 88 nodes, 534 samples] a sweep took 86 ms against 183 ms, with the
    divergence equal to six decimals. Every value stays finite by construction (the overflow branch), which fastmath
    assumes."""
    sample_count = design.shape[0]
    member_count = members.shape[0]
    node_count = node_variance.shape[1]
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
            variance_node = node_variance[member, node]
            ratio = variance_node * omega
            if ratio == np.inf:
                # The kernel's own limit (``scale_mixture_ep._kernel_terms``): log(1 + v omega) = log v + log omega
                # to rounding, so the weight is small and finite, never -inf.
                conditional[node] = 1.0 / omega
                log_weights[node] = log_density[row, node] - 0.5 * (log_node_variance[member, node] + np.log(omega)) + 0.5 * h * h * conditional[node]
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
        new_third = 0.0
        new_fourth = 0.0
        for node in range(node_count):
            offset = h * conditional[node] - new_mean
            weight = log_weights[node] / total
            new_variance += weight * (conditional[node] + offset * offset)
            new_third += weight * (offset * offset * offset + 3.0 * conditional[node] * offset)
            new_fourth += weight * (offset**4 + 6.0 * conditional[node] * offset * offset + 3.0 * conditional[node] * conditional[node])
        step = new_mean - old_mean
        if step != 0.0:
            for sample in range(sample_count):
                residual[sample] -= design[sample, group] * step
        mean[member] = new_mean
        variance[member] = new_variance
        third[member] = new_third
        fourth[member] = new_fourth
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


@numba.njit(cache=True)
def _sample_nodes(responsibility, uniform, nodes):
    """``nodes[j, k]`` is the number of row j's cumulative responsibilities strictly below ``uniform[j, k]``, capped at
    the last node: the inverse CDF of the row's node law (``MeanFieldFixedPoints.draws``), by bisection. One row's
    cumulative sums live at a time, so nothing of the size (rows x nodes x draws) is formed."""
    row_count, node_count = responsibility.shape
    draw_count = uniform.shape[1]
    cumulative = np.empty(node_count)
    for row in range(row_count):
        total = 0.0
        for node in range(node_count):
            total += responsibility[row, node]
            cumulative[node] = total
        for draw in range(draw_count):
            value = uniform[row, draw]
            low = 0
            high = node_count
            while low < high:
                middle = (low + high) // 2
                if cumulative[middle] < value:
                    low = middle + 1
                else:
                    high = middle
            nodes[row, draw] = min(low, node_count - 1)


class _SweepBudget(Exception):
    """A second-start solve used the first start's count of sweeps without converging: abandoned, not refused."""


class _Response:
    """(Xp'Xp + diag t)^-1 applied to columns, for sites t of either sign: the mean-field fixed point's response
    matrix diag(tau) + Xp'Xp / sigma^2 (t = sigma^2 tau, module docstring) is symmetric but need not be positive
    definite. A member whose tilted variance exceeds its pseudo-likelihood's, v_j > 1 / omega_j, has tau_j < 0, and
    two tied members with such sites make it indefinite along their difference; it is still nonsingular wherever the
    fixed point has a linear response.

    The bulk P (t_j >= ||x_j||^2, where Woodbury keeps every digit: ``small_n._Kernel``) enters through the kernel
    K = I + Xp_P T_P^-1 Xp_P' (positive definite), and the rest N exactly through its Schur complement
    S = T_N + Xp_N' K^-1 Xp_N (|N| x |N|, symmetric, any sign), factored by LU. A dead row (a point-mass tilted law,
    t = infinity) has no response: it is a bulk row with T^-1 = 0.

    |N| is what the split costs, and nothing in the split bounds it a priori, so it was measured on bench-real
    genes [real], loso/AFR snv: over the first 40 responses of ENSG00000254709.8 (p 37,106 members, n 534) |N| ran
    0 to 5, median 4, and over the first 25 of ENSG00000100385.14 (p 28,920) 1 to 6, median 4. Every member was
    live in both, and the bulk was never empty (at worst 5 of 37,106 and 6 of 28,920 rows outside it), so S is a
    handful of rows and the K = I case never arose. The cost is the kernel's own n x n dsyrk over the p columns,
    0.18 s and 0.10 s median per response, which is that product's floor. Should a fit ever reach a large N,
    |N|^2 memory and |N|^3 time are what it pays."""

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
        self.third = np.zeros(prior.variant_count)
        self.fourth = np.zeros(prior.variant_count)
        self.shift = np.zeros(prior.variant_count)
        self.residual = np.array(statistics.projected_target, dtype=np.float64, copy=True)
        self.site_precision = np.zeros(prior.variant_count)
        self.effective = float(prior.variant_count)
        self.mean_move = np.inf
        self.noise_gain = np.inf
        self.refusals: list[str] = []
        self.profile = _new_profile() | {"sweeps": 0, "sweep_seconds": 0.0, "elbo": -np.inf}
        self._node_variance_key: bytes | None = None
        self._node_variance_table = (np.zeros((0, 0)), np.zeros((0, 0)))
        # The last fixed point's response factorization and its noise: the metric of the corrections between sweeps.
        self._response: _Response | None = None
        self._response_noise = float(start_noise)
        # A parallel pass (``_pass``) leaves the third and fourth moments to the fixed point that needs them.
        self._moments_stale = False
        self._held_device: dict | None = None
        # The start's state, from which every call is also solved cold (``__call__``).
        self._cold: dict | None = self._snapshot()

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

    def _node_variance(self, hyperparameters: MixtureHyperparameters) -> tuple[F64Array, F64Array]:
        """(members x nodes) u_j e^{t_k} at these hyperparameters and its log, held for the next sweep at the same ones."""
        key = hyperparameters.coefficients.tobytes()
        if self._node_variance_key != key:
            log_table = log_scale(self.prior, hyperparameters.coefficients)[:, None] + self.prior.log_variance_grid[None, :]
            with np.errstate(over="ignore"):
                self._node_variance_table = (np.exp(log_table), log_table)
            self._node_variance_key = key
        return self._node_variance_table

    def _sweep(self, hyperparameters: MixtureHyperparameters) -> tuple[float, float, float, float]:
        started = time.perf_counter()
        prior = self.prior
        values = _sweep(
            self.projected, self.group_squares, self.members, self.class_index,
            np.ascontiguousarray(class_log_density(prior, hyperparameters.coefficients)), *self._node_variance(hyperparameters),
            self.noise, self.mean, self.residual, self.variance, self.shift, self.third, self.fourth,
        )
        self.profile["sweeps"] += 1
        self.profile["passes"] += 1
        self.profile["sweep_seconds"] += time.perf_counter() - started
        return values

    def _device(self) -> dict | None:
        """The pass's member-side arrays on the fit's device (``scale_mixture_ep.device_scope``), held; None on the host."""
        xp = _DEVICE.get()
        if xp is np:
            return None
        held = self._held_device
        if held is None or held["xp"] is not xp:
            held = {
                "xp": xp, "members": xp.asarray(self.members), "squares": xp.asarray(self.member_squares),
                "class_index": xp.asarray(self.class_index), "target": xp.asarray(np.asarray(self.statistics.projected_target, dtype=np.float64)),
                "scales_key": None, "scales": None,
            }
            self._held_device = held
        return held

    def _pass(self, hyperparameters: MixtureHyperparameters, baseline: float | None, direction: F64Array | None) -> tuple[float, float, float, float]:
        """One pass over the members on the device, every site at once, in place of a sweep: the same ELBO, the same
        fixed point, no serial chain.

        Every q_j is the prior tilted at a site (h_j, omega_j), so the ELBO is a function L(h) of the sites (each
        term of ``_sweep``'s sum, summed in parallel), with dL/dh_j = v_j (h*_j - h_j) where h*_j is the site the
        residual now puts member j at (``_stale_gap``): a sweep sets each h_j to h*_j in turn, this pass moves
        all of them along a direction dh at once and takes the step the ELBO prefers. The direction is Newton's in
        the means through the last fixed point's response, dm = R^-1 (h* - h) (``_decrement``), in the sites
        dh = dm / v, or the gap itself where there is no response yet (the Jacobi step: every site to h*_j). The
        step is the unit one, the vertex of the parabola through L(0), L'(0) and L(1) when it lies inside, then
        halvings while a step could still gain the tolerance; the best of them is taken. ``baseline`` is the ELBO
        the state has (with the noise's pending gain applied), and a pass that beats it nowhere is replaced by a
        sweep, which cannot lose (each of its steps maximizes over one q_j): the pass is a faster route to the same
        ascent, never a different one. On ENSG00000254709.8 [real, p 37,106, K 88, n 534, one A40] one evaluation
        of L took 3.9 ms against 72 ms for a sweep, at the same fixed point (the prototype, lead/pmf1_*.log).

        Returns ``_sweep``'s pieces at the state it leaves."""
        held = self._device()
        assert held is not None
        xp = held["xp"]
        _xp, carriers, basis = self.design._device()
        prior = self.prior
        tolerance = 0.5 / self.draw_count
        key = hyperparameters.coefficients.tobytes()
        if held["scales_key"] != key:
            held["scales"] = xp.asarray(log_scale(prior, hyperparameters.coefficients))
            held["scales_key"] = key
        log_density = np.ascontiguousarray(class_log_density(prior, hyperparameters.coefficients))
        squares, members, target = held["squares"], held["members"], held["target"]
        noise = self.noise
        omega = squares / noise
        mean, variance = xp.asarray(self.mean), xp.asarray(self.variance)
        sites, residual = xp.asarray(self.shift), xp.asarray(self.residual)
        tied = self.design.tied
        group_count = self.design.group_count

        def image(values):
            grouped = xp.bincount(members, weights=values, minlength=group_count) if tied else values
            product = carriers @ grouped
            return product - basis @ (basis.T @ product) if basis.shape[1] else product

        def back(samples):
            projected = samples - basis @ (basis.T @ samples) if basis.shape[1] else samples
            grouped = carriers.T @ projected
            return grouped[members] if tied else grouped

        located = (back(residual) + squares * mean) / noise
        gap = located - sites
        directions = [gap]
        if direction is not None:
            live = variance > 0.0
            directions.insert(0, xp.where(live, xp.asarray(direction) / xp.where(live, variance, 1.0), gap))

        def evaluate(alpha: float):
            moved = sites + alpha * step
            log_normalizer, new_mean, new_variance, _flag = engine_kernels.tilted_moments(
                xp, held["class_index"], log_density, held["scales"], prior.log_variance_grid, omega, moved, self.working_bytes, check=False,
            )
            new_residual = target - image(new_mean)
            pull = moved * new_mean
            shrink = 0.5 * omega * (new_mean * new_mean + new_variance)
            pieces = xp.asnumpy(xp.stack([
                xp.sum(pull - shrink - log_normalizer), squares @ new_variance, new_residual @ new_residual,
                xp.sum(xp.abs(pull) + shrink + xp.abs(log_normalizer)),
            ]))
            pieces = tuple(float(piece) for piece in pieces)
            value, _rounding = self._elbo(*pieces)
            return value, pieces, (moved, new_mean, new_variance, new_residual)

        # Newton's direction first, where there is one and it ascends here (the response is the last fixed point's,
        # and the noise may have moved since: a stale direction can point downhill); the gap's own direction next
        # (the Jacobi step, always an ascent direction where the gap is not zero); the sweep last.
        candidates: list = []
        for step in directions:
            slope = float(variance @ (gap * step))
            if slope <= 0.0 and step is not gap:
                continue
            candidates.append(evaluate(1.0))
            if baseline is None:
                break
            curvature = candidates[-1][0] - baseline - slope
            alpha = 1.0
            if curvature < 0.0 and 0.0 < -slope / (2.0 * curvature) < 1.0:
                alpha = -slope / (2.0 * curvature)
                candidates.append(evaluate(alpha))
            while max(candidate[0] for candidate in candidates) <= baseline and alpha * slope > tolerance:
                alpha = 0.5 * alpha
                candidates.append(evaluate(alpha))
            if max(candidate[0] for candidate in candidates) > baseline:
                break
        best = max(candidates, key=lambda candidate: candidate[0])
        self.profile["parallel_passes"] = self.profile.get("parallel_passes", 0) + 1
        self.profile["parallel_evaluations"] = self.profile.get("parallel_evaluations", 0) + len(candidates)
        if baseline is not None and best[0] <= baseline:
            self.profile["parallel_fallbacks"] = self.profile.get("parallel_fallbacks", 0) + 1
            return self._sweep(hyperparameters)
        moved, new_mean, new_variance, new_residual = best[2]
        self.shift, self.mean, self.variance, self.residual = (xp.asnumpy(values) for values in (moved, new_mean, new_variance, new_residual))
        self._moments_stale = True
        self.profile["sweeps"] += 1
        self.profile["passes"] += 1
        return best[1]

    def _snapshot(self) -> dict:
        return {
            "mean": self.mean.copy(), "variance": self.variance.copy(), "shift": self.shift.copy(), "residual": self.residual.copy(),
            "third": self.third.copy(), "fourth": self.fourth.copy(),
            "noise": self.noise, "site_precision": self.site_precision.copy(), "effective": self.effective,
            # The response factor belongs to the state it was built at: a rejected trial's must not steer the next
            # solve from the restored state (the factor is immutable, so the reference is the state).
            "response": self._response, "response_noise": self._response_noise,
            "mean_move": self.mean_move, "noise_gain": self.noise_gain, "elbo": self.profile["elbo"],
            "moments_stale": self._moments_stale,
        }

    def _restore(self, snapshot: dict) -> None:
        self.mean, self.variance, self.shift, self.residual, self.third, self.fourth = (
            snapshot[name].copy() for name in ("mean", "variance", "shift", "residual", "third", "fourth")
        )
        self.noise, self.site_precision, self.effective = snapshot["noise"], snapshot["site_precision"].copy(), snapshot["effective"]
        self._response, self._response_noise = snapshot["response"], snapshot["response_noise"]
        self.mean_move, self.noise_gain, self.profile["elbo"] = snapshot["mean_move"], snapshot["noise_gain"], snapshot["elbo"]
        self._moments_stale = snapshot["moments_stale"]

    def __call__(self, hyperparameters: Sequence[MixtureHyperparameters]) -> list[FixedPoint | None]:
        """The fixed point of the higher ELBO between the solve from the carried state and the solve from the start.

        Coordinate ascent has several fixed points at one x, and neither start finds the better one always: on
        bench-real chr22 [real, loso/AFR, snv] the cold solve's ELBO was above the carried state's by up to 24 nats
        at 11 of 13 calls on ENSG00000075234.17 (19 at the fit's end, where the held-out r^2 was 0.457 cold against
        0.388 carried), below it by 0.6 on ENSG00000100385.14, and equal to 0.01 nats on two other genes. The ELBO
        is a lower bound on the evidence at this x, so the higher one is the better approximation, and the oracle
        is then nearer a function of x than of the path that reached it. The first call's two starts are one.

        The cold solve gets the carried solve's own effort, its count of sweeps, and no more: it is the second
        candidate, not a requirement, and from the start's zero means a trial's wide prior can leave it an ELBO of
        -1e5 climbing by 4e4 a sweep with an extrapolated remainder of 3e8 (the 500-gene run [real]: fits of hours
        where the carried solve had converged in seconds)."""
        (model_hyperparameters,) = hyperparameters
        self.profile["fixed_point_calls"] += 1
        entry = self._snapshot()
        solved: list[tuple[float, FixedPoint, dict]] = []
        budget: int | None = None
        for start in (entry, self._cold) if self.profile["fixed_point_calls"] > 1 else (entry,):
            self._restore(start)
            before = self.profile["sweeps"]
            try:
                point = self._solve(model_hyperparameters, sweep_budget=budget)
            except _SweepBudget:
                self.profile["cold_abandoned"] = self.profile.get("cold_abandoned", 0) + 1
                continue
            except (FloatingPointError, np.linalg.LinAlgError) as error:
                self.refusals.append(str(error))
                continue
            if budget is None:
                budget = self.profile["carried_sweeps"] = self.profile["sweeps"] - before
            solved.append((float(self.profile["elbo"]), point, self._snapshot()))
        if not solved:
            self._restore(entry)
            return [None]
        _value, point, state = max(solved, key=lambda item: item[0])
        self._restore(state)
        return [point]

    def _stale_gap(self) -> F64Array:
        """h'_j - h_j on the live rows: each pseudo-likelihood's location recomputed from the sweep's final residual
        against the one its site was built from, the ELBO's gradient in q's means at the sweep's end (a site updated
        early in the sweep is stale by the later sites' moves). Zero where q_j is a point mass."""
        live = self.variance > 0.0
        located = (self.design.back(self.residual) + self.member_squares * self.mean) / self.noise
        return np.where(live, located - self.shift, 0.0)

    def _decrement(self, gap: F64Array, response: _Response, response_noise: float) -> tuple[float, F64Array]:
        """(the Newton decrement of the fixed-point equation in the means, its step): the map h -> T(h) whose fixed
        point q is has Jacobian -(Xp'Xp - diag ||x_j||^2) diag(v) / sigma^2, so Newton's step in the means is
        dm = R^-1 (h' - h) with R = diag(tau) + Xp'Xp / sigma^2 the response matrix (module docstring), and the
        decrement (h' - h)' dm / 2 is the gain of the ELBO's own quadratic model in the means at fixed noise: in the
        means the ELBO is -||y_P - Xp m||^2 / (2 sigma^2) - sum_j [J_j(m_j) - omega_j m_j^2 / 2] (the likelihood's
        variance term cancels the KL's), J_j the Legendre transform of log Z_j with J_j'' = 1 / v_j, so its negative
        Hessian is R itself. R need not be positive definite (``_Response``): where the form is negative the model
        has no maximum along the gap and the decrement is no bound at all (``_solve`` then neither steps nor
        certifies by it). With a response held from an earlier fixed point the decrement is that metric's estimate;
        the fresh one at the returned fixed point is the certificate's."""
        step = response_noise * response.solve(gap[:, None])[:, 0]
        return 0.5 * float(gap @ step), step

    def _solve(self, hyperparameters: MixtureHyperparameters, sweep_budget: int | None = None) -> FixedPoint:
        """Sweeps at the current noise, with Newton corrections in the means between them, until the remaining gain
        (module docstring) is within the tolerance. The noise moves to its stationary value only between sweeps, so
        the returned state (q, sigma^2) is the one the last sweep built: its pseudo-likelihoods are the cavity,
        exactly. The noise's pending gain counts as remaining.

        Coordinate ascent alone crawls along the design's correlated directions (gene 1 [real]: sweeps stopped by
        the geometric extrapolation of their ELBO gains left q's means where the held-out r^2 was 0.0339, and
        sweeping until the means settled gave 0.0385: the two-gain extrapolation understates the remaining gain
        where the slow modes have not yet shown their rate). So after each sweep the stale gap h' - h (the ELBO's
        gradient in the means) is taken through the last fixed point's response factorization as Newton's step in
        the means, the next sweep re-tilts every site at the moved residual, and the remaining gain is the Newton
        decrement, read with the fresh factorization at the returned fixed point before it is returned. A correction
        the next sweep does not confirm (its ELBO below the pre-correction sweep's) is undone, and the call sweeps
        on without corrections."""
        tolerance = 0.5 / self.draw_count
        # The hyperparameters changed since the last call, so the state's ELBO is unknown until a sweep measures it:
        # the first sweep's gain is not a gain, and the extrapolation starts at the second.
        elbo: float | None = None
        gain: float | None = None
        previous_gain: float | None = None
        pending_noise: float | None = None
        corrections = self._response is not None
        correction: tuple[dict, float] | None = None
        sweeps_at_entry = self.profile["sweeps"]
        # On the device the pass takes Newton's direction itself (``_pass``); the sweep path applies it between sweeps.
        parallel = self._device() is not None
        direction: F64Array | None = None
        promised = 0.0
        while True:
            if sweep_budget is not None and self.profile["sweeps"] - sweeps_at_entry >= sweep_budget:
                raise _SweepBudget()
            if pending_noise is not None:
                elbo = elbo + self.noise_gain if elbo is not None else None
                self.noise = pending_noise
            if parallel:
                along_newton = direction is not None
                divergence, weighted_variance, residual_square, sizes = self._pass(hyperparameters, elbo, direction)
                if along_newton and elbo is not None and promised > tolerance:
                    value, _rounding = self._elbo(divergence, weighted_variance, residual_square, sizes)
                    if value - elbo <= tolerance:
                        # The step's decrement promised more than the tolerance and the pass along it realized less:
                        # the quadratic model in the sites is wrong at this scale (a site far from its maximum with
                        # a small variance takes a huge Newton step that the line search cuts to nothing; the tilted
                        # moments saturate where the model is quadratic). A sweep maximizes each site exactly, as the
                        # sweep path's corrections are confirmed by one, and the pass resumes from where it leaves
                        # (ENSG00000179399.15 snv in the GPU benchmark chunk [real]: passes of gains below the
                        # tolerance against a stale decrement above it, for an hour).
                        self.profile["parallel_sweeps"] = self.profile.get("parallel_sweeps", 0) + 1
                        divergence, weighted_variance, residual_square, sizes = self._sweep(hyperparameters)
                direction = None
                promised = 0.0
                if self._response is None:
                    # Before the first fixed point the passes would take the gap's own direction, which crawls along
                    # the design's correlated directions as the sweeps do (84 passes on ENSG00000100385.14 [real]
                    # against 4 sweeps): the response at this state gives them Newton's direction at once.
                    self._response, self._response_noise = self._response_at(), self.noise
                    self.profile["factorizations"] += 1
                    corrections = True
            else:
                divergence, weighted_variance, residual_square, sizes = self._sweep(hyperparameters)
            if not (np.isfinite(divergence) and np.isfinite(weighted_variance) and np.isfinite(residual_square)):
                raise FloatingPointError("a mean-field sweep is not finite")
            value, rounding = self._elbo(divergence, weighted_variance, residual_square, sizes)
            # The noise's stationary value at this q, and the gain it would bring (both exact); applied before the
            # next sweep, if there is one.
            pending_noise = (residual_square + weighted_variance) / self.residual_dimension
            self.noise_gain = noise_gain(pending_noise, self.noise, self.sample_count, self.covariate_count)
            if correction is not None:
                # The sweep after a correction: confirmed where the ELBO is at or above the pre-correction sweep's.
                snapshot, before = correction
                correction = None
                if value < before - rounding:
                    self._restore(snapshot)
                    corrections = False
                    elbo, gain, previous_gain, pending_noise = before, None, None, None
                    continue
                elbo, gain, previous_gain = value, None, None
            else:
                gain = (value - elbo) if elbo is not None else None
                elbo = value
                if gain is not None and gain < -rounding:
                    raise FloatingPointError(f"a mean-field sweep lowered the ELBO by {-gain:.3g} nats: the bound's ascent is broken")
                if parallel and along_newton and gain is not None and gain <= rounding:
                    # The pass along the held response's Newton step gained nothing above rounding while that step's
                    # decrement promised more: the response is another fixed point's and its model is wrong here, as
                    # a correction the next sweep does not confirm is on the sweep path. Its decrement no longer
                    # bounds the remainder; the sweeps' extrapolation does, and the fresh response at the fixed point
                    # gives the next direction (ENSG00000115806.13 sv [real, p 49]: 415 passes of 1e-12 nats each
                    # against a stale decrement above the tolerance, until the run was killed).
                    corrections = False
            self.profile["elbo"] = value
            gap = self._stale_gap()
            newton: float | None = None
            if corrections and self._response is not None:
                newton, step = self._decrement(gap, self._response, self._response_noise)
                if newton < 0.0:
                    # R is indefinite along the gap: no step and no bound from it; the sweeps' extrapolation governs.
                    newton, corrections = None, False
                elif parallel:
                    # The next pass moves along this step; the decrement is the remaining gain's estimate.
                    direction, promised = step, newton
                elif newton > tolerance:
                    # Newton's step in the means; the next sweep re-tilts every site at the moved residual.
                    correction = (self._snapshot(), value)
                    self.mean = self.mean + step
                    self.residual = self.residual - self.design.image(step)
                    self.profile["passes"] += 1
                    continue
            if newton is not None:
                remaining = newton
            else:
                # Without a response yet (the fit's first fixed point), or after an unconfirmed correction: the
                # sweeps' geometric extrapolation from the last two gains.
                if gain is None:
                    remaining = np.inf
                elif gain <= rounding:
                    remaining = 0.0
                elif previous_gain is not None and previous_gain > 0.0:
                    rate = gain / previous_gain
                    remaining = gain * rate / (1.0 - rate) if rate < 1.0 else np.inf
                else:
                    remaining = np.inf
                if gain is not None:
                    previous_gain = max(float(gain), 0.0)
            # The remainder in the certificate's units: KL(q || q') = move / 2, so the move is twice the remaining gain.
            self.mean_move = 2.0 * remaining
            if remaining + self.noise_gain <= tolerance:
                point = self._fixed_point(hyperparameters)
                # The certificate reads the decrement with the fresh factorization; where it is not within the
                # tolerance the corrections continue in that metric.
                assert self._response is not None
                fresh, fresh_step = self._decrement(gap, self._response, self._response_noise)
                if fresh >= 0.0:
                    # A negative form is no bound (R indefinite along the gap): the measured remainder stands.
                    remaining = fresh
                    if parallel:
                        direction, promised = fresh_step, fresh
                self.mean_move = 2.0 * remaining
                if remaining + self.noise_gain <= tolerance:
                    return point
                corrections = True

    def _response_at(self) -> _Response:
        """The response factorization at the current state (``_Response``): the fixed point's own when built there."""
        omega = self.member_squares / self.noise
        live = self.variance > 0.0
        with np.errstate(divide="ignore"):
            tau = np.where(live, 1.0 / np.where(live, self.variance, 1.0), np.inf) - omega
        return _Response(self.design, self.noise * tau, live)

    def _fixed_point(self, hyperparameters: MixtureHyperparameters) -> FixedPoint:
        """The certified state as the outer loop's fixed point: the pseudo-likelihoods as the cavity, q's own local
        metric for the prediction check (``norm``), p_eff = sum_j omega_j v_j, and the linear response through
        ``_Response``."""
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
        self._response, self._response_noise = response, self.noise
        if self._moments_stale:
            self.third, self.fourth = tilted_cumulants(self.prior, hyperparameters, Cavity(precision=omega, shift=self.shift), self.working_bytes)
            self._moments_stale = False
        design, noise, squares, variance = self.design, self.noise, self.member_squares, self.variance.copy()
        mean, shift, residual = self.mean.copy(), self.shift.copy(), self.residual.copy()
        third, fourth = self.third.copy(), self.fourth.copy()
        # The tilted moments' responses to the pseudo-likelihood's precision (module docstring), zero on a dead row.
        mean_by_omega = np.where(live, -0.5 * (third + 2.0 * mean * variance), 0.0)
        variance_by_shift = np.where(live, third, 0.0)
        variance_by_omega = np.where(live, -0.5 * (fourth - variance * variance) - mean * third, 0.0)
        residual_dimension = float(self.residual_dimension)
        noise_solve: dict[str, object] = {}

        def off_diagonal_gram(columns: F64Array) -> F64Array:
            return design.back(design.image(columns)) - squares[:, None] * columns

        def noise_terms() -> tuple[F64Array, F64Array, float]:
            # dm_1 = R^-1 c, dh_1 and the scalar on dsigma^2 (module docstring): once per fixed point.
            if not noise_solve:
                coupling = np.where(live, shift + mean_by_omega * omega / np.where(live, variance, 1.0), 0.0)
                mean_one = noise * response.solve(coupling[:, None])
                shift_one = off_diagonal_gram(mean_one) / noise - shift[:, None]
                scalar = residual_dimension - (
                    2.0 * float(residual @ design.image(mean_one)[:, 0])
                    + float(squares @ (variance_by_shift * shift_one[:, 0] - variance_by_omega * omega))
                ) / noise
                noise_solve.update(mean_one=mean_one, shift_one=shift_one, scalar=scalar)
            return noise_solve["mean_one"], noise_solve["shift_one"], noise_solve["scalar"]  # type: ignore[return-value]

        def cavity_response(mean_by_z: F64Array, variance_by_z: F64Array, _relative_tolerance: float = 0.0) -> tuple[F64Array, F64Array]:
            # (The tolerance is the streamed route's, whose solves are iterative; this factor is exact.)
            # dm_0 = (diag(tau) + Xp'Xp / sigma^2)^-1 diag(1 / v) (m_x E) on the live rows, 0 on the rest, then the
            # noise's own response and through it every pseudo-likelihood's (module docstring).
            started = time.perf_counter()
            scaled = np.where(live[:, None], mean_by_z / np.where(live, variance, 1.0)[:, None], 0.0)
            mean_step = noise * response.solve(scaled)
            shift_step = -off_diagonal_gram(mean_step) / noise
            _mean_one, shift_one, scalar = noise_terms()
            right = -2.0 * (residual @ design.image(mean_step)) + squares @ (
                np.where(live[:, None], variance_by_z, 0.0) + variance_by_shift[:, None] * shift_step
            )
            noise_step = right / scalar
            relative = (noise_step / noise)[None, :]
            shift_step = shift_step + shift_one * relative
            precision_step = -omega[:, None] * relative
            self.profile["response_seconds"] += time.perf_counter() - started
            self.profile["responses"] += 1
            return shift_step, precision_step

        def norm(direction: F64Array) -> float:
            # The squared length of a shift of q's means in q's own LOCAL metric. Member j's law is the prior tilted
            # by exp(h_j b - omega_j b^2 / 2); at fixed omega_j that is an exponential family in h_j with
            # d log Z_j / dh_j = m_j and d2 log Z_j / dh_j^2 = v_j, so its Fisher information in the mean coordinate
            # m_j is 1 / v_j and a product's is diag(1 / v). The move is sum_j d_j^2 / v_j, whose half is the leading
            # term of KL(q || q moved) and not that KL: the expansion's remainder is O(d^3), and the identity holds
            # exactly only where q_j is Gaussian. A non-Gaussian scale mixture's location Fisher information is not
            # its inverse variance (0.9 N(0, 0.01) + 0.1 N(0, 10) has 86.7 against 1 / 1.009), so this metric is the
            # tilted family's own, in its mean coordinate, not a location family's. A dead row moves nowhere.
            values = np.asarray(direction, dtype=np.float64)
            moving = values != 0.0
            if np.any(moving & ~live):
                return np.inf
            return float(np.sum(np.square(values[live]) / variance[live]))

        posterior = GaussianPosterior(cavity_response=cavity_response, exact=True)
        # The solver's state at this point, so the outer loop can put it back before a trial (``FixedPoint.restore``).
        snapshot = self._snapshot()
        cavity = Cavity(precision=omega, shift=self.shift.copy())
        # E's offset (``FixedPoint.evidence_offset``): the ELBO the last sweep built less the fixed-cavity F there.
        offset = float(self.profile["elbo"]) - _data_value(self.prior, hyperparameters.coefficients, cavity, self.working_bytes)
        return FixedPoint(
            cavity=cavity, posterior=posterior, mean=self.mean.copy(), precision_norm=norm, effective_effects=float(self.effective),
            restore=lambda: self._restore(snapshot), evidence_offset=offset,
        )

    def draws(self, hyperparameters: MixtureHyperparameters, generator: np.random.Generator, draw_count: int) -> F64Array:
        """(p x K) draws from q itself, member by member: each member's node from its responsibilities at its
        pseudo-likelihood, then its conditional normal N(h c_k, c_k) (``scale_mixture_ep._components``).

        These are conditional variational draws of the product approximation q = prod_j q_j at the fitted
        hyperparameters, not draws of the posterior q approximates. A direction that mixes members in LD has a
        variance under the product that is neither an upper nor a lower bound on the posterior's (for a two-effect
        Gaussian posterior with unit diagonal precision and off-diagonal 0.9 the product gives 2 against the
        posterior's 20 along [1, -1] and 1.05 along [1, 1]), and the hyperparameters are held fixed, so the draws
        carry the posterior's marginal spread per member and no joint-posterior or predictive-interval coverage.

        The rows are taken in pieces whose per-row intermediates fit ``working_bytes``
        (``scale_mixture_ep._row_chunks``): the widest of them are the kernel's node-wide forms and the sampler's
        draw-wide arrays, so the width to budget is their sum. ``_sample_nodes`` samples by inverse CDF inside a
        piece, so nothing of the size (rows x nodes x draws) is ever formed. A piece boundary moves which value of
        the generator's stream lands where, so the law is preserved and the numbers are not."""
        return product_draws(
            self.prior, hyperparameters.coefficients, self.member_squares / self.noise, self.shift, self.class_index, generator, draw_count, self.working_bytes
        )


def product_draws(
    prior: ScaleMixturePrior, coefficients: F64Array, omega: F64Array, shift: F64Array, class_index: I64Array,
    generator: np.random.Generator, draw_count: int, working_bytes: int,
) -> F64Array:
    """``MeanFieldFixedPoints.draws`` for any product q given by its pseudo-likelihoods (omega, shift) per member:
    the dense route's and the streamed full-data route's are one function."""
    count = int(draw_count)
    log_density = class_log_density(prior, coefficients)
    scales = log_scale(prior, coefficients)
    draws = np.empty((prior.variant_count, count))
    for class_position in range(log_density.shape[0]):
        rows = np.flatnonzero(class_index == class_position)
        if rows.size == 0:
            continue
        for piece in _row_chunks(rows, prior.grid_size + count, working_bytes):
            terms = _components(log_density[class_position], scales[piece], prior.log_variance_grid, omega[piece], shift[piece])
            uniform = generator.random((piece.shape[0], count))
            nodes = np.empty((piece.shape[0], count), dtype=np.int64)
            _sample_nodes(np.ascontiguousarray(terms.responsibility), uniform, nodes)
            conditional = np.take_along_axis(terms.conditional_variance, nodes, axis=1)
            draws[piece] = shift[piece][:, None] * conditional + np.sqrt(conditional) * generator.standard_normal(conditional.shape)
    return draws

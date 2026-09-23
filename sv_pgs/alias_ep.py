"""EP's fixed point at gene scale on the alias groups' sums, with joint cluster sites where a group's own site breaks
EP's numerical contract.

Units. Every exact alias group is one coordinate of the likelihood (``alias_groups``) with its induced prior
(``alias_laws.GroupLaws``). A unit is one group with its own site, or a cluster of groups with one joint Gaussian site
on their sums whose tilted law is the cluster's exact joint law: the members' scale mixtures times the Gaussian cavity
on the sums, by ``scale_sampler.cluster_tilted_moments`` over the sums' law atoms (enumerated where that is cheaper,
sampled to the resolution otherwise; the decode's once over the cluster's members), so the joint site carries every dependence inside the cluster that one site per
group cannot.

The contract and the dial. A sweep updates every single group by its exact sequential update
(``sequential_ep.SequentialSweep``) and then every cluster; afterwards every unit's moment-matching residual (the KL
of its tilted law from q's marginal, nats) is measured. EP has reached its fixed point when the residuals summed over the units are
within the resolution 1 / (2 K) (less each sampled cluster's own Monte Carlo floor, the KL its moments' errors make in
expectation) and q has stopped moving (the sweep's summed KL move of the groups' marginals within the resolution). A unit breaks the contract where its update is refused (no damping keeps every cavity finite) or where
its residual, above the resolution, did not fall over two consecutive sweeps after the first (the sweep is not contracting there; the first sweeps are the start's transient, not evidence); such a
unit joins the unit of the group whose column is most correlated with its own, and the sweeps continue from the
joined sites (the cluster's site starts as the block of its groups' sites). Clusters are thus admitted by measured
failure, never by a hand-picked correlation.

Cost. A single group's update is O(n'^2) plus the cavity refresh (``sequential_ep``); a cluster's is one tilted law
of its members and a damped rebuild, O(G n'^2).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numba
import numpy as np
from scipy.optimize import least_squares

from sv_pgs._typing import F64Array, I64Array
from sv_pgs.alias_laws import GroupLaws, group_terms
from sv_pgs.scale_sampler import NodePrior, cluster_tilted_moments
from sv_pgs.sequential_ep import SequentialSweep

_EPSILON = float(np.finfo(np.float64).eps)


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _tilted_sums(start, log_weight, log_variance, precision, shift, mean, variance, proper):
    """Every group's sum's tilted mean and variance at its cavity (``alias_groups.tilted_sum``), in parallel."""
    for g in numba.prange(start.shape[0] - 1):
        a, b = start[g], start[g + 1]
        P, h = precision[g], shift[g]
        peak = -np.inf
        values = np.empty(b - a)
        conditional = np.empty(b - a)
        ok = True
        for c in range(a, b):
            v = np.exp(log_variance[c])
            if v * P <= -1.0:
                ok = False
                break
            if v == np.inf:
                conditional[c - a] = 1.0 / P
                spread = log_variance[c] + np.log(P)
            else:
                conditional[c - a] = v / (1.0 + v * P)
                spread = np.log1p(v * P)
            values[c - a] = log_weight[c] - 0.5 * spread + 0.5 * h * h * conditional[c - a]
            if values[c - a] > peak:
                peak = values[c - a]
        proper[g] = ok
        if not ok:
            continue
        total = 0.0
        for c in range(b - a):
            values[c] = np.exp(values[c] - peak)
            total += values[c]
        first = 0.0
        for c in range(b - a):
            values[c] /= total
            first += values[c] * conditional[c]
        spread_sum = 0.0
        for c in range(b - a):
            spread_sum += values[c] * (conditional[c] - first) * (conditional[c] - first)
        mean[g] = h * first
        variance[g] = first + h * h * spread_sum


def _gaussian_kl(mean_from: F64Array, covariance_from: F64Array, mean_to: F64Array, covariance_to: F64Array) -> float:
    """KL(N(mean_from, covariance_from) || N(mean_to, covariance_to))."""
    inverse = np.linalg.inv(covariance_to)
    difference = mean_to - mean_from
    return 0.5 * float(
        np.trace(inverse @ covariance_from) + difference @ inverse @ difference - mean_from.shape[0]
        + np.linalg.slogdet(covariance_to)[1] - np.linalg.slogdet(covariance_from)[1]
    )


def cluster_draws(size: int, draw_count: int) -> int:
    """Effective draws for a cluster of ``size`` sums: its moments' Monte Carlo KL is (m + m (m + 1) / 2) / (2 N) in
    expectation (m means and m (m + 1) / 2 covariance entries from N effective draws), at most the resolution
    1 / (2 K) at N = K (m + m (m + 1) / 2)."""
    return draw_count * (size + size * (size + 1) // 2)


def monte_carlo_floor(mean_error: F64Array, covariance: F64Array, enumerated: bool) -> float:
    """The KL the estimated moments make from the exact ones in expectation: 1/2 e' V^-1 e for the means' standard
    errors e, and m (m + 1) / (4 N) for the covariance at the means' effective draws N (0 when enumerated)."""
    if enumerated:
        return 0.0
    size = covariance.shape[0]
    effective = float(np.median(np.diag(covariance) / np.maximum(mean_error**2, np.finfo(np.float64).tiny)))
    return 0.5 * float(mean_error @ np.linalg.solve(covariance, mean_error)) + 0.5 * (size * (size + 1) / 2) / effective


@dataclass
class AliasFixedPoint:
    """The fixed point: sites (``precision`` t scaled, ``shift`` nu), the clusters (arrays of groups), each member's
    posterior mean and variance (oriented), and the run's record."""

    precision: F64Array
    shift: F64Array
    clusters: list[I64Array]
    mean: F64Array
    variance: F64Array
    converged: bool
    sweeps: int
    record: list[dict] = field(default_factory=list)


class AliasEP:
    """EP on the alias sums with clusters admitted by the contract (module docstring).

    ``rows`` (groups x n') is the design over the groups in the covariates' complement, ``score`` X'y there, ``laws``
    the groups' induced priors, ``nodes`` the prior per member (``scale_sampler.NodePrior``, members in the design's
    order, oriented), ``groups`` each member's alias group."""

    def __init__(self, rows: F64Array, score: F64Array, laws: GroupLaws, nodes: NodePrior, groups: I64Array, noise: float, draw_count: int,
                 seed: int) -> None:
        self.rows = np.ascontiguousarray(rows, dtype=np.float64)
        self.squares = np.einsum("ij,ij->i", self.rows, self.rows)
        self.score = np.asarray(score, dtype=np.float64)
        self.laws = laws
        self.nodes = nodes
        self.groups = np.asarray(groups, dtype=np.int64)
        self.noise = float(noise)
        self.draw_count = int(draw_count)
        self.seed = int(seed)
        self.resolution = 0.5 / draw_count
        self.group_count = int(self.rows.shape[0])
        order = np.argsort(self.groups, kind="stable")
        bounds = np.concatenate([[0], np.cumsum(np.bincount(self.groups, minlength=self.group_count))])
        self.members_of = [order[bounds[g]:bounds[g + 1]] for g in range(self.group_count)]
        self.unit_rows = self.rows / np.sqrt(self.squares)[:, None]
        with np.errstate(over="ignore"):
            self.largest = np.exp(laws.largest_log_variance())
        self.sweep = SequentialSweep(self.rows, self.squares, self.score, laws, self.noise)
        self.states: dict[tuple, I64Array] = {}
        self.calls = 0

    def _indicator(self, cluster: I64Array) -> tuple[I64Array, F64Array]:
        members = np.concatenate([self.members_of[g] for g in cluster])
        indicator = np.zeros((members.shape[0], cluster.shape[0]))
        position = 0
        for column, g in enumerate(cluster):
            size = self.members_of[g].shape[0]
            indicator[position:position + size, column] = 1.0
            position += size
        return members, indicator

    def _sum_prior(self, cluster: I64Array) -> NodePrior:
        """The cluster's sums as the sampler's rows: each sum's law atoms (``alias_laws.GroupLaws``) as its nodes, padded
        with zero masses. The lattice spacing is set to 0, so the sampler's scale exchange moves nothing between rows
        whose atoms lie on different grids (``scale_sampler._swap_offsets``)."""
        laws = self.laws
        sizes = laws.law_start[cluster + 1] - laws.law_start[cluster]
        width = int(sizes.max())
        log_weights = np.full((cluster.shape[0], width), -np.inf)
        variances = np.ones((cluster.shape[0], width))
        for row, g in enumerate(cluster):
            a, b = laws.law_start[g], laws.law_start[g + 1]
            log_weights[row, : b - a] = laws.law_log_mass[a:b]
            with np.errstate(over="ignore"):
                variances[row, : b - a] = np.exp(laws.law_log_variance[a:b])
        return NodePrior(log_weights=log_weights, log_scale=np.zeros(cluster.shape[0]), grid=np.zeros(width), variances=variances)

    def cluster_tilted(self, cluster: I64Array, precision: F64Array, shift: F64Array, stamp: int):
        """(proper, mean, covariance, Monte Carlo floor) of a cluster's sums' tilted law: the product of the sums'
        induced laws and the Gaussian cavity on the sums, enumerated or sampled over the sums' atoms."""
        key = ("sums", *cluster.tolist())
        state = self.states.get(key)
        moments = cluster_tilted_moments(
            self._sum_prior(cluster), [np.arange(cluster.shape[0])], [np.asarray(precision)], [np.asarray(shift)],
            cluster_draws(cluster.shape[0], self.draw_count), self.seed * self.group_count + stamp, None if state is None else [state],
        )
        self.calls += 1
        if not moments.proper[0]:
            return False, None, None, 0.0
        self.states[key] = moments.state.copy()
        mean, covariance = moments.cluster(0)
        return True, mean.copy(), covariance.copy(), monte_carlo_floor(moments.mean_error, covariance, bool(moments.exact[0]))

    def member_tilted(self, cluster: I64Array, precision: F64Array, shift: F64Array, stamp: int):
        """(proper, mean, covariance, Monte Carlo floor, members, member means) of a cluster's members' exact tilted
        law, the decode's (``_decode``)."""
        members, indicator = self._indicator(cluster)
        key = tuple(cluster.tolist())
        state = self.states.get(key)
        moments = cluster_tilted_moments(
            self.nodes, [members], [indicator @ precision @ indicator.T], [indicator @ shift], cluster_draws(cluster.shape[0], self.draw_count),
            self.seed * self.group_count + stamp, None if state is None else [state],
        )
        self.calls += 1
        if not moments.proper[0]:
            return False, None, None, 0.0, members, None
        self.states[key] = moments.state.copy()
        member_mean, member_covariance = moments.cluster(0)
        covariance = indicator.T @ member_covariance @ indicator
        floor = monte_carlo_floor(np.sqrt(indicator.T @ moments.mean_error**2), covariance, bool(moments.exact[0]))
        return True, indicator.T @ member_mean, covariance, floor, members, (member_mean, np.diag(member_covariance))

    def _update_cluster(self, index: int, precision: F64Array, shift: F64Array, stamp: int) -> tuple[float | None, float]:
        """One cluster's damped site update; (residual or None where refused, Monte Carlo floor)."""
        sweep = self.sweep
        cluster = sweep.clusters[index]
        cavity = sweep.cluster_cavity(index, precision, shift)
        if cavity is None:
            return None, 0.0
        cavity_precision, cavity_shift, mean, covariance = cavity
        with np.errstate(divide="ignore"):
            if not np.linalg.eigvalsh(cavity_precision + np.diag(1.0 / self.largest[cluster]))[0] > 0.0:
                return None, 0.0
        proper, tilted_mean, tilted_covariance, floor = self.cluster_tilted(cluster, cavity_precision, cavity_shift, stamp)
        # A covariance that is not positive definite to its rounding (nearly collinear sums, sampled) has no Gaussian
        # to match: the update is refused.
        if not proper or not np.linalg.eigvalsh(tilted_covariance)[0] > 0.0:
            return None, 0.0
        residual = _gaussian_kl(tilted_mean, tilted_covariance, mean, covariance)
        inverse = np.linalg.inv(tilted_covariance)
        target_precision, target_shift = inverse - cavity_precision, inverse @ tilted_mean - cavity_shift
        old_precision = (np.diag(precision[cluster]) + sweep.coupling[index]) / self.noise
        old_shift = shift[cluster].copy()
        old_t, old_nu, old_coupling = precision.copy(), shift.copy(), sweep.coupling[index].copy()
        fraction = 1.0
        while fraction > _EPSILON:
            sweep.set_cluster_site(index, old_precision + fraction * (target_precision - old_precision),
                                   old_shift + fraction * (target_shift - old_shift), precision, shift)
            if sweep.valid(precision, shift):
                return residual, floor
            precision[:], shift[:] = old_t, old_nu
            sweep.coupling[index] = old_coupling.copy()
            fraction *= 0.5
        return None, floor

    def _single_residuals(self, precision: F64Array, shift: F64Array) -> tuple[F64Array, np.ndarray]:
        """Every group's residual KL (tilted law from q's marginal), and whether its tilted law is proper."""
        cavity_precision, cavity_shift = self.sweep.cavities(precision, shift)
        laws = self.laws
        mean = np.empty(self.group_count)
        variance = np.empty(self.group_count)
        proper = np.empty(self.group_count, dtype=np.bool_)
        _tilted_sums(laws.law_start, laws.law_log_mass, laws.law_log_variance, cavity_precision, cavity_shift, mean, variance, proper)
        site = precision / self.noise
        q_variance = 1.0 / (cavity_precision + site)
        q_mean = q_variance * (cavity_shift + shift)
        with np.errstate(divide="ignore", invalid="ignore"):
            residual = 0.5 * (variance / q_variance + (q_mean - mean) ** 2 / q_variance - 1.0 + np.log(q_variance / variance))
        return np.where(proper, residual, np.inf), proper

    def _partner(self, unit: I64Array, unit_of: I64Array) -> int:
        """The unit holding the group whose column is most correlated with any of ``unit``'s."""
        correlation = np.abs(self.unit_rows[unit] @ self.unit_rows.T)
        correlation[:, unit] = -np.inf
        return int(unit_of[int(np.argmax(np.max(correlation, axis=0)))])

    def _marginals(self, precision: F64Array, shift: F64Array) -> tuple[F64Array, F64Array]:
        """q's marginal (mean, variance) of every group's sum."""
        cavity_precision, cavity_shift = self.sweep.cavities(precision, shift)
        variance = 1.0 / (cavity_precision + precision / self.noise)
        return variance * (cavity_shift + shift), variance

    def fit(self, precision: F64Array, shift: F64Array, clusters: list[I64Array] | None = None, progress=None) -> AliasFixedPoint:
        """EP from the sites (``precision`` t, ``shift`` nu, copied) to its fixed point with clusters admitted by the
        contract (module docstring); ``progress`` (optional) is called with each sweep's record."""
        precision = np.array(precision, dtype=np.float64, copy=True)
        shift = np.array(shift, dtype=np.float64, copy=True)
        sweep = self.sweep
        sweep.set_clusters(list(clusters or []))
        # Each unit's last residual and its count of consecutive sweeps without a fall (``fit``'s contract).
        history: dict[int, float] = {}
        streak: dict[int, int] = {}
        sweeps_since_change = 0
        record: list[dict] = []
        order = np.arange(self.group_count, dtype=np.int64)
        stamp = 0
        converged = False
        while True:
            stamp += 1
            before_mean, before_variance = self._marginals(precision, shift)
            refused_count = sweep.run(precision, shift, order)
            if refused_count is None:
                raise FloatingPointError("the sites leave q's precision not positive definite")
            cluster_residual: dict[int, float] = {}
            refused_clusters: set[int] = set()
            floors: dict[int, float] = {}
            for index in range(len(sweep.clusters)):
                residual, floor = self._update_cluster(index, precision, shift, stamp)
                floors[index] = floor
                if residual is None:
                    refused_clusters.add(index)
                else:
                    cluster_residual[index] = residual
            single, _proper = self._single_residuals(precision, shift)
            unit_of = np.arange(self.group_count)
            for index, groups in enumerate(sweep.clusters):
                unit_of[groups] = self.group_count + index
            current: dict[int, float] = {}
            allowed: dict[int, float] = {}
            for g in np.flatnonzero(~sweep.clustered):
                current[int(g)], allowed[int(g)] = float(single[g]), self.resolution
            for index, value in cluster_residual.items():
                current[self.group_count + index], allowed[self.group_count + index] = value, self.resolution + floors[index]
            failing = {self.group_count + index for index in refused_clusters}
            # Non-contraction: above the resolution and not falling over two consecutive sweeps, counted from the second
            # sweep since the units last changed (the first is the transient's, and is not recorded).
            if sweeps_since_change:
                for key, value in current.items():
                    if key in history:
                        streak[key] = streak.get(key, 0) + 1 if value >= history[key] else 0
                    history[key] = value
            sweeps_since_change += 1
            failing |= {key for key, count in streak.items() if key in current and count >= 2 and current[key] > allowed[key]}
            # A single group whose update the sweep refused shows as an infinite residual (improper tilted law).
            failing |= {key for key, value in current.items() if not np.isfinite(value)}
            # The fixed point's test is q's, not each unit's: the residuals summed over the units (each one's own
            # Monte Carlo floor allowed) within the resolution.
            total = sum(current.values()) - sum(allowed[key] - self.resolution for key in current)
            # And q has stopped moving: the sweep's move, KL between the groups' marginals before and after it,
            # summed, within the resolution (a small residual alone can be a sweep still on its way).
            after_mean, after_variance = self._marginals(precision, shift)
            with np.errstate(divide="ignore", invalid="ignore"):
                move = float(np.sum(0.5 * (before_variance / after_variance + (after_mean - before_mean) ** 2 / after_variance - 1.0
                                           + np.log(after_variance / before_variance))))
            if not np.isfinite(move):
                move = np.inf
            record.append({"sweep": stamp, "refused": int(refused_count), "failing": len(failing), "residual": total, "move": move,
                           "clusters": sorted((int(c.shape[0]) for c in sweep.clusters), reverse=True)})
            if progress is not None:
                progress(record[-1])
            if not failing and total <= self.resolution and move <= self.resolution and not refused_clusters:
                converged = True
                break
            if not failing:
                continue
            clusters_now = [c.copy() for c in sweep.clusters]
            coupling_now = [c.copy() for c in sweep.coupling]
            members_of_unit = {g: np.array([g]) for g in range(self.group_count) if not sweep.clustered[g]}
            for index, groups in enumerate(clusters_now):
                members_of_unit[self.group_count + index] = groups
            joined: set[int] = set()
            merges: list[tuple[int, int]] = []
            for key in sorted(failing):
                if key in joined:
                    continue
                other = self._partner(members_of_unit[key], unit_of)
                if other in joined or other == key:
                    continue
                merges.append((key, other))
                joined.update((key, other))
            keep = [index for index in range(len(clusters_now)) if self.group_count + index not in joined]
            new_clusters = [clusters_now[index] for index in keep]
            new_coupling = [coupling_now[index] for index in keep]
            for first, second in merges:
                union = np.concatenate([members_of_unit[first], members_of_unit[second]])
                block = np.zeros((union.shape[0], union.shape[0]))
                position = 0
                for key in (first, second):
                    size = members_of_unit[key].shape[0]
                    if key >= self.group_count:
                        block[position:position + size, position:position + size] = coupling_now[key - self.group_count]
                    position += size
                new_clusters.append(union)
                new_coupling.append(block)
            sweep.set_clusters(new_clusters, new_coupling)
            history, streak, sweeps_since_change = {}, {}, 0
        return self._decode(precision, shift, converged, stamp, record)

    def _matching_cavity(self, group: int, mean: float, variance: float) -> tuple[float, float]:
        """The one-dimensional cavity (P, h) under which the group's sum's tilted law has this mean and variance,
        solved in (P, h) with P above -1 / V_max (where the tilted law is proper) by trust-region least squares on the
        standardized mean error and the log variance ratio, from the Gaussian answer with the prior's second moment
        as its variance; (nan, nan) where no cavity reproduces them to half precision."""
        laws = self.laws
        a, b = laws.law_start[group], laws.law_start[group + 1]
        start = np.array([0, b - a])
        log_mass, log_variance = laws.law_log_mass[a:b], laws.law_log_variance[a:b]
        floor = -1.0 / float(self.largest[group]) if np.isfinite(self.largest[group]) else 0.0

        def errors(point):
            mean_out, variance_out, proper = np.empty(1), np.empty(1), np.empty(1, dtype=np.bool_)
            _tilted_sums(start, log_mass, log_variance, np.array([point[0]]), np.array([point[1]]), mean_out, variance_out, proper)
            if not proper[0] or not variance_out[0] > 0.0:
                return np.array([np.inf, np.inf])
            return np.array([(mean_out[0] - mean) / np.sqrt(variance), np.log(variance_out[0] / variance)])

        prior_second = float(np.exp(log_mass) @ np.exp(log_variance))
        lowest = floor * (1.0 - _EPSILON**0.5)
        guess = np.array([max(1.0 / variance - 1.0 / prior_second, lowest + _EPSILON**0.5 / variance), mean / variance])
        solution = least_squares(errors, guess, bounds=([lowest, -np.inf], [np.inf, np.inf]), xtol=_EPSILON, ftol=_EPSILON, gtol=_EPSILON)
        if not np.all(np.abs(errors(solution.x)) <= _EPSILON**0.5):
            return np.nan, np.nan
        return float(solution.x[0]), float(solution.x[1])

    def _decode(self, precision: F64Array, shift: F64Array, converged: bool, sweeps: int, record: list[dict]) -> AliasFixedPoint:
        """Members' means and variances: single groups from their laws at their cavities (``alias_laws.group_terms``),
        clusters from their members' exact tilted law at the final cavity."""
        sweep = self.sweep
        cavity_precision, cavity_shift = sweep.cavities(precision, shift)
        clustered = sweep.clustered
        cavity_precision = np.where(clustered, 1.0, cavity_precision)
        cavity_shift = np.where(clustered, 0.0, cavity_shift)
        terms = group_terms(self.laws, cavity_precision, cavity_shift)
        mean, variance = terms.mean.copy(), terms.variance.copy()
        fallback: list[int] = []
        for index, cluster in enumerate(sweep.clusters):
            cavity = sweep.cluster_cavity(index, precision, shift)
            proper, _m, _v, _floor, members, decoded = self.member_tilted(cluster, cavity[0], cavity[1], sweeps + 1)
            if proper:
                mean[members], variance[members] = decoded
            else:
                fallback.extend(int(g) for g in cluster)
        if fallback:
            # The members' law over the cluster's cavity has no proper form where the sums' has (each member's own
            # variance bound is tighter than its sum's): each group is decoded at the one-dimensional cavity whose
            # tilted law of its sum has q's marginal moments, the members given their sum's marginal law.
            groups = np.asarray(fallback, dtype=np.int64)
            q_mean, q_variance = self._marginals(precision, shift)
            matched_precision, matched_shift = cavity_precision.copy(), cavity_shift.copy()
            for g in groups:
                matched_precision[g], matched_shift[g] = self._matching_cavity(int(g), float(q_mean[g]), float(q_variance[g]))
            matched = group_terms(self.laws, matched_precision, matched_shift)
            for g in groups:
                members = self.members_of[g]
                mean[members], variance[members] = matched.mean[members], matched.variance[members]
        return AliasFixedPoint(precision=precision, shift=shift, clusters=[c.copy() for c in sweep.clusters], mean=mean, variance=variance,
                               converged=converged, sweeps=sweeps, record=record)

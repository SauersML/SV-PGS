"""The induced law of every exact alias group's sum, with each member's leave-one-out law, and from them at a cavity the
group's exact log Z, its members' node marginals and scale derivatives (the empirical Bayes gradient) and the decode.

An alias group G (``alias_groups``) enters the likelihood only through T = sum_{j in G} gamma_j, and given the
members' nodes k_j the sum is N(0, V) with V = sum_j v_{j k_j}, v_jk = u_j e^{t_k}. So everything the fit asks of G at a
Gaussian cavity exp(-P T^2 / 2 + h T) is an expectation over the law of V against the kernel
K(V) = (1 + V P)^-1/2 exp(h^2 V / (2 (1 + V P))) (``scale_mixture_ep._kernel_terms``), positive terms only, so nothing
cancels however strong the signal:

    Z = E_V[K(V)],   P(k_j = a | y) = pi_ja E_{V_-j}[K(V_-j + v_ja)] / Z,
    d log Z / d log u_j = E[(v_j / V) L_1(V)],   L_1 = d log K / d log V = V (h^2 / (1 + V P)^2 - P / (1 + V P)) / 2,

with V_-j the sum without member j, whose law is the leave-one-out law. Member j's posterior mean and variance follow
from the same leave-one-out law: given its node a and V_-j = W the sum's tilted law is N(h c, c) with c = V / (1 + V P)
(V = W + v_ja), and given T the member is N((v_ja / V) T, v_ja W / V), so E[gamma_j] = E[(v_ja / V) h c] and
E[gamma_j^2] = E[v_ja W / V + (v_ja / V)^2 (c + h^2 c^2)].

The laws. A group of one member is its member's K atoms; a pair's law is its K^2 atoms and each leave-one-out law the
other member's K atoms, all exact. A larger group's laws are formed on nodes at the lattice's spacing in log V (the
lattice is a quadrature rule in log variance at that spacing, ``scale_mixture_ep.derived_lattice``): prefix laws
member by member (at a spacing refined by a measured factor, ``law_resolution``), each sum's mass split between its two neighbouring nodes linearly in log V (mass and the mean of
log V kept), the suffix laws the same way from the other end, and member j's leave-one-out law the convolution of the
prefix before it with the suffix after it. Masses are held in logs throughout (a tilted law can put its weight on
masses far below the largest). The laws depend on the hyperparameters, not on the cavity: they are formed once per
hyperparameter setting (``GroupLaws.of``) and read at every cavity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numba
import numpy as np

from sv_pgs._typing import F64Array, I64Array


@numba.njit(cache=True, error_model="numpy")
def _log_add(left, right):
    if left == -np.inf:
        return right
    if right == -np.inf:
        return left
    if left > right:
        return left + np.log1p(np.exp(right - left))
    return right + np.log1p(np.exp(left - right))


@numba.njit(cache=True, error_model="numpy")
def _convolve(first_log_variance, first_log_mass, second_log_variance, second_log_mass, low, spacing, node_count):
    """The law of the sum of two independent variances, each given as atoms (log V, log mass), on ``node_count`` nodes at
    ``spacing`` in log V from ``low``: each pair's sum split between its neighbouring nodes linearly in log V."""
    result = np.full(node_count, -np.inf)
    for i in range(first_log_variance.shape[0]):
        if first_log_mass[i] == -np.inf:
            continue
        for k in range(second_log_variance.shape[0]):
            if second_log_mass[k] == -np.inf:
                continue
            log_total = _log_add(first_log_variance[i], second_log_variance[k])
            position = (log_total - low) / spacing
            lower = int(np.floor(position))
            if lower < 0:
                lower = 0
            if lower > node_count - 2:
                lower = node_count - 2
            fraction = position - lower
            if fraction < 0.0:
                fraction = 0.0
            if fraction > 1.0:
                fraction = 1.0
            mass = first_log_mass[i] + second_log_mass[k]
            if fraction < 1.0:
                result[lower] = _log_add(result[lower], mass + np.log1p(-fraction))
            if fraction > 0.0:
                result[lower + 1] = _log_add(result[lower + 1], mass + np.log(fraction))
    return result


@numba.njit(cache=True, error_model="numpy")
def _log_kernel(log_variance, precision, shift):
    """(log K(V), L_1(V), c(V)) at log V (``scale_mixture_ep._kernel_terms``'s overflow limits where V overflows)."""
    variance = np.exp(log_variance)
    if variance == np.inf:
        conditional = 1.0 / precision
        log_kernel = -0.5 * (log_variance + np.log(precision)) + 0.5 * shift * shift / precision
        # L_1 -> -1/2 as V -> infinity.
        return log_kernel, -0.5, conditional
    ratio = variance * precision
    conditional = variance / (1.0 + ratio)
    log_kernel = -0.5 * np.log1p(ratio) + 0.5 * shift * shift * conditional
    first = 0.5 * variance * (shift * shift / ((1.0 + ratio) * (1.0 + ratio)) - precision / (1.0 + ratio))
    return log_kernel, first, conditional


@dataclass(frozen=True)
class GroupLaws:
    """Every alias group's law of V and its members' leave-one-out laws at one hyperparameter setting, packed as atoms
    (log V, log mass): group g's law is ``law_start[g]:law_start[g + 1]``; member j's leave-one-out law is
    ``loo_start[j]:loo_start[j + 1]`` (empty for a group of one); each member's own atoms are its K nodes
    (``member_log_variance`` j x K, ``member_log_weight`` j x K)."""

    groups: I64Array
    member_start: I64Array
    member_index: I64Array
    law_start: I64Array
    law_log_variance: F64Array
    law_log_mass: F64Array
    loo_start: I64Array
    loo_log_variance: F64Array
    loo_log_mass: F64Array
    member_log_variance: F64Array
    member_log_weight: F64Array

    @property
    def group_count(self) -> int:
        return int(self.member_start.shape[0] - 1)

    @classmethod
    def of(cls, groups: I64Array, log_density: F64Array, class_index: I64Array, log_scale: F64Array, grid: F64Array, refinement: int = 1) -> "GroupLaws":
        """The laws at the class log densities ``log_density`` (classes x K, normalized), each member's class and log
        scale log u_j, on the lattice ``grid`` (module docstring); a larger group's nodes are at the lattice's spacing
        divided by ``refinement`` (the splitting's error falls as its square; ``law_resolution`` measures it)."""
        groups = np.asarray(groups, dtype=np.int64)
        group_count = int(groups.max()) + 1
        member_index = np.argsort(groups, kind="stable").astype(np.int64)
        member_start = np.concatenate([[0], np.cumsum(np.bincount(groups, minlength=group_count))]).astype(np.int64)
        grid = np.asarray(grid, dtype=np.float64)
        spacing = (float(grid[1] - grid[0]) if grid.shape[0] > 1 else 1.0) / refinement
        member_log_variance = np.asarray(log_scale, dtype=np.float64)[:, None] + grid[None, :]
        member_log_weight = np.asarray(log_density, dtype=np.float64)[np.asarray(class_index, dtype=np.int64)]
        law_variance, law_mass, law_start = [], [], [0]
        loo_variance = [np.zeros(0) for _ in range(groups.shape[0])]
        loo_mass = [np.zeros(0) for _ in range(groups.shape[0])]
        for group in range(group_count):
            members = member_index[member_start[group]:member_start[group + 1]]
            count = members.shape[0]
            atoms_v = [member_log_variance[j] for j in members]
            atoms_m = [member_log_weight[j] for j in members]
            if count == 1:
                law_v, law_m = atoms_v[0], atoms_m[0]
            elif count == 2:
                law_v = np.logaddexp(atoms_v[0][:, None], atoms_v[1][None, :]).ravel()
                law_m = (atoms_m[0][:, None] + atoms_m[1][None, :]).ravel()
                loo_variance[members[0]], loo_mass[members[0]] = atoms_v[1], atoms_m[1]
                loo_variance[members[1]], loo_mass[members[1]] = atoms_v[0], atoms_m[0]
            else:
                low = float(np.min([values.min() for values in atoms_v]))
                high = float(np.logaddexp.reduce([values.max() for values in atoms_v]))
                node_count = int(np.ceil((high - low) / spacing)) + 2
                nodes = low + spacing * np.arange(node_count)
                prefix = [(atoms_v[0], atoms_m[0])]
                for position in range(1, count):
                    previous_v, previous_m = prefix[-1]
                    prefix.append((nodes, _convolve(previous_v, previous_m, atoms_v[position], atoms_m[position], low, spacing, node_count)))
                suffix = [None] * count
                suffix[count - 1] = (atoms_v[count - 1], atoms_m[count - 1])
                for position in range(count - 2, -1, -1):
                    later_v, later_m = suffix[position + 1]
                    suffix[position] = (nodes, _convolve(later_v, later_m, atoms_v[position], atoms_m[position], low, spacing, node_count))
                law_v, law_m = prefix[count - 1]
                for position, member in enumerate(members):
                    if position == 0:
                        loo_variance[member], loo_mass[member] = suffix[1]
                    elif position == count - 1:
                        loo_variance[member], loo_mass[member] = prefix[count - 2]
                    else:
                        before_v, before_m = prefix[position - 1]
                        after_v, after_m = suffix[position + 1]
                        loo_variance[member] = nodes
                        loo_mass[member] = _convolve(before_v, before_m, after_v, after_m, low, spacing, node_count)
            law_variance.append(np.asarray(law_v, dtype=np.float64))
            law_mass.append(np.asarray(law_m, dtype=np.float64))
            law_start.append(law_start[-1] + law_variance[-1].shape[0])
        loo_start = np.concatenate([[0], np.cumsum([values.shape[0] for values in loo_variance])]).astype(np.int64)
        return cls(
            groups=groups, member_start=member_start, member_index=member_index, law_start=np.asarray(law_start, dtype=np.int64),
            law_log_variance=np.concatenate(law_variance), law_log_mass=np.concatenate(law_mass), loo_start=loo_start,
            loo_log_variance=np.concatenate(loo_variance) if loo_start[-1] else np.zeros(0),
            loo_log_mass=np.concatenate(loo_mass) if loo_start[-1] else np.zeros(0),
            member_log_variance=member_log_variance, member_log_weight=member_log_weight,
        )

    @property
    def component_start(self) -> I64Array:
        """The group laws' atoms as the EP sweep's components (``sequential_ep.SequentialSweep``)."""
        return self.law_start

    @property
    def log_weight(self) -> F64Array:
        return self.law_log_mass

    @property
    def log_variance(self) -> F64Array:
        return self.law_log_variance

    def largest_log_variance(self) -> F64Array:
        """Each group's largest atom: its tilted law is proper exactly where 1 + V_max P > 0."""
        return np.array([self.law_log_variance[a:b][self.law_log_mass[a:b] > -np.inf].max() for a, b in zip(self.law_start[:-1], self.law_start[1:])])


@numba.njit(cache=True, error_model="numpy")
def _group_terms(law_start, law_log_variance, law_log_mass, member_start, member_index, loo_start, loo_log_variance, loo_log_mass,
                 member_log_variance, member_log_weight, precision, shift, log_normalizer, marginal, scale_derivative, member_mean, member_second):
    """Per group at its cavity: log Z, and per member its node marginals, d log Z / d log u_j and its posterior mean and
    second moment, in place (module docstring). A group of one member reads its own atoms for everything."""
    group_count = member_start.shape[0] - 1
    node_count = member_log_variance.shape[1]
    for group in range(group_count):
        P = precision[group]
        h = shift[group]
        start, stop = law_start[group], law_start[group + 1]
        total = -np.inf
        for a in range(start, stop):
            if law_log_mass[a] == -np.inf:
                continue
            log_k, _first, _conditional = _log_kernel(law_log_variance[a], P, h)
            total = _log_add(total, law_log_mass[a] + log_k)
        log_normalizer[group] = total
        for position in range(member_start[group], member_start[group + 1]):
            j = member_index[position]
            size = member_start[group + 1] - member_start[group]
            node_terms = np.full(node_count, -np.inf)
            share_first = np.zeros(node_count)
            mean_part = np.zeros(node_count)
            second_part = np.zeros(node_count)
            for k in range(node_count):
                if member_log_weight[j, k] == -np.inf:
                    continue
                log_v = member_log_variance[j, k]
                if size == 1:
                    log_k, first, conditional = _log_kernel(log_v, P, h)
                    node_terms[k] = member_log_weight[j, k] + log_k
                    share_first[k] = first
                    mean_part[k] = h * conditional
                    second_part[k] = conditional + h * h * conditional * conditional
                    continue
                # Over the leave-one-out law: sum_W m(W) K(W + v) with the share-weighted pieces as weighted means.
                weighted_first = 0.0
                weighted_mean = 0.0
                weighted_second = 0.0
                peak = -np.inf
                terms = np.empty(loo_start[j + 1] - loo_start[j])
                for index in range(loo_start[j], loo_start[j + 1]):
                    if loo_log_mass[index] == -np.inf:
                        terms[index - loo_start[j]] = -np.inf
                        continue
                    log_total = _log_add(loo_log_variance[index], log_v)
                    log_k, _f, _c = _log_kernel(log_total, P, h)
                    value = loo_log_mass[index] + log_k
                    terms[index - loo_start[j]] = value
                    if value > peak:
                        peak = value
                if peak == -np.inf:
                    continue
                weight_sum = 0.0
                for index in range(loo_start[j], loo_start[j + 1]):
                    value = terms[index - loo_start[j]]
                    if value == -np.inf:
                        continue
                    weight = np.exp(value - peak)
                    log_total = _log_add(loo_log_variance[index], log_v)
                    _lk, first, conditional = _log_kernel(log_total, P, h)
                    share = np.exp(log_v - log_total)
                    weight_sum += weight
                    weighted_first += weight * share * first
                    weighted_mean += weight * share * h * conditional
                    # E[gamma_j^2 | node, W] = v W / V + share^2 (c + h^2 c^2), the first term in logs.
                    allocation = np.exp(log_v + loo_log_variance[index] - log_total)
                    weighted_second += weight * (allocation + share * share * (conditional + h * h * conditional * conditional))
                inner = peak + np.log(weight_sum)
                node_terms[k] = member_log_weight[j, k] + inner
                share_first[k] = weighted_first / weight_sum
                mean_part[k] = weighted_mean / weight_sum
                second_part[k] = weighted_second / weight_sum
            peak = -np.inf
            for k in range(node_count):
                if node_terms[k] > peak:
                    peak = node_terms[k]
            norm = 0.0
            for k in range(node_count):
                if node_terms[k] > -np.inf:
                    norm += np.exp(node_terms[k] - peak)
            derivative = 0.0
            mean_value = 0.0
            second_value = 0.0
            for k in range(node_count):
                probability = np.exp(node_terms[k] - peak) / norm if node_terms[k] > -np.inf else 0.0
                marginal[j, k] = probability
                derivative += probability * share_first[k]
                mean_value += probability * mean_part[k]
                second_value += probability * second_part[k]
            scale_derivative[j] = derivative
            member_mean[j] = mean_value
            member_second[j] = second_value


@dataclass(frozen=True)
class GroupTerms:
    """``group_terms``' answer: per group log Z; per member its node marginals (members x K), d log Z / d log u_j and its
    posterior mean and variance."""

    log_normalizer: F64Array
    marginal: F64Array
    scale_derivative: F64Array
    mean: F64Array
    variance: F64Array


def group_terms(laws: GroupLaws, cavity_precision: F64Array, cavity_shift: F64Array) -> GroupTerms:
    """Every group's terms at its cavity (unscaled precision P and shift h on its sum)."""
    member_count, node_count = laws.member_log_variance.shape
    log_normalizer = np.empty(laws.group_count)
    marginal = np.zeros((member_count, node_count))
    scale_derivative = np.empty(member_count)
    mean = np.empty(member_count)
    second = np.empty(member_count)
    _group_terms(
        laws.law_start, laws.law_log_variance, laws.law_log_mass, laws.member_start, laws.member_index, laws.loo_start, laws.loo_log_variance,
        laws.loo_log_mass, laws.member_log_variance, laws.member_log_weight, np.asarray(cavity_precision, dtype=np.float64),
        np.asarray(cavity_shift, dtype=np.float64), log_normalizer, marginal, scale_derivative, mean, second,
    )
    return GroupTerms(log_normalizer=log_normalizer, marginal=marginal, scale_derivative=scale_derivative, mean=mean, variance=second - mean * mean)


def law_resolution(coarse: GroupLaws, fine: GroupLaws, cavity_precision: F64Array, cavity_shift: F64Array) -> float:
    """The larger groups' splitting error measured at their cavities: the largest change of any group's log Z between
    two refinements of its laws (the splitting's error falls as the spacing's square)."""
    return float(np.max(np.abs(group_terms(coarse, cavity_precision, cavity_shift).log_normalizer - group_terms(fine, cavity_precision, cavity_shift).log_normalizer)))

"""Exact observational aliases compiled into one training coordinate each, and their members decoded from it.

Members of an exact tie group (``small_n.dense_statistics``: training columns that are proportional, sign-reversed
included, so equal in the oriented standardized coordinates) enter the likelihood only through their sum
T_g = sum_j gamma_j. The posterior therefore factors: T_g's law is the likelihood's with the exact induced prior of the
sum, and the members given T_g follow the prior's own conditional. EP runs on T_g (one site per group, never per
member), and the members are decoded from it: E[gamma_j | y] = E[m_j(T_g) | y] and Var = E[V_j(T_g) | y] +
Var(m_j(T_g) | y), with m_j and V_j the prior's conditional mean and variance given the sum, so the within-group
allocation variance is kept and no mean is split deterministically. Exchangeable members come out exactly equal at
every effect size.

The induced prior. Each member's effect is a scale mixture, gamma_j ~ sum_k pi_{c(j)k} N(0, v_jk) with v_jk =
u_j e^{t_k}; given every member's node the sum is N(0, V), V = sum_j v_{j k_j}, so T_g is a scale mixture over the
members' node tuples, and given a tuple the members given T are Gaussian with m_j = (v_j / V) T and
V_j = v_j - v_j^2 / V. A group of one member is its member's own mixture. A pair is enumerated exactly, K^2 tuples.
A larger group's law is formed member by member, convolving the running law of V with the next member's nodes and
returning each sum to nodes on the lattice's own spacing in log V, its mass split between the two neighbouring nodes
linearly in log V (mass and the log-variance's mean kept), as the lattice itself is a quadrature rule in log
variance at that spacing (``scale_mixture_ep.derived_lattice``). Each node also carries, for each member, the mass-
weighted mean of s_j = v_j / V and of s_j (1 - s_j) over the sums it received (the allocation variance is V s_j
(1 - s_j)), all in logs where V overflows, which the decode reads.
"""

from __future__ import annotations

from dataclasses import dataclass

import numba
import numpy as np

from sv_pgs._typing import F64Array, I64Array


@dataclass(frozen=True)
class GroupPriors:
    """Every group's induced prior on its sum, packed: group g's components are ``component_start[g]:
    component_start[g + 1]`` of ``log_weight`` (log mass, normalized) and ``log_variance`` (log V); its members are
    ``member_index[member_start[g]:member_start[g + 1]]``, and for member position i of the group and component c the
    decode's share v_j / V and its s (1 - s) (the allocation variance in units of V) are at ``share_start[g] + i * C_g +
    c`` of ``share`` and ``allocation``."""

    component_start: I64Array
    log_weight: F64Array
    log_variance: F64Array
    member_start: I64Array
    member_index: I64Array
    share_start: I64Array
    share: F64Array
    allocation: F64Array
    # Each member's exchangeability class within its group (members with one class row and one scale share it): their
    # posterior moments are equal, and the decode gives them the same numbers.
    exchangeable: I64Array

    @property
    def group_count(self) -> int:
        return int(self.component_start.shape[0] - 1)

    def largest_log_variance(self) -> F64Array:
        """Each group's largest component log variance: its tilted law is proper exactly where 1 + V_max P > 0."""
        return np.maximum.reduceat(self.log_variance, self.component_start[:-1])


def _log_sum_exp(values: F64Array) -> float:
    peak = float(np.max(values))
    return peak + float(np.log(np.sum(np.exp(values - peak))))


def _pair(log_weights: F64Array, log_variances: F64Array) -> tuple[F64Array, F64Array, F64Array, F64Array]:
    """The exact law of a pair's sum over its K^2 node tuples: (log weight, log V, shares v_j / V (2 x K^2), their
    s (1 - s)), all in logs where V could overflow."""
    first, second = log_variances
    log_weight = (log_weights[0][:, None] + log_weights[1][None, :]).ravel()
    log_first = (first[:, None] + np.zeros(second.shape[0])[None, :]).ravel()
    log_second = (np.zeros(first.shape[0])[:, None] + second[None, :]).ravel()
    log_variance = np.logaddexp(log_first, log_second)
    share = np.array([np.exp(log_first - log_variance), np.exp(log_second - log_variance)])
    # s (1 - s) of each member is the product of the two shares.
    product = share[0] * share[1]
    return log_weight, log_variance, share, np.array([product, product])


def _convolved(log_weights: F64Array, log_variances: F64Array, spacing: float) -> tuple[F64Array, F64Array, F64Array, F64Array]:
    """A group's sum law formed member by member on nodes at ``spacing`` in log V (module docstring), in logs: each node
    carries its mass and, per member, the mass-weighted means of the share s_j = v_j / V and of s_j (1 - s_j)."""
    count = log_weights.shape[0]
    low = float(np.min(log_variances))
    high = float(_log_sum_exp(np.max(log_variances, axis=1)))
    nodes = low + spacing * np.arange(int(np.ceil((high - low) / spacing)) + 2)
    node_count = nodes.shape[0]
    log_node = log_variances[0].copy()
    mass = np.exp(log_weights[0] - _log_sum_exp(log_weights[0]))
    shares = np.zeros((count, log_node.shape[0]))
    shares[0] = 1.0
    ratios = np.zeros((count, log_node.shape[0]))
    for member in range(1, count):
        weights = np.exp(log_weights[member] - _log_sum_exp(log_weights[member]))
        log_total = np.logaddexp(log_node[:, None], log_variances[member][None, :]).ravel()
        joint = (mass[:, None] * weights[None, :]).ravel()
        kept_fraction = np.exp((log_node[:, None] - log_total.reshape(log_node.shape[0], -1))).ravel()
        new_fraction = np.exp((log_variances[member][None, :] - log_total.reshape(log_node.shape[0], -1))).ravel()
        carried = [(shares[j][:, None] * np.ones_like(weights)[None, :]).ravel() * kept_fraction for j in range(count)]
        carried[member] = new_fraction
        # Back onto the nodes: mass split linearly in log V between the neighbouring nodes.
        position = (log_total - low) / spacing
        lower = np.clip(np.floor(position).astype(np.int64), 0, node_count - 2)
        upper_fraction = np.clip(position - lower, 0.0, 1.0)
        pieces = ((lower, joint * (1.0 - upper_fraction)), (lower + 1, joint * upper_fraction))
        new_mass = np.zeros(node_count)
        for index, value in pieces:
            np.add.at(new_mass, index, value)
        kept = new_mass > 0.0
        new_shares = np.zeros((count, node_count))
        new_ratios = np.zeros((count, node_count))
        for j in range(count):
            for index, value in pieces:
                np.add.at(new_shares[j], index, value * carried[j])
                np.add.at(new_ratios[j], index, value * carried[j] * (1.0 - carried[j]))
        log_node = nodes[kept]
        mass = new_mass[kept]
        shares = new_shares[:, kept] / mass[None, :]
        ratios = new_ratios[:, kept] / mass[None, :]
    return np.log(mass), log_node, shares, ratios


def group_priors(groups: I64Array, log_density: F64Array, class_index: I64Array, log_scale: F64Array, grid: F64Array) -> GroupPriors:
    """The induced prior on every group's sum (module docstring) from each member's class log density row
    (``log_density[class_index[j]]``), its log scale log u_j and the lattice t."""
    groups = np.asarray(groups, dtype=np.int64)
    group_count = int(groups.max()) + 1
    member_index = np.argsort(groups, kind="stable").astype(np.int64)
    member_start = np.concatenate([[0], np.cumsum(np.bincount(groups, minlength=group_count))]).astype(np.int64)
    spacing = float(grid[1] - grid[0]) if grid.shape[0] > 1 else 1.0
    weights, variances, shares, allocations = [], [], [], []
    component_start, share_start = [0], [0]
    for group in range(group_count):
        members = member_index[member_start[group]:member_start[group + 1]]
        log_weights = np.asarray(log_density, dtype=np.float64)[np.asarray(class_index)[members]]
        log_variances = np.asarray(log_scale, dtype=np.float64)[members][:, None] + np.asarray(grid)[None, :]
        if members.shape[0] == 1:
            log_weight = log_weights[0] - _log_sum_exp(log_weights[0])
            log_variance = log_variances[0]
            share = np.ones((1, log_weight.shape[0]))
            allocation = np.zeros((1, log_weight.shape[0]))
        elif members.shape[0] == 2:
            log_weight, log_variance, share, allocation = _pair(log_weights, log_variances)
            log_weight = log_weight - _log_sum_exp(log_weight)
        else:
            log_weight, log_variance, share, allocation = _convolved(log_weights, log_variances, spacing)
            log_weight = log_weight - _log_sum_exp(log_weight)
        weights.append(log_weight)
        variances.append(log_variance)
        shares.append(share.ravel())
        allocations.append(allocation.ravel())
        component_start.append(component_start[-1] + log_weight.shape[0])
        share_start.append(share_start[-1] + share.size)
    keys = np.column_stack([groups, np.asarray(class_index, dtype=np.float64), np.asarray(log_scale, dtype=np.float64)])
    _unique, exchangeable = np.unique(keys, axis=0, return_inverse=True)
    return GroupPriors(
        component_start=np.asarray(component_start, dtype=np.int64), log_weight=np.concatenate(weights), log_variance=np.concatenate(variances),
        member_start=member_start, member_index=member_index, share_start=np.asarray(share_start, dtype=np.int64),
        share=np.concatenate(shares), allocation=np.concatenate(allocations), exchangeable=np.asarray(exchangeable, dtype=np.int64).ravel(),
    )


@numba.njit(cache=True, error_model="numpy")
def tilted_sum(log_weight, log_variance, precision, shift):
    """(proper, log Z, mean, variance) of a group's sum's tilted law, its induced prior times the cavity
    exp(-P T^2 / 2 + h T): per component log weight - log(1 + V P) / 2 + h^2 c / 2 with c = V / (1 + V P), mean
    h E_w[c] and variance E_w[c] + h^2 Var_w(c) (``scale_mixture_ep._kernel_terms``, with its overflow limits)."""
    count = log_weight.shape[0]
    largest = -np.inf
    weights = np.empty(count)
    conditional = np.empty(count)
    for c in range(count):
        variance = np.exp(log_variance[c])
        ratio = variance * precision
        if ratio <= -1.0:
            return False, 0.0, 0.0, 0.0
        if variance == np.inf:
            conditional[c] = 1.0 / precision
            spread = log_variance[c] + np.log(precision)
        else:
            conditional[c] = variance / (1.0 + ratio)
            spread = np.log1p(ratio)
        weights[c] = log_weight[c] - 0.5 * spread + 0.5 * shift * shift * conditional[c]
        if weights[c] > largest:
            largest = weights[c]
    total = 0.0
    for c in range(count):
        weights[c] = np.exp(weights[c] - largest)
        total += weights[c]
    first = 0.0
    for c in range(count):
        weights[c] /= total
        first += weights[c] * conditional[c]
    spread_sum = 0.0
    for c in range(count):
        offset = conditional[c] - first
        spread_sum += weights[c] * offset * offset
    return True, largest + np.log(total), shift * first, first + shift * shift * spread_sum


def decode(priors: GroupPriors, cavity_precision: F64Array, cavity_shift: F64Array, member_count: int) -> tuple[F64Array, F64Array, np.ndarray]:
    """Every member's posterior mean and variance from its group's tilted law at the final cavities (module docstring):
    per component c the tilted weight w_c, the sum's conditional mean mu_c = h c_c and variance c_c, and each member's
    share a_jc and allocation r_jc = V_c a_jc (1 - a_jc) give E[gamma_j] = sum_c w_c a_jc mu_c and Var = sum_c w_c (r_jc
    + a_jc^2 c_c + a_jc^2 mu_c^2) - E[gamma_j]^2. Returns (mean, variance, proper): a group whose cavity's tilted
    integral is infinite (1 + V_max P <= 0) has no tilted law, its members' moments are NaN, and ``proper`` is False
    for it, which the caller reports as the numerical contract's failure rather than substitute a number."""
    mean = np.empty(member_count)
    variance = np.empty(member_count)
    with np.errstate(over="ignore"):
        proper = (np.asarray(cavity_precision) >= 0.0) | (1.0 + np.exp(priors.largest_log_variance()) * np.asarray(cavity_precision) > 0.0)
    for group in range(priors.group_count):
        start, stop = priors.component_start[group], priors.component_start[group + 1]
        log_weight, log_variance = priors.log_weight[start:stop], priors.log_variance[start:stop]
        precision, shift = float(cavity_precision[group]), float(cavity_shift[group])
        members = priors.member_index[priors.member_start[group]:priors.member_start[group + 1]]
        if not proper[group]:
            mean[members] = np.nan
            variance[members] = np.nan
            continue
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            total = np.exp(log_variance)
            overflowed = ~np.isfinite(total)
            # ``tilted_sum``'s overflow limits: c = 1/P and log(1 + V P) = log V + log P where V overflowed.
            conditional = np.where(overflowed, 1.0 / precision, total / (1.0 + total * precision))
            spread = np.where(overflowed, log_variance + np.log(precision), np.log1p(total * precision))
        component = log_weight - 0.5 * spread + 0.5 * shift * shift * conditional
        weight = np.exp(component - component.max())
        weight /= weight.sum()
        location = shift * conditional
        size = stop - start
        base = priors.share_start[group]
        for position, member in enumerate(members):
            share = priors.share[base + position * size: base + (position + 1) * size]
            ratio = priors.allocation[base + position * size: base + (position + 1) * size]
            first = float(weight @ (share * location))
            # The allocation variance V s (1 - s), formed in logs (V may overflow where its weight does not vanish).
            with np.errstate(divide="ignore", over="ignore"):
                allocation = np.where((weight > 0.0) & (ratio > 0.0), np.exp(np.log(np.where(weight > 0.0, weight, 1.0)) + np.log(np.where(ratio > 0.0, ratio, 1.0)) + log_variance), 0.0)
            second = float(np.sum(allocation) + weight @ (share * share * (conditional + location * location)))
            mean[member] = first
            variance[member] = second - first * first
    # Exchangeable members' moments are equal; the convolution's order is not symmetric in them to the last digit, so
    # each takes its class's average.
    counts = np.bincount(priors.exchangeable)
    mean = (np.bincount(priors.exchangeable, weights=mean) / counts)[priors.exchangeable]
    variance = (np.bincount(priors.exchangeable, weights=variance) / counts)[priors.exchangeable]
    return mean, variance, proper


@numba.njit(cache=True, error_model="numpy")
def cluster_tilted(counts, log_weight, log_variance, precision, shift):
    """(proper, log Z, mean, covariance) of a cluster's joint tilted law by enumeration of its component tuples: the
    cluster's m sums, each with its own induced prior (``counts[i]`` components, packed in order in ``log_weight`` and
    ``log_variance``), times the joint cavity exp(-T' Lambda T / 2 + h' T). Given a tuple the law is Gaussian with
    precision D^-1 + Lambda (D = diag V, D^-1 = 0 where V overflows) and mean (D^-1 + Lambda)^-1 h, and its integral is
    det(I + D^1/2 Lambda D^1/2)^-1/2 exp(h' (D^-1 + Lambda)^-1 h / 2) with log det(I + D^1/2 Lambda D^1/2) = sum log V +
    log det(D^-1 + Lambda). The law is proper exactly where D^-1 + Lambda is positive definite at every tuple, i.e. at
    the tuple of each sum's largest variance."""
    size = counts.shape[0]
    offsets = np.zeros(size + 1, dtype=np.int64)
    for i in range(size):
        offsets[i + 1] = offsets[i] + counts[i]
    index = np.zeros(size, dtype=np.int64)
    mean = np.zeros(size)
    second = np.zeros((size, size))
    largest = -np.inf
    total = 0.0
    matrix = np.empty((size, size))
    within = np.zeros((size, size))
    delta = np.empty(size)
    while True:
        log_prior = 0.0
        log_det_variance = 0.0
        for i in range(size):
            c = offsets[i] + index[i]
            log_prior += log_weight[c]
            log_det_variance += log_variance[c]
        for i in range(size):
            for j in range(size):
                matrix[i, j] = precision[i, j]
            c = offsets[i] + index[i]
            matrix[i, i] += np.exp(-log_variance[c])
        # Cholesky by hand: positive definite or the law is improper.
        lower = np.zeros((size, size))
        for i in range(size):
            for j in range(i + 1):
                value = matrix[i, j]
                for k in range(j):
                    value -= lower[i, k] * lower[j, k]
                if i == j:
                    if not value > 0.0:
                        return False, 0.0, mean, second
                    lower[i, i] = np.sqrt(value)
                else:
                    lower[i, j] = value / lower[j, j]
        forward = np.empty(size)
        for i in range(size):
            value = shift[i]
            for k in range(i):
                value -= lower[i, k] * forward[k]
            forward[i] = value / lower[i, i]
        location = np.empty(size)
        for i in range(size - 1, -1, -1):
            value = forward[i]
            for k in range(i + 1, size):
                value -= lower[k, i] * location[k]
            location[i] = value / lower[i, i]
        log_det = 0.0
        quadratic = 0.0
        for i in range(size):
            log_det += 2.0 * np.log(lower[i, i])
            quadratic += forward[i] * forward[i]
        log_term = log_prior - 0.5 * (log_det_variance + log_det) + 0.5 * quadratic
        # The tuple's covariance (D^-1 + Lambda)^-1 from the factor.
        inverse = np.zeros((size, size))
        for column in range(size):
            unit = np.zeros(size)
            unit[column] = 1.0
            for i in range(size):
                value = unit[i]
                for k in range(i):
                    value -= lower[i, k] * unit[k]
                unit[i] = value / lower[i, i]
            for i in range(size - 1, -1, -1):
                value = unit[i]
                for k in range(i + 1, size):
                    value -= lower[k, i] * unit[k]
                unit[i] = value / lower[i, i]
            for i in range(size):
                inverse[i, column] = unit[i]
        # Weighted running moments, rescaled when the largest log term moves: the tuples' mean location, the scatter
        # of the locations about it (Welford, so no difference of second moments cancels) and their covariances'
        # sum; the covariance is their sum over the total, positive definite by construction.
        if log_term > largest:
            scale = np.exp(largest - log_term) if largest > -np.inf else 0.0
            total *= scale
            for i in range(size):
                for j in range(size):
                    second[i, j] *= scale
                    within[i, j] *= scale
            largest = log_term
        weight = np.exp(log_term - largest)
        if weight > 0.0:
            updated = total + weight
            for i in range(size):
                delta[i] = location[i] - mean[i]
            share = weight * total / updated
            for i in range(size):
                mean[i] += delta[i] * (weight / updated)
                for j in range(size):
                    second[i, j] += share * delta[i] * delta[j]
                    within[i, j] += weight * inverse[i, j]
            total = updated
        # The next tuple (mixed radix).
        position = 0
        while position < size:
            index[position] += 1
            if index[position] < counts[position]:
                break
            index[position] = 0
            position += 1
        if position == size:
            break
    for i in range(size):
        for j in range(size):
            second[i, j] = (second[i, j] + within[i, j]) / total
    return True, largest + np.log(total), mean, second

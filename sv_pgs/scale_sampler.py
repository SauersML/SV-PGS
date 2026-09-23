"""Exact posterior inference for SV-PGS's scale-mixture prior by collapsed sampling of the scales.

**The prior and the identity everything rests on.** Every effect is beta_j ~ sum_k pi_{c(j) k} N(0, u_j e^{t_k}) on
the lattice t (``scale_mixture_ep.ScaleMixturePrior``): a discrete scale index z_j in {0..K-1} with prior pi_{c(j)} and
beta_j | z_j ~ N(0, v_j(z_j)), v_j(k) = u_j e^{t_k}. Given every z, beta's joint law given any Gaussian likelihood is
Gaussian, LD included. So the only non-Gaussian part of the posterior is the discrete law of z, and every quantity
below integrates beta out analytically.

**One algebra for both engines.** Let the likelihood of beta be Gaussian with natural parameters (Lambda, h):
exp(-beta' Lambda beta / 2 + h' beta) (for the data, Lambda = X'X / sigma^2 and h = X'y / sigma^2; for an EP cavity,
its precision and shift). With D = diag v(z),

    Q = Lambda (I + D Lambda)^-1,   b = (I + Lambda D)^-1 h,

and every conditional follows from (Q, b):
- beta | z ~ N(D b, D - D Q D) (the covariance without inverting D, whose entries span the lattice's 10^20);
- the collapsed likelihood of z is det(I + D Lambda)^-1/2 exp(h' D b / 2);
- changing one variance, v_j -> v_j + delta, is a rank-one change of D, and Sherman-Morrison gives
  Q' = Q - c q q', b' = b - c q b_j with q = Q e_j and c = delta / (1 + delta Q_jj), and the log likelihood moves by
  -1/2 log(1 + delta Q_jj) + delta b_j^2 / (2 (1 + delta Q_jj));
- removing j's own variance gives its cavity: precision P_j = Q_jj / (1 - v_j Q_jj) and shift h_j = b_j / (1 - v_j Q_jj),
  and the collapsed conditional of z_j is pi_{c(j) k} (1 + v_k P_j)^-1/2 exp(h_j^2 v_k / (2 (1 + v_k P_j))), the
  single-site tilted law of ``scale_mixture_ep._kernel_terms`` at an exact cavity. (1 - v_j Q_jj = 1 / (1 + v_j P_j)
  loses digits only where the data dominate the prior, v_j P_j >> 1, never where the prior dominates, which is most
  columns: the variance-side form 1/Sigma_jj - 1/v_j would lose them there.)
For the sample side of the whole data (n samples, p >> n columns), Q restricted to a panel of columns S is
X_S' K^-1 X_S with the n x n kernel K = sigma^2 I + X D X', which is how the global sampler holds it.

**The local engine (``cluster_tilted_moments``).** EP's precision dial replaces an LD cluster's per-variant sites by
the cluster's exact joint tilted law: its prior times its Gaussian cavity (Lambda, h). This engine returns that law's
mean, covariance and log normalizer, for many clusters at once (numba, one thread per cluster): by exact enumeration
of the cluster's K^m scale configurations where that costs no more than the sampling it replaces, and otherwise by
the collapsed sampler warm-started from the caller's scale configuration, with Rao-Blackwellized moments (each
sweep contributes the exact Gaussian moments given its z, never raw beta draws) and their Monte Carlo errors.

**The global reference (``fit_scale_sampler``).** The whole window's posterior on the sample side, two chains, with
empirical Bayes of the hyperparameters on the marginal likelihood itself: its gradient by Fisher's identity (the
posterior mean of the complete-data score, Rao-Blackwellized) and Newton steps on Louis' observed information
(Gu and Kong 1998, "A stochastic approximation algorithm with Markov chain Monte-carlo method for incomplete data
estimation problems", PNAS 95:7270). It is exact but costs O(n^2 p) per sweep, so it is the reference for small and
gene-scale problems, not a production route.

**Moves.** A sweep updates each z_j from its exact collapsed conditional (Gibbs), then proposes to exchange the scales
of j and its LD partner (the column of highest |correlation|, or its exact-tie twin): (k_j, k_i) -> (k_i + d, k_j - d),
d the lattice offset between the two members' own scales u, an involution, so the Metropolis ratio is the target's.
A signal carried by j moves to a proxy in one move, where single-site Gibbs would have to pass through a state with
both (or neither) carrying it.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numba
import numpy as np
from scipy import linalg

from sv_pgs._typing import F64Array, I64Array
from sv_pgs.scale_mixture_ep import (
    MixtureHyperparameters,
    ScaleMixturePrior,
    _Objective,
    _penalty_groups,
    _penalty_matrix,
    _pseudo_inverse_trace,
    _resolved_spectrum,
    _smoothing_bounds,
    _spectrum,
    _trust_region_step,
    class_log_density,
    log_scale,
    noise_gain,
)

_EPSILON = float(np.finfo(np.float64).eps)
# The splitmix64 generator (Steele, Lea and Flood 2014, "Fast splittable pseudorandom number generators", OOPSLA):
# its golden-ratio increment and its two multipliers and three shifts. A counter-based generator per cluster keeps the
# local engine's draws reproducible whatever the thread count.
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX_FIRST = np.uint64(0xBF58476D1CE4E5B9)
_MIX_SECOND = np.uint64(0x94D049BB133111EB)
_SHIFT_FIRST = np.uint64(30)
_SHIFT_SECOND = np.uint64(27)
_SHIFT_THIRD = np.uint64(31)
# A uniform double from the top 53 bits of a 64-bit word.
_MANTISSA_BITS = np.finfo(np.float64).nmant + 1
_UNIFORM_SHIFT = np.uint64(np.iinfo(np.uint64).bits - _MANTISSA_BITS)
_UNIFORM_UNIT = float(np.ldexp(1.0, -_MANTISSA_BITS))
# The uniforms one member's update uses: its node's draw, the exchange's coin and the exchange's acceptance.
_UNIFORMS_PER_UPDATE = 3


# ------------------------------------------------------------------ the shared kernel


@numba.njit(cache=True)
def _next_uniform(generator: np.ndarray) -> float:
    """One uniform in [0, 1) from the splitmix64 state ``generator[0]``, advanced in place."""
    generator[0] += _GOLDEN
    value = generator[0]
    value = (value ^ (value >> _SHIFT_FIRST)) * _MIX_FIRST
    value = (value ^ (value >> _SHIFT_SECOND)) * _MIX_SECOND
    value = value ^ (value >> _SHIFT_THIRD)
    return float(value >> _UNIFORM_SHIFT) * _UNIFORM_UNIT


@numba.njit(cache=True)
def _apply(gram: np.ndarray, shift: np.ndarray, group: int, change: float) -> float:
    """v_group += change: Q -= c q q', b -= c q b_g with q = Q e_g, c = change / (1 + change Q_gg), in place. Returns
    the collapsed log likelihood's change, -1/2 log(1 + change Q_gg) + change b_g^2 / (2 (1 + change Q_gg))."""
    diagonal = gram[group, group]
    field_value = shift[group]
    one_plus = 1.0 + change * diagonal
    coefficient = change / one_plus
    column = gram[:, group].copy()
    size = gram.shape[0]
    for row in range(size):
        scaled = coefficient * column[row]
        shift[row] -= scaled * field_value
        for other in range(size):
            gram[row, other] -= scaled * column[other]
    return -0.5 * math.log(one_plus) + 0.5 * change * field_value * field_value / one_plus


@numba.njit(cache=True)
def _scan(
    gram, shift, delta, group_of, order, partner, offset, state, log_weights, variances, uniforms,
    occupancy, scaled_square, fourth, conditional_mean, swaps,
):
    """One scan over the members ``order`` (local member indices): each z_j from its exact collapsed conditional, then
    the scale exchange with ``partner`` (a local member index, or -1) at lattice ``offset``. In place on (``gram``,
    ``shift``) = (Q, b) over the local groups, ``state`` (each local member's node), ``delta`` (each group's total
    variance change) and the Rao-Blackwellized statistics of every updated member, each an expectation over its
    conditional given the rest: ``occupancy`` += P(z_j = k), ``scaled_square`` += E[1{z_j = k} beta_j^2 / v_j(k)],
    ``fourth`` += E[beta_j^4 / v_j(z_j)^2] and ``conditional_mean`` = E[beta_j]. ``swaps`` counts (proposed, accepted).
    ``uniforms`` holds three per position (the draw, the exchange's coin and its acceptance). Returns False where a conditional is improper (1 + v P <= 0: an indefinite
    cavity the prior's variances do not bound)."""
    node_count = variances.shape[1]
    log_node = np.empty(node_count)
    for position in range(order.shape[0]):
        member = order[position]
        group = group_of[member]
        current = state[member]
        own = variances[member, current]
        denominator = 1.0 - own * gram[group, group]
        precision = gram[group, group] / denominator
        field_value = shift[group] / denominator
        largest = -np.inf
        for node in range(node_count):
            variance = variances[member, node]
            if 1.0 + variance * precision <= 0.0:
                return False
            conditional = 1.0 / (1.0 / variance + precision)
            log_node[node] = log_weights[member, node] - 0.5 * math.log1p(variance * precision) + 0.5 * field_value * field_value * conditional
            if log_node[node] > largest:
                largest = log_node[node]
        total = 0.0
        for node in range(node_count):
            log_node[node] = math.exp(log_node[node] - largest)
            total += log_node[node]
        target = uniforms[_UNIFORMS_PER_UPDATE * position] * total
        # The inverse CDF; the last node where rounding leaves the running sum a hair below the total.
        chosen = -1
        running = 0.0
        mean = 0.0
        for node in range(node_count):
            weight = log_node[node] / total
            running += log_node[node]
            if chosen < 0 and running > target:
                chosen = node
            variance = variances[member, node]
            conditional = 1.0 / (1.0 / variance + precision)
            component = field_value * conditional
            square = component * component
            occupancy[member, node] += weight
            scaled_square[member, node] += weight * (square + conditional) / variance
            fourth[member] += weight * (square * square + 6.0 * square * conditional + 3.0 * conditional * conditional) / (variance * variance)
            mean += weight * component
        conditional_mean[member] = mean
        if chosen < 0:
            chosen = node_count - 1
        if chosen != current:
            change = variances[member, chosen] - own
            _apply(gram, shift, group, change)
            delta[group] += change
            state[member] = chosen
        other = partner[position]
        # The exchange is attempted on a fair coin, whatever the state: a deterministic attempt at every member made the
        # scan periodic where the exchange is always accepted (two identical columns: exchanged at the first member,
        # exchanged back at the second, so every sweep began in one phase and both chains' estimates sat at 1.485 and
        # 0 against the exact 0.743 with a batch-means error of 0). A state-independent coin keeps each step
        # posterior-invariant and the chain aperiodic.
        if other < 0 or uniforms[_UNIFORMS_PER_UPDATE * position + 1] >= 0.5:
            continue
        first, second = state[member], state[other]
        first_new, second_new = second + offset[position], first - offset[position]
        if first_new == first or first_new < 0 or first_new >= node_count or second_new < 0 or second_new >= node_count:
            continue
        swaps[0] += 1
        first_change = variances[member, first_new] - variances[member, first]
        second_change = variances[other, second_new] - variances[other, second]
        log_ratio = log_weights[member, first_new] + log_weights[other, second_new] - log_weights[member, first] - log_weights[other, second]
        first_group, second_group = group_of[member], group_of[other]
        if first_group == second_group:
            change = first_change + second_change
            one_plus = 1.0 + change * gram[first_group, first_group]
            if one_plus <= 0.0:
                continue
            log_ratio += -0.5 * math.log(one_plus) + 0.5 * change * shift[first_group] ** 2 / one_plus
        else:
            a11, a22, a12 = gram[first_group, first_group], gram[second_group, second_group], gram[first_group, second_group]
            b1, b2 = shift[first_group], shift[second_group]
            determinant = (1.0 + first_change * a11) * (1.0 + second_change * a22) - first_change * second_change * a12 * a12
            if determinant <= 0.0:
                continue
            quadratic = (
                first_change * (1.0 + a22 * second_change) * b1 * b1 - 2.0 * first_change * second_change * a12 * b1 * b2
                + second_change * (1.0 + a11 * first_change) * b2 * b2
            ) / determinant
            log_ratio += -0.5 * math.log(determinant) + 0.5 * quadratic
        if math.log(uniforms[_UNIFORMS_PER_UPDATE * position + 2]) < log_ratio:
            swaps[1] += 1
            if first_group == second_group:
                _apply(gram, shift, first_group, first_change + second_change)
                delta[first_group] += first_change + second_change
            else:
                _apply(gram, shift, first_group, first_change)
                _apply(gram, shift, second_group, second_change)
                delta[first_group] += first_change
                delta[second_group] += second_change
            state[member] = first_new
            state[other] = second_new
    return True


# ------------------------------------------------------------------ the local engine: one cluster


@numba.njit(cache=True)
def _cluster_start(precision, shift, variances, state):
    """(Q, b, log likelihood) at the configuration ``state``, from D = 0 (Q = Lambda, b = h) by one rank-one step per
    member: at most m Sherman-Morrison steps from an exact start, so no drift accumulates. ``ok`` is False where a
    step's 1 + delta Q_jj <= 0 (an improper configuration)."""
    size = precision.shape[0]
    gram = precision.copy()
    field_values = shift.copy()
    log_likelihood = 0.0
    for member in range(size):
        change = variances[member, state[member]]
        if 1.0 + change * gram[member, member] <= 0.0:
            return gram, field_values, log_likelihood, False
        log_likelihood += _apply(gram, field_values, member, change)
    return gram, field_values, log_likelihood, True


@numba.njit(cache=True)
def _add_moments(gram, field_values, variances, state, weight, mean, second):
    """mean += weight D b and second += weight (D - D Q D + D b b' D): the Gaussian moments of beta given z."""
    size = gram.shape[0]
    scale = np.empty(size)
    for member in range(size):
        scale[member] = variances[member, state[member]]
    for row in range(size):
        component = scale[row] * field_values[row]
        mean[row] += weight * component
        for column in range(size):
            value = scale[row] * scale[column] * (field_values[row] * field_values[column] - gram[row, column])
            if row == column:
                value += scale[row]
            second[row, column] += weight * value


@numba.njit(cache=True)
def _enumerate(precision, shift, log_weights, variances):
    """The exact tilted law of one cluster by enumerating its K^m configurations depth-first: level j holds (Q, b) with
    members 0..j-1 at their chosen variances (each level one rank-one step from its parent's copy, so no drift), and
    every leaf adds its Gaussian moments with weight pi(z) L(z), accumulated relative to the running largest log
    weight. Returns (log Z, mean, second moment, ok)."""
    size, node_count = variances.shape
    grams = np.empty((size + 1, size, size))
    fields = np.empty((size + 1, size))
    logs = np.zeros(size + 1)
    grams[0] = precision
    fields[0] = shift
    choice = np.zeros(size, dtype=np.int64)
    mean = np.zeros(size)
    second = np.zeros((size, size))
    peak = -np.inf
    total = 0.0
    level = 0
    choice[0] = -1
    while level >= 0:
        choice[level] += 1
        if choice[level] >= node_count:
            level -= 1
            continue
        change = variances[level, choice[level]]
        grams[level + 1] = grams[level]
        fields[level + 1] = fields[level]
        if 1.0 + change * grams[level + 1, level, level] <= 0.0:
            return np.nan, mean, second, False
        logs[level + 1] = logs[level] + log_weights[level, choice[level]] + _apply(grams[level + 1], fields[level + 1], level, change)
        if level + 1 < size:
            level += 1
            choice[level] = -1
            continue
        value = logs[size]
        if value > peak:
            rescale = math.exp(peak - value) if np.isfinite(peak) else 0.0
            total *= rescale
            mean *= rescale
            second *= rescale
            peak = value
        weight = math.exp(value - peak)
        total += weight
        _add_moments(grams[size], fields[size], variances, choice, weight, mean, second)
    return peak + math.log(total), mean / total, second / total, True


@numba.njit(cache=True)
def _partners(precision):
    """Each member's partner in a cluster: the other member of the largest squared correlation Lambda_ij^2 /
    (Lambda_ii Lambda_jj), -1 for a cluster of one."""
    size = precision.shape[0]
    partner = np.full(size, -1, dtype=np.int64)
    for row in range(size):
        best = -1.0
        for column in range(size):
            if column == row:
                continue
            scale = precision[row, row] * precision[column, column]
            value = precision[row, column] ** 2 / scale if scale > 0.0 else 0.0
            if value > best:
                best = value
                partner[row] = column
    return partner


@numba.njit(cache=True)
def _batch_variances(series, kept_start, kept_stop):
    """Batch-means estimates (Flegal and Jones 2010, batch size floor(sqrt(N))) of the Monte Carlo variance of the mean
    of each column of ``series`` over rows [kept_start, kept_stop), and the columns' sample variances."""
    count = kept_stop - kept_start
    columns = series.shape[1]
    size = max(1, int(math.floor(math.sqrt(count))))
    batches = count // size
    variance = np.zeros(columns)
    spread = np.zeros(columns)
    for column in range(columns):
        mean = 0.0
        for row in range(kept_start, kept_stop):
            mean += series[row, column]
        mean /= count
        for row in range(kept_start, kept_stop):
            spread[column] += (series[row, column] - mean) ** 2
        spread[column] /= max(count - 1, 1)
        if batches < 2:
            variance[column] = np.inf
            continue
        total = 0.0
        for batch in range(batches):
            value = 0.0
            for row in range(kept_start + batch * size, kept_start + (batch + 1) * size):
                value += series[row, column]
            value = value / size - mean
            total += value * value
        variance[column] = size * total / (batches - 1) / count
    return variance, spread


@numba.njit(cache=True)
def _swap_offsets(partner, log_scale, spacing):
    """Each member's lattice offset to its partner, rint((log u_partner - log u_member) / h): the node shift that gives
    the member its partner's variance (0 without a partner, or on a one-node lattice)."""
    offset = np.zeros(partner.shape[0], dtype=np.int64)
    if spacing <= 0.0:
        return offset
    for member in range(partner.shape[0]):
        if partner[member] >= 0:
            offset[member] = int(np.rint((log_scale[partner[member]] - log_scale[member]) / spacing))
    return offset


@numba.njit(cache=True)
def _cluster_chain(precision, shift, log_weights, variances, log_scale, spacing, state, generator, draw_count):
    """One cluster's collapsed chain, warm-started at ``state`` (left at the last state, the next call's warm start),
    in doubling rounds that keep the chain's last half: draw_count burn-in sweeps, then draw_count kept ones, then as
    many new sweeps as the chain has run (the doubled chain's last half), each round's kept sweeps being exactly the
    ones it ran. A round resolves the cluster when every mean's Monte Carlo variance (batch means) is at most its
    tilted variance over ``draw_count``: an effective sample size of at least the scorer's draw count, the resolution
    ``fit_model.DRAW_COUNT`` gives every posterior quantity. The chain stops unresolved when a doubling raises no mean's
    effective sample size: the work no longer buys precision, the signature of a chain that is not mixing at this
    length. Every kept sweep contributes the exact Gaussian moments of beta given its z (D b, D - D Q D).

    Returns (mean, second moment, the mean's MC variance, the members' node occupancy over the kept sweeps, sweeps,
    resolved, proper)."""
    size, node_count = variances.shape
    partner = _partners(precision) if node_count > 1 else np.full(size, -1, dtype=np.int64)
    offset = _swap_offsets(partner, log_scale, spacing)
    order = np.arange(size)
    uniforms = np.empty(_UNIFORMS_PER_UPDATE * size)
    swaps = np.zeros(2, dtype=np.int64)
    delta = np.zeros(size)
    scaled = np.zeros((size, node_count))
    fourth = np.zeros(size)
    conditional = np.zeros(size)
    sweeps = 0
    length = draw_count
    burn_in = True
    best = 0.0
    while True:
        series = np.empty((length, size))
        mean = np.zeros(size)
        second = np.zeros((size, size))
        occupancy = np.zeros((size, node_count))
        for row in range(length):
            gram, field_values, _log_likelihood, ok = _cluster_start(precision, shift, variances, state)
            if not ok:
                return mean, second, mean, occupancy, sweeps, False, False
            for index in range(_UNIFORMS_PER_UPDATE * size):
                uniforms[index] = _next_uniform(generator)
            if not _scan(gram, field_values, delta, order, order, partner, offset, state, log_weights, variances, uniforms,
                         occupancy, scaled, fourth, conditional, swaps):
                return mean, second, mean, occupancy, sweeps, False, False
            row_mean = np.zeros(size)
            _add_moments(gram, field_values, variances, state, 1.0, row_mean, second)
            series[row] = row_mean
            mean += row_mean
            sweeps += 1
        if burn_in:
            burn_in = False
            continue
        mean /= length
        second /= length
        occupancy /= length
        mc_variance, _spread = _batch_variances(series, 0, length)
        resolved = True
        ess = np.inf
        for member in range(size):
            tilted = second[member, member] - mean[member] * mean[member]
            if mc_variance[member] > 0.0:
                ess = min(ess, tilted / mc_variance[member])
            if mc_variance[member] * draw_count > tilted:
                resolved = False
        if resolved or ess <= best:
            return mean, second, mc_variance, occupancy, sweeps, resolved, True
        best = ess
        length = sweeps


@numba.njit(cache=True)
def _importance_log_normalizer(precision, shift, log_weights, variances, proposal, generator, draws):
    """log Z of one cluster by importance sampling from the product proposal q(z) = prod_j proposal[j, z_j], and the
    delta-method standard error of log Z-hat, sd(w) / (mean(w) sqrt(N)). The proposal is each member's posterior
    marginal from the chain mixed half and half with its prior (a defensive mixture, Hesterberg 1995: every weight is
    at most twice the prior's likelihood ratio, so no configuration the prior reaches is starved)."""
    size, node_count = variances.shape
    log_terms = np.empty(draws)
    state = np.zeros(size, dtype=np.int64)
    cumulative = np.empty((size, node_count))
    for member in range(size):
        running = 0.0
        for node in range(node_count):
            running += proposal[member, node]
            cumulative[member, node] = running
    for draw in range(draws):
        log_proposal = 0.0
        log_prior = 0.0
        for member in range(size):
            target = _next_uniform(generator) * cumulative[member, node_count - 1]
            chosen = node_count - 1
            for node in range(node_count):
                if cumulative[member, node] > target:
                    chosen = node
                    break
            state[member] = chosen
            log_proposal += math.log(proposal[member, chosen] / cumulative[member, node_count - 1])
            log_prior += log_weights[member, chosen]
        _gram, _field, log_likelihood, _ok = _cluster_start(precision, shift, variances, state)
        log_terms[draw] = log_prior + log_likelihood - log_proposal
    peak = np.max(log_terms)
    weights = np.exp(log_terms - peak)
    average = np.mean(weights)
    spread = math.sqrt(np.sum((weights - average) ** 2) / max(draws - 1, 1))
    return peak + math.log(average), spread / (average * math.sqrt(draws))


@numba.njit(cache=True)
def _enumeration_is_cheaper(size, node_count, draw_count):
    """Enumeration's K^m configurations against the least the sampler spends, draw_count sweeps of m conditionals of K
    nodes each (both at O(m^2) per configuration or node): enumerate when K^(m-1) <= m draw_count."""
    return (size - 1) * math.log(node_count) <= math.log(size * draw_count)


@numba.njit(cache=True, parallel=True)
def _clusters(
    member_offsets, block_offsets, precision_values, shift_values, log_weights, variances, log_scale, spacing, states, seed, draw_count,
    means, covariances, mean_errors, log_normalizers, log_normalizer_errors, sweeps, exact, resolved, proper,
):
    """Every cluster's tilted law, one thread per cluster (``cluster_tilted_moments``)."""
    count = member_offsets.shape[0] - 1
    node_count = variances.shape[1]
    for cluster in numba.prange(count):
        start, stop = member_offsets[cluster], member_offsets[cluster + 1]
        size = stop - start
        precision = precision_values[block_offsets[cluster] : block_offsets[cluster + 1]].copy().reshape(size, size)
        shift = shift_values[start:stop].copy()
        weights = log_weights[start:stop]
        node_variances = variances[start:stop]
        if _enumeration_is_cheaper(size, node_count, draw_count):
            log_z, mean, second, ok = _enumerate(precision, shift, weights, node_variances)
            proper[cluster] = ok
            exact[cluster] = True
            resolved[cluster] = ok
            sweeps[cluster] = 0
            mean_errors[start:stop] = 0.0
            log_normalizer_errors[cluster] = 0.0
            log_normalizers[cluster] = log_z
        else:
            generator = np.empty(1, dtype=np.uint64)
            generator[0] = seed ^ (np.uint64(cluster + 1) * _GOLDEN)
            state = states[start:stop].copy()
            mean, second, mc_variance, occupancy, used, is_resolved, ok = _cluster_chain(
                precision, shift, weights, node_variances, log_scale[start:stop], spacing, state, generator, draw_count
            )
            proper[cluster] = ok
            exact[cluster] = False
            sweeps[cluster] = used
            resolved[cluster] = is_resolved
            if ok:
                states[start:stop] = state
                for member in range(size):
                    mean_errors[start + member] = math.sqrt(mc_variance[member])
                proposal = 0.5 * occupancy + 0.5 * np.exp(weights)
                log_z, log_z_error = _importance_log_normalizer(precision, shift, weights, node_variances, proposal, generator, used // 2)
                log_normalizers[cluster] = log_z
                log_normalizer_errors[cluster] = log_z_error
        if not proper[cluster]:
            means[start:stop] = np.nan
            covariances[block_offsets[cluster] : block_offsets[cluster + 1]] = np.nan
            log_normalizers[cluster] = np.nan
            continue
        means[start:stop] = mean
        covariance = np.empty((size, size))
        for row in range(size):
            for column in range(size):
                covariance[row, column] = second[row, column] - mean[row] * mean[column]
        covariances[block_offsets[cluster] : block_offsets[cluster + 1]] = covariance.ravel()


# ------------------------------------------------------------------ the local engine: interface


@dataclass(frozen=True)
class NodePrior:
    """The prior in node form, one row per member: ``log_weights`` (members x K) its class's log pi_{c(j) k},
    ``log_scale`` (members,) log u_j, ``grid`` (K,) the lattice t (evenly spaced), and ``variances`` (members x K)
    u_j e^{t_k}."""

    log_weights: F64Array
    log_scale: F64Array
    grid: F64Array
    variances: F64Array

    @classmethod
    def build(cls, log_weights: F64Array, log_scale: F64Array, grid: F64Array) -> NodePrior:
        log_weights = np.ascontiguousarray(log_weights, dtype=np.float64)
        log_scale = np.ascontiguousarray(log_scale, dtype=np.float64)
        grid = np.ascontiguousarray(grid, dtype=np.float64)
        if log_weights.shape != (log_scale.shape[0], grid.shape[0]):
            raise ValueError("log_weights must be (members, nodes) for log_scale (members,) and grid (nodes,)")
        spacing = np.diff(grid)
        if spacing.size and not np.allclose(spacing, spacing[0], rtol=np.sqrt(_EPSILON), atol=0.0):
            raise ValueError("the lattice must be evenly spaced (the scale exchange moves by whole nodes)")
        with np.errstate(over="ignore"):
            variances = np.exp(log_scale[:, None] + grid[None, :])
        return cls(log_weights=log_weights, log_scale=log_scale, grid=grid, variances=variances)

    @classmethod
    def of(cls, prior: ScaleMixturePrior, coefficients: F64Array) -> NodePrior:
        """The prior of ``prior`` at ``coefficients`` (``class_log_density``, ``log_scale``)."""
        log_density = class_log_density(prior, coefficients)
        return cls.build(log_density[prior.class_index], log_scale(prior, coefficients), prior.log_variance_grid)

    @property
    def spacing(self) -> float:
        return float(self.grid[1] - self.grid[0]) if self.grid.shape[0] > 1 else 0.0

    def rows(self, members: I64Array) -> NodePrior:
        return NodePrior(self.log_weights[members], self.log_scale[members], self.grid, self.variances[members])


@dataclass(frozen=True)
class ClusterMoments:
    """``cluster_tilted_moments``' answer, packed in the order of the clusters and their members: cluster c's members
    are rows ``member_offsets[c]:member_offsets[c + 1]`` of ``mean``, ``mean_error`` and ``state``, and its covariance
    is ``covariance[block_offsets[c]:block_offsets[c + 1]]`` reshaped (m, m) row-major.

    ``mean_error`` is each mean's Monte Carlo standard error (0 for an enumerated cluster), ``log_normalizer`` each
    cluster's log Z = log of the integral of prior x cavity (exact by enumeration, by importance sampling otherwise,
    with ``log_normalizer_error`` its standard error), ``state`` the last scale configuration (pass it back as the
    next call's warm start), ``sweeps`` the chain's length (0 when enumerated), ``exact`` whether it was enumerated,
    ``resolved`` whether its means reached the resolution (always for an enumerated cluster), and ``proper`` whether
    the tilted law is proper (False: the cavity is indefinite past what the prior's variances bound; every moment is
    then NaN)."""

    member_offsets: I64Array
    block_offsets: I64Array
    mean: F64Array
    covariance: F64Array
    mean_error: F64Array
    log_normalizer: F64Array
    log_normalizer_error: F64Array
    state: I64Array
    sweeps: I64Array
    exact: np.ndarray
    resolved: np.ndarray
    proper: np.ndarray

    def cluster(self, index: int) -> tuple[F64Array, F64Array]:
        """(mean, covariance) of cluster ``index``."""
        start, stop = self.member_offsets[index], self.member_offsets[index + 1]
        size = stop - start
        return self.mean[start:stop], self.covariance[self.block_offsets[index] : self.block_offsets[index + 1]].reshape(size, size)


def _cavity_modes(prior: NodePrior, precision: F64Array, shift: F64Array) -> I64Array:
    """Each member's most probable node under its own diagonal cavity (Lambda_jj, h_j): the cold start where the
    caller has no configuration to warm-start from."""
    variances = prior.variances
    diagonal = np.diag(precision)[:, None]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log_node = prior.log_weights - 0.5 * np.log1p(variances * diagonal) + 0.5 * np.square(shift)[:, None] / (1.0 / variances + diagonal)
    return np.argmax(np.where(np.isfinite(log_node), log_node, -np.inf), axis=1).astype(np.int64)


def cluster_tilted_moments(
    prior: NodePrior,
    clusters: Sequence[I64Array],
    precision: Sequence[F64Array],
    shift: Sequence[F64Array],
    draw_count: int,
    seed: int,
    state: Sequence[I64Array] | None = None,
) -> ClusterMoments:
    """The exact tilted law of every cluster: for cluster c with members ``clusters[c]`` (rows of ``prior``, e.g.
    ``NodePrior.of(prior, coefficients)``) and Gaussian cavity exp(-beta' Lambda beta / 2 + h' beta) with Lambda =
    ``precision[c]`` (m x m, symmetric; it may be indefinite where the prior's variances keep the law proper) and
    h = ``shift[c]``, the law prod_j p_j(beta_j) x cavity(beta) over the cluster's effects, p_j member j's scale
    mixture.

    This is EP's cluster site update: the caller divides the cluster's sites out of q to get the cavity, calls this,
    and matches the returned mean and covariance (the new cluster site is the returned law's Gaussian projection
    divided by the cavity). ``state[c]`` (each member's node, e.g. the previous call's ``ClusterMoments.state``)
    warm-starts the chains; None starts each member at the mode of its own diagonal cavity. Clusters run in parallel
    (numba threads, one cluster each); each is enumerated when its K^m configurations cost no more than the sampling
    it replaces (``_enumeration_is_cheaper``), and sampled otherwise to the resolution of ``draw_count`` effective
    draws per mean (``_cluster_chain``). The draws are reproducible from ``seed`` whatever the thread count. See
    ``ClusterMoments`` for the packing."""
    count = len(clusters)
    member_offsets = np.zeros(count + 1, dtype=np.int64)
    block_offsets = np.zeros(count + 1, dtype=np.int64)
    for position, members in enumerate(clusters):
        size = int(np.asarray(members).shape[0])
        if np.shape(precision[position]) != (size, size) or np.shape(shift[position]) != (size,):
            raise ValueError(f"cluster {position}: the cavity must be ({size}, {size}) and ({size},)")
        member_offsets[position + 1] = member_offsets[position] + size
        block_offsets[position + 1] = block_offsets[position] + size * size
    member_lists = [np.asarray(members, dtype=np.int64) for members in clusters]
    rows = np.concatenate(member_lists) if count else np.zeros(0, np.int64)
    local = prior.rows(rows)
    precision_values = np.concatenate([np.asarray(block, dtype=np.float64).ravel() for block in precision]) if count else np.zeros(0)
    shift_values = np.concatenate([np.asarray(values, dtype=np.float64) for values in shift]) if count else np.zeros(0)
    if state is None:
        pieces = [_cavity_modes(prior.rows(members), np.asarray(block, dtype=np.float64), np.asarray(values, dtype=np.float64))
                  for members, block, values in zip(member_lists, precision, shift)]
    else:
        pieces = [np.asarray(values, dtype=np.int64) for values in state]
    states = np.ascontiguousarray(np.concatenate(pieces) if count else np.zeros(0, np.int64), dtype=np.int64)
    total = int(member_offsets[-1])
    means, mean_errors = np.empty(total), np.empty(total)
    covariances = np.empty(int(block_offsets[-1]))
    log_normalizers, log_normalizer_errors = np.empty(count), np.empty(count)
    sweeps = np.zeros(count, dtype=np.int64)
    exact, resolved, proper = np.zeros(count, dtype=np.bool_), np.zeros(count, dtype=np.bool_), np.zeros(count, dtype=np.bool_)
    _clusters(
        member_offsets, block_offsets, precision_values, shift_values, local.log_weights, local.variances, local.log_scale, prior.spacing,
        states, np.uint64(int(seed) % (1 << int(np.iinfo(np.uint64).bits))), int(draw_count),
        means, covariances, mean_errors, log_normalizers, log_normalizer_errors, sweeps, exact, resolved, proper,
    )
    return ClusterMoments(
        member_offsets=member_offsets, block_offsets=block_offsets, mean=means, covariance=covariances, mean_error=mean_errors,
        log_normalizer=log_normalizers, log_normalizer_error=log_normalizer_errors, state=states, sweeps=sweeps, exact=exact,
        resolved=resolved, proper=proper,
    )


def exact_tilted_moments(prior: NodePrior, precision: F64Array, shift: F64Array) -> tuple[float, F64Array, F64Array]:
    """(log Z, mean, covariance) of one cluster's tilted law by exact enumeration of all K^m configurations, whatever
    the cost: the reference every sampled answer is checked against (``prior`` holds the cluster's own rows)."""
    log_z, mean, second, ok = _enumerate(
        np.ascontiguousarray(precision, dtype=np.float64), np.ascontiguousarray(shift, dtype=np.float64), prior.log_weights, prior.variances
    )
    if not ok:
        raise FloatingPointError("the tilted law is improper: the cavity is indefinite past what the prior's variances bound")
    return float(log_z), mean, second - np.outer(mean, mean)


# ------------------------------------------------------------------ the global reference: the sample side


def _group_partners(columns: F64Array) -> I64Array:
    """Each group's most correlated other group (the largest squared correlation of their columns), -1 in a window of
    one group; formed in panels n wide, so the correlations' temporary is groups x n."""
    count = columns.shape[1]
    if count < 2:
        return np.full(count, -1, dtype=np.int64)
    norms = np.sqrt(np.einsum("ij,ij->j", columns, columns))
    partner = np.empty(count, dtype=np.int64)
    width = max(1, columns.shape[0])
    for start in range(0, count, width):
        stop = min(start + width, count)
        correlation = columns.T @ columns[:, start:stop]
        correlation /= norms[:, None]
        correlation /= norms[None, start:stop]
        np.square(correlation, out=correlation)
        correlation[np.arange(start, stop), np.arange(stop - start)] = -1.0
        partner[start:stop] = np.argmax(correlation, axis=0)
    return partner


def _member_partners(group: I64Array, group_partner: I64Array) -> I64Array:
    """Each member's partner: the next member of its own exact-tie group (a correlation of one) where it has one, else
    the first member of its group's most correlated group."""
    order = np.argsort(group, kind="stable")
    ordered = group[order]
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    stops = np.r_[starts[1:], ordered.shape[0]]
    first = np.full(int(group.max()) + 1 if group.size else 0, -1, dtype=np.int64)
    first[ordered[starts]] = order[starts]
    partner = np.full(group.shape[0], -1, dtype=np.int64)
    for start, stop in zip(starts, stops):
        members = order[start:stop]
        if members.shape[0] > 1:
            partner[members] = np.roll(members, -1)
        else:
            other = group_partner[group[members[0]]]
            partner[members[0]] = first[other] if other >= 0 else -1
    return partner


@dataclass(frozen=True)
class _Block:
    """One panel of a sweep: ``members`` (updated in this order) and ``local`` (those members, then their partners
    outside the panel), the local members' groups ``groups`` with each local member's index into them ``group_of``,
    and each update's partner as a local member index (-1 for none) ``partner``."""

    members: I64Array
    local: I64Array
    groups: I64Array
    group_of: I64Array
    partner: I64Array


def block_width(sample_dimension: int) -> int:
    """The sweep's panel width: floor(sqrt(r)) members. A panel's rank-one steps cost |S|^2 each on its local Gram
    (|S| <= twice the width with the partners), O(r) per update, below the O(r^2) per member that the panel's
    sample-side products (M X_S, X_S' M X_S and the Woodbury update of M) cost, while those products stay matrix-matrix
    (BLAS-3)."""
    return max(1, math.isqrt(max(int(sample_dimension), 1)))


@dataclass(frozen=True)
class SampleSideDesign:
    """The whole window's likelihood on the covariates' orthogonal complement (r = n - k dimensions, where the flat
    prior on the covariate effects leaves it): ``columns`` (r x groups, Fortran) the groups' columns there, ``target``
    (r,), ``group`` each member's column, ``partner`` each member's scale-exchange partner, and the sweep's panels."""

    columns: F64Array
    target: F64Array
    group: I64Array
    partner: I64Array
    blocks: tuple[_Block, ...]

    @property
    def dimension(self) -> int:
        return int(self.columns.shape[0])

    @property
    def member_count(self) -> int:
        return int(self.group.shape[0])

    @classmethod
    def build(cls, carriers: F64Array, target: F64Array, basis: F64Array, group: I64Array) -> SampleSideDesign:
        """From the groups' columns ``carriers`` (n x groups), ``target`` (n,), an orthonormal basis of the covariates'
        span ``basis`` (n x k) and each member's group."""
        carriers = np.asarray(carriers, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if basis.shape[1]:
            complement = np.linalg.qr(basis, mode="complete")[0][:, basis.shape[1] :]
            columns = np.asfortranarray(complement.T @ carriers)
            reduced = complement.T @ target
        else:
            columns, reduced = np.asfortranarray(carriers), target.copy()
        group = np.asarray(group, dtype=np.int64)
        partner = _member_partners(group, _group_partners(columns))
        width = block_width(columns.shape[0])
        blocks = []
        for start in range(0, group.shape[0], width):
            members = np.arange(start, min(start + width, group.shape[0]), dtype=np.int64)
            partners = partner[members]
            outside = np.setdiff1d(partners[partners >= 0], members)
            local = np.concatenate([members, outside])
            sorter = np.argsort(local)
            partner_local = np.full(members.shape[0], -1, dtype=np.int64)
            has = partners >= 0
            partner_local[has] = sorter[np.searchsorted(local[sorter], partners[has])]
            groups, group_of = np.unique(group[local], return_inverse=True)
            blocks.append(_Block(members=members, local=local, groups=groups.astype(np.int64), group_of=np.asarray(group_of, dtype=np.int64).ravel(), partner=partner_local))
        return cls(columns=columns, target=reduced, group=group, partner=partner, blocks=tuple(blocks))

    @classmethod
    def of(cls, statistics) -> SampleSideDesign:
        """From ``small_n.DenseStatistics``: its members' groups and the covariates' basis."""
        return cls.build(statistics.design.carriers, statistics.target, statistics.design.basis, statistics.design.members)


class _Batches:
    """Batch sums of a vector series whose length is not known in advance: batches of b sweeps, merged pairwise (b
    doubles) whenever their number passes 2 sqrt(T), so there are always between sqrt(T) and 2 sqrt(T) of them, the
    consistent batch-means regime (Flegal and Jones 2010: batch size ~ sqrt(T))."""

    def __init__(self, width: int) -> None:
        self.sums: list[F64Array] = []
        self.size = 1
        self.partial = np.zeros(width)
        self.count = 0
        self.total = np.zeros(width)
        self.sweeps = 0

    def add(self, values: F64Array) -> None:
        self.total += values
        self.sweeps += 1
        self.partial += values
        self.count += 1
        if self.count == self.size:
            self.sums.append(self.partial)
            self.partial = np.zeros_like(self.total)
            self.count = 0
            if len(self.sums) > 2 * math.sqrt(self.sweeps):
                merged = [self.sums[index] + self.sums[index + 1] for index in range(0, len(self.sums) - 1, 2)]
                if len(self.sums) % 2:
                    self.partial = self.sums[-1] + self.partial
                    self.count = self.size
                self.sums = merged
                self.size *= 2

    def means(self) -> F64Array:
        """The complete batches' means (batches x width)."""
        return np.array(self.sums) / self.size if self.sums else np.zeros((0, self.total.shape[0]))

    @property
    def mean(self) -> F64Array:
        return self.total / max(self.sweeps, 1)


@dataclass
class _ChainStatistics:
    """One chain's Rao-Blackwellized statistics over some sweeps: node occupancies, E[1{z_j = k} beta_j^2 / v_j(k)],
    E[beta_j^4 / v_j^2] (sums over sweeps), the fitted genetic values E[X beta | z] and the noise statistic
    E[||y - X beta||^2 | z] / r at each sweep's start, the members' conditional means per sweep (batched), and the
    exchange counts."""

    occupancy: F64Array
    scaled: F64Array
    fourth: F64Array
    coefficients: _Batches
    fitted: list = field(default_factory=list)
    noise: list = field(default_factory=list)
    sweeps: int = 0
    swaps: F64Array = field(default_factory=lambda: np.zeros(2, dtype=np.int64))

    @classmethod
    def empty(cls, members: int, nodes: int) -> _ChainStatistics:
        return cls(occupancy=np.zeros((members, nodes)), scaled=np.zeros((members, nodes)), fourth=np.zeros(members), coefficients=_Batches(members))

    def averages(self) -> tuple[F64Array, F64Array, F64Array]:
        count = max(self.sweeps, 1)
        return self.occupancy / count, self.scaled / count, self.fourth / count


class _Chain:
    """One chain of the global collapsed sampler: every member's node ``state`` and, within a sweep, M = K^-1 for the
    sample-side kernel K = sigma^2 I + X D X' (r x r), formed afresh from the state at each sweep's start (so rounding
    never accumulates past one sweep) and moved by one Woodbury update per panel."""

    def __init__(self, design: SampleSideDesign, state: I64Array, generator: np.random.Generator) -> None:
        self.design = design
        self.state = np.ascontiguousarray(state, dtype=np.int64)
        self.generator = generator
        self.inverse: F64Array | None = None

    def _refresh(self, prior: NodePrior, noise: float) -> tuple[F64Array, float]:
        """M = (sigma^2 I + X D X')^-1 at the current state, (M y, tr M)."""
        design = self.design
        weights = np.bincount(design.group, weights=prior.variances[np.arange(design.member_count), self.state], minlength=design.columns.shape[1])
        roots = np.sqrt(weights)
        kernel = np.zeros((design.dimension, design.dimension), order="F")
        width = max(1, design.dimension)
        for start in range(0, design.columns.shape[1], width):
            panel = slice(start, min(start + width, design.columns.shape[1]))
            kernel = linalg.blas.dsyrk(1.0, design.columns[:, panel] * roots[None, panel], beta=1.0, c=kernel, overwrite_c=True)
        kernel = np.triu(kernel) + np.triu(kernel, 1).T
        kernel[np.diag_indices_from(kernel)] += noise
        factor = linalg.cho_factor(kernel, lower=False, check_finite=False)
        self.inverse = linalg.cho_solve(factor, np.eye(design.dimension), check_finite=False)
        return self.inverse @ design.target, float(np.trace(self.inverse))

    def sweep(self, prior: NodePrior, noise: float, statistics: _ChainStatistics) -> None:
        """One sweep over every member (panel by panel), adding its Rao-Blackwellized statistics."""
        design = self.design
        solved, trace = self._refresh(prior, noise)
        dimension = design.dimension
        statistics.fitted.append(design.target - noise * solved)
        statistics.noise.append((noise * noise * float(solved @ solved) + dimension * noise - noise * noise * trace) / dimension)
        conditional = np.zeros(design.member_count)
        spacing = prior.spacing
        for block in design.blocks:
            columns = design.columns[:, block.groups]
            product = self.inverse @ columns
            gram = columns.T @ product
            original = gram.copy()
            shift = product.T @ design.target
            delta = np.zeros(block.groups.shape[0])
            local = block.local
            state = self.state[local].copy()
            updated = block.members.shape[0]
            order = np.arange(updated, dtype=np.int64)
            log_scales = prior.log_scale[local]
            offset = np.zeros(updated, dtype=np.int64)
            has = block.partner >= 0
            if spacing > 0.0:
                offset[has] = np.rint((log_scales[block.partner[has]] - log_scales[order[has]]) / spacing).astype(np.int64)
            occupancy = np.zeros((local.shape[0], prior.grid.shape[0]))
            scaled = np.zeros_like(occupancy)
            fourth = np.zeros(local.shape[0])
            means = np.zeros(local.shape[0])
            proper = _scan(
                gram, shift, delta, block.group_of, order, block.partner, offset, state, prior.log_weights[local], prior.variances[local],
                self.generator.random(_UNIFORMS_PER_UPDATE * updated), occupancy, scaled, fourth, means, statistics.swaps,
            )
            if not proper:
                raise FloatingPointError("a collapsed conditional is improper on the data's own likelihood")
            statistics.occupancy[block.members] += occupancy[:updated]
            statistics.scaled[block.members] += scaled[:updated]
            statistics.fourth[block.members] += fourth[:updated]
            conditional[block.members] = means[:updated]
            self.state[local] = state
            changed = np.flatnonzero(delta)
            if changed.size:
                # K' = K + X_c diag(delta) X_c', so M' = M - W_c diag(delta) (I + G_c diag(delta))^-1 W_c' with W = M X_S
                # and G = X_S' M X_S at the panel's start (Woodbury; I + G_c diag(delta) is invertible because its
                # determinant is det K' / det K > 0).
                changes = delta[changed]
                system = original[np.ix_(changed, changed)] * changes[None, :]
                system[np.diag_indices_from(system)] += 1.0
                solved_panel = np.linalg.solve(system, product[:, changed].T)
                self.inverse -= (product[:, changed] * changes[None, :]) @ solved_panel
        statistics.coefficients.add(conditional)
        statistics.sweeps += 1

    def run(self, prior: NodePrior, noise: float, sweeps: int, statistics: _ChainStatistics | None = None) -> _ChainStatistics:
        if statistics is None:
            statistics = _ChainStatistics.empty(self.design.member_count, prior.grid.shape[0])
        for _ in range(sweeps):
            self.sweep(prior, noise, statistics)
        return statistics


def _prior_draws(prior: NodePrior, generator: np.random.Generator) -> I64Array:
    """Each member's node drawn from its prior pi: an overdispersed start, so the chains' agreement means something."""
    cumulative = np.cumsum(np.exp(prior.log_weights - prior.log_weights.max(axis=1, keepdims=True)), axis=1)
    target = generator.random(cumulative.shape[0]) * cumulative[:, -1]
    return np.minimum((cumulative < target[:, None]).sum(axis=1), prior.grid.shape[0] - 1).astype(np.int64)


# ------------------------------------------------------------------ diagnostics


def split_rhat(series: Sequence[F64Array]) -> tuple[F64Array, float]:
    """Split R-hat (Gelman et al., Bayesian Data Analysis, 3rd ed., section 11.4) of every column of the chains'
    series (each T x d), and its aggregate over the columns: the pooled between- and within-sequence variances summed
    over the columns, sqrt(((N - 1)/N sum W + sum B / N) / sum W), which weighs each column by its own spread."""
    length = min(values.shape[0] for values in series) // 2
    halves = [values[offset : offset + length] for values in series for offset in (values.shape[0] - 2 * length, values.shape[0] - length)]
    stacked = np.stack(halves)
    within = stacked.var(axis=1, ddof=1).mean(axis=0)
    between = length * stacked.mean(axis=1).var(axis=0, ddof=1)
    pooled = (length - 1) / length * within + between / length
    with np.errstate(divide="ignore", invalid="ignore"):
        per_column = np.sqrt(pooled / within)
        aggregate = float(np.sqrt(pooled.sum() / within.sum()))
    return per_column, aggregate


def pooled_mc_variance(series: Sequence[F64Array]) -> F64Array:
    """The Monte Carlo variance of the chains' pooled mean of every column, from each chain's batch means."""
    variances = [_batch_variances(np.ascontiguousarray(values), 0, values.shape[0])[0] for values in series]
    return np.sum(variances, axis=0) / len(series) ** 2


def batch_mc_variance(batch_means: Sequence[F64Array]) -> F64Array:
    """The Monte Carlo variance of the chains' pooled mean from each chain's batch means (a batches x d array of means
    of equal-size batches): Var(batch means) / batches per chain, averaged over the chains' squares."""
    variances = [values.var(axis=0, ddof=1) / values.shape[0] for values in batch_means]
    return np.sum(variances, axis=0) / len(batch_means) ** 2


@dataclass(frozen=True)
class PosteriorSample:
    """The global sampler's answer at fixed hyperparameters: ``mean`` the Rao-Blackwellized posterior mean of every
    member's effect, ``batch_means`` each chain's batch means of it (for any linear prediction's Monte Carlo error and
    split R-hat), ``occupancy`` every member's posterior node probabilities, and the precision dials: the fitted genetic
    values' Monte Carlo error in the posterior metric (sum_i Var_MC(f_i) / sigma^2, resolved at <= 1/K), their aggregate
    split R-hat (resolved at <= sqrt(1 + 1/K)), its per-value maximum, the effective sample size, sweeps per chain,
    the exchange acceptance, and ``status`` ("resolved", or "unresolved" when a doubling no longer raised the
    effective sample size)."""

    mean: F64Array
    batch_means: tuple[F64Array, ...]
    occupancy: F64Array
    fitted: F64Array
    fitted_error: float
    rhat: float
    rhat_max: float
    effective_size: float
    sweeps: int
    exchange_acceptance: float
    status: str


def _sampling_phase(chains: Sequence[_Chain], prior: NodePrior, noise: float, start_length: int, draw_count: int) -> PosteriorSample:
    """Every chain at fixed hyperparameters, doubling (each round runs as many sweeps as the chains have), until the
    fitted genetic values' pooled mean is resolved: Monte Carlo error sum_i Var_MC(f_i) / sigma^2 <= 1/K (the draws'
    resolution ||X d||^2 / sigma^2 <= 1/K that ``small_n._mode_mixture`` stops at) and aggregate split R-hat
    <= sqrt(1 + 1/K) (the chains' means disagree by at most 1/K of the posterior spread). A doubling that raises no
    effective sample size ends the phase unresolved."""
    statistics = [chain.run(prior, noise, start_length) for chain in chains]
    best = 0.0
    while True:
        fitted = [np.array(values.fitted) for values in statistics]
        mc_variance = pooled_mc_variance(fitted)
        spread = np.mean([values.var(axis=0, ddof=1) for values in fitted], axis=0)
        per_value, rhat = split_rhat(fitted)
        error = float(mc_variance.sum()) / noise
        effective = float(spread.sum() / mc_variance.sum()) if mc_variance.sum() > 0.0 else np.inf
        resolved = error <= 1.0 / draw_count and rhat <= math.sqrt(1.0 + 1.0 / draw_count)
        if resolved or effective <= best:
            break
        best = effective
        for chain, values in zip(chains, statistics):
            chain.run(prior, noise, values.sweeps, values)
    occupancy = np.mean([values.averages()[0] for values in statistics], axis=0)
    swaps = np.sum([values.swaps for values in statistics], axis=0)
    return PosteriorSample(
        mean=np.mean([values.coefficients.mean for values in statistics], axis=0),
        batch_means=tuple(values.coefficients.means() for values in statistics),
        occupancy=occupancy,
        fitted=np.mean([values.mean(axis=0) for values in fitted], axis=0),
        fitted_error=error,
        rhat=rhat,
        rhat_max=float(np.nanmax(per_value)),
        effective_size=effective,
        sweeps=int(statistics[0].sweeps),
        exchange_acceptance=float(swaps[1] / swaps[0]) if swaps[0] else float("nan"),
        status="resolved" if resolved else "unresolved",
    )


def sample_posterior(design: SampleSideDesign, prior: NodePrior, noise: float, draw_count: int, seed: int, chains: int = 2) -> PosteriorSample:
    """The exact posterior at fixed hyperparameters (``prior`` over the design's members, noise variance ``noise``) by
    ``chains`` independent chains from overdispersed prior draws, to the resolution of ``_sampling_phase``: the
    reference against which every approximation at the same hyperparameters is checked."""
    runs = [_Chain(design, _prior_draws(prior, generator), generator) for generator in (np.random.default_rng([seed, index]) for index in range(chains))]
    return _sampling_phase(runs, prior, noise, draw_count, draw_count)


# ------------------------------------------------------------------ empirical Bayes on the marginal likelihood


def complete_data_score(prior: ScaleMixturePrior, coefficients: F64Array, occupancy: F64Array, scaled: F64Array) -> F64Array:
    """The gradient of log p(y | x) in z = (each class's eta, the scale coefficients) by Fisher's identity: the
    posterior mean of the complete-data score. With the node probabilities m_j = P(z_j = . | y) and
    A_j = E[beta_j^2 / v_j(z_j) | y] (``occupancy`` and ``scaled`` summed over nodes), class c's eta gets
    sum_{j in c} m_j - N_c pi_c and the scale coefficients sum_j d_j (A_j - 1) / 2, since log N(beta; 0, u e^t)
    moves with a = log u as -1/2 + beta^2 / (2 v)."""
    density = np.exp(class_log_density(prior, coefficients))
    grid_size = prior.grid_size
    gradient = np.zeros(prior.density_size + prior.scale_size)
    for class_position, rows in enumerate(prior.class_rows):
        gradient[class_position * grid_size : (class_position + 1) * grid_size] = occupancy[rows].sum(axis=0) - rows.shape[0] * density[class_position]
    if prior.scale_size:
        gradient[prior.density_size :] = prior.scale_design.T @ (0.5 * (scaled.sum(axis=1) - 1.0))
    return gradient


def observed_information(prior: ScaleMixturePrior, coefficients: F64Array, occupancy: F64Array, scaled: F64Array, fourth: F64Array) -> F64Array:
    """The Hessian of log p(y | x) in z by Louis' identity, E[complete-data Hessian] + Cov[complete-data score], with
    the covariance taken variant by variant (each member's own score covariance exactly; the covariance between two
    members' scores, which LD couples, is left out: it moves the Newton metric, never the stationary point, which the
    exact gradient fixes). Per member of class c: eta block diag(m) - m m' - (diag pi - pi pi') (the complete data's
    -(diag pi - pi pi') plus Cov(e_z) = diag m - m m'), scale block d d' (-A/2 + (E[beta^4 / v^2] - A^2) / 4), cross
    block (q - m A) d' / 2 with q_k = E[1{z = k} beta^2 / v_k]."""
    density = np.exp(class_log_density(prior, coefficients))
    grid_size = prior.grid_size
    size = prior.density_size + prior.scale_size
    hessian = np.zeros((size, size))
    squares = scaled.sum(axis=1)
    scale_span = slice(prior.density_size, size)
    design = prior.scale_design
    if prior.scale_size:
        curvature = -0.5 * squares + 0.25 * (fourth - squares * squares)
        hessian[scale_span, scale_span] = design.T @ (curvature[:, None] * design)
    for class_position, rows in enumerate(prior.class_rows):
        span = slice(class_position * grid_size, (class_position + 1) * grid_size)
        marginals = occupancy[rows]
        class_density = density[class_position]
        hessian[span, span] = (
            np.diag(marginals.sum(axis=0)) - marginals.T @ marginals - rows.shape[0] * (np.diag(class_density) - np.outer(class_density, class_density))
        )
        if prior.scale_size:
            cross = (0.5 * (scaled[rows] - marginals * squares[rows, None])).T @ design[rows]
            hessian[span, scale_span] = cross
            hessian[scale_span, span] = cross.T
    return hessian


def _magnitude_inverse(matrix: F64Array) -> F64Array:
    """|M|^-1 by the eigendecomposition with every eigenvalue replaced by its magnitude, those within rounding of zero
    raised to that floor (``scale_mixture_ep._resolved_spectrum``): the metric of a Newton step where M is indefinite."""
    eigenvalues, eigenvectors = _resolved_spectrum(_spectrum(matrix))
    return (eigenvectors / np.abs(eigenvalues)[None, :]) @ eigenvectors.T


def fellner_schall(
    prior: ScaleMixturePrior, log_smoothing: F64Array, coefficients: F64Array, information: F64Array, lower: F64Array, upper: F64Array,
) -> tuple[F64Array, float]:
    """The penalty weights' update for the Laplace marginal likelihood of x (Wood and Fasiolo 2017, "A generalized
    Fellner-Schall method for smoothing parameter optimization", Biometrics 73:1071): lambda_i <- lambda_i
    (tr(S^+ S_i) - tr((I + S)^-1 S_i)) / x' S_i x with I the data's information in x, S the total penalty and S^+ its
    pseudo-inverse over each connected penalty group; each log weight clipped to its resolvable range
    (``_smoothing_bounds``; the upper end is the lambda = infinity edge to the tolerance). Returns the new log weights
    and the Laplace V's predicted gain of the move, sum_i (dV/drho_i)^2 / (lambda_i x' S_i x) with
    dV/drho_i = lambda_i (tr(S^+ S_i) - tr((I + S)^-1 S_i) - x' S_i x) / 2 (V's curvature in rho_i is
    lambda_i x' S_i x / 2 to leading order), zero for a weight held at a bound its slope pushes past."""
    penalty = _penalty_matrix(prior, log_smoothing)
    covariance = _magnitude_inverse(information + penalty)
    group_of = {int(coordinate): group for group in _penalty_groups(prior) for coordinate in group}
    updated = np.array(log_smoothing, dtype=np.float64, copy=True)
    gain = 0.0
    for position, block in enumerate(prior.smoothing_blocks):
        weight = float(np.exp(log_smoothing[position]))
        coordinates = block.coordinates
        group = group_of[int(coordinates[0])]
        embedded = np.zeros((group.shape[0], group.shape[0]))
        inside = np.searchsorted(group, coordinates)
        embedded[np.ix_(inside, inside)] = block.matrix
        prior_trace = _pseudo_inverse_trace(penalty[np.ix_(group, group)], embedded)
        posterior_trace = float(np.sum(covariance[np.ix_(coordinates, coordinates)] * block.matrix))
        residual = block.factor @ coefficients[coordinates]
        quadratic = float(residual @ residual)
        numerator = prior_trace - posterior_trace
        if numerator <= 0.0:
            proposal = lower[position]
        elif quadratic <= 0.0:
            proposal = upper[position]
        else:
            proposal = log_smoothing[position] + math.log(numerator / quadratic)
        updated[position] = float(np.clip(proposal, lower[position], upper[position]))
        slope = 0.5 * weight * (numerator - quadratic)
        pinned = (log_smoothing[position] >= upper[position] and slope > 0.0) or (log_smoothing[position] <= lower[position] and slope < 0.0)
        if not pinned:
            gain += slope * slope / (weight * quadratic) if weight * quadratic > 0.0 else np.inf
    return updated, gain


@dataclass(frozen=True)
class EmpiricalBayes:
    """The empirical Bayes' answer: the hyperparameters (log weights finite, the upper resolvable end standing for the
    lambda = infinity edge), the noise variance, the rounds' history (sweeps per chain, the cross-validated Newton
    decrement 1/2 g_A' I^-1 g_B, its Monte Carlo part (g_A - g_B)' I^-1 (g_A - g_B) / 8, the noise's and the weights'
    gains) and ``status``: "resolved" when every one of those is within the tolerance, "unresolved" when doubling the
    sweeps twice in a row did not shrink the Monte Carlo part (the chains are not mixing at this length)."""

    hyperparameters: MixtureHyperparameters
    noise: float
    history: tuple[dict, ...]
    status: str
    sweeps: int


def empirical_bayes(
    design: SampleSideDesign, prior: ScaleMixturePrior, start: MixtureHyperparameters, start_noise: float, chains: Sequence[_Chain],
    draw_count: int,
) -> EmpiricalBayes:
    """Empirical Bayes of (x, the penalty weights, sigma^2) on the marginal likelihood p(y | x, sigma^2) itself, by
    stochastic approximation with Newton steps (Gu and Kong 1998): each round runs every chain L sweeps at fixed
    hyperparameters and forms the complete-data score's posterior mean per chain (``complete_data_score``) and the
    pooled observed information (``observed_information``); the weights move by one Fellner-Schall step on the
    Laplace V of x, sigma^2 by its EM step (the posterior mean of ||y - X beta||^2 / r, whose missing information is
    small: the effects' posterior spread is a small part of the residual's), and x by a trust-region Newton step on
    the penalized log marginal likelihood.

    Monte Carlo noise is kept out of the steps by cross-validation between the chains: the two chains' scores g_A and
    g_B are independent given the hyperparameters, so 1/2 g_A' I^-1 g_B is an unbiased estimate of the Newton
    decrement of the true gradient, and (g_A - g_B)' I^-1 (g_A - g_B) / 8 that of the pooled score's Monte Carlo part.
    Where the Monte Carlo part is at least the cross decrement, a step would follow noise: the round's length doubles
    instead (Booth and Hobert 1999's rule of growing the Monte Carlo sample when the step is not resolved). The loop
    ends when the cross decrement, the Monte Carlo part, the noise's gain (``noise_gain``) and the weights' predicted
    gain are all within the tolerance 1/(2K) nats (``small_n.fit_small_n``'s). A step is judged on the next round's
    gradient: its realized gain is the trapezoid of the two gradients along it (the outer loop's rule,
    ``scale_mixture_ep._path_gain``'s leading term); a step that realized no gain is undone and the radius halved, and
    one that realized at least half its model's gain at the radius doubles it (Nocedal and Wright, Numerical
    Optimization, Algorithm 4.1, with its acceptance threshold at 0 and its expansion threshold at 1/2)."""
    tolerance = 0.5 / draw_count
    coefficients = np.array(start.coefficients, dtype=np.float64, copy=True)
    mapping = prior.coefficient_map
    noise = float(start_noise)
    log_smoothing: F64Array | None = None
    length = 1
    radius: float | None = None
    previous: dict | None = None
    history: list[dict] = []
    stalled = 0
    last_noise_part = np.inf
    sweeps = 0
    while True:
        node = NodePrior.of(prior, coefficients)
        rounds = [chain.run(node, noise, length) for chain in chains]
        sweeps += length
        averages = [values.averages() for values in rounds]
        scores = [complete_data_score(prior, coefficients, occupancy, scaled) for occupancy, scaled, _fourth in averages]
        pooled = [np.mean([values[index] for values in averages], axis=0) for index in range(len(averages[0]))]
        hessian = observed_information(prior, coefficients, *pooled)
        information = -(mapping.T @ hessian @ mapping)
        bounds = _smoothing_bounds(prior, _Objective(value=0.0, gradient=np.mean(scores, axis=0), hessian=hessian, magnitude=0.0, rounding=0.0))
        lower = np.array([bound[0] for bound in bounds])
        upper = np.array([bound[1] for bound in bounds])
        if log_smoothing is None:
            # Every weight starts at the centre of its resolvable range, where the penalty's geometric-mean eigenvalue
            # matches the data's curvature (``scale_mixture_ep._maximize_evidence``'s release point): a weight at its
            # edge has no Fellner-Schall step out of it (x' S_i x = 0 there).
            log_smoothing = 0.5 * (lower + upper)
        log_smoothing = np.clip(log_smoothing, lower, upper)
        if previous is not None:
            penalty = _penalty_matrix(prior, previous["log_smoothing"])
            gradient_now = mapping.T @ np.mean(scores, axis=0) - penalty @ coefficients
            realized = 0.5 * float((previous["gradient"] + gradient_now) @ previous["step"])
            step_length = float(np.linalg.norm(previous["step"]))
            history[-1]["realized"] = realized
            if not realized > 0.0:
                coefficients = previous["coefficients"]
                radius = 0.5 * step_length
                previous = None
                continue
            if realized >= 0.5 * previous["predicted"] and step_length >= radius * (1.0 - math.sqrt(_EPSILON)):
                radius = 2.0 * radius
        log_smoothing, smoothing_gain = fellner_schall(prior, log_smoothing, coefficients, information, lower, upper)
        penalty = _penalty_matrix(prior, log_smoothing)
        gradients = [mapping.T @ score - penalty @ coefficients for score in scores]
        gradient = np.mean(gradients, axis=0)
        total = information + penalty
        metric = _magnitude_inverse(total)
        cross = 0.5 * float(gradients[0] @ metric @ gradients[-1])
        difference = gradients[0] - gradients[-1]
        noise_part = 0.125 * float(difference @ metric @ difference)
        new_noise = float(np.mean([value for values in rounds for value in values.noise]))
        gained = noise_gain(new_noise, noise, design.dimension, 0)
        noise = new_noise
        history.append({
            "sweeps": length, "cross_decrement": cross, "monte_carlo_part": noise_part, "noise_gain": gained, "smoothing_gain": smoothing_gain,
            "noise": noise, "log_smoothing": [float(value) for value in log_smoothing],
        })
        if max(cross, noise_part, gained, smoothing_gain) <= tolerance:
            status = "resolved"
            break
        if noise_part >= cross:
            stalled = stalled + 1 if noise_part >= last_noise_part else 0
            last_noise_part = noise_part
            if stalled >= 2:
                status = "unresolved"
                break
            length *= 2
            previous = None
            continue
        spectrum = _resolved_spectrum(_spectrum(total))
        if radius is None:
            # The Cauchy step's length on |I + S|, ||g||^3 / g'|I + S|g (``scale_mixture_ep._cauchy_radius``'s rule).
            curvature = float(np.sum(np.abs(spectrum[0]) * np.square(spectrum[1].T @ gradient)))
            radius = float(np.linalg.norm(gradient)) ** 3 / curvature
        step = _trust_region_step(total, gradient, radius, spectrum)
        predicted = float(gradient @ step) - 0.5 * float(step @ total @ step)
        previous = {"coefficients": coefficients, "gradient": gradient, "step": step, "predicted": predicted, "log_smoothing": log_smoothing}
        coefficients = coefficients + step
    return EmpiricalBayes(
        hyperparameters=MixtureHyperparameters(coefficients=coefficients, log_smoothing=log_smoothing), noise=noise, history=tuple(history),
        status=status, sweeps=sweeps,
    )


# ------------------------------------------------------------------ the fit

# Two chains: the fewest whose disagreement (split R-hat, the cross-validated Newton decrement) measures what one
# chain's own autocorrelation cannot, a mode the chain has not left.
_CHAINS = 2


@dataclass(frozen=True)
class ScaleSamplerFit:
    """One window's fit by the global reference sampler: Stage 0 (``small_n.DenseStatistics``), the prior, its
    empirical Bayes, the posterior at the fitted hyperparameters, the covariates' effects ``alpha`` (intercept first)
    and where the time went. ``coefficients`` are the members' posterior-mean effects on their standardized columns
    (in the store rows' own orientation, beta_j = s_j gamma_j)."""

    statistics: object
    prior: ScaleMixturePrior
    empirical_bayes: EmpiricalBayes
    posterior: PosteriorSample
    alpha: F64Array
    profile: dict

    @property
    def coefficients(self) -> F64Array:
        return self.statistics.signs * self.posterior.mean

    @property
    def coefficient_batches(self) -> tuple[F64Array, ...]:
        """Each chain's batch means of ``coefficients`` (batches x members)."""
        return tuple(self.statistics.signs[None, :] * values for values in self.posterior.batch_means)


def fit_scale_sampler(
    *,
    codes: np.ndarray,
    covariates: F64Array,
    target: F64Array,
    variant_class: np.ndarray,
    log_variance_offset: F64Array | None,
    draw_count: int,
    working_bytes: int,
    seed: int,
    annotations: Mapping[str, np.ndarray] | None = None,
    codes_per_unit: np.ndarray | None = None,
) -> ScaleSamplerFit:
    """The exact posterior of SV-PGS's prior on one window's dense training codes (as ``small_n.fit_small_n`` takes
    them): Stage 0 dense, the same prior (``small_n.small_n_prior``, annotations included) from the same moment start
    (``small_n.small_n_start``), empirical Bayes on the marginal likelihood (``empirical_bayes``), then the posterior
    at the fitted hyperparameters to the resolution of ``draw_count`` (``_sampling_phase``), from two chains started
    at independent prior draws."""
    from sv_pgs.small_n import dense_kernel_bytes, dense_stage0_bytes, dense_statistics, refuse_dense_route, small_n_prior, small_n_start

    started = time.perf_counter()
    sample_count, record_count = (int(size) for size in np.shape(codes))
    refuse_dense_route(dense_stage0_bytes(sample_count, record_count), working_bytes, "Stage 0's dense pass")
    offsets = np.zeros(record_count) if log_variance_offset is None else np.asarray(log_variance_offset, dtype=np.float64)
    statistics = dense_statistics(codes, covariates, target, offsets)
    refuse_dense_route(dense_kernel_bytes(sample_count, statistics.design.group_count), working_bytes, "the dense design and its n x n kernel")
    prior = small_n_prior(statistics, variant_class, offsets, draw_count, annotations, codes_per_unit)
    start, start_noise, moment = small_n_start(statistics, prior)
    design = SampleSideDesign.of(statistics)
    stage0_seconds = time.perf_counter() - started
    start_prior = NodePrior.of(prior, start.coefficients)
    generators = [np.random.default_rng([seed, index]) for index in range(_CHAINS)]
    chains = [_Chain(design, _prior_draws(start_prior, generator), generator) for generator in generators]
    fitted = empirical_bayes(design, prior, start, start_noise, chains, draw_count)
    eb_seconds = time.perf_counter() - started - stage0_seconds
    node = NodePrior.of(prior, fitted.hyperparameters.coefficients)
    posterior = _sampling_phase(chains, node, fitted.noise, max(fitted.history[-1]["sweeps"], draw_count), draw_count)
    mean = posterior.mean
    alpha = statistics.covariate_pseudo_inverse @ (statistics.covariates.T @ statistics.target - statistics.loading @ mean)
    profile = {
        "stage0_seconds": stage0_seconds,
        "empirical_bayes_seconds": eb_seconds,
        "sampling_seconds": time.perf_counter() - started - stage0_seconds - eb_seconds,
        "total_seconds": time.perf_counter() - started,
        "samples": statistics.sample_count,
        "members": int(design.member_count),
        "groups": int(design.columns.shape[1]),
        "grid": int(prior.grid_size),
        "classes": int(prior.class_count),
        "start_heritability": float(moment.heritability),
        "empirical_bayes_status": fitted.status,
        "empirical_bayes_rounds": len(fitted.history),
        "empirical_bayes_sweeps": fitted.sweeps,
        "empirical_bayes_last": fitted.history[-1],
        "noise": fitted.noise,
        "log_smoothing": [float(value) for value in fitted.hyperparameters.log_smoothing],
        "posterior_status": posterior.status,
        "posterior_sweeps": posterior.sweeps,
        "fitted_error": posterior.fitted_error,
        "rhat": posterior.rhat,
        "rhat_max": posterior.rhat_max,
        "effective_size": posterior.effective_size,
        "exchange_acceptance": posterior.exchange_acceptance,
    }
    return ScaleSamplerFit(statistics=statistics, prior=prior, empirical_bayes=fitted, posterior=posterior, alpha=alpha, profile=profile)



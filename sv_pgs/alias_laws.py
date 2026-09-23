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
other member's K atoms, all exact. A larger group's laws are formed on nodes evenly spaced in log V at the lattice's
spacing divided by a measured refinement (``law_resolution``; the lattice is a quadrature rule in log variance at its
spacing, ``scale_mixture_ep.derived_lattice``): prefix laws member by member, each sum's mass split between its two
neighbouring nodes linearly in log V (mass and the mean of log V kept), suffix laws the same way from the other end,
and a member's leave-one-out law the convolution of the prefix before it with the suffix after it. Every leave-one-out
law and every member's atoms lie on even grids in log V, so where a sum W_i + v_k lands and the shares v / V depend on
i - r k alone (r the grids' spacing ratio): each convolution and each member's terms read those from one table, and
the kernel is evaluated at the exact sum. Members of one class with one scale are exchangeable: their leave-one-out
laws are one law, formed once, and their terms are computed once and are exactly equal. Masses are held in logs (a
tilted law can put its weight on masses far below the largest), and convolved linearly in bands of half the double
range, which loses no term that logs would keep. The laws depend on the hyperparameters,
not on the cavity: they are formed once per hyperparameter setting (``GroupLaws.of``) and read at every cavity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numba
import numpy as np

from sv_pgs._typing import F64Array, I64Array

# Half the double range in logs: a product of two masses each within it of its band's largest is representable.
_BAND_WIDTH = -float(np.log(np.finfo(np.float64).tiny)) / 2


@numba.njit(cache=True, error_model="numpy")
def _softplus(value):
    """log(1 + e^value), without overflow."""
    if value > 0.0:
        return value + np.log1p(np.exp(-value))
    return np.log1p(np.exp(value))


@numba.njit(cache=True, error_model="numpy")
def _landing(origin, stride, count, node_count, spacing):
    """Where a sum lands, for a law on nodes i (spacing ``spacing`` in log V from its origin) plus atoms at
    ``origin`` + ``stride`` k nodes (k < ``count``): with u = i - stride k and D = u - origin, the sum is at node
    i + L(u), L(u) = softplus(-spacing D) / spacing. Returns (L, log share of the atom, log share of the law) over u
    from -stride (count - 1) to node_count - 1."""
    size = node_count + stride * (count - 1)
    offset = np.empty(size)
    log_atom_share = np.empty(size)
    log_law_share = np.empty(size)
    for index in range(size):
        distance = spacing * (index - stride * (count - 1) - origin)
        offset[index] = _softplus(-distance) / spacing
        log_atom_share[index] = -_softplus(distance)
        log_law_share[index] = -_softplus(-distance)
    return offset, log_atom_share, log_law_share


@numba.njit(cache=True, error_model="numpy")
def _split(position, node_count):
    """(lower node, fraction to the upper) of a position, the upper end clamped."""
    lower = int(np.floor(position))
    if lower > node_count - 2:
        lower = node_count - 2
    fraction = position - lower
    if fraction > 1.0:
        fraction = 1.0
    if fraction < 0.0:
        fraction = 0.0
    return lower, fraction


@numba.njit(cache=True, error_model="numpy")
def _bin(origin, stride, log_mass, node_count, out):
    """Atoms at ``origin`` + ``stride`` k nodes onto the nodes, each split between its neighbours linearly in log V."""
    out[:] = -np.inf
    for k in range(log_mass.shape[0]):
        if log_mass[k] == -np.inf:
            continue
        lower, fraction = _split(origin + stride * k, node_count)
        if fraction < 1.0:
            out[lower] = np.logaddexp(out[lower], log_mass[k] + np.log1p(-fraction))
        if fraction > 0.0:
            out[lower + 1] = np.logaddexp(out[lower + 1], log_mass[k] + np.log(fraction))


@numba.njit(cache=True, error_model="numpy")
def _bands(log_mass, band_width):
    """(largest, band count, each mass's band): band b holds the masses between b and b + 1 band widths below the
    largest (-1 for a zero mass)."""
    largest = -np.inf
    for value in log_mass:
        if value > largest:
            largest = value
    band = np.full(log_mass.shape[0], -1, dtype=np.int64)
    count = 0
    if largest == -np.inf:
        return largest, count, band
    for index in range(log_mass.shape[0]):
        if log_mass[index] > -np.inf:
            band[index] = int((largest - log_mass[index]) // band_width)
            if band[index] + 1 > count:
                count = band[index] + 1
    return largest, count, band


@numba.njit(cache=True, error_model="numpy")
def _convolve(law, origin, stride, log_mass, spacing, out, peak, band_width):
    """The law of the sum of a law on the nodes and independent atoms at ``origin`` + ``stride`` k nodes, split onto the
    nodes (module docstring); ``peak`` is scratch."""
    node_count = law.shape[0]
    count = log_mass.shape[0]
    offset, _atom, _law = _landing(origin, stride, count, node_count, spacing)
    base = stride * (count - 1)
    # The split depends on u alone (the landing's whole and fractional parts) except at the clamped upper end.
    whole = np.empty(offset.shape[0], dtype=np.int64)
    upper_weight = np.empty(offset.shape[0])
    for index in range(offset.shape[0]):
        whole[index] = int(np.floor(offset[index]))
        upper_weight[index] = offset[index] - whole[index]
    lower_weight = 1.0 - upper_weight
    # Linear accumulation in bands: each input's masses are grouped by their distance below its largest in steps of
    # ``band_width``, half the double range, and each pair of bands is summed linearly relative to the bands' largest
    # masses, so every product of two band-relative masses is representable; the bands' sums are combined in logs. One
    # band each is the common case.
    _law_peak, law_bands, law_band = _bands(law, band_width)
    _mass_peak, mass_bands, mass_band = _bands(log_mass, band_width)
    out[:] = -np.inf
    law_linear = np.empty(node_count)
    mass_linear = np.empty(count)
    for first_band in range(law_bands):
        law_top = -np.inf
        for i in range(node_count):
            if law_band[i] == first_band and law[i] > law_top:
                law_top = law[i]
        if law_top == -np.inf:
            continue
        for i in range(node_count):
            law_linear[i] = np.exp(law[i] - law_top) if law_band[i] == first_band else 0.0
        for second_band in range(mass_bands):
            mass_top = -np.inf
            for k in range(count):
                if mass_band[k] == second_band and log_mass[k] > mass_top:
                    mass_top = log_mass[k]
            if mass_top == -np.inf:
                continue
            for k in range(count):
                mass_linear[k] = np.exp(log_mass[k] - mass_top) if mass_band[k] == second_band else 0.0
            peak[:] = 0.0
            for i in range(node_count):
                if law_linear[i] == 0.0:
                    continue
                for k in range(count):
                    if mass_linear[k] == 0.0:
                        continue
                    product = law_linear[i] * mass_linear[k]
                    index = i - stride * k + base
                    lower = i + whole[index]
                    if lower <= node_count - 2:
                        peak[lower] += product * lower_weight[index]
                        peak[lower + 1] += product * upper_weight[index]
                    else:
                        lower, fraction = _split(i + offset[index], node_count)
                        peak[lower] += product * (1.0 - fraction)
                        peak[lower + 1] += product * fraction
            for node in range(node_count):
                if peak[node] > 0.0:
                    out[node] = np.logaddexp(out[node], np.log(peak[node]) + law_top + mass_top)


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _larger_laws(larger, with_loo, member_start, member_index, representative, member_origin, member_log_weight, stride, spacing, low,
                 law_start, law_log_mass, loo_start, loo_log_mass, band_width):
    """Each binned group's law and, where ``with_loo``, its representatives' leave-one-out laws on its nodes (module
    docstring)."""
    for position in numba.prange(larger.shape[0]):
        group = larger[position]
        start, stop = member_start[group], member_start[group + 1]
        count = stop - start
        node_count = law_start[group + 1] - law_start[group]
        prefix = np.empty((count, node_count))
        peak = np.empty(node_count)
        origins = np.empty(count)
        for p in range(count):
            origins[p] = (member_origin[member_index[start + p]] - low[group]) / spacing
        _bin(origins[0], stride, member_log_weight[member_index[start]], node_count, prefix[0])
        for p in range(1, count):
            _convolve(prefix[p - 1], origins[p], stride, member_log_weight[member_index[start + p]], spacing, prefix[p], peak, band_width)
        law_log_mass[law_start[group]:law_start[group + 1]] = prefix[count - 1]
        if not with_loo[position]:
            continue
        single_run = representative[member_index[stop - 1]] == member_index[start]
        if single_run:
            j = member_index[start]
            loo_log_mass[loo_start[j]:loo_start[j + 1]] = prefix[count - 2]
            continue
        suffix = np.empty((count, node_count))
        _bin(origins[count - 1], stride, member_log_weight[member_index[stop - 1]], node_count, suffix[count - 1])
        for p in range(count - 2, 0, -1):
            _convolve(suffix[p + 1], origins[p], stride, member_log_weight[member_index[start + p]], spacing, suffix[p], peak, band_width)
        for p in range(count):
            j = member_index[start + p]
            if representative[j] != j:
                continue
            target = loo_log_mass[loo_start[j]:loo_start[j + 1]]
            if p == 0:
                target[:] = suffix[1]
            elif p == count - 1:
                target[:] = prefix[count - 2]
            else:
                _convolve(prefix[p - 1], 0.0, 1, suffix[p + 1], spacing, target, peak, band_width)


@dataclass(frozen=True)
class GroupLaws:
    """Every alias group's law of V and its members' leave-one-out laws at one hyperparameter setting. Group g's law is
    the atoms ``law_start[g]:law_start[g + 1]`` (log V, log mass). Member j's own atoms are at log V =
    ``member_origin[j]`` + ``grid_spacing`` k with log weights ``member_log_weight[j]``; its leave-one-out law (a
    representative's, empty for a group of one and for a member that copies its ``representative``) is the masses
    ``loo_start[j]:loo_start[j + 1]`` on nodes at ``loo_low[j]`` + ``loo_spacing[j]`` i, where its own atoms are every
    ``loo_stride[j]``-th node."""

    groups: I64Array
    member_start: I64Array
    member_index: I64Array
    representative: I64Array
    law_start: I64Array
    law_log_variance: F64Array
    law_log_mass: F64Array
    member_origin: F64Array
    member_log_weight: F64Array
    grid_spacing: float
    loo_start: I64Array
    loo_log_mass: F64Array
    loo_low: F64Array
    loo_spacing: F64Array
    loo_stride: I64Array

    @property
    def group_count(self) -> int:
        return int(self.member_start.shape[0] - 1)

    @property
    def member_log_variance(self) -> F64Array:
        """Member j's atoms' log variances (members x K)."""
        return self.member_origin[:, None] + self.grid_spacing * np.arange(self.member_log_weight.shape[1])[None, :]

    @classmethod
    def of(cls, groups: I64Array, log_density: F64Array, class_index: I64Array, log_scale: F64Array, grid: F64Array, refinement: int = 1) -> "GroupLaws":
        """The laws at the class log densities ``log_density`` (classes x K, normalized), each member's class and log
        scale log u_j, on the evenly spaced lattice ``grid`` (module docstring); a larger group's nodes are at the
        lattice's spacing divided by ``refinement`` (the splitting's error falls as its square; ``law_resolution``
        measures it)."""
        groups = np.asarray(groups, dtype=np.int64)
        class_index = np.asarray(class_index, dtype=np.int64)
        log_scale = np.asarray(log_scale, dtype=np.float64)
        grid = np.asarray(grid, dtype=np.float64)
        member_count, node_count = groups.shape[0], grid.shape[0]
        group_count = int(groups.max()) + 1
        grid_spacing = float(grid[1] - grid[0]) if node_count > 1 else 1.0
        spacing = grid_spacing / refinement
        member_origin = log_scale + grid[0]
        member_log_weight = np.ascontiguousarray(np.asarray(log_density, dtype=np.float64)[class_index])
        # Within a group, exchangeable members (one class, one scale) are adjacent; each run's first is its representative.
        member_index = np.lexsort((log_scale, class_index, groups)).astype(np.int64)
        sizes = np.bincount(groups, minlength=group_count)
        member_start = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        ordered_group, ordered_class, ordered_scale = groups[member_index], class_index[member_index], log_scale[member_index]
        new_run = np.ones(member_count, dtype=bool)
        new_run[1:] = (ordered_group[1:] != ordered_group[:-1]) | (ordered_class[1:] != ordered_class[:-1]) | (ordered_scale[1:] != ordered_scale[:-1])
        run_first = member_index[np.maximum.accumulate(np.where(new_run, np.arange(member_count), 0))]
        representative = np.empty(member_count, dtype=np.int64)
        representative[member_index] = run_first
        member_size = sizes[groups]
        # Laws: a single's K atoms; binned, nodes from the group's smallest atom to the log of the sum of its members'
        # largest, and one more.
        low = np.full(group_count, np.inf)
        np.minimum.at(low, groups, member_origin)
        high = np.full(group_count, -np.inf)
        np.logaddexp.at(high, groups, log_scale + grid[-1])
        # A pair's law is its K^2 atoms exactly where they are no more than its binned nodes; past that it is binned
        # like a larger group's (the sweep reads every atom of every law at every update), its leave-one-out laws exact.
        binned_size = np.ceil((high - low) / spacing).astype(np.int64) + 2
        exact_pair = (sizes == 2) & (node_count * node_count <= binned_size)
        binned = np.flatnonzero((sizes > 2) | ((sizes == 2) & ~exact_pair)).astype(np.int64)
        law_size = np.where(sizes == 1, node_count, np.where(exact_pair, node_count * node_count, binned_size))
        law_start = np.concatenate([[0], np.cumsum(law_size)]).astype(np.int64)
        law_log_variance = np.empty(law_start[-1])
        law_log_mass = np.empty(law_start[-1])
        loo_low = np.zeros(member_count)
        loo_spacing = np.full(member_count, grid_spacing)
        loo_stride = np.ones(member_count, dtype=np.int64)
        loo_size = np.zeros(member_count, dtype=np.int64)
        computes = (representative == np.arange(member_count)) & (member_size > 1)
        singles = np.flatnonzero(sizes == 1)
        single_members = member_index[member_start[singles]]
        single_atoms = law_start[singles][:, None] + np.arange(node_count)[None, :]
        law_log_variance[single_atoms] = member_origin[single_members][:, None] + grid_spacing * np.arange(node_count)[None, :]
        law_log_mass[single_atoms] = member_log_weight[single_members]
        exact_pairs = np.flatnonzero(exact_pair)
        first, second = member_index[member_start[exact_pairs]], member_index[member_start[exact_pairs] + 1]
        pair_atoms = law_start[exact_pairs][:, None] + np.arange(node_count * node_count)[None, :]
        first_variance = member_origin[first][:, None] + grid_spacing * np.arange(node_count)[None, :]
        second_variance = member_origin[second][:, None] + grid_spacing * np.arange(node_count)[None, :]
        law_log_variance[pair_atoms] = np.logaddexp(first_variance[:, :, None], second_variance[:, None, :]).reshape(exact_pairs.shape[0], node_count * node_count)
        law_log_mass[pair_atoms] = (member_log_weight[first][:, :, None] + member_log_weight[second][:, None, :]).reshape(exact_pairs.shape[0], node_count * node_count)
        pairs = np.flatnonzero(sizes == 2)
        first, second = member_index[member_start[pairs]], member_index[member_start[pairs] + 1]
        # A pair's leave-one-out law is the other member's atoms: on their grid, at the lattice's spacing.
        loo_low[first], loo_low[second] = member_origin[second], member_origin[first]
        loo_size[first[computes[first]]] = node_count
        loo_size[second[computes[second]]] = node_count
        larger_members = np.flatnonzero(member_size > 2)
        loo_low[larger_members] = low[groups[larger_members]]
        loo_spacing[larger_members] = spacing
        loo_stride[larger_members] = refinement
        larger_computes = larger_members[computes[larger_members]]
        loo_size[larger_computes] = law_size[groups[larger_computes]]
        loo_start = np.concatenate([[0], np.cumsum(loo_size)]).astype(np.int64)
        loo_log_mass = np.empty(loo_start[-1])
        paired = np.concatenate([first[computes[first]], second[computes[second]]])
        other = np.concatenate([second[computes[first]], first[computes[second]]])
        loo_log_mass[loo_start[paired][:, None] + np.arange(node_count)[None, :]] = member_log_weight[other]
        for group in binned:
            law_log_variance[law_start[group]:law_start[group + 1]] = low[group] + spacing * np.arange(law_size[group])
        if binned.shape[0]:
            _larger_laws(binned, sizes[binned] > 2, member_start, member_index, representative, member_origin, member_log_weight, refinement, spacing, low,
                         law_start, law_log_mass, loo_start, loo_log_mass, _BAND_WIDTH)
        return cls(
            groups=groups, member_start=member_start, member_index=member_index, representative=representative, law_start=law_start,
            law_log_variance=law_log_variance, law_log_mass=law_log_mass, member_origin=member_origin, member_log_weight=member_log_weight,
            grid_spacing=grid_spacing, loo_start=loo_start, loo_log_mass=loo_log_mass, loo_low=loo_low, loo_spacing=loo_spacing,
            loo_stride=loo_stride,
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
        finite = np.where(self.law_log_mass > -np.inf, self.law_log_variance, -np.inf)
        return np.maximum.reduceat(finite, self.law_start[:-1])


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


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _group_terms(law_start, law_log_variance, law_log_mass, member_start, member_index, representative, member_origin, member_log_weight,
                 grid_spacing, loo_start, loo_log_mass, loo_low, loo_spacing, loo_stride, precision, shift, log_normalizer, marginal,
                 scale_derivative, member_mean, member_second):
    """Per group at its cavity: log Z, and per member its node marginals, d log Z / d log u_j and its posterior mean and
    second moment, in place (module docstring)."""
    group_count = member_start.shape[0] - 1
    node_count = member_log_weight.shape[1]
    for group in numba.prange(group_count):
        P = precision[group]
        h = shift[group]
        start, stop = law_start[group], law_start[group + 1]
        peak = -np.inf
        for a in range(start, stop):
            if law_log_mass[a] > -np.inf:
                log_k, _first, _conditional = _log_kernel(law_log_variance[a], P, h)
                if law_log_mass[a] + log_k > peak:
                    peak = law_log_mass[a] + log_k
        total = 0.0
        for a in range(start, stop):
            if law_log_mass[a] > -np.inf:
                log_k, _first, _conditional = _log_kernel(law_log_variance[a], P, h)
                total += np.exp(law_log_mass[a] + log_k - peak)
        log_normalizer[group] = peak + np.log(total)
        size = member_start[group + 1] - member_start[group]
        node_terms = np.empty(node_count)
        share_first = np.zeros(node_count)
        mean_part = np.zeros(node_count)
        second_part = np.zeros(node_count)
        for position in range(member_start[group], member_start[group + 1]):
            j = member_index[position]
            if representative[j] != j:
                continue
            node_terms[:] = -np.inf
            if size == 1:
                for k in range(node_count):
                    if member_log_weight[j, k] == -np.inf:
                        continue
                    log_k, first, conditional = _log_kernel(member_origin[j] + grid_spacing * k, P, h)
                    node_terms[k] = member_log_weight[j, k] + log_k
                    share_first[k] = first
                    mean_part[k] = h * conditional
                    second_part[k] = conditional + h * h * conditional * conditional
            else:
                # Over the leave-one-out law W on its nodes, with the member's atom v_k: sum_W m(W) K(W + v_k) and the
                # share-weighted pieces as weighted means; where W_i + v_k lands and its shares depend on i - r k alone.
                law = loo_log_mass[loo_start[j]:loo_start[j + 1]]
                count = law.shape[0]
                spacing = loo_spacing[j]
                stride = loo_stride[j]
                origin = (member_origin[j] - loo_low[j]) / spacing
                offset, log_atom_share, log_law_share = _landing(origin, stride, node_count, count, spacing)
                base = stride * (node_count - 1)
                values = np.empty(count)
                firsts = np.empty(count)
                conditionals = np.empty(count)
                totals = np.empty(count)
                for k in range(node_count):
                    if member_log_weight[j, k] == -np.inf:
                        continue
                    inner_peak = -np.inf
                    for i in range(count):
                        values[i] = -np.inf
                        if law[i] == -np.inf:
                            continue
                        totals[i] = loo_low[j] + spacing * (i + offset[i - stride * k + base])
                        log_k, firsts[i], conditionals[i] = _log_kernel(totals[i], P, h)
                        values[i] = law[i] + log_k
                        if values[i] > inner_peak:
                            inner_peak = values[i]
                    if inner_peak == -np.inf:
                        continue
                    weight_sum = weighted_first = weighted_mean = weighted_second = 0.0
                    for i in range(count):
                        if values[i] == -np.inf:
                            continue
                        weight = np.exp(values[i] - inner_peak)
                        index = i - stride * k + base
                        log_total = totals[i]
                        first, conditional = firsts[i], conditionals[i]
                        share = np.exp(log_atom_share[index])
                        weight_sum += weight
                        weighted_first += weight * share * first
                        weighted_mean += weight * share * h * conditional
                        # E[gamma_j^2 | node, W] = v W / V + share^2 (c + h^2 c^2), v W / V = V s (1 - s) in logs.
                        allocation = np.exp(log_total + log_atom_share[index] + log_law_share[index])
                        weighted_second += weight * (allocation + share * share * (conditional + h * h * conditional * conditional))
                    node_terms[k] = member_log_weight[j, k] + inner_peak + np.log(weight_sum)
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
            derivative = mean_value = second_value = 0.0
            for k in range(node_count):
                probability = np.exp(node_terms[k] - peak) / norm if node_terms[k] > -np.inf else 0.0
                marginal[j, k] = probability
                derivative += probability * share_first[k]
                mean_value += probability * mean_part[k]
                second_value += probability * second_part[k]
            scale_derivative[j] = derivative
            member_mean[j] = mean_value
            member_second[j] = second_value
        for position in range(member_start[group], member_start[group + 1]):
            j = member_index[position]
            source = representative[j]
            if source != j:
                marginal[j] = marginal[source]
                scale_derivative[j] = scale_derivative[source]
                member_mean[j] = member_mean[source]
                member_second[j] = member_second[source]


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
    member_count, node_count = laws.member_log_weight.shape
    log_normalizer = np.empty(laws.group_count)
    marginal = np.zeros((member_count, node_count))
    scale_derivative = np.empty(member_count)
    mean = np.empty(member_count)
    second = np.empty(member_count)
    _group_terms(
        laws.law_start, laws.law_log_variance, laws.law_log_mass, laws.member_start, laws.member_index, laws.representative, laws.member_origin,
        laws.member_log_weight, laws.grid_spacing, laws.loo_start, laws.loo_log_mass, laws.loo_low, laws.loo_spacing, laws.loo_stride,
        np.asarray(cavity_precision, dtype=np.float64), np.asarray(cavity_shift, dtype=np.float64), log_normalizer, marginal, scale_derivative,
        mean, second,
    )
    return GroupTerms(log_normalizer=log_normalizer, marginal=marginal, scale_derivative=scale_derivative, mean=mean, variance=second - mean * mean)


def law_resolution(coarse: GroupLaws, fine: GroupLaws, cavity_precision: F64Array, cavity_shift: F64Array) -> float:
    """The larger groups' splitting error measured at their cavities: the largest change of any group's log Z between
    two refinements of its laws (the splitting's error falls as the spacing's square)."""
    return float(np.max(np.abs(group_terms(coarse, cavity_precision, cavity_shift).log_normalizer - group_terms(fine, cavity_precision, cavity_shift).log_normalizer)))

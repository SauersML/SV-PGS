"""Stage 2 in Gram space: the quantitative mean-field fit from Stage 0's banded Grams, at a cost that does not depend on n.

For a quantitative trait the likelihood of the covariate-projected problem depends on the data only through
G = Xp'Xp, s = Xp'y and y'Py (n' = n - rank C its residual dimension):

    -2 log p(y | beta, sigma^2) = n' log(2 pi sigma^2) + (y'Py - 2 beta's + beta'G beta) / sigma^2.

Stage 0 streams the genotypes once and stores exactly these in float32 (G) and float64 (s, y'Py): G within each LD
block, R_b, and between genome-adjacent blocks, R_{b,b+1} (``genotype_statistics.LdGramStore``). The mean-field fixed
point (``full_data_fit._FullDataMeanField``, ``mean_field``) reads the data only through the fields
c = s - G mbar (mbar the groups' signed mean sums, ``tie_members``) and the residual's square
||y_P - Xp mbar||^2 = y'Py - 2 mbar's + mbar'G mbar, and its responses only through products with G. So with G_band, the
block-tridiagonal part of G that Stage 0 stores, every sweep, product and solve here costs O(sum_b |b| |band b|) and
reads the band once, never the n samples:

- the sweep (``GramBand.sweep``): at block b, c_b = s_b - R_{b-1,b}' mbar_{b-1} - R_b mbar_b - R_{b,b+1} mbar_{b+1} from
  the current means, then each member's update in the sweep's order, c_b -= R_b[:, g] dm_g after each (exactly
  x_g'r after x_g dm_g has left r), and mbar_b' R_b mbar_b + 2 mbar_{b-1}' R_{b-1,b} mbar_b are added to mbar'G mbar
  once the block's means are final. Where G_band = G (the band holds all the LD) the sweep is the n-space sweep's
  coordinate ascent, update for update, to rounding.

The far field. Whatever LD lies beyond the band (G - G_band: blocks two apart, and other chromosomes) is left out of
the band. True LD between blocks two apart is what Stage 0's partition cuts at the LD's minimum, and its reach is the
data's (``marginal_variances.ld_extent``). Chance LD does not vanish: an unlinked pair has r^2 of mean 1/n, so the far
field adds to each field c_j a term sum_{k beyond j's band} G_jk E beta_k, of variance G_jj (1/n) sum_k G_kk E beta_k^2
(``far_field_ratio``): relative to the noise's own G_jj sigma^2 that is the far variants' share of the genetic variance
over sigma^2, whatever n is (0.71 on the two-chromosome test store at h2 = 0.5, where a fit on the band alone ended
uncertified [own-sim]). So the band is an accelerator only (``full_data_fit._GramMeanField`` with the dual solver):
the band's sweeps with the far field corrected from the samples by iterative refinement, a few reads of the store per
solve in place of one per sweep, and the dual solver's solves. The far field's size at the fitted state is logged
(``far_field_ratio``).

Memory. The band is read block by block from Stage 0's files, never mapped whole (``LdGramStore.read_gram``); each
block read is charged to the shared ledger (``memory_broker``) while it lives, and a block is kept for later sweeps
and products only as a cache lease of what the ledger has left (on the device and on the host), so the band's size
never enters the job's resident set beyond what the ledger grants.
"""

from __future__ import annotations

import json
import time
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numba
import numpy as np

from sv_pgs._typing import F64Array, I64Array
from sv_pgs.memory_broker import HOST, current_broker, device_pool, transient_device_arrays
from sv_pgs.phase_timer import GB, timing
from sv_pgs.progress import log

_EPSILON = float(np.finfo(np.float64).eps)
_FLOAT_BYTES = np.dtype(np.float64).itemsize


def _host(values: Any) -> np.ndarray:
    return values.get() if hasattr(values, "get") else np.asarray(values)


@numba.njit(cache=True, fastmath={"reassoc", "contract"})
def _tilted(h, omega, row, member, log_density, node_variance, log_node_variance, log_weights, conditional):
    """Member ``member``'s q_j at precision ``omega`` and shift ``h``: its prior's scale mixture times exp(h b - omega
    b^2 / 2) over the lattice, as (log normalizer, mean, variance, third and fourth central moments); ``log_weights``
    and ``conditional`` are scratch (the nodes' weights and conditional variances)."""
    node_count = node_variance.shape[1]
    peak = -np.inf
    for node in range(node_count):
        variance_node = node_variance[member, node]
        ratio = variance_node * omega
        if ratio == np.inf:
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
    new_variance = 0.0
    new_third = 0.0
    new_fourth = 0.0
    for node in range(node_count):
        offset = h * conditional[node] - new_mean
        weight = log_weights[node] / total
        new_variance += weight * (conditional[node] + offset * offset)
        new_third += weight * (offset * offset * offset + 3.0 * conditional[node] * offset)
        new_fourth += weight * (offset**4 + 6.0 * conditional[node] * offset * offset + 3.0 * conditional[node] * conditional[node])
    return log_normalizer, new_mean, new_variance, new_third, new_fourth


@numba.njit(cache=True, fastmath={"reassoc", "contract"})
def _proximal_shift(h, omega, kappa, target, row, member, log_density, node_variance, log_node_variance, log_weights, conditional):
    """The shift of member ``member``'s coordinate update under the proximal term -(kappa / 2) (m_j - target)^2 on its
    mean m_j = E_q b_j (``GramBand.sweep``'s): q_j is then its law at shift h - kappa (m_j - target), so m_j solves
    m = M(h - kappa (m - target)), M the law's mean as a function of its shift. g(m) = m - M(...) rises with slope at
    least one (M rises with the shift, by the law's variance), so its root lies within |g(m)| of any m: Newton's steps
    inside that bracket, halving it where a step leaves it, until no double lies strictly inside it."""
    mean = target
    moments = _tilted(h - kappa * (mean - target), omega, row, member, log_density, node_variance, log_node_variance, log_weights, conditional)
    gap = mean - moments[1]
    lower, upper = (mean - gap, mean) if gap > 0.0 else (mean, mean - gap)
    while gap != 0.0:
        candidate = mean - gap / (1.0 + kappa * moments[2])
        if not lower < candidate < upper:
            candidate = 0.5 * (lower + upper)
        if candidate == lower or candidate == upper:
            break
        mean = candidate
        moments = _tilted(h - kappa * (mean - target), omega, row, member, log_density, node_variance, log_node_variance, log_weights, conditional)
        gap = mean - moments[1]
        if gap > 0.0:
            upper = mean
        else:
            lower = mean
    return h - kappa * (mean - target)


@numba.njit(cache=True, fastmath={"reassoc", "contract"})
def _gram_sweep(
    gram, field, groups, signs, squares, class_index, log_density, node_variance, log_node_variance, noise, mean, variance, shift, third, fourth,
    moved, proximal, anchor,
):
    """One block's coordinate ascent in Gram space, in place, the members in the order given.

    ``gram`` is the block's Gram R_b (w x w, float32 as Stage 0 stores it, or float64), ``field`` c_b = x_g' r over the
    block's groups (w), ``groups[j]`` member j's group within the block and ``signs[j]`` its tie sign, ``squares`` the
    groups' ||x_g||^2 (R_b's diagonal), the per-member arrays (class, node variances and moments) in sweep order, and
    ``moved`` (w) receives each group's signed mean change. Member j's update is ``mean_field._sweep``'s, with its
    projection x_j'r = s_j c_g; after it, c -= R_b[g, :] s_j dm_j over the block (R_b symmetric, so its row g is its
    column g). With ``proximal`` lambda > 0 each update ascends the ELBO less (lambda / 2 sigma^2) ||mbar - a||^2 on the
    groups' signed mean sums, ``anchor`` a - mbar at the block's start per group (``_proximal_shift``). Returns
    (sum_j KL(q_j || p_j), sum_j ||x_j||^2 v_j, the KL terms' pieces' sizes)."""
    width = field.shape[0]
    member_count = groups.shape[0]
    node_count = node_variance.shape[1]
    divergence = 0.0
    weighted_variance = 0.0
    sizes = 0.0
    log_weights = np.empty(node_count)
    conditional = np.empty(node_count)
    for member in range(member_count):
        group = groups[member]
        sign = signs[member]
        omega = squares[group] / noise
        old_mean = mean[member]
        h = (sign * field[group] + squares[group] * old_mean) / noise
        row = class_index[member]
        if proximal > 0.0:
            # The member's mean that puts its group's sum at the anchor, the others' held: the term's target.
            h = _proximal_shift(
                h, omega, proximal / noise, old_mean + sign * (anchor[group] - moved[group]), row, member, log_density, node_variance,
                log_node_variance, log_weights, conditional,
            )
        log_normalizer, new_mean, new_variance, new_third, new_fourth = _tilted(
            h, omega, row, member, log_density, node_variance, log_node_variance, log_weights, conditional,
        )
        step = new_mean - old_mean
        if step != 0.0:
            signed_step = sign * step
            moved[group] += signed_step
            for index in range(width):
                field[index] -= gram[group, index] * signed_step
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
    return divergence, weighted_variance, sizes


@dataclass
class SweepResult:
    """A Gram-space sweep's ELBO pieces: as ``mean_field._sweep``'s, with ``residual_size`` the size of the residual
    square's own terms, y'Py + 2 |mbar|'|s| + |mbar|'|G_band||mbar| (its rounding bound's scale)."""

    divergence: float
    weighted_variance: float
    residual_square: float
    sizes: float
    residual_size: float


class GramBand:
    """Stage 0's banded Gram, read block by block on demand: R_b within each block and R_{b,b+1} between genome-adjacent
    blocks (float32, as stored), the groups' squares ||x_g||^2 (R's diagonal, float32-exact) and the covariates'
    cross-products X'C, which every target of Stage 0 shares, and per target (a column each) the projected scores s
    (float64, groups x targets), y'Py and C'y for the covariate coefficients.

    ``working_bytes`` sizes the float64 row chunks a stored block is promoted in for a product. Each block read is
    charged to the shared ledger for as long as its array lives, and kept for later passes only as a ledger cache
    (the device's when the fit runs on one, else the host's), admitted from what the ledger has left."""

    def __init__(self, statistics: Any, working_bytes: int, store: Any = None) -> None:
        # The Grams from ``store`` where given (a ``BandStore``: the summary band's parts, in the reduced columns' order),
        # else Stage 0's own blocks; the scores, squares and covariate products are Stage 0's either way.
        ld = statistics.ld
        grams = ld if store is None else store
        self.ld = grams
        self.block_count = int(grams.block_count)
        widths = np.array([grams.block_width(block) for block in range(self.block_count)], dtype=np.int64)
        self.starts = np.concatenate([[0], np.cumsum(widths)]).astype(np.int64)
        self.group_count = int(self.starts[-1])
        # linked[b]: a stored Gram between block b and block b + 1 (none across a chromosome's end).
        self.linked = np.array([grams.has_adjacent(block + 1) for block in range(self.block_count - 1)], dtype=bool)
        self.sample_count = int(statistics.sample_count)
        self.residual_dimension = float(statistics.sample_count - statistics.covariate_rank)
        self.working_bytes = int(working_bytes)
        blocks = [ld.block(block) for block in range(int(ld.block_count))]
        target_count = int(np.shape(statistics.target_gram)[0])
        self.scores = np.concatenate([np.asarray(block.projected_score, dtype=np.float64) for block in blocks]) if blocks else np.zeros((0, target_count))
        self.covariate_cross = np.concatenate([np.asarray(block.covariate_cross, dtype=np.float64) for block in blocks], axis=0)
        del blocks
        # R_jj stored in float32, kept as R_jj / n in float64: n times it rounds back to the stored float32 exactly (its
        # relative error, two float64 roundings, is far below float32's half unit).
        self.squares = (np.asarray(ld.ld_diagonal(), dtype=np.float64) * statistics.sample_count).astype(np.float32).astype(np.float64)
        covariate_target = np.asarray(statistics.covariate_target, dtype=np.float64)
        self.covariate_target = covariate_target
        self.covariate_pseudo_inverse = np.asarray(statistics.covariate_gram_pseudo_inverse, dtype=np.float64)
        target_gram = np.asarray(statistics.target_gram, dtype=np.float64)
        self.target_square = np.array([
            float(target_gram[target, target] - covariate_target[:, target] @ self.covariate_pseudo_inverse @ covariate_target[:, target])
            for target in range(target_count)
        ])
        self.band_width = max(
            (int(widths[block]) + (int(widths[block - 1]) if block and self.linked[block - 1] else 0)
             + (int(widths[block + 1]) if block + 1 < self.block_count and self.linked[block] else 0)
             for block in range(self.block_count)), default=0,
        )
        self._arrays: dict | None = None
        self._broker = current_broker()
        self._cache: dict[tuple, tuple[Any, Any]] = {}
        self.reads = 0
        self.read_bytes = 0
        self._unscaled()

    def _unscaled(self) -> None:
        """The far field's measured scale unset (1: chance's own 1/n per pair; ``far_field_scale`` sets the store's)."""
        self.far_scale = 1.0

    @classmethod
    def from_arrays(
        cls, within: list, cross: list, scores: F64Array, target_square: float, sample_count: int, residual_dimension: float, working_bytes: int,
        covariate_cross: F64Array | None = None, covariate_target: F64Array | None = None, covariate_pseudo_inverse: F64Array | None = None,
    ) -> "GramBand":
        """A band held in memory: ``within[b]`` R_b and ``cross[b]`` R_{b,b+1} (None where no Gram links the blocks),
        the rest as ``__init__`` reads it from Stage 0 for one target (tests, and callers with their own Grams)."""
        band = cls.__new__(cls)
        widths = np.array([np.shape(values)[0] for values in within], dtype=np.int64)
        band.ld = None
        band.block_count = len(within)
        band.starts = np.concatenate([[0], np.cumsum(widths)]).astype(np.int64)
        band.group_count = int(band.starts[-1])
        band.linked = np.array([values is not None for values in cross], dtype=bool)
        band.sample_count = int(sample_count)
        band.residual_dimension = float(residual_dimension)
        band.working_bytes = int(working_bytes)
        band.scores = np.asarray(scores, dtype=np.float64)[:, None]
        band.squares = np.concatenate([np.asarray(_host(values.diagonal()), dtype=np.float64) for values in within])
        band.target_square = np.array([target_square], dtype=np.float64)
        band.covariate_cross = np.zeros((band.group_count, 0)) if covariate_cross is None else np.asarray(covariate_cross, dtype=np.float64)
        band.covariate_target = np.zeros((0, 1)) if covariate_target is None else np.asarray(covariate_target, dtype=np.float64)[:, None]
        band.covariate_pseudo_inverse = np.zeros((0, 0)) if covariate_pseudo_inverse is None else np.asarray(covariate_pseudo_inverse, dtype=np.float64)
        band.band_width = max(
            (int(widths[block]) + (int(widths[block - 1]) if block and band.linked[block - 1] else 0)
             + (int(widths[block + 1]) if block + 1 < band.block_count and band.linked[block] else 0)
             for block in range(band.block_count)), default=0,
        )
        band._arrays = {("within", block): values for block, values in enumerate(within)}
        band._arrays.update({("cross", block): values for block, values in enumerate(cross) if values is not None})
        band._broker = current_broker()
        band._cache = {}
        band.reads = 0
        band.read_bytes = 0
        band._unscaled()
        return band

    # blocks

    def span(self, block: int) -> slice:
        return slice(int(self.starts[block]), int(self.starts[block + 1]))

    def within(self, block: int, array_module: Any = np) -> Any:
        """R_b (float32) on ``array_module``."""
        return self._block(("within", block), array_module)

    def cross(self, block: int, array_module: Any = np) -> Any:
        """R_{b,b+1} (float32, |b| x |b+1|) on ``array_module``, or None where no Gram links the two blocks."""
        if block < 0 or block + 1 >= self.block_count or not self.linked[block]:
            return None
        return self._block(("cross", block), array_module)

    def _read(self, key: tuple) -> np.ndarray:
        kind, block = key
        if self._arrays is not None:
            return self._arrays[key]
        values = read_block(self.ld, kind, block)
        self.reads += 1
        self.read_bytes += int(values.nbytes)
        return values

    def _block(self, key: tuple, array_module: Any) -> Any:
        on_device = array_module is not np
        place = (key, "device" if on_device else "host")
        held = self._cache.get(place)
        if held is not None:
            return held[0]
        host = self._cache.get((key, "host"))
        values = host[0] if host is not None else self._read(key)
        if host is None and self._arrays is None:
            # Kept on the host as well where the fit runs on a device: a block the device cannot hold is then uploaded
            # from memory at the next pass, not read from the disk again.
            self._keep((key, "host"), values, HOST)
        if not on_device:
            return values
        device = array_module.asarray(values)
        pool = device_pool(int(array_module.cuda.runtime.getDevice()))
        if self._broker is not None and pool in self._broker.meters:
            self._keep(place, device, pool, allocated=True)
        return device

    def _keep(self, place: tuple, values: Any, pool: str, allocated: bool = False) -> None:
        if self._broker is None:
            return
        lease = self._broker.admit(pool, int(values.nbytes), "a cached Stage 0 Gram block", lambda place=place: self._cache.pop(place, None), allocated=allocated)
        if lease is not None:
            self._cache[place] = (values, lease)

    def release(self) -> None:
        for _values, lease in self._cache.values():
            lease.release()
        self._cache.clear()

    # products

    def _chunk_rows(self, columns: int, xp: Any = np, copies: int = 1) -> int:
        """Rows of a stored block promoted at once: ``copies`` float64 copies of them (the rows, and with the
        magnitudes their absolute values) within the fit's working set, and on a device within what the ledger can
        grant there (``memory_broker``: its capacity less every byte the device holds, with the idle caches counted as
        reclaimable, since the allocator evicts them before it refuses)."""
        available = self.working_bytes
        if xp is not np and self._broker is not None:
            pool = device_pool(int(xp.cuda.runtime.getDevice()))
            if pool in self._broker.meters:
                available = min(available, self._broker.reclaimable(pool))
        return max(1, available // max(1, _FLOAT_BYTES * columns * copies))

    def _times(self, matrix: Any, values: Any, xp: Any, transposed: bool = False, absolute: bool = False) -> Any:
        """matrix @ values (or matrix' @ values) in float64, the stored float32 promoted a row chunk at a time; with
        ``absolute``, also |matrix| @ |values|."""
        rows, columns = int(matrix.shape[0]), int(matrix.shape[1])
        out = xp.zeros((columns if transposed else rows,) + tuple(values.shape[1:]))
        magnitude = xp.zeros_like(out) if absolute else None
        step = self._chunk_rows(columns, xp, 2 if absolute else 1)
        for first in range(0, rows, step):
            last = min(first + step, rows)
            # The promoted rows (and their magnitudes) are transient: reserved on their own and returned when freed.
            with transient_device_arrays(xp):
                chunk = xp.asarray(matrix[first:last], dtype=xp.float64)
                sizes = xp.abs(chunk) if absolute else None
            if transposed:
                out += chunk.T @ values[first:last]
                if absolute:
                    magnitude += sizes.T @ xp.abs(values[first:last])
            else:
                out[first:last] = chunk @ values
                if absolute:
                    magnitude[first:last] = sizes @ xp.abs(values)
            del chunk, sizes
        return (out, magnitude) if absolute else out

    def product(self, values: Any, array_module: Any = np) -> Any:
        """G_band @ values for values over the groups (groups x r, float64), one read of the band."""
        xp = array_module
        values = xp.asarray(values, dtype=xp.float64)
        out = xp.zeros_like(values)
        for block in range(self.block_count):
            own = self.span(block)
            out[own] += self._times(self.within(block, xp), values[own], xp)
            cross = self.cross(block, xp)
            if cross is not None:
                following = self.span(block + 1)
                out[own] += self._times(cross, values[following], xp)
                out[following] += self._times(cross, values[own], xp, transposed=True)
        return out

    def quadratic(self, values: F64Array) -> float:
        return float(values @ _host(self.product(values[:, None]))[:, 0])

    # the sweep

    def sweep(
        self, xp: Any, *, member_blocks: tuple, group: I64Array, sign: F64Array, class_index: I64Array, log_density: F64Array,
        scales: F64Array, grid: F64Array, noise: float, mean: F64Array, variance: F64Array, shift: F64Array, third: F64Array,
        fourth: F64Array, field_shift: F64Array | None = None, target: int = 0, proximal: float = 0.0, anchor: F64Array | None = None,
    ) -> SweepResult:
        """One coordinate-ascent sweep of Stage 0's target ``target`` over every block's members in ``member_blocks``'
        order, in place on the members' moments (host arrays), on ``xp`` (numpy, or cupy with ``device_sweep``'s panel
        kernel). ``field_shift`` (groups) is added to every field and to the residual square's linear term,
        s -> s + field_shift: the far field held fixed through a round of the exact refinement
        (``full_data_fit._GramMeanField``). ``proximal`` lambda > 0 ascends the ELBO less (lambda / 2 sigma^2)
        ||mbar - anchor||^2 on the groups' signed mean sums instead (that round's minorizer), on the host kernel
        (``_gram_sweep``; the returned residual square is the band's, without the term)."""
        if len(member_blocks) != self.block_count or any(
            members.size and not (self.starts[block] <= int(group[members].min()) and int(group[members].max()) < self.starts[block + 1])
            for block, members in enumerate(member_blocks)
        ):
            # The kernels index a block's fields by its members' groups unchecked: a member outside its block would write
            # past the arrays.
            raise ValueError("every member of a sweep block must be one of the band's block's groups")
        grouped = np.zeros(self.group_count)
        np.add.at(grouped, group, sign * mean)
        scores = self.scores[:, target]
        self._linear = scores if field_shift is None else scores + np.asarray(field_shift, dtype=np.float64)
        self._target_square = float(self.target_square[target])
        if xp is np or proximal > 0.0:
            # Each group's anchor less its sum at the sweep's start, which each block's kernel reads at its own start.
            offsets = np.zeros(self.group_count) if anchor is None else np.asarray(anchor, dtype=np.float64) - grouped
            return self._host_sweep(
                member_blocks, group, sign, class_index, log_density, scales, grid, noise, mean, variance, shift, third, fourth, grouped,
                proximal, offsets,
            )
        return self._device_sweep(xp, member_blocks, group, sign, class_index, log_density, scales, grid, noise, mean, variance, shift, third, fourth, grouped)

    def _block_field(self, block: int, grouped: Any, xp: Any, within: Any) -> Any:
        """c_b = s_b - R_{b-1,b}' mbar_{b-1} - R_b mbar_b - R_{b,b+1} mbar_{b+1} at the current means."""
        own = self.span(block)
        field = xp.asarray(self._linear[own]) - self._times(within, grouped[own][:, None], xp)[:, 0]
        previous = self.cross(block - 1, xp)
        if previous is not None:
            field -= self._times(previous, grouped[self.span(block - 1)][:, None], xp, transposed=True)[:, 0]
        following = self.cross(block, xp)
        if following is not None:
            field -= self._times(following, grouped[self.span(block + 1)][:, None], xp)[:, 0]
        return field

    def _block_quadratic(self, block: int, grouped: Any, xp: Any, within: Any) -> tuple[float, float]:
        """mbar_b' R_b mbar_b + 2 mbar_{b-1}' R_{b-1,b} mbar_b once both blocks' means are final, and the same with
        absolute values (the terms' size)."""
        own = self.span(block)
        values = grouped[own][:, None]
        image, magnitude = self._times(within, values, xp, absolute=True)
        total = float(_host(values[:, 0] @ image[:, 0]))
        size = float(_host(xp.abs(values[:, 0]) @ magnitude[:, 0]))
        previous = self.cross(block - 1, xp)
        if previous is not None:
            before = grouped[self.span(block - 1)][:, None]
            image, magnitude = self._times(previous, values, xp, absolute=True)
            total += 2.0 * float(_host(before[:, 0] @ image[:, 0]))
            size += 2.0 * float(_host(xp.abs(before[:, 0]) @ magnitude[:, 0]))
        return total, size

    def _finish(self, divergence: float, weighted_variance: float, sizes: float, grouped: np.ndarray, quadratic: float, quadratic_size: float) -> SweepResult:
        linear = float(grouped @ self._linear)
        residual_square = self._target_square - 2.0 * linear + quadratic
        residual_size = abs(self._target_square) + 2.0 * float(np.abs(grouped) @ np.abs(self._linear)) + quadratic_size
        return SweepResult(divergence, weighted_variance, residual_square, sizes, residual_size)

    def _host_sweep(
        self, member_blocks, group, sign, class_index, log_density, scales, grid, noise, mean, variance, shift, third, fourth, grouped, proximal, offsets,
    ) -> SweepResult:
        divergence = weighted_variance = sizes = quadratic = quadratic_size = 0.0
        for block, members in enumerate(member_blocks):
            own = self.span(block)
            within = self.within(block)
            field = self._block_field(block, grouped, np, within)
            log_node_variance = np.ascontiguousarray(scales[members][:, None] + grid[None, :])
            with np.errstate(over="ignore"):
                node_variance = np.exp(log_node_variance)
            state = [np.ascontiguousarray(values[members]) for values in (mean, variance, shift, third, fourth)]
            moved = np.zeros(own.stop - own.start)
            # A block's groups are its own, so their sums at its start are the sweep's start's (``offsets``).
            part = _gram_sweep(
                within, field, np.ascontiguousarray(group[members] - own.start), np.ascontiguousarray(sign[members]),
                np.ascontiguousarray(self.squares[own]), np.ascontiguousarray(class_index[members]), log_density, node_variance,
                log_node_variance, float(noise), *state, moved, float(proximal), np.ascontiguousarray(offsets[own]),
            )
            for values, piece in zip((mean, variance, shift, third, fourth), state):
                values[members] = piece
            grouped[own] += moved
            divergence += part[0]
            weighted_variance += part[1]
            sizes += part[2]
            terms, size = self._block_quadratic(block, grouped, np, within)
            quadratic += terms
            quadratic_size += size
            del within, field
        return self._finish(divergence, weighted_variance, sizes, grouped, quadratic, quadratic_size)

    def _device_sweep(self, cupy, member_blocks, group, sign, class_index, log_density, scales, grid, noise, mean, variance, shift, third, fourth, grouped) -> SweepResult:
        import cupyx

        from sv_pgs.device_sweep import PANEL, PIECE_COLUMNS, _kernel, step_threads

        kernel = _kernel(cupy)
        log_density_device = cupy.asarray(np.ascontiguousarray(log_density))
        node_count = int(grid.shape[0])
        # The step kernel's per-(member, node) scratch (``device_sweep.sweep_piece``): one panel's worth each.
        scratch = [cupy.empty(PANEL * node_count, dtype=cupy.float64) for _name in ("conditional", "base", "weight")]
        device_grouped = cupy.asarray(grouped)
        squares = cupy.asarray(self.squares)
        divergence = weighted_variance = sizes = quadratic = quadratic_size = 0.0
        for block, members in enumerate(member_blocks):
            own = self.span(block)
            within = self.within(block, cupy)
            field = self._block_field(block, device_grouped, cupy, within)
            local = cupy.asarray(group[members] - own.start)
            signs = cupy.asarray(sign[members])
            log_node_variance = cupy.ascontiguousarray(cupy.asarray(scales[members])[:, None] + cupy.asarray(grid)[None, :])
            node_variance = cupy.exp(log_node_variance)
            state = [cupy.asarray(np.ascontiguousarray(values[members])) for values in (mean, variance, shift, third, fourth)]
            member_squares = squares[own][local]
            classes = cupy.asarray(np.ascontiguousarray(class_index[members]))
            pieces = cupy.zeros((members.shape[0], PIECE_COLUMNS))
            moved = cupy.zeros(own.stop - own.start)
            for first in range(0, members.shape[0], PANEL):
                last = min(first + PANEL, members.shape[0])
                rows, panel_signs = local[first:last], signs[first:last]
                # The panel's own Gram in the members' signs, and their fields: the kernel's c_P and Xp_P'Xp_P.
                gram_rows = cupy.asarray(within[rows], dtype=cupy.float64)
                panel_gram = cupy.ascontiguousarray(gram_rows[:, rows] * panel_signs[:, None] * panel_signs[None, :])
                projection = cupy.ascontiguousarray(field[rows] * panel_signs)
                step = cupy.empty(last - first, dtype=cupy.float64)
                kernel(
                    (1,), (step_threads(kernel),),
                    (
                        panel_gram, projection, cupy.ascontiguousarray(member_squares[first:last]), classes[first:last], log_density_device,
                        np.int32(node_count), node_variance[first:last], log_node_variance[first:last], np.float64(float(noise)),
                        np.int32(last - first),
                        state[0][first:last], state[1][first:last], state[2][first:last], state[3][first:last], state[4][first:last], step, pieces[first:last],
                        *scratch,
                    ),
                )
                signed_step = step * panel_signs
                # c_b -= R_b[:, g_P] s_P dm_P: R_b's rows g_P (its columns, R_b symmetric), transposed.
                field -= gram_rows.T @ signed_step
                cupyx.scatter_add(moved, rows, signed_step)
                del gram_rows
            for values, piece in zip((mean, variance, shift, third, fourth), state):
                values[members] = cupy.asnumpy(piece)
            device_grouped[own] += moved
            totals = cupy.asnumpy(pieces.sum(axis=0))
            divergence += float(totals[0])
            weighted_variance += float(totals[1])
            sizes += float(totals[2])
            terms, size = self._block_quadratic(block, device_grouped, cupy, within)
            quadratic += terms
            quadratic_size += size
            del within, field
        return self._finish(divergence, weighted_variance, sizes, cupy.asnumpy(device_grouped), quadratic, quadratic_size)

    # the far field

    def far_variances(self, second_moments: F64Array) -> F64Array:
        """phi_b = (kappa / n') sum_{k beyond b's band} ||x_k||^2 E beta_k^2 per block: the variance the LD beyond the
        band adds to each field of block b per unit of its ||x_j||^2, at E beta_g^2 per group. A pair beyond the band
        has mean r^2 kappa / n' (``far_field_scale``: kappa measured on the store, 1 for chance alone), so the fields'
        far terms sum_k G_jk E beta_k have that variance."""
        load = self.squares * np.asarray(second_moments, dtype=np.float64) * float(self.far_scale) / self.residual_dimension
        per_block = np.array([float(np.sum(load[self.span(block)])) for block in range(self.block_count)])
        total = float(per_block.sum())
        near = per_block.copy()
        near[1:] += np.where(self.linked, per_block[:-1], 0.0)
        near[:-1] += np.where(self.linked, per_block[1:], 0.0)
        return np.maximum(total - near, 0.0)

    def far_field_ratio(self, second_moments: F64Array, noise: float) -> float:
        """max_b phi_b / sigma^2 (``far_variances``): the far field's variance per unit of a field's own noise."""
        return float(np.max(self.far_variances(second_moments), initial=0.0)) / float(noise)


def read_block(ld: Any, kind: str, block: int) -> np.ndarray:
    """A stored Gram read from Stage 0's files (``LdGramStore.read_gram``, not mapped): "within", R_b, or "cross",
    R_{b,b+1}; its bytes charged to the current ledger for as long as the array lives."""
    shape = (ld.block_width(block),) * 2 if kind == "within" else (ld.block_width(block), ld.block_width(block + 1))
    broker = current_broker()
    values = np.empty(shape, dtype=np.float32)
    if broker is not None:
        lease = broker.reserve(HOST, int(values.nbytes), "a Stage 0 Gram block")
        weakref.finalize(values, lease.release)
    return ld.read_gram(block, values) if kind == "within" else ld.read_adjacent(block + 1, values)


def pass_costs(band: GramBand, source: Any, array_module: Any) -> tuple[float, float]:
    """Seconds per pass over the design in sample space (the store's codes: ``dual_solve``'s tiles) and in Gram space
    (the band), each measured on the first LD block and scaled to every block: its read by the bytes (n x |b| codes;
    |b| (|b| + |b+1|) float32 Grams) and its products by the work (n |b|; |b| (|b| + 2 |b+1|)). Every sweep, product
    and solve of the fit is a whole number of such passes in either space, so the cheaper pass is the cheaper fit. The
    samples' read counts only where the devices cannot hold the codes (``holds_codes``): a held set is read from the
    store once, to fill it, and every later pass is its products alone. Charged a read per pass, an All of Us chr22
    fit's sample pass measured 7 s where its sweeps took 0.4 s [real], and at the whole genome the band route chosen
    on that count read an 82 GB band the devices could not hold at every sweep."""
    xp = array_module

    def synchronized() -> float:
        if xp is not np:
            xp.cuda.Stream.null.synchronize()
        return time.perf_counter()

    own = band.span(0)
    width = own.stop - own.start
    following = band.cross(0, np) is not None
    started = synchronized()
    within = band.within(0, xp)
    cross = band.cross(0, xp)
    gram_read = synchronized() - started
    values = xp.ones((band.group_count, 1))

    def gram_products() -> None:
        band._times(within, values[own], xp)
        if cross is not None:
            later = band.span(1)
            band._times(cross, values[later], xp)
            band._times(cross, values[own], xp, transposed=True)

    gram_products()
    started = synchronized()
    gram_products()
    gram_compute = synchronized() - started
    widths = np.diff(band.starts)
    next_widths = np.concatenate([widths[1:], [0]]) * np.concatenate([band.linked, [False]])
    read_total = float(np.sum(widths * (widths + next_widths)))
    work_total = float(np.sum(widths * (widths + 2 * next_widths)))
    first_next = float(next_widths[0]) if following else 0.0
    gram_seconds = gram_read * read_total / (width * (width + first_next)) + gram_compute * work_total / (width * (width + 2.0 * first_next))
    del within, cross

    blocks = iter(source.blocks())
    started = synchronized()
    start, stop, tile = next(blocks)
    sample_read = synchronized() - started
    left = xp.ones((int(source.sample_count), 1))
    right = xp.ones((stop - start, 1))
    tile.rmatmat(left)
    tile.matmat(right)
    started = synchronized()
    tile.rmatmat(left)
    tile.matmat(right)
    sample_compute = synchronized() - started
    held = bool(source.holds_codes())
    close = getattr(blocks, "close", None)
    if close is not None:
        close()
    sample_seconds = ((0.0 if held else sample_read) + sample_compute) * float(band.group_count) / float(stop - start)
    return sample_seconds, gram_seconds


_BAND_INDEX = "band.json"
_BAND_FILES = {"within": "band_within.f32", "cross": "band_cross.f32"}


class BandStore:
    """The summary band on disk: Stage 0's Grams re-cut into parts no wider than their region's LD extent, each part's
    Gram and each part's Gram with the next (``build_band_store``), float32. It reads as ``LdGramStore`` does for
    ``GramBand`` (``block_width``, ``has_adjacent``, ``read_gram``, ``read_adjacent``), the parts in the reduced columns'
    order, so a ``GramBand`` over it indexes the same groups."""

    def __init__(self, directory: Path) -> None:
        from sv_pgs.genotype_statistics import read_array

        self.directory = Path(directory)
        index = json.loads((self.directory / _BAND_INDEX).read_text(encoding="utf-8"))
        self.widths = np.asarray(index["widths"], dtype=np.int64)
        self.within_offsets = np.asarray(index["within_offsets"], dtype=np.int64)
        # cross_offsets[b]: the Gram of part b - 1 against part b, or -1 where none (a chromosome's first part).
        self.cross_offsets = np.asarray(index["cross_offsets"], dtype=np.int64)
        self.extents = index["extents"]
        self.stored_bytes = int(index["stored_bytes"])
        self._read = read_array

    @property
    def block_count(self) -> int:
        return int(self.widths.shape[0])

    def block_width(self, block: int) -> int:
        return int(self.widths[block])

    def has_adjacent(self, block: int) -> bool:
        return 0 < block < self.block_count and int(self.cross_offsets[block]) >= 0

    def read_gram(self, block: int, out: np.ndarray | None = None) -> np.ndarray:
        width = self.block_width(block)
        return self._read(self.directory / _BAND_FILES["within"], int(self.within_offsets[block]), (width, width), np.dtype(np.float32), out)

    def read_adjacent(self, block: int, out: np.ndarray | None = None) -> np.ndarray:
        shape = (self.block_width(block - 1), self.block_width(block))
        return self._read(self.directory / _BAND_FILES["cross"], int(self.cross_offsets[block]), shape, np.dtype(np.float32), out)


def build_band_store(
    ld: Any, sample_count: int, directory: Path, working_bytes: int, level: float, array_module: Any = np, null_scale: float = 1.0,
) -> BandStore:
    """Stage 0's Grams (``ld``, an ``LdGramStore``) as the summary band: each Stage 0 block's LD extent measured on its
    own Gram (``marginal_variances.ld_extent``, the whole tail's test within the block: the lag beyond which its pairs'
    excess r^2 is within its null resolution), the block cut into near-equal consecutive parts no wider than it, and
    each part's Gram with itself and with the next part written once (across a block's end from Stage 0's adjacent
    Gram; none across a chromosome's end). Every pair closer than its block's extent is then within one part or two
    neighbouring ones, so the band holds the LD the data show and 2 p w float32 entries in place of Stage 0's sum of
    squared block widths. An extent is never capped: a region whose LD reaches far (admixture's local-ancestry tracts,
    an inversion) keeps its wide parts, and each block's extent is logged. ``null_scale`` is kappa, the unlinked pairs'
    measured mean r^2 n (``far_field_scale``): each extent is read against kappa / n (``extent_from_profile``), without
    which a pooled cross-ancestry cohort's extents ran to every block's edge. One read of Stage 0's Grams."""
    from sv_pgs.marginal_variances import BlockGrams, ld_extent

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    widths: list[int] = []
    within_offsets: list[int] = []
    cross_offsets: list[int] = []
    extents: list[dict] = []
    positions = {"within": 0, "cross": 0}
    handles = {name: open(directory / file_name, "wb") for name, file_name in _BAND_FILES.items()}
    started = time.perf_counter()

    def write(name: str, values: np.ndarray) -> int:
        offset = positions[name]
        handles[name].write(np.ascontiguousarray(values, dtype=np.float32))
        positions[name] += int(values.size)
        return offset

    try:
        previous: slice | None = None
        for block in range(ld.block_count):
            width = int(ld.block_width(block))
            if width == 0:
                previous = None
                continue
            within = read_block(ld, "within", block)
            single = BlockGrams(blocks=(np.arange(width, dtype=np.int64),), within=(within,), next_cross=())
            extent = int(ld_extent(single, int(sample_count), int(working_bytes), level, array_module, null_scale)) if width > 1 else 1
            parts = max(1, -(-width // extent))
            cuts = np.linspace(0, width, parts + 1).round().astype(np.int64)
            spans = [slice(int(cuts[part]), int(cuts[part + 1])) for part in range(parts)]
            extents.append({"block": block, "width": width, "extent": extent, "parts": parts})
            first_cross = -1
            if previous is not None and ld.has_adjacent(block):
                adjacent = read_block(ld, "cross", block - 1)
                first_cross = write("cross", adjacent[previous, spans[0]])
                del adjacent
            for position, span in enumerate(spans):
                widths.append(span.stop - span.start)
                within_offsets.append(write("within", within[span, span]))
                cross_offsets.append(first_cross if position == 0 else write("cross", within[spans[position - 1], span]))
            previous = spans[-1]
            del within
    finally:
        for handle in handles.values():
            handle.close()
    stored = sum(positions.values()) * np.dtype(np.float32).itemsize
    index = {"widths": widths, "within_offsets": within_offsets, "cross_offsets": cross_offsets, "extents": extents, "stored_bytes": stored}
    (directory / _BAND_INDEX).write_text(json.dumps(index), encoding="utf-8")
    values = np.array([entry["extent"] for entry in extents])
    log(
        f"summary band: {len(extents)} Stage 0 blocks re-cut into {len(widths)} parts by their LD extents against kappa {float(null_scale):.4g} / n "
        f"(median {np.median(values):.0f}, "
        f"90% {np.quantile(values, 0.9):.0f}, max {values.max()} variants; blocks at their full width: "
        f"{sum(entry['extent'] >= entry['width'] for entry in extents)}), {stored / 1e9:.2f} GB against Stage 0's {ld.stored_bytes / 1e9:.2f} GB, "
        f"in {time.perf_counter() - started:.0f} s"
    )
    for entry in extents:
        log(f"summary band: block {entry['block']}: {entry['width']} columns, LD extent {entry['extent']}, {entry['parts']} parts")
    timing("summary_band", time.perf_counter() - started, blocks=len(extents), parts=len(widths), gb=stored / GB)
    return BandStore(directory)


def far_field_scale(
    store: Any, training_columns: I64Array, covariates: F64Array, statistics: Any, starts: I64Array, linked: np.ndarray, draw_count: int, seed: int,
    codes: Any = None,
) -> tuple[float, float, int]:
    """kappa, the mean r^2 n' of the pairs beyond the band (chance alone: 1), with its standard error and the pairs it
    is taken over, measured on the store: m reduced columns drawn at random, read on the training samples, standardized
    and projected on the covariates as Stage 0 projects its Grams, and their pairwise r^2 over the pairs no block reaches
    (``starts`` the blocks' first columns and ``linked[b]`` whether blocks b and b + 1 are neighbours: pairs in other
    blocks than a column's own and its linked neighbours'). Stage 0's blocks serve before the band is cut: the extent
    test reads its excess against this kappa (``build_band_store``). On bench-sim chr22 [sim] the pooled
    cohort's pairs beyond 20 Mb sit at 1.30 / n' (0.30 / n' above chance at 150 standard errors: between-group structure
    the PCs leave), so chance alone would under-state the far field. m is the least count whose chance standard error,
    2 / m for m (m - 1) / 2 pairs of chi-square(1) r^2 n', is within the fit's resolution 1 / (2K) of the scale: 4K
    columns, one read of each, never a pass over the store; none at all where ``codes``, the training codes Stage 0's pass
    kept (``training_codes.TrainingCodes``), hold them, which give the store's own codes."""
    started = time.perf_counter()
    generator = np.random.default_rng(seed)
    starts = np.asarray(starts, dtype=np.int64)
    group_count = int(starts[-1])
    count = min(group_count, 4 * int(draw_count))
    chosen = np.sort(generator.choice(group_count, size=count, replace=False))
    ties = statistics.tie_map
    kept = np.asarray(ties.kept_indices, dtype=np.int64)
    rows = np.asarray(statistics.active_rows, dtype=np.int64)[kept[chosen]]
    columns = np.asarray(training_columns, dtype=np.int64)
    values = np.empty((columns.shape[0], count))
    kept = codes.unsigned(rows) if codes is not None and codes.serves(columns) else None
    if kept is not None:
        values[:] = kept.T
    else:
        for index, row in enumerate(rows.tolist()):
            values[:, index] = np.asarray(store.read_codes(row, row + 1, columns), dtype=np.float64)[0]
    design = np.asarray(covariates, dtype=np.float64)
    left, singular, _ = np.linalg.svd(design, full_matrices=False)
    basis = left[:, singular > max(design.shape) * np.finfo(np.float64).eps * singular[0]]
    values -= basis @ (basis.T @ values)
    norms = np.linalg.norm(values, axis=0)
    values = values[:, norms > 0.0] / norms[norms > 0.0]
    chosen = chosen[norms > 0.0]
    residual = float(columns.shape[0] - basis.shape[1])
    part = np.searchsorted(starts, chosen, side="right") - 1
    near = np.abs(part[:, None] - part[None, :]) <= 1
    step = part[:, None] - part[None, :]
    # Neighbouring parts reach each other only where a Gram links them (none across a chromosome's end).
    lower = np.minimum(part[:, None], part[None, :])
    links = np.concatenate([np.asarray(linked, dtype=bool), [False]])
    near &= (step == 0) | links[np.clip(lower, 0, links.shape[0] - 1)]
    upper = np.triu(~near, 1)
    squares = (values.T @ values)[upper] ** 2 * residual
    pairs = int(squares.shape[0])
    timing("far_field", time.perf_counter() - started, columns=count)
    if pairs < 2:
        return 1.0, 0.0, pairs
    return float(np.mean(squares)), float(np.std(squares) / np.sqrt(pairs)), pairs



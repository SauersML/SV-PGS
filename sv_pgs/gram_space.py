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
- the posterior solves (``GramGaussian``): A = G/sigma^2 + diag(tau) by conjugate gradients on
  B = I + D^1/2 G D^1/2 / sigma^2 (D = 1/tau on the sites with tau > 0), block-Jacobi preconditioned, with the sites
  of non-positive precision eliminated exactly by their Schur complement, as the dual solver does in sample space
  (``dual_solve``: the two are the same system, B's spectrum being S = I + Xt D Xt''s apart from ones).

The far field. Whatever LD lies beyond the band (G - G_band: blocks two apart, and other chromosomes) is the
approximation. True LD between blocks two apart is what Stage 0's partition cuts at the LD's minimum, and its reach
is the data's (``marginal_variances.ld_extent``). Chance LD does not vanish: an unlinked pair has r^2 of mean 1/n, so
the far field adds to each field c_j a term sum_{k beyond j's band} G_jk E beta_k, of variance
G_jj (1/n) sum_k G_kk E beta_k^2 (``far_field_ratio``): relative to the noise's own G_jj sigma^2 that is the far
variants' share of the genetic variance over sigma^2, whatever n is. The fit reports it (``FitCertificate.far_field``),
largest over the blocks, at the fitted state; it is exactly 0 on the sample-space route.

Memory. The band is read block by block from Stage 0's files, never mapped whole (``LdGramStore.read_gram``); each
block read is charged to the shared ledger (``memory_broker``) while it lives, and a block is kept for later sweeps
and products only as a cache lease of what the ledger has left (on the device and on the host), so the band's size
never enters the job's resident set beyond what the ledger grants.
"""

from __future__ import annotations

import time
import weakref
from dataclasses import dataclass
from typing import Any, Callable

import numba
import numpy as np

from sv_pgs._typing import F64Array, I64Array
from sv_pgs.memory_broker import HOST, current_broker, device_pool
from sv_pgs.progress import log

_EPSILON = float(np.finfo(np.float64).eps)
_FLOAT_BYTES = np.dtype(np.float64).itemsize
_STORED_BYTES = np.dtype(np.float32).itemsize


def _host(values: Any) -> np.ndarray:
    return values.get() if hasattr(values, "get") else np.asarray(values)


@numba.njit(cache=True, fastmath={"reassoc", "contract"})
def _gram_sweep(gram, field, groups, signs, squares, class_index, log_density, node_variance, log_node_variance, noise, mean, variance, shift, third, fourth, moved):
    """One block's coordinate ascent in Gram space, in place, the members in the order given.

    ``gram`` is the block's Gram R_b (w x w, float32 as Stage 0 stores it, or float64), ``field`` c_b = x_g' r over the
    block's groups (w), ``groups[j]`` member j's group within the block and ``signs[j]`` its tie sign, ``squares`` the
    groups' ||x_g||^2 (R_b's diagonal), the per-member arrays (class, node variances and moments) in sweep order, and
    ``moved`` (w) receives each group's signed mean change. Member j's update is ``mean_field._sweep``'s, with its
    projection x_j'r = s_j c_g; after it, c -= R_b[g, :] s_j dm_j over the block (R_b symmetric, so its row g is its
    column g). Returns (sum_j KL(q_j || p_j), sum_j ||x_j||^2 v_j, the KL terms' pieces' sizes)."""
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
    """Stage 0's banded Gram of one target, read block by block on demand: R_b within each block and R_{b,b+1} between
    genome-adjacent blocks (float32, as stored), the projected scores s (float64), the groups' squares ||x_g||^2 (R's
    diagonal, float32-exact), y'Py, and the covariates' cross-products X'C for the covariate coefficients.

    ``working_bytes`` sizes the float64 row chunks a stored block is promoted in for a product. Each block read is
    charged to the shared ledger for as long as its array lives, and kept for later passes only as a ledger cache
    (the device's when the fit runs on one, else the host's), admitted from what the ledger has left."""

    def __init__(self, statistics: Any, working_bytes: int, target: int = 0) -> None:
        ld = statistics.ld
        self.ld = ld
        self.block_count = int(ld.block_count)
        widths = np.array([ld.block_width(block) for block in range(self.block_count)], dtype=np.int64)
        self.starts = np.concatenate([[0], np.cumsum(widths)]).astype(np.int64)
        self.group_count = int(self.starts[-1])
        # linked[b]: a stored Gram between block b and block b + 1 (none across a chromosome's end).
        self.linked = np.array([ld.has_adjacent(block + 1) for block in range(self.block_count - 1)], dtype=bool)
        self.sample_count = int(statistics.sample_count)
        self.residual_dimension = float(statistics.sample_count - statistics.covariate_rank)
        self.working_bytes = int(working_bytes)
        blocks = [ld.block(block) for block in range(self.block_count)]
        self.scores = np.concatenate([np.asarray(block.projected_score[:, target], dtype=np.float64) for block in blocks]) if blocks else np.zeros(0)
        self.covariate_cross = np.concatenate([np.asarray(block.covariate_cross, dtype=np.float64) for block in blocks], axis=0)
        del blocks
        # R_jj stored in float32, kept as R_jj / n in float64: n times it rounds back to the stored float32 exactly (its
        # relative error, two float64 roundings, is far below float32's half unit).
        self.squares = (np.asarray(ld.ld_diagonal(), dtype=np.float64) * statistics.sample_count).astype(np.float32).astype(np.float64)
        covariate_target = np.asarray(statistics.covariate_target, dtype=np.float64)[:, target]
        self.covariate_target = covariate_target
        self.covariate_pseudo_inverse = np.asarray(statistics.covariate_gram_pseudo_inverse, dtype=np.float64)
        self.target_square = float(np.asarray(statistics.target_gram, dtype=np.float64)[target, target] - covariate_target @ self.covariate_pseudo_inverse @ covariate_target)
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

    @classmethod
    def from_arrays(
        cls, within: list, cross: list, scores: F64Array, target_square: float, sample_count: int, residual_dimension: float, working_bytes: int,
        covariate_cross: F64Array | None = None, covariate_target: F64Array | None = None, covariate_pseudo_inverse: F64Array | None = None,
    ) -> "GramBand":
        """A band held in memory: ``within[b]`` R_b and ``cross[b]`` R_{b,b+1} (None where no Gram links the blocks),
        the rest as ``__init__`` reads it from Stage 0 (tests, and callers with their own Grams)."""
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
        band.scores = np.asarray(scores, dtype=np.float64)
        band.squares = np.concatenate([np.diagonal(np.asarray(values)).astype(np.float64) for values in within])
        band.target_square = float(target_square)
        band.covariate_cross = np.zeros((band.group_count, 0)) if covariate_cross is None else np.asarray(covariate_cross, dtype=np.float64)
        band.covariate_target = np.zeros(0) if covariate_target is None else np.asarray(covariate_target, dtype=np.float64)
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

    def _chunk_rows(self, columns: int, xp: Any = np) -> int:
        """Rows of a stored block promoted at once: two float64 copies (the rows and their magnitudes) within the fit's
        working set, and on a device within half of what its pool has left without evicting a cache (a promotion is
        transient; the cached blocks are what the next pass reads)."""
        available = self.working_bytes
        if xp is not np and self._broker is not None:
            pool = device_pool(int(xp.cuda.runtime.getDevice()))
            if pool in self._broker.meters:
                available = min(available, self._broker.remaining(pool) // 2)
        return max(1, available // max(1, _FLOAT_BYTES * columns * 2))

    def _times(self, matrix: Any, values: Any, xp: Any, transposed: bool = False, absolute: bool = False) -> Any:
        """matrix @ values (or matrix' @ values) in float64, the stored float32 promoted a row chunk at a time; with
        ``absolute``, also |matrix| @ |values|."""
        rows, columns = int(matrix.shape[0]), int(matrix.shape[1])
        out = xp.zeros((columns if transposed else rows,) + tuple(values.shape[1:]))
        magnitude = xp.zeros_like(out) if absolute else None
        step = self._chunk_rows(columns, xp)
        for first in range(0, rows, step):
            last = min(first + step, rows)
            chunk = xp.asarray(matrix[first:last], dtype=xp.float64)
            if transposed:
                out += chunk.T @ values[first:last]
                if absolute:
                    magnitude += xp.abs(chunk).T @ xp.abs(values[first:last])
            else:
                out[first:last] = chunk @ values
                if absolute:
                    magnitude[first:last] = xp.abs(chunk) @ xp.abs(values)
            del chunk
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
        fourth: F64Array,
    ) -> SweepResult:
        """One coordinate-ascent sweep over every block's members in ``member_blocks``' order, in place on the members'
        moments (host arrays), on ``xp`` (numpy, or cupy with ``device_sweep``'s panel kernel)."""
        grouped = np.zeros(self.group_count)
        np.add.at(grouped, group, sign * mean)
        if xp is np:
            return self._host_sweep(member_blocks, group, sign, class_index, log_density, scales, grid, noise, mean, variance, shift, third, fourth, grouped)
        return self._device_sweep(xp, member_blocks, group, sign, class_index, log_density, scales, grid, noise, mean, variance, shift, third, fourth, grouped)

    def _block_field(self, block: int, grouped: Any, xp: Any, within: Any) -> Any:
        """c_b = s_b - R_{b-1,b}' mbar_{b-1} - R_b mbar_b - R_{b,b+1} mbar_{b+1} at the current means."""
        own = self.span(block)
        field = xp.asarray(self.scores[own]) - self._times(within, grouped[own][:, None], xp)[:, 0]
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
        linear = float(grouped @ self.scores)
        residual_square = self.target_square - 2.0 * linear + quadratic
        residual_size = abs(self.target_square) + 2.0 * float(np.abs(grouped) @ np.abs(self.scores)) + quadratic_size
        return SweepResult(divergence, weighted_variance, residual_square, sizes, residual_size)

    def _host_sweep(self, member_blocks, group, sign, class_index, log_density, scales, grid, noise, mean, variance, shift, third, fourth, grouped) -> SweepResult:
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
            part = _gram_sweep(
                within, field, np.ascontiguousarray(group[members] - own.start), np.ascontiguousarray(sign[members]), np.ascontiguousarray(self.squares[own]),
                np.ascontiguousarray(class_index[members]), log_density, node_variance, log_node_variance, float(noise), *state, moved,
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

        from sv_pgs.device_sweep import PANEL, PIECE_COLUMNS, _kernel

        kernel = _kernel(cupy)
        log_density_device = cupy.asarray(np.ascontiguousarray(log_density))
        node_count = int(grid.shape[0])
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
                    (1,), (PANEL,),
                    (
                        panel_gram, projection, cupy.ascontiguousarray(member_squares[first:last]), classes[first:last], log_density_device,
                        np.int32(node_count), node_variance[first:last], log_node_variance[first:last], np.float64(noise), np.int32(last - first),
                        state[0][first:last], state[1][first:last], state[2][first:last], state[3][first:last], state[4][first:last], step, pieces[first:last],
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

    def far_field_ratio(self, second_moments: F64Array, noise: float) -> float:
        """max_b (1/n) sum_{k beyond b's band} ||x_k||^2 E beta_k^2 / sigma^2: the variance chance LD beyond the band
        would add to each field of block b, per unit of the field's own noise G_jj sigma^2 (an unlinked pair's r^2 has
        mean 1/n), at the given E beta_g^2 per group; the largest over the blocks."""
        load = self.squares * np.asarray(second_moments, dtype=np.float64) / self.sample_count
        per_block = np.array([float(np.sum(load[self.span(block)])) for block in range(self.block_count)])
        total = float(per_block.sum())
        worst = 0.0
        for block in range(self.block_count):
            near = per_block[block]
            if block and self.linked[block - 1]:
                near += per_block[block - 1]
            if block + 1 < self.block_count and self.linked[block]:
                near += per_block[block + 1]
            worst = max(worst, total - near)
        return worst / float(noise)


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
    and solve of the fit is a whole number of such passes in either space, so the cheaper pass is the cheaper fit."""
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
    close = getattr(blocks, "close", None)
    if close is not None:
        close()
    sample_seconds = (sample_read + sample_compute) * float(band.group_count) / float(stop - start)
    return sample_seconds, gram_seconds


class _GramSource:
    """The shape a ``DualModels`` over the members needs (``full_data_fit``): the groups and the samples, no data."""

    def __init__(self, variant_count: int, sample_count: int) -> None:
        self.variant_count = int(variant_count)
        self.sample_count = int(sample_count)


class GramGaussian:
    """The dual solver's role (``dual_solve.DualGaussian``) for the mean-field fit in Gram space: a quantitative model's
    sample-side arrays (its training mask, targets and covariates, which the oracle's projector reads, O(n) each), the
    groups' squares from the band, and A = G_band / sigma^2 + diag(tau) at the sites of the last ``iterate``, whose
    solves (``posterior_solve``) and mean are conjugate gradients on the band.

    One model (Stage 0's statistics are one target's). ``mean`` is solved when first read after an ``iterate``, so a
    fixed point's refresh (the mean-field route never reads it) costs nothing."""

    def __init__(self, band: GramBand, *, training: Any, targets: Any, covariates: Any, array_module: Any = np, probe_count: int = 0) -> None:
        self.band = band
        self.array_module = array_module
        xp = array_module
        self.training = xp.asarray(training, dtype=xp.float64)
        if int(self.training.shape[1]) != 1:
            raise ValueError("a Gram-space fit is one model's: Stage 0's statistics hold one target")
        self.targets = xp.where(self.training == 0.0, 0.0, xp.asarray(targets, dtype=xp.float64))
        self.covariates = xp.asarray(covariates, dtype=xp.float64)
        self.model_count = 1
        self.training_counts = _host(self.training.sum(axis=0))
        self.sample_weights = self.training
        self.unit_squares = xp.asarray(band.squares[:, None])
        self.metric_key = None
        self.source = _GramSource(band.group_count, int(self.training.shape[0]))
        self.noise_variance = np.ones(1)
        self.probe_count = int(probe_count)
        self._precision: F64Array | None = None
        self._shift: F64Array | None = None
        self._mean: F64Array | None = None
        self._resolved: dict | None = None
        self._factors: Callable[[Any], Any] | None = None
        self.last_posterior_duals: Any = None
        self.iterations = 0
        self.solves = 0

    def reweight(self, **_arguments: Any) -> None:
        raise ValueError("the Gram-space route fits quantitative models only: a binary model's metric moves with its sites")

    def iterate(self, *, site_precision: Any, site_shift: Any, noise_variance: Any, **_arguments: Any) -> None:
        """The sites (tau, nu) of the groups and the noise: A = G_band / sigma^2 + diag(tau), with q's mean
        A^-1 (s / sigma^2 + nu). Nothing is solved here."""
        self._precision = np.array(np.asarray(_host(site_precision), dtype=np.float64).reshape(self.band.group_count, -1)[:, 0], copy=True)
        self._shift = np.array(np.asarray(_host(site_shift), dtype=np.float64).reshape(self.band.group_count, -1)[:, 0], copy=True)
        self.noise_variance = np.array(np.asarray(noise_variance, dtype=np.float64).reshape(-1)[:1], copy=True)
        self._mean = None
        self._resolved = None
        self._factors = None

    @property
    def mean(self) -> F64Array:
        if self._mean is None:
            assert self._precision is not None and self._shift is not None
            right = (self.band.scores / float(self.noise_variance[0]) + np.where(np.isfinite(self._precision), self._shift, 0.0))[:, None]
            # The mean to the fit's resolution: ||mu_hat - mu||_A <= sqrt(1/K) for the scorer's K draws, the dual solver's.
            bound = np.full(1, np.sqrt(1.0 / self.probe_count)) if self.probe_count else np.full(1, np.sqrt(_EPSILON))
            solution, _certificate = self.posterior_solve(right, 0, bound)
            self._mean = np.asarray(solution, dtype=np.float64)
        return self._mean

    # the operator

    def _scaled(self) -> tuple[F64Array, I64Array, float]:
        """D^1/2 on the bulk (tau > 0; 0 where tau is infinite: that group is fixed), 0 elsewhere; the resolved sites
        (tau <= 0); and the noise."""
        assert self._precision is not None
        tau = self._precision
        bulk = tau > 0.0
        with np.errstate(divide="ignore"):
            root = np.where(bulk & np.isfinite(tau), 1.0 / np.sqrt(np.where(bulk, tau, 1.0)), 0.0)
        return root, np.flatnonzero(~bulk).astype(np.int64), float(self.noise_variance[0])

    def _bulk_operator(self, root: Any, noise: float) -> Callable[[Any], Any]:
        """y -> B y = y + D^1/2 G_band D^1/2 y / sigma^2 (identity on the rows D does not reach)."""
        xp = self.array_module
        band = self.band

        def apply(values: Any) -> Any:
            self.iterations += 1
            return values + root[:, None] * band.product(root[:, None] * values, xp) / noise

        return apply

    def _preconditioner(self, root: Any, noise: float) -> Callable[[Any], Any]:
        """M^-1 for M the block-Jacobi part of B: B's diagonal blocks over parts of each LD block no wider than the
        widest whose factors fit what the ledger grants the fit's working set (``_part_width``), each factored once per
        ``iterate``. A part whose B-block has no Cholesky factor (the band is not positive semidefinite there) is left
        at the identity; CG's own curvature check refuses such a band."""
        if self._factors is not None:
            return self._factors
        xp = self.array_module
        band = self.band
        width = self._part_width()
        if self.band._broker is not None and xp is np:
            # The factors over every group: at most groups x width float64, for as long as these sites hold.
            lease = self.band._broker.reserve(HOST, _FLOAT_BYTES * band.group_count * width, "the Gram-space preconditioner's factors")
        else:
            lease = None
        factors: list[tuple[slice, Any]] = []
        for block in range(band.block_count):
            own = band.span(block)
            within = band.within(block, xp)
            for first in range(own.start, own.stop, width):
                last = min(first + width, own.stop)
                local = slice(first - own.start, last - own.start)
                part_root = root[first:last]
                matrix = xp.asarray(within[local, local], dtype=xp.float64) * (part_root[:, None] * part_root[None, :]) / noise
                matrix[xp.arange(last - first), xp.arange(last - first)] += 1.0
                try:
                    factor = xp.linalg.cholesky(matrix)
                except np.linalg.LinAlgError:
                    continue
                if not bool(xp.all(xp.isfinite(factor))):
                    continue
                factors.append((slice(first, last), factor))
            del within

        def apply(values: Any) -> Any:
            out = values.copy()
            for rows, factor in factors:
                out[rows] = _cholesky_solve(xp, factor, values[rows])
            return out

        if lease is not None:
            weakref.finalize(apply, lease.release)
        self._factors = apply
        return apply

    def _part_width(self) -> int:
        """The widest preconditioner part whose factors over every group, groups x width float64, fit the fit's working
        set, never wider than the widest block."""
        widest = int(np.max(np.diff(self.band.starts))) if self.band.block_count else 1
        return max(1, min(widest, self.band.working_bytes // max(1, _FLOAT_BYTES * self.band.group_count)))

    def _cg(self, operator: Callable[[Any], Any], preconditioner: Callable[[Any], Any], right: Any, bound: Any, root: Any, noise: float) -> tuple[Any, Any, Any]:
        """Preconditioned CG on B y = right, column by column in one block of products, until each column's exact
        residual ||right - B y|| is within its bound or stops falling (float64's floor). Returns (y, residual, norms).
        B >= I holds where G_band is positive semidefinite; a direction with p'Bp < ||p||^2 beyond rounding says it is not,
        which is refused (LinAlgError), since the certificate ||y - y*||_B <= ||residual|| needs it."""
        xp = self.array_module
        solution = xp.zeros_like(right)
        residual = right.copy()
        norms = xp.sqrt(xp.sum(residual * residual, axis=0))
        target = xp.asarray(bound, dtype=xp.float64)
        best = norms.copy()
        stalled = xp.zeros(norms.shape, dtype=bool)
        since = xp.zeros(norms.shape, dtype=xp.int64)
        preconditioned = preconditioner(residual)
        direction = preconditioned.copy()
        inner = xp.sum(residual * preconditioned, axis=0)
        # Rounding of p'Bp: the product's float64 sums over at most the band's width plus the identity's term.
        rounding = (self.band.band_width + 2) * _EPSILON
        while True:
            open_columns = (norms > target) & ~stalled
            if not bool(xp.any(open_columns)):
                return solution, residual, norms
            image = operator(direction)
            curvature = xp.sum(direction * image, axis=0)
            length = xp.sum(direction * direction, axis=0)
            magnitude = length + xp.sum(xp.abs(direction) * xp.abs(image - direction), axis=0)
            if bool(xp.any(open_columns & (curvature - length < -rounding * magnitude))):
                raise np.linalg.LinAlgError(
                    "the banded Gram is not positive semidefinite along a conjugate-gradient direction (p'Bp < ||p||^2): the "
                    "blocks' and their neighbours' Grams do not bound a Gram, so the Gram-space posterior is not certified"
                )
            safe = xp.where(open_columns & (curvature > 0.0), curvature, 1.0)
            step = xp.where(open_columns & (curvature > 0.0), inner / safe, 0.0)
            solution += direction * step[None, :]
            residual -= image * step[None, :]
            norms = xp.sqrt(xp.sum(residual * residual, axis=0))
            # A column whose residual has not fallen below its best for as many products as its band has blocks is at
            # float64's floor: CG's residual norm is not monotone, but it cannot stay above its best for longer than the
            # Krylov space takes to cover the band's coupling, one block per product.
            improved = norms < best
            best = xp.where(improved, norms, best)
            since = xp.where(improved, 0, since + 1)
            stalled |= since > self.band.block_count + 1
            preconditioned = preconditioner(residual)
            updated = xp.sum(residual * preconditioned, axis=0)
            beta = xp.where(inner > 0.0, updated / xp.where(inner > 0.0, inner, 1.0), 0.0)
            direction = preconditioned + direction * beta[None, :]
            inner = updated

    def _resolved_block(self, root: F64Array, resolved: I64Array, noise: float, bound: float) -> dict:
        """The resolved sites' elimination: U = D^1/2 G[:, L] / sigma^2, Z = B^-1 U by CG (residuals R_Z), and the core
        A_LL - U'Z factored (Cholesky, or an eigendecomposition where the core is a linear response's symmetric
        indefinite one), as ``dual_solve.resolved_block``."""
        xp = self.array_module
        band = self.band
        unit = xp.zeros((band.group_count, resolved.shape[0]))
        unit[xp.asarray(resolved), xp.arange(resolved.shape[0])] = 1.0
        columns = band.product(unit, xp)
        del unit
        root_device = xp.asarray(root)
        design = root_device[:, None] * columns / noise
        tau = xp.asarray(self._precision[resolved])
        core_data = columns[xp.asarray(resolved)] / noise
        core_data[xp.arange(resolved.shape[0]), xp.arange(resolved.shape[0])] += tau
        del columns
        duals, residual, _norms = self._cg(self._bulk_operator(root_device, noise), self._preconditioner(root_device, noise), design, xp.full(resolved.shape[0], bound), root_device, noise)
        core = core_data - design.T @ duals
        core = 0.5 * (core + core.T)
        try:
            factor = xp.linalg.cholesky(core)
            signs = None if bool(xp.all(xp.isfinite(factor))) else False
        except np.linalg.LinAlgError:
            signs = False
        if signs is False:
            values, vectors = xp.linalg.eigh(core)
            magnitude = xp.abs(values)
            if not float(xp.min(magnitude)) > _EPSILON * core.shape[0] * float(xp.max(magnitude)):
                raise np.linalg.LinAlgError("the resolved sites' core is singular: the linear response has no solution.")
            factor, signs = vectors * xp.sqrt(magnitude)[None, :], xp.sign(values)
        from sv_pgs.dual_solve import ResolvedBlock

        return {"block": ResolvedBlock(design, None, duals, residual, core, factor, signs), "core_data": core_data, "rows": resolved}

    def posterior_solve(self, right: Any, model: int, error_bound: Any, start: Any = None) -> tuple[F64Array, F64Array]:
        """A^-1 right (groups x r) at the last ``iterate``'s sites and noise, and each column's certified ||x_hat - x||_A
        (``dual_solve.DualGaussian.posterior_solve``'s contract): on the bulk, x_S = D^1/2 y with B y = D^1/2 r_S, whose
        error in A's metric is y's in B's, at most the exact residual's norm (B >= I); the resolved rows by their Schur
        complement, x_L = core^-1 (r_L - U'y), and the certificate ``dual_solve.split_columns``'s bound with the core's
        and Z's residuals. Each column is solved to its bound or to float64's floor, the certificate saying which. The
        bulk duals y are left in ``last_posterior_duals``; ``start`` (a previous solve's) is accepted for the dual
        solver's contract and not read: a warm start saves products only where the solve takes many, and each iteration
        here is already a band solve."""
        from sv_pgs.dual_solve import _core_solve, core_bounds

        if model != 0:
            raise ValueError("a Gram-space fit is one model's")
        xp = self.array_module
        started = time.perf_counter()
        before = self.iterations
        root, resolved, noise = self._scaled()
        values = xp.asarray(np.asarray(_host(right), dtype=np.float64).reshape(self.band.group_count, -1))
        target = xp.broadcast_to(xp.asarray(error_bound, dtype=xp.float64), (int(values.shape[1]),)).copy()
        root_device = xp.asarray(root)
        operator = self._bulk_operator(root_device, noise)
        preconditioner = self._preconditioner(root_device, noise)
        bulk_right = root_device[:, None] * values
        if resolved.shape[0]:
            bulk_right[xp.asarray(resolved)] = 0.0
        if not resolved.shape[0]:
            duals, residual, norms = self._cg(operator, preconditioner, bulk_right, target, root_device, noise)
            solution, certificate = root_device[:, None] * duals, norms
        else:
            if self._resolved is None:
                # Z to float64's working accuracy: its residuals enter every column's certificate (split_columns).
                self._resolved = self._resolved_block(root, resolved, noise, float(np.sqrt(_EPSILON)))
            block = self._resolved["block"]
            rows = xp.asarray(resolved)
            duals, residual, norms = self._cg(operator, preconditioner, bulk_right, target, root_device, noise)
            shift = values[rows]
            resolved_mean = _core_solve(xp, block, shift - block.design.T @ duals)
            duals = duals - block.duals @ resolved_mean
            # b - B y - U x_L = (b - B y_b) - (U - B Z_hat) x_L.
            bulk_residual = residual - block.residual @ resolved_mean
            stationarity = shift - block.design.T @ duals - self._resolved["core_data"] @ resolved_mean
            lowest, residual_norm, core_error = core_bounds(xp, block)
            bulk_norms = xp.sqrt(xp.sum(bulk_residual * bulk_residual, axis=0))
            if core_error >= 1.0:
                certificate = xp.full(bulk_norms.shape, np.inf)
            else:
                projected = xp.linalg.solve(block.factor, stationarity - block.duals.T @ bulk_residual)
                quadratic = (xp.sqrt(xp.sum(projected * projected, axis=0)) + float(np.sqrt(1.0 / lowest)) * residual_norm * bulk_norms) ** 2 / (1.0 - core_error)
                certificate = xp.sqrt(bulk_norms * bulk_norms + quadratic)
            solution = root_device[:, None] * duals
            solution[rows] = resolved_mean
        self.last_posterior_duals = duals
        self.solves += 1
        log(f"gram posterior solve: {int(values.shape[1])} columns, {self.iterations - before} band products, {resolved.shape[0]} resolved sites, {time.perf_counter() - started:.1f} s")
        return np.asarray(_host(solution), dtype=np.float64), np.asarray(_host(certificate), dtype=np.float64)


def _cholesky_solve(xp: Any, lower: Any, right: Any) -> Any:
    if xp is np:
        from scipy.linalg import solve_triangular

        return solve_triangular(lower.T, solve_triangular(lower, right, lower=True), lower=False)
    from cupyx.scipy.linalg import solve_triangular as device_triangular

    return device_triangular(lower.T, device_triangular(lower, right, lower=True), lower=False)

"""The mean-field coordinate-ascent sweep on a CUDA device, for the streamed full-data route.

The host sweep (``mean_field._sweep``) updates each q_j in order from c_j = x_j' r, then moves the residual r by
x_j dm_j, which costs one pass over the n samples per coordinate twice over. Run on a streamed store it also needs the
projected columns on the host: at 40,000 samples x 518k records, 166 GB copied off the device every sweep.

Here the same updates, in the same order, run on the device a panel of ``PANEL`` columns at a time. For a panel
with projected columns Xp_P:

  c_P = X_P' M r          (r already lies in the projected space, so Xp_P' r = X_P' M r, = X_P' r for a 0/1 mask);
  within the panel, member j's update reads c_j and then c_i -= (Xp_P' Xp_P)_ij dm_j for the panel's members,
  which is exactly x_i' r after x_j dm_j has left it: one warp, one lane per member, the node sums by shuffles;
  r -= (I - H) M X_P dm_P  (M the root weights: the training mask, or a binary model's sqrt(omega); H the covariate projector).

Nothing but the panel's moments leaves the device. The panel Grams Xp_P' Xp_P depend on neither the prior nor the
noise, so each is formed once per fit (``PanelGrams``) and reused by every sweep. Every quantity is float64 and the
arithmetic is the host kernel's: the node terms with their overflow limits (``scale_mixture_ep._kernel_terms``),
Var = E_w[c] + Var_w(h c), the third and fourth central moments, and the KL pieces the ELBO needs.

``PANEL`` is the warp width: one lane per panel member for the Gram update, and the node sums reduce across the same
32 lanes by shuffles, with no shared-memory barrier.

``sweep_models`` runs the same sweep for every model a solve round asks at once, on the codes where each block's
device holds them, one kernel launch per panel. A sweep is a chain: each panel's c_P reads the residual every earlier
panel moved, and within a panel each member's h reads the members before it, so its time is the members' chain, not
the codes' bandwidth (a sweep reads 2 n bytes of codes per member, 88 GB at 44k samples x 1.0e6 members, 60 ms at an
A100's 1.5 TB/s). Through ``ShardedDualSource.blocks`` every panel was decoded on its block's device, copied home and
moved by some 25 array operations, about 516k submissions and 1.1-2.2 ms a panel per model: the All of Us
whole-genome sweeps' 35-70 s [real]. Here one launch per panel (``panel_step``) moves the residual by the previous
panel's steps on every sample, forms the partial sums of this panel's c_P from the moved residual in the same pass,
and its last block to finish (the CUDA programming guide's threadfence reduction: a ticket per block, no grid
barrier) completes the sums and runs the panel's members, one warp per model; the next launch moves the residual by
them. The models of a round are independent problems, each with its own residual, mask, sites, noise and prior, so
batching them changes no model's arithmetic: one model's sweep is the same whichever others share its launches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator, Sequence

import numpy as np

from sv_pgs.compute_budget import _cupy_device_context
from sv_pgs.device_plan import moved
from sv_pgs.memory_broker import device_pool
from sv_pgs.phase_timer import launch_timer

PANEL = 32
"""Columns per panel: the CUDA warp width (one lane per member of the panel)."""
PIECE_COLUMNS = 3
"""Per member: its KL term, ||x_j||^2 v_j, and the KL term's pieces' sizes (the ELBO and its rounding bound)."""
MOMENTS = ("mean", "variance", "shift", "third", "fourth")
"""A member's state a sweep moves, in ``panel_step``'s order of its arguments."""

_SOURCE = r"""
extern "C" __global__ void panel_sweep(
    const double* __restrict__ gram, double* __restrict__ projection, const double* __restrict__ squares,
    const long long* __restrict__ class_index, const double* __restrict__ log_density, const int node_count,
    const double* __restrict__ node_variance, const double* __restrict__ log_node_variance, const double noise,
    const int width, double* __restrict__ mean, double* __restrict__ variance, double* __restrict__ shift,
    double* __restrict__ third, double* __restrict__ fourth, double* __restrict__ step_out, double* __restrict__ pieces,
    double* __restrict__ conditional_scratch, double* __restrict__ base_scratch, double* __restrict__ weight_scratch)
{
    // Phase 0, every thread of the block: each (member, node)'s h-free terms, the conditional variance
    // c = 1 / (1 / v + omega) and the log weight's base log pi - log(1 + v omega) / 2 (or its overflow limit), which do
    // not depend on the member's cavity shift h. The log weight is then base + h^2 c / 2, the same expression, in the same
    // order, as when it was formed whole per pass.
    const int thread = threadIdx.x;
    for (int index = thread; index < width * node_count; index += blockDim.x) {
        const int member = index / node_count;
        const int node = index - member * node_count;
        const double omega = squares[member] / noise;
        const double variance_node = node_variance[member * (long long)node_count + node];
        const double ratio = variance_node * omega;
        const double density = log_density[class_index[member] * (long long)node_count + node];
        double conditional, base;
        if (isinf(ratio)) {
            conditional = 1.0 / omega;
            base = density - 0.5 * (log_node_variance[member * (long long)node_count + node] + log(omega));
        } else {
            conditional = variance_node > 0.0 ? 1.0 / (1.0 / variance_node + omega) : 0.0;
            base = density - 0.5 * log1p(ratio);
        }
        conditional_scratch[index] = conditional;
        base_scratch[index] = base;
    }
    __syncthreads();
    if (thread >= 32) return;
    // Phase 1, one warp, one lane per panel member for the Gram update and the node sums by shuffles, the members in order.
    const int lane = thread;
    const unsigned full = 0xffffffffu;
    double c = lane < width ? projection[lane] : 0.0;
    for (int member = 0; member < width; ++member) {
        const double square = squares[member];
        const double omega = square / noise;
        const double old_mean = mean[member];
        const double h = (__shfl_sync(full, c, member) + square * old_mean) / noise;
        const double* conditional_row = conditional_scratch + member * (long long)node_count;
        const double* base_row = base_scratch + member * (long long)node_count;
        double* weight_row = weight_scratch + member * (long long)node_count;
        // pass 1: the peak of the log weights
        double peak = __longlong_as_double(0xfff0000000000000ULL);  // -inf: NVRTC has no INFINITY
        for (int node = lane; node < node_count; node += 32) {
            const double log_weight = base_row[node] + 0.5 * h * h * conditional_row[node];
            weight_row[node] = log_weight;
            peak = fmax(peak, log_weight);
        }
        for (int offset = 16; offset > 0; offset >>= 1) peak = fmax(peak, __shfl_xor_sync(full, peak, offset));
        // pass 2: the normalizer and the mean
        double total = 0.0, first = 0.0;
        for (int node = lane; node < node_count; node += 32) {
            const double weight = exp(weight_row[node] - peak);
            weight_row[node] = weight;
            total += weight;
            first += weight * h * conditional_row[node];
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            total += __shfl_xor_sync(full, total, offset);
            first += __shfl_xor_sync(full, first, offset);
        }
        const double new_mean = first / total;
        // pass 3: the central moments about the new mean
        double second = 0.0, third_sum = 0.0, fourth_sum = 0.0;
        for (int node = lane; node < node_count; node += 32) {
            const double conditional = conditional_row[node];
            const double weight = weight_row[node] / total;
            const double offset_value = h * conditional - new_mean;
            const double offset_square = offset_value * offset_value;
            second += weight * (conditional + offset_square);
            third_sum += weight * (offset_square * offset_value + 3.0 * conditional * offset_value);
            fourth_sum += weight * (offset_square * offset_square + 6.0 * conditional * offset_square + 3.0 * conditional * conditional);
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            second += __shfl_xor_sync(full, second, offset);
            third_sum += __shfl_xor_sync(full, third_sum, offset);
            fourth_sum += __shfl_xor_sync(full, fourth_sum, offset);
        }
        const double step = new_mean - old_mean;
        const double log_normalizer = peak + log(total);
        if (lane == 0) {
            mean[member] = new_mean;
            variance[member] = second;
            third[member] = third_sum;
            fourth[member] = fourth_sum;
            shift[member] = h;
            step_out[member] = step;
            const double pull = h * new_mean;
            const double shrink = 0.5 * omega * (new_mean * new_mean + second);
            pieces[3 * member] = pull - shrink - log_normalizer;
            pieces[3 * member + 1] = square * second;
            pieces[3 * member + 2] = fabs(pull) + shrink + fabs(log_normalizer);
        }
        // x_i' r after x_member dm has left r, for the panel's other members
        if (lane < width) c -= gram[lane * (long long)width + member] * step;
    }
}
"""

_CODE_SOURCE = r"""
// A panel's products read straight from the held int8 codes, standardized on the fly: x_j = s_j (c_j - mu_j) / sigma_j
// with c_j the code row rows[j] of the tile, s_j the member's tie sign. Neither forms the float64 columns.
extern "C" __global__ void panel_project(
    const signed char* __restrict__ codes, const long long stride, const long long* __restrict__ rows, const int sample_count,
    const double* __restrict__ mask, const double* __restrict__ residual, const double* __restrict__ means,
    const double* __restrict__ scales, const double* __restrict__ signs, double* __restrict__ out
) {
    // out_j = x_j' M r, one block per member: sum_i c_ji w_i and sum_i w_i with w = M r, then the standardization.
    __shared__ double products[PROJECT_THREADS];
    __shared__ double totals[PROJECT_THREADS];
    const int j = blockIdx.x;
    const long long row = rows[j];
    const signed char* code = codes + row * stride;
    double product = 0.0, total = 0.0;
    for (int i = threadIdx.x; i < sample_count; i += blockDim.x) {
        const double weighted = mask[i] * residual[i];
        product += (double)code[i] * weighted;
        total += weighted;
    }
    products[threadIdx.x] = product;
    totals[threadIdx.x] = total;
    __syncthreads();
    for (int half = blockDim.x / 2; half > 0; half /= 2) {
        if (threadIdx.x < half) {
            products[threadIdx.x] += products[threadIdx.x + half];
            totals[threadIdx.x] += totals[threadIdx.x + half];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[j] = signs[j] * (products[0] - means[row] * totals[0]) / scales[row];
}

extern "C" __global__ void panel_apply(
    const signed char* __restrict__ codes, const long long stride, const long long* __restrict__ rows, const int width,
    const int sample_count, const double* __restrict__ step, const double* __restrict__ means, const double* __restrict__ scales,
    const double* __restrict__ signs, const double* __restrict__ mask, const double* __restrict__ covariates,
    const double* __restrict__ covariate_step, const int covariate_count, double* __restrict__ residual
) {
    // r -= M X_P s - (M C) u, u = A s (``sweep_piece``), one thread per sample.
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= sample_count) return;
    double moved = 0.0;
    for (int j = 0; j < width; ++j) {
        const long long row = rows[j];
        moved += signs[j] * step[j] * ((double)codes[row * stride + i] - means[row]) / scales[row];
    }
    double covariate = 0.0;
    for (int c = 0; c < covariate_count; ++c) covariate += covariates[(long long)i * covariate_count + c] * covariate_step[c];
    residual[i] -= moved * mask[i] - covariate;
}
"""

PROJECT_THREADS = 1024
"""Threads per block in the code kernels: CUDA's maximum threads per block (a power of two, which the tree reduction
halves), so one block covers the most samples per pass that a block can."""

_KERNELS: dict[int, Any] = {}
_CODE_KERNELS: dict[int, tuple[Any, Any]] = {}


def _kernel(cupy: Any) -> Any:
    device = int(cupy.cuda.runtime.getDevice())
    if device not in _KERNELS:
        _KERNELS[device] = cupy.RawKernel(_SOURCE, "panel_sweep", options=("--std=c++11",))
    return _KERNELS[device]


def step_threads(kernel: Any) -> int:
    """The step kernel's block: every thread forms the h-free terms, then its first warp runs the members. The kernel's
    own limit on this device (its registers per thread against the multiprocessor's file: an H100 refused 1,024 with
    CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES), at most ``PROJECT_THREADS``, and whole warps (at least the one the loop needs)."""
    return max(PANEL, min(PROJECT_THREADS, int(kernel.max_threads_per_block)) // PANEL * PANEL)


def _code_kernels(cupy: Any) -> tuple[Any, Any]:
    device = int(cupy.cuda.runtime.getDevice())
    if device not in _CODE_KERNELS:
        source = _CODE_SOURCE.replace("PROJECT_THREADS", str(PROJECT_THREADS))
        _CODE_KERNELS[device] = (
            cupy.RawKernel(source, "panel_project", options=("--std=c++11",)),
            cupy.RawKernel(source, "panel_apply", options=("--std=c++11",)),
        )
    return _CODE_KERNELS[device]


class CodePanels:
    """A piece's panel products read from its tile's held int8 codes (``code_products.CodeBlockTile``): c_P = X_P' M r
    and the residual's move r -= M X_P s - (M C) A s, each one kernel over the codes, never forming X_P in float64.

    A sweep reads every column twice per pass (c_P, then the move), so where the columns were decoded each time a
    sweep moved 8 bytes per entry through the device three times over (decoded, read, read again) against 1 byte of
    code: decoding was 35% of a genome fit's stage 2 and the panel products most of the rest (bench-sim 015 [sim],
    py-spy). ``rows`` are the members' tile rows, ``signs`` their tie signs, both in sweep order."""

    def __init__(self, cupy: Any, tile: Any, rows: Any, signs: Any) -> None:
        self.cupy = cupy
        self.codes = tile.aligned_codes
        self.stride = int(self.codes.shape[1])
        self.sample_count = int(tile.sample_count)
        self.means = cupy.ascontiguousarray(cupy.asarray(tile.means, dtype=cupy.float64))
        self.scales = cupy.ascontiguousarray(cupy.asarray(tile.scales, dtype=cupy.float64))
        self.rows = cupy.ascontiguousarray(cupy.asarray(rows, dtype=cupy.int64))
        self.signs = cupy.ascontiguousarray(cupy.asarray(signs, dtype=cupy.float64))
        self.project_kernel, self.apply_kernel = _code_kernels(cupy)

    @staticmethod
    def supports(tile: Any) -> bool:
        return all(hasattr(tile, name) for name in ("aligned_codes", "means", "scales", "sample_count"))

    def project(self, first: int, last: int, mask: Any, residual: Any) -> Any:
        out = self.cupy.empty(last - first, dtype=self.cupy.float64)
        timer = launch_timer()
        mark = timer.before()
        self.project_kernel(
            (last - first,), (PROJECT_THREADS,),
            (self.codes, np.int64(self.stride), self.rows[first:last], np.int32(self.sample_count), mask, residual,
             self.means, self.scales, self.signs[first:last], out),
        )
        timer.after(mark)
        return out

    def apply(self, first: int, last: int, step: Any, mask: Any, covariates: Any, covariate_step: Any, residual: Any) -> None:
        threads = PROJECT_THREADS
        timer = launch_timer()
        mark = timer.before()
        self.apply_kernel(
            (-(-self.sample_count // threads),), (threads,),
            (self.codes, np.int64(self.stride), self.rows[first:last], np.int32(last - first), np.int32(self.sample_count),
             step, self.means, self.scales, self.signs[first:last], mask, covariates, covariate_step,
             np.int32(covariates.shape[1]), residual),
        )
        timer.after(mark)


class PanelGrams:
    """Each panel's projected Gram Xp_P' Xp_P (float64, width x width) and covariate coupling, formed on the device
    and kept for later sweeps as a cache of the shared ledger (``memory_broker``): the design is fixed through a fit,
    so a kept panel is never rebuilt. A panel is admitted only from what the device's pool has left once the fit's
    mandatory buffers are reserved, and a later mandatory lease evicts it (the panel is then rebuilt when met), so the
    cache never takes what a mandatory buffer needs. A panel is named by its device's pool, its model and its members'
    own indices in sweep order, so a panel of another order or of other members never reads it (keyed by a piece's
    first member plus an offset, two panels of a random within-block order could share a key and one read the other's
    Gram), and is charged to the pool of the device that built it (``sweep_models`` builds each block's on its own
    device). Sweeps visit the panels in one cycle, where any eviction order misses every evicted panel once per sweep,
    so the cache keeps the panels it met first and rebuilds the rest. With no ``broker`` every panel is kept."""

    def __init__(self, broker: Any = None, pool: str | None = None) -> None:
        self._grams: dict[tuple, Any] = {}
        self._leases: dict[tuple, Any] = {}
        self._broker = broker
        self._pool = pool

    def get(self, key: tuple, build, *, pool: str | None = None) -> Any:
        """The panel ``key`` (its first entry the model) held in ``pool`` (the cache's own by default), built where it
        is not held."""
        pool = self._pool if pool is None else pool
        key = (pool, key)
        held = self._grams.get(key)
        if held is not None:
            return held
        built = build()
        if self._broker is None:
            self._grams[key] = built
            return built
        lease = None
        if pool in self._broker.capacities:
            lease = self._broker.admit(pool, sum(int(part.nbytes) for part in built), "a panel Gram", lambda key=key: self._evict(key), allocated=True)
        if lease is not None:
            self._grams[key] = built
            self._leases[key] = lease
        return built

    def _evict(self, key: tuple) -> None:
        self._grams.pop(key, None)
        self._leases.pop(key, None)

    def drop(self, model: int) -> None:
        """Release one model's panels (its metric moved: ``full_data_fit._FullDataMeanField._reweight``); the other
        models' are theirs alone and stay."""
        for key in [key for key in self._grams if key[1][0] == model]:
            lease = self._leases.pop(key, None)
            if lease is not None:
                lease.release()
            self._grams.pop(key, None)

    def clear(self) -> None:
        for lease in self._leases.values():
            lease.release()
        self._grams.clear()
        self._leases.clear()


def sweep_piece(
    cupy: Any,
    *,
    decode,
    width: int,
    mask: Any,
    covariates: Any,
    covariate_pinv: Any,
    residual: Any,
    grams: PanelGrams,
    model: int,
    members: np.ndarray,
    metric: bytes = b"",
    squares: Any,
    class_index: Any,
    log_density: Any,
    node_variance: Any,
    log_node_variance: Any,
    noise: float,
    mean: Any,
    variance: Any,
    shift: Any,
    third: Any,
    fourth: Any,
    pieces: Any,
    panels: CodePanels | None = None,
) -> None:
    """One sweep over a piece's columns in order, in place on the device.

    ``decode(first, last)`` returns the piece's standardized columns first..last-1 (n x panel, before masking and
    projection; one panel at a time, so the device holds a panel's columns, never a block's), ``width`` the piece's
    member count, ``mask`` M the model's root weights W^1/2 per unit noise (n: its training indicator for a
    quantitative model, sqrt(omega) on the training rows for a binary one, ``binary_likelihood``), ``covariates`` its
    masked covariates M C (n x k) and ``covariate_pinv`` (C'M^2 C)^+ (k x k): the complement of M v is
    (I - H) M v = M v - M C (C'M^2 C)^+ C' M^2 v (the noise's weights cancel), so each panel keeps
    A = (C'M^2 C)^+ C' M^2 X_panel (k x panel) beside its Gram and a sweep's residual update is M X s - M C (A s), two
    small products in place of a general projection per panel (35% of a bench-sim Stage 2 [sim, scenario_001]);
    ``residual`` r (n), which lies in the projected space, so Xp_P' r = X_P' M r (M r = r for a 0/1 mask);
    the per-member arrays are the piece's own slices, and ``pieces`` (width x 3) receives each member's KL term,
    ||x_j||^2 v_j and the terms' sizes. ``model``, ``metric`` (the weights' key: a binary model's Grams belong to one
    set of Polya-Gamma weights) and ``members`` (the piece's member indices, host, in sweep order) name each panel for
    the Gram cache. With ``panels`` (``CodePanels``) the panel products read the codes and ``decode`` runs only to
    build a Gram the cache does not hold."""
    kernel = _kernel(cupy)
    # The step kernel's launches, timed where the sweep is (``phase_timer.timed_sweep``).
    timer = launch_timer()
    node_count = int(node_variance.shape[1])
    # The step kernel's per-(member, node) terms that do not depend on the cavity shift, formed by the whole block
    # before its one warp's sequential member loop (``panel_sweep``), and the loop's weights: one panel's worth each.
    conditional_scratch = cupy.empty(PANEL * node_count, dtype=cupy.float64)
    base_scratch = cupy.empty(PANEL * node_count, dtype=cupy.float64)
    weight_scratch = cupy.empty(PANEL * node_count, dtype=cupy.float64)
    for first in range(0, width, PANEL):
        last = min(first + PANEL, width)
        columns = decode(first, last) if panels is None else None

        def build(columns=columns, first=first, last=last):
            columns = decode(first, last) if columns is None else columns
            masked = columns * mask[:, None]
            coupling = cupy.ascontiguousarray(covariate_pinv @ (covariates.T @ masked))
            projected = masked - covariates @ coupling
            return cupy.ascontiguousarray(projected.T @ projected), coupling

        gram, coupling = grams.get((int(model), metric, np.ascontiguousarray(members[first:last], dtype=np.int64).tobytes()), build)
        projection = cupy.ascontiguousarray(columns.T @ (mask * residual)) if panels is None else panels.project(first, last, mask, residual)
        step = cupy.empty(last - first, dtype=cupy.float64)
        mark = timer.before()
        kernel(
            (1,), (step_threads(kernel),),
            (
                gram, projection, squares[first:last], class_index[first:last], log_density, np.int32(node_count),
                node_variance[first:last], log_node_variance[first:last], np.float64(noise), np.int32(last - first),
                mean[first:last], variance[first:last], shift[first:last], third[first:last], fourth[first:last], step,
                pieces[first:last], conditional_scratch, base_scratch, weight_scratch,
            ),
        )
        timer.after(mark)
        if panels is None:
            residual -= (columns @ step) * mask - covariates @ (coupling @ step)
        else:
            panels.apply(first, last, step, mask, covariates, cupy.ascontiguousarray(coupling @ step), residual)


_BATCH_SOURCE = r"""
#define FULL_MASK 0xffffffffu

// Each (member, model, node)'s h-free terms of a piece (``panel_sweep``'s phase 0, the same expressions in the same
// order), every member of the piece at once, so that no panel's chain waits for them: rows ``first`` to
// ``first + width`` of the device's layout, written as [member][model][node].
extern "C" __global__ void node_terms(
    const double* __restrict__ log_scale, const double* __restrict__ squares, const long long* __restrict__ class_index,
    const double* __restrict__ log_density, const double* __restrict__ grid, const double* __restrict__ noise,
    const long long first, const int width, const int models, const int node_count, const int class_count,
    double* __restrict__ conditional, double* __restrict__ base)
{
    const long long total = (long long)width * models * node_count;
    for (long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x; index < total; index += (long long)gridDim.x * blockDim.x) {
        const int node = (int)(index % node_count);
        const long long rest = index / node_count;
        const int model = (int)(rest % models);
        const long long member = first + rest / models;
        const long long at = member * models + model;
        const double omega = squares[at] / noise[model];
        const double log_variance_node = log_scale[at] + grid[node];
        const double variance_node = exp(log_variance_node);
        const double ratio = variance_node * omega;
        const double density = log_density[((long long)model * class_count + class_index[member]) * node_count + node];
        double conditional_value, base_value;
        if (isinf(ratio)) {
            conditional_value = 1.0 / omega;
            base_value = density - 0.5 * (log_variance_node + log(omega));
        } else {
            conditional_value = variance_node > 0.0 ? 1.0 / (1.0 / variance_node + omega) : 0.0;
            base_value = density - 0.5 * log1p(ratio);
        }
        conditional[index] = conditional_value;
        base[index] = base_value;
    }
}

// One panel of every model's sweep, one sample per thread:
//   1. the residual's move by the previous panel's steps s (``applied``, ``applied_width`` members, 0 for none):
//      r -= M X s - M C u, u = A s, as ``panel_apply``, each member's term a_j (c_ij - mu_j) with a_j = s_j dm_j / sigma_j
//      formed once per block (one division per member, not one per sample and member);
//   2. this panel's partial sums over the block's samples (``projected``, ``projected_width`` members, 0 for none):
//      lane j of each warp sums member j's codes against its warp's PANEL weighted residuals M r, the warps' sums
//      added in order, with the sum of the weights beside them; each warp first copies the panel's codes at its
//      samples to shared memory, one coalesced row per member, read there by every model (lane j reading member j's
//      row from the codes themselves touches 32 rows at once, 32 sectors per load, again for each model);
//   3. the block that takes the last ticket completes c_j = s_j (sum_i c_ji w_i - mu_j sum_i w_i) / sigma_j from
//      every block's partial sums, and runs the panel's members (``panel_sweep``'s phase 1, one warp per model, the
//      node terms held in registers, the next member's loaded while this one runs), writing the steps and u = A s for
//      the next launch.
// Members are positions in the device's layout: ``rows`` their code rows in the tile, ``signs`` their tie signs;
// member arrays are [position][model] and sample arrays [model][sample].
extern "C" __global__ void panel_step(
    const signed char* __restrict__ codes, const long long stride, const double* __restrict__ code_means,
    const double* __restrict__ code_scales, const long long* __restrict__ rows, const double* __restrict__ signs,
    const long long applied, const int applied_width, const long long projected, const int projected_width,
    const double* __restrict__ covariates, const int covariate_count, const double* __restrict__ masks,
    double* residuals, const int models, const int sample_count, double* steps, double* covariate_steps,
    double* partials, double* partial_totals, unsigned int* counter,
    const double* __restrict__ conditional, const double* __restrict__ base, const long long node_first, const int node_count,
    const double* __restrict__ squares, const double* __restrict__ noise, const double* __restrict__ grams,
    const double* __restrict__ couplings, const long long panel,
    double* mean, double* variance, double* shift, double* third, double* fourth, double* pieces)
{
    extern __shared__ double shared[];
    __shared__ unsigned int ticket;
    __shared__ long long applied_rows[PANEL];
    __shared__ long long projected_rows[PANEL];
    const int threads = blockDim.x;
    const int warps = threads / PANEL;
    double* weighted = shared;
    double* member_sums = shared + threads;
    double* weight_sums = member_sums + warps * PANEL;
    double* scaled_steps = weight_sums + warps;
    double* centers = scaled_steps + 2 * PANEL;
    const int lane = threadIdx.x % PANEL;
    const int warp = threadIdx.x / PANEL;
    // the warp's tile: member j's codes at the warp's samples in row j, PANEL + 1 bytes apart (lane j's reads along
    // its row then fall in 32 different banks)
    signed char* tile = (signed char*)(centers + PANEL) + warp * PANEL * (PANEL + 1);
    const int block_first = blockIdx.x * threads;
    const int sample = block_first + threadIdx.x;
    const int covered = min(threads, sample_count - block_first);
    for (int j = threadIdx.x; j < applied_width; j += threads) {
        applied_rows[j] = rows[applied + j];
        centers[j] = code_means[applied_rows[j]];
    }
    for (int j = threadIdx.x; j < projected_width; j += threads) projected_rows[j] = rows[projected + j];
    __syncthreads();
    if (projected_width > 0) {
        for (int j = 0; j < PANEL; ++j)
            tile[j * (PANEL + 1) + lane] = j < projected_width && sample < sample_count ? codes[projected_rows[j] * stride + sample] : 0;
        __syncwarp();
    }
    for (int model = 0; model < models; ++model) {
        // this model's a_j, in one of two buffers by its parity: every thread passes the barrier after them only once
        // it has read the buffer's previous model's, so a block's shared memory is the same whatever the models
        double* scaled_step = scaled_steps + (model % 2) * PANEL;
        if (applied_width > 0) {
            if (threadIdx.x < PANEL) {
                const int j = threadIdx.x;
                scaled_step[j] = j < applied_width ? signs[applied + j] * steps[(long long)model * PANEL + j] / code_scales[applied_rows[j]] : 0.0;
            }
            __syncthreads();
        }
        double value = 0.0;
        if (sample < sample_count) {
            const long long at = (long long)model * sample_count + sample;
            const double mask = masks[at];
            double residual = residuals[at];
            if (applied_width > 0) {
                double moved = 0.0;
                for (int j = 0; j < applied_width; ++j) moved += scaled_step[j] * ((double)codes[applied_rows[j] * stride + sample] - centers[j]);
                const double* covariate_step = covariate_steps + (long long)model * covariate_count;
                double covariate = 0.0;
                for (int c = 0; c < covariate_count; ++c) covariate += covariates[(long long)c * sample_count + sample] * covariate_step[c];
                residual -= moved * mask - covariate * mask;
                residuals[at] = residual;
            }
            value = mask * residual;
        }
        if (projected_width == 0) continue;
        weighted[threadIdx.x] = value;
        __syncthreads();
        const int warp_first = warp * PANEL;
        double product = 0.0, total = 0.0;
        if (lane < projected_width) {
            const signed char* code = tile + lane * (PANEL + 1);
            for (int offset = 0; offset < PANEL && warp_first + offset < covered; ++offset) product += (double)code[offset] * weighted[warp_first + offset];
        }
        for (int offset = 0; offset < PANEL && warp_first + offset < covered; ++offset) total += weighted[warp_first + offset];
        member_sums[warp * PANEL + lane] = product;
        if (lane == 0) weight_sums[warp] = total;
        __syncthreads();
        if (warp == 0) {
            double block_product = 0.0;
            for (int other = 0; other < warps; ++other) block_product += member_sums[other * PANEL + lane];
            partials[((long long)model * gridDim.x + blockIdx.x) * PANEL + lane] = block_product;
            if (lane == 0) {
                double block_total = 0.0;
                for (int other = 0; other < warps; ++other) block_total += weight_sums[other];
                partial_totals[(long long)model * gridDim.x + blockIdx.x] = block_total;
            }
        }
        __syncthreads();
    }
    if (projected_width == 0) return;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) ticket = atomicAdd(counter, 1u);
    __syncthreads();
    if (ticket != gridDim.x - 1) return;
    if (threadIdx.x == 0) *counter = 0u;
    for (int model = warp; model < models; model += warps) {
        double total = 0.0;
        for (int block = 0; block < (int)gridDim.x; ++block) total += __ldcg(partial_totals + (long long)model * gridDim.x + block);
        double c = 0.0;
        if (lane < projected_width) {
            double product = 0.0;
            for (int block = 0; block < (int)gridDim.x; ++block) product += __ldcg(partials + ((long long)model * gridDim.x + block) * PANEL + lane);
            const long long row = projected_rows[lane];
            c = signs[projected + lane] * (product - code_means[row] * total) / code_scales[row];
        }
        const double model_noise = noise[model];
        const double* gram = grams + (panel * models + model) * (long long)(PANEL * PANEL);
        const double* coupling = couplings + (panel * models + model) * (long long)covariate_count * PANEL;
        double* step_out = steps + (long long)model * PANEL;
        // The next member's terms, its square, mean and Gram entry, loaded a member ahead: the chain never waits on memory.
        double next_held[NODES_PER_LANE], next_base[NODES_PER_LANE];
        double next_square = 0.0, next_mean = 0.0, next_gram = 0.0;
        {
            const long long at = projected * models + model;
            const long long terms = (node_first * models + model) * (long long)node_count;
            #pragma unroll
            for (int slot = 0; slot < NODES_PER_LANE; ++slot) {
                const int node = lane + PANEL * slot;
                next_held[slot] = node < node_count ? conditional[terms + node] : 0.0;
                next_base[slot] = node < node_count ? base[terms + node] : 0.0;
            }
            next_square = squares[at];
            next_mean = mean[at];
            next_gram = lane < projected_width ? gram[lane * PANEL] : 0.0;
        }
        for (int member = 0; member < projected_width; ++member) {
            const long long at = (projected + member) * models + model;
            const double square = next_square;
            const double old_mean = next_mean;
            const double gram_entry = next_gram;
            double held[NODES_PER_LANE];
            double weight[NODES_PER_LANE];
            #pragma unroll
            for (int slot = 0; slot < NODES_PER_LANE; ++slot) {
                held[slot] = next_held[slot];
                weight[slot] = next_base[slot];
            }
            if (member + 1 < projected_width) {
                const long long next_at = at + models;
                const long long terms = ((node_first + member + 1) * models + model) * (long long)node_count;
                #pragma unroll
                for (int slot = 0; slot < NODES_PER_LANE; ++slot) {
                    const int node = lane + PANEL * slot;
                    next_held[slot] = node < node_count ? conditional[terms + node] : 0.0;
                    next_base[slot] = node < node_count ? base[terms + node] : 0.0;
                }
                next_square = squares[next_at];
                next_mean = mean[next_at];
                next_gram = lane < projected_width ? gram[lane * PANEL + member + 1] : 0.0;
            }
            const double omega = square / model_noise;
            const double h = (__shfl_sync(FULL_MASK, c, member) + square * old_mean) / model_noise;
            // pass 1: the peak of the log weights
            double peak = __longlong_as_double(0xfff0000000000000ULL);  // -inf: NVRTC has no INFINITY
            #pragma unroll
            for (int slot = 0; slot < NODES_PER_LANE; ++slot) {
                if (lane + PANEL * slot < node_count) {
                    const double log_weight = weight[slot] + 0.5 * h * h * held[slot];
                    weight[slot] = log_weight;
                    peak = fmax(peak, log_weight);
                }
            }
            for (int offset = PANEL / 2; offset > 0; offset >>= 1) peak = fmax(peak, __shfl_xor_sync(FULL_MASK, peak, offset));
            // pass 2: the normalizer and the mean
            double total_weight = 0.0, first = 0.0;
            #pragma unroll
            for (int slot = 0; slot < NODES_PER_LANE; ++slot) {
                if (lane + PANEL * slot < node_count) {
                    const double node_weight = exp(weight[slot] - peak);
                    weight[slot] = node_weight;
                    total_weight += node_weight;
                    first += node_weight * h * held[slot];
                }
            }
            for (int offset = PANEL / 2; offset > 0; offset >>= 1) {
                total_weight += __shfl_xor_sync(FULL_MASK, total_weight, offset);
                first += __shfl_xor_sync(FULL_MASK, first, offset);
            }
            const double new_mean = first / total_weight;
            // pass 3: the central moments about the new mean
            double second = 0.0, third_sum = 0.0, fourth_sum = 0.0;
            #pragma unroll
            for (int slot = 0; slot < NODES_PER_LANE; ++slot) {
                if (lane + PANEL * slot < node_count) {
                    const double conditional_value = held[slot];
                    const double node_weight = weight[slot] / total_weight;
                    const double offset_value = h * conditional_value - new_mean;
                    const double offset_square = offset_value * offset_value;
                    second += node_weight * (conditional_value + offset_square);
                    third_sum += node_weight * (offset_square * offset_value + 3.0 * conditional_value * offset_value);
                    fourth_sum += node_weight * (offset_square * offset_square + 6.0 * conditional_value * offset_square + 3.0 * conditional_value * conditional_value);
                }
            }
            for (int offset = PANEL / 2; offset > 0; offset >>= 1) {
                second += __shfl_xor_sync(FULL_MASK, second, offset);
                third_sum += __shfl_xor_sync(FULL_MASK, third_sum, offset);
                fourth_sum += __shfl_xor_sync(FULL_MASK, fourth_sum, offset);
            }
            const double step = new_mean - old_mean;
            const double log_normalizer = peak + log(total_weight);
            if (lane == 0) {
                mean[at] = new_mean;
                variance[at] = second;
                third[at] = third_sum;
                fourth[at] = fourth_sum;
                shift[at] = h;
                step_out[member] = step;
                const double pull = h * new_mean;
                const double shrink = 0.5 * omega * (new_mean * new_mean + second);
                pieces[3 * at] = pull - shrink - log_normalizer;
                pieces[3 * at + 1] = square * second;
                pieces[3 * at + 2] = fabs(pull) + shrink + fabs(log_normalizer);
            }
            // x_i' r after x_member dm has left r, for the panel's other members
            if (lane < projected_width) c -= gram_entry * step;
        }
        __syncwarp();
        // u = A s, the covariates' share of the panel's move, for the next launch
        for (int covariate = lane; covariate < covariate_count; covariate += PANEL) {
            double moved = 0.0;
            for (int member = 0; member < projected_width; ++member) moved += coupling[(long long)covariate * PANEL + member] * step_out[member];
            covariate_steps[(long long)model * covariate_count + covariate] = moved;
        }
    }
}
"""

_BATCH_KERNELS: dict[tuple[int, int], tuple[Any, Any]] = {}


def _batch_kernels(cupy: Any, node_count: int) -> tuple[Any, Any]:
    """(node_terms, panel_step) for the current device, compiled once per node count: each lane holds its nodes'
    terms in registers, ceil(nodes / PANEL) of them."""
    device = int(cupy.cuda.runtime.getDevice())
    lanes = -(-int(node_count) // PANEL)
    key = (device, lanes)
    if key not in _BATCH_KERNELS:
        options = ("--std=c++11", f"-DPANEL={PANEL}", f"-DNODES_PER_LANE={lanes}")
        _BATCH_KERNELS[key] = (
            cupy.RawKernel(_BATCH_SOURCE, "node_terms", options=options),
            cupy.RawKernel(_BATCH_SOURCE, "panel_step", options=options),
        )
    return _BATCH_KERNELS[key]


def _panel_step_shared_bytes(threads: int) -> int:
    """``panel_step``'s dynamic shared memory: a weighted residual per thread, a sum per (warp, lane), one per warp, the
    previous panel's scaled steps of two models per lane and its members' code means, then each warp's tile of the
    panel's codes (PANEL rows of PANEL + 1 bytes). No share depends on the models, so neither does a block."""
    warps = threads // PANEL
    return (threads + warps * PANEL + warps + 2 * PANEL + PANEL) * np.dtype(np.float64).itemsize + warps * PANEL * (PANEL + 1)


class CudaPanelKernels:
    """``sweep_models``' two kernels on the current CUDA device (``_BATCH_SOURCE``)."""

    def __init__(self, cupy: Any, node_count: int) -> None:
        self.cupy = cupy
        self.node_count = int(node_count)

    def threads(self, sample_count: int) -> int:
        """A panel step's block: a share of the samples per multiprocessor (one block on each, every one busy, and as
        few partial sums for the chain to add as the device allows), in whole warps, within the kernel's own limit
        (``step_threads``) and the shared memory a block of it may take (48 KB less its static arrays unless raised:
        1,024 threads would need 50 KB). It depends on the samples alone, so a model's partial sums are over the same
        blocks of samples whichever models share its launches."""
        _terms, step = _batch_kernels(self.cupy, self.node_count)
        processors = int(self.cupy.cuda.Device().attributes["MultiProcessorCount"])
        share = -(-int(sample_count) // processors)
        threads = min(step_threads(step), max(PANEL, -(-share // PANEL) * PANEL))
        while threads > PANEL and _panel_step_shared_bytes(threads) > int(step.max_dynamic_shared_size_bytes):
            threads -= PANEL
        return threads

    def node_terms(self, arguments: tuple, total: int) -> None:
        terms, _step = _batch_kernels(self.cupy, self.node_count)
        attributes = self.cupy.cuda.Device().attributes
        threads = step_threads(terms)
        # a grid-stride loop needs no more blocks than the device keeps resident at once
        resident = int(attributes["MultiProcessorCount"]) * (int(attributes["MaxThreadsPerMultiProcessor"]) // threads)
        terms((max(1, min(-(-int(total) // threads), resident)),), (threads,), arguments)

    def panel_step(self, blocks: int, threads: int, models: int, arguments: tuple) -> None:
        _terms, step = _batch_kernels(self.cupy, self.node_count)
        step((blocks,), (threads,), arguments, shared_mem=_panel_step_shared_bytes(threads))


class HostPanelKernels:
    """The two kernels in numpy, operation for operation where numpy allows (no fused multiply-adds, its own order
    within the node and sample sums), for ``sweep_models`` on the host: the reference its CUDA kernels are checked
    against, with the same layouts, the same blocks of ``threads`` samples and the same partial sums."""

    def __init__(self, threads: int) -> None:
        if threads <= 0 or threads % PANEL:
            raise ValueError("a panel step's block is whole warps")
        self._threads = int(threads)

    def threads(self, sample_count: int) -> int:
        return self._threads

    def node_terms(self, arguments: tuple, total: int) -> None:
        log_scale, squares, class_index, log_density, grid, noise, first, width, models, _node_count, _class_count, conditional, base = arguments
        rows = slice(int(first), int(first) + int(width))
        omega = (squares[rows] / noise[None, :])[:, :, None]
        log_variance = log_scale[rows][:, :, None] + grid[None, None, :]
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            variance = np.exp(log_variance)
            ratio = variance * omega
            density = log_density[np.arange(int(models))[None, :], class_index[rows][:, None], :]
            overflow = np.isinf(ratio)
            conditional[: int(width)] = np.where(overflow, 1.0 / omega, np.where(variance > 0.0, 1.0 / (1.0 / variance + omega), 0.0))
            base[: int(width)] = np.where(overflow, density - 0.5 * (log_variance + np.log(omega)), density - 0.5 * np.log1p(ratio))

    def panel_step(self, blocks: int, threads: int, models: int, arguments: tuple) -> None:
        (codes, _stride, code_means, code_scales, rows, signs, applied, applied_width, projected, projected_width, covariates,
         covariate_count, masks, residuals, _models, sample_count, steps, covariate_steps, partials, partial_totals, _counter,
         conditional, base, node_first, _node_count, squares, noise, grams, couplings, panel, mean, variance, shift, third, fourth,
         pieces) = arguments
        count, models = int(sample_count), int(models)
        applied, applied_width, projected, projected_width = int(applied), int(applied_width), int(projected), int(projected_width)
        if applied_width:
            applied_rows = rows[applied:applied + applied_width].astype(np.int64)
            scaled_steps = signs[applied:applied + applied_width][None, :] * steps[:, :applied_width] / code_scales[applied_rows][None, :]
            for model in range(models):
                moved = np.zeros(count)
                for member in range(applied_width):
                    row = int(applied_rows[member])
                    moved += scaled_steps[model, member] * (codes[row, :count].astype(np.float64) - code_means[row])
                covariate = np.zeros(count)
                for index in range(int(covariate_count)):
                    covariate += covariates[index] * covariate_steps[model, index]
                residuals[model] -= moved * masks[model] - covariate * masks[model]
        if not projected_width:
            return
        code_rows = rows[projected:projected + projected_width].astype(np.int64)
        for model in range(models):
            weighted = masks[model] * residuals[model]
            for block in range(int(blocks)):
                samples = slice(block * threads, min((block + 1) * threads, count))
                partials[model, block, :projected_width] = codes[code_rows, samples].astype(np.float64) @ weighted[samples]
                partial_totals[model, block] = np.sum(weighted[samples])
        for model in range(models):
            total = np.sum(partial_totals[model, : int(blocks)])
            product = np.sum(partials[model, : int(blocks), :projected_width], axis=0)
            c = signs[projected:projected + projected_width] * (product - code_means[code_rows] * total) / code_scales[code_rows]
            gram, coupling, model_noise = grams[int(panel), model], couplings[int(panel), model], float(noise[model])
            for member in range(projected_width):
                at = projected + member
                square = float(squares[at, model])
                omega = square / model_noise
                old_mean = float(mean[at, model])
                h = (c[member] + square * old_mean) / model_noise
                held = conditional[int(node_first) + member, model]
                log_weight = base[int(node_first) + member, model] + 0.5 * h * h * held
                peak = np.max(log_weight)
                weight = np.exp(log_weight - peak)
                total_weight = np.sum(weight)
                new_mean = np.sum(weight * h * held) / total_weight
                normalized = weight / total_weight
                offset = h * held - new_mean
                square_offset = offset * offset
                second = np.sum(normalized * (held + square_offset))
                third_sum = np.sum(normalized * (square_offset * offset + 3.0 * held * offset))
                fourth_sum = np.sum(normalized * (square_offset * square_offset + 6.0 * held * square_offset + 3.0 * held * held))
                step = new_mean - old_mean
                log_normalizer = peak + np.log(total_weight)
                mean[at, model], variance[at, model], third[at, model], fourth[at, model], shift[at, model] = new_mean, second, third_sum, fourth_sum, h
                steps[model, member] = step
                pull = h * new_mean
                shrink = 0.5 * omega * (new_mean * new_mean + second)
                pieces[at, model] = (pull - shrink - log_normalizer, square * second, abs(pull) + shrink + abs(log_normalizer))
                c = c - gram[:projected_width, member] * step
            covariate_steps[model, : int(covariate_count)] = coupling[: int(covariate_count), :projected_width] @ steps[model, :projected_width]


@dataclass(frozen=True)
class _BlockLayout:
    """One LD block of a sweep: its device (a position in ``SweepLayout.devices``), the first position of its members
    in that device's layout, its first group column, its members in sweep order and its pieces (member ranges)."""

    device: int
    offset: int
    start: int
    members: np.ndarray
    pieces: tuple[tuple[int, int], ...]


@dataclass
class _DeviceLayout:
    """One device's members, its blocks' in block order (host), and on the device each one's code row in its block's
    tile, its tie sign and its prior class; the covariates there (n x k, and k x n for the residual's move), copied at a
    sweep's first use."""

    device_id: int
    members: np.ndarray
    rows: Any
    signs: Any
    classes: Any
    covariates: Any = None
    covariates_t: Any = None


class SweepLayout:
    """Where a sweep's members live: each block's device, its members in sweep order and its pieces, and each device's
    members laid out in the order it visits them, so that a panel is a run of consecutive positions there and its
    launch takes offsets, never a gather. Fixed for an oracle (its member order, the source's blocks and devices)."""

    def __init__(
        self, array_module: Any, devices: Sequence[int], member_blocks: Sequence[np.ndarray], bounds: Sequence[tuple[int, int]],
        group: np.ndarray, sign: np.ndarray, class_index: np.ndarray, pieces: Callable[[int], Iterator[tuple[int, int]]],
    ) -> None:
        if not (len(devices) == len(member_blocks) == len(bounds)):
            raise ValueError("a sweep layout needs one device and one member list per block")
        positions: dict[int, int] = {}
        counts: list[int] = []
        held: list[list[np.ndarray]] = []
        code_rows: list[list[np.ndarray]] = []
        self.blocks: list[_BlockLayout] = []
        for device_id, members, (start, _stop) in zip(devices, member_blocks, bounds):
            position = positions.setdefault(int(device_id), len(counts))
            if position == len(counts):
                counts.append(0)
                held.append([])
                code_rows.append([])
            members = np.asarray(members, dtype=np.int64)
            self.blocks.append(_BlockLayout(position, counts[position], int(start), members, tuple(pieces(int(members.shape[0])))))
            counts[position] += int(members.shape[0])
            held[position].append(members)
            code_rows[position].append(np.asarray(group, dtype=np.int64)[members] - int(start))
        self.devices: list[_DeviceLayout] = []
        for device_id, position in positions.items():
            members = np.concatenate(held[position])
            with _cupy_device_context(array_module, device_id):
                self.devices.append(_DeviceLayout(
                    device_id, members, array_module.asarray(np.concatenate(code_rows[position])),
                    array_module.asarray(np.asarray(sign, dtype=np.float64)[members]), array_module.asarray(np.asarray(class_index, dtype=np.int64)[members]),
                ))
        self.widest_piece = max((last - first for block in self.blocks for first, last in block.pieces), default=0)

    @property
    def panel_count(self) -> int:
        return sum(-(-(last - first) // PANEL) for block in self.blocks for first, last in block.pieces)


class NoCodes(TypeError):
    """A block's tile holds no codes (``CodePanels.supports``): its sweeps decode it (``sweep_piece``)."""


def _piece_grams(
    xp: Any, tile: Any, rows: Any, signs: Any, mask: Any, masked_covariates: Any, covariate_pinv: Any, columns: int,
) -> tuple[Any, Any]:
    """One model's panel Grams and couplings of a piece (members ``rows`` of ``tile``, signed), (panels, PANEL, PANEL)
    and (panels, k, PANEL), each panel's zero-padded to PANEL: ``sweep_piece``'s Gram and coupling, A = (C'M^2 C)^+
    C'M^2 X_P and Xp_P' Xp_P, the columns decoded ``columns`` at a time (a multiple of PANEL) and each chunk's Grams
    one batched product (a GEMM per panel was three submissions per 32 columns: 280k at 1.0e6 members x 3 models)."""
    width = int(rows.shape[0])
    sample_count = int(mask.shape[0])
    covariate_count = int(masked_covariates.shape[1])
    panels = -(-width // PANEL)
    grams = xp.empty((panels, PANEL, PANEL))
    couplings = xp.empty((panels, covariate_count, PANEL))
    for first in range(0, width, columns):
        last = min(first + columns, width)
        count = -(-(last - first) // PANEL)
        dense = xp.asarray(tile.columns(rows[first:last]), dtype=xp.float64) * signs[first:last][None, :]
        projected = dense * mask[:, None]
        del dense
        # The chunk's columns padded to whole panels with zero columns: a short last panel's Gram and coupling are then
        # zero beyond its members, and every panel's Gram is one product of a batch. At most two (n x columns) arrays
        # live at once beside the decode's own.
        coupling = xp.zeros((covariate_count, count * PANEL))
        coupling[:, : last - first] = covariate_pinv @ (masked_covariates.T @ projected)
        projected -= masked_covariates @ coupling[:, : last - first]
        stacked = xp.zeros((count, PANEL, sample_count))
        stacked.reshape(count * PANEL, sample_count)[: last - first] = projected.T
        del projected
        grams[first // PANEL:first // PANEL + count] = xp.matmul(stacked, stacked.transpose(0, 2, 1))
        del stacked
        couplings[first // PANEL:first // PANEL + count] = coupling.reshape(covariate_count, count, PANEL).transpose(1, 0, 2)
    return grams, couplings


def sweep_models(
    xp: Any, kernels: Any, layout: SweepLayout, tiles: Iterator[tuple[int, int, Any, int]], *, grams: PanelGrams, keys: Sequence[tuple],
    masks: np.ndarray, residuals: np.ndarray, noise: np.ndarray, log_density: np.ndarray, grid: np.ndarray, log_scale: np.ndarray,
    squares: np.ndarray, state: dict[str, np.ndarray], covariates: Any, covariate_pinvs: Sequence[Any], gram_columns: Callable[[int], int],
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """One sweep of each of M models over every member, in place on the host arrays ``state`` (name -> (members, M):
    mean, variance, shift, third, fourth); returns (each model's residual (n), the ELBO pieces' totals (M,
    PIECE_COLUMNS), the residuals' squares (M)), on the host.

    ``tiles`` yields (start, stop, tile, device) per block in order (``device_plan.native_blocks``), each tile a
    ``code_products.CodeBlockTile`` on its device; ``layout`` places the members. Per model: ``keys`` its panel Grams'
    name (model, metric), ``masks`` M (n), ``residuals`` r (n, in the projected space), ``noise`` sigma^2,
    ``log_density`` its classes' log pi (C x K) on the lattice ``grid``, ``log_scale`` log u_j and ``squares``
    ||xp_j||^2 per member (members x M), ``covariate_pinvs`` (C'M^2 C)^+ (k x k). ``covariates`` C (n x k) and the
    pinvs live on the caller's device. Each block's work runs on its device's current stream, the residuals handed
    from one block's device to the next (``moved``), the panel Grams built there on a miss (``gram_columns(width)``
    columns decoded at once) and named by that device's pool.

    The arithmetic is ``sweep_piece``'s with ``CodePanels`` per member and per panel, the same piece and panel
    boundaries and member order; its sums are in other orders: c_P over each block's warps of 32 samples, the blocks
    then in order (``panel_project``: one tree over 1,024 threads), u = A s in member order (a cuBLAS GEMV), the
    covariates' move as (C u) M (as (M C) u: the same bits for a 0/1 mask), the panel Grams in batched GEMMs over
    several panels' columns, and the ELBO pieces and the residual squares summed on the host, the devices in order.
    Within a panel's chain the node terms and moments are ``panel_sweep``'s expressions in its order. Every sum is a
    model's own, in an order set by the samples, the members and the devices alone, so a model's sweep is the same to
    the bit whichever models share it."""
    model_count = len(keys)
    sample_count = int(masks.shape[1])
    node_count = int(grid.shape[0])
    class_count = int(log_density.shape[1])
    covariate_count = int(covariates.shape[1])
    runtime = getattr(getattr(xp, "cuda", None), "runtime", None)
    home_id = int(runtime.getDevice()) if runtime is not None else 0
    if set(state) != set(MOMENTS):
        raise ValueError(f"a sweep's state is its members' {', '.join(MOMENTS)}")
    arrays, scratch = [], []
    for device in layout.devices:
        with _cupy_device_context(xp, device.device_id):
            members = device.members

            def put(values: np.ndarray) -> Any:
                return xp.asarray(np.ascontiguousarray(values))

            held = {name: put(state[name][members]) for name in MOMENTS}
            held.update(
                log_scale=put(log_scale[members]), squares=put(squares[members]), pieces=xp.zeros((members.shape[0], model_count, PIECE_COLUMNS)),
                masks=put(masks), noise=put(noise), log_density=put(log_density), grid=put(grid),
            )
            if device.covariates is None:
                device.covariates = moved(xp, covariates, device.device_id)
                device.covariates_t = xp.ascontiguousarray(device.covariates.T)
            threads = kernels.threads(sample_count)
            blocks = -(-sample_count // threads)
            arrays.append(held)
            scratch.append(dict(
                threads=threads, blocks=blocks, steps=xp.zeros((model_count, PANEL)), covariate_steps=xp.zeros((model_count, covariate_count)),
                partials=xp.zeros((model_count, blocks, PANEL)), partial_totals=xp.zeros((model_count, blocks)), counter=xp.zeros(1, dtype=xp.uint32),
                conditional=xp.empty((layout.widest_piece, model_count, node_count)), base=xp.empty((layout.widest_piece, model_count, node_count)),
                masked_covariates={}, pinvs={},
            ))
    residual: Any = np.array(residuals, dtype=np.float64)
    for block, (start, _stop, tile, device_id) in zip(layout.blocks, tiles, strict=True):
        device = layout.devices[block.device]
        if int(device_id) != device.device_id or int(start) != block.start:
            raise ValueError("a block's tile is not where the sweep's layout places it")
        if not CodePanels.supports(tile):
            raise NoCodes("a block's tile holds no codes for the panel sweep")
        held, work = arrays[block.device], scratch[block.device]
        pool = device_pool(device.device_id)
        with _cupy_device_context(xp, device.device_id):
            residual = moved(xp, residual, device.device_id)
            codes = tile.aligned_codes
            # Every launch of the block reads these; only the panels' positions (and each piece's Grams) change.
            fixed = (codes, np.int64(codes.shape[1]), xp.ascontiguousarray(tile.means), xp.ascontiguousarray(tile.scales), device.rows, device.signs)
            moving = (
                device.covariates_t, np.int32(covariate_count), held["masks"], residual, np.int32(model_count), np.int32(sample_count),
                work["steps"], work["covariate_steps"], work["partials"], work["partial_totals"], work["counter"], work["conditional"], work["base"],
            )
            members_out = tuple(held[name] for name in MOMENTS) + (held["pieces"],)
            applied, applied_width = 0, 0
            for first, last in block.pieces:
                offset, width = block.offset + first, last - first
                piece = block.members[first:last]
                piece_rows, piece_signs = device.rows[offset:offset + width], device.signs[offset:offset + width]
                parts = []
                for index, key in enumerate(keys):

                    def build(index: int = index) -> tuple[Any, Any]:
                        if index not in work["masked_covariates"]:
                            work["masked_covariates"][index] = device.covariates * held["masks"][index][:, None]
                            work["pinvs"][index] = moved(xp, covariate_pinvs[index], device.device_id)
                        return _piece_grams(
                            xp, tile, piece_rows, piece_signs, held["masks"][index], work["masked_covariates"][index], work["pinvs"][index],
                            gram_columns(width),
                        )

                    # A piece's stack, named apart from ``sweep_piece``'s one-panel Grams (a piece of at most PANEL members
                    # has a panel's members).
                    parts.append(grams.get((*key, "piece", np.ascontiguousarray(piece, dtype=np.int64).tobytes()), build, pool=pool))
                piece_grams = xp.ascontiguousarray(xp.stack([part[0] for part in parts], axis=1))
                piece_couplings = xp.ascontiguousarray(xp.stack([part[1] for part in parts], axis=1))
                del parts
                kernels.node_terms(
                    (held["log_scale"], held["squares"], device.classes, held["log_density"], held["grid"], held["noise"], np.int64(offset),
                     np.int32(width), np.int32(model_count), np.int32(node_count), np.int32(class_count), work["conditional"], work["base"]),
                    width * model_count * node_count,
                )
                chain = (np.int32(node_count), held["squares"], held["noise"], piece_grams, piece_couplings)
                for panel_first in range(0, width, PANEL):
                    projected, projected_width = offset + panel_first, min(PANEL, width - panel_first)
                    kernels.panel_step(work["blocks"], work["threads"], model_count, (
                        *fixed, np.int64(applied), np.int32(applied_width), np.int64(projected), np.int32(projected_width), *moving,
                        np.int64(panel_first), *chain, np.int64(panel_first // PANEL), *members_out,
                    ))
                    applied, applied_width = projected, projected_width
            if applied_width:
                # The block's last panel's move: no panel follows on these codes (the next block's first launch projects).
                kernels.panel_step(work["blocks"], work["threads"], model_count, (
                    *fixed, np.int64(applied), np.int32(applied_width), np.int64(0), np.int32(0), *moving, np.int64(0), *chain, np.int64(0),
                    *members_out,
                ))
    totals = np.zeros((model_count, PIECE_COLUMNS))
    for device, held in zip(layout.devices, arrays):
        with _cupy_device_context(xp, device.device_id):
            for name in MOMENTS:
                state[name][device.members] = _to_host(held[name])
            pieces = _to_host(held["pieces"])
        for index in range(model_count):
            # Each model's own column summed alone, in one order whichever models share the sweep.
            totals[index] += np.ascontiguousarray(pieces[:, index]).sum(axis=0)
    with _cupy_device_context(xp, int(_device_of(residual, home_id))):
        swept = _to_host(residual)
    residuals = [np.array(swept[index]) for index in range(model_count)]
    return residuals, totals, np.array([float(values @ values) for values in residuals])


def _to_host(values: Any) -> np.ndarray:
    return np.asarray(values.get() if hasattr(values, "get") else values)


def _device_of(values: Any, default: int) -> int:
    """The device an array lives on (``default`` for a host array)."""
    device = getattr(values, "device", None)
    return int(device.id) if hasattr(device, "id") else int(default)

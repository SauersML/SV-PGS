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
"""

from __future__ import annotations

from typing import Any

import numpy as np

PANEL = 32
"""Columns per panel: the CUDA warp width (one lane per member of the panel)."""
PIECE_COLUMNS = 3
"""Per member: its KL term, ||x_j||^2 v_j, and the KL term's pieces' sizes (the ELBO and its rounding bound)."""

_SOURCE = r"""
extern "C" __global__ void panel_sweep(
    const double* __restrict__ gram, double* __restrict__ projection, const double* __restrict__ squares,
    const long long* __restrict__ class_index, const double* __restrict__ log_density, const int node_count,
    const double* __restrict__ node_variance, const double* __restrict__ log_node_variance, const double noise,
    const int width, double* __restrict__ mean, double* __restrict__ variance, double* __restrict__ shift,
    double* __restrict__ third, double* __restrict__ fourth, double* __restrict__ step_out, double* __restrict__ pieces)
{
    const int lane = threadIdx.x;
    const unsigned full = 0xffffffffu;
    double c = lane < width ? projection[lane] : 0.0;
    for (int member = 0; member < width; ++member) {
        const double square = squares[member];
        const double omega = square / noise;
        const double old_mean = mean[member];
        const double h = (__shfl_sync(full, c, member) + square * old_mean) / noise;
        const double* density = log_density + class_index[member] * (long long)node_count;
        const double* variance_row = node_variance + member * (long long)node_count;
        const double* log_variance_row = log_node_variance + member * (long long)node_count;
        // pass 1: the peak of the log weights
        double peak = __longlong_as_double(0xfff0000000000000ULL);  // -inf: NVRTC has no INFINITY
        for (int node = lane; node < node_count; node += 32) {
            const double variance_node = variance_row[node];
            const double ratio = variance_node * omega;
            double log_weight;
            if (isinf(ratio)) {
                const double conditional = 1.0 / omega;
                log_weight = density[node] - 0.5 * (log_variance_row[node] + log(omega)) + 0.5 * h * h * conditional;
            } else {
                const double conditional = variance_node > 0.0 ? 1.0 / (1.0 / variance_node + omega) : 0.0;
                log_weight = density[node] - 0.5 * log1p(ratio) + 0.5 * h * h * conditional;
            }
            peak = fmax(peak, log_weight);
        }
        for (int offset = 16; offset > 0; offset >>= 1) peak = fmax(peak, __shfl_xor_sync(full, peak, offset));
        // pass 2: the normalizer and the mean
        double total = 0.0, first = 0.0;
        for (int node = lane; node < node_count; node += 32) {
            const double variance_node = variance_row[node];
            const double ratio = variance_node * omega;
            double conditional, log_weight;
            if (isinf(ratio)) {
                conditional = 1.0 / omega;
                log_weight = density[node] - 0.5 * (log_variance_row[node] + log(omega)) + 0.5 * h * h * conditional;
            } else {
                conditional = variance_node > 0.0 ? 1.0 / (1.0 / variance_node + omega) : 0.0;
                log_weight = density[node] - 0.5 * log1p(ratio) + 0.5 * h * h * conditional;
            }
            const double weight = exp(log_weight - peak);
            total += weight;
            first += weight * h * conditional;
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            total += __shfl_xor_sync(full, total, offset);
            first += __shfl_xor_sync(full, first, offset);
        }
        const double new_mean = first / total;
        // pass 3: the central moments about the new mean
        double second = 0.0, third_sum = 0.0, fourth_sum = 0.0;
        for (int node = lane; node < node_count; node += 32) {
            const double variance_node = variance_row[node];
            const double ratio = variance_node * omega;
            double conditional, log_weight;
            if (isinf(ratio)) {
                conditional = 1.0 / omega;
                log_weight = density[node] - 0.5 * (log_variance_row[node] + log(omega)) + 0.5 * h * h * conditional;
            } else {
                conditional = variance_node > 0.0 ? 1.0 / (1.0 / variance_node + omega) : 0.0;
                log_weight = density[node] - 0.5 * log1p(ratio) + 0.5 * h * h * conditional;
            }
            const double weight = exp(log_weight - peak) / total;
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
        self.project_kernel(
            (last - first,), (PROJECT_THREADS,),
            (self.codes, np.int64(self.stride), self.rows[first:last], np.int32(self.sample_count), mask, residual,
             self.means, self.scales, self.signs[first:last], out),
        )
        return out

    def apply(self, first: int, last: int, step: Any, mask: Any, covariates: Any, covariate_step: Any, residual: Any) -> None:
        threads = PROJECT_THREADS
        self.apply_kernel(
            (-(-self.sample_count // threads),), (threads,),
            (self.codes, np.int64(self.stride), self.rows[first:last], np.int32(last - first), np.int32(self.sample_count),
             step, self.means, self.scales, self.signs[first:last], mask, covariates, covariate_step,
             np.int32(covariates.shape[1]), residual),
        )


class PanelGrams:
    """Each panel's projected Gram Xp_P' Xp_P (float64, width x width) and covariate coupling, formed on the device
    and kept for later sweeps as a cache of the shared ledger (``memory_broker``): the design is fixed through a fit,
    so a kept panel is never rebuilt. A panel is admitted only from what the device's pool has left once the fit's
    mandatory buffers are reserved, and a later mandatory lease evicts it (the panel is then rebuilt when met), so the
    cache never takes what a mandatory buffer needs. A panel is named by its model and its members' own indices in
    sweep order, so a panel of another order or of other members never reads it (keyed by a piece's first member plus
    an offset, two panels of a random within-block order could share a key and one read the other's Gram). Sweeps
    visit the panels in one cycle, where any eviction order misses every evicted panel once per sweep, so the cache
    keeps the panels it met first and rebuilds the rest. With no ``broker`` every panel is kept."""

    def __init__(self, broker: Any = None, pool: str | None = None) -> None:
        self._grams: dict[tuple, Any] = {}
        self._leases: dict[tuple, Any] = {}
        self._broker = broker
        self._pool = pool

    def get(self, key: tuple, build) -> Any:
        held = self._grams.get(key)
        if held is not None:
            return held
        built = build()
        if self._broker is None:
            self._grams[key] = built
            return built
        lease = self._broker.admit(self._pool, sum(int(part.nbytes) for part in built), "a panel Gram", lambda key=key: self._evict(key), allocated=True)
        if lease is not None:
            self._grams[key] = built
            self._leases[key] = lease
        return built

    def _evict(self, key: tuple) -> None:
        self._grams.pop(key, None)
        self._leases.pop(key, None)

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
    node_count = int(node_variance.shape[1])
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
        kernel(
            (1,), (PANEL,),
            (
                gram, projection, squares[first:last], class_index[first:last], log_density, np.int32(node_count),
                node_variance[first:last], log_node_variance[first:last], np.float64(noise), np.int32(last - first),
                mean[first:last], variance[first:last], shift[first:last], third[first:last], fourth[first:last], step,
                pieces[first:last],
            ),
        )
        if panels is None:
            residual -= (columns @ step) * mask - covariates @ (coupling @ step)
        else:
            panels.apply(first, last, step, mask, covariates, cupy.ascontiguousarray(coupling @ step), residual)

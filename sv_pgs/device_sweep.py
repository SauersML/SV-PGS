"""The mean-field coordinate-ascent sweep on a CUDA device, for the streamed full-data route.

The host sweep (``mean_field._sweep``) updates each q_j in order from c_j = x_j' r, then moves the residual r by
x_j dm_j, which costs one pass over the n samples per coordinate twice over. Run on a streamed store it also needs the
projected columns on the host: at 40,000 samples x 518k records, 166 GB copied off the device every sweep.

Here the same updates, in the same order, run on the device a panel of ``PANEL`` columns at a time. For a panel
with projected columns Xp_P:

  c_P = X_P' r            (r already lies in the projected, training-masked space, so Xp_P' r = X_P' r);
  within the panel, member j's update reads c_j and then c_i -= (Xp_P' Xp_P)_ij dm_j for the panel's members,
  which is exactly x_i' r after x_j dm_j has left it: one warp, one lane per member, the node sums by shuffles;
  r -= (I - H) M X_P dm_P  (M the training mask, H the covariate projector).

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

_KERNELS: dict[int, Any] = {}


def _kernel(cupy: Any) -> Any:
    device = int(cupy.cuda.runtime.getDevice())
    if device not in _KERNELS:
        _KERNELS[device] = cupy.RawKernel(_SOURCE, "panel_sweep", options=("--std=c++11",))
    return _KERNELS[device]


class PanelGrams:
    """Each panel's projected Gram Xp_P' Xp_P (float64, width x width) and covariate coupling, formed on the device
    and kept for later sweeps while their bytes fit ``capacity_bytes``: the design is fixed through a fit, so a kept
    panel is never rebuilt. A panel is named by its model and its members' own indices in sweep order, so a panel of
    another order or of other members never reads it (keyed by a piece's first member plus an offset, two panels of a
    random within-block order could share a key and one read the other's Gram). Sweeps visit the panels in one cycle,
    where any eviction order misses every evicted panel once per sweep, so the cache keeps the panels it met first
    and rebuilds the rest; ``None`` capacity keeps every panel."""

    def __init__(self, capacity_bytes: int | None = None) -> None:
        self._grams: dict[tuple[int, bytes], Any] = {}
        self._capacity = capacity_bytes
        self._bytes = 0

    def get(self, key: tuple[int, bytes], build) -> Any:
        held = self._grams.get(key)
        if held is not None:
            return held
        built = build()
        size = sum(int(part.nbytes) for part in built)
        if self._capacity is None or self._bytes + size <= self._capacity:
            self._grams[key] = built
            self._bytes += size
        return built

    def clear(self) -> None:
        self._grams.clear()
        self._bytes = 0


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
) -> None:
    """One sweep over a piece's columns in order, in place on the device.

    ``decode(first, last)`` returns the piece's standardized columns first..last-1 (n x panel, before masking and
    projection; one panel at a time, so the device holds a panel's columns, never a block's), ``width`` the piece's
    member count, ``mask`` the model's
    training indicator (n), ``covariates`` its masked covariates M C (n x k) and ``covariate_pinv`` (C'MC)^+ (k x k):
    the complement of M v is (I - H) M v = M v - M C (C'MC)^+ C' M v (the noise's weights cancel), so each panel keeps
    A = (C'MC)^+ C' M X_panel (k x panel) beside its Gram and a sweep's residual update is M X s - M C (A s), two
    small products in place of a general projection per panel (35% of a bench-sim Stage 2 [sim, scenario_001]);
    ``residual`` r (n);
    the per-member arrays are the piece's own slices, and ``pieces`` (width x 3) receives each member's KL term,
    ||x_j||^2 v_j and the terms' sizes. ``model`` and ``members`` (the piece's member indices, host, in sweep order)
    name each panel for the Gram cache."""
    kernel = _kernel(cupy)
    node_count = int(node_variance.shape[1])
    for first in range(0, width, PANEL):
        last = min(first + PANEL, width)
        columns = decode(first, last)

        def build(columns=columns):
            masked = columns * mask[:, None]
            coupling = cupy.ascontiguousarray(covariate_pinv @ (covariates.T @ masked))
            projected = masked - covariates @ coupling
            return cupy.ascontiguousarray(projected.T @ projected), coupling

        gram, coupling = grams.get((int(model), np.ascontiguousarray(members[first:last], dtype=np.int64).tobytes()), build)
        projection = cupy.ascontiguousarray(columns.T @ residual)
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
        residual -= (columns @ step) * mask - covariates @ (coupling @ step)

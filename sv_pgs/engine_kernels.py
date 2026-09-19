"""The engine's per-variant loops over the mixing lattice as fused CUDA kernels.

``scale_mixture_ep`` evaluates, for every effect j of class c and every lattice node k with variance
v = u_j e^t_k (v = 0 below the kernel floor), the tilted component

    log pi_ck - log1p(v P_j) / 2 + h_j^2 v / (2 (1 + v P_j)),

and reduces over k. On the host that is chunks of (rows x K) float64 arrays; here one thread holds one
effect and walks its nodes once, so nothing of size rows x K exists except the M-step's Gram operands.

The arithmetic is float64 and follows the host's (``scale_mixture_ep._kernel_terms``) with fewer
operations, each change bounded to a few units of rounding: v is e^(log u) e^t from precomputed factors
while all three are normal numbers (else exp(log u + t), as on the host), and r = 1/(1 + q) is the node's
only division, with v r and q r its products while r is normal (else the host's reciprocal forms, which
take the limits of an overflowing v or q). A component is pi e^(a/2) sqrt(r), a = h^2 v r, so a node costs a
sqrt and one exp where the host takes log1p and exp. A negative cavity precision is allowed while every
1 + v P stays positive; 1 + v P <= 0 anywhere is an error, as on the host.

The reductions are single passes: a running log-sum-exp over the exponents log pi + a/2, whose earlier weights
are rescaled when the largest moves (one exp per node either way), with weighted Welford updates of the moments. The M-step's density
block is written through the deviations D = r - pi: its gradient sum_j D_j and Hessian
diag(sum D) - D'D - (sum D) pi' - pi (sum D)' equal the host's sum r - n pi and
diag(sum r) - r'r - n (diag pi - pi pi') exactly, without their cancellation. Its Gram products (D'D and
G'S over a long inner dimension of rows) are split into a batch of rows per multiprocessor, one GEMM each.

Nothing here imports the engine: callers pass arrays and get arrays.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any, Sequence

import numpy as np

_SOURCE = r"""
// One effect's terms at one node at or above the kernel floor: r = 1/(1 + q), q r, v r and a = h^2 v r, with q = v P.
// ``scale_exp`` is e^(log u) and ``node_exp`` e^t.
__device__ __forceinline__ void node_terms(
    const double log_scale_value, const double scale_exp, const double node, const double node_exp,
    const double precision, const double shift_square, const double tiny, const double huge,
    double* retained, double* ratio_retained, double* conditional, double* signal, int* improper)
{
    double variance = scale_exp * node_exp;
    if (!(scale_exp >= tiny && scale_exp <= huge && node_exp >= tiny && node_exp <= huge && variance >= tiny && variance <= huge))
        variance = exp(log_scale_value + node);
    const double ratio = variance * precision;
    if (ratio <= -1.0) *improper = 1;
    *retained = 1.0 / (1.0 + ratio);
    if (fabs(ratio) * tiny < 0.5) {
        *ratio_retained = ratio * *retained;
        *conditional = variance * *retained;
    } else {
        *ratio_retained = 1.0 / (1.0 + 1.0 / ratio);
        *conditional = 1.0 / (1.0 / variance + precision);
    }
    *signal = shift_square * *conditional;
}

// The running log-sum-exp over e^(exponent) sqrt(r): the node's weight relative to the largest exponent so far (a
// finite one), rescaling the running sums when it moves.
__device__ __forceinline__ double running_weight(const double exponent, double* peak, double* total, double* square)
{
    if (exponent > *peak) {
        const double rescale = exp(*peak - exponent);
        *total *= rescale;
        *square *= rescale;
        *peak = exponent;
        return 1.0;
    }
    return exp(exponent - *peak);
}
"""

_SOURCE += r"""
extern "C" __global__ void tilted_rows(
    const long long rows, const int node_count,
    const long long* __restrict__ class_index, const double* __restrict__ log_density,
    const double* __restrict__ nodes, const double* __restrict__ node_exp, const double floor,
    const double tiny, const double huge, const double log_zero,
    const double* __restrict__ log_scale, const double* __restrict__ precision, const double* __restrict__ shift,
    double* __restrict__ log_normalizer, double* __restrict__ mean, double* __restrict__ variance,
    int* __restrict__ improper)
{
    const long long row = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= rows) return;
    const double* mass = log_density + class_index[row] * (long long)node_count;
    const double scale_value = log_scale[row], scale_exp = exp(scale_value), cavity_precision = precision[row];
    const double shift_value = shift[row], shift_square = shift_value * shift_value;
    int bad = 0;
    double peak = log_zero, total = 0.0, running_mean = 0.0, running_square = 0.0;
    for (int node = 0; node < node_count; ++node) {
        double exponent = mass[node], root = 1.0, conditional = 0.0;
        if (nodes[node] >= floor) {
            double retained, ratio_retained, signal;
            node_terms(scale_value, scale_exp, nodes[node], node_exp[node], cavity_precision, shift_square, tiny, huge,
                       &retained, &ratio_retained, &conditional, &signal, &bad);
            exponent += 0.5 * signal;
            root = sqrt(retained);
        }
        if (exponent == log_zero || root == 0.0) continue;
        const double weight = running_weight(exponent, &peak, &total, &running_square) * root;
        const double updated = total + weight;
        const double deviation = conditional - running_mean;
        const double step = deviation * (weight / updated);
        running_mean += step;
        running_square += total * deviation * step;
        total = updated;
    }
    log_normalizer[row] = peak + log(total);
    mean[row] = shift_value * running_mean;
    variance[row] = running_mean + shift_square * (running_square / total);
    if (bad) *improper = 1;
}

// The M-step's per-effect terms for rows of one class: deviations D = r - pi and centred derivatives G = r (d1 - mean d1)
// in batches of ``width`` rows, node-major within a batch ([(batch * K + node) * width + row % width]) so a warp's stores
// coalesce and each batch is one GEMM operand; and per row log Z, mean d1 and Var_r(d1) + E_r[d2]. The first pass
// parks each node's exponent in D and d1 in G; the second recomputes sqrt(r) and finishes them.
extern "C" __global__ void objective_rows(
    const long long rows, const long long width, const int node_count,
    const double* __restrict__ class_log_density, const double* __restrict__ class_density,
    const double* __restrict__ nodes, const double* __restrict__ node_exp, const double floor,
    const double tiny, const double huge, const double log_zero,
    const double* __restrict__ log_scale, const double* __restrict__ precision, const double* __restrict__ shift,
    double* __restrict__ deviations, double* __restrict__ centred,
    double* __restrict__ log_normalizer, double* __restrict__ mean_first, double* __restrict__ curvature,
    int* __restrict__ improper)
{
    const long long row = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= rows) return;
    const double scale_value = log_scale[row], scale_exp = exp(scale_value), cavity_precision = precision[row];
    const double shift_square = shift[row] * shift[row];
    const long long base = (row / width) * node_count * width + row % width;
    int bad = 0;
    double retained, ratio_retained, conditional, signal;
    double peak = log_zero, total = 0.0, first_mean = 0.0, first_square = 0.0, second_mean = 0.0;
    for (int node = 0; node < node_count; ++node) {
        double exponent = class_log_density[node], root = 1.0, first = 0.0, second = 0.0;
        if (nodes[node] >= floor) {
            node_terms(scale_value, scale_exp, nodes[node], node_exp[node], cavity_precision, shift_square, tiny, huge,
                       &retained, &ratio_retained, &conditional, &signal, &bad);
            exponent += 0.5 * signal;
            root = sqrt(retained);
            first = 0.5 * (retained * signal - ratio_retained);
            second = 0.5 * signal * retained * (2.0 * retained - 1.0) - 0.5 * ratio_retained * retained;
        }
        const long long at = base + node * width;
        deviations[at] = exponent;
        centred[at] = first;
        if (exponent == log_zero || root == 0.0) continue;
        const double weight = running_weight(exponent, &peak, &total, &first_square) * root;
        const double updated = total + weight;
        const double share = weight / updated;
        const double deviation = first - first_mean;
        const double step = deviation * share;
        first_mean += step;
        first_square += total * deviation * step;
        second_mean += (second - second_mean) * share;
        total = updated;
    }
    const double log_total = peak + log(total);
    for (int node = 0; node < node_count; ++node) {
        const long long at = base + node * width;
        double root = 1.0;
        if (nodes[node] >= floor) {
            node_terms(scale_value, scale_exp, nodes[node], node_exp[node], cavity_precision, shift_square, tiny, huge,
                       &retained, &ratio_retained, &conditional, &signal, &bad);
            root = sqrt(retained);
        }
        const double responsibility = root > 0.0 ? exp(deviations[at] - log_total) * root : 0.0;
        deviations[at] = responsibility - class_density[node];
        centred[at] = responsibility * (centred[at] - first_mean);
    }
    log_normalizer[row] = log_total;
    mean_first[row] = first_mean;
    curvature[row] = first_square / total + second_mean;
    if (bad) *improper = 1;
}
"""

_KERNELS: dict[tuple[int, int, str], tuple[Any, int]] = {}
_FLOAT64 = np.finfo(np.float64)


def _occupancy_block(cupy: ModuleType, kernel: Any) -> int:
    """The warp multiple that maximizes resident warps per SM for ``kernel``, from the device's per-SM limits and the
    kernel's register count; ties go to the smallest block, which leaves the most blocks to balance across SMs."""
    attributes = cupy.cuda.Device().attributes
    warp = int(attributes["WarpSize"])
    registers = max(int(kernel.num_regs), 1)
    best, best_warps = warp, -1
    for block in range(warp, int(kernel.max_threads_per_block) + 1, warp):
        resident_blocks = min(
            int(attributes["MaxThreadsPerMultiProcessor"]) // block,
            int(attributes["MaxRegistersPerMultiprocessor"]) // (registers * block),
            int(attributes["MaxBlocksPerMultiprocessor"]),
        )
        resident_warps = resident_blocks * block // warp
        if resident_warps > best_warps:
            best, best_warps = block, resident_warps
    return best


def _kernel(cupy: ModuleType, name: str) -> tuple[Any, int]:
    """The named kernel for the current device, compiled once, and its occupancy-optimal block size."""
    key = (id(cupy), int(cupy.cuda.Device().id), name)
    if key not in _KERNELS:
        kernel = cupy.RawKernel(_SOURCE, name, options=("--std=c++14",))
        _KERNELS[key] = (kernel, _occupancy_block(cupy, kernel))
    return _KERNELS[key]


def _launch(cupy: ModuleType, name: str, rows: int, arguments: Sequence[Any]) -> None:
    kernel, block = _kernel(cupy, name)
    kernel((-(-rows // block),), (block,), tuple(arguments))


def _range_arguments() -> tuple[np.float64, np.float64, np.float64]:
    """The smallest normal and the largest finite float64, and log 0."""
    return np.float64(_FLOAT64.tiny), np.float64(_FLOAT64.max), np.float64(-np.inf)


def _raise_if_improper(improper: Any) -> None:
    if bool(improper[0]):
        raise FloatingPointError("a cavity is improper on the lattice: 1 + v P <= 0")


def _column(cupy: ModuleType, values: Any, dtype: Any) -> Any:
    return cupy.ascontiguousarray(cupy.asarray(values, dtype=dtype))


def _tilted_row_bytes() -> int:
    """Device bytes one row of a tilted-moments chunk adds: copies of its class index and three inputs (the three
    outputs are the call's result, as on the host)."""
    return np.dtype(np.int64).itemsize + 3 * np.dtype(np.float64).itemsize


def tilted_moments(
    cupy: ModuleType,
    class_index: Any,
    log_density: Any,
    log_scale_rows: Any,
    grid: Any,
    floor: float,
    precision: Any,
    shift: Any,
    working_bytes: int,
) -> tuple[Any, Any, Any]:
    """log Z_j, the tilted mean and the tilted variance of every effect (device float64), as
    ``scale_mixture_ep.tilted_moments`` computes them. ``log_density`` is the (C x K) normalized log pi; inputs may
    live on the host or the device, and rows go in chunks whose device copies fit ``working_bytes``."""
    if working_bytes <= 0:
        raise ValueError("working_bytes must be positive")
    rows = int(class_index.shape[0])
    log_normalizer = cupy.empty(rows, dtype=cupy.float64)
    mean = cupy.empty(rows, dtype=cupy.float64)
    variance = cupy.empty(rows, dtype=cupy.float64)
    density = _column(cupy, log_density, cupy.float64)
    nodes = _column(cupy, grid, cupy.float64)
    node_exp = cupy.exp(nodes)
    improper = cupy.zeros(1, dtype=cupy.int32)
    chunk = max(1, int(working_bytes) // _tilted_row_bytes())
    for start in range(0, rows, chunk):
        stop = min(start + chunk, rows)
        count = stop - start
        _launch(cupy, "tilted_rows", count, (
            np.int64(count), np.int32(density.shape[1]),
            _column(cupy, class_index[start:stop], cupy.int64), density, nodes, node_exp, np.float64(floor), *_range_arguments(),
            _column(cupy, log_scale_rows[start:stop], cupy.float64),
            _column(cupy, precision[start:stop], cupy.float64),
            _column(cupy, shift[start:stop], cupy.float64),
            log_normalizer[start:stop], mean[start:stop], variance[start:stop], improper,
        ))
    _raise_if_improper(improper)
    return log_normalizer, mean, variance


def _objective_row_bytes(node_count: int, scale_size: int) -> int:
    """Device bytes one row of an objective chunk holds at once: its D and G columns, its design row and the
    curvature-scaled copy of it, copies of its three inputs, its three outputs and the |log Z| of the magnitude.
    A chunk's rows fill its batches, so no padding row exceeds the budget."""
    return np.dtype(np.float64).itemsize * (2 * node_count + 2 * scale_size + 7)


def objective_statistics(
    cupy: ModuleType,
    class_rows: Sequence[np.ndarray],
    log_density: np.ndarray,
    log_scale_rows: np.ndarray,
    grid: np.ndarray,
    floor: float,
    precision: np.ndarray,
    shift: np.ndarray,
    scale_design: np.ndarray,
    working_bytes: int,
) -> tuple[float, np.ndarray, np.ndarray, float]:
    """sum_j log Z_j, its gradient and Hessian in z = (eta_1, ..., eta_C, theta), and sum_j |log Z_j| (host float64),
    as ``scale_mixture_ep._data_objective`` defines them; chunks of rows are sized from ``working_bytes`` of device memory."""
    if working_bytes <= 0:
        raise ValueError("working_bytes must be positive")
    class_count, node_count = log_density.shape
    scale_size = int(scale_design.shape[1])
    scale_span = slice(class_count * node_count, class_count * node_count + scale_size)
    dimension = class_count * node_count + scale_size
    gradient = np.zeros(dimension)
    hessian = np.zeros((dimension, dimension))
    # A full chunk is whole batches: one per multiprocessor, as wide as the budget allows. A class's last, shorter
    # chunk spreads over as many of them, padded to at most the full chunk.
    multiprocessors = int(cupy.cuda.Device().attributes["MultiProcessorCount"])
    capacity = max(1, int(working_bytes) // _objective_row_bytes(node_count, scale_size))
    chunk = min(multiprocessors, capacity) * (capacity // min(multiprocessors, capacity))
    nodes = _column(cupy, grid, cupy.float64)
    node_exp = cupy.exp(nodes)
    scale_gradient = cupy.zeros(scale_size, dtype=cupy.float64)
    scale_hessian = cupy.zeros((scale_size, scale_size), dtype=cupy.float64)
    value = cupy.zeros((), dtype=cupy.float64)
    magnitude = cupy.zeros((), dtype=cupy.float64)
    improper = cupy.zeros(1, dtype=cupy.int32)
    density = np.exp(log_density)
    for class_position, all_rows in enumerate(class_rows):
        class_log_density = _column(cupy, log_density[class_position], cupy.float64)
        class_density = _column(cupy, density[class_position], cupy.float64)
        deviation_sum = cupy.zeros(node_count, dtype=cupy.float64)
        deviation_outer = cupy.zeros((node_count, node_count), dtype=cupy.float64)
        cross = cupy.zeros((node_count, scale_size), dtype=cupy.float64)
        for start in range(0, all_rows.shape[0], chunk):
            rows = all_rows[start : start + chunk]
            count = int(rows.shape[0])
            batches = min(multiprocessors, count)
            width = -(-count // batches)
            deviations = cupy.zeros((batches, node_count, width), dtype=cupy.float64)
            centred = cupy.zeros((batches, node_count, width), dtype=cupy.float64)
            log_normalizer = cupy.empty(count, dtype=cupy.float64)
            mean_first = cupy.empty(count, dtype=cupy.float64)
            curvature = cupy.empty(count, dtype=cupy.float64)
            _launch(cupy, "objective_rows", count, (
                np.int64(count), np.int64(width), np.int32(node_count), class_log_density, class_density, nodes, node_exp, np.float64(floor),
                *_range_arguments(),
                _column(cupy, log_scale_rows[rows], cupy.float64),
                _column(cupy, precision[rows], cupy.float64),
                _column(cupy, shift[rows], cupy.float64),
                deviations, centred, log_normalizer, mean_first, curvature, improper,
            ))
            padded = cupy.zeros((batches * width, scale_size), dtype=cupy.float64)
            padded[:count] = cupy.asarray(scale_design[rows], dtype=cupy.float64)
            design = padded[:count]
            value += log_normalizer.sum()
            magnitude += cupy.abs(log_normalizer).sum()
            deviation_sum += deviations.sum(axis=(0, 2))
            deviation_outer += cupy.matmul(deviations, deviations.transpose(0, 2, 1)).sum(axis=0)
            cross += cupy.matmul(centred, padded.reshape(batches, width, scale_size)).sum(axis=0)
            scale_gradient += design.T @ mean_first
            scale_hessian += design.T @ (curvature[:, None] * design)
        span = slice(class_position * node_count, (class_position + 1) * node_count)
        summed = cupy.asnumpy(deviation_sum)
        outer = cupy.asnumpy(deviation_outer)
        mass = density[class_position]
        gradient[span] = summed
        hessian[span, span] = np.diag(summed) - outer - np.outer(summed, mass) - np.outer(mass, summed)
        hessian[span, scale_span] = cupy.asnumpy(cross)
        hessian[scale_span, span] = hessian[span, scale_span].T
    _raise_if_improper(improper)
    gradient[scale_span] = cupy.asnumpy(scale_gradient)
    hessian[scale_span, scale_span] = cupy.asnumpy(scale_hessian)
    return float(value), gradient, hessian, float(magnitude)

"""Standardized-genotype products of one LD block from its signed dosage codes.

The store delivers a block as signed codes ``s = code - 127`` in [-127, 127] (int8,
variant-major ``[p_b, n]``). The model's standardized genotypes are
``x_j = (s_j - mean_j) / scale_j`` (training mean, population SD, in signed-code units), and every
product folds the centering into a rank-one correction, so ``X_b`` is never formed:

    X_b R       = S^T (R / scale) - 1 (mean / scale)^T R                     R [p_b, K]
    X_b^T L     = (S L - mean (1^T L)) / scale                               L [n, K]
    X_b^T W X_b = D^-1 (S W S^T - mean (S w)^T - (S w) mean^T + (1^T w) mean mean^T) D^-1
    X_b^T W C   = (S (W C) - mean (w^T C)) / scale

Both devices return fp64-accurate products.

* CUDA: the float operand of ``S`` is written as ``m`` balanced base-128 int8 digits per entry
  (``OPERAND_DIGITS`` for fp64-equivalent products, fewer for a caller's error budget), with a
  power-of-two scale per column, laid side by side. One int8 x int8 ->
  int32 cuBLAS GEMM per sample chunk (IMMA tensor cores on sm_75+, DP4A on sm_70) computes every
  digit product exactly (``INT32_EXACT_DIGIT_ROWS`` rows per chunk keep int32 exact), and
  recombining the digits in fp64 leaves only the operand's quantization: at most
  ``2^-(7 m - 2)`` of its column maximum per entry.
* CPU: the codes are converted to float64 per sample chunk (exact) and multiplied by DGEMM.

One Stage 2 read multiplies every block by the same sample-side operand ``L``, so
``CodeBlockTile.sample_operand(L, relative_error)`` prepares it once (its digits on CUDA, its padded copy on the
CPU, and its column sums) and every block's ``rmatmat`` reuses it.
``accumulate_matmat(R, image, relative_error)`` adds ``X_b R`` into the read's image in place,
with the recombination, centering and sum fused on CUDA. The error budget is normwise per
column: the operand the GEMMs multiply is within ``relative_error * ||L_k||_2`` of ``L_k`` (so the
product is within ``||X_b||_2`` times that), and the digit count is the least that guarantees it.
At a budget of fp64 rounding both give exactly the values of ``rmatmat(L)`` and
``image += matmat(R)``.

Memory: the caller's plan hands each tile ``workspace_bytes`` for the transients of one product.
Every product counts its fixed buffers (padded operand, output, integer products) and its
per-sample ones from the expressions that allocate them, and its sample chunk is the most that
fits. A ``SampleOperand`` lives for a whole read, so the plan counts its ``nbytes`` separately.
CuPy is passed in by the caller as the array module; this module never imports it.
"""

from __future__ import annotations

import math
from types import ModuleType
from typing import Any

import numpy as np

from sv_pgs.genotype_buffers import (
    SIGNED_CODE_OFFSET,
    _CUBLAS_COMPUTE_32I,
    _CUBLAS_GEMM_DEFAULT,
    _CUDA_R_8I,
    _CUDA_R_32I,
)

DIGIT_BITS = 7
"""Each operand digit covers 7 bits: balanced digits lie in [-64, 63], safely inside int8."""

OPERAND_DIGITS = 8
"""The most digits an operand entry takes: 7 * 8 - 2 = 54 bits hold the 53-bit fp64 mantissa of
the column maximum, so quantization is at most 2^-54 of it and the products carry fp64 GEMM
accuracy. A caller's error budget asks for fewer (``operand_digits_for``)."""

_DIGIT_HALF = 1 << (DIGIT_BITS - 1)
_DIGIT_MASK = (1 << DIGIT_BITS) - 1

INT32_EXACT_DIGIT_ROWS = (2**31 - 1) // (SIGNED_CODE_OFFSET * _DIGIT_HALF)
"""Largest sample count whose int32 sum of ``s_i d_i`` (|d| <= 64) cannot overflow (264,208)."""

_FLOAT64_BYTES = 8
_INT32_BYTES = 4


def _digit_working_bytes(digit_count: int) -> int:
    """Peak bytes ``operand_digits`` holds per operand entry: the int8 digits plus four live 8-byte
    temporaries while one digit is split off (the remaining integers, the digit, and two
    intermediate expressions)."""
    return digit_count + 4 * _FLOAT64_BYTES


_FLOAT64_ROUNDING = float(np.finfo(np.float64).eps) / 2
"""A budget no digit count below ``OPERAND_DIGITS`` meets: products exact to fp64 rounding."""


def _digits_for_ratio(ratio: float, relative_error: float) -> int:
    """The least digit count m with ratio * 2^-(7m-2) <= relative_error, at most OPERAND_DIGITS."""
    if not relative_error > 0:
        raise ValueError("relative_error must be positive")
    if ratio == 0:
        return 1
    return max(1, min(OPERAND_DIGITS, math.ceil((math.log2(ratio / relative_error) + 2) / DIGIT_BITS)))


def operand_digits_for(dense: Any, relative_error: float, array_module: ModuleType) -> int:
    """The fewest digits whose split of ``dense`` [rows, K] moves no column k by more than
    ``relative_error * ||dense_k||_2`` (2-norm), at most ``OPERAND_DIGITS``.

    The split rounds each entry to within 2^-(7m-2) of its column's largest magnitude and keeps
    zeros exact, so column k moves by at most sqrt(nnz_k) 2^-(7m-2) max|dense_k|.
    """
    xp = array_module
    values = xp.asarray(dense, dtype=xp.float64)
    if values.size == 0:
        return _digits_for_ratio(0.0, relative_error)
    magnitude = xp.abs(values)
    norm = xp.sqrt(xp.square(values).sum(axis=0))
    live = norm > 0
    ratio = xp.where(live, xp.sqrt((magnitude > 0).sum(axis=0)) * magnitude.max(axis=0) / xp.where(live, norm, 1.0), 0.0)
    return _digits_for_ratio(float(ratio.max()), relative_error)


def operand_digits(dense: Any, array_module: ModuleType, digit_count: int = OPERAND_DIGITS) -> tuple[Any, Any]:
    """Split ``dense`` [rows, K] into ``digit_count`` balanced base-128 digits.

    Returns ``(digits, scale)``: ``digits`` is int8 [rows, digit_count * K] in column-major
    order, digit ``d`` of column ``k`` in column ``d * K + k``, and
    ``sum_d digits[:, d * K + k] * 128^d == rint(dense[:, k] * scale[k])`` exactly. ``scale`` is a
    power of two per column that puts the column maximum in (2^(7m-3), 2^(7m-2)].
    """
    values = array_module.asarray(dense, dtype=array_module.float64)
    rows, columns = values.shape
    magnitude = array_module.max(array_module.abs(values), axis=0) if rows else array_module.zeros(columns)
    exponent = (DIGIT_BITS * digit_count - 2) - array_module.ceil(array_module.log2(array_module.where(magnitude > 0, magnitude, 1.0)))
    scale = array_module.exp2(exponent)
    integers = array_module.rint(values * scale[None, :]).astype(array_module.int64)
    digits = array_module.empty((rows, digit_count * columns), dtype=array_module.int8, order="F")
    for digit_index in range(digit_count):
        low = ((integers + _DIGIT_HALF) & _DIGIT_MASK) - _DIGIT_HALF
        digits[:, digit_index * columns : (digit_index + 1) * columns] = low.astype(array_module.int8)
        integers = (integers - low) >> DIGIT_BITS
    return digits, scale


def recombine_digit_products(products: Any, scale: Any, array_module: ModuleType) -> Any:
    """Undo ``operand_digits`` on the right of an exact integer product [rows, m * K]."""
    columns = int(scale.shape[0])
    digit_count = int(products.shape[1]) // columns
    total = products[:, (digit_count - 1) * columns :].astype(array_module.float64)
    for digit_index in range(digit_count - 2, -1, -1):
        total = total * float(1 << DIGIT_BITS) + products[:, digit_index * columns : (digit_index + 1) * columns]
    return total / scale[None, :]


INT8_GEMM_ALIGNMENT = 4
"""cuBLAS runs int8 GEMMs only when the reduction length, both leading dimensions and every operand
offset are multiples of 4 (CUBLAS_STATUS_NOT_SUPPORTED otherwise). A block is zero-padded to
multiples of 4 on both axes: a zero signed code adds nothing to any product."""


def _aligned(count: int) -> int:
    return -(-count // INT8_GEMM_ALIGNMENT) * INT8_GEMM_ALIGNMENT


_TRANSPOSE_SOURCE = r"""
extern "C" __global__
void transpose_codes(const signed char* __restrict__ source, signed char* __restrict__ target,
                     long long rows, long long lead, long long column_start, long long columns) {
    // target [columns, rows] = source[:, column_start:column_start + columns].T, source row-major [rows, lead]
    __shared__ int tile[WARP][WARP + 1];
    long long column = column_start + (long long)blockIdx.x * WARP + threadIdx.x;
    long long row0 = (long long)blockIdx.y * WARP;
    for (int step = threadIdx.y; step < WARP; step += blockDim.y) {
        long long row = row0 + step;
        tile[step][threadIdx.x] = (row < rows && column < column_start + columns) ? (int)source[row * lead + column] : 0;
    }
    __syncthreads();
    long long target_row0 = (long long)blockIdx.x * WARP;
    long long target_column = row0 + threadIdx.x;
    for (int step = threadIdx.y; step < WARP; step += blockDim.y) {
        long long target_row = target_row0 + step;
        if (target_row < columns && target_column < rows) {
            target[target_row * rows + target_column] = (signed char)tile[threadIdx.x][step];
        }
    }
}
"""

_ACCUMULATE_SOURCE = r"""
extern "C" __global__
void accumulate_recombined(const int* __restrict__ products, long long lead, int digits, long long columns,
                           const double* __restrict__ scale, const double* __restrict__ offset,
                           double* __restrict__ image, long long row_start, long long rows) {
    // image[row_start + i, k] += recombined(products[i, :, k]) / scale[k] - offset[k], image row-major,
    // products column-major [lead, digits * columns]: recombine_digit_products' operations in its order,
    // with explicit roundings so that no multiply-add is contracted.
    __shared__ double tile[WARP][WARP + 1];
    long long row0 = (long long)blockIdx.x * WARP;
    long long column0 = (long long)blockIdx.y * WARP;
    for (int step = threadIdx.y; step < WARP; step += blockDim.y) {
        long long column = column0 + step;
        long long row = row0 + threadIdx.x;
        double total = 0.0;
        if (column < columns && row < rows) {
            total = (double)products[((long long)(digits - 1) * columns + column) * lead + row];
            for (int digit = digits - 2; digit >= 0; --digit) {
                total = __dadd_rn(__dmul_rn(total, (double)(1 << DIGIT_BITS)), (double)products[((long long)digit * columns + column) * lead + row]);
            }
            total = __dsub_rn(__ddiv_rn(total, scale[column]), offset[column]);
        }
        tile[step][threadIdx.x] = total;
    }
    __syncthreads();
    for (int step = threadIdx.y; step < WARP; step += blockDim.y) {
        long long row = row0 + step;
        long long column = column0 + threadIdx.x;
        if (column < columns && row < rows) {
            long long at = (row_start + row) * columns + column;
            image[at] = __dadd_rn(image[at], tile[threadIdx.x][step]);
        }
    }
}
"""

_KERNELS: dict[tuple[int, int, str], Any] = {}


def _cuda_kernel(cupy: ModuleType, name: str) -> tuple[Any, int]:
    """The named kernel for the current device, compiled once, with its warp-wide tile edge.

    The tile edge is the device's warp size, so each warp moves one contiguous tile row; a block
    holds as many warp rows as the device allows, up to one per tile row.
    """
    device = cupy.cuda.Device()
    key = (id(cupy), int(device.id), name)
    if key not in _KERNELS:
        warp = int(device.attributes["WarpSize"])
        source = _TRANSPOSE_SOURCE if name == "transpose_codes" else _ACCUMULATE_SOURCE
        options = (f"-DWARP={warp}", f"-DDIGIT_BITS={DIGIT_BITS}")
        _KERNELS[key] = (cupy.RawKernel(source, name, options=options), warp)
    return _KERNELS[key]


def _launch_shape(cupy: ModuleType, warp: int, rows: int, columns: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Grid and block of a warp-tiled kernel over [rows, columns] (rows along grid x)."""
    warp_rows = min(warp, int(cupy.cuda.Device().attributes["MaxThreadsPerBlock"]) // warp)
    return (-(-rows // warp), -(-columns // warp)), (warp, warp_rows)


def _transpose_codes(cupy: ModuleType, codes: Any, column_start: int, column_stop: int) -> Any:
    """``cupy.ascontiguousarray(codes[:, column_start:column_stop].T)`` for C-contiguous int8 codes."""
    kernel, warp = _cuda_kernel(cupy, "transpose_codes")
    rows, lead = (int(extent) for extent in codes.shape)
    columns = column_stop - column_start
    target = cupy.empty((columns, rows), dtype=cupy.int8)
    grid, block = _launch_shape(cupy, warp, columns, rows)
    kernel(grid, block, (codes, target, np.int64(rows), np.int64(lead), np.int64(column_start), np.int64(columns)))
    return target


class SampleOperand:
    """The sample-side operand ``L`` [n, K] of one read, prepared once for every block's ``X_b' L``.

    Built by ``CodeBlockTile.sample_operand``. On CUDA it holds the balanced digits of ``L`` (one
    set per int32-exact span of samples, which the read's memory plan holds whole) and on the CPU
    the zero-padded ``L``; both hold the column sums ``1' L``. ``nbytes`` counts what it holds.
    """

    def __init__(
        self, column_sums: Any, padded: Any, chunks: list[tuple[int, int, Any, Any]], padded_samples: int, digit_count: int
    ) -> None:
        self.column_sums = column_sums
        self.digit_count = digit_count
        self.columns = int(column_sums.shape[0])
        self.padded = padded
        self.chunks = chunks
        self.padded_samples = padded_samples
        self.nbytes = int(column_sums.nbytes) + (0 if padded is None else int(padded.nbytes)) + sum(
            int(digits.nbytes) + int(scale.nbytes) for _, _, digits, scale in chunks
        )


class CodeBlockTile:
    """Standardized genotypes ``X_b = (S - mean) / scale`` of one LD block, from its signed codes.

    ``signed_codes`` is int8 [p_b, n] on the device of ``array_module`` (``numpy`` or ``cupy``);
    ``means`` and ``scales`` are per variant in signed-code units. Implements the Stage 2
    ``GenotypeBlockTile`` protocol with fp64-accurate products. The codes are held zero-padded to
    ``INT8_GEMM_ALIGNMENT`` on both axes; every product returns the unpadded shape.
    """

    def __init__(self, signed_codes: Any, means: Any, scales: Any, array_module: ModuleType, workspace_bytes: int) -> None:
        codes = array_module.asarray(signed_codes)
        if codes.dtype != array_module.int8 or codes.ndim != 2:
            raise ValueError("signed_codes must be a 2-D int8 [variants, samples] array")
        variant_count, sample_count = (int(extent) for extent in codes.shape)
        aligned = array_module.zeros((_aligned(variant_count), _aligned(sample_count)), dtype=array_module.int8)
        aligned[:variant_count, :sample_count] = codes
        scale_values = array_module.asarray(scales, dtype=array_module.float64)
        spread = float(scale_values.max() / scale_values.min()) if variant_count else 1.0
        self._hold(aligned, variant_count, sample_count, means, scale_values, spread, array_module, workspace_bytes)

    @classmethod
    def from_aligned(
        cls, aligned_codes: Any, variant_count: int, sample_count: int, means: Any, scales: Any, scale_spread: float | None,
        array_module: ModuleType, workspace_bytes: int, sample_major: tuple[Any, int] | None = None,
    ) -> CodeBlockTile:
        """A tile over codes the caller already holds in an ``INT8_GEMM_ALIGNMENT``-aligned buffer (a
        streamed block's device buffer, say), without copying them. The alignment padding, at most
        three rows and three columns, is cleared in place.

        ``scale_spread`` is max(scales) / min(scales) when the caller knows it on the host and the tile must
        never wait on the device (a streamed block, whose next block's host work overlaps this one's products):
        ``accumulate_matmat`` then takes its digits from the a-priori bound. ``None`` measures them on each
        call's R instead (``accumulate_digits``), one device reduction and wait, never more digits.

        ``sample_major`` is ``(codes_t, column)`` when the caller also holds the same codes sample-major: a
        C-contiguous int8 [padded samples, lead] array with ``codes_t[i, column + j] == aligned_codes[j, i]``
        (``column`` a multiple of ``INT8_GEMM_ALIGNMENT``). The products that reduce over variants then read
        it directly instead of transposing each sample chunk of the codes (CUDA)."""
        codes = array_module.asarray(aligned_codes)
        if codes.dtype != array_module.int8 or codes.shape != (_aligned(variant_count), _aligned(sample_count)):
            raise ValueError(f"aligned_codes must be int8 [{_aligned(variant_count)}, {_aligned(sample_count)}]")
        codes[variant_count:] = 0
        codes[:, sample_count:] = 0
        if sample_major is not None:
            major, column = sample_major
            if (
                major.dtype != array_module.int8 or major.ndim != 2 or not major.flags.c_contiguous or int(major.shape[0]) != int(codes.shape[1])
                or column % INT8_GEMM_ALIGNMENT or column + int(codes.shape[0]) > int(major.shape[1])
            ):
                raise ValueError("sample_major must be C-contiguous int8 [padded samples, lead] holding this tile's aligned columns")
        tile = cls.__new__(cls)
        tile._hold(
            codes, int(variant_count), int(sample_count), means, scales, None if scale_spread is None else float(scale_spread),
            array_module, workspace_bytes, sample_major,
        )
        return tile

    def _hold(
        self, aligned_codes: Any, variant_count: int, sample_count: int, means: Any, scales: Any, scale_spread: float | None,
        array_module: ModuleType, workspace_bytes: int, sample_major: tuple[Any, int] | None = None,
    ) -> None:
        self._array_module = array_module
        self._scale_spread = scale_spread
        self._variant_count, self._sample_count = variant_count, sample_count
        self._codes = aligned_codes
        self._sample_major = sample_major
        self._means = array_module.asarray(means, dtype=array_module.float64)
        self._scales = array_module.asarray(scales, dtype=array_module.float64)
        if self._means.shape != (variant_count,) or self._scales.shape != (variant_count,):
            raise ValueError("means and scales need one entry per variant")
        self._workspace_bytes = int(workspace_bytes)

    @property
    def variant_count(self) -> int:
        return self._variant_count

    @property
    def sample_count(self) -> int:
        return self._sample_count

    @property
    def aligned_codes(self) -> Any:
        """The held int8 codes [variants, samples], zero-padded to ``INT8_GEMM_ALIGNMENT`` on both axes."""
        return self._codes

    @property
    def means(self) -> Any:
        return self._means

    @property
    def scales(self) -> Any:
        return self._scales

    def _variant_contiguous(self, start: int, stop: int) -> tuple[Any, int, int]:
        """(left, byte offset, lead) of samples start..stop with each sample's variants contiguous, the TN
        GEMM's operand when a product reduces over variants: the held sample-major codes, else a transpose."""
        if self._sample_major is None:
            return _transpose_codes(self._array_module, self._codes, start, stop), 0, int(self._codes.shape[0])
        major, column = self._sample_major
        lead = int(major.shape[1])
        return major, start * lead + column, lead

    def matmat(self, right: Any) -> Any:
        """X_b @ right for right of shape (p_b, K); returns (n, K)."""
        xp = self._array_module
        scaled = xp.asarray(right, dtype=xp.float64) / self._scales[:, None]
        return self._codes_transposed_times(scaled) - (self._means @ scaled)[None, :]

    def accumulate_matmat(self, right: Any, image: Any, relative_error: float) -> None:
        """``image += X_b R~`` in place for an R~ with ``||R~_k - right_k||_2 <= relative_error ||right_k||_2``.

        ``image`` is the read's C-contiguous float64 (n, K) sum. No (n, K) temporary is formed: on
        CUDA one kernel recombines each chunk's integer products, centers them and adds them in.
        The CUDA split holds R / scale, whose rounding moves column k of R by at most
        2^-(7m-2) max|R_k / scale| (sum of scale^2 over R_k's nonzero rows)^(1/2) <= sqrt(p_b)
        (max scale / min scale) 2^-(7m-2) ||R_k||_2. The tile's scale spread sets m by the right-hand
        bound without waiting on the device; a tile without one measures the left (``accumulate_digits``).
        At a budget of fp64
        rounding (m = OPERAND_DIGITS) the values equal ``image += self.matmat(right)`` exactly; the
        CPU products are exact fp64 whatever the budget.
        """
        xp = self._array_module
        scaled = xp.asarray(right, dtype=xp.float64) / self._scales[:, None]
        offset = self._means @ scaled
        variants, samples = (int(extent) for extent in self._codes.shape)
        columns = int(scaled.shape[1])
        if image.shape != (self._sample_count, columns) or image.dtype != xp.float64 or not image.flags.c_contiguous:
            raise ValueError(f"image must be C-contiguous float64 [{self._sample_count}, {columns}]")
        padded = self._padded_rows(scaled, variants)
        if xp is np:
            # the chunks of _codes_transposed_times, so every GEMM call and hence every value matches
            fixed = _FLOAT64_BYTES * variants * columns + _FLOAT64_BYTES * samples * columns
            chunk = self._sample_chunk(fixed, _FLOAT64_BYTES * (variants + columns), samples)
            for start in range(0, min(samples, self._sample_count), chunk):
                stop = min(start + chunk, samples)
                rows = min(stop, self._sample_count) - start
                image[start : start + rows] += (self._codes[:, start:stop].T.astype(np.float64) @ padded)[:rows] - offset[None, :]
            return
        if variants > INT32_EXACT_DIGIT_ROWS:
            raise ValueError(f"an LD block of {variants} variants exceeds the int32-exact depth {INT32_EXACT_DIGIT_ROWS}")
        if self._scale_spread is None:
            digit_count = self.accumulate_digits(right, relative_error)
        else:
            digit_count = _digits_for_ratio(math.sqrt(self._variant_count) * self._scale_spread, relative_error)
        digits, scale = operand_digits(padded, xp, digit_count)
        if digit_count < OPERAND_DIGITS:
            # center with the operand the digits represent, so the image gets exactly X_b R~
            represented = recombine_digit_products(digits.astype(xp.int32), scale, xp)[: self._variant_count]
            offset = self._means @ represented
        digit_columns = digit_count * columns
        # fixed: the operand and its digits; per sample: its variant-contiguous codes (unless held sample-major)
        # and integer products
        fixed = _FLOAT64_BYTES * variants * columns + digit_columns * variants
        transposed = variants if self._sample_major is None else 0
        chunk = self._sample_chunk(fixed, transposed + _INT32_BYTES * digit_columns, samples)
        products = xp.empty((chunk, digit_columns), dtype=xp.int32, order="F")
        kernel, warp = _cuda_kernel(xp, "accumulate_recombined")
        scale, offset = xp.ascontiguousarray(scale), xp.ascontiguousarray(offset)
        for start in range(0, min(samples, self._sample_count), chunk):
            stop = min(start + chunk, samples)
            left, left_offset, left_lead = self._variant_contiguous(start, stop)
            _cuda_int8_gemm(
                xp, rows=stop - start, columns=digit_columns, depth=variants,
                left=left, left_offset=left_offset, left_lead=left_lead,
                right=digits, right_lead=variants, output=products, output_lead=chunk,
            )
            rows = min(stop, self._sample_count) - start
            grid, block = _launch_shape(xp, warp, rows, columns)
            kernel(
                grid, block,
                (products, np.int64(chunk), np.int32(digit_count), np.int64(columns), scale, offset, image, np.int64(start), np.int64(rows)),
            )

    def accumulate_digits(self, right: Any, relative_error: float) -> int:
        """The fewest digits whose split of R / scale moves no column k of R = ``right`` by more than
        ``relative_error * ||R_k||_2``, at most ``OPERAND_DIGITS``.

        The split rounds each entry of column k of Q = R / scale to within 2^-(7m-2) of max|Q_k| and keeps zeros
        exact, so R_jk = scale_j Q_jk moves by at most scale_j 2^-(7m-2) max|Q_k| on Q_k's nonzero rows, and R_k by
        2^-(7m-2) max|Q_k| (sum of scale_j^2 over them)^(1/2). The count is measured on R, one device reduction.
        """
        xp = self._array_module
        values = xp.asarray(right, dtype=xp.float64)
        quotient = values / self._scales[:, None]
        magnitude = xp.abs(quotient).max(axis=0)
        support = xp.sqrt(((quotient != 0) * xp.square(self._scales)[:, None]).sum(axis=0))
        norm = xp.sqrt(xp.square(values).sum(axis=0))
        live = norm > 0
        ratio = xp.where(live, magnitude * support / xp.where(live, norm, 1.0), 0.0)
        return _digits_for_ratio(float(ratio.max()) if ratio.size else 0.0, relative_error)

    def sample_operand(self, left: Any, relative_error: float) -> SampleOperand:
        """``left`` [n, K] prepared once for the ``rmatmat`` of every block of a read.

        Every tile of the read shares this tile's sample count. ``rmatmat(operand)`` is X_b' L~ for
        an L~ with ``||L~_k - left_k||_2 <= relative_error ||left_k||_2`` (``operand_digits_for``;
        exact zeros, such as a fold's masked rows, stay zero); the CPU products are exact fp64
        whatever the budget. At a budget of fp64 rounding it equals ``rmatmat(left)`` bit for bit
        whenever ``left``'s product fits one chunk of its own (a workspace that holds it and n
        within the int32-exact depth); otherwise the two group the fp64 sum over samples
        differently.
        """
        xp = self._array_module
        values = xp.asarray(left, dtype=xp.float64)
        samples = int(self._codes.shape[1])
        if values.ndim != 2 or values.shape[0] != self._sample_count:
            raise ValueError(f"left must be [{self._sample_count}, K]")
        padded = self._padded_rows(values, samples)
        column_sums = values.sum(axis=0)
        if xp is np:
            return SampleOperand(column_sums, padded, [], samples, OPERAND_DIGITS)
        digit_count = operand_digits_for(values, relative_error, xp)
        span = INT32_EXACT_DIGIT_ROWS // INT8_GEMM_ALIGNMENT * INT8_GEMM_ALIGNMENT
        chunks = [
            (start, min(start + span, samples), *operand_digits(padded[start : start + span], xp, digit_count))
            for start in range(0, samples, span)
        ]
        if digit_count < OPERAND_DIGITS:
            # center with the operand the digits represent, so rmatmat gives exactly X_b' L~
            column_sums = sum(recombine_digit_products(digits.astype(xp.int32), scale, xp).sum(axis=0) for _, _, digits, scale in chunks)
        return SampleOperand(column_sums, None, chunks, samples, digit_count)

    def rmatmat(self, left: Any) -> Any:
        """X_b.T @ left for left of shape (n, K), or for its read's ``SampleOperand``; returns (p_b, K)."""
        xp = self._array_module
        if isinstance(left, SampleOperand):
            products, column_sums = self._codes_times_operand(left), left.column_sums
        else:
            values = xp.asarray(left, dtype=xp.float64)
            products, column_sums = self._codes_times(values), values.sum(axis=0)
        centered = products - self._means[:, None] * column_sums[None, :]
        return centered / self._scales[:, None]

    def weighted_column_squares(self, weights: Any) -> Any:
        """(X_b * X_b).T @ weights for weights of shape (n, c); returns (p_b, c).

        With x = (s - mean) / scale, sum_i w_i x_i^2 = (S2 w - 2 mean (S w) + mean^2 (1' w)) / scale^2
        for S2 = s * s. On CUDA, S2 = 128 A + B with A = S2 >> 7 and B = S2 & 127, both in [0, 127],
        so each digit GEMM stays within the int32-exact bound of the codes themselves.
        """
        operand = self.sample_operand(weights, _FLOAT64_ROUNDING)
        linear = self._codes_times_operand(operand)
        squares = self._squared_codes_times_operand(operand)
        centered = squares - 2.0 * self._means[:, None] * linear + (self._means * self._means)[:, None] * operand.column_sums[None, :]
        return centered / (self._scales * self._scales)[:, None]

    def columns(self, local: Any) -> Any:
        """The standardized columns X_b[:, local] as a dense float64 (n, len(local)) array."""
        xp = self._array_module
        index = xp.asarray(local, dtype=xp.int64)
        codes = self._codes[index, : self._sample_count].astype(xp.float64)
        return ((codes - self._means[index][:, None]) / self._scales[index][:, None]).T

    def weighted_gram(self, weights: Any) -> Any:
        """X_b.T diag(weights) X_b; returns (p_b, p_b)."""
        xp = self._array_module
        weight_vector = xp.asarray(weights, dtype=xp.float64)
        cross = self._codes_times_weighted_codes(weight_vector)
        code_weights = self._codes_times(weight_vector[:, None])[:, 0]
        centered = (
            cross
            - self._means[:, None] * code_weights[None, :]
            - code_weights[:, None] * self._means[None, :]
            + float(weight_vector.sum()) * self._means[:, None] * self._means[None, :]
        )
        return centered / (self._scales[:, None] * self._scales[None, :])

    def weighted_cross(self, weights: Any, covariates: Any) -> Any:
        """X_b.T diag(weights) C for C of shape (n, k); returns (p_b, k)."""
        xp = self._array_module
        weighted = xp.asarray(weights, dtype=xp.float64)[:, None] * xp.asarray(covariates, dtype=xp.float64)
        return self.rmatmat(weighted)

    def _sample_chunk(self, fixed_bytes: int, bytes_per_sample: int, exact_rows: int) -> int:
        """Samples per chunk: the largest multiple of ``INT8_GEMM_ALIGNMENT`` whose buffers fit the workspace."""
        padded_samples = int(self._codes.shape[1])
        fitting = (self._workspace_bytes - fixed_bytes) // bytes_per_sample
        chunk = min(exact_rows, padded_samples, fitting) // INT8_GEMM_ALIGNMENT * INT8_GEMM_ALIGNMENT
        if chunk < INT8_GEMM_ALIGNMENT:
            raise MemoryError(
                f"a workspace of {self._workspace_bytes} bytes cannot hold a product's fixed buffers "
                f"({fixed_bytes} bytes) and a {INT8_GEMM_ALIGNMENT}-sample chunk ({bytes_per_sample} bytes per sample)"
            )
        return chunk

    def _padded_rows(self, operand: Any, rows: int) -> Any:
        """``operand`` [r, K] float64 with zero rows appended up to ``rows``."""
        xp = self._array_module
        padded = xp.zeros((rows, int(operand.shape[1])), dtype=xp.float64)
        padded[: operand.shape[0]] = operand
        return padded

    def _codes_times(self, operand: Any) -> Any:
        """S @ operand for operand [n, K] float64; returns [p_b, K] float64."""
        xp = self._array_module
        variants, samples = (int(extent) for extent in self._codes.shape)
        columns = int(operand.shape[1])
        padded = self._padded_rows(operand, samples)
        total = xp.zeros((variants, columns), dtype=xp.float64)
        operand_bytes = _FLOAT64_BYTES * columns * samples
        if xp is np:
            # fixed: the operand, the total and one GEMM result; per sample: its fp64 codes
            fixed = operand_bytes + 2 * _FLOAT64_BYTES * variants * columns
            chunk = self._sample_chunk(fixed, _FLOAT64_BYTES * variants, samples)
            for start in range(0, samples, chunk):
                stop = min(start + chunk, samples)
                total += self._codes[:, start:stop].astype(np.float64) @ padded[start:stop]
            return total[: self._variant_count]
        # fixed: the operand, the total, the integer products and two fp64 recombination terms;
        # per sample: the digit split of its operand row
        fixed = operand_bytes + variants * columns * (3 * _FLOAT64_BYTES + OPERAND_DIGITS * _INT32_BYTES)
        chunk = self._sample_chunk(fixed, _digit_working_bytes(OPERAND_DIGITS) * columns, INT32_EXACT_DIGIT_ROWS)
        chunks = ((start, min(start + chunk, samples), *operand_digits(padded[start : start + chunk], xp)) for start in range(0, samples, chunk))
        return self._digit_products(self._codes, chunks, columns, OPERAND_DIGITS)[: self._variant_count]

    def _digit_products(self, left_codes: Any, chunks: Any, columns: int, digit_count: int) -> Any:
        """sum over the operand's sample chunks of left_codes @ chunk, from its digits (CUDA); [p_b, K]."""
        xp = self._array_module
        variants, samples = (int(extent) for extent in left_codes.shape)
        total = xp.zeros((variants, columns), dtype=xp.float64)
        products = xp.empty((variants, digit_count * columns), dtype=xp.int32, order="F")
        for start, stop, digits, scale in chunks:
            _cuda_int8_gemm(
                xp, rows=variants, columns=digit_count * columns, depth=stop - start,
                left=left_codes, left_offset=start, left_lead=samples,
                right=digits, right_lead=stop - start, output=products, output_lead=variants,
            )
            total += recombine_digit_products(products, scale, xp)
        return total

    def _require_workspace(self, fixed_bytes: int) -> None:
        if fixed_bytes > self._workspace_bytes:
            raise MemoryError(f"a workspace of {self._workspace_bytes} bytes cannot hold a product's fixed buffers ({fixed_bytes} bytes)")

    def _operand_codes_times(self, operand: SampleOperand, converted: Any) -> Any:
        """sum over samples of converted(codes chunk) @ operand, on the CPU; [p_b, K]."""
        variants, samples = (int(extent) for extent in self._codes.shape)
        columns = operand.columns
        # fixed: the total and one GEMM result; per sample: its converted fp64 codes
        chunk = self._sample_chunk(2 * _FLOAT64_BYTES * variants * columns, _FLOAT64_BYTES * variants, samples)
        total = np.zeros((variants, columns), dtype=np.float64)
        for start in range(0, samples, chunk):
            stop = min(start + chunk, samples)
            total += converted(self._codes[:, start:stop]) @ operand.padded[start:stop]
        return total

    def _check_operand(self, operand: SampleOperand) -> None:
        if operand.padded_samples != int(self._codes.shape[1]):
            raise ValueError("the operand was prepared for tiles of another sample count")

    def _codes_times_operand(self, operand: SampleOperand) -> Any:
        """S @ L for the read's prepared L; returns [p_b, K] float64."""
        self._check_operand(operand)
        if self._array_module is np:
            return self._operand_codes_times(operand, lambda codes: codes.astype(np.float64))[: self._variant_count]
        variants = int(self._codes.shape[0])
        # fixed: the total, the integer products and two fp64 recombination terms
        self._require_workspace(variants * operand.columns * (3 * _FLOAT64_BYTES + operand.digit_count * _INT32_BYTES))
        return self._digit_products(self._codes, operand.chunks, operand.columns, operand.digit_count)[: self._variant_count]

    def _squared_codes_times_operand(self, operand: SampleOperand) -> Any:
        """(S * S) @ L for the read's prepared L; returns [p_b, K] float64."""
        self._check_operand(operand)
        xp = self._array_module
        if xp is np:
            return self._operand_codes_times(operand, lambda codes: np.square(codes.astype(np.float64)))[: self._variant_count]
        variants = int(self._codes.shape[0])
        # fixed: the int16 squares, their two int8 halves, the total, the integer products and two
        # fp64 recombination terms
        code_bytes = int(self._codes.size)
        self._require_workspace(4 * code_bytes + variants * operand.columns * (3 * _FLOAT64_BYTES + operand.digit_count * _INT32_BYTES))
        squares = self._codes.astype(xp.int16)
        squares *= squares
        high = (squares >> DIGIT_BITS).astype(xp.int8)
        low = (squares & _DIGIT_MASK).astype(xp.int8)
        del squares
        return (
            self._digit_products(high, operand.chunks, operand.columns, operand.digit_count) * float(1 << DIGIT_BITS)
            + self._digit_products(low, operand.chunks, operand.columns, operand.digit_count)
        )[: self._variant_count]

    def _codes_transposed_times(self, operand: Any) -> Any:
        """S^T @ operand for operand [p_b, K] float64; returns [n, K] float64."""
        xp = self._array_module
        variants, samples = (int(extent) for extent in self._codes.shape)
        columns = int(operand.shape[1])
        padded = self._padded_rows(operand, variants)
        output_bytes = _FLOAT64_BYTES * samples * columns
        if xp is np:
            # fixed: the operand and the output; per sample: its fp64 codes and GEMM result row
            fixed = _FLOAT64_BYTES * variants * columns + output_bytes
            chunk = self._sample_chunk(fixed, _FLOAT64_BYTES * (variants + columns), samples)
            out = np.empty((samples, columns), dtype=np.float64)
            for start in range(0, samples, chunk):
                stop = min(start + chunk, samples)
                out[start:stop] = self._codes[:, start:stop].T.astype(np.float64) @ padded
            return out[: self._sample_count]
        if variants > INT32_EXACT_DIGIT_ROWS:
            raise ValueError(f"an LD block of {variants} variants exceeds the int32-exact depth {INT32_EXACT_DIGIT_ROWS}")
        # The reduction runs over variants, which the variant-major codes do not hold contiguously,
        # so each sample chunk is transposed to variant-contiguous order for the TN GEMM.
        digits, scale = operand_digits(padded, xp)
        digit_columns = OPERAND_DIGITS * columns
        # fixed: the operand, its digits and the output; per sample: its variant-contiguous codes,
        # its integer products and two fp64 recombination terms
        fixed = _FLOAT64_BYTES * variants * columns + digit_columns * variants + output_bytes
        per_sample = (variants if self._sample_major is None else 0) + _INT32_BYTES * digit_columns + 2 * _FLOAT64_BYTES * columns
        chunk = self._sample_chunk(fixed, per_sample, samples)
        products = xp.empty((chunk, digit_columns), dtype=xp.int32, order="F")
        out = xp.empty((samples, columns), dtype=xp.float64)
        for start in range(0, samples, chunk):
            stop = min(start + chunk, samples)
            left, left_offset, left_lead = self._variant_contiguous(start, stop)
            _cuda_int8_gemm(
                xp, rows=stop - start, columns=digit_columns, depth=variants,
                left=left, left_offset=left_offset, left_lead=left_lead,
                right=digits, right_lead=variants, output=products, output_lead=chunk,
            )
            out[start:stop] = recombine_digit_products(products[: stop - start], scale, xp)
        return out[: self._sample_count]

    def _codes_times_weighted_codes(self, weights: Any) -> Any:
        """S diag(weights) S^T; returns [p_b, p_b] float64."""
        xp = self._array_module
        variants, samples = (int(extent) for extent in self._codes.shape)
        padded_weights = self._padded_rows(weights[:, None], samples)[:, 0]
        if xp is np:
            # fixed: the total and one GEMM result; per sample: its fp64 codes and their weighted copy
            fixed = 2 * _FLOAT64_BYTES * variants * variants
            chunk = self._sample_chunk(fixed, 2 * _FLOAT64_BYTES * variants, samples)
            total = np.zeros((variants, variants), dtype=np.float64)
            for start in range(0, samples, chunk):
                stop = min(start + chunk, samples)
                converted = self._codes[:, start:stop].astype(np.float64)
                total += converted @ (padded_weights[start:stop, None] * converted.T)
            return total[: self._variant_count, : self._variant_count]
        total = xp.zeros((variants, variants), dtype=xp.float64)
        products = xp.empty((variants, OPERAND_DIGITS * variants), dtype=xp.int32, order="F")
        # fixed: the weights, the total, the integer products and two fp64 recombination terms;
        # per sample: its weighted fp64 codes and their digit split
        fixed = _FLOAT64_BYTES * samples + variants * variants * (3 * _FLOAT64_BYTES + OPERAND_DIGITS * _INT32_BYTES)
        chunk = self._sample_chunk(fixed, (_FLOAT64_BYTES + _digit_working_bytes(OPERAND_DIGITS)) * variants, INT32_EXACT_DIGIT_ROWS)
        for start in range(0, samples, chunk):
            stop = min(start + chunk, samples)
            weighted_codes = padded_weights[start:stop, None] * self._codes[:, start:stop].T.astype(xp.float64)
            digits, scale = operand_digits(weighted_codes, xp)
            _cuda_int8_gemm(
                xp, rows=variants, columns=OPERAND_DIGITS * variants, depth=stop - start,
                left=self._codes, left_offset=start, left_lead=samples,
                right=digits, right_lead=stop - start, output=products, output_lead=variants,
            )
            total += recombine_digit_products(products, scale, xp)
        return total[: self._variant_count, : self._variant_count]


def _cuda_int8_gemm(
    cupy: ModuleType, *, rows: int, columns: int, depth: int,
    left: Any, left_offset: int, left_lead: int, right: Any, right_lead: int, output: Any, output_lead: int,
) -> None:
    """Column-major ``output[rows, columns] = left^T @ right`` in exact int32 (the TN form).

    cuBLAS supports int8 GEMMs only as TN: both operands keep the reduction dimension ``depth``
    contiguous. ``left`` is column-major ``[depth, rows]`` with lead ``left_lead`` starting
    ``left_offset`` bytes in; ``right`` is column-major int8 ``[depth, columns]``.
    """
    cublas = cupy.cuda.cublas
    handle = cupy.cuda.device.get_cublas_handle()
    cublas.setStream(handle, cupy.cuda.get_current_stream().ptr)
    one = np.ones(1, dtype=np.int32)
    zero = np.zeros(1, dtype=np.int32)
    cublas.gemmEx(
        handle,
        cublas.CUBLAS_OP_T,
        cublas.CUBLAS_OP_N,
        rows, columns, depth,
        one.ctypes.data, left.data.ptr + left_offset, _CUDA_R_8I, left_lead,
        right.data.ptr, _CUDA_R_8I, right_lead,
        zero.ctypes.data, output.data.ptr, _CUDA_R_32I, output_lead,
        _CUBLAS_COMPUTE_32I, _CUBLAS_GEMM_DEFAULT,
    )

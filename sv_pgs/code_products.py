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

* CUDA: the float operand of ``S`` is written as ``OPERAND_DIGITS`` balanced base-128 int8
  digits per entry, with a power-of-two scale per column, laid side by side. One int8 x int8 ->
  int32 cuBLAS GEMM per sample chunk (IMMA tensor cores on sm_75+, DP4A on sm_70) computes every
  digit product exactly (``INT32_EXACT_DIGIT_ROWS`` rows per chunk keep int32 exact), and
  recombining the digits in fp64 leaves only the operand's quantization: at most
  ``2^-(7 OPERAND_DIGITS - 3)`` of its column maximum per entry.
* CPU: the codes are converted to float64 per sample chunk (exact) and multiplied by DGEMM.

CuPy is passed in by the caller as the array module; this module never imports it.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np

from sv_pgs.compute_budget import ComputeBudget
from sv_pgs.genotype_buffers import (
    SIGNED_CODE_OFFSET,
    _CUBLAS_COMPUTE_32I,
    _CUBLAS_GEMM_DEFAULT,
    _CUDA_R_8I,
    _CUDA_R_32I,
)

DIGIT_BITS = 7
"""Each operand digit covers 7 bits: balanced digits lie in [-64, 63], safely inside int8."""

OPERAND_DIGITS = 6
"""Digits per operand entry: quantization at most 2^-39 (1.8e-12) of the column maximum, so a
dot product over 1e5 samples keeps a relative error far below Stage 2's 1e-7 gradient tolerance."""

_DIGIT_HALF = 1 << (DIGIT_BITS - 1)
_DIGIT_MASK = (1 << DIGIT_BITS) - 1

INT32_EXACT_DIGIT_ROWS = (2**31 - 1) // (SIGNED_CODE_OFFSET * _DIGIT_HALF)
"""Largest sample count whose int32 sum of ``s_i d_i`` (|d| <= 64) cannot overflow (264,208)."""

_CPU_CONVERSION_SHARE = 0.25
"""Share of the host budget one float64 conversion chunk of the codes may use."""

_CUDA_OPERAND_SHARE = 0.25
"""Share of the device budget the digit operand and integer products of one chunk may use."""


def operand_digits(dense: Any, array_module: ModuleType) -> tuple[Any, Any]:
    """Split ``dense`` [rows, K] into balanced base-128 digits.

    Returns ``(digits, scale)``: ``digits`` is int8 [rows, OPERAND_DIGITS * K] in column-major
    order, digit ``d`` of column ``k`` in column ``d * K + k``, and
    ``sum_d digits[:, d * K + k] * 128^d == rint(dense[:, k] * scale[k])`` exactly. ``scale`` is a
    power of two per column that puts the column maximum in (2^(7m-3), 2^(7m-2)].
    """
    values = array_module.asarray(dense, dtype=array_module.float64)
    rows, columns = values.shape
    magnitude = array_module.max(array_module.abs(values), axis=0) if rows else array_module.zeros(columns)
    exponent = (DIGIT_BITS * OPERAND_DIGITS - 2) - array_module.ceil(array_module.log2(array_module.where(magnitude > 0, magnitude, 1.0)))
    scale = array_module.exp2(exponent)
    integers = array_module.rint(values * scale[None, :]).astype(array_module.int64)
    digits = array_module.empty((rows, OPERAND_DIGITS * columns), dtype=array_module.int8, order="F")
    for digit_index in range(OPERAND_DIGITS):
        low = ((integers + _DIGIT_HALF) & _DIGIT_MASK) - _DIGIT_HALF
        digits[:, digit_index * columns : (digit_index + 1) * columns] = low.astype(array_module.int8)
        integers = (integers - low) >> DIGIT_BITS
    return digits, scale


def recombine_digit_products(products: Any, scale: Any, array_module: ModuleType) -> Any:
    """Undo ``operand_digits`` on the right of an exact integer product [rows, OPERAND_DIGITS * K]."""
    columns = int(scale.shape[0])
    total = products[:, (OPERAND_DIGITS - 1) * columns :].astype(array_module.float64)
    for digit_index in range(OPERAND_DIGITS - 2, -1, -1):
        total = total * float(1 << DIGIT_BITS) + products[:, digit_index * columns : (digit_index + 1) * columns]
    return total / scale[None, :]


INT8_GEMM_ALIGNMENT = 4
"""cuBLAS runs int8 GEMMs only when the reduction length, both leading dimensions and every operand
offset are multiples of 4 (CUBLAS_STATUS_NOT_SUPPORTED otherwise). A block is zero-padded to
multiples of 4 on both axes: a zero signed code adds nothing to any product."""


def _aligned(count: int) -> int:
    return -(-count // INT8_GEMM_ALIGNMENT) * INT8_GEMM_ALIGNMENT


class CodeBlockTile:
    """Standardized genotypes ``X_b = (S - mean) / scale`` of one LD block, from its signed codes.

    ``signed_codes`` is int8 [p_b, n] on the device of ``array_module`` (``numpy`` or ``cupy``);
    ``means`` and ``scales`` are per variant in signed-code units. Implements the Stage 2
    ``GenotypeBlockTile`` protocol with fp64-accurate products. The codes are held zero-padded to
    ``INT8_GEMM_ALIGNMENT`` on both axes; every product returns the unpadded shape.
    """

    def __init__(self, signed_codes: Any, means: Any, scales: Any, array_module: ModuleType, budget: ComputeBudget) -> None:
        self._array_module = array_module
        codes = array_module.asarray(signed_codes)
        if codes.dtype != array_module.int8 or codes.ndim != 2:
            raise ValueError("signed_codes must be a 2-D int8 [variants, samples] array")
        self._variant_count, self._sample_count = (int(extent) for extent in codes.shape)
        self._codes = array_module.zeros((_aligned(self._variant_count), _aligned(self._sample_count)), dtype=array_module.int8)
        self._codes[: self._variant_count, : self._sample_count] = codes
        self._means = array_module.asarray(means, dtype=array_module.float64)
        self._scales = array_module.asarray(scales, dtype=array_module.float64)
        if self._means.shape != (self._variant_count,) or self._scales.shape != (self._variant_count,):
            raise ValueError("means and scales need one entry per variant")
        self._budget = budget

    @property
    def variant_count(self) -> int:
        return self._variant_count

    @property
    def sample_count(self) -> int:
        return self._sample_count

    def matmat(self, right: Any) -> Any:
        """X_b @ right for right of shape (p_b, K); returns (n, K)."""
        xp = self._array_module
        scaled = xp.asarray(right, dtype=xp.float64) / self._scales[:, None]
        return self._codes_transposed_times(scaled) - (self._means @ scaled)[None, :]

    def rmatmat(self, left: Any) -> Any:
        """X_b.T @ left for left of shape (n, K); returns (p_b, K)."""
        xp = self._array_module
        values = xp.asarray(left, dtype=xp.float64)
        centered = self._codes_times(values) - self._means[:, None] * values.sum(axis=0)[None, :]
        return centered / self._scales[:, None]

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

    def _sample_chunk(self, bytes_per_sample: int, exact_rows: int, share: float) -> int:
        """Samples per chunk: a multiple of ``INT8_GEMM_ALIGNMENT`` that fits ``share`` of the budget."""
        padded_samples = int(self._codes.shape[1])
        fitting = int(self._budget.working_bytes * share) // max(bytes_per_sample, 1)
        chunk = min(exact_rows, padded_samples, fitting) // INT8_GEMM_ALIGNMENT * INT8_GEMM_ALIGNMENT
        return max(INT8_GEMM_ALIGNMENT, chunk)

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
        if xp is np:
            chunk = self._sample_chunk(8 * variants, samples, _CPU_CONVERSION_SHARE)
            for start in range(0, samples, chunk):
                stop = min(start + chunk, samples)
                total += self._codes[:, start:stop].astype(np.float64) @ padded[start:stop]
            return total[: self._variant_count]
        chunk = self._sample_chunk(OPERAND_DIGITS * columns * 9 + 8 * columns, INT32_EXACT_DIGIT_ROWS, _CUDA_OPERAND_SHARE)
        products = xp.empty((variants, OPERAND_DIGITS * columns), dtype=xp.int32, order="F")
        for start in range(0, samples, chunk):
            stop = min(start + chunk, samples)
            digits, scale = operand_digits(padded[start:stop], xp)
            _cuda_int8_gemm(
                xp, rows=variants, columns=OPERAND_DIGITS * columns, depth=stop - start,
                left=self._codes, left_offset=start, left_lead=samples,
                right=digits, right_lead=stop - start, output=products, output_lead=variants,
            )
            total += recombine_digit_products(products, scale, xp)
        return total[: self._variant_count]

    def _codes_transposed_times(self, operand: Any) -> Any:
        """S^T @ operand for operand [p_b, K] float64; returns [n, K] float64."""
        xp = self._array_module
        variants, samples = (int(extent) for extent in self._codes.shape)
        columns = int(operand.shape[1])
        padded = self._padded_rows(operand, variants)
        if xp is np:
            chunk = self._sample_chunk(8 * variants, samples, _CPU_CONVERSION_SHARE)
            out = np.empty((samples, columns), dtype=np.float64)
            for start in range(0, samples, chunk):
                stop = min(start + chunk, samples)
                out[start:stop] = self._codes[:, start:stop].T.astype(np.float64) @ padded
            return out[: self._sample_count]
        if variants > INT32_EXACT_DIGIT_ROWS:
            raise ValueError(f"an LD block of {variants} variants exceeds the int32-exact depth {INT32_EXACT_DIGIT_ROWS}")
        # The reduction runs over variants, which the sample-major codes do not hold contiguously,
        # so each sample chunk is transposed to variant-contiguous order for the TN GEMM.
        digits, scale = operand_digits(padded, xp)
        digit_columns = OPERAND_DIGITS * columns
        chunk = self._sample_chunk(variants + 4 * digit_columns + 8 * columns, samples, _CUDA_OPERAND_SHARE)
        products = xp.empty((chunk, digit_columns), dtype=xp.int32, order="F")
        out = xp.empty((samples, columns), dtype=xp.float64)
        for start in range(0, samples, chunk):
            stop = min(start + chunk, samples)
            variant_contiguous = xp.ascontiguousarray(self._codes[:, start:stop].T)
            _cuda_int8_gemm(
                xp, rows=stop - start, columns=digit_columns, depth=variants,
                left=variant_contiguous, left_offset=0, left_lead=variants,
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
            chunk = self._sample_chunk(16 * variants, samples, _CPU_CONVERSION_SHARE)
            total = np.zeros((variants, variants), dtype=np.float64)
            for start in range(0, samples, chunk):
                stop = min(start + chunk, samples)
                converted = self._codes[:, start:stop].astype(np.float64)
                total += converted @ (padded_weights[start:stop, None] * converted.T)
            return total[: self._variant_count, : self._variant_count]
        total = xp.zeros((variants, variants), dtype=xp.float64)
        products = xp.empty((variants, OPERAND_DIGITS * variants), dtype=xp.int32, order="F")
        chunk = self._sample_chunk(OPERAND_DIGITS * variants * 9 + 8 * variants, INT32_EXACT_DIGIT_ROWS, _CUDA_OPERAND_SHARE)
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

"""Stage 0 on one CUDA device: exact int8 x int8 -> int32 cuBLAS GEMMs.

The device buffer holds ``s = code - 127`` as int8, one variant per row, columns in
the sample layout. Every genotype product is a cuBLAS ``gemmEx`` with int8 inputs and
int32 accumulation (IMMA tensor cores on sm_75 and newer, DP4A on sm_70), over at most
``INT32_EXACT_ROWS`` samples, so it is exact. Longer sample ranges are split and summed
in int64. Host tiles upload on a copy stream while the previous tile computes, and pair
weights use the same correctly rounded fp64 steps as the CPU reference, so both
backends choose identical blocks.
"""

from __future__ import annotations

import cupy as cp
import numpy as np
from cupy.cuda import cublas
from numpy.typing import NDArray

from sv_pgs.stage0.layout import LAYOUT_ALIGNMENT, SampleLayout
from sv_pgs.stage0.partition import PAIR_WEIGHT_SCALE
from sv_pgs.stage0.statistics import INT32_EXACT_ROWS, MAXIMUM_STORED_CODE, SIGNED_CODE_OFFSET

_CUDA_R_8I = 3
_CUDA_R_32I = 10
_CUBLAS_COMPUTE_32I = 72
_CUBLAS_GEMM_DEFAULT = -1
_ROW_ALIGNMENT = 16
_GRAM_PANEL = 512
_SAMPLE_CHUNK = INT32_EXACT_ROWS // LAYOUT_ALIGNMENT * LAYOUT_ALIGNMENT
_STAGING_BUFFERS = 2
_CROSS_SAMPLE_CHUNK = 1024

_KERNEL_SOURCE = rf"""
extern "C" __global__ void stage0_shift_gather(
    const unsigned char* __restrict__ codes, const long long code_stride,
    const long long* __restrict__ source_columns, const int width,
    signed char* __restrict__ out, const long long out_stride, int* __restrict__ invalid)
{{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const long long row = blockIdx.y;
    if (column >= width) return;
    const long long source = source_columns[column];
    signed char value = 0;
    if (source >= 0) {{
        const int code = codes[row * code_stride + source];
        if (code > {MAXIMUM_STORED_CODE}) atomicOr(invalid, 1);
        value = (signed char)(code - {SIGNED_CODE_OFFSET});
    }}
    out[row * out_stride + column] = value;
}}

extern "C" __global__ void stage0_pair_weights(
    const int* __restrict__ band, const int window_rows, const int tile_rows,
    const long long* __restrict__ row_sums, const long long* __restrict__ row_squares,
    const long long* __restrict__ column_sums, const long long* __restrict__ column_squares,
    const long long profile_count, const double null_value, const int tile_offset,
    const int maximum_distance, long long* __restrict__ weights)
{{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int column = blockIdx.y;
    if (row >= window_rows) return;
    const long long index = (long long)column * window_rows + row;
    const int distance = tile_offset + column - row;
    const long long row_variance = profile_count * row_squares[row] - row_sums[row] * row_sums[row];
    const long long column_variance = profile_count * column_squares[column] - column_sums[column] * column_sums[column];
    long long weight = 0;
    if (distance >= 1 && distance <= maximum_distance && row_variance > 0 && column_variance > 0) {{
        const long long numerator = profile_count * (long long)band[index] - row_sums[row] * column_sums[column];
        const double numerator_float = (double)numerator;
        const double squared = __ddiv_rn(
            __dmul_rn(numerator_float, numerator_float),
            __dmul_rn((double)row_variance, (double)column_variance));
        weight = __double2ll_rn(__dmul_rn(__dsub_rn(squared, null_value), {PAIR_WEIGHT_SCALE!r}));
    }}
    weights[index] = weight;
}}
"""

_SHIFT_GATHER = cp.RawKernel(_KERNEL_SOURCE, "stage0_shift_gather")
_PAIR_WEIGHTS = cp.RawKernel(_KERNEL_SOURCE, "stage0_pair_weights")
_SQUARE_SUM = cp.ReductionKernel(
    "T value", "int64 total", "(long long)value * value", "a + b", "total = a", "0", "stage0_square_sum"
)


def _aligned_rows(count: int) -> int:
    return -(-count // _ROW_ALIGNMENT) * _ROW_ALIGNMENT


def cuda_stage0_device_bytes(
    layout: SampleLayout, capacity_rows: int, tile_rows: int, block_cap: int, cross_product_columns: int
) -> int:
    """Device bytes the backend allocates, plus its largest per-call scratch."""
    buffer = (capacity_rows + _ROW_ALIGNMENT) * layout.width
    moments = 2 * 8 * (capacity_rows + _ROW_ALIGNMENT)
    staging = _STAGING_BUFFERS * tile_rows * layout.store_width + 8 * layout.width
    window_rows = _aligned_rows(block_cap + tile_rows)
    band = window_rows * _aligned_rows(tile_rows) * (4 + 8 + 8)
    gram_rows = _aligned_rows(block_cap)
    long_group = int(layout.group_widths.max()) > _SAMPLE_CHUNK
    gram = gram_rows * gram_rows * (4 + 4 + (8 if long_group else 0)) + block_cap * block_cap * 4
    cross = 0
    if cross_product_columns:
        cross = layout.width * cross_product_columns * 8 + _cross_scratch_bytes(block_cap)
    return buffer + moments + staging + max(band, gram + cross)


def _cross_scratch_bytes(block_cap: int) -> int:
    """fp64 scratch for one sample chunk of a block's cross-products."""
    return block_cap * _CROSS_SAMPLE_CHUNK * 8


class CudaStage0Backend:
    """Genotype buffer and exact products on one device (see the module docstring)."""

    def __init__(
        self,
        device_id: int,
        layout: SampleLayout,
        capacity_rows: int,
        tile_rows: int,
        cross_product_columns: NDArray[np.float64] | None,
    ) -> None:
        self.layout = layout
        self._device = cp.cuda.Device(device_id)
        with self._device:
            self._buffer = cp.zeros((capacity_rows + _ROW_ALIGNMENT, layout.width), dtype=cp.int8)
            self._profile_sums = cp.zeros(capacity_rows + _ROW_ALIGNMENT, dtype=cp.int64)
            self._profile_squares = cp.zeros(capacity_rows + _ROW_ALIGNMENT, dtype=cp.int64)
            self._source_columns = cp.asarray(layout.source_columns, dtype=cp.int64)
            self._staging = [cp.empty((tile_rows, layout.store_width), dtype=cp.uint8) for _ in range(_STAGING_BUFFERS)]
            self._invalid = cp.zeros(1, dtype=cp.int32)
            self._compute = cp.cuda.Stream(non_blocking=True)
            self._copy = cp.cuda.Stream(non_blocking=True)
            self._staged = [cp.cuda.Event() for _ in range(_STAGING_BUFFERS)]
            self._consumed = [cp.cuda.Event() for _ in range(_STAGING_BUFFERS)]
            for event in self._consumed:
                event.record(self._compute)
            self._cross = None
            if cross_product_columns is not None:
                columns = np.asarray(cross_product_columns, dtype=np.float64)
                if columns.ndim != 2 or columns.shape[0] != layout.store_width:
                    raise ValueError("cross_product_columns must be (store samples, columns)")
                gather = np.where(layout.source_columns >= 0, layout.source_columns, 0)
                laid_out = np.where(layout.source_columns[:, None] >= 0, columns[gather], 0.0)
                self._cross = cp.asarray(laid_out)
        self._int_one = np.ones(1, dtype=np.int32)
        self._int_zero = np.zeros(1, dtype=np.int32)

    def host_tile(self, tile_rows: int) -> NDArray[np.uint8]:
        """A pinned host buffer for one tile, so its upload runs asynchronously."""
        with self._device:
            memory = cp.cuda.alloc_pinned_memory(tile_rows * self.layout.store_width)
        return np.frombuffer(memory, dtype=np.uint8, count=tile_rows * self.layout.store_width).reshape(
            tile_rows, self.layout.store_width
        )

    def stage_tile(self, codes: NDArray[np.uint8], staging_index: int) -> None:
        """Start uploading ``codes`` into staging buffer ``staging_index`` on the copy stream."""
        if codes.dtype != np.uint8 or codes.ndim != 2 or codes.shape[1] != self.layout.store_width:
            raise ValueError("a tile must be uint8 [rows, store samples]")
        with self._device:
            self._copy.wait_event(self._consumed[staging_index])
            self._staging[staging_index][: codes.shape[0]].set(codes, stream=self._copy)
            self._staged[staging_index].record(self._copy)

    def release_staged_tile(self, staging_index: int) -> None:
        """Block until the upload from the host tile of ``staging_index`` finished."""
        self._staged[staging_index].synchronize()

    def load_staged_tile(self, staging_index: int, rows: int, slot: int) -> None:
        """Shift and lay out staged rows at ``slot`` and record their profile moments."""
        with self._device, self._compute:
            self._compute.wait_event(self._staged[staging_index])
            target = self._buffer[slot : slot + rows]
            threads = 256
            _SHIFT_GATHER(
                ((self.layout.width + threads - 1) // threads, rows),
                (threads,),
                (
                    self._staging[staging_index],
                    np.int64(self.layout.store_width),
                    self._source_columns,
                    np.int32(self.layout.width),
                    target,
                    np.int64(self.layout.width),
                    self._invalid,
                ),
            )
            self._consumed[staging_index].record(self._compute)
            sums = cp.zeros(rows, dtype=cp.int64)
            squares = cp.zeros(rows, dtype=cp.int64)
            for group in range(self.layout.group_count):
                low, high = self.layout.profile_range(group)
                profile = target[:, low:high]
                sums += profile.sum(axis=1, dtype=cp.int64)
                squares += _SQUARE_SUM(profile, axis=1)
            self._profile_sums[slot : slot + rows] = sums
            self._profile_squares[slot : slot + rows] = squares

    def check_codes(self) -> None:
        """Raise if any loaded code exceeded the dosage range (code 255 = missing)."""
        with self._device:
            if int(self._invalid.get(stream=self._compute)[0]):
                raise ValueError(f"a dosage code above {MAXIMUM_STORED_CODE} (missing) reached Stage 0")

    def _gemm(self, left_row: int, left_rows: int, right_row: int, right_rows: int, sample_low: int, samples: int,
              output: cp.ndarray, output_offset: int, output_lead: int, accumulate: bool) -> None:
        """``output[c, r] (+)= sum_k s[left_row + r, k] s[right_row + c, k]`` via column-major cuBLAS."""
        handle = cp.cuda.device.get_cublas_handle()
        cublas.setStream(handle, self._compute.ptr)
        width = self.layout.width
        base = self._buffer.data.ptr
        cublas.gemmEx(
            handle,
            cublas.CUBLAS_OP_T,
            cublas.CUBLAS_OP_N,
            left_rows,
            right_rows,
            samples,
            self._int_one.ctypes.data,
            base + left_row * width + sample_low,
            _CUDA_R_8I,
            width,
            base + right_row * width + sample_low,
            _CUDA_R_8I,
            width,
            (self._int_one if accumulate else self._int_zero).ctypes.data,
            output.data.ptr + 4 * output_offset,
            _CUDA_R_32I,
            output_lead,
            _CUBLAS_COMPUTE_32I,
            _CUBLAS_GEMM_DEFAULT,
        )

    def pair_weights(
        self, window_slot: int, tile_slot: int, tile_rows: int, maximum_distance: int
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Row and column sums of the fixed-point pair weights between the window
        ``[window_slot, tile_slot + tile_rows)`` and the tile ``[tile_slot, tile_slot + tile_rows)``."""
        window_rows = tile_slot + tile_rows - window_slot
        padded_window = _aligned_rows(window_rows)
        padded_tile = _aligned_rows(tile_rows)
        with self._device, self._compute:
            band = cp.empty((padded_tile, padded_window), dtype=cp.int32)
            for group in range(self.layout.group_count):
                low, high = self.layout.profile_range(group)
                self._gemm(window_slot, padded_window, tile_slot, padded_tile, low, high - low, band, 0, padded_window,
                           accumulate=group > 0)
            weights = cp.empty((tile_rows, padded_window), dtype=cp.int64)
            threads = 256
            _PAIR_WEIGHTS(
                ((window_rows + threads - 1) // threads, tile_rows),
                (threads,),
                (
                    band,
                    np.int32(padded_window),
                    np.int32(tile_rows),
                    self._profile_sums[window_slot:],
                    self._profile_squares[window_slot:],
                    self._profile_sums[tile_slot:],
                    self._profile_squares[tile_slot:],
                    np.int64(self.layout.profile_count),
                    np.float64(1.0 / (self.layout.profile_count - 1)),
                    np.int32(tile_slot - window_slot),
                    np.int32(maximum_distance),
                    weights,
                ),
            )
            row_weights = weights[:, :window_rows].sum(axis=0).get(stream=self._compute)
            column_weights = weights[:, :window_rows].sum(axis=1).get(stream=self._compute)
        return row_weights, column_weights

    def block_statistics(
        self, slot: int, rows: int
    ) -> tuple[NDArray[np.int64], NDArray[np.integer], NDArray[np.float64] | None]:
        """Exact per-group sums ``(G, rows)``, Grams ``(G, rows, rows)`` and cross-products."""
        groups = self.layout.group_count
        padded = _aligned_rows(rows)
        sums = np.zeros((groups, rows), dtype=np.int64)
        long_group = int(self.layout.group_widths.max()) > _SAMPLE_CHUNK
        grams = np.zeros((groups, rows, rows), dtype=np.int64 if long_group else np.int32)
        with self._device, self._compute:
            block = self._buffer[slot : slot + rows]
            output = cp.empty((padded, padded), dtype=cp.int32)
            for group in range(groups):
                low, high = self.layout.group_range(group)
                sums[group] = block[:, low:high].sum(axis=1, dtype=cp.int64).get(stream=self._compute)
                total = None
                for chunk_low in range(low, high, _SAMPLE_CHUNK):
                    chunk_samples = min(_SAMPLE_CHUNK, high - chunk_low)
                    for panel_row in range(0, padded, _GRAM_PANEL):
                        panel_rows = min(_GRAM_PANEL, padded - panel_row)
                        self._gemm(slot + panel_row, panel_rows, slot + panel_row, padded - panel_row, chunk_low,
                                   chunk_samples, output, panel_row * padded + panel_row, padded, accumulate=False)
                    if long_group:
                        total = output.astype(cp.int64) if total is None else total + output
                lower = output if total is None else total
                symmetric = cp.tril(lower[:rows, :rows]) + cp.tril(lower[:rows, :rows], -1).T
                grams[group] = symmetric.get(stream=self._compute)
            self._compute.synchronize()
        return sums, grams, None if self._cross is None else self.cross_products(slot, rows)

    def cross_products(self, slot: int, rows: int) -> NDArray[np.float64]:
        """Per-group ``sum_i s_i y_i^T`` ``(G, rows, columns)`` in fp64 cuBLAS GEMMs."""
        if self._cross is None:
            raise ValueError("the backend was built without cross-product columns")
        result = np.zeros((self.layout.group_count, rows, self._cross.shape[1]), dtype=np.float64)
        with self._device, self._compute:
            block = self._buffer[slot : slot + rows]
            for group in range(self.layout.group_count):
                low, high = self.layout.group_range(group)
                total = cp.zeros((rows, self._cross.shape[1]), dtype=cp.float64)
                for chunk_low in range(low, high, _CROSS_SAMPLE_CHUNK):
                    chunk_high = min(chunk_low + _CROSS_SAMPLE_CHUNK, high)
                    total += block[:, chunk_low:chunk_high].astype(cp.float64) @ self._cross[chunk_low:chunk_high]
                result[group] = total.get(stream=self._compute)
            self._compute.synchronize()
        return result

    def move_rows(self, source_slot: int, target_slot: int, rows: int) -> None:
        """Copy rows to a lower slot, front to back in pieces that never overlap."""
        step = source_slot - target_slot
        if step == 0:
            return
        with self._device, self._compute:
            for offset in range(0, rows, step):
                count = min(step, rows - offset)
                for array in (self._buffer, self._profile_sums, self._profile_squares):
                    array[target_slot + offset : target_slot + offset + count] = array[
                        source_slot + offset : source_slot + offset + count
                    ]

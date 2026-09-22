"""Device buffers of the Stage 0 genotype pass: exact integer genotype products on a CPU or a GPU.

A buffer holds ``s = code - 127`` (``code`` = round(127 DS) in 0..254) as int8, one variant
per row, with the samples in a ``SampleLayout``. Its products are exact integers:

* CUDA: cuBLAS ``gemmEx`` with int8 inputs and int32 accumulation (IMMA tensor cores on
  sm_75 and newer, DP4A on sm_70) over at most ``INT32_EXACT_ROWS`` samples per call;
  longer sample ranges are split and summed in int64.
* Host: float32 GEMMs over sample chunks of at most ``FLOAT32_EXACT_ROWS`` columns, whose
  every partial sum is an integer below 2^24 whatever the BLAS blocking; chunk results
  accumulate in fp64 (exact below 2^53). Float32 operands are stored chunk-major
  (``[chunk, variant, sample]``) so every GEMM operand is contiguous and never copied.
  Independent output panels run on a thread pool of single-threaded BLAS workers, so the
  summation order never depends on the threads.

The pair weights that choose LD-block boundaries are ``ld_partition.fixed_point_pair_weights``
on the host and one CUDA kernel taking the same correctly rounded steps on the device, so both
choose the same blocks. CuPy is passed in by the caller (``compute_budget._try_import_cupy``); this
module never imports it.
"""

from __future__ import annotations

import contextlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable, Iterator

import numpy as np
from numpy.typing import NDArray
from threadpoolctl import ThreadpoolController

from sv_pgs.dosage_store import MAXIMUM_CODE
from sv_pgs.ld_partition import PAIR_WEIGHT_SCALE, fixed_point_pair_weights

SIGNED_CODE_OFFSET = 127
"""``s = code - SIGNED_CODE_OFFSET`` lies in [-127, 127] for every stored code."""

INT32_EXACT_ROWS = int(np.iinfo(np.int32).max) // (SIGNED_CODE_OFFSET * SIGNED_CODE_OFFSET)
"""Largest sample count whose int32 sum of ``s_i s_j`` cannot overflow (133,144)."""

FLOAT32_EXACT_ROWS = 2 ** (np.finfo(np.float32).nmant + 1) // (SIGNED_CODE_OFFSET * SIGNED_CODE_OFFSET)
"""Largest sample count whose float32 sum of ``s_i s_j`` stays an exact integer in any order (1,040)."""

LAYOUT_ALIGNMENT = 64
"""Column alignment of every group range: int8 tensor-core operands need aligned rows."""

PROFILE_SAMPLE_TARGET = 16384
"""Size of the cut-cost profile subsample. A cut cost ``C`` (a sum of squared correlations)
has relative standard error about ``2 / sqrt(C n)``: below 1.6% for ``C >= 1`` here."""

_ROW_ALIGNMENT = 16
_STAGING_BUFFERS = 2
_HOST_SAMPLE_CHUNK = 1 << (FLOAT32_EXACT_ROWS.bit_length() - 1)
_HOST_PANEL = 512
_HOST_CROSS_ROWS = 64
_CROSS_SAMPLE_CHUNK = 8192
_CUDA_SAMPLE_CHUNK = INT32_EXACT_ROWS // LAYOUT_ALIGNMENT * LAYOUT_ALIGNMENT
_CUDA_GRAM_PANEL = 512


@dataclass(frozen=True, slots=True)
class SampleLayout:
    """Buffer columns: included samples grouped, each group aligned, profile samples first.

    ``source_columns[k]`` is the store column behind buffer column ``k`` (``-1``: a zero pad
    column, which adds nothing to any product). Group ``g`` spans ``group_range(g)``, and its
    profile samples, every ``stride``-th included sample in store order (independent of the
    group labels), span ``profile_range(g)`` at its front.
    """

    source_columns: NDArray[np.int64]
    group_offsets: NDArray[np.int64]
    group_widths: NDArray[np.int64]
    group_counts: NDArray[np.int64]
    profile_counts: NDArray[np.int64]
    store_width: int

    @property
    def width(self) -> int:
        return int(self.source_columns.shape[0])

    @property
    def group_count(self) -> int:
        return int(self.group_counts.shape[0])

    @property
    def profile_count(self) -> int:
        return int(self.profile_counts.sum())

    @property
    def profile_width(self) -> int:
        return sum(_aligned(int(count)) for count in self.profile_counts)

    def group_range(self, group: int) -> tuple[int, int]:
        start = int(self.group_offsets[group])
        return start, start + int(self.group_widths[group])

    def profile_range(self, group: int) -> tuple[int, int]:
        start = int(self.group_offsets[group])
        return start, start + _aligned(int(self.profile_counts[group]))


def _aligned(count: int) -> int:
    return -(-count // LAYOUT_ALIGNMENT) * LAYOUT_ALIGNMENT


def _aligned_rows(count: int) -> int:
    return -(-count // _ROW_ALIGNMENT) * _ROW_ALIGNMENT


def build_sample_layout(sample_groups: NDArray[np.int64], profile_target: int = PROFILE_SAMPLE_TARGET) -> SampleLayout:
    """Lay out store columns labelled ``0..G-1`` by ``sample_groups`` (``-1`` excludes a column)."""
    groups = np.asarray(sample_groups, dtype=np.int64)
    if groups.ndim != 1 or np.any(groups < -1):
        raise ValueError("sample_groups must be 1-D labels in -1 (excluded) or 0..G-1")
    included = np.flatnonzero(groups >= 0)
    if included.shape[0] < 3:
        raise ValueError("Stage 0 needs at least three included samples")
    group_count = int(groups.max()) + 1
    is_profile = np.zeros(groups.shape[0], dtype=np.bool_)
    is_profile[included[:: -(-included.shape[0] // profile_target)]] = True
    columns: list[NDArray[np.int64]] = []
    offsets = np.zeros(group_count, dtype=np.int64)
    widths = np.zeros(group_count, dtype=np.int64)
    counts = np.zeros(group_count, dtype=np.int64)
    profile_counts = np.zeros(group_count, dtype=np.int64)
    cursor = 0
    for group in range(group_count):
        members = np.flatnonzero(groups == group)
        if members.shape[0] == 0:
            raise ValueError(f"sample group {group} is empty")
        profile_members = members[is_profile[members]]
        profile_pad = np.full(_aligned(profile_members.shape[0]) - profile_members.shape[0], -1, dtype=np.int64)
        ordered = np.concatenate((profile_members, profile_pad, members[~is_profile[members]]))
        padded = np.concatenate((ordered, np.full(_aligned(ordered.shape[0]) - ordered.shape[0], -1, dtype=np.int64)))
        offsets[group], widths[group] = cursor, padded.shape[0]
        counts[group], profile_counts[group] = members.shape[0], profile_members.shape[0]
        columns.append(padded)
        cursor += padded.shape[0]
    return SampleLayout(
        source_columns=np.concatenate(columns),
        group_offsets=offsets,
        group_widths=widths,
        group_counts=counts,
        profile_counts=profile_counts,
        store_width=int(groups.shape[0]),
    )


def _laid_out_columns(layout: SampleLayout, columns: NDArray[np.float64] | None) -> NDArray[np.float64] | None:
    """Cross-product columns ``[store samples, q]`` in buffer order, zero on pad columns."""
    if columns is None:
        return None
    values = np.asarray(columns, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != layout.store_width:
        raise ValueError("cross-product columns must be (store samples, columns)")
    gather = np.where(layout.source_columns >= 0, layout.source_columns, 0)
    return np.where(layout.source_columns[:, None] >= 0, values[gather], 0.0)


def _chunks(samples: int) -> int:
    return -(-samples // _HOST_SAMPLE_CHUNK)


def _store_chunked(target: NDArray[np.float32], rows: slice, values: NDArray[np.int8]) -> None:
    """Write ``values`` ``[rows, samples]`` into chunk-major ``target[:, rows, :]``, zero-padding the last chunk."""
    samples = values.shape[1]
    for chunk in range(target.shape[0]):
        low = chunk * _HOST_SAMPLE_CHUNK
        high = min(low + _HOST_SAMPLE_CHUNK, samples)
        target[chunk, rows, : high - low] = values[:, low:high]
        target[chunk, rows, high - low :] = 0.0


def host_buffer_bytes(layout: SampleLayout, capacity_rows: int, block_cap: int, cross_columns: int, workers: int) -> int:
    """Host bytes a ``HostGenotypeBuffer`` holds at its peak (one block's products in flight)."""
    rows = capacity_rows + _ROW_ALIGNMENT
    buffer = rows * (layout.width + 4 * _chunks(layout.profile_width) * _HOST_SAMPLE_CHUNK + 16)
    staging = (_STAGING_BUFFERS + 1) * (block_cap // 2 + LAYOUT_ALIGNMENT) * layout.store_width
    block = block_cap * _chunks(int(layout.group_widths.max())) * _HOST_SAMPLE_CHUNK * 4 + 3 * block_cap * block_cap * 8
    # the Gram against the previous block: its float32 rows and the int64 result per group
    adjacent = block_cap * _chunks(int(layout.group_widths.max())) * _HOST_SAMPLE_CHUNK * 4 + (layout.group_count + 1) * block_cap * block_cap * 8
    workers_scratch = workers * (_HOST_PANEL * _HOST_PANEL * 8 + 2 * _HOST_PANEL * _HOST_SAMPLE_CHUNK * 4)
    return buffer + staging + block + adjacent + workers_scratch + layout.width * cross_columns * 8


def cuda_buffer_bytes(layout: SampleLayout, capacity_rows: int, tile_rows: int, block_cap: int, cross_columns: int) -> int:
    """Device bytes a ``CudaGenotypeBuffer`` holds at its peak."""
    rows = capacity_rows + _ROW_ALIGNMENT
    resident = rows * (layout.width + 16) + _STAGING_BUFFERS * tile_rows * layout.store_width + 8 * layout.width
    band_rows = _aligned_rows(block_cap + tile_rows)
    band = band_rows * _aligned_rows(tile_rows) * (4 + 8)
    gram_rows = _aligned_rows(block_cap)
    long_group = int(layout.group_widths.max()) > _CUDA_SAMPLE_CHUNK
    gram = layout.group_count * gram_rows * gram_rows * 8 + gram_rows * gram_rows * (4 + (8 if long_group else 0))
    # the Gram against the previous block has the same shape bound as the block's own
    adjacent = gram
    cross = layout.width * cross_columns * 8 + block_cap * _CROSS_SAMPLE_CHUNK * 8
    # the projection step's fp64 correlation matrix and the one full-width panel temporary its rank-one update
    # forms (the device runs every row as one panel): 2 x 11.6 GB at a 38k cap, which the model omitted and the
    # A40 refused (bench-sim scenario_000, 40,000 samples, 2026-09-22)
    projection = 2 * gram_rows * gram_rows * 8
    return resident + max(band, gram + adjacent + cross + projection)


class HostGenotypeBuffer:
    """The genotype buffer in host memory (see the module docstring)."""

    array_module = np

    def __init__(
        self,
        layout: SampleLayout,
        capacity_rows: int,
        cross_product_columns: NDArray[np.float64] | None,
        worker_count: int,
    ) -> None:
        self.layout = layout
        rows = capacity_rows + _ROW_ALIGNMENT
        self._buffer = np.zeros((rows, layout.width), dtype=np.int8)
        included_column = int(layout.source_columns[layout.source_columns >= 0][0])
        self._gather = np.where(layout.source_columns >= 0, layout.source_columns, included_column)
        self._pad_columns = np.flatnonzero(layout.source_columns < 0)
        self._profile_columns = np.concatenate(
            [np.arange(*layout.profile_range(group)) for group in range(layout.group_count)]
        )
        self._profile = np.zeros((_chunks(self._profile_columns.shape[0]), rows, _HOST_SAMPLE_CHUNK), dtype=np.float32)
        self._profile_sums = np.zeros(rows, dtype=np.int64)
        self._profile_squares = np.zeros(rows, dtype=np.int64)
        self._cross = _laid_out_columns(layout, cross_product_columns)
        self._staged: dict[int, NDArray[np.uint8]] = {}
        self._pool = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="stage0-host")
        self._worker_count = worker_count
        self._blas = ThreadpoolController()

    def close(self) -> None:
        self._pool.shutdown()

    def context(self) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()

    def _parallel(self, function: Callable[[Any], None], tasks: list[Any]) -> None:
        with self._blas.limit(limits=1, user_api="blas"):
            for _ in self._pool.map(function, tasks):
                pass

    def _row_ranges(self, total: int, piece: int) -> list[tuple[int, int]]:
        return [(start, min(start + piece, total)) for start in range(0, total, piece)]

    def to_host(self, array: NDArray) -> NDArray:
        """Host arrays are already on the host."""
        return array

    def parallel_rows(self, function: Callable[[int, int], None], rows: int) -> None:
        """Run ``function(start, stop)`` over row panels on the worker threads."""
        self._parallel(lambda row_range: function(*row_range), self._row_ranges(rows, _HOST_PANEL))

    def host_tile(self, tile_rows: int) -> NDArray[np.uint8]:
        return np.empty((tile_rows, self.layout.store_width), dtype=np.uint8)

    def stage_tile(self, codes: NDArray[np.uint8], staging_index: int) -> None:
        _check_tile(codes, self.layout)
        self._staged[staging_index] = codes

    def release_staged_tile(self, staging_index: int) -> None:
        """Loading copies the host tile, so it is free as soon as it is loaded."""

    def check_codes(self) -> None:
        """Codes are validated while they load."""

    def load_staged_tile(self, staging_index: int, rows: int, slot: int) -> None:
        """Shift, validate and lay out the staged rows at ``slot``; record their profile moments."""
        codes = self._staged.pop(staging_index)[:rows]
        invalid = np.zeros(1, dtype=np.bool_)

        def load(row_range: tuple[int, int]) -> None:
            start, stop = row_range
            gathered = np.take(codes[start:stop], self._gather, axis=1)
            if int(gathered.max()) > MAXIMUM_CODE:
                invalid[0] = True
            target = self._buffer[slot + start : slot + stop]
            np.subtract(gathered.view(np.int8), np.int8(SIGNED_CODE_OFFSET), out=target)
            target[:, self._pad_columns] = 0
            profile = np.take(target, self._profile_columns, axis=1)
            _store_chunked(self._profile, slice(slot + start, slot + stop), profile)
            self._profile_sums[slot + start : slot + stop] = profile.sum(axis=1, dtype=np.int64)
            self._profile_squares[slot + start : slot + stop] = np.square(profile.astype(np.int32)).sum(axis=1, dtype=np.int64)

        self._parallel(load, self._row_ranges(rows, -(-rows // self._worker_count)))
        if invalid[0]:
            raise ValueError(f"a dosage code above {MAXIMUM_CODE} (missing) reached Stage 0")

    def pair_weights(
        self, window_slot: int, tile_slot: int, tile_rows: int, maximum_distance: int
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Row and column sums of the fixed-point pair weights of window ``[window_slot, tile end)``
        against tile ``[tile_slot, tile_slot + tile_rows)``."""
        window_stop = tile_slot + tile_rows
        band = _host_exact_products(
            self._profile[:, window_slot:window_stop], self._profile[:, tile_slot:window_stop], self._parallel,
            self._worker_count, symmetric=False,
        )
        row_panels = self._row_ranges(band.shape[0], _HOST_PANEL)
        column_panels = self._row_ranges(tile_rows, _HOST_PANEL // 2)
        row_parts = np.zeros((len(column_panels), band.shape[0]), dtype=np.int64)
        column_parts = np.zeros((len(row_panels), tile_rows), dtype=np.int64)
        panels = [(row_index, column_index) for row_index in range(len(row_panels)) for column_index in range(len(column_panels))]

        def weigh(panel: tuple[int, int]) -> None:
            (row_start, row_stop), (column_start, column_stop) = row_panels[panel[0]], column_panels[panel[1]]
            weights = fixed_point_pair_weights(
                band[row_start:row_stop, column_start:column_stop],
                self._profile_sums[window_slot + row_start : window_slot + row_stop],
                self._profile_squares[window_slot + row_start : window_slot + row_stop],
                self._profile_sums[tile_slot + column_start : tile_slot + column_stop],
                self._profile_squares[tile_slot + column_start : tile_slot + column_stop],
                self.layout.profile_count,
                tile_slot - window_slot + column_start - row_start,
                maximum_distance,
            )
            row_parts[panel[1], row_start:row_stop] = weights.sum(axis=1)
            column_parts[panel[0], column_start:column_stop] = weights.sum(axis=0)

        self._parallel(weigh, panels)
        return row_parts.sum(axis=0), column_parts.sum(axis=0)

    def _group_values(self, slot: int, rows: int, group: int, sums: NDArray[np.int64] | None) -> NDArray[np.float32]:
        """Rows ``[slot, slot + rows)`` of one group's samples as chunk-major float32; their sums
        go into ``sums`` when it is given."""
        low, high = self.layout.group_range(group)
        block = self._buffer[slot : slot + rows]
        values = np.empty((_chunks(high - low), rows, _HOST_SAMPLE_CHUNK), dtype=np.float32)

        def convert(row_range: tuple[int, int]) -> None:
            start, stop = row_range
            _store_chunked(values, slice(start, stop), block[start:stop, low:high])
            if sums is not None:
                sums[start:stop] = block[start:stop, low:high].sum(axis=1, dtype=np.int64)

        self._parallel(convert, self._row_ranges(rows, -(-rows // self._worker_count)))
        return values

    def block_statistics(
        self, slot: int, rows: int
    ) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64] | None]:
        """Exact per-group sums ``(G, rows)``, Grams ``(G, rows, rows)`` and cross-products."""
        groups = self.layout.group_count
        sums = np.zeros((groups, rows), dtype=np.int64)
        grams = np.zeros((groups, rows, rows), dtype=np.int64)
        for group in range(groups):
            values = self._group_values(slot, rows, group, sums[group])
            grams[group] = _host_exact_products(values, values, self._parallel, self._worker_count, symmetric=True)
        return sums, grams, None if self._cross is None else self.cross_products(slot, rows)

    def cross_block_grams(self, first_slot: int, first_rows: int, second_slot: int, second_rows: int) -> NDArray[np.int64]:
        """Exact per-group Grams between two blocks' rows, ``sum_{i in g} s_i^(1) s_i^(2)T``
        ``(G, first_rows, second_rows)``: adjacent LD blocks' coupling."""
        grams = np.zeros((self.layout.group_count, first_rows, second_rows), dtype=np.int64)
        for group in range(self.layout.group_count):
            first = self._group_values(first_slot, first_rows, group, None)
            second = self._group_values(second_slot, second_rows, group, None)
            grams[group] = _host_exact_products(first, second, self._parallel, self._worker_count, symmetric=False)
        return grams

    def cross_products(self, slot: int, rows: int) -> NDArray[np.float64]:
        """Per-group ``sum_i s_i y_i^T`` ``(G, rows, columns)`` in fp64, in a fixed summation order."""
        if self._cross is None:
            raise ValueError("the buffer was built without cross-product columns")
        cross = self._cross
        block = self._buffer[slot : slot + rows]
        result = np.zeros((self.layout.group_count, rows, cross.shape[1]), dtype=np.float64)
        for group in range(self.layout.group_count):
            low, high = self.layout.group_range(group)

            def panel(row_range: tuple[int, int]) -> None:
                start, stop = row_range
                total = np.zeros((stop - start, cross.shape[1]), dtype=np.float64)
                for sample_low in range(low, high, _CROSS_SAMPLE_CHUNK):
                    sample_high = min(sample_low + _CROSS_SAMPLE_CHUNK, high)
                    total += block[start:stop, sample_low:sample_high].astype(np.float64) @ cross[sample_low:sample_high]
                result[group, start:stop] = total

            self._parallel(panel, self._row_ranges(rows, _HOST_CROSS_ROWS))
        return result

    def move_rows(self, source_slot: int, target_slot: int, rows: int) -> None:
        """Copy rows to a lower slot (NumPy resolves overlapping copies)."""
        for array in (self._buffer, self._profile_sums, self._profile_squares):
            array[target_slot : target_slot + rows] = array[source_slot : source_slot + rows]
        self._profile[:, target_slot : target_slot + rows] = self._profile[:, source_slot : source_slot + rows]


def _check_tile(codes: NDArray[np.uint8], layout: SampleLayout) -> None:
    if codes.dtype != np.uint8 or codes.ndim != 2 or codes.shape[1] != layout.store_width:
        raise ValueError("a tile must be uint8 [rows, store samples]")


def _host_exact_products(
    left: NDArray[np.float32],
    right: NDArray[np.float32],
    parallel: Callable[[Callable[[Any], None], list[Any]], None],
    workers: int,
    symmetric: bool,
) -> NDArray[np.int64]:
    """``left @ right.T`` over all chunks for chunk-major integer-valued float32 operands
    ``[chunk, rows, _HOST_SAMPLE_CHUNK]`` with ``|value| <= 127``, exactly.

    Output panels shrink from ``_HOST_PANEL`` until every worker has two; ``symmetric``
    (``left is right``) computes the panels on and above the diagonal and mirrors them.
    NumPy's matmul releases the GIL (SciPy's BLAS wrappers do not), so the workers overlap,
    and it calls SYRK for the diagonal panels.
    """
    rows, columns = left.shape[1], right.shape[1]
    result = np.zeros((rows, columns), dtype=np.int64)
    size = _HOST_PANEL
    while size > _ROW_ALIGNMENT and -(-rows // size) * -(-columns // size) < 2 * workers * (2 if symmetric else 1):
        size //= 2
    panels = [
        (row_start, column_start)
        for row_start in range(0, rows, size)
        for column_start in range(0, columns, size)
        if not symmetric or column_start >= row_start
    ]

    def panel(corner: tuple[int, int]) -> None:
        row_start, column_start = corner
        row_stop, column_stop = min(row_start + size, rows), min(column_start + size, columns)
        total = np.zeros((row_stop - row_start, column_stop - column_start), dtype=np.float64)
        for chunk in range(left.shape[0]):
            total += left[chunk, row_start:row_stop] @ right[chunk, column_start:column_stop].T
        result[row_start:row_stop, column_start:column_stop] = total

    parallel(panel, panels)
    if symmetric:
        result = np.triu(result) + np.triu(result, 1).T
    return result


_CUDA_SHIFT_GATHER = r"""
extern "C" __global__ void stage0_shift_gather(
    const unsigned char* __restrict__ codes, const long long code_stride,
    const long long* __restrict__ source_columns, const int width,
    signed char* __restrict__ out, const long long out_stride, int* __restrict__ invalid)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const long long row = blockIdx.y;
    if (column >= width) return;
    const long long source = source_columns[column];
    signed char value = 0;
    if (source >= 0) {
        const int code = codes[row * code_stride + source];
        if (code > MAXIMUM_CODE) atomicOr(invalid, 1);
        value = (signed char)(code - SIGNED_CODE_OFFSET);
    }
    out[row * out_stride + column] = value;
}
""".replace("MAXIMUM_CODE", str(MAXIMUM_CODE)).replace("SIGNED_CODE_OFFSET", str(SIGNED_CODE_OFFSET))

_CUDA_PAIR_WEIGHTS = r"""
extern "C" __global__ void stage0_pair_weights(
    const int* __restrict__ band, const int band_lead, const int window_rows, const int tile_rows,
    const long long* __restrict__ row_sums, const long long* __restrict__ row_squares,
    const long long* __restrict__ column_sums, const long long* __restrict__ column_squares,
    const long long profile_count, const double null_value, const int first_distance,
    const int maximum_distance, long long* __restrict__ weights)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int column = blockIdx.y;
    if (row >= window_rows) return;
    const long long row_sum = row_sums[row];
    const long long column_sum = column_sums[column];
    const long long row_variance = profile_count * row_squares[row] - row_sum * row_sum;
    const long long column_variance = profile_count * column_squares[column] - column_sum * column_sum;
    const int distance = first_distance + column - row;
    long long weight = 0;
    if (distance >= 1 && distance <= maximum_distance && row_variance > 0 && column_variance > 0) {
        const long long numerator = profile_count * (long long)band[(long long)column * band_lead + row] - row_sum * column_sum;
        const double numerator_float = (double)numerator;
        const double squared = __ddiv_rn(__dmul_rn(numerator_float, numerator_float),
                                         __dmul_rn((double)row_variance, (double)column_variance));
        weight = __double2ll_rn(__dmul_rn(__dsub_rn(squared, null_value), PAIR_WEIGHT_SCALE));
    }
    weights[(long long)column * window_rows + row] = weight;
}
""".replace("PAIR_WEIGHT_SCALE", repr(PAIR_WEIGHT_SCALE))

_CUDA_R_8I = 3
_CUDA_R_32I = 10
_CUBLAS_COMPUTE_32I = 72
_CUBLAS_GEMM_DEFAULT = -1


class CudaGenotypeBuffer:
    """The genotype buffer on one CUDA device (see the module docstring).

    Host tiles upload on a copy stream into one of two staging buffers while the previous
    tile computes on the compute stream.
    """

    def __init__(
        self,
        cupy: ModuleType,
        device_id: int,
        layout: SampleLayout,
        capacity_rows: int,
        tile_rows: int,
        cross_product_columns: NDArray[np.float64] | None,
    ) -> None:
        self.layout = layout
        self.array_module = cupy
        self._cupy = cupy
        self._device = cupy.cuda.Device(device_id)
        rows = capacity_rows + _ROW_ALIGNMENT
        with self._device:
            self._buffer = cupy.zeros((rows, layout.width), dtype=cupy.int8)
            self._profile_sums = cupy.zeros(rows, dtype=cupy.int64)
            self._profile_squares = cupy.zeros(rows, dtype=cupy.int64)
            self._source_columns = cupy.asarray(layout.source_columns, dtype=cupy.int64)
            self._staging = [cupy.empty((tile_rows, layout.store_width), dtype=cupy.uint8) for _ in range(_STAGING_BUFFERS)]
            self._invalid = cupy.zeros(1, dtype=cupy.int32)
            self._compute = cupy.cuda.Stream(non_blocking=True)
            self._copy = cupy.cuda.Stream(non_blocking=True)
            self._staged = [cupy.cuda.Event() for _ in range(_STAGING_BUFFERS)]
            self._consumed = [cupy.cuda.Event() for _ in range(_STAGING_BUFFERS)]
            for event in self._consumed:
                event.record(self._compute)
            laid_out = _laid_out_columns(layout, cross_product_columns)
            self._cross = None if laid_out is None else cupy.asarray(laid_out)
        self._shift_gather = cupy.RawKernel(_CUDA_SHIFT_GATHER, "stage0_shift_gather")
        self._pair_weights = cupy.RawKernel(_CUDA_PAIR_WEIGHTS, "stage0_pair_weights")
        self._square_sum = cupy.ReductionKernel(
            "T value", "int64 total", "(long long)value * value", "a + b", "total = a", "0", "stage0_square_sum"
        )
        self._one = np.ones(1, dtype=np.int32)
        self._zero = np.zeros(1, dtype=np.int32)

    def close(self) -> None:
        with self._device:
            self._compute.synchronize()
            self._copy.synchronize()

    @contextlib.contextmanager
    def context(self) -> Iterator[None]:
        """The device and compute stream every array of this buffer belongs to."""
        with self._device, self._compute:
            yield

    def parallel_rows(self, function: Callable[[int, int], None], rows: int) -> None:
        """The device runs all rows as one panel."""
        function(0, rows)

    def to_host(self, array: Any) -> NDArray:
        """Copy a device array into pinned host memory (recycled by CuPy's pinned pool) at full link speed."""
        with self.context():
            source = self._cupy.ascontiguousarray(array)
            memory = self._cupy.cuda.alloc_pinned_memory(max(source.nbytes, 1))
            host = np.frombuffer(memory, dtype=source.dtype, count=source.size).reshape(source.shape)
            source.get(stream=self._compute, out=host)
        return host

    def host_tile(self, tile_rows: int) -> NDArray[np.uint8]:
        """A pinned host tile, so its upload runs asynchronously."""
        size = tile_rows * self.layout.store_width
        with self._device:
            memory = self._cupy.cuda.alloc_pinned_memory(size)
        return np.frombuffer(memory, dtype=np.uint8, count=size).reshape(tile_rows, self.layout.store_width)

    def stage_tile(self, codes: NDArray[np.uint8], staging_index: int) -> None:
        """Start uploading ``codes`` into staging buffer ``staging_index`` on the copy stream."""
        _check_tile(codes, self.layout)
        with self._device:
            self._copy.wait_event(self._consumed[staging_index])
            self._staging[staging_index][: codes.shape[0]].set(codes, stream=self._copy)
            self._staged[staging_index].record(self._copy)

    def release_staged_tile(self, staging_index: int) -> None:
        """Block until the upload from the host tile of ``staging_index`` finished."""
        self._staged[staging_index].synchronize()

    def load_staged_tile(self, staging_index: int, rows: int, slot: int) -> None:
        """Shift and lay out staged rows at ``slot`` and record their profile moments."""
        cupy = self._cupy
        with self.context():
            self._compute.wait_event(self._staged[staging_index])
            target = self._buffer[slot : slot + rows]
            threads = 256
            self._shift_gather(
                ((self.layout.width + threads - 1) // threads, rows),
                (threads,),
                (self._staging[staging_index], np.int64(self.layout.store_width), self._source_columns,
                 np.int32(self.layout.width), target, np.int64(self.layout.width), self._invalid),
            )
            self._consumed[staging_index].record(self._compute)
            sums = cupy.zeros(rows, dtype=cupy.int64)
            squares = cupy.zeros(rows, dtype=cupy.int64)
            for group in range(self.layout.group_count):
                low, high = self.layout.profile_range(group)
                sums += target[:, low:high].sum(axis=1, dtype=cupy.int64)
                squares += self._square_sum(target[:, low:high], axis=1)
            self._profile_sums[slot : slot + rows] = sums
            self._profile_squares[slot : slot + rows] = squares

    def check_codes(self) -> None:
        """Raise if any loaded code exceeded the dosage range (code 255 = missing)."""
        with self._device:
            if int(self._invalid.get(stream=self._compute)[0]):
                raise ValueError(f"a dosage code above {MAXIMUM_CODE} (missing) reached Stage 0")

    def _gemm(self, left_row: int, left_rows: int, right_row: int, right_rows: int, sample_low: int, samples: int,
              output: Any, output_offset: int, output_lead: int, accumulate: bool) -> None:
        """Column-major ``output[r + c * lead] (+)= sum_k s[left_row + r, k] s[right_row + c, k]``."""
        cublas = self._cupy.cuda.cublas
        handle = self._cupy.cuda.device.get_cublas_handle()
        cublas.setStream(handle, self._compute.ptr)
        width = self.layout.width
        base = self._buffer.data.ptr
        cublas.gemmEx(
            handle, cublas.CUBLAS_OP_T, cublas.CUBLAS_OP_N, left_rows, right_rows, samples,
            self._one.ctypes.data, base + left_row * width + sample_low, _CUDA_R_8I, width,
            base + right_row * width + sample_low, _CUDA_R_8I, width,
            (self._one if accumulate else self._zero).ctypes.data, output.data.ptr + 4 * output_offset,
            _CUDA_R_32I, output_lead, _CUBLAS_COMPUTE_32I, _CUBLAS_GEMM_DEFAULT,
        )

    def pair_weights(
        self, window_slot: int, tile_slot: int, tile_rows: int, maximum_distance: int
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Row and column sums of the fixed-point pair weights of window ``[window_slot, tile end)``
        against tile ``[tile_slot, tile_slot + tile_rows)``."""
        cupy = self._cupy
        window_rows = tile_slot + tile_rows - window_slot
        padded_window, padded_tile = _aligned_rows(window_rows), _aligned_rows(tile_rows)
        with self.context():
            band = cupy.empty((padded_tile, padded_window), dtype=cupy.int32)
            for group in range(self.layout.group_count):
                low, high = self.layout.profile_range(group)
                self._gemm(window_slot, padded_window, tile_slot, padded_tile, low, high - low, band, 0,
                           padded_window, accumulate=group > 0)
            weights = cupy.empty((tile_rows, window_rows), dtype=cupy.int64)
            threads = 256
            self._pair_weights(
                ((window_rows + threads - 1) // threads, tile_rows),
                (threads,),
                (band, np.int32(padded_window), np.int32(window_rows), np.int32(tile_rows),
                 self._profile_sums[window_slot:], self._profile_squares[window_slot:],
                 self._profile_sums[tile_slot:], self._profile_squares[tile_slot:],
                 np.int64(self.layout.profile_count), np.float64(1.0 / (self.layout.profile_count - 1)),
                 np.int32(tile_slot - window_slot), np.int32(maximum_distance), weights),
            )
            return weights.sum(axis=0).get(stream=self._compute), weights.sum(axis=1).get(stream=self._compute)

    def block_statistics(self, slot: int, rows: int) -> tuple[Any, Any, Any]:
        """Exact per-group sums ``(G, rows)``, Grams ``(G, rows, rows)`` and cross-products, on the device."""
        cupy = self._cupy
        padded = _aligned_rows(rows)
        long_group = int(self.layout.group_widths.max()) > _CUDA_SAMPLE_CHUNK
        with self.context():
            block = self._buffer[slot : slot + rows]
            sums = cupy.stack([block[:, low:high].sum(axis=1, dtype=cupy.int64)
                               for low, high in map(self.layout.group_range, range(self.layout.group_count))])
            grams = cupy.empty((self.layout.group_count, rows, rows), dtype=cupy.int64 if long_group else cupy.int32)
            output = cupy.empty((padded, padded), dtype=cupy.int32)
            for group in range(self.layout.group_count):
                low, high = self.layout.group_range(group)
                total = None
                for chunk_low in range(low, high, _CUDA_SAMPLE_CHUNK):
                    chunk_samples = min(_CUDA_SAMPLE_CHUNK, high - chunk_low)
                    for panel_row in range(0, padded, _CUDA_GRAM_PANEL):
                        self._gemm(slot + panel_row, min(_CUDA_GRAM_PANEL, padded - panel_row), slot + panel_row,
                                   padded - panel_row, chunk_low, chunk_samples, output,
                                   panel_row * padded + panel_row, padded, accumulate=False)
                    if long_group:
                        total = output.astype(cupy.int64) if total is None else total + output
                lower = (output if total is None else total)[:rows, :rows]
                index = cupy.arange(rows)
                grams[group] = cupy.where(index[:, None] >= index[None, :], lower, lower.T)
            cross = None if self._cross is None else self.cross_products(slot, rows)
        return sums, grams, cross

    def cross_block_grams(self, first_slot: int, first_rows: int, second_slot: int, second_rows: int) -> Any:
        """Exact per-group Grams between two blocks' rows ``(G, first_rows, second_rows)``, on the device."""
        cupy = self._cupy
        first_padded, second_padded = _aligned_rows(first_rows), _aligned_rows(second_rows)
        long_group = int(self.layout.group_widths.max()) > _CUDA_SAMPLE_CHUNK
        with self.context():
            grams = cupy.empty((self.layout.group_count, first_rows, second_rows), dtype=cupy.int64 if long_group else cupy.int32)
            # column-major [first_padded, second_padded]: entry (r, c) at c * first_padded + r
            output = cupy.empty((second_padded, first_padded), dtype=cupy.int32)
            for group in range(self.layout.group_count):
                low, high = self.layout.group_range(group)
                total = None
                for chunk_low in range(low, high, _CUDA_SAMPLE_CHUNK):
                    self._gemm(first_slot, first_padded, second_slot, second_padded, chunk_low, min(_CUDA_SAMPLE_CHUNK, high - chunk_low),
                               output, 0, first_padded, accumulate=False)
                    if long_group:
                        total = output.astype(cupy.int64) if total is None else total + output
                grams[group] = (output if total is None else total)[:second_rows, :first_rows].T
        return grams

    def cross_products(self, slot: int, rows: int) -> Any:
        """Per-group ``sum_i s_i y_i^T`` ``(G, rows, columns)`` in fp64 cuBLAS GEMMs, on the device."""
        cupy = self._cupy
        if self._cross is None:
            raise ValueError("the buffer was built without cross-product columns")
        with self.context():
            block = self._buffer[slot : slot + rows]
            result = cupy.zeros((self.layout.group_count, rows, self._cross.shape[1]), dtype=cupy.float64)
            for group in range(self.layout.group_count):
                low, high = self.layout.group_range(group)
                for chunk_low in range(low, high, _CROSS_SAMPLE_CHUNK):
                    chunk_high = min(chunk_low + _CROSS_SAMPLE_CHUNK, high)
                    result[group] += block[:, chunk_low:chunk_high].astype(cupy.float64) @ self._cross[chunk_low:chunk_high]
        return result

    def move_rows(self, source_slot: int, target_slot: int, rows: int) -> None:
        """Copy rows to a lower slot, front to back in pieces that never overlap."""
        step = source_slot - target_slot
        if step == 0:
            return
        with self.context():
            for offset in range(0, rows, step):
                count = min(step, rows - offset)
                for array in (self._buffer, self._profile_sums, self._profile_squares):
                    array[target_slot + offset : target_slot + offset + count] = array[
                        source_slot + offset : source_slot + offset + count
                    ]

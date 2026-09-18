"""Stage 0 on the CPU: exact integer genotype products through float32 BLAS.

The buffer holds ``s = code - 127`` as int8. Products are float32 GEMMs over
sample chunks of at most ``FLOAT32_EXACT_ROWS`` columns: every partial sum of such a
chunk is an integer of magnitude below 2^24, so the GEMM result is exact whatever
the BLAS blocking. Chunk results accumulate in fp64, which stays exact below 2^53,
and are returned as int64. Work is split into independent output panels on a
thread pool whose workers run single-threaded BLAS, so no two workers write the
same memory and nothing is summed in a thread-dependent order.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np
import scipy.linalg.blas as scipy_blas
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits

from sv_pgs.stage0.layout import SampleLayout
from sv_pgs.stage0.partition import fixed_point_pair_weights
from sv_pgs.stage0.statistics import FLOAT32_EXACT_ROWS, MAXIMUM_STORED_CODE, SIGNED_CODE_OFFSET

_SAMPLE_CHUNK = FLOAT32_EXACT_ROWS // 16 * 16
_OUTPUT_PANEL = 512


class CpuStage0Backend:
    """Genotype buffer and exact products on the host (see the module docstring)."""

    def __init__(
        self,
        layout: SampleLayout,
        capacity_rows: int,
        cross_product_columns: NDArray[np.float64] | None,
        worker_count: int,
    ) -> None:
        self.layout = layout
        self._buffer = np.zeros((capacity_rows, layout.width), dtype=np.int8)
        self._gather_index = np.where(layout.source_columns >= 0, layout.source_columns, 0)
        self._pad_columns = np.flatnonzero(layout.source_columns < 0)
        self._profile_columns = np.concatenate(
            [np.arange(*layout.profile_range(group)) for group in range(layout.group_count)]
        )
        self._profile = np.zeros((capacity_rows, self._profile_columns.shape[0]), dtype=np.float32)
        self._profile_sums = np.zeros(capacity_rows, dtype=np.int64)
        self._profile_squares = np.zeros(capacity_rows, dtype=np.int64)
        self._cross = None
        if cross_product_columns is not None:
            columns = np.asarray(cross_product_columns, dtype=np.float64)
            if columns.ndim != 2 or columns.shape[0] != layout.store_width:
                raise ValueError("cross_product_columns must be (store samples, columns)")
            self._cross = np.where(layout.source_columns[:, None] >= 0, columns[self._gather_index], 0.0)
        self._staged: dict[int, NDArray[np.uint8]] = {}
        self._pool = ThreadPoolExecutor(max_workers=worker_count)
        self._worker_count = worker_count

    def close(self) -> None:
        self._pool.shutdown()

    def _map(self, function: Callable[[tuple[int, int]], None], total: int, piece: int) -> None:
        ranges = [(start, min(start + piece, total)) for start in range(0, total, piece)]
        with threadpool_limits(limits=1, user_api="blas"):
            for _ in self._pool.map(function, ranges):
                pass

    def host_tile(self, tile_rows: int) -> NDArray[np.uint8]:
        """A host buffer for one tile."""
        return np.empty((tile_rows, self.layout.store_width), dtype=np.uint8)

    def stage_tile(self, codes: NDArray[np.uint8], staging_index: int) -> None:
        """Hold ``codes`` (``[rows, store samples]`` uint8) until it is loaded."""
        if codes.dtype != np.uint8 or codes.ndim != 2 or codes.shape[1] != self.layout.store_width:
            raise ValueError("a tile must be uint8 [rows, store samples]")
        self._staged[staging_index] = codes

    def release_staged_tile(self, staging_index: int) -> None:
        """The host tile is free once loaded: loading copies it."""

    def check_codes(self) -> None:
        """Codes are validated while loading."""

    def load_staged_tile(self, staging_index: int, rows: int, slot: int) -> None:
        """Shift, validate and lay out the staged rows at ``slot``."""
        codes = self._staged.pop(staging_index)[:rows]
        invalid = np.zeros(1, dtype=np.int64)

        def load(rows: tuple[int, int]) -> None:
            start, stop = rows
            gathered = np.take(codes[start:stop], self._gather_index, axis=1)
            if gathered.shape[0] and int(gathered.max()) > MAXIMUM_STORED_CODE:
                invalid[0] = 1
            target = self._buffer[slot + start : slot + stop]
            np.subtract(gathered.view(np.int8), np.int8(SIGNED_CODE_OFFSET), out=target)
            target[:, self._pad_columns] = 0
            profile = self._profile[slot + start : slot + stop]
            profile[...] = np.take(target, self._profile_columns, axis=1)
            self._profile_sums[slot + start : slot + stop] = profile.sum(axis=1, dtype=np.float64)
            self._profile_squares[slot + start : slot + stop] = np.square(profile).sum(axis=1, dtype=np.float64)

        self._map(load, codes.shape[0], max(1, -(-codes.shape[0] // self._worker_count)))
        if invalid[0]:
            raise ValueError(f"a dosage code above {MAXIMUM_STORED_CODE} (missing) reached Stage 0")

    def pair_weights(
        self, window_slot: int, tile_slot: int, tile_rows: int, maximum_distance: int
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Row and column sums of the fixed-point pair weights between the window
        ``[window_slot, tile_slot + tile_rows)`` and the tile ``[tile_slot, tile_slot + tile_rows)``."""
        window_stop = tile_slot + tile_rows
        window = self._profile[window_slot:window_stop]
        tile = self._profile[tile_slot:window_stop]
        band = _exact_products(window, tile, self._map)
        column_offset = np.arange(tile_slot - window_slot, window_stop - window_slot, dtype=np.int64)
        row_weights = np.zeros(window.shape[0], dtype=np.int64)
        column_partials = np.zeros((-(-window.shape[0] // _OUTPUT_PANEL), tile_rows), dtype=np.int64)

        def weigh(rows: tuple[int, int]) -> None:
            start, stop = rows
            weights = fixed_point_pair_weights(
                band[start:stop],
                self._profile_sums[window_slot + start : window_slot + stop],
                self._profile_squares[window_slot + start : window_slot + stop],
                self._profile_sums[tile_slot:window_stop],
                self._profile_squares[tile_slot:window_stop],
                self.layout.profile_count,
                column_offset - start,
                maximum_distance,
            )
            row_weights[start:stop] = weights.sum(axis=1)
            column_partials[start // _OUTPUT_PANEL] = weights.sum(axis=0)

        self._map(weigh, window.shape[0], _OUTPUT_PANEL)
        return row_weights, column_partials.sum(axis=0)

    def block_statistics(
        self, slot: int, rows: int
    ) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64] | None]:
        """Exact per-group sums ``(G, rows)``, Grams ``(G, rows, rows)`` and cross-products."""
        groups = self.layout.group_count
        sums = np.zeros((groups, rows), dtype=np.int64)
        grams = np.zeros((groups, rows, rows), dtype=np.int64)
        cross = None if self._cross is None else np.zeros((groups, rows, self._cross.shape[1]), dtype=np.float64)
        block = self._buffer[slot : slot + rows]
        for group in range(groups):
            low, high = self.layout.group_range(group)
            values = np.empty((rows, high - low), dtype=np.float32)

            def convert(row_range: tuple[int, int]) -> None:
                start, stop = row_range
                values[start:stop] = block[start:stop, low:high]
                sums[group, start:stop] = values[start:stop].sum(axis=1, dtype=np.float64)

            self._map(convert, rows, max(1, -(-rows // self._worker_count)))
            grams[group] = _exact_products(values, values, self._map, symmetric=True)
            if cross is not None:
                cross[group] = _cross_products(values, self._cross[low:high], self._map)
        return sums, grams, cross

    def move_rows(self, source_slot: int, target_slot: int, rows: int) -> None:
        """Copy rows to a lower slot (numpy resolves the overlap)."""
        for array in (self._buffer, self._profile, self._profile_sums, self._profile_squares):
            array[target_slot : target_slot + rows] = array[source_slot : source_slot + rows]


def _exact_products(
    left: NDArray[np.float32],
    right: NDArray[np.float32],
    parallel_map: Callable[[Callable[[tuple[int, int]], None], int, int], None],
    symmetric: bool = False,
) -> NDArray[np.int64]:
    """``left @ right.T`` for integer-valued float32 rows with ``|value| <= 127``, exactly.

    ``symmetric`` (``left is right``) computes the panels on and above the diagonal and
    mirrors them.
    """
    rows, columns = left.shape[0], right.shape[0]
    samples = left.shape[1]
    result = np.zeros((rows, columns), dtype=np.int64)
    row_panels = range(0, rows, _OUTPUT_PANEL)
    column_panels = range(0, columns, _OUTPUT_PANEL)
    panels = [
        (row_start, column_start)
        for row_start in row_panels
        for column_start in column_panels
        if not symmetric or column_start >= row_start
    ]

    def panel(index_range: tuple[int, int]) -> None:
        row_start, column_start = panels[index_range[0]]
        row_stop = min(row_start + _OUTPUT_PANEL, rows)
        column_stop = min(column_start + _OUTPUT_PANEL, columns)
        total = np.zeros((row_stop - row_start, column_stop - column_start), dtype=np.float64)
        for sample_start in range(0, samples, _SAMPLE_CHUNK):
            sample_stop = min(sample_start + _SAMPLE_CHUNK, samples)
            total += scipy_blas.sgemm(
                1.0,
                left[row_start:row_stop, sample_start:sample_stop].T,
                right[column_start:column_stop, sample_start:sample_stop].T,
                trans_a=True,
            )
        result[row_start:row_stop, column_start:column_stop] = total

    parallel_map(panel, len(panels), 1)
    if symmetric:
        upper = np.triu(result, 1)
        result = np.triu(result) + upper.T
    return result


def _cross_products(
    values: NDArray[np.float32],
    columns: NDArray[np.float64],
    parallel_map: Callable[[Callable[[tuple[int, int]], None], int, int], None],
) -> NDArray[np.float64]:
    """``values @ columns`` in fp64, split over row panels."""
    result = np.zeros((values.shape[0], columns.shape[1]), dtype=np.float64)

    def panel(row_range: tuple[int, int]) -> None:
        start, stop = row_range
        result[start:stop] = values[start:stop].astype(np.float64) @ columns

    parallel_map(panel, values.shape[0], _OUTPUT_PANEL)
    return result

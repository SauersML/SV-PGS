"""Stage 2's genotype blocks streamed from the dosage store: ``exact_polish.GenotypeBlockSource``.

One Stage 2 read visits every LD block once. Three stages of it overlap:

1. the host: ``DosageStore.iter_codes`` reads and decodes the next blocks' row spans ahead, on
   its reader threads, into a ring of (pinned, on CUDA) host buffers;
2. the copy engine: the next block's span goes host -> device on its own stream while the
   current block computes;
3. the device: one kernel picks the block's rows out of its span and writes them as signed codes
   ``s = code - 127`` into an aligned buffer, which a ``CodeBlockTile`` wraps without copying.

Each block's rows are ascending store rows (its reduced columns' representatives), read as the
one contiguous span from its first to its last row.
"""

from __future__ import annotations

from typing import Any, Iterator, Sequence

import numpy as np
from numpy.typing import NDArray

from sv_pgs.code_products import INT8_GEMM_ALIGNMENT, CodeBlockTile
from sv_pgs.compute_budget import ComputeBudget, _try_import_cupy
from sv_pgs.dosage_store import DosageStore
from sv_pgs.genotype_buffers import SIGNED_CODE_OFFSET
from sv_pgs.genotype_statistics import GenotypeSufficientStatistics

_GATHER_SOURCE = r"""
extern "C" __global__
void gather_signed_codes(const unsigned char* __restrict__ span, const long long* __restrict__ rows_in_span,
                         signed char* __restrict__ target, long long rows, long long samples,
                         long long padded_rows, long long padded_samples) {
    // target[r, c] = span[rows_in_span[r], c] - SIGNED_CODE_OFFSET inside [rows, samples], 0 in the padding
    long long total = padded_rows * padded_samples;
    for (long long at = (long long)blockIdx.x * blockDim.x + threadIdx.x; at < total; at += (long long)gridDim.x * blockDim.x) {
        long long row = at / padded_samples;
        long long column = at - row * padded_samples;
        target[at] = (row < rows && column < samples)
            ? (signed char)((int)span[rows_in_span[row] * samples + column] - SIGNED_CODE_OFFSET) : (signed char)0;
    }
}
"""


def _aligned(count: int) -> int:
    return -(-count // INT8_GEMM_ALIGNMENT) * INT8_GEMM_ALIGNMENT


class StoreGenotypeBlockSource:
    """A dosage store's LD blocks as Stage 2 tiles, streamed once per read.

    ``block_rows[b]`` holds block b's store rows, ascending; ``block_variant_indices[b]`` its
    columns in the model's variant order (what ``exact_polish`` hands its local step); ``means``
    and ``scales`` are per block row, in signed-code units, concatenated over blocks.
    ``workspace_bytes`` is each tile's product workspace from the caller's memory plan, and
    ``resident_bytes`` is what the source itself holds on the compute device.
    """

    def __init__(
        self,
        store: DosageStore,
        block_rows: Sequence[NDArray[np.int64]],
        block_variant_indices: Sequence[NDArray[np.int64]],
        means: NDArray[np.float64],
        scales: NDArray[np.float64],
        budget: ComputeBudget,
        workspace_bytes: int,
    ) -> None:
        self._store = store
        self._block_rows = [np.asarray(rows, dtype=np.int64) for rows in block_rows]
        self._block_variant_indices = [np.asarray(indices, dtype=np.int64) for indices in block_variant_indices]
        if len(self._block_rows) != len(self._block_variant_indices):
            raise ValueError("block_rows and block_variant_indices need one entry per block")
        for rows, indices in zip(self._block_rows, self._block_variant_indices):
            if rows.ndim != 1 or rows.shape[0] == 0 or rows.shape != indices.shape or np.any(np.diff(rows) <= 0):
                raise ValueError("every block needs ascending, distinct store rows, one per variant index")
        self._offsets = np.concatenate([[0], np.cumsum([rows.shape[0] for rows in self._block_rows])]).astype(np.int64)
        if np.shape(means) != (int(self._offsets[-1]),) or np.shape(scales) != (int(self._offsets[-1]),):
            raise ValueError("means and scales need one entry per block row")
        self._spans = [(int(rows[0]), int(rows[-1]) + 1) for rows in self._block_rows]
        self._budget = budget
        self._workspace_bytes = int(workspace_bytes)
        self._samples = int(store.n_samples)
        self._padded_samples = _aligned(self._samples)
        widest_span = max(stop - start for start, stop in self._spans)
        widest_block = _aligned(max(rows.shape[0] for rows in self._block_rows))
        self._cupy = _try_import_cupy() if budget.device_kind == "cuda" else None
        xp = self._cupy if self._cupy is not None else np
        self._means = xp.asarray(means, dtype=xp.float64)
        self._scales = xp.asarray(scales, dtype=xp.float64)
        # Two slots, so that the next block's span and codes fill while the current block computes.
        self._signed = [xp.empty((widest_block, self._padded_samples), dtype=xp.int8) for _ in range(2)]
        self.resident_bytes = sum(int(slot.nbytes) for slot in self._signed) + int(self._means.nbytes) + int(self._scales.nbytes)
        if self._cupy is not None:
            cupy = self._cupy
            self._spans_on_device = [cupy.empty((widest_span, self._samples), dtype=cupy.uint8) for _ in range(2)]
            self._rows_in_span = [cupy.asarray(rows - rows[0]) for rows in self._block_rows]
            self.resident_bytes += sum(int(slot.nbytes) for slot in self._spans_on_device) + sum(int(rows.nbytes) for rows in self._rows_in_span)
            self._gather = cupy.RawKernel(_GATHER_SOURCE.replace("SIGNED_CODE_OFFSET", str(SIGNED_CODE_OFFSET)), "gather_signed_codes")
            self._copy_stream = cupy.cuda.Stream(non_blocking=True)

    @classmethod
    def from_statistics(
        cls, store: DosageStore, statistics: GenotypeSufficientStatistics, budget: ComputeBudget, workspace_bytes: int
    ) -> StoreGenotypeBlockSource:
        """The reduced model's LD blocks of one Stage 0 pass: each reduced column is its tie
        group's representative, whose store row, mean and scale Stage 0 recorded."""
        kept = np.asarray(statistics.tie_map.kept_indices, dtype=np.int64)
        boundaries = statistics.ld.block_boundaries
        block_rows, block_indices = [], []
        for block_index in range(statistics.ld.block_count):
            reduced = np.arange(int(boundaries[block_index]), int(boundaries[block_index + 1]), dtype=np.int64)
            block_rows.append(statistics.active_rows[kept[reduced]])
            block_indices.append(reduced)
        order = np.concatenate([kept[indices] for indices in block_indices])
        return cls(store, block_rows, block_indices, statistics.means[order], statistics.scales[order], budget, workspace_bytes)

    @property
    def sample_count(self) -> int:
        return self._samples

    @property
    def array_module(self) -> Any:
        return self._cupy if self._cupy is not None else np

    @property
    def block_variant_indices(self) -> Sequence[NDArray[np.int64]]:
        return self._block_variant_indices

    def _tile(self, block_index: int, slot: int) -> CodeBlockTile:
        rows = int(self._block_rows[block_index].shape[0])
        offset = slice(int(self._offsets[block_index]), int(self._offsets[block_index + 1]))
        return CodeBlockTile.from_aligned(
            self._signed[slot][: _aligned(rows)], rows, self._samples, self._means[offset], self._scales[offset],
            self.array_module, self._workspace_bytes,
        )

    def iter_tiles(self) -> Iterator[tuple[int, CodeBlockTile]]:
        """Yield (block_index, tile) in block order; a tile is valid until the next is requested."""
        spans = self._store.iter_codes(self._spans, None, self._budget)
        if self._cupy is None:
            for block_index, (start, _stop, codes) in enumerate(spans):
                rows = self._block_rows[block_index]
                target = self._signed[block_index % 2][: rows.shape[0], : self._samples]
                gathered = codes[rows - start]
                np.subtract(gathered.view(np.int8), np.int8(SIGNED_CODE_OFFSET), out=target)
                yield block_index, self._tile(block_index, block_index % 2)
            return
        yield from self._iter_device_tiles(spans)

    def _iter_device_tiles(self, spans: Iterator[tuple[int, int, NDArray[np.uint8]]]) -> Iterator[tuple[int, CodeBlockTile]]:
        cupy = self._cupy
        compute = cupy.cuda.get_current_stream()
        copied = [cupy.cuda.Event() for _ in range(2)]
        computed = [cupy.cuda.Event() for _ in range(2)]
        for event in computed:
            event.record(compute)

        def upload(position: int, host_codes: NDArray[np.uint8]) -> None:
            slot = position % 2
            # The span buffer is free once the block that last used it has been computed.
            self._copy_stream.wait_event(computed[slot])
            self._spans_on_device[slot][: host_codes.shape[0]].set(host_codes, stream=self._copy_stream)
            copied[slot].record(self._copy_stream)

        count = len(self._spans)
        _start, _stop, first = next(spans)
        upload(0, first)
        attributes = cupy.cuda.Device().attributes
        threads = int(attributes["MaxThreadsPerBlock"])
        # a grid-stride loop needs no more blocks than the device keeps resident at once
        resident_blocks = int(attributes["MultiProcessorCount"]) * (int(attributes["MaxThreadsPerMultiProcessor"]) // threads)
        for block_index in range(count):
            slot = block_index % 2
            # iter_codes reuses the host buffer of this span once the next span is requested.
            copied[slot].synchronize()
            if block_index + 1 < count:
                _start, _stop, following = next(spans)
                upload(block_index + 1, following)
            rows = int(self._block_rows[block_index].shape[0])
            compute.wait_event(copied[slot])
            padded_rows = _aligned(rows)
            elements = padded_rows * self._padded_samples
            self._gather(
                (min(-(-elements // threads), resident_blocks),), (threads,),
                (
                    self._spans_on_device[slot], self._rows_in_span[block_index], self._signed[slot],
                    np.int64(rows), np.int64(self._samples), np.int64(padded_rows), np.int64(self._padded_samples),
                ),
            )
            yield block_index, self._tile(block_index, slot)
            computed[slot].record(compute)

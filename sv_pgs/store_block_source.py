"""Stage 2's genotype blocks streamed from the dosage store (the dual solver's `GenotypeBlockSource`).

One Stage 2 read visits every LD block once. Three stages of it overlap:

1. the host: ``DosageStore.iter_codes`` reads and decodes the next blocks' row spans ahead, on
   its reader threads, into a ring of (pinned, on CUDA) host buffers;
2. the copy engine: the next block's span goes host -> device on its own stream while the
   current block computes. A rowdict store skips the host decode: only the chunks holding the
   block's rows go to the device, and ``DosageStore.read_codes_to_device`` checks and decodes
   just those rows there, on the copy stream, into a compact [block rows, n] buffer; the host
   only reads bytes;
3. the device: one kernel picks the block's rows out of its span and writes them as signed codes
   ``s = code - 127`` into an aligned buffer, which a ``CodeBlockTile`` wraps without copying.

Each block's rows are ascending store rows (its reduced columns' representatives), read as the
one contiguous span from its first to its last row.
"""

from __future__ import annotations

import time
import weakref
from typing import Any, Iterator, Sequence

import numpy as np
from numpy.typing import NDArray

from sv_pgs import rowdict_codec
from sv_pgs.code_products import INT8_GEMM_ALIGNMENT, CodeBlockTile
from sv_pgs.compute_budget import ComputeBudget, _try_import_cupy
from sv_pgs.dosage_store import DosageStore
from sv_pgs.genotype_buffers import SIGNED_CODE_OFFSET
from sv_pgs.genotype_statistics import GenotypeSufficientStatistics
from sv_pgs.memory_broker import HOST, MemoryBroker, current_broker, device_pool, meter_device
from sv_pgs.phase_timer import GB, timing
from sv_pgs.progress import log

_GATHER_SOURCE = r"""
extern "C" __global__
void gather_signed_codes(const unsigned char* __restrict__ span, const long long* __restrict__ rows_in_span,
                         const long long* __restrict__ sample_columns,
                         signed char* __restrict__ target, long long rows, long long samples, long long source_samples,
                         long long padded_rows, long long padded_samples) {
    // Gather selected rows and samples, subtract SIGNED_CODE_OFFSET, and zero padding.
    long long total = padded_rows * padded_samples;
    for (long long at = (long long)blockIdx.x * blockDim.x + threadIdx.x; at < total; at += (long long)gridDim.x * blockDim.x) {
        long long row = at / padded_samples;
        long long column = at - row * padded_samples;
        target[at] = (row < rows && column < samples)
            ? (signed char)((int)span[rows_in_span[row] * source_samples + sample_columns[column]] - SIGNED_CODE_OFFSET) : (signed char)0;
    }
}
"""


def _aligned(count: int) -> int:
    return -(-count // INT8_GEMM_ALIGNMENT) * INT8_GEMM_ALIGNMENT


def consecutive_runs(rows: NDArray[np.int64]) -> list[tuple[int, int]]:
    """The maximal runs of consecutive store rows in the ascending, distinct ``rows``, as [start, stop) ranges."""
    breaks = np.flatnonzero(np.diff(rows) != 1) + 1
    starts = rows[np.concatenate([[0], breaks])]
    stops = rows[np.concatenate([breaks, [rows.shape[0]]]) - 1] + 1
    return list(zip(starts.tolist(), stops.tolist()))


def reduced_block_layout(
    statistics: GenotypeSufficientStatistics,
) -> tuple[list[NDArray[np.int64]], list[NDArray[np.int64]], list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    """Per LD block of one Stage 0 pass: its store rows, its reduced columns, and their means and
    scales. Each reduced column is its tie group's representative, whose store row, mean and scale
    Stage 0 recorded."""
    kept = np.asarray(statistics.tie_map.kept_indices, dtype=np.int64)
    boundaries = statistics.ld.block_boundaries
    block_rows, block_indices, means, scales = [], [], [], []
    for block_index in range(statistics.ld.block_count):
        reduced = np.arange(int(boundaries[block_index]), int(boundaries[block_index + 1]), dtype=np.int64)
        representatives = kept[reduced]
        block_rows.append(statistics.active_rows[representatives])
        block_indices.append(reduced)
        means.append(statistics.means[representatives])
        scales.append(statistics.scales[representatives])
    return block_rows, block_indices, means, scales


class StoreGenotypeBlockSource:
    """A dosage store's LD blocks as Stage 2 tiles, streamed once per read.

    ``block_rows[b]`` holds block b's store rows, ascending; ``block_variant_indices[b]`` its
    columns in the model's variant order (what the dual solver hands its local step); ``means``
    and ``scales`` are per block row, in signed-code units, concatenated over blocks.
    ``workspace_bytes`` is each tile's product workspace from the caller's memory plan, and
    ``resident_bytes`` is what the source itself holds on the compute device.
    ``sample_indices`` optionally restricts all products to those store samples;
    means and scales still come from the caller's training statistics. GPU rowdict
    staging decodes full rows before gathering selected samples into compact tiles.
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
        sample_indices: NDArray[np.int64] | None = None,
    ) -> None:
        self._store = store
        self._sample_indices = None
        if sample_indices is not None:
            selected = np.asarray(sample_indices)
            if (selected.ndim != 1 or not selected.size or selected.dtype.kind not in "iu"
                    or np.any(selected >= store.n_samples) or np.any(selected < 0)
                    or np.any(selected[1:] <= selected[:-1])):
                raise ValueError("sample_indices must be ascending, distinct, nonempty store sample indices")
            if selected.size != store.n_samples:
                self._sample_indices = selected.astype(np.int64, copy=True)
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
        self._samples = int(store.n_samples if self._sample_indices is None else self._sample_indices.size)
        self._padded_samples = _aligned(self._samples)
        widest_block = _aligned(max(rows.shape[0] for rows in self._block_rows))
        self._cupy = _try_import_cupy() if budget.device_kind == "cuda" else None
        xp = self._cupy if self._cupy is not None else np
        host_scales = np.asarray(scales, dtype=np.float64)
        self._scale_spreads = [
            float(host_scales[start:stop].max() / host_scales[start:stop].min()) for start, stop in zip(self._offsets[:-1], self._offsets[1:])
        ]
        self._means = xp.asarray(means, dtype=xp.float64)
        self._scales = xp.asarray(host_scales)
        # Two slots, so that the next block's span and codes fill while the current block computes. On the host they are
        # this source's own mandatory buffers, charged to the shared ledger for its lifetime before they are allocated
        # (on a device, the ledger's allocator meters them).
        broker = current_broker()
        if self._cupy is None and broker is not None:
            lease = broker.reserve(HOST, 2 * widest_block * self._padded_samples, "the block source's code slots")
            weakref.finalize(self, lease.release)
        self._signed = [xp.empty((widest_block, self._padded_samples), dtype=xp.int8) for _ in range(2)]
        self.resident_bytes = sum(int(slot.nbytes) for slot in self._signed) + int(self._means.nbytes) + int(self._scales.nbytes)
        if self._cupy is not None:
            cupy = self._cupy
            self._decoder = rowdict_codec.GpuRowDecoder(cupy) if store.codecs == frozenset({"rowdict"}) else None
            # Rowdict decoding stays on the GPU; gather just the selected samples
            # into the smaller signed tile. Other codecs select before uploading.
            self._input_samples = int(store.n_samples) if self._decoder is not None else self._samples
            self._sample_columns = (cupy.asarray(self._sample_indices) if self._decoder is not None and self._sample_indices is not None
                                    else cupy.arange(self._samples, dtype=cupy.int64))
            # A block's rows are staged compactly, so the gather is the identity: a rowdict read lands them so, and other
            # codecs read only them (``_iter_device_tiles``). Staged whole, a span is every record between the block's
            # first and last: chr22's HapMap3 block of 14,542 sites spanned 861k records, 37 GB of an A100's 42.
            self._spans_on_device = [cupy.empty((widest_block, self._input_samples), dtype=cupy.uint8) for _ in range(2)]
            self._rows_in_span = [cupy.arange(rows.shape[0], dtype=cupy.int64) for rows in self._block_rows]
            self.resident_bytes += sum(int(slot.nbytes) for slot in self._spans_on_device) + sum(int(rows.nbytes) for rows in self._rows_in_span) + int(self._sample_columns.nbytes)
            self._gather = cupy.RawKernel(_GATHER_SOURCE.replace("SIGNED_CODE_OFFSET", str(SIGNED_CODE_OFFSET)), "gather_signed_codes")
            self._copy_stream = cupy.cuda.Stream(non_blocking=True)
        # Every block's signed codes held on the device after the first read, as a cache of the shared ledger
        # (``memory_broker``): each later read (every mean-field sweep, every dual-solver pass) then builds its tiles from
        # device memory. Streamed, 43% of a 518k x 40k bench-sim Stage 2 was the store's reads [sim, scenario_000, py-spy
        # on the A40]. The cache is admitted only from what the device's pool has left once everything the fit holds is
        # counted, and never from a free-memory snapshot; a later allocation that needs its bytes evicts it
        # (``_drop_resident``), and the read in progress then streams the blocks it has not reached, so the eviction
        # frees every block but the one a tile still references. An evicted set is asked for again at the next read,
        # from what the ledger then has left: the peak that evicted it can be a step's own (the moment starts square
        # each Stage 0 block in float64 on the device, 10 GB at a 29k-column block), and a set never readmitted
        # streamed every later read from the store (an All of Us whole-genome fit [real]: three reads of 6-10 minutes
        # each after its moment starts, where the resident reads are seconds). A readmission costs nothing beyond the
        # read that refills it, which streams the store either way.
        self._resident: list[Any] | None = None
        self._resident_complete = False
        self._evictions = 0
        self._refused = False
        self.resident_codes_bytes = sum(_aligned(int(rows.shape[0])) * self._padded_samples for rows in self._block_rows)
        self._broker = None
        if self._cupy is not None:
            device_id = int(self._cupy.cuda.runtime.getDevice())
            self._pool = device_pool(device_id)
            self._broker = current_broker()
            if self._broker is None or self._pool not in self._broker.meters:
                # Outside a CUDA scope: this source's own ledger of its device, metered as a scope meters it (no allocator
                # is routed through it, so only the cache's admission reads it).
                self._broker = MemoryBroker.from_budget(budget)
                meter_device(self._broker, self._cupy, device_id)

    def _admit_resident(self) -> bool:
        """Admit the resident set as a device cache from what the ledger has left (the class docstring)."""
        if self._cupy is None or self._broker is None:
            return False
        lease = self._broker.admit(self._pool, self.resident_codes_bytes, "the resident genotype codes", self._drop_resident)
        if lease is not None:
            log(f"resident genotype codes: {self.resident_codes_bytes / 1e9:.1f} GB admitted on {self._pool}{' again' if self._evictions else ''}")
            self._refused = False
        elif not self._refused:
            # Logged once per refusal spell: every read asks again.
            log(f"resident genotype codes: {self.resident_codes_bytes / 1e9:.1f} GB not admitted on {self._pool}, "
                f"{self._broker.remaining(self._pool) / 1e9:.1f} GB left: reads stream the store")
            self._refused = True
        return lease is not None

    def _drop_resident(self) -> None:
        """The ledger evicts the resident set: its blocks are freed as their last tile releases them, and the next read
        asks for it again."""
        self._resident = None
        self._resident_complete = False
        self._evictions += 1
        log(f"resident genotype codes: evicted from {self._pool} (eviction {self._evictions}); the next read streams the store and asks again")

    def holds_codes(self) -> bool:
        """Whether this source's codes are held on its device, or would be at its next read (within what the ledger has
        left): its reads after the one that fills them are then device passes, not reads of the store.

        What is left is counted as the admission counts it, the pool's free blocks returned first (``remaining``), not
        as ``admits`` reads the pool as it stands: there, on the All of Us whole-genome fits, the device the route
        choice's first read had not reached yet answered no while CuPy's pool held freed blocks, admitted the set at
        its own first read a minute later, and half the fits took the band route."""
        if self._resident is not None:
            return True
        return self._cupy is not None and self._broker is not None and self._broker.remaining(self._pool) >= self.resident_codes_bytes

    def _keep(self, block_index: int, slot: int) -> None:
        """Copy block ``block_index``'s signed codes into the resident set (on the first read), or drop the set if the
        device cannot hold it after all."""
        resident = self._resident
        if resident is None:
            return
        rows = _aligned(int(self._block_rows[block_index].shape[0]))
        try:
            copy = self._signed[slot][:rows].copy()
        except self._cupy.cuda.memory.OutOfMemoryError:
            for lease in [lease for lease in self._broker.leases if lease.evict == self._drop_resident]:
                lease.release()
            self._drop_resident()
            return
        if self._resident is resident:
            resident[block_index] = copy

    def _resident_tile(self, block_index: int, codes: Any) -> CodeBlockTile:
        rows = int(self._block_rows[block_index].shape[0])
        offset = slice(int(self._offsets[block_index]), int(self._offsets[block_index + 1]))
        return CodeBlockTile.from_aligned(
            codes, rows, self._samples, self._means[offset], self._scales[offset],
            self._scale_spreads[block_index], self.array_module, self._workspace_bytes,
        )

    @classmethod
    def from_statistics(
        cls, store: DosageStore, statistics: GenotypeSufficientStatistics, budget: ComputeBudget, workspace_bytes: int,
        sample_indices: NDArray[np.int64] | None = None,
    ) -> StoreGenotypeBlockSource:
        """The reduced model's LD blocks of one Stage 0 pass: each reduced column is its tie
        group's representative, whose store row, mean and scale Stage 0 recorded."""
        block_rows, block_indices, means, scales = reduced_block_layout(statistics)
        return cls(store, block_rows, block_indices, np.concatenate(means), np.concatenate(scales), budget, workspace_bytes, sample_indices)

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
            self._scale_spreads[block_index], self.array_module, self._workspace_bytes,
        )

    def iter_tiles(self) -> Iterator[tuple[int, CodeBlockTile]]:
        """Yield (block_index, tile) in block order; a tile is valid until the next is requested."""
        if self._cupy is not None and self._resident_complete and self._resident is not None:
            for block_index in range(len(self._block_rows)):
                resident = self._resident
                if resident is None:
                    # Evicted mid-read (``_drop_resident``): the blocks not yet reached are streamed.
                    yield from self._iter_streamed(block_index)
                    return
                yield block_index, self._resident_tile(block_index, resident[block_index])
            return
        if self._cupy is not None and self._resident is None and not self._resident_complete and self._admit_resident():
            self._resident = [None] * len(self._block_rows)
        started = time.perf_counter()
        yield from self._iter_streamed(0)
        if self._cupy is not None:
            # A device read of the store: the one that fills the resident set, or one that streams (``_admit_resident``),
            # its wall time the reader's work on its tiles included.
            timing("resident_fill" if self._resident is not None else "store_read", time.perf_counter() - started, gb=self.resident_codes_bytes / GB, pool=self._pool)
        # only a read that reached every block leaves a complete resident set
        self._resident_complete = self._cupy is not None and self._resident is not None

    def _iter_streamed(self, first_block: int) -> Iterator[tuple[int, CodeBlockTile]]:
        """The blocks from ``first_block`` on, read from the store."""
        if self._cupy is not None:
            yield from (self._iter_decoded_tiles(first_block) if self._decoder is not None else self._iter_device_tiles(first_block))
            return
        spans = self._store.iter_codes(self._spans[first_block:], self._sample_indices, self._budget)
        for position, (start, _stop, codes) in enumerate(spans):
            block_index = first_block + position
            rows = self._block_rows[block_index]
            target = self._signed[block_index % 2][: rows.shape[0], : self._samples]
            gathered = codes[rows - start]
            np.subtract(gathered.view(np.int8), np.int8(SIGNED_CODE_OFFSET), out=target)
            yield block_index, self._tile(block_index, block_index % 2)

    def _iter_device_tiles(self, first_block: int = 0) -> Iterator[tuple[int, CodeBlockTile]]:
        cupy = self._cupy
        compute = cupy.cuda.get_current_stream()
        copied = [cupy.cuda.Event() for _ in range(2)]
        computed = [cupy.cuda.Event() for _ in range(2)]
        for event in computed:
            event.record(compute)
        # Only the blocks' own rows are read, each maximal run of consecutive store rows as one range: a block's span is
        # every record between its first and last, and read whole (chr22's HapMap3 block of 14,542 sites spanned 861k
        # records) its pinned read-ahead slots outgrew what the host could pin.
        runs = [consecutive_runs(rows) for rows in self._block_rows[first_block:]]
        pieces = self._store.iter_codes([run for block in runs for run in block], self._sample_indices, self._budget)
        # Each slot's block, assembled on the host from its runs; a slot is refilled only after its last copy has run.
        assembled = [np.empty(self._spans_on_device[slot].shape, dtype=np.uint8) for slot in range(2)]

        def upload(position: int) -> None:
            slot = position % 2
            filled = 0
            for _run in runs[position - first_block]:
                _start, _stop, codes = next(pieces)
                assembled[slot][filled : filled + codes.shape[0]] = codes
                filled += codes.shape[0]
            # The staging buffer is free once the block that last used it has been computed.
            self._copy_stream.wait_event(computed[slot])
            self._spans_on_device[slot][:filled].set(assembled[slot][:filled], stream=self._copy_stream)
            copied[slot].record(self._copy_stream)

        count = len(self._block_rows)
        upload(first_block)
        for block_index in range(first_block, count):
            slot = block_index % 2
            copied[slot].synchronize()
            if block_index + 1 < count:
                upload(block_index + 1)
            compute.wait_event(copied[slot])
            self._gather_block(block_index, slot)
            self._keep(block_index, slot)
            yield block_index, self._tile(block_index, slot)
            computed[slot].record(compute)

    def _gather_block(self, block_index: int, slot: int) -> None:
        """Queue, on the current stream, block ``block_index``'s rows of its span as signed codes."""
        cupy = self._cupy
        attributes = cupy.cuda.Device().attributes
        threads = int(attributes["MaxThreadsPerBlock"])
        # a grid-stride loop needs no more blocks than the device keeps resident at once
        resident_blocks = int(attributes["MultiProcessorCount"]) * (int(attributes["MaxThreadsPerMultiProcessor"]) // threads)
        rows = int(self._block_rows[block_index].shape[0])
        padded_rows = _aligned(rows)
        self._gather(
            (min(-(-padded_rows * self._padded_samples // threads), resident_blocks),), (threads,),
            (
                self._spans_on_device[slot], self._rows_in_span[block_index], self._sample_columns, self._signed[slot],
                np.int64(rows), np.int64(self._samples), np.int64(self._input_samples), np.int64(padded_rows), np.int64(self._padded_samples),
            ),
        )

    def _iter_decoded_tiles(self, first_block: int = 0) -> Iterator[tuple[int, CodeBlockTile]]:
        """The rowdict path: each block's rows decode on the device, on the copy stream.

        Block b + 1's span is fetched after block b is yielded, so its host work (read, crc32c,
        frame location, the host -> device copy) overlaps the products the caller queued for b.
        """
        cupy = self._cupy
        compute = cupy.cuda.get_current_stream()
        decoded = [cupy.cuda.Event() for _ in range(2)]
        gathered = [cupy.cuda.Event() for _ in range(2)]
        for event in gathered:
            event.record(compute)

        def fetch(position: int) -> None:
            slot = position % 2
            start, stop = self._spans[position]
            rows = self._block_rows[position]
            # the staging buffer is free once the gather of the block that last used it has run
            self._copy_stream.wait_event(gathered[slot])
            with self._copy_stream:
                self._store.read_codes_to_device(start, stop, self._spans_on_device[slot][: rows.shape[0]], self._decoder, rows=rows)
            decoded[slot].record(self._copy_stream)

        fetch(first_block)
        for block_index in range(first_block, len(self._spans)):
            slot = block_index % 2
            compute.wait_event(decoded[slot])
            self._gather_block(block_index, slot)
            self._keep(block_index, slot)
            gathered[slot].record(compute)
            yield block_index, self._tile(block_index, slot)
            if block_index + 1 < len(self._spans):
                fetch(block_index + 1)


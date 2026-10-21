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

Where Stage 0's pass kept its training codes (``training_codes.TrainingCodes``: every record Stage 2
reads, as the signed codes these slots hold), a read takes its blocks from them instead of the
store: on the host a copy into the slot, on a device a copy into a host slot laid out as the
device's (aligned rows, padded samples, the padding zero) and its upload on the copy stream. The
slots then hold the same bytes either way, so the tiles and every product are the same. On a
device a block's restart unit (``_iter_prepared``), where every block has one, comes first: a
restarted fit's Stage 0 is its unit's, which reads no tile, so it keeps no codes.
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
from sv_pgs.restart_store import CodeCache, definition_sources, facts, unit_key

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
    ``codes``, the training codes Stage 0's pass kept (``training_codes.TrainingCodes``), serve
    every read while they hold every block's rows on this source's samples (the module docstring).
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
        code_cache: CodeCache | None = None,
        codes: Any = None,
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
        # Stage 0's kept codes where they are this source's samples' (their columns in this source's order); each block's
        # rows' positions among them are found at the first read that asks (``_kept``).
        sample_columns = np.arange(store.n_samples, dtype=np.int64) if self._sample_indices is None else self._sample_indices
        self._codes = codes if codes is not None and codes.serves(sample_columns) else None
        self._code_positions: list[NDArray[np.int64]] | None = None
        # Where the read in progress takes its codes from (the timing line's ``codes``): the store, Stage 0's kept codes
        # or the blocks' restart units.
        self._origin = "store"
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
        # Each block's signed codes as a restart unit (``_iter_prepared``), on a device only: the host route's tiles are
        # the store's own reads.
        self._code_cache = code_cache if self._cupy is not None else None
        self._code_keys: list[tuple[str, dict[str, str]]] | None = None
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

    @property
    def resident_complete(self) -> bool:
        """Whether every block's codes are held on the device: every read from here on is a device pass."""
        return self._resident_complete and self._resident is not None

    def fill(self) -> bool:
        """The read that fills the resident codes, made now where the device would hold them (``holds_codes``) and does
        not yet hold them whole, so that the fit's first pass is a device pass and the fill can run on every device at
        once (``device_plan.ShardedDualSource.fill``); whether they are held whole after it. Nothing is read on the host,
        which holds no resident set, nor where the ledger would not admit one (the fit's reads then stream as they
        would have)."""
        if self._cupy is not None and not self.resident_complete and self.holds_codes():
            for _block in self.iter_tiles():
                pass
        return self.resident_complete

    def _kept(self) -> list[NDArray[np.int64]] | None:
        """Each block's rows' positions among the codes Stage 0's pass kept, where those still hold every block's rows;
        None where they do not (a read then streams the store). Codes that never held every row are dropped at once."""
        codes = self._codes
        if codes is None or codes.off is not None:
            return None
        if self._code_positions is None:
            positions = [codes.positions(rows) for rows in self._block_rows]
            if any(entry is None for entry in positions):
                self._codes = None
                log("training codes: Stage 0's pass did not keep every block's records; Stage 2 reads the store")
                return None
            self._code_positions = positions
        return self._code_positions

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
        sample_indices: NDArray[np.int64] | None = None, code_cache: CodeCache | None = None, codes: Any = None,
    ) -> StoreGenotypeBlockSource:
        """The reduced model's LD blocks of one Stage 0 pass: each reduced column is its tie
        group's representative, whose store row, mean and scale Stage 0 recorded."""
        block_rows, block_indices, means, scales = reduced_block_layout(statistics)
        return cls(
            store, block_rows, block_indices, np.concatenate(means), np.concatenate(scales), budget, workspace_bytes, sample_indices, code_cache, codes,
        )

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
            # A device read of the store (or of Stage 0's kept codes, or of the blocks' restart units: ``codes``): the one
            # that fills the resident set, or one that streams (``_admit_resident``), its wall time the reader's work on
            # its tiles included.
            timing(
                "resident_fill" if self._resident is not None else "store_read", time.perf_counter() - started, gb=self.resident_codes_bytes / GB,
                pool=self._pool, codes=self._origin,
            )
        # only a read that reached every block leaves a complete resident set
        self._resident_complete = self._cupy is not None and self._resident is not None

    def _iter_streamed(self, first_block: int) -> Iterator[tuple[int, CodeBlockTile]]:
        """The blocks from ``first_block`` on: on a device from their restart units where every one has one
        (``_iter_prepared``), else, as on the host, from the codes Stage 0's pass kept while they hold them, else from
        the store (``_iter_read``)."""
        if self._cupy is not None and self._code_cache is not None:
            yield from self._iter_prepared(first_block)
            return
        yield from self._iter_read(first_block)

    def _iter_read(self, first_block: int) -> Iterator[tuple[int, CodeBlockTile]]:
        """The blocks from ``first_block`` on from the codes Stage 0's pass kept while they hold them, else from the
        store."""
        positions = self._kept()
        self._origin = "store" if positions is None else "stage0"
        if positions is None:
            yield from self._iter_store(first_block)
        elif self._cupy is not None:
            yield from self._iter_kept_device_tiles(first_block, positions)
        else:
            yield from self._iter_kept_tiles(first_block, positions)

    def _iter_kept_tiles(self, first_block: int, positions: list[NDArray[np.int64]]) -> Iterator[tuple[int, CodeBlockTile]]:
        """The host's blocks from ``first_block`` on, each block's kept codes copied into the slot the store's codes fill
        (``_iter_store``): the same signed codes in the same places, so the same tile. Where the codes are evicted
        between two blocks, the blocks left are read from the store."""
        for block_index in range(first_block, len(self._block_rows)):
            slot = block_index % 2
            rows = int(self._block_rows[block_index].shape[0])
            if not self._codes.copy_rows(positions[block_index], self._signed[slot][:rows, : self._samples]):
                yield from self._iter_store(block_index)
                return
            yield block_index, self._tile(block_index, slot)

    def _iter_kept_device_tiles(self, first_block: int, positions: list[NDArray[np.int64]]) -> Iterator[tuple[int, CodeBlockTile]]:
        """The device's blocks from ``first_block`` on from the kept codes: each block's rows copied into a host slot laid
        out as the device's slot (its rows aligned and its samples padded, the padding zero, as the gather kernel leaves
        it) and uploaded on the copy stream while the block before computes, then kept and tiled as a store read's
        block is. The two host slots are a mandatory lease of the host ledger for the read's life. Where the codes are
        evicted between two blocks, the blocks left are read from the store."""
        cupy = self._cupy
        compute = cupy.cuda.get_current_stream()
        copied = [cupy.cuda.Event() for _ in range(2)]
        computed = [cupy.cuda.Event() for _ in range(2)]
        for event in computed:
            event.record(compute)
        shape = tuple(int(extent) for extent in self._signed[0].shape)
        broker = current_broker()
        lease = None if broker is None else broker.reserve(HOST, 2 * shape[0] * shape[1], "the kept codes' upload slots")
        try:
            staged = [np.zeros(shape, dtype=np.int8) for _ in range(2)]

            def upload(position: int) -> bool:
                slot = position % 2
                rows = int(self._block_rows[position].shape[0])
                padded_rows = _aligned(rows)
                if not self._codes.copy_rows(positions[position], staged[slot][:rows, : self._samples]):
                    return False
                # A wider block before may have left rows here; the padded samples are never written.
                staged[slot][rows:padded_rows] = 0
                # The device slot is free once the block that last used it has been computed.
                self._copy_stream.wait_event(computed[slot])
                self._signed[slot][:padded_rows].set(staged[slot][:padded_rows], stream=self._copy_stream)
                copied[slot].record(self._copy_stream)
                return True

            count = len(self._block_rows)
            if not upload(first_block):
                yield from self._iter_store(first_block)
                return
            for block_index in range(first_block, count):
                slot = block_index % 2
                copied[slot].synchronize()
                following = block_index + 1 < count and upload(block_index + 1)
                compute.wait_event(copied[slot])
                self._keep(block_index, slot)
                yield block_index, self._tile(block_index, slot)
                computed[slot].record(compute)
                if block_index + 1 < count and not following:
                    yield from self._iter_store(block_index + 1)
                    return
        finally:
            if lease is not None:
                lease.release()

    def _iter_store(self, first_block: int) -> Iterator[tuple[int, CodeBlockTile]]:
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

    def _iter_prepared(self, first_block: int) -> Iterator[tuple[int, CodeBlockTile]]:
        """The blocks from ``first_block`` on from their prepared codes where every one of them is a whole restart unit
        (``restart_store``: ``codes``, one per block, keyed by the store, the block's rows, the samples and the source of
        this class with every definition it reaches, the store's reads among them, never the engine's), else from the store, each block missing its unit then kept as one. A
        restarted fit's resident set then fills from the compact signed codes (``resident_codes_bytes``, 8-15 GB a
        device on All of Us whole-genome arms [real]) on the local disk, not from a read of the whole store (547 GB,
        about 20 minutes [real]). The device placement, the means and the scales are not the codes' bytes: a unit serves
        either device, and every tile takes its standardization from this fit's Stage 0. A block's host copy lives for
        that block alone, charged to the ledger. A partly prepared set is read from the codes Stage 0's pass kept, where
        they are held, else from the store (``_iter_read``), which completes it."""
        cache = self._code_cache
        if self._code_keys is None:
            source = definition_sources([("sv_pgs.store_block_source", "StoreGenotypeBlockSource")])
            samples = self._sample_indices if self._sample_indices is not None else np.arange(self._samples, dtype=np.int64)
            self._code_keys = []
            for rows in self._block_rows:
                named = facts({"store": cache.store_key, "rows": rows, "samples": samples, "padded_samples": self._padded_samples, "source": source})
                self._code_keys.append((unit_key(named), named))
        units = [cache.store.unit("codes", key) for key, _named in self._code_keys[first_block:]]
        if all(unit is not None for unit in units):
            self._origin = "units"
            log(f"resident genotype codes: blocks {first_block} to {len(self._block_rows) - 1} read from their prepared codes on {self._pool}")
            for block_index, (path, _record) in zip(range(first_block, len(self._block_rows)), units):
                slot = block_index % 2
                shape = (_aligned(int(self._block_rows[block_index].shape[0])), self._padded_samples)
                values = np.load(path / "codes.npy", mmap_mode="r", allow_pickle=False)
                if values.shape != shape or values.dtype != np.dtype(np.int8):
                    raise ValueError(f"{path}'s prepared codes are {values.shape} {values.dtype}, not {shape} int8")
                with self._broker.reserve(HOST, values.nbytes, "a prepared genotype-code block"):
                    self._signed[slot][:shape[0]].set(values)
                    self._cupy.cuda.get_current_stream().synchronize()
                del values
                self._keep(block_index, slot)
                yield block_index, self._tile(block_index, slot)
            return
        for (block_index, tile), unit in zip(self._iter_read(first_block), units):
            if unit is None:
                key, named = self._code_keys[block_index]
                rows = _aligned(int(self._block_rows[block_index].shape[0]))
                with self._broker.reserve(HOST, rows * self._padded_samples, "a prepared genotype-code block"):
                    values = self._signed[block_index % 2][:rows].get()
                    with cache.store.build("codes", key, group=cache.group, named=named, supersede=False) as directory:
                        np.save(directory / "codes.npy", values, allow_pickle=False)
                    del values
            yield block_index, tile

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


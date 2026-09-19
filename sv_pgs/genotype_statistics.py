"""Stage 0: the single phenotype-independent pass over the 8-bit dosage store.

Each chromosome streams once, in variant order, as tiles of half a block. For every
tile the device buffer (``genotype_buffers``)

1. uploads and lays out the codes (``s = code - 127``, samples grouped),
2. multiplies the tile's profile samples against the previous ``cap - 1`` variants and turns
   the band into fixed-point pair weights, the LD each cut position would separate.

``ld_partition.OnlineBlockPartitioner`` solves the minimum-cost block partition online and
releases each block as soon as its boundaries are certain; the buffer then computes the
block's exact per-group Gram, sums and cross-products from the rows it still holds. Chromosome
ends are always cuts, so chromosomes are independent: with several devices each takes whole
chromosomes (longest first, onto the least-loaded device) with its own read-ahead thread and
a double-buffered upload, so reading, uploading and computing overlap.

Per block and sample group ``g`` the pass keeps exact integers,

    S_g = sum_{i in g} s_i s_i^T,    u_g = sum_{i in g} s_i,    n_g = |g|,

which add across groups (every fold but one is an exact sum). Standardization happens once,
in fp64, from ``N = n S - u u^T`` (exact int64; ``|N| <= 127^2 n^2``, so its fp64 value is
exact below n = 747,000): ``X^T X = n R`` with ``R = N / sqrt(diag N diag N^T)`` for the
model's population-sd standardized columns. The 1/127 of ``DS = (s + 127) / 127`` cancels.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from sv_pgs.compute_budget import ComputeBudget
from sv_pgs.config import ModelConfig
from sv_pgs.data import TieMap
from sv_pgs.genotype import _try_import_cupy
from sv_pgs.genotype_buffers import (
    INT32_EXACT_ROWS,
    SIGNED_CODE_OFFSET,
    CudaGenotypeBuffer,
    HostGenotypeBuffer,
    SampleLayout,
    build_sample_layout,
    cuda_buffer_bytes,
    host_buffer_bytes,
)
from sv_pgs.ld_partition import LdBlockBoundaries, OnlineBlockPartitioner, cut_allowed_from_groups
from sv_pgs.preprocessing import _compact_identity_tie_map, tie_map_from_groups
from sv_pgs.progress import log

LAG_BLOCKS = 2
"""Blocks a boundary decision may trail the stream before ``force_cut`` commits one. It sets the
retained buffer, ``(LAG_BLOCKS + 2) * cap + tile_rows`` rows, and never depends on the hardware."""

BLOCK_CAP_STEP = 256

_HOST_TILES = 3


class GenotypeTileSource(Protocol):
    """Variant-major 8-bit dosage codes, one chromosome at a time. Must be thread-safe
    across chromosomes: each device reads its own."""

    @property
    def sample_count(self) -> int: ...

    def chromosomes(self) -> Sequence[str]: ...

    def variant_count(self, chromosome: str) -> int: ...

    def unsplittable_groups(self, chromosome: str) -> NDArray[np.int64]:
        """Per variant: adjacent variants with equal ids (one bubble or repeat locus) share a block."""
        ...

    def read_rows(self, chromosome: str, start: int, stop: int, out: NDArray[np.uint8]) -> None:
        """Fill ``out`` (``stop - start`` rows) with the codes ``[variants, samples]`` of ``start..stop``."""
        ...

    def store_rows(self, chromosome: str) -> NDArray[np.int64]:
        """The store row of each variant the chromosome streams."""
        ...


class GenotypeBuffer(Protocol):
    """One device's genotype buffer (``genotype_buffers.HostGenotypeBuffer`` or ``CudaGenotypeBuffer``)."""

    array_module: ModuleType

    def context(self) -> Any: ...

    def host_tile(self, tile_rows: int) -> NDArray[np.uint8]: ...

    def stage_tile(self, codes: NDArray[np.uint8], staging_index: int) -> None: ...

    def release_staged_tile(self, staging_index: int) -> None: ...

    def load_staged_tile(self, staging_index: int, rows: int, slot: int) -> None: ...

    def check_codes(self) -> None: ...

    def pair_weights(
        self, window_slot: int, tile_slot: int, tile_rows: int, maximum_distance: int
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]: ...

    def block_statistics(self, slot: int, rows: int) -> tuple[Any, Any, Any]: ...

    def parallel_rows(self, function: Callable[[int, int], None], rows: int) -> None: ...

    def to_host(self, array: Any) -> NDArray: ...

    def cross_products(self, slot: int, rows: int) -> Any: ...

    def move_rows(self, source_slot: int, target_slot: int, rows: int) -> None: ...


@dataclass(frozen=True, slots=True)
class BlockStatistics:
    """Exact statistics of the variants ``[start, stop)`` of one chromosome (source order).

    Arrays live in ``array_module`` (NumPy, or CuPy on the buffer's device): ``grams``
    ``(G, width, width)`` int32 when every group has at most ``INT32_EXACT_ROWS`` samples,
    else int64; ``sums`` ``(G, width)`` int64; ``cross_products`` ``(G, width, columns)``
    float64 ``sum_{i in g} s_i y_i^T``, or ``None``. ``group_counts`` is NumPy.
    """

    chromosome: str
    start: int
    stop: int
    group_counts: NDArray[np.int64]
    sums: Any
    grams: Any
    cross_products: Any
    array_module: ModuleType = np

    @property
    def width(self) -> int:
        return self.stop - self.start

    def on_host(self, buffer: GenotypeBuffer) -> BlockStatistics:
        """The same statistics as NumPy arrays (``buffer`` holds the block)."""
        return replace(
            self,
            sums=buffer.to_host(self.sums),
            grams=buffer.to_host(self.grams),
            cross_products=None if self.cross_products is None else buffer.to_host(self.cross_products),
            array_module=np,
        )


@dataclass(frozen=True, slots=True)
class GenotypePassPlan:
    """Sizes of one pass: every block has at most ``block_cap`` variants."""

    block_cap: int
    tile_rows: int
    capacity_rows: int


def plan_genotype_pass(block_cap: int) -> GenotypePassPlan:
    """Tiles of half a block (a multiple of 64 variants) and the retained buffer they need.

    A tile's band costs ``(cap + tile) * tile`` products and each tile pays a fixed launch and
    synchronization cost, so half a block keeps the band within 1.5x its minimum while halving
    the per-tile overhead that bounds the pass on fast GPUs.
    """
    if block_cap < 64:
        raise ValueError("the LD block cap must be at least 64 variants")
    tile_rows = max(64, block_cap // 2 // 64 * 64)
    return GenotypePassPlan(block_cap=block_cap, tile_rows=tile_rows, capacity_rows=(LAG_BLOCKS + 2) * block_cap + tile_rows)


def largest_block_cap(available_bytes: int, bytes_for_cap: Callable[[int], int]) -> int:
    """The largest block cap (a multiple of ``BLOCK_CAP_STEP``) whose working set fits.

    ``bytes_for_cap`` must grow with the cap; ``genotype_buffers.host_buffer_bytes`` and
    ``cuda_buffer_bytes`` model Stage 0, and a caller takes the minimum over its stages.
    """
    if bytes_for_cap(BLOCK_CAP_STEP) > available_bytes:
        raise MemoryError(
            f"{available_bytes / 1e9:.2f} GB cannot hold the working set of a {BLOCK_CAP_STEP}-variant "
            f"LD block ({bytes_for_cap(BLOCK_CAP_STEP) / 1e9:.2f} GB)"
        )
    low = BLOCK_CAP_STEP
    while bytes_for_cap(2 * low) <= available_bytes:
        low *= 2
    high = 2 * low
    while high - low > BLOCK_CAP_STEP:
        middle = (low + high) // 2 // BLOCK_CAP_STEP * BLOCK_CAP_STEP
        if bytes_for_cap(middle) <= available_bytes:
            low = middle
        else:
            high = middle
    return low


@dataclass(slots=True)
class ChromosomeSummary:
    chromosome: str
    boundaries: NDArray[np.int64]
    cut_costs: NDArray[np.int64]
    forced_cuts: int
    seconds: float


@dataclass(slots=True)
class GenotypePassSummary:
    chromosomes: dict[str, ChromosomeSummary] = field(default_factory=dict)
    seconds: float = 0.0


def assign_chromosomes(variant_counts: dict[str, int], device_count: int) -> list[list[str]]:
    """Longest chromosome first onto the least-loaded device (ties: lowest device, source order)."""
    loads = [0] * device_count
    assignment: list[list[str]] = [[] for _ in range(device_count)]
    for chromosome in sorted(variant_counts, key=lambda name: -variant_counts[name]):
        device = min(range(device_count), key=lambda index: (loads[index], index))
        assignment[device].append(chromosome)
        loads[device] += variant_counts[chromosome]
    return assignment


def _run_on_devices(
    source: GenotypeTileSource,
    buffers: Sequence[GenotypeBuffer],
    work: Callable[[GenotypeBuffer, str], None],
) -> None:
    """Run ``work(buffer, chromosome)`` for every chromosome, one thread per buffer."""
    counts = {chromosome: source.variant_count(chromosome) for chromosome in source.chromosomes()}
    failures: list[BaseException] = []

    def device_worker(buffer: GenotypeBuffer, chromosomes: list[str]) -> None:
        try:
            for chromosome in chromosomes:
                work(buffer, chromosome)
        except BaseException as error:
            failures.append(error)

    workers = [
        threading.Thread(target=device_worker, args=(buffer, chromosomes), name=f"stage0-device-{index}")
        for index, (buffer, chromosomes) in enumerate(zip(buffers, assign_chromosomes(counts, len(buffers))))
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    if failures:
        raise failures[0]


class _TileStream:
    """Tiles of one chromosome, read ahead on a thread and uploaded one tile ahead.

    ``current`` is the staged tile ``(start, stop, staging_index)``; once it is loaded,
    ``advance`` frees its host tile and stages the next, so the upload of tile ``i + 1`` and
    the read of tile ``i + 2`` overlap the computation on tile ``i``.
    """

    def __init__(self, source: GenotypeTileSource, chromosome: str, buffer: GenotypeBuffer, tile_rows: int) -> None:
        self._buffer = buffer
        self._free: queue.Queue = queue.Queue()
        for _ in range(_HOST_TILES):
            self._free.put(buffer.host_tile(tile_rows))
        self._filled: queue.Queue = queue.Queue()
        self._reader = threading.Thread(
            target=self._read_ahead,
            args=(source, chromosome, source.variant_count(chromosome), tile_rows),
            name=f"stage0-read-{chromosome}",
            daemon=True,
        )
        self._reader.start()
        self._staging = 0
        self._host: NDArray[np.uint8] | None = None
        self.current: tuple[int, int, int] | None = None
        self._stage_next()

    def _read_ahead(self, source: GenotypeTileSource, chromosome: str, variant_count: int, tile_rows: int) -> None:
        try:
            for start in range(0, variant_count, tile_rows):
                stop = min(start + tile_rows, variant_count)
                host = self._free.get()
                source.read_rows(chromosome, start, stop, host[: stop - start])
                self._filled.put((start, stop, host))
            self._filled.put(None)
        except BaseException as error:
            self._filled.put(error)

    def _stage_next(self) -> None:
        item = self._filled.get()
        if isinstance(item, BaseException):
            raise item
        if item is None:
            self.current = None
            self._reader.join()
            return
        start, stop, host = item
        self._buffer.stage_tile(host[: stop - start], self._staging)
        self._host = host
        self.current = (start, stop, self._staging)

    def advance(self) -> None:
        """Call once the current tile is loaded."""
        self._buffer.release_staged_tile(self._staging)
        self._free.put(self._host)
        self._staging = 1 - self._staging
        self._stage_next()


def run_genotype_pass(
    source: GenotypeTileSource,
    layout: SampleLayout,
    buffers: Sequence[GenotypeBuffer],
    plan: GenotypePassPlan,
    sink: Callable[[BlockStatistics, GenotypeBuffer], None],
) -> GenotypePassSummary:
    """Stream every chromosome of ``source`` once and hand each finished block to ``sink``.

    ``sink(block, buffer)`` runs on the block's device thread inside ``buffer.context()``, so
    it may keep working on the device arrays (``buffer.parallel_rows`` spreads row panels over
    the device); calls from different devices overlap, so it must be thread-safe. Blocks of
    one chromosome arrive in order.
    """
    if layout.store_width != source.sample_count:
        raise ValueError("the sample layout does not match the store's sample count")
    started = time.monotonic()
    summary = GenotypePassSummary()
    summary_lock = threading.Lock()

    def chromosome_work(buffer: GenotypeBuffer, chromosome: str) -> None:
        result = _run_chromosome(source, layout, buffer, plan, chromosome, sink)
        with summary_lock:
            summary.chromosomes[chromosome] = result

    _run_on_devices(source, buffers, chromosome_work)
    summary.seconds = time.monotonic() - started
    return summary


def _run_chromosome(
    source: GenotypeTileSource,
    layout: SampleLayout,
    buffer: GenotypeBuffer,
    plan: GenotypePassPlan,
    chromosome: str,
    sink: Callable[[BlockStatistics, GenotypeBuffer], None],
) -> ChromosomeSummary:
    started = time.monotonic()
    variant_count = source.variant_count(chromosome)
    groups = np.asarray(source.unsplittable_groups(chromosome), dtype=np.int64)
    if groups.shape[0] != variant_count:
        raise ValueError(f"{chromosome}: {groups.shape[0]} unsplittable group ids for {variant_count} variants")
    partitioner = OnlineBlockPartitioner(cut_allowed_from_groups(groups), plan.block_cap)
    maximum_distance = plan.block_cap - 1
    gram_dtype = "int32" if int(layout.group_counts.max()) <= INT32_EXACT_ROWS else "int64"
    stream = _TileStream(source, chromosome, buffer, plan.tile_rows)
    boundaries = [0]
    forced = 0
    base = 0

    def emit(cuts: list[int]) -> None:
        for cut in cuts:
            start = boundaries[-1]
            sums, grams, cross = buffer.block_statistics(start - base, cut - start)
            with buffer.context():
                sink(
                    BlockStatistics(
                        chromosome=chromosome,
                        start=start,
                        stop=cut,
                        group_counts=layout.group_counts.copy(),
                        sums=sums,
                        grams=grams.astype(gram_dtype, copy=False),
                        cross_products=cross,
                        array_module=buffer.array_module,
                    ),
                    buffer,
                )
            boundaries.append(cut)

    while stream.current is not None:
        start, stop, staging = stream.current
        rows = stop - start
        while stop - base > plan.capacity_rows:
            if partitioner.committed > base:
                buffer.move_rows(partitioner.committed - base, 0, start - partitioner.committed)
                base = partitioner.committed
            else:
                emit(partitioner.force_cut())
                forced += 1
        buffer.load_staged_tile(staging, rows, start - base)
        stream.advance()
        window_start = max(0, start - maximum_distance)
        row_weights, column_weights = buffer.pair_weights(window_start - base, start - base, rows, maximum_distance)
        buffer.check_codes()
        partitioner.add_pair_weights(window_start, row_weights, start, column_weights)
        emit(partitioner.advance(stop - maximum_distance))
    emit(partitioner.finish())
    seconds = time.monotonic() - started
    log(
        f"stage0 {chromosome}: {variant_count:,} variants in {len(boundaries) - 1:,} LD blocks "
        f"(cap {plan.block_cap}, {forced} forced cuts) in {seconds:.1f}s"
    )
    return ChromosomeSummary(
        chromosome=chromosome,
        boundaries=np.asarray(boundaries, dtype=np.int64),
        cut_costs=partitioner.cut_costs.copy(),
        forced_cuts=forced,
        seconds=seconds,
    )


def run_cross_product_pass(
    source: GenotypeTileSource,
    buffers: Sequence[GenotypeBuffer],
    tile_rows: int,
    sink: Callable[[str, int, int, Any], None],
) -> float:
    """Stream the store once more for cross-product columns that arrive after the Stage 0 pass.

    Each buffer must be built with the columns and room for ``tile_rows`` rows.
    ``sink(chromosome, start, stop, products)`` receives ``(G, stop - start, columns)``
    ``sum_{i in g} s_i y_i^T`` per tile, on the device thread inside the buffer's context
    (thread-safe, as for ``run_genotype_pass``). Returns the wall time in seconds.
    """
    started = time.monotonic()

    def chromosome_work(buffer: GenotypeBuffer, chromosome: str) -> None:
        stream = _TileStream(source, chromosome, buffer, tile_rows)
        while stream.current is not None:
            start, stop, staging = stream.current
            buffer.load_staged_tile(staging, stop - start, 0)
            stream.advance()
            products = buffer.cross_products(0, stop - start)
            buffer.check_codes()
            with buffer.context():
                sink(chromosome, start, stop, products)

    _run_on_devices(source, buffers, chromosome_work)
    return time.monotonic() - started


TIE_CORRELATION_SCREEN = 1.0 - 1e-9
"""Pairs with fp64 ``|r|`` above this are tested for an exact tie in integer arithmetic."""


@dataclass(frozen=True, slots=True)
class ProjectedLdBlock:
    """One LD block of the reduced model, covariate-projected (Frisch-Waugh-Lovell).

    With ``X`` the population-sd standardized training genotypes of the block's reduced
    columns and ``C`` the covariates (intercept included):
    ``projected_gram = X^T (I - H_C) X`` (fp32), ``projected_score = X^T (I - H_C) Y`` and
    ``covariate_cross = X^T C`` (unprojected).
    """

    chromosome: str
    reduced_columns: NDArray[np.int64]
    projected_gram: NDArray[np.float32]
    projected_score: NDArray[np.float64]
    covariate_cross: NDArray[np.float64]


_LD_INDEX = "ld_blocks.json"
_PENDING_BLOCK_WRITES = 4
"""Blocks the writer thread may hold while the disk catches up (bounds the host memory in flight)."""
_LD_ARRAYS = {
    "gram": ("ld_grams.f32", np.float32),
    "score": ("ld_projected_scores.f64", np.float64),
    "cross": ("ld_covariate_cross.f64", np.float64),
    "diagonal": ("ld_diagonal.f64", np.float64),
    "ld_score": ("ld_score.f64", np.float64),
}


class LdGramStore:
    """The projected LD blocks of one fit on disk (memory-mapped, read by Stage 1 and 2).

    It is ``ld_space_fit.LDBlockSource``: ``correlation_block(b) = X~_b^T X~_b / n``, with
    the per-column diagonal and within-block LD scores computed once in Stage 0.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        index = json.loads((self.directory / _LD_INDEX).read_text(encoding="utf-8"))
        self.sample_count = int(index["sample_count"])
        self.target_count = int(index["target_count"])
        self.covariate_count = int(index["covariate_count"])
        self._blocks = index["blocks"]
        self.block_boundaries = np.asarray([0] + [int(entry["reduced_stop"]) for entry in self._blocks], dtype=np.int64)
        self._arrays = {
            name: np.memmap(self.directory / file_name, dtype=dtype, mode="r")
            for name, (file_name, dtype) in _LD_ARRAYS.items()
            if (self.directory / file_name).stat().st_size
        }

    @property
    def block_count(self) -> int:
        return len(self._blocks)

    @property
    def stored_bytes(self) -> int:
        return sum(array.nbytes for array in self._arrays.values())

    def block(self, block_index: int) -> ProjectedLdBlock:
        entry = self._blocks[block_index]
        width = int(entry["reduced_stop"]) - int(entry["reduced_start"])

        def view(name: str, columns: int) -> NDArray:
            if width * columns == 0:
                return np.zeros((width, columns), dtype=_LD_ARRAYS[name][1])
            offset = int(entry[f"{name}_offset"])
            return np.asarray(self._arrays[name][offset : offset + width * columns]).reshape(width, columns)

        return ProjectedLdBlock(
            chromosome=entry["chromosome"],
            reduced_columns=np.arange(int(entry["reduced_start"]), int(entry["reduced_stop"]), dtype=np.int64),
            projected_gram=view("gram", width),
            projected_score=view("score", self.target_count),
            covariate_cross=view("cross", self.covariate_count),
        )

    def correlation_block(self, block_index: int) -> NDArray[np.float32]:
        """``R_b = X~_b^T X~_b / n`` (exactly symmetric, float32)."""
        return self.block(block_index).projected_gram * np.float32(1.0 / self.sample_count)

    def _concatenated(self, name: str) -> NDArray[np.float64]:
        if name not in self._arrays:
            return np.zeros(0, dtype=np.float64)
        return np.asarray(self._arrays[name], dtype=np.float64)

    def ld_diagonal(self) -> NDArray[np.float64]:
        """``R_jj`` for every reduced column, in reduced order."""
        return self._in_reduced_order("diagonal")

    def ld_scores(self) -> NDArray[np.float64]:
        """``sum_i R_ij^2`` within its block for every reduced column, in reduced order."""
        return self._in_reduced_order("ld_score")

    def _in_reduced_order(self, name: str) -> NDArray[np.float64]:
        values = self._concatenated(name)
        pieces = [
            values[int(entry[f"{name}_offset"]) : int(entry[f"{name}_offset"]) + int(entry["reduced_stop"]) - int(entry["reduced_start"])]
            for entry in self._blocks
        ]
        return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float64)


class _LdGramWriter:
    """Appends projected blocks as they arrive from any device, on its own thread so the disk
    overlaps the devices; the index is written in store order."""

    def __init__(self, directory: Path, sample_count: int, target_count: int, covariate_count: int) -> None:
        self.directory = Path(directory)
        self._sample_count = sample_count
        self.directory.mkdir(parents=True, exist_ok=True)
        self._files = {name: open(self.directory / file_name, "wb") for name, (file_name, _) in _LD_ARRAYS.items()}
        self._offsets = {name: 0 for name in _LD_ARRAYS}
        self._target_count = target_count
        self._covariate_count = covariate_count
        self._pending: queue.Queue = queue.Queue(maxsize=_PENDING_BLOCK_WRITES)
        self._failures: list[BaseException] = []
        self._thread = threading.Thread(target=self._write_all, name="stage0-ld-writer", daemon=True)
        self._thread.start()
        self.entries: dict[tuple[str, int], dict[str, Any]] = {}

    def append(self, chromosome: str, start: int, arrays: dict[str, NDArray]) -> None:
        """Queue one block (the arrays must stay unchanged until written)."""
        if self._failures:
            raise self._failures[0]
        self._pending.put((chromosome, start, arrays))

    def _write_all(self) -> None:
        while True:
            item = self._pending.get()
            if item is None:
                return
            chromosome, start, arrays = item
            try:
                entry: dict[str, Any] = {"chromosome": chromosome}
                for name, values in arrays.items():
                    entry[f"{name}_offset"] = self._offsets[name]
                    self._files[name].write(memoryview(np.ascontiguousarray(values, dtype=_LD_ARRAYS[name][1])).cast("B"))
                    self._offsets[name] += int(values.size)
                self.entries[(chromosome, start)] = entry
            except BaseException as error:
                self._failures.append(error)

    def finish(self, ordered_blocks: list[tuple[str, int, int, int]]) -> LdGramStore:
        """Write the index for ``(chromosome, start, reduced_start, reduced_stop)`` in store order."""
        self._pending.put(None)
        self._thread.join()
        if self._failures:
            raise self._failures[0]
        for handle in self._files.values():
            handle.close()
        blocks = []
        for chromosome, start, reduced_start, reduced_stop in ordered_blocks:
            entry = dict(self.entries[(chromosome, start)])
            entry.update(reduced_start=reduced_start, reduced_stop=reduced_stop)
            blocks.append(entry)
        index = {
            "sample_count": self._sample_count,
            "target_count": self._target_count,
            "covariate_count": self._covariate_count,
            "blocks": blocks,
        }
        (self.directory / _LD_INDEX).write_text(json.dumps(index), encoding="utf-8")
        return LdGramStore(self.directory)


@dataclass(frozen=True, slots=True)
class GenotypeSufficientStatistics:
    """Everything the fast path needs from the genotypes of one training set (reduced space)."""

    sample_count: int
    active_rows: NDArray[np.int64]
    means: NDArray[np.float64]
    scales: NDArray[np.float64]
    allele_frequency: NDArray[np.float64]
    tie_map: TieMap
    block_of_reduced: NDArray[np.int32]
    covariate_gram: NDArray[np.float64]
    covariate_target: NDArray[np.float64]
    target_gram: NDArray[np.float64]
    ld: LdGramStore
    boundaries: LdBlockBoundaries


@dataclass(frozen=True, slots=True)
class _Projection:
    covariate_count: int
    column_sums: NDArray[np.float64]
    covariate_gram_inverse: NDArray[np.float64]
    covariate_target: NDArray[np.float64]
    minimum_minor_allele_frequency: float
    minimum_scale: float


@dataclass(frozen=True, slots=True)
class _BlockSummary:
    """Host results of one block: per streamed row, and ties among its active rows."""

    active: NDArray[np.bool_]
    means: NDArray[np.float64]
    scales: NDArray[np.float64]
    frequency: NDArray[np.float64]
    representative: NDArray[np.int64]
    sign: NDArray[np.float32]
    reduced_count: int


def _exact_ties(
    candidates: NDArray[np.int64], block: BlockStatistics, active_index: NDArray[np.int64], buffer: GenotypeBuffer
) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    """Representative (lowest index) and sign of every active column.

    ``candidates`` are active-index pairs whose fp64 ``|r|`` passed the screen; a pair is an
    exact tie iff ``N_jk^2 = N_jj N_kk`` with ``N = n S - u u^T``, decided in Python integers
    (the product exceeds int64).
    """
    width = active_index.shape[0]
    representative = np.arange(width, dtype=np.int64)
    sign = np.ones(width, dtype=np.float32)
    if candidates.shape[0] == 0:
        return representative, sign
    count = int(block.group_counts[0])
    rows = active_index[candidates]
    xp = block.array_module
    grams = buffer.to_host(block.grams[0][xp.asarray(rows[:, 0]), xp.asarray(rows[:, 1])]).tolist()
    first_diagonal = buffer.to_host(block.grams[0][xp.asarray(rows[:, 0]), xp.asarray(rows[:, 0])]).tolist()
    second_diagonal = buffer.to_host(block.grams[0][xp.asarray(rows[:, 1]), xp.asarray(rows[:, 1])]).tolist()
    sums = buffer.to_host(block.sums[0])
    for (first, second), (first_row, second_row), gram, first_square, second_square in zip(
        candidates.tolist(), rows.tolist(), grams, first_diagonal, second_diagonal
    ):
        first_sum, second_sum = int(sums[first_row]), int(sums[second_row])
        cross = count * gram - first_sum * second_sum
        if cross * cross != (count * first_square - first_sum * first_sum) * (count * second_square - second_sum * second_sum):
            continue
        root_first, root_second = representative[first], representative[second]
        if root_first == root_second:
            continue
        merged_sign = sign[first] * sign[second] * (1.0 if cross > 0 else -1.0)
        low, high = min(root_first, root_second), max(root_first, root_second)
        members = representative == high
        sign[members] *= merged_sign
        representative[members] = low
    return representative, sign


def _project_block(
    block: BlockStatistics, projection: _Projection, buffer: GenotypeBuffer
) -> tuple[_BlockSummary, dict[str, NDArray]]:
    """Standardize, find exact ties, drop inactive and tied columns and project out the
    covariates on the block's device; return the host summary and the projected arrays.

    ``N = n S - u u^T`` is formed in fp64, where every term is an integer below 2^53 and so
    exact; the ``width^2`` work runs as row panels through ``buffer.parallel_rows``.
    """
    xp = block.array_module
    count = int(block.group_counts[0])
    width = block.width
    sums = block.sums[0].astype(xp.float64)
    diagonal = xp.diagonal(block.grams[0]).astype(xp.float64)
    variance_numerator = count * diagonal - sums * sums
    means = sums / count
    scales = xp.sqrt(variance_numerator) / count
    frequency = (means + SIGNED_CODE_OFFSET) / (2 * SIGNED_CODE_OFFSET)
    active = (variance_numerator > 0) & (xp.minimum(frequency, 1.0 - frequency) >= projection.minimum_minor_allele_frequency)
    active &= scales / SIGNED_CODE_OFFSET >= projection.minimum_scale
    active_host = buffer.to_host(active)
    active_index = np.flatnonzero(active_host)
    inverse_root = xp.where(active, 1.0 / xp.sqrt(xp.where(active, variance_numerator, 1.0)), 0.0)
    correlation = xp.empty((width, width), dtype=xp.float64)
    screened: list[NDArray[np.int64]] = []

    def correlate(start: int, stop: int) -> None:
        panel = correlation[start:stop]
        panel[...] = block.grams[0][start:stop]
        panel *= count
        panel -= sums[start:stop, None] * sums[None, :]
        panel *= inverse_root[start:stop, None]
        panel *= inverse_root[None, :]
        above = xp.arange(width)[None, :] > xp.arange(start, stop)[:, None]
        hits = xp.argwhere(((panel > TIE_CORRELATION_SCREEN) | (panel < -TIE_CORRELATION_SCREEN)) & above)
        screened.append(buffer.to_host(hits) + np.array([start, 0]))

    buffer.parallel_rows(correlate, width)
    pairs = np.concatenate(screened) if screened else np.zeros((0, 2), dtype=np.int64)
    pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
    position = np.full(width, -1, dtype=np.int64)
    position[active_index] = np.arange(active_index.shape[0])
    representative, sign = _exact_ties(position[pairs], block, active_index, buffer)
    kept = active_index[representative == np.arange(representative.shape[0])]
    kept_device = xp.asarray(kept)
    raw_cross = block.cross_products[0][kept_device]
    centered = raw_cross - xp.outer(sums[kept_device], xp.asarray(projection.column_sums)) / count
    standardized = centered * (count * inverse_root[kept_device])[:, None]
    covariate_cross = standardized[:, : projection.covariate_count]
    loading = covariate_cross @ xp.asarray(projection.covariate_gram_inverse)
    projected_score = standardized[:, projection.covariate_count :] - loading @ xp.asarray(projection.covariate_target)
    gram = xp.empty((kept.shape[0], kept.shape[0]), dtype=xp.float32)
    every_column = kept.shape[0] == width

    def project(start: int, stop: int) -> None:
        panel = correlation[start:stop] if every_column else correlation[kept_device[start:stop]][:, kept_device]
        panel *= count
        panel -= loading[start:stop] @ covariate_cross.T
        gram[start:stop] = panel

    buffer.parallel_rows(project, kept.shape[0])
    ld_score = xp.empty(kept.shape[0], dtype=xp.float64)

    def mirror_lower(start: int, stop: int) -> None:
        """Copy the lower triangle over the upper one (panels write disjoint upper parts),
        so the stored Gram is exactly symmetric."""
        gram[start:stop, stop:] = gram[stop:, start:stop].T
        square = gram[start:stop, start:stop]
        index = xp.arange(stop - start)
        square[...] = xp.where(index[:, None] <= index[None, :], square.T, square)

    def score_rows(start: int, stop: int) -> None:
        correlation_rows = gram[start:stop].astype(xp.float64) / count
        ld_score[start:stop] = (correlation_rows * correlation_rows).sum(axis=1)

    buffer.parallel_rows(mirror_lower, kept.shape[0])
    buffer.parallel_rows(score_rows, kept.shape[0])
    summary = _BlockSummary(
        active=active_host,
        means=buffer.to_host(means),
        scales=buffer.to_host(scales),
        frequency=buffer.to_host(frequency),
        representative=representative,
        sign=sign,
        reduced_count=int(kept.shape[0]),
    )
    arrays = {
        "gram": gram,
        "score": projected_score,
        "cross": covariate_cross,
        "diagonal": xp.diagonal(gram).astype(xp.float64) / count,
        "ld_score": ld_score,
    }
    return summary, {name: buffer.to_host(values) for name, values in arrays.items()}


def build_genotype_buffers(
    budget: ComputeBudget, layout: SampleLayout, plan: GenotypePassPlan, cross_product_columns: NDArray[np.float64] | None
) -> list[GenotypeBuffer]:
    """One buffer per CUDA device of ``budget``, or one host buffer on its CPU threads."""
    if budget.device_kind == "cuda":
        cupy = _try_import_cupy()
        if cupy is None:
            raise RuntimeError("the compute budget names CUDA devices but CuPy cannot be imported")
        return [
            CudaGenotypeBuffer(cupy, device_id, layout, plan.capacity_rows, plan.tile_rows, cross_product_columns)
            for device_id in budget.device_ids
        ]
    return [HostGenotypeBuffer(layout, plan.capacity_rows, cross_product_columns, worker_count=budget.cpu_threads)]


def stage0_block_cap(budget: ComputeBudget, layout: SampleLayout, cross_product_columns: int) -> int:
    """The largest LD block cap the Stage 0 working set of ``budget`` allows."""
    if budget.device_kind == "cuda":
        def device_bytes(cap: int) -> int:
            plan = plan_genotype_pass(cap)
            return cuda_buffer_bytes(layout, plan.capacity_rows, plan.tile_rows, cap, cross_product_columns)
        return largest_block_cap(budget.working_bytes, device_bytes)

    def host_bytes(cap: int) -> int:
        plan = plan_genotype_pass(cap)
        return host_buffer_bytes(layout, plan.capacity_rows, cap, cross_product_columns, budget.cpu_threads)
    return largest_block_cap(budget.host_bytes, host_bytes)


def compute_genotype_statistics(
    source: GenotypeTileSource,
    sample_indices: NDArray[np.int64],
    covariates: NDArray[np.float64],
    targets: NDArray[np.float64],
    config: ModelConfig,
    budget: ComputeBudget,
    block_cap: int,
    out_dir: Path,
) -> GenotypeSufficientStatistics:
    """Stage 0 for one training set: one genotype pass, LD blocks, projected LD on disk.

    ``covariates`` ``[n, k]`` (intercept included) and ``targets`` ``[n, T]`` follow
    ``sample_indices`` (sorted store columns). Variants whose training MAF is below
    ``config.minimum_minor_allele_frequency`` or whose dosage SD is below
    ``config.minimum_scale`` are inactive; exact duplicate or negated standardized columns
    within a block collapse into tie groups.
    """
    indices = np.asarray(sample_indices, dtype=np.int64)
    if indices.ndim != 1 or np.any(np.diff(indices) <= 0):
        raise ValueError("sample_indices must be sorted, distinct store columns")
    covariate_matrix = np.asarray(covariates, dtype=np.float64)
    target_matrix = np.asarray(targets, dtype=np.float64)
    if covariate_matrix.shape[0] != indices.shape[0] or target_matrix.shape[0] != indices.shape[0]:
        raise ValueError("covariates and targets must have one row per training sample")
    sample_side = np.concatenate((covariate_matrix, target_matrix), axis=1)
    sample_groups = np.full(source.sample_count, -1, dtype=np.int64)
    sample_groups[indices] = 0
    layout = build_sample_layout(sample_groups)
    maximum_cap = stage0_block_cap(budget, layout, sample_side.shape[1])
    if block_cap > maximum_cap:
        raise MemoryError(f"an LD block cap of {block_cap} exceeds the {maximum_cap} this device's memory allows")
    plan = plan_genotype_pass(block_cap)
    store_columns = np.zeros((source.sample_count, sample_side.shape[1]), dtype=np.float64)
    store_columns[indices] = sample_side
    covariate_gram = covariate_matrix.T @ covariate_matrix
    covariate_target = covariate_matrix.T @ target_matrix
    projection = _Projection(
        covariate_count=covariate_matrix.shape[1],
        column_sums=sample_side.sum(axis=0),
        covariate_gram_inverse=np.linalg.inv(covariate_gram),
        covariate_target=covariate_target,
        minimum_minor_allele_frequency=config.minimum_minor_allele_frequency,
        minimum_scale=config.minimum_scale,
    )
    writer = _LdGramWriter(out_dir, indices.shape[0], target_matrix.shape[1], covariate_matrix.shape[1])
    summaries: dict[tuple[str, int], tuple[int, _BlockSummary]] = {}
    summaries_lock = threading.Lock()

    buffers = build_genotype_buffers(budget, layout, plan, store_columns)
    log(f"stage0: {len(buffers)} {budget.device_kind} buffer(s), block cap {block_cap}, {layout.width:,} laid-out samples")

    def sink(block: BlockStatistics, buffer: GenotypeBuffer) -> None:
        block_summary, arrays = _project_block(block, projection, buffer)
        writer.append(block.chromosome, block.start, arrays)
        with summaries_lock:
            summaries[(block.chromosome, block.start)] = (block.stop, block_summary)

    summary = run_genotype_pass(source, layout, buffers, plan, sink)
    for buffer in buffers:
        buffer.close()
    return _assemble_statistics(source, summary, summaries, writer, indices.shape[0], block_cap,
                                covariate_gram, covariate_target, target_matrix.T @ target_matrix)


def _tie_map(representative: NDArray[np.int64], sign: NDArray[np.float32]) -> TieMap:
    """The TieMap of the active variants from each one's representative and sign."""
    members = np.arange(representative.shape[0], dtype=np.int64)
    if np.array_equal(representative, members):
        return _compact_identity_tie_map(representative.shape[0])
    groups: dict[int, list[tuple[int, float]]] = {}
    for member, (root, member_sign) in enumerate(zip(representative.tolist(), sign.tolist())):
        groups.setdefault(root, []).append((member, member_sign))
    return tie_map_from_groups(representative.shape[0], groups)


def _assemble_statistics(
    source: GenotypeTileSource,
    summary: GenotypePassSummary,
    summaries: dict[tuple[str, int], tuple[int, _BlockSummary]],
    writer: _LdGramWriter,
    sample_count: int,
    block_cap: int,
    covariate_gram: NDArray[np.float64],
    covariate_target: NDArray[np.float64],
    target_gram: NDArray[np.float64],
) -> GenotypeSufficientStatistics:
    """Put the per-block results in store order and index the reduced space."""
    chromosomes = tuple(source.chromosomes())
    active_rows, means, scales, frequency = [], [], [], []
    representatives: list[NDArray[np.int64]] = []
    signs: list[NDArray[np.float32]] = []
    ordered_blocks, block_of_reduced = [], []
    block_chromosomes, block_starts, block_stops = [], [], []
    active_cursor = reduced_cursor = 0
    for chromosome_index, chromosome in enumerate(chromosomes):
        store_rows = np.asarray(source.store_rows(chromosome), dtype=np.int64)
        boundaries = summary.chromosomes[chromosome].boundaries
        for start, stop in zip(boundaries[:-1].tolist(), boundaries[1:].tolist()):
            _, block_summary = summaries[(chromosome, start)]
            local_active = np.flatnonzero(block_summary.active)
            active_rows.append(store_rows[start + local_active])
            means.append(block_summary.means[local_active])
            scales.append(block_summary.scales[local_active])
            frequency.append(block_summary.frequency[local_active])
            representatives.append(active_cursor + block_summary.representative)
            signs.append(block_summary.sign)
            ordered_blocks.append((chromosome, start, reduced_cursor, reduced_cursor + block_summary.reduced_count))
            block_of_reduced.append(np.full(block_summary.reduced_count, len(ordered_blocks) - 1, dtype=np.int32))
            block_chromosomes.append(chromosome_index)
            block_starts.append(int(store_rows[start]))
            block_stops.append(int(store_rows[stop - 1]) + 1)
            active_cursor += local_active.shape[0]
            reduced_cursor += block_summary.reduced_count
    tie_map = _tie_map(np.concatenate(representatives), np.concatenate(signs))
    boundaries = LdBlockBoundaries(
        block_cap=block_cap,
        chromosomes=chromosomes,
        block_chromosomes=np.asarray(block_chromosomes, dtype=np.int32),
        block_starts=np.asarray(block_starts, dtype=np.int64),
        block_stops=np.asarray(block_stops, dtype=np.int64),
        cut_costs={name: item.cut_costs for name, item in summary.chromosomes.items()},
        forced_cuts=sum(item.forced_cuts for item in summary.chromosomes.values()),
    )
    ld = writer.finish(ordered_blocks)
    log(
        f"stage0: {active_cursor:,} active variants, {reduced_cursor:,} reduced columns in "
        f"{len(ordered_blocks):,} LD blocks ({boundaries.forced_cuts} forced cuts); pass {summary.seconds:.1f}s, "
        f"LD store {ld.stored_bytes / 1e9:.2f} GB"
    )
    return GenotypeSufficientStatistics(
        sample_count=sample_count,
        active_rows=np.concatenate(active_rows),
        means=np.concatenate(means),
        scales=np.concatenate(scales),
        allele_frequency=np.concatenate(frequency),
        tie_map=tie_map,
        block_of_reduced=np.concatenate(block_of_reduced),
        covariate_gram=covariate_gram,
        covariate_target=covariate_target,
        target_gram=target_gram,
        ld=ld,
        boundaries=boundaries,
    )

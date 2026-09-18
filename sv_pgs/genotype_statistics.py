"""Stage 0: the single phenotype-independent pass over the 8-bit dosage store.

Each chromosome streams once, in variant order, as tiles of a quarter block. For every
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

import queue
import threading
import time
from dataclasses import dataclass, field, replace
from types import ModuleType
from typing import Any, Callable, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from sv_pgs.genotype_buffers import INT32_EXACT_ROWS, SIGNED_CODE_OFFSET, SampleLayout
from sv_pgs.ld_partition import OnlineBlockPartitioner, cut_allowed_from_groups
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

    def on_host(self) -> BlockStatistics:
        """The same statistics as NumPy arrays."""
        if self.array_module is np:
            return self
        return replace(
            self,
            sums=self.sums.get(),
            grams=self.grams.get(),
            cross_products=None if self.cross_products is None else self.cross_products.get(),
            array_module=np,
        )


def pooled_integer_statistics(block: BlockStatistics, groups: Sequence[int]) -> tuple[int, Any, Any]:
    """``(n, S, u)`` summed over ``groups``, exact int64, in the block's array module."""
    xp = block.array_module
    selected = np.asarray(groups, dtype=np.int64)
    count = int(block.group_counts[selected].sum())
    gram = block.grams[xp.asarray(selected)].astype(xp.int64).sum(axis=0)
    sums = block.sums[xp.asarray(selected)].sum(axis=0)
    return count, gram, sums


def block_correlation(block: BlockStatistics, groups: Sequence[int]) -> tuple[int, Any, Any]:
    """``(n, R, constant)`` over ``groups``: the fp64 correlation from exact integers, so that
    ``X^T X = n R``, and the zero-variance flags (their rows and columns of ``R`` are zero)."""
    xp = block.array_module
    count, gram, sums = pooled_integer_statistics(block, groups)
    numerator = count * gram - xp.outer(sums, sums)
    diagonal = xp.diagonal(numerator).copy()
    constant = diagonal == 0
    inverse_root = xp.where(constant, 0.0, 1.0 / xp.sqrt(xp.where(constant, 1, diagonal).astype(xp.float64)))
    return count, numerator.astype(xp.float64) * inverse_root[:, None] * inverse_root[None, :], constant


def column_moments(block: BlockStatistics, groups: Sequence[int]) -> tuple[Any, Any]:
    """Mean and population standard deviation of the dosage ``DS = code / 127`` over ``groups``."""
    xp = block.array_module
    count, gram, sums = pooled_integer_statistics(block, groups)
    variance_numerator = (count * xp.diagonal(gram) - sums * sums).astype(xp.float64)
    mean = (sums.astype(xp.float64) / count + SIGNED_CODE_OFFSET) / SIGNED_CODE_OFFSET
    return mean, xp.sqrt(variance_numerator) / (count * SIGNED_CODE_OFFSET)


def centered_cross_products(block: BlockStatistics, groups: Sequence[int], column_sums: NDArray[np.float64]) -> Any:
    """``X^T (Y - 1 ybar^T)`` for population-sd standardized ``X`` over ``groups``;
    ``column_sums[g]`` is ``sum_{i in g} y_i`` per cross-product column."""
    if block.cross_products is None:
        raise ValueError("the block was computed without cross-product columns")
    xp = block.array_module
    selected = np.asarray(groups, dtype=np.int64)
    count, gram, sums = pooled_integer_statistics(block, selected)
    raw = block.cross_products[xp.asarray(selected)].sum(axis=0)
    y_sums = xp.asarray(np.asarray(column_sums, dtype=np.float64)[selected].sum(axis=0))
    centered = raw - xp.outer(sums.astype(xp.float64), y_sums) / count
    variance_numerator = (count * xp.diagonal(gram) - sums * sums).astype(xp.float64)
    varying = variance_numerator > 0
    scale = xp.where(varying, count / xp.sqrt(xp.where(varying, variance_numerator, 1.0)), 0.0)
    return centered * scale[:, None]


@dataclass(frozen=True, slots=True)
class GenotypePassPlan:
    """Sizes of one pass: every block has at most ``block_cap`` variants."""

    block_cap: int
    tile_rows: int
    capacity_rows: int


def plan_genotype_pass(block_cap: int) -> GenotypePassPlan:
    """Tiles of a quarter block (a multiple of 64 variants) and the retained buffer they need."""
    if block_cap < 64:
        raise ValueError("the LD block cap must be at least 64 variants")
    tile_rows = max(64, block_cap // 4 // 64 * 64)
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
    sink: Callable[[BlockStatistics], None],
) -> GenotypePassSummary:
    """Stream every chromosome of ``source`` once and hand each finished block to ``sink``.

    ``sink`` runs on the block's device thread inside its buffer's ``context()``, so it may
    keep working on the device arrays; calls from different devices overlap, so it must be
    thread-safe. Blocks of one chromosome arrive in order.
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
    sink: Callable[[BlockStatistics], None],
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
                    )
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

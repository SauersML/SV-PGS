"""Stage 0: the single phenotype-independent pass over the 8-bit dosage store.

Each chromosome streams once, in variant order, as tiles of ``tile_rows`` variants. For
every tile the backend

1. uploads and lays out the codes (``s = code - 127``, samples grouped, see ``layout``),
2. multiplies the tile's profile samples against the previous ``cap - 1`` variants, and
   turns the band into fixed-point pair weights, the LD every cut position separates.

The partitioner (``partition``) solves the exact minimum-cost block partition online and
releases a block as soon as its boundaries are certain; the backend then computes the
block's exact per-group Gram, sums and cross-products from the rows it still holds.
Chromosome ends are always cuts, so chromosomes are independent: with several devices,
each takes whole chromosomes (longest first, onto the least-loaded device), with its own
read-ahead thread and a double-buffered upload, so reading, uploading and computing
overlap.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from sv_pgs.progress import log
from sv_pgs.stage0.layout import SampleLayout
from sv_pgs.stage0.partition import OnlineBlockPartitioner, cut_allowed_from_groups
from sv_pgs.stage0.statistics import INT32_EXACT_ROWS, BlockStatistics

LAG_BLOCKS = 2
"""Blocks a boundary decision may lag the stream before ``force_cut`` commits one; it sets
the retained buffer, ``(LAG_BLOCKS + 2) * cap + tile_rows`` rows, and never the hardware."""

_HOST_TILES = 3


class GenotypeTileSource(Protocol):
    """Variant-major 8-bit dosage codes, one chromosome at a time."""

    @property
    def sample_count(self) -> int: ...

    def chromosomes(self) -> Sequence[str]: ...

    def variant_count(self, chromosome: str) -> int: ...

    def unsplittable_groups(self, chromosome: str) -> NDArray[np.int64]:
        """Per variant: adjacent variants with equal ids (one bubble or repeat locus) stay in one block."""
        ...

    def read_rows(self, chromosome: str, start: int, stop: int, out: NDArray[np.uint8]) -> None:
        """Fill ``out[: stop - start]`` with the codes ``[variants, samples]`` of variants ``start..stop``."""
        ...


class Stage0Backend(Protocol):
    """One device's genotype buffer and exact products (``cpu_backend`` or ``cuda_backend``)."""

    def host_tile(self, tile_rows: int) -> NDArray[np.uint8]: ...

    def stage_tile(self, codes: NDArray[np.uint8], staging_index: int) -> None: ...

    def release_staged_tile(self, staging_index: int) -> None: ...

    def load_staged_tile(self, staging_index: int, rows: int, slot: int) -> None: ...

    def check_codes(self) -> None: ...

    def pair_weights(
        self, window_slot: int, tile_slot: int, tile_rows: int, maximum_distance: int
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]: ...

    def block_statistics(
        self, slot: int, rows: int
    ) -> tuple[NDArray[np.int64], NDArray[np.integer], NDArray[np.float64] | None]: ...

    def move_rows(self, source_slot: int, target_slot: int, rows: int) -> None: ...


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
    return GenotypePassPlan(
        block_cap=block_cap,
        tile_rows=tile_rows,
        capacity_rows=(LAG_BLOCKS + 2) * block_cap + tile_rows,
    )


@dataclass(slots=True)
class ChromosomeSummary:
    chromosome: str
    boundaries: NDArray[np.int64]
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


def run_genotype_pass(
    source: GenotypeTileSource,
    layout: SampleLayout,
    backends: Sequence[Stage0Backend],
    plan: GenotypePassPlan,
    sink: Callable[[BlockStatistics], None],
) -> GenotypePassSummary:
    """Stream every chromosome of ``source`` once and hand each finished block to ``sink``.

    ``sink`` is called from one thread at a time. Blocks of one chromosome arrive in order.
    """
    if layout.store_width != source.sample_count:
        raise ValueError("the sample layout does not match the store's sample count")
    started = time.monotonic()
    counts = {chromosome: source.variant_count(chromosome) for chromosome in source.chromosomes()}
    assignment = assign_chromosomes(counts, len(backends))
    summary = GenotypePassSummary()
    sink_lock = threading.Lock()
    failures: list[BaseException] = []

    def locked_sink(block: BlockStatistics) -> None:
        with sink_lock:
            sink(block)

    def device_worker(backend: Stage0Backend, chromosomes: list[str]) -> None:
        try:
            for chromosome in chromosomes:
                result = _run_chromosome(source, layout, backend, plan, chromosome, locked_sink)
                with sink_lock:
                    summary.chromosomes[chromosome] = result
        except BaseException as error:
            failures.append(error)

    workers = [
        threading.Thread(target=device_worker, args=(backend, chromosomes), name=f"stage0-device-{index}")
        for index, (backend, chromosomes) in enumerate(zip(backends, assignment))
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    if failures:
        raise failures[0]
    summary.seconds = time.monotonic() - started
    return summary


def _read_ahead(
    source: GenotypeTileSource,
    chromosome: str,
    variant_count: int,
    tile_rows: int,
    free: queue.Queue,
    filled: queue.Queue,
) -> None:
    try:
        for start in range(0, variant_count, tile_rows):
            stop = min(start + tile_rows, variant_count)
            buffer = free.get()
            source.read_rows(chromosome, start, stop, buffer[: stop - start])
            filled.put((start, stop, buffer))
        filled.put(None)
    except BaseException as error:
        filled.put(error)


def _next_tile(filled: queue.Queue) -> tuple[int, int, NDArray[np.uint8]] | None:
    item = filled.get()
    if isinstance(item, BaseException):
        raise item
    return item


def _run_chromosome(
    source: GenotypeTileSource,
    layout: SampleLayout,
    backend: Stage0Backend,
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
    int32_grams = int(layout.group_counts.max()) <= INT32_EXACT_ROWS
    free: queue.Queue = queue.Queue()
    for _ in range(_HOST_TILES):
        free.put(backend.host_tile(plan.tile_rows))
    filled: queue.Queue = queue.Queue()
    reader = threading.Thread(
        target=_read_ahead,
        args=(source, chromosome, variant_count, plan.tile_rows, free, filled),
        name=f"stage0-read-{chromosome}",
        daemon=True,
    )
    reader.start()
    boundaries = [0]
    forced = 0
    base = 0

    def emit(cuts: list[int]) -> None:
        for cut in cuts:
            start = boundaries[-1]
            sums, grams, cross = backend.block_statistics(start - base, cut - start)
            sink(
                BlockStatistics(
                    chromosome=chromosome,
                    start=start,
                    stop=cut,
                    group_counts=layout.group_counts.copy(),
                    sums=sums,
                    grams=grams.astype(np.int32 if int32_grams else np.int64, copy=False),
                    cross_products=cross,
                )
            )
            boundaries.append(cut)

    current = _next_tile(filled)
    staging = 0
    if current is not None:
        backend.stage_tile(current[2][: current[1] - current[0]], staging)
    while current is not None:
        start, stop, host = current
        rows = stop - start
        while stop - base > plan.capacity_rows:
            if partitioner.committed > base:
                backend.move_rows(partitioner.committed - base, 0, start - partitioner.committed)
                base = partitioner.committed
            else:
                emit(partitioner.force_cut())
                forced += 1
        backend.load_staged_tile(staging, rows, start - base)
        backend.release_staged_tile(staging)
        free.put(host)
        upcoming = _next_tile(filled)
        if upcoming is not None:
            backend.stage_tile(upcoming[2][: upcoming[1] - upcoming[0]], 1 - staging)
        window_start = max(0, start - maximum_distance)
        row_weights, column_weights = backend.pair_weights(window_start - base, start - base, rows, maximum_distance)
        partitioner.add_pair_weights(window_start, row_weights, start, column_weights)
        emit(partitioner.advance(stop - maximum_distance))
        current = upcoming
        staging = 1 - staging
    backend.check_codes()
    emit(partitioner.finish())
    reader.join()
    seconds = time.monotonic() - started
    log(
        f"stage0 {chromosome}: {variant_count:,} variants in {len(boundaries) - 1:,} LD blocks "
        f"(cap {plan.block_cap}, {forced} forced cuts) in {seconds:.1f}s"
    )
    return ChromosomeSummary(
        chromosome=chromosome,
        boundaries=np.asarray(boundaries, dtype=np.int64),
        forced_cuts=forced,
        seconds=seconds,
    )

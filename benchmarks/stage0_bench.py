"""Stage 0 and the steps between it and the EB, old against new, on one GPU, with the numbers each pair must share.

    python -m benchmarks.stage0_bench [--samples N] [--columns P] [--density D] [--read-density R] [--cap C] [--device cuda|cpu] [--tiny]

A synthetic zstd store of two halves holds ``--columns`` fitted records (mosaic LD, as ``gpu_smoke_fixtures``'s, in
variant chunks) over two chromosomes, one in ``--density`` of its records as the whole-genome store holds its fitted
records (about 1 in 120 genome-wide, 1 in 59 on chr22 [real]); the rest are scrambled copies of them. A compact
store holds the fitted records alone, as a fit store of only those records would. Each phase prints its seconds,
old and new where the change has an old, and its bytes, so it scales to an arm by them (an arm's LD store is 150-169
GB, its band 82-98 GB [real]):

- Stage 0's tile reads alone, from a cold page cache, of ``--read-density`` records scattered at random (1 in 120 as
  the whole-genome arm's ~1.0e6 fitted columns of 126.7M records [real]), read one after another as Stage 0 read them
  (``SerialTileSource``) and at once (``genotype_statistics.DosageStoreTileSource``): the bytes must agree, and the
  milliseconds a candidate scale to an arm by its candidates and its samples (a chunk's bytes);
- Stage 0 on the scattered records with the old reads and the new, and on the compact ones (the LD files must agree
  byte for byte, the lag profile to a device's atomic-add rounding), and the time its new sums take on the device
  (``stage0 sums``);
- the summary band: the old copy of every block (``copied_band_store``) against the layout from Stage 0's lag
  profiles (``gram_space.summary_band``) and its completion; the two bands' files must agree byte for byte;
- the EB start: reading every Gram again (``moment_starts_read_all``) against Stage 0's sums (``moment_starts``);
- the resident codes' fill, the first full read of the fitted records, scattered and compact.

``copied_band_store`` and ``moment_starts_read_all`` are the versions the package ran before Stage 0 kept its sums,
and ``SerialTileSource`` the reads before they ran at once, kept here as the new ones' references (the tests check
each pair agrees). ``--tiny`` is the CPU run the tests make.
Timing only: the genotypes are random [own-sim]."""
from __future__ import annotations

import argparse
import filecmp
import hashlib
import json
import os
import tempfile
import time
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator

import numpy as np

from sv_pgs.dosage_store import (
    CODES_PER_DOSAGE, VARIANT_CLASS_LEGEND, CodeShardWriter, DosageStore, create_code_array, dosage_array_directory,
    sites_md5, statistic_column_directory, variant_column_directory, write_column, write_manifest, write_variant_ids,
)
from sv_pgs.genotype_statistics import DosageStoreTileSource
from sv_pgs.gram_space import _BAND_FILES, _BAND_INDEX, BandStore, read_block, summary_band

# Scenario sizes and generator rates belong to this benchmark, not to the inference package.
SAMPLES = 16384
COLUMNS = 32768
DENSITY = 8
READ_DENSITY = 120
BLOCK_CAP = 4096
CHROMOSOMES = ("chr21", "chr22")
CHUNK_VARIANTS = 512
SHARD_ROWS = 4096
INNER_ROWS = 64
DRAWS = 64
FOUNDERS = 10
HOTSPOT_SPACING = 60
SCRAMBLE = 2654435761
TINY = {"samples": 512, "columns": 1280, "density": 4, "read_density": 16, "cap": 256}


def mosaic_chunks(generator: np.random.Generator, samples: int, variants: int, chunk: int) -> Iterator[np.ndarray]:
    """``gpu_smoke_fixtures.mosaic_codes``' LD made a chunk of variants at a time, each haplotype's founder carried over a
    chunk's end so the LD runs on through it: variant-major 8-bit codes, a chunk's haplotypes in memory at once."""
    haplotypes = 2 * samples
    current = generator.integers(0, FOUNDERS, size=haplotypes)
    for start in range(0, variants, chunk):
        width = min(chunk, variants - start)
        frequency = generator.uniform(0.02, 0.5, size=width)
        founder_alleles = generator.random((FOUNDERS, width)) < frequency
        switch_rate = np.full(width, 0.004)
        switch_rate[generator.random(width) < 1.0 / HOTSPOT_SPACING] = 0.6
        segment = np.cumsum(generator.random((haplotypes, width)) < switch_rate, axis=1)
        drawn = generator.integers(0, FOUNDERS, size=(haplotypes, width))
        founder = np.where(segment == 0, current[:, None], np.take_along_axis(drawn, np.maximum(segment - 1, 0), axis=1))
        current = founder[:, -1].copy()
        alleles = founder_alleles[founder, np.arange(width)[None, :]]
        dosage = alleles[0::2].astype(np.float32) + alleles[1::2]
        noisy = np.clip(dosage + generator.normal(0.0, 0.08, size=dosage.shape).astype(np.float32), 0.0, 2.0)
        codes = np.rint(noisy * 127.0).astype(np.uint8).T.copy()
        codes[generator.random(width) < 0.01] = 3
        yield codes


def write_store(root: Path, rows_of: dict[str, Callable[[int, int], np.ndarray]], record_counts: dict[str, int], samples: int) -> None:
    """A two-half zstd dosage store, as ``gpu_smoke_fixtures._write_store`` writes one, from each chromosome's rows
    ``rows_of[chromosome](start, stop)`` (variant-major codes over every sample), a shard at a time."""
    chromosomes = list(record_counts)
    digests = []
    for chromosome in chromosomes:
        count = record_counts[chromosome]
        positions = np.arange(1, count + 1, dtype=np.int64) * 100
        lengths = np.ones(count, dtype=np.int32)
        columns = {
            "pos": (positions, {}),
            "ref_len": (lengths, {}),
            "alt_len": (lengths, {}),
            "cm": (positions / 1e6, {}),
            "variant_class": (np.zeros(count, dtype=np.uint8), {"legend": VARIANT_CLASS_LEGEND}),
            "codes_per_unit": (np.full(count, CODES_PER_DOSAGE, dtype=np.uint8), {}),
            "value_origin": (np.zeros(count, dtype=np.int16), {}),
            "group_first": (np.arange(count, dtype=np.int64), {}),
            "class": (np.zeros(count, dtype=np.uint8), {"legend": ["SNV", "INDEL", "SV"]}),
            "has_pl": (np.zeros(count, dtype=bool), {}),
            "panel_af": (np.full(count, 0.25, dtype=np.float32), {}),
        }
        for name, (values, attributes) in columns.items():
            write_column(variant_column_directory(root, chromosome, name), values, attributes)
        write_variant_ids(root, chromosome, [f"{chromosome}-{position}-allele0-1" for position in positions])
        digests.append(sites_md5(positions, lengths, lengths))
    halves = [samples // 2, samples - samples // 2]
    write_manifest(root, chromosomes=chromosomes, record_counts=[record_counts[name] for name in chromosomes], half_sample_counts=halves, chromosome_sites_md5=digests)
    for chromosome in chromosomes:
        count = record_counts[chromosome]
        layouts = [
            create_code_array(dosage_array_directory(root, half, chromosome), count, halves[half], codec="zstd", shard_rows=SHARD_ROWS, inner_rows=INNER_ROWS)
            for half in range(2)
        ]
        sums = [np.zeros(count, dtype=np.uint64) for _ in range(2)]
        squares = [np.zeros(count, dtype=np.uint64) for _ in range(2)]
        for shard in range(layouts[0].shard_count):
            start, stop = shard * SHARD_ROWS, min((shard + 1) * SHARD_ROWS, count)
            codes = rows_of[chromosome](start, stop)
            for half, (low, high) in enumerate(((0, halves[0]), (halves[0], samples))):
                piece = np.ascontiguousarray(codes[:, low:high])
                with CodeShardWriter(dosage_array_directory(root, half, chromosome), layouts[half], shard) as writer:
                    writer.write_rows(piece)
                wide = piece.astype(np.uint64)
                sums[half][start:stop] = wide.sum(axis=1)
                squares[half][start:stop] = (wide * wide).sum(axis=1)
        for half in range(2):
            write_column(statistic_column_directory(root, half, chromosome, "sum_code"), sums[half])
            write_column(statistic_column_directory(root, half, chromosome, "sum_code2"), squares[half])


class SerialTileSource(DosageStoreTileSource):
    """Stage 0's tile source as it read before its reads ran at once: each contiguous run of candidates one
    ``read_codes``, one after another."""

    def read_rows(self, chromosome: str, start: int, stop: int, out: np.ndarray) -> None:
        rows = self._rows[chromosome][start:stop]
        breaks = np.flatnonzero(np.diff(rows) > 1) + 1
        for run_start, run_stop in zip(np.concatenate(([0], breaks)).tolist(), np.concatenate((breaks, [rows.shape[0]])).tolist()):
            self._store.read_codes(int(rows[run_start]), int(rows[run_stop - 1]) + 1, None, out=out[run_start:run_stop])


def evict(root: Path) -> bool:
    """Drop a store's files from the page cache (written back first), so the reads that follow come from the disk as
    a fit's first reads do; False where the platform cannot (the reads are then warm)."""
    if not hasattr(os, "posix_fadvise"):
        return False
    os.sync()
    for path in root.rglob("*"):
        if path.is_file():
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(descriptor)
    return True


def read_tiles(source: DosageStoreTileSource, tile_rows: int) -> str:
    """Every tile of every chromosome read once, in order, as Stage 0's reader reads them; the digest of their bytes."""
    digest = hashlib.sha256()
    for chromosome in source.chromosomes():
        count = source.variant_count(chromosome)
        tile = np.empty((min(tile_rows, count), source.sample_count), dtype=np.uint8)
        for start in range(0, count, tile_rows):
            stop = min(start + tile_rows, count)
            source.read_rows(chromosome, start, stop, tile[: stop - start])
            digest.update(tile[: stop - start].tobytes())
    return digest.hexdigest()


def copied_band_store(
    ld: Any, sample_count: int, directory: Path, working_bytes: int, level: float, array_module: Any = np, null_scale: float = 1.0,
) -> BandStore:
    """The summary band as the package built it before Stage 0 kept its lag profiles: each Stage 0 block's Gram read,
    its LD extent measured on it (``marginal_variances.ld_extent``) and every part copied out of it at once. Written
    whole, ``gram_space.summary_band`` writes the same files, byte for byte."""
    from sv_pgs.marginal_variances import BlockGrams, ld_extent

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    widths: list[int] = []
    within_offsets: list[int] = []
    cross_offsets: list[int] = []
    extents: list[dict] = []
    positions = {"within": 0, "cross": 0}
    handles = {name: open(directory / file_name, "wb") for name, file_name in _BAND_FILES.items()}

    def write(name: str, values: np.ndarray) -> int:
        offset = positions[name]
        handles[name].write(np.ascontiguousarray(values, dtype=np.float32))
        positions[name] += int(values.size)
        return offset

    try:
        previous: slice | None = None
        for block in range(ld.block_count):
            width = int(ld.block_width(block))
            if width == 0:
                previous = None
                continue
            within = read_block(ld, "within", block)
            single = BlockGrams(blocks=(np.arange(width, dtype=np.int64),), within=(within,), next_cross=())
            extent = int(ld_extent(single, int(sample_count), int(working_bytes), level, array_module, null_scale)) if width > 1 else 1
            parts = max(1, -(-width // extent))
            cuts = np.linspace(0, width, parts + 1).round().astype(np.int64)
            spans = [slice(int(cuts[part]), int(cuts[part + 1])) for part in range(parts)]
            extents.append({"block": block, "width": width, "extent": extent, "parts": parts})
            first_cross = -1
            if previous is not None and ld.has_adjacent(block):
                adjacent = read_block(ld, "cross", block - 1)
                first_cross = write("cross", adjacent[previous, spans[0]])
                del adjacent
            for position, span in enumerate(spans):
                widths.append(span.stop - span.start)
                within_offsets.append(write("within", within[span, span]))
                cross_offsets.append(first_cross if position == 0 else write("cross", within[spans[position - 1], span]))
            previous = spans[-1]
            del within
    finally:
        for handle in handles.values():
            handle.close()
    stored = sum(positions.values()) * np.dtype(np.float32).itemsize
    index = {"widths": widths, "within_offsets": within_offsets, "cross_offsets": cross_offsets, "extents": extents, "stored_bytes": stored}
    (directory / _BAND_INDEX).write_text(json.dumps(index), encoding="utf-8")
    return BandStore(directory)


def moment_starts_read_all(statistics: Any, prior: Any) -> list:
    """``full_data_fit.moment_starts`` as the package ran it before Stage 0 kept its Grams' sums: every within-block
    and adjacent Gram read again and squared on the fit's device (``scale_mixture_ep.device_scope``)."""
    from sv_pgs.dual_solve import _host
    from sv_pgs.scale_mixture_ep import _DEVICE, moment_start
    from sv_pgs.tie_members import TieGroups

    xp = _DEVICE.get()
    ld = statistics.ld
    ties = TieGroups.from_tie_map(statistics.tie_map)
    weights = np.zeros(ties.group_count)
    np.add.at(weights, ties.group, np.exp(prior.log_variance_offset))
    covariate_count = statistics.covariate_rank
    fitted = statistics.covariate_target.T @ statistics.covariate_gram_pseudo_inverse @ statistics.covariate_target
    target_square = np.diag(statistics.target_gram) - np.diag(fitted)
    score_square = np.zeros(target_square.shape[0])
    column_square = np.zeros(ties.group_count)
    gram_trace = 0.0
    gram_square = 0.0
    weighted_diagonal = 0.0
    previous_columns = None
    for block_index in range(ld.block_count):
        block = ld.block(block_index)
        columns = np.asarray(block.reduced_columns, dtype=np.int64)
        gram = xp.asarray(read_block(ld, "within", block_index)).astype(xp.float64)
        score_square += np.sum(np.square(np.asarray(block.projected_score, dtype=np.float64)), axis=0)
        diagonal = _host(xp.diag(gram))
        gram_trace += float(diagonal.sum())
        weighted_diagonal += float(weights[columns] @ diagonal)
        gram *= gram
        column_square[columns] += _host(gram.sum(axis=0))
        gram_square += float(_host(gram.sum()))
        del gram
        if ld.has_adjacent(block_index):
            cross_squares = xp.asarray(read_block(ld, "cross", block_index - 1)).astype(xp.float64)
            cross_squares *= cross_squares
            column_square[previous_columns] += _host(cross_squares.sum(axis=1))
            column_square[columns] += _host(cross_squares.sum(axis=0))
            gram_square += 2.0 * float(_host(cross_squares.sum()))
            del cross_squares
        previous_columns = columns
    return [
        moment_start(
            target_square=float(target_square[target]), residual_dimension=float(statistics.sample_count - covariate_count),
            score_square=float(score_square[target]), gram_trace=gram_trace, weighted_diagonal=weighted_diagonal,
            weighted_square=float(weights @ column_square), gram_square=gram_square,
        )
        for target in range(target_square.shape[0])
    ]


def stage0_sums_seconds(ld: Any, xp: Any, synchronize: Callable[[], None]) -> float:
    """The device seconds of Stage 0's new sums over every stored block (``genotype_statistics._project_block``'s lag
    profile and squared Gram and coupling), the Grams already on the device as Stage 0 holds them."""
    from sv_pgs.lag_profiles import LAG_PROFILE_ROWS, lag_excess_rows, lag_profile

    seconds = 0.0
    for block in range(ld.block_count):
        gram = xp.asarray(ld.read_gram(block))
        adjacent = xp.asarray(ld.read_adjacent(block)) if ld.has_adjacent(block) else None
        synchronize()
        started = time.perf_counter()
        width = int(gram.shape[0])
        diagonal = xp.diagonal(gram).astype(xp.float64)
        lag_profile((lag_excess_rows(gram, diagonal, first, min(first + LAG_PROFILE_ROWS, width), ld.sample_count, xp)
                     for first in range(0, width, LAG_PROFILE_ROWS)), width, xp)
        scratch = xp.empty(max(width * width, 0 if adjacent is None else int(adjacent.size)), dtype=xp.float64)
        square = scratch[: width * width].reshape(width, width)
        square[...] = gram
        square *= square
        square.sum(axis=0)
        float(square.sum())
        if adjacent is not None:
            coupling = scratch[: adjacent.size].reshape(adjacent.shape)
            coupling[...] = adjacent
            coupling *= coupling
            coupling.sum(axis=1)
            coupling.sum(axis=0)
            float(coupling.sum())
        synchronize()
        seconds += time.perf_counter() - started
        del gram, adjacent, scratch, square
    return seconds


def same_files(first: Path, second: Path, names: list[str]) -> list[str]:
    """The files of ``names`` that differ between the two directories."""
    return [name for name in names if not filecmp.cmp(first / name, second / name, shallow=False)]


def main(arguments: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--samples", type=int, default=SAMPLES)
    parser.add_argument("--columns", type=int, default=COLUMNS)
    parser.add_argument("--density", type=int, default=DENSITY)
    parser.add_argument("--read-density", type=int, default=READ_DENSITY)
    parser.add_argument("--cap", type=int, default=BLOCK_CAP)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--tiny", action="store_true", help="the tests' CPU-sized problem")
    parser.add_argument("--out", type=Path, default=None, help="working directory (a temporary one by default)")
    options = parser.parse_args(arguments)
    if options.tiny:
        for name, value in TINY.items():
            setattr(options, name, value)
    if options.out is None:
        with tempfile.TemporaryDirectory(prefix="stage0-bench-") as directory:
            return run(options, Path(directory))
    options.out.mkdir(parents=True, exist_ok=True)
    return run(options, options.out)


def run(options: argparse.Namespace, root: Path) -> dict:
    from sv_pgs.compute_budget import ComputeBudget, detect_compute_budget
    from sv_pgs.config import ModelConfig
    from sv_pgs.dual_solve import StreamedDualSource
    from sv_pgs.full_data_fit import moment_starts
    from sv_pgs.genotype_statistics import _LD_ARRAYS, _LD_INDEX, compute_genotype_statistics, plan_genotype_pass
    from sv_pgs.marginal_variances import certificate_level
    from sv_pgs.memory_broker import memory_scope
    from sv_pgs.scale_mixture_ep import device_scope
    from sv_pgs.store_block_source import StoreGenotypeBlockSource

    if options.device == "cuda":
        import cupy as xp

        budget = detect_compute_budget()
        if budget.device_kind != "cuda":
            raise SystemExit("no CUDA device: run with --device cpu")
        budget = replace(budget, device_ids=budget.device_ids[:1], device_names=budget.device_names[:1],
                         device_bytes=budget.device_bytes[:1], device_compute_capabilities=budget.device_compute_capabilities[:1])

        def synchronize() -> None:
            xp.cuda.Device(budget.device_ids[0]).synchronize()
    else:
        xp = np
        budget = ComputeBudget(device_kind="cpu", device_ids=(), device_names=(), device_bytes=(), device_compute_capabilities=(),
                               host_bytes=1 << 31, cpu_threads=2)

        def synchronize() -> None:
            return None

    report: dict[str, Any] = {"samples": options.samples, "columns": options.columns, "density": options.density,
                              "read_density": options.read_density, "cap": options.cap, "device": options.device, "phases": {}, "checks": {}}
    phases, checks = report["phases"], report["checks"]

    def timed(name: str, work: Callable[[], Any]) -> Any:
        synchronize()
        started = time.perf_counter()
        result = work()
        synchronize()
        phases[name] = round(time.perf_counter() - started, 3)
        print(f"PHASE {name}: {phases[name]:.2f} s", flush=True)
        return result

    generator = np.random.default_rng(20261020)
    per_chromosome = options.columns // len(CHROMOSOMES)
    fitted = {}

    def generate() -> None:
        for chromosome in CHROMOSOMES:
            fitted[chromosome] = np.concatenate(list(mosaic_chunks(generator, options.samples, per_chromosome, CHUNK_VARIANTS)))

    timed("generate genotypes", generate)
    full_counts = {chromosome: per_chromosome * options.density for chromosome in CHROMOSOMES}

    def scattered_rows(chromosome: str) -> Callable[[int, int], np.ndarray]:
        def rows(start: int, stop: int) -> np.ndarray:
            records = np.arange(start, stop, dtype=np.int64)
            source = np.where(records % options.density == 0, records // options.density, (records * SCRAMBLE) % per_chromosome)
            return fitted[chromosome][source]
        return rows

    timed("write scattered store", lambda: write_store(root / "scattered", {name: scattered_rows(name) for name in CHROMOSOMES}, full_counts, options.samples))
    timed("write compact store", lambda: write_store(
        root / "compact", {name: (lambda start, stop, name=name: fitted[name][start:stop]) for name in CHROMOSOMES},
        {name: per_chromosome for name in CHROMOSOMES}, options.samples,
    ))
    fitted.clear()
    stores = {"scattered": DosageStore.open(root / "scattered"), "compact": DosageStore.open(root / "compact")}
    candidates = {
        "scattered": np.concatenate([offset * options.density * per_chromosome + np.arange(0, full_counts[name], options.density) for offset, name in enumerate(CHROMOSOMES)]),
        "compact": np.arange(options.columns, dtype=np.int64),
    }
    training = np.arange(options.samples, dtype=np.int64)
    covariates = np.column_stack([np.ones(options.samples), generator.standard_normal((options.samples, 2))])
    targets = generator.standard_normal((options.samples, 1))

    # Stage 0's reads alone at the whole genome's density, old and new, each from a cold page cache.
    records = int(stores["scattered"].n_variants)
    sparse = np.sort(generator.choice(records, size=max(1, records // options.read_density), replace=False))
    report["read candidates"] = int(sparse.shape[0])
    report["reads cold"] = evict(root / "scattered")
    tile_rows = plan_genotype_pass(options.cap).tile_rows
    old_digest = timed("tile reads old (one after another)", lambda: read_tiles(SerialTileSource(stores["scattered"], sparse), tile_rows))
    evict(root / "scattered")
    new_digest = timed("tile reads new (at once)", lambda: read_tiles(DosageStoreTileSource(stores["scattered"], sparse), tile_rows))
    checks["tile reads equal"] = old_digest == new_digest
    for name in ("tile reads old (one after another)", "tile reads new (at once)"):
        report[f"{name} ms a candidate"] = round(1e3 * phases[name] / sparse.shape[0], 4)

    statistics = {}
    sources = {
        "scattered, old reads": lambda: SerialTileSource(stores["scattered"], candidates["scattered"]),
        "scattered": lambda: DosageStoreTileSource(stores["scattered"], candidates["scattered"]),
        "compact": lambda: DosageStoreTileSource(stores["compact"], candidates["compact"]),
    }
    with memory_scope(budget):
        for name, source in sources.items():
            evict(root / name.split(",")[0])
            statistics[name] = timed(f"stage0 {name}", lambda name=name, source=source: compute_genotype_statistics(
                source(), training, covariates, targets, ModelConfig(), budget, options.cap, root / f"ld_{name.replace(', ', '_').replace(' ', '_')}",
            ))
        ld = statistics["compact"].ld
        report["ld_store_gb"] = round(ld.stored_bytes / 1e9, 3)
        report["ld_blocks"] = int(ld.block_count)
        exact = [file_name for name, (file_name, _dtype) in _LD_ARRAYS.items() if name != "lag_excess"]
        checks["stage0 old and new reads give the same LD files"] = not same_files(root / "ld_scattered_old_reads", root / "ld_scattered", exact)
        checks["stage0 LD files equal"] = not same_files(root / "ld_scattered", root / "ld_compact", exact)
        scattered_index = json.loads((root / "ld_scattered" / _LD_INDEX).read_text())
        checks["stage0 LD index equal"] = scattered_index == json.loads((root / "ld_compact" / _LD_INDEX).read_text())
        profiles = [statistics[layout].ld.lag_profiles()[0] for layout in ("scattered", "compact")]
        checks["stage0 lag profiles equal"] = bool(np.allclose(profiles[0], profiles[1], rtol=1e-12, atol=1e-9))
        phases["stage0 sums (inside stage0)"] = round(stage0_sums_seconds(ld, xp, synchronize), 3)
        print(f"PHASE stage0 sums (inside stage0): {phases['stage0 sums (inside stage0)']:.2f} s", flush=True)

        level = certificate_level(DRAWS)
        sample_count = statistics["compact"].sample_count
        timed("band old (copy every block)", lambda: copied_band_store(ld, sample_count, root / "band_old", budget.working_bytes // 2, level, xp))
        band = timed("band new (layout, first parts)", lambda: summary_band(ld, sample_count, root / "band_new", level))
        report["band_gb"] = round(band.stored_bytes / 1e9, 3)
        report["band_written_for_pricing_gb"] = round(band.written_bytes / 1e9, 3)
        timed("band new complete (band route only)", band.complete)
        checks["band files equal"] = not same_files(root / "band_old", root / "band_new", list(_BAND_FILES.values()))
        old_index, new_index = (json.loads((root / name / _BAND_INDEX).read_text()) for name in ("band_old", "band_new"))
        checks["band index equal"] = all(old_index[key] == new_index[key] for key in old_index)

        prior = SimpleNamespace(log_variance_offset=generator.normal(scale=0.5, size=int(np.asarray(statistics["compact"].active_rows).shape[0])))

        def read_all() -> list:
            with device_scope(xp):
                return moment_starts_read_all(statistics["compact"], prior)

        old_starts = timed("moment starts old (read every Gram)", read_all)
        new_starts = timed("moment starts new (Stage 0's sums)", lambda: moment_starts(statistics["compact"], prior))
        differences = [abs(getattr(old, field.name) - getattr(new, field.name)) / max(abs(getattr(old, field.name)), 1e-300)
                       for old, new in zip(old_starts, new_starts) for field in fields(old)]
        report["moment starts largest relative difference"] = float(max(differences))
        checks["moment starts equal"] = float(max(differences)) <= 1e-12

        for layout in ("scattered", "compact"):
            source = StreamedDualSource(StoreGenotypeBlockSource.from_statistics(
                stores[layout], statistics[layout], budget, budget.working_bytes // 2, sample_indices=training,
            ))
            timed(f"fill {layout} (first read)", lambda source=source: sum(1 for _ in source.blocks()))
            timed(f"read {layout} (held: {bool(source.holds_codes())})", lambda source=source: sum(1 for _ in source.blocks()))
            del source
    for store in stores.values():
        store.close()
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}", flush=True)
    print(json.dumps(report), flush=True)
    return report


if __name__ == "__main__":
    raise SystemExit(0 if all(main()["checks"].values()) else 1)

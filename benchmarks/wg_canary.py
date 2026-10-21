"""A synthetic column-scale canary through the workspace pipeline, without a remote input.

    python -m benchmarks.wg_canary --out canary --device cpu
    python -m benchmarks.wg_canary --out canary-gpu --device cuda --gpus 2

The default has 1,060,000 fitted records, 22 chromosomes and 512 invented samples. Founder mosaics make local LD,
with independent blocks, within-block recombination and dosage noise; the noise prevents exact ties from collapsing
millions of columns to a few founders. Eight structural types occupy one record in twenty. The other records are
synthetic stand-ins for the selected SNV panel, not public or participant genotypes. --records can exceed --columns:
the extra rows have zero reported reliability and exercise the store and filtering without entering the fit. Use
--records 126700000 explicitly for the live record axis; the default does NOT claim to cover that axis. That option
also writes and decodes all those records and is substantially more expensive.

The converter, measurement model, family split, full-data fitter, EB stopping criteria, lattice refinement, mode
mixture, artifact writer, scoring, effects and report are production code. Two arm processes share read-only upstream
checkpoints, fit separately, and a fresh process merges their model directories through the pipeline's own validation.
The local phenotype step replaces the remote phenotype acquisition; the plain-VCF site reader is extended with the
popped annotations n_paths, n_paths_total and cx. Their values are simulated, but their names, types and downstream
annotation design are the real ones. This does not test the popped GP decoder or a remote restore.

Memory is deliberately not scaled by samples alone. The default plan reserves 24 GiB on the host and 8 GiB per GPU
for column work, then scales the rest of a reference 160 GiB host / 40 GiB GPU by samples / 50,000. At 512 that is
25.39 GiB / 8.33 GiB. These are scenario assumptions, not measured live working sets. --host-gib and --device-gib
allow a measured plan to replace them. The runner fails if its selected budget exceeds detected capacity, if the
sample full-data route is not used, or if a CUDA fit never fills resident codes. It records cache admissions and
evictions; matching every cache decision requires comparing these records with the live run. A single scalar budget
cannot preserve both n*p and p*k allocations when only n changes.

Memory checks run at EB call boundaries after cyclic collection. A rolling window detects sustained growth in RSS,
Python traced live bytes, CUDA live bytes, ledger bytes and ledger lease counts; one-time high-water allocations are
allowed. Every EB outer loop must end with no curvature warm start leased, which a tiny run's short loops check where
their windows never fill. The configurable byte tolerance is printed with the results, not silently treated as proof of no leak.
No finite run can detect every leak, and the sampling misses temporaries within a call; kernel high-water RSS and the
broker's peaks cover peaks separately. The deadline is a failure, never a successful early stop of EB.

Planning arithmetic, not a benchmark: 512 / 50,000 = .01024; the selected int8 matrix is .543 GB. Pure sample-linear
Stage 0 would take 65--73 min * .01024 = .67--.75 min per arm, but the 9--14 min band construction and column/lattice
work need not shrink. At 20 EB rounds and 15--60 s of column work per round, two arms need 10--40 min for EB alone;
allow 10--20 min for generation/store, 18--28 min for both bands, 5--15 min for starts/mixtures/postfit and 5--10 min
for environment setup: roughly 50--115 min on A100, with a 3 h GPU deadline. CPU speed is unmeasured: at 3--5 times
the fit work, plan about 2--8 h (an 8 h CPU deadline), not a promised cheap smoke test. At illustrative, user-supplied hourly rates .50
(CPU Spot) and 1.50 (A100 Spot), those ranges are $1--4 and $1.25--2.88 plus disk. No current cloud price is asserted.
The profile supplies measured phase times for replacing these assumptions before scaling to all eleven traits.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import gzip
import json
import logging
import math
import os
import re
import resource
import shutil
import signal
import socket
import subprocess
import sys
import time
import tracemalloc
from collections import Counter, deque
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np

from benchmarks import profile_log, round_timeline
from sv_pgs import phase_timer, scale_mixture_ep
from sv_pgs import workspace_pipeline as pipeline
from sv_pgs.compute_budget import ComputeBudget, detect_compute_budget
from sv_pgs.model_acceptance import acceptance
from sv_pgs.progress import log

GIB = 1 << 30
MIB = 1 << 20
SV_TYPES = ("DEL", "INS", "DUP", "INS:ME:ALU", "BND", "STR", "CPX", "INV")
ANNOTATIONS = {"n_paths", "n_paths_total", "cx", "tr_locus", "sv_length", "in_gene", "in_exon", "log_tss_distance"}
SYNTHETIC_WORKSPACE = {"WORKSPACE_CDR": "synthetic_canary.cdr", "GOOGLE_PROJECT": "synthetic-canary"}
"""The workspace names the phenotype step's key reads (``instrumentation``), never a real project or CDR."""
# GRCh38 autosome lengths, used only to spread the synthetic columns over a realistic chromosome axis.
CHROMOSOME_MB = (249, 242, 198, 190, 182, 171, 159, 145, 138, 134, 135, 133, 114, 107, 102, 90, 83, 80, 59, 64, 47, 51)


@dataclasses.dataclass(frozen=True)
class Scenario:
    columns: int = 1_060_000
    samples: int = 512
    chromosomes: int = 22
    block_columns: int = 256
    seed: int = 20261020
    records: int | None = None
    sv_every: int = 20
    traits: tuple[str, ...] = ("total_bilirubin",)
    diseases: tuple[str, ...] = ()

    def __post_init__(self):
        if not 1 <= self.chromosomes <= len(CHROMOSOME_MB):
            raise ValueError("chromosomes must be in 1..22")
        if self.columns < self.chromosomes or self.samples < 128 or self.samples % 2:
            raise ValueError("at least one column per chromosome and an even synthetic cohort of at least 128 are required")
        if self.block_columns < 2 or self.sv_every < 2 or self.seed < 0:
            raise ValueError("block_columns and sv_every must exceed one; seed must be nonnegative")
        if self.records is not None and self.records < self.columns:
            raise ValueError("records cannot be smaller than the fitted column target")
        names = self.diseases + self.traits
        if not names or len(set(names)) != len(names):
            raise ValueError("name each phenotype once")

    def counts(self, total: int) -> list[int]:
        lengths = np.array(CHROMOSOME_MB[:self.chromosomes], dtype=float)
        # One guaranteed row per chromosome, then distribute the rest by chromosome length.
        counts = np.floor((total - self.chromosomes) * lengths / lengths.sum()).astype(int) + 1
        counts[:total - int(counts.sum())] += 1
        return counts.tolist()


def mosaic_block(scenario: Scenario, chromosome: int, block: int, width: int) -> np.ndarray:
    """One block's imputed dosages [records, samples], independent of writer chunking and of any earlier RNG calls.

    Two haplotypes copy balanced founder alleles, changing founder rarely inside the block. An independent block
    starts with new founders. A continuous dosage perturbation supplies imputation-like variation, including on a
    row whose drawn hard calls happen to be constant. The .03 dosage floor keeps even those rows above a
    one-percent frequency threshold in every training split, and the noise keeps them polymorphic after quantization.
    """
    rng = np.random.default_rng(np.random.SeedSequence(scenario.seed, spawn_key=(chromosome, block)))
    founders = 16
    frequency = rng.uniform(.1, .5, width)
    alleles = rng.random((width, founders)) < frequency[:, None]
    alleles[:, 0], alleles[:, 1] = False, True
    copies = rng.integers(founders, size=2 * scenario.samples)
    dosage = np.empty((width, scenario.samples), dtype=np.float64)
    for row in range(width):
        change = rng.random(copies.size) < .004
        copies[change] = rng.integers(founders, size=int(change.sum()))
        haplotypes = alleles[row, copies].astype(float)
        dosage[row] = haplotypes[::2] + haplotypes[1::2]
    dosage += rng.normal(0, .04, dosage.shape)
    return np.round(np.clip(dosage, .03, 1.97), 3)


def generate(root: Path, scenario: Scenario) -> pipeline.WorkspaceConfig:
    """Write the bounded-memory local inputs, then return the real workspace config that consumes them.

    No whole-genome genotype array is allocated: one LD block and the sample-length phenotype signal are live. Gzip
    headers have a fixed mtime, and chromosome/block RNG streams are independent, so identical scenarios make identical
    input bytes in different directories. Source files contain invented sample labels; no source table is published.
    """
    root.mkdir(parents=True, exist_ok=False)
    chromosomes = tuple(f"chr{i + 1}" for i in range(scenario.chromosomes))
    fitted_counts = scenario.counts(scenario.columns)
    extra_total = (scenario.records or scenario.columns) - scenario.columns
    extras = np.full(scenario.chromosomes, extra_total // scenario.chromosomes, dtype=int)
    extras[:extra_total % scenario.chromosomes] += 1
    rng = np.random.default_rng(np.random.SeedSequence(scenario.seed, spawn_key=(len(CHROMOSOME_MB),)))
    names = [f"synthetic_{i:06d}" for i in range(scenario.samples)]
    persons = [str(1_000_000 + i) for i in range(scenario.samples)]
    signal = np.zeros(scenario.samples)
    dosage_text = np.array([f"{value / 1000:.3f}" for value in range(2001)], dtype=object)
    with (root / "genes.gtf").open("w") as genes, (root / "repeats.bed").open("w") as repeats:
        for index, (chromosome, fitted, extra) in enumerate(zip(chromosomes, fitted_counts, extras)):
            count = fitted + int(extra)
            spacing = max(1, CHROMOSOME_MB[index] * 1_000_000 // (count + 1))
            for start in range(1, count * spacing + 1, max(spacing, scenario.block_columns * spacing)):
                end = start + scenario.block_columns * spacing // 2
                for feature, last in (("gene", end), ("transcript", end), ("exon", start + spacing)):
                    genes.write(f'{chromosome}\tcanary\t{feature}\t{start}\t{last}\t.\t+\t.\tgene_id "synthetic";\n')
                repeats.write(f"{chromosome}\t{start + spacing}\t{start + spacing * 2}\n")
            with (root / f"{chromosome}.vcf.gz").open("wb") as raw:
                with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0, compresslevel=1) as handle:
                    header = [
                        "##fileformat=VCFv4.2", f"##contig=<ID={chromosome},length={CHROMOSOME_MB[index] * 1_000_000}>",
                        '##INFO=<ID=CM,Number=1,Type=Float,Description="synthetic genetic map">',
                        '##INFO=<ID=R2,Number=1,Type=Float,Description="synthetic imputation quality">',
                        '##INFO=<ID=SVTYPE,Number=1,Type=String,Description="synthetic type">',
                        '##INFO=<ID=SVLEN,Number=1,Type=Integer,Description="synthetic length">',
                        '##FORMAT=<ID=DS,Number=1,Type=Float,Description="synthetic dosage">',
                        "\t".join(["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", *names]),
                    ]
                    handle.write(("\n".join(header) + "\n").encode())
                    for first in range(0, count, scenario.block_columns):
                        width = min(scenario.block_columns, count - first)
                        dosages = mosaic_block(scenario, index, first // scenario.block_columns, width)
                        if first == 0:
                            signal += dosages[:min(width, 4)].sum(axis=0)
                        for local, dosage in enumerate(dosages):
                            row = first + local
                            position = (row + 1) * spacing
                            alt, info = "G", f"CM={position / 1_000_000:.6f};R2={.95 if row < fitted else 0}"
                            if row % scenario.sv_every == scenario.sv_every - 1:
                                kind = SV_TYPES[(row // scenario.sv_every) % len(SV_TYPES)]
                                length = int(50 * 10 ** ((row // scenario.sv_every % 41) / 10))
                                alt = f"<{kind}>"
                                info += f";SVTYPE={kind};SVLEN={length}"
                            fields = [chromosome, str(position), f"canary_{index}_{row}", "A", alt, ".", "PASS", info, "DS"]
                            handle.write(("\t".join(fields + dosage_text[np.rint(dosage * 1000).astype(np.int64)].tolist()) + "\n").encode())
    (root / "crosswalk.tsv").write_text("research_id\tsequencing_id\n" + "".join(f"{person}\t{name}\n" for person, name in zip(persons, names)))
    pcs = rng.normal(size=(scenario.samples, 3))
    (root / "ancestry.tsv").write_text("research_id\tancestry_pred\tpca_features\n" + "".join(
        f"{person}\teur\t{json.dumps(values.tolist())}\n" for person, values in zip(persons, pcs)
    ))
    (root / "relatedness.tsv").write_text("i.s\tj.s\tkin\n" + "".join(
        f"{persons[i]}\t{persons[i + 1]}\t0.25\n" for i in range(0, scenario.samples, 2)
    ))
    signal = (signal - signal.mean()) / signal.std()
    for trait in scenario.diseases + scenario.traits:
        target = .5 * signal + np.sqrt(.75) * rng.normal(size=scenario.samples) + .2 * pcs[:, 0]
        if trait in scenario.diseases:
            target = (target > np.median(target)).astype(float)
        (root / f"{trait}.tsv").write_text("person_id\ttarget\n" + "".join(f"{person}\t{value:.17g}\n" for person, value in zip(persons, target)))
        (root / f"{trait}.tsv.metadata.json").write_text(json.dumps({"covariate_columns": []}))
    raw_config = dict(
        run_directory=str(root.parent / "prepare"), marker_directory=None, chromosomes=chromosomes,
        imputed_halves=[dict(label="synthetic", batches=["one"], batch_path=str(root / "{chromosome}.vcf.gz"))],
        truth_calls=None, imputation_reference_ids=None, strata_directory=None, bubble_split=None, genetic_map=None,
        tandem_repeats=str(root / "repeats.bed"), gene_annotation=str(root / "genes.gtf"), crosswalk=str(root / "crosswalk.tsv"),
        crosswalk_columns=["research_id", "sequencing_id"], ancestry=str(root / "ancestry.tsv"), relatedness=str(root / "relatedness.tsv"),
        relatedness_columns=["i.s", "j.s", "kin"], diseases=scenario.diseases, traits=scenario.traits, disease_likelihood="bernoulli", fold_count=5, cross_fit=False,
        seed=scenario.seed, codec="zstd", export=[], direct_sv=None, genome_build="synthetic-GRCh38-layout",
        fit_minimum_minor_allele_frequency=0.0, fit_minimum_r2=0.1, annotated_prior=True,
    )
    config = pipeline.WorkspaceConfig.from_mapping(raw_config)
    (root.parent / "config.json").write_text(json.dumps(raw_config, indent=2))
    return config


def local_phenotypes(run, directory):
    """Supply invented targets at the acquisition boundary; cohort construction and every later step stay real."""
    for trait in (*run.config.diseases, *run.config.traits):
        for suffix in (".tsv", ".tsv.metadata.json"):
            shutil.copyfile(run.config.crosswalk.parent / (trait + suffix), directory / (trait + suffix))
    return {"traits": list(run.config.traits), "diseases": list(run.config.diseases), "source": "synthetic local targets"}


def no_remote(*args, **kwargs):
    raise RuntimeError("the synthetic canary must never open a network connection or a remote phenotype client")


def no_fit(*args, **kwargs):
    raise RuntimeError("the merged arms must finalize without another fit")


def planned_budget(detected: ComputeBudget, samples: int, host_gib: float | None, device_gib: float | None) -> ComputeBudget:
    """Keep the column reserve, scale only the sample share, and refuse a plan the machine cannot accommodate."""
    host = int((24 + (160 - 24) * samples / 50_000 if host_gib is None else host_gib) * GIB)
    device = int((8 + (40 - 8) * samples / 50_000 if device_gib is None else device_gib) * GIB)
    if host <= 0 or device <= 0 or host > detected.host_bytes or any(device > free for free in detected.device_bytes):
        raise MemoryError("the canary budget exceeds detected capacity or is nonpositive; choose an explicit smaller budget")
    return dataclasses.replace(detected, host_bytes=host, device_bytes=tuple(device for _ in detected.device_ids))


class CanaryFailure(RuntimeError):
    """An instrumentation failure is not a numerical refusal the fitter may catch and turn into a null model."""


ACCEPTED_STEP = re.compile(
    r"inner step accepted: length (\S+) \(radius (\S+)\), decrement (\S+) (definite|indefinite), gain (\S+), "
    r"state error (\S+) \(decrement (\S+)\)"
)
"""scale_mixture_ep.fit_hyperparameters' line for an accepted inner step of EB."""


def accepted_step_problem(message: str):
    """What is wrong with an accepted EB step's line, or None. Some nonfinite values are EB's own design, not a fault:
    a step accepted outside every basin (no certified state at the trial) prints state error inf (decrement inf) and
    gain nan, and an indefinite Newton model prints decrement inf; the first canary run on the CPU met exactly that line
    and passed it afterwards. A nonfinite step length or radius, a definite model's nonfinite decrement, a nan state
    error, an infinite gain, or a line no longer of this form is a failure."""
    matched = ACCEPTED_STEP.search(message)
    if matched is None:
        return "a line whose form the canary no longer reads (update ACCEPTED_STEP)"
    length, radius, decrement, kind, gain, error, error_decrement = (
        value if index == 3 else float(value) for index, value in enumerate(matched.groups())
    )
    if not (math.isfinite(length) and math.isfinite(radius)):
        return "a nonfinite length or radius"
    if kind == "definite" and not math.isfinite(decrement):
        return "a nonfinite decrement of a definite model"
    if math.isnan(error) or math.isnan(error_decrement) or math.isinf(error) != math.isinf(error_decrement):
        return "a nonfinite state error of a certified state"
    if math.isinf(gain):
        return "a nonfinite gain"
    return None


class GrowthGuard:
    """Bounded history of equivalent call boundaries: fail when successive window thirds rise beyond the allowance.

    The minima ignore a transient live trial, but two consecutively elevated windows cannot be dismissed as a peak.
    Warm-up and tolerance are explicit scenario controls. A lease count has no byte tolerance; it fails where each
    third's floor rose by a lease a call or more: a leak of one lease per call (every curvature solve's warm start
    before full_data_fit._SolveStarts) raises it by the third's width, while the outer loop's working set (its kept
    model snapshots and latest warm starts) moves by a few as steps are accepted, refused or halved: on the two-arm
    fixture it rose from 26 to 28 over the last eight calls of a loop that ended with three snapshots and one start.
    """

    def __init__(self, window=12, warmup=4, tolerance=32 * MIB):
        if window < 6 or window % 3 or warmup < 0 or tolerance < 0:
            raise ValueError("growth window must be a multiple of three >= 6; warmup and tolerance are nonnegative")
        self.window, self.warmup, self.tolerance = window, warmup, tolerance
        self.calls = 0
        self.history = deque(maxlen=window)

    def observe(self, values):
        self.calls += 1
        if self.calls <= self.warmup:
            return
        self.history.append(dict(values))
        if len(self.history) != self.window:
            return
        width = self.window // 3
        rows = list(self.history)
        for key in values:
            floors = [min(row[key] for row in rows[start:start + width]) for start in range(0, self.window, width)]
            allowance = width - 1 if key.endswith("leases") else self.tolerance
            if floors[1] - floors[0] > allowance and floors[2] - floors[1] > allowance:
                raise CanaryFailure(f"sustained per-EB-call growth in {key}: window floors {floors}, allowance {allowance}")


def memory_snapshot():
    from sv_pgs.memory_broker import current_broker

    gc.collect()
    values = {"python_live": tracemalloc.get_traced_memory()[0]}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            values["rss"] = int(line.split()[1]) * 1024
    broker = current_broker()
    if broker is not None:
        values.update({f"ledger_{pool}": broker.held(pool) for pool in broker.capacities})
        values["ledger_leases"] = len(broker.leases)
        if broker.meters:
            import cupy
            for device in broker.meters:
                with cupy.cuda.Device(int(device.removeprefix("device"))):
                    cupy.cuda.get_current_stream().synchronize()
                    values[f"{device}_live"] = cupy.get_default_memory_pool().used_bytes()
    return values


def held_purposes() -> Counter:
    """The broker's held leases counted by purpose."""
    from sv_pgs.memory_broker import current_broker

    broker = current_broker()
    return Counter(lease.purpose for lease in (broker.leases if broker is not None else ()))


def lease_purposes():
    """The broker's held leases counted by purpose, most first: what a lease-count failure names."""
    return ", ".join(f"{purpose} x{count}" for purpose, count in held_purposes().most_common())


@contextmanager
def instrumentation(scenario: Scenario, counters: dict, growth_mib: float = 32, allow_band: bool = False):
    """Benchmark-only observation hooks, restored on exit; no fit answer, derivative, stopping rule or draw is replaced.

    The phenotype step's code key includes this file. Its inputs still fingerprint each trait's real definition, whose
    SQL names the workspace's CDR, so WORKSPACE_CDR and GOOGLE_PROJECT are set to SYNTHETIC_WORKSPACE for the canary's
    run: nothing is queried (every connection is refused). The extra annotation columns enter through the real
    converter's site-reader result, before its table and identity are written. Timestamps use local time including its
    offset, also in the pipeline completion records; progress/profile times are local or elapsed.
    """
    from sv_pgs import full_data_fit, stage2_wiring

    original_sites = pipeline._plain_chromosome_sites
    original_call = phase_timer.TimedOracle.__call__
    original_mixture = full_data_fit._mode_mixture
    original_fit = full_data_fit.fit_full_data
    original_eb_log = scale_mixture_ep.log
    original_loop = full_data_fit._timed_hyperparameters

    def loop(*args, **kwargs):
        # An outer loop releases its oracle's curvature warm starts as it ends (full_data_fit._SolveStarts): one left
        # leased outlives its oracle in the ledger, as every start of every stage did before, one lease per solve.
        fits = original_loop(*args, **kwargs)
        left = held_purposes()["curvature solve warm start"]
        counters["loops_checked_for_warm_starts"] = counters.get("loops_checked_for_warm_starts", 0) + 1
        if left:
            raise CanaryFailure(f"an EB loop ended with {left} curvature warm starts still leased; held: {lease_purposes()}")
        return fits

    def eb_log(message):
        original_eb_log(message)
        problem = accepted_step_problem(message) if "inner step accepted" in message else None
        if problem is not None:
            raise CanaryFailure(f"EB accepted a step with {problem}")

    def sites(config, chromosome):
        result = original_sites(config, chromosome)
        rows = np.arange(result.expected.positions.size)
        paths = 1 + rows % 4
        return dataclasses.replace(result, annotations={**result.annotations, "n_paths": paths.astype(float),
            "n_paths_total": (paths + rows % 3).astype(float), "cx": (rows % 3).astype(np.int32)},
            annotation_legends={**result.annotation_legends, "cx": ("simple", "nested", "complex")})

    def call(oracle, hyperparameters):
        guard = getattr(oracle, "_canary_guard", None)
        if guard is None:
            guard = oracle._canary_guard = GrowthGuard(tolerance=int(growth_mib * MIB))
        before = memory_snapshot()
        try:
            guard.observe(before)
        except CanaryFailure as failure:
            if "leases" not in str(failure):
                raise
            raise CanaryFailure(f"{failure}; held: {lease_purposes()}") from None
        counters["max_calls_per_eb_loop"] = max(counters.get("max_calls_per_eb_loop", 0), guard.calls)
        counters["growth_windows"] = counters.get("growth_windows", 0) + int(len(guard.history) == guard.window)
        counters["eb_calls"] = counters.get("eb_calls", 0) + 1
        for key, value in before.items():
            counters[f"peak_boundary_{key}"] = max(counters.get(f"peak_boundary_{key}", 0), value)
        log("canary memory: " + " ".join(f"{key}={value}" for key, value in before.items()))
        return original_call(oracle, hyperparameters)

    def mixture(*args, **kwargs):
        counters["mode_mixture"] = counters.get("mode_mixture", 0) + 1
        started = time.perf_counter()
        try:
            return original_mixture(*args, **kwargs)
        finally:
            phase_timer.timing("mode_mixture", time.perf_counter() - started)

    def fit(*args, **kwargs):
        counters["full_data"] = counters.get("full_data", 0) + 1
        if kwargs.get("band") is None:
            counters["sample_full_data"] = counters.get("sample_full_data", 0) + 1
        else:
            counters["band_full_data"] = counters.get("band_full_data", 0) + 1
        prior, statistics = kwargs["prior"], kwargs["statistics"]
        for name, value in (("lattice_nodes", prior.log_variance_grid.size), ("annotation_columns", prior.scale_design.shape[1]),
                            ("ld_blocks", statistics.ld.block_count)):
            counters[f"max_{name}"] = max(counters.get(f"max_{name}", 0), int(value))
        if kwargs.get("band") is not None and not allow_band:
            raise CanaryFailure("the canary selected the band accelerator; the live run used the sample full-data route")
        return original_fit(*args, **kwargs)

    def timed_function(function, phase):
        def timed(*args, **kwargs):
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                phase_timer.timing(phase, time.perf_counter() - started)
        return timed

    steps = tuple(dataclasses.replace(entry, body=local_phenotypes, code=(sys.modules[__name__], local_phenotypes)) if entry.name == "phenotypes" else entry
                  for entry in pipeline._STEPS)
    tracing = tracemalloc.is_tracing()
    if not tracing:
        tracemalloc.start(1)
    with ExitStack() as stack:
        for owner, name, value in (
            (pipeline, "_STEPS", steps), (pipeline, "_plain_chromosome_sites", sites),
            (pipeline, "_utc", lambda: time.strftime("%Y%m%dT%H%M%S%z")),
            (phase_timer.TimedOracle, "__call__", call), (full_data_fit, "_mode_mixture", mixture), (full_data_fit, "_timed_hyperparameters", loop),
            (scale_mixture_ep, "log", eb_log),
            (stage2_wiring, "fit_full_data", fit),
            (full_data_fit, "moment_starts", timed_function(full_data_fit.moment_starts, "moment_starts")),
            (stage2_wiring, "annotation_design", timed_function(stage2_wiring.annotation_design, "annotation_design")),
            (socket, "create_connection", no_remote), (socket.socket, "connect", no_remote), (socket, "getaddrinfo", no_remote),
        ):
            stack.enter_context(patch.object(owner, name, value))
        stack.enter_context(patch.dict(os.environ, SYNTHETIC_WORKSPACE))
        try:
            yield
        finally:
            if not tracing:
                tracemalloc.stop()


def link_upstream(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=False)
    for name in pipeline.STEP_NAMES[:pipeline.STEP_NAMES.index("fit")]:
        (target / name).symlink_to((source / name).resolve(), target_is_directory=True)


def merge_arms(root: Path) -> None:
    """Copy whole model arms, with a common fit key, onto a fresh run; production code checks every payload on use."""
    link_upstream(root / "prepare", root / "merge")
    partial = root / "merge" / "fit.partial"
    partial.mkdir()
    keys = [(root / arm / "fit.partial" / pipeline.KEY_FILE).read_bytes() for arm in pipeline.ARMS]
    if keys[0] != keys[1]:
        raise CanaryFailure("the independent arms have different fit keys")
    (partial / pipeline.KEY_FILE).write_bytes(keys[0])
    for arm in pipeline.ARMS:
        shutil.copytree(root / arm / "fit.partial" / "models" / arm, partial / "models" / arm)


def validate_models(config, bindings, scenario):
    """Require certified genetic fits in both arms, and the intended column axis after the real filters and tie logic."""
    for arm in pipeline.ARMS:
        for trait in (*config.diseases, *config.traits):
            model = bindings.load_model(config.run_directory / "fit" / "models" / arm / trait)
            verdict = acceptance(model)
            if not verdict["accepted"]:
                raise CanaryFailure(f"{arm}/{trait} is uncertified: " + ", ".join(issue["code"] for issue in verdict["issues"] if issue["severity"] == "fatal"))
            # The short arm loses the structural records; allow only rounding at chromosome boundaries, not tie collapse.
            expected = sum(count - count // scenario.sv_every for count in scenario.counts(scenario.columns)) if arm == "snv_indel" else scenario.columns
            if any(scoring.store_rows.size != expected for scoring in model.scoring):
                raise CanaryFailure(f"{arm} fitted a different column count than its synthetic design ({expected})")


def run_stage(args, scenario):
    counters = {}
    started = time.perf_counter()
    root = args.out.resolve()
    os.environ["TZ"] = "America/Los_Angeles"
    time.tzset()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    detected = detect_compute_budget()
    if args.device == "cuda" and (detected.device_kind != "cuda" or len(detected.device_ids) != args.gpus):
        raise CanaryFailure("the requested CUDA device count is not available")
    if args.device == "cpu" and detected.device_kind != "cpu":
        raise CanaryFailure("CPU canary unexpectedly detected CUDA; CUDA_VISIBLE_DEVICES must be empty")
    budget = planned_budget(detected, scenario.samples, args.host_gib, args.device_gib)
    counters["budget"] = dataclasses.asdict(budget)
    counters["growth_tolerance_bytes"] = int(args.growth_mib * MIB)
    bindings = dataclasses.replace(pipeline.workspace_bindings(), bigquery_client=no_remote)
    try:
        with instrumentation(scenario, counters, args.growth_mib, args.allow_band_accelerator):
            if args.stage == "prepare":
                generating = time.perf_counter()
                config = generate(root / "inputs", scenario)
                phase_timer.timing("canary_generate", time.perf_counter() - generating)
                pipeline.run_pipeline(config, bindings, budget, through="direct_sv")
                with pipeline.DosageStore.open(root / "prepare" / "store" / "store") as store:
                    if not ANNOTATIONS <= set(store.variant_table.annotations):
                        raise CanaryFailure("the converter lost a real annotation column")
                    counters["records"] = store.n_variants
            else:
                config = pipeline.WorkspaceConfig.from_mapping(json.loads((root / "config.json").read_text()))
                config = dataclasses.replace(config, run_directory=root / args.stage)
                if args.stage in pipeline.ARMS:
                    link_upstream(root / "prepare", config.run_directory)
                    pipeline.run_pipeline(config, bindings, budget, through="fit", fit_arms=(args.stage,))
                    if not all(counters.get(name) for name in ("eb_calls", "sample_full_data", "mode_mixture")):
                        raise CanaryFailure("the fit did not exercise EB, full-data inference and the mode mixture")
                else:
                    merge_arms(root)
                    pipeline.run_pipeline(config, dataclasses.replace(bindings, fit=no_fit), budget, through="report")
                    validate_models(config, bindings, scenario)
                    with np.load(config.run_directory / "score" / "predictions.npz") as scores:
                        if not np.all(np.isfinite(scores["genetic_mean"])):
                            raise CanaryFailure("nonfinite genetic scores after arm merge")
                    for table in (config.run_directory / "report").glob("*.tsv"):
                        pipeline.check_exportable(table)
                    for step in ("fit", "score", "effects", "report"):
                        if not (config.run_directory / step / pipeline.COMPLETE_FILE).is_file():
                            raise CanaryFailure(f"the merged run did not complete {step}")
                    for trait in (*config.diseases, *config.traits):
                        for arm in pipeline.ARMS:
                            if not (config.run_directory / "effects" / f"{trait}.{arm}.tsv.gz").is_file():
                                raise CanaryFailure("a required effects table is absent")
            counters["passed"] = True
    finally:
        counters["seconds"] = time.perf_counter() - started
        # Linux ru_maxrss is KiB; RUSAGE_CHILDREN also covers the converter's completed worker processes.
        counters["peak_host_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        counters["peak_worker_bytes"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024
        (root / f"{args.stage}.metrics.json").write_text(json.dumps(counters, indent=2) + "\n")
        phase_timer.timing(f"canary_{args.stage}", counters["seconds"], gb=counters["peak_host_bytes"] / 1e9)


def process_tree_rss(pid: int, proc: Path = Path("/proc")) -> int:
    """Sum the current RSS of a task and its descendants; shared pages are counted in each process, conservatively.

    This is sampled, unlike each process's kernel high-water mark. A child may exit between the children list and its
    status read. The benchmark writes only the byte total, never process identifiers or command lines.
    """
    total = 0
    pending, seen = [pid], set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            status = (proc / str(current) / "status").read_text()
            total += next(int(line.split()[1]) * 1024 for line in status.splitlines() if line.startswith("VmRSS:"))
            pending.extend(int(child) for child in (proc / str(current) / "task" / str(current) / "children").read_text().split())
        except (FileNotFoundError, ProcessLookupError, StopIteration):
            continue
    return total


def supervise(command, environment, handle, seconds, metrics):
    """Run one isolated phase, sample its entire process tree, and kill its whole group when its deadline expires."""
    started = time.perf_counter()
    peak = 0
    process = subprocess.Popen(command, env=environment, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        while process.poll() is None:
            peak = max(peak, process_tree_rss(process.pid))
            if time.perf_counter() - started >= seconds:
                raise CanaryFailure("canary deadline expired; EB did not complete")
            time.sleep(.25)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)
        metrics["passed"] = True
    finally:
        metrics.update(seconds=time.perf_counter() - started, peak_process_tree_rss_bytes=peak)
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=1_060_000)
    parser.add_argument("--records", type=int)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--chromosomes", type=int, default=22)
    parser.add_argument("--block-columns", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--sv-every", type=int, default=20)
    parser.add_argument("--traits", nargs="*", default=["total_bilirubin"])
    parser.add_argument("--diseases", nargs="*", default=[], help="Canonical disease names; synthetic balanced targets use the real Bernoulli route.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--gpus", type=int, choices=(1, 2), default=1)
    parser.add_argument("--host-gib", type=float)
    parser.add_argument("--device-gib", type=float)
    parser.add_argument("--growth-mib", type=float, default=32)
    parser.add_argument("--allow-band-accelerator", action="store_true",
                        help="Permit measured band acceleration on tiny fixtures; final sample-route validation is still required.")
    parser.add_argument("--deadline-seconds", type=float, help="Default: 8 hours on CPU, 3 hours on CUDA.")
    parser.add_argument("--stage", choices=("prepare", *pipeline.ARMS, "merge"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.deadline_seconds is None:
        args.deadline_seconds = 28800 if args.device == "cpu" else 10800
    if args.deadline_seconds <= 0 or not math.isfinite(args.deadline_seconds) or args.growth_mib < 0 or not math.isfinite(args.growth_mib):
        parser.error("deadline must be finite and positive; growth allowance finite and nonnegative")
    scenario = Scenario(args.columns, args.samples, args.chromosomes, args.block_columns, args.seed, args.records, args.sv_every, tuple(args.traits), tuple(args.diseases))
    if args.stage:
        run_stage(args, scenario)
        return 0
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "scenario.json").write_text(json.dumps(dataclasses.asdict(scenario), indent=2) + "\n")
    environment = dict(os.environ, TZ="America/Los_Angeles", PYTHONUNBUFFERED="1")
    visible = environment.get("CUDA_VISIBLE_DEVICES", ",".join(map(str, range(args.gpus)))).split(",")
    environment["CUDA_VISIBLE_DEVICES"] = "" if args.device == "cpu" else ",".join(visible[:args.gpus])
    arguments = list(sys.argv[1:] if argv is None else argv)
    started = time.perf_counter()
    status = 1
    stages = {}
    try:
        with (args.out / "progress.log").open("w") as handle:
            for stage in ("prepare", *pipeline.ARMS, "merge"):
                remaining = args.deadline_seconds - (time.perf_counter() - started)
                if remaining <= 0:
                    raise CanaryFailure("canary deadline expired before completion")
                print(f"canary: {stage}; progress in {args.out / 'progress.log'}", flush=True)
                stages[stage] = {"passed": False}
                supervise(
                    [sys.executable, "-m", "benchmarks.wg_canary", *arguments, "--stage", stage], environment, handle, remaining, stages[stage],
                )
        # Every step's line and every phase of its work, read by the round's gate: a step or a phase that went unlogged,
        # a reused step that did work or a fitted group without its Stage 0 is a failure here as in a round's log.
        with (args.out / "progress.log").open(errors="replace") as handle:
            found = round_timeline.problems(round_timeline.read(handle.read().splitlines()))
        if found:
            raise CanaryFailure("the progress log fails the round's gate: " + "; ".join(found))
        if args.device == "cuda":
            with (args.out / "progress.log").open() as handle:
                pools = {match.group(1) for line in handle if "timing: resident_fill " in line
                         for match in re.finditer(r"pool=(device\d+)", line)}
            if len(pools) != args.gpus:
                raise CanaryFailure("the CUDA fit did not fill resident codes on every requested device")
        status = 0
        return 0
    finally:
        with (args.out / "progress.log").open(errors="replace") as handle:
            profile = profile_log.parse(handle)
        cache = {"admissions": 0, "refusals": 0, "evictions": 0}
        with (args.out / "progress.log").open(errors="replace") as handle:
            for line in handle:
                if "resident genotype codes:" in line:
                    if "not admitted" in line:
                        cache["refusals"] += 1
                    elif "admitted" in line:
                        cache["admissions"] += 1
                    elif "evicted" in line:
                        cache["evictions"] += 1
        with (args.out / "progress.log").open(errors="replace") as handle:
            report = profile_log.report(profile) + "\n\n" + round_timeline.report(round_timeline.read(handle.read().splitlines()))
        (args.out / "profile.txt").write_text(report + "\n")
        (args.out / "result.json").write_text(json.dumps({"passed": status == 0, "seconds": time.perf_counter() - started, "stages": stages, "resident_cache": cache,
            "peak_process_tree_rss_bytes": max((entry.get("peak_process_tree_rss_bytes", 0) for entry in stages.values()), default=0)}) + "\n")
        print(report, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

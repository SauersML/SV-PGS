"""Biobank-scale timing of the full-data fit: Stage 0 and Stage 2 on one synthetic chromosome at n samples.

``generate`` writes a one-half dosage store of ``--samples`` samples x ``--variants`` records of one chromosome:
each haplotype copies one of a few founder haplotypes over segments of consecutive records (a haplotype switches
founder at a segment's start with probability one half, so LD decays over a few segments), with each allele flipped
independently at a small rate, so LD is block-structured and no two columns are exactly equal. Hard calls: the code
is 127 times the dosage. A quantitative trait of ``--causal`` causal records and heritability ``--heritability``
is drawn with it (the genetic value accumulated as the codes are written), with two standard normal covariates.

``fit`` runs ``stage2_wiring.fit_models`` on it, timing Stage 0 (``compute_genotype_statistics``) and the rest of the
fit separately, and reports the route the measured pass costs chose, the peak resident set and the device pool's
peak. Nothing here is an accuracy claim: the trait is this script's own simulation [own-sim], and the run measures
time and memory only.
"""
from __future__ import annotations

import argparse
import json
import resource
import shutil
import threading
import time
from pathlib import Path

import numpy as np

from sv_pgs.config import TraitType, VariantClass
from sv_pgs.dosage_store import CODES_PER_DOSAGE, VARIANT_CLASSES, DosageStore, VariantTable, write_dosage_store
from sv_pgs.progress import log

CODE_PER_ALLELE = 127
"""A hard call's code per ALT allele: codes are 127 x the dosage (``dosage_store``: 254 codes span dosage 0..2)."""


def generate(root: Path, samples: int, variants: int, causal: int, heritability: float, founders: int, segment: tuple[int, int], flip: float, seed: int) -> None:
    generator = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    haplotypes = 2 * samples
    frequency = generator.uniform(0.05, 0.5, size=variants)
    founder_alleles = generator.random((founders, variants)) < frequency
    lengths = generator.integers(segment[0], segment[1] + 1, size=variants // segment[0] + 1)
    cuts = np.concatenate([[0], np.cumsum(lengths)])
    cuts = cuts[cuts < variants].tolist() + [variants]
    founder = generator.integers(0, founders, size=haplotypes)
    causal_rows = np.sort(generator.choice(variants, size=causal, replace=False))
    effects = generator.standard_normal(causal)
    genetic = np.zeros(samples)
    codes = np.lib.format.open_memmap(root / "codes.npy", mode="w+", dtype=np.uint8, shape=(variants, samples))
    sums = np.zeros(variants, dtype=np.uint64)
    squares = np.zeros(variants, dtype=np.uint64)
    started = time.time()
    for first, last in zip(cuts[:-1], cuts[1:]):
        switch = generator.random(haplotypes) < 0.5
        founder = np.where(switch, generator.integers(0, founders, size=haplotypes), founder)
        alleles = founder_alleles[:, first:last][founder]
        flat = alleles.reshape(-1)
        flips = generator.integers(0, flat.shape[0], size=generator.binomial(flat.shape[0], flip))
        flat[flips] = ~flat[flips]
        dosage = alleles[0::2].astype(np.uint8) + alleles[1::2].astype(np.uint8)
        block = np.ascontiguousarray((dosage * CODE_PER_ALLELE).T)
        codes[first:last] = block
        wide = block.astype(np.uint64)
        sums[first:last] = wide.sum(axis=1)
        squares[first:last] = (wide * wide).sum(axis=1)
        inside = (causal_rows >= first) & (causal_rows < last)
        if inside.any():
            values = dosage[:, causal_rows[inside] - first].astype(np.float64)
            genetic += (values - values.mean(axis=0)) @ effects[inside]
        del alleles, dosage, block, wide
    codes.flush()
    log(f"biobank scale: {variants:,} records x {samples:,} samples generated in {time.time() - started:.0f} s")
    covariates = generator.standard_normal((samples, 2))
    genetic *= np.sqrt(heritability) / np.std(genetic)
    phenotype = genetic + 0.2 * covariates[:, 0] + np.sqrt(1.0 - heritability) * generator.standard_normal(samples)
    np.savez(root / "trait.npz", phenotype=phenotype, genetic=genetic, covariates=covariates, causal=causal_rows, effects=effects)
    ids = [f"v{row}" for row in range(variants)]
    positions = np.arange(1, variants + 1, dtype=np.int64) * 100
    table = VariantTable(
        chromosome=np.ones(variants, dtype=np.int8),
        position=positions,
        genetic_position_cm=positions / 1e6,
        ref_length=np.ones(variants, dtype=np.int32),
        alt_length=np.ones(variants, dtype=np.int32),
        variant_class=np.full(variants, VARIANT_CLASSES.index(VariantClass.SNV), dtype=np.uint8),
        codes_per_unit=np.full(variants, CODES_PER_DOSAGE, dtype=np.uint8),
        value_origin=np.zeros(variants, dtype=np.int64),
        group_first=np.arange(variants, dtype=np.int64),
        sum_code=sums,
        sum_code2=squares,
        annotations={},
        annotation_legends={},
        id_bytes=np.frombuffer("".join(ids).encode(), dtype=np.uint8),
        id_offsets=np.concatenate([[0], np.cumsum([len(name) for name in ids])]).astype(np.int64),
    )
    started = time.time()
    rows = 4096
    write_dosage_store(root / "store", samples, table, (codes[first:first + rows] for first in range(0, variants, rows)), codec="raw")
    del codes
    (root / "codes.npy").unlink()
    log(f"biobank scale: store written in {time.time() - started:.0f} s")


def fit(root: Path, work: Path, seed: int) -> dict:
    from benchmarks.bench_sim.submissions.svpgs_full import _array_module, task_budget
    from sv_pgs import stage2_wiring
    from sv_pgs.fit_model import DRAW_COUNT
    from sv_pgs.scale_mixture_ep import device_scope

    peaks = {"pool_bytes": 0}

    def sample() -> None:
        try:
            import cupy
        except ImportError:
            return
        pool = cupy.get_default_memory_pool()
        while True:
            peaks["pool_bytes"] = max(peaks["pool_bytes"], int(pool.total_bytes()))
            time.sleep(0.25)

    timings: dict[str, float] = {}
    stage0 = stage2_wiring.compute_genotype_statistics

    def timed_stage0(*arguments, **keywords):
        started = time.time()
        result = stage0(*arguments, **keywords)
        timings["stage0_seconds"] = time.time() - started
        timings["stage0_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        return result

    stage2_wiring.compute_genotype_statistics = timed_stage0
    choice: dict[str, float] = {}
    measured = stage2_wiring.pass_costs

    def recorded(band, source, array_module):
        sample_seconds, gram_seconds = measured(band, source, array_module)
        choice.update(sample_pass_seconds=sample_seconds, gram_pass_seconds=gram_seconds)
        return sample_seconds, gram_seconds

    stage2_wiring.pass_costs = recorded
    trait = np.load(root / "trait.npz")
    phenotype = np.asarray(trait["phenotype"], dtype=np.float64)
    samples = phenotype.shape[0]
    covariates = np.column_stack([np.ones(samples), trait["covariates"]])
    budget = task_budget()
    # The sampler starts once the device's context and libraries exist: a thread polling CuPy's pool while the main
    # thread created them failed the first allocation (cudaErrorInvalidValue in memsetAsync, twice on A100 nodes).
    threading.Thread(target=sample, daemon=True).start()
    work.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with DosageStore.open(root / "store") as store, device_scope(_array_module(budget)):
        fitted = stage2_wiring.fit_models(
            store=store, store_columns=np.arange(samples, dtype=np.int64), covariates=covariates,
            covariate_columns=np.ones((1, covariates.shape[1]), dtype=bool), targets=phenotype[:, None],
            training=np.ones((samples, 1), dtype=bool), trait_types=[TraitType.QUANTITATIVE], log_variance_offset=None, budget=budget,
            work_dir=work, seed=seed, draw_count=DRAW_COUNT,
        )
    total = time.time() - started
    (scoring,) = fitted.scoring
    certificate = fitted.certificate
    report = {
        "samples": samples,
        "records": int(np.asarray(scoring.store_rows).shape[0]),
        "device": budget.device_kind,
        "total_seconds": total,
        "stage0_seconds": timings.get("stage0_seconds"),
        "stage2_seconds": total - timings.get("stage0_seconds", 0.0),
        "route": "gram" if choice and choice["gram_pass_seconds"] < choice["sample_pass_seconds"] else "samples",
        **choice,
        "noise_variance": float(fitted.noise_variance[0]),
        "far_field": np.asarray(certificate.far_field).tolist(),
        "budget_unresolved": np.asarray(certificate.budget_unresolved).tolist(),
        "outer_iterations": np.asarray(certificate.outer_iterations).tolist(),
        "stage0_peak_rss_bytes": timings.get("stage0_rss_bytes"),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "device_pool_peak_bytes": peaks["pool_bytes"],
    }
    log(f"biobank scale: {json.dumps(report)}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("generate", "fit"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--variants", type=int)
    parser.add_argument("--causal", type=int)
    parser.add_argument("--heritability", type=float)
    parser.add_argument("--founders", type=int)
    parser.add_argument("--segment", type=int, nargs=2)
    parser.add_argument("--flip", type=float)
    parser.add_argument("--seed", type=int, default=20260923)
    arguments = parser.parse_args()
    if arguments.command == "generate":
        generate(arguments.root, arguments.samples, arguments.variants, arguments.causal, arguments.heritability, arguments.founders, tuple(arguments.segment), arguments.flip, arguments.seed)
    else:
        work = arguments.work or arguments.root / "work"
        try:
            report = fit(arguments.root, work, arguments.seed)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        (arguments.root / "report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

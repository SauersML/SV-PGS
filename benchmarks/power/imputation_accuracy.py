"""How well a bench-sim measurement arm recovers true genotypes, by minor allele frequency: the squared correlation of
each record's observed dosage with its true genotype over every person of the cohort, for every measured SV record and
a random sample of measured SNV records, summarized per frequency bin.

    python -m benchmarks.power.imputation_accuracy --cohort COHORT --out imputation_accuracy.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.bench_sim import harness, truth

FREQUENCY_EDGES = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5)
SNV_SAMPLE = 6000
BLOCK = 500


def record_r2(cohort: Path, records: np.ndarray, arm: str) -> tuple[np.ndarray, np.ndarray]:
    """(minor allele frequency, dosage r^2 with the true genotype) of each record; r^2 is nan for a monomorphic one."""
    true_genotypes = np.load(cohort / "truth_G.npy", mmap_mode="r")
    observed = np.load(cohort / harness.ARMS[arm][0], mmap_mode="r")
    frequency, r2 = [], []
    for start in range(0, records.size, BLOCK):
        block = records[start : start + BLOCK]
        genotypes = np.asarray(true_genotypes[block], dtype=np.float64)
        dosages = np.asarray(observed[block], dtype=np.float64) / harness.CODES_PER_DOSAGE
        allele = genotypes.mean(axis=1) / 2.0
        frequency.append(np.minimum(allele, 1.0 - allele))
        genotypes -= genotypes.mean(axis=1, keepdims=True)
        dosages -= dosages.mean(axis=1, keepdims=True)
        scale = np.sqrt(np.sum(genotypes**2, axis=1) * np.sum(dosages**2, axis=1))
        product = np.sum(genotypes * dosages, axis=1)
        r2.append(np.where(scale > 0, np.square(product / np.where(scale > 0, scale, 1.0)), np.nan))
    return np.concatenate(frequency), np.concatenate(r2)


def summary(frequency: np.ndarray, r2: np.ndarray) -> dict:
    bins = []
    for low, high in zip(FREQUENCY_EDGES[:-1], FREQUENCY_EDGES[1:]):
        values = r2[(frequency >= low) & (frequency < high) & np.isfinite(r2)]
        bins.append({"low": low, "high": high, "count": int(values.size)} | (
            {"mean": float(values.mean()), "median": float(np.median(values)),
             "q25": float(np.percentile(values, 25)), "q75": float(np.percentile(values, 75))} if values.size else {}
        ))
    return {"records": int(frequency.size), "median_maf": float(np.median(frequency)),
            "below_1pct": float(np.mean(frequency < 0.01)), "bins": bins}


def imputation_accuracy(cohort: Path, arm: str = "beagle", seed: int = 0) -> dict:
    variants = np.load(cohort / "variants.npz")
    measured = np.load(cohort / "measured.npy")
    classes = variants["cls"]
    snvs = np.flatnonzero(measured & (classes == truth.SNV))
    groups = {
        "SV": np.flatnonzero(measured & (classes == truth.SV)),
        "SNV": np.sort(np.random.default_rng(seed).choice(snvs, min(SNV_SAMPLE, snvs.size), replace=False)),
    }
    return {"arm": arm, "measurement": harness.ARMS[arm][2], "edges": list(FREQUENCY_EDGES)} | {
        name: summary(*record_r2(cohort, records, arm)) for name, records in groups.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cohort", type=Path, required=True, help="a bench-sim cohort chromosome directory")
    parser.add_argument("--arm", choices=list(harness.ARMS), default="beagle")
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.out.write_text(json.dumps(imputation_accuracy(arguments.cohort, arguments.arm), indent=1))
    print(arguments.out)


if __name__ == "__main__":
    main()

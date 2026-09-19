"""bench-sim: is imputed DS a calibrated posterior mean or draw-like? Aggregates per class x MAF stratum x group.

For every record, over the calibration samples: kappa = Cov(G, DS) / Var(DS) (the slope of truth on DS:
1 for a posterior mean, about sqrt(r2) for a draw from a calibrated posterior), r2 = corr^2(G, DS), and the
variance ratio Var(DS) / Var(G). Strata pool the centred cross-products of their records. MAF strata here are
reporting cells for aggregates, not model terms.

    python -m benchmarks.bench_sim.calibration --dir <cohort/chr22> --samples 2500 \
        --arm glimpse2=observed_smoke.npy --arm beagle=observed_beagle.npy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.bench_sim.measurement import measured_records

CODES_PER_DOSAGE = 127
CLASS_NAMES = ("SNV", "INDEL", "TR", "SV")
MAF_EDGES = (0.0, 0.001, 0.01, 0.05, 0.5)


def stratum_statistics(truth: np.ndarray, dosage: np.ndarray) -> dict:
    truth = truth - truth.mean(axis=1, keepdims=True)
    dosage = dosage - dosage.mean(axis=1, keepdims=True)
    cross = (truth * dosage).sum(axis=1)
    truth_ss = (truth * truth).sum(axis=1)
    dosage_ss = (dosage * dosage).sum(axis=1)
    live = (truth_ss > 0) & (dosage_ss > 0)
    if not live.any():
        return {"records": 0}
    per_record_kappa = cross[live] / dosage_ss[live]
    per_record_r2 = cross[live] ** 2 / (truth_ss[live] * dosage_ss[live])
    return {
        "records": int(live.sum()),
        "kappa_pooled": float(cross[live].sum() / dosage_ss[live].sum()),
        "kappa_median": float(np.median(per_record_kappa)),
        "r2_median": float(np.median(per_record_r2)),
        "sqrt_r2_median": float(np.median(np.sqrt(per_record_r2))),
        "variance_ratio_median": float(np.median(dosage_ss[live] / truth_ss[live])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--arm", action="append", required=True, help="name=observed file in --dir")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.dir)
    cls = np.load(root / "variants.npz")["cls"]
    measured = measured_records(root)
    samples = np.load(root / "samples.npz")
    group = samples["group"][:args.samples]
    group_names = [str(name) for name in samples["group_names"]]
    truth = np.load(root / "truth_G.npy", mmap_mode="r")
    truth_block = np.empty((cls.size, args.samples), dtype=np.uint8)
    for start in range(0, cls.size, 20_000):
        truth_block[start:start + 20_000] = np.asarray(truth[start:start + 20_000])[:, :args.samples]
    frequency = truth_block.mean(axis=1, dtype=np.float64) / 2.0
    minor = np.minimum(frequency, 1.0 - frequency)
    report: dict = {
        "samples": args.samples,
        "group_counts": {name: int((group == index).sum()) for index, name in enumerate(group_names)},
        "arms": {},
    }
    for arm in args.arm:
        name, filename = arm.split("=")
        observed = np.load(root / filename, mmap_mode="r")
        dosage_block = np.empty((cls.size, args.samples), dtype=np.float32)
        for start in range(0, cls.size, 20_000):
            dosage_block[start:start + 20_000] = np.asarray(observed[start:start + 20_000])[:, :args.samples] / CODES_PER_DOSAGE
        table: dict = {}
        for class_index, class_name in enumerate(CLASS_NAMES):
            for low, high in zip(MAF_EDGES[:-1], MAF_EDGES[1:]):
                rows = np.flatnonzero(measured & (cls == class_index) & (minor > low) & (minor <= high))
                if rows.size == 0:
                    continue
                cell = f"{class_name}|maf({low},{high}]"
                table[cell] = {"all": stratum_statistics(truth_block[rows].astype(np.float64), dosage_block[rows].astype(np.float64))}
                for group_index, group_name in enumerate(group_names):
                    members = group == group_index
                    if members.sum() > 2:
                        table[cell][group_name] = stratum_statistics(
                            truth_block[np.ix_(rows, members)].astype(np.float64), dosage_block[np.ix_(rows, members)].astype(np.float64))
        report["arms"][name] = table
    Path(args.out).write_text(json.dumps(report, indent=1))
    for name, table in report["arms"].items():
        for cell, stats in table.items():
            overall = stats["all"]
            if overall.get("records"):
                print(f"{name:8s} {cell:22s} n={overall['records']:7d} kappa={overall['kappa_pooled']:.3f} "
                      f"sqrt(r2)={overall['sqrt_r2_median']:.3f} r2={overall['r2_median']:.3f} var_ratio={overall['variance_ratio_median']:.3f}")


if __name__ == "__main__":
    main()

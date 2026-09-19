"""bench-sim neutral baselines, on the kernels prepared by gpu_kernels.

oracle_observed applies the true additive effects to observed codes: the imputation-limited ceiling.
ridge_inf is GBLUP with lambda = (1 - h2)/h2, h2 from Haseman-Elston on the training kernel, in two arms:
"simple" (SNV+INDEL records only) and "all". Predictions go to <results>/<method>/<scenario>/prediction.npz.

    python -m benchmarks.bench_sim.baselines --cohort <cohort/chr22> --scenarios <dirs...> --results <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.bench_sim.harness import ARMS, covariate_matrix
from benchmarks.bench_sim.measurement import measured_records

CODES_PER_DOSAGE = 127


def residualized(outcome: np.ndarray, covariates: np.ndarray) -> tuple[np.ndarray, float]:
    design = np.column_stack([np.ones(outcome.size), covariates])
    coefficients, *_ = np.linalg.lstsq(design, outcome, rcond=None)
    residual = outcome - design @ coefficients
    return residual / residual.std(), float(residual.std())


def load_shared(cohort: Path, arm: str) -> dict:
    """Scenario-independent inputs, read once: split, covariates, kernel blocks and eigendecompositions."""
    samples = np.load(cohort / "samples.npz")
    train = np.flatnonzero(~samples["is_test"])
    test = np.flatnonzero(samples["is_test"])
    covariates, _ = covariate_matrix(cohort, arm)
    counts = json.loads((cohort / f"kernel_counts_{arm}.json").read_text())
    weight_structural = counts["structural"] / (counts["simple"] + counts["structural"])
    shared = {"train": train, "test": test, "covariates": covariates, "weight_structural": weight_structural,
              "observed": np.load(cohort / ARMS[arm][0], mmap_mode="r"), "cls": np.load(cohort / "variants.npz")["cls"],
              "measured": measured_records(cohort)}
    diagonals, crosses = {}, {}
    for name in ("simple", "structural"):
        kernel = np.load(cohort / f"kernel_{name}_{arm}.npy", mmap_mode="r")
        diagonals[name] = np.asarray(kernel[train, train], dtype=np.float64)
        crosses[name] = np.asarray(kernel[test][:, train], dtype=np.float64)
    shared["cross"] = crosses
    shared["diagonal"] = {"simple": diagonals["simple"],
                          "all": (1 - weight_structural) * diagonals["simple"] + weight_structural * diagonals["structural"]}
    shared["eigen"] = {name: (np.load(cohort / f"eig_{name}_{arm}_values.npy").astype(np.float64),
                              np.load(cohort / f"eig_{name}_{arm}_vectors.npy").astype(np.float64))
                       for name in ("simple", "all")}
    return shared


def baselines(shared: dict, scenario: Path, results: Path, arm: str) -> None:
    train, test, covariates = shared["train"], shared["test"], shared["covariates"]
    truth = np.load(scenario / "truth.npz")

    # oracle_observed: the true additive effects of the measured causal records, applied to their observed
    # dosages. Causal records the imputed callset lacks are invisible to every method.
    causal = truth["causal"]
    order = np.argsort(causal)
    rows, effects = causal[order], truth["per_allele"][order]
    visible = shared["measured"][rows]
    rows, effects = rows[visible], effects[visible]
    dosage = np.asarray(shared["observed"][rows], dtype=np.float64)[:, test] / CODES_PER_DOSAGE
    dosage -= dosage.mean(axis=1, keepdims=True)
    structural = shared["cls"][rows] >= 2
    out = results / "oracle_observed" / scenario.name
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "prediction.npz", total=effects @ dosage, structural=effects[structural] @ dosage[structural])
    (out / "meta.json").write_text(json.dumps({"measurement": ARMS[arm][2]}))

    # ridge_inf, two kernel arms.
    standardized, phenotype_scale = residualized(truth["phenotype"][train], covariates[train])
    weight_structural = shared["weight_structural"]
    for kernel_arm in ("simple", "all"):
        values, vectors = shared["eigen"][kernel_arm]
        projected = vectors.T @ standardized
        quadratic = float(projected @ (values * projected))
        diagonal = shared["diagonal"][kernel_arm]
        heritability = (quadratic - diagonal @ (standardized * standardized)) / (float(values @ values) - diagonal @ diagonal)
        out = results / f"ridge_inf_{kernel_arm}" / scenario.name
        out.mkdir(parents=True, exist_ok=True)
        # h2 is truncated to its parameter space [0, 1]; at h2 = 1 the solve is the pseudo-inverse.
        heritability_used = float(np.clip(heritability, 0.0, 1.0))
        if heritability_used == 0.0:
            np.savez(out / "prediction.npz", total=np.zeros(test.size), structural=np.zeros(test.size))
        else:
            ridge = (1.0 - heritability_used) / heritability_used
            shifted = values + ridge
            inverse = np.divide(1.0, shifted, out=np.zeros_like(shifted), where=shifted > 0)
            alpha = vectors @ (projected * inverse)
            if kernel_arm == "simple":
                total, structural_part = shared["cross"]["simple"] @ alpha, np.zeros(test.size)
            else:
                structural_part = weight_structural * (shared["cross"]["structural"] @ alpha)
                total = (1 - weight_structural) * (shared["cross"]["simple"] @ alpha) + structural_part
            np.savez(out / "prediction.npz", total=phenotype_scale * total, structural=phenotype_scale * structural_part)
        (out / "meta.json").write_text(json.dumps({"he_h2": heritability, "h2_used": heritability_used, "measurement": ARMS[arm][2]}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--scenarios", nargs="+", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    args = parser.parse_args()
    shared = load_shared(Path(args.cohort), args.arm)
    for scenario in args.scenarios:
        baselines(shared, Path(scenario), Path(args.results) / args.arm, args.arm)
        print("baselines", scenario, flush=True)


if __name__ == "__main__":
    main()

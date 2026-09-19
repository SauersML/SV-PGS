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

from benchmarks.bench_sim.harness import covariate_matrix

CODES_PER_DOSAGE = 127


def residualized(outcome: np.ndarray, covariates: np.ndarray) -> tuple[np.ndarray, float]:
    design = np.column_stack([np.ones(outcome.size), covariates])
    coefficients, *_ = np.linalg.lstsq(design, outcome, rcond=None)
    residual = outcome - design @ coefficients
    return residual / residual.std(), float(residual.std())


def baselines(cohort: Path, scenario: Path, results: Path) -> None:
    samples = np.load(cohort / "samples.npz")
    train = np.flatnonzero(~samples["is_test"])
    test = np.flatnonzero(samples["is_test"])
    truth = np.load(scenario / "truth.npz")
    covariates, _ = covariate_matrix(cohort)
    observed = np.load(cohort / "observed.npy", mmap_mode="r")
    cls = np.load(cohort / "variants.npz")["cls"]

    # oracle_observed: the true additive effects applied to observed dosages.
    causal = truth["causal"]
    order = np.argsort(causal)
    rows, effects = causal[order], truth["per_allele"][order]
    dosage = np.asarray(observed[rows], dtype=np.float64)[:, test] / CODES_PER_DOSAGE
    dosage -= dosage.mean(axis=1, keepdims=True)
    structural = cls[rows] >= 2
    out = results / "oracle_observed" / scenario.name
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "prediction.npz", total=effects @ dosage, structural=effects[structural] @ dosage[structural])

    # ridge_inf, two arms.
    standardized, phenotype_scale = residualized(truth["phenotype"][train], covariates[train])
    counts = json.loads((cohort / "kernel_counts.json").read_text())
    simple = np.load(cohort / "kernel_simple.npy", mmap_mode="r")
    structural_kernel = np.load(cohort / "kernel_structural.npy", mmap_mode="r")
    weight_structural = counts["structural"] / (counts["simple"] + counts["structural"])
    cross_simple = np.asarray(simple[test][:, train], dtype=np.float64)
    cross_structural = np.asarray(structural_kernel[test][:, train], dtype=np.float64)
    for arm in ("simple", "all"):
        values = np.load(cohort / f"eig_{arm}_values.npy").astype(np.float64)
        vectors = np.load(cohort / f"eig_{arm}_vectors.npy", mmap_mode="r")
        projected = np.asarray(vectors.T @ standardized, dtype=np.float64)
        quadratic = float(projected @ (values * projected))
        diagonal = np.asarray(simple[train, train] if arm == "simple" else
                              (1 - weight_structural) * simple[train, train] + weight_structural * structural_kernel[train, train], dtype=np.float64)
        frobenius = float(values @ values)
        heritability = (quadratic - diagonal @ (standardized * standardized)) / (frobenius - diagonal @ diagonal)
        out = results / f"ridge_inf_{arm}" / scenario.name
        out.mkdir(parents=True, exist_ok=True)
        # h2 is truncated to its parameter space [0, 1]; at h2 = 1 the solve is the pseudo-inverse.
        heritability_used = float(np.clip(heritability, 0.0, 1.0))
        if heritability_used == 0.0:
            np.savez(out / "prediction.npz", total=np.zeros(test.size), structural=np.zeros(test.size))
        else:
            ridge = (1.0 - heritability_used) / heritability_used
            shifted = values + ridge
            inverse = np.divide(1.0, shifted, out=np.zeros_like(shifted), where=shifted > 0)
            alpha = np.asarray(vectors @ (projected * inverse), dtype=np.float64)
            if arm == "simple":
                total, structural_part = cross_simple @ alpha, np.zeros(test.size)
            else:
                structural_part = weight_structural * (cross_structural @ alpha)
                total = (1 - weight_structural) * (cross_simple @ alpha) + structural_part
            np.savez(out / "prediction.npz", total=phenotype_scale * total, structural=phenotype_scale * structural_part)
        (out / "meta.json").write_text(json.dumps({"he_h2": heritability, "h2_used": heritability_used}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--scenarios", nargs="+", required=True)
    parser.add_argument("--results", required=True)
    args = parser.parse_args()
    for scenario in args.scenarios:
        baselines(Path(args.cohort), Path(scenario), Path(args.results))
        print("baselines", scenario, flush=True)


if __name__ == "__main__":
    main()

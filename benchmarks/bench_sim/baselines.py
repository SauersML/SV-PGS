"""bench-sim neutral baselines, on the kernels prepared by gpu_kernels.

oracle_observed applies the true additive effects to observed codes: the imputation-limited ceiling.
ridge_inf is GBLUP with lambda = (1 - h2)/h2, h2 from Haseman-Elston on the training kernel, solved by Cholesky, in two arms:
"simple" (SNV+INDEL records only) and "all". Predictions go to <results>/<method>/<scenario>/prediction.npz.

    python -m benchmarks.bench_sim.baselines --cohort <cohort/chr22> --scenarios <dirs...> --results <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from benchmarks.bench_sim.harness import ARMS, covariate_matrix
from benchmarks.bench_sim.records import measured_records

CODES_PER_DOSAGE = 127


def residualized(outcome: np.ndarray, covariates: np.ndarray) -> tuple[np.ndarray, float]:
    design = np.column_stack([np.ones(outcome.size), covariates])
    coefficients, *_ = np.linalg.lstsq(design, outcome, rcond=None)
    residual = outcome - design @ coefficients
    return residual / residual.std(), float(residual.std())


PROJECTION_BLOCK_ROWS = 2048


def project_out_covariates(train_kernel: np.ndarray, cross_kernel: np.ndarray, design: np.ndarray) -> None:
    """In place: K <- P K P and C <- C P with P = I - Q Q' (Q an orthonormal basis of the training design).

    The covariates and PCs are fixed effects, so the kernel must be projected the same way the phenotype is
    residualized; an unprojected kernel keeps the ancestry structure the PCs removed from y, inflates tr(K^2),
    and drives the Haseman-Elston h2 toward zero.
    """
    basis, _ = np.linalg.qr(design)
    kernel_basis = train_kernel @ basis
    middle = basis.T @ kernel_basis
    for first in range(0, train_kernel.shape[0], PROJECTION_BLOCK_ROWS):
        rows = slice(first, first + PROJECTION_BLOCK_ROWS)
        train_kernel[rows] -= basis[rows] @ kernel_basis.T + kernel_basis[rows] @ basis.T - (basis[rows] @ middle) @ basis.T
    cross_kernel -= (cross_kernel @ basis) @ basis.T


def load_shared(cohort: Path, arm: str) -> dict:
    """Scenario-independent inputs, read once: split, covariates, and the training and test-train kernel blocks."""
    samples = np.load(cohort / "samples.npz")
    train = np.flatnonzero(~samples["is_test"])
    test = np.flatnonzero(samples["is_test"])
    covariates, _ = covariate_matrix(cohort, arm)
    counts = json.loads((cohort / f"kernel_counts_{arm}.json").read_text())
    weight_structural = counts["structural"] / (counts["simple"] + counts["structural"])
    blocks = {}
    for name in ("simple", "structural"):
        kernel = np.load(cohort / f"kernel_{name}_{arm}.npy", mmap_mode="r")
        rows = np.asarray(kernel[train])
        blocks[name] = (rows[:, train].astype(np.float64), np.asarray(kernel[test])[:, train].astype(np.float64))
        del rows
    design = np.column_stack([np.ones(train.size), covariates[train]])
    for name in ("simple", "structural"):
        project_out_covariates(blocks[name][0], blocks[name][1], design)
    train_kernels = {"simple": blocks["simple"][0],
                     "all": (1 - weight_structural) * blocks["simple"][0] + weight_structural * blocks["structural"][0]}
    return {
        "train": train, "test": test, "covariates": covariates, "weight_structural": weight_structural,
        "observed": np.load(cohort / ARMS[arm][0], mmap_mode="r"), "cls": np.load(cohort / "variants.npz")["cls"],
        "measured": measured_records(cohort), "train_kernel": train_kernels,
        "frobenius": {name: float(np.sum(matrix * matrix)) for name, matrix in train_kernels.items()},
        "cross": {"simple": blocks["simple"][1], "structural": blocks["structural"][1]},
    }


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

    # ridge_inf, two kernel arms: Haseman-Elston h2, then (K + lambda I) alpha = y by Cholesky.
    standardized, phenotype_scale = residualized(truth["phenotype"][train], covariates[train])
    weight_structural = shared["weight_structural"]
    for kernel_arm in ("simple", "all"):
        train_kernel = shared["train_kernel"][kernel_arm]
        diagonal = np.diagonal(train_kernel)
        quadratic = float(standardized @ (train_kernel @ standardized))
        heritability = (quadratic - diagonal @ (standardized * standardized)) / (shared["frobenius"][kernel_arm] - diagonal @ diagonal)
        out = results / f"ridge_inf_{kernel_arm}" / scenario.name
        out.mkdir(parents=True, exist_ok=True)
        # h2 is truncated to its parameter space [0, 1]. At h2 = 0 the prediction is zero; at h2 = 1 the ridge is
        # zero and the training kernel itself is factored (positive definite: more measured records than samples).
        heritability_used = float(np.clip(heritability, 0.0, 1.0))
        if heritability_used == 0.0:
            np.savez(out / "prediction.npz", total=np.zeros(test.size), structural=np.zeros(test.size))
        else:
            ridge = (1.0 - heritability_used) / heritability_used
            factor = cho_factor(train_kernel + ridge * np.eye(train.size), lower=True)
            alpha = cho_solve(factor, standardized)
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

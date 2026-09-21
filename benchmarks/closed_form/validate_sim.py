"""[sim-only] Check the closed-form r^2 against realized estimators on REAL LD (bench-real gene windows, known truth).

For each gene window (training and test genotypes of one split, SNV set), phenotypes are simulated from a known
scale-mixture prior. For each architecture it compares:
- the Bayes-optimal replica prediction `replica.predicted_r2`;
- the posterior mean under a Gaussian prior at the true h^2 (ridge, exact kernel form);
- oracle SuSiE with the true number of effects, slab variance and noise (sparse architectures only);
VAMP itself is not run here: on real cis-LD its iterates overflow (the design is far from right-rotationally
invariant), which the first run recorded.
The realized accuracy is the population rho^2 over the test sample's genotype covariance, so no test-noise sampling
enters. Rows are appended to a JSONL checkpoint, and each (gene, architecture) row has its own seed, so a resumed run
skips finished rows and reproduces the rest.

usage: validate_sim.py DATASET GENE_ROWS(comma) SPLIT REPLICATES OUT.jsonl
"""
import json
import os
import sys

import numpy as np

from benchmarks.bench_real import harness
from benchmarks.closed_form import replica
from benchmarks.closed_form.susie_oracle import susie

ARCHITECTURES = {
    "gaussian_h0.2": dict(heritability=0.2, causal=None),
    "sparse2_h0.1": dict(heritability=0.1, causal=2),
    "sparse2_h0.3": dict(heritability=0.3, causal=2),
    "sparse10_h0.3": dict(heritability=0.3, causal=10),
}


def moments(beta, estimate, test, noise):
    """(cov(yhat, y), var(y), var(yhat)) in the test genotype distribution Sigma_t = X_t^T X_t / n_t (never formed)."""
    signal, fitted = test @ beta, test @ estimate
    count = test.shape[0]
    return float(signal @ fitted) / count, float(signal @ signal) / count + noise, float(fitted @ fitted) / count


def rho2(beta, estimate, test, noise):
    covariance, variance_y, variance_hat = moments(beta, estimate, test, noise)
    return covariance ** 2 / (variance_y * variance_hat) if variance_hat > 0 else 0.0


def ridge(design, kernel, response, noise, variance):
    """Posterior mean under beta ~ N(0, variance I): X^T (X X^T + noise / variance I)^{-1} y."""
    return design.T @ np.linalg.solve(kernel + noise / variance * np.eye(kernel.shape[0]), response)


def finished(out_path):
    if not os.path.exists(out_path):
        return set()
    with open(out_path) as handle:
        return {(row["gene_row"], row["architecture"]) for row in map(json.loads, handle) if row}


def main():
    dataset_dir, gene_rows, split_name, replicates, out_path = sys.argv[1], [int(value) for value in sys.argv[2].split(",")], sys.argv[3], int(sys.argv[4]), sys.argv[5]
    dataset = harness.Dataset(dataset_dir)
    done = finished(out_path)
    for gene_row in gene_rows:
        if all((gene_row, name) in done for name in ARCHITECTURES):
            continue
        window = harness.load_gene_window(dataset, gene_row)
        train_all, test_all, _, _ = harness.build_gene_task(dataset, window, dataset.splits[split_name])
        train, test = harness.subset(train_all, test_all, "snv", split_name)
        raw = np.asarray(train.genotypes, dtype=np.float64)
        center, scale = raw.mean(axis=0), raw.std(axis=0)
        x_train = (raw - center) / scale
        x_test = (np.asarray(test, dtype=np.float64) - center) / scale
        kernel = x_train @ x_train.T
        count, dimension = x_train.shape
        for index, (name, spec) in enumerate(ARCHITECTURES.items()):
            if (gene_row, name) in done:
                continue
            generator = np.random.default_rng([20260920, gene_row, index])
            heritability = spec["heritability"]
            noise = 1.0 - heritability
            if spec["causal"] is None:
                variances, weights = np.array([heritability / dimension]), np.array([1.0])
            else:
                share = spec["causal"] / dimension
                variances = np.array([0.0, heritability / spec["causal"]])
                weights = np.array([1.0 - share, share])
            predicted, detail = replica.predicted_r2(x_train, x_test, heritability, variances, weights)
            # Ridge is linear, so its pooled r^2 depends on the prior only through E[beta beta^T] = (h^2 / p) I: the
            # Gaussian-prior formula at the same h^2 is exact for it under every architecture.
            predicted_ridge, _ = replica.predicted_r2(x_train, x_test, heritability, [1.0], [1.0])
            methods = ["ridge"] + ([] if spec["causal"] is None else ["susie_oracle"])
            realized = {method: [] for method in methods}
            pooled = {method: [] for method in methods}
            for _ in range(replicates):
                component = generator.choice(len(weights), size=dimension, p=weights)
                beta = generator.standard_normal(dimension) * np.sqrt(variances[component])
                response = x_train @ beta + generator.standard_normal(count) * np.sqrt(noise)
                estimates = {"ridge": ridge(x_train, kernel, response, noise, heritability / dimension)}
                if spec["causal"] is not None:
                    estimates["susie_oracle"] = susie(x_train, response, spec["causal"], variances[1], noise)
                for method, estimate in estimates.items():
                    realized[method].append(rho2(beta, estimate, x_test, noise))
                    pooled[method].append(moments(beta, estimate, x_test, noise))
            row = {"gene_row": gene_row, "gene_id": window.gene_id, "split": split_name, "n": count, "p": dimension, "architecture": name,
                   "replicates": replicates, "predicted_bayes_rho2": predicted, "predicted_ridge_pooled_r2": predicted_ridge,
                   "gamma2": detail["gamma2"], "excess_risk": detail["excess_risk"]}
            for method in methods:
                values = np.array(realized[method])
                row[f"{method}_rho2_mean"] = float(values.mean())
                row[f"{method}_rho2_se"] = float(values.std(ddof=1) / np.sqrt(values.size))
                covariance, variance_y, variance_hat = np.mean(np.array(pooled[method]), axis=0)
                row[f"{method}_pooled_r2"] = float(covariance ** 2 / (variance_y * variance_hat)) if variance_hat > 0 else 0.0
            with open(out_path, "a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()

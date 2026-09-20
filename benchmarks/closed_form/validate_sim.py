"""[sim-only] Check the closed-form r^2 against realized estimators on REAL LD (bench-real gene windows, known truth).

For each gene window (training and test genotypes of one split, SNV set), phenotypes are simulated from a known
scale-mixture prior. For each architecture it compares:
- the Bayes-optimal replica prediction `replica.predicted_r2`;
- VAMP with the true prior (the algorithm the prediction describes);
- the posterior mean under a Gaussian prior (ridge at the true h^2);
- mr.ash (bench-real's port of mr.ash.alpha, learning its own mixture).
The realized accuracy is the population rho^2 over the test sample's genotype covariance, so no test-noise sampling
enters.

usage: validate_sim.py DATASET GENE_ROWS(comma) SPLIT REPLICATES OUT.json
"""
import json
import sys
import types

import numpy as np

from benchmarks.bench_real import baselines, harness
from benchmarks.closed_form import replica
from benchmarks.closed_form.vamp import vamp

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


def main():
    dataset_dir, gene_rows, split_name, replicates, out_path = sys.argv[1], [int(value) for value in sys.argv[2].split(",")], sys.argv[3], int(sys.argv[4]), sys.argv[5]
    dataset = harness.Dataset(dataset_dir)
    generator = np.random.default_rng(20260920)
    results = []
    for gene_row in gene_rows:
        window = harness.load_gene_window(dataset, gene_row)
        train_all, test_all, _, _ = harness.build_gene_task(dataset, window, dataset.splits[split_name])
        train, test = harness.subset(train_all, test_all, "snv", split_name)
        raw = np.asarray(train.genotypes, dtype=np.float64)
        center, scale = raw.mean(axis=0), raw.std(axis=0)
        x_train = (raw - center) / scale
        x_test = (np.asarray(test, dtype=np.float64) - center) / scale
        count, dimension = x_train.shape
        for name, spec in ARCHITECTURES.items():
            heritability = spec["heritability"]
            noise = 1.0 - heritability
            if spec["causal"] is None:
                variances, weights = np.array([heritability / dimension]), np.array([1.0])
            else:
                share = spec["causal"] / dimension
                variances = np.array([0.0, heritability / spec["causal"]])
                weights = np.array([1.0 - share, share])
            predicted, detail = replica.predicted_r2(x_train, x_test, heritability, variances, weights)
            realized = {"vamp": [], "ridge": [], "mr_ash": []}
            pooled = {method: [] for method in realized}
            for _ in range(replicates):
                component = generator.choice(len(weights), size=dimension, p=weights)
                beta = generator.standard_normal(dimension) * np.sqrt(variances[component])
                response = x_train @ beta + generator.standard_normal(count) * np.sqrt(noise)
                estimate, _ = vamp(x_train, response, noise, variances, weights, damping=0.5, iterations=300)
                realized["vamp"].append(rho2(beta, estimate, x_test, noise))
                pooled["vamp"].append(moments(beta, estimate, x_test, noise))
                ridge, _ = vamp(x_train, response, noise, np.array([heritability / dimension]), np.array([1.0]), damping=0.0, iterations=3)
                realized["ridge"].append(rho2(beta, ridge, x_test, noise))
                pooled["ridge"].append(moments(beta, ridge, x_test, noise))
                fitted = baselines.mr_ash(types.SimpleNamespace(genotypes=x_train, phenotype=response))
                mr_ash_estimate = np.asarray(fitted.coefficients, dtype=np.float64)
                realized["mr_ash"].append(rho2(beta, mr_ash_estimate, x_test, noise))
                pooled["mr_ash"].append(moments(beta, mr_ash_estimate, x_test, noise))
            row = {"gene_row": gene_row, "gene_id": window.gene_id, "split": split_name, "n": count, "p": dimension, "architecture": name,
                   "predicted_bayes_rho2": predicted, "gamma2": detail["gamma2"], "excess_risk": detail["excess_risk"]}
            for method, values in realized.items():
                row[f"{method}_rho2_mean"] = float(np.mean(values))
                row[f"{method}_rho2_se"] = float(np.std(values, ddof=1) / np.sqrt(len(values)))
                covariance, variance_y, variance_hat = np.mean(np.array(pooled[method]), axis=0)
                row[f"{method}_pooled_r2"] = float(covariance ** 2 / (variance_y * variance_hat)) if variance_hat > 0 else 0.0
            results.append(row)
            print(json.dumps(row), flush=True)
    with open(out_path, "w") as handle:
        json.dump(results, handle, indent=1)


if __name__ == "__main__":
    main()

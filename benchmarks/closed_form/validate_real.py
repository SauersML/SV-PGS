"""[real] Predict GBLUP-REML's held-out r^2 per gene and split from its REML h^2 and the held-out genotypes alone.

Under the Gaussian-prior model GBLUP fits, the posterior mean's expected accuracy in a test population with genotype
covariance Sigma_t is (h^2 tr(Sigma_t)/p - tr(Sigma_t C)) / (h^2 tr(Sigma_t)/p + 1 - h^2), with C the posterior
covariance (replica.predicted_r2 with a one-component prior). This is exact for that model. Comparing it with the
realized r^2 on real expression tests the model's own account of accuracy, including the portability loss that LD
and allele-frequency differences alone predict when a held-out ancestry has a different Sigma_t.

Rows are appended per gene to a JSONL checkpoint; a resumed run skips the genes already written.

usage: validate_real.py DATASET CHROM SPLITS(comma) WORKERS OUT.jsonl
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

from benchmarks.bench_real import baselines, harness
from benchmarks.closed_form import replica

_DATASET = None


def _init(dataset_dir):
    global _DATASET
    _DATASET = harness.Dataset(dataset_dir)


def _gene(arguments):
    gene_row, split_names = arguments
    window = harness.load_gene_window(_DATASET, gene_row)
    rows = []
    for split_name in split_names:
        train_all, test_all, test_phenotype, _ = harness.build_gene_task(_DATASET, window, _DATASET.splits[split_name])
        train, test = harness.subset(train_all, test_all, "snv", split_name)
        predictor = baselines.gblup_reml(train)
        heritability = float(getattr(predictor, "heritability", 0.0))
        prediction = np.asarray(predictor.predict(test), dtype=np.float64)
        centered_prediction, centered_truth = prediction - prediction.mean(), test_phenotype - test_phenotype.mean()
        denominator = np.sqrt(np.sum(centered_prediction ** 2) * np.sum(centered_truth ** 2))
        realized = 0.0 if denominator == 0 else float((centered_prediction @ centered_truth / denominator) ** 2)
        raw = np.asarray(train.genotypes, dtype=np.float64)
        center, scale = raw.mean(axis=0), raw.std(axis=0)
        x_train, x_test = (raw - center) / scale, (np.asarray(test, dtype=np.float64) - center) / scale
        if heritability <= 0.0 or heritability >= 1.0:
            predicted = 0.0
        else:
            predicted, _ = replica.predicted_r2(x_train, x_test, heritability, [1.0], [1.0])
        rows.append({"gene_row": gene_row, "gene_id": window.gene_id, "split": split_name, "n_train": x_train.shape[0], "n_test": x_test.shape[0],
                     "p": x_train.shape[1], "reml_h2": heritability, "predicted_rho2": predicted,
                     "expected_sample_r2": float(replica.expected_sample_r2(predicted, x_test.shape[0])), "realized_r2": realized})
    return rows


def main():
    dataset_dir, chrom, split_names, workers, out_path = sys.argv[1], sys.argv[2], sys.argv[3].split(","), int(sys.argv[4]), sys.argv[5]
    dataset = harness.Dataset(dataset_dir)
    gene_rows = list(dataset.gene_rows([chrom]))
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as handle:
            done = {json.loads(line)["gene_row"] for line in handle if line.strip()}
    remaining = [gene_row for gene_row in gene_rows if gene_row not in done]
    with Pool(workers, initializer=_init, initargs=(dataset_dir,)) as pool, open(out_path, "a") as handle:
        for rows in pool.imap_unordered(_gene, [(gene_row, split_names) for gene_row in remaining]):
            handle.writelines(json.dumps(row) + "\n" for row in rows)
            handle.flush()
    print("genes", len(gene_rows), "already done", len(done), "run", len(remaining))


if __name__ == "__main__":
    main()

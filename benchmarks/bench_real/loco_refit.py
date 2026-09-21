"""Training-side leave-one-carrier-out refits for the SV-driven genes (critique/CRITIQUE_METHOD.md item 5B).

sv_artifacts.py's influence check removes a carrier from the test set only. Here each minor-allele carrier of a gene's
driving SV is removed from every loso training set that holds them. The snv and snv_sv arms are refitted with the
harness's own code and scored the way the benchmark scores them (harness.predict_for_truth, then report.py's
within-group partial r^2), and the gene's pooled gain over the five held-out groups is recomputed; the test sets
don't change. Genes run in increasing order of carrier count, where one person can matter most, until the deadline.
"""
import argparse
import datetime
import pathlib
import time

import numpy as np
import pandas as pd

from benchmarks.bench_real import harness, report


def split_gains(dataset, window, fit, splits: dict):
    """Per loso split: r2 of snv_sv minus r2 of snv on that split's held-out group, under the benchmark's own rule.

    A refit is only comparable with the run it re-examines if it is scored like it: the score is the one
    harness.predict_for_truth compares with the truth, and the metric report.py's within-group partial r^2 against
    [1, C_T] over the group's people. A group [1, C_T] leaves no residual dimension in is left out, as the report
    leaves it out."""
    gains = {}
    for name, split in splits.items():
        train_all, test_all, test_phenotype, test_index = harness.build_gene_task(dataset, window, split)
        basis, rank = report.group_basis(dataset.covariates[test_index])
        if len(test_index) <= rank:
            continue
        truth = report.residual_on(basis, np.asarray(test_phenotype, dtype=np.float64)[None])
        values = {}
        for feature_set in ("snv", "snv_sv"):
            train, test = harness.subset(train_all, test_all, feature_set, name)
            score, _ = harness.predict_for_truth(fit(train), train, test, dataset.covariates[test_index])
            values[feature_set] = float(report.partial_scores(report.residual_on(basis, score[None]), truth)[1][0])
        gains[name] = values["snv_sv"] - values["snv"]
    return gains


def run(dataset_dir, artifacts_path, method_spec, method_name, design, deadline, out_dir):
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dataset = harness.Dataset(dataset_dir)
    fit = harness.load_method(method_spec)
    artifacts = pd.read_csv(artifacts_path, sep="\t")
    artifacts = artifacts[(artifacts["method"] == method_name) & (artifacts["design"] == design)].sort_values("minor_allele_carriers")
    loso = {name: split for name, split in dataset.splits.items() if name.startswith("loso/")}
    row_of_gene = {gene: row for row, gene in enumerate(dataset.genes["gene_id"])}
    rows = []
    for _, gene in artifacts.iterrows():
        if time.time() > deadline:
            break
        window = harness.load_gene_window(dataset, row_of_gene[gene["gene_id"]])
        table, dosage = dataset.chromosome(gene["chrom"])
        sv = np.asarray(dosage[int(np.flatnonzero(table["id"].to_numpy() == gene["sv_id"])[0])], dtype=np.float64)
        carriers = dataset.samples["sample"].to_numpy()[sv > 0 if sv.mean() <= 1.0 else sv < 2]
        base = split_gains(dataset, window, fit, loso)
        for carrier in carriers:
            if time.time() > deadline:
                break
            changed = {name: {**split, "train": [sample for sample in split["train"] if sample != carrier]}
                       for name, split in loso.items() if carrier in split["train"]}
            gains = {**base, **split_gains(dataset, window, fit, changed)}
            rows.append({"gene_id": gene["gene_id"], "gene_name": gene.get("gene_name", ""), "sv_id": gene["sv_id"], "carrier": carrier,
                         "carriers": len(carriers), "gain": float(np.mean(list(base.values()))), "gain_without_carrier": float(np.mean(list(gains.values())))})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / f"loco_refit_{method_name}_{design}.tsv", sep="\t", index=False)
    if len(frame):
        summary = frame.groupby(["gene_id", "gene_name", "sv_id", "carriers", "gain"])["gain_without_carrier"].agg(["min", "max", "count"]).reset_index()
    else:
        summary = pd.DataFrame()
    summary.to_csv(out / f"loco_refit_{method_name}_{design}_summary.tsv", sep="\t", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--artifacts", required=True, help="sv_artifacts.tsv from sv_artifacts.py")
    parser.add_argument("--method", required=True, help="path/to/file.py:callable, as for the harness")
    parser.add_argument("--name", required=True)
    parser.add_argument("--design", default="loso", choices=["loso"])
    parser.add_argument("--deadline", required=True, help="ISO-8601 UTC time after which no new refit starts")
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    deadline = datetime.datetime.fromisoformat(arguments.deadline.replace("Z", "+00:00")).timestamp()
    summary = run(arguments.dataset, arguments.artifacts, arguments.method, arguments.name, arguments.design, deadline, arguments.out)
    with pd.option_context("display.width", 250):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

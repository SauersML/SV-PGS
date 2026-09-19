"""Score out-of-fold predictions from the harness.

Per gene and superpopulation, r^2 = squared Pearson correlation between prediction and the covariate-adjusted
held-out expression (0 when the prediction is constant). Paired differences between two arms are averaged over
genes; their standard error is the delete-one-chromosome jackknife (genes on one chromosome share variants and
trans structure, chromosomes do not), and a gene-level SE is shown when only one chromosome is scored.
"""
import argparse
import itertools
import pathlib

import numpy as np
import pandas as pd

SUPERPOPULATIONS = ("AFR", "AMR", "EAS", "EUR", "SAS")


def squared_correlation(prediction, truth):
    prediction = prediction - prediction.mean()
    truth = truth - truth.mean()
    denominator = np.sqrt((prediction ** 2).sum() * (truth ** 2).sum())
    return 0.0 if denominator == 0 else float((prediction @ truth / denominator) ** 2)


def per_gene_scores(results_dir: pathlib.Path, dataset_dir: pathlib.Path, method: str, design: str):
    samples = pd.read_csv(dataset_dir / "samples.tsv", sep="\t")
    frames = []
    for genes_file in sorted((results_dir / method / design).glob("*.genes.tsv")):
        tag = genes_file.name.removesuffix(".genes.tsv")
        genes = pd.read_csv(genes_file, sep="\t")
        truth = np.load(results_dir / method / design / f"{tag}.truth.npy")
        for feature_set in ("snv", "snv_sv", "snv_pgsv"):
            path = results_dir / method / design / f"{tag}.{feature_set}.predictions.npy"
            if not path.exists():
                continue
            predictions = np.load(path)
            for superpopulation in SUPERPOPULATIONS:
                members = np.flatnonzero(samples["Superpopulation"].to_numpy() == superpopulation)
                scores = [squared_correlation(predictions[row, members].astype(np.float64), truth[row, members].astype(np.float64)) for row in range(len(genes))]
                frames.append(pd.DataFrame({"gene_id": genes["gene_id"], "chrom": genes["chrom"], "method": method, "feature_set": feature_set,
                                            "design": design, "superpopulation": superpopulation, "r2": scores}))
    return pd.concat(frames, ignore_index=True)


def jackknife(differences: pd.Series, blocks: pd.Series):
    labels = blocks.unique()
    if len(labels) < 2:
        return float(differences.mean()), float(differences.std(ddof=1) / np.sqrt(len(differences))), "gene-level"
    estimates = np.array([differences[blocks != label].mean() for label in labels])
    count = len(labels)
    return float(differences.mean()), float(np.sqrt((count - 1) / count * ((estimates - estimates.mean()) ** 2).sum())), "chromosome jackknife"


def paired(scores: pd.DataFrame, arm_a, arm_b):
    rows = []
    key = ["gene_id", "chrom", "design", "superpopulation"]
    left = scores[(scores["method"] == arm_a[0]) & (scores["feature_set"] == arm_a[1])][key + ["r2"]]
    right = scores[(scores["method"] == arm_b[0]) & (scores["feature_set"] == arm_b[1])][key + ["r2"]]
    merged = left.merge(right, on=key, suffixes=("_a", "_b"))
    for (design, superpopulation), group in merged.groupby(["design", "superpopulation"]):
        mean, error, kind = jackknife(group["r2_a"] - group["r2_b"], group["chrom"])
        rows.append({"arm_a": "/".join(arm_a), "arm_b": "/".join(arm_b), "design": design, "superpopulation": superpopulation, "genes": len(group),
                     "mean_r2_a": group["r2_a"].mean(), "mean_r2_b": group["r2_b"].mean(), "difference": mean, "se": error, "se_kind": kind})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    results_dir, dataset_dir = pathlib.Path(arguments.results), pathlib.Path(arguments.dataset)
    scores = pd.concat([per_gene_scores(results_dir, dataset_dir, method, design) for method in arguments.methods for design in ("random5", "loso")
                        if (results_dir / method / design).exists()], ignore_index=True)
    scores.to_csv(pathlib.Path(arguments.out) / "per_gene_r2.tsv.gz", sep="\t", index=False)
    present = set(zip(scores["method"], scores["feature_set"]))
    arms = [(method, feature_set) for method in arguments.methods for feature_set in ("snv", "snv_sv", "snv_pgsv") if (method, feature_set) in present]
    comparisons = []
    for method in arguments.methods:
        for feature_set in ("snv_sv", "snv_pgsv"):
            if (method, feature_set) in present and (method, "snv") in present:
                comparisons += paired(scores, (method, feature_set), (method, "snv"))
    for arm_a, arm_b in itertools.combinations(arms, 2):
        if arm_a[1] == arm_b[1] and arm_a[0] != arm_b[0]:
            comparisons += paired(scores, arm_a, arm_b)
    table = pd.DataFrame(comparisons)
    table.to_csv(pathlib.Path(arguments.out) / "paired_differences.tsv", sep="\t", index=False)
    summary = scores.groupby(["design", "superpopulation", "method", "feature_set"])["r2"].agg(["mean", "count"]).reset_index()
    summary.to_csv(pathlib.Path(arguments.out) / "mean_r2.tsv", sep="\t", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 500, "display.float_format", "{:.5f}".format):
        print(summary.pivot_table(index=["design", "superpopulation"], columns=["method", "feature_set"], values="mean"))
        print(table)


if __name__ == "__main__":
    main()

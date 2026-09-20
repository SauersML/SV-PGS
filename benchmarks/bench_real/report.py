"""Score out-of-fold predictions from the harness.

Per gene and superpopulation, r^2 = squared Pearson correlation between prediction and the covariate-adjusted
held-out expression (0 when the prediction is constant). Paired differences between two arms are averaged over
genes; their standard error is the delete-one-chromosome jackknife (genes on one chromosome share variants and
trans structure, chromosomes do not), and a gene-level SE is shown when only one chromosome is scored.

Headline: both designs hold every person out exactly once, so the five superpopulations are pooled into one test.
Per gene, r^2 and paired differences are averaged over the groups and covariances are summed over them; the result
is one estimate per method and arm with one chromosome-jackknife SE, reported under superpopulation "pooled". The
per-group rows follow as secondary detail (the African-ancestry drop under loso is the per-group result that matters).

SV credit, per method and feature set with SV columns: the harness also predicts with every SV column held at its
training mean, so the SV part of a prediction is prediction - prediction_without_sv. The credit is that part's
share of the held-out covariance with expression, summed over genes (an exact additive split for linear
predictors), and the r^2 lost when SVs are held at their means.
"""
import argparse
import itertools
import pathlib

import numpy as np
import pandas as pd

SUPERPOPULATIONS = ("AFR", "AMR", "EAS", "EUR", "SAS")
POOLED = "pooled"
FEATURE_SETS = ("snv", "snv_sv", "snv_pgsv", "sv", "pgsv", "snv_matched", "hgsvc3", "snv_hgsvc3", "ont", "snv_ont")
JOINT_SETS = ("snv_sv", "snv_pgsv", "snv_hgsvc3", "snv_ont")
# Within a method: adding each SV source to SNVs, each long-read source against the panel SVs, and SVs alone against an
# equal number of matched SNVs (and against each other).
WITHIN_METHOD_COMPARISONS = (("snv_sv", "snv"), ("snv_pgsv", "snv"), ("snv_hgsvc3", "snv"), ("snv_ont", "snv"), ("snv_hgsvc3", "snv_sv"),
                             ("snv_ont", "snv_sv"), ("sv", "snv_matched"), ("pgsv", "snv_matched"), ("sv", "pgsv"))


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
        for feature_set in FEATURE_SETS:
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
    pooled = merged.groupby(["gene_id", "chrom", "design"], as_index=False)[["r2_a", "r2_b"]].mean().assign(superpopulation=POOLED)
    for (design, superpopulation), group in pd.concat([pooled, merged], ignore_index=True).groupby(["design", "superpopulation"], sort=False):
        mean, error, kind = jackknife(group["r2_a"] - group["r2_b"], group["chrom"])
        rows.append({"arm_a": "/".join(arm_a), "arm_b": "/".join(arm_b), "design": design, "superpopulation": superpopulation, "genes": len(group),
                     "mean_r2_a": group["r2_a"].mean(), "mean_r2_b": group["r2_b"].mean(), "difference": mean, "se": error, "se_kind": kind})
    return rows


def pooled_r2(scores: pd.DataFrame):
    """Per method, feature set and design: the mean over genes of each gene's r^2 averaged over the held-out groups."""
    per_gene = scores.groupby(["method", "feature_set", "design", "gene_id", "chrom"], as_index=False)["r2"].mean()
    rows = []
    for key, group in per_gene.groupby(["method", "feature_set", "design"]):
        mean, error, kind = jackknife(group["r2"], group["chrom"])
        rows.append(dict(zip(["method", "feature_set", "design"], key), superpopulation=POOLED, genes=len(group), mean_r2=mean, se=error, se_kind=kind))
    return pd.DataFrame(rows)


def covariance(first, second):
    return float(((first - first.mean()) * (second - second.mean())).sum())


def sv_credit(results_dir: pathlib.Path, dataset_dir: pathlib.Path, method: str, design: str):
    samples = pd.read_csv(dataset_dir / "samples.tsv", sep="\t")
    rows = []
    for genes_file in sorted((results_dir / method / design).glob("*.genes.tsv")):
        tag = genes_file.name.removesuffix(".genes.tsv")
        genes = pd.read_csv(genes_file, sep="\t")
        truth = np.load(results_dir / method / design / f"{tag}.truth.npy").astype(np.float64)
        for feature_set in JOINT_SETS:
            full_path = results_dir / method / design / f"{tag}.{feature_set}.predictions.npy"
            masked_path = results_dir / method / design / f"{tag}.{feature_set}.predictions_without_sv.npy"
            if not (full_path.exists() and masked_path.exists()):
                continue
            full, masked = np.load(full_path).astype(np.float64), np.load(masked_path).astype(np.float64)
            for superpopulation in SUPERPOPULATIONS:
                members = np.flatnonzero(samples["Superpopulation"].to_numpy() == superpopulation)
                for row in range(len(genes)):
                    observed, predicted, predicted_without = truth[row, members], full[row, members], masked[row, members]
                    rows.append({"gene_id": genes["gene_id"].iloc[row], "chrom": genes["chrom"].iloc[row], "method": method, "feature_set": feature_set,
                                 "design": design, "superpopulation": superpopulation, "covariance_full": covariance(observed, predicted),
                                 "covariance_sv": covariance(observed, predicted - predicted_without),
                                 "r2_drop": squared_correlation(predicted, observed) - squared_correlation(predicted_without, observed)})
    return pd.DataFrame(rows)


def summarize_sv_credit(credit: pd.DataFrame):
    pooled = credit.groupby(["method", "feature_set", "design", "gene_id", "chrom"], as_index=False).agg(
        covariance_full=("covariance_full", "sum"), covariance_sv=("covariance_sv", "sum"), r2_drop=("r2_drop", "mean")).assign(superpopulation=POOLED)
    rows = []
    for key, group in pd.concat([pooled, credit], ignore_index=True).groupby(["method", "feature_set", "design", "superpopulation"], sort=False):
        chromosomes = group["chrom"].unique()
        share = group["covariance_sv"].sum() / group["covariance_full"].sum()
        if len(chromosomes) > 1:
            leave_out = np.array([group.loc[group["chrom"] != label, "covariance_sv"].sum() / group.loc[group["chrom"] != label, "covariance_full"].sum() for label in chromosomes])
            share_se = float(np.sqrt((len(chromosomes) - 1) / len(chromosomes) * ((leave_out - leave_out.mean()) ** 2).sum()))
        else:
            share_se = float("nan")
        drop, drop_se, kind = jackknife(group["r2_drop"], group["chrom"])
        rows.append(dict(zip(["method", "feature_set", "design", "superpopulation"], key), genes=len(group), sv_share_of_covariance=share,
                         share_se=share_se, r2_drop=drop, r2_drop_se=drop_se, se_kind=kind))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", required=True, help="one or more results directories; each (method, design, feature set) must come from one")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    results_dirs, dataset_dir = [pathlib.Path(path) for path in arguments.results], pathlib.Path(arguments.dataset)
    runs = [(results_dir, method, design) for results_dir in results_dirs for method in arguments.methods for design in ("random5", "loso")
            if (results_dir / method / design).exists()]
    scores = pd.concat([per_gene_scores(results_dir, dataset_dir, method, design).assign(results=str(results_dir)) for results_dir, method, design in runs],
                       ignore_index=True)
    sources = scores.groupby(["method", "design", "feature_set"])["results"].nunique()
    if (sources > 1).any():
        raise ValueError(f"an arm appears in more than one results directory: {list(sources[sources > 1].index)}")
    scores = scores.drop(columns="results")
    scores.to_csv(pathlib.Path(arguments.out) / "per_gene_r2.tsv.gz", sep="\t", index=False)
    present = set(zip(scores["method"], scores["feature_set"]))
    arms = [(method, feature_set) for method in arguments.methods for feature_set in FEATURE_SETS if (method, feature_set) in present]
    comparisons = []
    for method in arguments.methods:
        for feature_set, baseline in WITHIN_METHOD_COMPARISONS:
            if (method, feature_set) in present and (method, baseline) in present:
                comparisons += paired(scores, (method, feature_set), (method, baseline))
    for arm_a, arm_b in itertools.combinations(arms, 2):
        if arm_a[1] == arm_b[1] and arm_a[0] != arm_b[0]:
            comparisons += paired(scores, arm_a, arm_b)
    table = pd.DataFrame(comparisons)
    table.to_csv(pathlib.Path(arguments.out) / "paired_differences.tsv", sep="\t", index=False)
    credit = pd.concat([sv_credit(results_dir, dataset_dir, method, design) for results_dir, method, design in runs], ignore_index=True)
    if len(credit):
        summarize_sv_credit(credit).to_csv(pathlib.Path(arguments.out) / "sv_credit.tsv", sep="\t", index=False)
    summary = scores.groupby(["design", "superpopulation", "method", "feature_set"])["r2"].agg(["mean", "count"]).reset_index()
    summary.to_csv(pathlib.Path(arguments.out) / "mean_r2.tsv", sep="\t", index=False)
    headline = pooled_r2(scores)
    headline.to_csv(pathlib.Path(arguments.out) / "pooled_r2.tsv", sep="\t", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 500, "display.float_format", "{:.5f}".format):
        print("pooled held-out r^2 (headline)")
        print(headline.to_string(index=False))
        print("pooled paired differences (headline)")
        print(table[table["superpopulation"] == POOLED].to_string(index=False))
        print("per held-out group")
        print(summary.pivot_table(index=["design", "superpopulation"], columns=["method", "feature_set"], values="mean"))


if __name__ == "__main__":
    main()

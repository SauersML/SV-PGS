"""SNVs alone against SNVs + SVs on held-out expression, per method, with leave-one-chromosome-out intervals.

Reads one or more per_gene_r2.tsv.gz tables written by report.py (the within-group partial r^2, one row per gene,
method, feature set, design and held-out group). Per method and design, a gene's r^2 under each arm is its mean over
the held-out groups (report.py's pooled headline), or one group's value in the per-group rows; the SV gain is the
joint arm's r^2 minus the baseline's, on the genes both arms completed.

Each mean over genes gets two 95% normal intervals:
  loco  the delete-one-chromosome jackknife weighted for unequal chromosome sizes (report.weighted_jackknife). Genes
        of one chromosome share variants, LD and local trans structure, so the chromosome is the resampling unit.
  gene  the gene-level bootstrap, which treats genes as independent. For a mean its variance is known exactly,
        (1/n^2) sum (x_i - mean)^2, so no resampling (and no replicate count) is needed. It is reported only to show
        what ignoring the chromosome structure would claim.
"""
import argparse
import pathlib

import numpy as np
import pandas as pd
from scipy import stats

from benchmarks.bench_real import report

Z = float(stats.norm.ppf(0.975))


def interval(values: pd.Series, blocks: pd.Series) -> dict:
    """The mean of values over genes with its chromosome-jackknife and gene-bootstrap 95% intervals."""
    mean, loco_se, kind = report.jackknife(values, blocks)
    boot_se = float(np.sqrt(((values - mean) ** 2).sum()) / len(values))
    return {"genes": len(values), "chromosomes": int(blocks.nunique()), "mean": mean, "loco_se": loco_se, "loco_lo": mean - Z * loco_se,
            "loco_hi": mean + Z * loco_se, "loco_kind": kind, "boot_se": boot_se, "boot_lo": mean - Z * boot_se, "boot_hi": mean + Z * boot_se}


def paired_genes(scores: pd.DataFrame, baseline: str, joint: str) -> pd.DataFrame:
    """Per method, design, held-out group (and the pooled mean over groups) and gene: the r^2 of both arms and the gain,
    on genes where both arms completed in every group."""
    key = ["method", "design", "superpopulation", "gene_id", "chrom"]
    arms = scores[scores["feature_set"].isin([baseline, joint])]
    wide = arms.set_index(key + ["feature_set"])["r2"].unstack("feature_set").reindex(columns=[baseline, joint]).reset_index()
    failed = wide[[baseline, joint]].isna().any(axis=1)
    bad = set(map(tuple, wide.loc[failed, ["method", "design", "gene_id"]].to_numpy()))
    wide = wide[[tuple(row) not in bad for row in wide[["method", "design", "gene_id"]].to_numpy()]]
    pooled = wide.groupby(["method", "design", "gene_id", "chrom"], as_index=False)[[baseline, joint]].mean().assign(superpopulation=report.POOLED)
    both = pd.concat([pooled, wide], ignore_index=True)
    return both.assign(gain=both[joint] - both[baseline])


def summarize(genes: pd.DataFrame, baseline: str, joint: str) -> pd.DataFrame:
    rows = []
    for (method, design, superpopulation), group in genes.groupby(["method", "design", "superpopulation"], sort=False):
        for quantity in (baseline, joint, "gain"):
            rows.append({"method": method, "design": design, "superpopulation": superpopulation, "quantity": quantity,
                         **interval(group[quantity].reset_index(drop=True), group["chrom"].reset_index(drop=True))})
    return pd.DataFrame(rows)


def per_chromosome(genes: pd.DataFrame) -> pd.DataFrame:
    pooled = genes[genes["superpopulation"] == report.POOLED]
    return pooled.groupby(["method", "design", "chrom"], as_index=False).agg(genes=("gain", "size"), mean_gain=("gain", "mean"))


def load(tables, methods=None, genes_file=None) -> pd.DataFrame:
    scores = pd.concat([pd.read_csv(path, sep="\t") for path in tables], ignore_index=True)
    if methods:
        scores = scores[scores["method"].isin(methods)]
    if genes_file:
        scores = scores[scores["gene_id"].isin(set(pd.read_csv(genes_file, sep="\t")["gene_id"]))]
    identity = ["method", "design", "feature_set", "superpopulation", "gene_id"]
    if scores.duplicated(identity).any():
        raise ValueError("a gene is scored more than once for one arm across the tables")
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", nargs="+", required=True, help="report.py per_gene_r2.tsv.gz tables")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--genes", help="a table with a gene_id column: score only these genes")
    parser.add_argument("--common", action="store_true", help="keep only genes every method completed, so the methods share one gene set")
    parser.add_argument("--baseline", default="snv")
    parser.add_argument("--joint", default="snv_sv")
    parser.add_argument("--design", default="loso")
    parser.add_argument("--out", required=True, help="output prefix")
    arguments = parser.parse_args()
    scores = load(arguments.scores, arguments.methods, arguments.genes)
    genes = paired_genes(scores[scores["design"] == arguments.design], arguments.baseline, arguments.joint)
    if arguments.common:
        pooled = genes[genes["superpopulation"] == report.POOLED]
        shared = set.intersection(*[set(group["gene_id"]) for _, group in pooled.groupby("method")])
        genes = genes[genes["gene_id"].isin(shared)]
    out = pathlib.Path(arguments.out)
    genes.to_csv(f"{out}.genes.tsv.gz", sep="\t", index=False)
    summary = summarize(genes, arguments.baseline, arguments.joint)
    summary.to_csv(f"{out}.summary.tsv", sep="\t", index=False)
    per_chromosome(genes).to_csv(f"{out}.per_chromosome.tsv", sep="\t", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 500, "display.float_format", "{:.5f}".format):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

"""Within-ancestry permutation null for the SV gain of bench-real's top_variant baseline (critique/CRITIQUE_METHOD.md
item 5, EVALUATION.md's C-null).

Each replicate permutes people within every superpopulation and applies that one permutation to every panel SV column
of every gene, in training and test alike. That keeps each SV's allele frequency per ancestry and the SV-SV linkage, and
breaks the SVs' linkage to expression and to the SNVs. The snv_sv arm is then refitted and scored exactly as the harness
does, on the loso splits; the snv arm doesn't depend on the SV columns. Under loso a within-superpopulation permutation
never moves a person between training and test, so no training column changes its polymorphism.

top_variant is exact and cheap under this null. Permuting column x's rows by p leaves its variance unchanged, and
corr(x o p, y) = corr(x, y o p^-1), so every replicate's SV correlations are one matrix product against permuted
phenotypes. When an SV wins the selection, its test r2 is corr(x_test, y_test o p^-1)^2, because r2 doesn't depend on
the slope. The identity permutation reproduces the harness's own top_variant snv_sv predictions; the run checks this.
"""
import argparse
import hashlib
import json
import pathlib
from multiprocessing import get_context

import numpy as np
import pandas as pd

from benchmarks.bench_real import harness, robust

GROUPS = ("AFR", "AMR", "EAS", "EUR", "SAS")


def seed(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")


def within_group_permutations(groups: np.ndarray, count: int, generator: np.random.Generator):
    """(count + 1, n) inverse permutations; row 0 is the identity. Each maps a person to someone in their own group."""
    inverse = np.tile(np.arange(len(groups)), (count + 1, 1))
    for group in np.unique(groups):
        members = np.flatnonzero(groups == group)
        for row in range(1, count + 1):
            inverse[row, members] = members[generator.permutation(len(members))]
    return inverse


def column_correlations(genotypes: np.ndarray, phenotypes: np.ndarray):
    """corr of every column with every phenotype column: genotypes (n, p), phenotypes (n, k) -> (p, k)."""
    centred = genotypes - genotypes.mean(axis=0)
    responses = phenotypes - phenotypes.mean(axis=0)
    numerator = centred.T @ responses
    return numerator / np.sqrt(np.outer((centred ** 2).sum(axis=0), (responses ** 2).sum(axis=0)))


def gene_null(dataset, gene_row: int, inverse: np.ndarray):
    """Per loso split: (observed and null) r2 gain of top_variant snv_sv over snv, and whether an SV won the selection."""
    window = harness.load_gene_window(dataset, gene_row)
    gains, wins = [], []
    for name in sorted(split for split in dataset.splits if split.startswith("loso/")):
        train_all, test_all, test_phenotype, test_index = harness.build_gene_task(dataset, window, dataset.splits[name])
        train, test = harness.subset(train_all, test_all, "snv_sv", name)
        train_index = np.array([dataset.sample_index[sample] for sample in dataset.splits[name]["train"]])
        is_sv = train.variants.is_sv
        genotypes, phenotype = np.asarray(train.genotypes, dtype=np.float64), train.phenotype
        test_genotypes = np.asarray(test, dtype=np.float64)
        snv_columns = np.flatnonzero(~is_sv)
        snv_corr = column_correlations(genotypes[:, snv_columns], phenotype[:, None])[:, 0]
        best_snv = snv_columns[int(np.argmax(np.abs(snv_corr)))] if len(snv_columns) else None
        best_snv_abs = float(np.max(np.abs(snv_corr))) if len(snv_columns) else -np.inf
        snv_r2 = float(np.nan_to_num(column_correlations(test_genotypes[:, [best_snv]], test_phenotype[:, None])[0, 0] ** 2)) if best_snv is not None else 0.0
        sv_columns = np.flatnonzero(is_sv)
        if len(sv_columns) == 0:
            gains.append(np.zeros(inverse.shape[0]))
            wins.append(np.zeros(inverse.shape[0], dtype=bool))
            continue
        position = np.empty(len(dataset.samples), dtype=int)
        position[train_index] = np.arange(len(train_index))
        train_permuted = phenotype[position[inverse[:, train_index]]].T
        position[test_index] = np.arange(len(test_index))
        test_permuted = test_phenotype[position[inverse[:, test_index]]].T
        sv_corr = np.abs(column_correlations(genotypes[:, sv_columns], train_permuted))
        winner = np.argmax(sv_corr, axis=0)
        sv_wins = sv_corr[winner, np.arange(inverse.shape[0])] > best_snv_abs
        if best_snv is not None:
            ties = sv_corr[winner, np.arange(inverse.shape[0])] == best_snv_abs
            sv_wins |= ties & (sv_columns[winner] < best_snv)
        test_r2 = np.nan_to_num(column_correlations(test_genotypes[:, sv_columns], test_permuted) ** 2)  # a test-constant column predicts a constant: r2 0
        gain = np.where(sv_wins, test_r2[winner, np.arange(inverse.shape[0])] - snv_r2, 0.0)
        gains.append(gain)
        wins.append(sv_wins)
    return window.gene_id, window.chrom, np.mean(gains, axis=0), np.sum(wins, axis=0)


_STATE = {}


def _init(dataset_dir, inverse):
    _STATE["dataset"] = harness.Dataset(dataset_dir)
    _STATE["inverse"] = inverse


def _work(gene_row):
    return gene_null(_STATE["dataset"], gene_row, _STATE["inverse"])


def run(dataset_dir, gene_prefix, permutations, workers, observed_dir, out_dir):
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dataset = harness.Dataset(dataset_dir)
    inverse = within_group_permutations(dataset.samples["Superpopulation"].to_numpy(), permutations,
                                        np.random.default_rng(seed("bench-real/perm_null/top_variant")))
    rows = dataset.gene_rows([f"chr{number}" for number in range(1, 23)], gene_prefix)
    results = []
    with get_context("fork").Pool(workers, initializer=_init, initargs=(dataset_dir, inverse)) as pool:
        for result in pool.imap_unordered(_work, rows, chunksize=4):
            results.append(result)
    results.sort(key=lambda item: item[0])
    genes = pd.DataFrame({"gene_id": [item[0] for item in results], "chrom": [item[1] for item in results]})
    gains = np.stack([item[2] for item in results])
    wins = np.stack([item[3] for item in results])
    np.save(out / "gains.npy", gains)
    np.save(out / "sv_wins.npy", wins)
    genes.to_csv(out / "genes.tsv", sep="\t", index=False)
    pooled = gains.mean(axis=0)
    null = pooled[1:]
    per_gene_p = (1 + (gains[:, 1:] >= gains[:, :1]).sum(axis=1)) / (permutations + 1)
    genes.assign(observed_gain=gains[:, 0], null_mean=gains[:, 1:].mean(axis=1), permutation_p=per_gene_p,
                 observed_sv_wins=wins[:, 0], null_sv_wins_mean=wins[:, 1:].mean(axis=1)).to_csv(out / "per_gene.tsv", sep="\t", index=False)
    summary = {"genes": len(genes), "permutations": permutations, "observed_pooled_gain": float(pooled[0]),
               "null_pooled_gain_mean": float(null.mean()), "null_pooled_gain_sd": float(null.std(ddof=1)),
               "pooled_p_one_sided": float((1 + (null >= pooled[0]).sum()) / (permutations + 1)),
               "observed_gene_split_sv_wins": int(wins[:, 0].sum()), "null_gene_split_sv_wins_mean": float(wins[:, 1:].sum(axis=0).mean()),
               "null_gene_split_sv_wins_max": int(wins[:, 1:].sum(axis=0).max()),
               "genes_with_min_attainable_p": int((per_gene_p == 1 / (permutations + 1)).sum())}
    if observed_dir is not None:
        directory = pathlib.Path(observed_dir) / "top_variant" / "loso"
        truth_gain = {}
        for genes_file in directory.glob("*.genes.tsv"):
            tag = genes_file.name.removesuffix(".genes.tsv")
            table = pd.read_csv(genes_file, sep="\t")
            truth = np.load(directory / f"{tag}.truth.npy").astype(np.float64)
            full = np.load(directory / f"{tag}.snv_sv.predictions.npy").astype(np.float64)
            snv = np.load(directory / f"{tag}.snv.predictions.npy").astype(np.float64)
            groups = dataset.samples["Superpopulation"].to_numpy()
            for row, gene_id in enumerate(table["gene_id"]):
                values = []
                for group in GROUPS:
                    index = np.flatnonzero(groups == group)
                    ones = np.ones((1, len(index)))
                    values.append(robust.group_metrics(ones, full[row:row + 1, index], truth[row:row + 1, index])["r2"][0, 0]
                                  - robust.group_metrics(ones, snv[row:row + 1, index], truth[row:row + 1, index])["r2"][0, 0])
                truth_gain[gene_id] = float(np.mean(values))
        matched = [(gain, truth_gain[gene]) for gene, gain in zip(genes["gene_id"], gains[:, 0]) if gene in truth_gain]
        summary["identity_vs_harness_max_abs_difference"] = float(max(abs(a - b) for a, b in matched)) if matched else None
        summary["identity_vs_harness_genes"] = len(matched)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--gene-prefix", type=int, required=True)
    parser.add_argument("--permutations", type=int, required=True, help="the one-sided p value's resolution is 1/(permutations + 1)")
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--observed", help="bench-real results directory, to check the identity permutation against the harness")
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.dataset, arguments.gene_prefix, arguments.permutations, arguments.workers, arguments.observed, arguments.out), indent=1))


if __name__ == "__main__":
    main()

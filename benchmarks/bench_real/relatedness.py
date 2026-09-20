"""Relatedness among the MAGE samples, and the fold checks it drives (critique/CRITIQUE_REAL.md item 5).

Kinship is KING-robust, the between-family estimator of Manichaikul et al. 2010 (Bioinformatics 26:2867). It is robust to
population structure because it normalizes by the less heterozygous member of each pair:
  phi_ij = [2 (N_Aa,Aa - 2 N_AA,aa) + N_i - N_j] / (4 N_i),   N_i = min(het_i, het_j), N_j = max(het_i, het_j),
where N_Aa,Aa counts sites where both are heterozygous, N_AA,aa sites with opposite homozygotes, and het_i is i's
heterozygous site count. It uses the bench-real dataset's own small variants (SNVs and indels < 50 bp, every autosome),
as hard calls.

Degrees follow the same paper: a d-th degree pair has expected kinship 2^-(d+1), and the class boundaries are the
geometric midpoints 2^-(d+1.5): > 2^-1.5 duplicate or MZ twin, > 2^-2.5 first degree, > 2^-3.5 second, > 2^-4.5 third.
Pairs below 2^-4.5 are what KING calls unrelated. The SE of each kinship is the delete-one-chromosome jackknife.
"""
import argparse
import json
import pathlib

import numpy as np
import pandas as pd

AUTOSOMES = tuple(f"chr{number}" for number in range(1, 23))
DEGREES = {0: 2.0 ** -1.5, 1: 2.0 ** -2.5, 2: 2.0 ** -3.5, 3: 2.0 ** -4.5}


def degree(kinship: float):
    """0 (duplicate or MZ) to 3 (third degree), or None below KING's third-degree boundary."""
    for level, boundary in DEGREES.items():
        if kinship > boundary:
            return level
    return None


def king_counts(dosage: np.ndarray, rows: np.ndarray, chunk: int):
    """(both heterozygous, opposite homozygotes, heterozygous per sample) summed over the given variant rows."""
    samples = dosage.shape[1]
    both = np.zeros((samples, samples))
    opposite = np.zeros((samples, samples))
    heterozygous = np.zeros(samples)
    for start in range(0, len(rows), chunk):
        block = np.asarray(dosage[rows[start:start + chunk]])
        het = (block == 1).astype(np.float32)
        alt = (block == 2).astype(np.float32)
        ref = (block == 0).astype(np.float32)
        both += (het.T @ het).astype(np.float64)
        cross = (alt.T @ ref).astype(np.float64)
        opposite += cross + cross.T
        heterozygous += het.sum(axis=0)
    return both, opposite, heterozygous


def king_robust(both: np.ndarray, opposite: np.ndarray, heterozygous: np.ndarray):
    smaller = np.minimum.outer(heterozygous, heterozygous)
    larger = np.maximum.outer(heterozygous, heterozygous)
    with np.errstate(divide="ignore", invalid="ignore"):
        kinship = (2.0 * (both - 2.0 * opposite) + smaller - larger) / (4.0 * smaller)
    return kinship


def kinship_by_chromosome(dataset_dir: pathlib.Path, chunk: int):
    """Per-autosome KING counts, so a genome-wide kinship and its chromosome jackknife come from one pass."""
    counts = {}
    for chrom in AUTOSOMES:
        table = pd.read_csv(dataset_dir / f"{chrom}.variants.tsv", sep="\t", usecols=lambda column: column in {"is_sv", "source"})
        small = ~table["is_sv"].to_numpy(dtype=bool)
        if "source" in table:
            small &= table["source"].to_numpy() == "panel"
        dosage = np.load(dataset_dir / f"{chrom}.dosage.npy", mmap_mode="r")
        counts[chrom] = king_counts(dosage, np.flatnonzero(small), chunk)
    return counts


def genome_kinship(counts: dict, leave_out=None):
    kept = [value for chrom, value in counts.items() if chrom != leave_out]
    return king_robust(*(sum(part[index] for part in kept) for index in range(3)))


def related_pairs(kinship: np.ndarray, jackknife: list, samples: pd.DataFrame):
    """Every pair above KING's third-degree boundary, with its degree and jackknife SE."""
    upper = np.triu_indices(len(samples), k=1)
    values = kinship[upper]
    keep = values > DEGREES[3]
    first, second = upper[0][keep], upper[1][keep]
    count = len(jackknife)
    leave = np.stack([matrix[first, second] for matrix in jackknife])
    se = np.sqrt((count - 1) / count * ((leave - leave.mean(axis=0)) ** 2).sum(axis=0))
    table = pd.DataFrame({"sample_a": samples["sample"].to_numpy()[first], "sample_b": samples["sample"].to_numpy()[second],
                          "kinship": values[keep], "kinship_se": se})
    table["degree"] = [degree(value) for value in table["kinship"]]
    lookup = samples.set_index("sample")
    for column in ("FamilyID", "Superpopulation", "Population"):
        table[f"{column}_a"] = lookup.loc[table["sample_a"], column].to_numpy()
        table[f"{column}_b"] = lookup.loc[table["sample_b"], column].to_numpy()
    table["same_pedigree_family"] = table["FamilyID_a"] == table["FamilyID_b"]
    return table.sort_values("kinship", ascending=False).reset_index(drop=True)


def fold_of_sample(splits: list, design: str):
    fold = {}
    for split in splits:
        if split["name"].startswith(design + "/"):
            for sample in split["test"]:
                fold[sample] = split["name"]
    return fold


def cross_fold_report(pairs: pd.DataFrame, splits: list):
    """Related pairs whose members are in different test folds, per design and degree."""
    report = {}
    for design in ("random5", "loso"):
        fold = fold_of_sample(splits, design)
        crossing = pairs[[fold[a] != fold[b] for a, b in zip(pairs["sample_a"], pairs["sample_b"])]]
        report[design] = {"related_pairs": int(len(pairs)), "pairs_across_folds": int(len(crossing)),
                          "across_folds_by_degree": {str(level): int((crossing["degree"] == level).sum()) for level in DEGREES},
                          "samples_with_a_relative_in_another_fold": int(len(set(crossing["sample_a"]) | set(crossing["sample_b"])))}
    return report


def relatedness_clusters(samples: pd.DataFrame, pairs: pd.DataFrame):
    """Connected components of pedigree families joined by every KING-related pair (union-find)."""
    parent = {family: family for family in samples["FamilyID"]}

    def find(label):
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    for first, second in zip(pairs["FamilyID_a"], pairs["FamilyID_b"]):
        root_a, root_b = find(first), find(second)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)
    return samples["FamilyID"].map(find)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--chunk", type=int, required=True, help="variants per matrix product (memory only)")
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    dataset_dir, out = pathlib.Path(arguments.dataset), pathlib.Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    samples = pd.read_csv(dataset_dir / "samples.tsv", sep="\t")
    splits = json.loads((dataset_dir / "splits.json").read_text())
    counts = kinship_by_chromosome(dataset_dir, arguments.chunk)
    kinship = genome_kinship(counts)
    jackknife = [genome_kinship(counts, leave_out=chrom) for chrom in AUTOSOMES]
    np.save(out / "king_robust_kinship.npy", kinship)
    pairs = related_pairs(kinship, jackknife, samples)
    pairs.to_csv(out / "related_pairs.tsv", sep="\t", index=False)
    clusters = relatedness_clusters(samples, pairs)
    samples.assign(relatedness_cluster=clusters).to_csv(out / "samples_with_clusters.tsv", sep="\t", index=False)
    report = {"pairs_by_degree": {str(level): int((pairs["degree"] == level).sum()) for level in DEGREES},
              "pairs_not_in_the_same_pedigree_family": int((~pairs["same_pedigree_family"]).sum()),
              "cross_superpopulation_pairs": int((pairs["Superpopulation_a"] != pairs["Superpopulation_b"]).sum()),
              "folds": cross_fold_report(pairs, splits),
              "clusters_merging_pedigree_families": int((samples.groupby(clusters)["FamilyID"].nunique() > 1).sum()),
              "boundaries": {str(level): boundary for level, boundary in DEGREES.items()}}
    (out / "relatedness_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

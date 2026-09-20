"""Checks of the KING-robust kinship and the alternative bench-real splits, on synthetic genotypes only."""

import json

import numpy as np
import pandas as pd

from benchmarks.bench_real import relatedness, split_variants


def synthetic_pedigree(sites: int, seed: int):
    """Genotypes (sites x samples) for: two unrelated founders, their child, the child's duplicate, and a full sib."""
    generator = np.random.default_rng(seed)
    frequency = generator.uniform(0.05, 0.5, size=sites)
    haplotypes = generator.random((4, sites)) < frequency

    def child():
        pick = generator.integers(0, 2, size=(2, sites))
        return haplotypes[pick[0], np.arange(sites)].astype(int) + haplotypes[2 + pick[1], np.arange(sites)].astype(int)

    father, mother = haplotypes[0].astype(int) + haplotypes[1], haplotypes[2].astype(int) + haplotypes[3]
    first = child()
    stranger = (generator.random((2, sites)) < frequency).sum(axis=0)
    return np.stack([father, mother, first, first, child(), stranger], axis=1).astype(np.int8)


def test_king_counts_match_a_direct_count():
    dosage = np.random.default_rng(0).integers(0, 3, size=(200, 5)).astype(np.int8)
    both, opposite, heterozygous = relatedness.king_counts(dosage, np.arange(200), chunk=37)
    for i in range(5):
        for j in range(5):
            assert both[i, j] == np.sum((dosage[:, i] == 1) & (dosage[:, j] == 1))
            assert opposite[i, j] == np.sum(((dosage[:, i] == 2) & (dosage[:, j] == 0)) | ((dosage[:, i] == 0) & (dosage[:, j] == 2)))
        assert heterozygous[i] == np.sum(dosage[:, i] == 1)


def test_king_recovers_the_pedigree_degrees():
    dosage = synthetic_pedigree(40000, 1)
    kinship = relatedness.king_robust(*relatedness.king_counts(dosage, np.arange(dosage.shape[0]), chunk=8192))
    assert relatedness.degree(kinship[2, 3]) == 0
    assert relatedness.degree(kinship[0, 2]) == 1 and relatedness.degree(kinship[1, 2]) == 1
    assert relatedness.degree(kinship[2, 4]) == 1
    assert relatedness.degree(kinship[0, 1]) is None and relatedness.degree(kinship[0, 5]) is None


def samples_table(count: int):
    groups = np.repeat(["AFR", "AMR", "EAS", "EUR", "SAS"], count // 5)
    table = pd.DataFrame({"sample": [f"S{index:03d}" for index in range(len(groups))], "Superpopulation": groups})
    table["FamilyID"] = table["sample"]
    table.loc[[0, 1], "FamilyID"] = "F0"
    return table


def test_clusters_join_families_through_related_pairs():
    samples = samples_table(20)
    pairs = pd.DataFrame({"FamilyID_a": ["F0", "S005"], "FamilyID_b": ["S003", "S006"]})
    clusters = relatedness.relatedness_clusters(samples, pairs)
    assert clusters[0] == clusters[1] == clusters[3]
    assert clusters[5] == clusters[6] and clusters[5] != clusters[0]


def test_size_matched_training_sets_are_stratified_subsets_of_the_target_size():
    samples = samples_table(100)
    sealed = [{"name": f"loso/{group}", "test": sorted(samples.loc[samples["Superpopulation"] == group, "sample"]),
               "train": sorted(samples.loc[samples["Superpopulation"] != group, "sample"])} for group in ("AFR", "AMR")]
    sealed[0]["train"] = sealed[0]["train"][:-7]
    matched, target = split_variants.size_matched(sealed, samples)
    assert target == len(sealed[0]["train"])
    group = samples.set_index("sample")["Superpopulation"]
    for before, after in zip(sealed, matched):
        assert after["test"] == before["test"] and len(after["train"]) == target and set(after["train"]) <= set(before["train"])
        share = group.loc[before["train"]].value_counts() / len(before["train"]) * target
        assert (abs(group.loc[after["train"]].value_counts().reindex(share.index, fill_value=0) - share) < 1).all()


def test_king_random5_never_splits_a_cluster(tmp_path):
    samples = samples_table(50)
    clusters = samples["FamilyID"].copy()
    clusters[[2, 3, 4]] = "C"
    folds = split_variants.king_random5(samples, clusters)
    fold_of = {sample: index for index, fold in enumerate(folds) for sample in fold["test"]}
    assert sorted(fold_of) == sorted(samples["sample"])
    assert len({fold_of[samples.loc[index, "sample"]] for index in (0, 1)}) == 1
    assert len({fold_of[samples.loc[index, "sample"]] for index in (2, 3, 4)}) == 1
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "samples.tsv").write_text("x")
    (dataset / "splits.json").write_text("[]")
    digest = split_variants.shadow_dataset(dataset, tmp_path / "shadow", folds)
    assert (tmp_path / "shadow" / "samples.tsv").is_symlink() and json.loads((tmp_path / "shadow" / "splits.json").read_text()) == folds
    assert (tmp_path / "shadow" / "splits.sha256").read_text().strip() == digest

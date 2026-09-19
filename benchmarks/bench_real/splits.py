"""Sealed train/test splits of the MAGE benchmark samples.

Two designs:
  random5  five folds; whole 1kGP families stay in one fold, and every superpopulation is spread evenly over the folds.
  loso     leave one superpopulation out: train on four superpopulations, test on the fifth.
It also writes gene_order.tsv, a seeded permutation of all genes: any prefix of it is a random gene sample, so
methods too costly for every gene run on the same leading prefix.

The random5 fold count is the conventional five (Hastie, Tibshirani & Friedman 2009, §7.10); it only sets the
precision of the out-of-fold estimate, not which method wins. The seed is derived from the split name, so the file
is reproducible, and its sha256 is recorded in the data card.
"""
import hashlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd

RANDOM_FOLD_COUNT = 5


def seed_from_name(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")


def random_folds(samples: pd.DataFrame) -> list:
    generator = np.random.default_rng(seed_from_name("bench-real/random5"))
    fold_of_sample = {}
    for _, members in samples.groupby("Superpopulation"):
        families = members.groupby("FamilyID")["sample"].apply(list).tolist()
        order = generator.permutation(len(families))
        fold_sizes = np.zeros(RANDOM_FOLD_COUNT, dtype=int)
        for family_index in order:
            family = families[family_index]
            target = int(np.argmin(fold_sizes))
            fold_sizes[target] += len(family)
            for sample in family:
                fold_of_sample[sample] = target
    family_folds = pd.Series(fold_of_sample).groupby(samples.set_index("sample")["FamilyID"]).nunique()
    if (family_folds > 1).any():
        raise ValueError("a family spans two folds")
    return [{"name": f"random5/fold{fold}", "test": sorted(sample for sample, value in fold_of_sample.items() if value == fold)} for fold in range(RANDOM_FOLD_COUNT)]


def leave_one_superpopulation_out(samples: pd.DataFrame) -> list:
    return [{"name": f"loso/{superpopulation}", "test": sorted(members["sample"])} for superpopulation, members in samples.groupby("Superpopulation")]


def main(dataset_dir: str):
    directory = pathlib.Path(dataset_dir)
    samples = pd.read_csv(directory / "samples.tsv", sep="\t")
    everyone = set(samples["sample"])
    splits = random_folds(samples) + leave_one_superpopulation_out(samples)
    for split in splits:
        split["train"] = sorted(everyone - set(split["test"]))
    text = json.dumps(splits, indent=0)
    (directory / "splits.json").write_text(text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    genes = pd.read_csv(directory / "genes.tsv", sep="\t")
    order = np.random.default_rng(seed_from_name("bench-real/gene_order")).permutation(len(genes))
    genes.iloc[order][["gene_id"]].to_csv(directory / "gene_order.tsv", sep="\t", index=False)
    (directory / "splits.sha256").write_text(digest + "\n")
    for split in splits:
        test = samples[samples["sample"].isin(split["test"])]
        print(split["name"], len(split["train"]), len(split["test"]), dict(test["Superpopulation"].value_counts().sort_index()))
    print("sha256", digest)


if __name__ == "__main__":
    main(sys.argv[1])

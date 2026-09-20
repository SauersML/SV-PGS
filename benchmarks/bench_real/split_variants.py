"""Alternative sealed splits for bench-real's portability controls (critique/CRITIQUE_REAL.md item 5).

size_matched  every loso and random5 training set is subsampled to the smallest loso training set, which is the
              AFR-held-out one. Test sets don't change, so portability (loso vs random5) is compared at equal training n.
              The draw is stratified by superpopulation with proportional largest-remainder allocation, so each
              training set keeps its composition. The seed is sha256("bench-real/size_matched/<split name>").
random5_king  random5 rebuilt with the fold unit widened from the pedigree family to the relatedness cluster:
              pedigree families joined by every KING-robust pair above the third-degree boundary (relatedness.py).
              No test person then has a relative in training. It is the same balanced greedy fill as splits.py, seeded
              with sha256("bench-real/random5_king").
A shadow dataset directory links every dataset file except splits.json, which holds the alternative splits under the
standard design names. So the harness runs them unchanged, into a separate results directory.
"""
import argparse
import hashlib
import json
import pathlib

import numpy as np
import pandas as pd

from benchmarks.bench_real import splits as sealed_splits


def seed(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")


def proportional_allocation(counts: pd.Series, total: int) -> pd.Series:
    """Largest-remainder allocation of total across strata in proportion to counts; ties go to the earlier label."""
    exact = counts / counts.sum() * total
    allocation = np.floor(exact).astype(int)
    remainder = total - int(allocation.sum())
    order = sorted(counts.index, key=lambda label: (-(exact[label] - allocation[label]), label))
    for label in order[:remainder]:
        allocation[label] += 1
    return allocation


def size_matched(splits: list, samples: pd.DataFrame):
    """Every loso and random5 training set subsampled to the smallest loso training size."""
    target = min(len(split["train"]) for split in splits if split["name"].startswith("loso/"))
    group = samples.set_index("sample")["Superpopulation"]
    matched = []
    for split in splits:
        train = sorted(split["train"])
        strata = group.loc[train]
        allocation = proportional_allocation(strata.value_counts().sort_index(), target)
        generator = np.random.default_rng(seed(f"bench-real/size_matched/{split['name']}"))
        chosen = []
        for label, size in allocation.items():
            members = sorted(strata.index[strata == label])
            chosen += [str(sample) for sample in generator.choice(members, size=size, replace=False)]
        matched.append({"name": split["name"], "test": split["test"], "train": sorted(chosen)})
    return matched, target


def king_random5(samples: pd.DataFrame, clusters: pd.Series):
    """random5 whose fold unit is the relatedness cluster; a cluster spanning superpopulations goes with its majority."""
    table = samples.copy()
    table["FamilyID"] = clusters.to_numpy()
    majority = table.groupby("FamilyID")["Superpopulation"].agg(lambda values: values.value_counts().sort_index().idxmax())
    table["Superpopulation"] = table["FamilyID"].map(majority)
    generator = np.random.default_rng(seed("bench-real/random5_king"))
    fold_of = {}
    for _, members in table.groupby("Superpopulation"):
        families = members.groupby("FamilyID")["sample"].apply(list).tolist()
        sizes = np.zeros(sealed_splits.RANDOM_FOLD_COUNT, dtype=int)
        for index in generator.permutation(len(families)):
            fold = int(np.argmin(sizes))
            sizes[fold] += len(families[index])
            for sample in families[index]:
                fold_of[sample] = fold
    everyone = set(samples["sample"])
    folds = []
    for fold in range(sealed_splits.RANDOM_FOLD_COUNT):
        test = sorted(sample for sample, value in fold_of.items() if value == fold)
        folds.append({"name": f"random5/fold{fold}", "test": test, "train": sorted(everyone - set(test))})
    return folds


def shadow_dataset(dataset_dir: pathlib.Path, out_dir: pathlib.Path, splits: list):
    """A dataset directory that links every file but splits.json, which holds these splits."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in dataset_dir.iterdir():
        if path.name in ("splits.json", "splits.sha256"):
            continue
        link = out_dir / path.name
        if not link.exists():
            link.symlink_to(path.resolve())
    text = json.dumps(splits, indent=0)
    (out_dir / "splits.json").write_text(text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    (out_dir / "splits.sha256").write_text(digest + "\n")
    return digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--clusters", required=True, help="samples_with_clusters.tsv from relatedness.py")
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    dataset_dir, out = pathlib.Path(arguments.dataset), pathlib.Path(arguments.out)
    samples = pd.read_csv(dataset_dir / "samples.tsv", sep="\t")
    splits = json.loads((dataset_dir / "splits.json").read_text())
    matched, target = size_matched(splits, samples)
    clusters = pd.read_csv(arguments.clusters, sep="\t").set_index("sample").loc[samples["sample"], "relatedness_cluster"]
    king = king_random5(samples, clusters)
    loso = [split for split in splits if split["name"].startswith("loso/")]
    report = {"size_matched_training_n": target,
              "size_matched_sha256": shadow_dataset(dataset_dir, out / "dataset_size_matched", matched),
              "random5_king_sha256": shadow_dataset(dataset_dir, out / "dataset_random5_king", king + loso),
              "random5_king_fold_sizes": [len(fold["test"]) for fold in king],
              "multi_family_clusters": int((samples.groupby(clusters.to_numpy())["FamilyID"].nunique() > 1).sum())}
    (out / "split_variants.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

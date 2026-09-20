"""Sealed splits of the cytotoxicity lines. The DREAM challenge's own train/test split is not used.

  random5  five folds (the conventional count: Hastie, Tibshirani & Friedman 2009, §7.10). Whole relatedness clusters
           stay in one fold: 1kGP pedigree families merged with any pair KING calls related (third degree or closer,
           Manichaikul et al. 2010). Every continental group is spread evenly over the folds.
  lococ    leave one continental group out (EUR, EAS, AFR, AMR): train on three, test on the fourth.
The seed comes from the design's name, so the splits are reproducible, and their sha256 is recorded in the data card.
"""
import hashlib

import numpy as np
import pandas as pd

RANDOM_FOLD_COUNT = 5
CONTINENT = {"GBR": "EUR", "CEU": "EUR", "TSI": "EUR", "YRI": "AFR", "LWK": "AFR", "CHB": "EAS", "JPT": "EAS", "CLM": "AMR", "MXL": "AMR"}


def seed_from_name(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")


def relatedness_clusters(lines: pd.DataFrame, related_pairs) -> pd.Series:
    """Cluster label per line: the connected components of 'same pedigree family' plus 'KING-related pair'."""
    parent = {line: line for line in lines["line"]}

    def root(line):
        while parent[line] != line:
            parent[line] = parent[parent[line]]
            line = parent[line]
        return line

    def join(first, second):
        parent[root(first)] = root(second)

    for _, members in lines.groupby("family"):
        members = list(members["line"])
        for other in members[1:]:
            join(members[0], other)
    for first, second in related_pairs:
        if first in parent and second in parent:
            join(first, second)
    return pd.Series({line: root(line) for line in lines["line"]})


def random_folds(lines: pd.DataFrame, related_pairs=()) -> list:
    """lines has columns line, family, continent. Returns the five test sets."""
    generator = np.random.default_rng(seed_from_name("bench-tox/random5"))
    clusters = relatedness_clusters(lines, related_pairs)
    continent = lines.set_index("line")["continent"]
    fold_of_line = {}
    fold_sizes = {group: np.zeros(RANDOM_FOLD_COUNT, dtype=int) for group in sorted(continent.unique())}
    grouped = sorted(clusters.groupby(clusters).groups.items(), key=lambda item: item[0])
    for index in generator.permutation(len(grouped)):
        members = list(grouped[index][1])
        # A mixed-ancestry cluster is placed by its most common group; ties break by the sorted group name.
        counts = continent[members].value_counts()
        group = sorted(counts[counts == counts.max()].index)[0]
        target = int(np.argmin(fold_sizes[group]))
        fold_sizes[group][target] += len(members)
        for line in members:
            fold_of_line[line] = target
    return [{"name": f"random5/fold{fold}", "test": sorted(line for line, value in fold_of_line.items() if value == fold)}
            for fold in range(RANDOM_FOLD_COUNT)]


def leave_one_continent_out(lines: pd.DataFrame) -> list:
    return [{"name": f"lococ/{group}", "test": sorted(members["line"])} for group, members in lines.groupby("continent")]


def all_splits(lines: pd.DataFrame, related_pairs=()) -> list:
    everyone = set(lines["line"])
    splits = random_folds(lines, related_pairs) + leave_one_continent_out(lines)
    for split in splits:
        split["train"] = sorted(everyone - set(split["test"]))
    return splits

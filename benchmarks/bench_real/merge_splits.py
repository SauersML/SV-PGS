"""Merge per-split harness runs (--splits) of one method and design into one result per gene set.

A run restricted to some splits fills only those splits' test samples. Runs of the same genes and feature sets over
disjoint splits merge by taking, per sample, the one run that predicted it; the merged set is written under the tag
<base>.merged, with a run.json listing its parts. Two parts predicting the same sample is an error.
Usage: merge_splits.py <results>/<method>/<design> <base tag>
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd


def merge(directory: pathlib.Path, base: str):
    parts = sorted(path.name.removesuffix(".genes.tsv") for path in directory.glob(f"{base}.*.genes.tsv") if not path.name.endswith(".merged.genes.tsv"))
    if not parts:
        raise ValueError(f"no parts for {base} in {directory}")
    genes = pd.read_csv(directory / f"{parts[0]}.genes.tsv", sep="\t")
    for part in parts[1:]:
        if not pd.read_csv(directory / f"{part}.genes.tsv", sep="\t")["gene_id"].equals(genes["gene_id"]):
            raise ValueError(f"{part} scores different genes")
    arrays = sorted({path.name.removeprefix(f"{parts[0]}.") for path in directory.glob(f"{parts[0]}.*.npy")})
    for name in arrays:
        merged = None
        for part in parts:
            values = np.load(directory / f"{part}.{name}")
            if merged is None:
                merged = values.copy()
                continue
            overlap = ~np.isnan(merged) & ~np.isnan(values)
            if overlap.any():
                raise ValueError(f"{part} and an earlier part both predict {int(overlap.sum())} cells of {name}")
            merged = np.where(np.isnan(merged), values, merged)
        np.save(directory / f"{base}.merged.{name}", merged)
    genes.to_csv(directory / f"{base}.merged.genes.tsv", sep="\t", index=False)
    pd.concat([pd.read_csv(directory / f"{part}.log.tsv", sep="\t") for part in parts]).to_csv(directory / f"{base}.merged.log.tsv", sep="\t", index=False)
    records = [json.loads((directory / f"{part}.run.json").read_text()) for part in parts]
    (directory / f"{base}.merged.run.json").write_text(json.dumps({"merged_from": parts, "parts": records}, indent=1))
    return parts


if __name__ == "__main__":
    print(merge(pathlib.Path(sys.argv[1]), sys.argv[2]))

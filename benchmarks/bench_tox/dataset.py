"""The cytotoxicity benchmark's analysis set: the 793 DREAM lines in the 1kGP 30x panel, their pooled ComBat-normalized
cytotoxicity values for the development compounds, their covariates, and the pre-registered splits (PREREG.md).

Every per-line table stays under ``WORK`` (mode 700) on MSI; this module writes nothing else. The confirmation
compounds (``seal.py``) are dropped at load and never read by any development analysis.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib

import numpy as np
import pandas as pd

from benchmarks.bench_tox import genotypes, seal, splits

RESTRICTED = pathlib.Path("/projects/standard/hsiehph/sauer354/svpgs-team/restricted/tox_dream")
WORK = pathlib.Path("/scratch.global/sauer354/svpgs-team/bench-tox/work")
PEDIGREE = pathlib.Path("/scratch.global/sauer354/svpgs-team/bench-real/data/kgp/20130606_g1k_3202_samples_ped_population.txt")
# KING-robust on this many random autosomal SNVs per chromosome (some ten thousand markers genome-wide suffice for
# the third-degree boundary; Manichaikul et al. 2010, Table 2, with markers in the tens of thousands).
KING_MARKERS_PER_CHROMOSOME = 2_000


def _restricted(name: str) -> pd.DataFrame:
    return pd.read_csv(RESTRICTED / name, sep="\t", index_col=0)


def load_values() -> pd.DataFrame:
    """Lines x development compounds: the pooled DREAM training (extended) and gold-standard test values, for the
    analysis lines; the confirmation compounds are dropped here."""
    train = _restricted("ToxChallenge_CytotoxicityData_Train_Subchal1_Extended.txt")
    test = _restricted("ToxChallenge_Subchall1_Test_Data.txt")
    values = pd.concat([train, test.loc[~test.index.isin(train.index)]])
    analysis = [line.strip() for line in (RESTRICTED / "derived/analysis_lines.txt").read_text().split()]
    development, _confirmation = seal.split_compounds(list(values.columns))
    values = values.loc[analysis, development].astype(np.float64)
    if values.isna().any().any():
        raise ValueError("the analysis lines' cytotoxicity values have a missing entry")
    return values


def load_lines() -> pd.DataFrame:
    """One row per analysis line, in the genotype files' order: line, sex, population, batch, continent, family."""
    covariates = pd.read_csv(RESTRICTED / "ToxChallenge_Covariates.txt", sep="\t", dtype=str).set_index("ID")
    pedigree = pd.read_csv(PEDIGREE, sep=r"\s+", dtype=str).set_index("SampleID")
    order = genotypes.samples()
    analysis = set(line.strip() for line in (RESTRICTED / "derived/analysis_lines.txt").read_text().split())
    rows = []
    for line in order:
        if line not in analysis:
            continue
        population = covariates.loc[line, "Population"]
        continent = splits.CONTINENT.get(population, pedigree.loc[line, "Superpopulation"] if line in pedigree.index else None)
        if continent is None:
            raise ValueError(f"no continental group for {line} ({population})")
        rows.append({
            "line": line, "sex": int(covariates.loc[line, "Sex"]), "population": population, "batch": covariates.loc[line, "Cytotoxicity_Batch"],
            "continent": continent, "family": pedigree.loc[line, "FamilyID"] if line in pedigree.index else line,
        })
    lines = pd.DataFrame(rows)
    if len(lines) != len(analysis):
        raise ValueError(f"{len(analysis) - len(lines)} analysis lines are not in the genotype files")
    return lines


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def build_splits(lines: pd.DataFrame) -> list:
    """The pre-registered splits, with KING pairs from the beds (cached under WORK), and their sha256."""
    WORK.mkdir(parents=True, exist_ok=True)
    os.chmod(WORK, 0o700)
    pairs_path = WORK / "king_pairs.json"
    if pairs_path.exists():
        pairs = [tuple(pair) for pair in json.loads(pairs_path.read_text())]
    else:
        pairs = genotypes.related_pairs(lines["line"].tolist(), KING_MARKERS_PER_CHROMOSOME, splits.seed_from_name("bench-tox/king"))
        pairs_path.write_text(json.dumps(pairs))
        os.chmod(pairs_path, 0o600)
    result = splits.all_splits(lines[["line", "family", "continent"]], pairs)
    text = json.dumps(result, sort_keys=True)
    (WORK / "splits.json").write_text(text)
    os.chmod(WORK / "splits.json", 0o600)
    (WORK / "splits.sha256").write_text(_sha256(text) + "\n")
    return result


def summary(lines: pd.DataFrame, values: pd.DataFrame, built: list, pairs_count: int) -> dict:
    """Aggregates only: counts per group, fold sizes, the splits' sha256."""
    return {
        "lines": int(len(lines)), "development_compounds": int(values.shape[1]),
        "continents": {group: int(count) for group, count in lines["continent"].value_counts().sort_index().items()},
        "related_pairs_third_degree_or_closer": pairs_count,
        "splits": {split["name"]: {"train": len(split["train"]), "test": len(split["test"])} for split in built},
        "splits_sha256": (WORK / "splits.sha256").read_text().strip(),
    }


def main():
    lines = load_lines()
    values = load_values()
    built = build_splits(lines)
    pairs = json.loads((WORK / "king_pairs.json").read_text())
    print(json.dumps(summary(lines, values, built, len(pairs)), indent=1))


if __name__ == "__main__":
    main()

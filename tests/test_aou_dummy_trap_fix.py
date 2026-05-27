"""Regression test for the dummy-variable trap fix in covariate expansion.

`_expand_one_hot_covariates` must drop exactly one reference category per
one-hot OMOP prefix so the resulting covariate matrix has full column rank
against the implicit intercept (i.e. no `sum(dummies) == 1` per row).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sv_pgs.aou_runner import _expand_one_hot_covariates


def _write_sample_table(path: Path, columns: dict) -> None:
    names = list(columns.keys())
    n = len(next(iter(columns.values())))
    lines = ["\t".join(names)]
    for i in range(n):
        lines.append("\t".join(str(columns[c][i]) for c in names))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_drops_largest_frequency_reference_per_group(tmp_path: Path) -> None:
    cols = {
        "person_id": ["s1", "s2", "s3", "s4", "s5"],
        "age_at_observation_start": [40.0, 50.0, 60.0, 70.0, 80.0],
        "age_squared": [1600.0, 2500.0, 3600.0, 4900.0, 6400.0],
        # gender: 8507 majority (3/5), 8532 (1/5), 0 (1/5) -> drop 8507
        "gender_concept_id_8507": [1, 1, 1, 0, 0],
        "gender_concept_id_8532": [0, 0, 0, 1, 0],
        "gender_concept_id_0":    [0, 0, 0, 0, 1],
        # race: 8527 majority (4/5), 8516 (1/5) -> drop 8527
        "race_concept_id_8527": [1, 1, 1, 1, 0],
        "race_concept_id_8516": [0, 0, 0, 0, 1],
        # ethnicity: 38003564 majority (4/5), 38003563 (1/5) -> drop 38003564
        "ethnicity_concept_id_38003564": [1, 1, 1, 1, 0],
        "ethnicity_concept_id_38003563": [0, 0, 0, 0, 1],
        "PC1": [0.1, -0.2, 0.3, -0.1, 0.0],
    }
    path = tmp_path / "samples.tsv"
    _write_sample_table(path, cols)

    requested = [
        "age_at_observation_start",
        "age_squared",
        "gender_concept_id",
        "race_concept_id",
        "ethnicity_concept_id",
        "PC1",
    ]
    expanded = _expand_one_hot_covariates(requested, path)

    assert "gender_concept_id_8507" not in expanded
    assert "race_concept_id_8527" not in expanded
    assert "ethnicity_concept_id_38003564" not in expanded
    assert "gender_concept_id_8532" in expanded
    assert "gender_concept_id_0" in expanded
    assert "race_concept_id_8516" in expanded
    assert "ethnicity_concept_id_38003563" in expanded
    assert expanded[0] == "age_at_observation_start"
    assert expanded[1] == "age_squared"
    assert expanded[-1] == "PC1"


def test_covariate_matrix_has_full_column_rank(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 200
    gender = rng.choice(["8507", "8532", "0"], size=n, p=[0.5, 0.4, 0.1])
    race = rng.choice(["8527", "8516", "8515"], size=n, p=[0.7, 0.2, 0.1])
    ethnicity = rng.choice(["38003564", "38003563"], size=n, p=[0.85, 0.15])

    def _one_hot(prefix: str, values: np.ndarray) -> dict:
        levels = sorted(set(values.tolist()))
        return {f"{prefix}_{lv}": (values == lv).astype(np.int64) for lv in levels}

    arrays: dict = {
        "person_id": np.array([f"s{i}" for i in range(n)]),
        "age_at_observation_start": rng.uniform(30, 90, size=n),
    }
    arrays["age_squared"] = arrays["age_at_observation_start"] ** 2
    arrays.update(_one_hot("gender_concept_id", gender))
    arrays.update(_one_hot("race_concept_id", race))
    arrays.update(_one_hot("ethnicity_concept_id", ethnicity))
    arrays["PC1"] = rng.standard_normal(n)
    arrays["PC2"] = rng.standard_normal(n)

    cols = {k: (v.tolist() if hasattr(v, "tolist") else list(v)) for k, v in arrays.items()}
    path = tmp_path / "samples.tsv"
    _write_sample_table(path, cols)

    requested = [
        "age_at_observation_start",
        "age_squared",
        "gender_concept_id",
        "race_concept_id",
        "ethnicity_concept_id",
        "PC1",
        "PC2",
    ]
    expanded = _expand_one_hot_covariates(requested, path)

    C = np.column_stack(
        [np.ones(n)] + [np.asarray(arrays[c], dtype=np.float64) for c in expanded]
    )
    rank = np.linalg.matrix_rank(C)
    assert rank == C.shape[1], (
        f"expected full column rank {C.shape[1]} after dropping reference "
        f"categories, got rank={rank}"
    )

    all_cols = [c for c in arrays.keys() if c != "person_id"]
    C_bad = np.column_stack(
        [np.ones(n)] + [np.asarray(arrays[c], dtype=np.float64) for c in all_cols]
    )
    assert np.linalg.matrix_rank(C_bad) < C_bad.shape[1]


def test_explicit_column_name_passthrough(tmp_path: Path) -> None:
    """If the user passes a fully-qualified one-hot column directly, it's
    not in `_OMOP_ONE_HOT_PREFIXES` so it must pass through unchanged
    (preserves the `--covariates` override path)."""
    cols = {
        "person_id": ["s1", "s2"],
        "gender_concept_id_8507": [1, 0],
        "gender_concept_id_8532": [0, 1],
        "PC1": [0.1, -0.2],
    }
    path = tmp_path / "samples.tsv"
    _write_sample_table(path, cols)

    expanded = _expand_one_hot_covariates(
        ["gender_concept_id_8507", "PC1"], path
    )
    assert expanded == ["gender_concept_id_8507", "PC1"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

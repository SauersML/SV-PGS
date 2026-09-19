"""Regression test for the dummy-variable trap fix in covariate expansion.

`_expand_one_hot_covariates` must drop exactly one reference category per
one-hot OMOP prefix so the resulting covariate matrix has full column rank
against the implicit intercept (i.e. no `sum(dummies) == 1` per row).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from sv_pgs.all_of_us import AllOfUsDiseaseRequest, prepare_all_of_us_disease_sample_table
from sv_pgs.aou_runner import _expand_one_hot_covariates
from tests.test_all_of_us import _FakeBigQueryClient, _disease_row

FEMALE = 45878463
MALE = 45880669
INTERSEX = 1585848
SKIPPED = 903096


def _write_sample_table(path: Path, columns: dict) -> None:
    names = list(columns.keys())
    row_count = len(next(iter(columns.values())))
    lines = ["\t".join(names)]
    for row_index in range(row_count):
        lines.append("\t".join(str(columns[name][row_index]) for name in names))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_drops_largest_frequency_reference(tmp_path: Path) -> None:
    columns = {
        "person_id": ["s1", "s2", "s3", "s4", "s5"],
        "age_at_observation_end": [40.0, 50.0, 60.0, 70.0, 80.0],
        # sex at birth: female majority (3/5), male (1/5), skipped (1/5) -> drop female
        f"sex_at_birth_concept_id_{FEMALE}": [1, 1, 1, 0, 0],
        f"sex_at_birth_concept_id_{MALE}": [0, 0, 0, 1, 0],
        f"sex_at_birth_concept_id_{SKIPPED}": [0, 0, 0, 0, 1],
        "PC1": [0.1, -0.2, 0.3, -0.1, 0.0],
    }
    path = tmp_path / "samples.tsv"
    _write_sample_table(path, columns)

    expanded = _expand_one_hot_covariates(["age_at_observation_end", "sex_at_birth_concept_id", "PC1"], path)

    assert expanded == [
        "age_at_observation_end",
        f"sex_at_birth_concept_id_{MALE}",
        f"sex_at_birth_concept_id_{SKIPPED}",
        "PC1",
    ]


def test_covariate_matrix_has_full_column_rank(tmp_path: Path) -> None:
    generator = np.random.default_rng(0)
    person_count = 200
    sexes = generator.choice([str(FEMALE), str(MALE), str(SKIPPED)], size=person_count, p=[0.5, 0.4, 0.1])
    arrays: dict = {
        "person_id": np.array([f"s{index}" for index in range(person_count)]),
        "age_at_observation_end": generator.uniform(30, 90, size=person_count),
    }
    for level in sorted(set(sexes.tolist())):
        arrays[f"sex_at_birth_concept_id_{level}"] = (sexes == level).astype(np.int64)
    arrays["PC1"] = generator.standard_normal(person_count)
    arrays["PC2"] = generator.standard_normal(person_count)
    path = tmp_path / "samples.tsv"
    _write_sample_table(path, {name: values.tolist() for name, values in arrays.items()})

    expanded = _expand_one_hot_covariates(["age_at_observation_end", "sex_at_birth_concept_id", "PC1", "PC2"], path)

    covariates = np.column_stack(
        [np.ones(person_count)] + [np.asarray(arrays[name], dtype=np.float64) for name in expanded]
    )
    assert np.linalg.matrix_rank(covariates) == covariates.shape[1]
    every_level = [name for name in arrays if name != "person_id"]
    trapped = np.column_stack(
        [np.ones(person_count)] + [np.asarray(arrays[name], dtype=np.float64) for name in every_level]
    )
    assert np.linalg.matrix_rank(trapped) < trapped.shape[1]


def test_explicit_column_name_passthrough(tmp_path: Path) -> None:
    """A fully-qualified one-hot column is not a prefix, so it passes through
    unchanged (preserves the `--covariates` override path)."""
    columns = {
        "person_id": ["s1", "s2"],
        f"sex_at_birth_concept_id_{FEMALE}": [1, 0],
        f"sex_at_birth_concept_id_{MALE}": [0, 1],
        "PC1": [0.1, -0.2],
    }
    path = tmp_path / "samples.tsv"
    _write_sample_table(path, columns)

    expanded = _expand_one_hot_covariates([f"sex_at_birth_concept_id_{FEMALE}", "PC1"], path)
    assert expanded == [f"sex_at_birth_concept_id_{FEMALE}", "PC1"]


def test_prepared_phenotype_table_drops_exactly_one_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The phenotype writer keeps every level; the expander drops the majority only.

    Each stage used to drop its own reference (the writer the lowest concept
    id, the expander the majority), merging two levels into the reference.
    """
    monkeypatch.setenv("GOOGLE_PROJECT", "billing-project")
    monkeypatch.setenv("WORKSPACE_CDR", "aou_workspace.cdr_dataset")
    sexes = [FEMALE] * 5 + [MALE] * 3 + [INTERSEX] + [SKIPPED]
    rows = [
        _disease_row(100 + index, occurrence_count=2 * (index % 2), sex_at_birth_concept_id=sexes[index])
        for index in range(10)
    ]
    path = tmp_path / "atrial_fibrillation.samples.tsv"
    prepare_all_of_us_disease_sample_table(
        request=AllOfUsDiseaseRequest(disease="atrial_fibrillation"),
        output_path=path,
        client=_FakeBigQueryClient(rows=rows),
    )

    expanded = _expand_one_hot_covariates(["sex_at_birth_concept_id"], path)

    assert sorted(expanded) == [
        f"sex_at_birth_concept_id_{INTERSEX}",
        f"sex_at_birth_concept_id_{MALE}",
        f"sex_at_birth_concept_id_{SKIPPED}",
    ]
    with path.open(newline="", encoding="utf-8") as handle:
        table_rows = list(csv.DictReader(handle, delimiter="\t"))
    covariate_matrix = np.column_stack(
        [np.ones(len(table_rows))]
        + [np.asarray([float(row[column]) for row in table_rows]) for column in expanded]
    )
    assert np.linalg.matrix_rank(covariate_matrix) == covariate_matrix.shape[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

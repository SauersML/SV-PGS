from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from sv_pgs.all_of_us import (
    AllOfUsDiseaseRequest,
    available_disease_names,
    build_all_of_us_disease_query_config,
    build_all_of_us_disease_sql,
    prepare_all_of_us_disease_sample_table,
    resolve_disease_definition,
)
from sv_pgs.cli import main


class _FakeQueryJob:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def result(self):
        return [_FakeRow(row) for row in self._rows]


class _FakeRow:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def items(self):
        return self._payload.items()


class _FakeBigQueryClient:
    def __init__(self, rows: list[dict[str, object]], project: str | None = None) -> None:
        self.rows = rows
        self.project = project
        self.sql: str | None = None
        self.job_config = None

    def query(self, sql: str, job_config):
        self.sql = sql
        self.job_config = job_config
        return _FakeQueryJob(self.rows)


def test_build_all_of_us_disease_sql_uses_workspace_cdr_and_prefix_filters(monkeypatch):
    monkeypatch.setenv("WORKSPACE_CDR", "aou_workspace.cdr_dataset")
    disease_definition = resolve_disease_definition("heart_failure")
    sql = build_all_of_us_disease_sql(disease_definition)
    query_config = build_all_of_us_disease_query_config(disease_definition)

    assert "`aou_workspace.cdr_dataset.condition_occurrence`" in sql
    assert "`aou_workspace.cdr_dataset.observation_period`" in sql
    assert "`aou_workspace.cdr_dataset.person`" in sql
    assert "condition_source_concept_id" in sql
    assert "vocabulary_id = 'ICD10CM'" in sql
    assert "vocabulary_id = 'ICD9CM'" in sql
    assert "FROM `aou_workspace.cdr_dataset.concept` AS concept" in sql
    assert "JOIN `aou_workspace.cdr_dataset.concept_ancestor` AS concept_ancestor" in sql
    assert "JOIN `aou_workspace.cdr_dataset.observation` AS observation" in sql
    assert "STARTS_WITH(concept_code, icd10_prefix)" in sql
    assert "primary_consent_date" in sql
    parameter_values = {parameter.name: parameter for parameter in query_config.query_parameters}
    assert parameter_values["icd10_prefixes"].values == ["I50"]


def test_prepare_all_of_us_disease_sample_table_writes_outputs(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GOOGLE_PROJECT", "billing-project")
    monkeypatch.setenv("WORKSPACE_CDR", "aou_workspace.cdr_dataset")
    fake_client = _FakeBigQueryClient(
        rows=[
            {
                "person_id": "101",
                "sample_id": "101",
                "target": 1,
                "phenotype_occurrence_count": 3,
                "first_condition_date": "2020-01-01",
                "observation_start_date": "2018-01-01",
                "observation_end_date": "2024-01-01",
                "age_at_observation_start": 42,
                "gender_concept_id": 45880669,
                "race_concept_id": 8527,
                "ethnicity_concept_id": 38003564,
            },
            {
                "person_id": "102",
                "sample_id": "102",
                "target": 0,
                "phenotype_occurrence_count": 0,
                "first_condition_date": None,
                "observation_start_date": "2019-01-01",
                "observation_end_date": "2024-01-01",
                "primary_consent_date": "2018-12-01",
                "age_at_observation_start": 37,
                "gender_concept_id": 45878463,
                "race_concept_id": 8516,
                "ethnicity_concept_id": 38003563,
            },
        ]
    )

    output_path = tmp_path / "heart_failure.tsv"
    outputs = prepare_all_of_us_disease_sample_table(
        request=AllOfUsDiseaseRequest(
            disease="heart_failure",
        ),
        output_path=output_path,
        client=fake_client,
    )

    assert outputs.sample_table_path.is_file()
    assert outputs.sql_path.is_file()
    assert outputs.metadata_path.is_file()
    assert fake_client.sql is not None
    assert "heart_failure" not in fake_client.sql
    assert fake_client.job_config is not None
    parameter_values = {parameter.name: parameter for parameter in fake_client.job_config.query_parameters}
    assert parameter_values["icd10_prefixes"].values == ["I50"]

    rows = _read_tsv_rows(output_path)
    assert rows[0]["sample_id"] == "101"
    assert rows[0]["person_id"] == "101"
    assert rows[0]["target"] == "1"
    assert rows[1]["sample_id"] == "102"
    metadata_payload = json.loads(outputs.metadata_path.read_text(encoding="utf-8"))
    assert metadata_payload["row_count"] == 2
    assert metadata_payload["disease"] == "heart_failure"
    assert metadata_payload["icd10_prefixes"] == ["I50"]
    assert metadata_payload["min_occurrences"] == 2
    assert metadata_payload["cdr_dataset"] == "aou_workspace.cdr_dataset"


def test_prepare_all_of_us_disease_requires_all_of_us_env(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("GOOGLE_PROJECT", raising=False)
    monkeypatch.delenv("WORKSPACE_CDR", raising=False)

    with pytest.raises(ValueError, match="GOOGLE_PROJECT"):
        prepare_all_of_us_disease_sample_table(
            request=AllOfUsDiseaseRequest(disease="asthma"),
            output_path=tmp_path / "out.tsv",
        )


def test_prepare_all_of_us_disease_requires_workspace_cdr_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("GOOGLE_PROJECT", "billing-project")
    monkeypatch.delenv("WORKSPACE_CDR", raising=False)
    fake_client = _FakeBigQueryClient(
        rows=[
            {
                "person_id": "101",
                "sample_id": "101",
                "target": 1,
                "phenotype_occurrence_count": 2,
                "first_condition_date": "2020-01-01",
                "observation_start_date": "2018-01-01",
                "observation_end_date": "2024-01-01",
                "primary_consent_date": "2017-01-01",
                "age_at_observation_start": 42,
                "gender_concept_id": 45880669,
                "race_concept_id": 8527,
                "ethnicity_concept_id": 38003564,
            }
        ]
    )

    with pytest.raises(ValueError, match="WORKSPACE_CDR"):
        prepare_all_of_us_disease_sample_table(
            request=AllOfUsDiseaseRequest(disease="heart failure"),
            output_path=tmp_path / "heart_failure.tsv",
            client=fake_client,
        )


def test_prepare_all_of_us_disease_uses_client_project_without_google_project_env(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("GOOGLE_PROJECT", raising=False)
    monkeypatch.setenv("WORKSPACE_CDR", "aou_workspace.cdr_dataset")
    fake_client = _FakeBigQueryClient(
        rows=[
            {
                "person_id": "101",
                "sample_id": "101",
                "target": 1,
                "phenotype_occurrence_count": 2,
                "first_condition_date": "2020-01-01",
                "observation_start_date": "2018-01-01",
                "observation_end_date": "2024-01-01",
                "primary_consent_date": "2017-01-01",
                "age_at_observation_start": 42,
                "gender_concept_id": 45880669,
                "race_concept_id": 8527,
                "ethnicity_concept_id": 38003564,
            }
        ],
        project="client-project",
    )

    outputs = prepare_all_of_us_disease_sample_table(
        request=AllOfUsDiseaseRequest(disease="heart_failure"),
        output_path=tmp_path / "heart_failure.tsv",
        client=fake_client,
    )

    metadata_payload = json.loads(outputs.metadata_path.read_text(encoding="utf-8"))
    assert metadata_payload["billing_project"] == "client-project"


def test_prepare_all_of_us_disease_uses_workspace_cdr_from_env_in_query_and_metadata(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("GOOGLE_PROJECT", "billing-project")
    monkeypatch.setenv("WORKSPACE_CDR", "fc-aou-cdr-prod-ct.C2024Q3R9")
    fake_client = _FakeBigQueryClient(
        rows=[
            {
                "person_id": "101",
                "sample_id": "101",
                "target": 1,
                "phenotype_occurrence_count": 2,
                "first_condition_date": "2020-01-01",
                "observation_start_date": "2018-01-01",
                "observation_end_date": "2024-01-01",
                "primary_consent_date": "2017-01-01",
                "age_at_observation_start": 42,
                "gender_concept_id": 45880669,
                "race_concept_id": 8527,
                "ethnicity_concept_id": 38003564,
            }
        ]
    )

    outputs = prepare_all_of_us_disease_sample_table(
        request=AllOfUsDiseaseRequest(disease="heart_failure"),
        output_path=tmp_path / "heart_failure.tsv",
        client=fake_client,
    )

    assert fake_client.sql is not None
    assert "fc-aou-cdr-prod-ct.C2024Q3R9" in fake_client.sql
    metadata_payload = json.loads(outputs.metadata_path.read_text(encoding="utf-8"))
    assert metadata_payload["cdr_dataset"] == "fc-aou-cdr-prod-ct.C2024Q3R9"



def test_cli_prepare_all_of_us_disease_wires_request_and_outputs(monkeypatch, tmp_path: Path):
    calls: dict[str, object] = {}

    def fake_prepare(request, output_path, **kwargs):
        calls["request"] = request
        calls["output_path"] = output_path
        calls["kwargs"] = kwargs
        output_file = Path(output_path)
        output_file.write_text("sample_id\tperson_id\ttarget\n", encoding="utf-8")
        sql_path = output_file.with_suffix(output_file.suffix + ".sql")
        sql_path.write_text("SELECT 1\n", encoding="utf-8")
        metadata_path = output_file.with_suffix(output_file.suffix + ".metadata.json")
        metadata_path.write_text("{}", encoding="utf-8")
        return type(
            "Prepared",
            (),
            {
                "sample_table_path": output_file,
                "sql_path": sql_path,
                "metadata_path": metadata_path,
            },
        )()

    monkeypatch.setattr("sv_pgs.cli.prepare_all_of_us_disease_sample_table", fake_prepare)
    output_path = tmp_path / "prepared.tsv"
    exit_code = main(
        [
            "prepare-all-of-us-disease",
            "--disease",
            "copd",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert calls["output_path"] == output_path
    request = calls["request"]
    assert request.disease == "copd"


def test_cli_prepare_all_of_us_disease_accepts_aliases(monkeypatch, tmp_path: Path):
    calls: dict[str, object] = {}

    def fake_prepare(request, output_path, **kwargs):
        calls["request"] = request
        output_file = Path(output_path)
        output_file.write_text("sample_id\tperson_id\ttarget\n", encoding="utf-8")
        sql_path = output_file.with_suffix(output_file.suffix + ".sql")
        sql_path.write_text("SELECT 1\n", encoding="utf-8")
        metadata_path = output_file.with_suffix(output_file.suffix + ".metadata.json")
        metadata_path.write_text("{}", encoding="utf-8")
        return type(
            "Prepared",
            (),
            {
                "sample_table_path": output_file,
                "sql_path": sql_path,
                "metadata_path": metadata_path,
            },
        )()

    monkeypatch.setattr("sv_pgs.cli.prepare_all_of_us_disease_sample_table", fake_prepare)
    exit_code = main(
        [
            "prepare-all-of-us-disease",
            "--disease",
            "heart failure",
            "--output",
            str(tmp_path / "prepared.tsv"),
        ]
    )

    assert exit_code == 0
    request = calls["request"]
    assert request.disease == "heart failure"


def test_cli_lists_available_all_of_us_diseases(capsys):
    exit_code = main(["list-all-of-us-diseases"])
    assert exit_code == 0
    printed = capsys.readouterr().out.strip().splitlines()
    assert "heart_failure" in printed
    assert "type2_diabetes" in printed
    assert printed == sorted(available_disease_names())


def _read_tsv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [{str(key): "" if value is None else str(value) for key, value in row.items()} for row in reader]

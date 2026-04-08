from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest

import sv_pgs.aou_runner as aou_runner
from sv_pgs.all_of_us import (
    AllOfUsDiseaseRequest,
    available_disease_names,
    build_all_of_us_disease_query_config,
    build_all_of_us_disease_sql,
    prepare_all_of_us_disease_sample_table,
    resolve_disease_definition,
)
from sv_pgs.cli import main
from sv_pgs.config import ModelConfig
from sv_pgs.io import load_multi_vcf_dataset_from_files


def _write_aou_variant_metadata_stub(path: Path) -> Path:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("variant_id", "variant_class"))
    return path


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
    request = cast(AllOfUsDiseaseRequest, calls["request"])
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
    request = cast(AllOfUsDiseaseRequest, calls["request"])
    assert request.disease == "heart failure"


def test_cli_lists_available_all_of_us_diseases(capsys):
    exit_code = main(["list-all-of-us-diseases"])
    assert exit_code == 0
    printed = capsys.readouterr().out.strip().splitlines()
    assert "heart_failure" in printed
    assert "type2_diabetes" in printed
    assert printed == sorted(available_disease_names())


def test_cli_run_all_of_us_forwards_core_settings(monkeypatch, tmp_path: Path):
    calls: dict[str, object] = {}

    def fake_run_all_of_us(**kwargs):
        calls.update(kwargs)

    monkeypatch.setattr("sv_pgs.cli.run_all_of_us", fake_run_all_of_us)

    exit_code = main(
        [
            "run-all-of-us",
            "--disease",
            "heart_failure",
            "--chromosomes",
            "1,2",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert calls == {
        "disease": "heart_failure",
        "chromosomes": [1, 2],
        "output_base": str(tmp_path),
        "n_pcs": 10,
        "max_outer_iterations": 30,
        "pipeline_validation_fraction": 0.0,
        "pipeline_validation_min_samples": 0,
        "random_seed": 0,
    }


def test_cli_run_builds_config(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}
    dataset = type(
        "Dataset",
        (),
        {
            "targets": np.array([0.0, 1.0], dtype=np.float32),
            "sample_ids": np.array(["sample-1", "sample-2"]),
            "genotypes": np.zeros((2, 1), dtype=np.float32),
            "covariates": np.zeros((2, 0), dtype=np.float32),
        },
    )()

    monkeypatch.setattr("sv_pgs.cli.jax_runtime_snapshot", lambda: "jax")
    monkeypatch.setattr("sv_pgs.cli.gpu_memory_snapshot", lambda: "gpu")
    monkeypatch.setattr("sv_pgs.cli.nvidia_smi_snapshot", lambda: "nvidia")
    monkeypatch.setattr("sv_pgs.cli.log", lambda message: None)

    def fake_load_dataset_from_files(**kwargs):
        captured["load_config"] = kwargs["config"]
        return dataset

    def fake_run_training_pipeline(**kwargs):
        captured["pipeline_config"] = kwargs["config"]
        captured["output_dir"] = kwargs["output_dir"]
        return type(
            "Outputs",
            (),
            {
                "artifact_dir": tmp_path / "artifact",
                "summary_path": tmp_path / "summary.json.gz",
                "predictions_path": tmp_path / "predictions.tsv.gz",
                "coefficients_path": tmp_path / "coefficients.tsv.gz",
            },
        )()

    monkeypatch.setattr("sv_pgs.cli.load_dataset_from_files", fake_load_dataset_from_files)
    monkeypatch.setattr("sv_pgs.cli.run_training_pipeline", fake_run_training_pipeline)

    exit_code = main(
        [
            "run",
            "--genotypes",
            str(tmp_path / "input.bed"),
            "--genotype-format",
            "plink1",
            "--sample-table",
            str(tmp_path / "samples.tsv"),
            "--target-column",
            "target",
            "--output-dir",
            str(tmp_path / "out"),
            "--pipeline-validation-fraction",
            "0.25",
            "--pipeline-validation-min-samples",
            "7",
        ]
    )

    assert exit_code == 0
    load_config = cast(ModelConfig, captured["load_config"])
    pipeline_config = cast(ModelConfig, captured["pipeline_config"])
    assert load_config.max_outer_iterations == 30
    assert load_config.pipeline_validation_fraction == pytest.approx(0.25)
    assert load_config.pipeline_validation_min_samples == 7
    assert pipeline_config.max_outer_iterations == 30
    assert pipeline_config.pipeline_validation_fraction == pytest.approx(0.25)
    assert pipeline_config.pipeline_validation_min_samples == 7
    assert pipeline_config.trait_type == aou_runner.TraitType.BINARY
    assert captured["output_dir"] == tmp_path / "out"


def test_resolve_ancestry_predictions_path_uses_documented_cdr_v8_location(monkeypatch):
    monkeypatch.setenv("CDR_STORAGE_PATH", "gs://bucket/cdr")

    assert aou_runner.resolve_ancestry_predictions_path() == (
        "gs://bucket/cdr/wgs/short_read/snpindel/aux/ancestry/ancestry_preds.tsv"
    )


def test_download_ancestry_preds_uses_documented_remote_path(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        aou_runner,
        "resolve_ancestry_predictions_path",
        lambda: "gs://bucket/cdr/wgs/short_read/snpindel/aux/ancestry/ancestry_preds.tsv",
    )

    copied: list[tuple[str, str]] = []

    def fake_cp(src: str, dst: str) -> None:
        copied.append((src, dst))
        Path(dst).write_text("research_id\tpc1\n1\t0.1\n", encoding="utf-8")

    monkeypatch.setattr(aou_runner, "_gsutil_cp", fake_cp)

    local_path = aou_runner.download_ancestry_preds(tmp_path)

    assert local_path == tmp_path / "ancestry_preds.tsv"
    assert copied == [
        ("gs://bucket/cdr/wgs/short_read/snpindel/aux/ancestry/ancestry_preds.tsv", str(local_path))
    ]


def test_merge_pcs_into_sample_table_recomputes_existing_output_for_requested_n_pcs(tmp_path: Path):
    sample_table_path = tmp_path / "samples.tsv"
    ancestry_path = tmp_path / "ancestry.tsv"
    output_path = tmp_path / "samples.with_pcs.tsv"

    _write_table(
        sample_table_path,
        header=("sample_id", "person_id", "target"),
        rows=(("sample-1", "person-1", "1"), ("sample-2", "person-2", "0")),
    )
    _write_table(
        ancestry_path,
        header=("person_id", "PC1", "PC2", "PC3"),
        rows=(("person-1", "0.1", "0.2", "0.3"), ("person-2", "0.4", "0.5", "0.6")),
    )

    _, pc_cols = aou_runner.merge_pcs_into_sample_table(
        sample_table_path=sample_table_path,
        ancestry_path=ancestry_path,
        output_path=output_path,
        n_pcs=2,
    )
    assert pc_cols == ["PC1", "PC2"]
    assert "PC3" not in _read_tsv_rows(output_path)[0]

    _, pc_cols = aou_runner.merge_pcs_into_sample_table(
        sample_table_path=sample_table_path,
        ancestry_path=ancestry_path,
        output_path=output_path,
        n_pcs=3,
    )
    assert pc_cols == ["PC1", "PC2", "PC3"]
    assert _read_tsv_rows(output_path)[0]["PC3"] == "0.3"


def test_merge_pcs_into_sample_table_uses_sample_id_when_person_id_is_absent(tmp_path: Path):
    sample_table_path = tmp_path / "samples.tsv"
    ancestry_path = tmp_path / "ancestry.tsv"
    output_path = tmp_path / "samples.with_pcs.tsv"

    _write_table(
        sample_table_path,
        header=("sample_id", "target"),
        rows=(("sample-1", "1"), ("sample-2", "0")),
    )
    _write_table(
        ancestry_path,
        header=("sample_id", "PC1", "PC2"),
        rows=(("sample-1", "0.1", "0.2"), ("sample-2", "0.3", "0.4")),
    )

    merged_path, pc_cols = aou_runner.merge_pcs_into_sample_table(
        sample_table_path=sample_table_path,
        ancestry_path=ancestry_path,
        output_path=output_path,
        n_pcs=2,
    )

    assert merged_path == output_path
    assert pc_cols == ["PC1", "PC2"]
    rows = _read_tsv_rows(output_path)
    assert rows[0]["sample_id"] == "sample-1"
    assert rows[0]["PC1"] == "0.1"
    assert rows[1]["PC2"] == "0.4"


def test_download_sv_vcf_downloads_one_chromosome_at_a_time(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CDR_STORAGE_PATH", "gs://bucket/cdr")
    monkeypatch.setenv("GOOGLE_PROJECT", "billing-project")
    monkeypatch.setattr(aou_runner, "_check_disk_space", lambda path, required_bytes: None)
    monkeypatch.setattr(aou_runner, "_gsutil_size", lambda path: 100)

    copied: list[tuple[str, str]] = []

    def fake_cp(src: str, dst: str) -> None:
        copied.append((src, dst))
        Path(dst).write_text("data\n", encoding="utf-8")

    monkeypatch.setattr(aou_runner, "_gsutil_cp", fake_cp)

    work_dir = tmp_path / "run"
    cache_dir = aou_runner.local_sv_vcf_cache_dir(work_dir)
    local_vcf = aou_runner.download_sv_vcf(22, work_dir)

    assert local_vcf == cache_dir / "AoU_srWGS_SV.v8.chr22.vcf.gz"
    assert copied == [
        (
            "gs://bucket/cdr/wgs/short_read/structural_variants/vcf/full/AoU_srWGS_SV.v8.chr22.vcf.gz",
            str(cache_dir / "AoU_srWGS_SV.v8.chr22.vcf.gz.partial"),
        ),
        (
            "gs://bucket/cdr/wgs/short_read/structural_variants/vcf/full/AoU_srWGS_SV.v8.chr22.vcf.gz.tbi",
            str(cache_dir / "AoU_srWGS_SV.v8.chr22.vcf.gz.tbi.partial"),
        ),
    ]
    assert local_vcf.exists()
    assert Path(f"{local_vcf}.tbi").exists()


def test_download_sv_vcf_downloads_missing_index_for_existing_vcf(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CDR_STORAGE_PATH", "gs://bucket/cdr")
    monkeypatch.setenv("GOOGLE_PROJECT", "billing-project")
    monkeypatch.setattr(aou_runner, "_check_disk_space", lambda path, required_bytes: None)

    work_dir = tmp_path / "run"
    cache_dir = aou_runner.local_sv_vcf_cache_dir(work_dir)
    local_vcf = cache_dir / "AoU_srWGS_SV.v8.chr22.vcf.gz"
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_vcf.write_text("data\n", encoding="utf-8")

    sized: list[str] = []
    copied: list[tuple[str, str]] = []

    def fake_size(path: str) -> int:
        sized.append(path)
        return 100

    def fake_cp(src: str, dst: str) -> None:
        copied.append((src, dst))
        Path(dst).write_text("data\n", encoding="utf-8")

    monkeypatch.setattr(aou_runner, "_gsutil_size", fake_size)
    monkeypatch.setattr(aou_runner, "_gsutil_cp", fake_cp)

    local_path = aou_runner.download_sv_vcf(22, work_dir)

    assert local_path == local_vcf
    assert sized == [
        "gs://bucket/cdr/wgs/short_read/structural_variants/vcf/full/AoU_srWGS_SV.v8.chr22.vcf.gz.tbi"
    ]
    assert copied == [
        (
            "gs://bucket/cdr/wgs/short_read/structural_variants/vcf/full/AoU_srWGS_SV.v8.chr22.vcf.gz.tbi",
            str(cache_dir / "AoU_srWGS_SV.v8.chr22.vcf.gz.tbi.partial"),
        )
    ]


def test_merge_pcs_into_sample_table_raises_when_ids_do_not_overlap(tmp_path: Path):
    sample_table_path = tmp_path / "samples.tsv"
    ancestry_path = tmp_path / "ancestry.tsv"
    output_path = tmp_path / "merged.tsv"
    _write_table(
        sample_table_path,
        header=("sample_id", "person_id", "target"),
        rows=(
            ("sample-1", "person-1", "1"),
            ("sample-2", "person-2", "0"),
        ),
    )
    _write_table(
        ancestry_path,
        header=("research_id", "PC1", "PC2"),
        rows=(
            ("ancestry-1", "0.1", "0.2"),
            ("ancestry-2", "0.3", "0.4"),
        ),
    )

    with pytest.raises(RuntimeError, match="No ID overlap"):
        aou_runner.merge_pcs_into_sample_table(
            sample_table_path=sample_table_path,
            ancestry_path=ancestry_path,
            output_path=output_path,
            n_pcs=2,
        )


def test_merge_pcs_into_sample_table_raises_when_merge_produces_zero_pcs(tmp_path: Path):
    sample_table_path = tmp_path / "samples.tsv"
    ancestry_path = tmp_path / "ancestry.tsv"
    output_path = tmp_path / "merged.tsv"
    _write_table(
        sample_table_path,
        header=("sample_id", "person_id", "target"),
        rows=(
            ("sample-1", "person-1", "1"),
            ("sample-2", "person-2", "0"),
        ),
    )
    _write_table(
        ancestry_path,
        header=("person_id", "PC1", "PC2"),
        rows=(
            ("person-1", "", ""),
            ("person-2", "", ""),
        ),
    )

    with pytest.raises(RuntimeError, match="zero rows with PCs"):
        aou_runner.merge_pcs_into_sample_table(
            sample_table_path=sample_table_path,
            ancestry_path=ancestry_path,
            output_path=output_path,
            n_pcs=2,
        )


def test_run_all_of_us_runs_single_unified_fit_and_reuses_cached_downloads(monkeypatch, tmp_path: Path):
    class _Dataset:
        def __init__(self) -> None:
            self.targets = np.array([0.0, 1.0], dtype=np.float32)

    ancestry_path = tmp_path / "ancestry_preds.tsv"
    ancestry_path.write_text("research_id\tpca_features\n1\t[0.1,0.2]\n", encoding="utf-8")

    release_calls: list[str] = []
    loader_calls: list[list[str]] = []
    pipeline_calls: list[tuple[int, Path, float, int]] = []

    def fake_prepare(request, output_path, **kwargs):
        Path(output_path).write_text("sample_id\tperson_id\ttarget\tage_at_observation_start\tgender_concept_id\trace_concept_id\tethnicity_concept_id\n", encoding="utf-8")

    def fake_merge(sample_table_path, ancestry_path, output_path, n_pcs):
        Path(output_path).write_text("sample_id\tperson_id\ttarget\tage_at_observation_start\tage_squared\tgender_concept_id\trace_concept_id\tethnicity_concept_id\n", encoding="utf-8")
        return output_path, []

    def fake_download_sv_vcf(chromosome: int, work_dir: Path) -> Path:
        vcf_path = aou_runner.local_sv_vcf_path(chromosome, work_dir)
        vcf_path.parent.mkdir(parents=True, exist_ok=True)
        Path(vcf_path).write_text("vcf\n", encoding="utf-8")
        Path(f"{vcf_path}.tbi").write_text("tbi\n", encoding="utf-8")
        return vcf_path

    monkeypatch.setattr(aou_runner, "prepare_all_of_us_disease_sample_table", fake_prepare)
    monkeypatch.setattr(aou_runner, "download_ancestry_preds", lambda work_dir: ancestry_path)
    monkeypatch.setattr(aou_runner, "merge_pcs_into_sample_table", fake_merge)
    monkeypatch.setattr(aou_runner, "download_sv_vcf", fake_download_sv_vcf)
    monkeypatch.setattr("sv_pgs.io.precache_vcfs_parallel", lambda vcf_paths, config: None)
    monkeypatch.setattr(
        aou_runner,
        "build_aou_sv_variant_metadata",
        lambda *, vcf_paths, output_path: _write_aou_variant_metadata_stub(output_path),
    )
    monkeypatch.setattr(aou_runner, "release_process_memory", lambda: release_calls.append("released"))
    def fake_load_multi_vcf_dataset_from_files(**kwargs):
        loader_calls.append([str(path) for path in kwargs["genotype_paths"]])
        assert kwargs["variant_metadata_path"] == tmp_path / "variant_metadata.tsv.gz"
        return _Dataset()

    def fake_run_training_pipeline(**kwargs):
        pipeline_calls.append(
            (
                kwargs["dataset"].targets.shape[0],
                Path(kwargs["output_dir"]),
            )
        )
        return None

    monkeypatch.setattr(aou_runner, "load_multi_vcf_dataset_from_files", fake_load_multi_vcf_dataset_from_files)
    monkeypatch.setattr(aou_runner, "run_training_pipeline", fake_run_training_pipeline)

    aou_runner.run_all_of_us(
        disease="heart_failure",
        chromosomes=[1, 2],
        output_base=str(tmp_path),
    )

    cache_dir = aou_runner.local_sv_vcf_cache_dir(tmp_path)
    assert (cache_dir / "AoU_srWGS_SV.v8.chr1.vcf.gz").exists()
    assert (cache_dir / "AoU_srWGS_SV.v8.chr1.vcf.gz.tbi").exists()
    assert (cache_dir / "AoU_srWGS_SV.v8.chr2.vcf.gz").exists()
    assert (cache_dir / "AoU_srWGS_SV.v8.chr2.vcf.gz.tbi").exists()
    assert loader_calls == [[
        str(cache_dir / "AoU_srWGS_SV.v8.chr1.vcf.gz"),
        str(cache_dir / "AoU_srWGS_SV.v8.chr2.vcf.gz"),
    ]]
    assert pipeline_calls == [(2, tmp_path)]
    assert release_calls == ["released"]


def test_run_all_of_us_rejects_duplicate_or_invalid_chromosomes(tmp_path: Path):
    with pytest.raises(ValueError, match="chromosomes must be unique"):
        aou_runner.run_all_of_us(
            disease="heart_failure",
            chromosomes=[1, 1],
            output_base=str(tmp_path),
        )

    with pytest.raises(ValueError, match="chromosomes must be autosomes 1-22"):
        aou_runner.run_all_of_us(
            disease="heart_failure",
            chromosomes=[0, 23],
            output_base=str(tmp_path),
        )


def test_build_aou_sv_variant_metadata_extracts_prior_annotation_features(tmp_path: Path):
    vcf_path = tmp_path / "annotated_sv.vcf"
    vcf_path.write_text(
        "\n".join(
            [
                "##fileformat=VCFv4.2",
                "##contig=<ID=1>",
                "##INFO=<ID=AF,Number=A,Type=Float,Description=\"Allele frequency\">",
                "##INFO=<ID=SVTYPE,Number=1,Type=String,Description=\"SV type\">",
                "##INFO=<ID=SVLEN,Number=1,Type=Integer,Description=\"SV length\">",
                "##INFO=<ID=ALGORITHMS,Number=.,Type=String,Description=\"Calling algorithms\">",
                "##INFO=<ID=PREDICTED_LOF,Number=.,Type=String,Description=\"Predicted LOF genes\">",
                "##INFO=<ID=PREDICTED_NONCODING_SPAN,Number=.,Type=String,Description=\"Noncoding span annotations\">",
                "##INFO=<ID=CNQ,Number=1,Type=Float,Description=\"Copy number quality\">",
                "##INFO=<ID=LOEUF,Number=1,Type=Float,Description=\"Constraint score\">",
                "##INFO=<ID=PHYLOP,Number=1,Type=Float,Description=\"Conservation score\">",
                "##INFO=<ID=REPEATMASKER,Number=1,Type=String,Description=\"Repeat annotation\">",
                "##INFO=<ID=PREDICTED_PROMOTER,Number=.,Type=String,Description=\"Promoter overlaps\">",
                "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">",
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1",
                "1\t100\tsv_lof\tN\t<DEL>\t80\tPASS\tAF=0.01;SVTYPE=DEL;SVLEN=-1200;ALGORITHMS=MANTA,WHAM;PREDICTED_LOF=GENE1,GENE2;PREDICTED_NONCODING_SPAN=enhancer,promoter;CNQ=42;LOEUF=0.12;PHYLOP=1.5\tGT\t0/1",
                "1\t300\tsv_repeat\tN\t<DUP>\t50\tPASS\tAF=0.05;SVTYPE=DUP;SVLEN=600;ALGORITHMS=DEPTH;PREDICTED_PROMOTER=GENE3;REPEATMASKER=LINE\tGT\t0/1",
                "",
            ]
        ),
        encoding="utf-8",
    )

    metadata_path = aou_runner.build_aou_sv_variant_metadata(
        vcf_paths=[vcf_path],
        output_path=tmp_path / "variant_metadata.tsv.gz",
    )

    rows = _read_tsv_rows(metadata_path)
    assert [row["variant_id"] for row in rows] == ["sv_lof", "sv_repeat"]
    assert rows[0]["prior_binary__predicted_lof"] == "true"
    assert rows[0]["prior_continuous__lof_gene_count"] == "2"
    assert rows[0]["prior_membership__calling_algorithms"] == "manta=0.5,wham=0.5"
    assert rows[0]["prior_membership__noncoding_span"] == "enhancer=0.5,promoter=0.5"
    assert rows[0]["prior_categorical__strongest_effect"] == "lof"
    assert rows[0]["prior_nested__functional_context"] == "genic>loss_of_function"
    assert rows[0]["prior_continuous__constraint_score"] == "0.12"
    assert rows[0]["prior_continuous__conservation_score"] == "1.5"
    assert rows[1]["prior_binary__is_repeat"] == "true"
    assert rows[1]["prior_binary__predicted_promoter"] == "true"
    assert rows[1]["prior_categorical__strongest_effect"] == "promoter"

    sample_table_path = tmp_path / "samples.tsv"
    sample_table_path.write_text("sample_id\ttarget\ns1\t1\n", encoding="utf-8")
    dataset = load_multi_vcf_dataset_from_files(
        genotype_paths=[vcf_path],
        config=ModelConfig(),
        sample_table_path=sample_table_path,
        sample_id_column="sample_id",
        target_column="target",
        covariate_columns=(),
        variant_metadata_path=metadata_path,
    )
    assert dataset.variant_records[0].prior_binary_features["predicted_lof"] is True
    assert dataset.variant_records[0].prior_membership_features["calling_algorithms"] == {"manta": 0.5, "wham": 0.5}
    assert dataset.variant_records[0].prior_categorical_features["strongest_effect"] == "lof"
    assert dataset.variant_records[0].prior_nested_features["functional_context"] == ("genic", "loss_of_function")


def test_run_all_of_us_skips_existing_fit_only_when_run_metadata_matches(monkeypatch, tmp_path: Path):
    disease = "heart_failure"
    sample_table_path = tmp_path / f"{disease}.samples.tsv"
    sample_table_path.write_text(
        "sample_id\tperson_id\ttarget\tage_at_observation_start\tgender_concept_id\trace_concept_id\tethnicity_concept_id\n",
        encoding="utf-8",
    )
    ancestry_path = tmp_path / "ancestry_preds.tsv"
    ancestry_path.write_text("research_id\tpca_features\n1\t[0.1,0.2]\n", encoding="utf-8")
    with gzip.open(tmp_path / "summary.json.gz", "wt", encoding="utf-8") as handle:
        handle.write("{}")

    pc_cols = ["PC1", "PC2"]
    covariates = aou_runner.DEFAULT_COVARIATES + pc_cols
    aou_runner._aou_run_metadata_path(tmp_path).write_text(
        json.dumps(
            aou_runner._build_aou_run_metadata(
                disease=disease,
                chromosomes=[1, 2],
                n_pcs=2,
                pc_cols=pc_cols,
                covariates=covariates,
                max_outer_iterations=30,
                pipeline_validation_fraction=0.0,
                pipeline_validation_min_samples=0,
                random_seed=0,
                variant_metadata_schema_version=aou_runner._AOU_VARIANT_METADATA_SCHEMA_VERSION,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    def fake_merge(sample_table_path, ancestry_path, output_path, n_pcs):
        Path(output_path).write_text(
            "sample_id\tperson_id\ttarget\tage_at_observation_start\tage_squared\tgender_concept_id\trace_concept_id\tethnicity_concept_id\tPC1\tPC2\n",
            encoding="utf-8",
        )
        return output_path, pc_cols

    monkeypatch.setattr(aou_runner, "download_ancestry_preds", lambda work_dir: ancestry_path)
    monkeypatch.setattr(aou_runner, "merge_pcs_into_sample_table", fake_merge)
    monkeypatch.setattr(
        aou_runner,
        "download_sv_vcf",
        lambda chromosome, work_dir: (_ for _ in ()).throw(AssertionError("download_sv_vcf should not run")),
    )
    monkeypatch.setattr(
        aou_runner,
        "load_multi_vcf_dataset_from_files",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("dataset loading should not run")),
    )
    monkeypatch.setattr(
        aou_runner,
        "run_training_pipeline",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("training should not run")),
    )

    aou_runner.run_all_of_us(
        disease=disease,
        chromosomes=[1, 2],
        output_base=str(tmp_path),
        n_pcs=2,
    )


def test_run_all_of_us_reruns_when_existing_fit_metadata_differs(monkeypatch, tmp_path: Path):
    class _Dataset:
        def __init__(self) -> None:
            self.targets = np.array([0.0, 1.0], dtype=np.float32)

    disease = "heart_failure"
    sample_table_path = tmp_path / f"{disease}.samples.tsv"
    sample_table_path.write_text(
        "sample_id\tperson_id\ttarget\tage_at_observation_start\tgender_concept_id\trace_concept_id\tethnicity_concept_id\n",
        encoding="utf-8",
    )
    ancestry_path = tmp_path / "ancestry_preds.tsv"
    ancestry_path.write_text("research_id\tpca_features\n1\t[0.1,0.2,0.3]\n", encoding="utf-8")
    with gzip.open(tmp_path / "summary.json.gz", "wt", encoding="utf-8") as handle:
        handle.write("{}")
    aou_runner._aou_run_metadata_path(tmp_path).write_text(
        json.dumps(
            aou_runner._build_aou_run_metadata(
                disease=disease,
                chromosomes=[1, 2],
                n_pcs=2,
                pc_cols=["PC1", "PC2"],
                covariates=aou_runner.DEFAULT_COVARIATES + ["PC1", "PC2"],
                max_outer_iterations=30,
                pipeline_validation_fraction=0.0,
                pipeline_validation_min_samples=0,
                random_seed=0,
                variant_metadata_schema_version=aou_runner._AOU_VARIANT_METADATA_SCHEMA_VERSION,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    loader_calls: list[list[str]] = []
    pipeline_calls: list[Path] = []

    def fake_merge(sample_table_path, ancestry_path, output_path, n_pcs):
        Path(output_path).write_text(
            "sample_id\tperson_id\ttarget\tage_at_observation_start\tage_squared\tgender_concept_id\trace_concept_id\tethnicity_concept_id\tPC1\tPC2\tPC3\n",
            encoding="utf-8",
        )
        return output_path, ["PC1", "PC2", "PC3"]

    def fake_download_sv_vcf(chromosome: int, work_dir: Path) -> Path:
        vcf_path = aou_runner.local_sv_vcf_path(chromosome, work_dir)
        vcf_path.parent.mkdir(parents=True, exist_ok=True)
        Path(vcf_path).write_text("vcf\n", encoding="utf-8")
        Path(f"{vcf_path}.tbi").write_text("tbi\n", encoding="utf-8")
        return vcf_path

    monkeypatch.setattr(aou_runner, "download_ancestry_preds", lambda work_dir: ancestry_path)
    monkeypatch.setattr(aou_runner, "merge_pcs_into_sample_table", fake_merge)
    monkeypatch.setattr(aou_runner, "download_sv_vcf", fake_download_sv_vcf)
    monkeypatch.setattr("sv_pgs.io.precache_vcfs_parallel", lambda vcf_paths, config: None)
    monkeypatch.setattr(
        aou_runner,
        "build_aou_sv_variant_metadata",
        lambda *, vcf_paths, output_path: _write_aou_variant_metadata_stub(output_path),
    )
    monkeypatch.setattr(aou_runner, "release_process_memory", lambda: None)
    def fake_load_multi_vcf_dataset_from_files(**kwargs):
        loader_calls.append([str(path) for path in kwargs["genotype_paths"]])
        assert kwargs["variant_metadata_path"] == tmp_path / "variant_metadata.tsv.gz"
        return _Dataset()

    def fake_run_training_pipeline(**kwargs):
        pipeline_calls.append(Path(kwargs["output_dir"]))
        return None

    monkeypatch.setattr(aou_runner, "load_multi_vcf_dataset_from_files", fake_load_multi_vcf_dataset_from_files)
    monkeypatch.setattr(aou_runner, "run_training_pipeline", fake_run_training_pipeline)

    aou_runner.run_all_of_us(
        disease=disease,
        chromosomes=[1, 2],
        output_base=str(tmp_path),
        n_pcs=3,
    )

    cache_dir = aou_runner.local_sv_vcf_cache_dir(tmp_path)
    assert loader_calls == [[
        str(cache_dir / "AoU_srWGS_SV.v8.chr1.vcf.gz"),
        str(cache_dir / "AoU_srWGS_SV.v8.chr2.vcf.gz"),
    ]]
    assert pipeline_calls == [tmp_path]
    rerun_metadata = json.loads(aou_runner._aou_run_metadata_path(tmp_path).read_text(encoding="utf-8"))
    assert rerun_metadata["requested_n_pcs"] == 3
    assert rerun_metadata["effective_pc_columns"] == ["PC1", "PC2", "PC3"]


def test_run_all_of_us_raises_when_parallel_precache_fails(monkeypatch, tmp_path: Path):
    disease = "heart_failure"
    sample_table_path = tmp_path / f"{disease}.samples.tsv"
    sample_table_path.write_text(
        "sample_id\tperson_id\ttarget\tage_at_observation_start\tgender_concept_id\trace_concept_id\tethnicity_concept_id\n",
        encoding="utf-8",
    )
    ancestry_path = tmp_path / "ancestry_preds.tsv"
    ancestry_path.write_text("research_id\tpca_features\n1\t[0.1,0.2]\n", encoding="utf-8")
    vcf_path = tmp_path / "AoU_srWGS_SV.v8.chr1.vcf.gz"
    vcf_path.write_bytes(b"vcf")

    def fake_merge(sample_table_path, ancestry_path, output_path, n_pcs):
        Path(output_path).write_text(
            "sample_id\tperson_id\ttarget\tage_at_observation_start\tage_squared\tgender_concept_id\trace_concept_id\tethnicity_concept_id\tPC1\tPC2\n",
            encoding="utf-8",
        )
        return output_path, ["PC1", "PC2"]

    monkeypatch.setattr(aou_runner, "download_ancestry_preds", lambda work_dir: ancestry_path)
    monkeypatch.setattr(aou_runner, "merge_pcs_into_sample_table", fake_merge)
    monkeypatch.setattr(aou_runner, "download_sv_vcf", lambda chromosome, work_dir: vcf_path)
    monkeypatch.setattr(
        "sv_pgs.io.precache_vcfs_parallel",
        lambda vcf_paths, config: (_ for _ in ()).throw(ValueError("boom")),
    )
    monkeypatch.setattr(
        aou_runner,
        "build_aou_sv_variant_metadata",
        lambda *, vcf_paths, output_path: (_ for _ in ()).throw(AssertionError("metadata build should not run")),
    )
    monkeypatch.setattr(
        aou_runner,
        "load_multi_vcf_dataset_from_files",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("dataset loading should not run")),
    )
    monkeypatch.setattr(
        aou_runner,
        "run_training_pipeline",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("training should not run")),
    )

    with pytest.raises(RuntimeError, match="parallel VCF precache failed"):
        aou_runner.run_all_of_us(
            disease=disease,
            chromosomes=[1],
            output_base=str(tmp_path),
            n_pcs=2,
        )

def _read_tsv_rows(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [{str(key): "" if value is None else str(value) for key, value in row.items()} for row in reader]


def _write_table(path: Path, header: tuple[str, ...], rows: tuple[tuple[str, ...], ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)

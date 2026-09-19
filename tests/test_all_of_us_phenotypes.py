"""All of Us phenotypes: the disease panel and the quantitative traits.

Covers the disease and trait catalogues, the BigQuery SQL against the OMOP CDM
v5.4 schema and its parameters, the disease case and control rules with the
age-of-onset liability target, the per-person quantitative target
(treatment precedence and corrections, variance components, the empirical
BLUP against the dense Henderson mixed-model solution), the sample-table and
census writers, and the CLI commands that prepare them.
"""
from __future__ import annotations

import csv
import dataclasses
import datetime
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest
from google.cloud import bigquery
from scipy.stats import norm, truncnorm

from sv_pgs.all_of_us import (
    DIABETES,
    DISEASE_DEFINITIONS,
    LDL_LOWERING,
    MEASUREMENT_DEFINITIONS,
    MEASUREMENT_EXCLUSION_REASONS,
    UNBOUNDED_WINDOW_DAYS,
    ClinicalWindow,
    DiseaseDefinition,
    MeasurementDefinition,
    TreatmentCorrection,
    TreatmentRule,
    UnitConversion,
    WindowConcept,
    _liability_targets,
    _occasion_design,
    _prepare_training_rows,
    available_disease_names,
    available_measurement_names,
    build_all_of_us_disease_query_config,
    build_all_of_us_disease_query_parameters,
    build_all_of_us_disease_sql,
    build_all_of_us_lab_criterion_query_parameters,
    build_all_of_us_measurement_query_config,
    build_all_of_us_measurement_query_parameters,
    build_all_of_us_measurement_sql,
    build_all_of_us_measurement_targets,
    person_occasions,
    disease_covariate_columns,
    measurement_covariate_columns,
    phenotype_fingerprint,
    prepare_all_of_us_disease_sample_table,
    prepare_all_of_us_measurement_census,
    prepare_all_of_us_measurement_sample_table,
    resolve_all_of_us_phenotype,
    resolve_disease_definition,
    resolve_lab_criterion_measurement,
    resolve_measurement_definition,
)
from sv_pgs.cli import main
from sv_pgs.phenotype_measurement import Occasions, fit_at_exponent
from tests.phenotype_bounds import (
    rounding_gamma,
    sampling_bound,
    variance_component_standard_errors,
    within_rounding,
)

# Columns of the OMOP CDM v5.4 tables the query reads, from the official
# BigQuery DDL (github.com/OHDSI/CommonDataModel, inst/ddl/5.4/bigquery/
# OMOPCDM_bigquery_5.4_ddl.sql). person.sex_at_birth_concept_id is the All of
# Us CDR extension of the person table.
OMOP_CDM_V54_COLUMNS: dict[str, set[str]] = {
    "person": {
        "person_id", "gender_concept_id", "year_of_birth", "month_of_birth", "day_of_birth",
        "birth_datetime", "race_concept_id", "ethnicity_concept_id", "location_id", "provider_id",
        "care_site_id", "person_source_value", "gender_source_value", "gender_source_concept_id",
        "race_source_value", "race_source_concept_id", "ethnicity_source_value",
        "ethnicity_source_concept_id",
        "sex_at_birth_concept_id",
    },
    "measurement": {
        "measurement_id", "person_id", "measurement_concept_id", "measurement_date",
        "measurement_datetime", "measurement_time", "measurement_type_concept_id",
        "operator_concept_id", "value_as_number", "value_as_concept_id", "unit_concept_id",
        "range_low", "range_high", "provider_id", "visit_occurrence_id", "visit_detail_id",
        "measurement_source_value", "measurement_source_concept_id", "unit_source_value",
        "unit_source_concept_id", "value_source_value", "measurement_event_id",
        "meas_event_field_concept_id",
    },
    "observation": {
        "observation_id", "person_id", "observation_concept_id", "observation_date",
        "observation_datetime", "observation_type_concept_id", "value_as_number", "value_as_string",
        "value_as_concept_id", "qualifier_concept_id", "unit_concept_id", "provider_id",
        "visit_occurrence_id", "visit_detail_id", "observation_source_value",
        "observation_source_concept_id", "unit_source_value", "qualifier_source_value",
        "value_source_value", "observation_event_id", "obs_event_field_concept_id",
    },
    "condition_occurrence": {
        "condition_occurrence_id", "person_id", "condition_concept_id", "condition_start_date",
        "condition_start_datetime", "condition_end_date", "condition_end_datetime",
        "condition_type_concept_id", "condition_status_concept_id", "stop_reason", "provider_id",
        "visit_occurrence_id", "visit_detail_id", "condition_source_value",
        "condition_source_concept_id", "condition_status_source_value",
    },
    "drug_exposure": {
        "drug_exposure_id", "person_id", "drug_concept_id", "drug_exposure_start_date",
        "drug_exposure_start_datetime", "drug_exposure_end_date", "drug_exposure_end_datetime",
        "verbatim_end_date", "drug_type_concept_id", "stop_reason", "refills", "quantity",
        "days_supply", "sig", "route_concept_id", "lot_number", "provider_id",
        "visit_occurrence_id", "visit_detail_id", "drug_source_value", "drug_source_concept_id",
        "route_source_value", "dose_unit_source_value",
    },
    "visit_occurrence": {
        "visit_occurrence_id", "person_id", "visit_concept_id", "visit_start_date",
        "visit_start_datetime", "visit_end_date", "visit_end_datetime", "visit_type_concept_id",
        "provider_id", "care_site_id", "visit_source_value", "visit_source_concept_id",
        "admitted_from_concept_id", "admitted_from_source_value", "discharged_to_concept_id",
        "discharged_to_source_value", "preceding_visit_occurrence_id",
    },
    "concept": {
        "concept_id", "concept_name", "domain_id", "vocabulary_id", "concept_class_id",
        "standard_concept", "concept_code", "valid_start_date", "valid_end_date", "invalid_reason",
    },
    "procedure_occurrence": {
        "procedure_occurrence_id", "person_id", "procedure_concept_id", "procedure_date",
        "procedure_datetime", "procedure_end_date", "procedure_end_datetime",
        "procedure_type_concept_id", "modifier_concept_id", "quantity", "provider_id",
        "visit_occurrence_id", "visit_detail_id", "procedure_source_value",
        "procedure_source_concept_id", "modifier_source_value",
    },
    "concept_ancestor": {
        "ancestor_concept_id", "descendant_concept_id", "min_levels_of_separation",
        "max_levels_of_separation",
    },
}
DATASET = "aou_workspace.cdr_dataset"


@pytest.fixture(autouse=True)
def _workspace_cdr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORKSPACE_CDR", DATASET)


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


def _read_tsv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _disease_row(
    person_id: int,
    *,
    occurrence_count: int,
    sex_at_birth_concept_id: int = 45878463,
    pre_landmark_condition_dates: int = 12,
    control_exclusion: bool = False,
    ambiguous: bool = False,
    case_medication_dates: int = 0,
    control_exclusion_medication: bool = False,
    case_procedure_dates: int = 0,
) -> dict[str, object]:
    """One row shaped like the disease query output."""
    return {
        "sample_id": str(person_id),
        "person_id": str(person_id),
        "phenotype_occurrence_count": occurrence_count,
        "first_condition_date": "2020-01-01" if occurrence_count else None,
        "observation_start_date": "2015-01-01",
        "observation_end_date": "2024-01-01",
        "primary_consent_date": "2018-06-01",
        "year_of_birth": 1975,
        "age_at_first_condition": 40.0 + person_id % 30 if occurrence_count else None,
        "age_at_first_case_procedure": 50.0 if case_procedure_dates else None,
        "age_at_observation_end": 45.0 + person_id % 30,
        "pre_landmark_condition_dates": pre_landmark_condition_dates,
        "has_control_exclusion_code": control_exclusion,
        "has_ambiguous_code": ambiguous,
        "case_medication_dates": case_medication_dates,
        "has_control_exclusion_medication": control_exclusion_medication,
        "case_procedure_dates": case_procedure_dates,
        "sex_at_birth_concept_id": sex_at_birth_concept_id,
        "sex_at_birth_name": {45878463: "female", 45880669: "male"}.get(sex_at_birth_concept_id),
    }


def _lab_row(count: int, first: str, last: str) -> dict[str, object]:
    return {"qualifying_occasion_count": count, "first_qualifying_date": first, "last_qualifying_date": last}


def _person_days(
    person_id: int,
    occasions: list[tuple[float, float]] | None = None,
    *,
    treated_occasions: list[tuple[float, float]] | None = None,
    sex_at_birth_concept_id: int | None = 45878463,
    excluded: dict[str, int] | None = None,
    unrecognized_unit_labels: list[str] | None = None,
) -> list[dict[str, object]]:
    """Person-day rows shaped like the measurement query output: one per (age, value) occasion, untreated then
    treated, a day later each, plus one day holding the excluded rows when there are any."""
    sex_name = {45878463: "female", 45880669: "male"}.get(sex_at_birth_concept_id)
    days = [(age, value, False) for age, value in occasions or []] + [(age, value, True) for age, value in treated_occasions or []]
    rows: list[dict[str, object]] = []
    start = datetime.date(2015, 1, 1)
    for position, (age, value, treated) in enumerate(days):
        rows.append({
            "sample_id": str(person_id), "person_id": str(person_id),
            "measurement_date": start + datetime.timedelta(days=position), "age_at_occasion": age, "treated": treated,
            "row_count": 1, **{f"{reason}_row_count": 0 for reason in MEASUREMENT_EXCLUSION_REASONS},
            "retained_row_count": 1, "occasion_value": value, "unrecognized_unit_labels": [],
            "sex_at_birth_concept_id": sex_at_birth_concept_id, "sex_at_birth_name": sex_name,
        })
    if excluded or unrecognized_unit_labels:
        counts = {reason: 0 for reason in MEASUREMENT_EXCLUSION_REASONS} | (excluded or {})
        rows.append({
            "sample_id": str(person_id), "person_id": str(person_id),
            "measurement_date": start + datetime.timedelta(days=len(days)), "age_at_occasion": 50.0, "treated": False,
            "row_count": sum(counts.values()), **{f"{reason}_row_count": count for reason, count in counts.items()},
            "retained_row_count": 0, "occasion_value": None, "unrecognized_unit_labels": unrecognized_unit_labels or [],
            "sex_at_birth_concept_id": sex_at_birth_concept_id, "sex_at_birth_name": sex_name,
        })
    return rows


def _synthetic_person_rows(
    person_count: int,
    *,
    level_variance: float,
    noise_variance: float,
    seed: int,
) -> tuple[list[dict[str, object]], np.ndarray]:
    """Person-day rows simulated from the per-occasion model on the linear scale, recorded to 0.1 unit as a lab
    reports them; returns the rows and each person's true long-run level net of the age and sex terms."""
    generator = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    levels = generator.normal(0.0, math.sqrt(level_variance), person_count)
    for person_index in range(person_count):
        female = person_index % 2 == 0
        ages = generator.uniform(25.0, 80.0, int(generator.integers(1, 7)))
        values = 90.0 + 0.03 * (ages - 50.0) + (0.5 if female else 0.0) + levels[person_index]
        values = np.round(values + generator.normal(0.0, math.sqrt(noise_variance), ages.shape[0]), 1)
        rows += _person_days(
            person_index,
            [(float(age), float(value)) for age, value in zip(ages, values, strict=True)],
            sex_at_birth_concept_id=45878463 if female else 45880669,
        )
    return rows, levels


_BASE_DEFINITION = MeasurementDefinition(
    canonical_name="test_trait",
    aliases=(),
    description="test",
    loinc_codes=("1-1",),
    canonical_unit="milligram per deciliter",
    unit_conversions=(UnitConversion("milligram per deciliter", 1.0),),
)


def _definition(**overrides: object) -> MeasurementDefinition:
    return dataclasses.replace(_BASE_DEFINITION, **overrides)


def _captured_occasions(monkeypatch, definition: MeasurementDefinition, rows) -> tuple[Occasions, list, dict]:
    """Run the target builder with the model replaced by a stand-in; return the occasions it was given, and the
    training rows and summary it built from the stand-in's levels (each person's index, reliability one half)."""
    captured: dict[str, Occasions] = {}

    def stand_in(occasions: Occasions, working_bytes: int):
        captured["occasions"] = occasions
        persons = occasions.person_count
        return type("Fit", (), {
            "exponent": 1.0, "level_variance": 1.0, "noise_second_moment": 1.0, "log_evidence": 0.0,
            "level_mean": np.arange(persons, dtype=np.float64), "reliability": np.full(persons, 0.5),
        })()

    monkeypatch.setattr("sv_pgs.all_of_us.fit_occasion_model", stand_in)
    training_rows, _columns, summary = build_all_of_us_measurement_targets(definition, rows, 1 << 28)
    return captured["occasions"], training_rows, summary


# ---------------------------------------------------------------------------
# Diseases
# ---------------------------------------------------------------------------


def test_build_all_of_us_disease_sql_uses_workspace_cdr_and_snomed_concept_ancestor(monkeypatch):
    monkeypatch.setenv("WORKSPACE_CDR", "aou_workspace.cdr_dataset")
    disease_definition = resolve_disease_definition("atrial_fibrillation")
    sql = build_all_of_us_disease_sql(disease_definition)
    query_config = build_all_of_us_disease_query_config(disease_definition)

    assert "`aou_workspace.cdr_dataset.condition_occurrence`" in sql
    assert "`aou_workspace.cdr_dataset.observation_period`" in sql
    assert "`aou_workspace.cdr_dataset.person`" in sql
    assert "condition_concept_id" in sql
    assert "vocabulary_id = 'SNOMED'" in sql
    assert "standard_concept = 'S'" in sql
    assert "FROM `aou_workspace.cdr_dataset.concept`" in sql
    assert "`aou_workspace.cdr_dataset.concept_ancestor`" in sql
    assert "JOIN `aou_workspace.cdr_dataset.observation` AS observation" in sql
    assert "concept_code IN UNNEST(@case_snomed_codes)" in sql
    assert "primary_consent_date" in sql
    # A case needs MIN_DISEASE_OCCURRENCES distinct diagnosis dates, not rows.
    assert (
        "DISTINCT IF(disease_concepts.concept_id IS NOT NULL, condition_occurrence.condition_start_date, NULL)"
    ) in sql
    assert "COUNT(*) AS phenotype_occurrence_count" not in sql
    # The SNOMED codes must NOT be string-interpolated into the SQL itself.
    assert all(code not in sql for code in disease_definition.case_snomed_codes)
    parameter_values = {parameter.name: parameter for parameter in query_config.query_parameters}
    assert parameter_values["case_snomed_codes"].values == ["49436004", "5370000"]


def test_prepare_all_of_us_disease_sample_table_writes_outputs(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GOOGLE_PROJECT", "billing-project")
    monkeypatch.setenv("WORKSPACE_CDR", "aou_workspace.cdr_dataset")
    fake_client = _FakeBigQueryClient(
        rows=[
            _disease_row(101, occurrence_count=3, sex_at_birth_concept_id=45880669),
            _disease_row(102, occurrence_count=0, sex_at_birth_concept_id=45880669),
            _disease_row(103, occurrence_count=1, sex_at_birth_concept_id=45878463),
        ]
    )

    output_path = tmp_path / "atrial_fibrillation.tsv"
    outputs = prepare_all_of_us_disease_sample_table(
        "atrial_fibrillation",
        output_path=output_path,
        client=fake_client,
    )

    assert outputs.sample_table_path.is_file()
    assert outputs.sql_path.is_file()
    assert outputs.metadata_path.is_file()
    assert fake_client.sql is not None
    assert "atrial_fibrillation" not in fake_client.sql
    assert fake_client.job_config is not None
    parameter_values = {parameter.name: parameter for parameter in fake_client.job_config.query_parameters}
    assert parameter_values["case_snomed_codes"].values == ["49436004", "5370000"]

    rows = _read_tsv_rows(output_path)
    assert rows[0]["sample_id"] == "101"
    assert rows[0]["person_id"] == "101"
    assert rows[0]["target"] == "1"
    assert rows[1]["sample_id"] == "102"
    assert float(rows[0]["liability_target"]) > 0.0 > float(rows[1]["liability_target"])
    assert rows[0]["sex_at_birth_concept_id_45880669"] == "1"
    metadata_payload = json.loads(outputs.metadata_path.read_text(encoding="utf-8"))
    assert metadata_payload["row_count"] == 2
    assert metadata_payload["n_excluded_one_date"] == 1
    assert metadata_payload["covariate_columns"] == list(disease_covariate_columns())
    assert metadata_payload["phenotype_fingerprint"] == phenotype_fingerprint(resolve_disease_definition("afib"))
    assert metadata_payload["disease"] == "atrial_fibrillation"
    assert metadata_payload["case_snomed_codes"] == ["49436004", "5370000"]
    assert metadata_payload["case_medication"]["atc_codes"] == ["C01B", "C01AA"]
    assert metadata_payload["lab_criteria"] == []
    assert "icd10_prefixes" not in metadata_payload
    assert "icd9_prefixes" not in metadata_payload
    assert metadata_payload["min_occurrences"] == 2
    assert metadata_payload["cdr_dataset"] == "aou_workspace.cdr_dataset"


def _case_liability_within_rounding(probability: float):
    """ndtri(probability), a case's liability, to its float64 forward error bound.

    Either side reaches it in at most 10 rounded operations: two Kaplan-Meier
    steps (3 each), the survival mid-point (2) and ndtri (2). The relative
    condition number of z = ndtri(p) is p / (phi(z) |z|).
    """
    liability = norm.ppf(probability)
    return within_rounding(liability, 10, probability / (norm.pdf(liability) * abs(liability)))


def _control_liability_within_rounding(survival: float):
    """-phi(z) / S with z = ndtri(S), a control's liability, to its float64 forward error bound.

    Either side reaches it in at most 15 rounded operations: two Kaplan-Meier
    steps (3 each), ndtri (2), the normal density (6) and the quotient (1). The
    relative condition number with respect to S is |1 + z S / phi(z)|.
    """
    threshold = norm.ppf(survival)
    liability = truncnorm(-np.inf, threshold).mean()
    return within_rounding(liability, 15, abs(1.0 + threshold * survival / norm.pdf(threshold)))


def test_liability_target_matches_the_age_of_onset_threshold_model():
    # One stratum: cases diagnosed at 50 and 60, controls censored at 55, 65, 70.
    # Kaplan-Meier: S(50) = 4/5, S(60) = 4/5 * 2/3.
    targets = np.array([1.0, 0.0, 1.0, 0.0, 0.0])
    ages = np.array([50.0, 55.0, 60.0, 65.0, 70.0])
    liabilities = _liability_targets(targets, ages, ["female"] * 5)
    survival_50, survival_60 = 0.8, 0.8 * 2.0 / 3.0
    assert liabilities[0] == _case_liability_within_rounding((1.0 + survival_50) / 2.0)
    assert liabilities[1] == _control_liability_within_rounding(survival_50)
    assert liabilities[2] == _case_liability_within_rounding((survival_50 + survival_60) / 2.0)
    assert liabilities[3] == liabilities[4] == _control_liability_within_rounding(survival_60)
    # Earlier onset means higher liability; every control sits below zero.
    assert liabilities[0] > liabilities[2] > 0.0 > liabilities[1] > liabilities[3]


def test_liability_target_uses_sex_specific_incidence_and_pools_unknown_sex():
    targets = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    ages = np.array([50.0, 60.0, 60.0, 60.0, 70.0, 60.0])
    sexes = ["female", "female", "male", "male", "male", None]
    liabilities = _liability_targets(targets, ages, sexes)
    # Female curve: S(50) = 1/2, so the female case sits at norm.ppf(3/4).
    assert liabilities[0] == _case_liability_within_rounding(0.75)
    # No male event by 60: the male controls' threshold is +inf and their mean 0.
    assert liabilities[2] == liabilities[3] == 0.0
    # The unknown-sex control uses the pooled curve: S(60) = 5/6 after the
    # event at 50 among six people.
    assert liabilities[5] == _control_liability_within_rounding(5.0 / 6.0)


def test_the_disease_panel_is_the_ten_mixed_panel_diseases():
    assert available_disease_names() == sorted([
        "type2_diabetes", "atrial_fibrillation", "hypothyroidism", "copd", "depression", "gout", "cataract",
        "chronic_kidney_disease", "psoriasis", "prostate_cancer",
    ])


def test_panel_diseases_parameterize_their_case_and_control_rules():
    type2_diabetes = build_all_of_us_disease_query_parameters(resolve_disease_definition("t2d"))
    assert type2_diabetes["ambiguous_snomed_codes"] == ("STRING", ["46635009"])
    assert type2_diabetes["control_exclusion_snomed_codes"] == ("STRING", ["73211009", "11687002"])
    # Case support leaves metformin (A10BA) out; control exclusion takes all of A10.
    assert "A10BA" not in type2_diabetes["case_medication_atc_codes"][1]
    assert type2_diabetes["control_exclusion_medication_atc_codes"] == ("STRING", ["A10"])
    hypothyroidism = resolve_disease_definition("hypothyroidism")
    assert hypothyroidism.case_medication_minimum_dates == 2
    cataract = build_all_of_us_disease_query_parameters(resolve_disease_definition("cataract"))
    assert cataract["case_procedure_snomed_codes"] == ("STRING", ["54885007"])
    kidney = resolve_disease_definition("ckd")
    assert [(criterion.measurement, criterion.qualifies_at_or_above, criterion.threshold, criterion.minimum_span_days)
            for criterion in kidney.lab_criteria] == [
        ("egfr_ckd_epi_2021", False, 60.0, 90),
        ("urine_albumin_creatinine_ratio", True, 30.0, 90),
    ]
    prostate = resolve_disease_definition("prostate_cancer")
    assert prostate.required_sex == "male" and prostate.minimum_control_age_years == 50.0
    assert resolve_disease_definition("copd").minimum_case_age_years == 40.0


def test_every_lab_criterion_reads_a_known_measurement():
    for definition in (resolve_disease_definition(name) for name in available_disease_names()):
        for criterion in definition.lab_criteria:
            assert resolve_lab_criterion_measurement(criterion).canonical_name == criterion.measurement
            # Only the lab-criterion path bounds its rows, by the criterion's own range.
            parameters = build_all_of_us_lab_criterion_query_parameters(criterion)
            assert (parameters["plausible_low"][1], parameters["plausible_high"][1]) == criterion.plausible_range



def test_kidney_disease_cases_come_from_repeated_labs_or_staged_codes():
    kidney = resolve_disease_definition("chronic_kidney_disease")
    rows = [
        _disease_row(201, occurrence_count=0),  # eGFR < 60 twice, 120 days apart: case
        _disease_row(202, occurrence_count=0),  # eGFR < 60 twice, 30 days apart: neither
        _disease_row(203, occurrence_count=0),  # albuminuria twice, 200 days apart: case
        _disease_row(204, occurrence_count=2),  # two stage 3-5 diagnosis dates: case
        _disease_row(205, occurrence_count=0),  # nothing: control
        _disease_row(206, occurrence_count=0, control_exclusion=True),  # unstaged CKD code: neither
    ]
    egfr = {"201": _lab_row(3, "2019-01-01", "2019-05-01"), "202": _lab_row(2, "2019-01-01", "2019-01-31")}
    albuminuria = {"203": _lab_row(2, "2021-03-01", "2021-09-17")}

    training_rows, _columns, counts = _prepare_training_rows(kidney, rows, [egfr, albuminuria])

    by_person = {row["person_id"]: row for row in training_rows}
    assert {person: row["target"] for person, row in by_person.items()} == {"201": 1, "203": 1, "204": 1, "205": 0}
    # A lab case's onset is its first qualifying occasion (mid-year 1975 birthday).
    assert by_person["201"]["age_at_onset"] == (datetime.date(2019, 1, 1) - datetime.date(1975, 7, 1)).days / 365.25
    assert counts["n_cases_by_lab"] == 2 and counts["n_cases_by_diagnosis"] == 1
    assert counts["n_excluded_lab_evidence"] == 1 and counts["n_excluded_control_exclusion"] == 1


def test_case_support_medication_procedures_sex_and_age_rules():
    hypothyroidism = resolve_disease_definition("hypothyroidism")
    rows = [
        _disease_row(301, occurrence_count=1, case_medication_dates=2),  # one code + two levothyroxine dates
        _disease_row(302, occurrence_count=1, case_medication_dates=1),  # one levothyroxine date: neither
        _disease_row(303, occurrence_count=0, control_exclusion_medication=True),  # thyroid drug, no code
        _disease_row(304, occurrence_count=0),
    ]
    training_rows, _columns, counts = _prepare_training_rows(hypothyroidism, rows, [])
    assert [(row["person_id"], row["target"]) for row in training_rows] == [("301", 1), ("304", 0)]
    assert counts["n_excluded_one_date"] == 1 and counts["n_excluded_control_exclusion"] == 1

    cataract = resolve_disease_definition("cataract")
    procedure_only = [_disease_row(401, occurrence_count=0, case_procedure_dates=1), _disease_row(402, occurrence_count=0)]
    training_rows, _columns, counts = _prepare_training_rows(cataract, procedure_only, [])
    assert [(row["person_id"], row["target"], row["age_at_onset"]) for row in training_rows] == [
        ("401", 1, 50.0), ("402", 0, None)
    ]
    assert counts["n_cases_by_procedure"] == 1

    prostate = resolve_disease_definition("prostate_cancer")
    men_and_women = [
        _disease_row(501, occurrence_count=2, sex_at_birth_concept_id=45880669),
        _disease_row(502, occurrence_count=2, sex_at_birth_concept_id=45878463),
        # Controls must be men aged 50 or more: 45 + 503 % 30 = 58, 45 + 480 % 30 = 45.
        _disease_row(503, occurrence_count=0, sex_at_birth_concept_id=45880669),
        _disease_row(480, occurrence_count=0, sex_at_birth_concept_id=45880669),
    ]
    training_rows, _columns, counts = _prepare_training_rows(prostate, men_and_women, [])
    assert [(row["person_id"], row["target"]) for row in training_rows] == [("501", 1), ("503", 0)]
    assert counts["n_excluded_sex"] == 1 and counts["n_excluded_age"] == 1

    copd = resolve_disease_definition("copd")
    # First COPD diagnosis at 40 + 481 % 30 = 41 counts; one at 39 does not.
    early_onset = dict(_disease_row(482, occurrence_count=2), age_at_first_condition=39.0)
    training_rows, _columns, counts = _prepare_training_rows(copd, [_disease_row(481, occurrence_count=2), early_onset], [])
    assert [row["person_id"] for row in training_rows] == ["481"]
    assert counts["n_excluded_age"] == 1


def test_lab_evidence_must_cover_every_criterion():
    with pytest.raises(ValueError, match="one person table per lab criterion"):
        _prepare_training_rows(resolve_disease_definition("ckd"), [_disease_row(1, occurrence_count=0)], [])


def test_sparse_early_ehr_excludes_cases_and_controls_alike(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("GOOGLE_PROJECT", "billing-project")
    rows = [
        _disease_row(101, occurrence_count=3),
        _disease_row(102, occurrence_count=3, pre_landmark_condition_dates=4),
        _disease_row(103, occurrence_count=0),
        _disease_row(104, occurrence_count=0, pre_landmark_condition_dates=4),
    ]
    outputs = prepare_all_of_us_disease_sample_table(
        "gout",
        output_path=tmp_path / "gout.tsv",
        client=_FakeBigQueryClient(rows),
    )
    assert [row["person_id"] for row in _read_tsv_rows(outputs.sample_table_path)] == ["101", "103"]
    metadata = json.loads(outputs.metadata_path.read_text(encoding="utf-8"))
    assert metadata["n_excluded_sparse_ehr"] == 2
    assert metadata["min_pre_landmark_condition_dates"] == 5


def test_prepare_all_of_us_disease_requires_all_of_us_env(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("GOOGLE_PROJECT", raising=False)
    monkeypatch.delenv("WORKSPACE_CDR", raising=False)

    with pytest.raises(ValueError, match="GOOGLE_PROJECT"):
        prepare_all_of_us_disease_sample_table(
            "gout",
            output_path=tmp_path / "out.tsv",
        )


def test_prepare_all_of_us_disease_requires_workspace_cdr_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("GOOGLE_PROJECT", "billing-project")
    monkeypatch.delenv("WORKSPACE_CDR", raising=False)
    fake_client = _FakeBigQueryClient(
        rows=[_disease_row(101, occurrence_count=2), _disease_row(102, occurrence_count=0)]
    )

    with pytest.raises(ValueError, match="WORKSPACE_CDR"):
        prepare_all_of_us_disease_sample_table(
            "atrial fibrillation",
            output_path=tmp_path / "atrial_fibrillation.tsv",
            client=fake_client,
        )


def test_prepare_all_of_us_disease_uses_client_project_without_google_project_env(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("GOOGLE_PROJECT", raising=False)
    monkeypatch.setenv("WORKSPACE_CDR", "aou_workspace.cdr_dataset")
    fake_client = _FakeBigQueryClient(
        rows=[_disease_row(101, occurrence_count=2), _disease_row(102, occurrence_count=0)],
        project="client-project",
    )

    outputs = prepare_all_of_us_disease_sample_table(
        "atrial_fibrillation",
        output_path=tmp_path / "atrial_fibrillation.tsv",
        client=fake_client,
    )

    metadata_payload = json.loads(outputs.metadata_path.read_text(encoding="utf-8"))
    assert metadata_payload["billing_project"] == "client-project"


def test_prepare_all_of_us_disease_uses_workspace_cdr_from_env_in_query_and_metadata(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("GOOGLE_PROJECT", "billing-project")
    monkeypatch.setenv("WORKSPACE_CDR", "fc-aou-cdr-prod-ct.C2024Q3R9")
    fake_client = _FakeBigQueryClient(
        rows=[_disease_row(101, occurrence_count=2), _disease_row(102, occurrence_count=0)]
    )

    outputs = prepare_all_of_us_disease_sample_table(
        "atrial_fibrillation",
        output_path=tmp_path / "atrial_fibrillation.tsv",
        client=fake_client,
    )

    assert fake_client.sql is not None
    assert "fc-aou-cdr-prod-ct.C2024Q3R9" in fake_client.sql
    metadata_payload = json.loads(outputs.metadata_path.read_text(encoding="utf-8"))
    assert metadata_payload["cdr_dataset"] == "fc-aou-cdr-prod-ct.C2024Q3R9"


@pytest.mark.parametrize("disease", ["copd", "atrial fibrillation"])
def test_cli_prepare_all_of_us_disease_forwards_the_name_and_output(monkeypatch, tmp_path: Path, disease):
    calls: dict[str, object] = {}

    def fake_prepare(disease, output_path, **kwargs):
        calls.update(disease=disease, output_path=output_path)
        return type("Prepared", (), {"sample_table_path": output_path, "sql_path": output_path, "metadata_path": output_path})()

    monkeypatch.setattr("sv_pgs.cli.prepare_all_of_us_disease_sample_table", fake_prepare)
    output_path = tmp_path / "prepared.tsv"
    assert main(["prepare-all-of-us-disease", "--disease", disease, "--output", str(output_path)]) == 0
    assert calls == {"disease": disease, "output_path": output_path}


def test_cli_lists_available_all_of_us_diseases(capsys):
    exit_code = main(["list-all-of-us-diseases"])
    assert exit_code == 0
    printed = capsys.readouterr().out.strip().splitlines()
    assert "atrial_fibrillation" in printed
    assert "type2_diabetes" in printed
    assert printed == sorted(available_disease_names())


# ---------------------------------------------------------------------------
# Quantitative traits: catalogue
# ---------------------------------------------------------------------------


def test_every_disease_and_trait_name_resolves_to_exactly_its_own_definition():
    definitions: tuple[DiseaseDefinition | MeasurementDefinition, ...] = (
        *DISEASE_DEFINITIONS,
        *MEASUREMENT_DEFINITIONS,
    )
    names_per_definition = [
        {name.strip().lower().replace("-", "_").replace(" ", "_") for name in (definition.canonical_name, *definition.aliases)}
        for definition in definitions
    ]
    all_names = [name for names in names_per_definition for name in names]
    assert len(set(all_names)) == len(all_names)
    for definition in definitions:
        for name in (definition.canonical_name, *definition.aliases):
            assert resolve_all_of_us_phenotype(name) is definition


def test_catalogue_is_the_panels_eleven_quantitative_traits():
    assert available_measurement_names() == sorted([
        "height", "body_mass_index", "systolic_blood_pressure", "heart_rate", "mean_corpuscular_volume",
        "platelet_count", "white_blood_cell_count", "total_bilirubin", "egfr_ckd_epi_2021",
        "ldl_cholesterol", "hemoglobin_a1c",
    ])


def test_medication_rules_follow_the_published_conventions():
    treatments = {
        definition.canonical_name: definition.treatment
        for definition in MEASUREMENT_DEFINITIONS
        if definition.treatment is not None
    }
    assert treatments["ldl_cholesterol"] == TreatmentRule(
        LDL_LOWERING, TreatmentCorrection.DIVIDE, 0.7, treatments["ldl_cholesterol"].citation
    )
    assert treatments["systolic_blood_pressure"].correction is TreatmentCorrection.ADD
    assert treatments["systolic_blood_pressure"].amount == 15.0
    for trait in ("heart_rate", "body_mass_index"):
        assert treatments[trait].correction is TreatmentCorrection.EXCLUDE
    # HbA1c is restricted to people without diabetes.
    hba1c = resolve_measurement_definition("hba1c")
    assert hba1c.treatment is None
    assert hba1c.clinical_windows == (DIABETES,)
    assert DIABETES.days_before == DIABETES.days_after == UNBOUNDED_WINDOW_DAYS


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"canonical_unit": "millimole per liter"}, "canonical unit"),
        (
            {"unit_conversions": (UnitConversion("milligram per deciliter", 1.0), UnitConversion("milligram per deciliter", 2.0))},
            "unique and lower case",
        ),
        ({"unit_conversions": (UnitConversion("milligram per deciliter", 1.0), UnitConversion("mg/dL", 1.0))}, "lower case"),
        ({"value_formula": "mdrd"}, "unknown value formula"),
        ({"loinc_codes": ("1-1", "1-1")}, "LOINC codes"),
    ],
)
def test_measurement_definition_rejects_inconsistent_definitions(overrides, message):
    with pytest.raises(ValueError, match=message):
        _definition(**overrides)


def test_treatment_rules_and_windows_reject_inconsistent_arguments():
    with pytest.raises(ValueError, match="amount"):
        TreatmentRule(LDL_LOWERING, TreatmentCorrection.EXCLUDE, 0.7, "test")
    with pytest.raises(ValueError, match="amount"):
        TreatmentRule(LDL_LOWERING, TreatmentCorrection.DIVIDE, None, "test")
    with pytest.raises(ValueError, match="domains"):
        ClinicalWindow("test", (WindowConcept("visit", "SNOMED", "1"),), 0, 0)
    with pytest.raises(ValueError, match="needs concepts"):
        ClinicalWindow("test", (), 0, 0)
    with pytest.raises(ValueError, match="window days"):
        ClinicalWindow("test", (WindowConcept("drug", "ATC", "A10"),), -2, 0)


def test_resolve_measurement_definition_accepts_aliases_and_rejects_diseases():
    assert resolve_measurement_definition("MCV").canonical_name == "mean_corpuscular_volume"
    assert resolve_measurement_definition("creatinine").canonical_name == "egfr_ckd_epi_2021"
    with pytest.raises(ValueError, match="Unsupported trait"):
        resolve_measurement_definition("gout")


# ---------------------------------------------------------------------------
# SQL and parameters
# ---------------------------------------------------------------------------


def _table_aliases(sql: str) -> dict[str, str]:
    """alias -> OMOP table for every `dataset.table` [AS alias] in the query."""
    aliases: dict[str, str] = {}
    for table, alias in re.findall(r"`" + re.escape(DATASET) + r"\.(\w+)`(?:\s+AS\s+(\w+))?", sql):
        aliases[alias or table] = table
    return aliases


def test_measurement_sql_references_only_omop_cdm_v54_columns(monkeypatch):
    monkeypatch.setenv("WORKSPACE_CDR", DATASET)
    sql = build_all_of_us_measurement_sql()
    aliases = _table_aliases(sql)
    assert set(aliases.values()) == set(OMOP_CDM_V54_COLUMNS)
    references = re.findall(r"\b(\w+)\.(\w+)\b", sql)
    checked = 0
    for alias, column in references:
        if alias in aliases:
            assert column in OMOP_CDM_V54_COLUMNS[aliases[alias]], f"{alias}.{column}"
            checked += 1
    assert checked > 30
    # Unqualified base-table columns (inside single-table CTEs).
    for table, column in (
        ("condition_occurrence", "condition_start_date"),
        ("condition_occurrence", "condition_concept_id"),
        ("observation", "observation_date"),
        ("observation", "observation_concept_id"),
        ("procedure_occurrence", "procedure_date"),
        ("procedure_occurrence", "procedure_concept_id"),
        ("drug_exposure", "drug_exposure_start_date"),
        ("drug_exposure", "drug_concept_id"),
        ("visit_occurrence", "visit_start_date"),
        ("visit_occurrence", "visit_end_date"),
        ("visit_occurrence", "visit_concept_id"),
    ):
        assert re.search(rf"\b{column}\b", sql)
        assert column in OMOP_CDM_V54_COLUMNS[table]


def test_measurement_sql_is_one_parameterized_query_for_every_trait(monkeypatch):
    monkeypatch.setenv("WORKSPACE_CDR", DATASET)
    sql = build_all_of_us_measurement_sql()
    parameter_names = set(re.findall(r"@(\w+)", sql))
    for definition in MEASUREMENT_DEFINITIONS:
        parameters = build_all_of_us_measurement_query_parameters(definition)
        assert set(parameters) == parameter_names
        # Codes enter only as parameters, never as SQL text.
        for loinc_code in definition.loinc_codes:
            assert f"'{loinc_code}'" not in sql
    for reason in MEASUREMENT_EXCLUSION_REASONS:
        assert f"'{reason}'" in sql
        assert f"{reason}_row_count" in sql
    assert "vocabulary_id = 'LOINC'" in sql
    # Both concept columns are matched; units are matched by All of Us unit name.
    assert "measurement.measurement_source_concept_id IN (SELECT concept_id FROM analyte_concepts)" in sql
    assert "LOWER(unit.concept_name)" in sql
    assert "GROUP BY person_id, measurement_date, age_years, on_treatment" in sql
    assert "race_concept_id" not in sql
    assert "ethnicity_concept_id" not in sql


def test_measurement_sql_scans_the_measurement_table_once(monkeypatch):
    # BigQuery re-executes a non-recursive CTE at every reference, so each CTE
    # downstream of the measurement scan must be referenced exactly once.
    monkeypatch.setenv("WORKSPACE_CDR", DATASET)
    sql = build_all_of_us_measurement_sql()
    assert sql.count(f"`{DATASET}.measurement`") == 1
    for common_table in (
        "exclusion_windows", "measurement_rows", "analyte_rows", "classified_rows", "valued_rows",
        "person_days",
    ):
        assert f"{common_table} AS (" in sql
        assert len(re.findall(rf"\b(?:FROM|JOIN) {common_table}\b", sql)) == 1, common_table


def test_measurement_sql_uses_the_race_free_ckd_epi_2021_constants(monkeypatch):
    monkeypatch.setenv("WORKSPACE_CDR", DATASET)
    sql = build_all_of_us_measurement_sql()
    assert (
        "WHEN 'female' THEN 142 * POW(LEAST(canonical_value / 0.7, 1), -0.241)"
        " * POW(GREATEST(canonical_value / 0.7, 1), -1.2) * POW(0.9938, age_years) * 1.012"
    ) in sql
    assert (
        "WHEN 'male' THEN 142 * POW(LEAST(canonical_value / 0.9, 1), -0.302)"
        " * POW(GREATEST(canonical_value / 0.9, 1), -1.2) * POW(0.9938, age_years) * 1.0"
    ) in sql


def test_measurement_query_config_types_every_parameter():
    definition = resolve_measurement_definition("systolic_blood_pressure")
    config = build_all_of_us_measurement_query_config(definition)
    parameters = {parameter.name: parameter for parameter in config.query_parameters}
    assert isinstance(parameters["loinc_codes"], bigquery.ArrayQueryParameter)
    assert parameters["loinc_codes"].values == ["8480-6", "8459-0", "76534-7"]
    assert parameters["physical_measurement_concept_ids"].array_type == "INT64"
    assert parameters["physical_measurement_concept_ids"].values == [903118]
    assert parameters["excluded_source_concept_ids"].values == [903109, 903114, 903130]
    assert parameters["unit_labels"].values == ["millimeter mercury column", "no unit", "unit"]
    assert parameters["unit_scales"].array_type == "FLOAT64"
    assert parameters["treatment_atc_codes"].values == ["C02", "C03", "C07", "C08", "C09"]
    # A trait keeps every positive reading: its range parameters are typed NULLs.
    assert isinstance(parameters["plausible_low"], bigquery.ScalarQueryParameter)
    assert parameters["plausible_low"].type_ == "FLOAT64"
    assert parameters["plausible_low"].value is None and parameters["plausible_high"].value is None
    assert parameters["minimum_age_years"].value == 18
    assert parameters["window_domains"].values == []
    height = {
        parameter.name: parameter
        for parameter in build_all_of_us_measurement_query_config(resolve_measurement_definition("height")).query_parameters
    }
    assert height["treatment_atc_codes"].values == []
    assert height["minimum_age_years"].value == 20.0


def test_clinical_windows_flatten_to_one_parameter_entry_per_root_concept():
    parameters = build_all_of_us_measurement_query_parameters(resolve_measurement_definition("mcv"))
    roots = list(zip(
        parameters["window_domains"][1],
        parameters["window_vocabularies"][1],
        parameters["window_concept_codes"][1],
        parameters["window_days_before"][1],
        parameters["window_days_after"][1],
        strict=True,
    ))
    assert parameters["window_days_before"][0] == "INT64"
    assert roots == [
        ("condition", "SNOMED", "93143009", 180, UNBOUNDED_WINDOW_DAYS),
        ("condition", "SNOMED", "118600007", 180, UNBOUNDED_WINDOW_DAYS),
        ("condition", "SNOMED", "109989006", 180, UNBOUNDED_WINDOW_DAYS),
        ("condition", "SNOMED", "109995007", 180, UNBOUNDED_WINDOW_DAYS),
        ("condition", "SNOMED", "425333006", 180, UNBOUNDED_WINDOW_DAYS),
        ("drug", "ATC", "L01", 0, 90),
        ("procedure", "SNOMED", "116762002", 0, 120),
        ("procedure", "CPT4", "36430", 0, 120),
    ]


def test_hba1c_ifcc_units_convert_by_the_ngsp_master_equation():
    conversion = {
        unit.unit_label: unit for unit in resolve_measurement_definition("hemoglobin_a1c").unit_conversions
    }["millimole per mole"]
    # IFCC = 10.93 * NGSP - 23.50, so 53 mmol/mol is NGSP 7.0%. The two published
    # equations are rounded forms of one line, so they agree to half a unit in the
    # last digit of each of their four coefficients.
    ifcc = 53.0
    coefficient_rounding = 0.000005 * ifcc + 0.0005 + 0.005 / 10.93 + 0.005 * (ifcc + 23.50) / 10.93**2
    assert abs(conversion.scale * ifcc + conversion.offset - (ifcc + 23.50) / 10.93) <= coefficient_rounding


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


def test_untreated_occasions_take_precedence_and_treated_ones_are_corrected(monkeypatch):
    ldl = resolve_measurement_definition("ldl_cholesterol")
    rows, _levels = _synthetic_person_rows(30, level_variance=400.0, noise_variance=100.0, seed=4)
    rows += _person_days(1001, [(50.0, 150.0), (50.5, 140.0)], treated_occasions=[(55.0, 90.0), (56.0, 85.0), (57.0, 95.0)])
    rows += _person_days(1002, treated_occasions=[(60.0, 70.0), (61.0, 72.0)])
    rows += _person_days(1003, excluded={"nonpositive": 2})
    occasions, training_rows, summary = _captured_occasions(monkeypatch, ldl, rows)
    by_person = {row["person_id"]: row for row in training_rows}
    assert "1003" not in by_person
    assert len(by_person) == 32
    assert by_person["1001"]["measurement_source"] == "untreated"
    assert by_person["1001"]["occasion_count"] == 2
    assert by_person["1001"]["age_at_measurement"] == 50.25
    assert by_person["1002"]["measurement_source"] == "treated_corrected"
    assert summary["n_persons_by_source"] == {"untreated": 31, "treated_corrected": 1}
    assert summary["n_persons_without_retained_occasion"] == 1
    assert summary["excluded_row_counts"]["nonpositive"] == 2
    assert occasions.values.shape[0] == sum(row["occasion_count"] for row in training_rows)


@pytest.mark.parametrize(
    ("trait", "treated", "expected"),
    [
        # LDL / 0.7 and SBP + 15 mmHg, occasion by occasion on the linear scale.
        ("ldl_cholesterol", [70.0, 63.0], [100.0, 90.0]),
        ("systolic_blood_pressure", [130.0, 125.0], [145.0, 140.0]),
    ],
)
def test_treatment_corrections_apply_to_every_treated_occasion(monkeypatch, trait, treated, expected):
    rows = _person_days(1, treated_occasions=[(60.0, value) for value in treated])
    rows += _person_days(2, [(40.0, 100.0), (41.0, 110.0)], sex_at_birth_concept_id=45880669)
    occasions, _training_rows, _summary = _captured_occasions(monkeypatch, resolve_measurement_definition(trait), rows)
    corrected = occasions.values[occasions.person_index == 0]
    # One division or addition each.
    for value, target in zip(corrected, expected, strict=True):
        assert value == within_rounding(target, 2)


def test_treated_only_persons_are_dropped_when_no_correction_exists(monkeypatch):
    heart_rate = resolve_measurement_definition("heart_rate")
    rows, _levels = _synthetic_person_rows(40, level_variance=50.0, noise_variance=20.0, seed=3)
    rows += _person_days(1000, treated_occasions=[(62.0, 62.0), (63.0, 66.0)])
    _occasions, training_rows, summary = _captured_occasions(monkeypatch, heart_rate, rows)
    assert "1000" not in {row["person_id"] for row in training_rows}
    assert summary["n_persons_treated_only_excluded"] == 1
    assert summary["n_persons"] == 40


def test_treated_occasions_for_a_trait_without_a_medication_rule_raise():
    height = resolve_measurement_definition("height")
    with pytest.raises(ValueError, match="without a medication rule"):
        build_all_of_us_measurement_targets(height, _person_days(1, treated_occasions=[(50.0, 170.0)]), 1 << 28)


def test_occasion_design_models_age_by_sex_and_drops_columns_that_add_no_rank():
    ages = np.array([40.0, 50.0, 60.0, 70.0, 45.0, 55.0])
    female = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    levels = ["45880669" if value == 0.0 else "45878463" for value in female]
    design = _occasion_design(ages, female, levels)
    centered = ages - ages.mean()
    np.testing.assert_array_equal(design[:, -1], centered * female)
    assert design.shape == (6, 5)
    # One sex only: no sex indicator and no interaction column.
    assert _occasion_design(ages, np.zeros(6), ["45880669"] * 6).shape == (6, 3)
    # Four occasions cannot carry five columns: the last, the interaction, adds no rank.
    assert _occasion_design(ages[:4], female[:4], levels[:4]).shape == (4, 4)


def test_the_model_target_tracks_the_true_level_better_than_the_raw_mean():
    # The full fit, the Box-Cox exponent included.
    rows, levels = _synthetic_person_rows(150, level_variance=1.0, noise_variance=2.0, seed=7)
    definition = resolve_measurement_definition("mean_corpuscular_volume")
    training_rows, _columns, summary = build_all_of_us_measurement_targets(definition, rows, 1 << 28)
    people = {person.person_id: person for person in person_occasions(rows)}
    # At the fitted transform, both are compared with the true level on that scale's own standardization.
    targets = np.array([row["target"] for row in training_rows])
    raw_means = np.array([people[row["person_id"]].values.mean() for row in training_rows])
    true_levels = levels[[int(row["person_id"]) for row in training_rows]]
    assert np.corrcoef(targets, true_levels)[0, 1] > np.corrcoef(raw_means, true_levels)[0, 1]
    assert 0.0 < summary["repeatability"] < 1.0


def test_no_statistic_of_the_occasions_is_a_trait_covariate(monkeypatch):
    # How often a trait is measured depends on its level, so adjusting the
    # genetic fit for the occasion count attenuates every effect
    # (novel-pheno Theorem 5); precision enters through target_reliability.
    assert not any("occasion" in column for column in measurement_covariate_columns())
    rows, _levels = _synthetic_person_rows(50, level_variance=1.0, noise_variance=1.0, seed=6)
    _occasions, training_rows, _summary = _captured_occasions(monkeypatch, resolve_measurement_definition("mcv"), rows)
    assert "log_occasion_count" not in training_rows[0]


def test_training_rows_carry_targets_covariates_and_one_hot_sex(monkeypatch):
    rows, _levels = _synthetic_person_rows(200, level_variance=1.0, noise_variance=1.0, seed=1)
    rows += _person_days(900, [(50.0, 90.0)], unrecognized_unit_labels=["millimole per liter"], excluded={"unrecognized_unit": 2})
    rows += _person_days(901, [(51.0, 91.0)], unrecognized_unit_labels=["millimole per liter", "unit_concept_id=9999"], excluded={"unrecognized_unit": 2})
    occasions, training_rows, summary = _captured_occasions(monkeypatch, resolve_measurement_definition("mcv"), rows)
    assert _columns_of(training_rows) >= {"sex_at_birth_concept_id_45878463", "sex_at_birth_concept_id_45880669"}
    people = {person.person_id: person for person in person_occasions(rows)}
    female_row, male_row = training_rows[0], training_rows[1]
    ages = people[female_row["person_id"]].ages
    assert female_row["age_at_measurement"] == ages.mean()
    assert female_row["age_at_measurement_squared"] == np.mean(np.square(ages))
    assert female_row["age_at_measurement_x_female"] == ages.mean()
    assert male_row["age_at_measurement_x_female"] == 0.0
    assert female_row["occasion_count"] == ages.shape[0]
    assert female_row["sex_at_birth_concept_id_45878463"] == 1
    assert "sex_at_birth_concept_id" not in female_row
    assert female_row["target_reliability"] == 0.5
    assert "target_inverse_normal" not in female_row
    # Persons, not days, per unrecognized unit label.
    assert summary["unrecognized_unit_person_counts"] == {"millimole per liter": 2, "unit_concept_id=9999": 1}
    assert occasions.design.shape == (occasions.values.shape[0], 5)


def _columns_of(rows: list[dict[str, object]]) -> set[str]:
    return set().union(*(row.keys() for row in rows))


# ---------------------------------------------------------------------------
# Sample tables, census and CLI
# ---------------------------------------------------------------------------


def _identity_transform_fit(monkeypatch) -> None:
    """The occasion model at the identity transform in a 256 MiB budget, for tests of the plumbing around it (the
    exponent's search is tested in test_the_model_target_tracks_the_true_level_better_than_the_raw_mean)."""
    monkeypatch.setattr(
        "sv_pgs.all_of_us.fit_occasion_model", lambda occasions, working_bytes: fit_at_exponent(occasions, 1.0, min(working_bytes, 1 << 28))
    )


def test_prepare_measurement_sample_table_writes_table_sql_and_metadata(tmp_path: Path, monkeypatch):
    _identity_transform_fit(monkeypatch)
    monkeypatch.setenv("GOOGLE_PROJECT", "billing-project")
    monkeypatch.setenv("WORKSPACE_CDR", DATASET)
    rows, _levels = _synthetic_person_rows(120, level_variance=1.0, noise_variance=1.0, seed=2)
    fake_client = _FakeBigQueryClient(rows)
    definition = resolve_measurement_definition("total_bilirubin")
    outputs = prepare_all_of_us_measurement_sample_table(
        "total_bilirubin", tmp_path / "total_bilirubin.samples.tsv", client=fake_client
    )
    assert fake_client.sql == build_all_of_us_measurement_sql()
    parameters = {parameter.name: parameter for parameter in fake_client.job_config.query_parameters}
    assert parameters["loinc_codes"].values == ["1975-2"]
    with outputs.sample_table_path.open(encoding="utf-8") as handle:
        table = list(csv.DictReader(handle, delimiter="\t"))
    assert len(table) == 120
    assert list(table[0])[:9] == [
        "sample_id", "person_id", "target", "occasion_count", "target_reliability", "measurement_source",
        "age_at_measurement", "age_at_measurement_squared", "age_at_measurement_x_female",
    ]
    assert outputs.sql_path.read_text(encoding="utf-8").strip() == build_all_of_us_measurement_sql()
    metadata = json.loads(outputs.metadata_path.read_text(encoding="utf-8"))
    assert metadata["trait"] == "total_bilirubin"
    assert metadata["phenotype_fingerprint"] == phenotype_fingerprint(definition)
    assert metadata["covariate_columns"] == list(measurement_covariate_columns())
    assert metadata["analysis_scale"].startswith("Box-Cox exponent ")
    assert metadata["treatment"] is None
    assert [window["name"] for window in metadata["clinical_windows"]] == ["acute_hepatobiliary", "cirrhosis"]
    assert metadata["query_parameters"]["loinc_codes"] == ["1975-2"]
    assert metadata["cdr_dataset"] == DATASET
    assert metadata["n_persons"] == 120
    assert 0.0 < metadata["repeatability"] < 1.0


def test_prepare_measurement_sample_table_requires_workspace_cdr(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GOOGLE_PROJECT", "billing-project")
    monkeypatch.delenv("WORKSPACE_CDR", raising=False)
    with pytest.raises(ValueError, match="WORKSPACE_CDR"):
        prepare_all_of_us_measurement_sample_table(
            "total_bilirubin", tmp_path / "out.tsv", client=_FakeBigQueryClient([])
        )


def test_phenotype_fingerprint_changes_with_the_definition_and_the_cdr(monkeypatch):
    monkeypatch.setenv("WORKSPACE_CDR", DATASET)
    height = resolve_measurement_definition("height")
    fingerprint = phenotype_fingerprint(height)
    assert fingerprint == phenotype_fingerprint(height)
    assert fingerprint != phenotype_fingerprint(dataclasses.replace(height, minimum_age_years=21.0))
    assert fingerprint != phenotype_fingerprint(resolve_all_of_us_phenotype("gout"))
    monkeypatch.setenv("WORKSPACE_CDR", "fc-aou-cdr-prod-ct.C2025Q4R5")
    assert fingerprint != phenotype_fingerprint(height)



def test_cli_lists_traits_and_prepares_a_trait_table(monkeypatch, tmp_path: Path, capsys):
    assert main(["list-all-of-us-traits"]) == 0
    assert capsys.readouterr().out.strip().splitlines() == available_measurement_names()

    calls: dict[str, object] = {}

    def fake_prepare(trait, output_path, **kwargs):
        calls["trait"] = trait
        calls["output_path"] = output_path
        return type(
            "Prepared",
            (),
            {"sample_table_path": output_path, "sql_path": output_path, "metadata_path": output_path},
        )()

    monkeypatch.setattr("sv_pgs.cli.prepare_all_of_us_measurement_sample_table", fake_prepare)
    assert main(["prepare-all-of-us-trait", "--trait", "mcv", "--output", str(tmp_path / "mcv.tsv")]) == 0
    assert calls == {"trait": "mcv", "output_path": tmp_path / "mcv.tsv"}



def test_census_suppresses_cells_of_twenty_or_fewer_participants(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("GOOGLE_PROJECT", "billing-project")
    monkeypatch.setenv("WORKSPACE_CDR", DATASET)
    fake_client = _FakeBigQueryClient([
        {"trait_name": "height", "vocabulary_id": "LOINC", "concept_code": "8302-2",
         "matched_on_standard_concept": True, "unit_label": "inch (us)", "participant_count": 246040, "row_count": 900000},
        {"trait_name": "height", "vocabulary_id": "LOINC", "concept_code": "8302-2",
         "matched_on_standard_concept": True, "unit_label": "kilogram", "participant_count": 20, "row_count": 25},
        {"trait_name": "height", "vocabulary_id": "LOINC", "concept_code": "8302-2",
         "matched_on_standard_concept": True, "unit_label": "meter", "participant_count": 21, "row_count": 30},
    ])
    census_path = prepare_all_of_us_measurement_census(tmp_path / "census.tsv", client=fake_client)
    with census_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert [(row["unit_label"], row["participant_count"], row["row_count"], row["suppressed_below_minimum"]) for row in rows] == [
        ("inch (us)", "246040", "900000", "False"),
        ("kilogram", "", "", "True"),
        ("meter", "21", "30", "False"),
    ]
    parameters = {parameter.name: parameter for parameter in fake_client.job_config.query_parameters}
    assert parameters["census_loinc_traits"].values.count("height") == 2
    assert parameters["census_physical_measurement_concept_ids"].values == [903133, 903118, 903126]


def test_cli_census_all_of_us_traits_writes_the_census(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.setattr(
        "sv_pgs.cli.prepare_all_of_us_measurement_census", lambda output_path: Path(output_path)
    )
    assert main(["census-all-of-us-traits", "--output", str(tmp_path / "census.tsv")]) == 0
    assert capsys.readouterr().out.strip() == f"census\t{tmp_path / 'census.tsv'}"

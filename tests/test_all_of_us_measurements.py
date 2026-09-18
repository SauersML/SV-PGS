"""All of Us quantitative traits from the OMOP measurement table.

Covers the trait catalogue, the BigQuery SQL against the OMOP CDM v5.4 schema,
the query parameters, the per-person target construction (treatment
precedence and corrections, variance components, empirical BLUP against the
dense Henderson mixed-model solution, inverse normal transform), the sample
table writer and the runner/CLI wiring.
"""
from __future__ import annotations

import csv
import dataclasses
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest
from google.cloud import bigquery
from scipy.special import ndtri

import sv_pgs.aou_runner as aou_runner
from sv_pgs.all_of_us import (
    DISEASE_DEFINITIONS,
    MEASUREMENT_DEFINITIONS,
    MEASUREMENT_EXCLUSION_REASONS,
    DiseaseDefinition,
    MeasurementDefinition,
    TreatmentCorrection,
    TreatmentRule,
    UnitConversion,
    LIPID_LOWERING,
    _estimate_person_variance_components,
    _person_blup,
    _rank_inverse_normal,
    available_measurement_names,
    build_all_of_us_measurement_query_config,
    build_all_of_us_measurement_query_parameters,
    build_all_of_us_measurement_sql,
    build_all_of_us_measurement_targets,
    prepare_all_of_us_measurement_sample_table,
    resolve_all_of_us_phenotype,
    resolve_measurement_definition,
)
from sv_pgs.cli import main

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
    "concept_ancestor": {
        "ancestor_concept_id", "descendant_concept_id", "min_levels_of_separation",
        "max_levels_of_separation",
    },
}
DATASET = "aou_workspace.cdr_dataset"


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


def _person_row(
    person_id: int,
    *,
    untreated: tuple[int, float, float, float] | None = None,
    treated: tuple[int, float, float, float] | None = None,
    sex_at_birth_concept_id: int | None = 45878463,
    excluded: dict[str, int] | None = None,
    unrecognized_unit_labels: list[str] | None = None,
) -> dict[str, object]:
    """One row shaped like the measurement query output.

    untreated/treated = (occasion count, mean, population variance, mean age);
    the mean squared age is mean_age^2 + 4 (a fixed within-person age spread).
    """
    excluded_counts = {reason: 0 for reason in MEASUREMENT_EXCLUSION_REASONS} | (excluded or {})
    row: dict[str, object] = {
        "sample_id": str(person_id),
        "person_id": str(person_id),
        "unrecognized_unit_labels": unrecognized_unit_labels or [],
        "sex_at_birth_concept_id": sex_at_birth_concept_id,
        "race_concept_id": 8527,
        "ethnicity_concept_id": 38003564,
    }
    for reason, count in excluded_counts.items():
        row[f"{reason}_row_count"] = count
    occasion_rows = 0
    for group, statistics in (("untreated", untreated), ("treated", treated)):
        if statistics is None:
            row[f"{group}_occasion_count"] = 0
            for statistic in ("mean", "variance", "mean_age", "mean_age_squared"):
                row[f"{group}_{statistic}"] = None
            continue
        count, mean, variance, mean_age = statistics
        occasion_rows += count
        row[f"{group}_occasion_count"] = count
        row[f"{group}_mean"] = mean
        row[f"{group}_variance"] = variance
        row[f"{group}_mean_age"] = mean_age
        row[f"{group}_mean_age_squared"] = mean_age**2 + 4.0
    row["measurement_row_count"] = occasion_rows + sum(excluded_counts.values())
    return row


def _synthetic_person_rows(
    person_count: int,
    *,
    between_variance: float,
    within_variance: float,
    seed: int,
) -> tuple[list[dict[str, object]], np.ndarray]:
    """Per-person rows simulated from the random-intercept model; returns the
    rows and each person's true long-run mean."""
    generator = np.random.default_rng(seed)
    rows = []
    true_means = np.empty(person_count)
    for person_index in range(person_count):
        count = int(generator.integers(1, 7))
        mean_age = float(generator.uniform(25.0, 80.0))
        female = person_index % 2 == 0
        long_run_mean = (
            10.0 + 0.03 * (mean_age - 50.0) + (0.5 if female else 0.0)
            + generator.normal(0.0, math.sqrt(between_variance))
        )
        occasions = long_run_mean + generator.normal(0.0, math.sqrt(within_variance), size=count)
        true_means[person_index] = long_run_mean
        rows.append(
            _person_row(
                person_index,
                untreated=(count, float(occasions.mean()), float(occasions.var()), mean_age),
                sex_at_birth_concept_id=45878463 if female else 45880669,
            )
        )
    return rows, true_means


_BASE_DEFINITION = MeasurementDefinition(
    canonical_name="test_trait",
    aliases=(),
    description="test",
    loinc_codes=("1-1",),
    canonical_unit="mg/dL",
    unit_conversions=(UnitConversion("mg/dL", 1.0),),
    plausible_range=(1.0, 100.0),
    log_scale=False,
    rationale="test",
)


def _definition(**overrides: object) -> MeasurementDefinition:
    return dataclasses.replace(_BASE_DEFINITION, **overrides)


# ---------------------------------------------------------------------------
# Catalogue
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


def test_catalogue_covers_the_named_sv_traits_and_the_requested_panels():
    names = set(available_measurement_names())
    # Headline SV/TR traits: alpha-globin red-cell indices, HP deletion ->
    # haptoglobin, ACE Alu -> serum ACE, UGT1A1 (TA)n -> bilirubin, GGT1 VNTR,
    # MUC1 VNTR -> urea/urate, LPA KIV-2.
    assert {
        "mean_corpuscular_volume", "mean_corpuscular_hemoglobin", "haptoglobin",
        "angiotensin_converting_enzyme", "total_bilirubin", "gamma_glutamyl_transferase",
        "blood_urea_nitrogen", "urate", "lipoprotein_a_mass", "lipoprotein_a_molar",
    } <= names
    assert {
        "ldl_cholesterol", "hdl_cholesterol", "total_cholesterol", "triglycerides", "hemoglobin_a1c",
        "glucose", "egfr_ckd_epi_2021", "platelet_count", "white_blood_cell_count",
        "neutrophil_count", "lymphocyte_count", "monocyte_count", "eosinophil_count",
        "red_cell_distribution_width", "alanine_aminotransferase", "aspartate_aminotransferase",
        "alkaline_phosphatase", "albumin", "thyrotropin", "c_reactive_protein", "ferritin",
        "vitamin_d_25_hydroxy", "height", "body_mass_index", "systolic_blood_pressure",
        "diastolic_blood_pressure", "waist_circumference", "hip_circumference", "heart_rate",
    } <= names
    assert len(MEASUREMENT_DEFINITIONS) == len(names)


def test_medication_conventions_follow_the_published_constants():
    corrections = {
        definition.canonical_name: definition.treatment
        for definition in MEASUREMENT_DEFINITIONS
        if definition.treatment is not None
    }
    assert corrections["ldl_cholesterol"].correction is TreatmentCorrection.DIVIDE
    assert corrections["ldl_cholesterol"].amount == 0.7
    assert corrections["total_cholesterol"].amount == 0.8
    assert corrections["systolic_blood_pressure"].correction is TreatmentCorrection.ADD
    assert corrections["systolic_blood_pressure"].amount == 15.0
    assert corrections["diastolic_blood_pressure"].amount == 10.0
    for trait in ("hemoglobin_a1c", "glucose", "thyrotropin", "urate", "body_mass_index", "heart_rate"):
        assert corrections[trait].correction is TreatmentCorrection.EXCLUDE
    # By convention HDL and triglycerides are not corrected for therapy.
    assert "hdl_cholesterol" not in corrections
    assert "triglycerides" not in corrections


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"canonical_unit": "mmol/L"}, "canonical unit"),
        ({"unit_conversions": (UnitConversion("mg/dL", 1.0), UnitConversion("mg/dL", 2.0))}, "duplicate unit"),
        ({"plausible_range": (5.0, 5.0)}, "empty plausible range"),
        ({"log_scale": True, "plausible_range": (0.0, 5.0)}, "log scale needs"),
        ({"log_offset": 0.05}, "log_offset without log scale"),
        ({"value_formula": "mdrd"}, "unknown value formula"),
        (
            {
                "log_scale": True,
                "treatment": TreatmentRule(LIPID_LOWERING, TreatmentCorrection.ADD, 1.0, "test"),
            },
            "additive correction",
        ),
        (
            {
                "log_scale": True,
                "log_offset": 0.05,
                "treatment": TreatmentRule(LIPID_LOWERING, TreatmentCorrection.DIVIDE, 0.7, "test"),
            },
            "ratio correction",
        ),
        ({"loinc_codes": ("1-1", "1-1")}, "LOINC codes"),
    ],
)
def test_measurement_definition_rejects_inconsistent_definitions(overrides, message):
    with pytest.raises(ValueError, match=message):
        _definition(**overrides)


def test_treatment_rule_needs_an_amount_exactly_when_it_corrects():
    with pytest.raises(ValueError, match="amount"):
        TreatmentRule(LIPID_LOWERING, TreatmentCorrection.EXCLUDE, 0.7, "test")
    with pytest.raises(ValueError, match="amount"):
        TreatmentRule(LIPID_LOWERING, TreatmentCorrection.DIVIDE, None, "test")


def test_resolve_measurement_definition_accepts_aliases_and_rejects_diseases():
    assert resolve_measurement_definition("MCV").canonical_name == "mean_corpuscular_volume"
    assert resolve_measurement_definition("hs-crp").canonical_name == "c_reactive_protein"
    with pytest.raises(ValueError, match="Unsupported trait"):
        resolve_measurement_definition("hypertension")


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
        ("drug_exposure", "drug_exposure_start_date"),
        ("drug_exposure", "drug_concept_id"),
        ("concept", "standard_concept"),
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
    assert "vocabulary_id = 'UCUM'" in sql
    assert "GROUP BY person_id, measurement_date, age_years, on_treatment" in sql


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
    assert "race_concept_id" not in sql.split("valued_rows AS", 1)[1].split("person_days AS", 1)[0]


def test_measurement_sql_scans_the_measurement_table_once(monkeypatch):
    # BigQuery re-executes a non-recursive CTE at every reference, so each CTE
    # downstream of the measurement scan must be referenced exactly once.
    monkeypatch.setenv("WORKSPACE_CDR", DATASET)
    sql = build_all_of_us_measurement_sql()
    assert sql.count(f"`{DATASET}.measurement`") == 1
    for common_table in ("analyte_rows", "classified_rows", "valued_rows", "person_days", "person_summaries"):
        assert f"{common_table} AS (" in sql
        assert len(re.findall(rf"\b(?:FROM|JOIN) {common_table}\b", sql)) == 1, common_table


def test_measurement_query_config_types_every_parameter():
    definition = resolve_measurement_definition("ldl_cholesterol")
    config = build_all_of_us_measurement_query_config(definition)
    parameters = {parameter.name: parameter for parameter in config.query_parameters}
    assert isinstance(parameters["loinc_codes"], bigquery.ArrayQueryParameter)
    assert parameters["loinc_codes"].values == ["13457-7", "18262-6", "2089-1", "96259-7"]
    assert parameters["unit_codes"].values == ["mg/dL", "mmol/L"]
    assert parameters["unit_scales"].array_type == "FLOAT64"
    assert parameters["unit_scales"].values == [1.0, 38.67]
    assert parameters["treatment_atc_codes"].values == ["C10"]
    assert isinstance(parameters["log_scale"], bigquery.ScalarQueryParameter)
    assert parameters["log_scale"].type_ == "BOOL"
    assert parameters["log_scale"].value is False
    assert parameters["plausible_low"].value == 10.0
    assert parameters["value_formula"].value == "identity"
    untreated_trait = build_all_of_us_measurement_query_config(resolve_measurement_definition("haptoglobin"))
    assert {parameter.name: parameter for parameter in untreated_trait.query_parameters}[
        "treatment_atc_codes"
    ].values == []


def test_hba1c_ifcc_units_convert_by_the_ngsp_master_equation():
    conversion = {
        unit.ucum_code: unit for unit in resolve_measurement_definition("hemoglobin_a1c").unit_conversions
    }["mmol/mol"]
    # IFCC = 10.93 * NGSP - 23.50, so 53 mmol/mol is NGSP 7.0%.
    assert conversion.scale * 53.0 + conversion.offset == pytest.approx((53.0 + 23.50) / 10.93, abs=0.002)


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


def test_untreated_occasions_take_precedence_and_treated_ones_are_corrected():
    ldl = resolve_measurement_definition("ldl_cholesterol")
    rows, _true_means = _synthetic_person_rows(30, between_variance=400.0, within_variance=100.0, seed=4)
    rows += [
        _person_row(1001, untreated=(2, 150.0, 25.0, 50.0), treated=(3, 90.0, 16.0, 55.0)),
        _person_row(1002, treated=(2, 70.0, 9.0, 60.0)),
        _person_row(1003, excluded={"implausible": 2}),
    ]
    training_rows, _columns, summary = build_all_of_us_measurement_targets(ldl, rows)
    by_person = {row["person_id"]: row for row in training_rows}
    assert "1003" not in by_person
    assert len(by_person) == 32
    assert by_person["1001"]["measurement_source"] == "untreated"
    assert by_person["1001"]["occasion_count"] == 2
    assert by_person["1001"]["age_at_measurement"] == 50.0
    assert by_person["1002"]["measurement_source"] == "treated_corrected"
    assert summary["n_persons_untreated"] == 31
    assert summary["n_persons_treated_corrected"] == 1
    assert summary["n_persons_without_retained_occasion"] == 1
    assert summary["excluded_row_counts"]["implausible"] == 2


@pytest.mark.parametrize(
    ("trait", "treated_mean", "treated_variance", "expected_mean", "expected_variance"),
    [
        # LDL / 0.7 on the linear scale scales the mean and the variance.
        ("ldl_cholesterol", 70.0, 9.0, 100.0, 9.0 / 0.49),
        # SBP + 15 mmHg shifts the mean only.
        ("systolic_blood_pressure", 130.0, 25.0, 145.0, 25.0),
    ],
)
def test_treatment_corrections_map_the_occasion_statistics_exactly(
    monkeypatch, trait, treated_mean, treated_variance, expected_mean, expected_variance
):
    definition = resolve_measurement_definition(trait)
    captured: dict[str, np.ndarray] = {}

    def capture_components(occasion_counts, person_means, within_sum_squares, design):
        captured["means"] = person_means.copy()
        captured["within"] = within_sum_squares.copy()
        return 1.0, 1.0

    monkeypatch.setattr("sv_pgs.all_of_us._estimate_person_variance_components", capture_components)
    rows = [
        _person_row(1, treated=(2, treated_mean, treated_variance, 60.0)),
        _person_row(2, untreated=(2, 0.0, 1.0, 40.0), sex_at_birth_concept_id=45880669),
        _person_row(3, untreated=(2, 0.0, 1.0, 50.0)),
    ]
    build_all_of_us_measurement_targets(definition, rows)
    assert captured["means"][0] == pytest.approx(expected_mean)
    assert captured["within"][0] == pytest.approx(2 * expected_variance)


def test_ratio_correction_on_the_log_scale_shifts_the_mean_by_log_amount(monkeypatch):
    definition = _definition(
        log_scale=True,
        treatment=TreatmentRule(LIPID_LOWERING, TreatmentCorrection.DIVIDE, 0.7, "test"),
    )
    captured: dict[str, np.ndarray] = {}

    def capture_components(occasion_counts, person_means, within_sum_squares, design):
        captured["means"] = person_means.copy()
        captured["within"] = within_sum_squares.copy()
        return 1.0, 1.0

    monkeypatch.setattr("sv_pgs.all_of_us._estimate_person_variance_components", capture_components)
    rows = [
        _person_row(1, treated=(3, math.log(70.0), 0.04, 60.0)),
        _person_row(2, untreated=(2, 4.0, 0.1, 40.0), sex_at_birth_concept_id=45880669),
        _person_row(3, untreated=(2, 4.5, 0.1, 50.0)),
    ]
    build_all_of_us_measurement_targets(definition, rows)
    assert captured["means"][0] == pytest.approx(math.log(100.0))
    assert captured["within"][0] == pytest.approx(3 * 0.04)


def test_treated_only_persons_are_dropped_when_no_correction_exists():
    hba1c = resolve_measurement_definition("hemoglobin_a1c")
    rows, _true_means = _synthetic_person_rows(40, between_variance=0.5, within_variance=0.2, seed=3)
    rows.append(_person_row(1000, treated=(4, 8.5, 0.3, 62.0)))
    training_rows, _columns, summary = build_all_of_us_measurement_targets(hba1c, rows)
    assert "1000" not in {row["person_id"] for row in training_rows}
    assert summary["n_persons_treated_only_excluded"] == 1
    assert summary["n_persons"] == 40


def test_treated_occasions_for_a_trait_without_a_medication_rule_raise():
    haptoglobin = resolve_measurement_definition("haptoglobin")
    with pytest.raises(ValueError, match="without a medication rule"):
        build_all_of_us_measurement_targets(haptoglobin, [_person_row(1, treated=(2, 4.0, 0.1, 50.0))])


def test_person_blup_equals_the_dense_henderson_mixed_model_solution():
    generator = np.random.default_rng(11)
    occasion_counts = np.array([1, 2, 5, 3, 1, 4, 2, 6])
    person_count = len(occasion_counts)
    design = np.column_stack(
        [np.ones(person_count), generator.normal(size=person_count), generator.integers(0, 2, person_count)]
    )
    between_variance, within_variance = 0.7, 1.3
    occasions = [
        generator.normal(1.0 + design[index, 1], 1.0, size=count) for index, count in enumerate(occasion_counts)
    ]
    # Occasion-level model y = X gamma + Z b + e with V = sigma_b^2 Z Z' + sigma_e^2 I.
    occasion_values = np.concatenate(occasions)
    membership = np.repeat(np.arange(person_count), occasion_counts)
    occasion_design = design[membership]
    incidence = np.zeros((len(occasion_values), person_count))
    incidence[np.arange(len(occasion_values)), membership] = 1.0
    covariance = between_variance * incidence @ incidence.T + within_variance * np.eye(len(occasion_values))
    precision = np.linalg.inv(covariance)
    fixed_effects = np.linalg.solve(
        occasion_design.T @ precision @ occasion_design, occasion_design.T @ precision @ occasion_values
    )
    random_effects = between_variance * incidence.T @ precision @ (occasion_values - occasion_design @ fixed_effects)
    expected = design @ fixed_effects + random_effects

    person_means = np.array([values.mean() for values in occasions])
    targets, reliabilities = _person_blup(
        occasion_counts.astype(float), person_means, design, between_variance, within_variance
    )
    np.testing.assert_allclose(targets, expected, rtol=1e-10, atol=1e-10)
    repeatability = between_variance / (between_variance + within_variance)
    np.testing.assert_allclose(
        reliabilities, occasion_counts * repeatability / (1.0 + (occasion_counts - 1.0) * repeatability)
    )


def test_variance_component_estimators_are_unbiased():
    generator = np.random.default_rng(5)
    between_variance, within_variance = 0.8, 0.5
    person_count, replicate_count = 150, 600
    occasion_counts = generator.integers(1, 5, size=person_count).astype(float)
    ages = generator.uniform(20.0, 80.0, size=person_count)
    design = np.column_stack([np.ones(person_count), ages - ages.mean(), (ages - ages.mean()) ** 2])
    gamma = np.array([2.0, 0.05, -0.001])
    estimates = np.empty((replicate_count, 2))
    for replicate in range(replicate_count):
        person_effects = generator.normal(0.0, math.sqrt(between_variance), size=person_count)
        means = np.empty(person_count)
        within_sum_squares = np.empty(person_count)
        for index in range(person_count):
            values = design[index] @ gamma + person_effects[index] + generator.normal(
                0.0, math.sqrt(within_variance), size=int(occasion_counts[index])
            )
            means[index] = values.mean()
            within_sum_squares[index] = ((values - values.mean()) ** 2).sum()
        estimates[replicate] = _estimate_person_variance_components(
            occasion_counts, means, within_sum_squares, design
        )
    mean_estimates = estimates.mean(axis=0)
    standard_errors = estimates.std(axis=0) / math.sqrt(replicate_count)
    assert abs(mean_estimates[0] - between_variance) < 4 * standard_errors[0]
    assert abs(mean_estimates[1] - within_variance) < 4 * standard_errors[1]


def test_variance_components_raise_without_repeats_or_between_person_signal():
    design = np.ones((4, 1))
    with pytest.raises(ValueError, match="not identifiable"):
        _estimate_person_variance_components(np.ones(4), np.arange(4.0), np.zeros(4), design)
    # Person means identical, large within-person scatter: sigma_b^2 estimate <= 0.
    with pytest.raises(ValueError, match="no between-person variance"):
        _estimate_person_variance_components(np.full(4, 5.0), np.full(4, 2.0), np.full(4, 40.0), design)


def test_blup_target_tracks_the_true_long_run_mean_better_than_the_raw_mean():
    rows, true_means = _synthetic_person_rows(3000, between_variance=1.0, within_variance=2.0, seed=7)
    definition = resolve_measurement_definition("mean_corpuscular_volume")
    training_rows, _columns, summary = build_all_of_us_measurement_targets(definition, rows)
    targets = np.array([row["target"] for row in training_rows])
    raw_means = np.array([row["untreated_mean"] for row in rows])
    assert np.mean((targets - true_means) ** 2) < np.mean((raw_means - true_means) ** 2)
    assert summary["between_person_variance"] == pytest.approx(1.0, rel=0.15)
    assert summary["within_person_variance"] == pytest.approx(2.0, rel=0.1)
    assert summary["repeatability"] == pytest.approx(1.0 / 3.0, rel=0.15)


def test_rank_inverse_normal_uses_blom_scores_with_average_ties():
    transformed = _rank_inverse_normal(np.array([3.0, 1.0, 2.0, 2.0]))
    ranks = np.array([4.0, 1.0, 2.5, 2.5])
    np.testing.assert_allclose(transformed, ndtri((ranks - 0.375) / 4.25))


def test_training_rows_carry_targets_covariates_and_one_hot_sex():
    rows, _true_means = _synthetic_person_rows(200, between_variance=1.0, within_variance=1.0, seed=1)
    rows[0]["unrecognized_unit_labels"] = ["mmol/L"]
    rows[1]["unrecognized_unit_labels"] = ["mmol/L", "unit_concept_id=0"]
    definition = resolve_measurement_definition("mean_corpuscular_volume")
    training_rows, columns, summary = build_all_of_us_measurement_targets(definition, rows)
    assert columns == (
        "sex_at_birth_concept_id_45878463",
        "sex_at_birth_concept_id_45880669",
        "race_concept_id_8527",
        "ethnicity_concept_id_38003564",
    )
    first = training_rows[0]
    assert first["age_at_measurement"] == rows[0]["untreated_mean_age"]
    assert first["age_at_measurement_squared"] == rows[0]["untreated_mean_age_squared"]
    assert first["sex_at_birth_concept_id_45878463"] == 1
    assert "sex_at_birth_concept_id" not in first
    assert 0.0 < first["target_reliability"] < 1.0
    inverse_normal = np.array([row["target_inverse_normal"] for row in training_rows])
    assert inverse_normal.mean() == pytest.approx(0.0, abs=1e-9)
    assert summary["unrecognized_unit_person_counts"] == {"mmol/L": 2, "unit_concept_id=0": 1}


# ---------------------------------------------------------------------------
# Sample table, runner and CLI
# ---------------------------------------------------------------------------


def test_prepare_measurement_sample_table_writes_table_sql_and_metadata(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GOOGLE_PROJECT", "billing-project")
    monkeypatch.setenv("WORKSPACE_CDR", DATASET)
    rows, _true_means = _synthetic_person_rows(120, between_variance=1.0, within_variance=1.0, seed=2)
    fake_client = _FakeBigQueryClient(rows)
    outputs = prepare_all_of_us_measurement_sample_table(
        "haptoglobin", tmp_path / "haptoglobin.samples.tsv", client=fake_client
    )
    assert fake_client.sql == build_all_of_us_measurement_sql()
    parameters = {parameter.name: parameter for parameter in fake_client.job_config.query_parameters}
    assert parameters["loinc_codes"].values == ["4542-7"]
    with outputs.sample_table_path.open(encoding="utf-8") as handle:
        table = list(csv.DictReader(handle, delimiter="\t"))
    assert len(table) == 120
    assert list(table[0])[:9] == [
        "sample_id", "person_id", "target", "target_inverse_normal", "occasion_count",
        "target_reliability", "measurement_source", "age_at_measurement", "age_at_measurement_squared",
    ]
    assert outputs.sql_path.read_text(encoding="utf-8").strip() == build_all_of_us_measurement_sql()
    metadata = json.loads(outputs.metadata_path.read_text(encoding="utf-8"))
    assert metadata["trait"] == "haptoglobin"
    assert metadata["analysis_scale"] == "log(value + 0)"
    assert metadata["treatment"] is None
    assert metadata["query_parameters"]["loinc_codes"] == ["4542-7"]
    assert metadata["cdr_dataset"] == DATASET
    assert metadata["n_persons"] == 120
    assert 0.0 < metadata["repeatability"] < 1.0


def test_prepare_measurement_sample_table_requires_workspace_cdr(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GOOGLE_PROJECT", "billing-project")
    monkeypatch.delenv("WORKSPACE_CDR", raising=False)
    with pytest.raises(ValueError, match="WORKSPACE_CDR"):
        prepare_all_of_us_measurement_sample_table(
            "haptoglobin", tmp_path / "out.tsv", client=_FakeBigQueryClient([])
        )


def test_expand_one_hot_covariates_handles_sex_at_birth(tmp_path: Path):
    table = tmp_path / "trait.samples.with_pcs.tsv"
    table.write_text(
        "sample_id\ttarget\tage_at_measurement\tage_at_measurement_squared\tsex_at_birth_concept_id_1"
        "\tsex_at_birth_concept_id_2\trace_concept_id_3\tethnicity_concept_id_4\n"
        "a\t1.0\t50\t2504\t1\t0\t1\t1\n"
        "b\t2.0\t60\t3604\t1\t0\t1\t1\n"
        "c\t3.0\t40\t1604\t0\t1\t1\t1\n",
        encoding="utf-8",
    )
    expanded = aou_runner._expand_one_hot_covariates(aou_runner.MEASUREMENT_COVARIATES, table)
    assert expanded == ["age_at_measurement", "age_at_measurement_squared", "sex_at_birth_concept_id_2"]


def test_run_all_of_us_prepares_a_trait_table_and_uses_measurement_covariates(monkeypatch, tmp_path: Path):
    class _Dataset:
        def __init__(self) -> None:
            self.targets = np.array([0.1, 1.3, 2.2], dtype=np.float32)
            self.variant_stats = None
            self.variant_records: list = []
            self.variant_stats_minimum_scale: float | None = None

    monkeypatch.setattr(aou_runner, "check_aou_preflight", _successful_preflight)
    monkeypatch.setattr("sv_pgs.genotype.require_gpu", lambda: None)
    prepared: list[tuple[str, Path]] = []
    loaded_covariates: list[list[str]] = []
    trait_types: list[object] = []

    def fake_prepare_measurement(trait, output_path, **kwargs):
        prepared.append((trait, Path(output_path)))
        Path(output_path).write_text("sample_id\n", encoding="utf-8")
        Path(str(output_path) + ".metadata.json").write_text("{}", encoding="utf-8")

    def fake_merge(sample_table_path, ancestry_path, output_path, n_pcs):
        Path(output_path).write_text(
            "sample_id\tperson_id\ttarget\tage_at_measurement\tage_at_measurement_squared"
            "\tsex_at_birth_concept_id_1\tsex_at_birth_concept_id_2\trace_concept_id_3\tethnicity_concept_id_4\tPC1\n"
            "1\t1\t0.1\t50\t2504\t1\t0\t1\t1\t0.1\n"
            "2\t2\t1.3\t60\t3604\t1\t0\t1\t1\t0.2\n"
            "3\t3\t2.2\t40\t1604\t0\t1\t1\t1\t0.3\n",
            encoding="utf-8",
        )
        return output_path, ["PC1"]

    def fake_load(**kwargs):
        loaded_covariates.append(list(kwargs["covariate_columns"]))
        return _Dataset()

    def fake_pipeline(**kwargs):
        trait_types.append(kwargs["config"].trait_type)

    monkeypatch.setattr(aou_runner, "prepare_all_of_us_measurement_sample_table", fake_prepare_measurement)
    monkeypatch.setattr(
        aou_runner,
        "prepare_all_of_us_disease_sample_table",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("disease preparation must not run")),
    )
    monkeypatch.setattr(aou_runner, "download_ancestry_preds", lambda work_dir: tmp_path / "ancestry.tsv")
    monkeypatch.setattr(aou_runner, "merge_pcs_into_sample_table", fake_merge)
    monkeypatch.setattr(aou_runner, "download_array_plink", lambda work_dir: tmp_path / "arrays.bed")
    monkeypatch.setattr(aou_runner, "load_multi_source_dataset_from_files", fake_load)
    monkeypatch.setattr(aou_runner, "run_training_pipeline", fake_pipeline)
    monkeypatch.setattr(aou_runner, "release_process_memory", lambda: None)

    aou_runner.run_all_of_us(
        phenotype="mean_corpuscular_volume",
        chromosomes=[22],
        output_base=str(tmp_path / "mcv_results"),
        variants="snp",
    )

    assert prepared == [
        ("mean_corpuscular_volume", tmp_path / "mcv_results" / "mean_corpuscular_volume.samples.tsv")
    ]
    expected_covariates = [
        "age_at_measurement", "age_at_measurement_squared", "sex_at_birth_concept_id_2", "PC1",
    ]
    assert loaded_covariates == [expected_covariates, expected_covariates]
    assert trait_types == [aou_runner.TraitType.QUANTITATIVE]
    run_metadata = json.loads(aou_runner._aou_run_metadata_path(tmp_path / "mcv_results").read_text())
    assert run_metadata["disease"] == "mean_corpuscular_volume"


def _successful_preflight(cache_dir: Path, *, required_stage_bytes: int, required_temp_bytes: int):
    from sv_pgs.preflight import AouPreflightReport

    cache_dir.mkdir(parents=True, exist_ok=True)
    return AouPreflightReport(
        cdr_storage_path=None,
        workspace_bucket=None,
        google_project=None,
        cache_dir=cache_dir,
        cache_storage_class="local_hot",
        free_bytes=required_stage_bytes + required_temp_bytes,
        required_stage_bytes=required_stage_bytes,
        required_temp_bytes=required_temp_bytes,
        cuda_visible_devices=["GPU 0: unit-test device"],
        cupy_available=True,
        cupy_devices=1,
        jax_preallocate="false",
        jax_mem_fraction=None,
    )


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


def test_cli_run_all_of_us_forwards_a_trait_as_its_canonical_phenotype(monkeypatch, tmp_path: Path):
    calls: dict[str, object] = {}
    monkeypatch.setattr("sv_pgs.cli.run_all_of_us", lambda **kwargs: calls.update(kwargs))
    assert main(["run-all-of-us", "--trait", "ggt", "--chromosomes", "22", "--output-dir", str(tmp_path)]) == 0
    assert calls["phenotype"] == "gamma_glutamyl_transferase"


@pytest.mark.parametrize(
    "arguments",
    [
        ["--trait", "ggt", "--disease", "asthma"],
        ["--trait", "ggt", "--all-diseases"],
        ["--trait", "hypertension"],
        ["--disease", "ggt"],
    ],
)
def test_cli_run_all_of_us_rejects_mixed_or_wrong_kind_phenotypes(tmp_path: Path, arguments):
    with pytest.raises(ValueError):
        main(["run-all-of-us", *arguments, "--output-dir", str(tmp_path)])

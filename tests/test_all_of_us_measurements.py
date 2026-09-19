"""All of Us quantitative traits from the OMOP measurement table.

Covers the trait catalogue, the BigQuery SQL against the OMOP CDM v5.4 schema,
the query parameters, the per-person target construction (treatment
precedence, corrections and indicator pooling, variance components,
empirical BLUP against the dense Henderson mixed-model solution, inverse
normal transform), the sample table writer and the runner/CLI wiring.
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
    DIABETES,
    DISEASE_DEFINITIONS,
    LDL_LOWERING,
    LIPID_MODIFYING,
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
    _estimate_person_variance_components,
    _person_blup,
    _person_design,
    _rank_inverse_normal,
    available_measurement_names,
    build_all_of_us_measurement_query_config,
    build_all_of_us_measurement_query_parameters,
    build_all_of_us_measurement_sql,
    build_all_of_us_measurement_targets,
    measurement_covariate_columns,
    phenotype_fingerprint,
    prepare_all_of_us_measurement_census,
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
        "sex_at_birth_name": {45878463: "female", 45880669: "male"}.get(sex_at_birth_concept_id),
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
    canonical_unit="milligram per deciliter",
    unit_conversions=(UnitConversion("milligram per deciliter", 1.0),),
    plausible_range=(1.0, 100.0),
    log_scale=False,
)


def _definition(**overrides: object) -> MeasurementDefinition:
    return dataclasses.replace(_BASE_DEFINITION, **overrides)


def _captured_person_statistics(monkeypatch, definition: MeasurementDefinition, rows) -> dict[str, np.ndarray]:
    """Run the target builder with the variance components pinned and return
    the per-person means and within-person sums of squares it formed."""
    captured: dict[str, np.ndarray] = {}

    def capture_components(occasion_counts, person_means, within_sum_squares, design):
        captured["counts"] = occasion_counts.copy()
        captured["means"] = person_means.copy()
        captured["within"] = within_sum_squares.copy()
        return 1.0, 1.0

    monkeypatch.setattr("sv_pgs.all_of_us._estimate_person_variance_components", capture_components)
    build_all_of_us_measurement_targets(definition, rows)
    return captured


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


def test_catalogue_is_the_panels_sixteen_quantitative_traits():
    assert available_measurement_names() == sorted([
        "height", "body_mass_index", "systolic_blood_pressure", "diastolic_blood_pressure", "heart_rate",
        "mean_corpuscular_volume", "platelet_count", "white_blood_cell_count", "eosinophil_count",
        "total_bilirubin", "alkaline_phosphatase", "egfr_ckd_epi_2021", "hdl_cholesterol", "triglycerides",
        "ldl_cholesterol", "hemoglobin_a1c",
    ])
    # The panel's four diseases, each paired with a proxy above.
    for disease in ("hypertension", "type2_diabetes", "coronary_artery_disease", "asthma"):
        assert resolve_all_of_us_phenotype(disease).control_exclusion_snomed_codes


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
    assert treatments["diastolic_blood_pressure"].amount == 10.0
    for trait in ("heart_rate", "body_mass_index"):
        assert treatments[trait].correction is TreatmentCorrection.EXCLUDE
    # HDL and triglycerides are not corrected by convention; they carry the
    # on-treatment fraction instead.
    for trait in ("hdl_cholesterol", "triglycerides"):
        assert treatments[trait] == TreatmentRule(
            LIPID_MODIFYING, TreatmentCorrection.INDICATOR, None, treatments[trait].citation
        )
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
        ({"plausible_range": (5.0, 5.0)}, "empty plausible range"),
        ({"log_scale": True, "plausible_range": (0.0, 5.0)}, "log scale needs"),
        ({"log_offset": 0.05}, "log_offset without log scale"),
        ({"value_formula": "mdrd"}, "unknown value formula"),
        (
            {"log_scale": True, "treatment": TreatmentRule(LDL_LOWERING, TreatmentCorrection.ADD, 1.0, "test")},
            "additive correction",
        ),
        (
            {
                "log_scale": True,
                "log_offset": 0.05,
                "treatment": TreatmentRule(LDL_LOWERING, TreatmentCorrection.DIVIDE, 0.7, "test"),
            },
            "ratio correction",
        ),
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
        TreatmentRule(LDL_LOWERING, TreatmentCorrection.INDICATOR, 0.7, "test")
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
        "person_days", "person_summaries",
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
    assert isinstance(parameters["log_scale"], bigquery.ScalarQueryParameter)
    assert parameters["log_scale"].type_ == "BOOL"
    assert parameters["log_scale"].value is False
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
    assert by_person["1001"]["treated_occasion_fraction"] == 0.0
    assert by_person["1002"]["measurement_source"] == "treated_corrected"
    assert by_person["1002"]["treated_occasion_fraction"] == 1.0
    assert summary["n_persons_by_source"] == {"untreated": 31, "treated_corrected": 1}
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
    rows = [
        _person_row(1, treated=(2, treated_mean, treated_variance, 60.0)),
        _person_row(2, untreated=(2, 0.0, 1.0, 40.0), sex_at_birth_concept_id=45880669),
        _person_row(3, untreated=(2, 0.0, 1.0, 50.0)),
    ]
    captured = _captured_person_statistics(monkeypatch, resolve_measurement_definition(trait), rows)
    assert captured["means"][0] == pytest.approx(expected_mean)
    assert captured["within"][0] == pytest.approx(2 * expected_variance)


def test_ratio_correction_on_the_log_scale_shifts_the_mean_by_log_amount(monkeypatch):
    definition = _definition(
        log_scale=True,
        treatment=TreatmentRule(LDL_LOWERING, TreatmentCorrection.DIVIDE, 0.7, "test"),
    )
    rows = [
        _person_row(1, treated=(3, math.log(70.0), 0.04, 60.0)),
        _person_row(2, untreated=(2, 4.0, 0.1, 40.0), sex_at_birth_concept_id=45880669),
        _person_row(3, untreated=(2, 4.5, 0.1, 50.0)),
    ]
    captured = _captured_person_statistics(monkeypatch, definition, rows)
    assert captured["means"][0] == pytest.approx(math.log(100.0))
    assert captured["within"][0] == pytest.approx(3 * 0.04)


def test_indicator_rule_pools_both_groups_exactly_and_records_the_treated_fraction(monkeypatch):
    hdl = resolve_measurement_definition("hdl_cholesterol")
    untreated_values = np.array([3.8, 3.9])
    treated_values = np.array([4.1, 4.3, 4.0])
    rows = [
        _person_row(
            1,
            untreated=(2, float(untreated_values.mean()), float(untreated_values.var()), 50.0),
            treated=(3, float(treated_values.mean()), float(treated_values.var()), 56.0),
        ),
        _person_row(2, untreated=(2, 4.0, 0.1, 40.0), sex_at_birth_concept_id=45880669),
        _person_row(3, untreated=(2, 4.2, 0.1, 45.0)),
    ]
    captured = _captured_person_statistics(monkeypatch, hdl, rows)
    pooled = np.concatenate([untreated_values, treated_values])
    assert captured["counts"][0] == 5
    assert captured["means"][0] == pytest.approx(pooled.mean())
    assert captured["within"][0] == pytest.approx(((pooled - pooled.mean()) ** 2).sum())
    monkeypatch.undo()
    rows, _true_means = _synthetic_person_rows(40, between_variance=0.1, within_variance=0.05, seed=9)
    rows.append(
        _person_row(
            1000,
            untreated=(2, float(untreated_values.mean()), float(untreated_values.var()), 50.0),
            treated=(3, float(treated_values.mean()), float(treated_values.var()), 56.0),
        )
    )
    training_rows, _columns, summary = build_all_of_us_measurement_targets(hdl, rows)
    pooled_row = {row["person_id"]: row for row in training_rows}["1000"]
    assert pooled_row["treated_occasion_fraction"] == pytest.approx(0.6)
    assert pooled_row["age_at_measurement"] == pytest.approx((2 * 50.0 + 3 * 56.0) / 5)
    assert summary["n_persons_by_source"] == {"pooled": 41}
    assert "treated_occasion_fraction" in measurement_covariate_columns(hdl)
    assert "treated_occasion_fraction" not in measurement_covariate_columns(resolve_measurement_definition("mcv"))


def test_treated_only_persons_are_dropped_when_no_correction_exists():
    heart_rate = resolve_measurement_definition("heart_rate")
    rows, _true_means = _synthetic_person_rows(40, between_variance=50.0, within_variance=20.0, seed=3)
    rows.append(_person_row(1000, treated=(4, 62.0, 30.0, 62.0)))
    training_rows, _columns, summary = build_all_of_us_measurement_targets(heart_rate, rows)
    assert "1000" not in {row["person_id"] for row in training_rows}
    assert summary["n_persons_treated_only_excluded"] == 1
    assert summary["n_persons"] == 40


def test_treated_occasions_for_a_trait_without_a_medication_rule_raise():
    height = resolve_measurement_definition("height")
    with pytest.raises(ValueError, match="without a medication rule"):
        build_all_of_us_measurement_targets(height, [_person_row(1, treated=(2, 170.0, 1.0, 50.0))])


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


def test_person_design_models_age_by_sex_and_drops_empty_columns():
    ages = np.array([40.0, 50.0, 60.0, 70.0])
    female = np.array([1.0, 0.0, 1.0, 0.0])
    design = _person_design(ages, ages**2 + 4.0, female, [45878463, 45880669, 45878463, 45880669])
    centered = ages - ages.mean()
    np.testing.assert_allclose(design[:, -1], centered * female)
    assert design.shape == (4, 5)
    # One sex only: no sex indicator and no interaction column.
    single_sex = _person_design(ages, ages**2 + 4.0, np.zeros(4), [45880669] * 4)
    assert single_sex.shape == (4, 3)


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
    rows[0]["unrecognized_unit_labels"] = ["millimole per liter", "millimole per liter"]
    rows[1]["unrecognized_unit_labels"] = ["millimole per liter", "unit_concept_id=9999"]
    definition = resolve_measurement_definition("mean_corpuscular_volume")
    training_rows, columns, summary = build_all_of_us_measurement_targets(definition, rows)
    assert columns == ("sex_at_birth_concept_id_45878463", "sex_at_birth_concept_id_45880669")
    female_row, male_row = training_rows[0], training_rows[1]
    assert female_row["age_at_measurement"] == rows[0]["untreated_mean_age"]
    assert female_row["age_at_measurement_squared"] == rows[0]["untreated_mean_age_squared"]
    assert female_row["age_at_measurement_x_female"] == rows[0]["untreated_mean_age"]
    assert male_row["age_at_measurement_x_female"] == 0.0
    assert female_row["log_occasion_count"] == pytest.approx(math.log(rows[0]["untreated_occasion_count"]))
    assert female_row["sex_at_birth_concept_id_45878463"] == 1
    assert "sex_at_birth_concept_id" not in female_row
    assert 0.0 < female_row["target_reliability"] < 1.0
    inverse_normal = np.array([row["target_inverse_normal"] for row in training_rows])
    assert inverse_normal.mean() == pytest.approx(0.0, abs=1e-9)
    # Persons, not days, per unrecognized unit label.
    assert summary["unrecognized_unit_person_counts"] == {"millimole per liter": 2, "unit_concept_id=9999": 1}


# ---------------------------------------------------------------------------
# Sample table, runner and CLI
# ---------------------------------------------------------------------------


def test_prepare_measurement_sample_table_writes_table_sql_and_metadata(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GOOGLE_PROJECT", "billing-project")
    monkeypatch.setenv("WORKSPACE_CDR", DATASET)
    rows, _true_means = _synthetic_person_rows(120, between_variance=1.0, within_variance=1.0, seed=2)
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
    assert list(table[0])[:12] == [
        "sample_id", "person_id", "target", "target_inverse_normal", "occasion_count",
        "target_reliability", "measurement_source", "age_at_measurement", "age_at_measurement_squared",
        "age_at_measurement_x_female", "log_occasion_count", "treated_occasion_fraction",
    ]
    assert outputs.sql_path.read_text(encoding="utf-8").strip() == build_all_of_us_measurement_sql()
    metadata = json.loads(outputs.metadata_path.read_text(encoding="utf-8"))
    assert metadata["trait"] == "total_bilirubin"
    assert metadata["phenotype_fingerprint"] == phenotype_fingerprint(definition)
    assert metadata["covariate_columns"] == list(measurement_covariate_columns(definition))
    assert metadata["analysis_scale"] == "log(value + 0)"
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
    assert fingerprint != phenotype_fingerprint(resolve_all_of_us_phenotype("asthma"))
    monkeypatch.setenv("WORKSPACE_CDR", "fc-aou-cdr-prod-ct.C2025Q4R5")
    assert fingerprint != phenotype_fingerprint(height)


def test_expand_one_hot_covariates_handles_a_trait_covariate_list(tmp_path: Path):
    table = tmp_path / "trait.samples.with_pcs.tsv"
    table.write_text(
        "sample_id\ttarget\tage_at_measurement\tage_at_measurement_squared\tage_at_measurement_x_female"
        "\tlog_occasion_count\ttreated_occasion_fraction\tsex_at_birth_concept_id_1\tsex_at_birth_concept_id_2\n"
        "a\t1.0\t50\t2504\t50\t0.7\t0\t1\t0\n"
        "b\t2.0\t60\t3604\t60\t1.1\t0.5\t1\t0\n"
        "c\t3.0\t40\t1604\t0\t0\t1\t0\t1\n",
        encoding="utf-8",
    )
    covariates = list(measurement_covariate_columns(resolve_measurement_definition("hdl_cholesterol")))
    expanded = aou_runner._expand_one_hot_covariates(covariates, table)
    assert expanded == [
        "age_at_measurement",
        "age_at_measurement_squared",
        "age_at_measurement_x_female",
        "sex_at_birth_concept_id_2",
        "log_occasion_count",
        "treated_occasion_fraction",
    ]


def test_run_all_of_us_prepares_a_trait_table_and_uses_its_covariates(monkeypatch, tmp_path: Path):
    class _Dataset:
        def __init__(self) -> None:
            self.targets = np.array([0.1, 1.3, 2.2], dtype=np.float32)
            self.variant_stats = None
            self.variant_records: list = []
            self.variant_stats_minimum_scale: float | None = None

    monkeypatch.setenv("WORKSPACE_CDR", DATASET)
    monkeypatch.setattr(aou_runner, "check_aou_preflight", _successful_preflight)
    monkeypatch.setattr("sv_pgs.genotype.require_gpu", lambda: None)
    mcv = resolve_measurement_definition("mcv")
    prepared: list[tuple[str, Path]] = []
    loaded_covariates: list[list[str]] = []
    trait_types: list[object] = []

    def fake_prepare_measurement(trait, output_path, **kwargs):
        prepared.append((trait, Path(output_path)))
        Path(output_path).write_text("sample_id\n", encoding="utf-8")
        Path(str(output_path) + ".metadata.json").write_text(
            json.dumps(
                {
                    "phenotype_fingerprint": phenotype_fingerprint(mcv),
                    "covariate_columns": list(measurement_covariate_columns(mcv)),
                }
            ),
            encoding="utf-8",
        )

    def fake_merge(sample_table_path, ancestry_path, output_path, n_pcs):
        Path(output_path).write_text(
            "sample_id\tperson_id\ttarget\tage_at_measurement\tage_at_measurement_squared"
            "\tage_at_measurement_x_female\tlog_occasion_count\tsex_at_birth_concept_id_1"
            "\tsex_at_birth_concept_id_2\tPC1\n"
            "1\t1\t0.1\t50\t2504\t50\t0\t1\t0\t0.1\n"
            "2\t2\t1.3\t60\t3604\t60\t0.7\t1\t0\t0.2\n"
            "3\t3\t2.2\t40\t1604\t0\t1.1\t0\t1\t0.3\n",
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
    # A second run reuses the current table instead of preparing it again.
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
        "age_at_measurement", "age_at_measurement_squared", "age_at_measurement_x_female",
        "sex_at_birth_concept_id_2", "log_occasion_count", "PC1",
    ]
    assert loaded_covariates[:2] == [expected_covariates, expected_covariates]
    assert trait_types[0] == aou_runner.TraitType.QUANTITATIVE
    run_metadata = json.loads(aou_runner._aou_run_metadata_path(tmp_path / "mcv_results").read_text())
    assert run_metadata["disease"] == "mean_corpuscular_volume"
    assert run_metadata["phenotype_fingerprint"] == phenotype_fingerprint(mcv)


def test_run_all_of_us_rebuilds_a_table_prepared_from_another_definition(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("WORKSPACE_CDR", DATASET)
    monkeypatch.setattr(aou_runner, "check_aou_preflight", _successful_preflight)
    monkeypatch.setattr("sv_pgs.genotype.require_gpu", lambda: None)
    work_dir = tmp_path / "mcv_results"
    work_dir.mkdir()
    stale_table = work_dir / "mean_corpuscular_volume.samples.tsv"
    stale_table.write_text("sample_id\n", encoding="utf-8")
    Path(str(stale_table) + ".metadata.json").write_text(
        json.dumps({"phenotype_fingerprint": "an earlier definition", "covariate_columns": []}),
        encoding="utf-8",
    )
    prepared: list[str] = []

    def stop_after_prepare(trait, output_path, **kwargs):
        prepared.append(trait)
        raise RuntimeError("prepared")

    monkeypatch.setattr(aou_runner, "prepare_all_of_us_measurement_sample_table", stop_after_prepare)
    with pytest.raises(RuntimeError, match="prepared"):
        aou_runner.run_all_of_us(
            phenotype="mean_corpuscular_volume", chromosomes=[22], output_base=str(work_dir), variants="snp"
        )
    assert prepared == ["mean_corpuscular_volume"]


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
    assert main(["run-all-of-us", "--trait", "sbp", "--chromosomes", "22", "--output-dir", str(tmp_path)]) == 0
    assert calls["phenotype"] == "systolic_blood_pressure"


@pytest.mark.parametrize(
    "arguments",
    [
        ["--trait", "sbp", "--disease", "asthma"],
        ["--trait", "sbp", "--all-diseases"],
        ["--trait", "hypertension"],
        ["--disease", "sbp"],
    ],
)
def test_cli_run_all_of_us_rejects_mixed_or_wrong_kind_phenotypes(tmp_path: Path, arguments):
    with pytest.raises(ValueError):
        main(["run-all-of-us", *arguments, "--output-dir", str(tmp_path)])


def test_census_suppresses_cells_below_twenty_participants(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("GOOGLE_PROJECT", "billing-project")
    monkeypatch.setenv("WORKSPACE_CDR", DATASET)
    fake_client = _FakeBigQueryClient([
        {"trait_name": "height", "vocabulary_id": "LOINC", "concept_code": "8302-2",
         "matched_on_standard_concept": True, "unit_label": "inch (us)", "participant_count": 246040, "row_count": 900000},
        {"trait_name": "height", "vocabulary_id": "LOINC", "concept_code": "8302-2",
         "matched_on_standard_concept": True, "unit_label": "kilogram", "participant_count": 19, "row_count": 25},
    ])
    census_path = prepare_all_of_us_measurement_census(tmp_path / "census.tsv", client=fake_client)
    with census_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert [(row["unit_label"], row["participant_count"], row["row_count"], row["suppressed_below_minimum"]) for row in rows] == [
        ("inch (us)", "246040", "900000", "False"),
        ("kilogram", "", "", "True"),
    ]
    parameters = {parameter.name: parameter for parameter in fake_client.job_config.query_parameters}
    assert parameters["census_loinc_traits"].values.count("height") == 2
    assert parameters["census_physical_measurement_concept_ids"].values == [903133, 903118, 903115, 903126]


def test_cli_census_all_of_us_traits_writes_the_census(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.setattr(
        "sv_pgs.cli.prepare_all_of_us_measurement_census", lambda output_path: Path(output_path)
    )
    assert main(["census-all-of-us-traits", "--output", str(tmp_path / "census.tsv")]) == 0
    assert capsys.readouterr().out.strip() == f"census\t{tmp_path / 'census.tsv'}"

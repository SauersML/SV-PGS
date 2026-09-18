"""Execute the All of Us measurement query on synthetic OMOP CDM v5.4 tables.

The BigQuery query from build_all_of_us_measurement_sql is transpiled to
DuckDB with sqlglot, its @parameters replaced by the literal values of
build_all_of_us_measurement_query_parameters, and run against in-memory
tables laid out as `aou_workspace.cdr_dataset.<table>`. Each test checks the
per-person output rows against values computed independently here.
"""
from __future__ import annotations

import datetime
import math

import duckdb
import numpy as np
import pytest
import sqlglot
from sqlglot import exp

from sv_pgs.all_of_us import (
    MEASUREMENT_EXCLUSION_REASONS,
    MeasurementDefinition,
    build_all_of_us_measurement_query_parameters,
    build_all_of_us_measurement_sql,
    build_all_of_us_measurement_targets,
    resolve_measurement_definition,
)

DATASET = "aou_workspace.cdr_dataset"
OMOP_TABLES = {
    "person": (
        "person_id BIGINT, gender_concept_id BIGINT, year_of_birth BIGINT, race_concept_id BIGINT, "
        "ethnicity_concept_id BIGINT, sex_at_birth_concept_id BIGINT"
    ),
    "measurement": (
        "measurement_id BIGINT, person_id BIGINT, measurement_concept_id BIGINT, measurement_date DATE, "
        "measurement_type_concept_id BIGINT, operator_concept_id BIGINT, value_as_number DOUBLE, "
        "unit_concept_id BIGINT, visit_occurrence_id BIGINT, value_source_value VARCHAR"
    ),
    "observation": "observation_id BIGINT, person_id BIGINT, observation_concept_id BIGINT, observation_date DATE",
    "condition_occurrence": (
        "condition_occurrence_id BIGINT, person_id BIGINT, condition_concept_id BIGINT, condition_start_date DATE"
    ),
    "drug_exposure": "drug_exposure_id BIGINT, person_id BIGINT, drug_concept_id BIGINT, drug_exposure_start_date DATE",
    "visit_occurrence": "visit_occurrence_id BIGINT, person_id BIGINT, visit_concept_id BIGINT",
    "concept": (
        "concept_id BIGINT, concept_name VARCHAR, domain_id VARCHAR, vocabulary_id VARCHAR, "
        "concept_class_id VARCHAR, standard_concept VARCHAR, concept_code VARCHAR"
    ),
    "concept_ancestor": (
        "ancestor_concept_id BIGINT, descendant_concept_id BIGINT, min_levels_of_separation BIGINT, "
        "max_levels_of_separation BIGINT"
    ),
}
# (concept_id, concept_name, domain_id, vocabulary_id, concept_class_id, standard_concept, concept_code);
# the ids are the real OMOP vocabulary ids.
CONCEPTS = (
    (3016723, "Creatinine [Mass/volume] in Serum or Plasma", "Measurement", "LOINC", "Lab Test", "S", "2160-0"),
    (3020564, "Creatinine [Moles/volume] in Serum or Plasma", "Measurement", "LOINC", "Lab Test", "S", "14682-9"),
    (3004501, "Glucose [Mass/volume] in Serum or Plasma", "Measurement", "LOINC", "Lab Test", "S", "2345-7"),
    (3028437, "Cholesterol in LDL [Mass/volume] in Serum or Plasma by Direct assay", "Measurement", "LOINC",
     "Lab Test", "S", "18262-6"),
    (8840, "milligram per deciliter", "Unit", "UCUM", "Unit", "S", "mg/dL"),
    (8749, "micromole per liter", "Unit", "UCUM", "Unit", "S", "umol/L"),
    (8753, "millimole per liter", "Unit", "UCUM", "Unit", "S", "mmol/L"),
    (0, "No matching concept", "Metadata", "None", "Undefined", None, "No matching concept"),
    (4172703, "=", "Meas Value Operator", "SNOMED", "Qualifier Value", "S", "276136004"),
    (4171756, "<", "Meas Value Operator", "SNOMED", "Qualifier Value", "S", "276139006"),
    (9201, "Inpatient Visit", "Visit", "Visit", "Visit", "S", "IP"),
    (9202, "Outpatient Visit", "Visit", "Visit", "Visit", "S", "OP"),
    (9203, "Emergency Room Visit", "Visit", "Visit", "Visit", "S", "ER"),
    (262, "Emergency Room and Inpatient Visit", "Visit", "Visit", "Visit", "S", "ERIP"),
    (8717, "Inpatient Hospital", "Visit", "CMS Place of Service", "Visit", "S", "21"),
    (4088927, "Pregnancy, childbirth and puerperium finding", "Condition", "SNOMED", "Clinical Finding", "S",
     "248982007"),
    (4299535, "Pregnancy", "Condition", "SNOMED", "Clinical Finding", "S", "77386006"),
    (4243056, "Not pregnant", "Condition", "SNOMED", "Clinical Finding", "S", "60001007"),
    (4047564, "Routine antenatal care", "Observation", "SNOMED", "Procedure", "S", "134435003"),
    (21601853, "LIPID MODIFYING AGENTS", "Drug", "ATC", "ATC 2nd", "C", "C10"),
    (1545958, "atorvastatin", "Drug", "RxNorm", "Ingredient", "S", "83367"),
    (45878463, "Female", "Meas Value", "LOINC", "Answer", "S", "LA3-6"),
    (45880669, "Male", "Meas Value", "LOINC", "Answer", "S", "LA2-8"),
    (903096, "PMI: Skip", "Observation", "PPI", "Answer", None, "PMI_Skip"),
)
# (ancestor, descendant, min, max); every hierarchical concept is its own ancestor.
CONCEPT_ANCESTORS = (
    (9201, 9201, 0, 0), (9201, 8717, 1, 1), (9202, 9202, 0, 0), (9203, 9203, 0, 0), (262, 262, 0, 0),
    (4088927, 4088927, 0, 0), (4088927, 4299535, 2, 2), (4088927, 4243056, 2, 2),
    (4299535, 4299535, 0, 0), (4243056, 4243056, 0, 0), (4047564, 4047564, 0, 0),
    (21601853, 21601853, 0, 0), (21601853, 1545958, 3, 3), (1545958, 1545958, 0, 0),
)
FEMALE = 45878463
MALE = 45880669
LAB_RESULT_TYPE = 32856
SELF_REPORT_TYPE = 32865


def _duckdb_query(definition: MeasurementDefinition) -> str:
    parameters = build_all_of_us_measurement_query_parameters(definition)

    def literal(parameter_type: str, value: object) -> exp.Expression:
        if isinstance(value, list):
            return exp.Array(expressions=[literal(parameter_type, element) for element in value])
        if parameter_type == "BOOL":
            return exp.Boolean(this=value)
        if parameter_type == "STRING":
            return exp.Literal.string(str(value))
        return exp.cast(exp.Literal.number(repr(float(value))), "DOUBLE")

    def substitute(node: exp.Expression) -> exp.Expression:
        if isinstance(node, exp.Parameter):
            parameter_type, value = parameters[node.this.name]
            return literal(parameter_type, value)
        return node

    tree = sqlglot.parse_one(build_all_of_us_measurement_sql(), read="bigquery")
    return tree.transform(substitute).sql(dialect="duckdb")


class _Cdr:
    """In-memory OMOP CDR with the shared vocabulary preloaded."""

    def __init__(self) -> None:
        self.connection = duckdb.connect()
        self.connection.execute("ATTACH ':memory:' AS aou_workspace")
        self.connection.execute("CREATE SCHEMA aou_workspace.cdr_dataset")
        for table, columns in OMOP_TABLES.items():
            self.connection.execute(f"CREATE TABLE {DATASET}.{table} ({columns})")
        self.insert("concept", CONCEPTS)
        self.insert("concept_ancestor", CONCEPT_ANCESTORS)
        self._next_id = 1

    def insert(self, table: str, rows) -> None:
        rows = list(rows)
        placeholders = ", ".join("?" * len(rows[0]))
        self.connection.executemany(f"INSERT INTO {DATASET}.{table} VALUES ({placeholders})", rows)

    def next_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def person(self, person_id: int, year_of_birth: int, sex_at_birth_concept_id: int | None) -> None:
        self.insert("person", [(person_id, 0, year_of_birth, 8527, 38003564, sex_at_birth_concept_id)])

    def measurement(
        self,
        person_id: int,
        date: str,
        value: float | None,
        *,
        concept_id: int = 3016723,
        unit_concept_id: int | None = 8840,
        operator_concept_id: int | None = None,
        visit_concept_id: int | None = None,
        type_concept_id: int = LAB_RESULT_TYPE,
        value_source_value: str | None = None,
    ) -> None:
        visit_occurrence_id = None
        if visit_concept_id is not None:
            visit_occurrence_id = self.next_id()
            self.insert("visit_occurrence", [(visit_occurrence_id, person_id, visit_concept_id)])
        self.insert(
            "measurement",
            [(
                self.next_id(), person_id, concept_id, datetime.date.fromisoformat(date), type_concept_id,
                operator_concept_id, value, unit_concept_id, visit_occurrence_id, value_source_value,
            )],
        )

    def condition(self, person_id: int, concept_id: int, date: str) -> None:
        self.insert("condition_occurrence", [(self.next_id(), person_id, concept_id, datetime.date.fromisoformat(date))])

    def observation(self, person_id: int, concept_id: int, date: str) -> None:
        self.insert("observation", [(self.next_id(), person_id, concept_id, datetime.date.fromisoformat(date))])

    def drug(self, person_id: int, concept_id: int, date: str) -> None:
        self.insert("drug_exposure", [(self.next_id(), person_id, concept_id, datetime.date.fromisoformat(date))])

    def run(self, definition: MeasurementDefinition) -> dict[str, dict[str, object]]:
        cursor = self.connection.execute(_duckdb_query(definition))
        columns = [description[0] for description in cursor.description]
        return {row[0]: dict(zip(columns, row, strict=True)) for row in cursor.fetchall()}


@pytest.fixture
def cdr(monkeypatch: pytest.MonkeyPatch) -> _Cdr:
    monkeypatch.setenv("WORKSPACE_CDR", DATASET)
    return _Cdr()


def _age(date: str, year_of_birth: int) -> float:
    return (datetime.date.fromisoformat(date) - datetime.date(year_of_birth, 7, 1)).days / 365.25


def _ckd_epi_2021(creatinine_mg_dl: float, age: float, female: bool) -> float:
    kappa, alpha, sex_factor = (0.7, -0.241, 1.012) if female else (0.9, -0.302, 1.0)
    ratio = creatinine_mg_dl / kappa
    return 142.0 * min(ratio, 1.0) ** alpha * max(ratio, 1.0) ** -1.2 * 0.9938**age * sex_factor


def _excluded(row: dict[str, object]) -> dict[str, int]:
    counts = {reason: row[f"{reason}_row_count"] for reason in MEASUREMENT_EXCLUSION_REASONS}
    assert all(isinstance(count, int) for count in counts.values())
    return {reason: count for reason, count in counts.items() if count}


def test_row_rules_units_and_same_day_collapse(cdr: _Cdr):
    egfr = resolve_measurement_definition("egfr_ckd_epi_2021")
    cdr.person(1, 1960, FEMALE)
    cdr.measurement(1, "2015-03-01", 0.8, visit_concept_id=9202)
    cdr.measurement(1, "2015-03-01", 70.72, concept_id=3020564, unit_concept_id=8749)
    cdr.measurement(1, "2016-05-10", 1.0, operator_concept_id=4172703)
    cdr.measurement(1, "2017-01-01", 25.0)
    cdr.measurement(1, "2017-02-01", 0.3, operator_concept_id=4171756)
    cdr.measurement(1, "2017-03-01", 0.2, value_source_value="<0.2")
    cdr.measurement(1, "2017-04-01", 0.07, unit_concept_id=8753)
    cdr.measurement(1, "2017-05-01", 0.9, unit_concept_id=0)
    cdr.measurement(1, "2017-06-01", 0.9, visit_concept_id=8717)
    cdr.measurement(1, "2017-07-01", 0.9, visit_concept_id=262)
    cdr.measurement(1, "2017-07-02", 0.9, visit_concept_id=9203)
    cdr.measurement(1, "2017-08-01", 0.9, type_concept_id=SELF_REPORT_TYPE)
    cdr.measurement(1, "2018-01-01", None)
    cdr.measurement(1, "2018-02-01", 95.0, concept_id=3004501)
    cdr.person(2, 2000, MALE)
    cdr.measurement(2, "2015-01-01", 0.6)
    cdr.measurement(2, "2020-06-01", 1.1)
    cdr.measurement(2, "2021-06-01", 0.9)
    cdr.person(3, 1970, 903096)
    cdr.measurement(3, "2020-01-01", 1.0)

    rows = cdr.run(egfr)
    assert set(rows) == {"1", "2", "3"}

    first = rows["1"]
    assert first["measurement_row_count"] == 12
    assert _excluded(first) == {
        "censored": 2, "unrecognized_unit": 2, "implausible": 1, "self_reported": 1, "acute_care": 3,
    }
    assert sorted(first["unrecognized_unit_labels"]) == ["mmol/L", "unit_concept_id=0"]
    # 2015-03-01 has two rows (mg/dL and umol/L): one occasion, the mean of their log eGFRs.
    first_age = _age("2015-03-01", 1960)
    first_occasion = np.mean([
        math.log(_ckd_epi_2021(0.8, first_age, female=True)),
        math.log(_ckd_epi_2021(70.72 / 88.42, first_age, female=True)),
    ])
    second_age = _age("2016-05-10", 1960)
    second_occasion = math.log(_ckd_epi_2021(1.0, second_age, female=True))
    assert first["untreated_occasion_count"] == 2
    assert first["untreated_mean"] == pytest.approx(np.mean([first_occasion, second_occasion]), rel=1e-12)
    assert first["untreated_variance"] == pytest.approx(np.var([first_occasion, second_occasion]), rel=1e-9)
    assert first["untreated_mean_age"] == pytest.approx((first_age + second_age) / 2, rel=1e-12)
    assert first["untreated_mean_age_squared"] == pytest.approx((first_age**2 + second_age**2) / 2, rel=1e-12)
    assert first["treated_occasion_count"] == 0

    second = rows["2"]
    assert _excluded(second) == {"under_adult_age": 1}
    male_occasions = [
        math.log(_ckd_epi_2021(value, _age(date, 2000), female=False))
        for date, value in (("2020-06-01", 1.1), ("2021-06-01", 0.9))
    ]
    assert second["untreated_occasion_count"] == 2
    assert second["untreated_mean"] == pytest.approx(np.mean(male_occasions), rel=1e-12)

    # eGFR needs sex at birth; a skipped answer is not guessed.
    third = rows["3"]
    assert _excluded(third) == {"sex_unknown": 1}
    assert third["untreated_occasion_count"] == 0
    assert third["untreated_mean"] is None


def test_pregnancy_windows_come_from_conditions_and_antenatal_observations(cdr: _Cdr):
    haptoglobin = resolve_measurement_definition("haptoglobin")
    haptoglobin_concept = 3014431
    cdr.insert("concept", [(haptoglobin_concept, "Haptoglobin", "Measurement", "LOINC", "Lab Test", "S", "4542-7")])
    cdr.person(1, 1985, FEMALE)
    cdr.condition(1, 4299535, "2019-06-01")
    cdr.condition(1, 4243056, "2012-01-01")
    cdr.observation(1, 4047564, "2023-01-01")
    for date, value in (
        ("2018-08-12", 100.0),  # 293 days before the pregnancy code: excluded
        ("2018-08-10", 110.0),  # 295 days before: kept
        ("2020-06-13", 120.0),  # 378 days after: excluded
        ("2020-06-14", 130.0),  # 379 days after: kept
        ("2012-02-01", 140.0),  # after a "Not pregnant" finding: kept
        ("2023-02-01", 150.0),  # after routine antenatal care (observation): excluded
    ):
        cdr.measurement(1, date, value, concept_id=haptoglobin_concept)
    row = cdr.run(haptoglobin)["1"]
    assert _excluded(row) == {"pregnancy": 3}
    assert row["untreated_occasion_count"] == 3
    assert row["untreated_mean"] == pytest.approx(np.mean(np.log([110.0, 130.0, 140.0])), rel=1e-12)


def test_occasions_on_or_after_the_first_exposure_are_treated(cdr: _Cdr):
    ldl = resolve_measurement_definition("ldl_cholesterol")
    cdr.person(1, 1950, MALE)
    cdr.drug(1, 1545958, "2018-01-01")
    cdr.drug(1, 1545958, "2019-01-01")
    cdr.measurement(1, "2017-06-01", 160.0, concept_id=3028437)
    cdr.measurement(1, "2018-01-01", 110.0, concept_id=3028437)
    cdr.measurement(1, "2018-06-01", 90.0, concept_id=3028437)
    cdr.person(2, 1955, FEMALE)
    cdr.measurement(2, "2017-06-01", 4.0, concept_id=3028437, unit_concept_id=8753)
    rows = cdr.run(ldl)
    treated_person = rows["1"]
    assert treated_person["untreated_occasion_count"] == 1
    assert treated_person["untreated_mean"] == pytest.approx(160.0)
    assert treated_person["treated_occasion_count"] == 2
    assert treated_person["treated_mean"] == pytest.approx(100.0)
    assert treated_person["treated_variance"] == pytest.approx(100.0)
    # mmol/L cholesterol converts at 38.67 mg/dL per mmol/L.
    assert rows["2"]["untreated_mean"] == pytest.approx(4.0 * 38.67)
    assert rows["2"]["treated_occasion_count"] == 0


def test_query_rows_feed_the_target_builder_end_to_end(cdr: _Cdr):
    generator = np.random.default_rng(0)
    mcv = resolve_measurement_definition("mean_corpuscular_volume")
    mcv_concept = 3023599
    cdr.insert("concept", [(mcv_concept, "MCV", "Measurement", "LOINC", "Lab Test", "S", "787-2")])
    person_effects = {}
    for person_id in range(1, 301):
        cdr.person(person_id, int(generator.integers(1940, 1990)), FEMALE if person_id % 2 else MALE)
        person_effects[str(person_id)] = generator.normal(0.0, 5.0)
        for occasion in range(int(generator.integers(1, 6))):
            cdr.measurement(
                person_id,
                f"{2010 + occasion}-0{1 + occasion}-15",
                90.0 + person_effects[str(person_id)] + generator.normal(0.0, 3.0),
                concept_id=mcv_concept,
                unit_concept_id=8583,
            )
    cdr.insert("concept", [(8583, "femtoliter", "Unit", "UCUM", "Unit", "S", "fL")])
    rows = list(cdr.run(mcv).values())
    training_rows, _columns, summary = build_all_of_us_measurement_targets(mcv, rows)
    assert summary["n_persons"] == 300
    assert summary["between_person_variance"] == pytest.approx(25.0, rel=0.3)
    assert summary["within_person_variance"] == pytest.approx(9.0, rel=0.2)
    targets = np.array([row["target"] for row in training_rows])
    effects = np.array([person_effects[row["person_id"]] for row in training_rows])
    assert np.corrcoef(targets, effects)[0, 1] > 0.85

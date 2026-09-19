"""Execute the All of Us phenotype queries on synthetic OMOP CDM v5.4 tables.

The BigQuery queries from build_all_of_us_measurement_sql and
build_all_of_us_disease_sql are transpiled to DuckDB with sqlglot, their
@parameters replaced by the literal values of the parameter builders, and run
against in-memory tables laid out as `aou_workspace.cdr_dataset.<table>`. Each
test checks the per-person output rows against values computed independently
here.
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
    DiseaseDefinition,
    LabCriterion,
    MeasurementDefinition,
    PersonOccasions,
    _prepare_training_rows,
    build_all_of_us_disease_query_parameters,
    build_all_of_us_disease_sql,
    build_all_of_us_lab_criterion_query_parameters,
    build_all_of_us_measurement_census_query_parameters,
    build_all_of_us_measurement_census_sql,
    build_all_of_us_measurement_query_parameters,
    build_all_of_us_measurement_sql,
    build_all_of_us_measurement_targets,
    lab_criterion_evidence,
    person_occasions,
    resolve_disease_definition,
    resolve_measurement_definition,
)
from tests.phenotype_bounds import sampling_bound, within_rounding

WORKING_BYTES = 1 << 28

DATASET = "aou_workspace.cdr_dataset"


# A plain trait's occasion value: the unit conversion (2) and the same-day mean (2).
_IDENTITY_OPERATIONS = 4
# An eGFR occasion: the conversion (2), the ratio (1), two powers (4), the age term's power (2) and age (1),
# four products (4) and the same-day mean (2).
_EGFR_OPERATIONS = 16


OMOP_TABLES = {
    "person": (
        "person_id BIGINT, gender_concept_id BIGINT, year_of_birth BIGINT, race_concept_id BIGINT, "
        "ethnicity_concept_id BIGINT, sex_at_birth_concept_id BIGINT"
    ),
    "observation_period": (
        "observation_period_id BIGINT, person_id BIGINT, observation_period_start_date DATE, "
        "observation_period_end_date DATE"
    ),
    "measurement": (
        "measurement_id BIGINT, person_id BIGINT, measurement_concept_id BIGINT, measurement_date DATE, "
        "measurement_type_concept_id BIGINT, operator_concept_id BIGINT, value_as_number DOUBLE, "
        "unit_concept_id BIGINT, visit_occurrence_id BIGINT, measurement_source_concept_id BIGINT, "
        "value_source_value VARCHAR"
    ),
    "observation": "observation_id BIGINT, person_id BIGINT, observation_concept_id BIGINT, observation_date DATE",
    "condition_occurrence": (
        "condition_occurrence_id BIGINT, person_id BIGINT, condition_concept_id BIGINT, condition_start_date DATE"
    ),
    "procedure_occurrence": (
        "procedure_occurrence_id BIGINT, person_id BIGINT, procedure_concept_id BIGINT, procedure_date DATE"
    ),
    "drug_exposure": "drug_exposure_id BIGINT, person_id BIGINT, drug_concept_id BIGINT, drug_exposure_start_date DATE",
    "visit_occurrence": (
        "visit_occurrence_id BIGINT, person_id BIGINT, visit_concept_id BIGINT, visit_start_date DATE, "
        "visit_end_date DATE"
    ),
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
    (3004501, "Glucose [Mass/volume] in Serum or Plasma", "Measurement", "LOINC", "Lab Test", "S", "2345-7"),
    (3028437, "Cholesterol in LDL [Mass/volume] in Serum or Plasma by Direct assay", "Measurement", "LOINC",
     "Lab Test", "S", "18262-6"),
    (3023599, "MCV [Entitic volume] by Automated count", "Measurement", "LOINC", "Lab Test", "S", "787-2"),
    (3004410, "Hemoglobin A1c/Hemoglobin.total in Blood", "Measurement", "LOINC", "Lab Test", "S", "4548-4"),
    (3036277, "Body height", "Measurement", "LOINC", "Clinical Observation", "S", "8302-2"),
    (3004249, "Systolic blood pressure", "Measurement", "LOINC", "Clinical Observation", "S", "8480-6"),
    (903133, "Height", "Measurement", "PPI", "Clinical Observation", None, "height"),
    (903118, "Computed systolic blood pressure, mean of 2nd and 3rd measures", "Measurement", "PPI",
     "Clinical Observation", None, "blood-pressure-systolic-mean"),
    (903109, "Systolic blood pressure 1st reading", "Measurement", "PPI", "Clinical Observation", None,
     "blood-pressure-systolic-1"),
    (8840, "milligram per deciliter", "Unit", "UCUM", "Unit", "S", "mg/dL"),
    (8749, "micromole per liter", "Unit", "UCUM", "Unit", "S", "umol/L"),
    (8753, "millimole per liter", "Unit", "UCUM", "Unit", "S", "mmol/L"),
    (8582, "centimeter", "Unit", "UCUM", "Unit", "S", "cm"),
    (9330, "inch (US)", "Unit", "UCUM", "Unit", "S", "[in_us]"),
    (8876, "millimeter mercury column", "Unit", "UCUM", "Unit", "S", "mm[Hg]"),
    (8583, "femtoliter", "Unit", "UCUM", "Unit", "S", "fL"),
    (8554, "percent", "Unit", "UCUM", "Unit", "S", "%"),
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
    (317510, "Leukemia", "Condition", "SNOMED", "Clinical Finding", "S", "93143009"),
    (4022139, "Administration of blood product", "Procedure", "SNOMED", "Procedure", "S", "116762002"),
    (2788717, "Transfusion of Nonautologous Red Blood Cells into Peripheral Vein, Percutaneous Approach",
     "Procedure", "ICD10PCS", "ICD10PCS", "S", "30233N1"),
    (2108119, "Transfusion, blood or blood components", "Procedure", "CPT4", "CPT4", "S", "36430"),
    (21601386, "ANTINEOPLASTIC AGENTS", "Drug", "ATC", "ATC 2nd", "C", "L01"),
    (1305058, "methotrexate", "Drug", "RxNorm", "Ingredient", "S", "6851"),
    (201820, "Diabetes mellitus", "Condition", "SNOMED", "Clinical Finding", "S", "73211009"),
    (201826, "Type 2 diabetes mellitus", "Condition", "SNOMED", "Clinical Finding", "S", "44054006"),
    (201254, "Type 1 diabetes mellitus", "Condition", "SNOMED", "Clinical Finding", "S", "46635009"),
    (4024659, "Gestational diabetes mellitus", "Condition", "SNOMED", "Clinical Finding", "S", "11687002"),
    (21600712, "DRUGS USED IN DIABETES", "Drug", "ATC", "ATC 2nd", "C", "A10"),
    (1503297, "metformin", "Drug", "RxNorm", "Ingredient", "S", "6809"),
    (21601853, "LIPID MODIFYING AGENTS", "Drug", "ATC", "ATC 2nd", "C", "C10"),
    (21601855, "HMG CoA reductase inhibitors", "Drug", "ATC", "ATC 4th", "C", "C10AA"),
    (1545958, "atorvastatin", "Drug", "RxNorm", "Ingredient", "S", "83367"),
    (320128, "Essential hypertension", "Condition", "SNOMED", "Clinical Finding", "S", "59621000"),
    (316866, "Hypertensive disorder", "Condition", "SNOMED", "Clinical Finding", "S", "38341003"),
    (21600381, "ANTIHYPERTENSIVES", "Drug", "ATC", "ATC 2nd", "C", "C02"),
    (21601664, "BETA BLOCKING AGENTS", "Drug", "ATC", "ATC 2nd", "C", "C07"),
    (1307046, "metoprolol", "Drug", "RxNorm", "Ingredient", "S", "6918"),
    (4001450, "Consent PII", "Observation", "PPI", "Module", None, "ConsentPII"),
    (4329041, "Pain", "Condition", "SNOMED", "Clinical Finding", "S", "22253000"),
    (45878463, "Female", "Meas Value", "LOINC", "Answer", "S", "LA3-6"),
    (45880669, "Male", "Meas Value", "LOINC", "Answer", "S", "LA2-8"),
    (903096, "PMI: Skip", "Observation", "PPI", "Answer", None, "PMI_Skip"),
)
# (ancestor, descendant, min, max); every standard or classification concept is its own ancestor.
CONCEPT_ANCESTORS = (
    (9201, 9201, 0, 0), (9201, 8717, 1, 1), (9202, 9202, 0, 0), (9203, 9203, 0, 0), (262, 262, 0, 0),
    (4088927, 4088927, 0, 0), (4088927, 4299535, 2, 2), (4088927, 4243056, 2, 2),
    (4299535, 4299535, 0, 0), (4243056, 4243056, 0, 0), (4047564, 4047564, 0, 0),
    (317510, 317510, 0, 0), (4022139, 4022139, 0, 0), (4022139, 2788717, 1, 1), (2788717, 2788717, 0, 0),
    (2108119, 2108119, 0, 0), (21601386, 21601386, 0, 0), (21601386, 1305058, 3, 3), (1305058, 1305058, 0, 0),
    (201820, 201820, 0, 0), (201820, 201826, 1, 1), (201820, 201254, 1, 1), (201820, 4024659, 1, 1),
    (201826, 201826, 0, 0), (201254, 201254, 0, 0), (4024659, 4024659, 0, 0),
    (21600712, 21600712, 0, 0), (21600712, 1503297, 3, 3), (1503297, 1503297, 0, 0),
    (21601853, 21601853, 0, 0), (21601853, 21601855, 2, 2), (21601855, 21601855, 0, 0),
    (21601853, 1545958, 3, 3), (21601855, 1545958, 1, 1), (1545958, 1545958, 0, 0),
    (320128, 320128, 0, 0), (316866, 316866, 0, 0), (316866, 320128, 1, 1),
    (21600381, 21600381, 0, 0), (21601664, 21601664, 0, 0), (21601664, 1307046, 2, 2), (1307046, 1307046, 0, 0),
    (4329041, 4329041, 0, 0),
)
# Concepts of the disease panel that the tests below need, under synthetic ids
# (91000001-) that no real OMOP concept uses.
SYNTHETIC_CONCEPTS = (
    (91000001, "Sulfonylureas", "Drug", "ATC", "ATC 4th", "C", "A10BB"),
    (91000002, "glipizide", "Drug", "RxNorm", "Ingredient", "S", "4821"),
    (91000003, "Atrial fibrillation", "Condition", "SNOMED", "Clinical Finding", "S", "49436004"),
    (91000004, "Atrial flutter", "Condition", "SNOMED", "Clinical Finding", "S", "5370000"),
    (91000005, "Cataract", "Condition", "SNOMED", "Clinical Finding", "S", "193570009"),
    (91000006, "Extraction of cataract", "Procedure", "SNOMED", "Procedure", "S", "54885007"),
    (91000007, "Congenital cataract", "Condition", "SNOMED", "Clinical Finding", "S", "79410001"),
    (91000008, "levothyroxine sodium", "Drug", "ATC", "ATC 5th", "C", "H03AA01"),
    (91000009, "levothyroxine", "Drug", "RxNorm", "Ingredient", "S", "10582"),
    (91000010, "Hypothyroidism", "Condition", "SNOMED", "Clinical Finding", "S", "40930008"),
    (91000011, "THYROID PREPARATIONS", "Drug", "ATC", "ATC 3rd", "C", "H03A"),
    (91000012, "Chronic kidney disease stage 3", "Condition", "SNOMED", "Clinical Finding", "S", "433144002"),
    (91000013, "Chronic kidney disease", "Condition", "SNOMED", "Clinical Finding", "S", "709044004"),
)
SYNTHETIC_CONCEPT_ANCESTORS = (
    *((concept[0], concept[0], 0, 0) for concept in SYNTHETIC_CONCEPTS),
    (21600712, 91000001, 2, 2), (21600712, 91000002, 3, 3), (91000001, 91000002, 1, 1),
    (91000005, 91000007, 1, 1),
    (91000011, 91000008, 1, 1), (91000011, 91000009, 2, 2), (91000008, 91000009, 1, 1),
    (91000013, 91000012, 1, 1),
)
FEMALE = 45878463
MALE = 45880669
PAIN = 4329041
LAB_RESULT_TYPE = 32856
SELF_REPORT_TYPE = 32865


def _duckdb_query(sql: str, parameters: dict[str, tuple[str, object]]) -> str:
    def literal(parameter_type: str, value: object) -> exp.Expression:
        if isinstance(value, list):
            return exp.Array(expressions=[literal(parameter_type, element) for element in value])
        if value is None:
            return exp.cast(exp.Null(), "DOUBLE" if parameter_type == "FLOAT64" else "BIGINT")
        if parameter_type == "BOOL":
            return exp.Boolean(this=value)
        if parameter_type == "STRING":
            return exp.Literal.string(str(value))
        if parameter_type == "INT64":
            return exp.cast(exp.Literal.number(str(int(value))), "BIGINT")
        return exp.cast(exp.Literal.number(repr(float(value))), "DOUBLE")

    def substitute(node: exp.Expression) -> exp.Expression:
        if isinstance(node, exp.Parameter):
            parameter_type, value = parameters[node.this.name]
            return literal(parameter_type, value)
        return node

    return sqlglot.parse_one(sql, read="bigquery").transform(substitute).sql(dialect="duckdb")


def _date(text: str) -> datetime.date:
    return datetime.date.fromisoformat(text)


class _Cdr:
    """In-memory OMOP CDR with the shared vocabulary preloaded."""

    def __init__(self) -> None:
        self.connection = duckdb.connect()
        self.connection.execute("ATTACH ':memory:' AS aou_workspace")
        self.connection.execute("CREATE SCHEMA aou_workspace.cdr_dataset")
        for table, columns in OMOP_TABLES.items():
            self.connection.execute(f"CREATE TABLE {DATASET}.{table} ({columns})")
        self.insert("concept", CONCEPTS + SYNTHETIC_CONCEPTS)
        self.insert("concept_ancestor", CONCEPT_ANCESTORS + SYNTHETIC_CONCEPT_ANCESTORS)
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

    def observation_period(self, person_id: int, start: str, end: str) -> None:
        self.insert("observation_period", [(self.next_id(), person_id, _date(start), _date(end))])

    def measurement(
        self,
        person_id: int,
        date: str,
        value: float | None,
        *,
        concept_id: int = 3016723,
        source_concept_id: int | None = None,
        unit_concept_id: int | None = 8840,
        operator_concept_id: int | None = None,
        type_concept_id: int = LAB_RESULT_TYPE,
        value_source_value: str | None = None,
    ) -> None:
        self.insert(
            "measurement",
            [(
                self.next_id(), person_id, concept_id, _date(date), type_concept_id, operator_concept_id, value,
                unit_concept_id, None, source_concept_id, value_source_value,
            )],
        )

    def visit(self, person_id: int, concept_id: int, start: str, end: str) -> None:
        self.insert("visit_occurrence", [(self.next_id(), person_id, concept_id, _date(start), _date(end))])

    def condition(self, person_id: int, concept_id: int, date: str) -> None:
        self.insert("condition_occurrence", [(self.next_id(), person_id, concept_id, _date(date))])

    def procedure(self, person_id: int, concept_id: int, date: str) -> None:
        self.insert("procedure_occurrence", [(self.next_id(), person_id, concept_id, _date(date))])

    def observation(self, person_id: int, concept_id: int, date: str) -> None:
        self.insert("observation", [(self.next_id(), person_id, concept_id, _date(date))])

    def drug(self, person_id: int, concept_id: int, date: str) -> None:
        self.insert("drug_exposure", [(self.next_id(), person_id, concept_id, _date(date))])

    def _run(self, sql: str, parameters: dict[str, tuple[str, object]]) -> dict[str, dict[str, object]]:
        cursor = self.connection.execute(_duckdb_query(sql, parameters))
        columns = [description[0] for description in cursor.description]
        return {row[0]: dict(zip(columns, row, strict=True)) for row in cursor.fetchall()}

    def _all_rows(self, sql: str, parameters: dict[str, tuple[str, object]]) -> list[dict[str, object]]:
        cursor = self.connection.execute(_duckdb_query(sql, parameters))
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]

    def measurement_rows(self, definition: MeasurementDefinition) -> list[dict[str, object]]:
        return self._all_rows(build_all_of_us_measurement_sql(), build_all_of_us_measurement_query_parameters(definition))

    def measurement_people(self, definition: MeasurementDefinition) -> dict[str, PersonOccasions]:
        return {person.person_id: person for person in person_occasions(self.measurement_rows(definition))}

    def lab_evidence(self, criterion: LabCriterion) -> dict[str, dict[str, object]]:
        rows = self._all_rows(build_all_of_us_measurement_sql(), build_all_of_us_lab_criterion_query_parameters(criterion))
        return lab_criterion_evidence(criterion, person_occasions(rows))

    def disease_rows(self, definition: DiseaseDefinition) -> dict[str, dict[str, object]]:
        return self._run(build_all_of_us_disease_sql(definition), build_all_of_us_disease_query_parameters(definition))

    def census_rows(self) -> list[dict[str, object]]:
        cursor = self.connection.execute(
            _duckdb_query(build_all_of_us_measurement_census_sql(), build_all_of_us_measurement_census_query_parameters())
        )
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


@pytest.fixture
def cdr(monkeypatch: pytest.MonkeyPatch) -> _Cdr:
    monkeypatch.setenv("WORKSPACE_CDR", DATASET)
    return _Cdr()


def _age(date: str, year_of_birth: int) -> float:
    return (_date(date) - datetime.date(year_of_birth, 7, 1)).days / 365.25


def _ckd_epi_2021(creatinine_mg_dl: float, age: float, female: bool) -> float:
    kappa, alpha, sex_factor = (0.7, -0.241, 1.012) if female else (0.9, -0.302, 1.0)
    ratio = creatinine_mg_dl / kappa
    return 142.0 * min(ratio, 1.0) ** alpha * max(ratio, 1.0) ** -1.2 * 0.9938**age * sex_factor


def _excluded(person: PersonOccasions) -> dict[str, int]:
    return {reason: count for reason, count in person.excluded_row_counts.items() if count}


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------


def test_row_rules_units_and_same_day_collapse(cdr: _Cdr):
    egfr = resolve_measurement_definition("egfr_ckd_epi_2021")
    cdr.person(1, 1960, FEMALE)
    cdr.measurement(1, "2015-03-01", 0.8)
    cdr.measurement(1, "2015-03-01", 0.9, unit_concept_id=None)
    cdr.measurement(1, "2016-05-10", 1.0, operator_concept_id=4172703, source_concept_id=3016723, concept_id=0)
    # 25 mg/dL is a gross value, which a trait keeps for its noise density to weigh.
    cdr.measurement(1, "2017-01-01", 25.0)
    cdr.measurement(1, "2017-01-15", 0.0)
    cdr.measurement(1, "2017-02-01", 0.3, operator_concept_id=4171756)
    cdr.measurement(1, "2017-03-01", 0.2, value_source_value="<0.2")
    cdr.measurement(1, "2017-04-01", 0.07, unit_concept_id=8753)
    cdr.measurement(1, "2017-05-01", 0.9, unit_concept_id=8554)
    cdr.measurement(1, "2017-08-01", 0.9, type_concept_id=SELF_REPORT_TYPE)
    cdr.measurement(1, "2018-01-01", None)
    cdr.measurement(1, "2018-02-01", 95.0, concept_id=3004501)
    cdr.person(2, 2000, MALE)
    cdr.measurement(2, "2015-01-01", 0.6)
    cdr.measurement(2, "2020-06-01", 1.1)
    cdr.measurement(2, "2021-06-01", 0.9)
    cdr.person(3, 1970, 903096)
    cdr.measurement(3, "2020-01-01", 1.0)

    people = cdr.measurement_people(egfr)
    assert set(people) == {"1", "2", "3"}

    first = people["1"]
    # A row whose standard concept is unmapped (0) still counts through its source LOINC.
    assert first.row_count == 10
    assert _excluded(first) == {"censored": 2, "unrecognized_unit": 2, "nonpositive": 1, "self_reported": 1}
    assert sorted(first.unrecognized_unit_labels) == ["millimole per liter", "percent"]
    # 2015-03-01 has two rows (mg/dL and no unit): one occasion, the mean of their eGFRs on the linear scale.
    ages = [_age(date, 1960) for date in ("2015-03-01", "2016-05-10", "2017-01-01")]
    expected = [
        np.mean([_ckd_epi_2021(0.8, ages[0], female=True), _ckd_epi_2021(0.9, ages[0], female=True)]),
        _ckd_epi_2021(1.0, ages[1], female=True),
        _ckd_epi_2021(25.0, ages[2], female=True),
    ]
    assert first.dates == (_date("2015-03-01"), _date("2016-05-10"), _date("2017-01-01"))
    for value, target in zip(first.values, expected, strict=True):
        assert value == within_rounding(target, _EGFR_OPERATIONS)
    for age, target in zip(first.ages, ages, strict=True):
        assert age == within_rounding(target, 1)
    assert not first.treated.any()
    assert first.sex_at_birth_name == "female"

    second = people["2"]
    assert _excluded(second) == {"under_minimum_age": 1}
    male_values = [_ckd_epi_2021(value, _age(date, 2000), female=False) for date, value in (("2020-06-01", 1.1), ("2021-06-01", 0.9))]
    for value, target in zip(second.values, male_values, strict=True):
        assert value == within_rounding(target, _EGFR_OPERATIONS)

    # eGFR needs sex at birth; a skipped answer is not guessed.
    third = people["3"]
    assert _excluded(third) == {"sex_unknown": 1}
    assert third.values.size == 0


def test_acute_care_windows_span_thirty_days_around_each_stay(cdr: _Cdr):
    mcv = resolve_measurement_definition("mcv")
    cdr.person(1, 1960, FEMALE)
    cdr.visit(1, 8717, "2017-06-01", "2017-06-05")
    cdr.visit(1, 262, "2019-01-10", "2019-01-10")
    cdr.visit(1, 9202, "2020-01-10", "2020-01-10")
    for date, value in (
        ("2017-05-01", 80.0),  # 31 days before admission: kept
        ("2017-05-02", 81.0),  # 30 days before: excluded
        ("2017-06-03", 82.0),  # during the stay: excluded
        ("2017-07-05", 83.0),  # 30 days after discharge: excluded
        ("2017-07-06", 84.0),  # 31 days after: kept
        ("2019-01-20", 85.0),  # 10 days after an ER-and-inpatient visit: excluded
        ("2020-01-10", 86.0),  # at an outpatient visit: kept
    ):
        cdr.measurement(1, date, value, concept_id=3023599, unit_concept_id=8583)
    person = cdr.measurement_people(mcv)["1"]
    assert _excluded(person) == {"acute_care": 4}
    np.testing.assert_array_equal(person.values, [80.0, 84.0, 86.0])


def test_pregnancy_windows_come_from_conditions_and_antenatal_observations(cdr: _Cdr):
    mcv = resolve_measurement_definition("mcv")
    cdr.person(1, 1985, FEMALE)
    cdr.condition(1, 4299535, "2019-06-01")
    cdr.condition(1, 4243056, "2012-01-01")
    cdr.observation(1, 4047564, "2023-01-01")
    for date, value in (
        ("2018-08-12", 80.0),  # 293 days before the pregnancy code: excluded
        ("2018-08-10", 81.0),  # 295 days before: kept
        ("2020-06-13", 82.0),  # 378 days after: excluded
        ("2020-06-14", 83.0),  # 379 days after: kept
        ("2012-02-01", 84.0),  # after a "Not pregnant" finding: kept
        ("2023-02-01", 85.0),  # after routine antenatal care (observation): excluded
    ):
        cdr.measurement(1, date, value, concept_id=3023599, unit_concept_id=8583)
    person = cdr.measurement_people(mcv)["1"]
    assert _excluded(person) == {"pregnancy": 3}
    np.testing.assert_array_equal(np.sort(person.values), [81.0, 83.0, 84.0])


def test_clinical_windows_cover_conditions_procedures_and_drugs(cdr: _Cdr):
    mcv = resolve_measurement_definition("mcv")
    cdr.person(1, 1960, MALE)
    cdr.procedure(1, 2788717, "2016-01-01")  # ICD10PCS transfusion, a descendant of the SNOMED root
    cdr.procedure(1, 2108119, "2017-01-01")  # CPT 36430
    cdr.drug(1, 1305058, "2018-01-01")  # methotrexate (L01)
    cdr.condition(1, 317510, "2020-01-01")  # leukemia: excluded from 180 days before, forever after
    for date, value in (
        ("2015-12-31", 90.0),  # kept
        ("2016-04-30", 91.0),  # 120 days after the transfusion: excluded
        ("2016-05-01", 92.0),  # 121 days after: kept
        ("2017-03-01", 93.0),  # after the CPT transfusion: excluded
        ("2018-03-31", 94.0),  # 89 days after methotrexate: excluded
        ("2018-04-02", 95.0),  # 91 days after: kept
        ("2019-07-06", 96.0),  # 179 days before leukemia: excluded
        ("2019-07-01", 97.0),  # 184 days before: kept
        ("2024-01-01", 98.0),  # after leukemia: excluded
    ):
        cdr.measurement(1, date, value, concept_id=3023599, unit_concept_id=8583)
    person = cdr.measurement_people(mcv)["1"]
    assert _excluded(person) == {"clinical_exclusion": 5}
    np.testing.assert_array_equal(np.sort(person.values), [90.0, 92.0, 95.0, 97.0])


def test_diabetes_removes_every_hba1c_value_of_the_person(cdr: _Cdr):
    hba1c = resolve_measurement_definition("hba1c")
    cdr.person(1, 1960, FEMALE)
    cdr.condition(1, 201826, "2021-01-01")
    cdr.person(2, 1960, FEMALE)
    cdr.drug(2, 1503297, "2021-01-01")
    cdr.person(3, 1960, FEMALE)
    for person_id in (1, 2, 3):
        cdr.measurement(person_id, "2015-01-01", 5.6, concept_id=3004410, unit_concept_id=8554)
        cdr.measurement(person_id, "2016-01-01", 5.8, concept_id=3004410, unit_concept_id=None)
    people = cdr.measurement_people(hba1c)
    assert _excluded(people["1"]) == {"clinical_exclusion": 2}
    assert _excluded(people["2"]) == {"clinical_exclusion": 2}
    assert _excluded(people["3"]) == {}
    np.testing.assert_array_equal(people["3"].values, [5.6, 5.8])


def test_physical_measurements_use_protocol_means_and_convert_inches(cdr: _Cdr):
    height = resolve_measurement_definition("height")
    systolic = resolve_measurement_definition("sbp")
    cdr.person(1, 1960, MALE)
    # Enrollment height (PPI source concept, cm) and EHR heights in inches and with no unit.
    cdr.measurement(1, "2018-05-01", 180.0, concept_id=903133, source_concept_id=903133, unit_concept_id=8582)
    cdr.measurement(1, "2019-05-01", 71.0, concept_id=3036277, unit_concept_id=9330)
    cdr.measurement(1, "2020-05-01", 70.0, concept_id=3036277, unit_concept_id=0)
    # Enrollment blood pressure: the protocol mean counts, the individual readings do not.
    cdr.measurement(1, "2018-05-01", 131.0, concept_id=903118, source_concept_id=903118, unit_concept_id=8876)
    cdr.measurement(1, "2018-05-01", 150.0, concept_id=3004249, source_concept_id=903109, unit_concept_id=8876)
    cdr.measurement(1, "2019-05-01", 125.0, concept_id=3004249, unit_concept_id=None)
    heights = cdr.measurement_people(height)["1"]
    for value, target in zip(heights.values, [180.0, 71.0 * 2.54000508, 70.0 * 2.54], strict=True):
        assert value == within_rounding(target, _IDENTITY_OPERATIONS)
    pressures = cdr.measurement_people(systolic)["1"]
    assert pressures.row_count == 2
    np.testing.assert_array_equal(pressures.values, [131.0, 125.0])


def test_occasions_on_or_after_the_first_exposure_are_treated(cdr: _Cdr):
    ldl = resolve_measurement_definition("ldl_cholesterol")
    cdr.person(1, 1950, MALE)
    cdr.drug(1, 1545958, "2018-01-01")
    cdr.drug(1, 1545958, "2019-01-01")
    cdr.measurement(1, "2017-06-01", 160.0, concept_id=3028437)
    cdr.measurement(1, "2018-01-01", 110.0, concept_id=3028437)
    cdr.measurement(1, "2018-06-01", 90.0, concept_id=3028437)
    cdr.person(2, 1955, FEMALE)
    cdr.measurement(2, "2017-06-01", 104.0, concept_id=3028437, unit_concept_id=None)
    people = cdr.measurement_people(ldl)
    treated_person = people["1"]
    np.testing.assert_array_equal(treated_person.values, [160.0, 110.0, 90.0])
    np.testing.assert_array_equal(treated_person.treated, [False, True, True])
    np.testing.assert_array_equal(people["2"].values, [104.0])
    assert not people["2"].treated.any()


def _mcv_cohort(cdr: _Cdr, generator: np.random.Generator, persons: int) -> dict[str, float]:
    effects = {}
    for person_id in range(1, persons + 1):
        cdr.person(person_id, int(generator.integers(1940, 1990)), FEMALE if person_id % 2 else MALE)
        effects[str(person_id)] = generator.normal(0.0, 5.0)
        for occasion in range(int(generator.integers(1, 6))):
            cdr.measurement(
                person_id,
                f"{2010 + occasion}-0{1 + occasion}-15",
                90.0 + effects[str(person_id)] + generator.normal(0.0, 3.0),
                concept_id=3023599,
                unit_concept_id=8583,
            )
    return effects


def test_query_rows_feed_the_target_builder_end_to_end(cdr: _Cdr):
    generator = np.random.default_rng(0)
    mcv = resolve_measurement_definition("mean_corpuscular_volume")
    effects = _mcv_cohort(cdr, generator, 300)
    training_rows, _columns, summary = build_all_of_us_measurement_targets(mcv, cdr.measurement_rows(mcv), WORKING_BYTES)
    assert summary["n_persons"] == 300
    targets = np.array([row["target"] for row in training_rows])
    true_effects = np.array([effects[row["person_id"]] for row in training_rows])
    # The target correlates with the true effect by about the root mean reliability.
    reliability = np.array([row["target_reliability"] for row in training_rows])
    expected_correlation = float(np.sqrt(reliability.mean()))
    correlation_standard_error = (1.0 - expected_correlation**2) / np.sqrt(len(training_rows))
    assert np.corrcoef(targets, true_effects)[0, 1] > expected_correlation - sampling_bound(correlation_standard_error)


def test_gross_errors_reach_the_model_and_are_downweighted_by_its_learned_density(cdr: _Cdr):
    # Typos of x10 in MCV, and creatinine in umol/L under the unit label whose rows carry mg/dL values: the
    # trait keeps them (no range) and the learned noise density keeps them from moving a person's level.
    generator = np.random.default_rng(5)
    mcv = resolve_measurement_definition("mean_corpuscular_volume")
    clean = _Cdr()
    clean.connection.execute(f"SET search_path = '{DATASET}'")
    effects = _mcv_cohort(cdr, generator, 250)
    _mcv_cohort(clean, np.random.default_rng(5), 250)
    typo_people = [str(person_id) for person_id in range(1, 251, 25)]
    for position, person_id in enumerate(typo_people):
        cdr.measurement(int(person_id), f"2019-0{1 + position % 9}-20", 10.0 * (90.0 + effects[person_id]), concept_id=3023599, unit_concept_id=8583)
    contaminated_rows, _columns, contaminated_summary = build_all_of_us_measurement_targets(mcv, cdr.measurement_rows(mcv), WORKING_BYTES)
    clean_rows, _columns, clean_summary = build_all_of_us_measurement_targets(mcv, clean.measurement_rows(mcv), WORKING_BYTES)
    assert contaminated_summary["n_occasions"] == clean_summary["n_occasions"] + len(typo_people)
    contaminated = {row["person_id"]: row for row in contaminated_rows}
    reference = {row["person_id"]: row for row in clean_rows}
    level_variance = contaminated_summary["level_variance"]
    for person_id in typo_people:
        # One posterior sd of the person's level, on the contaminated fit's scale.
        sd = np.sqrt(level_variance * (1.0 - contaminated[person_id]["target_reliability"]))
        if contaminated_summary["box_cox_exponent"] == clean_summary["box_cox_exponent"]:
            assert abs(contaminated[person_id]["target"] - reference[person_id]["target"]) < sd
        # Whatever the transform, the typo leaves the person on the same side of the cohort's centre.
        assert np.sign(contaminated[person_id]["target"]) == np.sign(reference[person_id]["target"])


# ---------------------------------------------------------------------------
# Diseases
# ---------------------------------------------------------------------------


def _observed_with_early_ehr(cdr: _Cdr, person_id: int, year_of_birth: int, sex: int) -> None:
    """A person observed 2010-2024 with the five early condition dates the EHR-depth floor needs."""
    cdr.person(person_id, year_of_birth, sex)
    cdr.observation_period(person_id, "2010-01-01", "2024-01-01")
    for month in range(2, 7):
        cdr.condition(person_id, PAIN, f"2010-0{month}-01")


def test_type2_diabetes_cases_need_two_dates_or_non_metformin_support(cdr: _Cdr):
    type2_diabetes = resolve_disease_definition("t2d")
    for person_id in range(1, 8):
        _observed_with_early_ehr(cdr, person_id, 1950 + person_id, FEMALE if person_id % 2 else MALE)
    cdr.condition(1, 201826, "2015-01-01")
    cdr.condition(1, 201826, "2015-01-01")
    cdr.condition(1, 201826, "2016-01-01")
    cdr.condition(2, 201826, "2016-01-01")
    cdr.drug(2, 91000002, "2016-02-01")  # one code + a sulfonylurea: a case
    cdr.condition(3, 201826, "2017-01-01")
    cdr.drug(3, 1503297, "2017-02-01")  # one code + metformin only: neither
    cdr.condition(4, 201820, "2018-01-01")  # unspecified diabetes: not a control
    cdr.drug(5, 1503297, "2019-01-01")  # metformin without a code: not a control
    cdr.condition(6, 201826, "2015-01-01")
    cdr.condition(6, 201254, "2015-06-01")  # type 1 diabetes: ambiguous

    rows = cdr.disease_rows(type2_diabetes)
    assert rows["1"]["phenotype_occurrence_count"] == 2
    assert rows["1"]["age_at_first_condition"] == within_rounding(_age("2015-01-01", 1951), 1)
    assert rows["1"]["year_of_birth"] == 1951
    assert rows["2"]["case_medication_dates"] == 1
    assert rows["3"]["case_medication_dates"] == 0
    assert rows["3"]["has_control_exclusion_medication"] is True
    assert rows["4"]["has_control_exclusion_code"] is True
    assert rows["5"]["has_control_exclusion_medication"] is True
    assert rows["7"]["has_control_exclusion_medication"] is False

    training_rows, _columns, counts = _prepare_training_rows(type2_diabetes, list(rows.values()), [{}])
    assert {row["person_id"]: row["target"] for row in training_rows} == {"1": 1, "2": 1, "7": 0}
    assert counts["n_excluded_one_date"] == 1
    assert counts["n_excluded_control_exclusion"] == 2
    assert counts["n_excluded_ambiguous_code"] == 1


def test_diagnosis_roots_distinct_drug_dates_and_case_procedures(cdr: _Cdr):
    for person_id in range(1, 6):
        _observed_with_early_ehr(cdr, person_id, 1950, FEMALE)
    cdr.condition(1, 91000003, "2015-01-01")
    cdr.condition(1, 91000004, "2016-01-01")  # atrial fibrillation then flutter: two dates of the disease
    cdr.condition(2, 91000010, "2015-01-01")
    cdr.drug(2, 91000009, "2015-02-01")
    cdr.drug(2, 91000009, "2015-02-01")
    cdr.drug(2, 91000009, "2015-05-01")  # levothyroxine on two distinct dates
    cdr.procedure(3, 91000006, "2019-04-01")  # cataract extraction alone
    cdr.condition(4, 91000005, "2018-01-01")
    cdr.condition(4, 91000005, "2019-01-01")
    cdr.condition(4, 91000007, "2019-01-01")  # congenital cataract: ambiguous

    atrial_fibrillation = cdr.disease_rows(resolve_disease_definition("afib"))
    assert atrial_fibrillation["1"]["phenotype_occurrence_count"] == 2
    hypothyroidism = cdr.disease_rows(resolve_disease_definition("hypothyroidism"))
    assert hypothyroidism["2"]["case_medication_dates"] == 2
    assert hypothyroidism["2"]["has_control_exclusion_medication"] is True
    cataract_definition = resolve_disease_definition("cataract")
    cataract = cdr.disease_rows(cataract_definition)
    assert cataract["3"]["case_procedure_dates"] == 1
    assert cataract["3"]["age_at_first_case_procedure"] == within_rounding(_age("2019-04-01", 1950), 1)
    assert cataract["4"]["has_ambiguous_code"] is True

    training_rows, _columns, counts = _prepare_training_rows(cataract_definition, list(cataract.values()), [])
    assert {row["person_id"]: row["target"] for row in training_rows} == {"1": 0, "2": 0, "3": 1, "5": 0}
    assert counts["n_cases_by_procedure"] == 1 and counts["n_excluded_ambiguous_code"] == 1


def test_egfr_lab_criterion_counts_retained_occasions_below_sixty(cdr: _Cdr):
    kidney = resolve_disease_definition("ckd")
    egfr_criterion, albuminuria_criterion = kidney.lab_criteria
    for person_id in (1, 2, 3):
        _observed_with_early_ehr(cdr, person_id, 1960, FEMALE)
    # Person 1: eGFR about 44 and 41 (creatinine 1.4 and 1.5 mg/dL at 54), 151 days
    # apart, then a normal value: a case by the lab criterion.
    cdr.measurement(1, "2015-01-10", 1.4)
    cdr.measurement(1, "2015-06-10", 1.5)
    cdr.measurement(1, "2016-01-10", 0.7)
    # Person 2: one low value outside the hospital and one during an inpatient
    # stay (acute kidney injury), which never counts.
    cdr.measurement(2, "2015-01-10", 1.6)
    cdr.visit(2, 9201, "2016-03-01", "2016-03-05")
    cdr.measurement(2, "2016-03-02", 3.0)
    # Person 3: staged chronic kidney disease on two dates.
    cdr.condition(3, 91000012, "2017-01-01")
    cdr.condition(3, 91000012, "2018-01-01")

    # Person 4: creatinine 25 mg/dL twice, outside the criterion's plausible range: no evidence, although the
    # eGFR trait keeps such readings for its noise density.
    _observed_with_early_ehr(cdr, 4, 1960, FEMALE)
    cdr.measurement(4, "2015-01-10", 25.0)
    cdr.measurement(4, "2015-06-10", 25.0)
    egfr_evidence = cdr.lab_evidence(egfr_criterion)
    assert egfr_evidence["1"] == {
        "qualifying_occasion_count": 2, "first_qualifying_date": _date("2015-01-10"), "last_qualifying_date": _date("2015-06-10"),
    }
    assert egfr_evidence["2"]["qualifying_occasion_count"] == 1
    assert "3" not in egfr_evidence and "4" not in egfr_evidence
    assert cdr.measurement_people(resolve_measurement_definition("egfr"))["4"].values.shape == (2,)

    rows = list(cdr.disease_rows(kidney).values())
    training_rows, _columns, counts = _prepare_training_rows(kidney, rows, [egfr_evidence, {}])
    assert {row["person_id"]: row["target"] for row in training_rows} == {"1": 1, "3": 1, "4": 0}
    assert counts["n_cases_by_lab"] == 1 and counts["n_excluded_lab_evidence"] == 1
    assert albuminuria_criterion.measurement == "urine_albumin_creatinine_ratio"


# ---------------------------------------------------------------------------
# Census
# ---------------------------------------------------------------------------


def test_census_counts_participants_per_trait_concept_and_unit(cdr: _Cdr):
    for person_id in (1, 2, 3):
        cdr.person(person_id, 1960, FEMALE)
        cdr.measurement(person_id, "2018-05-01", 170.0, concept_id=903133, source_concept_id=903133, unit_concept_id=8582)
        cdr.measurement(person_id, "2019-05-01", 67.0, concept_id=3036277, unit_concept_id=9330)
        cdr.measurement(person_id, "2020-05-01", 67.0, concept_id=3036277, unit_concept_id=9330)
    cdr.measurement(1, "2021-05-01", 66.0, concept_id=0, source_concept_id=3036277, unit_concept_id=None)
    cdr.measurement(2, "2021-05-01", 90.0, concept_id=3023599, unit_concept_id=8583)
    cells = {
        (row["trait_name"], row["concept_code"], row["matched_on_standard_concept"], row["unit_label"]):
            (row["participant_count"], row["row_count"])
        for row in cdr.census_rows()
    }
    assert cells == {
        ("height", "903133", True, "centimeter"): (3, 3),
        ("height", "8302-2", True, "inch (us)"): (3, 6),
        ("height", "8302-2", False, "no unit"): (1, 1),
        ("mean_corpuscular_volume", "787-2", True, "femtoliter"): (1, 1),
    }

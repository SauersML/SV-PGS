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
from tests.oracles import fit_at_exponent
from tests.phenotype_bounds import sampling_bound, within_rounding

WORKING_BYTES = 1 << 28


def _identity_transform_fit(monkeypatch) -> None:
    """The occasion model at the identity transform, for tests of the SQL feeding it (the exponent's search is
    tested in tests/test_all_of_us_phenotypes.py)."""
    monkeypatch.setattr("sv_pgs.all_of_us.fit_occasion_model", lambda occasions, working_bytes: fit_at_exponent(occasions, 1.0, working_bytes))

DATASET = "aou_workspace.cdr_dataset"


# A plain trait's occasion value: the unit conversion (2) and the same-day mean (2).
_IDENTITY_OPERATIONS = 4
# An eGFR reading of given creatinine and age: the ratio (1), two powers (4), the age term's power (2) and four
# products (4).
_EGFR_OPERATIONS = 11


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
    # Hypertension.
    (91000014, "Hypertensive heart disease", "Condition", "SNOMED", "Clinical Finding", "S", "64715009"),
    (91000015, "Maternal hypertension", "Condition", "SNOMED", "Clinical Finding", "S", "288250001"),
    (91000016, "Pre-eclampsia", "Condition", "SNOMED", "Clinical Finding", "S", "398254007"),
    (91000017, "Secondary hypertension", "Condition", "SNOMED", "Clinical Finding", "S", "31992008"),
    (91000018, "Elevated blood-pressure reading without diagnosis of hypertension", "Condition", "SNOMED",
     "Clinical Finding", "S", "371622005"),
    (91000019, "AGENTS ACTING ON THE RENIN-ANGIOTENSIN SYSTEM", "Drug", "ATC", "ATC 2nd", "C", "C09"),
    (91000020, "lisinopril", "Drug", "RxNorm", "Ingredient", "S", "29046"),
    # Coronary artery disease.
    (91000021, "Ischemic heart disease", "Condition", "SNOMED", "Clinical Finding", "S", "414545008"),
    (91000022, "Myocardial infarction", "Condition", "SNOMED", "Clinical Finding", "S", "22298006"),
    (91000023, "Myocardial infarction due to demand ischemia", "Condition", "SNOMED", "Clinical Finding", "S",
     "16837681000119104"),
    (91000024, "Disorder of coronary artery", "Condition", "SNOMED", "Clinical Finding", "S", "414024009"),
    (91000025, "Coronary arteriosclerosis", "Condition", "SNOMED", "Clinical Finding", "S", "53741008"),
    (91000026, "Coronary artery stenosis", "Condition", "SNOMED", "Clinical Finding", "S", "233970002"),
    (91000027, "Percutaneous coronary intervention", "Procedure", "SNOMED", "Procedure", "S", "415070008"),
    (91000028, "Organic nitrates", "Drug", "ATC", "ATC 4th", "C", "C01DA"),
    (91000029, "isosorbide mononitrate", "Drug", "RxNorm", "Ingredient", "S", "6058"),
    # Asthma.
    (91000030, "Asthma", "Condition", "SNOMED", "Clinical Finding", "S", "195967001"),
    (91000031, "Allergic asthma", "Condition", "SNOMED", "Clinical Finding", "S", "389145006"),
    (91000032, "Chronic obstructive pulmonary disease", "Condition", "SNOMED", "Clinical Finding", "S", "13645005"),
    (91000033, "Asthma-chronic obstructive pulmonary disease overlap syndrome", "Condition", "SNOMED",
     "Clinical Finding", "S", "10692761000119107"),
    (91000034, "DRUGS FOR OBSTRUCTIVE AIRWAY DISEASES", "Drug", "ATC", "ATC 2nd", "C", "R03"),
    (91000035, "Glucocorticoids", "Drug", "ATC", "ATC 4th", "C", "R03BA"),
    (91000036, "fluticasone", "Drug", "RxNorm", "Ingredient", "S", "41126"),
    (91000037, "Selective beta-2-adrenoreceptor agonists", "Drug", "ATC", "ATC 4th", "C", "R03AC"),
    (91000038, "albuterol", "Drug", "RxNorm", "Ingredient", "S", "435"),
    # Hyperlipidemia.
    (91000039, "Hyperlipidemia", "Condition", "SNOMED", "Clinical Finding", "S", "55822004"),
    (91000040, "Mixed hyperlipidemia", "Condition", "SNOMED", "Clinical Finding", "S", "267434003"),
    (91000041, "Hypercholesterolemia", "Condition", "SNOMED", "Clinical Finding", "S", "13644009"),
    (91000042, "Hyperalphalipoproteinemia", "Condition", "SNOMED", "Clinical Finding", "S", "238080004"),
    (91000043, "Dyslipidemia", "Condition", "SNOMED", "Clinical Finding", "S", "370992007"),
    (91000044, "Fibrates", "Drug", "ATC", "ATC 4th", "C", "C10AB"),
    (91000045, "fenofibrate", "Drug", "RxNorm", "Ingredient", "S", "8703"),
    # Breast cancer.
    (91000046, "Malignant neoplasm of breast", "Condition", "SNOMED", "Clinical Finding", "S", "254837009"),
    (91000047, "Infiltrating duct carcinoma of breast", "Condition", "SNOMED", "Clinical Finding", "S", "408643008"),
    (91000048, "Malignant lymphoma of breast", "Condition", "SNOMED", "Clinical Finding", "S", "278052009"),
    (91000049, "Carcinoma in situ of breast", "Condition", "SNOMED", "Clinical Finding", "S", "189336000"),
    (91000050, "Ductal carcinoma in situ of breast", "Condition", "SNOMED", "Clinical Finding", "S", "109889007"),
    # CPT-4 procedures, which concept_ancestor does not file under any SNOMED procedure.
    (91000051, "Percutaneous transcatheter placement of intracoronary stent(s), single major coronary artery or branch",
     "Procedure", "CPT4", "CPT4", "S", "92928"),
    (91000052, "Extracapsular cataract removal with insertion of intraocular lens prosthesis", "Procedure", "CPT4", "CPT4", "S",
     "66984"),
)
SYNTHETIC_CONCEPT_ANCESTORS = (
    *((concept[0], concept[0], 0, 0) for concept in SYNTHETIC_CONCEPTS),
    (21600712, 91000001, 2, 2), (21600712, 91000002, 3, 3), (91000001, 91000002, 1, 1),
    (91000005, 91000007, 1, 1),
    (91000011, 91000008, 1, 1), (91000011, 91000009, 2, 2), (91000008, 91000009, 1, 1),
    (91000013, 91000012, 1, 1),
    # Hypertensive disorder over pre-eclampsia (by maternal hypertension) and secondary hypertension; hypertensive
    # heart disease is filed apart from it.
    (316866, 91000015, 1, 1), (316866, 91000016, 2, 2), (91000015, 91000016, 1, 1), (316866, 91000017, 1, 1),
    (91000019, 91000020, 2, 2),
    # Myocardial infarction (type 2 included) under ischemic heart disease; coronary arteriosclerosis and stenosis
    # under disorder of coronary artery.
    (91000021, 91000022, 1, 1), (91000021, 91000023, 2, 2), (91000022, 91000023, 1, 1),
    (91000024, 91000025, 1, 1), (91000024, 91000026, 1, 1),
    (91000028, 91000029, 1, 1),
    # The overlap syndrome under both asthma and COPD; fluticasone and albuterol under their R03 classes.
    (91000030, 91000031, 1, 1), (91000030, 91000033, 1, 1), (91000032, 91000033, 1, 1),
    (91000034, 91000035, 2, 2), (91000034, 91000036, 3, 3), (91000035, 91000036, 1, 1),
    (91000034, 91000037, 2, 2), (91000034, 91000038, 3, 3), (91000037, 91000038, 1, 1),
    # Mixed hyperlipidemia and hypercholesterolemia under hyperlipidemia, raised HDL under hypercholesterolemia.
    (91000039, 91000040, 1, 1), (91000039, 91000041, 1, 1), (91000039, 91000042, 3, 3), (91000041, 91000042, 2, 2),
    (21601853, 91000044, 2, 2), (21601853, 91000045, 3, 3), (91000044, 91000045, 1, 1),
    (91000046, 91000047, 2, 2), (91000046, 91000048, 1, 1), (91000049, 91000050, 1, 1),
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
    # 2015-03-01 has two rows (mg/dL and no unit): one occasion, the mean of their creatinine on the linear scale.
    ages = [_age(date, 1960) for date in ("2015-03-01", "2016-05-10", "2017-01-01")]
    assert first.dates.tolist() == [_date("2015-03-01"), _date("2016-05-10"), _date("2017-01-01")]
    for value, target in zip(first.values, (0.85, 1.0, 25.0), strict=True):
        assert value == within_rounding(target, _IDENTITY_OPERATIONS)
    for age, target in zip(first.ages, ages, strict=True):
        assert age == within_rounding(target, 1)
    assert not first.treated.any()
    assert first.sex_at_birth_name == "female"

    second = people["2"]
    assert _excluded(second) == {"under_minimum_age": 1}
    for value, target in zip(second.values, (1.1, 0.9), strict=True):
        assert value == within_rounding(target, _IDENTITY_OPERATIONS)

    # The query keeps creatinine whatever the sex at birth; the eGFR equation, which needs it, is applied in Python.
    third = people["3"]
    assert _excluded(third) == {}
    assert third.values.size == 1


def test_egfr_readings_are_the_ckd_epi_2021_equation_of_each_days_creatinine(cdr: _Cdr, monkeypatch):
    egfr = resolve_measurement_definition("egfr_ckd_epi_2021")
    for person_id, sex, creatinine in ((1, FEMALE, (0.6, 1.3)), (2, MALE, (0.8, 1.5)), (3, 903096, (1.0, 2.0))):
        cdr.person(person_id, 1960, sex)
        for year, value in zip((2015, 2016), creatinine, strict=True):
            cdr.measurement(person_id, f"{year}-03-01", value)
    cdr.measurement(1, "2015-03-01", 0.7, unit_concept_id=None)
    captured: dict[str, object] = {}

    def stand_in(occasions, working_bytes):
        captured["occasions"] = occasions
        persons = occasions.person_count
        return type("Fit", (), {
            "exponent": 1.0, "level_variance": 1.0, "noise_second_moment": 1.0, "log_evidence": 0.0, "split_identified": True,
            "level_mean": np.zeros(persons), "reliability": np.full(persons, 0.5),
        })()

    monkeypatch.setattr("sv_pgs.all_of_us.fit_occasion_model", stand_in)
    rows = cdr.measurement_rows(egfr)
    training_rows, _columns, summary = build_all_of_us_measurement_targets(egfr, rows, WORKING_BYTES)
    # A skipped sex at birth is not guessed: the eGFR is not one value, so the trait leaves the person out.
    assert [row["person_id"] for row in training_rows] == ["1", "2"]
    assert summary["n_persons_sex_at_birth_unknown"] == 1
    # Each day's creatinine (the same-day mean of 0.6 and 0.7 on the first) through the person's own equation.
    people = {person.person_id: person for person in person_occasions(rows)}
    expected = [
        _ckd_epi_2021(creatinine, age, female=person_id == "1")
        for person_id in ("1", "2")
        for creatinine, age in zip(people[person_id].values, people[person_id].ages, strict=True)
    ]
    assert people["1"].values[0] == within_rounding(0.65, _IDENTITY_OPERATIONS)
    occasions = captured["occasions"]
    for value, target in zip(occasions.values, expected, strict=True):
        assert value == within_rounding(target, _EGFR_OPERATIONS)
    # The recording resolution is creatinine's, the smallest gap between 0.65, 0.8, 1.3 and 1.5 mg/dL, carried to each
    # eGFR by the equation's derivative: the exponent of creatinine on its side of kappa times eGFR over creatinine.
    np.testing.assert_array_equal(occasions.recorded, np.concatenate([people["1"].values, people["2"].values]))
    assert occasions.resolution == within_rounding(0.15, _IDENTITY_OPERATIONS, (0.65 + 0.8) / 0.15)
    for slope, exponent, value, creatinine in zip(occasions.slopes, (-0.241, -1.2, -0.302, -1.2), occasions.values, occasions.recorded, strict=True):
        assert slope == within_rounding(exponent * value / creatinine, 2)


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
    """MCV readings recorded to 0.1 fL, as analyzers report them; returns each person's true effect."""
    effects = {}
    for person_id in range(1, persons + 1):
        cdr.person(person_id, int(generator.integers(1940, 1990)), FEMALE if person_id % 2 else MALE)
        effects[str(person_id)] = generator.normal(0.0, 5.0)
        for occasion in range(int(generator.integers(1, 6))):
            cdr.measurement(
                person_id,
                f"{2010 + occasion}-0{1 + occasion}-15",
                round(90.0 + effects[str(person_id)] + generator.normal(0.0, 3.0), 1),
                concept_id=3023599,
                unit_concept_id=8583,
            )
    return effects


def test_query_rows_feed_the_target_builder_end_to_end(cdr: _Cdr, monkeypatch):
    _identity_transform_fit(monkeypatch)
    generator = np.random.default_rng(0)
    mcv = resolve_measurement_definition("mean_corpuscular_volume")
    effects = _mcv_cohort(cdr, generator, 150)
    training_rows, _columns, summary = build_all_of_us_measurement_targets(mcv, cdr.measurement_rows(mcv), WORKING_BYTES)
    assert summary["n_persons"] == 150
    targets = np.array([row["target"] for row in training_rows])
    true_effects = np.array([effects[row["person_id"]] for row in training_rows])
    # The target correlates with the true effect by about the root mean reliability.
    reliability = np.array([row["target_reliability"] for row in training_rows])
    expected_correlation = float(np.sqrt(reliability.mean()))
    correlation_standard_error = (1.0 - expected_correlation**2) / np.sqrt(len(training_rows))
    assert np.corrcoef(targets, true_effects)[0, 1] > expected_correlation - sampling_bound(correlation_standard_error)


def test_gross_errors_reach_the_model_and_are_downweighted_by_its_learned_density(cdr: _Cdr, monkeypatch):
    # Typos of x10 in MCV: the trait keeps them (no plausible range) and the learned noise density keeps them from
    # moving a person's level. Both fits at the identity transform, so their levels share a scale.
    _identity_transform_fit(monkeypatch)
    generator = np.random.default_rng(5)
    mcv = resolve_measurement_definition("mean_corpuscular_volume")
    clean = _Cdr()
    clean.connection.execute(f"SET search_path = '{DATASET}'")
    effects = _mcv_cohort(cdr, generator, 150)
    _mcv_cohort(clean, np.random.default_rng(5), 150)
    typo_people = [str(person_id) for person_id in range(1, 151, 15)]
    for position, person_id in enumerate(typo_people):
        cdr.measurement(int(person_id), f"2019-0{1 + position % 9}-20", round(10.0 * (90.0 + effects[person_id]), 1), concept_id=3023599, unit_concept_id=8583)
    contaminated_rows, _columns, contaminated_summary = build_all_of_us_measurement_targets(mcv, cdr.measurement_rows(mcv), WORKING_BYTES)
    clean_rows, _columns, clean_summary = build_all_of_us_measurement_targets(mcv, clean.measurement_rows(mcv), WORKING_BYTES)
    assert contaminated_summary["n_occasions"] == clean_summary["n_occasions"] + len(typo_people)
    contaminated = {row["person_id"]: row for row in contaminated_rows}
    reference = {row["person_id"]: row for row in clean_rows}
    level_variance = contaminated_summary["level_variance"]
    for person_id in typo_people:
        # Within one posterior standard deviation of the person's level in the fit without the typo.
        sd = np.sqrt(level_variance * (1.0 - contaminated[person_id]["target_reliability"]))
        assert abs(contaminated[person_id]["target"] - reference[person_id]["target"]) < sd


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
    cdr.procedure(5, 91000052, "2020-06-01")  # a CPT-4 cataract extraction alone

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
    assert {row["person_id"]: row["target"] for row in training_rows} == {"1": 0, "2": 0, "3": 1, "5": 1}
    assert counts["n_cases_by_evidence"] == {"procedure": 2} and counts["n_excluded_ambiguous_code"] == 1


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
    # Sex at birth skipped: each eGFR lies between the female and male equations' values. Person 5's creatinine of
    # 1.2 mg/dL at 54 is an eGFR of about 54 by the female equation and 72 by the male one, so it may be below 60
    # but need not be: never a control, nor a case. Person 6's 2.0 mg/dL is below 60 by both: a case.
    for person_id, creatinine in ((5, 1.2), (6, 2.0)):
        _observed_with_early_ehr(cdr, person_id, 1960, 903096)
        cdr.measurement(person_id, "2015-01-10", creatinine)
        cdr.measurement(person_id, "2015-06-10", creatinine)
    egfr_evidence = cdr.lab_evidence(egfr_criterion)
    assert egfr_evidence["1"] == {
        "qualifying_occasion_count": 2, "first_qualifying_date": _date("2015-01-10"), "last_qualifying_date": _date("2015-06-10"),
    }
    assert egfr_evidence["2"]["qualifying_occasion_count"] == 1
    assert "3" not in egfr_evidence and "4" not in egfr_evidence
    assert cdr.measurement_people(resolve_measurement_definition("egfr"))["4"].values.shape == (2,)
    assert egfr_evidence["5"] == {"qualifying_occasion_count": 0, "first_qualifying_date": None, "last_qualifying_date": None}
    assert egfr_evidence["6"]["qualifying_occasion_count"] == 2

    rows = list(cdr.disease_rows(kidney).values())
    training_rows, _columns, counts = _prepare_training_rows(kidney, rows, [egfr_evidence, {}])
    assert {row["person_id"]: row["target"] for row in training_rows} == {"1": 1, "3": 1, "4": 0, "6": 1}
    assert counts["n_cases_by_evidence"] == {"diagnosis": 1, "lab": 2} and counts["n_excluded_lab_evidence"] == 2
    assert albuminuria_criterion.measurement == "urine_albumin_creatinine_ratio"


def test_hypertension_cases_need_codes_first_line_support_or_repeated_systolic_readings_of_160(cdr: _Cdr):
    hypertension = resolve_disease_definition("hypertension")
    for person_id in range(1, 12):
        _observed_with_early_ehr(cdr, person_id, 1950 + person_id, FEMALE if person_id % 2 else MALE)
    cdr.condition(1, 320128, "2015-01-01")
    cdr.condition(1, 320128, "2016-01-01")  # essential hypertension on two dates: a case
    cdr.condition(2, 91000014, "2015-01-01")
    cdr.drug(2, 91000020, "2015-02-01")  # hypertensive heart disease (filed apart) + lisinopril: a case
    cdr.condition(3, 320128, "2015-01-01")
    cdr.drug(3, 1307046, "2015-02-01")  # one code + metoprolol, a beta blocker that confirms nothing: neither
    cdr.drug(4, 1307046, "2015-02-01")  # metoprolol without a code: not a control
    cdr.condition(5, 320128, "2015-01-01")
    cdr.condition(5, 320128, "2019-01-01")
    cdr.condition(5, 91000016, "2016-03-01")  # pre-eclampsia: ambiguous, whatever else is coded
    cdr.condition(6, 91000017, "2015-01-01")
    cdr.condition(6, 91000017, "2016-01-01")  # secondary hypertension: ambiguous
    cdr.condition(7, 91000018, "2015-01-01")  # an elevated reading without a diagnosis: not a control
    # Systolic readings (mmHg): person 8's two of 160 or more make a case; person 9's 150 and 155 are past the
    # guidelines' 140 but not this criterion's 160, so person 9 is a control; person 10's single 162 is no case
    # and no control.
    for person_id, readings in ((8, (165.0, 172.0)), (9, (150.0, 155.0)), (10, (162.0, 128.0))):
        for date, value in zip(("2016-04-01", "2017-04-01"), readings, strict=True):
            cdr.measurement(person_id, date, value, concept_id=3004249, unit_concept_id=8876)
    [systolic] = hypertension.lab_criteria
    evidence = cdr.lab_evidence(systolic)
    assert {person: row["qualifying_occasion_count"] for person, row in evidence.items()} == {"8": 2, "10": 1}

    rows = cdr.disease_rows(hypertension)
    assert rows["2"]["phenotype_occurrence_count"] == 1 and rows["2"]["case_medication_dates"] == 1
    assert rows["3"]["case_medication_dates"] == 0 and rows["3"]["has_control_exclusion_medication"] is True
    assert rows["5"]["has_ambiguous_code"] is True and rows["6"]["has_ambiguous_code"] is True
    assert rows["7"]["has_control_exclusion_code"] is True

    training_rows, _columns, counts = _prepare_training_rows(hypertension, list(rows.values()), [evidence])
    assert {row["person_id"]: row["target"] for row in training_rows} == {"1": 1, "2": 1, "8": 1, "9": 0, "11": 0}
    assert counts["n_cases_by_evidence"] == {"diagnosis": 2, "lab": 1} and counts["n_excluded_lab_evidence"] == 1
    assert counts["n_excluded_ambiguous_code"] == 2 and counts["n_excluded_one_date"] == 1
    assert counts["n_excluded_control_exclusion"] == 2


def test_coronary_artery_disease_cases_come_from_codes_or_revascularization(cdr: _Cdr):
    coronary = resolve_disease_definition("coronary_artery_disease")
    for person_id in range(1, 10):
        _observed_with_early_ehr(cdr, person_id, 1945 + person_id, MALE if person_id % 2 else FEMALE)
    cdr.condition(1, 91000025, "2015-01-01")
    cdr.condition(1, 91000025, "2016-01-01")  # coronary arteriosclerosis, filed apart from ischemic heart disease
    cdr.condition(2, 91000022, "2015-01-01")
    cdr.condition(2, 91000025, "2015-06-01")  # myocardial infarction, then arteriosclerosis: two dates of the disease
    cdr.procedure(3, 91000027, "2018-03-01")  # percutaneous coronary intervention alone: a case
    cdr.condition(4, 91000022, "2015-01-01")  # one date: neither
    cdr.condition(5, 91000026, "2015-01-01")  # coronary artery stenosis: not a control
    cdr.drug(6, 91000029, "2015-01-01")  # isosorbide mononitrate without a code: not a control
    cdr.condition(7, 91000023, "2015-01-01")
    cdr.condition(7, 91000023, "2015-01-03")  # type 2 (demand) myocardial infarction: ambiguous
    cdr.procedure(9, 91000051, "2019-05-01")  # a CPT-4 coronary stent alone: a case

    rows = cdr.disease_rows(coronary)
    assert rows["2"]["phenotype_occurrence_count"] == 2
    assert rows["3"]["case_procedure_dates"] == 1
    assert rows["7"]["phenotype_occurrence_count"] == 2 and rows["7"]["has_ambiguous_code"] is True

    training_rows, _columns, counts = _prepare_training_rows(coronary, list(rows.values()), [])
    assert {row["person_id"]: row["target"] for row in training_rows} == {"1": 1, "2": 1, "3": 1, "8": 0, "9": 1}
    assert counts["n_cases_by_evidence"] == {"diagnosis": 2, "procedure": 2} and counts["n_excluded_one_date"] == 1
    assert counts["n_excluded_control_exclusion"] == 2 and counts["n_excluded_ambiguous_code"] == 1


def test_asthma_cases_need_two_dates_or_repeated_controllers_and_copd_is_ambiguous(cdr: _Cdr):
    asthma = resolve_disease_definition("asthma")
    for person_id in range(1, 8):
        _observed_with_early_ehr(cdr, person_id, 1960 + person_id, FEMALE if person_id % 2 else MALE)
    cdr.condition(1, 91000030, "2015-01-01")
    cdr.condition(1, 91000030, "2016-01-01")
    cdr.condition(2, 91000031, "2015-01-01")
    cdr.drug(2, 91000036, "2015-01-01")
    cdr.drug(2, 91000036, "2015-04-01")  # allergic asthma + fluticasone on two dates: a case
    cdr.condition(3, 91000030, "2015-01-01")
    cdr.drug(3, 91000036, "2015-01-01")  # one controller date: neither
    cdr.drug(4, 91000038, "2015-01-01")  # albuterol alone: no controller, and not a control
    cdr.condition(5, 91000030, "2015-01-01")
    cdr.condition(5, 91000030, "2016-01-01")
    cdr.condition(5, 91000032, "2019-01-01")  # COPD as well: ambiguous
    cdr.condition(6, 91000033, "2015-01-01")
    cdr.condition(6, 91000033, "2016-01-01")  # the asthma-COPD overlap syndrome: ambiguous

    rows = cdr.disease_rows(asthma)
    assert rows["2"]["case_medication_dates"] == 2
    assert rows["3"]["case_medication_dates"] == 1
    assert rows["4"]["case_medication_dates"] == 0 and rows["4"]["has_control_exclusion_medication"] is True

    training_rows, _columns, counts = _prepare_training_rows(asthma, list(rows.values()), [])
    assert {row["person_id"]: row["target"] for row in training_rows} == {"1": 1, "2": 1, "7": 0}
    assert counts["n_excluded_one_date"] == 1 and counts["n_excluded_control_exclusion"] == 1
    assert counts["n_excluded_ambiguous_code"] == 2


def test_hyperlipidemia_cases_come_from_codes_ldl_lowering_support_or_repeated_ldl_of_160(cdr: _Cdr):
    hyperlipidemia = resolve_disease_definition("hyperlipidemia")
    for person_id in range(1, 11):
        _observed_with_early_ehr(cdr, person_id, 1955 + person_id, FEMALE if person_id % 2 else MALE)
    cdr.condition(1, 91000040, "2015-01-01")
    cdr.condition(1, 91000040, "2016-01-01")  # mixed hyperlipidemia on two dates
    cdr.condition(2, 91000039, "2015-01-01")
    cdr.drug(2, 1545958, "2015-02-01")  # one code + atorvastatin: a case
    cdr.condition(3, 91000039, "2015-01-01")
    cdr.drug(3, 91000045, "2015-02-01")  # one code + fenofibrate, which barely moves LDL: neither
    cdr.drug(4, 91000045, "2015-02-01")  # fenofibrate without a code: not a control
    cdr.condition(5, 91000042, "2015-01-01")
    cdr.condition(5, 91000042, "2016-01-01")  # raised HDL, filed under hypercholesterolemia: ambiguous
    cdr.condition(6, 91000043, "2015-01-01")  # dyslipidemia: not a control
    # LDL cholesterol (mg/dL): person 7's two values of 160 or more make a case, person 8's single one is no case
    # and no control, and person 9's values below 160 leave a control.
    for person_id, values in ((7, (175.0, 182.0)), (8, (165.0, 120.0)), (9, (150.0, 155.0))):
        for date, value in zip(("2016-04-01", "2017-04-01"), values, strict=True):
            cdr.measurement(person_id, date, value, concept_id=3028437)
    [ldl] = hyperlipidemia.lab_criteria
    evidence = cdr.lab_evidence(ldl)
    assert {person: row["qualifying_occasion_count"] for person, row in evidence.items()} == {"7": 2, "8": 1}

    rows = cdr.disease_rows(hyperlipidemia)
    assert rows["2"]["case_medication_dates"] == 1
    assert rows["3"]["case_medication_dates"] == 0 and rows["3"]["has_control_exclusion_medication"] is True

    training_rows, _columns, counts = _prepare_training_rows(hyperlipidemia, list(rows.values()), [evidence])
    assert {row["person_id"]: row["target"] for row in training_rows} == {"1": 1, "2": 1, "7": 1, "9": 0, "10": 0}
    assert counts["n_cases_by_evidence"] == {"diagnosis": 2, "lab": 1} and counts["n_excluded_lab_evidence"] == 1
    assert counts["n_excluded_one_date"] == 1 and counts["n_excluded_control_exclusion"] == 2
    assert counts["n_excluded_ambiguous_code"] == 1


def test_breast_cancer_cases_are_invasive_and_female_and_controls_are_women_of_40_or_more(cdr: _Cdr):
    breast = resolve_disease_definition("breast_cancer")
    for person_id, year_of_birth, sex in (
        (1, 1950, FEMALE), (2, 1950, MALE), (3, 1955, FEMALE), (4, 1955, FEMALE),
        (5, 1960, FEMALE), (6, 1960, FEMALE), (7, 1990, FEMALE), (8, 1960, MALE),
    ):
        _observed_with_early_ehr(cdr, person_id, year_of_birth, sex)
    for person_id in (1, 2):
        cdr.condition(person_id, 91000047, "2015-01-01")
        cdr.condition(person_id, 91000047, "2015-03-01")  # infiltrating duct carcinoma on two dates
    cdr.condition(3, 91000050, "2015-01-01")
    cdr.condition(3, 91000050, "2016-01-01")  # ductal carcinoma in situ alone: neither
    cdr.condition(4, 91000050, "2014-01-01")
    cdr.condition(4, 91000046, "2016-01-01")
    cdr.condition(4, 91000046, "2017-01-01")  # in situ, then invasive: a case
    cdr.condition(5, 91000048, "2015-01-01")
    cdr.condition(5, 91000048, "2016-01-01")  # lymphoma of the breast: ambiguous

    rows = cdr.disease_rows(breast)
    assert rows["3"]["phenotype_occurrence_count"] == 0 and rows["3"]["has_control_exclusion_code"] is True
    assert rows["5"]["phenotype_occurrence_count"] == 2 and rows["5"]["has_ambiguous_code"] is True

    training_rows, _columns, counts = _prepare_training_rows(breast, list(rows.values()), [])
    # Person 7 is 33 at the end of observation, under the controls' 40; persons 2 and 8 are men.
    assert {row["person_id"]: row["target"] for row in training_rows} == {"1": 1, "4": 1, "6": 0}
    assert counts["n_excluded_sex"] == 2 and counts["n_excluded_age"] == 1
    assert counts["n_excluded_control_exclusion"] == 1 and counts["n_excluded_ambiguous_code"] == 1


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

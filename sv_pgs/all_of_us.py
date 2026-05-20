from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from google.cloud import bigquery

MIN_DISEASE_OCCURRENCES = 2


@dataclass(frozen=True, slots=True)
class DiseaseDefinition:
    canonical_name: str
    aliases: tuple[str, ...]
    description: str
    snomed_code: str
    snomed_concept_name: str


# Top-20 chronic disease phenotypes. Each is rooted at a single SNOMED CT
# disease concept; the OMOP concept_id is resolved at query time via the
# `concept` table (vocabulary_id='SNOMED', standard_concept='S'), and the
# `concept_ancestor` join expands to every descendant disorder. This mirrors
# the OHDSI Phenotype Library's canonical-cohort logic.
DISEASE_DEFINITIONS: tuple[DiseaseDefinition, ...] = (
    DiseaseDefinition(
        canonical_name="hypertension",
        aliases=("htn", "high_blood_pressure", "high blood pressure", "essential_hypertension"),
        description="Essential hypertension phenotype from EHR conditions.",
        snomed_code="59621000",
        snomed_concept_name="Essential hypertension",
    ),
    DiseaseDefinition(
        canonical_name="type2_diabetes",
        aliases=("t2d", "type_2_diabetes", "type 2 diabetes", "diabetes_type_2", "t2dm"),
        description="Type 2 diabetes mellitus phenotype from EHR conditions.",
        snomed_code="44054006",
        snomed_concept_name="Diabetes mellitus type 2",
    ),
    DiseaseDefinition(
        canonical_name="hyperlipidemia",
        aliases=("dyslipidemia", "high_cholesterol", "hypercholesterolemia"),
        description="Hyperlipidemia phenotype from EHR conditions.",
        snomed_code="55822004",
        snomed_concept_name="Hyperlipidemia",
    ),
    DiseaseDefinition(
        canonical_name="obesity",
        aliases=("obese", "morbid_obesity"),
        description="Obesity phenotype from EHR conditions.",
        snomed_code="414916001",
        snomed_concept_name="Obesity",
    ),
    DiseaseDefinition(
        canonical_name="asthma",
        aliases=("asthma_chronic",),
        description="Asthma phenotype from EHR conditions.",
        snomed_code="195967001",
        snomed_concept_name="Asthma",
    ),
    DiseaseDefinition(
        canonical_name="depression",
        aliases=("major_depression", "major depressive disorder", "mdd"),
        description="Major depressive disorder phenotype from EHR conditions.",
        snomed_code="370143000",
        snomed_concept_name="Major depressive disorder",
    ),
    DiseaseDefinition(
        canonical_name="anxiety",
        aliases=("anxiety_disorder", "generalized_anxiety"),
        description="Anxiety phenotype from EHR conditions.",
        snomed_code="48694002",
        snomed_concept_name="Anxiety",
    ),
    DiseaseDefinition(
        canonical_name="gerd",
        aliases=("reflux", "gastroesophageal_reflux", "gastroesophageal reflux disease"),
        description="Gastroesophageal reflux disease phenotype from EHR conditions.",
        snomed_code="235595009",
        snomed_concept_name="Gastroesophageal reflux disease",
    ),
    DiseaseDefinition(
        canonical_name="chronic_kidney_disease",
        aliases=("ckd", "chronic kidney disease"),
        description="Chronic kidney disease phenotype from EHR conditions.",
        snomed_code="709044004",
        snomed_concept_name="Chronic kidney disease",
    ),
    DiseaseDefinition(
        canonical_name="coronary_artery_disease",
        aliases=("cad", "coronary artery disease", "ischemic_heart_disease", "coronary_arteriosclerosis"),
        description="Coronary artery disease / coronary arteriosclerosis phenotype from EHR conditions.",
        snomed_code="53741008",
        snomed_concept_name="Coronary arteriosclerosis",
    ),
    DiseaseDefinition(
        canonical_name="heart_failure",
        aliases=("hf", "congestive_heart_failure", "chf", "heart failure"),
        description="Heart failure phenotype from EHR conditions.",
        snomed_code="84114007",
        snomed_concept_name="Heart failure",
    ),
    DiseaseDefinition(
        canonical_name="atrial_fibrillation",
        aliases=("af", "afib", "a_fib", "atrial fibrillation"),
        description="Atrial fibrillation phenotype from EHR conditions.",
        snomed_code="49436004",
        snomed_concept_name="Atrial fibrillation",
    ),
    DiseaseDefinition(
        canonical_name="osteoarthritis",
        aliases=("oa", "degenerative_joint_disease"),
        description="Osteoarthritis phenotype from EHR conditions.",
        snomed_code="396275006",
        snomed_concept_name="Osteoarthritis",
    ),
    DiseaseDefinition(
        canonical_name="copd",
        aliases=("chronic_obstructive_pulmonary_disease", "chronic obstructive pulmonary disease", "chronic_obstructive_lung_disease"),
        description="Chronic obstructive lung disease phenotype from EHR conditions.",
        snomed_code="13645005",
        snomed_concept_name="Chronic obstructive lung disease",
    ),
    DiseaseDefinition(
        canonical_name="stroke",
        aliases=("cerebrovascular_accident", "cva", "cerebrovascular_disease"),
        description="Cerebrovascular accident (stroke) phenotype from EHR conditions.",
        snomed_code="230690007",
        snomed_concept_name="Cerebrovascular accident",
    ),
    DiseaseDefinition(
        canonical_name="hypothyroidism",
        aliases=("underactive_thyroid", "low_thyroid"),
        description="Hypothyroidism phenotype from EHR conditions.",
        snomed_code="40930008",
        snomed_concept_name="Hypothyroidism",
    ),
    DiseaseDefinition(
        canonical_name="migraine",
        aliases=("migraine_headache", "migraines"),
        description="Migraine phenotype from EHR conditions.",
        snomed_code="37796009",
        snomed_concept_name="Migraine",
    ),
    DiseaseDefinition(
        canonical_name="rheumatoid_arthritis",
        aliases=("ra",),
        description="Rheumatoid arthritis phenotype from EHR conditions.",
        snomed_code="69896004",
        snomed_concept_name="Rheumatoid arthritis",
    ),
    DiseaseDefinition(
        canonical_name="atherosclerosis",
        aliases=("ascvd", "atherosclerotic_disease"),
        description="Atherosclerosis phenotype from EHR conditions.",
        snomed_code="38716007",
        snomed_concept_name="Atherosclerosis",
    ),
    DiseaseDefinition(
        canonical_name="sleep_apnea",
        aliases=("osa", "obstructive_sleep_apnea", "sleep apnea"),
        description="Sleep apnea phenotype from EHR conditions.",
        snomed_code="73430006",
        snomed_concept_name="Sleep apnea",
    ),
)


@dataclass(slots=True)
class AllOfUsDiseaseRequest:
    disease: str

    def __post_init__(self) -> None:
        if not self.disease.strip():
            raise ValueError("disease cannot be blank.")


@dataclass(slots=True)
class AllOfUsPreparedPhenotype:
    sample_table_path: Path
    sql_path: Path
    metadata_path: Path


def available_disease_names() -> list[str]:
    return sorted(disease_definition.canonical_name for disease_definition in DISEASE_DEFINITIONS)


def resolve_disease_definition(disease: str) -> DiseaseDefinition:
    normalized_disease = _normalize_name(disease)
    for disease_definition in DISEASE_DEFINITIONS:
        candidate_names = (disease_definition.canonical_name, *disease_definition.aliases)
        if normalized_disease in {_normalize_name(candidate_name) for candidate_name in candidate_names}:
            return disease_definition
    raise ValueError(
        "Unsupported disease: "
        + disease
        + ". Available diseases: "
        + ", ".join(available_disease_names())
    )


def build_all_of_us_disease_sql(disease_definition: DiseaseDefinition) -> str:
    dataset = _require_env("WORKSPACE_CDR")
    return f"""
WITH primary_consent AS (
  SELECT
    observation.person_id,
    MIN(observation.observation_date) AS primary_consent_date
  FROM `{dataset}.concept` AS concept
  JOIN `{dataset}.concept_ancestor` AS concept_ancestor
    ON concept.concept_id = concept_ancestor.ancestor_concept_id
  JOIN `{dataset}.observation` AS observation
    ON concept_ancestor.descendant_concept_id = observation.observation_concept_id
  WHERE concept.concept_name = 'Consent PII'
    AND concept.concept_class_id = 'Module'
  GROUP BY observation.person_id
),
ehr_participants AS (
  SELECT
    person_id,
    MIN(observation_period_start_date) AS observation_start_date,
    MAX(observation_period_end_date) AS observation_end_date
  FROM `{dataset}.observation_period`
  GROUP BY person_id
),
disease_root AS (
  SELECT concept_id
  FROM `{dataset}.concept`
  WHERE vocabulary_id = 'SNOMED'
    AND standard_concept = 'S'
    AND concept_code = @snomed_code
),
disease_concepts AS (
  SELECT DISTINCT concept_ancestor.descendant_concept_id AS concept_id
  FROM `{dataset}.concept_ancestor` AS concept_ancestor
  JOIN disease_root ON disease_root.concept_id = concept_ancestor.ancestor_concept_id
),
matched_conditions AS (
  SELECT
    condition_occurrence.person_id,
    condition_occurrence.condition_start_date
  FROM `{dataset}.condition_occurrence` AS condition_occurrence
  WHERE condition_occurrence.condition_concept_id IN (
    SELECT concept_id FROM disease_concepts
  )
),
aggregated_conditions AS (
  SELECT
    person_id,
    COUNT(*) AS phenotype_occurrence_count,
    MIN(condition_start_date) AS first_condition_date
  FROM matched_conditions
  GROUP BY person_id
)
SELECT
  CAST(ehr_participants.person_id AS STRING) AS sample_id,
  CAST(ehr_participants.person_id AS STRING) AS person_id,
  IF(COALESCE(aggregated_conditions.phenotype_occurrence_count, 0) >= {MIN_DISEASE_OCCURRENCES}, 1, 0) AS target,
  COALESCE(aggregated_conditions.phenotype_occurrence_count, 0) AS phenotype_occurrence_count,
  aggregated_conditions.first_condition_date,
  ehr_participants.observation_start_date,
  ehr_participants.observation_end_date,
  primary_consent.primary_consent_date,
  CASE
    WHEN person.year_of_birth IS NULL OR ehr_participants.observation_start_date IS NULL THEN NULL
    ELSE EXTRACT(YEAR FROM ehr_participants.observation_start_date) - person.year_of_birth
  END AS age_at_observation_start,
  person.gender_concept_id,
  person.race_concept_id,
  person.ethnicity_concept_id
FROM ehr_participants
JOIN `{dataset}.person` AS person
  ON person.person_id = ehr_participants.person_id
LEFT JOIN aggregated_conditions
  ON aggregated_conditions.person_id = ehr_participants.person_id
LEFT JOIN primary_consent
  ON primary_consent.person_id = ehr_participants.person_id
ORDER BY ehr_participants.person_id
""".strip()


def build_all_of_us_disease_query_config(disease_definition: DiseaseDefinition):
    from google.cloud import bigquery

    return bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("snomed_code", "STRING", disease_definition.snomed_code),
        ]
    )


def fetch_all_of_us_disease_rows(
    request: AllOfUsDiseaseRequest,
    client: bigquery.Client | None = None,
) -> list[dict[str, Any]]:
    from google.cloud import bigquery

    disease_definition = resolve_disease_definition(request.disease)
    active_client = client if client is not None else bigquery.Client(project=_require_env("GOOGLE_PROJECT"))
    query_job = active_client.query(
        build_all_of_us_disease_sql(disease_definition),
        job_config=build_all_of_us_disease_query_config(disease_definition),
    )
    return [dict(row.items()) for row in query_job.result()]


def prepare_all_of_us_disease_sample_table(
    request: AllOfUsDiseaseRequest,
    output_path: str | Path,
    *,
    client: bigquery.Client | None = None,
) -> AllOfUsPreparedPhenotype:
    disease_definition = resolve_disease_definition(request.disease)
    rows = fetch_all_of_us_disease_rows(request=request, client=client)
    billing_project = _resolve_billing_project(client)
    sample_table_path = Path(output_path)
    sample_table_path.parent.mkdir(parents=True, exist_ok=True)

    header = (
        "sample_id",
        "person_id",
        "target",
        "phenotype_occurrence_count",
        "first_condition_date",
        "observation_start_date",
        "observation_end_date",
        "primary_consent_date",
        "age_at_observation_start",
        "gender_concept_id",
        "race_concept_id",
        "ethnicity_concept_id",
    )
    with sample_table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header)
        for row in rows:
            writer.writerow([_format_value(row.get(column_name)) for column_name in header])

    sql_path = sample_table_path.with_suffix(sample_table_path.suffix + ".sql")
    sql_path.write_text(build_all_of_us_disease_sql(disease_definition) + "\n", encoding="utf-8")

    metadata_path = sample_table_path.with_suffix(sample_table_path.suffix + ".metadata.json")
    metadata_path.write_text(
        json.dumps(
            {
                "disease": disease_definition.canonical_name,
                "description": disease_definition.description,
                "snomed_code": disease_definition.snomed_code,
                "snomed_concept_name": disease_definition.snomed_concept_name,
                "min_occurrences": MIN_DISEASE_OCCURRENCES,
                "billing_project_env": "GOOGLE_PROJECT",
                "cdr_dataset_env": "WORKSPACE_CDR",
                "billing_project": billing_project,
                "cdr_dataset": _require_env("WORKSPACE_CDR"),
                "row_count": len(rows),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return AllOfUsPreparedPhenotype(
        sample_table_path=sample_table_path,
        sql_path=sql_path,
        metadata_path=metadata_path,
    )


def _normalize_name(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise ValueError("Missing required All of Us environment variable: " + name)
    return value.strip()


def _resolve_billing_project(client: bigquery.Client | None) -> str:
    if client is not None:
        client_project = getattr(client, "project", None)
        if isinstance(client_project, str) and client_project.strip():
            return client_project.strip()
    return _require_env("GOOGLE_PROJECT")


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value)

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MIN_DISEASE_OCCURRENCES = 2


@dataclass(frozen=True, slots=True)
class DiseaseDefinition:
    canonical_name: str
    aliases: tuple[str, ...]
    description: str
    icd9_prefixes: tuple[str, ...]
    icd10_prefixes: tuple[str, ...]


DISEASE_DEFINITIONS: tuple[DiseaseDefinition, ...] = (
    DiseaseDefinition(
        canonical_name="type2_diabetes",
        aliases=("t2d", "type_2_diabetes", "type 2 diabetes", "diabetes_type_2"),
        description="Type 2 diabetes mellitus phenotype from EHR conditions.",
        # ICD-9 250.x0/250.x2 = T2D; 250.x1/250.x3 = T1D. Using broad prefix
        # 250 captures some T1D but matches standard EHR phenotyping practice.
        # ICD-10 E11 is specific to T2D.
        icd9_prefixes=("250",),
        icd10_prefixes=("E11",),
    ),
    DiseaseDefinition(
        canonical_name="heart_failure",
        aliases=("hf", "congestive_heart_failure", "chf", "heart failure"),
        description="Heart failure phenotype from EHR conditions.",
        icd9_prefixes=("428",),
        icd10_prefixes=("I50",),
    ),
    DiseaseDefinition(
        canonical_name="atrial_fibrillation",
        aliases=("af", "afib", "a_fib", "atrial fibrillation"),
        description="Atrial fibrillation and flutter phenotype from EHR conditions.",
        icd9_prefixes=("427.3",),
        icd10_prefixes=("I48",),
    ),
    DiseaseDefinition(
        canonical_name="asthma",
        aliases=("asthma_chronic",),
        description="Asthma phenotype from EHR conditions.",
        icd9_prefixes=("493",),
        icd10_prefixes=("J45",),
    ),
    DiseaseDefinition(
        canonical_name="copd",
        aliases=("chronic_obstructive_pulmonary_disease", "chronic obstructive pulmonary disease"),
        description="Chronic obstructive pulmonary disease phenotype from EHR conditions.",
        icd9_prefixes=("490", "491", "492", "496"),
        icd10_prefixes=("J40", "J41", "J42", "J43", "J44"),
    ),
    DiseaseDefinition(
        canonical_name="chronic_kidney_disease",
        aliases=("ckd", "chronic kidney disease"),
        description="Chronic kidney disease phenotype from EHR conditions.",
        icd9_prefixes=("585",),
        icd10_prefixes=("N18",),
    ),
    DiseaseDefinition(
        canonical_name="coronary_artery_disease",
        aliases=("cad", "coronary artery disease", "ischemic_heart_disease"),
        description="Coronary artery disease / ischemic heart disease phenotype from EHR conditions.",
        icd9_prefixes=("410", "411", "412", "413", "414"),
        icd10_prefixes=("I20", "I21", "I22", "I23", "I24", "I25"),
    ),
    DiseaseDefinition(
        canonical_name="stroke",
        aliases=("cerebrovascular_disease", "stroke_ischemic_or_hemorrhagic"),
        description="Broad stroke phenotype from EHR conditions.",
        icd9_prefixes=("430", "431", "432", "433", "434", "436"),
        icd10_prefixes=("I60", "I61", "I62", "I63", "I64"),
    ),
    DiseaseDefinition(
        canonical_name="depression",
        aliases=("major_depression", "major depressive disorder", "depression"),
        description="Broad depression phenotype from EHR conditions.",
        icd9_prefixes=("296.2", "296.3", "311"),
        icd10_prefixes=("F32", "F33"),
    ),
    DiseaseDefinition(
        canonical_name="hypertension",
        aliases=("htn", "high_blood_pressure", "high blood pressure"),
        description="Broad hypertension phenotype from EHR conditions.",
        icd9_prefixes=("401", "402", "403", "404", "405"),
        icd10_prefixes=("I10", "I11", "I12", "I13", "I15"),
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
source_icd_concepts AS (
  SELECT concept_id
  FROM `{dataset}.concept`
  WHERE (
      vocabulary_id = 'ICD9CM'
      AND EXISTS (
        SELECT 1
        FROM UNNEST(@icd9_prefixes) AS icd9_prefix
        WHERE STARTS_WITH(concept_code, icd9_prefix)
      )
    )
    OR (
      vocabulary_id = 'ICD10CM'
      AND EXISTS (
        SELECT 1
        FROM UNNEST(@icd10_prefixes) AS icd10_prefix
        WHERE STARTS_WITH(concept_code, icd10_prefix)
      )
    )
),
matched_conditions AS (
  SELECT
    condition_occurrence.person_id,
    condition_occurrence.condition_start_date
  FROM `{dataset}.condition_occurrence` AS condition_occurrence
  WHERE condition_occurrence.condition_source_concept_id IN (
    SELECT concept_id FROM source_icd_concepts
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
            bigquery.ArrayQueryParameter("icd9_prefixes", "STRING", list(disease_definition.icd9_prefixes)),
            bigquery.ArrayQueryParameter("icd10_prefixes", "STRING", list(disease_definition.icd10_prefixes)),
        ]
    )


def fetch_all_of_us_disease_rows(
    request: AllOfUsDiseaseRequest,
    client: "bigquery.Client | None" = None,
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
    client: "bigquery.Client | None" = None,
) -> AllOfUsPreparedPhenotype:
    from google.cloud import bigquery
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
                "icd9_prefixes": list(disease_definition.icd9_prefixes),
                "icd10_prefixes": list(disease_definition.icd10_prefixes),
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


def _resolve_billing_project(client: "bigquery.Client | None") -> str:
    if client is not None:
        client_project = getattr(client, "project", None)
        if isinstance(client_project, str) and client_project.strip():
            return client_project.strip()
    return _require_env("GOOGLE_PROJECT")


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value)

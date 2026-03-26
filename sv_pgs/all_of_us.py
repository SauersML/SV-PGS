from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

from google.cloud import bigquery, storage

MIN_DISEASE_OCCURRENCES = 2
GENOMICS_MANIFEST_ENV_VARS = (
    "MICROARRAY_IDAT_MANIFEST_PATH",
    "WGS_CRAM_MANIFEST_PATH",
)


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
        icd9_prefixes=("42731", "42732"),
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
        icd9_prefixes=("496", "491", "492"),
        icd10_prefixes=("J41", "J42", "J43", "J44"),
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
        icd9_prefixes=("2962", "2963", "311"),
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


@dataclass(frozen=True, slots=True)
class _ManifestSource:
    env_var: str
    uri_columns: tuple[str, ...]
    description: str


_MANIFEST_SOURCES: tuple[_ManifestSource, ...] = (
    _ManifestSource(
        env_var="MICROARRAY_IDAT_MANIFEST_PATH",
        uri_columns=("green_idat_uri", "red_idat_uri"),
        description="microarray_idat",
    ),
    _ManifestSource(
        env_var="WGS_CRAM_MANIFEST_PATH",
        uri_columns=("cram_uri",),
        description="wgs_cram",
    ),
)


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


def build_all_of_us_disease_query_config(disease_definition: DiseaseDefinition) -> bigquery.QueryJobConfig:
    return bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("icd9_prefixes", "STRING", list(disease_definition.icd9_prefixes)),
            bigquery.ArrayQueryParameter("icd10_prefixes", "STRING", list(disease_definition.icd10_prefixes)),
        ]
    )


def fetch_all_of_us_disease_rows(
    request: AllOfUsDiseaseRequest,
    client: bigquery.Client | None = None,
) -> list[dict[str, Any]]:
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
    storage_client: storage.Client | None = None,
) -> AllOfUsPreparedPhenotype:
    disease_definition = resolve_disease_definition(request.disease)
    rows = fetch_all_of_us_disease_rows(request=request, client=client)
    research_ids_by_person_id, manifest_sources = _load_all_of_us_research_ids_by_person_id(storage_client)
    mapped_rows = _attach_research_ids(rows, research_ids_by_person_id)
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
        for row in mapped_rows:
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
                "genomics_manifest_envs": list(GENOMICS_MANIFEST_ENV_VARS),
                "genomics_manifest_sources": manifest_sources,
                "billing_project": _resolve_billing_project(client),
                "cdr_dataset": _require_env("WORKSPACE_CDR"),
                "input_row_count": len(rows),
                "row_count": len(mapped_rows),
                "unmapped_person_count": len(rows) - len(mapped_rows),
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


def _load_all_of_us_research_ids_by_person_id(
    storage_client: storage.Client | None,
) -> tuple[dict[str, str], list[str]]:
    manifest_rows_by_source: list[tuple[_ManifestSource, list[dict[str, str]]]] = []
    for manifest_source in _MANIFEST_SOURCES:
        manifest_path = os.environ.get(manifest_source.env_var, "").strip()
        if not manifest_path:
            continue
        manifest_rows_by_source.append(
            (
                manifest_source,
                _read_gcs_csv_rows(
                    manifest_path,
                    storage_client=storage_client,
                ),
            )
        )

    if not manifest_rows_by_source:
        raise ValueError(
            "Missing required All of Us genomics manifest environment variable. Expected one of: "
            + ", ".join(GENOMICS_MANIFEST_ENV_VARS)
        )

    merged_research_ids_by_person_id: dict[str, str] = {}
    active_sources: list[str] = []
    for manifest_source, manifest_rows in manifest_rows_by_source:
        active_sources.append(manifest_source.description)
        source_mapping = _extract_research_ids_from_manifest_rows(manifest_rows, manifest_source)
        for person_id, research_id in source_mapping.items():
            existing_research_id = merged_research_ids_by_person_id.get(person_id)
            if existing_research_id is not None and existing_research_id != research_id:
                raise ValueError(
                    "Conflicting All of Us research_id mappings for person_id "
                    + person_id
                    + ": "
                    + existing_research_id
                    + " vs "
                    + research_id
                )
            merged_research_ids_by_person_id[person_id] = research_id
    return merged_research_ids_by_person_id, active_sources


def _read_gcs_csv_rows(
    manifest_path: str,
    *,
    storage_client: storage.Client | None,
) -> list[dict[str, str]]:
    bucket_name, blob_name = _parse_gcs_uri(manifest_path)
    active_storage_client = storage_client if storage_client is not None else storage.Client(project=_require_env("GOOGLE_PROJECT"))
    text = active_storage_client.bucket(bucket_name).blob(blob_name).download_as_text(encoding="utf-8")
    reader = csv.DictReader(StringIO(text))
    if reader.fieldnames is None:
        raise ValueError("Manifest has no header row: " + manifest_path)
    return [
        {str(key): "" if value is None else str(value) for key, value in row.items()}
        for row in reader
    ]


def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    normalized_uri = uri.strip()
    if not normalized_uri.startswith("gs://"):
        raise ValueError("Expected gs:// URI for All of Us manifest: " + normalized_uri)
    bucket_and_path = normalized_uri[5:]
    if "/" not in bucket_and_path:
        raise ValueError("Expected object path in All of Us manifest URI: " + normalized_uri)
    bucket_name, blob_name = bucket_and_path.split("/", 1)
    if not bucket_name or not blob_name:
        raise ValueError("Malformed All of Us manifest URI: " + normalized_uri)
    return bucket_name, blob_name


def _extract_research_ids_from_manifest_rows(
    manifest_rows: list[dict[str, str]],
    manifest_source: _ManifestSource,
) -> dict[str, str]:
    if not manifest_rows:
        raise ValueError("All of Us manifest is empty for source: " + manifest_source.description)
    required_columns = ("person_id", *manifest_source.uri_columns)
    _require_manifest_columns(manifest_rows, required_columns, manifest_source.description)

    research_ids_by_person_id: dict[str, str] = {}
    for manifest_row in manifest_rows:
        person_id = str(manifest_row["person_id"]).strip()
        if not person_id:
            raise ValueError("Encountered blank person_id in All of Us manifest: " + manifest_source.description)
        extracted_research_ids = {
            _extract_research_id_from_uri(manifest_row[uri_column])
            for uri_column in manifest_source.uri_columns
            if str(manifest_row[uri_column]).strip()
        }
        if not extracted_research_ids:
            raise ValueError("No research_id-bearing URI found in All of Us manifest row for person_id " + person_id)
        if len(extracted_research_ids) != 1:
            raise ValueError(
                "Conflicting research_id values within All of Us manifest row for person_id "
                + person_id
                + ": "
                + ", ".join(sorted(extracted_research_ids))
            )
        research_id = next(iter(extracted_research_ids))
        existing_research_id = research_ids_by_person_id.get(person_id)
        if existing_research_id is not None and existing_research_id != research_id:
            raise ValueError(
                "Duplicate person_id with conflicting research_id values in All of Us manifest "
                + manifest_source.description
                + ": "
                + person_id
            )
        research_ids_by_person_id[person_id] = research_id
    return research_ids_by_person_id


def _extract_research_id_from_uri(uri: str) -> str:
    normalized_uri = uri.strip()
    if not normalized_uri:
        raise ValueError("Encountered blank All of Us URI while extracting research_id.")
    file_name = normalized_uri.rsplit("/", 1)[-1]
    match = re.match(r"(?P<research_id>\d{6,})", file_name)
    if match is None:
        raise ValueError("Could not extract research_id from All of Us URI: " + normalized_uri)
    return match.group("research_id")


def _attach_research_ids(
    rows: list[dict[str, Any]],
    research_ids_by_person_id: dict[str, str],
) -> list[dict[str, Any]]:
    mapped_rows: list[dict[str, Any]] = []
    for row in rows:
        person_id = str(row["person_id"]).strip()
        research_id = research_ids_by_person_id.get(person_id)
        if research_id is None:
            continue
        mapped_row = dict(row)
        mapped_row["sample_id"] = research_id
        mapped_rows.append(mapped_row)
    if not mapped_rows:
        raise ValueError("No All of Us disease rows matched any genomics research_id mapping.")
    return mapped_rows


def _require_manifest_columns(
    rows: list[dict[str, str]],
    required_columns: tuple[str, ...],
    context: str,
) -> None:
    missing_columns = [column_name for column_name in required_columns if column_name not in rows[0]]
    if missing_columns:
        raise ValueError(
            "All of Us manifest "
            + context
            + " is missing required columns: "
            + ", ".join(missing_columns)
        )


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value)

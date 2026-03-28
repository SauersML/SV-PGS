from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from bed_reader import open_bed
from cyvcf2 import VCF
from sklearn.metrics import log_loss, r2_score, roc_auc_score

from sv_pgs.config import ModelConfig, TraitType, VariantClass
from sv_pgs.data import VariantRecord
from sv_pgs.model import BayesianPGS

SV_LENGTH_THRESHOLD = 1_000.0
DEFAULT_SAMPLE_ID_COLUMNS = ("sample_id", "research_id", "person_id")


@dataclass(slots=True)
class LoadedDataset:
    sample_ids: list[str]
    genotypes: np.ndarray
    covariates: np.ndarray
    targets: np.ndarray
    variant_records: list[VariantRecord]


@dataclass(slots=True)
class PipelineOutputs:
    artifact_dir: Path
    summary_path: Path
    predictions_path: Path
    coefficients_path: Path


@dataclass(slots=True)
class _SampleTable:
    sample_ids: list[str]
    covariates: np.ndarray
    targets: np.ndarray


@dataclass(slots=True)
class _VariantDefaults:
    variant_id: str
    variant_class: VariantClass
    chromosome: str
    position: int
    length: float
    allele_frequency: float
    quality: float


def load_dataset_from_files(
    genotype_path: str | Path,
    sample_table_path: str | Path,
    target_column: str,
    covariate_columns: Sequence[str],
    *,
    genotype_format: str = "auto",
    sample_id_column: str = "auto",
    variant_metadata_path: str | Path | None = None,
) -> LoadedDataset:
    import sys, os, resource
    def _mem():
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            rss //= 1024 * 1024
        else:
            rss //= 1024
        return f"{rss} MB"
    print(f"[load] START  mem={_mem()}", flush=True)

    source_path = Path(genotype_path)
    resolved_format = _resolve_genotype_format(source_path, genotype_format)
    print(f"[load] format={resolved_format}  path={source_path}", flush=True)

    sample_table_rows = _read_delimited_rows(sample_table_path)
    if not sample_table_rows:
        raise ValueError("Sample table is empty: " + str(sample_table_path))
    print(f"[load] sample_table rows={len(sample_table_rows)}  columns={list(sample_table_rows[0].keys())}", flush=True)

    if resolved_format == "vcf":
        print("[load] loading VCF...", flush=True)
        source_sample_ids, genotype_matrix, default_variants = _load_vcf(source_path)
    elif resolved_format == "plink1":
        print("[load] loading PLINK metadata (sample IDs + variant info)...", flush=True)
        source_sample_ids, genotype_matrix, default_variants = _load_plink1(source_path)
    else:
        raise ValueError("Unsupported genotype format: " + resolved_format)

    print(f"[load] genotype source: {len(source_sample_ids)} samples, {len(default_variants)} variants", flush=True)
    print(f"[load] genotype_matrix shape={genotype_matrix.shape}  dtype={genotype_matrix.dtype}  mem={_mem()}", flush=True)

    resolved_sample_id_column = _resolve_sample_id_column(
        rows=sample_table_rows,
        requested_sample_id_column=sample_id_column,
        available_sample_ids=source_sample_ids,
    )
    print(f"[load] resolved sample_id_column={resolved_sample_id_column}", flush=True)

    sample_table = _build_sample_table(
        rows=sample_table_rows,
        sample_id_column=resolved_sample_id_column,
        target_column=target_column,
        covariate_columns=covariate_columns,
    )
    print(f"[load] sample_table: {len(sample_table.sample_ids)} samples, covariates shape={sample_table.covariates.shape}", flush=True)

    aligned_sample_indices = _align_sample_ids(
        expected_sample_ids=sample_table.sample_ids,
        available_sample_ids=source_sample_ids,
        context="genotype source",
    )
    print(f"[load] aligned {len(aligned_sample_indices)} samples", flush=True)

    aligned_genotypes = genotype_matrix[aligned_sample_indices, :]
    print(f"[load] aligned_genotypes shape={aligned_genotypes.shape}  mem={_mem()}", flush=True)

    variant_records = _build_variant_records(
        default_variants=default_variants,
        variant_metadata_path=variant_metadata_path,
    )
    print(f"[load] variant_records: {len(variant_records)}", flush=True)

    result = LoadedDataset(
        sample_ids=list(sample_table.sample_ids),
        genotypes=np.asarray(aligned_genotypes, dtype=np.float32),
        covariates=np.asarray(sample_table.covariates, dtype=np.float32),
        targets=np.asarray(sample_table.targets, dtype=np.float32),
        variant_records=variant_records,
    )
    print(f"[load] DONE  final genotypes shape={result.genotypes.shape}  mem={_mem()}", flush=True)
    return result


def run_training_pipeline(
    dataset: LoadedDataset,
    config: ModelConfig,
    output_dir: str | Path,
) -> PipelineOutputs:
    print(f"[pipeline] START  samples={len(dataset.sample_ids)}  genotypes={dataset.genotypes.shape}  trait={config.trait_type}", flush=True)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    print(f"[pipeline] fitting model...", flush=True)
    model = BayesianPGS(config).fit(
        dataset.genotypes,
        dataset.covariates,
        dataset.targets,
        dataset.variant_records,
    )

    print(f"[pipeline] model fitted, exporting artifacts...", flush=True)
    artifact_dir = destination / "artifact"
    model.export(artifact_dir)
    print(f"[pipeline] artifacts exported to {artifact_dir}", flush=True)

    coefficients_path = destination / "coefficients.tsv"
    coefficient_rows = model.coefficient_table()
    _write_delimited_rows(
        coefficients_path,
        header=("variant_id", "variant_class", "beta"),
        rows=(
            (
                str(coefficient_row["variant_id"]),
                str(coefficient_row["variant_class"]),
                _format_float(float(coefficient_row["beta"])),
            )
            for coefficient_row in coefficient_rows
        ),
    )

    predictions_path = destination / "predictions.tsv"
    summary_payload = _write_predictions_and_summary(
        predictions_path=predictions_path,
        dataset=dataset,
        model=model,
    )
    summary_payload.update(
        {
            "sample_count": int(dataset.genotypes.shape[0]),
            "variant_count": int(dataset.genotypes.shape[1]),
            "active_variant_count": int(model.state.active_variant_indices.shape[0]) if model.state is not None else 0,
            "trait_type": config.trait_type.value,
        }
    )

    print(f"[pipeline] writing summary...", flush=True)
    summary_path = destination / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(f"[pipeline] DONE", flush=True)
    return PipelineOutputs(
        artifact_dir=artifact_dir,
        summary_path=summary_path,
        predictions_path=predictions_path,
        coefficients_path=coefficients_path,
    )


def _build_sample_table(
    rows: Sequence[dict[str, str]],
    sample_id_column: str,
    target_column: str,
    covariate_columns: Sequence[str],
) -> _SampleTable:
    _require_columns(
        rows=rows,
        required_columns=(sample_id_column, target_column, *covariate_columns),
        context="sample table",
    )

    sample_ids: list[str] = []
    covariates: list[list[float]] = []
    targets: list[float] = []
    seen_sample_ids: set[str] = set()

    for row in rows:
        sample_id = str(row[sample_id_column]).strip()
        if not sample_id:
            raise ValueError("Encountered blank sample identifier in sample table.")
        if sample_id in seen_sample_ids:
            raise ValueError("Duplicate sample identifier in sample table: " + sample_id)
        seen_sample_ids.add(sample_id)
        sample_ids.append(sample_id)
        targets.append(_parse_float(row[target_column], column_name=target_column))
        covariates.append([
            _parse_float(row[column_name], column_name=column_name)
            for column_name in covariate_columns
        ])

    covariate_matrix = np.asarray(covariates, dtype=np.float32)
    if covariate_matrix.ndim != 2:
        covariate_matrix = covariate_matrix.reshape(len(sample_ids), len(covariate_columns))
    return _SampleTable(
        sample_ids=sample_ids,
        covariates=covariate_matrix,
        targets=np.asarray(targets, dtype=np.float32),
    )


def _resolve_sample_id_column(
    rows: Sequence[dict[str, str]],
    requested_sample_id_column: str,
    available_sample_ids: Sequence[str],
) -> str:
    available_columns = tuple(rows[0].keys())
    if requested_sample_id_column != "auto":
        if requested_sample_id_column not in rows[0]:
            raise ValueError("Sample table is missing required columns: " + requested_sample_id_column)
        return requested_sample_id_column

    candidate_columns = [
        column_name
        for column_name in DEFAULT_SAMPLE_ID_COLUMNS
        if column_name in rows[0]
    ]
    if not candidate_columns:
        raise ValueError(
            "Sample table must contain at least one identifier column: "
            + ", ".join(DEFAULT_SAMPLE_ID_COLUMNS)
            + ". Available columns: "
            + ", ".join(available_columns)
        )

    available_sample_id_set = set(available_sample_ids)
    match_counts = {
        column_name: sum(
            1
            for row in rows
            if str(row[column_name]).strip() in available_sample_id_set
        )
        for column_name in candidate_columns
    }
    best_match_count = max(match_counts.values())
    best_columns = [column_name for column_name in candidate_columns if match_counts[column_name] == best_match_count]
    if best_match_count == 0:
        raise ValueError(
            "Could not find a sample identifier column in the sample table that matches the genotype source. "
            + "Tried "
            + ", ".join(f"{column_name}({match_counts[column_name]})" for column_name in candidate_columns)
        )
    if len(best_columns) > 1:
        for preferred_column in DEFAULT_SAMPLE_ID_COLUMNS:
            if preferred_column in best_columns:
                return preferred_column
    return best_columns[0]


def _load_vcf(vcf_path: Path) -> tuple[list[str], np.ndarray, list[_VariantDefaults]]:
    reader = VCF(str(vcf_path))
    reader.set_threads(1)
    sample_ids = [str(sample_id) for sample_id in reader.samples]
    dosage_columns: list[np.ndarray] = []
    variants: list[_VariantDefaults] = []

    for record in reader:
        if len(record.ALT) != 1:
            raise ValueError(
                "Only biallelic VCF records are supported. Normalize multiallelic records before loading: "
                + _vcf_variant_key(record)
            )
        dosage = _vcf_record_dosage(record)
        dosage_columns.append(dosage)
        variants.append(
            _VariantDefaults(
                variant_id=_vcf_variant_id(record),
                variant_class=_infer_vcf_variant_class(record),
                chromosome=str(record.CHROM),
                position=int(record.POS),
                length=_infer_vcf_length(record),
                allele_frequency=_infer_vcf_allele_frequency(record, dosage),
                quality=_normalize_quality(record.QUAL),
            )
        )

    if not dosage_columns:
        raise ValueError("VCF contains no variants: " + str(vcf_path))
    genotype_matrix = np.column_stack(dosage_columns).astype(np.float32)
    return sample_ids, genotype_matrix, variants


def _load_plink1(bed_path: Path) -> tuple[list[str], np.ndarray, list[_VariantDefaults]]:
    print(f"[plink] opening {bed_path}", flush=True)
    reader = open_bed(bed_path)
    sample_ids = [str(sample_id) for sample_id in reader.iid]
    n_samples = len(sample_ids)
    n_variants = len(reader.sid)
    matrix_bytes = n_samples * n_variants * 4
    print(f"[plink] samples={n_samples}  variants={n_variants}  matrix_size={matrix_bytes / 1e9:.2f} GB (float32)", flush=True)
    print(f"[plink] reading genotype matrix into memory...", flush=True)
    genotype_matrix = np.asarray(reader.read(dtype="float32"), dtype=np.float32)
    print(f"[plink] genotype_matrix loaded: shape={genotype_matrix.shape}  dtype={genotype_matrix.dtype}", flush=True)

    variants: list[_VariantDefaults] = []

    for variant_index, (variant_id, chromosome, position, allele_1, allele_2) in enumerate(
        zip(
            reader.sid,
            reader.chromosome,
            reader.bp_position,
            reader.allele_1,
            reader.allele_2,
            strict=True,
        )
    ):
        variants.append(
            _VariantDefaults(
                variant_id=str(variant_id),
                variant_class=_infer_plink_variant_class(str(allele_1), str(allele_2)),
                chromosome=str(chromosome),
                position=int(position),
                length=1.0,
                allele_frequency=_infer_dosage_allele_frequency(genotype_matrix[:, variant_index]),
                quality=1.0,
            )
        )

    if not variants:
        raise ValueError("PLINK bed contains no variants: " + str(bed_path))
    print(f"[plink] built {len(variants)} variant records", flush=True)
    return sample_ids, genotype_matrix, variants


def _build_variant_records(
    default_variants: Sequence[_VariantDefaults],
    variant_metadata_path: str | Path | None,
) -> list[VariantRecord]:
    metadata_rows_by_id: dict[str, dict[str, str]] = {}
    if variant_metadata_path is not None:
        rows = _read_delimited_rows(variant_metadata_path)
        if not rows:
            raise ValueError("Variant metadata file is empty: " + str(variant_metadata_path))
        _require_columns(rows=rows, required_columns=("variant_id",), context="variant metadata")
        for row in rows:
            variant_id = str(row["variant_id"]).strip()
            if not variant_id:
                raise ValueError("Encountered blank variant_id in variant metadata.")
            if variant_id in metadata_rows_by_id:
                raise ValueError("Duplicate variant_id in variant metadata: " + variant_id)
            metadata_rows_by_id[variant_id] = row

    records: list[VariantRecord] = []
    seen_variant_ids: set[str] = set()
    for variant in default_variants:
        if variant.variant_id in seen_variant_ids:
            raise ValueError("Duplicate variant identifier in genotype data: " + variant.variant_id)
        seen_variant_ids.add(variant.variant_id)
        metadata_row = metadata_rows_by_id.pop(variant.variant_id, None)
        records.append(_merge_variant_metadata(variant, metadata_row))

    if metadata_rows_by_id:
        extra_variant_ids = sorted(metadata_rows_by_id)
        raise ValueError(
            "Variant metadata contains identifiers that do not exist in genotype data: "
            + ", ".join(extra_variant_ids[:10])
        )
    return records


def _merge_variant_metadata(
    default_variant: _VariantDefaults,
    metadata_row: dict[str, str] | None,
) -> VariantRecord:
    if metadata_row is None:
        return VariantRecord(
            variant_id=default_variant.variant_id,
            variant_class=default_variant.variant_class,
            chromosome=default_variant.chromosome,
            position=default_variant.position,
            length=default_variant.length,
            allele_frequency=default_variant.allele_frequency,
            quality=default_variant.quality,
        )

    prior_class_members = _parse_variant_classes(metadata_row.get("prior_class_members"))
    prior_class_membership = _parse_float_list(metadata_row.get("prior_class_membership"))
    return VariantRecord(
        variant_id=_coalesce_string(metadata_row.get("variant_id"), default_variant.variant_id),
        variant_class=_parse_variant_class(metadata_row.get("variant_class"), default_variant.variant_class),
        chromosome=_coalesce_string(metadata_row.get("chromosome"), default_variant.chromosome),
        position=_parse_int_or_default(metadata_row.get("position"), default_variant.position, column_name="position"),
        length=_parse_float_or_default(metadata_row.get("length"), default_variant.length, column_name="length"),
        allele_frequency=_parse_float_or_default(
            metadata_row.get("allele_frequency"),
            default_variant.allele_frequency,
            column_name="allele_frequency",
        ),
        quality=_parse_float_or_default(metadata_row.get("quality"), default_variant.quality, column_name="quality"),
        training_support=_parse_optional_int(metadata_row.get("training_support"), column_name="training_support"),
        is_repeat=_parse_bool_or_default(metadata_row.get("is_repeat"), False, column_name="is_repeat"),
        is_copy_number=_parse_bool_or_default(metadata_row.get("is_copy_number"), False, column_name="is_copy_number"),
        prior_class_members=prior_class_members,
        prior_class_membership=prior_class_membership,
    )


def _write_predictions_and_summary(
    predictions_path: Path,
    dataset: LoadedDataset,
    model: BayesianPGS,
) -> dict[str, Any]:
    if model.config.trait_type == TraitType.BINARY:
        probabilities = model.predict_proba(dataset.genotypes, dataset.covariates)[:, 1]
        predicted_labels = (probabilities >= 0.5).astype(np.int32)
        _write_delimited_rows(
            predictions_path,
            header=("sample_id", "target", "probability", "predicted_label"),
            rows=(
                (
                    sample_id,
                    _format_float(float(target)),
                    _format_float(float(probability)),
                    str(int(predicted_label)),
                )
                for sample_id, target, probability, predicted_label in zip(
                    dataset.sample_ids,
                    dataset.targets,
                    probabilities,
                    predicted_labels,
                    strict=True,
                )
            ),
        )
        unique_targets = np.unique(dataset.targets)
        training_auc = None if unique_targets.shape[0] < 2 else float(roc_auc_score(dataset.targets, probabilities))
        return {
            "training_auc": training_auc,
            "training_log_loss": float(log_loss(dataset.targets, probabilities, labels=[0.0, 1.0])),
            "training_accuracy": float(np.mean(predicted_labels == dataset.targets)),
        }

    predictions = model.predict(dataset.genotypes, dataset.covariates)
    _write_delimited_rows(
        predictions_path,
        header=("sample_id", "target", "prediction"),
        rows=(
            (
                sample_id,
                _format_float(float(target)),
                _format_float(float(prediction)),
            )
            for sample_id, target, prediction in zip(
                dataset.sample_ids,
                dataset.targets,
                predictions,
                strict=True,
            )
        ),
    )
    residuals = dataset.targets - predictions
    return {
        "training_r2": float(r2_score(dataset.targets, predictions)),
        "training_rmse": float(np.sqrt(np.mean(residuals * residuals))),
    }


def _write_delimited_rows(
    path: Path,
    header: Sequence[str],
    rows: Iterable[Sequence[str]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)


def _read_delimited_rows(path: str | Path) -> list[dict[str, str]]:
    resolved_path = Path(path)
    with resolved_path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        delimiter = _infer_delimiter(sample)
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError("Table has no header row: " + str(resolved_path))
        return [
            {str(key): "" if value is None else str(value) for key, value in row.items()}
            for row in reader
        ]


def _infer_delimiter(sample: str) -> str:
    tab_count = sample.count("\t")
    comma_count = sample.count(",")
    if tab_count == 0 and comma_count == 0:
        raise ValueError("Expected a tab-delimited or comma-delimited file with a header row.")
    return "\t" if tab_count >= comma_count else ","


def _require_columns(
    rows: Sequence[dict[str, str]],
    required_columns: Sequence[str],
    context: str,
) -> None:
    available_columns = set(rows[0])
    missing_columns = [column_name for column_name in required_columns if column_name not in available_columns]
    if missing_columns:
        raise ValueError(
            context
            + " is missing required columns: "
            + ", ".join(missing_columns)
        )


def _resolve_genotype_format(path: Path, requested_format: str) -> str:
    if requested_format != "auto":
        return requested_format
    suffixes = path.suffixes
    if path.suffix == ".bed":
        return "plink1"
    if path.suffix == ".bcf":
        return "vcf"
    if suffixes[-2:] == [".vcf", ".gz"] or path.suffix == ".vcf":
        return "vcf"
    raise ValueError("Could not infer genotype format from path: " + str(path))


def _align_sample_ids(
    expected_sample_ids: Sequence[str],
    available_sample_ids: Sequence[str],
    context: str,
) -> np.ndarray:
    sample_index_by_id: dict[str, int] = {}
    for sample_index, sample_id in enumerate(available_sample_ids):
        if sample_id in sample_index_by_id:
            raise ValueError("Duplicate sample identifier in " + context + ": " + sample_id)
        sample_index_by_id[sample_id] = sample_index

    missing_sample_ids = [sample_id for sample_id in expected_sample_ids if sample_id not in sample_index_by_id]
    if missing_sample_ids:
        raise ValueError(
            "Sample table contains identifiers missing from "
            + context
            + ": "
            + ", ".join(missing_sample_ids[:10])
        )
    return np.asarray([sample_index_by_id[sample_id] for sample_id in expected_sample_ids], dtype=np.int32)


def _vcf_record_dosage(record: Any) -> np.ndarray:
    dosage = np.full(record.gt_types.shape[0], np.nan, dtype=np.float32)
    dosage[record.gt_types == 0] = 0.0
    dosage[record.gt_types == 1] = 1.0
    dosage[record.gt_types == 3] = 2.0
    return dosage


def _vcf_variant_id(record: Any) -> str:
    if record.ID is None or str(record.ID) == ".":
        return _vcf_variant_key(record)
    return str(record.ID)


def _vcf_variant_key(record: Any) -> str:
    return str(record.CHROM) + ":" + str(record.POS) + ":" + str(record.REF) + ":" + str(record.ALT[0])


def _infer_vcf_variant_class(record: Any) -> VariantClass:
    if record.is_snp:
        return VariantClass.SNV
    if record.is_indel and not record.is_sv:
        return VariantClass.SMALL_INDEL
    variant_token = _normalize_variant_token(record.INFO.get("SVTYPE"))
    if variant_token is None:
        variant_token = _normalize_variant_token(record.ALT[0])
    if variant_token is None:
        return VariantClass.OTHER_COMPLEX_SV
    return _structural_variant_class_from_token(variant_token, _infer_vcf_length(record))


def _infer_plink_variant_class(allele_1: str, allele_2: str) -> VariantClass:
    structural_token = _symbolic_variant_token(allele_1, allele_2)
    if structural_token is not None:
        return _structural_variant_class_from_token(structural_token, length=1.0)
    if len(allele_1) == 1 and len(allele_2) == 1:
        return VariantClass.SNV
    return VariantClass.SMALL_INDEL


def _infer_vcf_length(record: Any) -> float:
    svlen_value = record.INFO.get("SVLEN")
    if svlen_value is not None:
        if isinstance(svlen_value, (tuple, list)):
            return float(abs(float(svlen_value[0])))
        return float(abs(float(svlen_value)))
    if record.is_snp:
        return 1.0
    if record.end is not None and int(record.end) >= int(record.POS):
        return float(int(record.end) - int(record.POS) + 1)
    return float(max(len(record.REF), len(record.ALT[0])))


def _infer_vcf_allele_frequency(record: Any, dosage: np.ndarray) -> float:
    af_value = record.INFO.get("AF")
    if af_value is not None:
        if isinstance(af_value, (tuple, list)):
            return float(af_value[0])
        return float(af_value)
    return _infer_dosage_allele_frequency(dosage)


def _infer_dosage_allele_frequency(dosage: np.ndarray) -> float:
    if np.all(np.isnan(dosage)):
        return 0.0
    return float(np.nanmean(dosage) / 2.0)


def _normalize_quality(value: Any) -> float:
    if value is None:
        return 1.0
    quality = float(value)
    if np.isnan(quality):
        return 1.0
    return quality


def _parse_variant_class(value: str | None, default: VariantClass) -> VariantClass:
    if value is None or not value.strip():
        return default
    return VariantClass(value.strip())


def _parse_variant_classes(value: str | None) -> tuple[VariantClass, ...]:
    if value is None or not value.strip():
        return ()
    return tuple(VariantClass(member.strip()) for member in value.split(",") if member.strip())


def _parse_float_list(value: str | None) -> tuple[float, ...]:
    if value is None or not value.strip():
        return ()
    return tuple(float(member.strip()) for member in value.split(",") if member.strip())


def _parse_optional_int(value: str | None, column_name: str) -> int | None:
    if value is None or not value.strip():
        return None
    return int(_parse_float(value, column_name=column_name))


def _parse_int_or_default(value: str | None, default: int, column_name: str) -> int:
    if value is None or not value.strip():
        return default
    return int(_parse_float(value, column_name=column_name))


def _parse_float_or_default(value: str | None, default: float, column_name: str) -> float:
    if value is None or not value.strip():
        return default
    return _parse_float(value, column_name=column_name)


def _parse_bool_or_default(value: str | None, default: bool, column_name: str) -> bool:
    if value is None or not value.strip():
        return default
    normalized_value = value.strip().lower()
    if normalized_value in {"1", "true", "yes"}:
        return True
    if normalized_value in {"0", "false", "no"}:
        return False
    raise ValueError("Could not parse boolean value for " + column_name + ": " + value)


def _coalesce_string(value: str | None, default: str) -> str:
    if value is None or not value.strip():
        return default
    return value.strip()


def _parse_float(value: str, column_name: str) -> float:
    try:
        return float(value)
    except ValueError as error:
        raise ValueError("Could not parse float value for " + column_name + ": " + value) from error


def _format_float(value: float) -> str:
    return format(value, ".8g")


def _normalize_variant_token(value: Any) -> str | None:
    if value is None:
        return None
    normalized_value = str(value).strip().upper()
    if not normalized_value:
        return None
    if normalized_value.startswith("<") and normalized_value.endswith(">"):
        normalized_value = normalized_value[1:-1]
    return normalized_value


def _symbolic_variant_token(allele_1: str, allele_2: str) -> str | None:
    for allele in (allele_1, allele_2):
        normalized_allele = _normalize_variant_token(allele)
        if normalized_allele is None:
            continue
        if any(token in normalized_allele for token in ("DEL", "DUP", "INS", "INV", "BND", "STR", "VNTR", "ME")):
            return normalized_allele
    return None


def _structural_variant_class_from_token(token: str, length: float) -> VariantClass:
    if "DEL" in token:
        return VariantClass.DELETION_LONG if length >= SV_LENGTH_THRESHOLD else VariantClass.DELETION_SHORT
    if "DUP" in token or "CNV" in token:
        return VariantClass.DUPLICATION_LONG if length >= SV_LENGTH_THRESHOLD else VariantClass.DUPLICATION_SHORT
    if "INS" in token or "ME" in token:
        return VariantClass.INSERTION_MEI
    if "INV" in token or "BND" in token:
        return VariantClass.INVERSION_BND_COMPLEX
    if "STR" in token or "VNTR" in token or "REPEAT" in token:
        return VariantClass.STR_VNTR_REPEAT
    return VariantClass.OTHER_COMPLEX_SV

"""All of Us orchestration: download VCFs, prepare phenotypes, merge PCs, run one unified fit."""
from __future__ import annotations

import csv
import gc
import gzip
import json
import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sv_pgs.all_of_us import AllOfUsDiseaseRequest, prepare_all_of_us_disease_sample_table, resolve_disease_definition
from sv_pgs.config import ModelConfig, TraitType, VariantClass
from sv_pgs.data import NESTED_PATH_DELIMITER
from sv_pgs.io import load_multi_vcf_dataset_from_files, run_training_pipeline
from sv_pgs.progress import log, mem

_LOCAL_CACHE_DIRNAME = ".sv_pgs_cache"
_AOU_SV_VCF_CACHE_SUBDIR = "aou_sv_vcfs"
_AOU_VARIANT_METADATA_SCHEMA_VERSION = 1
_AOU_VARIANT_METADATA_FILENAME = "variant_metadata.tsv.gz"
_AOU_VARIANT_METADATA_MANIFEST_FILENAME = "variant_metadata.manifest.json"

# ---------------------------------------------------------------------------
# AoU paths
# ---------------------------------------------------------------------------

def _cdr_storage_path() -> str:
    value = os.environ.get("CDR_STORAGE_PATH")
    if not value:
        raise RuntimeError("CDR_STORAGE_PATH is not set. Are you on an All of Us workbench?")
    return value


def _google_project() -> str:
    value = os.environ.get("GOOGLE_PROJECT")
    if not value:
        raise RuntimeError("GOOGLE_PROJECT is not set. Are you on an All of Us workbench?")
    return value


def sv_vcf_dir() -> str:
    return f"{_cdr_storage_path()}/wgs/short_read/structural_variants/vcf/full"


def resolve_ancestry_predictions_path() -> str:
    return f"{_cdr_storage_path()}/wgs/short_read/snpindel/aux/ancestry/ancestry_preds.tsv"


def local_ancestry_predictions_path(work_dir: Path) -> Path:
    return work_dir / "ancestry_preds.tsv"


def sv_vcf_name(chromosome: int) -> str:
    return f"AoU_srWGS_SV.v8.chr{chromosome}.vcf.gz"


def local_sv_vcf_cache_dir(work_dir: Path) -> Path:
    return work_dir.parent / _LOCAL_CACHE_DIRNAME / _AOU_SV_VCF_CACHE_SUBDIR


def local_sv_vcf_path(chromosome: int, work_dir: Path) -> Path:
    return local_sv_vcf_cache_dir(work_dir) / sv_vcf_name(chromosome)


def aou_variant_metadata_path(work_dir: Path) -> Path:
    return work_dir / _AOU_VARIANT_METADATA_FILENAME


def _aou_variant_metadata_manifest_path(work_dir: Path) -> Path:
    return work_dir / _AOU_VARIANT_METADATA_MANIFEST_FILENAME


# ---------------------------------------------------------------------------
# gsutil helpers
# ---------------------------------------------------------------------------

def _check_disk_space(path: Path, required_bytes: int) -> None:
    """Raise if the filesystem doesn't have enough free space."""
    stat = shutil.disk_usage(str(path))
    if stat.free < required_bytes:
        free_gb = stat.free / 1e9
        need_gb = required_bytes / 1e9
        raise RuntimeError(
            f"Not enough disk space: {free_gb:.1f} GB free, need {need_gb:.1f} GB "
            f"at {path}"
        )


def _gsutil_cp(src: str, dst: str) -> None:
    """Download with gsutil, showing real-time progress via -m flag."""
    cmd = ["gsutil", "-u", _google_project(), "-m", "cp", src, dst]
    log(f"  downloading {src}")
    # Stream output in real time so user sees progress
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if process.stdout is not None:
        for line in process.stdout:
            stripped = line.strip()
            if stripped:
                log(f"    {stripped}")
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"gsutil cp failed (exit {process.returncode}): {src}")


def _gsutil_size(path: str) -> int:
    """Get the size of a remote GCS object in bytes."""
    cmd = ["gsutil", "-u", _google_project(), "du", "-s", path]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return int(result.stdout.strip().split()[0])


def _download_gcs_object_if_missing(remote_path: str, local_path: Path) -> None:
    if local_path.exists():
        return
    local_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = local_path.with_name(local_path.name + ".partial")
    partial_path.unlink(missing_ok=True)
    try:
        _gsutil_cp(remote_path, str(partial_path))
        partial_path.replace(local_path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise


def download_sv_vcf(chromosome: int, work_dir: Path) -> Path:
    """Download one SV VCF + index when needed and return the local VCF path."""
    remote_dir = sv_vcf_dir()
    name = sv_vcf_name(chromosome)
    local_vcf = local_sv_vcf_path(chromosome, work_dir)
    local_vcf.parent.mkdir(parents=True, exist_ok=True)
    local_tbi = local_vcf.parent / f"{name}.tbi"
    vcf_remote = f"{remote_dir}/{name}"
    tbi_remote = f"{remote_dir}/{name}.tbi"

    missing_downloads: list[tuple[str, Path, str]] = []
    if not local_vcf.exists():
        missing_downloads.append((vcf_remote, local_vcf, "VCF"))
    if not local_tbi.exists():
        missing_downloads.append((tbi_remote, local_tbi, "index"))

    if not missing_downloads:
        log(f"  chr{chromosome}: VCF already present in local cache")
        return local_vcf

    required_bytes = sum(_gsutil_size(remote) for remote, _, _ in missing_downloads)
    cache_dir = local_sv_vcf_cache_dir(work_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    _check_disk_space(cache_dir, required_bytes)
    missing_labels = " + ".join(label for _, _, label in missing_downloads)
    log(
        f"  chr{chromosome}: downloading missing {missing_labels} into cache "
        + f"{cache_dir} ({required_bytes/1e9:.1f} GB)"
    )
    for remote, local_path, _ in missing_downloads:
        _download_gcs_object_if_missing(remote, local_path)
    return local_vcf


# ---------------------------------------------------------------------------
# Ancestry / PC merging
# ---------------------------------------------------------------------------

def download_ancestry_preds(work_dir: Path) -> Path:
    """Download the AoU ancestry predictions file (contains per-sample PCs)."""
    remote = resolve_ancestry_predictions_path()
    local = local_ancestry_predictions_path(work_dir)
    if local.exists():
        log(f"  ancestry predictions already present: {local.name}")
        return local
    log(f"  downloading ancestry predictions: {remote}")
    _gsutil_cp(remote, str(local))
    return local

def release_process_memory() -> None:
    gc.collect()
    try:
        import jax

        clear_caches = getattr(jax, "clear_caches", None)
        if callable(clear_caches):
            clear_caches()
    except Exception:
        pass
    try:
        import cupy as cp

        cp.cuda.Device().synchronize()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def _parse_pca_features_column(series: pd.Series, n_pcs: int) -> tuple[pd.DataFrame, list[str]]:
    """Parse a compound pca_features column like '[0.01,-0.02,...]' into PC1..PCn."""
    pc_names = [f"PC{i+1}" for i in range(n_pcs)]
    n_failed = 0

    def _parse_row(val: str) -> list[float]:
        nonlocal n_failed
        if not isinstance(val, str) or not val.strip():
            n_failed += 1
            return [float("nan")] * n_pcs
        cleaned = val.strip()
        try:
            if cleaned.startswith("["):
                values = json.loads(cleaned)
            else:
                values = [float(x.strip()) for x in cleaned.split(",") if x.strip()]
            return [float(values[i]) if i < len(values) else float("nan") for i in range(n_pcs)]
        except (json.JSONDecodeError, ValueError, IndexError):
            n_failed += 1
            return [float("nan")] * n_pcs

    # Log the format of the first non-empty value (truncated, no individual data)
    first_val = series.dropna().iloc[0] if len(series.dropna()) > 0 else ""
    preview = str(first_val)[:80] + ("..." if len(str(first_val)) > 80 else "")
    log(f"  pca_features format preview (first row, truncated): {preview}")

    parsed = series.apply(_parse_row)
    df = pd.DataFrame(parsed.tolist(), columns=pc_names, index=series.index)

    n_total = len(series)
    n_parsed = n_total - n_failed
    log(f"  parsed: {n_parsed}/{n_total} rows successfully ({n_failed} failed)")
    if n_parsed > 0:
        # Log aggregate stats only (no individual values)
        for col in pc_names[:3]:
            vals = df[col].dropna()
            if len(vals) > 0:
                log(f"    {col}: mean={vals.mean():.4f}  std={vals.std():.4f}  min={vals.min():.4f}  max={vals.max():.4f}")

    return df, pc_names


def merge_pcs_into_sample_table(
    sample_table_path: Path,
    ancestry_path: Path,
    output_path: Path,
    n_pcs: int = 10,
) -> tuple[Path, list[str]]:
    """Merge top N PCs from ancestry file into the sample table. Returns (output_path, pc_column_names)."""
    if output_path.exists():
        existing = pd.read_csv(output_path, sep="\t", nrows=0)
        existing_pc_cols = sorted(
            [c for c in existing.columns if c.startswith("PC") and c[2:].isdigit()],
            key=lambda x: int(x[2:]),
        )
        merged_mtime = output_path.stat().st_mtime
        inputs_mtime = max(sample_table_path.stat().st_mtime, ancestry_path.stat().st_mtime)
        if merged_mtime > inputs_mtime and len(existing_pc_cols) == int(n_pcs):
            log("  PC-merged table up to date, skipping recomputation")
            return output_path, existing_pc_cols
        log(f"  merged sample table already exists but is stale; recomputing with n_pcs={n_pcs}")

    log(f"  loading sample table: {sample_table_path}")
    samples = pd.read_csv(sample_table_path, sep="\t", dtype={"sample_id": str, "person_id": str})

    log(f"  loading ancestry predictions: {ancestry_path}")
    ancestry = pd.read_csv(ancestry_path, sep="\t", dtype=str)
    log(f"  ancestry columns ({len(ancestry.columns)}): {list(ancestry.columns)}")
    log(f"  ancestry rows: {len(ancestry)}, sample table rows: {len(samples)}")

    # Find the ID column in ancestry
    id_col = None
    for candidate in ["research_id", "person_id", "sample_id", "s"]:
        if candidate in ancestry.columns:
            id_col = candidate
            break
    if id_col is None:
        raise RuntimeError(f"No ID column found in ancestry file. Columns: {list(ancestry.columns)}")

    # Extract PCs: either individual columns (PC1, PC2, ...) or a compound pca_features column
    if "pca_features" in ancestry.columns:
        log(f"  parsing compound pca_features column into PC1..PC{n_pcs}")
        pc_df, pc_cols = _parse_pca_features_column(ancestry["pca_features"], n_pcs)
        for col in pc_cols:
            ancestry[col] = pc_df[col].values
    else:
        pc_cols = sorted(
            [c for c in ancestry.columns if c.startswith("PC") and c[2:].isdigit()],
            key=lambda x: int(x[2:]),
        )[:n_pcs]
        if not pc_cols:
            raise RuntimeError(f"No PC columns found in ancestry file. Columns: {list(ancestry.columns)}")
        for col in pc_cols:
            ancestry[col] = pd.to_numeric(ancestry[col], errors="coerce")

    log(f"  extracted {len(pc_cols)} PCs: {pc_cols}")

    # Add age^2
    if "age_at_observation_start" in samples.columns:
        samples["age_at_observation_start"] = pd.to_numeric(samples["age_at_observation_start"], errors="coerce")
        samples["age_squared"] = samples["age_at_observation_start"] ** 2
        log("  added age_squared covariate")

    # Check ID overlap before merging (aggregate counts only, no individual IDs)
    ancestry_ids = set(ancestry[id_col].dropna().astype(str))
    sample_id_sets = {
        column: set(samples[column].dropna().astype(str))
        for column in ("person_id", "sample_id")
        if column in samples.columns
    }
    if not sample_id_sets:
        raise RuntimeError("Sample table must contain at least one of: person_id, sample_id")

    overlap_counts = {column: len(ancestry_ids & ids) for column, ids in sample_id_sets.items()}
    log(f"  ancestry ID column: {id_col} ({len(ancestry_ids)} unique IDs)")
    for column in ("person_id", "sample_id"):
        if column in sample_id_sets:
            log(f"  overlap with {column}: {overlap_counts[column]}/{len(sample_id_sets[column])}")

    merge_key = max(overlap_counts, key=lambda column: overlap_counts[column])
    if overlap_counts[merge_key] == 0:
        available_keys = ", ".join(sample_id_sets)
        raise RuntimeError(
            f"No ID overlap between ancestry column '{id_col}' and sample table columns: {available_keys}"
        )
    log(f"  merging on {merge_key} ({overlap_counts[merge_key]} matches)")

    ancestry_subset = ancestry[[id_col] + pc_cols].copy()
    ancestry_subset = ancestry_subset.rename(columns={id_col: merge_key})
    merged = samples.merge(ancestry_subset, on=merge_key, how="left")
    n_with_pcs = int(merged[pc_cols[0]].notna().sum())

    log(f"  merged: {len(merged)} rows, {n_with_pcs} with PCs ({100*n_with_pcs/max(len(merged),1):.0f}%)")
    if n_with_pcs == 0:
        raise RuntimeError(
            "Ancestry merge produced zero rows with PCs. "
            + f"merge_key={merge_key} ancestry_path={ancestry_path} sample_table_path={sample_table_path}"
        )

    merged.to_csv(output_path, sep="\t", index=False)
    return output_path, pc_cols


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def _aou_run_metadata_path(work_dir: Path) -> Path:
    return work_dir / "aou_run_metadata.json"


def _validate_aou_chromosomes(chromosomes: list[int]) -> list[int]:
    if not chromosomes:
        raise ValueError("chromosomes cannot be empty.")
    normalized = [int(chromosome) for chromosome in chromosomes]
    invalid = [chromosome for chromosome in normalized if chromosome < 1 or chromosome > 22]
    if invalid:
        raise ValueError(f"chromosomes must be autosomes 1-22; got {invalid}")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"chromosomes must be unique; got {normalized}")
    return normalized


def _aou_vcf_source_signature(vcf_path: Path) -> dict[str, object]:
    stat = vcf_path.stat()
    return {
        "path": str(vcf_path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _normalize_info_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (tuple, list)):
        normalized_values: list[str] = []
        for member in value:
            normalized_values.extend(_normalize_info_strings(member))
        return normalized_values
    text = str(value).strip()
    if not text or text == ".":
        return []
    pieces = re.split(r"[,&|]", text)
    return [piece.strip() for piece in pieces if piece.strip() and piece.strip() != "."]


def _normalize_info_tokens(value: Any) -> list[str]:
    return [token.upper().replace(" ", "_") for token in _normalize_info_strings(value)]


def _info_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        numeric_values = [numeric_value for numeric_value in (_info_float(member) for member in value) if numeric_value is not None]
        if not numeric_values:
            return None
        return float(numeric_values[0])
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric_value):
        return None
    return numeric_value


def _info_best_numeric(info: Any, candidate_keys: tuple[str, ...]) -> float | None:
    for key in candidate_keys:
        try:
            raw_value = info.get(key)
        except Exception:
            raw_value = None
        numeric_value = _info_float(raw_value)
        if numeric_value is not None:
            return numeric_value
    return None


def _membership_weights(tokens: list[str]) -> str:
    if not tokens:
        return ""
    unique_tokens = sorted(set(tokens))
    weight = 1.0 / float(len(unique_tokens))
    return ",".join(f"{token.lower()}={weight:.8g}" for token in unique_tokens)


def _count_if_present(tokens: list[str]) -> str:
    if not tokens:
        return ""
    return str(len(sorted(set(tokens))))


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return ""
    if math.isnan(value):
        return ""
    return format(float(value), ".8g")


def _boolean_string(value: bool) -> str:
    return "true" if value else "false"


def _string_contains_any(text: str, needles: tuple[str, ...]) -> bool:
    normalized_text = text.lower()
    return any(needle in normalized_text for needle in needles)


def _strongest_effect_label(
    *,
    lof_tokens: list[str],
    copy_gain_tokens: list[str],
    promoter_tokens: list[str],
    intronic_tokens: list[str],
    utr_tokens: list[str],
    noncoding_span_tokens: list[str],
    noncoding_breakpoint_tokens: list[str],
    is_repeat: bool,
    is_copy_number: bool,
) -> str:
    if lof_tokens:
        return "lof"
    if copy_gain_tokens:
        return "copy_gain"
    if promoter_tokens:
        return "promoter"
    if intronic_tokens:
        return "intronic"
    if utr_tokens:
        return "utr"
    if noncoding_breakpoint_tokens:
        return "regulatory_breakpoint"
    if noncoding_span_tokens:
        return "regulatory_span"
    if is_repeat:
        return "repeat"
    if is_copy_number:
        return "copy_number"
    return "other"


def _strongest_effect_path(label: str) -> str:
    if label == "lof":
        return NESTED_PATH_DELIMITER.join(("genic", "loss_of_function"))
    if label == "copy_gain":
        return NESTED_PATH_DELIMITER.join(("genic", "copy_gain"))
    if label in {"promoter", "regulatory_breakpoint", "regulatory_span"}:
        return NESTED_PATH_DELIMITER.join(("regulatory", label))
    if label in {"intronic", "utr"}:
        return NESTED_PATH_DELIMITER.join(("genic", label))
    if label == "repeat":
        return NESTED_PATH_DELIMITER.join(("repeat", "context"))
    if label == "copy_number":
        return NESTED_PATH_DELIMITER.join(("copy_number", "structural"))
    return NESTED_PATH_DELIMITER.join(("other", "structural"))


def build_aou_sv_variant_metadata(
    *,
    vcf_paths: list[Path],
    output_path: Path,
) -> Path:
    from sv_pgs.io import _open_vcf_reader, _variant_defaults_from_vcf_record

    manifest_path = _aou_variant_metadata_manifest_path(output_path.parent)
    source_signatures = [_aou_vcf_source_signature(vcf_path) for vcf_path in vcf_paths]
    if output_path.exists() and manifest_path.exists():
        try:
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                int(existing_manifest.get("schema_version", -1)) == _AOU_VARIANT_METADATA_SCHEMA_VERSION
                and existing_manifest.get("source_vcfs") == source_signatures
            ):
                log(f"  variant metadata already up to date: {output_path}")
                return output_path
        except Exception:
            pass

    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    header = (
        "variant_id",
        "variant_class",
        "chromosome",
        "position",
        "length",
        "allele_frequency",
        "quality",
        "is_copy_number",
        "is_repeat",
        "prior_binary__is_copy_number",
        "prior_binary__is_repeat",
        "prior_binary__predicted_lof",
        "prior_binary__predicted_copy_gain",
        "prior_binary__predicted_promoter",
        "prior_binary__predicted_intronic",
        "prior_binary__predicted_utr",
        "prior_binary__predicted_intergenic",
        "prior_binary__pathogenic_or_disease_associated",
        "prior_continuous__sv_length_log10",
        "prior_continuous__site_quality",
        "prior_continuous__cohort_allele_frequency",
        "prior_continuous__algorithm_count",
        "prior_continuous__copy_number_quality",
        "prior_continuous__constraint_score",
        "prior_continuous__conservation_score",
        "prior_continuous__lof_gene_count",
        "prior_continuous__copy_gain_gene_count",
        "prior_continuous__promoter_gene_count",
        "prior_continuous__intronic_gene_count",
        "prior_continuous__utr_gene_count",
        "prior_membership__calling_algorithms",
        "prior_membership__noncoding_span",
        "prior_membership__noncoding_breakpoint",
        "prior_categorical__strongest_effect",
        "prior_nested__functional_context",
    )
    temporary_output_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with gzip.open(temporary_output_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header)
        for vcf_path in vcf_paths:
            reader = _open_vcf_reader(vcf_path)
            try:
                for record in reader:
                    if len(record.ALT) != 1:
                        continue
                    variant_defaults = _variant_defaults_from_vcf_record(record)
                    info = record.INFO
                    variant_class = variant_defaults.variant_class
                    is_copy_number = variant_class in {
                        VariantClass.DELETION_SHORT,
                        VariantClass.DELETION_LONG,
                        VariantClass.DUPLICATION_SHORT,
                        VariantClass.DUPLICATION_LONG,
                    }
                    repeat_tokens = [
                        key.lower()
                        for key in ("REPEATMASKER", "REPEATS", "TRF", "SEGDUP", "REPEAT_CLASS")
                        if info.get(key) not in (None, False, ".", "")
                    ]
                    is_repeat = (
                        variant_class == VariantClass.STR_VNTR_REPEAT
                        or bool(repeat_tokens)
                    )
                    lof_tokens = _normalize_info_tokens(info.get("PREDICTED_LOF"))
                    copy_gain_tokens = _normalize_info_tokens(info.get("PREDICTED_COPY_GAIN"))
                    promoter_tokens = _normalize_info_tokens(info.get("PREDICTED_PROMOTER"))
                    intronic_tokens = _normalize_info_tokens(info.get("PREDICTED_INTRONIC"))
                    utr_tokens = _normalize_info_tokens(info.get("PREDICTED_UTR"))
                    intergenic_tokens = _normalize_info_tokens(info.get("PREDICTED_INTERGENIC"))
                    noncoding_span_tokens = _normalize_info_tokens(info.get("PREDICTED_NONCODING_SPAN"))
                    noncoding_breakpoint_tokens = _normalize_info_tokens(info.get("PREDICTED_NONCODING_BREAKPOINT"))
                    algorithm_tokens = _normalize_info_tokens(info.get("ALGORITHMS"))
                    copy_number_quality = _info_best_numeric(info, ("CNQ", "RD_CNQ", "QS"))
                    constraint_score = _info_best_numeric(info, ("LOEUF", "LOEUF_MIN", "MIN_LOEUF", "CONSTRAINT", "CONSTRAINT_SCORE"))
                    conservation_score = _info_best_numeric(info, ("PHYLOP", "PHASTCONS", "GERP", "CONSERVATION"))
                    pathogenic_or_disease_associated = any(
                        info.get(key) not in (None, False, ".", "", "0", "false", "FALSE", "none", "None")
                        for key in (
                            "CLINVAR_PATHOGENIC",
                            "PATHOGENIC",
                            "PATHOGENICITY",
                            "DISEASE_ASSOCIATION",
                            "OMIM",
                        )
                    )
                    strongest_effect = _strongest_effect_label(
                        lof_tokens=lof_tokens,
                        copy_gain_tokens=copy_gain_tokens,
                        promoter_tokens=promoter_tokens,
                        intronic_tokens=intronic_tokens,
                        utr_tokens=utr_tokens,
                        noncoding_span_tokens=noncoding_span_tokens,
                        noncoding_breakpoint_tokens=noncoding_breakpoint_tokens,
                        is_repeat=is_repeat,
                        is_copy_number=is_copy_number,
                    )
                    writer.writerow(
                        (
                            variant_defaults.variant_id,
                            variant_defaults.variant_class.value,
                            variant_defaults.chromosome,
                            str(variant_defaults.position),
                            format(variant_defaults.length, ".8g"),
                            format(variant_defaults.allele_frequency, ".8g"),
                            format(variant_defaults.quality, ".8g"),
                            _boolean_string(is_copy_number),
                            _boolean_string(is_repeat),
                            _boolean_string(is_copy_number),
                            _boolean_string(is_repeat),
                            _boolean_string(bool(lof_tokens)),
                            _boolean_string(bool(copy_gain_tokens)),
                            _boolean_string(bool(promoter_tokens)),
                            _boolean_string(bool(intronic_tokens)),
                            _boolean_string(bool(utr_tokens)),
                            _boolean_string(bool(intergenic_tokens)),
                            _boolean_string(pathogenic_or_disease_associated),
                            format(math.log10(max(float(variant_defaults.length), 1.0)), ".8g"),
                            format(variant_defaults.quality, ".8g"),
                            format(variant_defaults.allele_frequency, ".8g"),
                            format(float(len(sorted(set(algorithm_tokens)))), ".8g") if algorithm_tokens else "",
                            _format_optional_float(copy_number_quality),
                            _format_optional_float(constraint_score),
                            _format_optional_float(conservation_score),
                            _count_if_present(lof_tokens),
                            _count_if_present(copy_gain_tokens),
                            _count_if_present(promoter_tokens),
                            _count_if_present(intronic_tokens),
                            _count_if_present(utr_tokens),
                            _membership_weights(algorithm_tokens),
                            _membership_weights(noncoding_span_tokens),
                            _membership_weights(noncoding_breakpoint_tokens),
                            strongest_effect,
                            _strongest_effect_path(strongest_effect),
                        )
                    )
                    row_count += 1
            finally:
                reader.close()
    temporary_output_path.replace(output_path)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": _AOU_VARIANT_METADATA_SCHEMA_VERSION,
                "source_vcfs": source_signatures,
                "row_count": row_count,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"  AoU variant metadata written: {output_path} ({row_count} variants)")
    return output_path


def _build_aou_run_metadata(
    *,
    disease: str,
    chromosomes: list[int],
    n_pcs: int,
    pc_cols: list[str],
    covariates: list[str],
    max_outer_iterations: int,
    pipeline_validation_fraction: float,
    pipeline_validation_min_samples: int,
    random_seed: int,
    variant_metadata_schema_version: int,
) -> dict[str, object]:
    return {
        "disease": disease,
        "chromosomes": chromosomes,
        "requested_n_pcs": n_pcs,
        "effective_pc_columns": pc_cols,
        "covariates": covariates,
        "max_outer_iterations": max_outer_iterations,
        "pipeline_validation_fraction": pipeline_validation_fraction,
        "pipeline_validation_min_samples": pipeline_validation_min_samples,
        "random_seed": random_seed,
        "variant_metadata_schema_version": variant_metadata_schema_version,
    }

DEFAULT_COVARIATES = [
    "age_at_observation_start",
    "age_squared",
    "gender_concept_id",
    "race_concept_id",
    "ethnicity_concept_id",
]


def run_all_of_us(
    disease: str,
    chromosomes: list[int],
    output_base: str,
    n_pcs: int = 10,
    max_outer_iterations: int = 30,
    pipeline_validation_fraction: float = 0.0,
    pipeline_validation_min_samples: int = 0,
    random_seed: int = 0,
) -> None:
    """Full AoU pipeline: download requested chromosomes, merge them, and run one fit."""
    chromosomes = _validate_aou_chromosomes(chromosomes)

    import os

    # Validate disease
    disease_def = resolve_disease_definition(disease)
    work_dir = Path(output_base)
    work_dir.mkdir(parents=True, exist_ok=True)

    from sv_pgs.progress import set_log_file
    log_path = work_dir / f"{disease_def.canonical_name}.{time.strftime('%Y%m%d_%H%M%S')}.log"
    set_log_file(log_path)

    from sv_pgs.genotype import require_gpu
    require_gpu()

    log(
        "=== ALL OF US PIPELINE ===  "
        + f"disease={disease_def.canonical_name}  chromosomes={chromosomes}  n_pcs={n_pcs}  cpus={os.cpu_count()}"
    )
    log(f"  ICD-9: {disease_def.icd9_prefixes}  ICD-10: {disease_def.icd10_prefixes}")
    log(f"  output: {work_dir}")
    config = ModelConfig(
        max_outer_iterations=max_outer_iterations,
        pipeline_validation_fraction=pipeline_validation_fraction,
        pipeline_validation_min_samples=pipeline_validation_min_samples,
        random_seed=random_seed,
    )

    # Status summary: what's done vs what's left
    log("=== STATUS CHECK ===")
    sample_table_path = work_dir / f"{disease_def.canonical_name}.samples.tsv"
    merged_path = work_dir / f"{disease_def.canonical_name}.samples.with_pcs.tsv"
    log(f"  phenotype table: {'DONE' if sample_table_path.exists() else 'NEEDED'}")
    log(f"  PC-merged table: {'DONE' if merged_path.exists() else 'NEEDED'}")
    ancestry_local = local_ancestry_predictions_path(work_dir)
    variant_metadata_path = aou_variant_metadata_path(work_dir)
    log(f"  ancestry file:   {'DONE' if ancestry_local.exists() else 'NEEDED'}")
    log(f"  variant metadata:{'DONE' if variant_metadata_path.exists() else 'NEEDED'}")

    # Cache key no longer depends on sample indices — only on VCF content + config.
    cached_chrs = []
    uncached_chrs = []
    from sv_pgs.io import _is_vcf_cache_bundle_complete, _vcf_cache_dir, _vcf_cache_paths
    for chrom in chromosomes:
        vcf_path = local_sv_vcf_path(chrom, work_dir)
        if not vcf_path.exists():
            uncached_chrs.append(f"chr{chrom}(no VCF)")
            continue
        cache_dir = _vcf_cache_dir(vcf_path)
        cache_paths = _vcf_cache_paths(vcf_path, config)
        has_cache = _is_vcf_cache_bundle_complete(cache_paths)
        has_partial = (cache_dir / f"{cache_paths.key}.inc.progress.json").exists() if cache_dir.exists() else False
        has_tmp = (cache_dir / f"{cache_paths.key}.tmp_parallel").exists() if cache_dir.exists() else False
        if has_cache:
            cached_chrs.append(f"chr{chrom}")
        elif has_partial:
            uncached_chrs.append(f"chr{chrom}(partial)")
        elif has_tmp:
            uncached_chrs.append(f"chr{chrom}(parallel-partial)")
        else:
            uncached_chrs.append(f"chr{chrom}")
    log(f"  VCF cached ({len(cached_chrs)}): {', '.join(cached_chrs) if cached_chrs else 'none'}")
    log(f"  VCF needed ({len(uncached_chrs)}): {', '.join(uncached_chrs) if uncached_chrs else 'none — all cached!'}")
    summary_path = work_dir / "summary.json.gz"
    log(f"  model fitted:    {'DONE' if summary_path.exists() else 'NEEDED'}")
    log("===================")

    # Step 1: Prepare phenotype
    log("=== STEP 1: Prepare phenotype ===")
    sample_table_path = work_dir / f"{disease_def.canonical_name}.samples.tsv"
    if not sample_table_path.exists():
        prepare_all_of_us_disease_sample_table(
            request=AllOfUsDiseaseRequest(disease=disease),
            output_path=sample_table_path,
        )
    else:
        log(f"  sample table already exists: {sample_table_path}")

    # Step 2: Download and merge PCs
    log("=== STEP 2: Merge genomic PCs ===")
    ancestry_path = download_ancestry_preds(work_dir)
    merged_path = work_dir / f"{disease_def.canonical_name}.samples.with_pcs.tsv"
    merged_path, pc_cols = merge_pcs_into_sample_table(
        sample_table_path=sample_table_path,
        ancestry_path=ancestry_path,
        output_path=merged_path,
        n_pcs=n_pcs,
    )

    # Build covariate list
    covariates = DEFAULT_COVARIATES + pc_cols
    log(f"  covariates ({len(covariates)}): {covariates}")

    summary_path = work_dir / "summary.json.gz"
    run_metadata_path = _aou_run_metadata_path(work_dir)
    run_metadata = _build_aou_run_metadata(
        disease=disease_def.canonical_name,
        chromosomes=chromosomes,
        n_pcs=n_pcs,
        pc_cols=pc_cols,
        covariates=covariates,
        max_outer_iterations=max_outer_iterations,
        pipeline_validation_fraction=pipeline_validation_fraction,
        pipeline_validation_min_samples=pipeline_validation_min_samples,
        random_seed=random_seed,
        variant_metadata_schema_version=_AOU_VARIANT_METADATA_SCHEMA_VERSION,
    )
    if summary_path.exists():
        if run_metadata_path.exists():
            existing_run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
            if existing_run_metadata == run_metadata:
                log(f"  unified fit already exists with matching configuration: {summary_path}")
                return
            log("  existing unified fit configuration differs; rerunning and overwriting outputs")
        else:
            log("  unified fit exists without run metadata; rerunning and overwriting outputs")

    log("=== STEP 3: Download chromosome VCFs ===")
    vcf_paths: list[Path] = []
    dataset = None
    try:
        for chrom in chromosomes:
            vcf_paths.append(download_sv_vcf(chrom, work_dir))

        log("=== STEP 3.5: Parallel VCF precache ===")
        from sv_pgs.io import precache_vcfs_parallel
        try:
            precache_vcfs_parallel(vcf_paths, config)
        except Exception as exc:
            raise RuntimeError("parallel VCF precache failed") from exc

        log("=== STEP 3.6: Build variant metadata priors ===")
        variant_metadata_path = build_aou_sv_variant_metadata(
            vcf_paths=vcf_paths,
            output_path=variant_metadata_path,
        )

        log("=== STEP 4: Load unified genome-wide dataset ===")
        dataset = load_multi_vcf_dataset_from_files(
            genotype_paths=vcf_paths,
            config=config,
            sample_table_path=str(merged_path),
            sample_id_column="auto",
            target_column="target",
            covariate_columns=covariates,
            variant_metadata_path=variant_metadata_path,
        )
        inferred_trait = TraitType.BINARY if len(np.unique(dataset.targets)) <= 2 else TraitType.QUANTITATIVE
        config.trait_type = inferred_trait

        log("=== STEP 5: Fit unified genome-wide model ===")
        log(f"  freeing dataset overhead before fit...  mem={mem()}")
        # Release the large variant_records list from the dataset — run_training_pipeline
        # will re-create training records from variant_stats inside fit()
        gc.collect()
        log(f"  memory after gc: {mem()}")
        run_training_pipeline(
            dataset=dataset,
            config=config,
            output_dir=work_dir,
        )
        run_metadata_path.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    finally:
        if dataset is not None:
            del dataset
        # Keep VCFs on disk for cache / reruns (do NOT delete)
        release_process_memory()
        log("=== UNIFIED FIT CLEANUP DONE ===")

    log("=== ALL OF US PIPELINE COMPLETE ===")

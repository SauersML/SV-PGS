"""All of Us orchestration: download VCFs, prepare phenotypes, merge PCs, run one unified fit."""
from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from sv_pgs.all_of_us import AllOfUsDiseaseRequest, prepare_all_of_us_disease_sample_table, resolve_disease_definition
from sv_pgs.config import ModelConfig, TraitType
from sv_pgs.io import load_multi_vcf_dataset_from_files, run_training_pipeline
from sv_pgs.progress import log


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


def sv_vcf_name(chromosome: int) -> str:
    return f"AoU_srWGS_SV.v8.chr{chromosome}.vcf.gz"


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


def download_sv_vcf(chromosome: int, work_dir: Path) -> Path:
    """Download one SV VCF + index when needed and return the local VCF path."""
    remote_dir = sv_vcf_dir()
    name = sv_vcf_name(chromosome)
    local_vcf = work_dir / name
    local_tbi = work_dir / f"{name}.tbi"
    vcf_remote = f"{remote_dir}/{name}"
    tbi_remote = f"{remote_dir}/{name}.tbi"

    missing_downloads: list[tuple[str, Path, str]] = []
    if not local_vcf.exists():
        missing_downloads.append((vcf_remote, local_vcf, "VCF"))
    if not local_tbi.exists():
        missing_downloads.append((tbi_remote, local_tbi, "index"))

    if not missing_downloads:
        log(f"  chr{chromosome}: VCF already present")
        return local_vcf

    required_bytes = sum(_gsutil_size(remote) for remote, _, _ in missing_downloads)
    _check_disk_space(work_dir, required_bytes)
    missing_labels = " + ".join(label for _, _, label in missing_downloads)
    log(f"  chr{chromosome}: downloading missing {missing_labels} ({required_bytes/1e9:.1f} GB)")
    for remote, local_path, _ in missing_downloads:
        _gsutil_cp(remote, str(local_path))
    return local_vcf


# ---------------------------------------------------------------------------
# Ancestry / PC merging
# ---------------------------------------------------------------------------

def download_ancestry_preds(work_dir: Path) -> Path:
    """Download the AoU ancestry predictions file (contains per-sample PCs)."""
    remote = resolve_ancestry_predictions_path()
    local = work_dir / Path(remote).name
    if local.exists():
        log(f"  ancestry predictions already present: {local.name}")
        return local
    log(f"  downloading ancestry predictions: {remote}")
    _gsutil_cp(remote, str(local))
    return local


def cleanup_local_sv_vcf(vcf_path: Path) -> None:
    tbi_path = Path(f"{vcf_path}.tbi")
    for path in (vcf_path, tbi_path):
        path.unlink(missing_ok=True)


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
        log(f"  merged sample table already exists; recomputing with n_pcs={n_pcs}")

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

    merge_key = max(overlap_counts, key=overlap_counts.get)
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


def _build_aou_run_metadata(
    *,
    disease: str,
    chromosomes: list[int],
    n_pcs: int,
    pc_cols: list[str],
    covariates: list[str],
    max_outer_iterations: int,
    min_sv_carriers: int,
    random_seed: int,
) -> dict[str, object]:
    return {
        "disease": disease,
        "chromosomes": chromosomes,
        "requested_n_pcs": n_pcs,
        "effective_pc_columns": pc_cols,
        "covariates": covariates,
        "max_outer_iterations": max_outer_iterations,
        "minimum_structural_variant_carriers": min_sv_carriers,
        "random_seed": random_seed,
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
    min_sv_carriers: int = 5,
    random_seed: int = 0,
) -> None:
    """Full AoU pipeline: download requested chromosomes, merge them, and run one fit."""

    # Validate disease
    disease_def = resolve_disease_definition(disease)
    log(f"=== ALL OF US PIPELINE ===  disease={disease_def.canonical_name}  chromosomes={chromosomes}  n_pcs={n_pcs}")
    log(f"  ICD-9: {disease_def.icd9_prefixes}  ICD-10: {disease_def.icd10_prefixes}")

    work_dir = Path(output_base)
    work_dir.mkdir(parents=True, exist_ok=True)

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

    summary_path = work_dir / "summary.json"
    run_metadata_path = _aou_run_metadata_path(work_dir)
    run_metadata = _build_aou_run_metadata(
        disease=disease_def.canonical_name,
        chromosomes=chromosomes,
        n_pcs=n_pcs,
        pc_cols=pc_cols,
        covariates=covariates,
        max_outer_iterations=max_outer_iterations,
        min_sv_carriers=min_sv_carriers,
        random_seed=random_seed,
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

        log("=== STEP 4: Load unified genome-wide dataset ===")
        dataset = load_multi_vcf_dataset_from_files(
            genotype_paths=vcf_paths,
            sample_table_path=str(merged_path),
            sample_id_column="auto",
            target_column="target",
            covariate_columns=covariates,
        )
        inferred_trait = TraitType.BINARY if len(np.unique(dataset.targets)) <= 2 else TraitType.QUANTITATIVE

        log("=== STEP 5: Fit unified genome-wide model ===")
        run_training_pipeline(
            dataset=dataset,
            config=ModelConfig(
                trait_type=inferred_trait,
                max_outer_iterations=max_outer_iterations,
                minimum_structural_variant_carriers=min_sv_carriers,
                random_seed=random_seed,
            ),
            output_dir=work_dir,
        )
        run_metadata_path.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    finally:
        if dataset is not None:
            del dataset
        for vcf_path in vcf_paths:
            cleanup_local_sv_vcf(vcf_path)
        release_process_memory()
        log("=== UNIFIED FIT CLEANUP DONE ===")

    log("=== ALL OF US PIPELINE COMPLETE ===")

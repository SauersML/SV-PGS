"""All of Us orchestration: download VCFs, prepare phenotypes, merge PCs, run per-chromosome."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from sv_pgs.all_of_us import AllOfUsDiseaseRequest, prepare_all_of_us_disease_sample_table, resolve_disease_definition
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
    import shutil
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


def download_sv_vcfs(chromosomes: list[int], work_dir: Path) -> dict[int, Path]:
    """Download SV VCFs for the requested chromosomes. Returns {chr: local_path}."""
    remote_dir = sv_vcf_dir()
    to_download: list[tuple[int, str, str]] = []  # (chrom, vcf_remote, tbi_remote)
    for chrom in chromosomes:
        if not (work_dir / sv_vcf_name(chrom)).exists():
            name = sv_vcf_name(chrom)
            to_download.append((chrom, f"{remote_dir}/{name}", f"{remote_dir}/{name}.tbi"))

    if to_download:
        # Query actual sizes from GCS
        log(f"  checking sizes for {len(to_download)} VCFs...")
        total_bytes = 0
        for chrom, vcf_remote, tbi_remote in to_download:
            vcf_size = _gsutil_size(vcf_remote)
            tbi_size = _gsutil_size(tbi_remote)
            total_bytes += vcf_size + tbi_size
            log(f"    chr{chrom}: {vcf_size/1e9:.1f} GB")
        _check_disk_space(work_dir, total_bytes)
        log(f"  total download: {total_bytes/1e9:.1f} GB, disk OK")

    vcf_paths: dict[int, Path] = {}
    remote_dir = sv_vcf_dir()
    for chrom in chromosomes:
        name = sv_vcf_name(chrom)
        local_vcf = work_dir / name
        local_tbi = work_dir / f"{name}.tbi"
        if local_vcf.exists():
            log(f"  chr{chrom}: already present")
        else:
            log(f"  chr{chrom}: downloading VCF + index...")
            _gsutil_cp(f"{remote_dir}/{name}", str(local_vcf))
            _gsutil_cp(f"{remote_dir}/{name}.tbi", str(local_tbi))
        vcf_paths[chrom] = local_vcf
    return vcf_paths


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


def _parse_pca_features_column(series: pd.Series, n_pcs: int) -> tuple[pd.DataFrame, list[str]]:
    """Parse a compound pca_features column like '[0.01,-0.02,...]' into PC1..PCn."""
    import json
    pc_names = [f"PC{i+1}" for i in range(n_pcs)]

    def _parse_row(val: str) -> list[float]:
        if not isinstance(val, str) or not val.strip():
            return [float("nan")] * n_pcs
        cleaned = val.strip()
        # Handle Hail array format: [0.01, -0.02, ...] or just 0.01,-0.02,...
        if cleaned.startswith("["):
            try:
                values = json.loads(cleaned)
            except json.JSONDecodeError:
                return [float("nan")] * n_pcs
        else:
            values = [float(x.strip()) for x in cleaned.split(",") if x.strip()]
        return [values[i] if i < len(values) else float("nan") for i in range(n_pcs)]

    parsed = series.apply(_parse_row)
    df = pd.DataFrame(parsed.tolist(), columns=pc_names, index=series.index)
    return df, pc_names


def merge_pcs_into_sample_table(
    sample_table_path: Path,
    ancestry_path: Path,
    output_path: Path,
    n_pcs: int = 10,
) -> tuple[Path, list[str]]:
    """Merge top N PCs from ancestry file into the sample table. Returns (output_path, pc_column_names)."""
    if output_path.exists():
        header = pd.read_csv(output_path, sep="\t", nrows=0).columns.tolist()
        pc_cols = [c for c in header if c.startswith("PC") and c[2:].isdigit()][:n_pcs]
        log(f"  merged sample table already exists with {len(pc_cols)} PCs")
        return output_path, pc_cols

    log(f"  loading sample table: {sample_table_path}")
    samples = pd.read_csv(sample_table_path, sep="\t", dtype={"sample_id": str, "person_id": str})

    log(f"  loading ancestry predictions: {ancestry_path}")
    ancestry = pd.read_csv(ancestry_path, sep="\t", dtype=str)
    log(f"  ancestry columns: {list(ancestry.columns)}")
    log(f"  ancestry rows: {len(ancestry)}, sample rows: {len(samples)}")

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

    # Merge: try person_id first, fall back to sample_id
    # In AoU controlled tier, person_id = research_id for most cases,
    # but if they differ we try both join keys.
    ancestry_subset = ancestry[[id_col] + pc_cols].copy()
    merge_key = "person_id"
    ancestry_subset = ancestry_subset.rename(columns={id_col: merge_key})
    merged = samples.merge(ancestry_subset, on=merge_key, how="left")
    n_with_pcs = int(merged[pc_cols[0]].notna().sum())

    # If merge on person_id gave 0 matches, try merging on sample_id instead
    if n_with_pcs == 0 and "sample_id" in samples.columns:
        log(f"  merge on person_id matched 0 rows, trying sample_id...")
        ancestry_subset2 = ancestry[[id_col] + pc_cols].copy()
        ancestry_subset2 = ancestry_subset2.rename(columns={id_col: "sample_id"})
        merged = samples.merge(ancestry_subset2, on="sample_id", how="left", suffixes=("", "_anc"))
        # Drop duplicate PC columns if any
        for col in pc_cols:
            if f"{col}_anc" in merged.columns:
                merged[col] = merged[f"{col}_anc"]
                merged.drop(columns=[f"{col}_anc"], inplace=True)
        n_with_pcs = int(merged[pc_cols[0]].notna().sum())

    log(f"  merged: {len(merged)} rows, {n_with_pcs} with PCs ({100*n_with_pcs/max(len(merged),1):.0f}%)")
    if n_with_pcs == 0:
        log("  WARNING: 0 samples matched ancestry file. PCs will be empty — check ID column mapping.")

    merged.to_csv(output_path, sep="\t", index=False)
    return output_path, pc_cols


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

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
    """Full AoU pipeline: download, prepare, merge PCs, fit per-chromosome."""
    from sv_pgs.io import load_dataset_from_files, run_training_pipeline
    from sv_pgs.config import ModelConfig

    # Validate disease
    disease_def = resolve_disease_definition(disease)
    log(f"=== ALL OF US PIPELINE ===  disease={disease_def.canonical_name}  chromosomes={chromosomes}  n_pcs={n_pcs}")
    log(f"  ICD-9: {disease_def.icd9_prefixes}  ICD-10: {disease_def.icd10_prefixes}")

    work_dir = Path(output_base)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download VCFs
    log("=== STEP 1: Download SV VCFs ===")
    vcf_paths = download_sv_vcfs(chromosomes, work_dir)

    # Step 2: Prepare phenotype
    log("=== STEP 2: Prepare phenotype ===")
    sample_table_path = work_dir / f"{disease_def.canonical_name}.samples.tsv"
    if not sample_table_path.exists():
        prepare_all_of_us_disease_sample_table(
            request=AllOfUsDiseaseRequest(disease=disease),
            output_path=sample_table_path,
        )
    else:
        log(f"  sample table already exists: {sample_table_path}")

    # Step 3: Download and merge PCs
    log("=== STEP 3: Merge genomic PCs ===")
    ancestry_path = download_ancestry_preds(work_dir)
    if ancestry_path is not None:
        merged_path = work_dir / f"{disease_def.canonical_name}.samples.with_pcs.tsv"
        merged_path, pc_cols = merge_pcs_into_sample_table(
            sample_table_path=sample_table_path,
            ancestry_path=ancestry_path,
            output_path=merged_path,
            n_pcs=n_pcs,
        )
    else:
        merged_path = sample_table_path
        pc_cols = []
        log("  skipping PCs (ancestry file not available)")

    # Build covariate list
    covariates = DEFAULT_COVARIATES + pc_cols
    log(f"  covariates ({len(covariates)}): {covariates}")

    # Step 4: Run per-chromosome
    log("=== STEP 4: Fit models per chromosome ===")
    for chrom in chromosomes:
        chr_output = work_dir / f"chr{chrom}"
        if (chr_output / "summary.json").exists():
            log(f"  chr{chrom}: already done, skipping")
            continue

        log(f"=== chr{chrom} START ===")
        vcf_path = vcf_paths[chrom]

        dataset = load_dataset_from_files(
            genotype_path=str(vcf_path),
            genotype_format="vcf",
            sample_table_path=str(merged_path),
            sample_id_column="auto",
            target_column="target",
            covariate_columns=covariates,
        )

        from sv_pgs.config import TraitType
        inferred_trait = "binary" if len(np.unique(dataset.targets)) <= 2 else "quantitative"
        run_training_pipeline(
            dataset=dataset,
            config=ModelConfig(
                trait_type=TraitType(inferred_trait),
                max_outer_iterations=max_outer_iterations,
                minimum_structural_variant_carriers=min_sv_carriers,
                random_seed=random_seed,
            ),
            output_dir=chr_output,
        )
        log(f"=== chr{chrom} DONE ===")

    log("=== ALL OF US PIPELINE COMPLETE ===")

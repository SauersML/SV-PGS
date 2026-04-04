"""All of Us orchestration: download VCFs, prepare phenotypes, merge PCs, run per-chromosome."""
from __future__ import annotations

import os
import subprocess
import shutil
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


def ancestry_preds_path() -> str:
    return f"{_cdr_storage_path()}/wgs/short_read/auxiliary/ancestry/ancestry_preds.tsv"


def sv_vcf_name(chromosome: int) -> str:
    return f"AoU_srWGS_SV.v8.chr{chromosome}.vcf.gz"


# ---------------------------------------------------------------------------
# gsutil helpers
# ---------------------------------------------------------------------------

def _gsutil_cp(src: str, dst: str) -> None:
    cmd = ["gsutil", "-u", _google_project(), "cp", src, dst]
    log(f"  gsutil cp {src} -> {dst}")
    subprocess.run(cmd, check=True)


def download_sv_vcfs(chromosomes: list[int], work_dir: Path) -> dict[int, Path]:
    """Download SV VCFs for the requested chromosomes. Returns {chr: local_path}."""
    vcf_paths: dict[int, Path] = {}
    remote_dir = sv_vcf_dir()
    for chrom in chromosomes:
        name = sv_vcf_name(chrom)
        local_vcf = work_dir / name
        local_tbi = work_dir / f"{name}.tbi"
        if local_vcf.exists():
            log(f"  chr{chrom}: already present ({local_vcf})")
        else:
            log(f"  chr{chrom}: downloading...")
            _gsutil_cp(f"{remote_dir}/{name}", str(local_vcf))
            _gsutil_cp(f"{remote_dir}/{name}.tbi", str(local_tbi))
        vcf_paths[chrom] = local_vcf
    return vcf_paths


# ---------------------------------------------------------------------------
# Ancestry / PC merging
# ---------------------------------------------------------------------------

def download_ancestry_preds(work_dir: Path) -> Path:
    """Download the AoU ancestry predictions file (contains per-sample PCs)."""
    local = work_dir / "ancestry_preds.tsv"
    if local.exists():
        log(f"  ancestry_preds.tsv already present")
        return local
    log(f"  downloading ancestry_preds.tsv...")
    _gsutil_cp(ancestry_preds_path(), str(local))
    return local


def merge_pcs_into_sample_table(
    sample_table_path: Path,
    ancestry_path: Path,
    output_path: Path,
    n_pcs: int = 10,
) -> tuple[Path, list[str]]:
    """Merge top N PCs from ancestry file into the sample table. Returns (output_path, pc_column_names)."""
    if output_path.exists():
        # Read back the PC column names from the header
        header = pd.read_csv(output_path, sep="\t", nrows=0).columns.tolist()
        pc_cols = [c for c in header if c.lower().startswith("pc") or c.lower().startswith("pca_")][:n_pcs]
        log(f"  merged sample table already exists with {len(pc_cols)} PCs")
        return output_path, pc_cols

    log(f"  loading sample table: {sample_table_path}")
    samples = pd.read_csv(sample_table_path, sep="\t", dtype={"sample_id": str, "person_id": str})

    log(f"  loading ancestry predictions: {ancestry_path}")
    ancestry = pd.read_csv(ancestry_path, sep="\t", dtype=str)

    # Find the ID column
    id_col = None
    for candidate in ["research_id", "person_id", "sample_id", "s"]:
        if candidate in ancestry.columns:
            id_col = candidate
            break
    if id_col is None:
        log(f"  WARNING: no ID column found in ancestry file. Columns: {list(ancestry.columns)[:15]}")
        samples.to_csv(output_path, sep="\t", index=False)
        return output_path, []

    # Find PC columns, sorted numerically
    pc_cols = sorted(
        [c for c in ancestry.columns if c.lower().startswith("pc") or c.lower().startswith("pca_")],
        key=lambda x: int("".join(filter(str.isdigit, x)) or "0"),
    )[:n_pcs]
    if not pc_cols:
        log(f"  WARNING: no PC columns found. Columns: {list(ancestry.columns)[:15]}")
        samples.to_csv(output_path, sep="\t", index=False)
        return output_path, []

    log(f"  merging {len(pc_cols)} PCs via {id_col}: {pc_cols}")
    for col in pc_cols:
        ancestry[col] = pd.to_numeric(ancestry[col], errors="coerce")

    # Add age^2 if age column exists
    if "age_at_observation_start" in samples.columns:
        samples["age_at_observation_start"] = pd.to_numeric(samples["age_at_observation_start"], errors="coerce")
        samples["age_squared"] = samples["age_at_observation_start"] ** 2
        log(f"  added age_squared covariate")

    ancestry_subset = ancestry[[id_col] + pc_cols].rename(columns={id_col: "person_id"})
    merged = samples.merge(ancestry_subset, on="person_id", how="left")
    n_with_pcs = int(merged[pc_cols[0]].notna().sum())
    log(f"  merged: {len(merged)} rows, {n_with_pcs} with PCs ({100*n_with_pcs/max(len(merged),1):.0f}%)")
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

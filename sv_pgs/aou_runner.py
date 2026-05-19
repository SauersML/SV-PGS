"""All of Us orchestration: download VCFs, prepare phenotypes, merge PCs, run one unified fit."""
from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sv_pgs.all_of_us import AllOfUsDiseaseRequest, prepare_all_of_us_disease_sample_table, resolve_disease_definition
from sv_pgs.config import ModelConfig, TraitType
from sv_pgs.io import load_multi_source_dataset_from_files
from sv_pgs.pipeline import run_training_pipeline
from sv_pgs.progress import log, mem

# Local on-workbench mirror of remote AoU buckets. We download each VCF once
# into work_dir.parent/.sv_pgs_cache/<subdir>/ and reuse it across runs; the
# bcftools precache step then writes an int8 .npy + variants/stats sidecar
# next to each downloaded VCF.
_LOCAL_CACHE_DIRNAME = ".sv_pgs_cache"
_AOU_SV_VCF_CACHE_SUBDIR = "aou_sv_vcfs"
# Microarray PLINK trio lives in its own subdir so that running --variants snp
# alongside --variants sv doesn't interleave files in one place. The trio is
# small enough (~80 GB across .bed + .bim + .fam) that we hold all three files
# locally without per-chromosome sharding.
_AOU_ARRAY_PLINK_CACHE_SUBDIR = "aou_array_plink"
# AoU's microarray PLINK files are emitted with the literal prefix "arrays".
# i.e. arrays.bed / arrays.bim / arrays.fam under .../microarray/plink/.
_AOU_ARRAY_PLINK_PREFIX = "arrays"

# ---------------------------------------------------------------------------
# AoU paths
#
# Everything lives under CDR_STORAGE_PATH (= gs://fc-aou-datasets-controlled/v8
# at the current Controlled Tier release). The workbench predefines a handful
# of env vars pointing at canonical assets; the source of truth is the
# "Controlled CDR Directory" article in the AoU User Support hub.
#
# Cohorts:
#   srWGS SNP & Indel : 414,830 participants
#   srWGS SVs         :  97,061 participants  (strict subset of the above)
#   genotyping array  : 447,278 participants
# Joint SV+SNP runs are capped at the 97,061 SV samples; the loader's
# _align_sample_ids does the intersection.
#
# srWGS Structural Variants (currently the only source we wire into the
# pipeline). One bgzipped .vcf.gz per autosome.
#   gs://.../v8/wgs/short_read/structural_variants/vcf/full/
#       AoU_srWGS_SV.v8.chr{N}.vcf.gz   (+ .tbi)
#
# srWGS SNP/Indel callsets (available, not yet wired). Same data in five
# formats; pick the one whose shape matches the downstream tool.
#   gs://.../v8/wgs/short_read/snpindel/
#     vds/hail.vds                                  WGS_VDS_PATH
#         Full sparse joint callset (Hail VDS), ~1B sites. Hail-only.
#     acaf_threshold/                               (AF>1% OR AC>100 / ancestry)
#       multiMT/hail.mt                             WGS_ACAF_THRESHOLD_MULTI_HAIL_PATH
#       splitMT/hail.mt                             WGS_ACAF_THRESHOLD_SPLIT_HAIL_PATH
#       vcf/                                        WGS_ACAF_THRESHOLD_VCF_PATH
#           Many .vcf.bgz shards per chromosome — enumerate with `gsutil ls`.
#       plink_bed/  bgen/  pgen/  bed/              (.bed/.bim/.fam etc.)
#         ACAF totals: 57M sites / 116M variants / 414,830 samples.
#     exome/                                        (Gencode v42 exons + 15 bp)
#       multiMT/hail.mt / splitMT/hail.mt           WGS_EXOME_MULTI/SPLIT_HAIL_PATH
#       vcf/                                        WGS_EXOME_VCF_PATH
#       plink_bed/  bgen/  pgen/  bed/
#         Exome totals: 38M sites / 46M variants / 414,830 samples.
#     clinvar/                                      (all ClinVar variants)
#       multiMT/hail.mt / splitMT/hail.mt           WGS_CLINVAR_MULTI/SPLIT_HAIL_PATH
#       vcf/                                        WGS_CLINVAR_VCF_PATH
#       plink_bed/  bgen/  pgen/  bed/
#         ClinVar totals: 1.5M sites / 2.2M variants / 414,830 samples.
#     cmrg/                                         WGS_CMRG_VCF_PATH
#         Challenging Medically Relevant Genes (33 genes) called against a
#         masked hg38 reference. Do NOT intersect with the other callsets.
#
# Auxiliary files we use today (the only path we touch outside the SV VCFs):
#   gs://.../v8/wgs/short_read/snpindel/aux/ancestry/ancestry_preds.tsv
#       Predicted continental ancestry + 16-dim PCA features per participant.
# Other aux directories that exist but we don't read:
#   vat/ (variant annotations), admixture_estimates/, pgx/ (PharmGKB stars),
#   phasing/, relatedness/ (kinship .tsvs), qc/ (sample QC flags + metrics).
#
# Other CDR roots, for reference:
#   gs://.../v8/wgs/cram/manifest.csv                           WGS_CRAM_MANIFEST_PATH
#   gs://.../v8/microarray/vcf/manifest.csv                     MICROARRAY_VCF_MANIFEST_PATH
#   gs://.../v8/microarray/hail.mt                              MICROARRAY_HAIL_STORAGE_PATH
#   gs://.../v8/microarray/plink/arrays.*                       (PLINK bed for array data)
#   gs://.../v8/microarray/idat/manifest.csv                    MICROARRAY_IDAT_MANIFEST_PATH
#   gs://.../v8/wgs/long_read/manifest.csv                      (lrWGS per-sample manifest)
#   gs://.../v8/known_issues/                                   (issue-specific sample lists)
# ---------------------------------------------------------------------------

def _cdr_storage_path() -> str:
    # gs://fc-aou-datasets-controlled/v8 on the current Controlled Tier; the
    # workbench sets this env var for every notebook + container.
    value = os.environ.get("CDR_STORAGE_PATH")
    if not value:
        raise RuntimeError("CDR_STORAGE_PATH is not set. Are you on an All of Us workbench?")
    return value


def _google_project() -> str:
    # Required as the gsutil -u billing project for any CDR egress (AoU does
    # not absorb bucket-read costs; the call fails without it).
    value = os.environ.get("GOOGLE_PROJECT")
    if not value:
        raise RuntimeError("GOOGLE_PROJECT is not set. Are you on an All of Us workbench?")
    return value


def sv_vcf_dir() -> str:
    # SV VCFs live under structural_variants/vcf/full — one .vcf.gz per
    # autosome (no env var; the path is documented in the Controlled CDR
    # Directory but not exposed via WGS_*_VCF_PATH).
    return f"{_cdr_storage_path()}/wgs/short_read/structural_variants/vcf/full"


def resolve_ancestry_predictions_path() -> str:
    # Per-participant ancestry predictions + PC features (.tsv keyed by
    # research_id). Same file is reused as the source of the 16 covariate PCs
    # the pipeline merges into the sample table.
    return f"{_cdr_storage_path()}/wgs/short_read/snpindel/aux/ancestry/ancestry_preds.tsv"


def local_ancestry_predictions_path(work_dir: Path) -> Path:
    # Downloaded copy inside the run's work_dir, so reruns reuse the file
    # without another gsutil cp.
    return work_dir / "ancestry_preds.tsv"


def sv_vcf_name(chromosome: int) -> str:
    # AoU's canonical SV VCF filename, hard-coded against the v8 release.
    # Bump the version literal when AoU rolls a new SV callset.
    return f"AoU_srWGS_SV.v8.chr{chromosome}.vcf.gz"


def local_sv_vcf_cache_dir(work_dir: Path) -> Path:
    # We park the SV VCF mirror one level above work_dir so multiple disease
    # runs (each with its own work_dir) share the same downloaded VCFs.
    return work_dir.parent / _LOCAL_CACHE_DIRNAME / _AOU_SV_VCF_CACHE_SUBDIR


def local_sv_vcf_path(chromosome: int, work_dir: Path) -> Path:
    return local_sv_vcf_cache_dir(work_dir) / sv_vcf_name(chromosome)


def array_plink_dir() -> str:
    # AoU's microarray PLINK 1 trio lives at .../microarray/plink/arrays.{bed,bim,fam}.
    # Single set (NOT chromosome-sharded), ~447,278 samples × ~700k SNPs,
    # lifted over to hg38 from the original genotyping-array calls. Used here
    # as the SNP source for joint SV+SNP runs because the file shape matches
    # PlinkRawGenotypeMatrix natively and the total size (~80 GB) fits a
    # standard workbench disk — unlike the ACAF SNP/indel callset at ~12 TB.
    return f"{_cdr_storage_path()}/microarray/plink"


def local_array_plink_cache_dir(work_dir: Path) -> Path:
    return work_dir.parent / _LOCAL_CACHE_DIRNAME / _AOU_ARRAY_PLINK_CACHE_SUBDIR


def local_array_plink_path(work_dir: Path) -> Path:
    # Returns the .bed path; the sibling .bim and .fam live in the same dir.
    return local_array_plink_cache_dir(work_dir) / f"{_AOU_ARRAY_PLINK_PREFIX}.bed"


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
    except (OSError, subprocess.SubprocessError, RuntimeError):
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


def download_array_plink(work_dir: Path) -> Path:
    """Download the AoU microarray PLINK 1 trio and return the local .bed path.

    The trio is a single file set (no chromosome sharding) covering all 447k
    array participants. After the first run the files stay on disk under
    work_dir.parent/.sv_pgs_cache/aou_array_plink/; subsequent runs reuse them.
    """
    remote_dir = array_plink_dir()
    cache_dir = local_array_plink_cache_dir(work_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_bed = local_array_plink_path(work_dir)

    missing_downloads: list[tuple[str, Path, str]] = []
    for extension in ("bed", "bim", "fam"):
        local_path = cache_dir / f"{_AOU_ARRAY_PLINK_PREFIX}.{extension}"
        if not local_path.exists():
            remote_path = f"{remote_dir}/{_AOU_ARRAY_PLINK_PREFIX}.{extension}"
            missing_downloads.append((remote_path, local_path, extension))

    if not missing_downloads:
        log("  microarray: PLINK trio already present in local cache")
        return local_bed

    required_bytes = sum(_gsutil_size(remote) for remote, _, _ in missing_downloads)
    _check_disk_space(cache_dir, required_bytes)
    missing_labels = " + ".join(label for _, _, label in missing_downloads)
    log(
        f"  microarray: downloading missing {missing_labels} into cache "
        + f"{cache_dir} ({required_bytes/1e9:.1f} GB)"
    )
    for remote, local_path, _ in missing_downloads:
        _download_gcs_object_if_missing(remote, local_path)
    return local_bed


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
    except (ImportError, RuntimeError):
        pass
    try:
        import cupy as cp

        cp.cuda.Device().synchronize()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except (ImportError, OSError, RuntimeError):
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


def _build_aou_run_metadata(
    *,
    disease: str,
    chromosomes: list[int],
    n_pcs: int,
    pc_cols: list[str],
    covariates: list[str],
    max_outer_iterations: int,
    random_seed: int,
    variant_metadata_path: str | None,
    variants: str,
    test_fraction: float = 0.0,
    marginal_screen_min_abs_z: float = 0.0,
) -> dict[str, object]:
    return {
        "disease": disease,
        "chromosomes": chromosomes,
        "requested_n_pcs": n_pcs,
        "effective_pc_columns": pc_cols,
        "covariates": covariates,
        "max_outer_iterations": max_outer_iterations,
        "random_seed": random_seed,
        "variant_metadata_path": variant_metadata_path,
        # Different --variants choices fit different models. Including this
        # in the run metadata makes existing-result-reuse skip a cached fit
        # when the source mix changes.
        "variants": variants,
        # Float so resuming a 0.2-test run picks up the cached fit but a flip
        # to 0.1 triggers a re-fit. Stored at 6-digit precision to keep the
        # JSON canonical across host float formatting differences.
        "test_fraction": round(float(test_fraction), 6),
        # |z| pre-screen threshold changes the active-variant set; if a user
        # tightens or loosens the screen, the cached fit no longer matches.
        "marginal_screen_min_abs_z": round(float(marginal_screen_min_abs_z), 6),
    }


# Canonical --variants choices. The CLI exposes a few aliases on top of these
# ("snv" for "snp", "sv+snp"/"snv+sv"/"sv+snv" for "snp+sv") which
# _normalize_variants_choice folds back into one of these three tokens. Keeping
# the canonical set tight means downstream branches stay easy to reason about.
_AOU_VARIANT_CHOICES = ("sv", "snp", "snp+sv")
_AOU_VARIANT_ALIASES: dict[str, str] = {
    "sv": "sv",
    "snp": "snp",
    "snv": "snp",
    "snp+sv": "snp+sv",
    "snv+sv": "snp+sv",
    "sv+snp": "snp+sv",
    "sv+snv": "snp+sv",
}


def _normalize_variants_choice(variants: str) -> str:
    """Map a user-facing --variants token to one of the three canonical choices.

    Accepts the technical-term variation "snv" wherever "snp" is meaningful,
    and both orderings of the +-separated joint form. Raises if the token is
    none of those — argparse already filters at the CLI boundary but the
    Python entrypoint is also reachable from tests and notebooks.
    """
    canonical = _AOU_VARIANT_ALIASES.get(variants)
    if canonical is None:
        raise ValueError(
            f"variants must be one of {sorted(_AOU_VARIANT_ALIASES)}; got {variants!r}"
        )
    return canonical


def _aou_metadata_equivalent(existing: dict[str, object], current: dict[str, object]) -> bool:
    """Compare run-metadata dicts with two back-compat tweaks.

    - Older runs predate the "variants" key; assume the historical SV default.
    - Older runs predate the "test_fraction" key; assume 0 (no holdout).
    - Older runs predate the "marginal_screen_min_abs_z" key; assume 0 (no screen),
      which is also the global ModelConfig default.
    Filling in those defaults stops a default-flip from spuriously invalidating
    every previously-produced result directory.
    """
    if existing == current:
        return True
    existing = dict(existing)
    current = dict(current)
    if "variants" not in existing:
        existing["variants"] = "sv"
    if "test_fraction" not in existing:
        existing["test_fraction"] = 0.0
    if "marginal_screen_min_abs_z" not in existing:
        existing["marginal_screen_min_abs_z"] = 0.0
    return existing == current


def _split_merged_sample_table(
    merged_path: Path,
    *,
    test_fraction: float,
    random_seed: int,
) -> tuple[Path, Path | None]:
    """Split the merged sample table into train + held-out test rows.

    Splitting is deterministic in `(random_seed, sample_id)`: a SHA-256 hash
    maps each row's sample_id into [0, 1); rows below `test_fraction` go to
    the test partition, everyone else to train. This way reruns with the same
    seed produce the same split, and adding/removing rows only moves the few
    affected rows (not the whole assignment).

    Returns (train_path, test_path). test_path is None when test_fraction is
    zero, in which case the original file is used unchanged.
    """
    if not (0.0 <= test_fraction < 1.0):
        raise ValueError(f"test_fraction must be in [0, 1); got {test_fraction!r}")
    if test_fraction == 0.0:
        return merged_path, None

    import hashlib

    base_name = merged_path.name
    train_path = merged_path.with_name(base_name.replace(".tsv", ".train.tsv", 1))
    test_path = merged_path.with_name(base_name.replace(".tsv", ".test.tsv", 1))
    salt = f"sv-pgs/split/seed={random_seed}".encode()

    train_lines: list[str] = []
    test_lines: list[str] = []
    with merged_path.open("rt", encoding="utf-8") as handle:
        header = handle.readline()
        train_lines.append(header)
        test_lines.append(header)
        for raw_line in handle:
            if not raw_line.strip():
                continue
            # sample_id is always column 0 (merge_pcs_into_sample_table emits
            # it as the leading column); slicing rather than full csv parse
            # keeps this O(bytes-in-row) for the 100k-row biobank case.
            sample_id = raw_line.split("\t", 1)[0]
            digest = hashlib.sha256(salt + b"|" + sample_id.encode()).digest()
            bucket = int.from_bytes(digest[:8], "big") / float(1 << 64)
            (test_lines if bucket < test_fraction else train_lines).append(raw_line)

    train_path.write_text("".join(train_lines), encoding="utf-8")
    test_path.write_text("".join(test_lines), encoding="utf-8")
    log(
        f"  train/test split (seed={random_seed}, test_fraction={test_fraction}): "
        f"{len(train_lines) - 1:,} train rows, {len(test_lines) - 1:,} test rows"
    )
    return train_path, test_path

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
    variant_metadata_path: str | Path | None = None,
    n_pcs: int = 10,
    max_outer_iterations: int = 40,
    random_seed: int = 0,
    variants: str = "snp+sv",
    test_fraction: float = 0.2,
    marginal_screen_min_abs_z: float = 1.0,
) -> None:
    """Full AoU pipeline: download requested chromosomes, merge them, and run one fit.

    `variants` selects the genotype sources fed into the joint model:
      "snp+sv"  — joint (default): AoU srWGS SV VCFs + AoU microarray PLINK,
                  intersected to the 97k-sample SV cohort.
      "sv"      — AoU srWGS SV VCFs only (97k samples).
      "snp"     — AoU microarray PLINK only (~447k samples).

    "snv" is accepted as an alias for "snp" and the joint form may be
    written in either order ("sv+snp", "snv+sv", etc.); the metadata file
    always records the canonical form.

    `test_fraction` carves a held-out evaluation set off the sample table by
    hashing sample_id with random_seed; defaults to 0.2 (80/20 train/test).
    Pass 0.0 to train on every available sample and skip the held-out AUC.
    The split is deterministic per seed so reruns reproduce the same
    train/test assignment.

    `marginal_screen_min_abs_z` is a univariate marginal-|z| floor applied
    after the MAF filter and before the joint Bayesian fit. Variants with
    |z| below this threshold (residualized on covariates; null distribution
    ~ N(0,1)) are dropped. The default 1.0 drops ~32% of pure-noise variants
    on biobank-scale data, typically cutting the joint matrix to fit a 16 GB
    GPU so the deterministic CAVI path runs (≈ 10× faster than the
    stochastic-block fallback) and removing noise that hurts held-out AUC.
    Set to 0.0 to disable; 1.5 or 2.0 for tighter screening.
    """
    variants = _normalize_variants_choice(variants)
    chromosomes = _validate_aou_chromosomes(chromosomes)
    if not (0.0 <= test_fraction < 1.0):
        raise ValueError(f"test_fraction must be in [0, 1); got {test_fraction!r}")

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
        random_seed=random_seed,
        marginal_screen_min_abs_z=marginal_screen_min_abs_z,
    )

    # Status summary: what's done vs what's left
    log("=== STATUS CHECK ===")
    sample_table_path = work_dir / f"{disease_def.canonical_name}.samples.tsv"
    merged_path = work_dir / f"{disease_def.canonical_name}.samples.with_pcs.tsv"
    log(f"  phenotype table: {'DONE' if sample_table_path.exists() else 'NEEDED'}")
    log(f"  PC-merged table: {'DONE' if merged_path.exists() else 'NEEDED'}")
    ancestry_local = local_ancestry_predictions_path(work_dir)
    resolved_variant_metadata_path = Path(variant_metadata_path) if variant_metadata_path is not None else None
    log(f"  ancestry file:   {'DONE' if ancestry_local.exists() else 'NEEDED'}")
    if resolved_variant_metadata_path is None:
        log("  variant metadata: not supplied")
    else:
        log(f"  variant metadata: {resolved_variant_metadata_path}")

    # Migrate old caches: previous versions stored caches in work_dir/.sv_pgs_cache/
    # with keys computed from the OLD VCF path.  The current code looks next to the
    # VCF files with keys computed from the NEW path.  We symlink old files under
    # the NEW key names so existing caches are found without re-parsing.
    from sv_pgs.io import _is_vcf_cache_bundle_complete, _vcf_cache_dir, _vcf_cache_key, _vcf_cache_paths
    import pickle as _pickle
    old_cache_dir = work_dir / _LOCAL_CACHE_DIRNAME
    if old_cache_dir.exists():
        first_vcf = local_sv_vcf_path(chromosomes[0], work_dir)
        if first_vcf.exists():
            new_cache_dir = _vcf_cache_dir(first_vcf)
            if old_cache_dir.resolve() != new_cache_dir.resolve():
                new_cache_dir.mkdir(parents=True, exist_ok=True)
                # Build map: chromosome → old key by reading variants.pkl
                old_chr_to_key: dict[str, str] = {}
                for geno_file in old_cache_dir.glob("*.genotypes.npy"):
                    old_k = geno_file.name.removesuffix(".genotypes.npy")
                    var_pkl = old_cache_dir / f"{old_k}.variants.pkl"
                    if not var_pkl.exists():
                        continue
                    try:
                        with open(var_pkl, "rb") as fh:
                            variants = _pickle.load(fh)
                        if variants:
                            chr_val = str(getattr(variants[0], "chromosome", "")).replace("chr", "")
                            old_chr_to_key[chr_val] = old_k
                    except (OSError, _pickle.UnpicklingError, ValueError, EOFError, AttributeError):
                        continue
                # Symlink old files under NEW key names
                migrated_chrs = 0
                for chrom in chromosomes:
                    vcf_path = local_sv_vcf_path(chrom, work_dir)
                    if not vcf_path.exists():
                        continue
                    new_key = _vcf_cache_key(vcf_path, config)
                    old_key_for_chrom = old_chr_to_key.get(str(chrom))
                    if old_key_for_chrom is None:
                        continue
                    # Check if new key already has a complete cache
                    new_geno = new_cache_dir / f"{new_key}.genotypes.npy"
                    if new_geno.exists():
                        continue
                    for suffix in (".genotypes.npy", ".variants.pkl", ".stats.npy", ".stats.npz", ".manifest.json"):
                        src = old_cache_dir / f"{old_key_for_chrom}{suffix}"
                        dst = new_cache_dir / f"{new_key}{suffix}"
                        if src.exists() and not dst.exists():
                            try:
                                dst.symlink_to(src.resolve())
                            except OSError:
                                pass
                    migrated_chrs += 1
                # Clean up .tmp_parallel directories from failed parallel parses
                for tmp_dir in new_cache_dir.glob("*.tmp_parallel"):
                    try:
                        shutil.rmtree(tmp_dir)
                    except OSError:
                        pass
                if migrated_chrs > 0:
                    log(f"  migrated {migrated_chrs} chromosome caches (old key → new key) from {old_cache_dir}")

    cached_chrs = []
    uncached_chrs = []
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
    # Skip SV-VCF status lines when running --variants snp, since we won't
    # download or precache them at all.
    if variants in ("sv", "snp+sv"):
        log(f"  SV VCF cached ({len(cached_chrs)}): {', '.join(cached_chrs) if cached_chrs else 'none'}")
        log(f"  SV VCF needed ({len(uncached_chrs)}): {', '.join(uncached_chrs) if uncached_chrs else 'none — all cached!'}")
    if variants in ("snp", "snp+sv"):
        # Microarray PLINK is one trio; either all three files are local or
        # we re-download. Report the aggregate state once instead of three
        # confusing lines.
        array_bed = local_array_plink_path(work_dir)
        array_present = all(
            array_bed.with_suffix(f".{extension}").exists()
            for extension in ("bed", "bim", "fam")
        )
        log(f"  microarray PLINK trio: {'DONE' if array_present else 'NEEDED'}")
    log(f"  variants source: {variants}")
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
        random_seed=random_seed,
        variant_metadata_path=str(resolved_variant_metadata_path) if resolved_variant_metadata_path is not None else None,
        variants=variants,
        test_fraction=test_fraction,
        marginal_screen_min_abs_z=marginal_screen_min_abs_z,
    )
    if summary_path.exists():
        if run_metadata_path.exists():
            existing_run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
            if _aou_metadata_equivalent(existing_run_metadata, run_metadata):
                log(f"  unified fit already exists with matching configuration: {summary_path}")
                return
            log("  existing unified fit configuration differs; rerunning and overwriting outputs")
        else:
            log("  unified fit exists without run metadata; rerunning and overwriting outputs")

    log(f"=== STEP 3: Download genotype sources (variants={variants}) ===")
    # `sources` is the ordered list of (kind, local_path) tuples handed to the
    # multi-source loader. SV VCFs come first (matching the historical sample
    # order); the microarray PLINK trio, if requested, is appended as the
    # single PLINK source.
    sources: list[tuple[str, Path]] = []
    vcf_paths: list[Path] = []
    dataset = None
    test_dataset = None
    try:
        if variants in ("sv", "snp+sv"):
            for chrom in chromosomes:
                local_vcf = download_sv_vcf(chrom, work_dir)
                vcf_paths.append(local_vcf)
                sources.append(("vcf", local_vcf))
        if variants in ("snp", "snp+sv"):
            log("  downloading microarray PLINK trio...")
            array_bed = download_array_plink(work_dir)
            sources.append(("plink1", array_bed))

        if vcf_paths:
            log("=== STEP 3.5: Parallel VCF precache ===")
            from sv_pgs.io import precache_vcfs_parallel
            try:
                precache_vcfs_parallel(vcf_paths, config)
            except (OSError, RuntimeError, ValueError) as exc:
                raise RuntimeError("parallel VCF precache failed") from exc
        else:
            log("=== STEP 3.5: skipping VCF precache (no VCF sources) ===")

        # Carve a deterministic held-out test partition off the merged sample
        # table BEFORE we load genotypes. The split uses sample_id+seed so
        # reruns are reproducible — and so two sibling runs (e.g. SV-only vs
        # SV+SNP) keep the same train/test assignment, which makes their
        # held-out AUCs comparable.
        train_table_path, test_table_path = _split_merged_sample_table(
            merged_path=merged_path,
            test_fraction=test_fraction,
            random_seed=random_seed,
        )

        log("=== STEP 4: Load unified genome-wide TRAIN dataset ===")
        dataset = load_multi_source_dataset_from_files(
            sources=[(kind, str(path)) for kind, path in sources],
            config=config,
            sample_table_path=str(train_table_path),
            sample_id_column="auto",
            target_column="target",
            covariate_columns=covariates,
            variant_metadata_path=resolved_variant_metadata_path,
        )
        if test_table_path is not None:
            log("=== STEP 4b: Load held-out TEST dataset ===")
            # Same sources, same variant order, different sample subset. The
            # multi-source loader's intersection logic will further restrict
            # to samples present in every source — for SV-only that's a no-op,
            # for SV+microarray that's the SV ∩ array intersection.
            # Pass train's variant_stats + variant_records straight through so
            # the test load skips its own ~10-110 min PLINK variant-stats
            # streaming pass entirely. The downstream pipeline only consumes
            # test_dataset.{genotypes, covariates, targets, sample_ids}; stats
            # and records are kept consistent with train so any consumer that
            # reads them sees one model, one standardization.
            test_dataset = load_multi_source_dataset_from_files(
                sources=[(kind, str(path)) for kind, path in sources],
                config=config,
                sample_table_path=str(test_table_path),
                sample_id_column="auto",
                target_column="target",
                covariate_columns=covariates,
                variant_metadata_path=resolved_variant_metadata_path,
                precomputed_variant_stats=dataset.variant_stats,
                precomputed_variant_records=dataset.variant_records,
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
            test_dataset=test_dataset,
        )
        run_metadata_path.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    finally:
        if dataset is not None:
            del dataset
        if test_dataset is not None:
            del test_dataset
        # Keep VCFs on disk for cache / reruns (do NOT delete)
        release_process_memory()
        log("=== UNIFIED FIT CLEANUP DONE ===")

    log("=== ALL OF US PIPELINE COMPLETE ===")

from __future__ import annotations

import csv
import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
from sklearn.metrics import log_loss, r2_score, roc_auc_score

from sv_pgs.config import ModelConfig, TraitType, VariantClass
from sv_pgs.data import VariantRecord, VariantStatistics
from sv_pgs.genotype import DenseRawGenotypeMatrix, PlinkRawGenotypeMatrix, RawGenotypeMatrix
from sv_pgs.model import BayesianPGS
from sv_pgs.preprocessing import compute_variant_statistics
from sv_pgs.progress import log, mem

SV_LENGTH_THRESHOLD = 1_000.0
DEFAULT_SAMPLE_ID_COLUMNS = ("sample_id", "research_id", "person_id")


@dataclass(slots=True)
class LoadedDataset:
    sample_ids: list[str]
    genotypes: RawGenotypeMatrix
    covariates: np.ndarray
    targets: np.ndarray
    variant_records: list[VariantRecord]
    variant_stats: VariantStatistics | None = None


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


@dataclass(slots=True)
class _PlinkMetadata:
    sample_ids: list[str]
    variant_count: int


@dataclass(slots=True)
class _DelimitedTableSpec:
    path: Path
    delimiter: str
    columns: tuple[str, ...]


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
    log(f"=== LOAD DATASET START === mem={mem()}")

    source_path = Path(genotype_path)
    resolved_format = _resolve_genotype_format(source_path, genotype_format)
    log(f"genotype format={resolved_format}  path={source_path}")

    log(f"reading sample table header: {sample_table_path}")
    sample_table_spec = _inspect_delimited_table(sample_table_path)
    log(f"sample table columns={list(sample_table_spec.columns)}")

    variant_stats: VariantStatistics | None = None
    if resolved_format == "vcf":
        # Read VCF header to get sample IDs without parsing genotypes
        log("reading VCF header for sample IDs...")
        from cyvcf2 import VCF

        vcf_header_reader = VCF(str(source_path))
        source_sample_ids = [str(s) for s in vcf_header_reader.samples]
        vcf_header_reader.close()
        log(f"VCF has {len(source_sample_ids)} samples")

        # Match samples against phenotype table BEFORE loading genotypes
        log("resolving sample ID column...")
        resolved_sample_id_column = _resolve_sample_id_column(
            table_spec=sample_table_spec,
            requested_sample_id_column=sample_id_column,
            available_sample_ids=source_sample_ids,
        )
        log(f"sample ID column: '{resolved_sample_id_column}'")

        log("building filtered sample table (parsing target + covariates for matched genotype IDs only)...")
        sample_table, total_sample_rows, unmatched_sample_rows = _build_sample_table(
            table_spec=sample_table_spec,
            sample_id_column=resolved_sample_id_column,
            target_column=target_column,
            covariate_columns=covariate_columns,
            available_sample_ids=source_sample_ids,
        )
        n_cases = int(np.sum(np.asarray(sample_table.targets) == 1.0))
        n_controls = int(np.sum(np.asarray(sample_table.targets) == 0.0))
        log(
            "sample table: "
            + f"{len(sample_table.sample_ids)} matched rows kept from {total_sample_rows}, "
            + f"{unmatched_sample_rows} dropped, {sample_table.covariates.shape[1]} covariates"
        )
        log(f"  target distribution: {n_cases} cases, {n_controls} controls (of {len(sample_table.sample_ids)} total)")

        log("aligning sample IDs between sample table and genotype source...")
        aligned_sample_indices = _align_sample_ids(
            expected_sample_ids=sample_table.sample_ids,
            available_sample_ids=source_sample_ids,
            context="genotype source",
        )
        log(f"aligned {len(aligned_sample_indices)} phenotype rows against {len(source_sample_ids)} genotype samples")

        # Load VCF genotypes, keeping only matched samples and accumulating as int8
        keep_indices = np.array(aligned_sample_indices, dtype=np.intp)
        log(f"loading VCF genotypes (keeping {len(keep_indices)} of {len(source_sample_ids)} samples, int8 accumulation)...")

        # Try disk cache first to skip VCF re-parsing on repeated runs
        cached = _load_vcf_from_cache(source_path, keep_sample_indices=keep_indices)
        if cached is not None:
            genotype_matrix, default_variants, variant_stats = cached
        else:
            genotype_matrix, default_variants, variant_stats = _load_vcf(source_path, keep_sample_indices=keep_indices)
            _save_vcf_to_cache(source_path, keep_indices, genotype_matrix, default_variants, variant_stats)

        log(f"VCF loaded: {genotype_matrix.shape[0]} samples x {len(default_variants)} variants  mem={mem()}")
        plink_metadata = None
    elif resolved_format == "plink1":
        log("reading PLINK .fam/.bim metadata (no genotype data yet)...")
        plink_metadata = _load_plink1_metadata(source_path)
        source_sample_ids = plink_metadata.sample_ids
        log(f"PLINK metadata: {len(source_sample_ids)} samples x {plink_metadata.variant_count} variants")
        bed_size = source_path.stat().st_size / 1e9
        full_matrix_gb = len(source_sample_ids) * plink_metadata.variant_count * 4 / 1e9
        log(f"  .bed file size: {bed_size:.2f} GB  |  full float32 matrix would be: {full_matrix_gb:.1f} GB")
        genotype_matrix = None
        default_variants = None
    else:
        raise ValueError("Unsupported genotype format: " + resolved_format)

    if resolved_format != "vcf":
        # PLINK path: sample matching happens after metadata load (same as before)
        log("resolving sample ID column...")
        resolved_sample_id_column = _resolve_sample_id_column(
            table_spec=sample_table_spec,
            requested_sample_id_column=sample_id_column,
            available_sample_ids=source_sample_ids,
        )
        log(f"sample ID column: '{resolved_sample_id_column}'")

        log("building filtered sample table (parsing target + covariates for matched genotype IDs only)...")
        sample_table, total_sample_rows, unmatched_sample_rows = _build_sample_table(
            table_spec=sample_table_spec,
            sample_id_column=resolved_sample_id_column,
            target_column=target_column,
            covariate_columns=covariate_columns,
            available_sample_ids=source_sample_ids,
        )
        n_cases = int(np.sum(np.asarray(sample_table.targets) == 1.0))
        n_controls = int(np.sum(np.asarray(sample_table.targets) == 0.0))
        log(
            "sample table: "
            + f"{len(sample_table.sample_ids)} matched rows kept from {total_sample_rows}, "
            + f"{unmatched_sample_rows} dropped, {sample_table.covariates.shape[1]} covariates"
        )
        log(f"  target distribution: {n_cases} cases, {n_controls} controls (of {len(sample_table.sample_ids)} total)")

        log("aligning sample IDs between sample table and genotype source...")
        aligned_sample_indices = _align_sample_ids(
            expected_sample_ids=sample_table.sample_ids,
            available_sample_ids=source_sample_ids,
            context="genotype source",
        )
        sample_table, aligned_sample_indices, reordered = _reorder_sample_table_by_source_index(
            sample_table=sample_table,
            source_indices=aligned_sample_indices,
        )
        if reordered:
            log("reordered matched phenotype rows into genotype order for contiguous PLINK access")
        log(f"aligned {len(aligned_sample_indices)} phenotype rows against {len(source_sample_ids)} genotype samples")

    if resolved_format == "vcf":
        # genotype_matrix is already subsetted to aligned samples (int8 for VCF)
        raw_genotypes: RawGenotypeMatrix = DenseRawGenotypeMatrix(genotype_matrix)
        if default_variants is None:
            raise RuntimeError("VCF defaults were not initialized.")
        log(f"VCF matrix: {raw_genotypes.shape}  mem={mem()}")
    else:
        if plink_metadata is None:
            raise RuntimeError("PLINK metadata were not initialized.")
        total_fam_samples = len(plink_metadata.sample_ids)
        log(f"creating lazy PLINK genotype reader ({len(aligned_sample_indices)} samples x {plink_metadata.variant_count} variants, {total_fam_samples} total in .fam)")
        subset_gb = len(aligned_sample_indices) * plink_metadata.variant_count * 4 / 1e9
        log(f"  subset float32 matrix would be: {subset_gb:.1f} GB (will stream in batches instead)")
        raw_genotypes = PlinkRawGenotypeMatrix(
            bed_path=source_path,
            sample_indices=aligned_sample_indices,
            variant_count=plink_metadata.variant_count,
            total_sample_count=total_fam_samples,
        )
        # Add intercept column to covariates for residual computation (same as model.fit)
        cov_with_intercept = np.concatenate([
            np.ones((sample_table.covariates.shape[0], 1), dtype=np.float32),
            sample_table.covariates,
        ], axis=1)
        log("computing variant statistics + screening scores (single pass, JAX)...")
        variant_stats = compute_variant_statistics(
            raw_genotypes, config=ModelConfig(),
            covariates=cov_with_intercept, targets=sample_table.targets,
        )
        log("building PLINK variant defaults from pre-computed allele frequencies...")
        default_variants = _build_plink_variant_defaults_from_stats(source_path, variant_stats)
        log(f"built {len(default_variants)} PLINK variant defaults  mem={mem()}")

    log("building variant records from defaults + optional metadata...")
    variant_records = _build_variant_records(
        default_variants=default_variants,
        variant_metadata_path=variant_metadata_path,
    )
    sv_count = sum(1 for vr in variant_records if vr.variant_class.value not in ("snv", "small_indel"))
    snv_count = sum(1 for vr in variant_records if vr.variant_class.value == "snv")
    log(f"variant records: {len(variant_records)} total ({snv_count} SNVs, {sv_count} structural variants)")

    log(f"=== LOAD DATASET DONE === final shape={raw_genotypes.shape}  mem={mem()}")
    return LoadedDataset(
        sample_ids=list(sample_table.sample_ids),
        genotypes=raw_genotypes,
        covariates=np.asarray(sample_table.covariates, dtype=np.float32),
        targets=np.asarray(sample_table.targets, dtype=np.float32),
        variant_records=variant_records,
        variant_stats=variant_stats,
    )


def run_training_pipeline(
    dataset: LoadedDataset,
    config: ModelConfig,
    output_dir: str | Path,
) -> PipelineOutputs:
    log(f"=== TRAINING PIPELINE START ===  samples={len(dataset.sample_ids)}  variants={dataset.genotypes.shape[1]}  trait={config.trait_type.value}  mem={mem()}")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    log("fitting Bayesian PGS model...")
    model = BayesianPGS(config).fit(
        dataset.genotypes,
        dataset.covariates,
        dataset.targets,
        dataset.variant_records,
        variant_stats=dataset.variant_stats,
    )
    log(f"model fitted  mem={mem()}")

    log("exporting model artifacts...")
    artifact_dir = destination / "artifact"
    model.export(artifact_dir)
    log(f"artifacts written to {artifact_dir}")

    log("writing coefficients table...")
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

    log("writing predictions...")
    predictions_path = destination / "predictions.tsv"
    summary_payload = _write_predictions_and_summary(
        predictions_path=predictions_path,
        dataset=dataset,
        model=model,
    )
    active_count = int(model.state.active_variant_indices.shape[0]) if model.state is not None else 0
    summary_payload.update(
        {
            "sample_count": int(dataset.genotypes.shape[0]),
            "variant_count": int(dataset.genotypes.shape[1]),
            "active_variant_count": active_count,
            "trait_type": config.trait_type.value,
        }
    )
    log(f"predictions written: {active_count} active variants out of {dataset.genotypes.shape[1]}")

    log("writing summary JSON...")
    summary_path = destination / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    log(f"=== TRAINING PIPELINE DONE ===  mem={mem()}")
    return PipelineOutputs(
        artifact_dir=artifact_dir,
        summary_path=summary_path,
        predictions_path=predictions_path,
        coefficients_path=coefficients_path,
    )


def _build_sample_table(
    table_spec: _DelimitedTableSpec,
    sample_id_column: str,
    target_column: str,
    covariate_columns: Sequence[str],
    available_sample_ids: Sequence[str],
) -> tuple[_SampleTable, int, int]:
    _require_columns(
        available_columns=table_spec.columns,
        required_columns=(sample_id_column, target_column, *covariate_columns),
        context="sample table",
    )

    sample_ids: list[str] = []
    covariates: list[list[float]] = []
    targets: list[float] = []
    seen_sample_ids: set[str] = set()
    available_sample_id_set = set(available_sample_ids)
    total_rows = 0
    unmatched_rows = 0

    for row in _iter_delimited_rows(table_spec):
        total_rows += 1
        sample_id = str(row[sample_id_column]).strip()
        if not sample_id:
            raise ValueError("Encountered blank sample identifier in sample table.")
        if sample_id not in available_sample_id_set:
            unmatched_rows += 1
            continue
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
    if not sample_ids:
        raise ValueError(
            "Sample table contains no rows that overlap the genotype source using column: " + sample_id_column
        )
    return (
        _SampleTable(
            sample_ids=sample_ids,
            covariates=covariate_matrix,
            targets=np.asarray(targets, dtype=np.float32),
        ),
        total_rows,
        unmatched_rows,
    )


def _resolve_sample_id_column(
    table_spec: _DelimitedTableSpec,
    requested_sample_id_column: str,
    available_sample_ids: Sequence[str],
) -> str:
    available_columns = table_spec.columns
    if requested_sample_id_column != "auto":
        if requested_sample_id_column not in available_columns:
            raise ValueError("Sample table is missing required columns: " + requested_sample_id_column)
        return requested_sample_id_column

    candidate_columns = [
        column_name
        for column_name in DEFAULT_SAMPLE_ID_COLUMNS
        if column_name in available_columns
    ]
    if not candidate_columns:
        raise ValueError(
            "Sample table must contain at least one identifier column: "
            + ", ".join(DEFAULT_SAMPLE_ID_COLUMNS)
            + ". Available columns: "
            + ", ".join(available_columns)
        )

    available_sample_id_set = set(available_sample_ids)
    # Sample first 1000 rows to pick the best column (avoids full 633k scan)
    match_counts = {column_name: 0 for column_name in candidate_columns}
    rows_checked = 0
    for row in _iter_delimited_rows(table_spec):
        for column_name in candidate_columns:
            if str(row[column_name]).strip() in available_sample_id_set:
                match_counts[column_name] += 1
        rows_checked += 1
        if rows_checked >= 1000:
            break
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


# ---------------------------------------------------------------------------
# VCF disk cache
# ---------------------------------------------------------------------------

_CACHE_DIR_NAME = ".sv_pgs_cache"
# Bump this when _VariantDefaults, VariantClass, or the cache format changes
# so stale caches are automatically invalidated.
_CACHE_VERSION = 2


def _vcf_cache_key(vcf_path: Path, keep_sample_indices: np.ndarray | None) -> str:
    """Compute a hex digest that uniquely identifies a VCF + sample-subset."""
    h = hashlib.sha256()
    h.update(f"v{_CACHE_VERSION}:".encode())
    h.update(str(vcf_path.resolve()).encode())
    stat = vcf_path.stat()
    h.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    if keep_sample_indices is not None:
        h.update(keep_sample_indices.tobytes())
    return h.hexdigest()[:24]


def _vcf_cache_dir(vcf_path: Path) -> Path:
    return vcf_path.resolve().parent / _CACHE_DIR_NAME


def _load_vcf_from_cache(
    vcf_path: Path,
    keep_sample_indices: np.ndarray | None,
) -> tuple[np.ndarray, list[_VariantDefaults], VariantStatistics] | None:
    """Try to load cached VCF parse results. Returns None on miss."""
    cache_dir = _vcf_cache_dir(vcf_path)
    if not cache_dir.exists():
        return None

    key = _vcf_cache_key(vcf_path, keep_sample_indices)
    geno_path = cache_dir / f"{key}.genotypes.npy"
    var_path = cache_dir / f"{key}.variants.pkl"
    stats_path = cache_dir / f"{key}.stats.npz"

    if not geno_path.exists() or not var_path.exists() or not stats_path.exists():
        log(f"VCF cache miss (key={key})")
        return None

    try:
        log(f"VCF cache hit — loading from {cache_dir.name}/{key}.*")
        genotype_matrix = np.load(geno_path, mmap_mode=None)
        with open(var_path, "rb") as f:
            variants = pickle.load(f)
        stats_payload = np.load(stats_path)
        variant_stats = VariantStatistics(
            means=np.asarray(stats_payload["means"], dtype=np.float32),
            scales=np.asarray(stats_payload["scales"], dtype=np.float32),
            allele_frequencies=np.asarray(stats_payload["allele_frequencies"], dtype=np.float32),
            support_counts=np.asarray(stats_payload["support_counts"], dtype=np.int32),
            marginal_scores=None,
        )
        log(f"  cached matrix {genotype_matrix.shape}, {len(variants)} variants")
        return genotype_matrix, variants, variant_stats
    except Exception as exc:
        log(f"VCF cache load failed ({exc}), will re-parse")
        return None


def _save_vcf_to_cache(
    vcf_path: Path,
    keep_sample_indices: np.ndarray | None,
    genotype_matrix: np.ndarray,
    variants: list[_VariantDefaults],
    variant_stats: VariantStatistics,
) -> None:
    """Persist parsed VCF results to disk cache."""
    cache_dir = _vcf_cache_dir(vcf_path)
    key = _vcf_cache_key(vcf_path, keep_sample_indices)
    geno_path = cache_dir / f"{key}.genotypes.npy"
    var_path = cache_dir / f"{key}.variants.pkl"
    stats_path = cache_dir / f"{key}.stats.npz"
    geno_tmp = cache_dir / f"{key}.genotypes.npy.tmp"
    var_tmp = var_path.with_suffix(".pkl.tmp")
    stats_tmp = cache_dir / f"{key}.stats.npz.tmp"

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Write to temp files first, then atomically rename to avoid
        # corrupt cache entries if the process is killed mid-write.
        with open(geno_tmp, "wb") as geno_handle:
            np.save(geno_handle, genotype_matrix)
        with open(str(var_tmp), "wb") as f:
            pickle.dump(variants, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(stats_tmp, "wb") as stats_handle:
            np.savez_compressed(
                stats_handle,
                means=np.asarray(variant_stats.means, dtype=np.float32),
                scales=np.asarray(variant_stats.scales, dtype=np.float32),
                allele_frequencies=np.asarray(variant_stats.allele_frequencies, dtype=np.float32),
                support_counts=np.asarray(variant_stats.support_counts, dtype=np.int32),
            )
        geno_tmp.rename(geno_path)
        var_tmp.rename(var_path)
        stats_tmp.rename(stats_path)
        total_mb = (geno_path.stat().st_size + var_path.stat().st_size + stats_path.stat().st_size) / 1e6
        log(f"VCF cache saved ({total_mb:.1f} MB) → {cache_dir.name}/{key}.*")
    except Exception as exc:
        log(f"VCF cache save failed ({exc}), continuing without cache")
        for p in (geno_tmp, var_tmp, stats_tmp, geno_path, var_path, stats_path):
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass


def _load_vcf(
    vcf_path: Path,
    keep_sample_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, list[_VariantDefaults], VariantStatistics]:
    """Load VCF genotypes into an int8 matrix.

    If keep_sample_indices is provided, only those sample positions are stored
    per variant (cuts sample dimension to phenotype-matched subset).
    Returns (int8 genotype matrix, variant defaults).

    Memory strategy: accumulate per-variant dosage as int8 columns (1 byte each),
    then stack into a single int8 matrix.  Values are 0/1/2 for dosage, -1 for
    missing.  Converted to float32 with NaN per-batch on the fly by
    DenseRawGenotypeMatrix.  Peak memory is ~2.5 GB for 97K×26K (4x smaller
    than the previous float32 approach).
    """
    import time
    import os

    # gt_types mapping: 0=HOM_REF, 1=HET, 2=UNKNOWN/MISSING, 3=HOM_ALT
    # int8 dosage: 0→0, 1→1, 3→2, 2→-1 (missing sentinel)
    _GT_TO_INT8 = np.array([0, 1, -1, 2], dtype=np.int8)

    vcf_size_mb = vcf_path.stat().st_size / 1e6
    log(f"opening VCF: {vcf_path.name} ({vcf_size_mb:.1f} MB)")

    from cyvcf2 import VCF

    reader = VCF(str(vcf_path))
    n_threads = os.cpu_count() or 4
    reader.set_threads(n_threads)
    log(f"VCF decompression threads: {n_threads}")
    n_all_samples = len(reader.samples)
    n_keep = len(keep_sample_indices) if keep_sample_indices is not None else n_all_samples
    log(f"VCF has {n_all_samples} samples, keeping {n_keep}")

    dosage_columns: list[np.ndarray] = []
    variants: list[_VariantDefaults] = []

    t_start = time.monotonic()
    last_log_time = t_start
    last_chrom = None

    # Local references avoid repeated global/attribute lookups in the hot loop
    _gt_to_i8 = _GT_TO_INT8
    _append_col = dosage_columns.append
    _append_var = variants.append
    _VariantDef = _VariantDefaults
    _vcf_id = _vcf_variant_id
    _vcf_cls = _infer_vcf_variant_class
    _vcf_len = _infer_vcf_length
    _norm_q = _normalize_quality
    _monotonic = time.monotonic

    for record in reader:
        if len(record.ALT) != 1:
            raise ValueError(
                "Only biallelic VCF records are supported. Normalize multiallelic records before loading: "
                + _vcf_variant_key(record)
            )

        gt = record.gt_types
        int8_col = _gt_to_i8[gt]
        if keep_sample_indices is not None:
            int8_col = int8_col[keep_sample_indices]
        _append_col(int8_col)

        # Defer AF to INFO field or post-hoc batch computation
        af_value = record.INFO.get("AF")
        af = float(af_value[0]) if isinstance(af_value, (tuple, list)) else float(af_value) if af_value is not None else -1.0
        _append_var(
            _VariantDef(
                variant_id=_vcf_id(record),
                variant_class=_vcf_cls(record),
                chromosome=str(record.CHROM),
                position=int(record.POS),
                length=_vcf_len(record),
                allele_frequency=af,
                quality=_norm_q(record.QUAL),
            )
        )

        n = len(variants)
        now = _monotonic()
        chrom = str(record.CHROM)

        if chrom != last_chrom:
            if last_chrom is not None:
                log(f"  chromosome {last_chrom} done — {n} variants so far  mem={mem()}")
            last_chrom = chrom
            last_log_time = now
        elif now - last_log_time >= 5.0:
            rate = n / (now - t_start)
            log(f"  {n} variants loaded ({rate:.0f} variants/s, {chrom})  mem={mem()}")
            last_log_time = now

    elapsed = time.monotonic() - t_start
    n_total = len(variants)

    if n_total == 0:
        raise ValueError("VCF contains no variants: " + str(vcf_path))

    log(f"parsed {n_total} variants in {elapsed:.1f}s ({n_total / elapsed:.0f} variants/s)")

    # Build int8 matrix directly — 4x smaller than float32 (2.5 GB vs 10 GB for 97K×26K).
    # Values: 0/1/2 for dosage, -1 for missing.  Converted to float32 per-batch on the fly.
    # Use Fortran (column-major) order for fast column writes, then convert to C order
    # for fast row-slice access during batch iteration.
    matrix_gb = n_keep * n_total / 1e9
    log(f"building int8 matrix ({n_keep} x {n_total}, {matrix_gb:.1f} GB)...")
    genotype_matrix = np.empty((n_keep, n_total), dtype=np.int8, order='F')
    for i in range(n_total):
        genotype_matrix[:, i] = dosage_columns[i]
        dosage_columns[i] = None  # free column immediately
    del dosage_columns
    genotype_matrix = np.ascontiguousarray(genotype_matrix)

    matrix_mb = genotype_matrix.nbytes / 1e6
    log(f"genotype matrix ready: {genotype_matrix.shape}, {matrix_mb:.1f} MB  mem={mem()}")

    # Compute variant stats in column chunks to avoid 20 GB intermediate.
    # Each chunk processes ~1024 variants at a time using int8→int64 accumulation.
    log(f"computing variant statistics (chunked over {n_total} variants)...")
    col_sums = np.empty(n_total, dtype=np.int64)
    col_sum_sq = np.empty(n_total, dtype=np.int64)
    n_valid = np.empty(n_total, dtype=np.int64)
    support_arr = np.empty(n_total, dtype=np.int32)
    chunk = 1024
    for start in range(0, n_total, chunk):
        end = min(start + chunk, n_total)
        block = genotype_matrix[:, start:end]  # int8 view, no copy
        valid = block != -1
        obs = np.where(valid, block.astype(np.int32), 0)
        col_sums[start:end] = np.sum(obs, axis=0, dtype=np.int64)
        col_sum_sq[start:end] = np.sum(obs * obs, axis=0, dtype=np.int64)
        n_valid[start:end] = np.count_nonzero(valid, axis=0)
        support_arr[start:end] = np.count_nonzero(obs > 0, axis=0)
    safe_n_valid = np.maximum(n_valid, 1).astype(np.float64)
    means_arr = (col_sums / safe_n_valid).astype(np.float32)
    allele_freqs = np.clip(means_arr / 2.0, 0.0, 1.0).astype(np.float32)
    centered_sum_sq = np.maximum(col_sum_sq.astype(np.float64) - col_sums.astype(np.float64) ** 2 / safe_n_valid, 0.0)
    scales_arr = np.sqrt(centered_sum_sq / max(n_keep, 1)).astype(np.float32)
    scales_arr = np.where(scales_arr < 1e-6, 1.0, scales_arr)
    # Fix AFs for variants where INFO/AF was missing (marked as -1)
    for i, v in enumerate(variants):
        if v.allele_frequency < 0:
            variants[i] = _VariantDefaults(
                variant_id=v.variant_id, variant_class=v.variant_class,
                chromosome=v.chromosome, position=v.position, length=v.length,
                allele_frequency=float(allele_freqs[i]), quality=v.quality,
            )
    del col_sums, col_sum_sq, centered_sum_sq, safe_n_valid
    log(f"variant statistics done  mem={mem()}")

    variant_stats = VariantStatistics(
        means=means_arr,
        scales=scales_arr,
        allele_frequencies=allele_freqs,
        support_counts=support_arr,
        marginal_scores=None,
    )
    return genotype_matrix, variants, variant_stats
def _load_plink1_metadata(bed_path: Path) -> _PlinkMetadata:
    fam_path = bed_path.with_suffix(".fam")
    bim_path = bed_path.with_suffix(".bim")

    log(f"parsing .fam file: {fam_path}")
    sample_ids = _read_plink_sample_ids(fam_path)
    log(f"  .fam: {len(sample_ids)} samples  mem={mem()}")

    # Count .bim lines without full parsing (fast: just count newlines)
    log(f"counting .bim variants: {bim_path}")
    bim_size = bim_path.stat().st_size
    with bim_path.open("rb") as handle:
        variant_count = sum(1 for _ in handle)
    log(f"  .bim: {variant_count} variants ({bim_size / 1e6:.1f} MB)  mem={mem()}")

    if variant_count == 0:
        raise ValueError("PLINK bed contains no variants: " + str(bed_path))
    return _PlinkMetadata(
        sample_ids=sample_ids,
        variant_count=variant_count,
    )


def _build_plink_variant_defaults_from_stats(
    bed_path: Path,
    variant_stats: VariantStatistics,
) -> list[_VariantDefaults]:
    """Build variant defaults using pre-computed allele frequencies (no data pass)."""
    variant_defaults: list[_VariantDefaults] = []
    for variant_index, bim_record in enumerate(_iter_plink_bim_records(bed_path.with_suffix(".bim"))):
        variant_defaults.append(
            _VariantDefaults(
                variant_id=bim_record.variant_id,
                variant_class=_infer_plink_variant_class(bim_record.allele_1, bim_record.allele_2),
                chromosome=bim_record.chromosome,
                position=bim_record.position,
                length=1.0,
                allele_frequency=float(variant_stats.allele_frequencies[variant_index]),
                quality=1.0,
            )
        )
    return variant_defaults


def _build_variant_records(
    default_variants: Sequence[_VariantDefaults],
    variant_metadata_path: str | Path | None,
) -> list[VariantRecord]:
    metadata_rows_by_id: dict[str, dict[str, str]] = {}
    if variant_metadata_path is not None:
        rows = _read_delimited_rows(variant_metadata_path)
        if not rows:
            raise ValueError("Variant metadata file is empty: " + str(variant_metadata_path))
        _require_columns(available_columns=tuple(rows[0].keys()), required_columns=("variant_id",), context="variant metadata")
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
    log(f"computing predictions for {len(dataset.sample_ids)} samples, trait={model.config.trait_type.value}  mem={mem()}")
    if model.config.trait_type == TraitType.BINARY:
        probabilities = model.predict_proba(dataset.genotypes, dataset.covariates)[:, 1]
        predicted_labels = (probabilities >= 0.5).astype(np.int32)
        log(f"binary predictions: mean_prob={float(np.mean(probabilities)):.4f}  pred_positive={int(np.sum(predicted_labels))}  pred_negative={int(np.sum(1-predicted_labels))}")
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
        training_accuracy = float(np.mean(predicted_labels == dataset.targets))
        training_log_loss_val = float(log_loss(dataset.targets, probabilities, labels=[0.0, 1.0]))
        log(f"training metrics: AUC={training_auc}  log_loss={training_log_loss_val:.4f}  accuracy={training_accuracy:.4f}  mem={mem()}")
        return {
            "training_auc": training_auc,
            "training_log_loss": training_log_loss_val,
            "training_accuracy": training_accuracy,
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
    return list(_iter_delimited_rows(_inspect_delimited_table(path)))


def _inspect_delimited_table(path: str | Path) -> _DelimitedTableSpec:
    resolved_path = Path(path)
    with resolved_path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        delimiter = _infer_delimiter(sample)
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError("Table has no header row: " + str(resolved_path))
        columns = tuple(str(field_name) for field_name in reader.fieldnames)
    return _DelimitedTableSpec(
        path=resolved_path,
        delimiter=delimiter,
        columns=columns,
    )


def _iter_delimited_rows(table_spec: _DelimitedTableSpec) -> Iterator[dict[str, str]]:
    with table_spec.path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=table_spec.delimiter)
        for row in reader:
            yield {
                str(key): "" if value is None else str(value)
                for key, value in row.items()
            }


def _infer_delimiter(sample: str) -> str:
    tab_count = sample.count("\t")
    comma_count = sample.count(",")
    if tab_count == 0 and comma_count == 0:
        raise ValueError("Expected a tab-delimited or comma-delimited file with a header row.")
    return "\t" if tab_count >= comma_count else ","


def _require_columns(
    available_columns: Sequence[str],
    required_columns: Sequence[str],
    context: str,
) -> None:
    available_column_set = set(available_columns)
    missing_columns = [column_name for column_name in required_columns if column_name not in available_column_set]
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


def _reorder_sample_table_by_source_index(
    sample_table: _SampleTable,
    source_indices: np.ndarray,
) -> tuple[_SampleTable, np.ndarray, bool]:
    sort_order = np.argsort(source_indices, kind="stable")
    if np.array_equal(sort_order, np.arange(sort_order.shape[0], dtype=sort_order.dtype)):
        return sample_table, np.asarray(source_indices, dtype=np.int32), False
    reordered_sample_table = _SampleTable(
        sample_ids=[sample_table.sample_ids[int(sample_position)] for sample_position in sort_order],
        covariates=np.asarray(sample_table.covariates[sort_order], dtype=np.float32),
        targets=np.asarray(sample_table.targets[sort_order], dtype=np.float32),
    )
    reordered_source_indices = np.asarray(source_indices[sort_order], dtype=np.int32)
    return reordered_sample_table, reordered_source_indices, True


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




@dataclass(slots=True)
class _PlinkBimRecord:
    chromosome: str
    variant_id: str
    position: int
    allele_1: str
    allele_2: str


def _read_plink_sample_ids(fam_path: Path) -> list[str]:
    sample_ids: list[str] = []
    with fam_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped_line = raw_line.strip()
            if not stripped_line:
                continue
            fields = stripped_line.split()
            if len(fields) < 2:
                raise ValueError(f"Malformed PLINK .fam row at line {line_number}: expected at least 2 columns.")
            sample_ids.append(fields[1])
    if not sample_ids:
        raise ValueError("PLINK .fam contains no samples: " + str(fam_path))
    return sample_ids


def _iter_plink_bim_records(bim_path: Path) -> Iterator[_PlinkBimRecord]:
    with bim_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped_line = raw_line.strip()
            if not stripped_line:
                continue
            fields = stripped_line.split()
            if len(fields) < 6:
                raise ValueError(f"Malformed PLINK .bim row at line {line_number}: expected 6 columns.")
            yield _PlinkBimRecord(
                chromosome=fields[0],
                variant_id=fields[1],
                position=int(fields[3]),
                allele_1=fields[4],
                allele_2=fields[5],
            )


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

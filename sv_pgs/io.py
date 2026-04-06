from __future__ import annotations

import csv
import gzip
import hashlib
import json
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Iterator, Literal, Sequence, TextIO

import numpy as np
from sklearn.metrics import log_loss, r2_score, roc_auc_score

from sv_pgs.config import ModelConfig, TraitType, VariantClass
from sv_pgs.data import VariantRecord, VariantStatistics
from sv_pgs.genotype import (
    PLINK_MISSING_INT8,
    ConcatenatedRawGenotypeMatrix,
    PlinkRawGenotypeMatrix,
    RawGenotypeMatrix,
    as_raw_genotype_matrix,
)
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
    column_index_by_name: dict[str, int]




@dataclass(slots=True)
class _TextVcfRecord:
    CHROM: str
    POS: int
    ID: str | None
    REF: str
    ALT: tuple[str, ...]
    QUAL: float | None
    INFO: dict[str, Any]
    gt_types: np.ndarray
    end: int | None

    @property
    def is_snp(self) -> bool:
        return len(self.ALT) == 1 and len(self.REF) == 1 and len(self.ALT[0]) == 1 and not self.ALT[0].startswith("<")

    @property
    def is_indel(self) -> bool:
        return len(self.ALT) == 1 and not self.ALT[0].startswith("<") and len(self.REF) != len(self.ALT[0])

    @property
    def is_sv(self) -> bool:
        alt = self.ALT[0] if self.ALT else ""
        if alt.startswith("<") and alt.endswith(">"):
            return True
        svtype = self.INFO.get("SVTYPE")
        if isinstance(svtype, (tuple, list)):
            return len(svtype) > 0
        return svtype is not None


class _TextVcfReader:
    __slots__ = ("_path", "samples", "seqnames", "seqlens")

    def __init__(self, vcf_path: Path) -> None:
        self._path = Path(vcf_path)
        self.samples: tuple[str, ...] = ()
        self.seqnames: tuple[str, ...] = ()
        self.seqlens: tuple[int, ...] = ()
        self._load_header()

    def _load_header(self) -> None:
        sample_names: tuple[str, ...] = ()
        contig_lengths: dict[str, int] = {}
        with _open_vcf_text(self._path) as handle:
            for raw_line in handle:
                line = raw_line.rstrip("\n")
                if line.startswith("##contig=<"):
                    contig_name, contig_length = _parse_contig_header(line)
                    if contig_name is not None:
                        contig_lengths[contig_name] = contig_length
                    continue
                if not line.startswith("#CHROM\t"):
                    continue
                header_fields = line.split("\t")
                sample_names = tuple(header_fields[9:])
                break
        self.samples = sample_names
        self.seqnames = tuple(contig_lengths.keys())
        self.seqlens = tuple(contig_lengths[contig_name] for contig_name in self.seqnames)

    def close(self) -> None:
        return None

    def set_threads(self, _threads: int) -> None:
        return None

    def __iter__(self) -> Iterator[_TextVcfRecord]:
        return self._iter_records(region=None)

    def __call__(self, region: str) -> Iterator[_TextVcfRecord]:
        return self._iter_records(region=region)

    def _iter_records(self, region: str | None) -> Iterator[_TextVcfRecord]:
        region_filter = _parse_vcf_region(region)
        with _open_vcf_text(self._path) as handle:
            for raw_line in handle:
                if raw_line.startswith("#"):
                    continue
                record = _parse_text_vcf_record(raw_line, sample_names=self.samples)
                if region_filter is not None and not _text_vcf_record_in_region(record, region_filter):
                    continue
                yield record


def _open_vcf_text(vcf_path: Path) -> Any:
    if vcf_path.suffix == ".gz":
        return gzip.open(vcf_path, "rt", encoding="utf-8")
    return vcf_path.open("r", encoding="utf-8")


def _parse_contig_header(line: str) -> tuple[str | None, int]:
    content = line.removeprefix("##contig=<").removesuffix(">")
    fields = [field.strip() for field in content.split(",")]
    contig_name: str | None = None
    contig_length = 0
    for field in fields:
        if field.startswith("ID="):
            contig_name = field.split("=", 1)[1]
        elif field.startswith("length="):
            contig_length = int(field.split("=", 1)[1])
    return contig_name, contig_length


def _parse_vcf_region(region: str | None) -> tuple[str, int, int | None] | None:
    if region is None:
        return None
    if ":" not in region:
        return region, 1, None
    chrom, coordinates = region.split(":", 1)
    if "-" not in coordinates:
        return chrom, int(coordinates), int(coordinates)
    start_text, end_text = coordinates.split("-", 1)
    return chrom, int(start_text), int(end_text)


def _text_vcf_record_in_region(record: _TextVcfRecord, region_filter: tuple[str, int, int | None]) -> bool:
    chrom, start, end = region_filter
    if record.CHROM != chrom:
        return False
    if record.POS < start:
        return False
    if end is not None and record.POS > end:
        return False
    return True


def _parse_text_vcf_record(line: str, *, sample_names: tuple[str, ...]) -> _TextVcfRecord:
    fields = line.rstrip("\n").split("\t")
    chrom = fields[0]
    pos = int(fields[1])
    record_id = None if fields[2] == "." else fields[2]
    ref = fields[3]
    alt_field = fields[4]
    alt = () if alt_field == "." else tuple(alt_field.split(","))
    qual = None if fields[5] == "." else float(fields[5])
    info = _parse_vcf_info(fields[7])
    gt_types = _parse_vcf_gt_types(fields[8], fields[9:9 + len(sample_names)])
    end = info.get("END")
    return _TextVcfRecord(
        CHROM=chrom,
        POS=pos,
        ID=record_id,
        REF=ref,
        ALT=alt,
        QUAL=qual,
        INFO=info,
        gt_types=gt_types,
        end=int(end) if end is not None else None,
    )


def _parse_vcf_info(info_field: str) -> dict[str, Any]:
    if info_field == ".":
        return {}
    parsed: dict[str, Any] = {}
    for token in info_field.split(";"):
        if "=" not in token:
            parsed[token] = True
            continue
        key, raw_value = token.split("=", 1)
        if "," in raw_value:
            parsed[key] = tuple(_parse_vcf_scalar(value) for value in raw_value.split(","))
        else:
            parsed[key] = _parse_vcf_scalar(raw_value)
    return parsed


def _parse_vcf_scalar(value: str) -> Any:
    if value == ".":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _parse_vcf_gt_types(format_field: str, sample_fields: Sequence[str]) -> np.ndarray:
    format_tokens = format_field.split(":")
    try:
        gt_index = format_tokens.index("GT")
    except ValueError:
        return np.full(len(sample_fields), 2, dtype=np.int8)
    gt_types = np.empty(len(sample_fields), dtype=np.int8)
    for sample_index, sample_field in enumerate(sample_fields):
        sample_tokens = sample_field.split(":")
        gt_token = sample_tokens[gt_index] if gt_index < len(sample_tokens) else "."
        gt_types[sample_index] = _gt_type_from_token(gt_token)
    return gt_types


def _gt_type_from_token(gt_token: str) -> int:
    if gt_token == "." or gt_token == "./." or gt_token == ".|.":
        return 2
    alleles = gt_token.replace("|", "/").split("/")
    if not alleles:
        return 2
    saw_ref = False
    saw_alt = False
    for allele in alleles:
        if allele == "." or allele == "":
            return 2
        if allele == "0":
            saw_ref = True
        else:
            saw_alt = True
    if saw_ref and saw_alt:
        return 1
    if saw_alt:
        return 3
    return 0


def _open_vcf_reader(vcf_path: Path) -> Any:
    try:
        from cyvcf2 import VCF

        return VCF(str(vcf_path))
    except ModuleNotFoundError:
        return _TextVcfReader(vcf_path)


def _vcf_record_count_hint(reader: Any) -> int | None:
    for attribute_name in ("num_records", "nrecords"):
        value = getattr(reader, attribute_name, None)
        if callable(value):
            try:
                value = value()
            except TypeError:
                continue
        if value is None:
            continue
        resolved_value = int(value)
        if resolved_value >= 0:
            return resolved_value
    return None


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
        vcf_header_reader = _open_vcf_reader(source_path)
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
        genotype_matrix, default_variants, variant_stats = _load_vcf_with_cache(
            source_path,
            keep_sample_indices=keep_indices,
            mmap_mode="r",
        )

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
        if genotype_matrix is None:
            raise RuntimeError("VCF genotype matrix was not initialized.")
        raw_genotypes = as_raw_genotype_matrix(genotype_matrix)
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
        log("computing variant statistics (single pass, JAX)...")
        variant_stats = compute_variant_statistics(
            raw_genotypes,
            config=ModelConfig(),
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


def load_multi_vcf_dataset_from_files(
    genotype_paths: Sequence[str | Path],
    sample_table_path: str | Path,
    target_column: str,
    covariate_columns: Sequence[str],
    *,
    sample_id_column: str = "auto",
    variant_metadata_path: str | Path | None = None,
) -> LoadedDataset:
    log(f"=== LOAD MULTI-VCF DATASET START ===  chromosomes={len(genotype_paths)}  mem={mem()}")
    source_paths = [Path(path) for path in genotype_paths]
    if not source_paths:
        raise ValueError("genotype_paths cannot be empty.")
    resolved_paths = [path.resolve() for path in source_paths]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("genotype_paths must be unique.")

    log(f"reading sample table header: {sample_table_path}")
    sample_table_spec = _inspect_delimited_table(sample_table_path)
    log(f"sample table columns={list(sample_table_spec.columns)}")

    first_source_path = source_paths[0]
    log(f"reading VCF header for sample IDs: {first_source_path}")
    source_sample_ids = _read_vcf_sample_ids(first_source_path)
    log(f"VCF has {len(source_sample_ids)} samples")

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
    keep_sample_indices = np.asarray(aligned_sample_indices, dtype=np.intp)
    log(f"aligned {len(aligned_sample_indices)} phenotype rows against {len(source_sample_ids)} genotype samples")

    raw_matrices: list[RawGenotypeMatrix] = []
    default_variants: list[_VariantDefaults] = []
    variant_stats_parts: list[VariantStatistics] = []
    for source_path in source_paths:
        log(f"loading chromosome VCF for unified fit: {source_path}")
        chromosome_sample_ids = _read_vcf_sample_ids(source_path)
        if chromosome_sample_ids != source_sample_ids:
            raise RuntimeError(f"VCF sample IDs do not match the first chromosome: {source_path}")
        genotype_matrix, chromosome_variants, chromosome_stats = _load_vcf_with_cache(
            source_path,
            keep_sample_indices=keep_sample_indices,
            mmap_mode="r",
        )
        raw_matrices.append(as_raw_genotype_matrix(genotype_matrix))
        default_variants.extend(chromosome_variants)
        variant_stats_parts.append(chromosome_stats)
        log(
            f"  chromosome ready: {genotype_matrix.shape[0]} samples x {genotype_matrix.shape[1]} variants  "
            f"total_variants_so_far={len(default_variants)}  mem={mem()}"
        )

    raw_genotypes: RawGenotypeMatrix = ConcatenatedRawGenotypeMatrix(tuple(raw_matrices))
    variant_stats = VariantStatistics(
        means=np.concatenate([stats.means for stats in variant_stats_parts]).astype(np.float32, copy=False),
        scales=np.concatenate([stats.scales for stats in variant_stats_parts]).astype(np.float32, copy=False),
        allele_frequencies=np.concatenate([stats.allele_frequencies for stats in variant_stats_parts]).astype(np.float32, copy=False),
        support_counts=np.concatenate([stats.support_counts for stats in variant_stats_parts]).astype(np.int32, copy=False),
    )
    _validate_multi_vcf_variant_keys(default_variants)

    log("building variant records from defaults + optional metadata...")
    variant_records = _build_variant_records(
        default_variants=default_variants,
        variant_metadata_path=variant_metadata_path,
    )
    sv_count = sum(1 for vr in variant_records if vr.variant_class.value not in ("snv", "small_indel"))
    snv_count = sum(1 for vr in variant_records if vr.variant_class.value == "snv")
    log(f"variant records: {len(variant_records)} total ({snv_count} SNVs, {sv_count} structural variants)")

    log(f"=== LOAD MULTI-VCF DATASET DONE === final shape={raw_genotypes.shape}  mem={mem()}")
    return LoadedDataset(
        sample_ids=list(sample_table.sample_ids),
        genotypes=raw_genotypes,
        covariates=np.asarray(sample_table.covariates, dtype=np.float32),
        targets=np.asarray(sample_table.targets, dtype=np.float32),
        variant_records=variant_records,
        variant_stats=variant_stats,
    )


def _validate_multi_vcf_variant_keys(default_variants: Sequence[_VariantDefaults]) -> None:
    seen_keys: set[tuple[str, int, str]] = set()
    duplicate_keys: list[tuple[str, int, str]] = []
    for variant in default_variants:
        variant_key = (str(variant.chromosome), int(variant.position), str(variant.variant_id))
        if variant_key in seen_keys:
            duplicate_keys.append(variant_key)
        else:
            seen_keys.add(variant_key)
    if duplicate_keys:
        preview = ", ".join(f"{chrom}:{position}:{variant_id}" for chrom, position, variant_id in duplicate_keys[:3])
        raise ValueError(f"duplicate variants detected across genotype_paths: {preview}")


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
                _format_float(_coerce_float(coefficient_row["beta"])),
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
    sample_id_index = table_spec.column_index_by_name[sample_id_column]
    target_index = table_spec.column_index_by_name[target_column]
    covariate_indices = tuple(table_spec.column_index_by_name[column_name] for column_name in covariate_columns)

    sample_ids: list[str] = []
    covariates: list[list[float]] = []
    targets: list[float] = []
    seen_sample_ids: set[str] = set()
    available_sample_id_set = set(available_sample_ids)
    total_rows = 0
    unmatched_rows = 0

    for row_values in _iter_delimited_row_values(table_spec):
        total_rows += 1
        sample_id = row_values[sample_id_index].strip()
        if not sample_id:
            raise ValueError("Encountered blank sample identifier in sample table.")
        if sample_id not in available_sample_id_set:
            unmatched_rows += 1
            continue
        if sample_id in seen_sample_ids:
            raise ValueError("Duplicate sample identifier in sample table: " + sample_id)
        seen_sample_ids.add(sample_id)
        # Parse target and covariates; drop rows with missing values
        try:
            target_value = float(row_values[target_index])
            covariate_values = [float(row_values[column_index]) for column_index in covariate_indices]
        except (ValueError, TypeError):
            unmatched_rows += 1
            continue
        # Drop rows where any value is NaN (e.g. missing PCs after merge)
        if np.isnan(target_value) or any(np.isnan(v) for v in covariate_values):
            unmatched_rows += 1
            continue
        sample_ids.append(sample_id)
        targets.append(target_value)
        covariates.append(covariate_values)

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
    candidate_indices = {
        column_name: table_spec.column_index_by_name[column_name]
        for column_name in candidate_columns
    }
    # Valid overlaps can start well after early rows in large phenotype exports,
    # so resolve against the full table instead of an initial sample.
    match_counts = {column_name: 0 for column_name in candidate_columns}
    for row_values in _iter_delimited_row_values(table_spec):
        for column_name in candidate_columns:
            if row_values[candidate_indices[column_name]].strip() in available_sample_id_set:
                match_counts[column_name] += 1
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
_VCF_CACHE_MANIFEST_VERSION = 1
_VCF_CACHE_STATS_DTYPE = np.dtype(
    [
        ("means", "<f4"),
        ("scales", "<f4"),
        ("allele_frequencies", "<f4"),
        ("support_counts", "<i4"),
    ]
)


@dataclass(slots=True)
class _VcfCachePaths:
    key: str
    cache_dir: Path
    geno_path: Path
    var_path: Path
    stats_path: Path
    legacy_stats_path: Path
    manifest_path: Path


def _cache_file_fingerprint(path: Path, sample_bytes: int = 1_048_576) -> bytes:
    """Return a cheap content fingerprint that is stable across re-downloads."""
    stat = path.stat()
    h = hashlib.sha256()
    h.update(f"{stat.st_size}:".encode())
    with path.open("rb") as handle:
        head = handle.read(sample_bytes)
        h.update(head)
        if stat.st_size > sample_bytes:
            tail_offset = max(len(head), stat.st_size - sample_bytes)
            handle.seek(tail_offset)
            h.update(handle.read(sample_bytes))
    return h.digest()


def _vcf_cache_key(vcf_path: Path, keep_sample_indices: np.ndarray | None) -> str:
    """Compute a hex digest that uniquely identifies a VCF + sample-subset."""
    h = hashlib.sha256()
    h.update(f"v{_CACHE_VERSION}:".encode())
    h.update(str(vcf_path.resolve()).encode())
    h.update(_cache_file_fingerprint(vcf_path))
    if keep_sample_indices is not None:
        h.update(keep_sample_indices.tobytes())
    return h.hexdigest()[:24]


def _vcf_cache_dir(vcf_path: Path) -> Path:
    return vcf_path.resolve().parent / _CACHE_DIR_NAME


def _vcf_cache_paths(vcf_path: Path, keep_sample_indices: np.ndarray | None) -> _VcfCachePaths:
    cache_dir = _vcf_cache_dir(vcf_path)
    key = _vcf_cache_key(vcf_path, keep_sample_indices)
    return _VcfCachePaths(
        key=key,
        cache_dir=cache_dir,
        geno_path=cache_dir / f"{key}.genotypes.npy",
        var_path=cache_dir / f"{key}.variants.pkl",
        stats_path=cache_dir / f"{key}.stats.npy",
        legacy_stats_path=cache_dir / f"{key}.stats.npz",
        manifest_path=cache_dir / f"{key}.manifest.json",
    )


def _cleanup_stale_vcf_cache_temps(cache_dir: Path, key: str) -> None:
    for stale_path in cache_dir.glob(f"{key}.*.tmp*"):
        if stale_path.is_dir():
            continue
        try:
            stale_path.unlink()
        except Exception:
            pass
    for stale_dir in cache_dir.glob(f"{key}.bundle.*"):
        if not stale_dir.is_dir():
            continue
        try:
            for child in stale_dir.iterdir():
                child.unlink(missing_ok=True)
            stale_dir.rmdir()
        except Exception:
            pass


def _load_vcf_cache_manifest(manifest_path: Path) -> dict[str, Any] | None:
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if int(manifest.get("manifest_version", 0)) != _VCF_CACHE_MANIFEST_VERSION:
        return None
    return manifest


def _write_vcf_cache_manifest(
    manifest_path: Path,
    *,
    sample_count: int,
    variant_count: int,
    stats_file: str,
) -> None:
    _atomic_write_text(
        manifest_path,
        json.dumps(
            {
                "manifest_version": _VCF_CACHE_MANIFEST_VERSION,
                "sample_count": int(sample_count),
                "variant_count": int(variant_count),
                "dtype": "int8",
                "fortran_order": True,
                "stats_file": stats_file,
            }
        ),
    )


def _load_vcf_cache_stats(stats_path: Path) -> VariantStatistics:
    stats_payload = np.load(stats_path, mmap_mode="r")
    try:
        if isinstance(stats_payload, np.lib.npyio.NpzFile):
            return VariantStatistics(
                means=np.asarray(stats_payload["means"], dtype=np.float32),
                scales=np.asarray(stats_payload["scales"], dtype=np.float32),
                allele_frequencies=np.asarray(stats_payload["allele_frequencies"], dtype=np.float32),
                support_counts=np.asarray(stats_payload["support_counts"], dtype=np.int32),
            )
        return VariantStatistics(
            means=np.asarray(stats_payload["means"], dtype=np.float32),
            scales=np.asarray(stats_payload["scales"], dtype=np.float32),
            allele_frequencies=np.asarray(stats_payload["allele_frequencies"], dtype=np.float32),
            support_counts=np.asarray(stats_payload["support_counts"], dtype=np.int32),
        )
    finally:
        if isinstance(stats_payload, np.lib.npyio.NpzFile):
            stats_payload.close()


def _write_vcf_cache_stats(stats_path: Path, variant_stats: VariantStatistics) -> None:
    stats_matrix = np.empty(variant_stats.means.shape[0], dtype=_VCF_CACHE_STATS_DTYPE)
    stats_matrix["means"] = np.asarray(variant_stats.means, dtype=np.float32)
    stats_matrix["scales"] = np.asarray(variant_stats.scales, dtype=np.float32)
    stats_matrix["allele_frequencies"] = np.asarray(variant_stats.allele_frequencies, dtype=np.float32)
    stats_matrix["support_counts"] = np.asarray(variant_stats.support_counts, dtype=np.int32)
    np.save(stats_path, stats_matrix, allow_pickle=False)


def _is_vcf_cache_bundle_complete(paths: _VcfCachePaths) -> bool:
    if paths.manifest_path.exists():
        manifest = _load_vcf_cache_manifest(paths.manifest_path)
        if manifest is None:
            return False
        stats_filename = str(manifest.get("stats_file", paths.stats_path.name))
        return paths.geno_path.exists() and paths.var_path.exists() and (paths.cache_dir / stats_filename).exists()
    return paths.geno_path.exists() and paths.var_path.exists() and paths.legacy_stats_path.exists()


def _ensure_vcf_cache_matrix_fast(paths: _VcfCachePaths, genotype_matrix: np.ndarray) -> np.ndarray:
    if genotype_matrix.flags.f_contiguous and not genotype_matrix.flags.c_contiguous:
        return genotype_matrix
    with open(paths.var_path, "rb") as handle:
        variants = pickle.load(handle)
    stats_path = paths.stats_path if paths.stats_path.exists() else paths.legacy_stats_path
    variant_stats = _load_vcf_cache_stats(stats_path)
    _save_vcf_to_cache(
        vcf_path=paths.geno_path,
        keep_sample_indices=None,
        genotype_matrix=np.asarray(genotype_matrix, dtype=np.int8),
        variants=variants,
        variant_stats=variant_stats,
        cache_paths=paths,
    )
    return np.load(paths.geno_path, mmap_mode="r")


def _upgrade_legacy_vcf_cache_bundle(
    paths: _VcfCachePaths,
    genotype_matrix: np.ndarray,
    variant_stats: VariantStatistics,
) -> None:
    created_stats = False
    try:
        if not paths.stats_path.exists():
            stats_tmp = paths.cache_dir / f"{paths.key}.stats.tmp.npy"
            _write_vcf_cache_stats(stats_tmp, variant_stats)
            stats_tmp.replace(paths.stats_path)
            created_stats = True
        _write_vcf_cache_manifest(
            paths.manifest_path,
            sample_count=int(genotype_matrix.shape[0]),
            variant_count=int(genotype_matrix.shape[1]),
            stats_file=paths.stats_path.name,
        )
    except Exception:
        try:
            (paths.cache_dir / f"{paths.key}.stats.tmp.npy").unlink(missing_ok=True)
        except Exception:
            pass
        if created_stats:
            try:
                paths.stats_path.unlink(missing_ok=True)
            except Exception:
                pass


def _load_vcf_from_cache(
    vcf_path: Path,
    keep_sample_indices: np.ndarray | None,
    *,
    mmap_mode: Literal["r", "r+", "w+", "c"] | None = None,
) -> tuple[np.ndarray, list[_VariantDefaults], VariantStatistics] | None:
    """Try to load cached VCF parse results. Returns None on miss."""
    paths = _vcf_cache_paths(vcf_path, keep_sample_indices)
    if not paths.cache_dir.exists():
        return None

    _cleanup_stale_vcf_cache_temps(paths.cache_dir, paths.key)
    if not _is_vcf_cache_bundle_complete(paths):
        log(f"VCF cache miss (key={paths.key})")
        return None

    try:
        log(f"VCF cache hit — loading from {paths.cache_dir.name}/{paths.key}.*")
        effective_mmap_mode: Literal["r", "r+", "w+", "c"] = "r" if mmap_mode is None else mmap_mode
        manifest = _load_vcf_cache_manifest(paths.manifest_path)
        stats_path = paths.legacy_stats_path
        expected_sample_count: int | None = None
        expected_variant_count: int | None = None
        if manifest is not None:
            expected_sample_count = int(manifest["sample_count"])
            expected_variant_count = int(manifest["variant_count"])
            stats_path = paths.cache_dir / str(manifest.get("stats_file", paths.stats_path.name))

        genotype_matrix = np.load(paths.geno_path, mmap_mode=effective_mmap_mode)
        with open(paths.var_path, "rb") as variant_handle:
            variants = pickle.load(variant_handle)
        variant_stats = _load_vcf_cache_stats(stats_path)
        if expected_sample_count is not None and genotype_matrix.shape[0] != expected_sample_count:
            raise ValueError(f"cached sample count mismatch: {genotype_matrix.shape[0]} != {expected_sample_count}")
        if expected_variant_count is not None and genotype_matrix.shape[1] != expected_variant_count:
            raise ValueError(f"cached variant count mismatch: {genotype_matrix.shape[1]} != {expected_variant_count}")
        stats_lengths = {
            int(variant_stats.means.shape[0]),
            int(variant_stats.scales.shape[0]),
            int(variant_stats.allele_frequencies.shape[0]),
            int(variant_stats.support_counts.shape[0]),
        }
        if len(stats_lengths) != 1:
            raise ValueError("cached stats shape mismatch")
        genotype_matrix = _ensure_vcf_cache_matrix_fast(paths, genotype_matrix)
        if manifest is None:
            _upgrade_legacy_vcf_cache_bundle(paths, genotype_matrix, variant_stats)
        log(f"  cached matrix {genotype_matrix.shape}, {len(variants)} variants")
        return genotype_matrix, variants, variant_stats
    except Exception as exc:
        log(f"VCF cache load failed ({exc}), will re-parse")
        return None


def _read_vcf_sample_ids(vcf_path: Path) -> list[str]:
    reader = _open_vcf_reader(vcf_path)
    try:
        return [str(sample_id) for sample_id in reader.samples]
    finally:
        reader.close()


_INCREMENTAL_CHECKPOINT_INTERVAL = 5000  # variants between disk flushes


def _vcf_contig_info(vcf_path: Path) -> tuple[str, int] | None:
    """Get (chromosome_name, length) for the chromosome that has data in this VCF.

    Reads the first record to find which chromosome has data, then looks up
    its length from the header contigs. Returns length=0 if unknown.
    """
    reader = _open_vcf_reader(vcf_path)
    try:
        seqnames = list(reader.seqnames) if hasattr(reader, "seqnames") else []
        seqlens = list(reader.seqlens) if hasattr(reader, "seqlens") else []
        contig_lengths = {
            str(name): int(length)
            for name, length in zip(seqnames, seqlens)
            if length and int(length) > 0
        }
        # Read first record to find which chromosome actually has data
        for record in reader:
            chrom = str(record.CHROM)
            length = contig_lengths.get(chrom, 0)
            return chrom, length
        return None
    finally:
        reader.close()


def _split_into_regions(chrom: str, chrom_length: int, n_regions: int) -> list[str]:
    """Split a chromosome into n_regions non-overlapping tabix region strings."""
    if chrom_length <= 0 or n_regions <= 1:
        return [chrom]  # can't split without known length
    step = max(chrom_length // n_regions, 1)
    regions = []
    for i in range(n_regions):
        start = i * step + 1
        end = (i + 1) * step if i < n_regions - 1 else chrom_length
        regions.append(f"{chrom}:{start}-{end}")
    return regions


def _region_parse_worker(args: tuple) -> tuple[int, str]:
    """Worker: parse one region of one VCF, write raw binary output files.
    Runs in a separate process. Returns (variant_count, output_prefix)."""
    import json as _json
    import struct
    import sys
    import time
    vcf_path_str, region, keep_indices_list, output_prefix = args
    keep_indices = np.array(keep_indices_list, dtype=np.intp) if keep_indices_list is not None else None
    vcf_name = Path(vcf_path_str).name

    reader = _open_vcf_reader(Path(vcf_path_str))
    reader.set_threads(1)

    gt_map = np.array([0, 1, PLINK_MISSING_INT8, 2], dtype=np.int8)
    stats_pack = struct.Struct("<qqii")
    geno_fh = open(f"{output_prefix}.geno", "wb")
    var_fh = open(f"{output_prefix}.var", "w")
    stats_fh = open(f"{output_prefix}.stats", "wb")

    count = 0
    t_start = time.monotonic()
    last_log = t_start
    iterator = reader(region) if region else reader
    for record in iterator:
        if len(record.ALT) != 1:
            continue
        col = gt_map[record.gt_types]
        if keep_indices is not None:
            col = col[keep_indices]
        geno_fh.write(col.tobytes())
        observed = col[col >= 0].astype(np.int32, copy=False)
        stats_fh.write(stats_pack.pack(
            int(np.sum(observed, dtype=np.int64)),
            int(np.sum(observed * observed, dtype=np.int64)),
            int(observed.shape[0]),
            int(np.count_nonzero(observed > 0)),
        ))
        vd = _variant_defaults_from_vcf_record(record)
        var_fh.write(_json.dumps({
            "variant_id": vd.variant_id, "variant_class": vd.variant_class.value,
            "chromosome": vd.chromosome, "position": vd.position,
            "length": vd.length, "allele_frequency": vd.allele_frequency,
            "quality": vd.quality,
        }) + "\n")
        count += 1
        now = time.monotonic()
        if now - last_log >= 10.0:
            rate = count / max(now - t_start, 0.01)
            print(f"  [worker] {vcf_name}: {count} variants ({rate:.0f}/s)", file=sys.stderr, flush=True)
            last_log = now

    geno_fh.close()
    var_fh.close()
    stats_fh.close()
    reader.close()
    elapsed = time.monotonic() - t_start
    print(f"  [worker] {vcf_name}: DONE {count} variants in {elapsed:.0f}s", file=sys.stderr, flush=True)
    return count, output_prefix


def precache_vcfs_parallel(
    vcf_paths: list[Path],
    keep_sample_indices: np.ndarray,
) -> None:
    """Parse and cache multiple VCFs in parallel, auto-detecting CPU count.

    Allocates workers across AND within chromosomes proportional to file size.
    Uses VCF header contig lengths for region splitting (no hardcoded lengths).
    If contig length is unknown, that VCF gets 1 worker (no splitting).

    Fully compatible with existing .npy cache — already-cached VCFs are skipped.
    Produces the same final cache format via the incremental → .npy pipeline.
    """
    import multiprocessing
    import os
    import shutil

    total_cpus = os.cpu_count() or 4

    # Skip already-cached VCFs (compatible with existing .npy cache)
    uncached: list[Path] = []
    for vcf_path in vcf_paths:
        cache_paths = _vcf_cache_paths(vcf_path, keep_sample_indices)
        if not _is_vcf_cache_bundle_complete(cache_paths):
            uncached.append(vcf_path)

    if not uncached:
        log(f"all {len(vcf_paths)} VCFs already cached")
        return

    # Get contig info and file sizes for allocation
    vcf_info: dict[Path, tuple[str | None, int, int]] = {}  # path → (chrom, length, file_size)
    for vcf_path in uncached:
        info = _vcf_contig_info(vcf_path)
        chrom = info[0] if info else None
        chrom_length = info[1] if info else 0
        vcf_info[vcf_path] = (chrom, chrom_length, vcf_path.stat().st_size)

    total_size = sum(v[2] for v in vcf_info.values())

    # Allocate workers proportional to file size
    # Each VCF gets at least 1, large VCFs get more if CPUs available
    allocation: dict[Path, int] = {}
    for vcf_path, (chrom, chrom_length, file_size) in vcf_info.items():
        share = file_size / max(total_size, 1)
        n_workers = max(1, round(share * total_cpus))
        # Can't split if we don't know the contig length
        if chrom_length <= 0:
            n_workers = 1
        allocation[vcf_path] = n_workers

    # Build task list
    keep_list = keep_sample_indices.tolist() if keep_sample_indices is not None else None
    tasks: list[tuple] = []
    for vcf_path, n_workers in allocation.items():
        chrom, chrom_length, _ = vcf_info[vcf_path]
        cache_dir = _vcf_cache_dir(vcf_path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = _vcf_cache_key(vcf_path, keep_sample_indices)
        tmp_dir = cache_dir / f"{key}.tmp_parallel"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        if n_workers <= 1 or chrom is None:
            # Single worker: parse entire VCF (no region filter)
            tasks.append((str(vcf_path), None, keep_list, str(tmp_dir / "region_0")))
        else:
            regions = _split_into_regions(chrom, chrom_length, n_workers)
            for i, region in enumerate(regions):
                tasks.append((str(vcf_path), region, keep_list, str(tmp_dir / f"region_{i}")))

    log(f"parallel VCF precache: {len(uncached)} VCFs, {len(tasks)} tasks, {total_cpus} CPUs")
    for vcf_path, n_workers in allocation.items():
        chrom, chrom_length, file_size = vcf_info[vcf_path]
        log(f"  {vcf_path.name}: {n_workers} workers, {chrom}:{chrom_length}, {file_size/1e9:.1f} GB")

    # Parse all regions in parallel
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=min(total_cpus, len(tasks))) as pool:
        for count, prefix in pool.imap_unordered(_region_parse_worker, tasks):
            log(f"  region done: {Path(prefix).name} ({count} variants)")

    # Merge region results per VCF → incremental cache → final .npy cache
    for vcf_path in uncached:
        cache_dir = _vcf_cache_dir(vcf_path)
        key = _vcf_cache_key(vcf_path, keep_sample_indices)
        tmp_dir = cache_dir / f"{key}.tmp_parallel"
        if not tmp_dir.exists():
            continue

        # Find region files sorted by index
        geno_files = sorted(tmp_dir.glob("region_*.geno"), key=lambda p: int(p.stem.split("_")[1]))
        if not geno_files:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            continue

        # Concatenate regions → incremental cache files
        inc_geno = cache_dir / f"{key}.inc.genotypes.bin"
        inc_var = cache_dir / f"{key}.inc.variants.jsonl"
        inc_stats = cache_dir / f"{key}.inc.stats.bin"
        n_total = 0
        with open(inc_geno, "wb") as gout, open(inc_var, "w") as vout, open(inc_stats, "wb") as sout:
            for geno_file in geno_files:
                prefix = str(geno_file).removesuffix(".geno")
                with open(f"{prefix}.geno", "rb") as f:
                    shutil.copyfileobj(f, gout)
                with open(f"{prefix}.var") as f:
                    vout.write(f.read())
                with open(f"{prefix}.stats", "rb") as f:
                    data = f.read()
                    sout.write(data)
                    n_total += len(data) // 24

        # Finalize: convert incremental binary directly to .npy cache.
        # Do NOT call _load_vcf_with_cache — that would re-open the VCF and
        # waste 20+ min skipping already-parsed variants.
        actual_n_keep = len(keep_sample_indices) if keep_sample_indices is not None else len(_read_vcf_sample_ids(vcf_path))
        log(f"  finalizing {vcf_path.name}: {n_total} variants from {len(geno_files)} regions")

        # Load incremental binary via memmap (zero copy)
        inc_matrix = np.memmap(inc_geno, dtype=np.int8, mode="r", shape=(n_total, actual_n_keep)).T

        # Load variant metadata from JSONL
        import json as _json_mod
        inc_variants: list[_VariantDefaults] = []
        with open(inc_var) as vf:
            for line in vf:
                d = _json_mod.loads(line)
                inc_variants.append(_VariantDefaults(
                    variant_id=d["variant_id"],
                    variant_class=VariantClass(d["variant_class"]),
                    chromosome=d["chromosome"],
                    position=d["position"],
                    length=d["length"],
                    allele_frequency=d["allele_frequency"],
                    quality=d["quality"],
                ))

        # Load stats via structured dtype
        stats_dtype = np.dtype([("sum", "<i8"), ("sum_sq", "<i8"), ("n_valid", "<i4"), ("support", "<i4")])
        stats_arr = np.fromfile(str(inc_stats), dtype=stats_dtype, count=n_total)
        col_sums = stats_arr["sum"]
        col_sum_sq = stats_arr["sum_sq"]
        n_valid_arr = stats_arr["n_valid"]
        support_arr = stats_arr["support"]
        safe_n = np.maximum(n_valid_arr.astype(np.int64), 1).astype(np.float64)
        means = (col_sums / safe_n).astype(np.float32)
        afs = np.clip(means / 2.0, 0.0, 1.0).astype(np.float32)
        css = np.maximum(col_sum_sq.astype(np.float64) - col_sums.astype(np.float64) ** 2 / safe_n, 0.0)
        scales = np.sqrt(css / max(actual_n_keep, 1)).astype(np.float32)
        scales = np.where(scales < 1e-6, 1.0, scales)
        inc_stats_obj = VariantStatistics(means=means, scales=scales, allele_frequencies=afs, support_counts=support_arr)

        # Save as final .npy cache
        _save_vcf_to_cache(vcf_path, keep_sample_indices, inc_matrix, inc_variants, inc_stats_obj)
        del inc_matrix

        # Clean up incremental files
        for p in (inc_geno, inc_var, inc_stats):
            p.unlink(missing_ok=True)

        # Clean up temp region files
        shutil.rmtree(tmp_dir, ignore_errors=True)
        log(f"  {vcf_path.name}: cached")


def _load_vcf_with_cache(
    vcf_path: Path,
    keep_sample_indices: np.ndarray,
    *,
    mmap_mode: Literal["r", "r+", "w+", "c"] | None,
) -> tuple[np.ndarray, list[_VariantDefaults], VariantStatistics]:
    effective_mmap_mode: Literal["r", "r+", "w+", "c"] = "r" if mmap_mode is None else mmap_mode
    # Check for completed cache first
    cached = _load_vcf_from_cache(
        vcf_path,
        keep_sample_indices=keep_sample_indices,
        mmap_mode=effective_mmap_mode,
    )
    if cached is not None:
        return cached

    # Parse with incremental checkpointing
    cache_dir = _vcf_cache_dir(vcf_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _vcf_cache_key(vcf_path, keep_sample_indices)
    genotype_matrix, variants, variant_stats = _load_vcf_incremental(
        vcf_path, keep_sample_indices, cache_dir, key, mmap_mode=effective_mmap_mode,
    )
    return genotype_matrix, variants, variant_stats


def _load_vcf_incremental(
    vcf_path: Path,
    keep_sample_indices: np.ndarray | None,
    cache_dir: Path,
    key: str,
    *,
    mmap_mode: Literal["r", "r+", "w+", "c"],
) -> tuple[np.ndarray, list[_VariantDefaults], VariantStatistics]:
    """Parse VCF with incremental checkpointing. Resumes from last checkpoint if interrupted."""
    import json as _json
    import os
    import struct
    import time

    reader = _open_vcf_reader(vcf_path)
    n_all_samples = len(reader.samples)
    n_keep = len(keep_sample_indices) if keep_sample_indices is not None else n_all_samples

    # Incremental files
    geno_bin = cache_dir / f"{key}.inc.genotypes.bin"
    var_jsonl = cache_dir / f"{key}.inc.variants.jsonl"
    stats_bin = cache_dir / f"{key}.inc.stats.bin"  # 4 int32/int64 values per variant
    progress_file = cache_dir / f"{key}.inc.progress.json"

    # Check for existing progress
    n_cached = 0
    resume_chrom: str | None = None
    resume_pos: int = 0
    if progress_file.exists():
        try:
            prog = _json.loads(progress_file.read_text())
            n_cached = int(prog["n_variants"])
            resume_chrom = prog.get("resume_chrom")
            resume_pos = int(prog.get("resume_pos", 0))
            # Truncate ALL incremental files to exact checkpoint (guard against partial writes)
            expected_geno_bytes = n_cached * n_keep
            expected_stats_bytes = n_cached * 24  # struct "<qqii" = 24 bytes
            if geno_bin.exists() and geno_bin.stat().st_size > expected_geno_bytes:
                with open(geno_bin, "r+b") as f:
                    f.truncate(expected_geno_bytes)
            if stats_bin.exists() and stats_bin.stat().st_size > expected_stats_bytes:
                with open(stats_bin, "r+b") as f:
                    f.truncate(expected_stats_bytes)
            if var_jsonl.exists():
                # Truncate JSONL to exactly n_cached lines
                lines = var_jsonl.read_text().splitlines()[:n_cached]
                var_jsonl.write_text("\n".join(lines) + "\n" if lines else "")
            log(f"  incremental cache: resuming from {n_cached} variants (last pos: {resume_chrom}:{resume_pos})")
        except Exception as exc:
            log(f"  incremental cache progress corrupt ({exc}), starting fresh")
            n_cached = 0
            for p in (geno_bin, var_jsonl, stats_bin, progress_file):
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
    else:
        for p in (geno_bin, var_jsonl, stats_bin):
            p.unlink(missing_ok=True)

    # Reuse the reader opened above for sample IDs
    n_threads = os.cpu_count() or 4
    reader.set_threads(n_threads)
    log(f"  VCF decompression threads: {n_threads}")

    _GT_TO_INT8 = np.array([0, 1, PLINK_MISSING_INT8, 2], dtype=np.int8)
    _gt_to_i8 = _GT_TO_INT8

    record_count_hint = _vcf_record_count_hint(reader) if n_cached == 0 else None
    if record_count_hint is not None and record_count_hint > 0:
        log(f"  record count hint: preallocating for {record_count_hint} variants")
        with open(geno_bin, "wb") as geno_prealloc_fh:
            geno_prealloc_fh.truncate(record_count_hint * n_keep)
        with open(stats_bin, "wb") as stats_prealloc_fh:
            stats_prealloc_fh.truncate(record_count_hint * 24)
        geno_fh: BinaryIO = open(geno_bin, "r+b")
        stats_fh: BinaryIO = open(stats_bin, "r+b")
    else:
        geno_fh = open(geno_bin, "ab")
        stats_fh = open(stats_bin, "ab")
    if n_cached > 0:
        geno_fh.seek(n_cached * n_keep)
        stats_fh.seek(n_cached * 24)
    var_fh: TextIO = open(var_jsonl, "a", encoding="utf-8")
    # Stats: 4 values per variant (sum_i64, sum_sq_i64, n_valid_i32, support_i32) = 24 bytes

    t_start = time.monotonic()
    last_log_time = t_start
    last_chrom = None
    variant_index = 0
    variants_since_checkpoint = 0
    _monotonic = time.monotonic

    # Skip already-cached records
    if n_cached > 0:
        log(f"  skipping {n_cached} already-cached variants...")
        skip_start = time.monotonic()
        skipped = 0
        for record in reader:
            skipped += 1
            if skipped >= n_cached:
                break
        skip_elapsed = time.monotonic() - skip_start
        log(f"  skipped {skipped} variants in {skip_elapsed:.1f}s ({skipped/max(skip_elapsed,0.01):.0f}/s)")
        variant_index = n_cached
        t_start = time.monotonic()
        last_log_time = t_start

    def _abort_incremental_load(message: str) -> None:
        geno_fh.close()
        var_fh.close()
        stats_fh.close()
        reader.close()
        for path in (geno_bin, var_jsonl, stats_bin, progress_file):
            path.unlink(missing_ok=True)
        raise ValueError(message)

    # Parse remaining variants
    for record in reader:
        if len(record.ALT) != 1:
            _abort_incremental_load(
                "Only biallelic VCF records are supported. Normalize multiallelic records before loading: "
                + _vcf_variant_key(record)
            )

        gt = record.gt_types
        int8_col = _gt_to_i8[gt]
        if keep_sample_indices is not None:
            int8_col = int8_col[keep_sample_indices]

        # Write genotype column to disk immediately
        geno_fh.write(int8_col.tobytes())

        # Compute per-variant stats
        observed = int8_col[int8_col >= 0].astype(np.int32, copy=False)
        dosage_sum = int(np.sum(observed, dtype=np.int64))
        dosage_sum_sq = int(np.sum(observed * observed, dtype=np.int64))
        n_valid = observed.shape[0]
        support = int(np.count_nonzero(observed > 0))

        # Write stats (fixed 24 bytes per variant)
        stats_fh.write(struct.pack("<qqii", dosage_sum, dosage_sum_sq, n_valid, support))

        # Write variant metadata
        vd = _variant_defaults_from_vcf_record(record)
        var_fh.write(_json.dumps({
            "variant_id": vd.variant_id, "variant_class": vd.variant_class.value,
            "chromosome": vd.chromosome, "position": vd.position,
            "length": vd.length, "allele_frequency": vd.allele_frequency,
            "quality": vd.quality,
        }) + "\n")

        variant_index += 1
        variants_since_checkpoint += 1

        # Periodic checkpoint
        if variants_since_checkpoint >= _INCREMENTAL_CHECKPOINT_INTERVAL:
            geno_fh.flush()
            var_fh.flush()
            stats_fh.flush()
            os.fsync(geno_fh.fileno())
            os.fsync(var_fh.fileno())
            os.fsync(stats_fh.fileno())
            _atomic_write_text(progress_file, _json.dumps({
                "n_variants": variant_index,
                "n_samples": n_keep,
                "resume_chrom": str(record.CHROM),
                "resume_pos": int(record.POS),
            }))
            variants_since_checkpoint = 0

        # Progress log
        now = _monotonic()
        chrom = str(record.CHROM)
        if chrom != last_chrom:
            if last_chrom is not None:
                log(f"  chromosome {last_chrom} done — {variant_index} variants so far  mem={mem()}")
            last_chrom = chrom
            last_log_time = now
        elif now - last_log_time >= 5.0:
            rate = (variant_index - n_cached) / max(now - t_start, 0.01)
            log(f"  {variant_index} variants loaded ({rate:.0f} variants/s, {chrom})  mem={mem()}")
            last_log_time = now

    # Final flush
    geno_fh.flush()
    var_fh.flush()
    stats_fh.flush()
    geno_fh.close()
    var_fh.close()
    stats_fh.close()
    reader.close()

    if record_count_hint is not None and record_count_hint > variant_index:
        with open(geno_bin, "r+b") as geno_trim_fh:
            geno_trim_fh.truncate(variant_index * n_keep)
        with open(stats_bin, "r+b") as stats_trim_fh:
            stats_trim_fh.truncate(variant_index * 24)

    n_total = variant_index
    if n_total == 0:
        raise ValueError("VCF contains no variants: " + str(vcf_path))

    elapsed = time.monotonic() - t_start
    new_variants = n_total - n_cached
    log(f"  parsed {new_variants} new variants in {elapsed:.1f}s ({n_total} total)")

    # Load variant metadata
    import json as _json2
    variants: list[_VariantDefaults] = []
    with open(var_jsonl, encoding="utf-8") as var_read_handle:
        for line in var_read_handle:
            d = _json2.loads(line)
            variants.append(_VariantDefaults(
                variant_id=d["variant_id"],
                variant_class=VariantClass(d["variant_class"]),
                chromosome=d["chromosome"],
                position=d["position"],
                length=d["length"],
                allele_frequency=d["allele_frequency"],
                quality=d["quality"],
            ))

    # Load stats via numpy structured dtype — no Python loop
    stats_dtype = np.dtype([
        ("sum", "<i8"), ("sum_sq", "<i8"), ("n_valid", "<i4"), ("support", "<i4"),
    ])
    stats_arr = np.fromfile(str(stats_bin), dtype=stats_dtype, count=n_total)
    col_sums = stats_arr["sum"]
    col_sum_sq = stats_arr["sum_sq"]
    n_valid_arr = stats_arr["n_valid"]
    support_arr = stats_arr["support"]

    safe_n_valid = np.maximum(n_valid_arr.astype(np.int64), 1).astype(np.float64)
    means_arr = (col_sums / safe_n_valid).astype(np.float32)
    allele_freqs = np.clip(means_arr / 2.0, 0.0, 1.0).astype(np.float32)
    centered_sum_sq = np.maximum(col_sum_sq.astype(np.float64) - col_sums.astype(np.float64) ** 2 / safe_n_valid, 0.0)
    scales_arr = np.sqrt(centered_sum_sq / max(n_keep, 1)).astype(np.float32)
    scales_arr = np.where(scales_arr < 1e-6, 1.0, scales_arr)

    variant_stats = VariantStatistics(
        means=means_arr,
        scales=scales_arr,
        allele_frequencies=allele_freqs,
        support_counts=support_arr,
    )

    log(f"  finalizing disk-backed VCF cache ({n_keep} samples x {n_total} variants)...")
    incremental_matrix = np.memmap(
        geno_bin,
        dtype=np.int8,
        mode="r",
        shape=(n_total, n_keep),
    ).T
    _save_vcf_to_cache(
        vcf_path=vcf_path,
        keep_sample_indices=keep_sample_indices,
        genotype_matrix=incremental_matrix,
        variants=variants,
        variant_stats=variant_stats,
    )
    del incremental_matrix

    # Clean up incremental files after the final cache is complete.
    for p in (geno_bin, var_jsonl, stats_bin, progress_file):
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass

    cached = _load_vcf_from_cache(
        vcf_path=vcf_path,
        keep_sample_indices=keep_sample_indices,
        mmap_mode=mmap_mode,
    )
    if cached is None:
        raise RuntimeError("VCF cache finalization failed.")
    genotype_matrix, _, _ = cached
    log(f"  incremental parse complete: {genotype_matrix.shape}  mem={mem()}")
    return genotype_matrix, variants, variant_stats


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically via temp file + rename."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.rename(path)


def _save_vcf_to_cache(
    vcf_path: Path,
    keep_sample_indices: np.ndarray | None,
    genotype_matrix: np.ndarray,
    variants: list[_VariantDefaults],
    variant_stats: VariantStatistics,
    *,
    cache_paths: _VcfCachePaths | None = None,
) -> None:
    """Persist parsed VCF results to disk cache."""
    paths = _vcf_cache_paths(vcf_path, keep_sample_indices) if cache_paths is None else cache_paths

    try:
        paths.cache_dir.mkdir(parents=True, exist_ok=True)
        _cleanup_stale_vcf_cache_temps(paths.cache_dir, paths.key)
        genotype_matrix_i8 = genotype_matrix if np.asarray(genotype_matrix).dtype == np.int8 else np.asarray(genotype_matrix, dtype=np.int8)
        sample_count, variant_count = genotype_matrix_i8.shape
        stats_lengths = {
            int(variant_stats.means.shape[0]),
            int(variant_stats.scales.shape[0]),
            int(variant_stats.allele_frequencies.shape[0]),
            int(variant_stats.support_counts.shape[0]),
        }
        if len(stats_lengths) != 1:
            raise ValueError("stats length mismatch during cache save")
        bundle_dir = Path(tempfile.mkdtemp(prefix=f"{paths.key}.bundle.", dir=paths.cache_dir))
        geno_tmp = bundle_dir / paths.geno_path.name
        var_tmp = bundle_dir / paths.var_path.name
        stats_tmp = bundle_dir / paths.stats_path.name
        manifest_tmp = bundle_dir / paths.manifest_path.name
        copy_batch_size = max(1, min(variant_count, 500_000_000 // max(sample_count, 1)))
        genotype_memmap = np.lib.format.open_memmap(
            geno_tmp,
            mode="w+",
            dtype=np.int8,
            shape=(sample_count, variant_count),
            fortran_order=True,
        )
        for start_index in range(0, variant_count, copy_batch_size):
            stop_index = min(start_index + copy_batch_size, variant_count)
            genotype_memmap[:, start_index:stop_index] = genotype_matrix_i8[:, start_index:stop_index]
        genotype_memmap.flush()
        del genotype_memmap
        with open(str(var_tmp), "wb") as f:
            pickle.dump(variants, f, protocol=pickle.HIGHEST_PROTOCOL)
        _write_vcf_cache_stats(stats_tmp, variant_stats)
        _atomic_write_text(
            manifest_tmp,
            json.dumps(
                {
                    "manifest_version": _VCF_CACHE_MANIFEST_VERSION,
                    "sample_count": int(sample_count),
                    "variant_count": int(variant_count),
                    "dtype": "int8",
                    "fortran_order": True,
                    "stats_file": paths.stats_path.name,
                }
            ),
        )
        geno_tmp.replace(paths.geno_path)
        var_tmp.replace(paths.var_path)
        stats_tmp.replace(paths.stats_path)
        manifest_tmp.replace(paths.manifest_path)
        try:
            paths.legacy_stats_path.unlink(missing_ok=True)
        except Exception:
            pass
        bundle_dir.rmdir()
        total_mb = (
            paths.geno_path.stat().st_size
            + paths.var_path.stat().st_size
            + paths.stats_path.stat().st_size
            + paths.manifest_path.stat().st_size
        ) / 1e6
        log(f"VCF cache saved ({total_mb:.1f} MB) → {paths.cache_dir.name}/{paths.key}.*")
    except Exception as exc:
        log(f"VCF cache save failed ({exc}), continuing without cache")
        _cleanup_stale_vcf_cache_temps(paths.cache_dir, paths.key)


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
        table_spec = _inspect_delimited_table(variant_metadata_path)
        if not table_spec.columns:
            raise ValueError("Variant metadata file is empty: " + str(variant_metadata_path))
        _require_columns(available_columns=table_spec.columns, required_columns=("variant_id",), context="variant metadata")
        saw_rows = False
        for row in _iter_delimited_rows(table_spec):
            saw_rows = True
            variant_id = str(row["variant_id"]).strip()
            if not variant_id:
                raise ValueError("Encountered blank variant_id in variant metadata.")
            if variant_id in metadata_rows_by_id:
                raise ValueError("Duplicate variant_id in variant metadata: " + variant_id)
            metadata_rows_by_id[variant_id] = row
        if not saw_rows:
            raise ValueError("Variant metadata file is empty: " + str(variant_metadata_path))

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
    prior_continuous_features = {
        column_name.removeprefix("prior_continuous__"): _parse_float_or_default(
            column_value,
            0.0,
            column_name=column_name,
        )
        for column_name, column_value in metadata_row.items()
        if column_name.startswith("prior_continuous__")
    }
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
        prior_continuous_features=prior_continuous_features,
        prior_class_members=prior_class_members,
        prior_class_membership=prior_class_membership,
    )


def _write_predictions_and_summary(
    predictions_path: Path,
    dataset: LoadedDataset,
    model: BayesianPGS,
) -> dict[str, Any]:
    log(f"computing predictions for {len(dataset.sample_ids)} samples, trait={model.config.trait_type.value}  mem={mem()}")
    genetic_score, covariate_score = model.decision_components(dataset.genotypes, dataset.covariates)
    linear_predictor = np.asarray(genetic_score + covariate_score, dtype=np.float32)
    if model.config.trait_type == TraitType.BINARY:
        probabilities = np.asarray(model.predict_proba(dataset.genotypes, dataset.covariates)[:, 1], dtype=np.float32)
        predicted_labels = (probabilities >= 0.5).astype(np.int32)
        log(f"binary predictions: mean_prob={float(np.mean(probabilities)):.4f}  pred_positive={int(np.sum(predicted_labels))}  pred_negative={int(np.sum(1-predicted_labels))}")
        _write_delimited_rows(
            predictions_path,
            header=("sample_id", "target", "genetic_score", "covariate_score", "linear_predictor", "probability", "predicted_label"),
            rows=(
                (
                    sample_id,
                    _format_float(float(target)),
                    _format_float(float(genetic_component)),
                    _format_float(float(covariate_component)),
                    _format_float(float(raw_score)),
                    _format_float(float(probability)),
                    str(int(predicted_label)),
                )
                for sample_id, target, genetic_component, covariate_component, raw_score, probability, predicted_label in zip(
                    dataset.sample_ids,
                    dataset.targets,
                    genetic_score,
                    covariate_score,
                    linear_predictor,
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

    predictions = np.asarray(model.predict(dataset.genotypes, dataset.covariates), dtype=np.float32)
    _write_delimited_rows(
        predictions_path,
        header=("sample_id", "target", "genetic_score", "covariate_score", "prediction"),
        rows=(
            (
                sample_id,
                _format_float(float(target)),
                _format_float(float(genetic_component)),
                _format_float(float(covariate_component)),
                _format_float(float(prediction)),
            )
            for sample_id, target, genetic_component, covariate_component, prediction in zip(
                dataset.sample_ids,
                dataset.targets,
                genetic_score,
                covariate_score,
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
        column_index_by_name={column_name: column_index for column_index, column_name in enumerate(columns)},
    )


def _iter_delimited_rows(table_spec: _DelimitedTableSpec) -> Iterator[dict[str, str]]:
    columns = table_spec.columns
    for row_values in _iter_delimited_row_values(table_spec):
        yield {
            column_name: row_values[column_index]
            for column_index, column_name in enumerate(columns)
        }


def _iter_delimited_row_values(table_spec: _DelimitedTableSpec) -> Iterator[list[str]]:
    with table_spec.path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter=table_spec.delimiter)
        next(reader, None)
        expected_width = len(table_spec.columns)
        for row in reader:
            normalized_row = ["" if value is None else str(value) for value in row[:expected_width]]
            if len(normalized_row) < expected_width:
                normalized_row.extend([""] * (expected_width - len(normalized_row)))
            yield normalized_row


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


def _vcf_variant_key(record: Any) -> str:
    return str(record.CHROM) + ":" + str(record.POS) + ":" + str(record.REF) + ":" + str(record.ALT[0])


def _variant_defaults_from_vcf_record(record: Any) -> _VariantDefaults:
    alt = str(record.ALT[0])
    info = record.INFO
    record_id = record.ID
    variant_id = _vcf_variant_key(record) if record_id is None or str(record_id) == "." else str(record_id)
    svlen_value = info.get("SVLEN")
    if svlen_value is not None:
        if isinstance(svlen_value, (tuple, list)):
            length = float(abs(float(svlen_value[0])))
        else:
            length = float(abs(float(svlen_value)))
    elif record.is_snp:
        length = 1.0
    elif record.end is not None and int(record.end) >= int(record.POS):
        length = float(int(record.end) - int(record.POS) + 1)
    else:
        length = float(max(len(record.REF), len(alt)))

    if record.is_snp:
        variant_class = VariantClass.SNV
    elif record.is_indel and not record.is_sv:
        variant_class = VariantClass.SMALL_INDEL
    else:
        variant_token = _normalize_variant_token(info.get("SVTYPE"))
        if variant_token is None:
            variant_token = _normalize_variant_token(alt)
        if variant_token is None:
            variant_class = VariantClass.OTHER_COMPLEX_SV
        else:
            variant_class = _structural_variant_class_from_token(variant_token, length)

    af_value = info.get("AF")
    if isinstance(af_value, (tuple, list)):
        allele_frequency = float(af_value[0])
    elif af_value is not None:
        allele_frequency = float(af_value)
    else:
        allele_frequency = -1.0

    return _VariantDefaults(
        variant_id=variant_id,
        variant_class=variant_class,
        chromosome=str(record.CHROM),
        position=int(record.POS),
        length=length,
        allele_frequency=allele_frequency,
        quality=_normalize_quality(record.QUAL),
    )


def _infer_plink_variant_class(allele_1: str, allele_2: str) -> VariantClass:
    structural_token = _symbolic_variant_token(allele_1, allele_2)
    if structural_token is not None:
        return _structural_variant_class_from_token(structural_token, length=1.0)
    if len(allele_1) == 1 and len(allele_2) == 1:
        return VariantClass.SNV
    return VariantClass.SMALL_INDEL




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


def _coerce_float(value: object) -> float:
    return float(str(value))


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

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Literal, Sequence, cast

import numpy as np

from sv_pgs._typing import F32Array, F64Array, I8Array, I32Array, NDArray
from sv_pgs.config import ModelConfig, VariantClass
from sv_pgs.data import NESTED_PATH_DELIMITER, VariantRecord, VariantStatistics
from sv_pgs.plink import PLINK_MISSING_INT8
from sv_pgs.genotype import (
    ConcatenatedRawGenotypeMatrix,
    IndexedRawGenotypeMatrix,
    PlinkRawGenotypeMatrix,
    RawGenotypeBatch,
    RawGenotypeMatrix,
    RowSubsetRawGenotypeMatrix,
    _has_sufficient_free_space_for_int8_npy,
    _int8_npy_header_bytes,
    as_raw_genotype_matrix,
    auto_batch_size_i8,
)
from sv_pgs.preprocessing import (
    _allele_frequencies_from_means,
    _batch_all_stats_i8,
    _means_and_scales_with_floor,
    compute_variant_statistics,
)
import jax.numpy as jnp
from sv_pgs.progress import log, mem

SV_LENGTH_THRESHOLD = 1_000.0
DEFAULT_SAMPLE_ID_COLUMNS = ("sample_id", "research_id", "person_id")
VARIANT_METADATA_BASE_COLUMNS = frozenset(
    {
        "variant_id",
        "variant_class",
        "chromosome",
        "position",
        "length",
        "allele_frequency",
        "quality",
        "training_support",
        "is_repeat",
        "is_copy_number",
        "prior_class_members",
        "prior_class_membership",
    }
)


@dataclass(slots=True)
class LoadedDataset:
    sample_ids: list[str]
    genotypes: RawGenotypeMatrix
    covariates: F32Array
    targets: F32Array
    variant_records: list[VariantRecord]
    variant_stats: VariantStatistics | None = None
    variant_stats_minimum_scale: float | None = None



@dataclass(slots=True)
class _SampleTable:
    sample_ids: list[str]
    covariates: F32Array
    targets: F32Array


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
    gt_types: I8Array
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


def _open_text_file(path: Path, mode: Literal["rt", "wt"], *, newline: str | None = None) -> Any:
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8", newline=newline)
    return path.open(mode.replace("t", ""), encoding="utf-8", newline=newline)


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
    # Region strings are routinely passed into shell-out commands (bcftools -r)
    # and into cache directory globs. A chromosome name like ``../../etc`` or
    # one containing a NUL byte could escape the intended scope; reject up
    # front. The VCF spec allows letters, digits, underscore, dot, hyphen,
    # asterisk, plus (and colon as the chrom:pos separator we already split).
    if "\x00" in region or any(ch in region for ch in ("/", "\\", "\n", "\r", "\t", " ")):
        raise ValueError(f"Refusing VCF region with disallowed characters: {region!r}")
    if ":" not in region:
        if region == "" or region == "." or region == "..":
            raise ValueError(f"Invalid VCF region: {region!r}")
        return region, 1, None
    chrom, coordinates = region.split(":", 1)
    if chrom in ("", ".", "..") or not chrom:
        raise ValueError(f"Invalid VCF region chromosome: {region!r}")
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
    # VCF spec mandates at least 8 fixed columns (CHROM..INFO); plus FORMAT
    # (1) when any genotypes are present. Without this check, the loop below
    # would raise an obscure IndexError on a truncated/garbled line.
    min_fields = 9 if sample_names else 8
    if len(fields) < min_fields:
        raise ValueError(
            f"Malformed VCF record: expected at least {min_fields} tab-separated "
            f"fields, got {len(fields)}: {line[:200]!r}"
        )
    chrom = fields[0]
    try:
        pos = int(fields[1])
    except ValueError as exc:
        raise ValueError(f"Malformed VCF POS field {fields[1]!r}: {exc}") from exc
    record_id = None if fields[2] == "." else fields[2]
    ref = fields[3]
    alt_field = fields[4]
    alt = () if alt_field == "." else tuple(alt_field.split(","))
    if fields[5] == ".":
        qual = None
    else:
        try:
            qual = float(fields[5])
        except ValueError as exc:
            raise ValueError(f"Malformed VCF QUAL field {fields[5]!r}: {exc}") from exc
    info = _parse_vcf_info(fields[7])
    gt_types = (
        _parse_vcf_gt_types(fields[8], fields[9:9 + len(sample_names)])
        if sample_names
        else np.empty(0, dtype=np.int8)
    )
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


def _parse_vcf_gt_types(format_field: str, sample_fields: Sequence[str]) -> I8Array:
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
        try:
            value = getattr(reader, attribute_name, None)
        except ValueError:
            continue
        if callable(value):
            try:
                value = value()
            except (TypeError, ValueError):
                continue
        if value is None:
            continue
        if not isinstance(value, (int, np.integer, float, np.floating, str)):
            continue
        try:
            resolved_value = int(value)
        except (TypeError, ValueError):
            continue
        if resolved_value >= 0:
            return resolved_value
    return None


def load_dataset_from_files(
    genotype_path: str | Path,
    config: ModelConfig,
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

        # Load VCF genotypes (full matrix from cache), then subset to matched samples
        keep_indices = np.array(aligned_sample_indices, dtype=np.intp)
        log(f"loading VCF genotypes (will subset to {len(keep_indices)} of {len(source_sample_ids)} samples)...")

        # Try disk cache first to skip VCF re-parsing on repeated runs
        genotype_matrix, default_variants, variant_stats = _load_vcf_with_cache(
            source_path,
            config=config,
            mmap_mode="r",
        )
        # Lazy row subset: wrap the mmap'd matrix instead of writing a temp
        # mmap with the row-permuted copy. See RowSubsetRawGenotypeMatrix.
        vcf_raw_genotypes: RawGenotypeMatrix | None
        vcf_raw_genotypes = _lazy_row_subset(genotype_matrix, keep_indices)

        log(f"VCF loaded: {vcf_raw_genotypes.shape[0]} samples x {len(default_variants)} variants  mem={mem()}")
        plink_metadata = None
    elif resolved_format == "plink1":
        log("reading PLINK .fam/.bim metadata (no genotype data yet)...")
        plink_metadata = _load_plink1_metadata(source_path)
        source_sample_ids = plink_metadata.sample_ids
        log(f"PLINK metadata: {len(source_sample_ids)} samples x {plink_metadata.variant_count} variants")
        bed_size = source_path.stat().st_size / 1e9
        full_matrix_gb = len(source_sample_ids) * plink_metadata.variant_count * 4 / 1e9
        log(f"  .bed file size: {bed_size:.2f} GB  |  full float32 matrix would be: {full_matrix_gb:.1f} GB")
        vcf_raw_genotypes = None
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
        # vcf_raw_genotypes is already subsetted to aligned samples (int8 for VCF)
        if vcf_raw_genotypes is None:
            raise RuntimeError("VCF genotype matrix was not initialized.")
        raw_genotypes = as_raw_genotype_matrix(vcf_raw_genotypes)
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
        log("computing variant statistics (single pass, JAX; disk-cached)...")
        variant_stats, plink_int8_cache_path = compute_plink_variant_statistics_cached(
            raw_genotypes,
            bed_path=source_path,
            sample_indices=aligned_sample_indices,
            config=config,
        )
        if plink_int8_cache_path is not None:
            int8_view = _open_plink_int8_cache_for_read(plink_int8_cache_path)
            if int8_view is not None:
                log(
                    f"  swapping PLINK source to int8 mmap "
                    f"({int8_view.shape[0]:,} × {int8_view.shape[1]:,}) — "
                    "downstream passes skip bed-decode entirely"
                )
                raw_genotypes = as_raw_genotype_matrix(int8_view)
        log("building PLINK variant defaults from pre-computed allele frequencies...")
        default_variants = _build_plink_variant_defaults_from_stats(source_path, variant_stats)
        log(f"built {len(default_variants)} PLINK variant defaults  mem={mem()}")

    log("building variant records from defaults + optional metadata...")
    variant_records = _build_variant_records(
        default_variants=default_variants,
        variant_metadata_path=variant_metadata_path,
    )
    snv_count, sv_count = _count_variant_record_classes(variant_records)
    log(f"variant records: {len(variant_records)} total ({snv_count} SNVs, {sv_count} structural variants)")

    log(f"=== LOAD DATASET DONE === final shape={raw_genotypes.shape}  mem={mem()}")
    return LoadedDataset(
        sample_ids=list(sample_table.sample_ids),
        genotypes=raw_genotypes,
        covariates=np.asarray(sample_table.covariates, dtype=np.float32),
        targets=np.asarray(sample_table.targets, dtype=np.float32),
        variant_records=variant_records,
        variant_stats=variant_stats,
        variant_stats_minimum_scale=(float(config.minimum_scale) if variant_stats is not None else None),
    )


def _lazy_row_subset(
    matrix: NDArray | RawGenotypeMatrix,
    row_indices: NDArray,
) -> RawGenotypeMatrix:
    """Apply a row subset lazily via RowSubsetRawGenotypeMatrix.

    Identity-row case is a pure passthrough (no wrapper). Otherwise wraps
    the source matrix and defers the row-permutation to per-batch fetches,
    eliminating the per-chromosome 11 GB temp-mmap write that dominated
    AoU-scale load time.
    """
    resolved_row_indices = np.asarray(row_indices, dtype=np.intp)
    source: RawGenotypeMatrix = as_raw_genotype_matrix(matrix)
    if resolved_row_indices.shape[0] == int(source.shape[0]) and np.array_equal(
        resolved_row_indices,
        np.arange(int(source.shape[0]), dtype=np.intp),
    ):
        return source
    return RowSubsetRawGenotypeMatrix(child=source, row_indices=resolved_row_indices)


def load_multi_vcf_dataset_from_files(
    genotype_paths: Sequence[str | Path],
    config: ModelConfig,
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

    # _build_sample_table emits rows in VCF order (available_sample_ids),
    # so alignment should produce identity indices → skip_subset=True.
    log("aligning sample IDs between sample table and genotype source...")
    aligned_sample_indices = _align_sample_ids(
        expected_sample_ids=sample_table.sample_ids,
        available_sample_ids=source_sample_ids,
        context="genotype source",
    )
    keep_sample_indices = np.asarray(aligned_sample_indices, dtype=np.intp)
    log(f"aligned {len(aligned_sample_indices)} phenotype rows against {len(source_sample_ids)} genotype samples")

    n_chromosomes = len(source_paths)
    expected_source_order = np.arange(len(source_sample_ids), dtype=np.intp)
    skip_subset = (
        len(keep_sample_indices) == len(source_sample_ids)
        and np.array_equal(keep_sample_indices, expected_source_order)
    )
    # Verify sample IDs match for every chromosome. Different VCFs in the
    # same dataset can list the same cohort in a different order; trusting
    # source_paths[0] would silently misalign genotypes against phenotypes.
    log(f"  verifying sample IDs for all {n_chromosomes} chromosomes against {source_paths[0].name}...")
    per_source_align_indices: list[NDArray | None] = [None] * n_chromosomes
    for chr_idx, source_path in enumerate(source_paths):
        chr_sample_ids = _read_vcf_sample_ids(source_path)
        if chr_sample_ids == source_sample_ids:
            continue
        # Different order (or different membership) — derive an explicit
        # per-source alignment from the phenotype sample table.
        if sorted(chr_sample_ids) != sorted(source_sample_ids):
            raise RuntimeError(
                f"VCF sample IDs do not match the reference VCF: {source_path} "
                f"(reference: {source_paths[0]})"
            )
        per_source_align_indices[chr_idx] = np.asarray(
            _align_sample_ids(
                expected_sample_ids=sample_table.sample_ids,
                available_sample_ids=chr_sample_ids,
                context=f"genotype source {source_path.name}",
            ),
            dtype=np.intp,
        )
        log(
            f"  {source_path.name}: sample order differs from reference VCF; "
            f"applying per-source alignment"
        )
    log(f"  sample IDs verified. skip_subset={skip_subset}. loading {n_chromosomes} chromosomes...")

    _t_start = time.monotonic()
    raw_matrices: list[RawGenotypeMatrix] = []
    per_chr_variants: list[list[_VariantDefaults]] = []
    per_chr_stats: list[VariantStatistics] = []
    total_variants = 0
    for chr_idx, source_path in enumerate(source_paths):
        _t_chr = time.monotonic()
        genotype_matrix, chromosome_variants, chromosome_stats = _load_vcf_with_cache(
            source_path, config=config, mmap_mode="r",
        )
        chromosome_raw: RawGenotypeMatrix = as_raw_genotype_matrix(genotype_matrix)
        source_specific_indices = per_source_align_indices[chr_idx]
        if source_specific_indices is not None:
            chromosome_raw = _lazy_row_subset(chromosome_raw, source_specific_indices)
        elif not skip_subset:
            chromosome_raw = _lazy_row_subset(chromosome_raw, keep_sample_indices)
        raw_matrices.append(chromosome_raw)
        per_chr_variants.append(list(chromosome_variants))
        per_chr_stats.append(chromosome_stats)
        total_variants += len(chromosome_variants)
        _elapsed = time.monotonic() - _t_start
        _chr_time = time.monotonic() - _t_chr
        log(
            f"  [{chr_idx+1}/{n_chromosomes}] {source_path.name}: "
            f"{len(chromosome_variants):,} variants in {_chr_time:.1f}s  "
            f"total={total_variants:,}  {_elapsed:.0f}s elapsed  mem={mem()}"
        )
    _elapsed = time.monotonic() - _t_start
    log(f"  all {n_chromosomes} chromosomes loaded: {total_variants:,} total variants in {_elapsed:.0f}s")

    raw_matrices, per_chr_variants, per_chr_stats = _dedupe_cross_source_variants(
        raw_matrices=raw_matrices,
        per_chr_variants=per_chr_variants,
        per_chr_stats=per_chr_stats,
        source_paths=source_paths,
    )

    default_variants: list[_VariantDefaults] = [variant for chrom in per_chr_variants for variant in chrom]
    raw_genotypes: RawGenotypeMatrix = ConcatenatedRawGenotypeMatrix(tuple(raw_matrices))
    variant_stats = VariantStatistics(
        means=np.concatenate([stats.means for stats in per_chr_stats]).astype(np.float32, copy=False),
        scales=np.concatenate([stats.scales for stats in per_chr_stats]).astype(np.float32, copy=False),
        allele_frequencies=np.concatenate([stats.allele_frequencies for stats in per_chr_stats]).astype(np.float32, copy=False),
        support_counts=np.concatenate([stats.support_counts for stats in per_chr_stats]).astype(np.int32, copy=False),
    )

    log("building variant records from defaults + optional metadata...")
    variant_records = _build_variant_records(
        default_variants=default_variants,
        variant_metadata_path=variant_metadata_path,
    )
    snv_count, sv_count = _count_variant_record_classes(variant_records)
    log(f"variant records: {len(variant_records)} total ({snv_count} SNVs, {sv_count} structural variants)")

    log(f"=== LOAD MULTI-VCF DATASET DONE === final shape={raw_genotypes.shape}  mem={mem()}")
    return LoadedDataset(
        sample_ids=list(sample_table.sample_ids),
        genotypes=raw_genotypes,
        covariates=np.asarray(sample_table.covariates, dtype=np.float32),
        targets=np.asarray(sample_table.targets, dtype=np.float32),
        variant_records=variant_records,
        variant_stats=variant_stats,
        variant_stats_minimum_scale=float(config.minimum_scale),
    )


def load_multi_vcf_dataset_as_bitpacked(
    vcf_paths: Sequence[str | Path],
    cache_dir: str | Path,
    sample_ids: Sequence[str] | None = None,
    *,
    sample_id_order: Sequence[str] | None = None,
    count_a1: bool = True,
    stream: Any | None = None,
) -> Any:
    """Transcode SV VCFs to a single PLINK BED trio and return a bitpacked GPU matrix.

    This is the bitpacked replacement for the historical int8 ``.npy`` cache
    path. It one-time-transcodes the input VCFs to PLINK 1.9 BED bytes via
    :func:`sv_pgs.sv_transcoder.transcode_sv_vcf_to_bed` (cached under
    ``cache_dir`` as ``<key>.bed`` / ``.bim`` / ``.fam``) and then uploads the
    bitpacked bytes to the GPU via
    :func:`sv_pgs.bitpacked_loader.load_bed_to_bitpacked_device`.

    Resume semantics: when the ``.bed`` / ``.bim`` / ``.fam`` trio already
    exists under ``cache_dir`` we skip transcoding and only do the upload.
    The legacy ``load_multi_vcf_dataset_from_files`` entry point is unchanged
    and remains available for back-compat.
    """
    from sv_pgs.sv_transcoder import transcode_sv_vcf_to_bed  # lazy
    from sv_pgs.bitpacked_loader import load_bed_to_bitpacked_device  # lazy

    paths = [Path(p) for p in vcf_paths]
    if not paths:
        raise ValueError("vcf_paths cannot be empty.")
    cache_dir_path = Path(cache_dir)
    cache_dir_path.mkdir(parents=True, exist_ok=True)

    # Stable per-cohort key: hash of resolved vcf paths + sample restriction.
    fp = hashlib.sha256()
    for path in paths:
        fp.update(str(path.resolve()).encode("utf-8"))
        fp.update(b"\0")
    if sample_ids is not None:
        for sid in sample_ids:
            fp.update(str(sid).encode("utf-8"))
            fp.update(b"\0")
    fp.update(b"|order|")
    if sample_id_order is not None:
        for sid in sample_id_order:
            fp.update(str(sid).encode("utf-8"))
            fp.update(b"\0")
    key = fp.hexdigest()[:16]
    bed_path = cache_dir_path / f"{key}.bed"
    bim_path = bed_path.with_suffix(".bim")
    fam_path = bed_path.with_suffix(".fam")

    if bed_path.exists() and bim_path.exists() and fam_path.exists():
        log(f"  SV BED cache hit: reusing {bed_path.name}")
        n_samples = sum(1 for _ in fam_path.open("r", encoding="utf-8"))
        n_variants = sum(1 for _ in bim_path.open("r", encoding="utf-8"))
    else:
        log(f"  SV BED cache miss: transcoding {len(paths)} VCF(s) -> {bed_path.name}")
        meta = transcode_sv_vcf_to_bed(
            paths,
            bed_path,
            sample_ids=list(sample_ids) if sample_ids is not None else None,
            sample_id_order=list(sample_id_order) if sample_id_order is not None else None,
        )
        n_samples = int(meta["n_samples"])
        n_variants = int(meta["n_variants"])

    return load_bed_to_bitpacked_device(
        bed_path,
        n_samples=n_samples,
        n_variants=n_variants,
        count_a1=count_a1,
        stream=stream,
    )


def _dedupe_cross_source_variants(
    *,
    raw_matrices: list[RawGenotypeMatrix],
    per_chr_variants: list[list[_VariantDefaults]],
    per_chr_stats: list[VariantStatistics],
    source_paths: Sequence[Path],
) -> tuple[list[RawGenotypeMatrix], list[list[_VariantDefaults]], list[VariantStatistics]]:
    """Drop variants whose (chr, pos, variant_id) already appeared in an earlier source.

    Past worker bugs occasionally produced cache bundles where a few records
    from one chromosome ended up duplicated inside another chromosome's cache
    (the per-cache audit cannot detect this because each cache passes its own
    chromosome and variant-id checks). Rather than invalidating those caches
    and forcing a multi-minute re-parse, we reconcile here: the first source
    to claim a key wins, and later sources expose only their remaining columns
    via IndexedRawGenotypeMatrix wrappers — zero-copy, no on-disk rewrite.
    """
    seen_keys: set[tuple[str, int, str]] = set()
    preview_duplicates: list[tuple[str, int, str, str]] = []
    dropped_per_source: list[int] = []
    new_matrices: list[RawGenotypeMatrix] = []
    new_variants: list[list[_VariantDefaults]] = []
    new_stats: list[VariantStatistics] = []

    for chr_idx, chromosome_variants in enumerate(per_chr_variants):
        keep_positions: list[int] = []
        for local_idx, variant in enumerate(chromosome_variants):
            variant_key = (str(variant.chromosome), int(variant.position), str(variant.variant_id))
            if variant_key in seen_keys:
                if len(preview_duplicates) < 3:
                    preview_duplicates.append((*variant_key, source_paths[chr_idx].name))
                continue
            seen_keys.add(variant_key)
            keep_positions.append(local_idx)
        dropped = len(chromosome_variants) - len(keep_positions)
        dropped_per_source.append(dropped)
        child = raw_matrices[chr_idx]
        stats = per_chr_stats[chr_idx]
        if dropped == 0:
            new_matrices.append(child)
            new_variants.append(chromosome_variants)
            new_stats.append(stats)
            continue
        keep_index = np.asarray(keep_positions, dtype=np.int64)
        new_matrices.append(IndexedRawGenotypeMatrix(child=child, selected_columns=keep_index))
        new_variants.append([chromosome_variants[i] for i in keep_positions])
        # When the caller supplies precomputed_variant_stats, per-source stats
        # are placeholder empties; preserve them as-is (the final dataset uses
        # the precomputed values, not the per-source concat).
        if stats.means.shape[0] == 0:
            new_stats.append(stats)
        else:
            new_stats.append(VariantStatistics(
                means=np.ascontiguousarray(stats.means[keep_index]),
                scales=np.ascontiguousarray(stats.scales[keep_index]),
                allele_frequencies=np.ascontiguousarray(stats.allele_frequencies[keep_index]),
                support_counts=np.ascontiguousarray(stats.support_counts[keep_index]),
            ))

    total_dropped = sum(dropped_per_source)
    if total_dropped > 0:
        per_source_summary = ", ".join(
            f"{source_paths[i].name}:-{dropped_per_source[i]}"
            for i in range(len(source_paths))
            if dropped_per_source[i] > 0
        )
        preview = ", ".join(
            f"{chrom}:{position}:{variant_id} (in {source_name})"
            for chrom, position, variant_id, source_name in preview_duplicates
        )
        qualifier = "cross-source " if len(source_paths) > 1 else ""
        log(
            f"  dropped {total_dropped} {qualifier}duplicate variants "
            f"(first occurrence kept). per-source: {per_source_summary}. "
            f"examples: {preview}"
        )
    return new_matrices, new_variants, new_stats


def load_multi_source_dataset_from_files(
    *,
    sources: Sequence[tuple[str, str | Path]],
    config: ModelConfig,
    sample_table_path: str | Path,
    target_column: str,
    covariate_columns: Sequence[str],
    sample_id_column: str = "auto",
    variant_metadata_path: str | Path | None = None,
    precomputed_variant_stats: VariantStatistics | None = None,
    precomputed_variant_records: Sequence[VariantRecord] | None = None,
) -> LoadedDataset:
    """Load a heterogeneous list of (kind, path) genotype sources.

    `sources` is a list of `(kind, path)` tuples where `kind` is `"vcf"` or
    `"plink1"`. Each source contributes its own variants and is exposed as a
    child of the resulting ConcatenatedRawGenotypeMatrix. The unified sample
    set is the intersection of sample IDs across sources (in the first
    source's order), so it's safe to mix the AoU SV cohort (97k) with the
    microarray cohort (447k) — the result is the 97k-sample intersection.

    Cross-source variant duplicates are dropped by the same first-occurrence
    rule as load_multi_vcf_dataset_from_files.

    If `precomputed_variant_stats` is supplied, the PLINK source skips its
    own ~10-110 min variant-stats streaming pass and uses the supplied
    values directly. The AoU runner uses this to load the held-out test
    cohort: stats from the test cohort would be computed-and-immediately-
    overwritten by the train stats anyway (the model expects a single
    consistent standardization between train and test), so skipping the
    test pass saves an entire bed-file scan on every run.
    """
    log(f"=== LOAD MULTI-SOURCE DATASET START === sources={len(sources)} mem={mem()}")
    if not sources:
        raise ValueError("sources cannot be empty.")
    source_specs: list[tuple[str, Path]] = [(str(kind), Path(path)) for kind, path in sources]
    for kind, _ in source_specs:
        if kind not in ("vcf", "plink1"):
            raise ValueError(f"Unsupported source kind: {kind!r}; expected 'vcf' or 'plink1'.")
    if len({path.resolve() for _, path in source_specs}) != len(source_specs):
        raise ValueError("source paths must be unique.")

    # Phase 1: read each source's native sample list. PLINK uses .fam,
    # VCF uses the header — both are cheap (no genotype data touched).
    per_source_sample_ids: list[list[str]] = []
    for kind, path in source_specs:
        log(f"  reading sample IDs from {kind} source: {path.name}")
        if kind == "vcf":
            ids = _read_vcf_sample_ids(path)
        else:
            ids = list(_load_plink1_metadata(path).sample_ids)
        log(f"    {len(ids):,} samples")
        per_source_sample_ids.append(ids)

    # Phase 2: intersect to samples present in every source. Preserve the
    # first source's order so subsequent runs see a stable sample ordering
    # (downstream caching keys off that order). De-dupe defensively — sample
    # IDs *should* be unique in every source, but a malformed VCF header or
    # corrupt .fam could leak a duplicate which would otherwise propagate
    # silently into the sample table.
    common_set = set(per_source_sample_ids[0])
    for ids in per_source_sample_ids[1:]:
        common_set &= set(ids)
    seen_common: set[str] = set()
    common_ordered: list[str] = []
    for sid in per_source_sample_ids[0]:
        if sid in common_set and sid not in seen_common:
            seen_common.add(sid)
            common_ordered.append(sid)
    log(f"  sample intersection across {len(source_specs)} sources: {len(common_ordered):,} samples")
    if not common_ordered:
        raise RuntimeError("No samples in common across genotype sources.")

    # Phase 3: build the filtered sample table against the intersection.
    log(f"reading sample table header: {sample_table_path}")
    sample_table_spec = _inspect_delimited_table(sample_table_path)
    resolved_sample_id_column = _resolve_sample_id_column(
        table_spec=sample_table_spec,
        requested_sample_id_column=sample_id_column,
        available_sample_ids=common_ordered,
    )
    log(f"sample ID column: '{resolved_sample_id_column}'")
    # Try the on-disk filtered-sample-table cache before the (slow, ~20-30 s)
    # rebuild. Cache key incorporates the TSV + first PLINK .fam fingerprints,
    # the resolved sample-id / target / covariate columns, and the intersected
    # sample ordering — i.e. every input that `_build_sample_table` consumes.
    _covariate_columns_tuple = tuple(str(c) for c in covariate_columns)
    _fam_path_for_cache: Path | None = None
    for _kind, _src_path in source_specs:
        if _kind == "plink1":
            _candidate_fam = _src_path.with_suffix(".fam")
            if _candidate_fam.exists():
                _fam_path_for_cache = _candidate_fam
                break
    _st_cache_root = _sample_table_cache_root(Path(sample_table_path))
    _st_cache_key = _sample_table_cache_key(
        sample_table_path=Path(sample_table_path),
        fam_path=_fam_path_for_cache,
        sample_id_column=resolved_sample_id_column,
        target_column=target_column,
        covariate_columns=_covariate_columns_tuple,
        intersected_samples=common_ordered,
    )
    _st_cache_path = _sample_table_cache_path(_st_cache_root, _st_cache_key)
    _cache_hit = (
        _read_sample_table_cache(_st_cache_path, expected_n_samples=len(common_ordered))
        if _st_cache_path.exists()
        else None
    )
    if _cache_hit is not None:
        sample_table, total_sample_rows, unmatched_sample_rows = _cache_hit
        log(
            f"sample table cache hit: {_st_cache_path} "
            f"({len(sample_table.sample_ids):,} rows, "
            f"{sample_table.covariates.shape[1]} cols)"
        )
    else:
        log("sample table cache miss; rebuilding and persisting...")
        log("building filtered sample table on intersected samples...")
        sample_table, total_sample_rows, unmatched_sample_rows = _build_sample_table(
            table_spec=sample_table_spec,
            sample_id_column=resolved_sample_id_column,
            target_column=target_column,
            covariate_columns=covariate_columns,
            available_sample_ids=common_ordered,
        )
        try:
            _write_sample_table_cache(
                _st_cache_path,
                sample_table=sample_table,
                total_sample_rows=total_sample_rows,
                unmatched_sample_rows=unmatched_sample_rows,
            )
        except OSError as exc:
            log(f"  warning: failed to persist sample-table cache to {_st_cache_path}: {exc!r}")
    n_cases = int(np.sum(np.asarray(sample_table.targets) == 1.0))
    n_controls = int(np.sum(np.asarray(sample_table.targets) == 0.0))
    log(
        "sample table: "
        + f"{len(sample_table.sample_ids):,} matched rows kept from {total_sample_rows:,}, "
        + f"{unmatched_sample_rows} dropped, {sample_table.covariates.shape[1]} covariates"
    )
    log(f"  target distribution: {n_cases:,} cases, {n_controls:,} controls")

    # Phase 4: per-source alignment — each source needs indices into its
    # OWN native sample order that produce the unified sample table order.
    per_source_keep_indices: list[NDArray] = []
    for (kind, path), source_ids in zip(source_specs, per_source_sample_ids):
        aligned = _align_sample_ids(
            expected_sample_ids=sample_table.sample_ids,
            available_sample_ids=source_ids,
            context=f"{kind} source {path.name}",
        )
        per_source_keep_indices.append(np.asarray(aligned, dtype=np.intp))

    # Phase 5: build a RawGenotypeMatrix + variants + stats for each source.
    # VCF sources go through the bcftools cache; PLINK reads lazily via
    # PlinkRawGenotypeMatrix and computes stats in one streaming pass.
    raw_matrices: list[RawGenotypeMatrix] = []
    per_source_variants: list[list[_VariantDefaults]] = []
    per_source_stats: list[VariantStatistics] = []
    total_variants = 0
    _t_start = time.monotonic()
    for src_idx, ((kind, path), source_ids, keep_indices) in enumerate(
        zip(source_specs, per_source_sample_ids, per_source_keep_indices)
    ):
        _t_src = time.monotonic()
        if kind == "vcf":
            genotype_matrix, variants, stats = _load_vcf_with_cache(path, config=config, mmap_mode="r")
            n_native = len(source_ids)
            skip_subset = (
                len(keep_indices) == n_native
                and np.array_equal(keep_indices, np.arange(n_native, dtype=np.intp))
            )
            raw: RawGenotypeMatrix = as_raw_genotype_matrix(genotype_matrix)
            if not skip_subset:
                raw = _lazy_row_subset(raw, keep_indices)
            variants_list = list(variants)
        else:
            meta = _load_plink1_metadata(path)
            raw = PlinkRawGenotypeMatrix(
                bed_path=path,
                sample_indices=keep_indices,
                variant_count=meta.variant_count,
                total_sample_count=len(meta.sample_ids),
            )
            if precomputed_variant_stats is not None:
                # Caller already has variant_stats and variant_records from a
                # prior load on the same source set (typically: train cohort
                # supplying values for the test load). Reuse an existing int8
                # cache if one is already available, but never build a fresh
                # full-genome cache here: on AoU that is a 144 GB validation
                # side quest before fitting starts, while the solver now works
                # on active LD/variant blocks.
                int8_path_existing = _plink_int8_cache_path(path, keep_indices, config)
                int8_view: I8Array | None = None
                if int8_path_existing.exists():
                    int8_view = _open_plink_int8_cache_for_read(int8_path_existing)
                    if int8_view is not None:
                        log(
                            f"  reusing precomputed variant statistics for {path.name} "
                            f"(skipping PLINK streaming pass); int8 cache present"
                        )
                    else:
                        log(
                            f"  PLINK int8 cache at {int8_path_existing.name} present but "
                            f"unreadable; removing it and continuing without eager rebuild"
                        )
                        try:
                            int8_path_existing.unlink()
                        except OSError as exc:
                            log(f"  could not remove stale int8 cache ({exc!r}); ignoring it")
                if int8_view is None:
                    log(
                        f"  reusing precomputed variant statistics for {path.name} "
                        "without building a full validation int8 cache; "
                        "active block kernels will stream from PLINK as needed"
                    )
                if int8_view is not None:
                    log(
                        f"  swapping PLINK source to int8 mmap "
                        f"({int8_view.shape[0]:,} × {int8_view.shape[1]:,}) — "
                        "downstream passes skip bed-decode entirely"
                    )
                    raw = as_raw_genotype_matrix(int8_view)
                stats = VariantStatistics(
                    means=np.empty(0, dtype=np.float32),
                    scales=np.empty(0, dtype=np.float32),
                    allele_frequencies=np.empty(0, dtype=np.float32),
                    support_counts=np.empty(0, dtype=np.int32),
                )
                variants_list = [
                    _VariantDefaults(
                        variant_id=bim_record.variant_id,
                        variant_class=_infer_plink_variant_class(bim_record.allele_1, bim_record.allele_2),
                        chromosome=bim_record.chromosome,
                        position=bim_record.position,
                        length=1.0,
                        allele_frequency=0.0,
                        quality=1.0,
                    )
                    for bim_record in _iter_plink_bim_records(path.with_suffix(".bim"))
                ]
            else:
                # Bitpacked-backend fast path: stream BED bytes straight to
                # the GPU via run_screening_pass (one-pass count/sum/sumsq
                # kernel) and skip the int8 .npy materialization entirely.
                # Always attempted; the legacy int8 streaming pass below
                # is the fallback when the bitpacked imports / GPU probe
                # fail or the helper otherwise returns ``None``. No env
                # vars, no config gating — fewer knobs to misconfigure.
                # Diagnostic logs are retained so a fallback always says
                # why.
                log("  variant stats: attempting bitpacked GPU streaming path")
                # ``n_samples`` must be the RAW on-disk BED iid_count
                # (``PlinkRawGenotypeMatrix.total_sample_count``), NOT the
                # post-intersect cohort size ``raw.shape[0]``. The screening
                # kernel uses it to compute byte offsets into the BED and
                # validates that ``sample_intersect`` indices fall inside
                # [0, n_samples). Passing the post-intersect count rejects
                # every index >= len(keep_indices).
                bitpacked_stats: VariantStatistics | None = _try_bitpacked_plink_variant_stats(
                    bed_path=path,
                    sample_indices=keep_indices,
                    n_samples=int(getattr(raw, "total_sample_count", raw.shape[0])),
                    n_variants=int(raw.shape[1]),
                    config=config,
                )
                if bitpacked_stats is None:
                    log("  variant stats: bitpacked path returned None (see prior log); falling back to legacy int8 streaming pass")
                if bitpacked_stats is not None:
                    stats = bitpacked_stats
                    variants_list = _build_plink_variant_defaults_from_stats(path, stats)
                else:
                    log(f"  computing variant statistics for {path.name} (single PLINK streaming pass; disk-cached)...")
                    stats, plink_int8_cache_path = compute_plink_variant_statistics_cached(
                        raw,
                        bed_path=path,
                        sample_indices=keep_indices,
                        config=config,
                    )
                    if plink_int8_cache_path is not None:
                        int8_view = _open_plink_int8_cache_for_read(plink_int8_cache_path)
                        if int8_view is not None:
                            log(
                                f"  swapping PLINK source to int8 mmap "
                                f"({int8_view.shape[0]:,} × {int8_view.shape[1]:,}) — "
                                "downstream passes skip bed-decode entirely"
                            )
                            raw = as_raw_genotype_matrix(int8_view)
                    variants_list = _build_plink_variant_defaults_from_stats(path, stats)
        raw_matrices.append(raw)
        per_source_variants.append(variants_list)
        per_source_stats.append(stats)
        total_variants += len(variants_list)
        _t = time.monotonic() - _t_src
        _elapsed = time.monotonic() - _t_start
        log(
            f"  [{src_idx+1}/{len(source_specs)}] {kind} {path.name}: "
            f"{len(variants_list):,} variants in {_t:.1f}s  "
            f"total={total_variants:,}  {_elapsed:.0f}s elapsed  mem={mem()}"
        )
    log(f"  all {len(source_specs)} sources loaded: {total_variants:,} total variants in {time.monotonic() - _t_start:.0f}s")

    # Phase 6: cross-source variant dedup (same first-occurrence rule used
    # by the multi-VCF path).
    raw_matrices, per_source_variants, per_source_stats = _dedupe_cross_source_variants(
        raw_matrices=raw_matrices,
        per_chr_variants=per_source_variants,
        per_chr_stats=per_source_stats,
        source_paths=[path for _, path in source_specs],
    )

    # Phase 7: concat + emit a LoadedDataset (same shape the rest of the
    # pipeline already consumes).
    default_variants: list[_VariantDefaults] = [variant for chunk in per_source_variants for variant in chunk]
    raw_genotypes: RawGenotypeMatrix = ConcatenatedRawGenotypeMatrix(tuple(raw_matrices))
    if precomputed_variant_stats is not None:
        expected_variants = int(raw_genotypes.shape[1])
        provided_variants = int(precomputed_variant_stats.means.shape[0])
        if expected_variants != provided_variants:
            raise ValueError(
                f"precomputed_variant_stats has {provided_variants} variants but the "
                f"loaded genotype matrix has {expected_variants} columns. The two loads "
                "must be over the same source set (same VCFs, same .bed) so dedup "
                "produces identical variant counts; otherwise stats and columns won't align."
            )
        variant_stats = precomputed_variant_stats
        log(
            f"reusing precomputed variant_stats ({variant_stats.means.shape[0]:,} variants) "
            "— skipping concatenation + variant-record rebuild"
        )
    else:
        variant_stats = VariantStatistics(
            means=np.concatenate([stats.means for stats in per_source_stats]).astype(np.float32, copy=False),
            scales=np.concatenate([stats.scales for stats in per_source_stats]).astype(np.float32, copy=False),
            allele_frequencies=np.concatenate([stats.allele_frequencies for stats in per_source_stats]).astype(np.float32, copy=False),
            support_counts=np.concatenate([stats.support_counts for stats in per_source_stats]).astype(np.int32, copy=False),
        )
    if precomputed_variant_records is not None:
        if isinstance(precomputed_variant_records, list):
            variant_records = precomputed_variant_records
        else:
            variant_records = list(precomputed_variant_records)
        log(f"reusing {len(variant_records):,} precomputed variant_records (skipping metadata join)")
    else:
        log("building variant records from defaults + optional metadata...")
        variant_records = _build_variant_records(
            default_variants=default_variants,
            variant_metadata_path=variant_metadata_path,
        )
    snv_count, sv_count = _count_variant_record_classes(variant_records)
    log(f"variant records: {len(variant_records):,} total ({snv_count:,} SNVs, {sv_count:,} structural variants)")

    log(f"=== LOAD MULTI-SOURCE DATASET DONE === final shape={raw_genotypes.shape}  mem={mem()}")
    return LoadedDataset(
        sample_ids=list(sample_table.sample_ids),
        genotypes=raw_genotypes,
        covariates=np.asarray(sample_table.covariates, dtype=np.float32),
        targets=np.asarray(sample_table.targets, dtype=np.float32),
        variant_records=variant_records,
        variant_stats=variant_stats,
        variant_stats_minimum_scale=float(config.minimum_scale),
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

    # Parse sample table into a dict keyed by sample_id so we can emit
    # rows in VCF order (available_sample_ids) rather than file order.
    # This guarantees the returned _SampleTable is already aligned with
    # the genotype matrices — no reindexing needed downstream.
    parsed: dict[str, tuple[float, list[float]]] = {}
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
        if sample_id in parsed:
            raise ValueError("Duplicate sample identifier in sample table: " + sample_id)
        try:
            target_value = float(row_values[target_index])
            covariate_values = [float(row_values[column_index]) for column_index in covariate_indices]
        except (ValueError, TypeError):
            unmatched_rows += 1
            continue
        if np.isnan(target_value) or any(np.isnan(v) for v in covariate_values):
            unmatched_rows += 1
            continue
        parsed[sample_id] = (target_value, covariate_values)

    # Emit rows in VCF sample order so genotype matrices need no reindexing.
    sample_ids: list[str] = []
    targets: list[float] = []
    covariates: list[list[float]] = []
    for vcf_sample_id in available_sample_ids:
        if vcf_sample_id in parsed:
            target_value, covariate_values = parsed[vcf_sample_id]
            sample_ids.append(vcf_sample_id)
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
_CACHE_VERSION = 3
_VCF_CACHE_MANIFEST_VERSION = 2
_VCF_CACHE_STATS_DTYPE = np.dtype(
    [
        ("means", "<f4"),
        ("scales", "<f4"),
        ("allele_frequencies", "<f4"),
        ("support_counts", "<i4"),
    ]
)
_VCF_CACHE_VARIANT_NUMERIC_DTYPE = np.dtype(
    [
        ("variant_class_code", "<u2"),
        ("position", "<i8"),
        ("length", "<f4"),
        ("allele_frequency", "<f4"),
        ("quality", "<f4"),
    ]
)
_VARIANT_CLASS_TO_CODE = {
    variant_class: code
    for code, variant_class in enumerate(VariantClass)
}
_VARIANT_CODE_TO_CLASS = {
    code: variant_class
    for variant_class, code in _VARIANT_CLASS_TO_CODE.items()
}

# ---------------------------------------------------------------------------
# Filtered sample table disk cache
# ---------------------------------------------------------------------------
#
# The filtered-sample-table build (`_build_sample_table` against an
# intersected sample-ID list) is fully deterministic given:
#   * the sample-table TSV (path + size + mtime_ns)
#   * the PLINK .fam (path + size + mtime_ns) — proxy for sample-source order
#   * the resolved sample_id_column
#   * the target_column
#   * the covariate_columns tuple
#   * the intersected-samples bytes (the unified sample order)
#
# but it costs ~20-30s on every AoU run because the TSV has 400k+ rows and
# millions of cells. Cache the result keyed by sha256 of the inputs above.
_SAMPLE_TABLE_CACHE_VERSION = 1
_SAMPLE_TABLE_CACHE_SUBDIR = "sample_table_cache"


def _sample_table_cache_root(sample_table_path: Path) -> Path:
    """Cache root for the filtered sample table.

    Mirrors `_vcf_cache_dir`: drop the cache beside the source file (typically
    a writable cache directory) without resolving symlinks (read-only mounts).
    """
    return Path(os.path.abspath(sample_table_path)).parent / _CACHE_DIR_NAME / _SAMPLE_TABLE_CACHE_SUBDIR


def _sample_table_cache_key(
    *,
    sample_table_path: Path,
    fam_path: Path | None,
    sample_id_column: str,
    target_column: str,
    covariate_columns: tuple[str, ...],
    intersected_samples: Sequence[str],
) -> str:
    h = hashlib.sha256()
    h.update(f"v{_SAMPLE_TABLE_CACHE_VERSION}:".encode())
    try:
        tsv_stat = sample_table_path.stat()
        h.update(f"tsv_path={os.path.abspath(sample_table_path)}\n".encode())
        h.update(f"tsv_size={tsv_stat.st_size}\n".encode())
        h.update(f"tsv_mtime_ns={tsv_stat.st_mtime_ns}\n".encode())
    except OSError:
        # If we can't stat the TSV the build will fail anyway; emit a
        # distinguishable marker so we don't collide with a real key.
        h.update(b"tsv_unstattable\n")
    if fam_path is not None:
        try:
            fam_stat = fam_path.stat()
            h.update(f"fam_path={os.path.abspath(fam_path)}\n".encode())
            h.update(f"fam_size={fam_stat.st_size}\n".encode())
            h.update(f"fam_mtime_ns={fam_stat.st_mtime_ns}\n".encode())
        except OSError:
            h.update(b"fam_unstattable\n")
    else:
        h.update(b"fam_path=<none>\n")
    h.update(f"sample_id_column={sample_id_column}\n".encode())
    h.update(f"target_column={target_column}\n".encode())
    h.update(("covariate_columns=" + "\x00".join(covariate_columns) + "\n").encode())
    # Hash the intersected sample IDs as a single utf-8 blob: cheap and stable.
    joined = "\x00".join(intersected_samples).encode("utf-8")
    h.update(f"intersected_n={len(intersected_samples)}\n".encode())
    h.update(joined)
    return h.hexdigest()[:32]


def _sample_table_cache_path(cache_root: Path, key: str) -> Path:
    return cache_root / f"{key}.npz"


def _read_sample_table_cache(
    cache_path: Path,
    *,
    expected_n_samples: int,
) -> tuple[_SampleTable, int, int] | None:
    """Return the cached filtered sample table, or None on any failure.

    Any IO / validation error is treated as a cache miss — the caller will
    rebuild. We deliberately swallow exceptions to keep the run robust.
    """
    try:
        with np.load(cache_path, allow_pickle=False) as bundle:
            sample_ids_arr = bundle["sample_ids"]
            covariates = np.asarray(bundle["covariates"], dtype=np.float32)
            targets = np.asarray(bundle["targets"], dtype=np.float32)
            total_sample_rows = int(bundle["total_sample_rows"])
            unmatched_sample_rows = int(bundle["unmatched_sample_rows"])
            version = int(bundle["version"])
    except (OSError, KeyError, ValueError, EOFError):
        return None
    if version != _SAMPLE_TABLE_CACHE_VERSION:
        return None
    n_rows = int(sample_ids_arr.shape[0])
    if n_rows > expected_n_samples:
        return None
    if covariates.ndim != 2 or covariates.shape[0] != n_rows:
        return None
    if targets.ndim != 1 or targets.shape[0] != n_rows:
        return None
    sample_ids = [
        s.decode("utf-8") if isinstance(s, (bytes, bytearray)) else str(s)
        for s in sample_ids_arr.tolist()
    ]
    return (
        _SampleTable(sample_ids=sample_ids, covariates=covariates, targets=targets),
        total_sample_rows,
        unmatched_sample_rows,
    )


def _fsync_parent_dir(path: Path) -> None:
    try:
        dir_fd = os.open(path.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _write_sample_table_cache(
    cache_path: Path,
    *,
    sample_table: _SampleTable,
    total_sample_rows: int,
    unmatched_sample_rows: int,
) -> None:
    """Atomically persist the filtered sample table.

    Pattern: `.partial.<pid>.<uuid4>` → fsync → os.replace → fsync_dir.
    On any IOError the caller logs a warning and proceeds — cache writes
    are best-effort.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(
        f"{cache_path.name}.partial.{os.getpid()}.{uuid.uuid4().hex}"
    )
    # Store sample IDs as a fixed-width UTF-8 byte array so that
    # `np.load(..., allow_pickle=False)` can read them back without resorting
    # to pickled object arrays.
    sample_ids_arr = np.asarray(sample_table.sample_ids, dtype=np.bytes_)
    try:
        with tmp_path.open("wb") as handle:
            np.savez(
                handle,
                version=np.int64(_SAMPLE_TABLE_CACHE_VERSION),
                sample_ids=sample_ids_arr,
                covariates=np.asarray(sample_table.covariates, dtype=np.float32),
                targets=np.asarray(sample_table.targets, dtype=np.float32),
                total_sample_rows=np.int64(total_sample_rows),
                unmatched_sample_rows=np.int64(unmatched_sample_rows),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, cache_path)
        _fsync_parent_dir(cache_path)
    except OSError:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise


@dataclass(slots=True)
class _VcfCachePaths:
    key: str
    cache_dir: Path
    geno_path: Path
    var_path: Path
    stats_path: Path
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


def _vcf_cache_key(vcf_path: Path, config: ModelConfig) -> str:
    """Compute a hex digest that uniquely identifies a VCF file.

    The key is independent of which samples are kept so that changing
    covariates (e.g. adding PCs) never invalidates the genotype cache.
    Sample subsetting is done at load time, not at cache time.
    """
    h = hashlib.sha256()
    h.update(f"v{_CACHE_VERSION}:".encode())
    h.update(str(vcf_path.resolve()).encode())
    h.update(_cache_file_fingerprint(vcf_path))
    h.update(f"minimum_scale={config.minimum_scale:.17g}".encode())
    return h.hexdigest()[:24]


def _vcf_cache_source_signature(vcf_path: Path) -> str:
    h = hashlib.sha256()
    h.update(str(vcf_path.resolve()).encode())
    h.update(_cache_file_fingerprint(vcf_path))
    return h.hexdigest()


def _vcf_cache_dir(vcf_path: Path) -> Path:
    # Use the unresolved parent so caches land beside the path as referenced
    # (typically a writable cache dir). Resolving would follow symlinks into
    # read-only mounted dataset directories (e.g. AoU workspace mounts).
    return Path(os.path.abspath(vcf_path)).parent / _CACHE_DIR_NAME


def _vcf_cache_paths(vcf_path: Path, config: ModelConfig) -> _VcfCachePaths:
    cache_dir = _vcf_cache_dir(vcf_path)
    key = _vcf_cache_key(vcf_path, config)
    return _VcfCachePaths(
        key=key,
        cache_dir=cache_dir,
        geno_path=cache_dir / f"{key}.genotypes.npy",
        var_path=cache_dir / f"{key}.variants.npz",
        stats_path=cache_dir / f"{key}.stats.npy",
        manifest_path=cache_dir / f"{key}.manifest.json",
    )


def _cleanup_stale_vcf_cache_temps(cache_dir: Path, key: str) -> None:
    for stale_path in cache_dir.glob(f"{key}.*.tmp*"):
        if stale_path.is_dir():
            continue
        try:
            stale_path.unlink()
        except OSError:
            pass
    for stale_dir in cache_dir.glob(f"{key}.bundle.*"):
        if not stale_dir.is_dir():
            continue
        try:
            for child in stale_dir.iterdir():
                child.unlink(missing_ok=True)
            stale_dir.rmdir()
        except OSError:
            pass


def _load_vcf_cache_manifest(manifest_path: Path) -> dict[str, Any] | None:
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(manifest, dict):
        return None
    if int(manifest.get("manifest_version", 0)) != _VCF_CACHE_MANIFEST_VERSION:
        return None
    return {str(key): value for key, value in manifest.items()}


def _variants_to_metadata_arrays(
    variants: Sequence[_VariantDefaults],
) -> tuple[NDArray, NDArray, NDArray]:
    variant_ids = np.asarray(
        [variant.variant_id for variant in variants],
        dtype=f"<U{max((len(variant.variant_id) for variant in variants), default=1)}",
    )
    chromosomes = np.asarray(
        [variant.chromosome for variant in variants],
        dtype=f"<U{max((len(variant.chromosome) for variant in variants), default=1)}",
    )
    numeric = np.empty(len(variants), dtype=_VCF_CACHE_VARIANT_NUMERIC_DTYPE)
    numeric["variant_class_code"] = np.asarray(
        [_VARIANT_CLASS_TO_CODE[variant.variant_class] for variant in variants],
        dtype=np.uint16,
    )
    numeric["position"] = np.asarray([variant.position for variant in variants], dtype=np.int64)
    numeric["length"] = np.asarray([variant.length for variant in variants], dtype=np.float32)
    numeric["allele_frequency"] = np.asarray([variant.allele_frequency for variant in variants], dtype=np.float32)
    numeric["quality"] = np.asarray([variant.quality for variant in variants], dtype=np.float32)
    return variant_ids, chromosomes, numeric


def _write_variant_metadata(path: Path, variants: Sequence[_VariantDefaults]) -> None:
    variant_ids, chromosomes, numeric = _variants_to_metadata_arrays(variants)
    with path.open("wb") as handle:
        np.savez(
            handle,
            variant_ids=variant_ids,
            chromosomes=chromosomes,
            numeric=numeric,
        )


def _classify_variant_from_record(rec: object) -> VariantClass:
    raw = getattr(rec, "variant_class", None)
    if raw is None and isinstance(rec, dict):
        raw = rec.get("variant_class")
    if isinstance(raw, VariantClass):
        return raw
    if raw is not None:
        try:
            return VariantClass(str(raw))
        except ValueError:
            pass
    return VariantClass.OTHER_COMPLEX_SV


def _load_variant_metadata(path: Path) -> list[_VariantDefaults]:
    if path.suffix == ".pkl":
        import pickle
        with open(path, "rb") as fh:
            legacy_variants = pickle.load(fh)

        def _legacy_field(rec: object, name: str, default: object = "") -> object:
            val = getattr(rec, name, None)
            if val is not None:
                return val
            if isinstance(rec, dict):
                return rec.get(name, default)
            return default

        return [
            _VariantDefaults(
                variant_id=str(_legacy_field(rec, "variant_id", "")),
                variant_class=_classify_variant_from_record(rec),
                chromosome=str(_legacy_field(rec, "chromosome", "")),
                position=int(str(_legacy_field(rec, "position", 0))),
                length=float(str(_legacy_field(rec, "length", 0.0))),
                allele_frequency=float(str(_legacy_field(rec, "allele_frequency", 0.0))),
                quality=float(str(_legacy_field(rec, "quality", 0.0))),
            )
            for rec in legacy_variants
        ]
    payload = np.load(path, allow_pickle=False)
    try:
        variant_ids = np.asarray(payload["variant_ids"], dtype=str)
        chromosomes = np.asarray(payload["chromosomes"], dtype=str)
        numeric = np.asarray(payload["numeric"], dtype=_VCF_CACHE_VARIANT_NUMERIC_DTYPE)
    finally:
        payload.close()
    return [
        _VariantDefaults(
            variant_id=str(variant_id),
            variant_class=_VARIANT_CODE_TO_CLASS[int(numeric_row["variant_class_code"])],
            chromosome=str(chromosome),
            position=int(numeric_row["position"]),
            length=float(numeric_row["length"]),
            allele_frequency=float(numeric_row["allele_frequency"]),
            quality=float(numeric_row["quality"]),
        )
        for variant_id, chromosome, numeric_row in zip(
            variant_ids,
            chromosomes,
            numeric,
            strict=True,
        )
    ]


def _load_vcf_cache_stats(stats_path: Path) -> VariantStatistics:
    stats_payload = np.load(stats_path, mmap_mode="r")
    try:
        return VariantStatistics(
            means=np.asarray(stats_payload["means"], dtype=np.float32),
            scales=np.asarray(stats_payload["scales"], dtype=np.float32),
            allele_frequencies=np.asarray(stats_payload["allele_frequencies"], dtype=np.float32),
            support_counts=np.asarray(stats_payload["support_counts"], dtype=np.int32),
        )
    finally:
        if hasattr(stats_payload, "close"):
            stats_payload.close()


def _write_vcf_cache_stats(stats_path: Path, variant_stats: VariantStatistics) -> None:
    stats_matrix = np.empty(variant_stats.means.shape[0], dtype=_VCF_CACHE_STATS_DTYPE)
    stats_matrix["means"] = np.asarray(variant_stats.means, dtype=np.float32)
    stats_matrix["scales"] = np.asarray(variant_stats.scales, dtype=np.float32)
    stats_matrix["allele_frequencies"] = np.asarray(variant_stats.allele_frequencies, dtype=np.float32)
    stats_matrix["support_counts"] = np.asarray(variant_stats.support_counts, dtype=np.int32)
    np.save(stats_path, stats_matrix, allow_pickle=False)


def _plink_stats_cache_key(
    bed_path: Path,
    sample_indices: NDArray,
    config: ModelConfig,
) -> str:
    """Stable hex digest identifying a PLINK variant-stats computation.

    Stats depend on the .bed contents, which samples we compute over, and
    the minimum_scale floor used in _scales_from_centered_sum_squares — but
    nothing else. The sample-subset goes into the hash so train/test splits
    keyed off the sample table produce distinct cache entries; the bed
    fingerprint goes in so re-downloaded bytes never reuse stale stats.
    """
    h = hashlib.sha256()
    h.update(f"plink-stats-v{_CACHE_VERSION}:".encode())
    h.update(str(bed_path.resolve()).encode())
    h.update(_cache_file_fingerprint(bed_path))
    h.update(f"minimum_scale={config.minimum_scale:.17g}".encode())
    sample_array = np.ascontiguousarray(np.asarray(sample_indices, dtype=np.int64))
    h.update(f"sample_count={sample_array.shape[0]}".encode())
    h.update(sample_array.tobytes())
    return h.hexdigest()[:24]


def _plink_stats_cache_path(
    bed_path: Path,
    sample_indices: NDArray,
    config: ModelConfig,
) -> Path:
    cache_dir = _vcf_cache_dir(bed_path)
    key = _plink_stats_cache_key(bed_path, sample_indices, config)
    return cache_dir / f"plink_stats_{key}.npy"


def _load_plink_stats_from_cache(stats_path: Path) -> VariantStatistics | None:
    if not stats_path.exists():
        return None
    try:
        return _load_vcf_cache_stats(stats_path)
    except (OSError, ValueError, KeyError) as exc:
        log(f"  PLINK stats cache at {stats_path.name} unreadable ({exc!r}); recomputing")
        return None


def _write_plink_stats_cache(stats_path: Path, variant_stats: VariantStatistics) -> None:
    """Write atomically: build a per-process .tmp.npy then rename so
    interrupted runs (or concurrent disease sweeps racing on the same
    cache key) never leave a half-written file the next run would read
    as a cache hit. The .tmp infix sits BEFORE the .npy extension
    because np.save auto-appends '.npy' to any path that doesn't already
    end in it.
    """
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = stats_path.with_suffix(f".tmp.{os.getpid()}.{uuid.uuid4().hex}.npy")
    try:
        _write_vcf_cache_stats(tmp_path, variant_stats)
        tmp_path.replace(stats_path)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _try_bitpacked_plink_variant_stats(
    *,
    bed_path: Path,
    sample_indices: NDArray,
    n_samples: int,
    n_variants: int,
    config: ModelConfig,
) -> VariantStatistics | None:
    """Compute PLINK per-variant stats via the bitpacked GPU streaming kernel.

    Streams raw BED bytes straight from disk into GPU HBM and runs
    :func:`sv_pgs.screening_pipeline.run_screening_pass`, which emits
    per-variant ``count`` / ``sum`` / ``sumsq`` in a single pass. The raw
    moments are then converted to the same ``VariantStatistics`` triple
    the legacy int8 streaming path emits and persisted to the same
    on-disk ``plink_stats_<key>.npy`` cache so future runs short-circuit
    via the existing ``_load_plink_stats_from_cache`` path.

    Returns ``None`` (no fallback side-effects) on any import / GPU
    failure; the caller then drops back to the legacy int8 streaming
    pass. The int8 ``.npy`` cache is never created on this path —
    downstream EM consumes the bitpacked device matrix directly via
    :func:`sv_pgs.pipeline._maybe_upgrade_to_bitpacked`.
    """
    stats_path = _plink_stats_cache_path(bed_path, sample_indices, config)
    cached = _load_plink_stats_from_cache(stats_path)
    if cached is not None:
        log(
            f"  variant stats: bitpacked GPU streaming path (cache hit @ {stats_path.name}, "
            f"{cached.means.shape[0]:,} variants)"
        )
        return cached

    log("  variant stats: bitpacked GPU streaming path (single pass)")
    # Broad except: a missing cupy build or a sv_pgs.bitpacked.* top-level
    # error must NOT silently disable the fast path — log the concrete
    # exception type + message so the next run shows the root cause.
    try:
        from sv_pgs.screening_pipeline import run_screening_pass  # lazy
    except Exception as exc:
        log(
            f"  variant stats: bitpacked path failed at screening_pipeline import "
            f"({type(exc).__name__}: {exc}); falling back to legacy int8 streaming pass"
        )
        return None

    sample_intersect_arr = np.asarray(sample_indices, dtype=np.int64)
    full_cohort = (
        sample_intersect_arr.size == n_samples
        and np.array_equal(
            sample_intersect_arr,
            np.arange(n_samples, dtype=sample_intersect_arr.dtype),
        )
    )
    intersect_arg: NDArray | None = None if full_cohort else sample_intersect_arr
    effective_n_samples = (
        int(sample_intersect_arr.shape[0]) if intersect_arg is not None else int(n_samples)
    )

    log(
        f"  variant stats: invoking run_screening_pass(bed={bed_path.name}, "
        f"n_samples={int(n_samples)}, n_variants={int(n_variants)}, "
        f"effective_n_samples={effective_n_samples}, "
        f"sample_intersect={'None' if intersect_arg is None else f'array[{intersect_arg.size}]'})"
    )
    try:
        screening = run_screening_pass(
            [Path(bed_path)],
            [int(n_samples)],
            [int(n_variants)],
            sample_intersect=intersect_arg,
            count_a1=True,
        )
    except Exception as exc:
        log(
            f"  variant stats: bitpacked path failed in run_screening_pass "
            f"({type(exc).__name__}: {exc}); falling back to legacy int8 streaming pass"
        )
        return None

    counts = np.asarray(screening["count"], dtype=np.int32)
    sums = np.asarray(screening["sum"], dtype=np.float64)
    sumsq = np.asarray(screening["sumsq"], dtype=np.float64)
    # centered_sum_squares = sum(d**2) - sum(d)**2 / count, with count==0
    # guarded so we never divide by zero. Matches the per-batch identity
    # accumulated by _batch_all_stats / _batch_all_stats_i8 (see
    # preprocessing.py:_means_and_scales_with_floor for the consumer).
    safe_counts = np.maximum(counts.astype(np.int64), 1)
    centered_sum_squares = np.maximum(sumsq - (sums * sums) / safe_counts, 0.0)
    dosage_like = np.ones(int(n_variants), dtype=bool)
    means, scales = _means_and_scales_with_floor(
        sums=sums,
        non_missing_counts=counts,
        centered_sum_squares=centered_sum_squares,
        sample_count=effective_n_samples,
        minimum_scale=float(config.minimum_scale),
    )
    allele_frequencies = _allele_frequencies_from_means(means=means, dosage_like=dosage_like)
    # support_counts is the non-zero-dosage count in the legacy path. The
    # screening kernel only emits non-missing count, which is an upper
    # bound. The downstream sparse/dense routing in
    # genotype.py:HYBRID_SPARSE_SUPPORT_THRESHOLD treats this conservatively
    # (variants get the dense path; correctness is unchanged).
    variant_stats = VariantStatistics(
        means=np.asarray(means, dtype=np.float32),
        scales=np.asarray(scales, dtype=np.float32),
        allele_frequencies=np.asarray(allele_frequencies, dtype=np.float32),
        support_counts=np.asarray(counts, dtype=np.int32),
    )
    try:
        _write_plink_stats_cache(stats_path, variant_stats)
        log(f"  PLINK stats cache saved via bitpacked path ({stats_path.stat().st_size / 1e6:.1f} MB)  mem={mem()}")
    except OSError as exc:
        log(f"  PLINK stats cache write failed ({exc!r}); continuing without cache")
    return variant_stats


def _plink_int8_cache_path(
    bed_path: Path,
    sample_indices: NDArray,
    config: ModelConfig,
) -> Path:
    """Path for the decoded int8 (n_samples, n_variants) PLINK cache.

    Shares the stats-cache key (which already includes bed fingerprint +
    sample subset + minimum_scale), so the int8 cache and the stats cache
    invalidate together. The file is a standard .npy mmap: future
    consumers can `np.lib.format.open_memmap(...)` it instantly.
    """
    cache_dir = _vcf_cache_dir(bed_path)
    key = _plink_stats_cache_key(bed_path, sample_indices, config)
    return cache_dir / f"plink_int8_{key}.npy"


def _open_int8_memmap(
    path: Path,
    *,
    mode: Literal["r", "r+", "w+", "c"],
    shape: tuple[int, ...] | None = None,
    fortran_order: bool = False,
) -> np.memmap[Any, np.dtype[np.int8]]:
    """Strict-mypy-safe wrapper around the untyped ``np.lib.format.open_memmap``."""
    open_memmap = cast(Any, np.lib.format.open_memmap)
    if mode == "w+":
        if shape is None:
            raise ValueError("shape is required for write-mode memmap")
        result = open_memmap(path, mode=mode, dtype=np.int8, shape=shape, fortran_order=fortran_order)
    else:
        result = open_memmap(path, mode=mode)
    return cast("np.memmap[Any, np.dtype[np.int8]]", result)


def _open_plink_int8_cache_for_read(int8_path: Path) -> I8Array | None:
    """Return a read-only mmap view of the int8 cache, or None on miss/error."""
    if not int8_path.exists():
        return None
    try:
        view: np.memmap[Any, np.dtype[np.int8]] = _open_int8_memmap(int8_path, mode="r")
    except (OSError, ValueError, EOFError) as exc:
        log(f"  PLINK int8 cache at {int8_path.name} unreadable ({exc!r}); will recompute")
        return None
    # Hint the kernel to pre-fault and retain pages: avoids per-block 3s
    # "upload" times that are really disk-read masquerading as H2D when
    # mmap pages get evicted under memory pressure between training blocks.
    _madvise_willneed(view)
    return view


def _madvise_willneed(array: NDArray) -> None:
    """Best-effort MADV_WILLNEED on the backing mmap of `array`.

    Prefers ``mmap.mmap.madvise`` (available on all platforms where
    ``mmap`` supports madvise, including macOS) and falls back to
    ``os.posix_madvise`` (Linux-only on CPython; missing on darwin).

    No-op on systems without madvise support or on non-mmap arrays.
    Failures are silent — this is a hint, not a correctness requirement.
    """
    try:
        import mmap as _mmap_module
    except ImportError:
        return
    base: object = array
    while getattr(base, "base", None) is not None:
        next_base = getattr(base, "base", None)
        if next_base is None:
            break
        base = next_base
    if not isinstance(base, _mmap_module.mmap):
        return

    # Preferred path: mmap.mmap.madvise(MADV_WILLNEED). Exists on Linux and
    # macOS in CPython 3.8+ and reliably reaches the kernel hint.
    mmap_madvise = getattr(base, "madvise", None)
    mmap_willneed = getattr(_mmap_module, "MADV_WILLNEED", None)
    if mmap_madvise is not None and mmap_willneed is not None:
        try:
            mmap_madvise(mmap_willneed)
            return
        except (OSError, ValueError):
            pass  # fall through to posix_madvise fallback

    # Fallback for older / non-standard runtimes: os.posix_madvise. This is
    # absent on darwin, so it is genuinely a fallback rather than the
    # primary path.
    posix_madvise = getattr(os, "posix_madvise", None)
    willneed = getattr(os, "POSIX_MADV_WILLNEED", None)
    if posix_madvise is None or willneed is None:
        return
    try:
        posix_madvise(base, 0, len(base), willneed)
    except (OSError, ValueError):
        pass


def compute_plink_variant_statistics_cached(
    raw_genotypes: PlinkRawGenotypeMatrix,
    bed_path: Path,
    sample_indices: NDArray,
    config: ModelConfig,
) -> tuple[VariantStatistics, Path | None]:
    """compute_variant_statistics on a PLINK source, with stats-only disk caching.

    Returns ``(variant_stats, int8_cache_path)``. ``int8_cache_path`` is the
    path to a *pre-existing* legacy decoded int8 mmap if one is on disk;
    None otherwise. This path no longer creates new int8 caches — the
    bitpacked GPU streaming path supersedes it. Old caches still load.
    """
    stats_path = _plink_stats_cache_path(bed_path, sample_indices, config)
    int8_path = _plink_int8_cache_path(bed_path, sample_indices, config)
    cached_stats = _load_plink_stats_from_cache(stats_path)
    if cached_stats is not None:
        log(
            f"  PLINK stats cache hit: {stats_path.name} "
            f"({cached_stats.means.shape[0]:,} variants)  mem={mem()}"
        )
        return cached_stats, (int8_path if int8_path.exists() else None)
    log(f"  PLINK stats cache miss: computing fresh and writing {stats_path.name}")
    variant_stats = compute_variant_statistics(raw_genotypes, config=config)
    try:
        _write_plink_stats_cache(stats_path, variant_stats)
        log(f"  PLINK stats cache saved ({stats_path.stat().st_size / 1e6:.1f} MB)  mem={mem()}")
    except OSError as exc:
        log(f"  PLINK stats cache write failed ({exc!r}); continuing without cache")
    return variant_stats, (int8_path if int8_path.exists() else None)


def _vcf_cache_audit_reason(vcf_path: Path, paths: _VcfCachePaths) -> str | None:
    """Cheap content audit on an existing cache bundle.

    Returns None if the cache looks healthy, or a short human-readable reason
    string if it should be invalidated and re-parsed. Detects two failure
    modes that previously slipped through the per-region merge check:

    - The cached variant_ids contain duplicates within a single VCF cache.
    - The cached chromosomes do not match the chromosome the source VCF
      actually covers (cross-contamination — e.g. chr13's cache file ends
      up with chr1 records).

    We only inspect the variants metadata file (~few MB) — no VCF record
    iteration — so this is cheap even when called on every VCF at startup.
    """
    variants_path = paths.var_path
    if not variants_path.exists():
        legacy = paths.cache_dir / f"{paths.key}.variants.pkl"
        if not legacy.exists():
            return None  # Bundle isn't complete; the normal check will skip it.
        variants_path = legacy
    try:
        if variants_path.suffix == ".npz":
            with np.load(variants_path, allow_pickle=False) as data:
                variant_ids = np.asarray(data["variant_ids"]).astype(str, copy=False)
                chromosomes = np.asarray(data["chromosomes"]).astype(str, copy=False)
        else:
            variants = _load_variant_metadata(variants_path)
            variant_ids = np.asarray([v.variant_id for v in variants], dtype=object)
            chromosomes = np.asarray([v.chromosome for v in variants], dtype=object)
    except (OSError, ValueError, KeyError, EOFError) as exc:
        return f"variant metadata unreadable ({exc})"

    if variant_ids.shape[0] == 0:
        return "variant metadata is empty"

    if np.unique(variant_ids).shape[0] != variant_ids.shape[0]:
        return f"variant_ids contain duplicates within the cache ({variant_ids.shape[0] - np.unique(variant_ids).shape[0]} duplicate rows)"

    info = _vcf_contig_info(vcf_path)
    if info is not None:
        expected_chrom = str(info[0])
        unique_chroms = set(chromosomes.tolist())
        if unique_chroms != {expected_chrom}:
            preview = sorted(unique_chroms - {expected_chrom})[:3]
            return (
                f"cached chromosomes {sorted(unique_chroms)} do not match "
                f"VCF contig {expected_chrom!r} (extras: {preview})"
            )

    return None


def _is_vcf_cache_bundle_complete(paths: _VcfCachePaths) -> bool:
    if not paths.geno_path.exists():
        return False
    # Support both new (.variants.npz) and legacy (.variants.pkl) formats
    var_exists = paths.var_path.exists()
    if not var_exists:
        legacy_var = paths.cache_dir / f"{paths.key}.variants.pkl"
        var_exists = legacy_var.exists()
    if not var_exists:
        return False
    # Check stats — manifest may reference old name, or may not exist
    manifest = _load_vcf_cache_manifest(paths.manifest_path)
    if manifest is not None:
        stats_filename = str(manifest.get("stats_file", paths.stats_path.name))
        if (paths.cache_dir / stats_filename).exists():
            return True
    # Fall through: check for stats files directly (legacy formats)
    if paths.stats_path.exists():
        return True
    legacy_stats_npy = paths.cache_dir / f"{paths.key}.stats.npy"
    legacy_stats_npz = paths.cache_dir / f"{paths.key}.stats.npz"
    return legacy_stats_npy.exists() or legacy_stats_npz.exists()


def _ensure_vcf_cache_matrix_fast(paths: _VcfCachePaths, genotype_matrix: I8Array) -> I8Array:
    if genotype_matrix.flags.f_contiguous and not genotype_matrix.flags.c_contiguous:
        return genotype_matrix
    variants = _load_variant_metadata(paths.var_path)
    variant_stats = _load_vcf_cache_stats(paths.stats_path)
    _save_vcf_to_cache(
        vcf_path=paths.geno_path,
        genotype_matrix=np.asarray(genotype_matrix, dtype=np.int8),
        variants=variants,
        variant_stats=variant_stats,
        cache_paths=paths,
    )
    return np.asarray(np.load(paths.geno_path, mmap_mode="r"), dtype=np.int8)


def _load_vcf_from_cache(
    vcf_path: Path,
    config: ModelConfig,
    *,
    mmap_mode: Literal["r", "r+", "w+", "c"] | None = None,
) -> tuple[I8Array, list[_VariantDefaults], VariantStatistics] | None:
    """Try to load cached VCF parse results. Returns None on miss."""
    paths = _vcf_cache_paths(vcf_path, config)
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
        # Legacy caches may not have manifests — infer from matrix shape
        _t0 = time.monotonic()
        genotype_matrix = np.load(paths.geno_path, mmap_mode=effective_mmap_mode)
        log(f"  mmap ready: {genotype_matrix.shape} {genotype_matrix.dtype} ({time.monotonic()-_t0:.1f}s)")
        if manifest is not None:
            expected_sample_count = int(manifest["sample_count"])
            expected_variant_count = int(manifest["variant_count"])
            stats_filename = str(manifest.get("stats_file", paths.stats_path.name))
            stats_path = paths.cache_dir / stats_filename
        else:
            expected_sample_count = genotype_matrix.shape[0]
            expected_variant_count = genotype_matrix.shape[1]
            # Find stats file — try all known locations
            stats_path = paths.stats_path
            if not stats_path.exists():
                for suffix in (".stats.npy", ".stats.npz"):
                    candidate = paths.cache_dir / f"{paths.key}{suffix}"
                    if candidate.exists():
                        stats_path = candidate
                        break
        if not stats_path.exists():
            raise ValueError(f"stats file not found: tried {stats_path.name}")
        _t1 = time.monotonic()
        # Support both .variants.npz (new) and .variants.pkl (legacy)
        var_path = paths.var_path
        if not var_path.exists():
            legacy_var = paths.cache_dir / f"{paths.key}.variants.pkl"
            if legacy_var.exists():
                var_path = legacy_var
        variants = _load_variant_metadata(var_path)
        log(f"  variants loaded: {len(variants)} ({time.monotonic()-_t1:.1f}s)")
        _t2 = time.monotonic()
        variant_stats = _load_vcf_cache_stats(stats_path)
        log(f"  stats loaded ({time.monotonic()-_t2:.1f}s)")
        if genotype_matrix.shape[0] != expected_sample_count:
            raise ValueError(f"cached sample count mismatch: {genotype_matrix.shape[0]} != {expected_sample_count}")
        if genotype_matrix.shape[1] != expected_variant_count:
            raise ValueError(f"cached variant count mismatch: {genotype_matrix.shape[1]} != {expected_variant_count}")
        stats_lengths = {
            int(variant_stats.means.shape[0]),
            int(variant_stats.scales.shape[0]),
            int(variant_stats.allele_frequencies.shape[0]),
            int(variant_stats.support_counts.shape[0]),
        }
        if len(stats_lengths) != 1:
            raise ValueError("cached stats shape mismatch")
        stats_variant_count = next(iter(stats_lengths))
        cached_variant_count = int(genotype_matrix.shape[1])
        variant_record_count = len(variants)
        if cached_variant_count != variant_record_count:
            raise ValueError(
                f"cached variant metadata mismatch: matrix has {cached_variant_count} columns, "
                f"but variants table has {variant_record_count} rows"
            )
        if cached_variant_count != stats_variant_count:
            raise ValueError(
                f"cached stats mismatch: matrix has {cached_variant_count} columns, "
                f"but stats describe {stats_variant_count} variants"
            )
        genotype_matrix = _ensure_vcf_cache_matrix_fast(paths, genotype_matrix)
        log(f"  cached matrix {genotype_matrix.shape}, {len(variants)} variants")
        return genotype_matrix, variants, variant_stats
    except (OSError, RuntimeError, ValueError, KeyError, EOFError) as exc:
        log(f"VCF cache load failed ({exc}), will re-parse")
        return None


def _read_vcf_sample_ids(vcf_path: Path) -> list[str]:
    reader = _open_vcf_reader(vcf_path)
    try:
        return [str(sample_id) for sample_id in reader.samples]
    finally:
        reader.close()


_INCREMENTAL_CHECKPOINT_BYTE_INTERVAL = 256 * 1024 * 1024
_INCREMENTAL_CHECKPOINT_TIME_SECONDS = 30.0


def _incremental_variant_chunk_path(cache_dir: Path, key: str, chunk_index: int) -> Path:
    return cache_dir / f"{key}.inc.variants.{chunk_index:06d}.npz"


def _iter_incremental_variant_chunk_paths(cache_dir: Path, key: str) -> list[Path]:
    return sorted(cache_dir.glob(f"{key}.inc.variants.*.npz"))


def _prepare_keep_sample_selector(
    keep_sample_indices: NDArray | Sequence[int] | None,
    total_sample_count: int,
) -> slice | NDArray | None:
    if keep_sample_indices is None:
        return None
    indices = np.asarray(keep_sample_indices, dtype=np.intp)
    if indices.ndim != 1:
        raise ValueError("keep_sample_indices must be one-dimensional.")
    if indices.size == 0:
        return slice(0, 0)
    if (
        indices.size == total_sample_count
        and int(indices[0]) == 0
        and int(indices[-1]) == total_sample_count - 1
        and np.all(np.diff(indices) == 1)
    ):
        return None
    if np.all(np.diff(indices) == 1):
        return slice(int(indices[0]), int(indices[-1]) + 1)
    return indices


def _record_gt_types_to_int8(
    gt_types: I8Array,
    gt_map: I8Array,
    keep_selector: slice | NDArray | None,
) -> I8Array:
    selected_gt_types = gt_types if keep_selector is None else gt_types[keep_selector]
    return np.asarray(gt_map[selected_gt_types], dtype=np.int8)


def _fast_int8_dosage_stats(col: I8Array) -> tuple[int, int, int, int]:
    """Per-variant (sum, sum_sq, n_observed, n_nonzero) for an int8 dosage column.

    Values are {0, 1, 2, PLINK_MISSING_INT8}. Counting each non-missing value
    once is cheaper than masking + astype(int64) + four reductions because each
    `(col == k).sum()` is a single fused int8 pass.
    """
    count_1 = int(np.count_nonzero(col == 1))
    count_2 = int(np.count_nonzero(col == 2))
    count_0 = int(np.count_nonzero(col == 0))
    n_observed = count_0 + count_1 + count_2
    dosage_sum = count_1 + 2 * count_2
    dosage_sum_sq = count_1 + 4 * count_2
    n_nonzero = count_1 + count_2
    return dosage_sum, dosage_sum_sq, n_observed, n_nonzero


def _region_output_complete(output_prefix: str | Path) -> bool:
    prefix = Path(output_prefix)
    return (
        Path(f"{prefix}.geno").exists()
        and Path(f"{prefix}.variants.npz").exists()
        and Path(f"{prefix}.stats").exists()
    )


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


_BCFTOOLS_QUERY_FORMAT = (
    r"%CHROM" "\t" r"%POS" "\t" r"%ID" "\t" r"%REF" "\t" r"%ALT" "\t"
    r"%QUAL" "\t" r"%INFO/SVTYPE" "\t" r"%INFO/SVLEN" "\t" r"%INFO/AF" "\t"
    r"%INFO/END" "\t" r"[%GT,]" "\n"
)


def _bcftools_executable() -> str:
    """Return the bcftools binary, hard-failing if it is not on PATH.

    The fast parser depends on bcftools query for genotype decoding; there is
    no cyvcf2 fallback path. Make the missing-dependency error obvious so the
    AoU image can be patched rather than silently regressing to ~15 variants/s.
    """
    import shutil

    path = shutil.which("bcftools")
    if path is None:
        raise RuntimeError(
            "bcftools is required for the VCF precache fast path but was not "
            "found on PATH. Install bcftools (e.g. `apt-get install bcftools` "
            "or `conda install -c bioconda bcftools`) and retry."
        )
    return path


def _parse_optional_bcftools_float(field: bytes) -> float | None:
    if not field or field == b".":
        return None
    if b"," in field:
        field = field.split(b",", 1)[0]
        if field == b".":
            return None
    return float(field)


def _parse_optional_bcftools_int(field: bytes) -> int | None:
    if not field or field == b".":
        return None
    if b"," in field:
        field = field.split(b",", 1)[0]
        if field == b".":
            return None
    return int(field)


_BCFTOOLS_VALID_BASES = frozenset("ACGTNacgtn")


def _alt_is_symbolic_or_bnd(alt: str) -> bool:
    return bool(alt) and (alt[0] == "<" or "[" in alt or "]" in alt)


def _is_atcgn_only(value: str) -> bool:
    return bool(value) and all(ch in _BCFTOOLS_VALID_BASES for ch in value)


def _variant_defaults_from_bcftools_fields(
    chrom: str,
    pos: int,
    record_id_field: bytes,
    ref: str,
    alt: str,
    qual_field: bytes,
    svtype_field: bytes,
    svlen_field: bytes,
    af_field: bytes,
    end_field: bytes,
) -> _VariantDefaults:
    """Build _VariantDefaults from one bcftools query line.

    Mirrors _variant_defaults_from_vcf_record exactly: same classification
    branches, same length-derivation fallback chain, same AF/quality defaults.
    """
    record_id_text = record_id_field.decode("utf-8") if record_id_field else ""
    variant_id = (
        f"{chrom}:{pos}:{ref}:{alt}"
        if not record_id_text or record_id_text == "."
        else record_id_text
    )

    alt_is_sv = _alt_is_symbolic_or_bnd(alt)
    ref_atcgn = _is_atcgn_only(ref)
    alt_atcgn = (not alt_is_sv) and _is_atcgn_only(alt)
    is_snp = len(ref) == 1 and len(alt) == 1 and ref_atcgn and alt_atcgn
    is_indel = (
        not is_snp
        and not alt_is_sv
        and ref_atcgn
        and alt_atcgn
        and len(ref) != len(alt)
    )

    svlen_value = _parse_optional_bcftools_float(svlen_field)
    if svlen_value is not None:
        length = float(abs(svlen_value))
    elif is_snp:
        length = 1.0
    else:
        end_value = _parse_optional_bcftools_int(end_field)
        if end_value is not None and end_value >= pos:
            length = float(end_value - pos + 1)
        else:
            length = float(max(len(ref), len(alt)))

    if is_snp:
        variant_class = VariantClass.SNV
    elif is_indel:
        variant_class = VariantClass.SMALL_INDEL
    else:
        svtype_text = svtype_field.decode("utf-8") if svtype_field else ""
        variant_token = (
            _normalize_variant_token(svtype_text)
            if svtype_text and svtype_text != "."
            else None
        )
        if variant_token is None:
            variant_token = _normalize_variant_token(alt)
        if variant_token is None:
            variant_class = VariantClass.OTHER_COMPLEX_SV
        else:
            variant_class = _structural_variant_class_from_token(variant_token, length)

    allele_frequency = _parse_optional_bcftools_float(af_field)
    if allele_frequency is None:
        allele_frequency = -1.0

    quality_value = _parse_optional_bcftools_float(qual_field)
    quality = _normalize_quality(quality_value)

    return _VariantDefaults(
        variant_id=variant_id,
        variant_class=variant_class,
        chromosome=chrom,
        position=pos,
        length=length,
        allele_frequency=allele_frequency,
        quality=quality,
    )


def _parse_gt_block_to_int8(gt_block: bytes, sample_count: int) -> I8Array:
    """Vectorized decode of a bcftools `[%GT,]` block to int8 dosages.

    Handles both diploid (`"X/X,"` = 4 bytes/sample) and haploid (`"X,"` =
    2 bytes/sample) layouts. AoU SV VCFs mix the two: most records are
    diploid biallelic, but some sites are emitted with haploid GTs (often
    when a sample has no call, producing `".,"`). We detect the layout from
    the comma positions per record rather than assuming a fixed width.

    Missing alleles ('.' in any position) collapse to PLINK_MISSING_INT8.
    Phasing characters ('/' vs '|') at the middle byte of a diploid GT are
    not read, so phased and unphased records decode identically.
    """
    buf = np.frombuffer(gt_block, dtype=np.uint8)
    # ord(',') = 44
    comma_positions = np.flatnonzero(buf == 44)
    if comma_positions.shape[0] != sample_count:
        raise ValueError(
            "bcftools GT block has unexpected comma count: "
            f"got {comma_positions.shape[0]} commas, expected {sample_count} "
            f"(one trailing comma per sample). Block length: {len(gt_block)} bytes."
        )
    # Each sample's GT spans [start, end) where start is the byte after the
    # previous comma (or 0 for the first sample) and end is the comma index.
    starts = np.empty(sample_count, dtype=np.int64)
    starts[0] = 0
    starts[1:] = comma_positions[:-1] + 1
    lengths = comma_positions - starts

    # 46 = ord('.'), 48 = ord('0')
    if bool(np.all(lengths == 3)):
        # Diploid: "X/X" or "X|X" or "./." etc.
        first_allele = buf[starts]
        second_allele = buf[starts + 2]
        missing_mask = (first_allele == 46) | (second_allele == 46)
        dosage_i16 = (
            (first_allele.astype(np.int16) - 48)
            + (second_allele.astype(np.int16) - 48)
        )
        dosage = dosage_i16.astype(np.int8)
        dosage[missing_mask] = PLINK_MISSING_INT8
        return np.asarray(dosage, dtype=np.int8)
    if bool(np.all(lengths == 1)):
        # Haploid: "X" or "." per sample. Treat the allele count directly as
        # dosage (0 -> 0, 1 -> 1) and any '.' as missing.
        allele = buf[starts]
        missing_mask = allele == 46
        dosage = (allele.astype(np.int16) - 48).astype(np.int8)
        dosage[missing_mask] = PLINK_MISSING_INT8
        return dosage

    unique_widths = sorted({int(value) for value in np.unique(lengths).tolist()})
    raise ValueError(
        "bcftools GT block has mixed per-sample widths within a single record: "
        f"observed widths {unique_widths}. All samples at a site must share "
        "the same ploidy; this VCF record violates that. Investigate the "
        "source file rather than papering over it."
    )


_REGION_CHECKPOINT_BYTE_INTERVAL = 64 * 1024 * 1024  # flush after 64 MB of genotype bytes
_REGION_CHECKPOINT_TIME_SECONDS = 20.0


def _region_variant_chunk_paths(output_prefix: str) -> list[Path]:
    """Return sorted list of intermediate variant-metadata chunks for one region."""
    prefix_path = Path(output_prefix)
    parent = prefix_path.parent
    glob_pattern = f"{prefix_path.name}.variants.[0-9][0-9][0-9][0-9][0-9][0-9].npz"
    return sorted(parent.glob(glob_pattern))


def _parse_region_string(region: str) -> tuple[str, int, int] | None:
    """Parse 'chr:start-end' into (chrom, start, end). Returns None on any
    deviation from that exact shape — caller falls back to a full re-parse.
    """
    if ":" not in region or "-" not in region:
        return None
    try:
        chrom_part, range_part = region.split(":", 1)
        start_text, end_text = range_part.split("-", 1)
        return chrom_part, int(start_text), int(end_text)
    except (ValueError, IndexError):
        return None


def _region_parse_worker(args: tuple[str, str | None, str, int]) -> tuple[int, str]:
    """Worker: stream one region of one VCF through bcftools query, write
    raw binary output files. Runs in a separate process. Returns
    (variant_count, output_prefix).

    Resumable: every ~64 MB of genotype output (or every 20 s, whichever
    comes first) we flush the genotype/stats binary streams, write the
    pending variant metadata to its own chunk file, and atomically rewrite
    a small progress.json. If the worker is killed and re-launched, we
    truncate any partial trailing writes and restart bcftools from the
    record after the last checkpointed position via `-r chr:N+1-end`, so
    we only ever lose the records written since the last checkpoint
    (seconds of work, not the entire region).
    """
    import json
    import struct
    import subprocess
    import sys
    import tempfile
    import time

    vcf_path_str, region, output_prefix, threads_per_reader = args
    vcf_path = Path(vcf_path_str)
    vcf_name = vcf_path.name

    # One-time header read for sample count. cyvcf2 here is fine — it doesn't
    # iterate any records, just parses the header (~tens of ms).
    sample_count = len(_read_vcf_sample_ids(vcf_path))

    geno_path = Path(f"{output_prefix}.geno")
    stats_path = Path(f"{output_prefix}.stats")
    progress_path = Path(f"{output_prefix}.progress.json")
    final_variants_path = Path(f"{output_prefix}.variants.npz")

    # If a prior run completed this region cleanly, the final .variants.npz
    # exists and progress.json is gone — nothing to do.
    if final_variants_path.exists() and not progress_path.exists():
        return 0, output_prefix

    # Attempt to resume from a previous checkpoint. Resume requires a region
    # string (so we can rewind bcftools via -r) and a progress.json whose
    # metadata matches this task.
    resume_count = 0
    resume_chunks = 0
    resume_chrom: str | None = None
    resume_pos: int | None = None
    resumed = False
    if progress_path.exists() and region is not None:
        try:
            state = json.loads(progress_path.read_text(encoding="utf-8"))
            if (
                int(state.get("sample_count", -1)) == sample_count
                and state.get("region") == region
            ):
                candidate_count = int(state["count"])
                candidate_chunks = int(state["chunk_count"])
                expected_geno_bytes = candidate_count * sample_count
                expected_stats_bytes = candidate_count * 24
                # Trim any trailing partial bytes from a write that crashed
                # mid-record. We only ever wrote whole records, so the
                # tail past expected_* is junk.
                if geno_path.exists() and geno_path.stat().st_size > expected_geno_bytes:
                    with open(geno_path, "r+b") as fh:
                        fh.truncate(expected_geno_bytes)
                if stats_path.exists() and stats_path.stat().st_size > expected_stats_bytes:
                    with open(stats_path, "r+b") as fh:
                        fh.truncate(expected_stats_bytes)
                geno_size = geno_path.stat().st_size if geno_path.exists() else 0
                stats_size = stats_path.stat().st_size if stats_path.exists() else 0
                if geno_size != expected_geno_bytes or stats_size != expected_stats_bytes:
                    raise ValueError(
                        f"size mismatch after truncate: geno={geno_size} stats={stats_size} "
                        f"expected geno={expected_geno_bytes} stats={expected_stats_bytes}"
                    )
                chunk_paths = _region_variant_chunk_paths(output_prefix)
                if len(chunk_paths) < candidate_chunks:
                    raise ValueError(
                        f"missing variant chunks: have {len(chunk_paths)}, "
                        f"progress says {candidate_chunks}"
                    )
                # Drop any chunks past the checkpointed count (incomplete tail).
                for extra in chunk_paths[candidate_chunks:]:
                    extra.unlink(missing_ok=True)
                resume_count = candidate_count
                resume_chunks = candidate_chunks
                resume_chrom = state.get("last_chrom")
                resume_pos = state.get("last_pos")
                resumed = True
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            print(
                f"  [worker] {vcf_name}: progress checkpoint unusable ({exc}); restarting region",
                file=sys.stderr,
                flush=True,
            )

    # Always clear stale .geno/.stats/chunks if we are NOT resuming. A prior
    # worker version (or a crash before the first checkpoint) can leave
    # partial binary files on disk without a matching progress.json; opening
    # them in append mode would silently concatenate stale bytes onto fresh
    # output and produce a cache where matrix col count > variant count.
    if not resumed:
        geno_path.unlink(missing_ok=True)
        stats_path.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)
        for chunk in _region_variant_chunk_paths(output_prefix):
            chunk.unlink(missing_ok=True)

    # Compute the bcftools region for this invocation. If we're resuming,
    # narrow the start to (last checkpointed pos + 1).
    effective_region = region
    if region is not None and resume_pos is not None and resume_chrom is not None:
        parsed = _parse_region_string(region)
        if parsed is not None:
            chrom_part, _, end_part = parsed
            if chrom_part == resume_chrom:
                effective_region = f"{chrom_part}:{int(resume_pos) + 1}-{end_part}"

    bcftools = _bcftools_executable()
    view_threads = max(int(threads_per_reader), 1)
    view_cmd = [bcftools, "view", "--threads", str(view_threads), "-Ou"]
    if effective_region:
        view_cmd += ["-r", effective_region]
    view_cmd += [str(vcf_path)]
    query_cmd = [bcftools, "query", "-f", _BCFTOOLS_QUERY_FORMAT, "-"]

    stats_pack = struct.Struct("<qqii")
    geno_fh = open(geno_path, "ab")
    stats_fh = open(stats_path, "ab")

    buffered_variants: list[_VariantDefaults] = []
    geno_buffer = bytearray()
    stats_buffer = bytearray()

    count = resume_count
    chunk_index = resume_chunks
    last_chrom: str | None = resume_chrom
    last_pos: int | None = resume_pos
    last_checkpoint_time = time.monotonic()
    bytes_since_checkpoint = 0
    t_start = time.monotonic()
    last_log = t_start

    def _atomic_write_progress() -> None:
        tmp = progress_path.with_name(progress_path.name + ".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "count": count,
                    "chunk_count": chunk_index,
                    "sample_count": sample_count,
                    "region": region,
                    "last_chrom": last_chrom,
                    "last_pos": last_pos,
                }
            ),
            encoding="utf-8",
        )
        tmp.replace(progress_path)

    def _checkpoint(force: bool = False) -> None:
        nonlocal chunk_index, last_checkpoint_time, bytes_since_checkpoint
        if not force:
            elapsed = time.monotonic() - last_checkpoint_time
            if (
                bytes_since_checkpoint < _REGION_CHECKPOINT_BYTE_INTERVAL
                and elapsed < _REGION_CHECKPOINT_TIME_SECONDS
            ):
                return
        if geno_buffer:
            geno_fh.write(bytes(geno_buffer))
            geno_buffer.clear()
        if stats_buffer:
            stats_fh.write(bytes(stats_buffer))
            stats_buffer.clear()
        geno_fh.flush()
        stats_fh.flush()
        if buffered_variants:
            chunk_path = Path(f"{output_prefix}.variants.{chunk_index:06d}.npz")
            tmp_chunk = chunk_path.with_name(chunk_path.name + ".tmp")
            _write_variant_metadata(tmp_chunk, buffered_variants)
            tmp_chunk.replace(chunk_path)
            buffered_variants.clear()
            chunk_index += 1
        _atomic_write_progress()
        last_checkpoint_time = time.monotonic()
        bytes_since_checkpoint = 0

    # If we resumed cleanly, persist a fresh progress.json immediately so a
    # second crash before the first new record still leaves a valid state.
    if resume_count > 0:
        _atomic_write_progress()

    view_stderr_tmp = tempfile.TemporaryFile()
    try:
        query_stderr_tmp = tempfile.TemporaryFile()
    except BaseException:
        view_stderr_tmp.close()
        raise
    view_proc = subprocess.Popen(
        view_cmd,
        stdout=subprocess.PIPE,
        stderr=view_stderr_tmp,
        bufsize=1 << 20,
    )
    assert view_proc.stdout is not None
    proc = subprocess.Popen(
        query_cmd,
        stdin=view_proc.stdout,
        stdout=subprocess.PIPE,
        stderr=query_stderr_tmp,
        bufsize=1 << 20,
    )
    view_proc.stdout.close()
    assert proc.stdout is not None
    try:
        for raw_line in proc.stdout:
            line = raw_line[:-1] if raw_line.endswith(b"\n") else raw_line
            fields = line.split(b"\t", 10)
            if len(fields) != 11:
                continue
            alt_field = fields[4]
            if b"," in alt_field:
                continue

            col = _parse_gt_block_to_int8(fields[10], sample_count)
            geno_bytes = col.tobytes()
            geno_buffer.extend(geno_bytes)
            dosage_sum, dosage_sum_sq, n_observed, n_nonzero = _fast_int8_dosage_stats(col)
            stats_buffer.extend(stats_pack.pack(dosage_sum, dosage_sum_sq, n_observed, n_nonzero))

            chrom = fields[0].decode("utf-8")
            pos = int(fields[1])
            ref = fields[3].decode("utf-8")
            alt = alt_field.decode("utf-8")
            buffered_variants.append(
                _variant_defaults_from_bcftools_fields(
                    chrom=chrom,
                    pos=pos,
                    record_id_field=fields[2],
                    ref=ref,
                    alt=alt,
                    qual_field=fields[5],
                    svtype_field=fields[6],
                    svlen_field=fields[7],
                    af_field=fields[8],
                    end_field=fields[9],
                )
            )

            count += 1
            last_chrom = chrom
            last_pos = pos
            bytes_since_checkpoint += len(geno_bytes) + 24
            _checkpoint(force=False)

            now = time.monotonic()
            if now - last_log >= 10.0:
                rate = (count - resume_count) / max(now - t_start, 0.01)
                print(f"  [worker] {vcf_name}: {count} variants ({rate:.0f}/s)", file=sys.stderr, flush=True)
                last_log = now

        proc.wait()
        view_proc.wait()
        if proc.returncode != 0 or view_proc.returncode != 0:
            view_stderr_tmp.seek(0)
            query_stderr_tmp.seek(0)
            view_err = view_stderr_tmp.read().decode("utf-8", errors="replace").strip()
            query_err = query_stderr_tmp.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"bcftools pipeline failed on {vcf_name}"
                + (f" region {effective_region}" if effective_region else "")
                + f" (view exit {view_proc.returncode}, query exit {proc.returncode}). "
                + f"view stderr: {view_err}  query stderr: {query_err}"
            )
    except BaseException:
        # On any error path, persist whatever we have so the next launch can
        # resume from this point rather than redoing the entire region.
        try:
            _checkpoint(force=True)
        except Exception:
            pass
        for sub in (proc, view_proc):
            try:
                sub.kill()
            except OSError:
                pass
            sub.wait()
        raise
    finally:
        try:
            proc.stdout.close()
        except OSError:
            pass
        view_stderr_tmp.close()
        query_stderr_tmp.close()

    # Final flush + chunk write so all remaining records are durable. Wrap so
    # an exception in ``_checkpoint`` (e.g. disk full on the final write) can't
    # leak the genotype/stats binary file descriptors.
    try:
        _checkpoint(force=True)
    finally:
        try:
            geno_fh.close()
        except OSError:
            pass
        try:
            stats_fh.close()
        except OSError:
            pass

    # Consolidate all variant-metadata chunks into the single .variants.npz
    # file the merge step expects. Atomic rename so a crash here can't
    # leave a half-written final file.
    chunk_paths = _region_variant_chunk_paths(output_prefix)
    all_variants: list[_VariantDefaults] = []
    for chunk_path in chunk_paths:
        all_variants.extend(_load_variant_metadata(chunk_path))
    tmp_final = final_variants_path.with_name(final_variants_path.name + ".tmp")
    _write_variant_metadata(tmp_final, all_variants)
    tmp_final.replace(final_variants_path)
    for chunk_path in chunk_paths:
        chunk_path.unlink(missing_ok=True)
    progress_path.unlink(missing_ok=True)

    elapsed = time.monotonic() - t_start
    print(f"  [worker] {vcf_name}: DONE {count} variants in {elapsed:.0f}s", file=sys.stderr, flush=True)
    return count, output_prefix


def precache_vcfs_parallel(
    vcf_paths: list[Path],
    config: ModelConfig,
) -> None:
    """Parse and cache multiple VCFs in parallel, auto-detecting CPU count.

    Allocates workers across AND within chromosomes proportional to file size.
    Uses VCF header contig lengths for region splitting (no hardcoded lengths).
    If contig length is unknown, that VCF gets 1 worker (no splitting).

    Already-cached VCFs under the current cache key are skipped.
    Produces the same final cache format via the incremental → .npy pipeline.
    """
    import multiprocessing
    import os
    import shutil

    total_cpus = os.cpu_count() or 4

    # Audit existing caches and surface anything suspicious — but never touch
    # the bytes on disk. The load path runs a global (chr, pos, variant_id)
    # dedup that reconciles within-cache and cross-cache duplicates in memory,
    # so rewriting the cache here would burn ~5 min per chromosome (a 14 GB
    # rewrite per file) for no behavioral change.
    audited = 0
    suspicious = 0
    for vcf_path in vcf_paths:
        cache_paths = _vcf_cache_paths(vcf_path, config)
        if not _is_vcf_cache_bundle_complete(cache_paths):
            continue
        audited += 1
        reason = _vcf_cache_audit_reason(vcf_path, cache_paths)
        if reason is None:
            continue
        suspicious += 1
        log(
            f"  audit note for {vcf_path.name}: {reason} "
            "(cache left in place; load-time dedup will drop residual duplicates)"
        )
    log(f"  audit complete: {audited} caches checked, {suspicious} flagged (none rewritten)")

    # Skip already-cached VCFs under the current cache key only.
    uncached: list[Path] = []
    for vcf_path in vcf_paths:
        cache_paths = _vcf_cache_paths(vcf_path, config)
        if _is_vcf_cache_bundle_complete(cache_paths):
            continue
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

    # Allocate workers proportional to file size. With the bcftools backend
    # `view -r chr:start-end` jumps directly to the requested BGZF blocks via
    # the .tbi index, so per-record overhead is minimal — unlike cyvcf2's
    # region iterator. We oversplit ~3x relative to CPU count so the pool has
    # small tasks: better tail-latency, better load balancing, and a kill
    # mid-run only loses the records since the last region boundary instead
    # of the whole chromosome.
    allocation: dict[Path, int] = {}
    oversplit_factor = 3
    for vcf_path, (chrom, chrom_length, file_size) in vcf_info.items():
        share = file_size / max(total_size, 1)
        n_workers = max(1, round(share * total_cpus * oversplit_factor))
        # Can't split if we don't know the contig length
        if chrom_length <= 0:
            n_workers = 1
        allocation[vcf_path] = n_workers

    # Build task list — cache stores ALL samples; no keep_sample_indices filtering
    tasks: list[tuple[str, str | None, str, int]] = []
    completed_regions_by_vcf: dict[Path, int] = {}
    total_regions_by_vcf: dict[Path, int] = {}
    for vcf_path, n_workers in allocation.items():
        chrom, chrom_length, _ = vcf_info[vcf_path]
        cache_dir = _vcf_cache_dir(vcf_path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = _vcf_cache_key(vcf_path, config)
        tmp_dir = cache_dir / f"{key}.tmp_parallel"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        if n_workers <= 1 or chrom is None:
            # Single worker: parse entire VCF (no region filter)
            region_prefix = tmp_dir / "region_0"
            total_regions_by_vcf[vcf_path] = 1
            if _region_output_complete(region_prefix):
                completed_regions_by_vcf[vcf_path] = 1
            else:
                completed_regions_by_vcf[vcf_path] = 0
                tasks.append((str(vcf_path), None, str(region_prefix), 1))
        else:
            regions = _split_into_regions(chrom, chrom_length, n_workers)
            total_regions_by_vcf[vcf_path] = len(regions)
            completed_count = 0
            for i, region in enumerate(regions):
                region_prefix = tmp_dir / f"region_{i}"
                if _region_output_complete(region_prefix):
                    completed_count += 1
                    continue
                tasks.append((str(vcf_path), region, str(region_prefix), 1))
            completed_regions_by_vcf[vcf_path] = completed_count

    # Reserve a couple of cores for the concurrent PLINK int8/stats warmup
    # (Step 3.25 background pass). Without this, observed wall-time on the
    # AoU run is dominated by the warmup starving — it stays at 0 bytes for
    # several minutes while 12/12 cores parse VCFs. The two reserved cores
    # let the warmup actually decode int8 batches in parallel, and the VCF
    # precache loses very little (already oversplit 3x relative to CPUs).
    _RESERVED_FOR_BG = 2 if total_cpus >= 6 else 0
    effective_cpus = max(total_cpus - _RESERVED_FOR_BG, 1)
    process_count = min(effective_cpus, max(len(tasks), 1))
    threads_per_worker = max(effective_cpus // max(process_count, 1), 1)
    if threads_per_worker > 1:
        tasks = [
            (vcf_path_str, region, output_prefix, threads_per_worker)
            for vcf_path_str, region, output_prefix, _ in tasks
        ]

    log(
        f"parallel VCF precache: {len(uncached)} VCFs, {len(tasks)} pending tasks, "
        + f"{total_cpus} CPUs, {process_count} workers x {threads_per_worker} reader threads"
    )
    for vcf_path, n_workers in allocation.items():
        chrom, chrom_length, file_size = vcf_info[vcf_path]
        completed_count = completed_regions_by_vcf.get(vcf_path, 0)
        total_regions = total_regions_by_vcf.get(vcf_path, 0)
        log(
            f"  {vcf_path.name}: {n_workers} regions, {chrom}:{chrom_length}, {file_size/1e9:.1f} GB"
            + f"  completed={completed_count}/{total_regions}"
        )

    # Parse all regions in parallel.
    # Must use "spawn" (not "fork"): JAX is already initialized in the parent
    # (cupy/jax import happens at module load), and os.fork() on a multithreaded
    # process deadlocks workers — they exit cleanly enough to drop their result
    # but hang the pool's terminate/join on shutdown. Seen on AoU: 35+ minutes
    # in pool.__exit__ → process.join() → popen_fork.poll() at 0% CPU after
    # the chr19 worker emitted its last batch.
    if tasks:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=process_count) as pool:
            for count, prefix in pool.imap_unordered(_region_parse_worker, tasks):
                log(f"  region done: {Path(prefix).name} ({count} variants)")
    else:
        log("  no region parsing needed; resuming from completed temporary region cache")

    # Merge region results per VCF → incremental cache → final .npy cache
    for vcf_path in uncached:
        cache_dir = _vcf_cache_dir(vcf_path)
        key = _vcf_cache_key(vcf_path, config)
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
        inc_stats = cache_dir / f"{key}.inc.stats.bin"
        region_variant_paths = [
            Path(f"{str(geno_file).removesuffix('.geno')}.variants.npz")
            for geno_file in geno_files
        ]
        n_total = 0
        with open(inc_geno, "wb") as gout, open(inc_stats, "wb") as sout:
            for geno_file in geno_files:
                prefix = str(geno_file).removesuffix(".geno")
                with open(f"{prefix}.geno", "rb") as f:
                    shutil.copyfileobj(f, gout)
                with open(f"{prefix}.stats", "rb") as f:
                    data = f.read()
                    sout.write(data)
                    n_total += len(data) // 24

        # Finalize: convert incremental binary directly to .npy cache.
        # Do NOT call _load_vcf_with_cache — that would re-open the VCF and
        # waste 20+ min skipping already-parsed variants.
        actual_n_keep = len(_read_vcf_sample_ids(vcf_path))
        log(f"  finalizing {vcf_path.name}: {n_total} variants from {len(geno_files)} regions")

        # Load incremental binary via memmap (zero copy)
        inc_matrix = np.memmap(inc_geno, dtype=np.int8, mode="r", shape=(n_total, actual_n_keep)).T

        inc_variants: list[_VariantDefaults] = []
        for region_variant_path in region_variant_paths:
            inc_variants.extend(_load_variant_metadata(region_variant_path))

        # The .stats file (n_total rows) and the consolidated variant metadata
        # must describe the same records. A mismatch means a region worker
        # left .geno/.stats out of sync with its variants.npz — write nothing
        # rather than persist a cache the loader can't trust.
        if len(inc_variants) != n_total:
            raise RuntimeError(
                f"region cache merge inconsistency for {vcf_path.name}: "
                f".stats has {n_total} rows but variant metadata has "
                f"{len(inc_variants)} rows. Delete "
                f"{cache_dir / f'{key}.tmp_parallel'} and re-run."
            )

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
        scales = np.where(scales < config.minimum_scale, 1.0, scales)
        inc_stats_obj = VariantStatistics(means=means, scales=scales, allele_frequencies=afs, support_counts=support_arr)

        # Save as final .npy cache
        cache_save_ok = _save_vcf_to_cache(vcf_path, inc_matrix, inc_variants, inc_stats_obj, config=config)
        del inc_matrix

        if cache_save_ok:
            # Only clean up intermediates if the final bundle actually landed.
            # If it failed (e.g. ENOSPC), keep region scratch so the next run
            # can resume instead of starting over.
            for p in (inc_geno, inc_stats, *region_variant_paths):
                p.unlink(missing_ok=True)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            log(f"  {vcf_path.name}: cached")
        else:
            log(
                f"  {vcf_path.name}: cache save FAILED — region scratch kept at "
                f"{tmp_dir.name} so the next run can resume. Free disk and retry."
            )
def _load_vcf_with_cache(
    vcf_path: Path,
    config: ModelConfig,
    *,
    mmap_mode: Literal["r", "r+", "w+", "c"] | None,
) -> tuple[I8Array, list[_VariantDefaults], VariantStatistics]:
    effective_mmap_mode: Literal["r", "r+", "w+", "c"] = "r" if mmap_mode is None else mmap_mode
    # Check for completed cache first (full genotype matrix, all samples)
    cached = _load_vcf_from_cache(
        vcf_path,
        config=config,
        mmap_mode=effective_mmap_mode,
    )
    if cached is not None:
        return cached

    # Parse with incremental checkpointing (all samples)
    cache_dir = _vcf_cache_dir(vcf_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _vcf_cache_key(vcf_path, config)
    genotype_matrix, variants, variant_stats = _load_vcf_incremental(
        vcf_path, None, cache_dir, key, config=config, mmap_mode=effective_mmap_mode,
    )
    return genotype_matrix, variants, variant_stats


def _load_vcf_incremental(
    vcf_path: Path,
    keep_sample_indices: NDArray | None,
    cache_dir: Path,
    key: str,
    *,
    config: ModelConfig,
    mmap_mode: Literal["r", "r+", "w+", "c"],
) -> tuple[I8Array, list[_VariantDefaults], VariantStatistics]:
    """Parse VCF with incremental checkpointing. Resumes from last checkpoint if interrupted."""
    import os
    import struct
    import time

    reader = _open_vcf_reader(vcf_path)
    n_all_samples = len(reader.samples)
    n_keep = len(keep_sample_indices) if keep_sample_indices is not None else n_all_samples

    # Incremental files
    geno_bin = cache_dir / f"{key}.inc.genotypes.bin"
    stats_bin = cache_dir / f"{key}.inc.stats.bin"  # 4 int32/int64 values per variant
    progress_file = cache_dir / f"{key}.inc.progress.json"

    # Check for existing progress
    n_cached = 0
    metadata_chunk_count = 0
    if progress_file.exists():
        try:
            prog = json.loads(progress_file.read_text(encoding="utf-8"))
            n_cached = int(prog["n_variants"])
            metadata_chunk_count = int(prog.get("metadata_chunk_count", 0))
            expected_geno_bytes = n_cached * n_keep
            expected_stats_bytes = n_cached * 24  # struct "<qqii" = 24 bytes
            if geno_bin.exists() and geno_bin.stat().st_size > expected_geno_bytes:
                with open(geno_bin, "r+b") as f:
                    f.truncate(expected_geno_bytes)
            if stats_bin.exists() and stats_bin.stat().st_size > expected_stats_bytes:
                with open(stats_bin, "r+b") as f:
                    f.truncate(expected_stats_bytes)
            chunk_paths = _iter_incremental_variant_chunk_paths(cache_dir, key)
            if len(chunk_paths) < metadata_chunk_count:
                raise ValueError("missing variant metadata chunks for incremental resume")
            for chunk_path in chunk_paths[metadata_chunk_count:]:
                chunk_path.unlink(missing_ok=True)
            log(f"  incremental cache: resuming from {n_cached} variants ({metadata_chunk_count} metadata chunks)")
        except (OSError, ValueError, EOFError) as exc:
            log(f"  incremental cache progress corrupt ({exc}), starting fresh")
            n_cached = 0
            metadata_chunk_count = 0
            for p in (geno_bin, stats_bin, progress_file, *_iter_incremental_variant_chunk_paths(cache_dir, key)):
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass
    else:
        for p in (geno_bin, stats_bin, progress_file, *_iter_incremental_variant_chunk_paths(cache_dir, key)):
            p.unlink(missing_ok=True)

    # Reuse the reader opened above for sample IDs
    n_threads = os.cpu_count() or 4
    reader.set_threads(n_threads)
    log(f"  VCF decompression threads: {n_threads}")

    _GT_TO_INT8 = np.array([0, 1, PLINK_MISSING_INT8, 2], dtype=np.int8)
    _gt_to_i8 = _GT_TO_INT8
    keep_selector = _prepare_keep_sample_selector(keep_sample_indices, len(reader.samples))

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

    t_start = time.monotonic()
    last_log_time = t_start
    last_chrom = None
    variant_index = 0
    bytes_since_checkpoint = 0
    metadata_chunk_index = metadata_chunk_count
    buffered_variants: list[_VariantDefaults] = []
    last_record_coordinates: tuple[str, int] | None = None
    _monotonic = time.monotonic
    checkpoint_started_at = t_start

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
        checkpoint_started_at = t_start

    def _clear_incremental_artifacts() -> None:
        for path in (geno_bin, stats_bin, progress_file, *_iter_incremental_variant_chunk_paths(cache_dir, key)):
            path.unlink(missing_ok=True)

    def _abort_incremental_load(message: str) -> None:
        geno_fh.close()
        stats_fh.close()
        reader.close()
        _clear_incremental_artifacts()
        raise ValueError(message)

    def _flush_incremental_checkpoint(*, force: bool = False) -> None:
        nonlocal buffered_variants
        nonlocal bytes_since_checkpoint
        nonlocal checkpoint_started_at
        nonlocal metadata_chunk_index
        if not buffered_variants and not force:
            return
        elapsed = _monotonic() - checkpoint_started_at
        if not force and bytes_since_checkpoint < _INCREMENTAL_CHECKPOINT_BYTE_INTERVAL and elapsed < _INCREMENTAL_CHECKPOINT_TIME_SECONDS:
            return
        geno_fh.flush()
        stats_fh.flush()
        if buffered_variants:
            _write_variant_metadata(
                _incremental_variant_chunk_path(cache_dir, key, metadata_chunk_index),
                buffered_variants,
            )
            metadata_chunk_index += 1
            buffered_variants = []
        resume_chrom, resume_pos = last_record_coordinates if last_record_coordinates is not None else (None, 0)
        _atomic_write_text(
            progress_file,
            json.dumps(
                {
                    "n_variants": variant_index,
                    "n_samples": n_keep,
                    "resume_chrom": resume_chrom,
                    "resume_pos": resume_pos,
                    "metadata_chunk_count": metadata_chunk_index,
                }
            ),
        )
        bytes_since_checkpoint = 0
        checkpoint_started_at = _monotonic()

    # Parse remaining variants. Wrap so that an unexpected exception during
    # reader iteration (e.g. cyvcf2 hits a malformed record, decode raises,
    # or a KeyboardInterrupt during a multi-hour parse) cannot leak the
    # genotype/stats binary fds and the VCF reader. ``_abort_incremental_load``
    # already closes these for one specific control path; this finally block
    # is the general-case safety net.
    parse_succeeded = False
    try:
        for record in reader:
            if len(record.ALT) != 1:
                _abort_incremental_load(
                    "Only biallelic VCF records are supported. Normalize multiallelic records before loading: "
                    + _vcf_variant_key(record)
                )

            int8_col = _record_gt_types_to_int8(record.gt_types, _gt_to_i8, keep_selector)

            # Write genotype column to disk immediately
            geno_fh.write(int8_col.tobytes())

            dosage_sum, dosage_sum_sq, n_valid, support = _fast_int8_dosage_stats(int8_col)
            stats_fh.write(struct.pack("<qqii", dosage_sum, dosage_sum_sq, n_valid, support))
            vd = _variant_defaults_from_vcf_record(record)
            buffered_variants.append(vd)
            last_record_coordinates = (str(record.CHROM), int(record.POS))

            variant_index += 1
            bytes_since_checkpoint += int(
                int8_col.nbytes
                + 24
                + len(vd.variant_id.encode("utf-8"))
                + len(vd.chromosome.encode("utf-8"))
                + 32
            )
            _flush_incremental_checkpoint()

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
        _flush_incremental_checkpoint(force=True)
        geno_fh.flush()
        stats_fh.flush()
        parse_succeeded = True
    finally:
        # If parse_succeeded is True we still need these closes; if False we
        # need them more (resource leak prevention). Either way close here so
        # the success path no longer needs the duplicate close trio below.
        if not parse_succeeded:
            for closer in (geno_fh.close, stats_fh.close, reader.close):
                try:
                    closer()
                except (OSError, ValueError):
                    pass
    geno_fh.close()
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

    variants: list[_VariantDefaults] = []
    metadata_chunk_paths = _iter_incremental_variant_chunk_paths(cache_dir, key)
    for metadata_chunk_path in metadata_chunk_paths:
        variants.extend(_load_variant_metadata(metadata_chunk_path))
    if len(variants) != n_total:
        _clear_incremental_artifacts()
        raise ValueError(f"incremental metadata mismatch: expected {n_total} variants, found {len(variants)}")

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
    scales_arr = np.where(scales_arr < config.minimum_scale, 1.0, scales_arr)

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
        genotype_matrix=incremental_matrix,
        variants=variants,
        variant_stats=variant_stats,
        config=config,
    )
    del incremental_matrix

    # Clean up incremental files after the final cache is complete.
    for p in (geno_bin, stats_bin, progress_file, *metadata_chunk_paths):
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass

    cached = _load_vcf_from_cache(
        vcf_path=vcf_path,
        config=config,
        mmap_mode=mmap_mode,
    )
    if cached is None:
        raise RuntimeError("VCF cache finalization failed.")
    genotype_matrix, _, _ = cached
    log(f"  incremental parse complete: {genotype_matrix.shape}  mem={mem()}")
    return genotype_matrix, variants, variant_stats


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically via per-process temp file + rename.

    Uses a per-process, per-call unique temp name so two concurrent writers
    racing on the same `path` cannot truncate or overwrite each other's
    staging files; the rename winner publishes a complete file. The loser
    leaves its temp behind on the failure path, which is cleaned below.
    """
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        tmp.write_text(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _save_vcf_to_cache(
    vcf_path: Path,
    genotype_matrix: I8Array,
    variants: list[_VariantDefaults],
    variant_stats: VariantStatistics,
    config: ModelConfig | None = None,
    *,
    cache_paths: _VcfCachePaths | None = None,
) -> bool:
    """Persist parsed VCF results to disk cache. Returns True on success, False on failure.

    Returning False instead of swallowing silently is load-bearing for the
    parallel precache path: the previous behavior masked an ENOSPC mid-run
    so the caller logged "cached" for a file that never actually hit disk,
    then the next chromosome's writable mmap turned the same disk-full
    state into SIGBUS and killed the process before any chromosome bundle
    finalized. With a real success signal the caller can fail loudly
    instead of corrupting the cache state.
    """
    if cache_paths is None:
        if config is None:
            raise ValueError("config is required when cache_paths is not provided.")
        paths = _vcf_cache_paths(vcf_path, config)
    else:
        paths = cache_paths

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
        genotype_memmap: np.memmap[Any, np.dtype[np.int8]] = _open_int8_memmap(
            geno_tmp,
            mode="w+",
            shape=(sample_count, variant_count),
            fortran_order=True,
        )
        for start_index in range(0, variant_count, copy_batch_size):
            stop_index = min(start_index + copy_batch_size, variant_count)
            genotype_memmap[:, start_index:stop_index] = genotype_matrix_i8[:, start_index:stop_index]
        genotype_memmap.flush()
        del genotype_memmap
        _write_variant_metadata(var_tmp, variants)
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
                    "source_signature": _vcf_cache_source_signature(vcf_path),
                }
            ),
        )
        geno_tmp.replace(paths.geno_path)
        var_tmp.replace(paths.var_path)
        stats_tmp.replace(paths.stats_path)
        manifest_tmp.replace(paths.manifest_path)
        bundle_dir.rmdir()
        total_mb = (
            paths.geno_path.stat().st_size
            + paths.var_path.stat().st_size
            + paths.stats_path.stat().st_size
            + paths.manifest_path.stat().st_size
        ) / 1e6
        log(f"VCF cache saved ({total_mb:.1f} MB) → {paths.cache_dir.name}/{paths.key}.*")
        return True
    except (OSError, ValueError) as exc:
        log(f"VCF cache save failed ({exc}), continuing without cache")
        _cleanup_stale_vcf_cache_temps(paths.cache_dir, paths.key)
        return False


_PLINK_METADATA_PROCESS_CACHE: dict[tuple[str, int, int, int, int, int, int], _PlinkMetadata] = {}


def _load_plink1_metadata(bed_path: Path) -> _PlinkMetadata:
    fam_path = bed_path.with_suffix(".fam")
    bim_path = bed_path.with_suffix(".bim")

    # In-process memoization: the AoU runner reaches this function 4x per run
    # (train load, test load, plus context resolves) for the same .bed file.
    # Each call rescans the 30 MB .fam and counts lines on the 72 MB .bim —
    # ~10s round-trip on the AoU workbench. Resolving the bed to a stable
    # (path, st_dev, st_ino, st_size, st_mtime_ns) tuple lets us replay the
    # parsed result and skip the work on every subsequent call in this
    # process. The cache is process-local (no disk persistence needed because
    # the on-disk .npy stats cache and the sample-table cache already cover
    # cross-run reuse).
    try:
        fam_st = fam_path.stat()
        bim_st = bim_path.stat()
        cache_key: tuple[str, int, int, int, int, int, int] | None = (
            str(bed_path.resolve()),
            int(fam_st.st_size),
            int(fam_st.st_mtime_ns),
            int(fam_st.st_ino),
            int(bim_st.st_size),
            int(bim_st.st_mtime_ns),
            int(bim_st.st_ino),
        )
    except OSError:
        cache_key = None
    if cache_key is not None and cache_key in _PLINK_METADATA_PROCESS_CACHE:
        cached = _PLINK_METADATA_PROCESS_CACHE[cache_key]
        log(
            f"  PLINK metadata cache hit (in-process): {len(cached.sample_ids)} "
            f"samples, {cached.variant_count} variants  mem={mem()}"
        )
        return cached

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
    metadata = _PlinkMetadata(
        sample_ids=sample_ids,
        variant_count=variant_count,
    )
    if cache_key is not None:
        _PLINK_METADATA_PROCESS_CACHE[cache_key] = metadata
    return metadata


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
    if variant_metadata_path is None:
        return [
            VariantRecord(
                variant_id=variant.variant_id,
                variant_class=variant.variant_class,
                chromosome=variant.chromosome,
                position=variant.position,
                length=variant.length,
                allele_frequency=variant.allele_frequency,
                quality=variant.quality,
            )
            for variant in default_variants
        ]

    metadata_rows_by_id: dict[str, dict[str, str]] = {}
    annotation_kinds: dict[str, str] = {}
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
    annotation_kinds = _infer_annotation_column_kinds(list(metadata_rows_by_id.values()))
    _log_annotation_column_kinds(table_spec.columns, annotation_kinds)

    records: list[VariantRecord] = []
    seen_variant_ids: set[str] = set()
    for variant in default_variants:
        if variant.variant_id in seen_variant_ids:
            raise ValueError("Duplicate variant identifier in genotype data: " + variant.variant_id)
        seen_variant_ids.add(variant.variant_id)
        metadata_row = metadata_rows_by_id.pop(variant.variant_id, None)
        records.append(_merge_variant_metadata(variant, metadata_row, annotation_kinds))

    if metadata_rows_by_id:
        extra_variant_ids = sorted(metadata_rows_by_id)
        raise ValueError(
            "Variant metadata contains identifiers that do not exist in genotype data: "
            + ", ".join(extra_variant_ids[:10])
        )
    return records


def _count_variant_record_classes(variant_records: Sequence[VariantRecord]) -> tuple[int, int]:
    snv_count = 0
    sv_count = 0
    for record in variant_records:
        variant_class_value = record.variant_class.value
        if variant_class_value == "snv":
            snv_count += 1
        elif variant_class_value != "small_indel":
            sv_count += 1
    return snv_count, sv_count


def _log_annotation_column_kinds(columns: Sequence[str], annotation_kinds: dict[str, str]) -> None:
    annotation_columns = [column_name for column_name in columns if column_name not in VARIANT_METADATA_BASE_COLUMNS]
    if not annotation_columns:
        log("variant metadata annotations: none")
        return
    log("variant metadata annotation column interpretations:")
    for column_name in annotation_columns:
        column_kind = annotation_kinds.get(column_name, "ignored_empty")
        log(f"  {column_name}: {column_kind}")


def _merge_variant_metadata(
    default_variant: _VariantDefaults,
    metadata_row: dict[str, str] | None,
    annotation_kinds: dict[str, str],
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
    prior_binary_features: dict[str, bool] = {}
    prior_continuous_features: dict[str, float] = {}
    prior_categorical_features: dict[str, str] = {}
    prior_membership_features: dict[str, dict[str, float]] = {}
    prior_nested_features: dict[str, tuple[str, ...]] = {}
    prior_nested_membership_features: dict[str, dict[str, float]] = {}
    for column_name, column_kind in annotation_kinds.items():
        column_value = metadata_row.get(column_name)
        parsed_value = _parse_string_feature_or_skip(column_value)
        if parsed_value is None:
            continue
        if column_kind == "binary":
            prior_binary_features[column_name] = _parse_bool_or_default(
                parsed_value,
                False,
                column_name=column_name,
            )
        elif column_kind == "continuous":
            prior_continuous_features[column_name] = _parse_float(parsed_value, column_name=column_name)
        elif column_kind == "membership":
            prior_membership_features[column_name] = _parse_weighted_levels(
                parsed_value,
                column_name=column_name,
            )
        elif column_kind == "nested":
            prior_nested_features[column_name] = _parse_nested_path(
                parsed_value,
                column_name=column_name,
            )
        elif column_kind == "nested_membership":
            prior_nested_membership_features[column_name] = _parse_weighted_nested_paths(
                parsed_value,
                column_name=column_name,
            )
        else:
            prior_categorical_features[column_name] = parsed_value
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
        prior_binary_features=prior_binary_features,
        prior_continuous_features=prior_continuous_features,
        prior_categorical_features=prior_categorical_features,
        prior_membership_features=prior_membership_features,
        prior_nested_features=prior_nested_features,
        prior_nested_membership_features=prior_nested_membership_features,
        prior_class_members=prior_class_members,
        prior_class_membership=prior_class_membership,
    )


def _inspect_delimited_table(path: str | Path) -> _DelimitedTableSpec:
    resolved_path = Path(path)
    with _open_text_file(resolved_path, "rt", newline="") as handle:
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
    with _open_text_file(table_spec.path, "rt", newline="") as handle:
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
) -> I32Array:
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
    source_indices: I32Array,
) -> tuple[_SampleTable, I32Array, bool]:
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
    # Each cyvcf2 property access crosses Python<->C, so pull every field we
    # need into a local exactly once before branching.
    chrom = str(record.CHROM)
    pos = int(record.POS)
    ref = record.REF
    alt = str(record.ALT[0])
    info = record.INFO
    record_id = record.ID
    is_snp = record.is_snp
    variant_id = (
        f"{chrom}:{pos}:{ref}:{alt}"
        if record_id is None or str(record_id) == "."
        else str(record_id)
    )
    svlen_value = info.get("SVLEN")
    if svlen_value is not None:
        if isinstance(svlen_value, (tuple, list)):
            length = float(abs(float(svlen_value[0])))
        else:
            length = float(abs(float(svlen_value)))
    elif is_snp:
        length = 1.0
    else:
        record_end = record.end
        if record_end is not None and int(record_end) >= pos:
            length = float(int(record_end) - pos + 1)
        else:
            length = float(max(len(ref), len(alt)))

    if is_snp:
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
        chromosome=chrom,
        position=pos,
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
            try:
                position_value = int(fields[3])
            except ValueError as exc:
                raise ValueError(
                    f"Malformed PLINK .bim row at line {line_number}: "
                    f"non-integer position {fields[3]!r}"
                ) from exc
            yield _PlinkBimRecord(
                chromosome=fields[0],
                variant_id=fields[1],
                position=position_value,
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


def _infer_annotation_column_kinds(rows: Sequence[dict[str, str]]) -> dict[str, str]:
    annotation_values: dict[str, list[str]] = {}
    for row in rows:
        for column_name, column_value in row.items():
            if column_name in VARIANT_METADATA_BASE_COLUMNS:
                continue
            parsed_value = _parse_string_feature_or_skip(column_value)
            if parsed_value is None:
                continue
            annotation_values.setdefault(column_name, []).append(parsed_value)

    annotation_kinds: dict[str, str] = {}
    for column_name, values in annotation_values.items():
        if all(_is_bool_text(value) for value in values):
            annotation_kinds[column_name] = "binary"
        elif all(_is_float_text(value) for value in values):
            annotation_kinds[column_name] = "continuous"
        elif all(_is_weighted_levels_text(value) for value in values):
            if all(_weighted_level_names_are_nested(value) for value in values):
                annotation_kinds[column_name] = "nested_membership"
            else:
                annotation_kinds[column_name] = "membership"
        elif all(NESTED_PATH_DELIMITER in value for value in values):
            annotation_kinds[column_name] = "nested"
        else:
            annotation_kinds[column_name] = "categorical"
    return annotation_kinds


def _parse_string_feature_or_skip(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    return value.strip()


def _is_bool_text(value: str) -> bool:
    return value.strip().lower() in {"1", "0", "true", "false", "yes", "no"}


def _is_float_text(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _is_weighted_levels_text(value: str) -> bool:
    assignments = [assignment.strip() for assignment in value.split(",") if assignment.strip()]
    if not assignments:
        return False
    for assignment in assignments:
        if "=" not in assignment:
            return False
        _, level_weight = assignment.split("=", 1)
        if not _is_float_text(level_weight.strip()):
            return False
    return True


def _weighted_level_names_are_nested(value: str) -> bool:
    assignments = [assignment.strip() for assignment in value.split(",") if assignment.strip()]
    return all(NESTED_PATH_DELIMITER in assignment.split("=", 1)[0] for assignment in assignments)


def _parse_weighted_levels(value: str, column_name: str) -> dict[str, float]:
    weighted_levels: dict[str, float] = {}
    for level_assignment in value.split(","):
        assignment_value = level_assignment.strip()
        if not assignment_value:
            continue
        if "=" not in assignment_value:
            raise ValueError("Could not parse weighted levels for " + column_name + ": " + value)
        level_name, level_weight = assignment_value.split("=", 1)
        weighted_levels[level_name.strip()] = _parse_float(level_weight.strip(), column_name=column_name)
    return weighted_levels


def _parse_nested_path(value: str, column_name: str) -> tuple[str, ...]:
    path_parts = tuple(path_part.strip() for path_part in value.split(NESTED_PATH_DELIMITER))
    if not path_parts or any(not path_part for path_part in path_parts):
        raise ValueError("Could not parse nested path for " + column_name + ": " + value)
    return path_parts


def _parse_weighted_nested_paths(value: str, column_name: str) -> dict[str, float]:
    weighted_paths = _parse_weighted_levels(value, column_name=column_name)
    normalized_paths: dict[str, float] = {}
    for path_name, path_weight in weighted_paths.items():
        path_parts = _parse_nested_path(path_name, column_name=column_name)
        normalized_paths[NESTED_PATH_DELIMITER.join(path_parts)] = path_weight
    return normalized_paths


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

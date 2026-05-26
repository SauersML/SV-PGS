"""SV VCF → PLINK 1.9 BED transcoder.

One-time, per-cohort utility that ingests structural-variant VCFs
(typically one per chromosome, as produced by the AoU SV release) and
emits a single PLINK 1.9 BED trio (.bed/.bim/.fam). The output is the
input format expected by the bitpacked GPU pipeline described in
``BITPACKED_SPEC.md``.

This module is intentionally self-contained: it only borrows the
low-level writer / encoder primitives from :mod:`sv_pgs.plink` and the
:class:`sv_pgs.data.VariantRecord` dataclass. It does NOT touch any of
the heavy ``sv_pgs.io`` parsing machinery — the SV ingest path is
simple enough that a small re-implementation is cheaper than reusing
the incremental cache plumbing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from sv_pgs.data import VariantRecord
from sv_pgs.gcsfuse_staging import is_gcsfuse_path
from sv_pgs.plink import (
    PLINK1_MAGIC,
    PLINK_MISSING_INT8,
    _bytes_per_variant,
    _encode_variant,
    to_bed,
)

_LOG = logging.getLogger(__name__)

# Match ``sv_pgs/io.py:SV_LENGTH_THRESHOLD``.
SV_LENGTH_THRESHOLD: float = 1_000.0

# Above this estimated buffered-matrix size, switch to streaming-write.
_STREAMING_THRESHOLD_BYTES: int = 8 * 1024 * 1024 * 1024  # 8 GiB

# cyvcf2 GT type codes (count_a1 / "count alt" semantics):
#   0 = 0/0 → dosage 2 (homozygous ref / A1)
#   1 = 0/1 → dosage 1
#   2 = ./. → missing
#   3 = 1/1 → dosage 0
# We map straight to int8 with PLINK_MISSING_INT8 = -127 for missing, to
# match ``sv_pgs.plink._DECODE_LOOKUP_A1`` (used by ``_encode_variant``).
_GT_TYPE_TO_INT8: np.ndarray = np.array(
    [2, 1, int(PLINK_MISSING_INT8), 0], dtype=np.int8
)


def _normalize_variant_token(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    if text.startswith("<") and text.endswith(">"):
        text = text[1:-1]
    return text


def _classify_sv_token(token: str | None, length: float) -> str:
    """Replicate ``_structural_variant_class_from_token`` from sv_pgs/io.py.

    Returns the SVTYPE label as a plain string (the bitpacked pipeline
    only cares about the label; full ``VariantClass`` enum membership is
    not needed at the transcoder layer).
    """
    if token is None:
        return "OTHER_COMPLEX_SV"
    if "DEL" in token:
        return "DELETION_LONG" if length >= SV_LENGTH_THRESHOLD else "DELETION_SHORT"
    if "DUP" in token or "CNV" in token:
        return (
            "DUPLICATION_LONG" if length >= SV_LENGTH_THRESHOLD else "DUPLICATION_SHORT"
        )
    if "INS" in token or "ME" in token:
        return "INSERTION_MEI"
    if "INV" in token or "BND" in token:
        return "INVERSION_BND_COMPLEX"
    if "STR" in token or "VNTR" in token or "REPEAT" in token:
        return "STR_VNTR_REPEAT"
    return "OTHER_COMPLEX_SV"


def _import_cyvcf2() -> Any:
    try:
        import cyvcf2  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "sv_transcoder requires cyvcf2 to parse SV VCFs. "
            "Install it with `pip install cyvcf2`."
        ) from exc
    return cyvcf2


def _resolve_sample_layout(
    vcf_samples: list[str],
    sample_ids: list[str] | None,
    sample_id_order: list[str] | None,
) -> tuple[list[str], np.ndarray]:
    """Return (final_sample_ids, gather_index_into_vcf_samples)."""
    vcf_index: dict[str, int] = {sid: idx for idx, sid in enumerate(vcf_samples)}

    if sample_ids is None and sample_id_order is None:
        return list(vcf_samples), np.arange(len(vcf_samples), dtype=np.intp)

    if sample_id_order is not None:
        # The explicit final ordering wins. If sample_ids was also passed,
        # we still trust sample_id_order — it is, by spec, the final order.
        requested = list(sample_id_order)
        if sample_ids is not None:
            allowed = set(sample_ids)
            missing = [sid for sid in requested if sid not in allowed]
            if missing:
                raise ValueError(
                    f"sample_id_order references {len(missing)} ids not in sample_ids "
                    f"(first 3: {missing[:3]})"
                )
    else:
        # sample_ids is set, sample_id_order is not — preserve VCF order
        # but restrict to the listed ids.
        requested_set = set(sample_ids or [])
        requested = [sid for sid in vcf_samples if sid in requested_set]

    missing_in_vcf = [sid for sid in requested if sid not in vcf_index]
    if missing_in_vcf:
        raise ValueError(
            f"{len(missing_in_vcf)} requested sample ids not found in VCF "
            f"(first 3: {missing_in_vcf[:3]})"
        )
    gather = np.fromiter(
        (vcf_index[sid] for sid in requested), dtype=np.intp, count=len(requested)
    )
    return requested, gather


def _variant_int8_from_record(record: Any, gather: np.ndarray) -> np.ndarray:
    """Extract per-sample int8 dosage for one cyvcf2 record."""
    gt_types = np.asarray(record.gt_types, dtype=np.int64)
    # cyvcf2 may emit values outside [0, 3] in unusual ploidy cases;
    # fold anything unexpected to "missing".
    safe = np.where((gt_types >= 0) & (gt_types <= 3), gt_types, 2)
    dosage = _GT_TYPE_TO_INT8[safe]
    if gather.size != gt_types.size or not np.array_equal(
        gather, np.arange(gt_types.size, dtype=np.intp)
    ):
        dosage = dosage[gather]
    return dosage


def _variant_record_from_cyvcf2(
    record: Any, positional_index: int
) -> tuple[VariantRecord, str, str, str]:
    """Return (VariantRecord, sid, ref_allele, alt_allele, svtype)."""
    chrom = str(record.CHROM)
    pos = int(record.POS)
    record_id = getattr(record, "ID", None)
    sid = str(record_id) if record_id not in (None, ".") else f"{chrom}:{pos}:{positional_index}"
    ref = str(record.REF) if record.REF is not None else "N"
    alt_seq = record.ALT if record.ALT else ()
    alt = str(alt_seq[0]) if alt_seq else "N"

    info = record.INFO
    try:
        svtype_raw = info.get("SVTYPE") if hasattr(info, "get") else info["SVTYPE"]
    except (KeyError, TypeError):
        svtype_raw = None
    svtype_token = _normalize_variant_token(svtype_raw)
    if svtype_token is None:
        svtype_token = _normalize_variant_token(alt)

    # Length computation (mirror sv_pgs/io.py:4380-4393)
    length: float = 1.0
    try:
        svlen_raw = info.get("SVLEN") if hasattr(info, "get") else info["SVLEN"]
    except (KeyError, TypeError):
        svlen_raw = None
    if svlen_raw is not None:
        if isinstance(svlen_raw, (tuple, list)):
            length = float(abs(float(svlen_raw[0])))
        else:
            length = float(abs(float(svlen_raw)))
    else:
        record_end = getattr(record, "end", None)
        if record_end is not None and int(record_end) >= pos:
            length = float(int(record_end) - pos + 1)
        else:
            length = float(max(len(ref), len(alt)))

    svtype_label = _classify_sv_token(svtype_token, length)

    try:
        from sv_pgs.data import VariantClass  # local import; avoids cycles

        variant_class = getattr(VariantClass, svtype_label, None)
        if variant_class is None:
            variant_class = VariantClass.OTHER_COMPLEX_SV
    except Exception:  # pragma: no cover — VariantClass should always import
        variant_class = None  # type: ignore[assignment]

    variant_record = VariantRecord(
        variant_id=sid,
        variant_class=variant_class,  # type: ignore[arg-type]
        chromosome=chrom,
        position=pos,
        length=length,
    )
    return variant_record, sid, ref, alt, svtype_label


def transcode_sv_vcf_to_bed(
    vcf_paths: list[Path],
    bed_out_path: Path,
    sample_ids: list[str] | None = None,
    sample_id_order: list[str] | None = None,
) -> dict[str, Any]:
    """Transcode one or more SV VCFs to a single PLINK 1.9 BED trio.

    Args:
        vcf_paths: One VCF per chromosome (or any partition; concatenated
            variant-major into the output BED in the order given).
        bed_out_path: Output ``.bed`` path. Sibling ``.bim`` and ``.fam``
            are written next to it by :func:`sv_pgs.plink.to_bed` (or by
            the streaming path below).
        sample_ids: Restrict to these ids (subset of VCF samples).
        sample_id_order: Final ordering of rows in the ``.fam``.

    Returns:
        ``{"n_variants", "n_samples", "variant_records",
           "svtype_per_variant"}``.
    """
    if not vcf_paths:
        raise ValueError("transcode_sv_vcf_to_bed requires at least one VCF path.")

    for _vcf_path in vcf_paths:
        if is_gcsfuse_path(Path(_vcf_path)):
            _LOG.warning(
                "transcode_sv_vcf_to_bed: VCF %s is on a gcsfuse mount; "
                "reading directly from gcsfuse is VERY slow (every read is an "
                "HTTP GET). SV VCFs can be tens of GB — consider calling "
                "sv_pgs.gcsfuse_staging.stage_to_local first to copy it to "
                "local NVMe before transcoding.",
                _vcf_path,
            )

    cyvcf2 = _import_cyvcf2()
    bed_out_path = Path(bed_out_path)
    if is_gcsfuse_path(bed_out_path.parent):
        raise ValueError(
            f"transcode_sv_vcf_to_bed: refusing to write BED output into a "
            f"gcsfuse-mounted directory ({bed_out_path.parent}). Writes to "
            f"gcsfuse are slow and unsafe for large BED trios — point "
            f"bed_out_path at local disk and stage afterwards if needed."
        )
    bed_out_path.parent.mkdir(parents=True, exist_ok=True)

    # Sample layout is determined from the first VCF. All inputs must share
    # the same sample set (typical for one-VCF-per-chromosome partitioning).
    first_reader = cyvcf2.VCF(str(vcf_paths[0]))
    try:
        first_samples = [str(s) for s in first_reader.samples]
    finally:
        first_reader.close()
    final_sample_ids, gather = _resolve_sample_layout(
        first_samples, sample_ids, sample_id_order
    )
    n_samples = len(final_sample_ids)
    if n_samples == 0:
        raise ValueError("transcode_sv_vcf_to_bed: 0 samples after restriction.")

    # Estimate buffered size: float32 matrix shape (n_samples, n_variants).
    # We do not know n_variants yet; probe the first VCF for a hint, and
    # otherwise default to "stream" for safety on large cohorts.
    variant_count_hint = 0
    for path in vcf_paths:
        reader = cyvcf2.VCF(str(path))
        try:
            for attr in ("num_records", "nrecords"):
                value = getattr(reader, attr, None)
                if callable(value):
                    try:
                        value = value()
                    except (TypeError, ValueError):
                        value = None
                if isinstance(value, (int, np.integer)) and value > 0:
                    variant_count_hint += int(value)
                    break
        finally:
            reader.close()

    use_streaming = (
        variant_count_hint == 0
        or n_samples * variant_count_hint * 4 > _STREAMING_THRESHOLD_BYTES
    )

    variant_records: list[VariantRecord] = []
    svtype_per_variant: list[str] = []
    sids: list[str] = []
    chromosomes: list[str] = []
    positions: list[int] = []
    ref_alleles: list[str] = []
    alt_alleles: list[str] = []

    if use_streaming:
        bytes_per_variant = _bytes_per_variant(n_samples)
        positional_index = 0
        with bed_out_path.open("wb") as bed_handle:
            bed_handle.write(PLINK1_MAGIC)
            for vcf_path in vcf_paths:
                reader = cyvcf2.VCF(str(vcf_path))
                try:
                    # Sanity-check that downstream VCFs share the sample list.
                    other_samples = [str(s) for s in reader.samples]
                    if other_samples != first_samples:
                        raise ValueError(
                            f"VCF {vcf_path} samples do not match the first VCF "
                            f"({len(other_samples)} vs {len(first_samples)})."
                        )
                    for record in reader:
                        dosage_i8 = _variant_int8_from_record(record, gather)
                        column = dosage_i8.astype(np.float32)
                        column[dosage_i8 == PLINK_MISSING_INT8] = np.nan
                        bed_handle.write(
                            _encode_variant(column, bytes_per_variant=bytes_per_variant)
                        )
                        vr, sid, ref, alt, svtype_label = _variant_record_from_cyvcf2(
                            record, positional_index
                        )
                        variant_records.append(vr)
                        svtype_per_variant.append(svtype_label)
                        sids.append(sid)
                        chromosomes.append(vr.chromosome)
                        positions.append(vr.position)
                        ref_alleles.append(ref)
                        alt_alleles.append(alt)
                        positional_index += 1
                finally:
                    reader.close()

        # Write .fam and .bim explicitly (streaming path bypassed to_bed).
        fam_path = bed_out_path.with_suffix(".fam")
        fam_path.write_text(
            "".join(
                f"{sid} {sid} 0 0 0 -9\n" for sid in final_sample_ids
            ),
            encoding="utf-8",
        )
        bim_path = bed_out_path.with_suffix(".bim")
        bim_path.write_text(
            "".join(
                f"{chrom} {sid} 0 {pos} {a1} {a2}\n"
                for chrom, sid, pos, a1, a2 in zip(
                    chromosomes, sids, positions, ref_alleles, alt_alleles, strict=True
                )
            ),
            encoding="utf-8",
        )
    else:
        # Buffered path: collect all variants into one float32 matrix, then
        # hand off to sv_pgs.plink.to_bed (which writes .bed/.bim/.fam).
        columns: list[np.ndarray] = []
        positional_index = 0
        for vcf_path in vcf_paths:
            reader = cyvcf2.VCF(str(vcf_path))
            try:
                other_samples = [str(s) for s in reader.samples]
                if other_samples != first_samples:
                    raise ValueError(
                        f"VCF {vcf_path} samples do not match the first VCF "
                        f"({len(other_samples)} vs {len(first_samples)})."
                    )
                for record in reader:
                    dosage_i8 = _variant_int8_from_record(record, gather)
                    column = dosage_i8.astype(np.float32)
                    column[dosage_i8 == PLINK_MISSING_INT8] = np.nan
                    columns.append(column)
                    vr, sid, ref, alt, svtype_label = _variant_record_from_cyvcf2(
                        record, positional_index
                    )
                    variant_records.append(vr)
                    svtype_per_variant.append(svtype_label)
                    sids.append(sid)
                    chromosomes.append(vr.chromosome)
                    positions.append(vr.position)
                    ref_alleles.append(ref)
                    alt_alleles.append(alt)
                    positional_index += 1
            finally:
                reader.close()

        if not columns:
            matrix = np.empty((n_samples, 0), dtype=np.float32)
        else:
            matrix = np.stack(columns, axis=1)

        properties: dict[str, list[Any]] = {
            "iid": list(final_sample_ids),
            "fid": list(final_sample_ids),
            "sid": sids,
            "chromosome": chromosomes,
            "bp_position": positions,
            "allele_1": ref_alleles,
            "allele_2": alt_alleles,
        }
        to_bed(bed_out_path, matrix, properties)

    return {
        "n_variants": len(variant_records),
        "n_samples": n_samples,
        "variant_records": variant_records,
        "svtype_per_variant": np.asarray(svtype_per_variant, dtype=object),
    }

"""GATK-SV call sets as a genotype source for the dosage store.

Reads a GATK-SV VCF (the AoU srWGS SV release, the public 1kGP freeze) record
by record and returns, per kept record, one small-integer value per sample
plus a no-call mask:

- **Copy-number records** (``SVTYPE=CNV`` or FILTER ``MULTIALLELIC``) carry the
  integer copy number from FORMAT/CN, falling back per sample to FORMAT/RD_CN.
  Their GT is not a genotype: AoU writes ``.``, the 1kGP freeze a placeholder
  ``0/1``. They get class ``COPY_NUMBER``.
- **Every other record** carries its ALT allele count 0/1/2 from GT. A call with
  any missing allele (``./.``, ``./1``) is a no-call.

FILTER policy: a record is kept only when all its FILTER values are PASS (or the
field is ``.``), with ``MULTIALLELIC`` also allowed on copy-number records.
Every UNRESOLVED breakend, HIGH_NCR, LIKELY_REFERENCE_ARTIFACT or
VARIABLE_ACROSS_BATCHES record is dropped. Breakends are dropped even when they
pass, and so is any other multi-ALT record. So is a copy-number record with a
copy number past the store's uint8 range: a satellite-scale array such as the
1kGP chr16 pericentromeric HGSV_208635, whose median copy number is 241 and
whose maximum is 652. Every drop is counted by reason.

No-calls are kept as a mask, never filled here. AoU's genotype filter turns
uncertain carriers into no-calls, so missingness depends on the genotype; the
store rows fill them inside the measurement model, from the imputed DS where
it has the same SV (``gatksv_store_rows``).
Samples keep this call set's own IDs (research IDs in AoU); joining them to
the store's samples is the crosswalk's job, never a join by name.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from cyvcf2 import VCF

from sv_pgs._typing import BoolArray, F64Array, I64Array, NDArray, U8Array
from sv_pgs.config import VariantClass
from sv_pgs.dosage_store import MAXIMUM_CODE
from sv_pgs.variant_typing import variant_class_and_length

PASSING_FILTER = "PASS"
COPY_NUMBER_FILTER = "MULTIALLELIC"


@dataclass(frozen=True, slots=True)
class GatksvBlock:
    """Kept records of one read, columnar, in file order.

    ``values[r, i]`` is the ALT allele count (0..2) or, where
    ``is_copy_number[r]``, the integer copy number (0..254) of sample ``i``;
    it is 0 wherever ``no_call[r, i]``. ``svtypes`` is the token each record
    was typed with: its SVTYPE, ``CNV`` for every copy-number record.
    """

    chromosomes: tuple[str, ...]
    positions: I64Array
    ends: I64Array
    lengths: F64Array
    variant_ids: tuple[str, ...]
    svtypes: tuple[str, ...]
    variant_classes: tuple[VariantClass, ...]
    is_copy_number: BoolArray
    values: U8Array
    no_call: BoolArray

    @property
    def record_count(self) -> int:
        return int(self.positions.shape[0])

    def aligned_to_store_samples(self, source_columns: I64Array) -> GatksvBlock:
        """This block with one column per store sample, in store order.

        ``source_columns`` comes from ``sample_crosswalk.source_columns_for_store_samples``;
        a store sample the call set does not carry (column -1) is a no-call.
        """
        present = source_columns >= 0
        values = np.zeros((self.record_count, source_columns.shape[0]), dtype=np.uint8)
        no_call = np.ones((self.record_count, source_columns.shape[0]), dtype=bool)
        values[:, present] = self.values[:, source_columns[present]]
        no_call[:, present] = self.no_call[:, source_columns[present]]
        return GatksvBlock(
            chromosomes=self.chromosomes,
            positions=self.positions,
            ends=self.ends,
            lengths=self.lengths,
            variant_ids=self.variant_ids,
            svtypes=self.svtypes,
            variant_classes=self.variant_classes,
            is_copy_number=self.is_copy_number,
            values=values,
            no_call=no_call,
        )


def _filter_values(record: Any) -> tuple[str, ...]:
    # cyvcf2 reports both PASS and "." as None.
    raw_filter = record.FILTER
    return (PASSING_FILTER,) if raw_filter is None else tuple(str(raw_filter).split(";"))


def _info_integer(record: Any, key: str) -> int | None:
    value = record.INFO.get(key)
    if isinstance(value, tuple):
        value = value[0]
    return None if value is None else int(value)


def _copy_number_values(record: Any, field: str) -> tuple[NDArray, BoolArray] | None:
    """Integer copy numbers of one FORMAT field and where they are missing."""
    values = record.format(field)
    if values is None:
        return None
    flat = np.asarray(values).reshape(values.shape[0], -1)[:, 0]
    if flat.dtype.kind in "US":
        text = flat.astype(str)
        missing = (text == ".") | (text == "")
        numbers = np.where(missing, "0", text).astype(np.int64)
    else:
        numbers = flat.astype(np.int64)
        missing = numbers < 0
    return numbers, missing


class GatksvSource:
    """One GATK-SV VCF (optionally one region of it) read as store records.

    ``skipped_records`` counts every dropped record by reason and
    ``kept_record_count`` the kept ones; both are complete once ``blocks()``
    is exhausted.
    """

    def __init__(self, vcf_path: str | Path, region: str | None = None) -> None:
        self.vcf_path = Path(vcf_path)
        self.region = region
        reader = VCF(str(self.vcf_path))
        self.sample_ids: tuple[str, ...] = tuple(str(sample) for sample in reader.samples)
        self._format_fields = tuple(field for field in ("CN", "RD_CN") if reader.contains(field))
        reader.close()
        self.skipped_records: Counter[str] = Counter()
        self.kept_record_count = 0

    @staticmethod
    def _skip_reason(record: Any, svtype: str | None, filter_values: tuple[str, ...], copy_number: bool) -> str | None:
        allowed = {PASSING_FILTER, COPY_NUMBER_FILTER} if copy_number else {PASSING_FILTER}
        if not set(filter_values) <= allowed:
            return "FILTER " + ";".join(filter_values)
        if svtype == "BND" or any("[" in alt or "]" in alt for alt in record.ALT):
            return "breakend"
        if not copy_number and len(record.ALT) != 1:
            return "multi-allelic " + (svtype if svtype is not None else "untyped")
        return None

    def _copy_numbers(self, record: Any) -> tuple[I64Array, BoolArray]:
        if not self._format_fields:
            raise ValueError(f"{self.vcf_path} has copy-number record {record.ID} but no FORMAT/CN or FORMAT/RD_CN.")
        numbers = np.zeros(len(self.sample_ids), dtype=np.int64)
        missing = np.ones(len(self.sample_ids), dtype=bool)
        for field in self._format_fields:
            parsed = _copy_number_values(record, field)
            if parsed is None:
                continue
            field_numbers, field_missing = parsed
            take = missing & ~field_missing
            numbers[take] = field_numbers[take]
            missing &= field_missing
        return numbers, missing

    def _allele_counts(self, record: Any) -> tuple[U8Array, BoolArray]:
        alleles = record.genotype.array()[:, :-1]
        missing = np.any(alleles < 0, axis=1)
        counts = np.where(missing, 0, np.sum(alleles > 0, axis=1))
        return counts.astype(np.uint8), missing

    def blocks(self, block_records: int) -> Iterator[GatksvBlock]:
        """Yield the kept records in blocks of at most ``block_records``."""
        if block_records < 1:
            raise ValueError("block_records must be positive.")
        reader = VCF(str(self.vcf_path))
        records = reader(self.region) if self.region is not None else reader
        pending: list[tuple[str, int, int, float, str, str, VariantClass, bool, U8Array, BoolArray]] = []
        try:
            for record in records:
                svtype_value = record.INFO.get("SVTYPE")
                svtype = None if svtype_value is None else str(svtype_value)
                filter_values = _filter_values(record)
                copy_number = svtype == "CNV" or COPY_NUMBER_FILTER in filter_values
                reason = self._skip_reason(record, svtype, filter_values, copy_number)
                if reason is not None:
                    self.skipped_records[reason] += 1
                    continue
                position = int(record.POS)
                svlen = _info_integer(record, "SVLEN")
                end = _info_integer(record, "END")
                # A copy-number record is typed as a CNV whatever its SVTYPE, so it
                # lands in COPY_NUMBER with the usual |SVLEN| / END length.
                typed_svtype = "CNV" if copy_number else svtype
                variant_class, length = variant_class_and_length(
                    pos=position,
                    ref=str(record.REF),
                    alt=str(record.ALT[0]),
                    svtype=typed_svtype,
                    svlen=None if svlen is None else float(svlen),
                    info_end=end,
                )
                if copy_number:
                    copy_numbers, no_call = self._copy_numbers(record)
                    if int(copy_numbers.max(initial=0)) > MAXIMUM_CODE:
                        self.skipped_records[f"copy number above {MAXIMUM_CODE}"] += 1
                        continue
                    values = copy_numbers.astype(np.uint8)
                else:
                    values, no_call = self._allele_counts(record)
                pending.append(
                    (
                        str(record.CHROM),
                        position,
                        int(record.end) if end is None else end,
                        length,
                        str(record.ID),
                        "" if typed_svtype is None else typed_svtype,
                        variant_class,
                        copy_number,
                        values,
                        no_call,
                    )
                )
                self.kept_record_count += 1
                if len(pending) == block_records:
                    yield _block_from_records(pending)
                    pending = []
            if pending:
                yield _block_from_records(pending)
        finally:
            reader.close()


def _block_from_records(
    pending: list[tuple[str, int, int, float, str, str, VariantClass, bool, U8Array, BoolArray]],
) -> GatksvBlock:
    chromosomes, positions, ends, lengths, variant_ids, svtypes, variant_classes, copy_numbers, values, no_calls = zip(*pending)
    return GatksvBlock(
        chromosomes=tuple(chromosomes),
        positions=np.asarray(positions, dtype=np.int64),
        ends=np.asarray(ends, dtype=np.int64),
        lengths=np.asarray(lengths, dtype=np.float64),
        variant_ids=tuple(variant_ids),
        svtypes=tuple(svtypes),
        variant_classes=tuple(variant_classes),
        is_copy_number=np.asarray(copy_numbers, dtype=bool),
        values=np.stack(values),
        no_call=np.stack(no_calls),
    )

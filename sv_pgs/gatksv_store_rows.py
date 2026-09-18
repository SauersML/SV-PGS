"""The rows a GATK-SV call set adds to the dosage store next to the imputed source.

Per chromosome, from the kept GATK-SV records aligned to the store's samples
(``GatksvBlock.aligned_to_store_samples``) and the imputed source's SV records
with their store codes:

1. Candidate pairs come from ``sv_fusion.candidate_pairs`` on the records'
   SVTYPEs, so only bi-allelic DEL, DUP, INS and CPX records pair. A
   copy-number record never does: a multi-allelic CNV is not a bi-allelic
   Hardy-Weinberg locus, and the two-source closure fails there
   (design-svcontent measured implied reliabilities of 1.0-9.6 at GBA, CYP2D6,
   CYP21A2 and RHD).
2. Every candidate is calibrated (``sv_fusion.calibrate_two_sources``) and the
   accepted ones are resolved one to one. An accepted pair becomes one fused
   row that replaces its imputed record, so the locus is one column.
3. Every other GATK-SV record is a row of its own, with each no-call filled by
   the best linear prediction of the call. When the record's strongest
   candidate pairing is significant (Fisher z >= ``MINIMUM_PAIRING_Z``), the
   prediction comes from that imputed DS, E[B | DS] = m_B + (C_AB / V_A)(DS - m_A);
   otherwise it is the mean of the record's observed calls. A record that no
   store sample has a call for carries nothing and is dropped (listed).

Calibration and fill statistics run over every store sample. They use
genotypes only, never a phenotype, and SPEC trains on all samples, so they are
the training statistics. ``observed_fractions`` keeps the share of each row
that was called. Codes follow the store encoding: a dosage or ALT count x is
round(127 x) and a copy number is stored as itself. A fused dosage or a
predicted call outside the stored range is clipped to it, and every clipped
entry is counted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sv_pgs._typing import BoolArray, F64Array, I64Array, NDArray, U8Array
from sv_pgs.gatksv_source import MAXIMUM_STORED_VALUE, GatksvBlock
from sv_pgs.sv_fusion import (
    MINIMUM_PAIRING_Z,
    SvSites,
    TwoSourceCalibration,
    calibrate_two_sources,
    candidate_pairs,
    fused_dosage,
    resolve_one_to_one,
)

CODES_PER_ALLELE = 127
MAXIMUM_ALLELE_COUNT = 2


@dataclass(frozen=True, slots=True)
class ImputedSvRecords:
    """The imputed source's SV records of one chromosome, with their store codes [records, samples]."""

    sites: SvSites
    codes: U8Array

    def __post_init__(self) -> None:
        if not self.sites.duplications_are_insertions:
            raise ValueError("the imputed source is sequence-resolved: its DUP records are inserted copies.")
        if self.codes.dtype != np.uint8 or self.codes.ndim != 2 or self.codes.shape[0] != self.sites.starts.shape[0]:
            raise ValueError("ImputedSvRecords.codes must be uint8 [records, samples], one row per site.")


@dataclass(frozen=True, slots=True)
class FusedRows:
    """One fused row per accepted pair; row ``r`` replaces imputed record ``imputed_records[r]``."""

    imputed_records: I64Array
    gatksv_records: I64Array
    calibrations: tuple[TwoSourceCalibration, ...]
    codes: U8Array
    clipped_counts: I64Array


@dataclass(frozen=True, slots=True)
class GatksvRows:
    """The GATK-SV records that stay rows of their own, every no-call filled.

    ``filled_from_imputed[r]`` says the no-calls of row ``r`` were predicted
    from a paired imputed DS. ``unobserved_records`` are the block records no
    store sample has a call for, dropped.
    """

    gatksv_records: I64Array
    codes: U8Array
    observed_fractions: F64Array
    filled_from_imputed: BoolArray
    clipped_counts: I64Array
    unobserved_records: I64Array


def gatksv_sites(block: GatksvBlock) -> SvSites:
    """The block's records on ``sv_fusion``'s coordinates.

    VCF POS is the padding base before a symbolic SV, so the first affected
    base is POS + 1; END is the last affected base, so the exclusive end is
    END + 1. An insertion is the point after its anchor.
    """
    kinds = np.asarray(block.svtypes)
    starts = block.positions + 1
    ends = np.where(kinds == "INS", starts + 1, np.maximum(block.ends + 1, starts + 1))
    return SvSites(
        chromosomes=np.asarray(block.chromosomes),
        starts=starts,
        ends=ends,
        sizes=np.maximum(np.rint(block.lengths).astype(np.int64), 1),
        kinds=kinds,
        duplications_are_insertions=False,
    )


def _clipped_codes(values: F64Array, maximum_value: float, codes_per_unit: int) -> tuple[U8Array, int]:
    clipped = np.clip(values, 0.0, maximum_value)
    codes = np.floor(clipped * codes_per_unit + 0.5).astype(np.uint8)
    return codes, int(np.count_nonzero(clipped != values))


def _dosage_codes(values: F64Array) -> tuple[U8Array, int]:
    return _clipped_codes(values, float(MAXIMUM_ALLELE_COUNT), CODES_PER_ALLELE)


def _copy_number_codes(values: F64Array) -> tuple[U8Array, int]:
    return _clipped_codes(values, float(MAXIMUM_STORED_VALUE), 1)


def gatksv_store_rows(
    gatksv: GatksvBlock,
    imputed: ImputedSvRecords,
    group_labels: NDArray,
) -> tuple[FusedRows, GatksvRows]:
    """Fuse the GATK-SV records that pair with an imputed record and fill the rest.

    ``gatksv`` holds one chromosome's kept records aligned to the store's
    samples, ``imputed`` the same chromosome's imputed SV records, and
    ``group_labels`` each store sample's genetic-similarity group.
    """
    sample_count = gatksv.values.shape[1]
    if imputed.codes.shape[1] != sample_count or np.asarray(group_labels).shape != (sample_count,):
        raise ValueError("the GATK-SV block, the imputed codes and the group labels need the store's samples.")
    observed = ~gatksv.no_call

    def imputed_dosage(record: int) -> F64Array:
        return imputed.codes[record] / float(CODES_PER_ALLELE)

    pairs = candidate_pairs(imputed.sites, gatksv_sites(gatksv))
    calibrations = [
        calibrate_two_sources(imputed_dosage(first_row), gatksv.values[second_row], observed[second_row], group_labels)
        for first_row, second_row in zip(pairs.first_rows.tolist(), pairs.second_rows.tolist())
    ]
    chosen = resolve_one_to_one(pairs, calibrations)

    fused_codes = np.empty((chosen.shape[0], sample_count), dtype=np.uint8)
    fused_clipped = np.zeros(chosen.shape[0], dtype=np.int64)
    for row, pair in enumerate(chosen.tolist()):
        first_row = int(pairs.first_rows[pair])
        second_row = int(pairs.second_rows[pair])
        dosage = fused_dosage(calibrations[pair], imputed_dosage(first_row), gatksv.values[second_row], observed[second_row])
        fused_codes[row], fused_clipped[row] = _dosage_codes(dosage)
    fused = FusedRows(
        imputed_records=pairs.first_rows[chosen],
        gatksv_records=pairs.second_rows[chosen],
        calibrations=tuple(calibrations[pair] for pair in chosen.tolist()),
        codes=fused_codes,
        clipped_counts=fused_clipped,
    )

    # The significant candidate with the largest Fisher z of each GATK-SV record.
    strongest_pair = np.full(gatksv.record_count, -1, dtype=np.int64)
    for pair, calibration in enumerate(calibrations):
        second_row = int(pairs.second_rows[pair])
        current = int(strongest_pair[second_row])
        if calibration.pairing_z >= MINIMUM_PAIRING_Z and (current < 0 or calibration.pairing_z > calibrations[current].pairing_z):
            strongest_pair[second_row] = pair

    fused_records = set(fused.gatksv_records.tolist())
    observed_counts = observed.sum(axis=1)
    kept_records: list[int] = []
    unobserved_records: list[int] = []
    for record in range(gatksv.record_count):
        if record in fused_records:
            continue
        if observed_counts[record] == 0:
            unobserved_records.append(record)
            continue
        kept_records.append(record)
    row_codes = np.empty((len(kept_records), sample_count), dtype=np.uint8)
    row_clipped = np.zeros(len(kept_records), dtype=np.int64)
    filled_from_imputed = np.zeros(len(kept_records), dtype=bool)
    for row, record in enumerate(kept_records):
        calls = gatksv.values[record].astype(np.float64)
        pair = int(strongest_pair[record])
        if pair >= 0:
            calibration = calibrations[pair]
            prediction = calibration.second_mean + calibration.second_slope * (
                imputed_dosage(int(pairs.first_rows[pair])) - calibration.first_mean
            )
            filled_from_imputed[row] = True
        else:
            prediction = np.full(sample_count, calls[observed[record]].mean())
        values = np.where(observed[record], calls, prediction)
        encode = _copy_number_codes if gatksv.is_copy_number[record] else _dosage_codes
        row_codes[row], row_clipped[row] = encode(values)
    records = np.asarray(kept_records, dtype=np.int64)
    rows = GatksvRows(
        gatksv_records=records,
        codes=row_codes,
        observed_fractions=observed_counts[records] / float(sample_count),
        filled_from_imputed=filled_from_imputed,
        clipped_counts=row_clipped,
        unobserved_records=np.asarray(unobserved_records, dtype=np.int64),
    )
    return fused, rows

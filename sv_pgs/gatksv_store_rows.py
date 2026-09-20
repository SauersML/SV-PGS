"""The rows a GATK-SV call set adds to the dosage store next to the imputed source.

Per chromosome, from the kept GATK-SV records aligned to the store's samples
(``GatksvBlock.aligned_to_store_samples``) and the imputed source's SV records
with their store codes:

1. Every GATK-SV record that some store sample is called at becomes a row of its
   own. A record no store sample has a call for carries nothing and is dropped
   (listed).
2. Candidate pairs with the imputed records come from ``sv_fusion.candidate_pairs``
   on the records' SVTYPEs, so only bi-allelic DEL, DUP, INS and CPX records pair;
   a copy-number record never does. A pair's evidence is the Fisher z of the
   correlation of the imputed DS with the call over the called samples. Pairs
   with z >= ``sv_fusion.MINIMUM_PAIRING_Z`` are resolved one to one in order of
   decreasing z and returned as ``FusionPairs``. They are fused by the
   measurement model (``measurement_model.LdBlock`` with the call absorbed), from
   truth pairs, never here: the truth-free two-source calibration assumed
   classical error on the imputed DS, which a draw-type DS violates.
3. Each no-call is filled with the best linear prediction of the call. When the
   record has a candidate imputed record with a defined correlation, the
   prediction comes from the imputed DS of the candidate with the largest Fisher
   z, E[B | DS] = m_B + (C_AB / V_A)(DS - m_A) over the called samples; a record
   with no such candidate takes the mean of its observed calls.

Fill statistics run over every store sample. They use genotypes only, never a
phenotype, and SPEC trains on all samples, so they are the training statistics.
``observed_fractions`` keeps the share of each row that was called. Codes follow
the store encoding: an ALT count x is round(127 x), and a copy-number row's copy
number c is round(c k) with k = floor(254 / its maximum called copy number), its
value c - its modal called copy number (``copy_number``); each row carries its
``codes_per_unit`` and ``value_origin``. A predicted call outside the stored range
is clipped to it, and every entry whose code the clip changed is counted.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from sv_pgs._typing import BoolArray, F64Array, I64Array, U8Array
from sv_pgs.copy_number import copy_number_codes_per_unit, modal_copy_numbers
from sv_pgs.dosage_store import CODES_PER_DOSAGE
from sv_pgs.gatksv_source import GatksvBlock
from sv_pgs.sv_fusion import MINIMUM_CALIBRATION_SAMPLES, MINIMUM_PAIRING_Z, SvSites, _fisher_z, candidate_pairs

MAXIMUM_ALLELE_COUNT = 2


@dataclass(frozen=True, slots=True)
class ImputedSvRecords:
    """The imputed source's SV records of one chromosome with their store codes [records, samples] (DS = code / 127)."""

    sites: SvSites
    codes: U8Array

    def __post_init__(self) -> None:
        if not self.sites.duplications_are_insertions:
            raise ValueError("the imputed source is sequence-resolved: its DUP records are inserted copies.")
        record_count = self.sites.starts.shape[0]
        if self.codes.dtype != np.uint8 or self.codes.ndim != 2 or self.codes.shape[0] != record_count:
            raise ValueError("ImputedSvRecords.codes must be uint8 [records, samples], one row per site.")


@dataclass(frozen=True, slots=True)
class FusionPairs:
    """One-to-one pairs of an imputed record and a GATK-SV row (an index into ``GatksvRows``) of one event."""

    imputed_records: I64Array
    gatksv_rows: I64Array
    pairing_z: F64Array


@dataclass(frozen=True, slots=True)
class GatksvRows:
    """The GATK-SV records that stay rows of their own, every no-call filled.

    ``filled_from_imputed[r]`` says the no-calls of row ``r`` were predicted
    from a paired imputed DS. ``lengths`` are the records' SV lengths (|SVLEN|
    or END span), which a symbolic record's REF/ALT do not carry, for the
    store's length annotation. A row's value is ``codes / codes_per_unit +
    value_origin``. ``unobserved_records`` are the block records no store
    sample has a call for, dropped.
    """

    gatksv_records: I64Array
    lengths: F64Array
    codes: U8Array
    codes_per_unit: U8Array
    value_origin: I64Array
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
    """Codes of ``values`` clipped to [0, maximum_value], and how many entries the clip changed."""
    codes = np.floor(np.clip(values, 0.0, maximum_value) * codes_per_unit + 0.5).astype(np.uint8)
    # Only a value past half a code outside the range stores a different code.
    half_code = 0.5 / codes_per_unit
    return codes, int(np.count_nonzero((values < -half_code) | (values >= maximum_value + half_code)))


def _dosage_codes(values: F64Array) -> tuple[U8Array, int]:
    return _clipped_codes(values, float(MAXIMUM_ALLELE_COUNT), CODES_PER_DOSAGE)


def _copy_number_codes(values: F64Array, calls: U8Array, observed: BoolArray) -> tuple[U8Array, int, int, int]:
    """A copy-number row's codes, clip count, codes per copy and value origin (minus its modal copy number)."""
    maximum = int(calls[observed].max())
    codes_per_unit = int(copy_number_codes_per_unit(np.array([maximum]))[0])
    codes, clipped = _clipped_codes(values, float(maximum), codes_per_unit)
    return codes, clipped, codes_per_unit, -int(modal_copy_numbers(calls[None], observed[None])[0])


def _pairing(dosage: F64Array, calls: F64Array) -> tuple[float, float]:
    """(Fisher z of corr(DS, B), least-squares slope of B on DS) over the called samples; z = 0 when undefined."""
    if dosage.shape[0] < MINIMUM_CALIBRATION_SAMPLES or np.ptp(dosage) == 0.0 or np.ptp(calls) == 0.0:
        return 0.0, float("nan")
    covariance = np.cov(np.vstack([dosage, calls]), bias=True)
    correlation = float(covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1]))
    return _fisher_z(correlation, dosage.shape[0]), float(covariance[0, 1] / covariance[0, 0])


def gatksv_store_rows(gatksv: GatksvBlock, imputed: ImputedSvRecords) -> tuple[GatksvRows, FusionPairs]:
    """Every called GATK-SV record as a row, each no-call filled, and the pairs the measurement model fuses.

    ``gatksv`` holds one chromosome's kept records aligned to the store's samples
    and ``imputed`` the same chromosome's imputed SV records.
    """
    sample_count = gatksv.values.shape[1]
    if imputed.codes.shape[1] != sample_count:
        raise ValueError("the GATK-SV block and the imputed codes need the store's samples.")
    observed = ~gatksv.no_call
    dosage = imputed.codes / float(CODES_PER_DOSAGE)
    pairs = candidate_pairs(imputed.sites, gatksv_sites(gatksv))
    evidence = [
        _pairing(dosage[first, observed[second]], gatksv.values[second, observed[second]].astype(np.float64))
        for first, second in zip(pairs.first_rows.tolist(), pairs.second_rows.tolist())
    ]
    observed_counts = observed.sum(axis=1)
    kept_records = np.flatnonzero(observed_counts > 0)
    row_of_record = np.full(gatksv.record_count, -1, dtype=np.int64)
    row_of_record[kept_records] = np.arange(kept_records.shape[0])

    # The candidate with a defined slope and the largest Fisher z of each GATK-SV record fills its no-calls.
    strongest_pair = np.full(gatksv.record_count, -1, dtype=np.int64)
    for pair, (z, slope) in enumerate(evidence):
        second = int(pairs.second_rows[pair])
        current = int(strongest_pair[second])
        if np.isfinite(slope) and (current < 0 or z > evidence[current][0]):
            strongest_pair[second] = pair
    row_codes = np.empty((kept_records.shape[0], sample_count), dtype=np.uint8)
    row_clipped = np.zeros(kept_records.shape[0], dtype=np.int64)
    row_codes_per_unit = np.full(kept_records.shape[0], CODES_PER_DOSAGE, dtype=np.uint8)
    row_origins = np.zeros(kept_records.shape[0], dtype=np.int64)
    filled_from_imputed = np.zeros(kept_records.shape[0], dtype=bool)
    for row, record in enumerate(kept_records.tolist()):
        calls = gatksv.values[record].astype(np.float64)
        called = observed[record]
        pair = int(strongest_pair[record])
        if pair >= 0:
            first = int(pairs.first_rows[pair])
            prediction = calls[called].mean() + evidence[pair][1] * (dosage[first] - dosage[first, called].mean())
            filled_from_imputed[row] = True
        else:
            prediction = np.full(sample_count, calls[called].mean())
        values = np.where(called, calls, prediction)
        if gatksv.is_copy_number[record]:
            row_codes[row], row_clipped[row], row_codes_per_unit[row], row_origins[row] = _copy_number_codes(
                values, gatksv.values[record], called
            )
        else:
            row_codes[row], row_clipped[row] = _dosage_codes(values)
    rows = GatksvRows(
        gatksv_records=kept_records.astype(np.int64),
        lengths=gatksv.lengths[kept_records],
        codes=row_codes,
        codes_per_unit=row_codes_per_unit,
        value_origin=row_origins,
        observed_fractions=observed_counts[kept_records] / float(sample_count),
        filled_from_imputed=filled_from_imputed,
        clipped_counts=row_clipped,
        unobserved_records=np.flatnonzero(observed_counts == 0).astype(np.int64),
    )

    # One to one, in order of decreasing evidence; weaker candidates stay separate columns.
    accepted = sorted((pair for pair, (z, _) in enumerate(evidence) if z >= MINIMUM_PAIRING_Z), key=lambda pair: -evidence[pair][0])
    taken_first: set[int] = set()
    taken_second: set[int] = set()
    chosen: list[int] = []
    for pair in accepted:
        first, second = int(pairs.first_rows[pair]), int(pairs.second_rows[pair])
        if first in taken_first or second in taken_second:
            continue
        taken_first.add(first)
        taken_second.add(second)
        chosen.append(pair)
    chosen.sort()
    fusion = FusionPairs(
        imputed_records=pairs.first_rows[chosen].astype(np.int64),
        gatksv_rows=row_of_record[pairs.second_rows[chosen]],
        pairing_z=np.array([evidence[pair][0] for pair in chosen], dtype=np.float64),
    )
    return rows, fusion

"""Per-record transforms and sidecar fields of the popped-BCF -> dosage-store converter.

The converter (store spec A4-A4.15, team/target-format/STORE_SPEC.md) turns every popped
record of the imputed panel into one store row. This module holds its exact per-record
pieces; each is a pure function of sites-only inputs or of one record block:

- ``core_spans``: the reference bases a record changes, on 0-based half-open coordinates
  (a pure insertion spans the two bases around its insertion point).
- ``value_matched_background`` (A4.10): the imputation's error-rate background is removed
  where the dosage equals a background value the site's paths predict, and nowhere else.
- ``linear_recalibration``: the stored column D* = mu_g + kappa (DS - mu_g), per ancestry group
  g, with the stratum's truth-calibrated kappa, so that Cov(G, D*) = Var(D*).
- ``tr_loci`` (A4.6/A4.7): each record's tandem-repeat locus, the connected components of
  repeat intervals that records bridge.
- ``sv_context`` (A4.12): every record's distance to the nearest common SV and the number
  of distinct SV-bearing bubbles around it.
- ``unbreakable_group_first``: the store's ``group_first``, merging bubbles, same-POS sets and
  TR loci into contiguous row spans a Stage 0 block may never cut.
- ``decode_batch`` / ``decode_called_batch`` / ``no_call_fill`` / ``assemble_half``: one batch
  file, checked in lockstep against the sidecar (gate G2), read as imputed DS (FORMAT gate G4,
  background removed) or as hard calls (the long-read panel members' half), and encoded; then a
  half's batches side by side in batch order, no-calls filled with the ancestry group's measured
  mean over the whole store, optionally recalibrated, written as the half's shards.
- ``write_store_manifest``: the MANIFEST with each half's measurement and the gate results.

Everything here is AoU panel-derived site structure or genotype-derived data once applied
to the panel, so its outputs are workspace-only (A4.9).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from cyvcf2 import VCF
import numpy as np

from sv_pgs._typing import BoolArray, I64Array, NDArray, U8Array
from sv_pgs.dosage_store import (
    CODES_PER_DOSAGE,
    MAXIMUM_CODE,
    MAXIMUM_DOSAGE_MILLI,
    MISSING_CODE,
    Codec,
    encode_dosage_milli,
    statistic_column_directory,
    write_column,
    write_half_codes,
    write_manifest,
)
from sv_pgs.variant_typing import trimmed_allele_cores

# GLIMPSE2 --err-imp of the aou2 imputation.
IMPUTATION_ERROR_RATE = 1e-3
# pop keeps at most this many of a bubble's paths per haplotype (max_alleles).
MAXIMUM_KEPT_PATHS = 10
NO_LOCUS = np.uint32(0xFFFFFFFF)
NO_RECORD = np.uint32(0xFFFFFFFF)
NO_DISTANCE = np.uint32(0xFFFFFFFF)
SV_CONTEXT_WINDOW = 50_000
# Records decoded, corrected and encoded per step of a batch's pass.
DECODE_BLOCK_ROWS = 4_096
# GP is written to 3 decimals, so its thousandths sum to 1000 within one unit of rounding.
GENOTYPE_PROBABILITY_SUM_SLACK_MILLI = 1
# DS and GP are rounded from one unrounded GP. With a, b in [0, 1) the fractional thousandths
# of GP1 and GP2, DS rounds a + 2b while GP1 + 2 GP2 rounds a and b separately; every case
# away from exact halves puts the two within one thousandth.
DOSAGE_FROM_PROBABILITY_SLACK_MILLI = 1


def core_spans(positions: NDArray, refs: Sequence[str], alts: Sequence[str]) -> tuple[I64Array, I64Array]:
    """0-based half-open spans of the reference bases each record changes.

    ``positions`` are VCF POS (1-based). The core starts after the shared prefix and covers the
    REF core; a pure insertion (empty REF core) covers the two bases around its insertion point.
    """
    count = len(refs)
    if len(alts) != count or np.asarray(positions).shape[0] != count:
        raise ValueError("core_spans needs one position, REF and ALT per record.")
    starts = np.empty(count, dtype=np.int64)
    ends = np.empty(count, dtype=np.int64)
    for row, (position, ref, alt) in enumerate(zip(np.asarray(positions, dtype=np.int64).tolist(), refs, alts)):
        prefix, ref_core, _ = trimmed_allele_cores(ref, alt)
        start = position - 1 + prefix
        if ref_core == 0:
            starts[row], ends[row] = start - 1, start + 1
        else:
            starts[row], ends[row] = start, start + ref_core
    return starts, ends


@dataclass(frozen=True, slots=True)
class BackgroundCorrection:
    """A block's corrected dosages and the floor-QC counts of the correction (A4.10)."""

    dosage_milli: NDArray
    zeroed: I64Array
    unmatched_low: I64Array


def value_matched_background(
    dosage_milli: NDArray,
    kept_paths: NDArray,
    carrying_paths: NDArray,
    error_rate: float = IMPUTATION_ERROR_RATE,
) -> BackgroundCorrection:
    """Zero each dosage that equals a background value its site predicts; leave every other value.

    Per record, with K = ``kept_paths`` = min(10, N_PATHS_TOTAL) and m = ``carrying_paths`` = the
    record's paths among the bubble's first K path records, a haplotype on the floor carries
    q = m w / (1 + K w), w = eps / (1 - eps), which is pop's odds normalization of GLIMPSE2's
    floor posterior. The backgrounds are {0, q, 2q} per sample; a dosage (3 decimals) equal to
    round_half_up(1000 q) or round_half_up(2000 q) is background and set to 0. This covers a
    single-path record (1 and 2 milli), a PL-bearing record (per-sample floors 0 and 2 eps) and
    a record none of whose paths was kept (m = 0, nothing changes). ``dosage_milli`` is
    [records, samples] integer; ``zeroed`` counts the changed entries per record and
    ``unmatched_low`` the entries left in 1..9 milli (few, and on carriers).
    """
    milli = np.asarray(dosage_milli)
    if milli.dtype.kind not in "iu" or milli.ndim != 2:
        raise ValueError("value_matched_background needs integer dosage thousandths [records, samples].")
    kept = np.asarray(kept_paths, dtype=np.float64)
    carrying = np.asarray(carrying_paths, dtype=np.float64)
    if kept.shape != (milli.shape[0],) or carrying.shape != (milli.shape[0],):
        raise ValueError("value_matched_background needs one kept-path and one carrying-path count per record.")
    if np.any(carrying > kept) or np.any(kept > MAXIMUM_KEPT_PATHS) or np.any(carrying < 0):
        raise ValueError(f"carrying paths must lie in [0, kept paths] and kept paths in [0, {MAXIMUM_KEPT_PATHS}].")
    odds = error_rate / (1.0 - error_rate)
    background = carrying * odds / (1.0 + kept * odds)
    single = np.floor(1000.0 * background + 0.5).astype(np.int64)[:, None]
    double = np.floor(2000.0 * background + 0.5).astype(np.int64)[:, None]
    values = milli.astype(np.int64)
    matched = (values != 0) & ((values == single) | (values == double))
    corrected = np.where(matched, 0, milli).astype(milli.dtype)
    low = (corrected > 0) & (corrected < 10)
    return BackgroundCorrection(
        dosage_milli=corrected,
        zeroed=matched.sum(axis=1).astype(np.int64),
        unmatched_low=low.sum(axis=1).astype(np.int64),
    )


@dataclass(frozen=True, slots=True)
class Recalibration:
    """A block's recalibrated dosages and how many entries the [0, 2] range clipped."""

    dosage_milli: NDArray
    clipped: I64Array


def linear_recalibration(dosage_milli: NDArray, sample_groups: NDArray, scales: NDArray) -> Recalibration:
    """D* = mu_g + kappa (DS - mu_g) per record and ancestry group, on thousandths of a dosage.

    ``sample_groups[s]`` is sample s's ancestry group (0..G-1) and ``scales[r, g]`` the kappa of
    record r's stratum in group g (``imputation_reliability.calibrated_scale`` from the triad
    r^2, never a regression slope on one noisy truth). mu_g is the record's mean over the
    group's samples, so the group means are unchanged. Results are rounded half up and clipped
    to [0, 2000]; ``clipped`` counts the entries the clip changed.
    """
    milli = np.asarray(dosage_milli)
    groups = np.asarray(sample_groups, dtype=np.int64)
    kappa = np.asarray(scales, dtype=np.float64)
    if milli.dtype.kind not in "iu" or milli.ndim != 2 or groups.shape != (milli.shape[1],):
        raise ValueError("linear_recalibration needs integer thousandths [records, samples] and one group per sample.")
    if groups.size and int(groups.min()) < 0:
        raise ValueError("ancestry groups must be 0..G-1.")
    group_count = int(groups.max()) + 1 if groups.size else 0
    if kappa.shape != (milli.shape[0], group_count) or not np.all(np.isfinite(kappa)) or np.any(kappa <= 0.0):
        raise ValueError("linear_recalibration needs a positive finite kappa per record and ancestry group.")
    values = milli.astype(np.float64)
    recalibrated = np.empty_like(values)
    for group in range(group_count):
        members = groups == group
        centre = values[:, members].mean(axis=1, keepdims=True)
        recalibrated[:, members] = centre + kappa[:, group : group + 1] * (values[:, members] - centre)
    rounded = np.floor(recalibrated + 0.5)
    clipped = (rounded < 0) | (rounded > MAXIMUM_DOSAGE_MILLI)
    return Recalibration(
        dosage_milli=np.clip(rounded, 0, MAXIMUM_DOSAGE_MILLI).astype(milli.dtype),
        clipped=clipped.sum(axis=1).astype(np.int64),
    )


@dataclass(frozen=True, slots=True)
class TrLoci:
    """One chromosome's tandem-repeat loci (A4.7) and each record's locus.

    ``record_locus[r]`` indexes the loci table (in coordinate order), ``NO_LOCUS`` for a record
    that overlaps no repeat interval. A locus is a connected component of repeat intervals under
    "a record's core overlaps both"; ``starts``/``ends`` span its intervals.
    """

    record_locus: NDArray
    starts: I64Array
    ends: I64Array
    interval_counts: I64Array
    record_counts: I64Array
    length_changing_record_counts: I64Array


def tr_loci(
    interval_starts: NDArray,
    interval_ends: NDArray,
    core_starts: NDArray,
    core_ends: NDArray,
    length_changes: NDArray,
) -> TrLoci:
    """Assign records to tandem-repeat loci on one chromosome.

    ``interval_starts``/``interval_ends`` are the repeat BED (0-based half-open, sorted and
    non-overlapping, as the GIAB AllTandemRepeatsandHomopolymers_slop5 file is); a record joins
    every interval its core overlaps, and intervals a record bridges merge into one locus, so each
    record, and each haplotype's length change, counts in exactly one locus.
    """
    interval_starts = np.asarray(interval_starts, dtype=np.int64)
    interval_ends = np.asarray(interval_ends, dtype=np.int64)
    core_starts = np.asarray(core_starts, dtype=np.int64)
    core_ends = np.asarray(core_ends, dtype=np.int64)
    if np.any(interval_ends <= interval_starts) or np.any(interval_starts[1:] < interval_ends[:-1]):
        raise ValueError("repeat intervals must be non-empty, sorted and non-overlapping.")
    if np.any(core_ends <= core_starts):
        raise ValueError("record cores must be non-empty.")
    # Intervals overlapping [start, end): those with end > start and start < end.
    first_interval = np.searchsorted(interval_ends, core_starts, side="right")
    past_interval = np.searchsorted(interval_starts, core_ends, side="left")
    overlaps = past_interval > first_interval
    # A record bridging intervals [first, past) links each to the next: one union of a run.
    interval_count = interval_starts.shape[0]
    links = np.zeros(interval_count, dtype=np.int64)
    bridging = overlaps & (past_interval - first_interval > 1)
    np.add.at(links, first_interval[bridging], 1)
    np.add.at(links, past_interval[bridging] - 1, -1)
    # links[i] > 0 over a run means interval i joins interval i + 1.
    joins_next = np.cumsum(links)[:-1] > 0 if interval_count else np.zeros(0, dtype=bool)
    component_of_interval = np.concatenate([[0], np.cumsum(~joins_next)]).astype(np.int64) if interval_count else np.zeros(0, dtype=np.int64)
    record_locus = np.full(core_starts.shape[0], NO_LOCUS, dtype=np.uint32)
    record_locus[overlaps] = component_of_interval[first_interval[overlaps]]
    component_count = int(component_of_interval[-1]) + 1 if interval_count else 0
    starts = np.full(component_count, np.iinfo(np.int64).max, dtype=np.int64)
    ends = np.full(component_count, np.iinfo(np.int64).min, dtype=np.int64)
    np.minimum.at(starts, component_of_interval, interval_starts)
    np.maximum.at(ends, component_of_interval, interval_ends)
    interval_counts = np.bincount(component_of_interval, minlength=component_count).astype(np.int64)
    loci_of_records = record_locus[overlaps].astype(np.int64)
    record_counts = np.bincount(loci_of_records, minlength=component_count).astype(np.int64)
    changing = np.asarray(length_changes)[overlaps] != 0
    length_changing = np.bincount(loci_of_records[changing], minlength=component_count).astype(np.int64)
    return TrLoci(
        record_locus=record_locus,
        starts=starts,
        ends=ends,
        interval_counts=interval_counts,
        record_counts=record_counts,
        length_changing_record_counts=length_changing,
    )


@dataclass(frozen=True, slots=True)
class SvContext:
    """Every record's SV context on one chromosome (A4.12)."""

    nearest_common_sv_distance: NDArray
    nearest_common_sv_record: NDArray
    sv_bubbles_nearby: NDArray


def sv_context(
    core_starts: NDArray,
    core_ends: NDArray,
    is_sv: BoolArray,
    is_common_sv: BoolArray,
    bubble_indices: NDArray,
    window: int = SV_CONTEXT_WINDOW,
) -> SvContext:
    """Distance to the nearest common SV and distinct SV-bearing bubbles within ``window`` bp.

    Distances are between core spans (0 when they overlap) and exclude the record itself;
    ties go to the lower record index. The bubble count covers every bubble whose SV records'
    cores (their hull; a bubble's records share one POS) overlap [start - window, end + window],
    other than the record's own bubble, and counts bubbles rather than records so that how a
    locus's alleles are split does not change it. Both are sorted-interval searches, with no
    per-record loop.
    """
    starts = np.asarray(core_starts, dtype=np.int64)
    ends = np.asarray(core_ends, dtype=np.int64)
    sv = np.asarray(is_sv, dtype=bool)
    common = np.asarray(is_common_sv, dtype=bool)
    bubbles = np.asarray(bubble_indices, dtype=np.int64)
    record_count = starts.shape[0]
    if not (ends.shape == sv.shape == common.shape == bubbles.shape == (record_count,)):
        raise ValueError("sv_context needs one core span, SV flag, common flag and bubble per record.")
    if np.any(common & ~sv):
        raise ValueError("a common SV record must be an SV record.")

    distance = np.full(record_count, NO_DISTANCE, dtype=np.uint32)
    nearest = np.full(record_count, NO_RECORD, dtype=np.uint32)
    common_records = np.flatnonzero(common)
    if common_records.shape[0]:
        order = common_records[np.lexsort((common_records, starts[common_records]))]
        sorted_starts = starts[order]
        sorted_ends = ends[order]
        # Prefix maximum of ends in start order, and where it is attained (first occurrence).
        running_end = np.maximum.accumulate(sorted_ends)
        new_maximum = np.concatenate([[True], sorted_ends[1:] > running_end[:-1]])
        attained = np.maximum.accumulate(np.where(new_maximum, np.arange(order.shape[0]), 0))
        rank_of_common = np.full(record_count, -1, dtype=np.int64)
        rank_of_common[order] = np.arange(order.shape[0])
        # Left candidates: common SVs strictly before the record in start order.
        insertion = np.searchsorted(sorted_starts, starts, side="left")
        own_rank = rank_of_common
        left_limit = np.where(own_rank >= 0, own_rank, insertion)
        right_rank = np.where(own_rank >= 0, own_rank + 1, insertion)
        big = np.iinfo(np.int64).max
        left_distance = np.full(record_count, big, dtype=np.int64)
        left_record = np.full(record_count, -1, dtype=np.int64)
        has_left = left_limit > 0
        left_position = attained[left_limit[has_left] - 1]
        left_distance[has_left] = np.maximum(0, starts[has_left] - running_end[left_limit[has_left] - 1])
        left_record[has_left] = order[left_position]
        right_distance = np.full(record_count, big, dtype=np.int64)
        right_record = np.full(record_count, -1, dtype=np.int64)
        has_right = right_rank < order.shape[0]
        right_position = right_rank[has_right]
        right_distance[has_right] = np.maximum(0, sorted_starts[right_position] - ends[has_right])
        right_record[has_right] = order[right_position]
        take_left = (left_distance < right_distance) | (
            (left_distance == right_distance) & (left_record >= 0) & ((right_record < 0) | (left_record < right_record))
        )
        best_distance = np.where(take_left, left_distance, right_distance)
        best_record = np.where(take_left, left_record, right_record)
        found = best_record >= 0
        distance[found] = best_distance[found].astype(np.uint32)
        nearest[found] = best_record[found].astype(np.uint32)

    nearby = np.zeros(record_count, dtype=np.uint16)
    sv_records = np.flatnonzero(sv)
    if sv_records.shape[0]:
        sv_bubbles = np.unique(bubbles[sv_records])
        bubble_start = np.full(sv_bubbles.shape[0], np.iinfo(np.int64).max, dtype=np.int64)
        bubble_end = np.full(sv_bubbles.shape[0], np.iinfo(np.int64).min, dtype=np.int64)
        slot = np.searchsorted(sv_bubbles, bubbles[sv_records])
        np.minimum.at(bubble_start, slot, starts[sv_records])
        np.maximum.at(bubble_end, slot, ends[sv_records])
        low = starts - window
        high = ends + window
        # #intervals overlapping [low, high) = total - #(end <= low) - #(start >= high).
        overlapping = (
            sv_bubbles.shape[0]
            - np.searchsorted(np.sort(bubble_end), low, side="right")
            - (sv_bubbles.shape[0] - np.searchsorted(np.sort(bubble_start), high, side="left"))
        )
        own_slot = np.searchsorted(sv_bubbles, bubbles)
        own_is_sv_bubble = (own_slot < sv_bubbles.shape[0]) & (sv_bubbles[np.minimum(own_slot, sv_bubbles.shape[0] - 1)] == bubbles)
        own_slot = np.minimum(own_slot, sv_bubbles.shape[0] - 1)
        own_overlaps = own_is_sv_bubble & (bubble_end[own_slot] > low) & (bubble_start[own_slot] < high)
        nearby = np.minimum(overlapping - own_overlaps.astype(np.int64), np.iinfo(np.uint16).max).astype(np.uint16)
    return SvContext(
        nearest_common_sv_distance=distance,
        nearest_common_sv_record=nearest,
        sv_bubbles_nearby=nearby,
    )


def unbreakable_group_first(*group_ids: NDArray) -> I64Array:
    """Each row's ``group_first``: the first row of the contiguous span it must share a block with.

    Each argument gives every row a group id (negative for no group) of one grouping (bubble,
    same-POS set, TR locus) over rows in store order. A group spans its first to its last row;
    overlapping spans merge, so a row between two members of a group is in the group's span.
    """
    row_count = np.asarray(group_ids[0]).shape[0]
    span_starts: list[I64Array] = []
    span_stops: list[I64Array] = []
    rows = np.arange(row_count, dtype=np.int64)
    for ids in group_ids:
        ids = np.asarray(ids, dtype=np.int64)
        if ids.shape != (row_count,):
            raise ValueError("unbreakable_group_first needs one group id per row in every grouping.")
        member = ids >= 0
        labels, inverse = np.unique(ids[member], return_inverse=True)
        first = np.full(labels.shape[0], row_count, dtype=np.int64)
        last = np.full(labels.shape[0], -1, dtype=np.int64)
        np.minimum.at(first, inverse, rows[member])
        np.maximum.at(last, inverse, rows[member])
        span_starts.append(first)
        span_stops.append(last + 1)
    starts = np.concatenate(span_starts)
    stops = np.concatenate(span_stops)
    group_first = rows.copy()
    if starts.size == 0:
        return group_first
    # Merge overlapping spans: a span continues while the running stop passes the next start.
    order = np.argsort(starts, kind="stable")
    starts, stops = starts[order], stops[order]
    running_stop = np.maximum.accumulate(stops)
    opens = np.concatenate([[True], starts[1:] >= running_stop[:-1]])
    merged_starts = starts[opens]
    merged_stops = np.maximum.reduceat(stops, np.flatnonzero(opens))
    covering = np.searchsorted(merged_starts, rows, side="right") - 1
    inside = covering >= 0
    inside[inside] = rows[inside] < merged_stops[covering[inside]]
    group_first[inside] = merged_starts[covering[inside]]
    return group_first


def refalt_digest(ref: str, alt: str) -> int:
    """The sidecar's refalt_md5: the first 16 hex digits of md5("REF\tALT"), as strata writes it."""
    return int(hashlib.md5(f"{ref}\t{alt}".encode()).hexdigest()[:16], 16)


@dataclass(frozen=True, slots=True)
class ExpectedSites:
    """One chromosome's sidecar keys, which every batch's records must match in order (gate G2)."""

    positions: I64Array
    refalt_digests: NDArray
    identifiers: tuple[str, ...]
    kept_paths: NDArray
    carrying_paths: NDArray

    def __post_init__(self) -> None:
        count = self.positions.shape[0]
        if not (self.refalt_digests.shape == self.kept_paths.shape == self.carrying_paths.shape == (count,)) or len(self.identifiers) != count:
            raise ValueError("ExpectedSites needs one digest, id, kept-path and carrying-path count per position.")


@dataclass(frozen=True, slots=True)
class DecodedBatch:
    """One batch's codes [records, batch samples] and its exact per-record, per-group sums and
    counts of measured dosages [records, groups]. A hard-call half's no-call is MISSING_CODE here
    until ``assemble_half`` fills it; ``no_calls`` counts them per record."""

    codes: U8Array
    group_sums: I64Array
    group_counts: I64Array
    zeroed: I64Array
    unmatched_low: I64Array
    no_calls: I64Array


def _gate(condition: bool, path: Path, record: int, detail: str) -> None:
    if not condition:
        raise ValueError(f"{path} record {record}: {detail}")


def _imputed_dosage_milli(record: Any, path: Path, row: int) -> NDArray:
    """DS in thousandths, gated on GP (gate G4): none missing, DS in [0, 2], GP summing to 1, DS = GP1 + 2 GP2."""
    dosage = record.format("DS")
    probabilities = record.format("GP")
    _gate(dosage is not None and probabilities is not None, path, row, "DS or GP missing")
    dosage_milli = np.rint(np.asarray(dosage, dtype=np.float64)[:, 0] * 1000.0)
    probability_milli = np.rint(np.asarray(probabilities, dtype=np.float64) * 1000.0)
    _gate(bool(np.all(np.isfinite(dosage_milli)) and np.all(np.isfinite(probability_milli))), path, row, "a DS or GP is missing")
    _gate(bool(np.all((dosage_milli >= 0) & (dosage_milli <= MAXIMUM_DOSAGE_MILLI))), path, row, "DS outside [0, 2]")
    _gate(
        bool(np.all(np.abs(probability_milli.sum(axis=1) - 1000.0) <= GENOTYPE_PROBABILITY_SUM_SLACK_MILLI)),
        path,
        row,
        "GP does not sum to 1",
    )
    implied = probability_milli[:, 1] + 2.0 * probability_milli[:, 2]
    _gate(bool(np.all(np.abs(dosage_milli - implied) <= DOSAGE_FROM_PROBABILITY_SLACK_MILLI)), path, row, "DS differs from GP1 + 2 GP2")
    return dosage_milli


# A hard-call half's no-call in the uint16 decode block.
NO_CALL_MILLI = np.iinfo(np.uint16).max
# cyvcf2 gt_types codes: HOM_REF, HET, UNKNOWN, HOM_ALT.
_MILLI_OF_GT_TYPE = np.array([0, MAXIMUM_DOSAGE_MILLI // 2, NO_CALL_MILLI, MAXIMUM_DOSAGE_MILLI])


def _called_dosage_milli(record: Any, path: Path, row: int) -> NDArray:
    """The ALT count of a hard genotype call in thousandths, NO_CALL_MILLI where it is a no-call."""
    _gate(len(record.ALT) == 1, path, row, "a called record must be biallelic")
    return _MILLI_OF_GT_TYPE[np.asarray(record.gt_types, dtype=np.int64)]


def _decode_records(
    vcf_path: str | Path,
    expected: ExpectedSites,
    sample_groups: NDArray,
    group_count: int,
    codes_path: str | Path,
    record_dosage_milli: Callable[[Any, Path, int], NDArray],
    remove_background: bool,
) -> DecodedBatch:
    path = Path(vcf_path)
    groups = np.asarray(sample_groups, dtype=np.int64)
    reader = VCF(str(path))
    try:
        sample_count = len(reader.samples)
        if groups.shape != (sample_count,) or (groups.size and (int(groups.min()) < 0 or int(groups.max()) >= group_count)):
            raise ValueError(f"{path}: need a group in 0..{group_count - 1} for each of its {sample_count} samples.")
        record_count = expected.positions.shape[0]
        codes = np.lib.format.open_memmap(Path(codes_path), mode="w+", dtype=np.uint8, shape=(record_count, sample_count))
        group_sums = np.zeros((record_count, group_count), dtype=np.int64)
        group_counts = np.zeros((record_count, group_count), dtype=np.int64)
        zeroed = np.zeros(record_count, dtype=np.int64)
        unmatched_low = np.zeros(record_count, dtype=np.int64)
        no_calls = np.zeros(record_count, dtype=np.int64)
        block = np.empty((DECODE_BLOCK_ROWS, sample_count), dtype=np.uint16)
        members = [groups == group for group in range(group_count)]

        def flush(stop: int) -> None:
            start = (stop - 1) // DECODE_BLOCK_ROWS * DECODE_BLOCK_ROWS
            called = block[: stop - start] != NO_CALL_MILLI
            dosage_milli = np.where(called, block[: stop - start], 0).astype(np.uint16)
            if remove_background:
                corrected = value_matched_background(dosage_milli, expected.kept_paths[start:stop], expected.carrying_paths[start:stop])
                dosage_milli = corrected.dosage_milli
                zeroed[start:stop] = corrected.zeroed
                unmatched_low[start:stop] = corrected.unmatched_low
            codes[start:stop] = np.where(called, encode_dosage_milli(dosage_milli), MISSING_CODE)
            no_calls[start:stop] = (~called).sum(axis=1)
            for group, mask in enumerate(members):
                group_sums[start:stop, group] = dosage_milli[:, mask].sum(axis=1, dtype=np.int64)
                group_counts[start:stop, group] = called[:, mask].sum(axis=1)

        row = 0
        for record in reader:
            _gate(row < record_count, path, row, "more records than the sidecar")
            _gate(record.POS == int(expected.positions[row]), path, row, "POS differs from the sidecar")
            _gate(refalt_digest(record.REF, ",".join(record.ALT)) == int(expected.refalt_digests[row]), path, row, "REF/ALT differ")
            _gate(str(record.INFO.get("ID")) == expected.identifiers[row], path, row, "INFO/ID differs")
            block[row % DECODE_BLOCK_ROWS] = record_dosage_milli(record, path, row).astype(np.uint16)
            row += 1
            if row % DECODE_BLOCK_ROWS == 0:
                flush(row)
        _gate(row == record_count, path, row, f"fewer records than the sidecar's {record_count}")
        if row % DECODE_BLOCK_ROWS:
            flush(row)
        codes.flush()
    finally:
        reader.close()
    return DecodedBatch(
        codes=np.load(Path(codes_path), mmap_mode="r"),
        group_sums=group_sums,
        group_counts=group_counts,
        zeroed=zeroed,
        unmatched_low=unmatched_low,
        no_calls=no_calls,
    )


def decode_batch(
    vcf_path: str | Path,
    expected: ExpectedSites,
    sample_groups: NDArray,
    group_count: int,
    codes_path: str | Path,
) -> DecodedBatch:
    """Decode one popped imputed batch file of one chromosome into corrected store codes.

    Records must match ``expected`` one for one (POS, md5 of REF/ALT, INFO/ID; gate G2), and
    every sample needs DS and GP with GP summing to 1 and DS = GP1 + 2 GP2 to within a
    thousandth (G4). The value-matched background is removed before encoding. ``codes_path``
    receives the codes as an .npy; ``sample_groups`` gives each of the file's samples, in header
    order, its ancestry group in 0..group_count-1.
    """
    return _decode_records(vcf_path, expected, sample_groups, group_count, codes_path, _imputed_dosage_milli, True)


def decode_called_batch(
    vcf_path: str | Path,
    expected: ExpectedSites,
    sample_groups: NDArray,
    group_count: int,
    codes_path: str | Path,
) -> DecodedBatch:
    """Decode hard genotype calls (the long-read panel members' half) on the same site list.

    The same lockstep gate as ``decode_batch``; each sample's dosage is its called ALT count, with
    no imputation background to remove. A no-call stays MISSING_CODE until ``assemble_half``.
    """
    return _decode_records(vcf_path, expected, sample_groups, group_count, codes_path, _called_dosage_milli, False)


def no_call_fill(group_sums: I64Array, group_counts: I64Array) -> NDArray:
    """Each record's fill for a no-call [records, groups], in thousandths rounded half up.

    ``group_sums``/``group_counts`` total the measured dosages of every batch of every half of the
    chromosome. The fill is the ancestry group's measured mean, the best linear predictor of a
    genotype with no measurement, or the record's pooled mean where the group has none. A record
    measured in no sample fails.
    """
    sums = np.asarray(group_sums, dtype=np.int64)
    counts = np.asarray(group_counts, dtype=np.int64)
    measured = counts.sum(axis=1)
    if np.any(measured == 0):
        raise ValueError(f"record {int(np.argmin(measured))}: no sample has a measurement.")
    pooled = sums.sum(axis=1) / measured
    means = np.where(counts > 0, sums / np.maximum(counts, 1), pooled[:, None])
    return np.floor(means + 0.5).astype(np.uint16)


def _half_code_blocks(
    batch_codes: Sequence[U8Array],
    sample_groups: NDArray,
    scales: NDArray | None,
    no_call_milli: NDArray,
    block_rows: int,
    no_calls: NDArray,
) -> Iterator[U8Array]:
    record_count = batch_codes[0].shape[0]
    for start in range(0, record_count, block_rows):
        stop = min(start + block_rows, record_count)
        codes = np.hstack([codes[start:stop] for codes in batch_codes])
        rows, columns = np.nonzero(codes == MISSING_CODE)
        codes[rows, columns] = encode_dosage_milli(no_call_milli[start:stop])[rows, sample_groups[columns]]
        no_calls[start:stop] = np.bincount(rows, minlength=stop - start)
        if scales is not None:
            # D* from the codes: DS = code / 127 to the nearest thousandth (rounded half up), recalibrated, re-encoded.
            dosage_milli = ((codes.astype(np.int64) * MAXIMUM_DOSAGE_MILLI + CODES_PER_DOSAGE) // MAXIMUM_CODE).astype(np.uint16)
            codes = encode_dosage_milli(linear_recalibration(dosage_milli, sample_groups, scales[start:stop]).dosage_milli)
        yield codes


def assemble_half(
    root: str | Path,
    half_index: int,
    chromosome: str,
    batch_codes: Sequence[U8Array],
    sample_groups: NDArray,
    scales: NDArray | None,
    no_call_milli: NDArray,
    *,
    codec: Codec,
    block_rows: int = DECODE_BLOCK_ROWS,
) -> tuple[I64Array, I64Array]:
    """Write one half of one chromosome from its batches' codes, columns in batch order.

    ``sample_groups`` covers the half's samples in that order. A no-call (MISSING_CODE) takes
    ``no_call_milli[record, group]`` (``no_call_fill`` of the chromosome's measurement totals).
    ``scales`` is the per-record, per-group kappa of the D* recalibration where
    design-reliability supplies one, or None before it exists (the stored codes are then the
    background-corrected DS, and the MANIFEST says so). Writes the half's per-record ``no_calls``
    statistic beside its code sums and returns the sums.
    """
    if not batch_codes:
        raise ValueError("assemble_half needs at least one batch.")
    record_count = batch_codes[0].shape[0]
    if any(codes.shape[0] != record_count for codes in batch_codes):
        raise ValueError("every batch of a chromosome needs the same records.")
    sample_count = sum(codes.shape[1] for codes in batch_codes)
    groups = np.asarray(sample_groups, dtype=np.int64)
    fill = np.asarray(no_call_milli)
    if fill.ndim != 2 or fill.shape[0] != record_count:
        raise ValueError("no_call_milli needs a row per record.")
    if groups.shape != (sample_count,) or (groups.size and (int(groups.min()) < 0 or int(groups.max()) >= fill.shape[1])):
        raise ValueError("sample_groups must cover the half's samples in batch order, one of no_call_milli's groups each.")
    no_calls = np.zeros(record_count, dtype=np.uint32)
    sums = write_half_codes(
        Path(root),
        half_index,
        chromosome,
        record_count,
        sample_count,
        _half_code_blocks(batch_codes, groups, scales, fill, block_rows, no_calls),
        codec=codec,
    )
    write_column(statistic_column_directory(Path(root), half_index, chromosome, "no_calls"), no_calls)
    return sums


HALF_MEASUREMENTS = ("imputed_dosage", "long_read_calls")


def write_store_manifest(
    root: str | Path,
    *,
    chromosomes: Sequence[str],
    record_counts: Sequence[int],
    chromosome_sites_md5: Sequence[str],
    half_sample_counts: Sequence[int],
    half_measurements: Sequence[str],
    gates: dict[str, str],
    recalibrated: bool,
) -> None:
    """The store MANIFEST: halves with their measurement kind, gate results, whether D* was applied.

    A long-read half shares the imputed halves' verified site list; the fit gives every half its
    own covariate.
    """
    if len(half_measurements) != len(half_sample_counts) or any(kind not in HALF_MEASUREMENTS for kind in half_measurements):
        raise ValueError(f"every half needs a measurement in {HALF_MEASUREMENTS}.")
    write_manifest(
        Path(root),
        chromosomes=chromosomes,
        record_counts=record_counts,
        half_sample_counts=half_sample_counts,
        chromosome_sites_md5=chromosome_sites_md5,
        attributes={"half_measurements": list(half_measurements), "gates": dict(gates), "recalibrated": recalibrated},
    )

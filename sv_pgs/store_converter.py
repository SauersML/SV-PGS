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

Everything here is AoU panel-derived site structure or genotype-derived data once applied
to the panel, so its outputs are workspace-only (A4.9).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from sv_pgs._typing import BoolArray, I64Array, NDArray
from sv_pgs.variant_typing import trimmed_allele_cores

# GLIMPSE2 --err-imp of the aou2 imputation.
IMPUTATION_ERROR_RATE = 1e-3
# pop keeps at most this many of a bubble's paths per haplotype (max_alleles).
MAXIMUM_KEPT_PATHS = 10
NO_LOCUS = np.uint32(0xFFFFFFFF)
NO_RECORD = np.uint32(0xFFFFFFFF)
NO_DISTANCE = np.uint32(0xFFFFFFFF)
SV_CONTEXT_WINDOW = 50_000
MAXIMUM_BUBBLE_COUNT = 65_535


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
    clipped = (rounded < 0) | (rounded > 2000)
    return Recalibration(
        dosage_milli=np.clip(rounded, 0, 2000).astype(milli.dtype),
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
        nearby = np.minimum(overlapping - own_overlaps.astype(np.int64), MAXIMUM_BUBBLE_COUNT).astype(np.uint16)
    return SvContext(
        nearest_common_sv_distance=distance,
        nearest_common_sv_record=nearest,
        sv_bubbles_nearby=nearby,
    )

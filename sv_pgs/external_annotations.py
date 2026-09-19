"""Map an external SV/VNTR association catalogue onto the store's records and TR loci.

Store spec A4.13-A4.15 (design-svcontent's matcher). An external association is evidence about
one allele, so a match is one-to-one in both directions, and an unmatched record has no row:
absence is explicit, never a zero or a mean that would leak the catalogue's coverage into the
prior. The per-trait payload (z^2 of the external association) is joined per trait by
external id; this module builds the trait-independent maps.

Record tiers, strongest first (the codes ``rec_map`` stores):

1. exact: same position, size and inserted/deleted sequence;
2. sequence: compatible kind, size ratio >= 0.5, breakpoints within 100 bp (insertions) or
   reciprocal overlap >= 0.5 (deletions), and k-mer Jaccard of the changed sequence >= 0.5;
3. coordinate: the same coordinate test without sequence agreement. On public chr20-22 these
   are mostly insertions with size ratio ~0.74: a different allele of the same repeat, which is
   why the tier stays its own code for empirical Bayes to weigh.

Assignment is greedy over (tier, quality, breakpoint distance, ids): each external record and
each store record is used at most once. VNTR loci match when their intervals overlap and their
motif lengths agree up to TRF's doubled or tripled periods.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from sv_pgs._typing import I64Array, NDArray
from sv_pgs.variant_typing import sequence_resolved_kind_and_length, trimmed_allele_cores

TIER_EXACT = 1
TIER_SEQUENCE = 2
TIER_COORDINATE = 3
BREAKPOINT_TOLERANCE = 100
SIZE_RATIO_FLOOR = 0.5
RECIPROCAL_OVERLAP_FLOOR = 0.5
JACCARD_FLOOR = 0.5
KMER_LENGTH = 11
CANDIDATE_BIN = 1_000
MOTIF_MULTIPLES = (1, 2, 3)


@dataclass(frozen=True, slots=True)
class ChangedSequences:
    """Sequence-resolved DEL/INS records, one row each, with the rows they came from.

    ``rows`` index the caller's input (store record indices for the panel side). ``starts`` is
    the first changed base (0-based), ``ends`` the exclusive end of the deleted bases (an
    insertion's end equals its start), ``sequences`` the deleted or inserted core.
    """

    rows: I64Array
    identifiers: tuple[str, ...]
    chromosomes: NDArray
    starts: I64Array
    ends: I64Array
    sizes: I64Array
    kinds: NDArray
    sequences: tuple[str, ...]


def changed_sequences(
    identifiers: Sequence[str],
    chromosomes: Sequence[str],
    positions: NDArray,
    refs: Sequence[str],
    alts: Sequence[str],
) -> ChangedSequences:
    """The DEL and INS records among sequence-resolved REF/ALT records (strata's kind rule)."""
    kept_rows: list[int] = []
    kept_starts: list[int] = []
    kept_ends: list[int] = []
    kept_sizes: list[int] = []
    kept_kinds: list[str] = []
    kept_sequences: list[str] = []
    for row, (position, ref, alt) in enumerate(zip(np.asarray(positions, dtype=np.int64).tolist(), refs, alts)):
        kind, length = sequence_resolved_kind_and_length(ref, alt)
        if kind not in ("DEL", "INS") or length < 1:
            continue
        prefix, ref_core, alt_core = trimmed_allele_cores(ref, alt)
        start = position - 1 + prefix
        kept_rows.append(row)
        kept_starts.append(start)
        kept_ends.append(start + ref_core if kind == "DEL" else start)
        kept_sizes.append(int(round(length)))
        kept_kinds.append(kind)
        kept_sequences.append((ref[prefix : prefix + ref_core] if kind == "DEL" else alt[prefix : prefix + alt_core]).upper())
    rows = np.asarray(kept_rows, dtype=np.int64)
    return ChangedSequences(
        rows=rows,
        identifiers=tuple(identifiers[row] for row in kept_rows),
        chromosomes=np.asarray([chromosomes[row] for row in kept_rows]),
        starts=np.asarray(kept_starts, dtype=np.int64),
        ends=np.asarray(kept_ends, dtype=np.int64),
        sizes=np.asarray(kept_sizes, dtype=np.int64),
        kinds=np.asarray(kept_kinds),
        sequences=tuple(kept_sequences),
    )


def _kmers(sequence: str) -> set[str]:
    if len(sequence) < KMER_LENGTH:
        return {sequence}
    return {sequence[offset : offset + KMER_LENGTH] for offset in range(len(sequence) - KMER_LENGTH + 1)}


def _jaccard(first: str, second: str) -> float:
    left, right = _kmers(first), _kmers(second)
    return len(left & right) / len(left | right)


def _score(external: ChangedSequences, external_row: int, panel: ChangedSequences, panel_row: int) -> tuple[int, float] | None:
    """(tier, quality in [0, 1]) of one candidate pair, or None when it fails the coordinate test."""
    kind = str(external.kinds[external_row])
    if kind != str(panel.kinds[panel_row]):
        return None
    external_size = int(external.sizes[external_row])
    panel_size = int(panel.sizes[panel_row])
    ratio = min(external_size, panel_size) / max(external_size, panel_size)
    distance = abs(int(external.starts[external_row]) - int(panel.starts[panel_row]))
    if kind == "DEL":
        overlap = max(0, min(int(external.ends[external_row]), int(panel.ends[panel_row])) - max(int(external.starts[external_row]), int(panel.starts[panel_row])))
        proximity = overlap / max(external_size, panel_size)
        passes = proximity >= RECIPROCAL_OVERLAP_FLOOR and ratio >= SIZE_RATIO_FLOOR
    else:
        proximity = max(0.0, 1.0 - distance / (BREAKPOINT_TOLERANCE + 1))
        passes = distance <= BREAKPOINT_TOLERANCE and ratio >= SIZE_RATIO_FLOOR
    if not passes:
        return None
    if distance == 0 and external_size == panel_size and external.sequences[external_row] == panel.sequences[panel_row]:
        return TIER_EXACT, 1.0
    similarity = _jaccard(external.sequences[external_row], panel.sequences[panel_row])
    if similarity >= JACCARD_FLOOR:
        return TIER_SEQUENCE, 0.5 * similarity + 0.3 * ratio + 0.2 * proximity
    return TIER_COORDINATE, 0.6 * ratio + 0.4 * proximity


@dataclass(frozen=True, slots=True)
class RecordMatches:
    """``rec_map``: one row per matched store record (store row, external id, tier)."""

    store_rows: I64Array
    external_identifiers: tuple[str, ...]
    tiers: NDArray
    displaced_pairs: int


def match_records(external: ChangedSequences, panel: ChangedSequences) -> RecordMatches:
    """One-to-one matches of external records to store records, strongest tier first."""
    bins: dict[tuple[str, int], list[int]] = collections.defaultdict(list)
    for panel_row in range(panel.rows.shape[0]):
        first_bin = (int(panel.starts[panel_row]) - BREAKPOINT_TOLERANCE) // CANDIDATE_BIN
        last_bin = (int(panel.ends[panel_row]) + BREAKPOINT_TOLERANCE) // CANDIDATE_BIN
        for bin_index in range(first_bin, last_bin + 1):
            bins[(str(panel.chromosomes[panel_row]), bin_index)].append(panel_row)
    candidates: list[tuple[int, float, int, str, str, int, int, int]] = []
    for external_row in range(external.rows.shape[0]):
        chromosome = str(external.chromosomes[external_row])
        first_bin = (int(external.starts[external_row]) - BREAKPOINT_TOLERANCE) // CANDIDATE_BIN
        last_bin = (int(external.ends[external_row]) + BREAKPOINT_TOLERANCE) // CANDIDATE_BIN
        nearby = {panel_row for bin_index in range(first_bin, last_bin + 1) for panel_row in bins.get((chromosome, bin_index), ())}
        for panel_row in sorted(nearby):
            scored = _score(external, external_row, panel, panel_row)
            if scored is None:
                continue
            tier, quality = scored
            candidates.append(
                (
                    tier,
                    -quality,
                    abs(int(external.starts[external_row]) - int(panel.starts[panel_row])),
                    external.identifiers[external_row],
                    panel.identifiers[panel_row],
                    external_row,
                    panel_row,
                    tier,
                )
            )
    candidates.sort(key=lambda candidate: candidate[:5])
    used_external: set[int] = set()
    used_panel: set[int] = set()
    store_rows: list[int] = []
    identifiers: list[str] = []
    tiers: list[int] = []
    displaced = 0
    for *_, external_row, panel_row, tier in candidates:
        if external_row in used_external:
            continue
        if panel_row in used_panel:
            displaced += 1
            continue
        used_external.add(external_row)
        used_panel.add(panel_row)
        store_rows.append(int(panel.rows[panel_row]))
        identifiers.append(external.identifiers[external_row])
        tiers.append(tier)
    order = np.argsort(np.asarray(store_rows, dtype=np.int64), kind="stable")
    return RecordMatches(
        store_rows=np.asarray(store_rows, dtype=np.int64)[order],
        external_identifiers=tuple(identifiers[index] for index in order.tolist()),
        tiers=np.asarray(tiers, dtype=np.uint8)[order],
        displaced_pairs=displaced,
    )


def majority_motif_length(
    locus_starts: NDArray,
    locus_ends: NDArray,
    repeat_starts: NDArray,
    repeat_ends: NDArray,
    repeat_periods: NDArray,
) -> NDArray:
    """Each locus's majority TRF period by overlapping bases (UCSC simpleRepeat); 0 if none overlaps.

    Loci and repeats are one chromosome's, 0-based half-open; ties go to the shorter period.
    """
    locus_starts = np.asarray(locus_starts, dtype=np.int64)
    locus_ends = np.asarray(locus_ends, dtype=np.int64)
    order = np.argsort(np.asarray(repeat_starts, dtype=np.int64), kind="stable")
    starts = np.asarray(repeat_starts, dtype=np.int64)[order]
    ends = np.asarray(repeat_ends, dtype=np.int64)[order]
    periods = np.asarray(repeat_periods, dtype=np.int64)[order]
    longest = int((ends - starts).max()) if starts.shape[0] else 0
    motif = np.zeros(locus_starts.shape[0], dtype=np.uint16)
    for locus, (locus_start, locus_end) in enumerate(zip(locus_starts.tolist(), locus_ends.tolist())):
        low = np.searchsorted(starts, locus_start - longest, side="left")
        high = np.searchsorted(starts, locus_end, side="left")
        overlap = np.minimum(ends[low:high], locus_end) - np.maximum(starts[low:high], locus_start)
        keep = overlap > 0
        if not keep.any():
            continue
        support: dict[int, int] = collections.defaultdict(int)
        for period, bases in zip(periods[low:high][keep].tolist(), overlap[keep].tolist()):
            support[period] += bases
        motif[locus] = min(support, key=lambda period: (-support[period], period))
    return motif


@dataclass(frozen=True, slots=True)
class LocusMatches:
    """``locus_map``: one row per matched TR locus (locus index, external locus id)."""

    loci: I64Array
    external_identifiers: tuple[str, ...]


def _motifs_agree(first: int, second: int) -> bool:
    return first > 0 and second > 0 and any(first * multiple == second or second * multiple == first for multiple in MOTIF_MULTIPLES)


def match_tr_loci(
    external_identifiers: Sequence[str],
    external_starts: NDArray,
    external_ends: NDArray,
    external_motifs: NDArray,
    locus_starts: NDArray,
    locus_ends: NDArray,
    locus_motifs: NDArray,
) -> LocusMatches:
    """One-to-one matches of one chromosome's external VNTR loci to its TR loci.

    A pair matches when the intervals overlap and the motif lengths agree up to a factor of
    1, 2 or 3 (TRF reports doubled or tripled periods). Pairs are taken by decreasing overlap,
    then external id, each locus and each external locus at most once.
    """
    external_starts = np.asarray(external_starts, dtype=np.int64)
    external_ends = np.asarray(external_ends, dtype=np.int64)
    locus_starts = np.asarray(locus_starts, dtype=np.int64)
    locus_ends = np.asarray(locus_ends, dtype=np.int64)
    candidates: list[tuple[int, str, int, int]] = []
    for external_row, (start, end) in enumerate(zip(external_starts.tolist(), external_ends.tolist())):
        low = np.searchsorted(locus_ends, start, side="right")
        high = np.searchsorted(locus_starts, end, side="left")
        for locus in range(low, high):
            if not _motifs_agree(int(external_motifs[external_row]), int(locus_motifs[locus])):
                continue
            overlap = min(end, int(locus_ends[locus])) - max(start, int(locus_starts[locus]))
            candidates.append((-overlap, external_identifiers[external_row], external_row, locus))
    candidates.sort()
    used_external: set[int] = set()
    used_loci: set[int] = set()
    matched: list[tuple[int, str]] = []
    for _, identifier, external_row, locus in candidates:
        if external_row in used_external or locus in used_loci:
            continue
        used_external.add(external_row)
        used_loci.add(locus)
        matched.append((locus, identifier))
    matched.sort()
    return LocusMatches(
        loci=np.asarray([locus for locus, _ in matched], dtype=np.int64),
        external_identifiers=tuple(identifier for _, identifier in matched),
    )


def squared_z(effects: NDArray, standard_errors: NDArray) -> NDArray:
    """The payload value: z^2 = (beta / se)^2, free of the effect scale and the allele orientation."""
    beta = np.asarray(effects, dtype=np.float64)
    error = np.asarray(standard_errors, dtype=np.float64)
    if beta.shape != error.shape or np.any(error <= 0.0) or not np.all(np.isfinite(beta)):
        raise ValueError("squared_z needs finite effects and positive standard errors, one each.")
    return (beta / error) ** 2

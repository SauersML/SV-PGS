"""Map an external SV/VNTR association catalogue onto the store's records and TR loci.

Store spec A4.13-A4.15 (design-svcontent's matcher). An external association is evidence about
one allele, so a match is one-to-one in both directions, and an unmatched record has no row:
absence is explicit, never a zero or a mean that would leak the catalogue's coverage into the
prior. The per-trait payload (z^2 of the external association) is joined per trait by
external id; this module builds the trait-independent maps and keeps them in the store at
``ext/<source>/rec_map/<chrom>`` (store row, tier, external id) and
``ext/<source>/locus_map/<chrom>`` (TR locus, external id).

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

The payload side is symmetric by construction, each source with its own
EB-learned weight: SNV records take the z^2 of an external SNV GWAS (Pan-UKB
EUR, lifted from GRCh37 by ``lift_positions``), SV records the z^2 of Bai et
al. 2026's SV GWAS and tandem-repeat length columns the z^2 of its VNTR GWAS.
``annotate_records`` joins one source's payload by key; a record without a row
stays absent (present flag 0).
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from sv_pgs._typing import BoolArray, F64Array, I64Array, NDArray
from sv_pgs.dosage_store import open_column, read_identifier_columns, write_column, write_identifier_columns
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


# Positions are below 2^32, so chromosome * stride + position is a unique site key.
_SITE_STRIDE = np.int64(1 << 32)


@dataclass(frozen=True)
class ExternalAssociations:
    """One external release: a unique key per tested variant and its squared z statistic."""

    keys: NDArray
    squared_z: F64Array


def _unique(keys: NDArray, values: NDArray, source: str) -> ExternalAssociations:
    keys = np.asarray(keys, dtype=object)
    if pd.Index(keys).has_duplicates:
        raise ValueError(f"{source} lists a variant more than once.")
    return ExternalAssociations(keys=keys, squared_z=np.asarray(values, dtype=np.float64))


def read_bai_structural(path: Path) -> ExternalAssociations:
    """A Bai 2026 SV release (columns CHR SNP POS A1 A2 N AF1 BETA SE P MAF), keyed by its SV id."""
    table = pd.read_csv(path, sep="\t", usecols=["SNP", "BETA", "SE"], dtype={"SNP": str})
    return _unique(table["SNP"].to_numpy(), squared_z(table["BETA"].to_numpy(), table["SE"].to_numpy()), str(path))


def read_bai_tandem_repeat(path: Path) -> ExternalAssociations:
    """A Bai 2026 VNTR release (effect per repeat unit), keyed by its locus id."""
    table = pd.read_csv(path, sep="\t", usecols=["ID", "BETA", "SE"], dtype={"ID": str})
    return _unique(table["ID"].to_numpy(), squared_z(table["BETA"].to_numpy(), table["SE"].to_numpy()), str(path))


@dataclass(frozen=True)
class SnvAssociations:
    """An SNV release before keying: its sites and their squared z."""

    chromosome: I64Array
    position: I64Array
    reference: NDArray
    alternate: NDArray
    squared_z: F64Array


def read_panukb_eur(path: Path) -> SnvAssociations:
    """A slim Pan-UKB EUR file (chr pos ref alt beta_EUR se_EUR ... low_confidence_EUR), autosomes only.

    Pan-UKB flags variants whose EUR statistics fail its QC as low confidence;
    those carry no usable evidence and are left out, so they become absent.
    """
    table = pd.read_csv(path, sep="\t", usecols=["chr", "pos", "ref", "alt", "beta_EUR", "se_EUR", "low_confidence_EUR"],
                        dtype={"chr": str, "ref": str, "alt": str})
    confident = ~table["low_confidence_EUR"].astype(str).str.lower().isin(["true", "1"])
    autosomal = table["chr"].str.fullmatch(r"\d+")
    table = table[confident & autosomal]
    return SnvAssociations(
        chromosome=table["chr"].astype(np.int64).to_numpy(),
        position=table["pos"].to_numpy(dtype=np.int64),
        reference=table["ref"].to_numpy(dtype=object),
        alternate=table["alt"].to_numpy(dtype=object),
        squared_z=squared_z(table["beta_EUR"].to_numpy(), table["se_EUR"].to_numpy()),
    )


def lift_positions(
    chromosome: NDArray, position: NDArray, map_chromosome: NDArray, map_from: NDArray, map_to: NDArray
) -> tuple[I64Array, BoolArray]:
    """Positions after a site map on the same chromosome; ``mapped`` is False (position -1) where the map has no entry."""
    map_key = np.asarray(map_chromosome, dtype=np.int64) * _SITE_STRIDE + np.asarray(map_from, dtype=np.int64)
    if map_key.shape[0] == 0:
        raise ValueError("the site map is empty.")
    order = np.argsort(map_key, kind="stable")
    sorted_key = map_key[order]
    if np.any(np.diff(sorted_key) == 0):
        raise ValueError("the site map lists a source position more than once.")
    query = np.asarray(chromosome, dtype=np.int64) * _SITE_STRIDE + np.asarray(position, dtype=np.int64)
    slot = np.minimum(np.searchsorted(sorted_key, query), sorted_key.shape[0] - 1)
    mapped = sorted_key[slot] == query
    lifted = np.full(query.shape[0], -1, dtype=np.int64)
    lifted[mapped] = np.asarray(map_to, dtype=np.int64)[order][slot[mapped]]
    return lifted, mapped


def snv_keys(chromosome: NDArray, position: NDArray, reference: NDArray, alternate: NDArray) -> NDArray:
    """chromosome:position:ref:alt keys, the join key between an SNV release and store records."""
    return np.array([f"{int(code)}:{int(site)}:{ref}:{alt}" for code, site, ref, alt in
                     zip(chromosome, position, reference, alternate)], dtype=object)


def keyed_snv_associations(associations: SnvAssociations, position: NDArray, mapped: NDArray) -> ExternalAssociations:
    """Key an SNV release at its (lifted) positions, keeping only the sites the lift mapped."""
    kept = np.asarray(mapped, dtype=bool)
    keys = snv_keys(associations.chromosome[kept], np.asarray(position)[kept],
                    associations.reference[kept], associations.alternate[kept])
    return _unique(keys, associations.squared_z[kept], "the lifted SNV release")


@dataclass(frozen=True)
class RecordAnnotation:
    """One external source, per record: log(1 + z^2) where present, and the present flag."""

    log_squared_z: F64Array
    present: BoolArray


def annotate_records(record_keys: NDArray, associations: ExternalAssociations) -> RecordAnnotation:
    """Exact key join of records to one source's payload; records without a row are absent."""
    positions = pd.Index(associations.keys).get_indexer(pd.Index(np.asarray(record_keys, dtype=object)))
    present = positions >= 0
    values = np.zeros(positions.shape[0])
    values[present] = np.log1p(associations.squared_z[positions[present]])
    return RecordAnnotation(log_squared_z=values, present=present)


def _map_directory(root: Path, source: str, kind: str, chromosome: str) -> Path:
    return Path(root) / "ext" / source / kind / chromosome


def write_record_map(root: Path, source: str, chromosome: str, matches: RecordMatches) -> None:
    """One source's record map of one chromosome: matched store rows, tiers and external ids."""
    directory = _map_directory(root, source, "rec_map", chromosome)
    write_column(directory / "rec_idx", matches.store_rows.astype(np.uint32))
    write_column(directory / "tier", matches.tiers.astype(np.uint8), {"legend": ["", "exact", "sequence", "coordinate"]})
    write_identifier_columns(directory / "id_bytes", directory / "id_offsets", matches.external_identifiers)


def read_record_map(root: Path, source: str, chromosome: str) -> RecordMatches:
    """A stored record map; ``displaced_pairs`` is a matching diagnostic the store does not keep."""
    directory = _map_directory(root, source, "rec_map", chromosome)
    return RecordMatches(
        store_rows=np.asarray(open_column(directory / "rec_idx")[0], dtype=np.int64),
        external_identifiers=read_identifier_columns(directory / "id_bytes", directory / "id_offsets"),
        tiers=np.asarray(open_column(directory / "tier")[0], dtype=np.uint8),
        displaced_pairs=0,
    )


def write_locus_map(root: Path, source: str, chromosome: str, matches: LocusMatches) -> None:
    """One source's TR-locus map of one chromosome: matched loci and their external locus ids."""
    directory = _map_directory(root, source, "locus_map", chromosome)
    write_column(directory / "tr_locus", matches.loci.astype(np.uint32))
    write_identifier_columns(directory / "id_bytes", directory / "id_offsets", matches.external_identifiers)


def read_locus_map(root: Path, source: str, chromosome: str) -> LocusMatches:
    directory = _map_directory(root, source, "locus_map", chromosome)
    return LocusMatches(
        loci=np.asarray(open_column(directory / "tr_locus")[0], dtype=np.int64),
        external_identifiers=read_identifier_columns(directory / "id_bytes", directory / "id_offsets"),
    )


def mapped_keys(rows: NDArray, external_identifiers: Sequence[str], count: int) -> NDArray:
    """Each of ``count`` store rows (records or loci) keyed by its matched external id, or by ""
    where it has none, which ``annotate_records`` then leaves absent."""
    keys = np.full(count, "", dtype=object)
    keys[np.asarray(rows, dtype=np.int64)] = list(external_identifiers)
    return keys

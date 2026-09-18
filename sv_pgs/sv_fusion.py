"""Find the SVs that two genotype sources both call, for fusing them into one column.

The imputed panel (sequence-resolved, <= 10 kb) and a GATK-SV call set overlap
for roughly 1-10 kb deletions, duplications and insertions. A pair of records is
a candidate for the same event when, on the same chromosome and of compatible
kinds, their sizes agree within a factor of two and either their intervals
overlap reciprocally by at least half (deletions, complex SVs) or their
breakpoints lie within 100 bp (insertions, and an insertion against a tandem
duplication). These are GATK-SV's own re-clustering rules. Whether a candidate
is really one event is then decided from the genotypes by the two-source
calibration.

Coordinates: ``starts`` is the first affected base (an insertion's point, the
base after its anchor), ``ends`` is exclusive (``start + 1`` for insertions),
``sizes`` the affected span or inserted length. A sequence-resolved source
writes a tandem duplication as an inserted copy, so its DUP is a point like an
insertion; a GATK-SV DUP is the duplicated interval, and an inserted copy can
sit at either of its ends.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sv_pgs._typing import F64Array, I64Array, NDArray

MINIMUM_SIZE_RATIO = 0.5
MINIMUM_RECIPROCAL_OVERLAP = 0.5
MAXIMUM_BREAKPOINT_DISTANCE = 100
# (kind in the first source, kind in the second) that may describe one event.
COMPATIBLE_KINDS = frozenset(
    {
        ("DEL", "DEL"),
        ("CPX", "CPX"),
        ("INS", "INS"),
        ("INS", "DUP"),
        ("DUP", "DUP"),
        ("DUP", "INS"),
    }
)


@dataclass(frozen=True, slots=True)
class SvSites:
    """SV records of one source, one row per record, on the common coordinates."""

    chromosomes: NDArray
    starts: I64Array
    ends: I64Array
    sizes: I64Array
    kinds: NDArray
    duplications_are_insertions: bool

    def __post_init__(self) -> None:
        lengths = {len(self.chromosomes), len(self.starts), len(self.ends), len(self.sizes), len(self.kinds)}
        if len(lengths) != 1:
            raise ValueError("every SvSites column needs one row per record.")
        if np.any(self.ends <= self.starts) or np.any(self.sizes < 1):
            raise ValueError("SvSites need end > start and size >= 1.")


@dataclass(frozen=True, slots=True)
class SvCandidatePairs:
    """Record pairs that may be one event: row indices into each source's SvSites."""

    first_rows: I64Array
    second_rows: I64Array
    size_ratios: F64Array
    # Reciprocal overlap for interval pairs; NaN when a point is involved.
    reciprocal_overlaps: F64Array
    # Breakpoint distance when a point is involved; -1 for interval pairs.
    breakpoint_distances: I64Array


def _is_point(sites: SvSites, kind: str) -> bool:
    return kind == "INS" or (kind == "DUP" and sites.duplications_are_insertions)


def _pair_geometry(
    first: SvSites, first_row: int, second: SvSites, second_row: int
) -> tuple[float, float, int] | None:
    first_kind = str(first.kinds[first_row])
    second_kind = str(second.kinds[second_row])
    if (first_kind, second_kind) not in COMPATIBLE_KINDS:
        return None
    size_ratio = min(first.sizes[first_row], second.sizes[second_row]) / max(first.sizes[first_row], second.sizes[second_row])
    if size_ratio < MINIMUM_SIZE_RATIO:
        return None
    first_point = _is_point(first, first_kind)
    second_point = _is_point(second, second_kind)
    if not first_point and not second_point:
        overlap = min(first.ends[first_row], second.ends[second_row]) - max(first.starts[first_row], second.starts[second_row])
        longer = max(first.ends[first_row] - first.starts[first_row], second.ends[second_row] - second.starts[second_row])
        reciprocal_overlap = max(overlap, 0) / longer
        if reciprocal_overlap < MINIMUM_RECIPROCAL_OVERLAP:
            return None
        return float(size_ratio), float(reciprocal_overlap), -1
    point_sites, point_row, other_sites, other_row, other_point = (
        (first, first_row, second, second_row, second_point) if first_point else (second, second_row, first, first_row, first_point)
    )
    point = int(point_sites.starts[point_row])
    if other_point:
        distance = abs(point - int(other_sites.starts[other_row]))
    else:
        # A copy inserted next to the duplicated interval sits at either end.
        distance = min(abs(point - int(other_sites.starts[other_row])), abs(point - int(other_sites.ends[other_row])))
    if distance > MAXIMUM_BREAKPOINT_DISTANCE:
        return None
    return float(size_ratio), float("nan"), distance


def candidate_pairs(first: SvSites, second: SvSites) -> SvCandidatePairs:
    """All record pairs of two sources that pass the size and position rules."""
    first_rows: list[int] = []
    second_rows: list[int] = []
    size_ratios: list[float] = []
    reciprocal_overlaps: list[float] = []
    breakpoint_distances: list[int] = []
    for chromosome in np.intersect1d(np.unique(first.chromosomes), np.unique(second.chromosomes)):
        first_on = np.flatnonzero(first.chromosomes == chromosome)
        order = first_on[np.argsort(first.starts[first_on], kind="stable")]
        sorted_starts = first.starts[order]
        # Any first-source record that can pair with a second-source record
        # starts within its longest span (plus the breakpoint slack) of it.
        reach = int((first.ends[first_on] - first.starts[first_on]).max()) + MAXIMUM_BREAKPOINT_DISTANCE
        for second_row in np.flatnonzero(second.chromosomes == chromosome):
            low = np.searchsorted(sorted_starts, second.starts[second_row] - reach, side="left")
            high = np.searchsorted(sorted_starts, second.ends[second_row] + reach, side="right")
            for first_row in order[low:high]:
                geometry = _pair_geometry(first, int(first_row), second, int(second_row))
                if geometry is None:
                    continue
                first_rows.append(int(first_row))
                second_rows.append(int(second_row))
                size_ratios.append(geometry[0])
                reciprocal_overlaps.append(geometry[1])
                breakpoint_distances.append(geometry[2])
    return SvCandidatePairs(
        first_rows=np.asarray(first_rows, dtype=np.int64),
        second_rows=np.asarray(second_rows, dtype=np.int64),
        size_ratios=np.asarray(size_ratios, dtype=np.float64),
        reciprocal_overlaps=np.asarray(reciprocal_overlaps, dtype=np.float64),
        breakpoint_distances=np.asarray(breakpoint_distances, dtype=np.int64),
    )

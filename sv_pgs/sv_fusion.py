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


# ---------------------------------------------------------------------------
# Two-source calibration and the fused column
# ---------------------------------------------------------------------------
#
# Model, per matched locus with latent ALT count g (mean 2p, variance G):
#   first source  A = the imputed DS, a calibrated posterior mean: E[g | A] = A,
#                 so E[A] = 2p and Cov(A, g) = Var(A) = V_A (Berkson error);
#   second source B = kappa g + a + e with e independent of (g, A) (classical
#                 error; the intercept a absorbs false-positive calls).
# Then Cov(A, B) = kappa V_A identifies kappa without truth, and within-group
# HWE gives G. The implied covariance of (g, A, B),
#   [[G, V_A, kappa G], [V_A, V_A, kappa V_A], [kappa G, kappa V_A, V_B]],
# is a valid covariance exactly when r2_A = V_A / G <= 1 and
# r2_B = kappa^2 G / V_B <= 1; the fused column is the best linear predictor
#   g_hat = 2p + w_A (A - m_A) + w_B (B - m_B),
#   w = [V_A, kappa G] Sigma^-1,  Sigma = [[V_A, kappa V_A], [kappa V_A, V_B]],
# with reliability r2_fused = w . [V_A, kappa G] / G. Where B is a no-call the
# predictor from A alone is A itself. (A classical error model for A as well,
# m_s = 2p kappa_s, does not hold for a posterior mean: it misreads the
# reliabilities of both sources.)

MINIMUM_PAIRING_Z = 5.0


@dataclass(frozen=True, slots=True)
class TwoSourceCalibration:
    """Truth-free calibration of one matched locus from its two sources."""

    sample_count: int
    first_mean: float
    second_mean: float
    genotype_variance: float
    second_slope: float
    first_reliability: float
    second_reliability: float
    fused_reliability: float
    first_weight: float
    second_weight: float
    pairing_z: float

    @property
    def accepted(self) -> bool:
        """The two records are one event, and the model's covariance is valid."""
        return (
            self.pairing_z >= MINIMUM_PAIRING_Z
            and 0.0 < self.first_reliability <= 1.0
            and 0.0 < self.second_reliability <= 1.0
        )


def _genotype_variance(first_dosage: F64Array, group_labels: NDArray) -> float:
    """Var(g) under HWE within each group: the mixture of the groups' binomials.

    Var(g) = sum_k w_k 2 p_k (1 - p_k) + sum_k w_k (2 p_k - 2 p)^2, with p_k from
    the calibrated source's group means, E[A | group] = 2 p_k.
    """
    overall_mean = float(first_dosage.mean())
    variance = 0.0
    for label in np.unique(group_labels):
        in_group = group_labels == label
        weight = float(in_group.mean())
        group_frequency = float(first_dosage[in_group].mean()) / 2.0
        variance += weight * (2.0 * group_frequency * (1.0 - group_frequency) + (2.0 * group_frequency - overall_mean) ** 2)
    return variance


def calibrate_two_sources(
    first_dosage: F64Array,
    second_values: NDArray,
    second_observed: NDArray,
    group_labels: NDArray,
) -> TwoSourceCalibration:
    """Calibrate a matched locus on the samples where both sources are observed.

    ``first_dosage`` is the imputed DS (calibrated), ``second_values`` the
    other source's allele count, ``group_labels`` the genetic-similarity group
    of each sample (one label everywhere for a single group).
    """
    observed = np.asarray(second_observed, dtype=bool)
    first = np.asarray(first_dosage, dtype=np.float64)[observed]
    second = np.asarray(second_values, dtype=np.float64)[observed]
    sample_count = int(observed.sum())
    first_mean = float(first.mean())
    second_mean = float(second.mean())
    first_variance = float(first.var())
    second_variance = float(second.var())
    covariance = float(np.mean((first - first_mean) * (second - second_mean)))
    genotype_variance = _genotype_variance(first, np.asarray(group_labels)[observed])
    second_slope = covariance / first_variance
    correlation = covariance / np.sqrt(first_variance * second_variance)
    sigma = np.array([[first_variance, covariance], [covariance, second_variance]])
    cross = np.array([first_variance, second_slope * genotype_variance])
    first_weight, second_weight = np.linalg.solve(sigma, cross)
    return TwoSourceCalibration(
        sample_count=sample_count,
        first_mean=first_mean,
        second_mean=second_mean,
        genotype_variance=genotype_variance,
        second_slope=second_slope,
        first_reliability=first_variance / genotype_variance,
        second_reliability=second_slope**2 * genotype_variance / second_variance,
        fused_reliability=float(first_weight * cross[0] + second_weight * cross[1]) / genotype_variance,
        first_weight=float(first_weight),
        second_weight=float(second_weight),
        pairing_z=float(np.arctanh(np.clip(correlation, -1.0 + 1e-15, 1.0 - 1e-15)) * np.sqrt(sample_count - 3)),
    )


def fused_dosage(
    calibration: TwoSourceCalibration,
    first_dosage: F64Array,
    second_values: NDArray,
    second_observed: NDArray,
) -> F64Array:
    """The fused column: the linear posterior mean of g given both sources.

    Samples where the second source is a no-call keep the imputed DS.
    """
    if not calibration.accepted:
        raise ValueError("the two records failed the calibration check; keep them as separate columns.")
    first = np.asarray(first_dosage, dtype=np.float64)
    observed = np.asarray(second_observed, dtype=bool)
    fused = first.copy()
    fused[observed] = (
        calibration.first_mean
        + calibration.first_weight * (first[observed] - calibration.first_mean)
        + calibration.second_weight * (np.asarray(second_values, dtype=np.float64)[observed] - calibration.second_mean)
    )
    return fused

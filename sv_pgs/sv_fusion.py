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

import dataclasses
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from sv_pgs._typing import F64Array, I64Array, NDArray
from sv_pgs.variant_typing import sequence_resolved_kind_and_length, trimmed_allele_cores

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


def sequence_resolved_sites(chromosomes: NDArray, positions: I64Array, refs: list[str], alts: list[str]) -> SvSites:
    """Sites of sequence-resolved SV alleles (an imputed panel) on the common coordinates.

    The kind (DEL, INS or CPX) and size come from the allele shape by the
    strata rule; the first affected base is POS plus the shared prefix, so a
    deletion spans its deleted core and an insertion is the point where its
    core goes in. A tandem duplication written this way is an insertion.
    """
    starts = np.empty(len(refs), dtype=np.int64)
    ends = np.empty(len(refs), dtype=np.int64)
    sizes = np.empty(len(refs), dtype=np.int64)
    kinds = []
    for row, (position, ref, alt) in enumerate(zip(np.asarray(positions).tolist(), refs, alts)):
        prefix, ref_core, _ = trimmed_allele_cores(ref, alt)
        kind, length = sequence_resolved_kind_and_length(ref, alt)
        starts[row] = position + prefix
        ends[row] = starts[row] + (1 if kind == "INS" else max(ref_core, 1))
        sizes[row] = max(int(round(length)), 1)
        kinds.append(kind)
    return SvSites(
        chromosomes=np.asarray(chromosomes),
        starts=starts,
        ends=ends,
        sizes=sizes,
        kinds=np.asarray(kinds),
        duplications_are_insertions=True,
    )


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
# Model, per matched locus with latent ALT count g (variance V_G), for the
# imputed dosage A and the other source's call B:
#   E[A | g] = alpha_A + rho_A g,   E[B | g] = alpha_B + rho_B g,
# with the two errors uncorrelated given g, and A's error independent of B's
# no-calls given g. Both lines are exact for any per-haplotype error model
# because a haplotype is 0 or 1. A is mean-calibrated (E[A] = E[g] after the
# background correction); that holds whether A is a calibrated posterior mean
# (Berkson, rho_A = r2_A) or a confident posterior draw (rho_A = r_A), and in
# general rho_A = sqrt(r2_A V_A / V_G). The moments of (A, B) leave exactly one
# number unidentified: A's reliability r2_A = corr^2(A, g). It is an input, from
# ``shrunk_imputed_reliabilities``. Given it, tau_A^2 = V_A (1 - r2_A),
# alpha_A = (1 - rho_A) m_A and, per group (B called / B a no-call),
# mu_G|group = (m_A|group - alpha_A) / rho_A. Where B is called the fused
# column is the best linear predictor from both sources,
#   F = mu_G|obs + w . (A - m_A|obs, B - m_B|obs),
#   w = Sigma_obs^-1 [rho_A V_G|obs, C_AB / rho_A],  V_G|obs = (V_A|obs - tau_A^2) / rho_A^2,
# and where B is a no-call it is the recalibrated imputed dosage
#   F = mu_G|miss + kappa_A (A - m_A|miss),  kappa_A = r2_A / rho_A.
# The group means absorb a no-call rate that depends on the genotype (GATK-SV
# sets uncertain carrier calls to no-call). V_G = 2p(1 - p), p = m_A / 2, only
# fixes the allele scale: the fused direction and its reliability
# r2_F = Var(F) / V_G do not depend on it. r2_B = corr^2(A, B) / r2_A.
# (theory-genouncertainty's sim F: with the locus's true r2_A the fused r2 is
# within 4e-4 of the truth-fitted oracle in Berkson, draw, tempered and
# deflated strata. Treating a draw-like A as a calibrated posterior mean, the
# closure this replaces, read r2_A = 1 and lost 0.31 of fused r2 in VNTRs.)

MINIMUM_PAIRING_Z = 5.0
# Fisher's z needs n > 3 for its standard error 1 / sqrt(n - 3).
MINIMUM_CALIBRATION_SAMPLES = 4
# The other source's implied reliability may pass 1 by sampling slack; past
# this a pair is not one event measured with independent errors (the
# cross-truth rejection distribution design-svcontent measured).
MAXIMUM_SECOND_RELIABILITY = 1.25


@dataclass(frozen=True, slots=True)
class TwoSourceCalibration:
    """One matched locus, calibrated from its two sources given A's reliability."""

    sample_count: int
    first_reliability: float
    first_slope: float
    genotype_variance: float
    observed_first_mean: float
    observed_second_mean: float
    missing_first_mean: float
    observed_genotype_mean: float
    missing_genotype_mean: float
    first_weight: float
    second_weight: float
    missing_slope: float
    # Least-squares slope of B on A where both are called: the prediction of
    # a no-call of B from A, for a record that stays a column of its own.
    second_slope: float
    second_reliability: float
    fused_reliability: float
    pairing_z: float

    @property
    def accepted(self) -> bool:
        """The records are one event and the model's covariance is valid."""
        return (
            self.pairing_z >= MINIMUM_PAIRING_Z
            and 0.0 < self.first_reliability <= 1.0
            and 0.0 < self.second_reliability <= MAXIMUM_SECOND_RELIABILITY
        )

    @property
    def second_reliability_flagged(self) -> bool:
        """Accepted with B's implied reliability past 1 (within the sampling slack)."""
        return self.accepted and self.second_reliability > 1.0


def _without_pairing_evidence(sample_count: int, first_reliability: float) -> TwoSourceCalibration:
    # A source constant on the shared samples, a monomorphic A, or fewer than
    # four shared samples leave the correlation undefined: no evidence that
    # the records pair.
    undefined = float("nan")
    return TwoSourceCalibration(
        sample_count=sample_count,
        first_reliability=first_reliability,
        first_slope=undefined,
        genotype_variance=undefined,
        observed_first_mean=undefined,
        observed_second_mean=undefined,
        missing_first_mean=undefined,
        observed_genotype_mean=undefined,
        missing_genotype_mean=undefined,
        first_weight=undefined,
        second_weight=undefined,
        missing_slope=undefined,
        second_slope=undefined,
        second_reliability=undefined,
        fused_reliability=undefined,
        pairing_z=0.0,
    )


def _fisher_z(correlation: float, sample_count: int) -> float:
    # Rounding can put a perfect correlation a hair past 1; |r| = 1 is infinite evidence.
    with np.errstate(divide="ignore"):
        return float(np.arctanh(np.clip(correlation, -1.0, 1.0)) * np.sqrt(sample_count - 3))


def _fused_values(calibration: TwoSourceCalibration, first: F64Array, second: F64Array, observed: NDArray) -> F64Array:
    fused = np.empty_like(first)
    fused[observed] = (
        calibration.observed_genotype_mean
        + calibration.first_weight * (first[observed] - calibration.observed_first_mean)
        + calibration.second_weight * (second[observed] - calibration.observed_second_mean)
    )
    missing = ~observed
    fused[missing] = calibration.missing_genotype_mean + calibration.missing_slope * (
        first[missing] - calibration.missing_first_mean
    )
    return fused


def calibrate_two_sources(
    first_dosage: F64Array,
    second_values: NDArray,
    second_observed: NDArray,
    first_reliability: float,
) -> TwoSourceCalibration:
    """Calibrate a matched locus: the imputed dosage (all samples), the other source where called.

    ``first_reliability`` is the imputed dosage's r2_A at this locus.
    """
    first = np.asarray(first_dosage, dtype=np.float64)
    second = np.asarray(second_values, dtype=np.float64)
    observed = np.asarray(second_observed, dtype=bool)
    sample_count = int(observed.sum())
    first_mean = float(first.mean())
    genotype_variance = first_mean - first_mean**2 / 2
    first_called = first[observed]
    second_called = second[observed]
    if (
        sample_count < MINIMUM_CALIBRATION_SAMPLES
        or genotype_variance <= 0.0
        or np.ptp(first_called) == 0.0
        or np.ptp(second_called) == 0.0
    ):
        return _without_pairing_evidence(sample_count, first_reliability)
    first_variance = float(first.var())
    first_slope = float(np.sqrt(first_reliability * first_variance / genotype_variance))
    first_noise = first_variance * (1.0 - first_reliability)
    first_intercept = (1.0 - first_slope) * first_mean

    observed_first_mean = float(first_called.mean())
    observed_second_mean = float(second_called.mean())
    sigma = np.cov(np.vstack([first_called, second_called]), bias=True)
    observed_genotype_variance = max(float(sigma[0, 0]) - first_noise, 0.0) / first_slope**2
    cross = np.array([first_slope * observed_genotype_variance, float(sigma[0, 1]) / first_slope])
    # Least squares keeps a perfectly correlated pair (singular Sigma) finite.
    first_weight, second_weight = np.linalg.lstsq(sigma, cross, rcond=None)[0]
    missing = ~observed
    missing_first_mean = float(first[missing].mean()) if missing.any() else observed_first_mean
    correlation = float(sigma[0, 1] / np.sqrt(sigma[0, 0] * sigma[1, 1]))
    calibration = TwoSourceCalibration(
        sample_count=sample_count,
        first_reliability=first_reliability,
        first_slope=first_slope,
        genotype_variance=genotype_variance,
        observed_first_mean=observed_first_mean,
        observed_second_mean=observed_second_mean,
        missing_first_mean=missing_first_mean,
        observed_genotype_mean=(observed_first_mean - first_intercept) / first_slope,
        missing_genotype_mean=(missing_first_mean - first_intercept) / first_slope,
        first_weight=float(first_weight),
        second_weight=float(second_weight),
        missing_slope=first_reliability / first_slope,
        second_slope=float(sigma[0, 1] / sigma[0, 0]),
        second_reliability=correlation**2 / first_reliability,
        fused_reliability=float("nan"),
        pairing_z=_fisher_z(correlation, sample_count),
    )
    fused = _fused_values(calibration, first, second, observed)
    return dataclasses.replace(calibration, fused_reliability=float(fused.var() / genotype_variance))


def fused_dosage(
    calibration: TwoSourceCalibration,
    first_dosage: F64Array,
    second_values: NDArray,
    second_observed: NDArray,
) -> F64Array:
    """The fused column on the allele-count scale: both sources where the other calls, else recalibrated A."""
    if not calibration.accepted:
        raise ValueError("the two records failed the calibration check; keep them as separate columns.")
    return _fused_values(
        calibration,
        np.asarray(first_dosage, dtype=np.float64),
        np.asarray(second_values, dtype=np.float64),
        np.asarray(second_observed, dtype=bool),
    )


def resolve_one_to_one(pairs: SvCandidatePairs, calibrations: Sequence[TwoSourceCalibration]) -> I64Array:
    """Indices of the candidate pairs to fuse, each record used at most once.

    Several popped alleles of one locus can match one GATK-SV record, and one
    allele several records. Accepted pairs are taken in order of decreasing
    pairing evidence (Fisher z); a pair is skipped once either record is
    taken. Records left unpaired stay separate columns.
    """
    if len(calibrations) != pairs.first_rows.shape[0]:
        raise ValueError("resolve_one_to_one needs one calibration per candidate pair.")
    accepted = [index for index, calibration in enumerate(calibrations) if calibration.accepted]
    accepted.sort(key=lambda index: -calibrations[index].pairing_z)
    taken_first: set[int] = set()
    taken_second: set[int] = set()
    chosen: list[int] = []
    for index in accepted:
        first_row = int(pairs.first_rows[index])
        second_row = int(pairs.second_rows[index])
        if first_row in taken_first or second_row in taken_second:
            continue
        taken_first.add(first_row)
        taken_second.add(second_row)
        chosen.append(index)
    return np.asarray(sorted(chosen), dtype=np.int64)


# ---------------------------------------------------------------------------
# The imputed source's reliability at each locus
# ---------------------------------------------------------------------------
#
# r2_A comes from three places (theory-genouncertainty's spec):
#   (a) a stratum verified to be Berkson (truth slope kappa within 5% of 1):
#       per locus r2_A = V_A / V_G, exact and free of B's false positives;
#   (b) the per-locus mean anchor: with A mean-calibrated and B = alpha_B +
#       rho_B g + e_B, rho_B = (m_B - alpha_B) / m_A, rho_A = C_AB / (rho_B V_G)
#       and r2_A = rho_A^2 V_G / V_A, i.e.
#         log r2_A = 2 log C + 2 log m_A - 2 log(m_B - alpha_B) - log V_G - log V_A,
#       where alpha_B = 2 f (1 - p) from B's per-haplotype false-positive rate f;
#   (c) the truth-calibrated reliability model's prediction mu (the store's
#       r2_truth), unbiased but blind to per-locus spread.
# The anchor's error is mostly systematic (a false-positive intercept that
# differs by locus, B's no-call selection); only truth loci measure it. On
# truth loci where B is also called on the long-read samples, e_t = x_t - y_t
# (y_t the log triad r2_A) gives a stratum bias b and an excess variance
# omega^2 = max(0, var e - mean sampling variances of x and y). The locus
# value is the normal-normal posterior mean on the log scale,
#   log r2_A = mu + B (x - b - mu),  B = s^2 / (s^2 + v + omega^2),
# with s^2 = max(0, var(x - b - mu) - mean v - omega^2) the true between-locus
# spread and v the anchor's sampling plus intercept variance; capped at 1.
# (sim F: fused-r2 gap to the oracle -0.001 to -0.0075 with random no-calls,
# against -0.011 to -0.087 for the stratum value alone; the weight falls to
# about 0 by itself when B's false-positive rate is large.)


@dataclass(frozen=True, slots=True)
class AnchorEstimate:
    """The mean anchor's log r2_A at one locus and its variance (sampling plus intercept)."""

    log_reliability: float
    variance: float


@dataclass(frozen=True, slots=True)
class AnchorErrorModel:
    """The mean anchor's error in one stratum, measured on truth loci.

    ``berkson`` marks a stratum whose truth slope showed the imputed dosage is
    a calibrated posterior mean; there r2_A = V_A / V_G per locus.
    """

    bias: float
    excess_variance: float
    berkson: bool


def berkson_reliability(first_dosage: F64Array) -> float:
    """r2_A = V_A / V_G of a calibrated posterior-mean dosage, V_G = 2p(1 - p)."""
    first = np.asarray(first_dosage, dtype=np.float64)
    first_mean = float(first.mean())
    genotype_variance = first_mean - first_mean**2 / 2
    if genotype_variance <= 0.0:
        return float("nan")
    return min(float(first.var()) / genotype_variance, 1.0)


def mean_anchor(
    first_dosage: F64Array,
    second_values: NDArray,
    second_observed: NDArray,
    false_positive_rate: float,
    false_positive_rate_variance: float,
) -> AnchorEstimate:
    """The per-locus mean anchor on the samples where both sources are called.

    ``false_positive_rate`` is the other source's per-haplotype false-positive
    rate for this record's class, and ``false_positive_rate_variance`` its
    uncertainty across the class's records.
    """
    observed = np.asarray(second_observed, dtype=bool)
    first = np.asarray(first_dosage, dtype=np.float64)[observed]
    second = np.asarray(second_values, dtype=np.float64)[observed]
    undefined = AnchorEstimate(log_reliability=float("nan"), variance=float("inf"))
    if first.shape[0] < MINIMUM_CALIBRATION_SAMPLES:
        return undefined
    first_mean = float(first.mean())
    second_mean = float(second.mean())
    frequency = first_mean / 2
    intercept = 2.0 * false_positive_rate * (1.0 - frequency)
    signal = second_mean - intercept
    first_centred = first - first_mean
    second_centred = second - second_mean
    covariance = float(np.mean(first_centred * second_centred))
    first_variance = float(np.mean(first_centred**2))
    genotype_variance = first_mean - first_mean**2 / 2
    if min(covariance, signal, genotype_variance, first_variance, first_mean) <= 0.0:
        return undefined
    log_reliability = (
        2.0 * np.log(covariance)
        + 2.0 * np.log(first_mean)
        - 2.0 * np.log(signal)
        - np.log(genotype_variance)
        - np.log(first_variance)
    )
    influence = (
        2.0 * (first_centred * second_centred - covariance) / covariance
        + 2.0 * first_centred / first_mean
        - 2.0 * second_centred / signal
        - (1.0 - first_mean) * first_centred / genotype_variance
        - (first_centred**2 - first_variance) / first_variance
    )
    intercept_variance = 4.0 * (4.0 * (1.0 - frequency) ** 2 * false_positive_rate_variance) / signal**2
    return AnchorEstimate(
        log_reliability=float(log_reliability),
        variance=float(np.mean(influence**2) / first.shape[0]) + intercept_variance,
    )


def _log_triad(cross: NDArray) -> NDArray:
    """log r2_A from centred cross-products of (A, T1, T2), [..., 3, 3]; NaN where the triad is undefined."""
    numerator = cross[..., 0, 1] * cross[..., 0, 2]
    denominator = cross[..., 0, 0] * cross[..., 1, 2]
    defined = (numerator > 0.0) & (denominator > 0.0)
    ratio = np.divide(numerator, denominator, out=np.full(numerator.shape, np.nan), where=defined)
    # r2_A is a squared correlation, so it cannot pass 1.
    return np.log(np.minimum(ratio, 1.0))


def triad_log_reliability(first_dosage: F64Array, first_truth: F64Array, second_truth: F64Array) -> tuple[float, float]:
    """log r2_A against two independent noisy truths, r(A,T1) r(A,T2) / r(T1,T2), and its jackknife variance.

    The delete-one jackknife is exact and needs no pass per sample: with z_i the samples centred
    on the full mean and C their cross-products, leaving sample i out gives C - n/(n-1) z_i z_i'.
    """
    values = np.vstack([np.asarray(column, dtype=np.float64) for column in (first_dosage, first_truth, second_truth)])
    sample_count = values.shape[1]
    centred = (values - values.mean(axis=1, keepdims=True)).T
    cross = centred.T @ centred
    leave_one_out = _log_triad(cross - sample_count / (sample_count - 1) * centred[:, :, None] * centred[:, None, :])
    variance = (sample_count - 1) / sample_count * float(np.sum((leave_one_out - leave_one_out.mean()) ** 2))
    return float(_log_triad(cross)), variance


def fit_anchor_error_model(
    anchors: Sequence[AnchorEstimate],
    truth_log_reliabilities: F64Array,
    truth_variances: F64Array,
    berkson: bool,
) -> AnchorErrorModel:
    """The stratum's anchor bias and excess error variance, from its truth loci."""
    anchor_values = np.array([anchor.log_reliability for anchor in anchors], dtype=np.float64)
    anchor_variances = np.array([anchor.variance for anchor in anchors], dtype=np.float64)
    truth_values = np.asarray(truth_log_reliabilities, dtype=np.float64)
    truth_spread = np.asarray(truth_variances, dtype=np.float64)
    usable = np.isfinite(anchor_values) & np.isfinite(anchor_variances) & np.isfinite(truth_values) & np.isfinite(truth_spread)
    if usable.sum() < 2:
        raise ValueError("an anchor error model needs at least two truth loci with finite anchors and triads.")
    errors = anchor_values[usable] - truth_values[usable]
    excess = float(errors.var() - anchor_variances[usable].mean() - truth_spread[usable].mean())
    return AnchorErrorModel(bias=float(errors.mean()), excess_variance=max(excess, 0.0), berkson=berkson)


def shrunk_imputed_reliabilities(
    anchors: Sequence[AnchorEstimate],
    prior_log_reliabilities: F64Array,
    error_model: AnchorErrorModel,
) -> F64Array:
    """Per-locus r2_A in one stratum: the anchors shrunk toward the reliability model's prediction.

    A locus whose anchor is undefined keeps the prediction.
    """
    anchor_values = np.array([anchor.log_reliability for anchor in anchors], dtype=np.float64)
    anchor_variances = np.array([anchor.variance for anchor in anchors], dtype=np.float64)
    prior = np.asarray(prior_log_reliabilities, dtype=np.float64)
    if prior.shape != anchor_values.shape:
        raise ValueError("shrunk_imputed_reliabilities needs one prior per anchor.")
    usable = np.isfinite(anchor_values) & np.isfinite(anchor_variances)
    deviation = np.where(usable, anchor_values - error_model.bias - prior, 0.0)
    spread = 0.0
    if usable.sum() >= 2:
        spread = max(
            float(deviation[usable].var() - anchor_variances[usable].mean() - error_model.excess_variance),
            0.0,
        )
    weight = np.zeros_like(prior)
    weight[usable] = spread / (spread + anchor_variances[usable] + error_model.excess_variance) if spread > 0.0 else 0.0
    return np.minimum(np.exp(prior + weight * deviation), 1.0)

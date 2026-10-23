"""The pre-specified replications of a trait's SV gain in held-out people of a later imputed callset.

Fixed before any replication value was computed (~/svpgs-watch/JOURNAL.txt, 2026-10-22). Nothing below is chosen
after seeing a replication number. ``REPLICATIONS`` holds one entry per pre-registered trait, each with the discovery
result it replicates. Every entry shares the statistic, the one-sided level, the calibration and the batch-drift
limitation below.

chronic_kidney_disease, the standard-method pair of run aou-m1p-k. Here the two arms are LDpred2-auto fitted on the
with_sv records and LDpred2-auto fitted on the snv_indel records, in one process on the same training rows with the
same settings, from one GWAS pass whose snv_indel LD band is the restriction of the with_sv band, and scored on the
same held-out people. Its saved effects are applied to the new people unchanged, centred on the training means they
were fitted with (``extra_test_people.fast_method_weights``); nothing is refitted and no mean is re-estimated on the
new people. Its discovery result is a gain of +3.98e-4 (95% CI +0.83 to +7.93e-4, two-sided p 0.027, so one-sided
0.013, +10.3% relative) over 7,185 held-out people, from the same partial-r^2 statistic. That cell is the strongest of
three methods by eleven traits and carries no multiplicity correction, and the pooled LDpred2-auto test over the
eleven traits is null (permutation p 0.55), which is exactly why it needs an independent test rather than a louder
claim. The secondary DeLong AUC difference there was +0.0042 (0.0012 to 0.0072).

Trait and models. hypothyroidism, the matched pair of pairing run x3: the with_sv arm against the separately fitted
snv_indel arm, both fitted by identical code (digest 7a0580a9) on the same training people, over the same store layout
(sites digest 244e0ed2). Their released models are scored unchanged; nothing is refitted. The within-model SV deletion
is not a gain test and plays no part here.

People. The held-out people are the new callset's, scored by ``extra_test_people``: people in no fit of either arm,
unrelated (above second degree) to any training person, with the trait's outcome and genetic PCs. The discovery
people are not reused, so the replication is an independent test of the same hypothesis, not a reanalysis.

Statistic. Accuracy is partial r^2 given the trait's own discovery covariates
(``held_out_comparison.weighted_partial_r2``), and the gain is the with_sv arm's less the snv_indel arm's. The test is
ONE-SIDED for with_sv > snv_indel at ALPHA, z = gain / SE with the SE from a family bootstrap
(``held_out_comparison.cluster_bootstrap_counts``, RESAMPLES resamples, families as in the discovery cohort, a new
person joined to a cohort family where they are related to it). Reported: the gain, its bootstrap 95% interval, the
one-sided p, and the one-sided upper 95% bound (estimate + 1.645 SE), absolute and relative to the snv_indel arm's
partial r^2. The relative gain is secondary.

Calibration. The family arm swap of ``sv_gain_meta``: in each replicate one fair coin per family exchanges that
family's two arm scores. Where the arms are equally accurate the swapped data have the observed data's distribution,
so the replicates give the null of z. Reported: the null mean and SD of z (the SE is right where the SD is near 1),
the share of replicates below ALPHA one-sided, and the permutation p of the observed z. The permutation p is primary
instead of the bootstrap p where the null SD of z is outside [0.8, 1.25].

Batch effects, fixed as a limitation before any replication value was computed. The discovery people's genotypes came
from one imputation run and the replication people's from a later, separately run one. Drift between the two runs is
not symmetric across the arms: an SV's imputed dosage is the less reproducible of the two kinds, so drift degrades the
with_sv arm's SV columns while leaving both arms' short-variant columns almost untouched. The models were fitted on the
earlier run, so such drift biases the replication gain DOWNWARD, and a null replication is therefore not evidence
against the discovery result on its own. ``batch_drift`` is run first and reported beside the gain: per record, the
ALT frequency and imputation quality of the two runs, summarized separately over the model's SV records and its
short-variant records. Drift in the SV records that is clearly larger than in the short-variant records makes the
replication a lower bound on the gain rather than a test of it, which the report says plainly.

Which traits are tested here. Only a trait whose discovery result is promising (``promising``), fixed now: a one-sided
discovery p below SCREENING_LEVEL in a valid matched pair (two separately fitted arms, identical code, one held-out
fold). Testing every trait would spend the new people's power on traits with no signal and need a multiplicity
correction that the screen avoids. hypothyroidism qualifies on its discovery result (gain +0.71e-4, 95% CI -0.09 to
+1.58e-4, two-sided p 0.086, so one-sided p 0.043, +1.5% relative). A trait that fails the screen is not tested in the
new people, and a trait tested here is reported whatever its result.

    python -m benchmarks.bench_real.sv_gain_replication SCORES.npz [--resamples N] [--permutations N]

``SCORES.npz`` is one trait's file from ``extra_test_people score``: ids, target, covariates, families, ancestry,
baseline (snv_indel) and candidate (with_sv). Only aggregates are printed, and every count of 1 to 20 is shown as <=20.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from sv_pgs.held_out_comparison import cluster_bootstrap_counts, weighted_partial_r2

TRAIT = "hypothyroidism"
PAIR = "x3"
"""The pairing run whose two arms are replicated (``aou-x3-o1`` holds both arms' models and the discovery report)."""


@dataclass(frozen=True)
class PreSpecified:
    """One trait's pre-registered replication: what is compared, where its weights are, and the discovery result it
    replicates. Written before any replication value for that trait was computed."""

    trait: str
    run: str
    arms: str
    statistic: str
    discovery_gain: float
    discovery_ci: tuple[float, float]
    discovery_p_one_sided: float
    discovery_relative: float
    fixed_on: str


REPLICATIONS = {
    "hypothyroidism": PreSpecified(
        trait="hypothyroidism", run="aou-x3-o1", arms="SV-PGS with_sv against the separately fitted snv_indel arm, identical code 7a0580a9",
        statistic="paired held-out partial r^2 difference, family-bootstrap SE, one-sided",
        discovery_gain=0.71e-4, discovery_ci=(-0.09e-4, 1.58e-4), discovery_p_one_sided=0.043, discovery_relative=0.015,
        fixed_on="2026-10-22"),
    "chronic_kidney_disease": PreSpecified(
        trait="chronic_kidney_disease", run="aou-m1p-k",
        arms="LDpred2-auto fitted on the with_sv records against LDpred2-auto fitted on the snv_indel records, one process, same code and training rows",
        statistic="paired held-out partial r^2 difference, family-bootstrap SE, one-sided",
        discovery_gain=3.98e-4, discovery_ci=(0.83e-4, 7.93e-4), discovery_p_one_sided=0.013, discovery_relative=0.103,
        fixed_on="2026-10-22"),
}
"""Every trait pre-registered for the new people, each entry fixed before that trait was scored there."""

ALPHA = 0.05
"""The replication's one-sided level."""
SCREENING_LEVEL = 0.10
"""A trait is tested in the new people where its one-sided discovery p is below this (``promising``)."""
RESAMPLES = 2000
PERMUTATIONS = 500
UPPER = float(stats.norm.ppf(0.95))
"""The one-sided 95% normal quantile behind every upper bound."""
NULL_SD_BOUNDS = (0.8, 1.25)
"""Outside this, the family arm swap's permutation p is primary instead of the bootstrap p."""
CHUNK = 250
"""Resamples per ``weighted_partial_r2`` call (``sv_gain_meta.CHUNK``)."""


MINIMUM_SHOWN = 40
"""No count below this is printed or read from All of Us: anything smaller is masked at the source (the user's rule of
2026-10-22, which tightens the program's own small-cell rule of 20)."""


def shown(count: int) -> str:
    """A count as it may be printed: below MINIMUM_SHOWN, only that it is below it."""
    return f"<{MINIMUM_SHOWN}" if 1 <= count < MINIMUM_SHOWN else str(count)


def promising(p_one_sided: float, matched_arms: bool, identical_code: bool) -> bool:
    """Whether a trait's discovery result qualifies it for the new people (the docstring's fixed rule)."""
    return bool(matched_arms and identical_code and p_one_sided < SCREENING_LEVEL)


def batch_drift(frequency: np.ndarray, other_frequency: np.ndarray, quality: np.ndarray, other_quality: np.ndarray,
                structural: np.ndarray) -> dict[str, dict[str, float]]:
    """Per-record drift between two imputation runs at the same sites, summarized over the SV records (``structural``)
    and the short-variant records on their own.

    The inputs are site-level fields only: each record's ALT frequency and the imputation's own quality
    (INFO/AF and INFO/INFO) in the run the models were fitted on and in the run the replication people come from.
    Each group gets the median absolute frequency difference, the frequency correlation, the median quality in each run
    and the median quality change. The SV group drifting much more than the short-variant group is what biases the
    replication against SVs (the docstring)."""
    def summary(member: np.ndarray) -> dict[str, float]:
        if int(member.sum()) < 2:
            return {"records": int(member.sum())}
        first, second = frequency[member], other_frequency[member]
        return {
            "records": int(member.sum()),
            "median_absolute_frequency_difference": float(np.median(np.abs(second - first))),
            "frequency_correlation": float(np.corrcoef(first, second)[0, 1]),
            "median_quality_fitted": float(np.median(quality[member])),
            "median_quality_replication": float(np.median(other_quality[member])),
            "median_quality_change": float(np.median(other_quality[member] - quality[member])),
        }
    structural = np.asarray(structural, dtype=bool)
    return {"structural": summary(structural), "short": summary(~structural)}


def drift_lines(drift: dict[str, dict[str, float]]) -> list[str]:
    """``batch_drift`` as the report prints it, with the verdict the pre-specification fixed."""
    lines = []
    for name, found in (("SV records", drift["structural"]), ("short-variant records", drift["short"])):
        if "frequency_correlation" not in found:
            lines.append(f"  {name}: {shown(found['records'])} records, too few to summarize")
            continue
        lines.append(f"  {name}: {found['records']} records, median |dAF| {found['median_absolute_frequency_difference']:.4f}, "
                     f"AF correlation {found['frequency_correlation']:.4f}, median quality "
                     f"{found['median_quality_fitted']:.3f} -> {found['median_quality_replication']:.3f} "
                     f"({found['median_quality_change']:+.3f})")
    both = all("frequency_correlation" in drift[key] for key in ("structural", "short"))
    if both:
        ratio = (drift["structural"]["median_absolute_frequency_difference"]
                 / max(drift["short"]["median_absolute_frequency_difference"], 1e-12))
        lines.append(f"  SV drift is {ratio:.1f}x the short-variant drift: "
                     + ("the replication is a lower bound on the gain, not a test of it" if ratio >= 2.0
                        else "no asymmetric drift, so the replication tests the gain"))
    return lines


def gains(weights: np.ndarray, target: np.ndarray, covariates: np.ndarray, baseline: np.ndarray, candidate: np.ndarray,
          swap: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per row of ``weights`` (resamples x people): the gain and the snv_indel arm's partial r^2, the two arms' scores
    exchanged where ``swap`` is true."""
    if swap is not None:
        baseline, candidate = np.where(swap, candidate, baseline), np.where(swap, baseline, candidate)
    scores = np.vstack([candidate, baseline])
    found = np.vstack([weighted_partial_r2(weights[start:start + CHUNK], covariates, target, scores)
                       for start in range(0, weights.shape[0], CHUNK)])
    return found[:, 0] - found[:, 1], found[:, 1]


def replicate(target: np.ndarray, covariates: np.ndarray, families: np.ndarray, baseline: np.ndarray, candidate: np.ndarray,
              seed: int, resamples: int = RESAMPLES, permutations: int = PERMUTATIONS) -> dict[str, float]:
    """The pre-specified one-sided test of with_sv > snv_indel on these people, with its family-swap calibration."""
    generator = np.random.default_rng(seed)
    clusters = np.unique(families, return_inverse=True)[1]
    counts = cluster_bootstrap_counts(np.zeros(int(clusters.max()) + 1, dtype=np.int64), resamples, generator)
    weights = counts[:, clusters]
    unit = np.ones((1, clusters.shape[0]))
    estimate, base = (value[0] for value in gains(unit, target, covariates, baseline, candidate))
    draws, base_draws = gains(weights, target, covariates, baseline, candidate)
    error = float(np.nanstd(draws, ddof=1))
    z = estimate / error if error > 0.0 else math.nan
    relative_draws = draws / base_draws
    relative_error = float(np.nanstd(relative_draws, ddof=1))
    result = {
        "people": int(target.shape[0]), "families": int(clusters.max()) + 1, "cases": int(np.sum(target == 1.0)),
        "baseline_r2": float(base), "estimate": float(estimate), "standard_error": error, "z": z,
        "p_one_sided": float(stats.norm.sf(z)) if math.isfinite(z) else math.nan,
        "ci_low": float(np.nanquantile(draws, 0.025)), "ci_high": float(np.nanquantile(draws, 0.975)),
        "upper_95": float(estimate + UPPER * error),
        "relative": float(estimate / base), "relative_standard_error": relative_error,
        "relative_upper_95": float(estimate / base + UPPER * relative_error),
    }
    if permutations:
        null = []
        swap_weights = weights[:min(resamples, 200)]
        for _ in range(permutations):
            coins = generator.random(int(clusters.max()) + 1) < 0.5
            swap = coins[clusters]
            swapped, _base = gains(unit, target, covariates, baseline, candidate, swap)
            swapped_draws, _ = gains(swap_weights, target, covariates, baseline, candidate, swap)
            swapped_error = float(np.nanstd(swapped_draws, ddof=1))
            null.append(swapped[0] / swapped_error if swapped_error > 0.0 else math.nan)
        null_array = np.array(null)
        finite = null_array[np.isfinite(null_array)]
        result.update({
            "permutations": int(finite.shape[0]), "null_z_mean": float(np.mean(finite)), "null_z_sd": float(np.std(finite, ddof=1)),
            "null_type_one_one_sided": float(np.mean(finite >= UPPER)),
            "p_permutation_one_sided": float((1 + np.sum(finite >= z)) / (1 + finite.shape[0])) if math.isfinite(z) else math.nan,
        })
        result["primary_p"] = (result["p_permutation_one_sided"]
                               if not NULL_SD_BOUNDS[0] <= result["null_z_sd"] <= NULL_SD_BOUNDS[1] else result["p_one_sided"])
    return result


def report(result: dict[str, float], trait: str) -> list[str]:
    lines = [
        f"{trait}: replication in new held-out people {shown(result['people'])} (families {shown(result['families'])}, "
        f"cases {shown(result['cases'])}); snv_indel partial r^2 {result['baseline_r2']:.5f}",
        f"  gain {result['estimate'] * 1e4:+.3f}e-4 [{result['ci_low'] * 1e4:+.3f}, {result['ci_high'] * 1e4:+.3f}], "
        f"one-sided p {result['p_one_sided']:.4g}, at most {result['upper_95'] * 1e4:+.3f}e-4",
        f"  relative {result['relative'] * 100:+.2f}% (at most {result['relative_upper_95'] * 100:+.2f}%)",
    ]
    if "permutations" in result:
        lines += [
            f"  family arm swap over {result['permutations']} replicates: null z mean {result['null_z_mean']:+.3f}, "
            f"SD {result['null_z_sd']:.3f}, one-sided type I at 0.05 {result['null_type_one_one_sided']:.3f}, "
            f"permutation p {result['p_permutation_one_sided']:.4g}",
            f"  primary one-sided p {result['primary_p']:.4g} vs alpha {ALPHA}: "
            f"{'replicated' if result['primary_p'] < ALPHA else 'not replicated'}",
        ]
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scores", help="One trait's npz from 'extra_test_people score'.")
    parser.add_argument("--trait", default=TRAIT)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    args = parser.parse_args(argv)
    with np.load(args.scores, allow_pickle=False) as arrays:
        result = replicate(arrays["target"], arrays["covariates"], arrays["families"], arrays["baseline"], arrays["candidate"],
                           args.seed, args.resamples, args.permutations)
    print("\n".join(report(result, args.trait)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

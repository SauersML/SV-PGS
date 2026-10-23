"""The pre-specified replication of one trait's matched SV gain in held-out people of a later imputed callset.

Fixed before any replication value was computed (~/svpgs-watch/JOURNAL.txt, 2026-10-22). Nothing below is chosen
after seeing a replication number.

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

import numpy as np
from scipy import stats

from sv_pgs.held_out_comparison import cluster_bootstrap_counts, weighted_partial_r2

TRAIT = "hypothyroidism"
PAIR = "x3"
"""The pairing run whose two arms are replicated (``aou-x3-o1`` holds both arms' models and the discovery report)."""
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


def shown(count: int) -> str:
    return "<=20" if 1 <= count <= 20 else str(count)


def promising(p_one_sided: float, matched_arms: bool, identical_code: bool) -> bool:
    """Whether a trait's discovery result qualifies it for the new people (the docstring's fixed rule)."""
    return bool(matched_arms and identical_code and p_one_sided < SCREENING_LEVEL)


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

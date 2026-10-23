"""Whether adding SVs to an SV-PGS fit raises held-out accuracy when every trait's test is combined, how large the gain
can be (its upper 95% bound), and whether the combination is calibrated.

Each trait's test compares two separately fitted models of one pairing run, fitted by identical code on the same
training people: the with_sv arm (every record) against the snv_indel arm (records under 50 bp), scored on the same
held-out people. Accuracy is partial r^2 given the trait's covariates (``held_out_comparison.weighted_partial_r2``,
the report's ``_partial_r2`` at unit weights). Only matched pairing runs are valid inputs. A run whose two arms were
fitted by different code compares the code as well as the SVs. Deleting the SV terms from one joint fit is no test
either: the SNVs that tag those SVs would have carried their effect in a refit.

Secondary and descriptive only, ``--contrast ablation``: one with_sv model's held-out score against the same score
with its SV terms zeroed (``_Scores.held_out_structural`` subtracted), on the same people. This is no measure of SV
gain, for the reason above. It says how much the joint fit's own SV terms carry, given the SNV weights fitted beside
them. Its relative gain is against the model's own short-variant part, and its output is labelled ABLATION_LABEL.

Uncertainty is a family bootstrap (``held_out_comparison.cluster_bootstrap_counts``) drawn once over the union of
every trait's held-out people. People are matched across runs by research id, and two people share a cluster where
any run puts them in one family (``union_clusters``). Every trait's gain is read from the same resamples, so the
correlation R of the traits' gains (shared people, correlated outcomes) is read from them too.

The analysis was fixed before any combined value was computed (~/svpgs-watch/JOURNAL.txt, 2026-10-22). The unit is one
test per trait on all its held-out people. It was amended at 17:40 PDT that day (benchmarks/bench_real/SV_GAIN_PRESPEC.txt),
when only aou-wl15 (LDL) had been scored, to the primary below. It weights LDL, the one result seen, the most.
- Primary (amended): the common relative gain g, the gains regressed on the arms' mean partial r^2 through the origin
  with weights 1 / var(gain): g = sum(a gain) / sum(a r^2), a = r^2 / var(gain) (``common_relative``). g is a
  fraction of the arms' mean partial r^2, not of the snv_indel arm's. Where every trait gains the same fraction of its
  accuracy, this is the most precise estimate of that fraction among weightings that ignore the traits' correlation
  (correlated traits would need GLS weights), and it needs no per-trait ratio, whose denominator can resample near 0.
  Its standard error comes from the joint resamples with the weights a fixed. Tests (amendment 5): one-sided, g > 0,
  at 0.05, both reported. The weak-null test of g = 0 is the family-clustered bootstrap z = g / SE, with g's one-sided
  95% lower and upper bounds. The sharp-null test is the family arm-swap permutation p (Calibration, below). A gain is
  claimed only if both p are below 0.05. A run's traits each enter once: several matched pairing runs of one trait are
  averaged per arm first, with their family memberships joined (``merge_pairs``), and people in an excluded list (the
  long-read panel) are dropped before anything is computed (``main``).
- Original primary, now secondary: the two-sided Stouffer combination of the per-trait z = gain / bootstrap SE,
  equal weights, Z = sum(z) / sqrt(1'R1).
- Secondary:
  - the Cauchy combination (ACAT; Liu and Xie 2020) of the per-trait two-sided p, calibrated under arbitrary
    dependence at small levels and powerful when one trait carries the signal;
  - an omnibus Cauchy combination of the Stouffer and ACAT p;
  - one-sided p for a gain (above 0);
  - the random-effects pooled relative gain (gain / the snv_indel arm's partial r^2) by REML with the Hartung-Knapp-
    Sidik-Jonkman interval on k - 1 degrees of freedom, a sensitivity analysis that treats the traits as independent
    although they share people;
  - the gain pooled by generalized least squares under the joint-resample covariance
    (``held_out_comparison.meta_analysis``), absolute and relative, over all held-out people and within the EUR, AFR
    and AMR groups.
- Every trait and every pooled gain also gets its one-sided upper 95% bound ("SVs add at most"): the estimate plus
  1.645 standard errors (HKSJ: the t quantile on k - 1 degrees of freedom), absolute and relative to the snv_indel
  arm's partial r^2.

Calibration: a family arm-swap null. In each replicate one fair coin per union family decides whether every member of
that family has the two arms' scores exchanged. Every trait reads the same coins, so the traits keep their dependence.
Where the two arms' scores are exchangeable within families, the swapped data have the observed data's distribution,
so the replicates give that null of every statistic. It is a sharp null, stronger than g = 0: two scores can be
equally accurate and still differ in distribution (their covariate loadings or error distributions), and g = 0 allows
traits whose gains have opposite signs. Each replicate repeats the analysis (PERMUTATION_RESAMPLES bootstrap resamples
per trait), and the observed statistics the replicates are ranked against are computed from the same resamples, as
many as each replicate uses. That yields:
- the null mean and SD of every z, of Z and of the primary's z (near 0 and 1 when the bootstrap SE is right);
- the share of replicates below 0.05 for each p;
- the permutation p of the observed statistics.

Inputs: the per-person arrays of ``PairedTrait`` (``analyse``). ``main`` reads them inside the All of Us workspace from
each named pairing run's cohort and score steps; only aggregates are printed, and every count of 1 to 20 is shown as
<=20.

    python sv_gain_meta.py RUN [RUN ...] [--exclude IDS] [--permutations N] [--resamples N]
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

from sv_pgs.held_out_comparison import PairedComparison, cluster_bootstrap_counts, meta_analysis, weighted_partial_r2

GROUPS = ("eur", "afr", "amr")
"""The ancestry groups pooled across traits (the three largest in the held-out sets)."""
RESAMPLES = 2000
"""Joint bootstrap resamples behind the observed statistics (the report's BOOTSTRAP_RESAMPLES)."""
PERMUTATIONS = 1000
"""Family arm swaps behind the primary's permutation p (SV_GAIN_PRESPEC.txt)."""
PERMUTATION_RESAMPLES = 200
"""Bootstrap resamples behind each permutation replicate's standard errors and R (the observed resamples' first ones)."""
CHUNK = 250
"""Resamples per ``weighted_partial_r2`` call: its Gram product holds CHUNK x people x (covariates + 3)^2 values."""
MINIMUM_GROUP_PEOPLE = 100
"""A trait's ancestry group enters the group's pooled gain with at least this many held-out people (and, for a
disease, more than 20 cases)."""
LEVEL = 0.05
UPPER = float(stats.norm.ppf(0.95))
"""The one-sided 95% normal quantile behind every upper bound."""
ABLATION_LABEL = ("SECONDARY / DESCRIPTIVE: SV terms zeroed in the joint fit, not a measure of SV gain "
                  "('snv_indel' below = the with_sv model less its SV terms)")


@dataclass(frozen=True)
class PairedTrait:
    """One trait's two arms on its held-out people, each person an index into the union of every trait's held-out
    people: the snv_indel arm's score (``baseline``) and the with_sv arm's (``candidate``)."""

    name: str
    binary: bool
    people: np.ndarray
    target: np.ndarray
    covariates: np.ndarray
    baseline: np.ndarray
    candidate: np.ndarray


def shown(count: int) -> str:
    """A count as a table may show it: 1 to 39 as <40 (the user's rule for All of Us counts, stricter than All of Us's
    own 1 to 20)."""
    return "<40" if 1 <= count < 40 else str(count)


def union_clusters(memberships: Sequence[tuple[np.ndarray, np.ndarray]], size: int) -> np.ndarray:
    """Clusters 0..C-1 of the union's ``size`` people: per run, (people, family) arrays, and two people share a cluster
    where any run puts them in one family, joined transitively (union-find)."""
    parent = np.arange(size)

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    for people, families in memberships:
        order = np.argsort(families, kind="stable")
        members, labels = np.asarray(people)[order], np.asarray(families)[order]
        starts = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
        for start, stop in zip(starts, np.r_[starts[1:], labels.shape[0]]):
            first = root(int(members[start]))
            for other in members[start + 1:stop]:
                found = root(int(other))
                if found != first:
                    parent[found] = first
    roots = np.array([root(index) for index in range(size)])
    return np.unique(roots, return_inverse=True)[1].astype(np.int64)


def gains(weights: np.ndarray, trait: PairedTrait, swap: np.ndarray | None = None, rows: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per row of ``weights`` (resamples x union people): the with_sv arm's partial r^2 less the snv_indel arm's, and
    the snv_indel arm's, on the trait's held-out people (``rows`` of them only, if given), the two arms' scores
    exchanged for the people where ``swap`` (per trait person) is true."""
    keep = np.arange(trait.people.shape[0]) if rows is None else rows
    candidate, baseline = trait.candidate, trait.baseline
    if swap is not None:
        candidate, baseline = np.where(swap, baseline, candidate), np.where(swap, candidate, baseline)
    scores = np.vstack([candidate[keep], baseline[keep]])
    found = [weighted_partial_r2(weights[start:start + CHUNK][:, trait.people[keep]], trait.covariates[keep], trait.target[keep], scores)
             for start in range(0, weights.shape[0], CHUNK)]
    values = np.vstack(found)
    return values[:, 0] - values[:, 1], values[:, 1]


def acat(p_values: Sequence[float]) -> float:
    """The Cauchy combination of p-values with equal weights: T = mean tan((0.5 - p) pi), p = 0.5 - arctan(T) / pi,
    read as 1 / (T pi) where T is large."""
    p = np.clip(np.asarray(p_values, dtype=np.float64), 1e-300, 1.0)
    terms = np.where(p < 1e-15, 1.0 / (p * math.pi), np.tan((0.5 - p) * math.pi))
    total = float(np.mean(terms))
    return 1.0 / (total * math.pi) if total > 1e15 else 0.5 - math.atan(total) / math.pi


def stouffer(z: np.ndarray, correlation: np.ndarray) -> tuple[float, float, float]:
    """Z = sum(z) / sqrt(1'R1), its two-sided p and its one-sided p for a positive sum."""
    ones = np.ones(z.shape[0])
    total = float(np.sum(z) / math.sqrt(float(ones @ correlation @ ones)))
    return total, float(2.0 * stats.norm.sf(abs(total))), float(stats.norm.sf(total))


def reml_hksj(estimates: np.ndarray, variances: np.ndarray) -> dict[str, float]:
    """The random-effects mean of independent ``estimates`` with sampling ``variances``: tau^2 by REML (Viechtbauer
    2005's fixed-point iteration, floored at 0), the mean weighted by 1 / (v + tau^2), and the Hartung-Knapp-Sidik-
    Jonkman standard error sqrt(q / sum w), q = sum w (y - mean)^2 / (k - 1), with its t interval, one-sided upper 95%
    bound and p on k - 1 degrees of freedom (NaN for one estimate)."""
    y, v = np.asarray(estimates, dtype=np.float64), np.asarray(variances, dtype=np.float64)
    k = y.shape[0]
    tau2 = max(0.0, float(np.var(y, ddof=1) - np.mean(v))) if k > 1 else 0.0
    for _ in range(500):
        w = 1.0 / (v + tau2)
        mean = float(np.sum(w * y) / np.sum(w))
        updated = max(0.0, float(np.sum(w ** 2 * ((y - mean) ** 2 - v)) / np.sum(w ** 2) + 1.0 / np.sum(w))) if k > 1 else 0.0
        if abs(updated - tau2) <= 1e-12 * max(1.0, tau2):
            tau2 = updated
            break
        tau2 = updated
    w = 1.0 / (v + tau2)
    mean = float(np.sum(w * y) / np.sum(w))
    if k < 2:
        return {"estimate": mean, "standard_error": math.nan, "ci_low": math.nan, "ci_high": math.nan, "upper_95": math.nan,
                "p_value": math.nan, "tau_squared": tau2}
    error = math.sqrt(float(np.sum(w * (y - mean) ** 2) / (k - 1)) / float(np.sum(w)))
    half = float(stats.t.ppf(0.975, k - 1)) * error
    statistic = mean / error if error > 0 else math.nan
    return {"estimate": mean, "standard_error": error, "ci_low": mean - half, "ci_high": mean + half,
            "upper_95": mean + float(stats.t.ppf(0.95, k - 1)) * error,
            "p_value": float(2.0 * stats.t.sf(abs(statistic), k - 1)) if math.isfinite(statistic) else math.nan, "tau_squared": tau2}


def common_relative(estimates: np.ndarray, bases: np.ndarray, draws: np.ndarray, base_draws: np.ndarray) -> dict[str, object]:
    """The primary (module docstring): the common relative gain g = sum(a gain) / sum(a r^2), a = r^2 / var(gain), its
    standard error over the joint resamples (resamples x traits) with a fixed, z, one- and two-sided p, 95% interval,
    one-sided lower and upper 95% bounds, and each trait's share of g: a r^2 / sum(a r^2), the weight its own relative
    gain gain / r^2 takes in g = sum(share gain / r^2). g is a fraction of r^2, the two arms' mean partial r^2
    (``bases``, the snv_indel arm's, plus half the gain): it is the same with the arms exchanged, so where they are
    equally accurate it carries no information on the gain's sign. The snv_indel arm's own r^2 would: it is high
    exactly where the gain is low, which weights the traits that lost and biases g down. A trait whose gain has no
    variance (both arms the same score) gets no weight."""
    levels, level_draws = bases + estimates / 2.0, base_draws + draws / 2.0
    variances = np.nanvar(draws, axis=0, ddof=1)
    varies = (np.nanmax(draws, axis=0) > np.nanmin(draws, axis=0)) & (variances > 0.0)
    weights = np.where(varies, levels / np.where(varies, variances, 1.0), 0.0)
    estimate = float(weights @ estimates / (weights @ levels))
    finite = np.all(np.isfinite(draws), axis=1) & np.all(np.isfinite(level_draws), axis=1)
    error = float(np.std((draws[finite] @ weights) / (level_draws[finite] @ weights), ddof=1))
    z = estimate / error if error > 0.0 else math.nan
    half = float(stats.norm.ppf(0.975)) * error
    return {"estimate": estimate, "standard_error": error, "z": z, "p_one_sided": float(stats.norm.sf(z)),
            "p_two_sided": float(2.0 * stats.norm.sf(abs(z))), "ci_low": estimate - half, "ci_high": estimate + half,
            "lower_95": estimate - UPPER * error, "upper_95": estimate + UPPER * error,
            "weight_share": weights * levels / (weights @ levels)}


def correlation_of(draws: np.ndarray) -> np.ndarray:
    """The correlation of the traits' resampled gains (resamples x traits), over the resamples where all are finite."""
    if draws.shape[1] == 1:
        return np.ones((1, 1))
    return np.corrcoef(draws[np.all(np.isfinite(draws), axis=1)], rowvar=False)


def covariance_of(draws: np.ndarray) -> np.ndarray:
    return np.atleast_2d(np.cov(draws[np.all(np.isfinite(draws), axis=1)], rowvar=False))


def combined(estimates: np.ndarray, draws: np.ndarray) -> dict[str, object]:
    """Per-trait z and p from the resampled gains, R, and the Stouffer, ACAT and omnibus combinations."""
    errors = np.nanstd(draws, axis=0, ddof=1)
    z = estimates / errors
    p = 2.0 * stats.norm.sf(np.abs(z))
    correlation = correlation_of(draws)
    total, p_stouffer, p_stouffer_one = stouffer(z, correlation)
    p_acat = acat(p)
    one_sided = stats.norm.sf(z)
    return {"standard_error": errors, "z": z, "p": p, "p_one_sided": one_sided, "R": correlation, "Z": total,
            "p_stouffer": p_stouffer, "p_stouffer_one_sided": p_stouffer_one, "p_acat": p_acat, "p_acat_one_sided": acat(one_sided),
            "p_omnibus": acat([p_stouffer, p_acat])}


def pooled(estimates: np.ndarray, draws: np.ndarray) -> dict[str, object]:
    """The traits' gains pooled under their joint-resample covariance (``meta_analysis``), fixed and random effects."""
    found = meta_analysis(estimates, covariance_of(draws))
    return {"fixed": found.fixed, "random": found.random, "q_p": found.heterogeneity_p, "i_squared": found.i_squared,
            "tau_squared": found.tau_squared}


def family_swaps(trait: PairedTrait, clusters: np.ndarray, coins: np.ndarray) -> np.ndarray:
    """Per person of the trait: whether its family's coin (one per union cluster) exchanges the arms."""
    return coins[clusters[trait.people]]


def group_rows(trait: PairedTrait, ancestry: np.ndarray, group: str) -> np.ndarray:
    return np.flatnonzero(ancestry[trait.people] == group)


def evaluated(traits: Sequence[PairedTrait], clusters: np.ndarray, weights: np.ndarray, coins: np.ndarray | None = None) -> dict[str, object]:
    """The combined statistics (``combined``) and the primary's z and one-sided p over ``weights``' resamples, with the
    arms exchanged in the families whose ``coins`` are true (none exchanged where ``coins`` is None). The observed
    statistic a permutation p ranks against and each swapped one are this one function on the same resamples."""
    unit = np.ones((1, clusters.shape[0]))
    estimates, bases, draws, base_draws = [], [], [], []
    for trait in traits:
        swap = None if coins is None else family_swaps(trait, clusters, coins)
        gain, base = gains(unit, trait, swap)
        gain_draws, base_draw = gains(weights, trait, swap)
        estimates.append(gain[0])
        bases.append(base[0])
        draws.append(gain_draws)
        base_draws.append(base_draw)
    found = combined(np.array(estimates), np.column_stack(draws))
    common = common_relative(np.array(estimates), np.array(bases), np.column_stack(draws), np.column_stack(base_draws))
    found["p_common_one_sided"] = common["p_one_sided"]
    found["common_z"] = common["z"]
    return found


def analyse(traits: Sequence[PairedTrait], clusters: np.ndarray, ancestry: np.ndarray, seed: int, resamples: int = RESAMPLES,
            permutations: int = PERMUTATIONS, permutation_resamples: int = PERMUTATION_RESAMPLES) -> dict[str, object]:
    """Every pre-specified statistic over ``traits`` (module docstring): ``clusters`` and ``ancestry`` per union person."""
    generator = np.random.default_rng(seed)
    counts = cluster_bootstrap_counts(np.zeros(int(clusters.max()) + 1, dtype=np.int64), resamples, generator)
    weights = counts[:, clusters]
    unit = np.ones((1, clusters.shape[0]))
    estimates, bases, draws, base_draws = [], [], [], []
    for trait in traits:
        gain, base = gains(unit, trait)
        gain_draws, base_draw = gains(weights, trait)
        estimates.append(gain[0])
        bases.append(base[0])
        draws.append(gain_draws)
        base_draws.append(base_draw)
    estimates_array, draws_array = np.array(estimates), np.column_stack(draws)
    observed = combined(estimates_array, draws_array)
    observed["common"] = common_relative(estimates_array, np.array(bases), draws_array, np.column_stack(base_draws))
    relative = estimates_array / np.array(bases)
    relative_draws = draws_array / np.column_stack(base_draws)
    relative_error = np.nanstd(relative_draws, axis=0, ddof=1)
    result: dict[str, object] = {
        "traits": [trait.name for trait in traits], "estimate": estimates_array, "baseline_r2": np.array(bases),
        "arm_correlation": np.array([float(np.corrcoef(trait.candidate, trait.baseline)[0, 1]) for trait in traits]),
        "ci_low": np.nanquantile(draws_array, 0.025, axis=0), "ci_high": np.nanquantile(draws_array, 0.975, axis=0), **observed,
        "upper_95": estimates_array + UPPER * observed["standard_error"],
        "relative": relative, "relative_standard_error": relative_error, "relative_upper_95": relative + UPPER * relative_error,
        "relative_random": reml_hksj(relative, relative_error ** 2),
        "absolute_pooled": pooled(estimates_array, draws_array), "relative_pooled": pooled(relative, relative_draws),
    }
    groups = {}
    for group in GROUPS:
        cells, used, used_draws, used_relative, used_relative_draws, used_bases, used_base_draws = [], [], [], [], [], [], []
        for trait in traits:
            rows = group_rows(trait, ancestry, group)
            cases = int(np.sum(trait.target[rows])) if trait.binary else None
            if rows.shape[0] >= MINIMUM_GROUP_PEOPLE and (cases is None or cases > 20):
                gain, base = gains(unit, trait, rows=rows)
                gain_draws, base_draw = gains(weights, trait, rows=rows)
                used.append(gain[0])
                used_draws.append(gain_draws)
                used_relative.append(gain[0] / base[0])
                used_relative_draws.append(gain_draws / base_draw)
                used_bases.append(base[0])
                used_base_draws.append(base_draw)
                cells.append((trait.name, rows.shape[0], cases, gain[0]))
            else:
                cells.append((trait.name, rows.shape[0], cases, math.nan))
        groups[group] = {"cells": cells,
                         "absolute": pooled(np.array(used), np.column_stack(used_draws)) if used else None,
                         "relative": pooled(np.array(used_relative), np.column_stack(used_relative_draws)) if used else None,
                         "common": common_relative(np.array(used), np.array(used_bases), np.column_stack(used_draws),
                                                   np.column_stack(used_base_draws)) if used else None}
    result["groups"] = groups
    if permutations:
        result["calibration"] = calibrate(traits, clusters, weights[:permutation_resamples], generator, permutations)
    return result


def calibrate(traits: Sequence[PairedTrait], clusters: np.ndarray, weights: np.ndarray, generator: np.random.Generator,
              permutations: int) -> dict[str, object]:
    """The analysis repeated on family arm swaps (module docstring), each permutation p ranking the observed statistic
    computed on the same resamples (``weights``) as every swapped one (``reference``)."""
    observed = evaluated(traits, clusters, weights)
    z_null, total_null, trait_p_null, common_null = [], [], [], []
    p_null: dict[str, list[float]] = {"p_stouffer": [], "p_acat": [], "p_omnibus": [], "p_stouffer_one_sided": [], "p_common_one_sided": []}
    for _ in range(permutations):
        coins = generator.random(int(clusters.max()) + 1) < 0.5
        found = evaluated(traits, clusters, weights, coins)
        z_null.append(found["z"])
        trait_p_null.append(found["p"])
        total_null.append(found["Z"])
        common_null.append(found["common_z"])
        for name in p_null:
            p_null[name].append(found[name])
    z_null_array, total_array, common_array = np.vstack(z_null), np.array(total_null), np.array(common_null)
    count = total_array.shape[0]
    return {
        "permutations": count, "resamples": int(weights.shape[0]),
        "reference": {"common_z": float(observed["common_z"]), "Z": float(observed["Z"])},
        "z_mean": np.mean(z_null_array, axis=0), "z_sd": np.std(z_null_array, axis=0, ddof=1),
        "trait_type_one": np.mean(np.vstack(trait_p_null) < LEVEL, axis=0),
        "Z_mean": float(np.mean(total_array)), "Z_sd": float(np.std(total_array, ddof=1)),
        "common_z_mean": float(np.mean(common_array)), "common_z_sd": float(np.std(common_array, ddof=1)),
        "p_permutation_common_upper": float((1 + np.sum(common_array >= observed["common_z"])) / (1 + count)),
        "type_one": {name: float(np.mean(np.array(values) < LEVEL)) for name, values in p_null.items()},
        "p_permutation_Z_upper": float((1 + np.sum(total_array >= observed["Z"])) / (1 + count)),
        "p_permutation_Z_two_sided": float((1 + np.sum(np.abs(total_array) >= abs(observed["Z"]))) / (1 + count)),
        "p_permutation_acat": float((1 + np.sum(np.array(p_null["p_acat"]) <= observed["p_acat"])) / (1 + count)),
        "p_permutation_omnibus": float((1 + np.sum(np.array(p_null["p_omnibus"]) <= observed["p_omnibus"])) / (1 + count)),
    }


def normal(comparison: PairedComparison, scale: float, unit: str) -> str:
    """A pooled gain (times ``scale``) with its normal 95% interval, two-sided p and one-sided upper 95% bound."""
    half = 1.959963984540054 * comparison.standard_error
    p = 2.0 * stats.norm.sf(abs(comparison.z_score)) if math.isfinite(comparison.z_score) else math.nan
    return (f"{comparison.estimate * scale:.3f}{unit} [{(comparison.estimate - half) * scale:.3f}, {(comparison.estimate + half) * scale:.3f}], "
            f"p = {p:.4g}, at most {(comparison.estimate + UPPER * comparison.standard_error) * scale:.3f}{unit}")


def pooled_lines(label: str, absolute: dict[str, object], relative: dict[str, object]) -> list[str]:
    return [f"{label} absolute gain (x1e-4), GLS fixed: {normal(absolute['fixed'], 1e4, '')}; random: {normal(absolute['random'], 1e4, '')}; "
            f"Q p = {absolute['q_p']:.3g}, I^2 = {absolute['i_squared']:.2f}",
            f"{label} relative gain (% of snv_indel r^2), GLS fixed: {normal(relative['fixed'], 100, '%')}; random: {normal(relative['random'], 100, '%')}"]


def report(result: dict[str, object], traits: Sequence[PairedTrait], contrast: str = "arms") -> list[str]:
    """The printed lines: aggregates only, every count of 1 to 20 shown as <=20; the ablation's under ABLATION_LABEL."""
    names = result["traits"]
    lines = [ABLATION_LABEL] if contrast == "ablation" else []
    lines += ["per trait: trait n cases snv_indel_r2 gain(x1e-4) CI SE z p_two p_one at_most(x1e-4) relative_gain(%) relative_SE(%) at_most(%)"]
    for index, trait in enumerate(traits):
        cases = shown(int(np.sum(trait.target))) if trait.binary else ""
        lines.append("  " + "\t".join(str(value) for value in (
            names[index], shown(trait.people.shape[0]), cases, f"{result['baseline_r2'][index]:.5g}", f"{result['estimate'][index] * 1e4:.3f}",
            f"[{result['ci_low'][index] * 1e4:.3f}, {result['ci_high'][index] * 1e4:.3f}]", f"{result['standard_error'][index] * 1e4:.3f}",
            f"{result['z'][index]:.3f}", f"{result['p'][index]:.4g}", f"{result['p_one_sided'][index]:.4g}", f"{result['upper_95'][index] * 1e4:.3f}",
            f"{result['relative'][index] * 100:.2f}", f"{result['relative_standard_error'][index] * 100:.2f}",
            f"{result['relative_upper_95'][index] * 100:.2f}")))
    lines.append("arm agreement (held-out correlation of the two arms' scores, outcome-free): "
                 + ", ".join(f"{name} {value:.4f}" for name, value in zip(names, result["arm_correlation"])))
    lines.append("R (joint family bootstrap correlation of the gains): " + np.array2string(result["R"], precision=3).replace("\n", " "))
    common = result["common"]
    lines.append(f"PRIMARY common relative gain g = {common['estimate'] * 100:.2f}% of the arms' mean partial r^2, 95% CI "
                 f"[{common['ci_low'] * 100:.2f}, {common['ci_high'] * 100:.2f}]; shares of g (a r^2 / sum a r^2) "
                 + ", ".join(f"{name} {share:.2f}" for name, share in zip(names, common["weight_share"])))
    lines.append(f"  weak null g = 0, family-clustered bootstrap test, one-sided (g > 0): z = {common['z']:.3f}, p = {common['p_one_sided']:.4g}; "
                 f"one-sided 95% bounds: g > {common['lower_95'] * 100:.2f}%, g < {common['upper_95'] * 100:.2f}% (two-sided p {common['p_two_sided']:.4g})")
    lines.append(f"original primary, now secondary: Stouffer Z = {result['Z']:.3f}, two-sided p = {result['p_stouffer']:.4g} "
                 f"(one-sided gain > 0: {result['p_stouffer_one_sided']:.4g})")
    lines.append(f"ACAT p = {result['p_acat']:.4g} (from one-sided p: {result['p_acat_one_sided']:.4g}); omnibus Cauchy(Stouffer, ACAT) p = {result['p_omnibus']:.4g}")
    random = result["relative_random"]
    lines.append(f"relative gain (% of snv_indel r^2), REML + HKSJ, sensitivity analysis treating the traits as independent: {random['estimate'] * 100:.2f}% [{random['ci_low'] * 100:.2f}, {random['ci_high'] * 100:.2f}], "
                 f"p = {random['p_value']:.4g}, at most {random['upper_95'] * 100:.2f}%, tau^2 = {random['tau_squared']:.3g}")
    lines.extend(pooled_lines("all held-out:", result["absolute_pooled"], result["relative_pooled"]))
    for group, entry in result["groups"].items():
        cells = "; ".join(f"{name} n={shown(people)}" + (f" cases={shown(cases)}" if cases is not None else "")
                          + (f" gain={gain * 1e4:.3f}e-4" if math.isfinite(gain) else " (not pooled)") for name, people, cases, gain in entry["cells"])
        lines.append(f"{group.upper()} cells: {cells}")
        if entry["absolute"] is not None:
            common = entry["common"]
            lines.append(f"{group.upper()} common relative gain (secondary, % of the arms' mean partial r^2): {common['estimate'] * 100:.2f}% "
                         f"[{common['ci_low'] * 100:.2f}, {common['ci_high'] * 100:.2f}], weak null one-sided p = {common['p_one_sided']:.4g}, "
                         f"one-sided 95% bounds {common['lower_95'] * 100:.2f}% to {common['upper_95'] * 100:.2f}%")
            lines.extend(pooled_lines(f"{group.upper()}:", entry["absolute"], entry["relative"]))
    calibration = result.get("calibration")
    if calibration:
        lines.append(f"calibration ({calibration['permutations']} family arm swaps, each and the observed statistics on the same "
                     f"{calibration['resamples']} resamples; sharp null: the arms' scores exchangeable within families): per-trait null z mean "
                     + np.array2string(calibration["z_mean"], precision=2) + ", SD " + np.array2string(calibration["z_sd"], precision=2)
                     + ", type-I at 0.05 " + np.array2string(calibration["trait_type_one"], precision=3))
        lines.append(f"  PRIMARY sharp null, one-sided permutation p (g > 0): {calibration['p_permutation_common_upper']:.4g} (observed z on these "
                     f"resamples {calibration['reference']['common_z']:.3f}); its z null mean "
                     f"{calibration['common_z_mean']:.2f}, SD {calibration['common_z_sd']:.2f}, normal one-sided type-I at 0.05 "
                     f"{calibration['type_one']['p_common_one_sided']:.3f}")
        lines.append(f"  Stouffer Z null mean {calibration['Z_mean']:.2f}, SD {calibration['Z_sd']:.2f}; type-I at 0.05: "
                     + ", ".join(f"{name} {value:.3f}" for name, value in calibration["type_one"].items()))
        lines.append(f"  permutation p: Stouffer Z upper {calibration['p_permutation_Z_upper']:.4g}, two-sided {calibration['p_permutation_Z_two_sided']:.4g}; "
                     f"ACAT {calibration['p_permutation_acat']:.4g}; omnibus {calibration['p_permutation_omnibus']:.4g}")
    return lines


OPS = "/home/jupyter/workspace/svpgs_ops/svpgs"


NULL_VARIANCE = 1e-12
"""A pair counts only if each arm's genetic score varies over the trait's held-out people by more than this (in the
trait's units squared; SV_GAIN_PRESPEC.txt, validity). An EB fit that collapsed to the null scores everyone alike: round
m3's warm-started asthma and CKD pairs had total genetic variances of 1.6e-35 and 1.4e-28. Standardizing such a score
turns its rounding error into a full-size score, so its "gain" is noise, and it is no null result either."""


def partners_of(name: str) -> list[str]:
    """The run that fitted the other arm of a pairing run named ...1 or ...2 (round x3's and m3's layout): the same name
    ending in the other digit, where that run exists; none otherwise."""
    if name[-1:] not in ("1", "2"):
        return []
    other = name[:-1] + ("2" if name[-1] == "1" else "1")
    return [other] if Path(f"{OPS}/{other}/run_config.json").exists() else []


def validity(name: str, partners: Sequence[str], cohort_trait: str, variances: dict[str, float], summary: dict[str, object]) -> tuple[list[str], list[str]]:
    """Why a pairing run is not a valid test of SV gain (SV_GAIN_PRESPEC.txt, validity), and its uncertified arms:
    - an arm whose fit is unscorable (the score step's quarantined) or null (held-out score variance at most NULL_VARIANCE,
      or a NULL_COLLAPSE.json a round's follower wrote in the run or its partner);
    - a run that fitted an arm (``partners``) under another fit step key, which holds the fit's code and inputs, or
      another seed.
    An uncertified arm (the score step's uncertified) is counted and flagged."""
    reasons = [f"its {cohort_trait} fit cannot be scored ({'; '.join(summary['quarantined'][cohort_trait])})"] if cohort_trait in summary.get("quarantined", {}) else []
    reasons += [f"its {arm} fit is null (held-out genetic score variance {variance:.2g}, at most {NULL_VARIANCE:g})"
                for arm, variance in variances.items() if not variance > NULL_VARIANCE]
    for marked in (name, *partners):
        # a round's follower (f7404f6 on) marks each run of a pair where an arm's saved coefficients have mean |b| < 1e-12
        marker = Path(f"{OPS}/{marked}/run/NULL_COLLAPSE.json")
        if marker.exists():
            reasons.append(f"EB fitted no genetic signal ({marked}/run/NULL_COLLAPSE.json: {json.loads(marker.read_text(encoding='utf-8')).get('why', '')})")
    key = Path(f"{OPS}/{name}/run/fit/_KEY").read_text(encoding="utf-8").strip()
    seed = json.load(open(f"{OPS}/{name}/run_config.json", encoding="utf-8"))["seed"]
    for partner in partners:
        keys = {path.read_text(encoding="utf-8").strip() for path in (Path(f"{OPS}/{partner}/run/fit.partial/_KEY"), Path(f"{OPS}/{partner}/run/fit/_KEY")) if path.exists()}
        if key not in keys:
            reasons.append(f"{partner}, which fitted an arm, has another fit step key (code or inputs)")
        if json.load(open(f"{OPS}/{partner}/run_config.json", encoding="utf-8"))["seed"] != seed:
            reasons.append(f"{partner}, which fitted an arm, has another seed")
    return reasons, list(summary.get("uncertified", {}).get(cohort_trait, []))


def read_run(name: str, contrast: str = "arms", partners: Sequence[str] | None = None) -> dict[str, object]:
    """One run's trait as the test reads it, from its cohort and score steps inside the workspace: per held-out person
    the research id, family, ancestry, outcome, covariates and the two scores compared (both arms, or for the ablation
    the with_sv model's score and the same score less its SV terms), with its validity (``validity``; ``partners`` are
    the runs that fitted its arms, by default ``partners_of``)."""
    from sv_pgs.compute_budget import detect_compute_budget
    from sv_pgs.config import TraitType
    from sv_pgs.workspace_pipeline import ARMS, WorkspaceConfig, _fit_cohort, _held_out_rows, _read_scores, _Run, workspace_bindings

    config = json.load(open(f"{OPS}/{name}/run_config.json", encoding="utf-8"))
    run_directory, source = Path(f"{OPS}/{name}/run"), config.get("accept_steps_from")
    cohort_directory = run_directory / "cohort"
    while not (cohort_directory / "cohort.npz").exists() and source:
        # a run that kept its cohort step from an earlier run's tree (accept_steps_from: .../svpgs/<run>/source) holds
        # no copy of it; the earlier run's cohort (or the one that run kept) is the one its score step indexes
        earlier_name = source.rstrip("/").split("/")[-2]
        cohort_directory = Path(f"{OPS}/{earlier_name}/run/cohort")
        source = json.load(open(f"{OPS}/{earlier_name}/run_config.json", encoding="utf-8")).get("accept_steps_from")
    if cohort_directory != run_directory / "cohort":
        # the score step checks its model names against the run's own cohort directory, so the run is read through a
        # view of its step directories with the cohort's taken from the earlier run
        view = Path(f"/tmp/sv_gain_meta_view_{name}")
        shutil.rmtree(view, ignore_errors=True)
        view.mkdir()
        for step in run_directory.iterdir():
            if step.name != "cohort":
                (view / step.name).symlink_to(step)
        (view / "cohort").symlink_to(cohort_directory)
        run_directory = view
    config["run_directory"], config["marker_directory"] = str(run_directory), f"/tmp/sv_gain_meta_markers_{name}"
    path = f"/tmp/sv_gain_meta_{name}.json"
    json.dump(config, open(path, "w", encoding="utf-8"))
    run = _Run(WorkspaceConfig.read(path), workspace_bindings(), detect_compute_budget(), {})
    cohort, scores = _fit_cohort(run), _read_scores(run)
    if scores.held_out_genetic.shape[1] != len(cohort.research_ids):
        raise ValueError(f"{name}: its scores hold {scores.held_out_genetic.shape[1]} people and its cohort {len(cohort.research_ids)}")
    (trait,) = range(len(cohort.trait_names))
    if len(cohort.evaluated_folds(trait, run.config.fold_count)) != 1:
        raise ValueError(f"{name}: the test reads one held-out fold")
    rows = np.flatnonzero(_held_out_rows(cohort, trait))
    variances = {arm: float(np.var(scores.held_out_genetic[ARMS.index(arm), rows, trait])) for arm in ARMS}
    partners = partners_of(name) if partners is None else list(partners)
    invalid, uncertified = validity(name, partners, cohort.trait_names[trait], variances, run.summary("score"))
    return {"name": cohort.trait_names[trait], "binary": cohort.trait_types[trait] == TraitType.BINARY,
            "invalid": invalid, "uncertified": uncertified, "variances": variances, "partners": partners,
            "ids": np.asarray(cohort.research_ids, dtype=object)[rows], "families": cohort.families[rows],
            "ancestry": np.asarray(cohort.ancestry, dtype=object)[rows], "target": cohort.targets[rows, trait],
            "covariates": cohort.covariates[np.ix_(rows, np.flatnonzero(cohort.covariate_columns[trait]))],
            "baseline": (scores.held_out_genetic[ARMS.index("with_sv"), rows, trait] - scores.held_out_structural[rows, trait]
                         if contrast == "ablation" else scores.held_out_genetic[ARMS.index("snv_indel"), rows, trait]),
            "candidate": scores.held_out_genetic[ARMS.index("with_sv"), rows, trait], "seed": run.config.seed}


PER_PERSON = ("ids", "families", "ancestry", "target", "covariates", "baseline", "candidate")


def excluded(run: dict[str, object], ids: set[str], keep: bool = False) -> tuple[dict[str, object], int]:
    """The run without its held-out people whose research id is in ``ids`` (with ``keep``, without those whose id is
    not), and how many it lost."""
    keep = np.array([(str(value) in ids) == keep for value in run["ids"]], dtype=bool)
    return {**run, **{key: np.asarray(run[key])[keep] for key in PER_PERSON}}, int(np.sum(~keep))


def merge_pairs(runs: Sequence[dict[str, object]]) -> dict[str, object]:
    """One trait's matched pairing runs as one ensemble comparison: on the held-out people every run has, each run's
    two scores standardized over those people, then each arm averaged over the runs. Averaging fits of one arm cuts the
    fit-to-fit noise both arms carry; both arms are averaged over the same runs. The outcome must agree between runs.
    Two people share a family where any run puts them in one (``union_clusters``), so the bootstrap and the swaps keep
    every run's family dependence."""
    first = runs[0]
    common = set(map(str, first["ids"]))
    for run in runs[1:]:
        common &= set(map(str, run["ids"]))
    order = [str(value) for value in first["ids"] if str(value) in common]
    merged = {**first}
    rows_of = []
    for run in runs:
        position = {str(value): row for row, value in enumerate(run["ids"])}
        rows_of.append(np.array([position[value] for value in order], dtype=np.int64))
    for key in ("ids", "ancestry", "target", "covariates"):
        merged[key] = np.asarray(first[key])[rows_of[0]]
    merged["families"] = union_clusters([(np.arange(len(order)), np.asarray(run["families"])[rows]) for run, rows in zip(runs, rows_of)], len(order))
    for run, rows in zip(runs[1:], rows_of[1:]):
        if run["name"] != first["name"] or not np.allclose(np.asarray(run["target"], dtype=np.float64)[rows], merged["target"]):
            raise ValueError(f"{run['name']}: runs merged as one trait disagree on the trait or its outcome")

    def standardized(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        spread = float(np.std(values))
        return (values - values.mean()) / spread if spread > 0.0 else values - values.mean()

    for arm in ("baseline", "candidate"):
        merged[arm] = np.mean([standardized(np.asarray(run[arm])[rows]) for run, rows in zip(runs, rows_of)], axis=0)
    merged["pairs"] = len(runs)
    return merged


def union(runs: Sequence[dict[str, object]]) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, int]:
    """The union of the traits' held-out people, matched by research id (each run's "ids", "families" and "ancestry"):
    each trait's people as indices into it, the union clusters (``union_clusters``), each union person's ancestry label
    (lower case; the first run's where runs differ) and how many labels differ between runs."""
    ids, index = np.unique(np.concatenate([run["ids"] for run in runs]).astype(str), return_inverse=True)
    offsets = np.cumsum([0] + [run["ids"].shape[0] for run in runs])
    people = [index[offsets[i]:offsets[i + 1]] for i in range(len(runs))]
    clusters = union_clusters([(people[i], np.asarray(run["families"])) for i, run in enumerate(runs)], ids.shape[0])
    ancestry = np.empty(ids.shape[0], dtype=object)
    disagreements = 0
    for i, run in enumerate(runs):
        labels = np.asarray(run["ancestry"], dtype=object)
        known = np.array([value is not None for value in ancestry[people[i]]])
        disagreements += int(np.sum(known & (ancestry[people[i]] != labels)))
        ancestry[people[i][~known]] = labels[~known]
    return people, clusters, np.char.lower(ancestry.astype(str)), disagreements


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", help="Matched pairing runs (both arms fitted by identical code), one trait each; "
                                                "RUN:PARTNER[,PARTNER] names the runs that fitted its arms (default: RUN with its last digit 1 and 2 swapped).")
    parser.add_argument("--contrast", choices=("arms", "ablation"), default="arms",
                        help="arms (primary): with_sv vs snv_indel; ablation (secondary, descriptive): SV terms zeroed in one fit.")
    parser.add_argument("--exclude", help="A file of research ids to drop from every trait's held-out people (one per line, or "
                                              "a tab-separated table with a research_id column): the long-read panel's members.")
    parser.add_argument("--only", action="store_true", help="Keep only the people in --exclude's file instead of dropping them "
                                                            "(the descriptive long-read stratum, SV_GAIN_PRESPEC.txt secondary 4).")
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    args = parser.parse_args(argv)
    named = [(spec.split(":")[0], spec.split(":")[1].split(",") if ":" in spec else None) for spec in args.runs]
    runs, names = [], []
    for name, partners in named:
        run = read_run(name, args.contrast, partners)
        checked = f"arms fitted by {', '.join([name, *run['partners']])}" if run["partners"] else "both arms fitted in this run"
        print(f"{name}: {run['name']}; {checked}; held-out genetic score variance "
              + ", ".join(f"{arm} {variance:.3g}" for arm, variance in run["variances"].items())
              + (f"; uncertified, counted and flagged: {'; '.join(run['uncertified'])}" if run["uncertified"] else "; both arms certified"))
        if run["invalid"]:
            print(f"{name}: EXCLUDED by the validity rule, not counted and not a null result: {'; '.join(run['invalid'])}")
            continue
        runs.append(run)
        names.append(name)
    if not runs:
        print("no valid matched pairing run")
        return 1
    args.runs = names
    if args.exclude:
        with open(args.exclude, encoding="utf-8") as handle:
            rows = [line.rstrip("\n").split("\t") for line in handle if line.strip()]
        column = rows[0].index("research_id") if "research_id" in rows[0] else None
        ids = {row[column] for row in rows[1:]} if column is not None else {row[0] for row in rows}
        for index, run in enumerate(runs):
            runs[index], lost = excluded(run, ids, keep=args.only)
            print(f"{args.runs[index]}: {shown(lost)} held-out people {'outside' if args.only else 'in'} {args.exclude} "
                  f"({shown(len(ids))} ids) {'dropped' if not args.only else 'dropped, the rest kept'}")
    by_trait: dict[str, list[dict[str, object]]] = {}
    for name, run in zip(args.runs, runs):
        by_trait.setdefault(str(run["name"]), []).append({**run, "run": name})
    runs = [merge_pairs(group) for group in by_trait.values()]
    for run in runs:
        print(f"{run['name']}: {run['pairs']} matched pairing run(s) averaged per arm, {shown(np.asarray(run['ids']).shape[0])} held-out people in all of them")
    people, clusters, labels, disagreements = union(runs)
    traits = [PairedTrait(run["name"], run["binary"], people[i], np.asarray(run["target"], dtype=np.float64), np.asarray(run["covariates"], dtype=np.float64),
                          np.asarray(run["baseline"], dtype=np.float64), np.asarray(run["candidate"], dtype=np.float64)) for i, run in enumerate(runs)]
    print(f"runs {' '.join(args.runs)}; contrast {args.contrast}; union of held-out people {shown(labels.shape[0])}, clusters {shown(int(clusters.max()) + 1)}, "
          f"ancestry labels differing between runs {shown(disagreements)}; resamples {args.resamples}, permutations {args.permutations}")
    result = analyse(traits, clusters, labels, int(runs[0]["seed"]), args.resamples, args.permutations)
    print("\n".join(report(result, traits, args.contrast)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
test per trait on all its held-out people.
- Primary: the two-sided Stouffer combination of the per-trait z = gain / bootstrap SE, equal weights,
  Z = sum(z) / sqrt(1'R1).
- Secondary:
  - the Cauchy combination (ACAT; Liu and Xie 2020) of the per-trait two-sided p, calibrated under arbitrary
    dependence at small levels and powerful when one trait carries the signal;
  - an omnibus Cauchy combination of the Stouffer and ACAT p;
  - one-sided p for a gain (above 0);
  - the random-effects pooled relative gain (gain / the snv_indel arm's partial r^2) by REML with the Hartung-Knapp-
    Sidik-Jonkman interval on k - 1 degrees of freedom;
  - the gain pooled by generalized least squares under the joint-resample covariance
    (``held_out_comparison.meta_analysis``), absolute and relative, over all held-out people and within the EUR, AFR
    and AMR groups.
- Every trait and every pooled gain also gets its one-sided upper 95% bound ("SVs add at most"): the estimate plus
  1.645 standard errors (HKSJ: the t quantile on k - 1 degrees of freedom), absolute and relative to the snv_indel
  arm's partial r^2.

Calibration: a family arm-swap null. In each replicate one fair coin per union family decides whether every member of
that family has the two arms' scores exchanged. Every trait reads the same coins, so the traits keep their dependence.
Where the arms are equally accurate the swapped data have the observed data's distribution, so the replicates give
the null of every statistic. Each replicate repeats the analysis (PERMUTATION_RESAMPLES bootstrap resamples per trait),
which yields:
- the null mean and SD of every z and of Z (near 0 and 1 when the bootstrap SE is right);
- the share of replicates below 0.05 for each p;
- the permutation p of the observed statistics.

Inputs: the per-person arrays of ``PairedTrait`` (``analyse``). ``main`` reads them inside the All of Us workspace from
each named pairing run's cohort and score steps; only aggregates are printed, and every count of 1 to 20 is shown as
<=20.

    python sv_gain_meta.py RUN [RUN ...] [--permutations N] [--resamples N]
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

from sv_pgs.held_out_comparison import PairedComparison, cluster_bootstrap_counts, meta_analysis, weighted_partial_r2

GROUPS = ("eur", "afr", "amr")
"""The ancestry groups pooled across traits (the three largest in the held-out sets)."""
RESAMPLES = 2000
"""Joint bootstrap resamples behind the observed statistics (the report's BOOTSTRAP_RESAMPLES)."""
PERMUTATIONS = 200
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
    """A count as a table may show it: 1 to 20 as <=20 (All of Us's rule for small cells)."""
    return "<=20" if 1 <= count <= 20 else str(count)


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
    relative = estimates_array / np.array(bases)
    relative_draws = draws_array / np.column_stack(base_draws)
    relative_error = np.nanstd(relative_draws, axis=0, ddof=1)
    result: dict[str, object] = {
        "traits": [trait.name for trait in traits], "estimate": estimates_array, "baseline_r2": np.array(bases),
        "ci_low": np.nanquantile(draws_array, 0.025, axis=0), "ci_high": np.nanquantile(draws_array, 0.975, axis=0), **observed,
        "upper_95": estimates_array + UPPER * observed["standard_error"],
        "relative": relative, "relative_standard_error": relative_error, "relative_upper_95": relative + UPPER * relative_error,
        "relative_random": reml_hksj(relative, relative_error ** 2),
        "absolute_pooled": pooled(estimates_array, draws_array), "relative_pooled": pooled(relative, relative_draws),
    }
    groups = {}
    for group in GROUPS:
        cells, used, used_draws, used_relative, used_relative_draws = [], [], [], [], []
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
                cells.append((trait.name, rows.shape[0], cases, gain[0]))
            else:
                cells.append((trait.name, rows.shape[0], cases, math.nan))
        groups[group] = {"cells": cells,
                         "absolute": pooled(np.array(used), np.column_stack(used_draws)) if used else None,
                         "relative": pooled(np.array(used_relative), np.column_stack(used_relative_draws)) if used else None}
    result["groups"] = groups
    if permutations:
        result["calibration"] = calibrate(traits, clusters, weights[:permutation_resamples], generator, permutations, observed)
    return result


def calibrate(traits: Sequence[PairedTrait], clusters: np.ndarray, weights: np.ndarray, generator: np.random.Generator,
              permutations: int, observed: dict[str, object]) -> dict[str, object]:
    """The analysis repeated on family arm swaps (module docstring)."""
    unit = np.ones((1, clusters.shape[0]))
    z_null, total_null, trait_p_null = [], [], []
    p_null: dict[str, list[float]] = {"p_stouffer": [], "p_acat": [], "p_omnibus": [], "p_stouffer_one_sided": []}
    for _ in range(permutations):
        coins = generator.random(int(clusters.max()) + 1) < 0.5
        estimates, draws = [], []
        for trait in traits:
            swap = family_swaps(trait, clusters, coins)
            estimates.append(gains(unit, trait, swap)[0][0])
            draws.append(gains(weights, trait, swap)[0])
        found = combined(np.array(estimates), np.column_stack(draws))
        z_null.append(found["z"])
        trait_p_null.append(found["p"])
        total_null.append(found["Z"])
        for name in p_null:
            p_null[name].append(found[name])
    z_null_array, total_array = np.vstack(z_null), np.array(total_null)
    count = total_array.shape[0]
    return {
        "permutations": count,
        "z_mean": np.mean(z_null_array, axis=0), "z_sd": np.std(z_null_array, axis=0, ddof=1),
        "trait_type_one": np.mean(np.vstack(trait_p_null) < LEVEL, axis=0),
        "Z_mean": float(np.mean(total_array)), "Z_sd": float(np.std(total_array, ddof=1)),
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
    lines.append("R (joint family bootstrap correlation of the gains): " + np.array2string(result["R"], precision=3).replace("\n", " "))
    lines.append(f"PRIMARY Stouffer Z = {result['Z']:.3f}, two-sided p = {result['p_stouffer']:.4g} (one-sided gain > 0: {result['p_stouffer_one_sided']:.4g})")
    lines.append(f"ACAT p = {result['p_acat']:.4g} (from one-sided p: {result['p_acat_one_sided']:.4g}); omnibus Cauchy(Stouffer, ACAT) p = {result['p_omnibus']:.4g}")
    random = result["relative_random"]
    lines.append(f"relative gain, REML + HKSJ: {random['estimate'] * 100:.2f}% [{random['ci_low'] * 100:.2f}, {random['ci_high'] * 100:.2f}], "
                 f"p = {random['p_value']:.4g}, at most {random['upper_95'] * 100:.2f}%, tau^2 = {random['tau_squared']:.3g}")
    lines.extend(pooled_lines("all held-out:", result["absolute_pooled"], result["relative_pooled"]))
    for group, entry in result["groups"].items():
        cells = "; ".join(f"{name} n={shown(people)}" + (f" cases={shown(cases)}" if cases is not None else "")
                          + (f" gain={gain * 1e4:.3f}e-4" if math.isfinite(gain) else " (not pooled)") for name, people, cases, gain in entry["cells"])
        lines.append(f"{group.upper()} cells: {cells}")
        if entry["absolute"] is not None:
            lines.extend(pooled_lines(f"{group.upper()}:", entry["absolute"], entry["relative"]))
    calibration = result.get("calibration")
    if calibration:
        lines.append(f"calibration ({calibration['permutations']} family arm swaps): per-trait null z mean "
                     + np.array2string(calibration["z_mean"], precision=2) + ", SD " + np.array2string(calibration["z_sd"], precision=2)
                     + ", type-I at 0.05 " + np.array2string(calibration["trait_type_one"], precision=3))
        lines.append(f"  Stouffer Z null mean {calibration['Z_mean']:.2f}, SD {calibration['Z_sd']:.2f}; type-I at 0.05: "
                     + ", ".join(f"{name} {value:.3f}" for name, value in calibration["type_one"].items()))
        lines.append(f"  permutation p: Stouffer Z upper {calibration['p_permutation_Z_upper']:.4g}, two-sided {calibration['p_permutation_Z_two_sided']:.4g}; "
                     f"ACAT {calibration['p_permutation_acat']:.4g}; omnibus {calibration['p_permutation_omnibus']:.4g}")
    return lines


OPS = "/home/jupyter/workspace/svpgs_ops/svpgs"


def read_run(name: str, contrast: str = "arms") -> dict[str, object]:
    """One run's trait as the test reads it, from its cohort and score steps inside the workspace: per held-out person
    the research id, family, ancestry, outcome, covariates and the two scores compared (both arms, or for the ablation
    the with_sv model's score and the same score less its SV terms)."""
    from sv_pgs.compute_budget import detect_compute_budget
    from sv_pgs.config import TraitType
    from sv_pgs.workspace_pipeline import ARMS, WorkspaceConfig, _held_out_rows, _read_cohort, _read_scores, _Run, workspace_bindings

    config = json.load(open(f"{OPS}/{name}/run_config.json", encoding="utf-8"))
    config["run_directory"], config["marker_directory"] = f"{OPS}/{name}/run", f"/tmp/sv_gain_meta_markers_{name}"
    path = f"/tmp/sv_gain_meta_{name}.json"
    json.dump(config, open(path, "w", encoding="utf-8"))
    run = _Run(WorkspaceConfig.read(path), workspace_bindings(), detect_compute_budget(), {})
    cohort, scores = _read_cohort(run), _read_scores(run)
    (trait,) = range(len(cohort.trait_names))
    if len(cohort.evaluated_folds(trait, run.config.fold_count)) != 1:
        raise ValueError(f"{name}: the test reads one held-out fold")
    rows = np.flatnonzero(_held_out_rows(cohort, trait))
    return {"name": cohort.trait_names[trait], "binary": cohort.trait_types[trait] == TraitType.BINARY,
            "ids": np.asarray(cohort.research_ids, dtype=object)[rows], "families": cohort.families[rows],
            "ancestry": np.asarray(cohort.ancestry, dtype=object)[rows], "target": cohort.targets[rows, trait],
            "covariates": cohort.covariates[np.ix_(rows, np.flatnonzero(cohort.covariate_columns[trait]))],
            "baseline": (scores.held_out_genetic[ARMS.index("with_sv"), rows, trait] - scores.held_out_structural[rows, trait]
                         if contrast == "ablation" else scores.held_out_genetic[ARMS.index("snv_indel"), rows, trait]),
            "candidate": scores.held_out_genetic[ARMS.index("with_sv"), rows, trait], "seed": run.config.seed}


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
    parser.add_argument("runs", nargs="+", help="Matched pairing runs (both arms fitted by identical code), one trait each.")
    parser.add_argument("--contrast", choices=("arms", "ablation"), default="arms",
                        help="arms (primary): with_sv vs snv_indel; ablation (secondary, descriptive): SV terms zeroed in one fit.")
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    args = parser.parse_args(argv)
    runs = [read_run(name, args.contrast) for name in args.runs]
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

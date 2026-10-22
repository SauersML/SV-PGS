"""Whether the SV-PGS models' own imputed SVs add held-out accuracy when every trait's test is combined, and whether
that combination is calibrated.

Each trait's test compares a baseline score with a candidate score, the baseline plus an ``extra`` part, on the same
held-out people: partial r^2 given the trait's covariates (``held_out_comparison.weighted_partial_r2``, the report's
``_partial_r2`` at unit weights). The within-model SV ablation takes one fitted with_sv model's held-out score as the
candidate and the same score less its structural part (``_Scores.held_out_structural``) as the baseline, so the extra
part is the model's SV terms. The arms contrast (``--contrast arms``, a run that scored both arms) takes the snv_indel
arm as the baseline and the with_sv arm as the candidate.

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
  - the random-effects pooled relative gain (gain / the baseline's partial r^2) by REML with the Hartung-Knapp-
    Sidik-Jonkman interval on k - 1 degrees of freedom;
  - the absolute gain pooled by generalized least squares under the joint-resample covariance
    (``held_out_comparison.meta_analysis``), over all held-out people and within the EUR, AFR and AMR groups.

Calibration: Freedman-Lane permutations (Anderson and Robinson 2001). Per trait, the extra part is regressed on the
covariates and the baseline score on the held-out people. Its residuals are permuted among whole families of one
size and one ancestry make-up, and added back to the fitted values, so the permuted extra part keeps its relation to
the baseline and covariates and loses any to the outcome beyond them. One random key per union family orders every
trait's families in a replicate, so traits on the same people get the same permutation and keep their dependence.
Each replicate repeats the whole analysis (PERMUTATION_RESAMPLES bootstrap resamples per trait): the null mean and SD of
every z and of Z, the share of replicates below 0.05 for each p, and the permutation p of the observed statistics.
Under this null the extra part adds only noise, so a gain's expectation is at or below 0; the null SD of z, near 1 when
the bootstrap SE is right, is the calibration read here.

Inputs: the per-person arrays of ``AblationTrait`` (``analyse``). ``main`` reads them inside the All of Us workspace
from each named run's cohort and score steps; only aggregates are printed, and every count of 1 to 20 is shown as <=20.

    python sv_gain_meta.py RUN [RUN ...] [--contrast ablation|arms] [--permutations N]
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

from sv_pgs.held_out_comparison import cluster_bootstrap_counts, meta_analysis, weighted_partial_r2

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


@dataclass(frozen=True)
class AblationTrait:
    """One trait's test on its held-out people, each person an index into the union of every trait's held-out people."""

    name: str
    binary: bool
    people: np.ndarray
    target: np.ndarray
    covariates: np.ndarray
    baseline: np.ndarray
    extra: np.ndarray


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


def gains(weights: np.ndarray, trait: AblationTrait, extra: np.ndarray | None = None, rows: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per row of ``weights`` (resamples x union people): the candidate's partial r^2 less the baseline's, and the
    baseline's, on the trait's held-out people (``rows`` of them only, if given), ``extra`` in place of its own."""
    extra = trait.extra if extra is None else extra
    keep = np.arange(trait.people.shape[0]) if rows is None else rows
    scores = np.vstack([trait.baseline[keep] + extra[keep], trait.baseline[keep]])
    found = [weighted_partial_r2(weights[start:start + CHUNK][:, trait.people[keep]], trait.covariates[keep], trait.target[keep], scores)
             for start in range(0, weights.shape[0], CHUNK)]
    values = np.vstack(found)
    return values[:, 0] - values[:, 1], values[:, 1]


def acat(p_values: Sequence[float]) -> float:
    """The Cauchy combination of two-sided p-values with equal weights: T = mean tan((0.5 - p) pi),
    p = 0.5 - arctan(T) / pi, read as 1 / (T pi) where T is large."""
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
    Jonkman standard error sqrt(q / sum w), q = sum w (y - mean)^2 / (k - 1), with its t interval and p on k - 1
    degrees of freedom."""
    y, v = np.asarray(estimates, dtype=np.float64), np.asarray(variances, dtype=np.float64)
    k = y.shape[0]
    tau2 = max(0.0, float(np.var(y, ddof=1) - np.mean(v))) if k > 1 else 0.0
    for _ in range(500):
        w = 1.0 / (v + tau2)
        mean = float(np.sum(w * y) / np.sum(w))
        updated = max(0.0, float(np.sum(w ** 2 * ((y - mean) ** 2 - v)) / np.sum(w ** 2) + 1.0 / np.sum(w)))
        if abs(updated - tau2) <= 1e-12 * max(1.0, tau2):
            tau2 = updated
            break
        tau2 = updated
    w = 1.0 / (v + tau2)
    mean = float(np.sum(w * y) / np.sum(w))
    q = float(np.sum(w * (y - mean) ** 2) / (k - 1)) if k > 1 else math.nan
    error = math.sqrt(q / float(np.sum(w))) if k > 1 else math.nan
    quantile = float(stats.t.ppf(0.975, k - 1)) if k > 1 else math.nan
    statistic = mean / error if error > 0 else math.nan
    return {"estimate": mean, "standard_error": error, "ci_low": mean - quantile * error, "ci_high": mean + quantile * error,
            "p_value": float(2.0 * stats.t.sf(abs(statistic), k - 1)) if k > 1 and math.isfinite(statistic) else math.nan, "tau_squared": tau2}


def correlation_of(draws: np.ndarray) -> np.ndarray:
    """The correlation of the traits' resampled gains (resamples x traits), over the resamples where all are finite."""
    finite = draws[np.all(np.isfinite(draws), axis=1)]
    return np.atleast_2d(np.corrcoef(finite, rowvar=False)) if draws.shape[1] > 1 else np.ones((1, 1))


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
            "p_stouffer": p_stouffer, "p_stouffer_one_sided": p_stouffer_one, "p_acat": p_acat,
            "p_acat_one_sided": acat(one_sided), "p_omnibus": acat([p_stouffer, p_acat])}


@dataclass(frozen=True)
class FamilyBlocks:
    """A trait's held-out families grouped into strata of one size and one ancestry make-up: per stratum, the families'
    union cluster ids and a families x size matrix of their members (indices into the trait's people, each family's
    members ordered by ancestry, then position)."""

    clusters: list[np.ndarray]
    members: list[np.ndarray]


def family_blocks(trait: AblationTrait, clusters: np.ndarray, ancestry: np.ndarray) -> FamilyBlocks:
    """The trait's ``FamilyBlocks``: its people's union clusters, each cluster's members on the trait's held-out rows."""
    own = clusters[trait.people]
    labels = ancestry[trait.people]
    order = np.lexsort((np.arange(own.shape[0]), labels, own))
    strata: dict[tuple, tuple[list[int], list[np.ndarray]]] = {}
    starts = np.flatnonzero(np.r_[True, own[order][1:] != own[order][:-1]])
    for start, stop in zip(starts, np.r_[starts[1:], order.shape[0]]):
        members = order[start:stop]
        key = (members.shape[0], tuple(labels[members]))
        entry = strata.setdefault(key, ([], []))
        entry[0].append(int(own[members[0]]))
        entry[1].append(members)
    return FamilyBlocks([np.array(ids) for ids, _ in strata.values()], [np.vstack(members) for _, members in strata.values()])


def permuted(residual: np.ndarray, blocks: FamilyBlocks, keys: np.ndarray) -> np.ndarray:
    """``residual`` with each stratum's families' blocks moved among them: family i (in cluster order) takes the block
    of the family ranked i-th by ``keys`` (one per union cluster)."""
    result = np.empty_like(residual)
    for ids, members in zip(blocks.clusters, blocks.members):
        result[members] = residual[members[np.argsort(keys[ids], kind="stable")]]
    return result


def freedman_lane(trait: AblationTrait) -> tuple[np.ndarray, np.ndarray]:
    """The extra part's least-squares fit on [covariates, baseline] over the trait's held-out people, and its residual."""
    design = np.column_stack([trait.covariates, trait.baseline])
    coefficients = np.linalg.lstsq(design, trait.extra, rcond=None)[0]
    fitted = design @ coefficients
    return fitted, trait.extra - fitted


def group_rows(trait: AblationTrait, ancestry: np.ndarray, group: str) -> np.ndarray:
    return np.flatnonzero(ancestry[trait.people] == group)


def analyse(traits: Sequence[AblationTrait], clusters: np.ndarray, ancestry: np.ndarray, seed: int, resamples: int = RESAMPLES,
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
    relative_variance = np.nanvar(relative_draws, axis=0, ddof=1)
    covariance = np.atleast_2d(np.cov(draws_array[np.all(np.isfinite(draws_array), axis=1)], rowvar=False))
    result: dict[str, object] = {
        "traits": [trait.name for trait in traits], "estimate": estimates_array, "baseline_r2": np.array(bases),
        "ci_low": np.nanquantile(draws_array, 0.025, axis=0), "ci_high": np.nanquantile(draws_array, 0.975, axis=0),
        "relative": relative, "relative_standard_error": np.sqrt(relative_variance), **observed,
        "relative_random": reml_hksj(relative, relative_variance), "absolute_pooled": meta_analysis(estimates_array, covariance),
    }
    groups = {}
    for group in GROUPS:
        entries = []
        for index, trait in enumerate(traits):
            rows = group_rows(trait, ancestry, group)
            cases = int(np.sum(trait.target[rows])) if trait.binary else None
            usable = rows.shape[0] >= MINIMUM_GROUP_PEOPLE and (cases is None or cases > 20)
            if usable:
                gain, _ = gains(unit, trait, rows=rows)
                gain_draws, _ = gains(weights, trait, rows=rows)
                entries.append((index, rows.shape[0], cases, gain[0], gain_draws))
            else:
                entries.append((index, rows.shape[0], cases, math.nan, None))
        used = [entry for entry in entries if entry[4] is not None]
        if used:
            group_draws = np.column_stack([entry[4] for entry in used])
            finite = group_draws[np.all(np.isfinite(group_draws), axis=1)]
            pooled = meta_analysis(np.array([entry[3] for entry in used]), np.atleast_2d(np.cov(finite, rowvar=False)))
        else:
            pooled = None
        groups[group] = {"cells": [(traits[index].name, people, cases, gain) for index, people, cases, gain, _ in entries], "pooled": pooled}
    result["groups"] = groups
    if permutations:
        result["calibration"] = calibrate(traits, clusters, ancestry, weights[:permutation_resamples], generator, permutations, observed)
    return result


def calibrate(traits: Sequence[AblationTrait], clusters: np.ndarray, ancestry: np.ndarray, weights: np.ndarray,
              generator: np.random.Generator, permutations: int, observed: dict[str, object]) -> dict[str, object]:
    """The analysis repeated on Freedman-Lane permutations of every trait's extra part (module docstring)."""
    fits = [freedman_lane(trait) for trait in traits]
    blocks = [family_blocks(trait, clusters, ancestry) for trait in traits]
    unit = np.ones((1, clusters.shape[0]))
    z_null, total_null, p_null = [], [], {"p_stouffer": [], "p_acat": [], "p_omnibus": [], "p_stouffer_one_sided": []}
    trait_p_null = []
    for _ in range(permutations):
        keys = generator.random(int(clusters.max()) + 1)
        estimates, draws = [], []
        for trait, (fitted, residual), block in zip(traits, fits, blocks):
            extra = fitted + permuted(residual, block, keys)
            estimates.append(gains(unit, trait, extra)[0][0])
            draws.append(gains(weights, trait, extra)[0])
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


def normal(comparison) -> str:
    """A pooled ``PairedComparison`` (x1e-4) with its normal 95% interval and two-sided p."""
    half = 1.959963984540054 * comparison.standard_error
    p = 2.0 * stats.norm.sf(abs(comparison.z_score)) if math.isfinite(comparison.z_score) else math.nan
    return (f"{comparison.estimate * 1e4:.3f}e-4 [{(comparison.estimate - half) * 1e4:.3f}, {(comparison.estimate + half) * 1e4:.3f}], "
            f"z = {comparison.z_score:.3f}, p = {p:.4g}")


def report(result: dict[str, object], traits: Sequence[AblationTrait], ancestry: np.ndarray) -> list[str]:
    """The printed lines: aggregates only, every count of 1 to 20 shown as <=20."""
    names = result["traits"]
    lines = ["per trait: trait n cases gain(x1e-4) CI SE z p_two p_one baseline_r2 relative_gain relative_SE"]
    for index, trait in enumerate(traits):
        cases = shown(int(np.sum(trait.target))) if trait.binary else ""
        lines.append("  " + "\t".join(str(value) for value in (
            names[index], shown(trait.people.shape[0]), cases, f"{result['estimate'][index] * 1e4:.3f}",
            f"[{result['ci_low'][index] * 1e4:.3f}, {result['ci_high'][index] * 1e4:.3f}]", f"{result['standard_error'][index] * 1e4:.3f}",
            f"{result['z'][index]:.3f}", f"{result['p'][index]:.4g}", f"{result['p_one_sided'][index]:.4g}", f"{result['baseline_r2'][index]:.5g}",
            f"{result['relative'][index]:.4f}", f"{result['relative_standard_error'][index]:.4f}")))
    lines.append("R (joint family bootstrap correlation of the gains): " + np.array2string(result["R"], precision=3).replace("\n", " "))
    lines.append(f"PRIMARY Stouffer Z = {result['Z']:.3f}, two-sided p = {result['p_stouffer']:.4g} (one-sided gain > 0: {result['p_stouffer_one_sided']:.4g})")
    lines.append(f"ACAT p = {result['p_acat']:.4g} (from one-sided p: {result['p_acat_one_sided']:.4g}); omnibus Cauchy(Stouffer, ACAT) p = {result['p_omnibus']:.4g}")
    random = result["relative_random"]
    lines.append(f"relative gain, REML + HKSJ: {random['estimate']:.4f} [{random['ci_low']:.4f}, {random['ci_high']:.4f}], p = {random['p_value']:.4g}, "
                 f"tau^2 = {random['tau_squared']:.3g}")
    pooled = result["absolute_pooled"]
    lines.append(f"absolute gain, GLS fixed (joint covariance): {normal(pooled.fixed)}; random: {normal(pooled.random)}; "
                 f"Q p = {pooled.heterogeneity_p:.3g}, I^2 = {pooled.i_squared:.2f}, tau^2 = {pooled.tau_squared:.3g}")
    for group, entry in result["groups"].items():
        cells = "; ".join(f"{name} n={shown(people)}" + (f" cases={shown(cases)}" if cases is not None else "")
                          + (f" gain={gain * 1e4:.3f}e-4" if math.isfinite(gain) else " (not pooled)") for name, people, cases, gain in entry["cells"])
        pooled = entry["pooled"]
        summary = f"pooled fixed {normal(pooled.fixed)}" if pooled is not None else "nothing pooled"
        lines.append(f"{group.upper()}: {summary} | {cells}")
    calibration = result.get("calibration")
    if calibration:
        lines.append(f"calibration ({calibration['permutations']} Freedman-Lane family permutations): per-trait null z mean "
                     + np.array2string(calibration["z_mean"], precision=2) + ", SD " + np.array2string(calibration["z_sd"], precision=2)
                     + ", type-I at 0.05 " + np.array2string(calibration["trait_type_one"], precision=3))
        lines.append(f"  Stouffer Z null mean {calibration['Z_mean']:.2f}, SD {calibration['Z_sd']:.2f}; type-I at 0.05: "
                     + ", ".join(f"{name} {value:.3f}" for name, value in calibration["type_one"].items()))
        lines.append(f"  permutation p: Stouffer Z upper {calibration['p_permutation_Z_upper']:.4g}, two-sided {calibration['p_permutation_Z_two_sided']:.4g}; "
                     f"ACAT {calibration['p_permutation_acat']:.4g}; omnibus {calibration['p_permutation_omnibus']:.4g}")
    return lines


OPS = "/home/jupyter/workspace/svpgs_ops/svpgs"


def read_run(name: str, contrast: str) -> dict[str, object]:
    """One run's trait as the test reads it, from its cohort and score steps inside the workspace: per held-out person
    the research id, family, ancestry, outcome, covariates, the baseline score and the extra part."""
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
    full = scores.held_out_genetic[ARMS.index("with_sv"), rows, trait]
    if contrast == "ablation":
        extra = scores.held_out_structural[rows, trait]
        baseline = full - extra
    else:
        baseline = scores.held_out_genetic[ARMS.index("snv_indel"), rows, trait]
        extra = full - baseline
    return {"name": cohort.trait_names[trait], "binary": cohort.trait_types[trait] == TraitType.BINARY,
            "ids": np.asarray(cohort.research_ids, dtype=object)[rows], "families": cohort.families[rows],
            "ancestry": np.asarray(cohort.ancestry, dtype=object)[rows], "target": cohort.targets[rows, trait],
            "covariates": cohort.covariates[np.ix_(rows, np.flatnonzero(cohort.covariate_columns[trait]))],
            "baseline": baseline, "extra": extra, "seed": run.config.seed}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--contrast", choices=("ablation", "arms"), default="ablation")
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    args = parser.parse_args(argv)
    runs = [read_run(name, args.contrast) for name in args.runs]
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
    traits = [AblationTrait(run["name"], run["binary"], people[i], np.asarray(run["target"], dtype=np.float64), np.asarray(run["covariates"], dtype=np.float64),
                            np.asarray(run["baseline"], dtype=np.float64), np.asarray(run["extra"], dtype=np.float64)) for i, run in enumerate(runs)]
    print(f"runs {' '.join(args.runs)}; contrast {args.contrast}; union of held-out people {shown(ids.shape[0])}, clusters {shown(int(clusters.max()) + 1)}, "
          f"ancestry labels differing between runs {shown(disagreements)}; resamples {args.resamples}, permutations {args.permutations}")
    labels = np.char.lower(ancestry.astype(str))
    result = analyse(traits, clusters, labels, int(runs[0]["seed"]), args.resamples, args.permutations)
    print("\n".join(report(result, traits, labels)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

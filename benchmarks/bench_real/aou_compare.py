"""The two arms and LDpred2-auto on the same held-out people: each score's partial r^2 and AUC with 95% intervals, and
their paired differences.

Per trait, the people are those held out of its models (the report's cell "all"), each scored by the model that left
their fold out: the snv_indel and with_sv genetic scores (the score step's ``held_out_genetic``) and the LDpred2 score
(aou_ldpred2.py's ``--scores``). Each score is read as the report reads it: its partial r^2 (``_partial_r2``: the
squared correlation with the outcome after both are residualized on the trait's covariates, the structure columns with
the genetic PCs and the trait's own) and, for a disease, the AUC of the score alone (raw: no covariate enters it). The
standard errors are the report's, family-clustered influence functions conditional on the fitted models: r^2's through
the correlation's (``held_out_comparison``), the AUC's through DeLong's placements, and the paired differences are the
report's comparisons (``paired_delta_r2`` of the residualized scores, ``paired_delta_auc``), positive where the
first-named score is the better. As in the report, every value is read on each fold's own people, on the folds
``_scored_folds`` keeps, and the folds are combined with weights n_f / n (``_fold_combined``): one fold in a single-split
run, where the error is the fold's own; across several folds the error leaves out the folds' shared-training
covariance and is descriptive. An interval is the estimate plus and minus the normal 97.5% quantile times its standard
error.

Only aggregates are written, with the report's suppression (``_shown``): a trait's values are shown only where its
people, and a disease's cases and controls, number at least MINIMUM_REPORTED_PARTICIPANTS. The three scores must cover
the same people wherever values are shown, so no count follows from this table less the report's.

A run that fits regions (``WorkspaceConfig.fit_regions``) is compared region by region, each region's arms (its run
directory's score step) beside its LDpred2 scores; the column "region" names it, and is empty for a run without them.
"""
from __future__ import annotations

import argparse
import csv
import math

import numpy as np
from scipy import stats

from sv_pgs import held_out_comparison
from sv_pgs.all_of_us import reportable_count
from sv_pgs.compute_budget import detect_compute_budget
from sv_pgs.config import TraitType
from sv_pgs.workspace_pipeline import (
    ARMS, COMPLETE_FILE, WorkspaceConfig, _cases_text, _evaluated_counts, _fold_combined, _HeldOutCell, _partial_r2, _fit_cohort, _read_scores,
    _residualized, _Run, _scored_folds, _shown, workspace_bindings,
)

COLUMNS = ("region", "trait", "score", "n", "cases", "metric", "estimate", "standard_error", "ci_lower", "ci_upper")
SCORES = (*ARMS, "ldpred2")
PAIRS = (("with_sv", "snv_indel"), ("snv_indel", "ldpred2"), ("with_sv", "ldpred2"))
"""The paired differences, each the first score's metric less the second's."""
INTERVAL_Z = float(stats.norm.ppf(0.975))


def partial_r2(outcome: np.ndarray, score: np.ndarray, covariates: np.ndarray, families: np.ndarray) -> tuple[float, float]:
    """The partial r^2 of ``score`` and its family-clustered standard error, whose influence is 2 r times the residualized
    correlation's (``paired_delta_r2``'s, with no baseline)."""
    correlation, influence = held_out_comparison._correlation_influence(_residualized(outcome, covariates), _residualized(score, covariates))
    totals = held_out_comparison.cluster_totals(2.0 * correlation * influence, families)
    return _partial_r2(outcome, score, covariates), math.sqrt(float(totals @ totals)) / outcome.shape[0]


def auc(labels: np.ndarray, score: np.ndarray, families: np.ndarray) -> tuple[float, float]:
    """The AUC of ``score`` and its family-clustered DeLong standard error: ``paired_delta_auc`` against a constant score,
    whose AUC is 1/2 and whose placements are all 1/2, so the difference's influence is the score's own."""
    comparison = held_out_comparison.paired_delta_auc(labels, np.zeros_like(score), score, families)
    return comparison.estimate + 0.5, comparison.standard_error


def trait_rows(cohort, genetic: np.ndarray, trait: int) -> list[tuple[str, ...]]:
    """One trait's rows of COLUMNS but the region from ``genetic`` [SCORES, cohort rows], each score's held-out values
    (NaN elsewhere)."""
    name, binary = cohort.trait_names[trait], cohort.trait_types[trait] == TraitType.BINARY
    targets = cohort.targets[:, trait]
    observed = np.isfinite(targets) & np.any(np.isfinite(genetic), axis=0)
    folds = _HeldOutCell(tuple(np.flatnonzero(observed & (cohort.folds == fold)) for fold in np.unique(cohort.folds[observed]).tolist()))
    selected = _scored_folds(folds, targets, binary)
    count, cases = _evaluated_counts(selected, targets, binary)
    counts = (reportable_count(count), _cases_text(count, cases))
    shown = _shown(count, cases)
    if shown and not all(np.all(np.isfinite(genetic[:, rows])) for rows in selected):
        raise ValueError(f"the arms and LDpred2 scored different held-out people of {name}, so their counts would differ by a few.")
    table: list[tuple[str, ...]] = []

    def single(metric: str, values: np.ndarray, rows: np.ndarray) -> held_out_comparison.PairedComparison:
        if metric == "pgs_partial_r2":
            covariates = cohort.covariates[np.ix_(rows, np.flatnonzero(cohort.covariate_columns[trait]))]
            return held_out_comparison.comparison_from_estimate(*partial_r2(targets[rows], values, covariates, cohort.families[rows]))
        return held_out_comparison.comparison_from_estimate(*auc(targets[rows], values, cohort.families[rows]))

    def paired(metric: str, baseline: np.ndarray, candidate: np.ndarray, rows: np.ndarray) -> held_out_comparison.PairedComparison:
        if metric == "pgs_partial_r2":
            covariates = cohort.covariates[np.ix_(rows, np.flatnonzero(cohort.covariate_columns[trait]))]
            residual = [_residualized(series, covariates) for series in (targets[rows], baseline, candidate)]
            return held_out_comparison.paired_delta_r2(*residual, cohort.families[rows])
        return held_out_comparison.paired_delta_auc(targets[rows], baseline, candidate, cohort.families[rows])

    def add(score: str, metric: str, per_fold: list[tuple[int, held_out_comparison.PairedComparison]]) -> None:
        combined = _fold_combined(per_fold)
        interval = (combined.estimate - INTERVAL_Z * combined.standard_error, combined.estimate + INTERVAL_Z * combined.standard_error)
        table.append((name, score, *counts, metric, *(f"{value:.6g}" for value in (combined.estimate, combined.standard_error, *interval))))

    for metric in ("pgs_partial_r2", "pgs_auc") if binary else ("pgs_partial_r2",):
        if not shown:
            table.extend((name, score, *counts, metric, "", "", "", "") for score in (*SCORES, *(f"{first} - {second}" for first, second in PAIRS)))
            continue
        for index, score in enumerate(SCORES):
            add(score, metric, [(rows.shape[0], single(metric, genetic[index, rows], rows)) for rows in selected])
        for first, second in PAIRS:
            add(f"{first} - {second}", f"delta_{metric}", [
                (rows.shape[0], paired(metric, genetic[SCORES.index(second), rows], genetic[SCORES.index(first), rows], rows)) for rows in selected
            ])
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="The run config the driver ran (its run directory holds the finished steps).")
    parser.add_argument("--scores", required=True, help="aou_ldpred2.py's held-out scores (--scores) of the same run.")
    parser.add_argument("--out", required=True, help="The aggregate table to write (COLUMNS, tab-separated).")
    args = parser.parse_args(argv)
    config = WorkspaceConfig.read(args.config)
    budget, bindings = detect_compute_budget(), workspace_bindings()
    regions = config.region_runs() or (("", config),)
    scored = [(index, name, _Run(region, bindings, budget, {})) for index, (name, region) in enumerate(regions)]
    scored = [(index, name, run) for index, name, run in scored if (run.output("score") / COMPLETE_FILE).is_file()]
    if not scored:
        print(f"{config.run_directory} holds no scored arms (its driver stopped before the score step): nothing to compare", flush=True)
        return 0
    cohort = _fit_cohort(_Run(config, bindings, budget, {}))
    with np.load(args.scores, allow_pickle=False) as ldpred2:
        if (
            tuple(ldpred2["regions"].tolist()) != tuple(name for name, _region in regions)
            or tuple(ldpred2["research_ids"].tolist()) != cohort.research_ids or tuple(ldpred2["trait_names"].tolist()) != cohort.trait_names
        ):
            raise ValueError(f"{args.scores} holds another run's regions, cohort rows or traits than {config.run_directory}'s.")
        held_out = ldpred2["held_out_score"]
    table = [COLUMNS]
    for index, name, run in scored:
        scores, quarantined = _read_scores(run), run.summary("score")["quarantined"]
        for trait in range(len(cohort.trait_names)):
            if cohort.trait_names[trait] in quarantined:
                # The score step left it out as unscorable (the report's quarantined.tsv): its arms have no held-out scores.
                print(f"{name or config.run_directory}: {cohort.trait_names[trait]} is quarantined: not compared", flush=True)
                continue
            genetic = np.vstack([scores.held_out_genetic[:, :, trait], held_out[index, None, :, trait]])
            table.extend((name, *row) for row in trait_rows(cohort, genetic, trait))
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle, delimiter="\t").writerows(table)
    for row in table:
        print("\t".join(row), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

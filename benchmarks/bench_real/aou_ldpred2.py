"""LDpred2-auto beside an All of Us run's fit: the summary-statistic baseline on the run's own cohort, store and split.

Everything that decides the comparison is the run's: each trait's model's training rows and held-out rows
(``_Cohort.training`` and ``held_out`` of each evaluated fold model), the trait's own covariates, the store the fit
reads (the measurement step's recalibrated dosages) and the snv_indel arm's records (``fit_filtered`` at the run
config's floors, site list and region, structural records left out). The GWAS, the 3 cM band of in-sample LD and
LDpred2-auto's settings are ldpred2_auto_allvar.py's, unchanged (the bench-sim LDpred2 arm): covariate-adjusted marginal
effects by Frisch-Waugh-Lovell, the projected dosages' correlations streamed column block by column block, h2_init from
LD score regression floored at 0.001, 30 chains on seq_log(1e-4, 0.2, 30), allow_jump_sign = FALSE, shrink_corr = 0.95.

Genetic positions restart at each chromosome, and the band is a window in them: each chromosome's are offset by
CHROMOSOME_SPACING_CM, so no window reaches another chromosome and the genome is one band in store order.

The held-out genetic score is the posterior mean effects on the centred held-out dosages, read as the report reads the
fit's (``_partial_r2``: the squared correlation with the target after both are residualized on the trait's covariates).
Only aggregates leave the VM: per trait, the held-out row count (never 1 to 20), the partial r^2 and the variant count.
Each held-out person's score, by its own model, goes to ``--scores`` by cohort row (``held_out_scores``), so the arms
and LDpred2 are compared on the same people (aou_compare.py): participant-level data, which stays in the workspace
beside the run's mirror, and which no step stages for export.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib

import numpy as np

from benchmarks.bench_sim.submissions import ldpred2_auto_allvar
from sv_pgs.all_of_us import MINIMUM_REPORTED_PARTICIPANTS
from sv_pgs.config import TraitType
from sv_pgs.compute_budget import detect_compute_budget
from sv_pgs.dosage_store import DosageStore
from sv_pgs.workspace_pipeline import (
    WorkspaceConfig, _final_measurement_model, _final_store, _fold_weighted, _partial_r2, _read_cohort, _Run, _shown, arm_log_reliability, fit_filtered, workspace_bindings,
)

CHROMOSOME_SPACING_CM = 1e4
"""Each chromosome's genetic positions are offset by its number times this: longer than any chromosome's map (chr1's
is about 290 cM) and the band's 3 cM window, so a window never reaches another chromosome."""
SMALLEST_SHOWN_COUNT = MINIMUM_REPORTED_PARTICIPANTS
"""The workspace report's threshold: no participant count from 1 to 20 leaves the workspace."""


class StoreView:
    """The bench-sim training view (``codes``, ``variants``, ``covariates``, ``phenotype``, ``n_samples``, ``cores``) of
    a store's ``records`` over the ascending store columns ``columns``: row r of the view is store record ``records[r]``."""

    def __init__(self, store: DosageStore, records: np.ndarray, columns: np.ndarray, covariates: np.ndarray, phenotype: np.ndarray) -> None:
        table = store.variant_table
        self.store, self.records, self.columns = store, records, np.asarray(columns, dtype=np.int64)
        self.variants = {
            "cm": np.asarray(table.chromosome, dtype=np.float64)[records] * CHROMOSOME_SPACING_CM + np.asarray(table.genetic_position_cm, dtype=np.float64)[records],
        }
        if np.any(np.diff(self.variants["cm"]) < 0.0):
            raise ValueError("the records' genetic positions do not rise in store order within each chromosome.")
        self.covariates, self.phenotype = covariates, phenotype
        self.n_samples = int(self.columns.shape[0])
        self.cores = os.cpu_count() or 1

    def codes(self, rows: np.ndarray) -> np.ndarray:
        """Codes [rows, columns] of the view's rows (dosage = code / 127), each run of consecutive records in one read."""
        records = self.records[np.asarray(rows, dtype=np.int64)]
        out = np.empty((records.shape[0], self.n_samples), dtype=np.uint8)
        breaks = np.flatnonzero(np.diff(records) != 1) + 1
        for first, last in zip(np.r_[0, breaks].tolist(), np.r_[breaks, records.shape[0]].tolist()):
            self.store.read_codes(int(records[first]), int(records[last - 1]) + 1, self.columns, out=out[first:last])
        return out


def cohort_rows(cohort, rows: np.ndarray) -> np.ndarray:
    """The cohort's ``rows`` (a mask) as indices in store column order, the order of their view (``cohort_view``)."""
    rows = np.flatnonzero(rows)
    return rows[np.argsort(cohort.store_columns[rows], kind="stable")]


def cohort_view(store: DosageStore, records: np.ndarray, cohort, rows: np.ndarray, trait: int) -> StoreView:
    """The view of the cohort's ``rows`` (a mask) for ``trait``: its target and own covariates, in store column order."""
    rows = cohort_rows(cohort, rows)
    covariates = np.asarray(cohort.covariates[np.ix_(rows, np.flatnonzero(cohort.covariate_columns[trait]))], dtype=np.float64)
    return StoreView(store, records, cohort.store_columns[rows], covariates, np.asarray(cohort.targets[rows, trait], dtype=np.float64))


def held_out_score(view: StoreView, means: np.ndarray, effects: np.ndarray) -> np.ndarray:
    """Sum over the view's records of (dosage - training mean) x effect, for each of the view's columns."""
    total = np.zeros(view.n_samples)
    for start in range(0, view.records.shape[0], ldpred2_auto_allvar.CHUNK):
        rows = np.arange(start, min(start + ldpred2_auto_allvar.CHUNK, view.records.shape[0]))
        total += (view.codes(rows).astype(np.float64) / 127.0 - means[rows, None]).T @ effects[rows]
    return total


def ldpred2_trait(store: DosageStore, records: np.ndarray, cohort, trait: int) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    """LDpred2-auto on each evaluated model's own training and held-out rows, on the report's partial-r^2 scale: the
    trait's aggregate line, and its held-out cohort rows with their scores, each person's by the model that left them out.

    The all-rows model has no held-out rows and never enters a comparison. Across folds the report averages the
    per-fold correlations with row-count weights, so the baseline does too; pooling differently scaled fold scores
    would answer another question. Empty and suppressed cells do not launch a fit, and a disease's case and control
    counts obey the same display rule even though this JSON row does not itself print those counts.
    """
    models = cohort.trait_models(trait) if hasattr(cohort, "trait_models") else (cohort.all_rows_model(trait),)
    evaluated = [(model, np.flatnonzero(cohort.held_out[:, model] & np.isfinite(cohort.targets[:, trait]))) for model in models]
    evaluated = [(model, rows) for model, rows in evaluated if rows.size]
    binary = hasattr(cohort, "trait_types") and cohort.trait_types[trait] == TraitType.BINARY

    def counts():
        return sum(rows.size for _, rows in evaluated), int(sum(cohort.targets[rows, trait].sum() for _, rows in evaluated)) if binary else None

    count, cases = counts()
    if _shown(count, cases):
        evaluated = [(model, rows) for model, rows in evaluated if
                     (min(int(cohort.targets[rows, trait].sum()), rows.size - int(cohort.targets[rows, trait].sum())) >= 2
                      if binary else np.ptp(cohort.targets[rows, trait]) > 0.0)]
        count, cases = counts()
    values, held_rows, held_scores = [], [np.zeros(0, dtype=np.int64)], [np.zeros(0)]
    if _shown(count, cases):
        for model, rows in evaluated:
            means, effects = ldpred2_auto_allvar.posterior_effects(
                cohort_view(store, records, cohort, cohort.training[:, model], trait), np.arange(records.shape[0]))
            held = np.zeros(cohort.targets.shape[0], dtype=bool)
            held[rows] = True
            test = cohort_view(store, records, cohort, held, trait)
            score = held_out_score(test, means, effects)
            values.append((test.n_samples, _partial_r2(test.phenotype, score, test.covariates)))
            held_rows.append(cohort_rows(cohort, held))
            held_scores.append(score)
    value = _fold_weighted(values)
    return {
        "trait": cohort.trait_names[trait],
        "held_out_rows": count if count == 0 or count >= SMALLEST_SHOWN_COUNT else f"<{SMALLEST_SHOWN_COUNT}",
        "variants": int(records.shape[0]),
        "pgs_partial_r2": value if np.isfinite(value) else None,
    }, np.concatenate(held_rows), np.concatenate(held_scores)


def held_out_scores(path: str | pathlib.Path, cohort, scores: np.ndarray, variants: np.ndarray) -> None:
    """Each held-out person's LDpred2 score, [cohort rows, traits] in the cohort's row and trait order (NaN where the
    person is not held out of the trait or its cell was not fitted), with the cohort's research IDs and trait names to
    check that order by."""
    with open(path, "wb") as handle:
        np.savez(
            handle, research_ids=np.array(cohort.research_ids), trait_names=np.array(cohort.trait_names), held_out_score=scores, variants=variants,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="The run config the driver ran (its run directory holds the finished steps).")
    parser.add_argument("--out", required=True, help="The aggregate table to write (one JSON line per trait).")
    parser.add_argument("--scores", required=True, help="The held-out people's scores to write (held_out_scores): kept in the workspace.")
    args = parser.parse_args(argv)
    config = WorkspaceConfig.read(args.config)
    run = _Run(config, workspace_bindings(), detect_compute_budget(), {})
    cohort = _read_cohort(run)
    measurement = run.bindings.load_measurement_model(_final_measurement_model(run))
    lines = []
    scores = np.full((len(cohort.research_ids), len(cohort.trait_names)), np.nan)
    with DosageStore.open(_final_store(run)) as store:
        table = store.variant_table
        kept = fit_filtered(
            table, store.n_samples, measurement.log_reliability, config.fit_minimum_minor_allele_frequency, config.fit_minimum_r2, config.fit_variant_sites,
            config.fit_region,
        )
        records = np.flatnonzero(np.isfinite(arm_log_reliability(table, kept, "snv_indel")))
        for trait in range(len(cohort.trait_names)):
            line, rows, score = ldpred2_trait(store, records, cohort, trait)
            scores[rows, trait] = score
            lines.append(json.dumps(line))
            print(lines[-1], flush=True)
    held_out_scores(args.scores, cohort, scores, np.full(len(cohort.trait_names), records.shape[0]))
    pathlib.Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

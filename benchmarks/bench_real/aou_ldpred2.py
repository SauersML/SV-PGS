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
Only aggregates leave the VM: per trait, the held-out row count (never 1 to 20), a disease's cases (``_cases_text``), the
partial r^2 and the variant count. Each held-out person's score, by its own model, goes to ``--scores`` by cohort row
(``held_out_scores``), so the arms and LDpred2 are compared on the same people (aou_compare.py), and to the run's baseline
directory (``WorkspaceConfig.baselines``, or ``--baselines``) with the digest of each fitted model's training rows
(``workspace_pipeline.write_baseline_scores``), where the report compares it with both arms and stacks it with them on
the same people, resamples and covariates (``partial_r2_tables``). Both are participant-level data, which stay in the
workspace beside the run's mirror, and which no step stages for export.

A run that fits regions (``WorkspaceConfig.fit_regions``) has LDpred2 fitted on each region's snv_indel records in
turn, on the run's one cohort and store, and each line names its region ("region" first); a run without them writes
the lines it always did.

Each fold model's fit is kept as it ends, in ``--units`` (``FoldUnits``): a whole-genome run has one such fit per trait
and evaluated fold, each a band and an R session estimated at an hour or more (not yet measured), after the driver and
in the same Spot task, so a preempted task that starts again fits only the folds it had not finished. A kept fold holds its training
means, posterior mean effects and held-out scores; its partial r^2 is read again from them, as the report would.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pathlib
import zipfile
from collections.abc import Mapping, Sequence

import numpy as np

from benchmarks.bench_sim.submissions import ldpred2_auto, ldpred2_auto_allvar, sbayesrc
from sv_pgs.all_of_us import MINIMUM_REPORTED_PARTICIPANTS
from sv_pgs.config import TraitType
from sv_pgs.compute_budget import detect_compute_budget
from sv_pgs.dosage_store import DosageStore
from sv_pgs.workspace_pipeline import (
    COMPLETE_FILE, STEP_NAMES, WorkspaceConfig, _cases_text, _final_measurement_model, _final_store, _fold_weighted, _partial_r2, _read_cohort,
    _Run, _shown, arm_log_reliability, fit_filtered, workspace_bindings, write_baseline_scores,
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


class FoldUnits:
    """Each fold model's LDpred2 fit, kept as it ends in ``directory`` (the ops mount, beside the run's mirror: the
    held-out scores are participant-level, as ``--scores`` is), one NPZ per fit named by the digest of what it was fitted
    from: the run's finished steps' keys up to the store LDpred2 reads, this code and the R script, and the fit's
    trait, model, records, training and held-out columns. Each is written aside and renamed into place, so a reader sees
    a whole one or none; one that cannot be read is fitted again."""

    FILES = ("means", "effects", "score")

    def __init__(self, directory: pathlib.Path, identity: Mapping[str, object]) -> None:
        self.directory, self.identity = pathlib.Path(directory), dict(identity)

    def path(self, **fit: object) -> pathlib.Path:
        content = json.dumps({**self.identity, **fit}, sort_keys=True)
        return self.directory / f"{hashlib.sha256(content.encode()).hexdigest()}.npz"

    def read(self, path: pathlib.Path) -> dict[str, np.ndarray] | None:
        try:
            with np.load(path, allow_pickle=False) as arrays:
                return {name: np.array(arrays[name]) for name in self.FILES}
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
            return None

    def write(self, path: pathlib.Path, arrays: Mapping[str, np.ndarray]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        aside = path.with_name(f".{path.name}.{os.getpid()}")
        with open(aside, "wb") as handle:
            np.savez(handle, **{name: arrays[name] for name in self.FILES})
        os.replace(aside, path)


def _digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values)).hexdigest()


def run_identity(run: _Run) -> dict[str, object]:
    """What every fold's fit of this run reads beyond its own rows: the keys of the run's finished steps through the one
    whose store it reads (direct_sv's), and the code: this module, the bench-sim LDpred2 arm, its R script and the two
    arms it borrows from (the band window and covariate basis, the row and GEMM helpers)."""
    steps = STEP_NAMES[: STEP_NAMES.index("direct_sv") + 1]
    keys = {step: json.loads((run.output(step) / COMPLETE_FILE).read_text(encoding="utf-8"))["key"] for step in steps if (run.output(step) / COMPLETE_FILE).is_file()}
    code = hashlib.sha256()
    for source in (pathlib.Path(__file__), *(pathlib.Path(module.__file__) for module in (ldpred2_auto_allvar, ldpred2_auto, sbayesrc)), ldpred2_auto_allvar._SCRIPT):
        code.update(source.read_bytes())
    return {"format": "aou-ldpred2-fold-v1", "steps": keys, "code": code.hexdigest()}


def _fold_fit(store: DosageStore, records: np.ndarray, cohort, trait: int, model: int, held: np.ndarray, units: FoldUnits | None) -> tuple[StoreView, np.ndarray]:
    """The held-out view of ``model`` and its scores: from its kept fit, or fitted now and kept."""
    test = cohort_view(store, records, cohort, held, trait)
    train = cohort_view(store, records, cohort, cohort.training[:, model], trait)
    path = None if units is None else units.path(
        trait=cohort.trait_names[trait], model=int(model), records=_digest(records), training=_digest(train.columns), held=_digest(test.columns),
    )
    kept = None if path is None or units is None else units.read(path)
    if kept is not None and kept["score"].shape == (test.n_samples,):
        return test, kept["score"]
    means, effects = ldpred2_auto_allvar.posterior_effects(train, np.arange(records.shape[0]))
    score = held_out_score(test, means, effects)
    if path is not None and units is not None:
        units.write(path, {"means": means, "effects": effects, "score": score})
    return test, score


def ldpred2_trait(store: DosageStore, records: np.ndarray, cohort, trait: int, units: FoldUnits | None = None) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    """LDpred2-auto on each evaluated model's own training and held-out rows, on the report's partial-r^2 scale: the
    trait's aggregate line, and its held-out cohort rows with their scores, each person's by the model that left them out.

    The all-rows model has no held-out rows and never enters a comparison. Across folds the report averages the
    per-fold correlations with row-count weights, so the baseline does too; pooling differently scaled fold scores
    would answer another question. Empty and suppressed cells do not launch a fit, and a disease's case count is shown
    only where neither it nor the controls it implies is 1 to 20 (``_cases_text``).
    With ``units`` each fold's fit is kept as it ends, and a kept one is not fitted again (``FoldUnits``).
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
            held = np.zeros(cohort.targets.shape[0], dtype=bool)
            held[rows] = True
            test, score = _fold_fit(store, records, cohort, trait, model, held, units)
            values.append((test.n_samples, _partial_r2(test.phenotype, score, test.covariates)))
            held_rows.append(cohort_rows(cohort, held))
            held_scores.append(score)
    value = _fold_weighted(values)
    line: dict[str, object] = {
        "trait": cohort.trait_names[trait],
        "held_out_rows": count if count == 0 or count >= SMALLEST_SHOWN_COUNT else f"<{SMALLEST_SHOWN_COUNT}",
        "variants": int(records.shape[0]),
        "pgs_partial_r2": value if np.isfinite(value) else None,
    }
    if binary:
        line["cases"] = _cases_text(count, cases)
    return line, np.concatenate(held_rows), np.concatenate(held_scores)


def held_out_scores(path: str | pathlib.Path, cohort, regions: Sequence[str], scores: np.ndarray, variants: np.ndarray) -> None:
    """Each held-out person's LDpred2 score, [regions, cohort rows, traits] in the run's region order (one region, "",
    for a run without fit_regions) and the cohort's row and trait order (NaN where the person is not held out of the
    trait or its cell was not fitted), with the region names and the cohort's research IDs and trait names to check
    that order by, and each region's variant count."""
    with open(path, "wb") as handle:
        np.savez(
            handle, regions=np.array(regions), research_ids=np.array(cohort.research_ids), trait_names=np.array(cohort.trait_names),
            held_out_score=scores, variants=variants,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="The run config the driver ran (its run directory holds the finished steps).")
    parser.add_argument("--out", required=True, help="The aggregate table to write (one JSON line per trait).")
    parser.add_argument("--scores", required=True, help="The held-out people's scores to write (held_out_scores): kept in the workspace.")
    parser.add_argument(
        "--baselines", type=pathlib.Path,
        help="The baseline directory the report reads (default the run config's, WorkspaceConfig.baselines): ldpred2_auto.npz there, a region's under regions/<name>.",
    )
    parser.add_argument("--units", help="Where each fold's fit is kept as it ends, and found again after a restart (FoldUnits): kept in the workspace.")
    args = parser.parse_args(argv)
    config = WorkspaceConfig.read(args.config)
    run = _Run(config, workspace_bindings(), detect_compute_budget(), {})
    units = None if args.units is None else FoldUnits(pathlib.Path(args.units), run_identity(run))
    cohort = _read_cohort(run)
    measurement = run.bindings.load_measurement_model(_final_measurement_model(run))
    lines = []
    if args.baselines is not None:
        config = dataclasses.replace(config, baseline_directory=args.baselines)
    regions = config.region_runs() or (("", config),)
    scores = np.full((len(regions), len(cohort.research_ids), len(cohort.trait_names)), np.nan)
    variants = np.zeros(len(regions), dtype=np.int64)
    with DosageStore.open(_final_store(run)) as store:
        table = store.variant_table
        for index, (name, region) in enumerate(regions):
            kept = fit_filtered(
                table, store.n_samples, measurement.log_reliability, config.fit_minimum_minor_allele_frequency, config.fit_minimum_r2,
                config.fit_variant_sites, region.fit_region,
            )
            records = np.flatnonzero(np.isfinite(arm_log_reliability(table, kept, "snv_indel")))
            variants[index] = records.shape[0]
            for trait in range(len(cohort.trait_names)):
                line, rows, score = ldpred2_trait(store, records, cohort, trait, units)
                scores[index, rows, trait] = score
                lines.append(json.dumps({"region": name, **line} if name else line))
                print(lines[-1], flush=True)
    held_out_scores(args.scores, cohort, [name for name, _region in regions], scores, variants)
    for index, (_name, region) in enumerate(regions):
        path = region.baselines() / "ldpred2_auto.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_baseline_scores(path, cohort, {name: scores[index, :, trait] for trait, name in enumerate(cohort.trait_names)})
    pathlib.Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

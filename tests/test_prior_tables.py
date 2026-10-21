"""benchmarks/bench_real/prior_tables.py writes a finished run's prior tables as its effects step does, without the step
runner, and prints no record."""
from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from benchmarks.bench_real import prior_tables
from sv_pgs import artifact, fit_model, stage2_wiring, workspace_pipeline
from sv_pgs.dosage_store import DosageStore
from sv_pgs.draw_laws import ProductMixtureDraws
from sv_pgs.scale_mixture_ep import prior_second_moment
from sv_pgs.workspace_pipeline import ARMS, KEY_FILE, WorkspaceConfig, run_pipeline
from tests import test_workspace_pipeline as pipeline_tests

workspace = pipeline_tests.workspace


def _with_prior(recorder: pipeline_tests._Recorder) -> None:
    """The synthetic fitter, each of whose models carries a mean-field draw law, so the prior is read from its fitted
    hyperparameters: per class a lattice of masses, per record a scale growing with its length."""
    fit = recorder.fit

    def fit_with_prior(request):
        fitted = fit(request)
        table = request.store.variant_table
        classes = np.asarray(table.variant_class, dtype=np.int64)
        lengths = np.asarray(table.annotations["sv_length"], dtype=np.float64)
        generator = np.random.default_rng(len(recorder.fit_calls))
        grid = np.linspace(-12.0, -4.0, 5)
        models = []
        for scoring in fitted.scoring:
            rows = scoring.store_rows
            kinds, index = np.unique(classes[rows], return_inverse=True)
            law = ProductMixtureDraws(
                class_index=index.astype(np.int64), log_scale=0.4 * np.log(np.maximum(lengths[rows], 1.0)) + generator.normal(0.0, 0.2, rows.shape[0]),
                log_density=np.log(generator.dirichlet(np.ones(grid.shape[0]), size=kinds.shape[0])), log_variance_grid=grid,
                shift=np.zeros((1, rows.shape[0])), omega=np.zeros((1, rows.shape[0])), weights=np.ones(1), key=(1, 2),
                draw_count=scoring.posterior_draws.shape[1],
            )
            models.append(dataclasses.replace(scoring, posterior_draws=law))
        fitted.scoring = tuple(models)
        return fitted

    recorder.fit = fit_with_prior


# The files the script reads (its docstring's list), relative to the run directory and to the direct_sv step's store.
RUN_INPUTS = ("fit/models/*/*/model.json", "fit/models/*/*/arrays.npz", "direct_sv/_COMPLETE.json", "cohort/cohort.json", "score/_COMPLETE.json")
STORE_INPUTS = ("MANIFEST.json", "variants/**/*", "stats/half*/*/sum_code/**/*", "stats/half*/*/sum_code2/**/*")


def _script_tables(run: Path, out: Path) -> dict[str, str]:
    """The script's tables in ``out``, from a copy of ``run`` that holds only the files it reads: no genotype array (the
    store's dosage/), no sample manifest, no step key and nothing the effects step wrote."""
    copied = out.parent / "copied_run"
    store = prior_tables.store_path(run).relative_to(run)
    for pattern in (*RUN_INPUTS, *(f"{store.as_posix()}/{pattern}" for pattern in STORE_INPUTS)):
        for path in run.glob(pattern):
            if path.is_file():
                (copied / path.relative_to(run)).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, copied / path.relative_to(run))
    assert (run / store / "dosage").is_dir() and sorted(path.name for path in (copied / store).iterdir()) == ["MANIFEST.json", "stats", "variants"]
    assert not list(copied.rglob(KEY_FILE)) and not (copied / "effects").exists()
    assert prior_tables.main([str(copied), "--out", str(out)]) == 0
    assert sorted(path.name for path in out.iterdir()) == sorted(prior_tables.TABLES)
    return {name: (out / name).read_text() for name in prior_tables.TABLES}


@pytest.mark.parametrize("flagged,threshold", [("quarantined", workspace_pipeline.MINIMUM_REPORTED_PARTICIPANTS), (None, 1)])
def test_a_finished_run_s_prior_tables_are_its_effects_step_s_without_the_step_runner(
    workspace, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, flagged: str | None, threshold: int,
) -> None:
    """The synthetic store holds 1 to 20 records of most classes, so at the export rule every value is hidden; with the
    rule's threshold at 1 for both the effects step and the script, every value is shown and compared too."""
    monkeypatch.setattr(workspace_pipeline, "MINIMUM_REPORTED_PARTICIPANTS", threshold)
    synthetic, client = workspace
    config = WorkspaceConfig.from_mapping(synthetic.raw)
    recorder = pipeline_tests._Recorder()
    _with_prior(recorder)
    summaries = run_pipeline(config, pipeline_tests._artifact_bindings(recorder, client, flagged=flagged), pipeline_tests.BUDGET, through="effects")
    quarantined = set(summaries["score"]["quarantined"])
    reported = [name for name in (*config.diseases, *config.traits) if name not in quarantined]
    run = config.run_directory
    expected = {name: (run / "effects" / name).read_text() for name in prior_tables.TABLES}
    lines = {name: [dict(zip(columns, line.split("\t"))) for line in expected[name].splitlines()[1:]] for name, columns in prior_tables.TABLES.items()}
    assert bool(quarantined) == (flagged == "quarantined") and summaries["effects"]["prior"].startswith(f"the learned prior read from {len(ARMS) * len(reported)} ")
    assert [*dict.fromkeys(line["trait"] for line in lines["prior_enrichment.tsv"])] == reported
    shown = [line for line in lines["prior_enrichment.tsv"] if line["variant_class"] == "all_structural" and line["prior_enrichment"]]
    assert bool(shown) == (threshold == 1) and all(line["arm"] == "with_sv" for line in shown)

    assert _script_tables(run, tmp_path / "prior") == expected
    assert prior_tables.traits(run) == reported
    printed = capsys.readouterr().out
    with DosageStore.open(prior_tables.store_path(run)) as store:
        identifiers = store.variant_table.variant_ids(np.arange(store.n_variants))
    assert printed.count("prior read") == len(ARMS) * len(reported) and not any(str(identifier) in printed for identifier in identifiers)


def test_a_full_data_fit_saves_its_learned_prior_and_the_script_reads_it(
    workspace, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real fitter (``fit_model.fit``: ``fit_full_data``'s empirical Bayes at its default, mean-field inference),
    saved and read back by ``artifact``: every saved law is the mean-field one, its prior at each record is the fit's
    own prior second moment at its fitted hyperparameters, the effects step reads it from every model, and the script's
    tables are the effects step's. A single split, as the All of Us runs are: each trait's one model is the one read. The
    export rule's threshold is 1, so every value shows."""
    monkeypatch.setattr(workspace_pipeline, "MINIMUM_REPORTED_PARTICIPANTS", 1)
    # Each model's prior second moment per record, from the fitted prior and hyperparameters, keyed by its posterior means.
    learned: list[tuple[np.ndarray, np.ndarray]] = []
    scoring_models = stage2_wiring.scoring_models

    def recorded(fit, prior, statistics, trait_types, draw_count, seed):
        models = scoring_models(fit, prior, statistics, trait_types, draw_count, seed)
        learned.extend((np.asarray(model.coefficients), prior_second_moment(prior, fit.hyperparameters[index])) for index, model in enumerate(models))
        return models

    monkeypatch.setattr(stage2_wiring, "scoring_models", recorded)
    synthetic, client = workspace
    config = WorkspaceConfig.from_mapping({**synthetic.raw, "cross_fit": False, "panel": None})
    bindings = dataclasses.replace(
        pipeline_tests._bindings(pipeline_tests._Recorder(), client), fit=fit_model.fit, save_model=artifact.save_model,
        load_model=artifact.load_model, predict=artifact.predict_models,
    )
    summaries = run_pipeline(config, bindings, pipeline_tests.BUDGET, through="effects")
    run = config.run_directory
    reported = prior_tables.traits(run)
    assert reported == [name for name in (*config.diseases, *config.traits) if name not in summaries["score"]["quarantined"]] and reported
    folders = [run / "fit" / "models" / arm / trait for arm in ARMS for trait in reported]
    assert all(json.loads((folder / "model.json").read_text())["draw_laws"] == ["product_mixture"] for folder in folders)
    assert summaries["effects"]["prior"].startswith(f"the learned prior read from {len(folders)} of {len(folders)} ")

    with DosageStore.open(prior_tables.store_path(run)) as store:
        codes_per_unit = np.asarray(store.variant_table.codes_per_unit, dtype=np.float64)
    for folder in folders:
        fitted = artifact.load_model(folder)
        for scoring in fitted.scoring:
            (variance,) = [variance for means, variance in learned if np.array_equal(means, scoring.coefficients)]
            unit = codes_per_unit[scoring.store_rows] / np.asarray(scoring.signed_scales, dtype=np.float64)
            np.testing.assert_allclose(workspace_pipeline.prior_moments(scoring, unit).variance, variance, rtol=1e-12)

    columns = prior_tables.TABLES["prior_enrichment.tsv"]
    lines = [dict(zip(columns, line.split("\t"))) for line in (run / "effects" / "prior_enrichment.tsv").read_text().splitlines()[1:]]
    structural = [line for line in lines if line["arm"] == "with_sv" and line["variant_class"] == "all_structural"]
    assert len(structural) == len(reported) and all(float(line["prior_enrichment"]) > 0.0 for line in structural)
    assert all(float(line["posterior_enrichment_ci_low"]) <= float(line["posterior_enrichment"]) <= float(line["posterior_enrichment_ci_high"]) for line in structural)

    expected = {name: (run / "effects" / name).read_text() for name in prior_tables.TABLES}
    assert _script_tables(run, tmp_path / "prior") == expected
    assert capsys.readouterr().out.count("prior read") == len(folders)

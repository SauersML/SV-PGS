"""Poster summaries must retain failures and report uncertainty at the actual replicate count."""
from __future__ import annotations

import importlib
import json
import math
import re
import sys

import numpy as np
import pytest
from scipy import stats

from docs.poster.ashg2026 import figures, simulation


def test_mean_interval_uses_student_t_and_counts_only_finite_replicates():
    values = [0.01, 0.02, 0.03]
    entry = simulation.interval(iter([*values, None, math.nan]))
    half = stats.t.ppf(0.975, 2) * np.std(values, ddof=1) / np.sqrt(3)
    assert entry["replicates"] == 3
    assert entry["lower"] == pytest.approx(0.02 - half)
    assert entry["upper"] == pytest.approx(0.02 + half)
    assert math.isnan(simulation.interval([0.01])["lower"])
    assert simulation.interval([])["replicates"] == 0


def test_summary_can_report_failures_before_any_paired_result_exists(tmp_path):
    folder = tmp_path / "grid/tasks/sv_20pct/n40000/replicate0/with_sv"
    folder.mkdir(parents=True)
    (folder / "attempt.json").write_text(json.dumps(dict(point="sv_20pct", train_count=40000, replicate=0,
                                                        arm="with_sv", status="failed")))
    summary = json.loads(simulation.collect(tmp_path, tmp_path / "summary.json").read_text())
    assert summary["tasks"] == 0 and summary["points"] == {} and summary["genotype_accuracy"] == {}
    assert summary["coverage"]["sv_20pct"]["40000"]["with_sv"] == {"completed": 0, "failed": 1, "running": 0}


def test_summary_keeps_each_comparisons_actual_denominator(tmp_path):
    for replicate in range(3):
        folder = tmp_path / f"grid/tasks/sv_20pct/n40000/replicate{replicate}"
        folder.mkdir(parents=True)
        result = dict(point="sv_20pct", train_count=40000, replicate=replicate,
                      snv_indel_auc=.6, with_sv_auc=.61 + replicate * .01, ldpred2_auc=.59,
                      delta_auc=.01 + replicate * .01, z=2.0 if replicate < 2 else math.nan,
                      truth=dict(structural_share=.2, causal_by_class=[160, 0, 0, 40], causal_count=200))
        if replicate == 0:
            result.update(ldpred2_allvar_auc=.62, delta_auc_vs_ldpred2_allvar=-.01, z_vs_ldpred2_allvar=-2.0)
        if replicate == 1:
            result.update(uncertified_arms=["snv_indel"])
        (folder / "result.json").write_text(json.dumps(result))
    failed = tmp_path / "grid/tasks/sv_20pct/n40000/replicate3/with_sv"
    failed.mkdir(parents=True)
    (failed / "attempt.json").write_text(json.dumps(dict(point="sv_20pct", train_count=40000, replicate=3,
                                                        arm="with_sv", status="failed")))
    summary = json.loads(simulation.collect(tmp_path, tmp_path / "summary.json").read_text())
    cell = summary["points"]["sv_20pct"]["sizes"]["40000"]
    assert cell["replicates"] == 3 and cell["failed_replicates"] == 1 and cell["uncertified_fits"] == 1
    assert cell["delta_auc"]["replicates"] == 3 and cell["power"]["replicates"] == 2
    assert cell["power"]["rate"] == 1.0
    assert cell["auc_intervals"]["ldpred2_allvar_auc"]["replicates"] == 1
    assert math.isnan(cell["delta_auc_vs_ldpred2_allvar"]["lower"])


def test_true_genotype_gain_is_paired_only_over_replicates_both_runs_finished(tmp_path):
    def write(grid, replicate, delta):
        folder = tmp_path / f"{grid}/tasks/sv_20pct/n40000/replicate{replicate}"
        folder.mkdir(parents=True)
        (folder / "result.json").write_text(json.dumps(dict(point="sv_20pct", train_count=40000, replicate=replicate,
                                                            delta_auc=delta)))
    for replicate, delta in enumerate((0.0, 0.01, 0.02)):
        write("grid", replicate, delta)
    for replicate, delta in ((1, 0.05), (2, 0.04), (3, 0.09)):
        write("grid-truth", replicate, delta)
    imputed = [json.loads(path.read_text()) for path in sorted((tmp_path / "grid").rglob("result.json"))]
    gains = simulation.genotype_accuracy(tmp_path, imputed)["delta_auc"]["sv_20pct"]
    assert gains["imputed"]["replicates"] == 3 and gains["true"]["replicates"] == 3
    assert gains["true"]["mean"] == pytest.approx(0.06)
    assert gains["true_minus_imputed"]["replicates"] == 2
    assert gains["true_minus_imputed"]["mean"] == pytest.approx(0.03)


def test_window_study_summarizes_each_measurement_apart_from_the_chromosome_grid(tmp_path):
    def write(grid, replicate, delta):
        folder = tmp_path / f"{grid}/tasks/tagged_sv/n10000/replicate{replicate}"
        folder.mkdir(parents=True)
        (folder / "result.json").write_text(json.dumps(dict(
            point="tagged_sv", train_count=10000, replicate=replicate, snv_indel_auc=.6, with_sv_auc=.6 + delta,
            delta_auc=delta, z=3.0, truth=dict(structural_share=.25, causal_by_class=[160, 0, 0, 40], causal_count=200))))
    for replicate in range(3):
        write("grid", replicate, 0.001)
        write("window", replicate, 0.01)
        write("window-truth", replicate, 0.03)
    (tmp_path / "public").mkdir()
    (tmp_path / "public/window.json").write_text(json.dumps(dict(first_record=0, last_record=60000)))
    summary = json.loads(simulation.collect(tmp_path, tmp_path / "summary.json").read_text())
    window = summary["window"]
    assert summary["points"]["tagged_sv"]["sizes"]["10000"]["delta_auc"]["mean"] == pytest.approx(0.001)
    assert window["imputed"]["tagged_sv"]["sizes"]["10000"]["delta_auc"]["mean"] == pytest.approx(0.01)
    assert window["true"]["tagged_sv"]["sizes"]["10000"]["delta_auc"]["mean"] == pytest.approx(0.03)
    assert "expected_auc" not in window["true"]["tagged_sv"] and window["cohort"]["last_record"] == 60000
    assert figures.window_cells(summary, "true", "tagged_sv")[10000]["replicates"] == 3


def test_zero_gains_have_a_nonzero_plot_range():
    low, high, step = figures.nice_range(0.0, 0.0)
    assert low < 0 < high and step > 0


def test_legacy_accuracy_has_no_invented_interval():
    entry = figures.auc_interval({"with_sv_auc": [0.6, 0.01]}, "with_sv_auc")
    assert entry["mean"] == 0.6 and math.isnan(entry["lower"])


def test_missing_results_remove_an_optional_figure_from_a_previous_run(monkeypatch, tmp_path):
    monkeypatch.setattr(figures, "OUT", tmp_path)
    monkeypatch.setattr(figures, "placeholder_figure", lambda name: tmp_path / f"{name}.png")
    for suffix in (".png", ".html"):
        (tmp_path / f"fig_sim_architectures{suffix}").write_text("stale result")
    figures.simulation_figures(None)
    assert not list(tmp_path.glob("fig_sim_architectures.*"))


def poster_module(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["build_pptx.py", "template.pptx", "poster.pptx", str(tmp_path / "figures")])
    return importlib.import_module("docs.poster.ashg2026.build_pptx")


def poster_copy(poster) -> dict:
    return poster.copy(json.loads((poster.HERE / "figures/gallery/numbers.json").read_text()))


def test_poster_design_lists_each_setting_value_the_grid_simulates(monkeypatch, tmp_path):
    from benchmarks.power import grid

    poster = poster_module(monkeypatch, tmp_path)
    simulated = {
        "h2": {point.get("truth", {}).get("h2", grid.HERITABILITY) for point in grid.POINTS.values()},
        "prevalence": {point.get("truth", {}).get("prevalence", grid.PREVALENCE) for point in grid.POINTS.values()},
        "causal_variants": {point.get("causal_variants", grid.CAUSAL_VARIANTS) for point in grid.POINTS.values()},
        "sv_causal_fraction": {point["sv_causal_fraction"] for point in grid.POINTS.values()},
    }
    assert {name: set(values) for name, values in poster.DESIGN_SETTINGS.items()} == simulated
    line = poster_copy(poster)["design"][1]
    # each value is named and none is given as a range
    assert "–" not in line
    assert "h² 0.01 or 0.1" in line and "prevalence 1% or 10%" in line
    assert "50, 200 or 2,000 causal variants" in line and "0%, 5% or 20% SVs" in line


def test_poster_cites_every_reference_in_order(monkeypatch, tmp_path):
    poster = poster_module(monkeypatch, tmp_path)
    words = poster_copy(poster)
    text = " ".join(line for key in ("introduction", "methods", "design", "aou", "discussion") for line in words[key])
    # h² and r² are exponents, not citations
    text = re.sub(r"[hr]²", "", text)
    marks = [poster.SUPERSCRIPT[index] for index in range(1, len(poster.REFERENCES) + 1)]
    assert all(mark in text for mark in marks)
    assert [text.index(mark) for mark in marks] == sorted(text.index(mark) for mark in marks)


def test_poster_gain_note_counts_the_simulations_the_forest_draws(monkeypatch, tmp_path):
    poster = poster_module(monkeypatch, tmp_path)
    rows = json.loads((poster.HERE / "figures/gallery/numbers.json").read_text())["poster_sv_gain"]["rows"]
    drawn = sum(row["true"][3] for row in rows.values() if row is not rows.get("well_imputed"))
    assert f"over {drawn} simulations" in poster_copy(poster)["gain_note"]

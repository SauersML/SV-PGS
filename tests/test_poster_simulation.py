"""Poster summaries must retain failures and report uncertainty at the actual replicate count."""
from __future__ import annotations

import importlib
import json
import math
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
    # A partial/stale effect output from a failed attempt must not enter the poster.
    np.savez(folder / "sv_effects.npz", record=[1], per_allele=[0.1])
    summary = json.loads(simulation.collect(tmp_path, tmp_path / "summary.json").read_text())
    assert summary["tasks"] == 0 and summary["points"] == {} and summary["effects"] == {}
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
        (folder / "result.json").write_text(json.dumps(result))
    failed = tmp_path / "grid/tasks/sv_20pct/n40000/replicate3/with_sv"
    failed.mkdir(parents=True)
    (failed / "attempt.json").write_text(json.dumps(dict(point="sv_20pct", train_count=40000, replicate=3,
                                                        arm="with_sv", status="failed")))
    summary = json.loads(simulation.collect(tmp_path, tmp_path / "summary.json").read_text())
    cell = summary["points"]["sv_20pct"]["sizes"]["40000"]
    assert cell["replicates"] == 3 and cell["failed_replicates"] == 1
    assert cell["delta_auc"]["replicates"] == 3 and cell["power"]["replicates"] == 2
    assert cell["power"]["rate"] == 1.0
    assert cell["auc_intervals"]["ldpred2_allvar_auc"]["replicates"] == 1
    assert math.isnan(cell["delta_auc_vs_ldpred2_allvar"]["lower"])


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


def test_poster_does_not_call_an_inconclusive_comparison_equivalent(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["build_pptx.py", "template.pptx", "poster.pptx", str(tmp_path / "missing.json")])
    poster = importlib.import_module("docs.poster.ashg2026.build_pptx")
    for name in ("SIMULATION_METHODS", "SIMULATION_ACCURACY", "SIMULATION_EFFECTS", "DISCUSSION"):
        monkeypatch.setattr(poster, name, getattr(poster, name))
    entry = simulation.interval([-0.01, 0.0, 0.01])
    cell = dict(replicates=3, delta_auc=entry, delta_auc_vs_ldpred2=entry,
                delta_auc_vs_ldpred2_allvar=simulation.interval([0.1, 0.2]),
                power=dict(replicates=3, rate=0.0, lower=0.0, upper=0.56), length_rho=[0.0, 0.01])
    poster.simulation_copy(dict(train_counts=[5000, 40000], points={"sv_20pct": {"sizes": {"40000": cell}}}))
    copy = " ".join(text for paragraph in poster.SIMULATION_ACCURACY for text, _flag in paragraph)
    assert "matched" not in copy and "outperformed" not in copy
    assert "95% CI" in copy and "SNV subset" in copy
    # Three core completions do not justify a sentence about an optional comparator with only two replicates.
    assert "all variant classes" not in copy

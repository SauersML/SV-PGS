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


def test_poster_design_lists_the_causal_sv_shares_the_grid_simulates(monkeypatch, tmp_path):
    from benchmarks.power import grid

    poster = poster_module(monkeypatch, tmp_path)
    assert set(poster.DESIGN_SHARES) == {point["sv_causal_fraction"] for point in grid.POINTS.values()}
    assert "causal SV share (0%, 5% or 20%)" in poster_copy(poster)["grid"][0]


def test_poster_cites_every_reference_in_order(monkeypatch, tmp_path):
    from docs.poster.ashg2026 import sim_gallery

    poster = poster_module(monkeypatch, tmp_path)
    words = poster_copy(poster)
    # in reading order: the left column, the simulation's column, then the Discussion; figures cite nothing
    lines = [*words["introduction"], *words["methods"], *words["design"], *words["aou"], *words["grid"],
             *words["discussion"]]
    assert not any(mark in name for _, name, *_ in sim_gallery.POSTER_METHODS for mark in "¹²³⁴⁵⁶⁷⁸⁹")
    # h² and r² are exponents, not citations
    text = re.sub(r"[hr]²", "", " ".join(lines))
    marks = [poster.SUPERSCRIPT[index] for index in range(1, len(poster.REFERENCES) + 1)]
    assert all(mark in text for mark in marks)
    assert [text.index(mark) for mark in marks] == sorted(text.index(mark) for mark in marks)


def test_every_poster_bullet_ends_with_a_full_stop(monkeypatch, tmp_path):
    words = poster_copy(poster_module(monkeypatch, tmp_path))
    for key in ("introduction", "methods", "design", "aou", "grid", "discussion"):
        for line in words[key]:
            # a citation's number follows the full stop
            assert line.rstrip("¹²³⁴⁵⁶⁷⁸⁹").endswith("."), line


COMMON = {"relative": 0.03, "relative_lower": -0.11, "relative_upper": 0.17, "traits": 5}
"""A matched common gain as aou_numbers.json holds it, in percent of the two arms' mean r²."""


def matched_common(poster, monkeypatch, tmp_path, common: dict) -> None:
    """aou_numbers.json in ``tmp_path`` with ``common`` as its matched traits' common gain, read as build_pptx's own."""
    (tmp_path / "aou_numbers.json").write_text(json.dumps({"matched": {"common": common}}))
    monkeypatch.setattr(poster, "HERE", tmp_path)


def test_discussion_states_the_imputed_share_of_the_true_sv_gain(monkeypatch, tmp_path):
    poster = poster_module(monkeypatch, tmp_path)
    numbers = {"fig5": {"people": 50_000}, "poster_sv_gain": {"rows": {"all": {"share": [0.4728, 0.1929, 0.7377]}}}}
    assert poster.copy(numbers)["discussion"][1] == (
        "Imputed SVs gave 47% of the AUC gain of true SV genotypes (95% CI 19–74%).")


def test_aou_result_states_the_common_change_over_its_traits(monkeypatch, tmp_path):
    poster = poster_module(monkeypatch, tmp_path)
    matched_common(poster, monkeypatch, tmp_path, COMMON)
    [sentence] = poster.aou_result()
    assert "changed r² by +0.03% (95% CI −0.11 to 0.17%) over 5 traits," in sentence


def test_references_sit_under_the_qr_code_within_the_right_column(monkeypatch, tmp_path):
    poster = poster_module(monkeypatch, tmp_path)
    heading, qr, words = poster.references()
    assert (heading.kind, heading.content) == ("bar", (96, "References"))
    assert (qr.kind, words.kind) == ("qr", "text")
    assert ["".join(text for text, _ in para.runs) for para in words.content] == [
        f"{poster.SUPERSCRIPT[index]}{reference}" for index, reference in enumerate(poster.REFERENCES, start=1)]
    longest = max(poster.width_of(para.runs, poster.REFERENCE) for para in words.content)
    assert longest + 2 * poster.INSET <= poster.RIGHT


def test_poster_copy_never_mentions_convergence_certification_or_a_dagger(monkeypatch, tmp_path):
    poster = poster_module(monkeypatch, tmp_path)
    words = poster_copy(poster)
    matched_common(poster, monkeypatch, tmp_path, COMMON)
    lines = [*(line for section in words.values() for line in section), *poster.aou_result()]
    for line in lines:
        assert not re.search("converg|certif|†", line, flags=re.IGNORECASE), line

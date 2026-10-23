"""Every figure on the poster keeps its type on its canvas, and the poster places each figure whole: a clipped axis
title, tick label, legend or note fails here, as does a picture set at another aspect ratio or cropped."""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest
from matplotlib.transforms import Bbox

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

POSTER = Path(__file__).resolve().parents[1] / "docs" / "poster" / "ashg2026"
sys.path.insert(0, str(POSTER))  # aou_gallery imports sim_gallery by name, as it runs beside it

import aou_gallery  # noqa: E402
import build_pptx  # noqa: E402
import sim_gallery  # noqa: E402
from sim_gallery import plt  # noqa: E402


def simulated_grids() -> dict[str, list[dict]]:
    """Three simulations of every architecture at every training size in each grid, with AUCs and gains in the
    ranges the real grids hold."""
    random = np.random.default_rng(1)
    grids = {}
    for grid in ("grid", "grid-truth", "window", "window-truth"):
        rows = []
        for point in sim_gallery.LABELS:
            for size in sim_gallery.SIZES:
                for replicate in range(3):
                    theirs = 0.52 + 0.02 * math.log2(size / sim_gallery.SIZES[0]) + random.normal(0, 0.02)
                    gain = 0.02 + random.normal(0, 0.01)
                    rows.append({"point": point, "train_count": size, "replicate": replicate, "ldpred2_auc": theirs,
                                 "with_sv_auc": theirs + gain, "snv_indel_auc": theirs + gain,
                                 "delta_auc": random.normal(0.001, 0.002), "delta_auc_vs_ldpred2": gain,
                                 "z_vs_ldpred2": gain / 0.005})
        grids[grid] = rows
    return grids


def simulated_summary() -> dict:
    """The summary's best possible AUC and per-frequency genotype r² bins."""
    edges = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    bins = [{"low": low, "high": high, "mean": 0.3 + 0.1 * index, "count": 100} for index, (low, high)
            in enumerate(zip(edges, edges[1:]))]
    return {"points": {"null": {"true_value_auc": 0.673}},
            "genotype_accuracy": {"r2_by_frequency": {"SNV": {"bins": bins}, "SV": {"bins": bins}}}}


def poster_figures():
    grids, summary = simulated_grids(), simulated_summary()
    with plt.rc_context(sim_gallery.POSTER_STYLE):
        yield "poster_accuracy_by_training_size", sim_gallery.poster_accuracy_figure(grids, summary)
        yield "poster_wins", sim_gallery.poster_wins_figure(grids)
        yield "poster_sv_gain", sim_gallery.poster_sv_gain_figure(grids)[0]
        yield "poster_imputation_accuracy", sim_gallery.poster_imputation_figure(summary)
        yield "poster_model", sim_gallery.poster_model_figure()
    rows = [{"trait": trait, "relative": mean, "relative_lower": low, "relative_upper": high}
            for trait, mean, low, high in (("ldl_cholesterol", -2.0, -7.0, 3.1), ("egfr", 0.3, -1.1, 1.7),
                                           ("hypothyroidism", 1.4, -0.4, 3.2), ("gout", 4.5, -24.6, 33.6))]
    with plt.rc_context({**sim_gallery.POSTER_STYLE, "axes.labelsize": sim_gallery.POSTER_LABEL}):
        yield "aou_1_gain", aou_gallery.gain_figure(rows)[0]
        # the final forest, with the converged traits' common g
        common = {"relative": 0.2, "relative_lower": -0.6, "relative_upper": 1.1, "traits": 4}
        yield "aou_1_gain_final", aou_gallery.gain_figure(rows, common)[0]
        swapped = {**common, "p_sharp": 0.3197}
        # and over it each trait's held-out r² for LDpred2-auto and SV-PGS, with the pooled rows' notes
        yield "aou_1_gain_compared", aou_gallery.gain_figure(rows, swapped, compared_rows(), POOLED)[0]
        # with asterisks past the intervals whose q is below 0.05, gout's at the end of its arrowhead
        yield "aou_1_gain_starred", aou_gallery.gain_figure(rows, swapped, compared_rows(0.01), POOLED)[0]


POOLED = {"trait": "All converged traits", "with_sv": {"relative": 17.45, "relative_lower": 4.01, "relative_upper": 32.79,
                                                       "p_one": 0.0047}}
"""aou_vs_ldpred2.json's "combined": SV-PGS with SVs pooled against LDpred2-auto, under the json's own name for it."""


def compared_rows(q: float = 0.3) -> list[dict]:
    """aou_vs_ldpred2.json's rows: each trait's held-out r² as a fraction with its 95% interval, per arm, in r² ranges
    like the real traits', and the Benjamini-Hochberg q of SV-PGS against LDpred2-auto and of the SV gain, ``q`` for
    LDL cholesterol's comparison and gout's gain and 0.3 for the rest."""
    rows = []
    for trait, r2 in (("egfr", 0.012), ("ldl_cholesterol", 0.05), ("gout", 0.003), ("hypothyroidism", 0.0008)):
        rows.append({"trait": trait, "r2": {arm: {"r2": r2 * scale, "lower": r2 * scale / 3, "upper": r2 * scale * 1.6}
                                            for arm, scale in (("ldpred2_auto", 0.9), ("with_sv", 1.0),
                                                               ("snv_indel", 1.0))},
                     "with_sv": {"q": q if trait == "ldl_cholesterol" else 0.3},
                     "sv_gain": {"q": q if trait == "gout" else 0.3}})
    return rows


def drawn_texts(fig):
    """Every piece of type a figure shows: each axes' notes, title and legend, the titles and tick labels of each axis
    it draws, and the figure's own text."""
    for ax in fig.axes:
        legend = ax.get_legend()
        axes = [axis for axis in (ax.xaxis, ax.yaxis) if ax.axison and axis.get_visible()]
        parts = [part for axis in axes for part in (axis.label, *axis.get_ticklabels())]
        keys = [*legend.texts, legend.get_title()] if legend else []
        yield from (text for text in (*ax.texts, ax.title, *parts, *keys) if text.get_visible() and text.get_text().strip())
    yield from fig.texts


def test_every_poster_figure_keeps_its_type_on_its_canvas():
    for name, fig in poster_figures():
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        across, up = fig.get_size_inches()
        texts = list(drawn_texts(fig))
        assert texts, name
        for text in texts:
            box = text.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())
            assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= across and box.y1 <= up, (name, text.get_text(), box)
        # and everything else drawn, marks left unclipped among it
        assert max(sim_gallery.overhang(fig)) <= 0, name
        plt.close(fig)


def test_the_poster_places_each_figure_whole(tmp_path):
    from PIL import Image

    picture = tmp_path / "figure.png"
    Image.new("RGBA", (300, 200)).save(picture)
    poster = object.__new__(build_pptx.Poster)
    poster.xml, poster.rels, poster.media = "", "<Relationships></Relationships>", tmp_path
    poster.shapes, poster.next_id, poster.images = [], 200, {}
    # room taller than the picture at the column's width, and room shorter than it
    for width, height in ((6.0, 10.0), (6.0, 2.0)):
        poster.place(build_pptx.figure(picture), 0.0, 0.0, width, height)
        shape = poster.shapes[-1]
        across, up = map(int, re.search(r'<a:ext cx="(\d+)" cy="(\d+)"/>', shape).groups())
        assert across / up == pytest.approx(300 / 200, rel=1e-4)
        assert across <= build_pptx.emu(width) and up <= build_pptx.emu(height)
        assert "srcRect" not in shape and "<a:fillRect/>" in shape


def row_names(ax) -> list[str]:
    """An All of Us panel's row names, top to bottom, a two-line name as one line."""
    return [text.get_text().replace("\n", " ") for text in ax.texts if text.get_gid() == "name"]


def notes(ax) -> str:
    """An All of Us panel's notes, their lines joined by spaces."""
    return " ".join(text.get_text() for text in ax.texts if text.get_gid() == "note")


def test_the_all_of_us_forest_skips_an_unconverged_fit_and_draws_the_common_gain():
    rows = [{"trait": "gout", "relative": 4.5, "relative_lower": -24.6, "relative_upper": 33.6},
            {"trait": "egfr", "relative": 0.3, "relative_lower": -1.1, "relative_upper": 1.7}]
    common = {"relative": 0.2, "relative_lower": -0.6, "relative_upper": 1.1, "traits": 2}
    with plt.rc_context({**sim_gallery.POSTER_STYLE, "axes.labelsize": sim_gallery.POSTER_LABEL}):
        fig, lines = aou_gallery.gain_figure(rows, common)
    assert row_names(fig.axes[-1]) == ["Gout", "eGFR", aou_gallery.COMBINED]
    assert "common change" in lines[0]
    plt.close(fig)
    with plt.rc_context({**sim_gallery.POSTER_STYLE, "axes.labelsize": sim_gallery.POSTER_LABEL}):
        fig, _ = aou_gallery.gain_figure(rows[1:])
    assert row_names(fig.axes[-1]) == ["eGFR"]
    plt.close(fig)
    # a trait whose fit did not converge is in neither panel
    with plt.rc_context({**sim_gallery.POSTER_STYLE, "axes.labelsize": sim_gallery.POSTER_LABEL}):
        fig, _ = aou_gallery.gain_figure([{**rows[0], "unconverged": True}, rows[1]], common, compared_rows())
    assert [row_names(ax) for ax in fig.axes] == [["eGFR", "LDL cholesterol", "Hypothyroidism"],
                                                  ["eGFR", aou_gallery.COMBINED]]
    plt.close(fig)


def test_both_all_of_us_panels_keep_the_compared_order_and_give_r2_in_percent():
    rows = [{"trait": trait, "relative": 0.5, "relative_lower": -1.0, "relative_upper": 2.0}
            for trait in ("hypothyroidism", "stroke", "ldl_cholesterol", "gout", "egfr")]
    common = {"relative": 0.2, "relative_lower": -0.6, "relative_upper": 1.1, "traits": 5}
    with plt.rc_context({**sim_gallery.POSTER_STYLE, "axes.labelsize": sim_gallery.POSTER_LABEL}):
        fig, lines = aou_gallery.gain_figure(rows, common, compared_rows())
    fig.canvas.draw()
    top, gain = fig.axes
    # the compared rows' order, then the trait they lack, then the common row
    assert row_names(top) == ["eGFR", "LDL cholesterol", "Gout", "Hypothyroidism"]
    assert row_names(gain) == ["eGFR", "LDL cholesterol", "Gout", "Hypothyroidism", "Stroke", aou_gallery.COMBINED]
    # r² is drawn in percent: the fractions times 100, ticked at the decades inside the axis without a % sign
    assert top.get_xlabel() == "Held-out r² (%)" and "percent" in lines[0]
    low, high = top.get_xlim()
    assert low <= 0.08 * 0.9 and high >= 5.0
    labels = [label.get_text() for label in top.get_xticklabels()]
    assert labels and set(labels) <= {"0.01", "0.1", "1", "10"}
    assert all(low <= tick <= high for tick in top.get_xticks())
    # the forest takes the panels as Panel lists, top to bottom
    panel = aou_gallery.Panel([("eGFR", [(0.5, 0.1, 0.9)])], [("", sim_gallery.SKY, "o", False)], "Change (%)")
    with plt.rc_context({**sim_gallery.POSTER_STYLE, "axes.labelsize": sim_gallery.POSTER_LABEL}):
        alone = aou_gallery.forest([panel])
    assert len(alone.axes) == 1 and max(sim_gallery.overhang(alone)) <= 0
    plt.close(alone)
    plt.close(fig)


def test_a_q_below_005_stars_its_interval_and_the_key_and_none_draws_neither():
    rows = [{"trait": trait, "relative": 0.5, "relative_lower": -1.0, "relative_upper": 2.0}
            for trait in ("ldl_cholesterol", "gout", "egfr")]
    for q, starred in ((0.01, True), (0.05, False), (0.3, False)):
        with plt.rc_context({**sim_gallery.POSTER_STYLE, "axes.labelsize": sim_gallery.POSTER_LABEL}):
            fig, lines = aou_gallery.gain_figure(rows, None, compared_rows(q))
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        top, gain = fig.axes
        notes = [text.get_text() for text in fig.texts]
        assert (aou_gallery.STAR_NOTE in notes) is starred and ("asterisk" in lines[0]) is starred
        for ax, name, index in ((top, "LDL cholesterol", 1), (gain, "Gout", 0)):
            stars = [text for text in ax.texts if text.get_text() == aou_gallery.STAR]
            assert len(stars) == (1 if starred else 0)
            if starred:
                # just past the starred trait's SV-PGS interval, on its line
                box = stars[0].get_window_extent(renderer)
                # each row's intervals, one an hlines collection, in the order of its series
                names = row_names(ax)
                segments = [segment for collection in ax.collections for segment in collection.get_segments()]
                (_, y), (x1, _) = segments[names.index(name) * (len(segments) // len(names)) + index]
                end = ax.transData.transform((x1, y))
                assert end[0] < box.x0 < end[0] + 0.5 * fig.dpi and box.y0 < end[1] < box.y1
        texts = [text.get_text() for text in drawn_texts(fig)]
        assert not any(word in text for text in texts for word in ("converged", "uncertified", "†"))
        assert max(sim_gallery.overhang(fig)) <= 0
        plt.close(fig)


def test_the_overall_rows_write_the_pooled_values_the_json_files_give():
    rows = [{"trait": trait, "relative": 0.5, "relative_lower": -1.0, "relative_upper": 2.0}
            for trait in ("ldl_cholesterol", "gout", "egfr")]
    common = {"name": "All converged traits", "relative": 0.2, "relative_lower": -0.6, "relative_upper": 1.1,
              "p_sharp": 0.3197, "traits": 3}
    for pooled, swap, top_note, gain_note in (
            (POOLED, common, "SV-PGS r² +17% vs LDpred2-auto, p = 0.005", "p = 0.32"),
            ({"with_sv": {"relative": -3.4, "p_one": 0.0123}}, {**common, "p_sharp": 0.0004},
             "SV-PGS r² −3% vs LDpred2-auto, p = 0.01", "p = 0.0004")):
        with plt.rc_context({**sim_gallery.POSTER_STYLE, "axes.labelsize": sim_gallery.POSTER_LABEL}):
            fig, lines = aou_gallery.gain_figure(rows, swap, compared_rows(), pooled)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        top, gain = fig.axes
        assert notes(top) == top_note and notes(gain) == gain_note
        # each panel's pooled row is last and named here, not by the json
        assert row_names(top)[-1] == row_names(gain)[-1] == aou_gallery.COMBINED == "Overall"
        assert "one-sided p" in lines[0]
        texts = list(drawn_texts(fig))
        assert not any(word in text.get_text() for text in texts for word in ("converged", "uncertified", "†"))
        # no two pieces of type overlap, and no note touches a mark, interval or arrowhead
        boxes = [text.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted()) for text in texts]
        for index, box in enumerate(boxes):
            for other in boxes[index + 1:]:
                assert min(box.x1, other.x1) - max(box.x0, other.x0) <= 0.01 or \
                    min(box.y1, other.y1) - max(box.y0, other.y0) <= 0.01, (box, other)
        for ax in (top, gain):
            # marks and arrowheads, but not the zero line, which runs the panel's height, and each interval's ends
            drawn = [line.get_window_extent(renderer) for line in ax.lines if line.get_marker() not in ("None", "")]
            drawn += [Bbox(ax.transData.transform(segment)) for collection in ax.collections
                      for segment in collection.get_segments()]
            for text in ax.texts:
                if text.get_gid() == "note":
                    box = text.get_window_extent(renderer)
                    assert not any(box.overlaps(mark) for mark in drawn), text.get_text()
        assert max(sim_gallery.overhang(fig)) <= 0
        plt.close(fig)
    # without the pooled values neither panel writes a note, and the r² panel has no pooled row
    with plt.rc_context({**sim_gallery.POSTER_STYLE, "axes.labelsize": sim_gallery.POSTER_LABEL}):
        fig, _ = aou_gallery.gain_figure(rows, {key: value for key, value in common.items() if key != "p_sharp"},
                                         compared_rows())
    top, gain = fig.axes
    assert notes(top) == notes(gain) == "" and aou_gallery.COMBINED not in row_names(top)
    plt.close(fig)


GROUPS = {"EUR": {"sv_gain": {"percent": 0.07, "lower": -0.21, "upper": 0.35, "p_one": 0.32},
                  "vs_ldpred2": {"percent": 21.9, "lower": 2.8, "upper": 45.1, "p_one": 0.011}},
          "AFR": {"sv_gain": {"percent": -0.5, "lower": -2.1, "upper": 1.2, "p_one": 0.61},
                  "vs_ldpred2": {"percent": 34.6, "lower": -10.0, "upper": 80.0, "p_one": 0.2}},
          "AMR": {"sv_gain": {"percent": 0.3, "lower": -1.5, "upper": 2.0, "p_one": 0.41},
                  "vs_ldpred2": {"percent": -2.6, "lower": -20.0, "upper": 18.0, "p_one": 0.0042}}}
"""aou_ancestry.json's "groups": each group's pooled SV gain and its pooled change over LDpred2-auto."""


def test_each_ancestry_group_has_a_row_under_overall_in_both_panels_when_the_file_gives_it():
    rows = [{"trait": trait, "relative": 0.5, "relative_lower": -1.0, "relative_upper": 2.0}
            for trait in ("ldl_cholesterol", "egfr")]
    common = {"relative": 0.2, "relative_lower": -0.6, "relative_upper": 1.1, "p_sharp": 0.3197, "traits": 2}
    with plt.rc_context({**sim_gallery.POSTER_STYLE, "axes.labelsize": sim_gallery.POSTER_LABEL}):
        fig, lines = aou_gallery.gain_figure(rows, common, compared_rows(), POOLED, GROUPS)
    fig.canvas.draw()
    top, gain = fig.axes
    assert row_names(top)[-4:] == row_names(gain)[-4:] == ["Overall", "EUR", "AFR", "AMR"]
    by_row = {}
    for ax in (top, gain):
        names = [text for text in ax.texts if text.get_gid() == "name"]
        for text in ax.texts:
            if text.get_gid() == "note":
                # a note's line belongs to the row whose name is nearest above it, or level with it
                owner = max((name for name in names if name.get_position()[1] <= text.get_position()[1] + 1e-9),
                            key=lambda name: name.get_position()[1])
                by_row.setdefault((ax is top, owner.get_text()), []).append(text.get_text())
    assert by_row[True, "EUR"] == ["+22%, p = 0.01"] and by_row[True, "AFR"] == ["+35%, p = 0.20"]
    assert by_row[True, "AMR"] == ["−3%, p = 0.004"]
    assert [by_row[False, group] for group in ("Overall", "EUR", "AFR", "AMR")] == \
        [["p = 0.32"], ["p = 0.32"], ["p = 0.61"], ["p = 0.41"]]
    # the gain panel draws each group's common g and its interval, in percent like Overall's
    segments = [segment for collection in gain.collections for segment in collection.get_segments()]
    assert [tuple(np.round(segment[:, 0] * 100, 2)) for segment in segments[-3:]] == [(-0.21, 0.35), (-2.1, 1.2),
                                                                                       (-1.5, 2.0)]
    assert "ancestry group" in lines[0] and max(sim_gallery.overhang(fig)) <= 0
    plt.close(fig)


def test_the_figure_is_drawn_without_ancestry_rows_when_aou_ancestry_json_is_missing(monkeypatch, tmp_path):
    numbers = {"matched": {"rows": [{"trait": "egfr", "relative": 0.3, "relative_lower": -1.1, "relative_upper": 1.7}],
                           "common": {"relative": 0.2, "relative_lower": -0.6, "relative_upper": 1.1,
                                      "p_sharp": 0.32, "traits": 1}}}
    (tmp_path / "aou_numbers.json").write_text(json.dumps(numbers))
    (tmp_path / "aou_vs_ldpred2.json").write_text(json.dumps({"rows": compared_rows()[:1], "combined": POOLED}))
    for name, path in (("NUMBERS", "aou_numbers.json"), ("COMPARED", "aou_vs_ldpred2.json"),
                       ("ANCESTRY", "aou_ancestry.json")):
        monkeypatch.setattr(aou_gallery, name, tmp_path / path)
    monkeypatch.setattr(aou_gallery, "OUT", tmp_path)
    aou_gallery.main()
    assert "ancestry" not in (tmp_path / "aou_1_gain.txt").read_text()
    (tmp_path / "aou_ancestry.json").write_text(json.dumps({"groups": GROUPS}))
    aou_gallery.main()
    assert "ancestry group" in (tmp_path / "aou_1_gain.txt").read_text()
    assert (tmp_path / "aou_1_gain.png").is_file()


def test_p_values_are_written_to_one_significant_figure_below_001_and_two_decimals_otherwise():
    assert [aou_gallery.p_text(p) for p in (0.0047, 0.00012, 0.0099, 0.0123, 0.3197, 0.05)] == \
        ["0.005", "0.0001", "0.01", "0.01", "0.32", "0.05"]
    assert [aou_gallery.signed(value) for value in (17.45, -3.4, 0.2)] == ["+17%", "−3%", "+0%"]


def test_the_two_marks_in_an_all_of_us_r2_row_stack_without_touching():
    with plt.rc_context({**sim_gallery.POSTER_STYLE, "axes.labelsize": sim_gallery.POSTER_LABEL}):
        fig, _ = aou_gallery.gain_figure([], None, compared_rows())
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = sorted((line.get_window_extent(renderer) for line in fig.axes[0].lines if line.get_marker() in ("s", "o")),
                   key=lambda box: -box.y1)
    assert len(boxes) == 8
    for upper, lower in zip(boxes, boxes[1:]):
        assert lower.y1 < upper.y0
    plt.close(fig)


def test_the_forest_gives_each_traits_change_relative_to_the_r2_without_svs():
    # c = 0.12 against b = 0.10: g = 0.02 / 0.11, and the change relative to b is 20%
    g = 0.02 / 0.11
    assert aou_gallery.baseline_relative(g) == pytest.approx(0.2)
    # a weighted g maps to the weighted change relative to b with the same weights
    weights, before, after = np.array([2.0, 1.0]), np.array([0.10, 0.05]), np.array([0.11, 0.04])
    weighted = (weights * (after - before)).sum() / (weights * (after + before) / 2).sum()
    assert aou_gallery.baseline_relative(weighted) == pytest.approx((weights * (after - before)).sum()
                                                                    / (weights * before).sum())


def test_the_discussion_states_the_common_all_of_us_gain_only_when_it_is_given(monkeypatch, tmp_path):
    monkeypatch.setattr(build_pptx, "HERE", tmp_path)
    assert build_pptx.aou_result() == []
    (tmp_path / "aou_numbers.json").write_text(json.dumps({"matched": {"rows": [], "common": {
        "relative": -2.0, "relative_lower": -4.0, "relative_upper": 0.5, "relative_bound": 0.1, "traits": 10}}}))
    # the primary's common g with its 95% interval
    assert build_pptx.aou_result() == ["In All of Us, imputed SVs changed r² by −2.00% (95% CI −4.00 to 0.50%) over "
                                       "10 traits, so these traits have few SVs with large effects or poorly imputed "
                                       "causal SVs."]

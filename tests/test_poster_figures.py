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
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from matplotlib.transforms import Bbox  # noqa: E402

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
    with plt.rc_context(aou_gallery.STYLE):
        # each trait's change alone, then with the common change, then the whole figure the poster prints
        yield "aou_1_gain", aou_gallery.gain_figure(MATCHED)[0]
        yield "aou_1_gain_final", aou_gallery.gain_figure(MATCHED, COMMON)[0]
        yield "aou_1_gain_poster", aou_gallery.gain_figure(MATCHED, COMMON, compared_rows(), POOLED, GROUPS)[0]
        # with asterisks past the intervals whose q is below 0.05, gout's at the end of its arrowhead
        yield "aou_1_gain_starred", aou_gallery.gain_figure(MATCHED, COMMON, compared_rows(0.01), POOLED, GROUPS)[0]


MATCHED = [{"trait": trait, "relative": mean, "relative_lower": low, "relative_upper": high}
           for trait, mean, low, high in (("ldl_cholesterol", -2.0, -7.0, 3.1), ("egfr", 0.3, -1.1, 1.7),
                                          ("hypothyroidism", 1.4, -0.4, 3.2), ("gout", 4.5, -24.6, 33.6),
                                          ("stroke", 0.5, -1.0, 2.0))]
"""aou_numbers.json's "matched" rows: each converged trait's g in percent with its 95% interval."""


COMMON = {"name": "All converged traits", "relative": 0.03, "relative_lower": -0.11, "relative_upper": 0.17,
          "p_sharp": 0.32, "traits": 5}
"""aou_numbers.json's "common": the traits' common g in percent with its 95% interval and one-sided swap p, under the
json's own name for it."""


POOLED = {"trait": "All converged traits", "with_sv": {"relative": 17.45, "relative_lower": 4.01, "relative_upper": 32.79,
                                                       "p_one": 0.0047}}
"""aou_vs_ldpred2.json's "combined": SV-PGS with SVs pooled against LDpred2-auto, under the json's own name for it."""


GROUPS = {"EUR": {"sv_gain": {"percent": 0.07, "lower": -0.21, "upper": 0.35, "p_one": 0.32},
                  "vs_ldpred2": {"percent": 21.9, "lower": 2.8, "upper": 45.1, "p_one": 0.011}},
          "AFR": {"sv_gain": {"percent": -0.5, "lower": -2.1, "upper": 1.2, "p_one": 0.61},
                  "vs_ldpred2": {"percent": 34.6, "lower": -10.0, "upper": 80.0, "p_one": 0.2}},
          "AMR": {"sv_gain": {"percent": 0.3, "lower": -1.5, "upper": 2.0, "p_one": 0.41},
                  "vs_ldpred2": {"percent": -2.6, "lower": -20.0, "upper": 18.0, "p_one": 0.0042}}}
"""aou_ancestry.json's "groups": each group's pooled SV gain and its pooled change over LDpred2-auto."""


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


def aou_figure(*args):
    """gain_figure's figure and description, made and drawn in the style main() saves it in."""
    with plt.rc_context(aou_gallery.STYLE):
        fig, lines = aou_gallery.gain_figure(*args)
        fig.canvas.draw()
    return fig, lines


PANELS = ["Held-out r² (%)", "SV-PGS vs LDpred2-auto (%)", "Change in r² with SVs (%)"]
"""The All of Us figure's panels' axis titles, top to bottom."""


def title(ax) -> str:
    """An All of Us panel's axis title, under its axis or beside it on its tick labels' line."""
    return ax.get_xlabel() or next(text.get_text() for text in ax.texts if text.get_gid() == "title")


def row_names(ax) -> list[str]:
    """An All of Us panel's row names, top to bottom."""
    return [text.get_text() for text in ax.texts if text.get_gid() == "name"]


def owner(ax, y: float) -> str:
    """The row of an All of Us panel whose name is nearest ``y``, in the axes' rows."""
    names = [text for text in ax.texts if text.get_gid() == "name"]
    return min(names, key=lambda text: abs(text.get_position()[1] - y)).get_text()


def marks(ax) -> dict[str, list[tuple[str, float, tuple[float, float]]]]:
    """Each row's marks, top to bottom, by row name: its marker, its estimate and its drawn interval's ends."""
    drawn = sorted((line.get_ydata()[0], line.get_marker(), line.get_xdata()[0]) for line in ax.lines
                   if line.get_marker() in ("s", "o"))
    intervals = sorted((segment[0, 1], tuple(segment[:, 0])) for collection in ax.collections
                       for segment in collection.get_segments())
    rows = {}
    for (y, marker, x), (_, ends) in zip(drawn, intervals, strict=True):
        rows.setdefault(owner(ax, y), []).append((marker, x, ends))
    return rows


def arrowheads(ax) -> dict[str, list[str]]:
    """Each row's arrowheads by row name, "<" at the axis's start and ">" at its end."""
    heads = {}
    for line in ax.lines:
        if line.get_marker() in ("<", ">"):
            heads.setdefault(owner(ax, line.get_ydata()[0]), []).append(line.get_marker())
    return heads


def row_notes(ax) -> dict[str, str]:
    """Each row's note by row name, its lines joined by spaces."""
    notes = {}
    for text in ax.texts:
        if text.get_gid() == "note":
            name = owner(ax, text.get_position()[1])
            notes[name] = f"{notes[name]} {text.get_text()}" if name in notes else text.get_text()
    return notes


def test_the_all_of_us_figure_fills_the_right_column_with_its_spare_height_between_the_panels(monkeypatch):
    fig, _ = aou_figure(MATCHED, COMMON, compared_rows(), POOLED, GROUPS)
    assert [title(ax) for ax in fig.axes] == PANELS
    height = aou_gallery.HEIGHT
    assert tuple(fig.get_size_inches()) == pytest.approx((aou_gallery.COLUMN, height))
    # the same panels at their own height: each axis as tall, the first as far under the canvas's top and the last as
    # far over its foot, and each gap between panels shorter by an equal share of the difference
    monkeypatch.setattr(aou_gallery, "HEIGHT", 0.0)
    natural, _ = aou_figure(MATCHED, COMMON, compared_rows(), POOLED, GROUPS)
    up = natural.get_size_inches()[1]
    spare = height - up
    assert spare > 0

    def spans(figure, tall):
        return [(tall * (1 - ax.get_position().y1), tall * ax.get_position().y0) for ax in figure.axes]

    filled, own = spans(fig, height), spans(natural, up)
    assert filled[-1][1] == pytest.approx(own[-1][1])
    for index, ((top, foot), (own_top, own_foot)) in enumerate(zip(filled, own)):
        assert height - top - foot == pytest.approx(up - own_top - own_foot)
        assert top == pytest.approx(own_top + index * spare / (len(own) - 1))
    plt.close(fig)
    plt.close(natural)


def test_every_all_of_us_label_is_one_line_on_the_canvas_and_touches_no_other_label_or_mark():
    for q in (0.3, 0.01):
        fig, _ = aou_figure(MATCHED, COMMON, compared_rows(q), POOLED, GROUPS)
        renderer = fig.canvas.get_renderer()
        across, up = fig.get_size_inches()
        texts = list(drawn_texts(fig))
        boxes = []
        for text in texts:
            words = text.get_text()
            box = text.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())
            assert "\n" not in words and box.height <= 1.5 * text.get_fontsize() / 72, words
            assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= across and box.y1 <= up, words
            # no word of a fit's convergence, of one method beating the other, or of the intervals' level
            assert not re.search(r"converg|beat|95% CI|uncertified|†", words, re.IGNORECASE), words
            boxes.append(box)
        for index, box in enumerate(boxes):
            for other in boxes[index + 1:]:
                assert min(box.x1, other.x1) - max(box.x0, other.x0) <= 0.01 or \
                    min(box.y1, other.y1) - max(box.y0, other.y0) <= 0.01, (box, other)
        for ax in fig.axes:
            # marks and arrowheads, but not the zero line, which runs the panel's height, and each interval's ends
            drawn = [line.get_window_extent(renderer) for line in ax.lines if line.get_marker() not in ("None", "")]
            drawn += [Bbox(ax.transData.transform(segment)) for collection in ax.collections
                      for segment in collection.get_segments()]
            for text in ax.texts:
                if text.get_gid() in ("note", "name", "title"):
                    box = text.get_window_extent(renderer)
                    assert not any(box.overlaps(mark) for mark in drawn), text.get_text()
        assert max(sim_gallery.overhang(fig)) <= 0
        plt.close(fig)


def test_the_per_trait_panel_draws_each_methods_held_out_r2_in_percent_on_an_absolute_axis():
    fig, lines = aou_figure(MATCHED, COMMON, compared_rows(), POOLED, GROUPS)
    r2 = fig.axes[0]
    assert title(r2) == "Held-out r² (%)" and r2.get_xscale() == "log" and "percent" in lines[0]
    low, high = r2.get_xlim()
    drawn = marks(r2)
    # compared's order; each method's r² times 100, LDpred2-auto's square over SV-PGS's circle, its interval ending at
    # the axis's end where it runs past it
    assert list(drawn) == [aou_gallery.trait_name(row["trait"]) for row in compared_rows()]
    for row in compared_rows():
        got = drawn[aou_gallery.trait_name(row["trait"])]
        assert [marker for marker, _, _ in got] == ["s", "o"]
        values = [(100 * row["r2"][arm]["r2"], 100 * row["r2"][arm]["lower"], 100 * row["r2"][arm]["upper"])
                  for arm in ("ldpred2_auto", "with_sv")]
        assert [number for _, x, ends in got for number in (x, *ends)] == pytest.approx(
            [number for value, lower, upper in values for number in (value, max(lower, low), min(upper, high))])
    # ticked at the decades inside the axis, without a % sign
    labels = [label.get_text() for label in r2.get_xticklabels()]
    assert labels and set(labels) <= {"0.01", "0.1", "1", "10"}
    assert all(low <= tick <= high for tick in r2.get_xticks())
    # no pooled row, whose r² would be relative
    assert not {aou_gallery.COMBINED, *aou_gallery.GROUPS} & set(row_names(r2)) and not row_notes(r2)
    plt.close(fig)


def test_the_pooled_rows_draw_the_json_values_with_their_one_sided_p():
    fig, lines = aou_figure(MATCHED, COMMON, compared_rows(), POOLED, GROUPS)
    _, versus, gain = fig.axes
    against = POOLED["with_sv"]
    for ax, overall, test in (
            (versus, [against[key] for key in ("relative", "relative_lower", "relative_upper", "p_one")], "vs_ldpred2"),
            (gain, [COMMON[key] for key in ("relative", "relative_lower", "relative_upper", "p_sharp")], "sv_gain")):
        expected = {aou_gallery.COMBINED: overall} | {
            group: [GROUPS[group][test][key] for key in ("percent", "lower", "upper", "p_one")]
            for group in aou_gallery.GROUPS}
        # named here, not by the json, and last, Overall in bold
        assert row_names(ax)[-4:] == list(expected) == ["Overall", "EUR", "AFR", "AMR"]
        assert [text.get_fontweight() for text in ax.texts if text.get_text() == "Overall"] == ["bold"]
        low, high = ax.get_xlim()
        drawn, notes, heads = marks(ax), row_notes(ax), arrowheads(ax)
        for name, (value, lower, upper, p) in expected.items():
            # a fraction on the percent ticks, its interval ending in an arrowhead at the axis's end where it runs past
            [(_, x, ends)] = drawn[name]
            assert [x, *ends] == pytest.approx([value / 100, max(lower / 100, low), min(upper / 100, high)])
            assert sorted(heads.get(name, [])) == [head for head, past in (("<", lower / 100 < low),
                                                                           (">", upper / 100 > high)) if past]
            assert notes[name] == f"p = {aou_gallery.p_text(p)}"
        # the pooled rows alone carry notes
        assert set(notes) == set(expected)
    # the SV gain panel's traits: each g relative to the r² without SVs, compared's order and then the rest
    assert row_names(gain)[:-4] == ["eGFR", "LDL cholesterol", "Gout", "Hypothyroidism", "Stroke"]
    traits = marks(gain)
    for row in MATCHED:
        [(_, x, _)] = traits[aou_gallery.trait_name(row["trait"])]
        assert x == pytest.approx(aou_gallery.baseline_relative(row["relative"] / 100))
    assert "overall and by ancestry group" in lines[0] and "one-sided p" in lines[0]
    plt.close(fig)


def test_a_fit_that_did_not_converge_and_a_value_the_json_lacks_are_in_no_panel():
    for unconverged in ({"unconverged": True}, {"flags": "uncertified"}):
        fig, _ = aou_figure([{**MATCHED[0], **unconverged}, *MATCHED[1:]], COMMON, compared_rows(), POOLED, GROUPS)
        assert [title(ax) for ax in fig.axes] == PANELS
        assert not any("LDL cholesterol" in row_names(ax) for ax in fig.axes)
        plt.close(fig)
    # without the pooled values, the groups, or the common change's p: neither a pooled panel nor a pooled row
    fig, lines = aou_figure(MATCHED, {key: value for key, value in COMMON.items() if key != "p_sharp"},
                            compared_rows())
    assert [title(ax) for ax in fig.axes] == [PANELS[0], PANELS[2]]
    assert row_names(fig.axes[-1]) == ["eGFR", "LDL cholesterol", "Gout", "Hypothyroidism", "Stroke"]
    assert not any(row_notes(ax) for ax in fig.axes) and "pooled" not in lines[0]
    plt.close(fig)
    # the groups without Overall's values: their rows alone, and a description that says so
    fig, lines = aou_figure(MATCHED, None, None, None, GROUPS)
    assert [title(ax) for ax in fig.axes] == PANELS[1:]
    assert [row_names(ax)[-3:] for ax in fig.axes] == [list(aou_gallery.GROUPS)] * 2
    assert "Overall" not in row_names(fig.axes[1]) and "overall" not in lines[0] and "by ancestry group" in lines[0]
    plt.close(fig)
    # each trait's change alone is one panel
    fig, _ = aou_figure(MATCHED[1:2])
    assert [row_names(ax) for ax in fig.axes] == [["eGFR"]]
    plt.close(fig)


def test_the_forest_draws_a_lone_panel_on_its_canvas():
    panel = aou_gallery.Panel([("eGFR", [(0.5, 0.1, 0.9)])], [("", sim_gallery.SKY, "o", False)], "Change (%)")
    with plt.rc_context(aou_gallery.STYLE):
        alone = aou_gallery.forest([panel])
    assert len(alone.axes) == 1 and max(sim_gallery.overhang(alone)) <= 0
    plt.close(alone)


def test_a_q_below_005_stars_its_interval_and_the_key_and_none_draws_neither():
    rows = [{"trait": trait, "relative": 0.5, "relative_lower": -1.0, "relative_upper": 2.0}
            for trait in ("ldl_cholesterol", "gout", "egfr")]
    for q, starred in ((0.01, True), (0.05, False), (0.3, False)):
        fig, lines = aou_figure(rows, None, compared_rows(q))
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
        assert max(sim_gallery.overhang(fig)) <= 0
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


def test_the_two_marks_in_an_all_of_us_r2_row_stack_without_touching():
    fig, _ = aou_figure([], None, compared_rows())
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
    assert build_pptx.aou_result() == ["In All of Us, imputed SVs changed r² by −2.00% (95% CI −4.00 to 0.50%) "
                                       "across 10 traits. These traits may have few SVs with large effects or poorly "
                                       "imputed causal SVs."]

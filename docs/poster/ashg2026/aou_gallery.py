"""The poster's All of Us figures, drawn in the simulation figures' poster style (sim_gallery.py) from aggregate values
alone: rows of the runs' exported tables, typed into NUMBERS from the workspace's terminal. No participant, genotype or
count of 1 to 20 is among them, and the file, like the figures, stays out of git until the results are released.

NUMBERS holds:
- "matched": the rows of SV-PGS with SVs against SV-PGS without them, fitted by identical code and scored on the same
  held-out people, one a trait: the held-out partial r² with SVs less without as a percentage of the two models' mean
  partial r², the pre-specified g of sv_gain_meta.py (relative, relative_lower, relative_upper: its 95% interval),
  for the traits whose two fits both converged alone (a row marked "unconverged" stops the figure). An optional
  "common" holds those traits' common g the same way, with "traits", the number of traits it combines;
- "enrichment": prior_enrichment.tsv's rows of arm with_sv, one a (trait, class) and one a trait for every SV together:
  the class's mean posterior E[beta^2 | data] per record over the SNVs' (posterior_enrichment, with its jackknife
  interval), the learned prior's (prior_enrichment), and the class's share of the model's genetic variance and of the
  fitted variants;
- "by_length" and "alpha": prior_by_length.tsv's and frequency_alpha.tsv's rows of arm with_sv.
None of the last three is drawn: their posterior moments E[beta^2 | data] are mostly the learned prior's mass where the
data say little about each record (gout's SVs: 98% posterior variance), so they are no measure of what the data put on
SVs.

Outputs, in figures/: aou_1_gain.png from "matched", a row per trait and one for "common" when it is given, drawn when
NUMBERS holds its rows (transparent, at the width of the poster's right column), with its .txt, whose first line
describes the picture. The figure gives each g as a change relative to the r² without SVs (baseline_relative), so its
axis title, "Change in r² with SVs (%)", reads as it says.

    uv run --with matplotlib --with scipy python docs/poster/ashg2026/aou_gallery.py
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.lines import Line2D

from sim_gallery import (INK, POSTER_BAR, POSTER_LABEL, POSTER_MARK, POSTER_PAD, POSTER_STYLE, SKY, frame, overhang, plt,
                         shifted, tallest, under_axes, widest)

HERE = Path(__file__).resolve().parent
NUMBERS = HERE / "aou_numbers.json"
OUT = HERE / "figures"
COLUMN = 11.4
"""Inches: the poster's right column (build_pptx.py's RIGHT), each figure's width, so its type prints at its own size."""
ROW_GAP = 0.05
"""Inches between one row's name and the next row's: a row is a line of poster type and this gap."""
LABEL_GAP = 0.3
"""Inches between the widest row name and the plot."""
MARK = 1.5
"""The forest's marks and intervals relative to the poster's (POSTER_MARK, POSTER_BAR) when its rows are a line of type
apart."""
HEIGHT = 17.22
"""Inches: the gain forest's least height, which fills the right column from its text to the Discussion band at the
poster's foot (build_pptx.py's layout). Rows further apart than a line of type have marks up to GROWTH times MARK."""
GROWTH = 4 / 3
TRAITS = {
    "ldl_cholesterol": "LDL cholesterol", "egfr": "eGFR", "hypertension": "Hypertension",
    "type_2_diabetes": "Type 2 diabetes", "coronary_artery_disease": "Coronary disease",
    "atrial_fibrillation": "Atrial fibrillation", "hypothyroidism": "Hypothyroidism", "gout": "Gout",
    "chronic_kidney_disease": "Kidney disease", "asthma": "Asthma", "stroke": "Stroke",
    "hyperlipidemia": "Hyperlipidemia",
}
"""Each trait's name on the poster; a trait not here is shown as its name with spaces."""


def trait_name(trait: str) -> str:
    """A trait's name on the poster: its TRAITS entry, or the name as aou_numbers.json gives it."""
    return TRAITS.get(trait, trait)


@dataclass
class Panel:
    """One panel of the All of Us figure: a row per (name, marks), each mark an (estimate, low, high) in its series'
    place, under the title ``xlabel``. A series is (its name in the key, colour, marker, filled)."""

    rows: list[tuple[str, list[tuple[float, float, float]]]]
    series: list[tuple[str, str, str, bool]]
    xlabel: str
    log: bool = False


def forest(panels: list[Panel], least: float = 0.0):
    """The panels stacked at the column's width, their rows' names once at the left in a margin they share. In each row
    the series' marks stand one above the next, each an estimate with its 95% interval, and a panel with more than one
    series has their key over it. A log panel spans its estimates' decades; a linear one is symmetric about 0 over its
    estimates, so that each mark clears an arrowhead at the axis's end, and an interval past the axis ends in one.
    Every margin and the rows' pitch are measured from the type, so the figure is as short as its type allows, unless
    ``least`` inches is taller: then its rows share that height and their marks grow with them, up to GROWTH."""
    fig = plt.figure(figsize=(COLUMN, 1.0))
    size = POSTER_STYLE["font.size"]
    names = [name for panel in panels for name, _ in panel.rows]
    weight = {name: "bold" if name == COMBINED else "normal" for name in names}
    line = max(tallest(fig, [name], fontsize=size, fontweight=weight[name]) for name in names) + ROW_GAP
    # one series' marks at MARK times the poster's, two or more at the poster's own, one above the next in a row
    marks = [POSTER_MARK * (MARK if len(panel.series) == 1 else 1.0) for panel in panels]
    pitches = [max(line, len(panel.series) * (mark / 72 + 2 * ROW_GAP)) for panel, mark in zip(panels, marks)]
    keys = [tallest(fig, [label for label, *_ in panel.series], fontsize=size) + POSTER_PAD if len(panel.series) > 1
            else 0.0 for panel in panels]
    bottoms = [under_axes(fig, [panel.xlabel]) for panel in panels]
    natural = sum(POSTER_PAD + key + pitch * len(panel.rows) + bottom
                  for panel, key, pitch, bottom in zip(panels, keys, pitches, bottoms))
    rows = sum(len(panel.rows) for panel in panels)
    height = max(least, natural)
    # an arrowhead at the axis's start reaches half its width to the left, toward the names
    arrow = max(marks) * GROWTH * 0.7 / 2 / 72
    left = max(widest(fig, [name], fontsize=size, fontweight=weight[name]) for name in names) + LABEL_GAP + arrow
    right = max(widest(fig, [tick_text(panel)(tick) for tick in ticks(panel)[-1:]], fontsize=size) / 2
                for panel in panels) + POSTER_PAD
    fig.set_size_inches(COLUMN, height)
    width = COLUMN - left - right
    top = height
    for panel, mark, pitch, key, bottom in zip(panels, marks, pitches, keys, bottoms):
        grown = pitch + (height - natural) / rows
        growth = min(grown / pitch, GROWTH)
        mark, bar, ring = mark * growth, POSTER_BAR * growth, 6.0 * growth
        top -= POSTER_PAD + key
        ax = frame(fig, left, top - grown * len(panel.rows), width, grown * len(panel.rows))
        low_limit, high_limit = limits(panel, width, (mark / 2 + ring / 2) / 72 + arrow + POSTER_PAD)
        if panel.log:
            ax.set_xscale("log")
        offsets = [(index - (len(panel.series) - 1) / 2) * (mark / 72 + 2 * ROW_GAP) / grown
                   for index in range(len(panel.series))]
        for row, (name, estimates) in enumerate(panel.rows):
            for (label, colour, marker, filled), offset, (value, low, high) in zip(panel.series, offsets, estimates):
                y = row + offset
                ax.hlines(y, max(low, low_limit), min(high, high_limit), color=colour, linewidth=bar, zorder=3,
                          capstyle="round")
                for end, edge, arrowhead in ((low, low_limit, "<"), (high, high_limit, ">")):
                    if (end < edge) if arrowhead == "<" else (end > edge):
                        ax.plot([edge], [y], marker=arrowhead, markersize=mark * 0.7, color=colour, markeredgewidth=0,
                                zorder=3, clip_on=False)
                ax.plot([value], [y], marker, markersize=mark, color=colour,
                        markerfacecolor=colour if filled else "white", markeredgecolor=colour,
                        markeredgewidth=0 if filled else ring, zorder=4)
            ax.text(0, row, name, transform=shifted(ax, -left), ha="left", va="center", fontweight=weight[name])
        if not panel.log:
            ax.axvline(0.0, color=INK, linewidth=4.0, zorder=2)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_ylim(len(panel.rows) - 0.5, -0.5)
        ax.set_xlim(low_limit, high_limit)
        ax.set_xticks(ticks(panel), [tick_text(panel)(tick) for tick in ticks(panel)])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel(panel.xlabel)
        # centred on the canvas, not on the axes, whose margins differ: a title wider than the axes then fits either side
        ax.xaxis.label.set_x((COLUMN / 2 - left) / width)
        if key:
            handles = [Line2D([0], [0], marker=marker, linestyle="none", markersize=mark, color=colour,
                              markerfacecolor=colour if filled else "white", markeredgecolor=colour,
                              markeredgewidth=0 if filled else ring, label=label)
                       for label, colour, marker, filled in panel.series]
            fig.legend(handles=handles, loc="lower left", ncol=len(handles), frameon=False, borderpad=0.0,
                       borderaxespad=0.0, handletextpad=0.3, columnspacing=1.2, handlelength=1.0,
                       bbox_to_anchor=(POSTER_PAD / COLUMN, (top + POSTER_PAD / 2) / height))
        top -= grown * len(panel.rows) + bottom
    return fig


def ticks(panel: Panel) -> list[float]:
    """A log panel's decades from below its smallest estimate to above its largest; a linear panel's span."""
    values = [value for _, estimates in panel.rows for value, _, _ in estimates]
    if panel.log:
        return [10.0 ** power for power in range(math.floor(math.log10(min(values))), math.ceil(math.log10(max(values))) + 1)]
    return span(values)


def tick_text(panel: Panel):
    """A log panel's ticks as plain decimals, a linear panel's in percent."""
    return (lambda tick: f"{tick:g}") if panel.log else percent


def limits(panel: Panel, width: float, clear: float) -> tuple[float, float]:
    """The axis's ends: each estimate ``clear`` inches from them at least, over ``width`` inches, and on a linear axis
    symmetric about 0 and wide enough for its ticks; on a log axis, its estimates' span alone."""
    values = [value for _, estimates in panel.rows for value, _, _ in estimates]
    if panel.log:
        low, high = math.log10(min(values)), math.log10(max(values))
        reach = (high - low) / (1 - 2 * clear / width)
        margin = (reach - (high - low)) / 2
        return 10.0 ** (low - margin), 10.0 ** (high + margin)
    reach = max(abs(value) for value in values) / (1 - 2 * clear / width)
    marks = ticks(panel)
    reach = max(reach, -marks[0], marks[-1])
    return -reach, reach


def percent(value: float) -> str:
    """A share as a tick label in percent, the % left to the axis title so the labels stay narrow."""
    return f"{value * 100:.0f}".replace("-", "−")


def span(values: list[float]) -> list[float]:
    """At most four ticks a round step apart, from below the smallest of ``values`` to above the largest, 0 among
    them."""
    low, high = min(min(values), 0.0), max(max(values), 0.0)
    step = next(step for step in (0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0) if math.ceil(high / step) - math.floor(low / step) <= 3)
    return list(np.arange(math.floor(low / step) * step, math.ceil(high / step) * step + step / 2, step))


def baseline_relative(g: float) -> float:
    """g, the change in r² from adding SVs as a fraction of the two models' mean r², as a fraction of the r² without SVs:
    (c - b) / ((c + b) / 2) = g gives (c - b) / b = 2g / (2 - g). The map increases, so it takes an interval's ends to
    the same interval's ends, and a weighted g, sum a (c - b) / sum a (c + b) / 2, to sum a (c - b) / sum a b with the
    same weights."""
    return 2 * g / (2 - g)


COMBINED = "All converged traits"
"""The All of Us forest's row for the traits' common g, its name in bold."""


def gain_figure(matched: list[dict], common: dict | None = None) -> tuple:
    """Each trait's change in held-out partial r² from adding SVs, relative to the r² without SVs (baseline_relative of
    the pre-specified g), with its 95% interval, a row per trait, and when ``common`` is given a bold row for the traits'
    common g itself, relative to the two arms' mean r². The two scales differ by about 0.1 percentage point at these
    gains, so one axis title serves both. No figure on the poster uses a fit that did not converge."""
    unconverged = [row["trait"] for row in matched if row.get("unconverged")]
    if unconverged:
        raise SystemExit(f"aou_numbers.json has fits that did not converge: {', '.join(unconverged)}")
    keys = ("relative", "relative_lower", "relative_upper")
    rows = [(trait_name(row["trait"]), *(baseline_relative(row[key] / 100) for key in keys)) for row in matched]
    if common:
        rows.append((COMBINED, *(common[key] / 100 for key in keys)))
    # the axis spans the estimates, not their intervals; an interval that runs past it ends in an arrowhead
    fig = forest(rows, "Change in r² with SVs (%)", span([value for _, value, _, _ in rows]), percent, log=False,
                 estimates=True, bold=frozenset({COMBINED}), least=HEIGHT)
    return fig, ["Change in held-out partial r² from adding SVs to SV-PGS for each trait, relative to the r² without "
                 "SVs, with 95% intervals, an arrowhead marking an interval that runs past the axis"
                 + (", and in bold the traits' common change relative to the two models' mean r²" if common else "")]


def main() -> None:
    numbers = json.loads(NUMBERS.read_text())
    with plt.rc_context({**POSTER_STYLE, "axes.labelsize": POSTER_LABEL}):
        drawn = []
        if numbers.get("matched", {}).get("rows"):
            drawn.append(("aou_1_gain", gain_figure(numbers["matched"]["rows"], numbers["matched"].get("common"))))
        for name, (fig, lines) in drawn:
            reach = overhang(fig)
            if max(reach) > 0:
                raise SystemExit(f"{name}: its type runs up to {max(reach):.2f} in past the canvas")
            fig.savefig(OUT / f"{name}.png", transparent=True)
            plt.close(fig)
            (OUT / f"{name}.txt").write_text("\n".join(lines) + "\n")
            print(OUT / f"{name}.png")


if __name__ == "__main__":
    main()

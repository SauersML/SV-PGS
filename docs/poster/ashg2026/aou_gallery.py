"""The poster's All of Us figures, drawn in the simulation figures' poster style (sim_gallery.py) from aggregate values
alone: rows of the runs' exported tables, typed into NUMBERS from the workspace's terminal. No participant, genotype or
count of 1 to 20 is among them, and the file, like the figures, stays out of git until the results are released.

NUMBERS holds:
- "matched": the rows of SV-PGS with SVs against SV-PGS without them, fitted by identical code and scored on the same
  held-out people, one a trait: the held-out partial r² with SVs less without as a percentage of the two models' mean
  partial r², the pre-specified g of sv_gain_meta.py (relative, relative_lower, relative_upper: its 95% interval), and
  "unconverged": true when the with-SV fit stopped before converging. An optional "common" holds the traits' common g
  the same way, with "traits", the number of traits it combines;
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
from pathlib import Path

import matplotlib
import numpy as np

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


def forest(rows: list[tuple[str, float, float, float]], xlabel: str, ticks: list[float], tick_text, log: bool,
           reference: float | None = None, estimates: bool = False, bold: frozenset = frozenset(),
           note: str | None = None, least: float = 0.0):
    """A forest at the column's width in the imputed SVs' sky blue: a row per (label, estimate, low, high), its label
    at the left of the canvas and its interval level with it, each estimate an open mark as imputed SVs' are in the
    simulation figures (every All of Us SV is imputed), with the reference drawn (0, or 1 on a log axis, unless given).
    The axis runs from the first tick to the last, or with ``estimates`` symmetric about the reference over the
    estimates alone, so that each estimate's mark clears an arrowhead at the axis's end. Every margin and the rows'
    pitch are measured from the type, so the forest is as short as its type allows, unless ``least`` inches is taller:
    then its rows share that height and its marks grow with them. The labels in ``bold`` are set in bold, and ``note``
    is one line under the axis title."""
    reference = (1.0 if log else 0.0) if reference is None else reference
    fig = plt.figure(figsize=(COLUMN, 1.0))
    labels = [label for label, *_ in rows]
    weight = {label: "bold" if label in bold else "normal" for label in labels}
    pitch = max(tallest(fig, [label], fontsize=POSTER_STYLE["font.size"], fontweight=weight[label])
                for label in labels) + ROW_GAP
    # the right margin holds half of the end tick's label; the bottom, the ticks and the axis title
    right = widest(fig, [tick_text(ticks[-1])], fontsize=POSTER_STYLE["font.size"]) / 2 + POSTER_PAD
    line = tallest(fig, [note], fontsize=POSTER_STYLE["font.size"]) + POSTER_PAD if note else 0.0
    top, bottom = POSTER_PAD, under_axes(fig, [xlabel]) + line
    height = max(top + bottom + pitch * len(rows), least)
    growth = MARK * min((height - top - bottom) / len(rows) / pitch, GROWTH)
    mark, bar, ring = POSTER_MARK * growth, POSTER_BAR * growth, 6.0 * growth
    # an arrowhead at the axis's start reaches half its width to the left, toward the names
    arrow = mark * 0.7 / 2 / 72
    left = max(widest(fig, [label], fontsize=POSTER_STYLE["font.size"], fontweight=weight[label])
               for label in labels) + LABEL_GAP + arrow
    fig.set_size_inches(COLUMN, height)
    ax = frame(fig, left, bottom, COLUMN - left - right, height - top - bottom)
    if log:
        ax.set_xscale("log")
    low_limit, high_limit = ticks[0], ticks[-1]
    if estimates:
        # inches each estimate keeps from the axis's end: its mark's radius and edge, an arrowhead's half, and a pad
        clear = (mark / 2 + ring / 2) / 72 + arrow + POSTER_PAD
        reach = max(abs(value - reference) for _, value, _, _ in rows) / (1 - 2 * clear / (COLUMN - left - right))
        reach = max(reach, reference - ticks[0], ticks[-1] - reference)
        low_limit, high_limit = reference - reach, reference + reach
    for row, (label, value, low, high) in enumerate(rows):
        ax.hlines(row, max(low, low_limit), min(high, high_limit), color=SKY, linewidth=bar, zorder=3,
                  capstyle="round")
        for end, edge, marker in ((low, low_limit, "<"), (high, high_limit, ">")):
            if (end < edge) if marker == "<" else (end > edge):
                ax.plot([edge], [row], marker=marker, markersize=mark * 0.7, color=SKY, markeredgewidth=0,
                        zorder=3, clip_on=False)
        ax.plot([value], [row], "o", markersize=mark, color=SKY, markerfacecolor="white", markeredgecolor=SKY,
                markeredgewidth=ring, zorder=4)
        ax.text(0, row, label, transform=shifted(ax, -left), ha="left", va="center", fontweight=weight[label])
    ax.axvline(reference, color=INK, linewidth=4.0, zorder=2)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(low_limit, high_limit)
    ax.set_xticks(ticks, [tick_text(tick) for tick in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel(xlabel)
    # centred on the canvas, not on the axes, whose margins differ: a title wider than the axes then fits either side
    ax.xaxis.label.set_x((COLUMN / 2 - left) / (COLUMN - left - right))
    if note:
        fig.text(0.5, POSTER_PAD / height, note, ha="center", va="bottom")
    return fig


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


UNCONVERGED = "† With-SV fit not converged"
"""The line under the All of Us forest's axis title when a trait's with-SV fit stopped before converging; the trait's
name then ends in the dagger."""
COMBINED = "All traits"
"""The All of Us forest's row for the traits' common g, its name in bold."""


def gain_figure(matched: list[dict], common: dict | None = None) -> tuple:
    """Each trait's change in held-out partial r² from adding SVs, relative to the r² without SVs (baseline_relative of
    the pre-specified g), with its 95% interval, a row per trait, and when ``common`` is given a bold row for the traits'
    common g itself, relative to the two arms' mean r². The two scales differ by about 0.1 percentage point at these
    gains, so one axis title serves both. A trait whose with-SV fit stopped before converging has a dagger, explained
    under the axis."""
    keys = ("relative", "relative_lower", "relative_upper")
    rows = [(trait_name(row["trait"]) + ("†" if row.get("unconverged") else ""),
             *(baseline_relative(row[key] / 100) for key in keys)) for row in matched]
    if common:
        rows.append((COMBINED, *(common[key] / 100 for key in keys)))
    note = UNCONVERGED if any(row.get("unconverged") for row in matched) else None
    # the axis spans the estimates, not their intervals; an interval that runs past it ends in an arrowhead
    fig = forest(rows, "Change in r² with SVs (%)", span([value for _, value, _, _ in rows]), percent, log=False,
                 estimates=True, bold=frozenset({COMBINED}), note=note, least=HEIGHT)
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

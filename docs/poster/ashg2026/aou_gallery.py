"""The poster's All of Us figures, drawn in the simulation figures' poster style (sim_gallery.py) from aggregate values
alone: rows of the runs' exported tables, typed into NUMBERS from the workspace's terminal. No participant, genotype or
count of 1 to 20 is among them, and the file, like the figures, stays out of git until the results are released.

NUMBERS holds:
- "matched": the rows of SV-PGS with SVs against SV-PGS without them, fitted by identical code and scored on the same
  held-out people, one a trait: the held-out partial r² with SVs less without as a percentage of the two models' mean
  partial r², the pre-specified g of sv_gain_meta.py (relative, relative_lower, relative_upper: its 95% interval);
- "enrichment": prior_enrichment.tsv's rows of arm with_sv, one a (trait, class) and one a trait for every SV together:
  the class's mean posterior E[beta^2 | data] per record over the SNVs' (posterior_enrichment, with its jackknife
  interval), the learned prior's (prior_enrichment), and the class's share of the model's genetic variance and of the
  fitted variants;
- "by_length" and "alpha": prior_by_length.tsv's and frequency_alpha.tsv's rows of arm with_sv.
None of the last three is drawn: their posterior moments E[beta^2 | data] are mostly the learned prior's mass where the
data say little about each record (gout's SVs: 98% posterior variance), so they are no measure of what the data put on
SVs.

Outputs, in figures/: aou_1_gain.png from "matched", a row per trait, drawn when NUMBERS holds its rows (transparent,
at the width of the poster's right column), with its .txt, whose first line describes the picture.

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
COLUMN = 9.8
"""Inches: the poster's right column (build_pptx.py's RIGHT, 10.4) less its INSETs, each figure's width."""
ROW_GAP = 0.2
"""Inches between one row's name and the next row's: a row is a line of poster type and this gap."""
LABEL_GAP = 0.3
"""Inches between the widest row name and the plot."""
TRAITS = {
    "ldl_cholesterol": "LDL cholesterol", "egfr": "eGFR", "hypertension": "Hypertension",
    "type_2_diabetes": "Type 2 diabetes", "coronary_artery_disease": "Coronary disease",
    "atrial_fibrillation": "Atrial fibrillation", "hypothyroidism": "Hypothyroidism", "gout": "Gout",
    "chronic_kidney_disease": "Kidney disease", "asthma": "Asthma", "stroke": "Stroke",
}
"""Each trait's name on the poster; a trait not here is shown as its name with spaces."""


def trait_name(trait: str) -> str:
    """A trait's name on the poster: its TRAITS entry, or the name as aou_numbers.json gives it."""
    return TRAITS.get(trait, trait)


def forest(rows: list[tuple[str, float, float, float]], xlabel: str, ticks: list[float], tick_text, log: bool,
           reference: float | None = None):
    """A forest at the column's width in the imputed SVs' sky blue: a row per (label, estimate, low, high), its label
    at the left of the canvas and its interval level with it, each estimate an open mark as imputed SVs' are in the
    simulation figures (every All of Us SV is imputed), with the reference drawn (0, or 1 on a log axis, unless given).
    Every margin and the rows' pitch are measured from the type, so the forest is as short as its type allows."""
    reference = (1.0 if log else 0.0) if reference is None else reference
    fig = plt.figure(figsize=(COLUMN, 1.0))
    labels = [label for label, *_ in rows]
    pitch = tallest(fig, labels, fontsize=POSTER_STYLE["font.size"]) + ROW_GAP
    left = widest(fig, labels, fontsize=POSTER_STYLE["font.size"]) + LABEL_GAP
    # the right margin holds half of the end tick's label; the bottom, the ticks and the axis title
    right = widest(fig, [tick_text(ticks[-1])], fontsize=POSTER_STYLE["font.size"]) / 2 + POSTER_PAD
    top, bottom = POSTER_PAD, under_axes(fig, [xlabel])
    height = top + bottom + pitch * len(rows)
    fig.set_size_inches(COLUMN, height)
    ax = frame(fig, left, bottom, COLUMN - left - right, height - top - bottom)
    if log:
        ax.set_xscale("log")
    low_limit, high_limit = ticks[0], ticks[-1]
    for row, (label, value, low, high) in enumerate(rows):
        ax.hlines(row, max(low, low_limit), min(high, high_limit), color=SKY, linewidth=POSTER_BAR, zorder=3,
                  capstyle="round")
        for end, edge, marker in ((low, low_limit, "<"), (high, high_limit, ">")):
            if (end < edge) if marker == "<" else (end > edge):
                ax.plot([edge], [row], marker=marker, markersize=POSTER_MARK * 0.7, color=SKY, markeredgewidth=0,
                        zorder=3, clip_on=False)
        ax.plot([value], [row], "o", markersize=POSTER_MARK, color=SKY, markerfacecolor="white", markeredgecolor=SKY,
                markeredgewidth=6.0, zorder=4)
        ax.text(0, row, label, transform=shifted(ax, -left), ha="left", va="center")
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


def gain_figure(matched: list[dict]) -> tuple:
    """Each trait's change in held-out partial r² from adding SVs, as a share of the two models' mean partial r², with
    its 95% interval, a row per trait."""
    rows = [(trait_name(row["trait"]), row["relative"] / 100, row["relative_lower"] / 100, row["relative_upper"] / 100)
            for row in matched]
    # the axis spans every estimate and each interval up to twice the median interval's width; a wider one runs off
    # the axis in arrowheads, so one imprecise trait does not flatten the rest
    typical = 2 * float(np.median([high - low for _, _, low, high in rows]))
    ends = [end for _, value, low, high in rows for end in ((value,) if high - low > typical else (low, high))]
    fig = forest(rows, "Change in r² with SVs (%)\nand 95% CI", span(ends), percent, log=False)
    return fig, ["Change in held-out partial r² from adding SVs to SV-PGS for each trait, as a percentage of the two "
                 "models' mean partial r², with 95% intervals, an arrowhead marking an interval that runs past the axis"]


def main() -> None:
    numbers = json.loads(NUMBERS.read_text())
    with plt.rc_context({**POSTER_STYLE, "axes.labelsize": POSTER_LABEL}):
        drawn = []
        if numbers.get("matched", {}).get("rows"):
            drawn.append(("aou_1_gain", gain_figure(numbers["matched"]["rows"])))
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

"""The poster's All of Us figures, drawn in the simulation figures' poster style (sim_gallery.py) from aggregate values
alone: rows of the runs' exported tables, typed into NUMBERS from the workspace's terminal. No participant, genotype or
count of 1 to 20 is among them, and the file, like the figures, stays out of git until the results are released.

NUMBERS holds:
- "paired": paired_differences.tsv's rows of candidate with_sv, baseline snv_indel, cell all: the held-out partial r²
  with SVs less without, on the same people, as a share of the partial r² without (relative_estimate, relative_ci_low,
  relative_ci_high: the family-clustered bootstrap's percentile interval), one row a trait;
- "enrichment": prior_enrichment.tsv's rows of arm with_sv, one a (trait, class) and one a trait for every SV together:
  the class's mean posterior E[beta^2 | data] per record over the SNVs' (posterior_enrichment, with its jackknife
  interval), the learned prior's (prior_enrichment), and the class's share of the model's genetic variance and of the
  fitted variants;
- "by_length" and "alpha": prior_by_length.tsv's and frequency_alpha.tsv's rows of arm with_sv, not drawn here.

Outputs, in figures/: aou_1_gain.png from "paired" and aou_2_svs.png from "enrichment", each drawn when NUMBERS holds
its rows (transparent, at the width of the poster's right column), each with its .txt, whose first
line build_pptx.py sets as the figure's headline.

    uv run --with matplotlib --with scipy python docs/poster/ashg2026/aou_gallery.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
import numpy as np

from matplotlib.transforms import ScaledTranslation, blended_transform_factory
from sim_gallery import BLUE, INK, POSTER_BAR, POSTER_LABEL, POSTER_MARK, POSTER_STYLE, SKY, frame, plt, shifted, widest

HERE = Path(__file__).resolve().parent
NUMBERS = HERE / "aou_numbers.json"
OUT = HERE / "figures"
COLUMN = 9.8
"""Inches: the poster's right column (build_pptx.py's RIGHT, 10.4) less its INSETs, each figure's width."""
PITCH = 0.68
"""Inches between a forest's rows: a line of poster type and its interval with room between."""
TRAITS = {
    "ldl_cholesterol": "LDL cholesterol", "egfr": "eGFR", "hypertension": "Hypertension",
    "type_2_diabetes": "Type 2 diabetes", "coronary_artery_disease": "Coronary disease",
    "atrial_fibrillation": "Atrial fibrillation", "hypothyroidism": "Hypothyroidism", "gout": "Gout",
    "chronic_kidney_disease": "Kidney disease", "asthma": "Asthma", "stroke": "Stroke",
}
"""Each trait's name on the poster; a trait not here is shown as its name with spaces."""


def trait_name(trait: str) -> str:
    return TRAITS.get(trait, trait.replace("_", " ").capitalize())


def forest(rows: list[tuple[str, float, float, float]], xlabel: str, ticks: list[float], tick_text, log: bool,
           reference: float | None = None):
    """A forest at the column's width: a row per (label, estimate, low, high), its label at the left, the estimate
    filled where its interval excludes the reference (0, or 1 on a log axis, unless given), with the reference drawn."""
    reference = (1.0 if log else 0.0) if reference is None else reference
    top, bottom, right = 0.35, 1.85, 0.75
    height = top + bottom + PITCH * len(rows)
    fig = plt.figure(figsize=(COLUMN, height))
    left = widest(fig, [label for label, *_ in rows], fontsize=POSTER_STYLE["font.size"]) + 0.45
    ax = frame(fig, left, bottom, COLUMN - left - right, height - top - bottom)
    if log:
        ax.set_xscale("log")
    low_limit, high_limit = ticks[0], ticks[-1]
    for y, (label, value, low, high) in enumerate(rows):
        clear = low > reference or high < reference
        ax.hlines(y, max(low, low_limit), min(high, high_limit), color=BLUE, linewidth=POSTER_BAR, zorder=3,
                  capstyle="round")
        for end, edge, marker in ((low, low_limit, "<"), (high, high_limit, ">")):
            if (end < edge) if marker == "<" else (end > edge):
                ax.plot([edge], [y], marker=marker, markersize=POSTER_MARK * 0.7, color=BLUE, markeredgewidth=0,
                        zorder=3, clip_on=False)
        ax.plot([value], [y], "o", markersize=POSTER_MARK, color=BLUE, markerfacecolor=BLUE if clear else "white",
                markeredgecolor=BLUE, markeredgewidth=0 if clear else 6.0, zorder=4)
        ax.text(0, y, label, transform=shifted(ax, -0.3), ha="right", va="center")
    ax.axvline(reference, color=INK, linewidth=4.0, zorder=2)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(low_limit, high_limit)
    ax.set_xticks(ticks, [tick_text(tick) for tick in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    # the axis title centred on the figure, not the plot, so a long title has the labels' width as well
    ax.set_xlabel(xlabel)
    ax.xaxis.label.set_x((COLUMN / 2 - left) / (COLUMN - left - right))
    return fig


def percent(value: float) -> str:
    return "0" if round(value * 100) == 0 else f"{value * 100:.0f}%".replace("-", "−")


def span(values: list[float]) -> list[float]:
    """At most four ticks a round step apart, from below the smallest of ``values`` to above the largest, 0 among
    them."""
    low, high = min(min(values), 0.0), max(max(values), 0.0)
    step = next(step for step in (0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0) if math.ceil(high / step) - math.floor(low / step) <= 3)
    return list(np.arange(math.floor(low / step) * step, math.ceil(high / step) * step + step / 2, step))


def gain_figure(paired: list[dict]) -> tuple:
    """Each trait's change in held-out partial r² from adding SVs, as a share of the partial r² without them, with its
    bootstrap 95% interval, largest first; filled where the interval excludes 0."""
    rows = sorted(((trait_name(row["trait"]), row["relative_estimate"], row["relative_ci_low"], row["relative_ci_high"])
                   for row in paired), key=lambda row: -row[1])
    # an interval past -100% or +200% runs off the axis in an arrowhead
    ends = [min(max(end, -1.0), 2.0) for _, _, low, high in rows for end in (low, high)]
    fig = forest(rows, "Change in test r² with SVs", span(ends), percent, log=False)
    clear = [label for label, _, low, _ in rows if low > 0]
    title = (f"SVs raised accuracy for {', '.join(clear)}" if 0 < len(clear) <= 2
             else f"SVs raised accuracy for {len(clear)} of {len(rows)} traits")
    return fig, [title]


SV_COLOUR = SKY
"""Every SV together: the poster's SV colour (sim_gallery.py's VARIANTS)."""


def rows_figure(rows: list[tuple[str, float, float, float]], xlabel: str, limits: tuple[float, float],
                ticks: list[float], tick_text, reference: float, reference_name: str):
    """A forest at the column's width on a log axis: a row per (label, estimate, low, high), an interval past an axis
    end running off it in an arrowhead, and the reference drawn and named over the plot."""
    top, bottom, right = 1.05, 2.1, 0.5
    height = top + bottom + PITCH * 1.3 * len(rows)
    fig = plt.figure(figsize=(COLUMN, height))
    left = widest(fig, [label for label, *_ in rows], fontsize=POSTER_STYLE["font.size"]) + 0.45
    ax = frame(fig, left, bottom, COLUMN - left - right, height - top - bottom)
    ax.set_xscale("log")
    for y, (label, value, low, high) in enumerate(rows):
        ax.hlines(y, max(low, limits[0]) if low > 0 else limits[0], min(high, limits[1]), color=SV_COLOUR,
                  linewidth=POSTER_BAR, zorder=3, capstyle="round")
        for end, edge, marker in ((low, limits[0], "<"), (high, limits[1], ">")):
            if (end < edge) if marker == "<" else (end > edge):
                ax.plot([edge], [y], marker=marker, markersize=POSTER_MARK * 0.7, color=SV_COLOUR, markeredgewidth=0,
                        zorder=3, clip_on=False)
        ax.plot([value], [y], "o", markersize=POSTER_MARK, color=SV_COLOUR, markeredgewidth=0, zorder=4)
        ax.text(0, y, label, transform=shifted(ax, -0.3), ha="right", va="center")
    ax.axvline(reference, color=INK, linewidth=4.0, zorder=2)
    ax.text(reference, 1, reference_name, transform=blended_transform_factory(ax.transData, ax.transAxes)
            + ScaledTranslation(0, 0.15, fig.dpi_scale_trans), ha="center", va="bottom")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(*limits)
    ax.set_xticks(ticks, [tick_text(tick) for tick in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    # the axis title centred on the figure, not the plot, so it has the labels' width as well
    ax.set_xlabel(xlabel)
    ax.xaxis.label.set_x((COLUMN / 2 - left) / (COLUMN - left - right))
    return fig


def svs_figure(enrichment: list[dict]) -> tuple:
    """Per trait, every SV's posterior effect variance per variant (E[beta^2 | data] on the standardized genotype)
    over the SNVs', with its block-jackknife 95% interval, on a log axis, the largest first; the SNVs' (1) drawn and
    named. The ticks are plain ratios: at the poster's type a "×" after each would run them together."""
    rows = sorted(((trait_name(row["trait"]), row["posterior_enrichment"], row["posterior_enrichment_ci_low"],
                    row["posterior_enrichment_ci_high"]) for row in enrichment if row["variant_class"] == "all_structural"),
                  key=lambda row: -row[1])
    fig = rows_figure(rows, "Effect variance vs SNVs", (0.03, 30.0), [0.1, 1.0, 10.0], lambda tick: f"{tick:g}",
                      reference=1.0, reference_name="SNVs")
    above = [label for label, _, low, _ in rows if low > 1.0]
    which = f"only in {above[0].lower()}" if len(above) == 1 else f"in {len(above)} of {len(rows)} traits"
    return fig, [f"Per variant, SVs beat SNVs {which}"]


def main() -> None:
    numbers = json.loads(NUMBERS.read_text())
    with plt.rc_context({**POSTER_STYLE, "axes.labelsize": POSTER_LABEL}):
        drawn = []
        if numbers.get("paired"):
            drawn.append(("aou_1_gain", gain_figure(numbers["paired"])))
        if numbers.get("enrichment"):
            drawn.append(("aou_2_svs", svs_figure(numbers["enrichment"])))
        for name, (fig, lines) in drawn:
            fig.savefig(OUT / f"{name}.png", transparent=True)
            plt.close(fig)
            (OUT / f"{name}.txt").write_text("\n".join(lines) + "\n")
            print(OUT / f"{name}.png")


if __name__ == "__main__":
    main()

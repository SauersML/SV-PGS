"""The poster's All of Us figures, drawn in the simulation figures' poster style (sim_gallery.py) from aggregate values
alone: rows of the runs' exported tables, typed into NUMBERS from the workspace's terminal. No participant, genotype or
count of 1 to 20 is among them, and the file, like the figures, stays out of git until the results are released.

NUMBERS holds:
- "paired": paired_differences.tsv's rows of candidate with_sv, baseline snv_indel, cell all: the held-out partial r²
  with SVs less without, on the same people, as a share of the partial r² without (relative_estimate, relative_ci_low,
  relative_ci_high: the family-clustered bootstrap's percentile interval), one row a trait;
- "enrichment": prior_enrichment.tsv's rows of arm with_sv: a class's mean posterior E[beta^2 | data] per record over
  the SNVs' (posterior_enrichment, its jackknife standard error and interval) and the learned prior's (prior_enrichment);
- "alpha", optionally: frequency_alpha.tsv's rows of arm with_sv.

Outputs, in figures/: aou_1_gain.png and aou_2_classes.png (transparent, at the width of the poster's right column),
each with its .txt, whose first line build_pptx.py sets as the figure's headline.

    uv run --with matplotlib --with scipy python docs/poster/ashg2026/aou_gallery.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
import numpy as np

from sim_gallery import (
    BLUE, GREEN, INK, ORANGE, POSTER_BAR, POSTER_LABEL, POSTER_MARK, POSTER_NAME, POSTER_STYLE, PURPLE, frame, plain,
    plt, shifted, widest,
)

HERE = Path(__file__).resolve().parent
NUMBERS = HERE / "aou_numbers.json"
OUT = HERE / "figures"
COLUMN = 10.8
"""Inches: the poster's right column (build_pptx.py's RIGHT, 11.4) less its INSETs, each figure's width."""
PITCH = 0.68
"""Inches between a forest's rows: a line of poster type and its interval with room between."""
TRAITS = {
    "ldl_cholesterol": "LDL cholesterol", "egfr": "eGFR", "hypertension": "Hypertension",
    "type_2_diabetes": "Type 2 diabetes", "coronary_artery_disease": "Coronary disease",
    "atrial_fibrillation": "Atrial fibrillation", "hypothyroidism": "Hypothyroidism", "gout": "Gout",
    "chronic_kidney_disease": "Kidney disease", "asthma": "Asthma", "stroke": "Stroke",
}
"""Each trait's name on the poster; a trait not here is shown as its name with spaces."""
CLASSES = {
    "deletion": "Deletions", "insertion": "Insertions", "duplication": "Duplications",
    "insertion_mei": "Mobile elements", "inversion": "Inversions", "inversion_bnd_complex": "Complex SVs",
    "str_vntr_repeat": "Tandem repeats", "other_complex_sv": "Other SVs", "all_structural": "All SVs",
}
"""Each structural class's name on the poster, in the order drawn; copy numbers are not among the classes fitted."""
Z = 1.959963984540054


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


def pooled(rows: list[dict]) -> tuple[float, float, float, int]:
    """The traits' posterior enrichments of one class combined on the log scale by inverse variance (each trait's
    standard error over its estimate), with its normal 95% interval; traits are treated as independent."""
    logs, weights = [], []
    for row in rows:
        value, error = row["posterior_enrichment"], row["posterior_enrichment_standard_error"]
        if value and value > 0 and error and error > 0:
            logs.append(math.log(value))
            weights.append((value / error) ** 2)
    if not logs:
        return math.nan, math.nan, math.nan, 0
    weights_array = np.asarray(weights)
    mean = float(np.dot(weights_array, logs) / weights_array.sum())
    half = Z / math.sqrt(weights_array.sum())
    return math.exp(mean), math.exp(mean - half), math.exp(mean + half), len(logs)


def classes_figure(enrichment: list[dict]) -> tuple:
    """Each structural class's posterior effect-size variance per variant relative to SNVs, combined over traits,
    with its 95% interval, largest first; 1 is the SNVs'."""
    rows = []
    for name, label in CLASSES.items():
        found = pooled([row for row in enrichment if row["variant_class"] == name])
        if found[3]:
            rows.append((label, *found[:3]))
    rows.sort(key=lambda row: -row[1])
    ends = [value for _, _, low, high in rows for value in (low, high)]
    ticks = [10.0 ** power for power in range(math.floor(math.log10(min(ends + [1.0]))), math.ceil(math.log10(max(ends + [1.0]))) + 1)]
    fig = forest(rows, "Effect variance vs SNVs", ticks, lambda tick: f"{tick:g}×", log=True)
    larger = [label for label, value, low, _ in rows if low > 1.0]
    title = f"{larger[0]} have the largest effects per variant" if larger else "SV effects per variant, relative to SNVs"
    return fig, [title]


def alpha_figure(alpha: list[dict]) -> tuple:
    """Each class's frequency dependence alpha, effect variance per allele proportional to (2p(1 - p))^alpha, combined
    over traits by inverse variance with its 95% interval, SNVs first; the SNVs' alpha drawn as the reference line."""
    rows = []
    for name, label in (("snv", "SNVs"), *CLASSES.items()):
        found = [(row["posterior_alpha"], row["posterior_alpha_standard_error"]) for row in alpha
                 if row["variant_class"] == name and row.get("posterior_alpha_standard_error")]
        if not found:
            continue
        weights = np.array([1.0 / error ** 2 for _, error in found])
        mean = float(np.dot(weights, [value for value, _ in found]) / weights.sum())
        half = Z / math.sqrt(weights.sum())
        rows.append((label, mean, mean - half, mean + half))
    snv = next((row[1] for row in rows if row[0] == "SNVs"), 0.0)
    ends = [end for _, _, low, high in rows for end in (low, high)] + [0.0]
    fig = forest(rows, "Frequency dependence α", span(ends), lambda tick: plain(tick, 1), log=False, reference=snv)
    return fig, ["Rare SVs have larger effects"]


LENGTH_CLASSES = (("deletion", "Deletions", ORANGE), ("insertion", "Insertions", GREEN), ("duplication", "Duplications", PURPLE))
"""The classes drawn by length, each in its colour in the simulation figures (sim_gallery.py's SV_CLASSES)."""


def bin_start(label: str) -> float:
    """A length bin's lower end in bp: "<50" (the short records) as 5, "50-499" as 50."""
    return 5.0 if label.startswith("<") else float(label.split("-")[0])


def bin_text(start: float) -> str:
    return "<50" if start < 50 else f"{start / 1000:g}k" if start >= 1000 else f"{start:g}"


def length_figure(by_length: list[dict]) -> tuple:
    """Each class's posterior effect variance per variant relative to SNVs by length, combined over traits, with its
    95% interval: a line per class over its length bins (a bin's lower end on a log axis; the short records, under
    50 bp, at the left), 1 being the SNVs'."""
    height, top, bottom, left, right = 6.0, 1.0, 1.75, 1.95, 0.5
    fig = plt.figure(figsize=(COLUMN, height))
    ax = frame(fig, left, bottom, COLUMN - left - right, height - top - bottom)
    ax.set_xscale("log")
    ax.set_yscale("log")
    starts, ends = set(), [1.0]
    for index, (name, label, colour) in enumerate(LENGTH_CLASSES):
        bins = sorted({row["length_bin"] for row in by_length if row["variant_class"] == name}, key=bin_start)
        points = [(bin_start(label_), *pooled([row for row in by_length if row["variant_class"] == name and row["length_bin"] == label_]))
                  for label_ in bins]
        points = [point for point in points if point[4]]
        if not points:
            continue
        # each class a little apart along x, so overlapping intervals stay readable
        x = np.array([point[0] for point in points]) * 1.12 ** (index - 1)
        ax.plot(x, [point[1] for point in points], color=colour, linewidth=POSTER_BAR, zorder=3)
        ax.vlines(x, [point[2] for point in points], [point[3] for point in points], color=colour, linewidth=POSTER_BAR, zorder=3)
        ax.plot(x, [point[1] for point in points], "o", color=colour, markersize=POSTER_MARK * 0.8, markeredgewidth=0, zorder=4)
        # the classes named in the empty upper left, top to bottom in their drawing order
        ax.text(0.03, 0.97 - 0.15 * index, label, transform=ax.transAxes, color=colour, fontsize=POSTER_NAME,
                fontweight="bold", va="top", ha="left")
        starts.update(point[0] for point in points)
        ends.extend(value for point in points for value in point[2:4])
    ax.axhline(1.0, color=INK, linewidth=4.0, zorder=2)
    ticks = sorted(starts)
    ax.set_xticks(ticks, [bin_text(tick) for tick in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    powers = range(math.floor(math.log10(min(ends))), math.ceil(math.log10(max(ends))) + 1)
    ax.set_yticks([10.0 ** power for power in powers], [f"{10.0 ** power:g}×" for power in powers])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("Variant length (bp)")
    ax.text(0, 1, "Effect variance vs SNVs", transform=shifted(ax, -(left - 0.12), 0.3, False), ha="left", va="bottom",
            fontsize=POSTER_LABEL)
    return fig, ["Longer SVs have larger effects"]


def main() -> None:
    numbers = json.loads(NUMBERS.read_text())
    with plt.rc_context({**POSTER_STYLE, "axes.labelsize": POSTER_LABEL}):
        drawn = [("aou_1_gain", gain_figure(numbers["paired"]))]
        if numbers.get("enrichment"):
            drawn.append(("aou_2_classes", classes_figure(numbers["enrichment"])))
        if numbers.get("by_length"):
            drawn.append(("aou_3_length", length_figure(numbers["by_length"])))
        if numbers.get("alpha"):
            drawn.append(("aou_4_alpha", alpha_figure(numbers["alpha"])))
        for name, (fig, lines) in drawn:
            fig.savefig(OUT / f"{name}.png", transparent=True)
            plt.close(fig)
            (OUT / f"{name}.txt").write_text("\n".join(lines) + "\n")
            print(OUT / f"{name}.png")


if __name__ == "__main__":
    main()

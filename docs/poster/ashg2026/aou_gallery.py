"""The poster's All of Us figures, drawn in the simulation figures' poster style (sim_gallery.py) from aggregate values
alone: rows of the runs' exported tables, typed into NUMBERS from the workspace's terminal. No participant, genotype or
count of 1 to 20 is among them, and the file, like the figures, stays out of git until the results are released.

NUMBERS holds:
- "matched": the rows of SV-PGS with SVs against SV-PGS without them, fitted by identical code and scored on the same
  held-out people, one a trait: the held-out partial r² with SVs less without as a percentage of the two models' mean
  partial r², the pre-specified g of sv_gain_meta.py (relative, relative_lower, relative_upper: its 95% interval),
  for the traits whose two fits both converged alone (a row marked "unconverged" is left out). An optional
  "common" holds those traits' common g the same way, with "traits", the number of traits it combines;
- "enrichment": prior_enrichment.tsv's rows of arm with_sv, one a (trait, class) and one a trait for every SV together:
  the class's mean posterior E[beta^2 | data] per record over the SNVs' (posterior_enrichment, with its jackknife
  interval), the learned prior's (prior_enrichment), and the class's share of the model's genetic variance and of the
  fitted variants;
- "by_length" and "alpha": prior_by_length.tsv's and frequency_alpha.tsv's rows of arm with_sv.
None of the last three is drawn: their posterior moments E[beta^2 | data] are mostly the learned prior's mass where the
data say little about each record (gout's SVs: 98% posterior variance), so they are no measure of what the data put on
SVs.

COMPARED, aou_vs_ldpred2.json, holds each converged trait's held-out partial r² for LDpred2-auto and SV-PGS with and
without SVs, as fractions with their 95% intervals ("r2"), and the Benjamini-Hochberg q of SV-PGS with SVs against
LDpred2-auto ("with_sv") and of the change with SVs ("sv_gain"). A q below Q puts STAR past the trait's SV-PGS interval
in that panel, and STAR_NOTE in the key.

Outputs, in figures/: aou_1_gain.png from "matched", a row per trait and one for "common" when it is given, drawn when
NUMBERS holds its rows (transparent, at the width of the poster's right column), with its .txt, whose first line
describes the picture. Over it, when COMPARED gives the r², a panel of each trait's r² in percent for LDpred2-auto and
SV-PGS, its traits in COMPARED's order, which the gain panel keeps. The figure gives each g as a change relative to the
r² without SVs (baseline_relative), so its axis title, "Change in r² with SVs (%)", reads as it says.

    uv run --with matplotlib --with scipy python docs/poster/ashg2026/aou_gallery.py
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.textpath import TextPath
from matplotlib.transforms import ScaledTranslation

from sim_gallery import (INK, LDPRED2, POSTER_BAR, POSTER_LABEL, POSTER_MARK, POSTER_PAD, POSTER_STYLE, SKY, SV_PGS, frame,
                         overhang, plt, shifted, tallest, widest)

HERE = Path(__file__).resolve().parent
NUMBERS = HERE / "aou_numbers.json"
COMPARED = HERE / "aou_vs_ldpred2.json"
OUT = HERE / "figures"
COLUMN = 11.4
"""Inches: the poster's right column (build_pptx.py's RIGHT), each figure's width, so its type prints at its own size."""
ROW_GAP = 0.05
"""Inches between one row's name and the next row's: a row is a line of poster type and this gap, or its marks and
this gap when they are taller."""
MARK_GAP = 0.02
"""Inches between the marks that stand one above the next in a row."""
EDGE = 0.02
"""Inches between the figure's outermost type or mark and its canvas's edge."""
TITLE_PAD = 4.0
"""Points between an axis's tick labels and its title."""
RING = 6.0
"""Points: an open mark's ring at MARK times the poster's mark."""
Q = 0.05
"""A Benjamini-Hochberg q below this earns its trait's SV-PGS interval an asterisk."""
STAR, STAR_NOTE = "*", "* BH q < 0.05"
"""The mark past a significant interval's end, and its entry in the key, drawn only when some mark is."""
STAR_GAP = 0.06
"""Inches between an interval's drawn end (or its arrowhead) and its asterisk."""
LABEL_GAP = 0.3
"""Inches between the widest row name and the plot."""
MARK = 1.5
"""The forest's marks and intervals relative to the poster's (POSTER_MARK, POSTER_BAR) when its rows are a line of type
apart."""
HEIGHT = 0.0
"""Inches: the gain forest's least height, which fills the right column from its text to the Discussion band at the
poster's foot (build_pptx.py's layout). Rows further apart than a line of type have marks up to GROWTH times MARK."""
GROWTH = 4 / 3
TRAITS = {
    "ldl_cholesterol": "LDL cholesterol", "egfr": "eGFR", "hypertension": "Hypertension",
    "type_2_diabetes": "Type 2 diabetes", "coronary_artery_disease": "Coronary disease",
    "atrial_fibrillation": "Atrial fibrillation", "hypothyroidism": "Hypothyroidism", "gout": "Gout",
    "chronic_kidney_disease": "Kidney disease", "asthma": "Asthma", "stroke": "Stroke",
    "hyperlipidemia": "Hyperlipidemia", "hdl_cholesterol": "HDL cholesterol",
}
"""Each trait's name on the poster; a trait not here is shown as its name with spaces."""


def trait_name(trait: str) -> str:
    """A trait's name on the poster: its TRAITS entry, or the name as aou_numbers.json gives it."""
    return TRAITS.get(trait, trait)


@dataclass
class Panel:
    """One panel of the All of Us figure: a row per (name, marks), each mark an (estimate, low, high) in its series'
    place, under the title ``xlabel``. A series is (its name in the key, colour, marker, filled). Each (row name, series
    index) in ``starred`` has STAR past its interval's end."""

    rows: list[tuple[str, list[tuple[float, float, float]]]]
    series: list[tuple[str, str, str, bool]]
    xlabel: str
    log: bool = False
    starred: set[tuple[str, int]] = field(default_factory=set)


def forest(panels: list[Panel], least: float = 0.0):
    """The panels stacked at the column's width, their rows' names at the left in a margin they share, COMBINED's in
    bold. In each row the series' marks stand one above the next, each an estimate with its 95% interval, and a panel
    with more than one series has their key over it, wrapped onto further lines when it is wider than the column. When
    a panel stars a mark, STAR_NOTE ends the first panel's key, which it starts when that panel has none. A log panel spans its estimates and is
    ticked at the decades inside; a linear one is symmetric about 0 over its estimates. Either way each mark clears an
    arrowhead at the axis's end, and an interval past the axis ends in one. Every margin and row height is measured from
    the type, so the figure is as short as its type allows, unless ``least`` inches is taller: then its rows share the
    extra height and their marks grow with it, up to GROWTH."""
    fig = plt.figure(figsize=(COLUMN, 1.0))
    size = POSTER_STYLE["font.size"]
    shown = {name: name for panel in panels for name, _ in panel.rows}
    weight = {name: "bold" if name == COMBINED else "normal" for name in shown}
    # one series' marks at MARK times the poster's, two or more at the poster's own, one above the next in a row
    marks = [POSTER_MARK * (MARK if len(panel.series) == 1 else 1.0) for panel in panels]
    # a row is a line of type, or as many lines as its name takes, or its marks one above the next when taller
    line = tallest(fig, ["Hg"], fontsize=size)
    stacks = [sum(mark / 72 + (0.0 if filled else RING / 72) for *_, filled in panel.series)
              + (len(panel.series) - 1) * MARK_GAP for panel, mark in zip(panels, marks)]
    heights = [[max(tallest(fig, [shown[name]], fontsize=size, fontweight=weight[name]) if "\n" in shown[name] else line,
                    stack) + ROW_GAP for name, _ in panel.rows] for panel, stack in zip(panels, stacks)]
    # each panel's key: its series, each a mark and its name, and after the first panel's STAR_NOTE when a mark is starred
    entries = [[(label, (colour, marker, filled)) for label, colour, marker, filled in panel.series]
               if len(panel.series) > 1 else [] for panel in panels]
    if any(panel.starred for panel in panels):
        entries[0].append((STAR_NOTE, None))
    lines = [key_lines(fig, entry, mark) for entry, mark in zip(entries, marks)]
    keys = [len(rows) * (max(tallest(fig, [label for label, _ in entry], fontsize=size), mark / 72) + ROW_GAP)
            for rows, entry, mark in zip(lines, entries, marks)]
    # under each axis: its ticks, tick labels and title, and then POSTER_PAD to the next panel or EDGE to the canvas's foot
    style = matplotlib.rcParams
    bottoms = [(style["xtick.major.size"] + style["xtick.major.pad"] + TITLE_PAD) / 72
               + tallest(fig, ["0.1"], fontsize=style["xtick.labelsize"])
               + tallest(fig, [panel.xlabel], fontsize=style["axes.labelsize"]) + POSTER_PAD for panel in panels]
    bottoms[-1] += EDGE - POSTER_PAD
    natural = EDGE + sum(key + sum(rows) + bottom for key, rows, bottom in zip(keys, heights, bottoms))
    height = max(least, natural)
    extra = (height - natural) / sum(len(rows) for rows in heights)
    # an arrowhead at the axis's start reaches half its width to the left, toward the names
    arrow = max(marks) * GROWTH * 0.7 / 2 / 72
    left = (EDGE + max(widest(fig, [shown[name]], fontsize=size, fontweight=weight[name]) for name in shown)
            + LABEL_GAP + arrow)
    fig.set_size_inches(COLUMN, height)
    growths = [min(1 + extra / min(rows), GROWTH) for rows in heights]
    # each mark clears the axis's ends by its half width, its ring, an arrowhead and a pad
    clears = [(mark * growth / 2 + RING * growth / 2) / 72 + arrow + POSTER_PAD for mark, growth in zip(marks, growths)]
    width = plot_width(fig, panels, clears, COLUMN - left, arrow)
    top = height - EDGE
    for panel, mark, rows, key, key_rows, bottom, growth, clear in zip(panels, marks, heights, keys, lines, bottoms,
                                                                       growths, clears):
        mark, bar, ring = mark * growth, POSTER_BAR * growth, RING * growth
        span = sum(rows) + extra * len(rows)
        # the key over the panel from the canvas's left edge, a line at a time: each series' mark, then its name
        for index, row in enumerate(key_rows):
            middle = top - index * key / len(key_rows) - (key / len(key_rows) - ROW_GAP) / 2
            for x, label, look in row:
                if look:
                    colour, marker, filled = look
                    fig.add_artist(Line2D([x + mark / 144], [middle], transform=fig.dpi_scale_trans, marker=marker,
                                          markersize=mark, color=colour, markerfacecolor=colour if filled else "white",
                                          markeredgecolor=colour, markeredgewidth=0 if filled else ring))
                    x += mark / 72 + POSTER_PAD
                fig.text(x / COLUMN, middle / height, label, ha="left", va="center")
        top -= key
        # the axes' y is inches down from its top, each row centred in its height
        ax = frame(fig, left, top - span, width, span)
        centres = np.cumsum([0.0, *(row + extra for row in rows)])[:-1] + (np.array(rows) + extra) / 2
        low_limit, high_limit = limits(panel, width, clear)
        if panel.log:
            ax.set_xscale("log")
        for (name, estimates), centre in zip(panel.rows, centres):
            for index, ((label, colour, marker, filled), (value, low, high)) in enumerate(zip(panel.series, estimates)):
                y = centre + (index - (len(panel.series) - 1) / 2) * (mark / 72 + (0.0 if filled else ring / 72)
                                                                      + MARK_GAP)
                ax.hlines(y, max(low, low_limit), min(high, high_limit), color=colour, linewidth=bar, zorder=3,
                          capstyle="round")
                for end, edge, arrowhead in ((low, low_limit, "<"), (high, high_limit, ">")):
                    if (end < edge) if arrowhead == "<" else (end > edge):
                        ax.plot([edge], [y], marker=arrowhead, markersize=mark * 0.7, color=colour, markeredgewidth=0,
                                zorder=3, clip_on=False)
                ax.plot([value], [y], marker, markersize=mark, color=colour,
                        markerfacecolor=colour if filled else "white", markeredgecolor=colour,
                        markeredgewidth=0 if filled else ring, zorder=4)
                if (name, index) in panel.starred:
                    # past the interval's drawn end, or its arrowhead, its ink centred on the interval
                    past = STAR_GAP + (arrow if high > high_limit else 0.0)
                    ax.text(min(high, high_limit), y, STAR, ha="left", va="baseline", color=INK,
                            transform=ax.transData + ScaledTranslation(past, -star_middle(size) / 72,
                                                                       fig.dpi_scale_trans))
            ax.text(0, centre, shown[name], transform=shifted(ax, EDGE - left), ha="left", va="center",
                    fontweight=weight[name])
        if not panel.log:
            ax.axvline(0.0, color=INK, linewidth=4.0, zorder=2)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_ylim(span, 0)
        ax.set_xlim(low_limit, high_limit)
        drawn = inside(panel, low_limit, high_limit)
        ax.set_xticks(drawn, [tick_text(panel)(tick) for tick in drawn])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.tick_params(axis="x", which="major", pad=style["xtick.major.pad"])
        ax.set_xlabel(panel.xlabel, labelpad=TITLE_PAD)
        # centred on the canvas, not on the axes, whose margins differ: a title wider than the axes then fits either side
        ax.xaxis.label.set_x((COLUMN / 2 - left) / width)
        top -= span + bottom
    return fig


def key_lines(fig, entries: list[tuple[str, tuple | None]], mark: float) -> list[list[tuple[float, str, tuple | None]]]:
    """A key's entries (label, its mark's (colour, marker, filled) or None) set left to right from EDGE, each at its x
    in inches, a new line begun where the next entry would pass the column's right edge."""
    size = matplotlib.rcParams["font.size"]
    rows, x = [[]], EDGE
    for label, look in entries:
        across = (mark / 72 + POSTER_PAD if look else 0.0) + widest(fig, [label], fontsize=size)
        if rows[-1] and x + across > COLUMN - EDGE:
            rows, x = [*rows, []], EDGE
        rows[-1].append((x, label, look))
        x += across + 4 * POSTER_PAD
    return rows if rows[0] else []


def star_middle(size: float) -> float:
    """Points from the baseline up to the middle of STAR's ink at ``size`` points."""
    extent = TextPath((0, 0), STAR, size=size, prop=FontProperties(size=size)).get_extents()
    return (extent.y0 + extent.y1) / 2


def plot_width(fig, panels: list[Panel], clears: list[float], room: float, arrow: float) -> float:
    """Inches of the panels' shared x axis in ``room`` inches right of the names: as wide as leaves EDGE past the
    farthest of an arrowhead at the axis's end, the half of each drawn tick label that sits past it, and a STAR past
    its interval's drawn end. Only drawn ticks count, so a decade outside a log axis takes no room. A narrower axis can
    draw another tick, so it narrows until its own ticks fit."""
    size = matplotlib.rcParams["xtick.labelsize"]
    star = widest(fig, [STAR], fontsize=matplotlib.rcParams["font.size"])
    width = room - arrow - EDGE
    for _ in range(20):
        reach = arrow
        for panel, clear in zip(panels, clears):
            low, high = limits(panel, width, clear)
            place = (lambda value: math.log10(value / low) / math.log10(high / low)) if panel.log \
                else (lambda value: (value - low) / (high - low))
            for tick in inside(panel, low, high):
                reach = max(reach, width * (place(tick) - 1) + widest(fig, [tick_text(panel)(tick)], fontsize=size) / 2)
            for name, estimates in panel.rows:
                for index, (_, _, end) in enumerate(estimates):
                    if (name, index) in panel.starred:
                        reach = max(reach, width * (place(min(end, high)) - 1) + (arrow if end > high else 0.0)
                                    + STAR_GAP + star)
        if width + reach + EDGE <= room + 1e-9:
            return width
        width = room - reach - EDGE
    raise SystemExit("the forest's x axis found no width whose tick labels fit the canvas")


def inside(panel: Panel, low: float, high: float) -> list[float]:
    """The panel's ticks between the axis's ends ``low`` and ``high``."""
    slack = 1e-9 * (high - low)
    return [tick for tick in ticks(panel) if low - slack <= tick <= high + slack]


def ticks(panel: Panel) -> list[float]:
    """A log panel's decades from two below its smallest estimate to two above its largest, the most its axis spans; a
    linear panel's span."""
    values = [value for _, estimates in panel.rows for value, _, _ in estimates]
    if panel.log:
        return [10.0 ** power for power in range(math.floor(math.log10(min(values))) - 1,
                                                 math.ceil(math.log10(max(values))) + 2)]
    return span(values)


def tick_text(panel: Panel):
    """A log panel's ticks as plain decimals (its values are already percentages: 0.1, 1, 10), a linear panel's in
    percent."""
    return (lambda tick: f"{tick:g}") if panel.log else percent


def limits(panel: Panel, width: float, clear: float) -> tuple[float, float]:
    """The axis's ends: each estimate ``clear`` inches from them at least, over ``width`` inches; on a linear axis also
    symmetric about 0 and wide enough for its ticks, on a log axis its estimates' span, each end that falls within
    DECADE_REACH of a decade moved out to it, so that decade is ticked."""
    values = [value for _, estimates in panel.rows for value, _, _ in estimates]
    if panel.log:
        low, high = math.log10(min(values)), math.log10(max(values))
        margin = (high - low) / (1 - 2 * clear / width) * clear / width
        low, high = low - margin, high + margin
        if low - math.floor(low) < DECADE_REACH:
            low = math.floor(low)
        if math.ceil(high) - high < DECADE_REACH:
            high = math.ceil(high)
        return 10.0 ** low, 10.0 ** high
    reach = max(abs(value) for value in values) / (1 - 2 * clear / width)
    marks = ticks(panel)
    reach = max(reach, -marks[0], marks[-1])
    return -reach, reach


DECADE_REACH = 0.25
"""Decades a log axis's end moves out at most, to reach a decade and tick it."""


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


COMBINED = "All traits"
"""The All of Us forest's row for the traits' common g, its name in bold."""


METHODS = [("LDpred2-auto", LDPRED2, "s", True), ("SV-PGS", SV_PGS, "o", True)]
"""The held-out r² panel's series, each method in its colour and marker in the simulation figures."""


def converged(row: dict) -> bool:
    """Whether both SV-PGS fits behind a trait's row converged: neither is marked unconverged or named uncertified."""
    return not row.get("unconverged") and "uncertified" not in row.get("flags", "")


def order(rows: list, compared: list[dict] | None) -> list:
    """``rows`` (name, marks) in the order of ``compared``'s traits, then the others in their own order."""
    place = {trait_name(row["trait"]): index for index, row in enumerate(compared or [])}
    return sorted(rows, key=lambda row: place.get(row[0], len(place)))


def significant(row: dict, test: str) -> bool:
    """Whether aou_vs_ldpred2.json's ``row`` gives ``test`` ("with_sv": SV-PGS with SVs against LDpred2-auto, "sv_gain":
    the change with SVs) a Benjamini-Hochberg q below Q."""
    q = (row.get(test) or {}).get("q")
    return q is not None and q < Q


def gain_figure(matched: list[dict], common: dict | None = None, compared: list[dict] | None = None) -> tuple:
    """Each trait's change in held-out partial r² from adding SVs, relative to the r² without SVs (baseline_relative of
    the pre-specified g), with its 95% interval, a row per trait, and when ``common`` is given a bold row for the traits'
    common g itself, relative to the two arms' mean r². The two scales differ by about 0.1 percentage point at these
    gains, so one axis title serves both. Over it, when ``compared`` (aou_vs_ldpred2.json's rows) gives the traits'
    held-out r², a panel of that r² in percent for LDpred2-auto and for SV-PGS with SVs on a log axis, and its q values
    star the SV-PGS intervals whose q is below Q in either panel (``significant``). Both panels list
    the traits in ``compared``'s order, then any other trait, then the common row. No figure on the poster uses a fit
    that did not converge, so a trait whose SV-PGS fit did not converge is in neither panel."""
    keys = ("relative", "relative_lower", "relative_upper")
    dropped = {trait_name(row["trait"]) for row in matched if not converged(row)}
    rows = order([(trait_name(row["trait"]), [tuple(baseline_relative(row[key] / 100) for key in keys)])
                  for row in matched if converged(row)], compared)
    if common:
        rows.append((COMBINED, [tuple(common[key] / 100 for key in keys)]))
    gained = {trait_name(row["trait"]) for row in compared or [] if significant(row, "sv_gain")}
    panels = [Panel(rows, [("", SKY, "o", False)], "Change in r² with SVs (%)",
                    starred={(name, 0) for name, _ in rows if name in gained})] if rows else []
    measured = [row for row in compared or [] if "r2" in row and trait_name(row["trait"]) not in dropped]
    if measured:
        panels.insert(0, Panel([(trait_name(row["trait"]), [tuple(100 * row["r2"][arm][key] for key in ("r2", "lower", "upper"))
                                                             for arm in ("ldpred2_auto", "with_sv")])
                                for row in measured], METHODS, "Held-out r² (%)", log=True,
                               starred={(trait_name(row["trait"]), 1) for row in measured if significant(row, "with_sv")}))
    stars = any(panel.starred for panel in panels)
    fig = forest(panels, least=HEIGHT)
    return fig, [("Held-out partial r² in percent of LDpred2-auto and SV-PGS for each trait on a log axis, and under it "
                  "the change " if measured
                  else "The change ")
                 + "in held-out partial r² from adding SVs to SV-PGS for each trait, relative to the r² without SVs, "
                 "with 95% intervals, an arrowhead marking an interval that runs past the axis"
                 + (", and in bold the traits' common change relative to the two models' mean r²" if common else "")
                 + (", an asterisk marking a Benjamini-Hochberg q below 0.05 for SV-PGS against LDpred2-auto (top) or "
                    "for the change with SVs (bottom)" if stars else "")]


def main() -> None:
    numbers = json.loads(NUMBERS.read_text())
    with plt.rc_context({**POSTER_STYLE, "axes.labelsize": POSTER_LABEL}):
        drawn = []
        if numbers.get("matched", {}).get("rows"):
            compared = json.loads(COMPARED.read_text())["rows"] if COMPARED.is_file() else None
            drawn.append(("aou_1_gain", gain_figure(numbers["matched"]["rows"], numbers["matched"].get("common"),
                                                    compared)))
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

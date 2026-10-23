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

from sim_gallery import (INK, LDPRED2, POSTER_BAR, POSTER_LABEL, POSTER_MARK, POSTER_PAD, POSTER_STYLE, SKY, SV_PGS, frame,
                         overhang, plt, shifted, tallest, widest)

HERE = Path(__file__).resolve().parent
NUMBERS = HERE / "aou_numbers.json"
COMPARED = HERE / "aou_vs_ldpred2.json"
ANCESTRY = HERE / "aou_ancestry.json"
OUT = HERE / "figures"
COLUMN = 11.4
"""Inches: the poster's right column (build_pptx.py's RIGHT), each figure's width, so its type prints at its own size."""
ROW_GAP = 0.03
"""Inches at least between the marks of one row and the next: a row is a line of poster type (its size in points
over 72 inches), or its marks and this gap when they are taller."""
AXIS_GAP = 0.375
"""Inches between a panel's last row and its x axis, about half a line of type, when that row is a note without marks,
whose type would otherwise sit on the axis."""
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
"""Inches between a mark's drawn end (its interval, arrowhead or mark) and its asterisk."""
NOTE_GAP = 0.2
"""Inches between a row's marks, or the widest name of the panel's rows with notes, and a note."""
MIN_AXIS = 3.5
"""Inches: the narrowest x axis a panel's notes may leave by following their rows' marks; below it they follow the
names instead."""
LABEL_GAP = 0.3
"""Inches between the widest row name and the plot."""
MARK = 1.5
"""The forest's marks and intervals relative to the poster's (POSTER_MARK, POSTER_BAR) when its rows are a line of type
apart."""
HEIGHT = 19.90
"""Inches: the All of Us figure's height, the room the poster's right column gives it (build_pptx.py's layout). What it
has beyond the figure's natural height goes between the panels; the rows stay a line of type apart."""
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
    index) in ``starred`` has STAR past its interval's end. A row's entry in ``notes`` is a note in the figure's type,
    as the pieces it may break between. In a row with marks it follows them on one line, at the row's end, unless the
    axis would then be narrower than MIN_AXIS; then it starts in a column NOTE_GAP right of the widest name among the
    panel's rows with notes. In a row without marks it starts in that column, wrapped onto further lines there."""

    rows: list[tuple[str, list[tuple[float, float, float]]]]
    series: list[tuple[str, str, str, bool]]
    xlabel: str
    log: bool = False
    starred: set[tuple[str, int]] = field(default_factory=set)
    notes: dict[str, list[str]] = field(default_factory=dict)


def forest(panels: list[Panel], least: float = 0.0):
    """The panels stacked at the column's width, their rows' names at the left in a margin they share, COMBINED's in
    bold. In each row the series' marks stand one above the next, each an estimate with its 95% interval, and a panel
    with more than one series has their key over it, wrapped onto further lines when it is wider than the column. When
    a panel stars a mark, STAR_NOTE ends the first panel's key, which it starts when that panel has none. A row's note
    follows its marks or its name (Panel). A log panel spans its estimates and is ticked at the
    decades inside; a linear one is symmetric about 0 over its estimates. Either way each mark clears an arrowhead at
    the axis's end, and an interval past the axis ends in one, its tip at the axis's end. The rows are a line of type
    apart, and the axis is under the last of them, AXIS_GAP under it when it is a note without marks. Every margin and row height is measured from the type,
    so the figure is as short as its type allows, unless ``least`` inches is taller: then the extra height goes
    between the panels, or over a lone panel."""
    fig = plt.figure(figsize=(COLUMN, 1.0))
    size = POSTER_STYLE["font.size"]
    shown = {name: name for panel in panels for name, _ in panel.rows}
    weight = {name: "bold" if name == COMBINED else "normal" for name in shown}
    named = {name: widest(fig, [name], fontsize=size, fontweight=weight[name]) for name in shown}
    # one series' marks at MARK times the poster's, two or more at the poster's own, one above the next in a row
    marks = [POSTER_MARK * (MARK if len(panel.series) == 1 else 1.0) for panel in panels]
    # a row is a line of type, or as many lines as its name or note takes, or its marks one above the next when taller
    line = size / 72
    stacks = [sum(mark / 72 + (0.0 if filled else RING / 72) for *_, filled in panel.series)
              + (len(panel.series) - 1) * MARK_GAP for panel, mark in zip(panels, marks)]
    # each panel's notes start in a column right of the widest name among its rows with notes
    columns = [max([EDGE + named[name] + NOTE_GAP for name, _ in panel.rows if name in panel.notes], default=EDGE)
               for panel in panels]
    wrapped = [{name: note_lines(fig, panel.notes[name], COLUMN - EDGE - column, COLUMN - EDGE - column)
                for name, estimates in panel.rows if not estimates and name in panel.notes}
               for panel, column in zip(panels, columns)]
    heights = [[len(notes[name]) * line if name in notes else max((name.count("\n") + 1) * line, stack + ROW_GAP)
                for name, _ in panel.rows] for panel, stack, notes in zip(panels, stacks, wrapped)]
    # each panel's key: its series, each a mark and its name, and after the first panel's STAR_NOTE when a mark is starred
    entries = [[(label, (colour, marker, filled)) for label, colour, marker, filled in panel.series]
               if len(panel.series) > 1 else [] for panel in panels]
    if any(panel.starred for panel in panels):
        entries[0].append((STAR_NOTE, None))
    lines = [key_lines(fig, entry, mark) for entry, mark in zip(entries, marks)]
    keys = [len(rows) * max(line, mark / 72 + ROW_GAP) for rows, mark in zip(lines, marks)]
    # under each axis: its ticks, tick labels and title, and then POSTER_PAD to the next panel or EDGE to the canvas's foot
    style = matplotlib.rcParams
    bottoms = [(style["xtick.major.size"] + style["xtick.major.pad"] + TITLE_PAD) / 72
               + tallest(fig, ["0.1"], fontsize=style["xtick.labelsize"])
               + tallest(fig, [panel.xlabel], fontsize=style["axes.labelsize"]) + POSTER_PAD for panel in panels]
    bottoms[-1] += EDGE - POSTER_PAD
    gaps = [AXIS_GAP if panel.notes else 0.0 for panel in panels]
    natural = EDGE + sum(key + sum(rows) + gap + bottom for key, rows, gap, bottom in zip(keys, heights, gaps, bottoms))
    height = max(least, natural)
    spare = height - natural
    # an arrowhead at the axis's start reaches half its width to the left, toward the names
    # an arrowhead ends at the axis's end, so it reaches its full length, twice this, inside the axis
    arrow = max(marks) * 0.7 / 2 / 72
    left = EDGE + max(named.values()) + LABEL_GAP
    fig.set_size_inches(COLUMN, height)
    # each mark clears the axis's ends by its half width, its ring, an arrowhead and a pad
    halves = [(mark + RING) / 2 / 72 for mark in marks]
    clears = [half + 2 * arrow + POSTER_PAD for half in halves]
    # the notes of rows with marks at the rows' ends, or, when that leaves too narrow an axis, in the notes' column
    width = plot_width(fig, panels, clears, halves, COLUMN - left, ends_noted=True)
    ending = width >= MIN_AXIS
    if not ending:
        left = max(left, *(column + widest(fig, [" ".join(panel.notes[name])], fontsize=size) + LABEL_GAP
                           for panel, column in zip(panels, columns) for name, estimates in panel.rows
                           if estimates and name in panel.notes))
        width = plot_width(fig, panels, clears, halves, COLUMN - left, ends_noted=False)
    star = widest(fig, [STAR], fontsize=size)
    top = height - EDGE
    top -= spare if len(panels) == 1 else 0.0
    for panel, mark, rows, key, key_rows, bottom, clear, half, notes, column, gap in zip(
            panels, marks, heights, keys, lines, bottoms, clears, halves, wrapped, columns, gaps):
        bar, ring = POSTER_BAR, RING
        span = sum(rows) + gap
        # the key over the panel from the canvas's left edge, a line at a time: each series' mark, then its name
        for index, row in enumerate(key_rows):
            middle = top - (index + 0.5) * key / len(key_rows)
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
        centres = np.cumsum([0.0, *rows])[:-1] + np.array(rows) / 2
        low_limit, high_limit = limits(panel, width, clear)
        if panel.log:
            ax.set_xscale("log")
        # the notes of rows with marks start at one x, NOTE_GAP past the farthest of those rows' ends
        aligned = max([ends(panel, name, estimates, width, low_limit, high_limit, half, star)[1]
                       for name, estimates in panel.rows if estimates and name in panel.notes], default=0.0) + NOTE_GAP
        for (name, estimates), centre in zip(panel.rows, centres):
            each, _ = ends(panel, name, estimates, width, low_limit, high_limit, half, star)
            for index, ((label, colour, marker, filled), (value, low, high)) in enumerate(zip(panel.series, estimates)):
                y = centre + (index - (len(panel.series) - 1) / 2) * (mark / 72 + (0.0 if filled else ring / 72)
                                                                      + MARK_GAP)
                ax.hlines(y, max(low, low_limit), min(high, high_limit), color=colour, linewidth=bar, zorder=3,
                          capstyle="round")
                # an interval past the axis ends in an arrowhead whose tip is the axis's end
                for end, edge, arrowhead, side in ((low, low_limit, "<", 0), (high, high_limit, ">", 1)):
                    if (end < edge) if arrowhead == "<" else (end > edge):
                        ax.plot([side], [y], marker=arrowhead, markersize=mark * 0.7, color=colour, markeredgewidth=0,
                                zorder=3, clip_on=False, transform=shifted(ax, (0.5 - side) * mark * 0.7 / 72))
                ax.plot([value], [y], marker, markersize=mark, color=colour,
                        markerfacecolor=colour if filled else "white", markeredgecolor=colour,
                        markeredgewidth=0 if filled else ring, zorder=4)
                if (name, index) in panel.starred:
                    # past the interval's drawn end, its arrowhead or its mark, its ink centred on the interval
                    ax.text(0, y, STAR, ha="left", va="baseline", color=INK, gid="star",
                            transform=shifted(ax, each[index] + STAR_GAP, -star_middle(size) / 72))
            if name in notes:
                # the note's lines in the notes' column, the name on its first line
                first = centre - len(notes[name]) * line / 2 + line / 2
                for index, text in enumerate(notes[name]):
                    ax.text(0, first + index * line, text, transform=shifted(ax, column - left), ha="left",
                            va="center", gid="note")
                centre = first
            elif name in panel.notes:
                ax.text(0, centre, " ".join(panel.notes[name]), ha="left", va="center", gid="note",
                        transform=shifted(ax, aligned if ending else column - left))
            ax.text(0, centre, shown[name], transform=shifted(ax, EDGE - left), ha="left", va="center",
                    fontweight=weight[name], gid="name")
        if not panel.log:
            ax.axvline(0.0, color=INK, linewidth=4.0, zorder=2)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_ylim(span, 0)
        ax.set_xlim(low_limit, high_limit)
        drawn = drawn_ticks(fig, panel, width, low_limit, high_limit)
        ax.set_xticks(drawn, [tick_text(panel)(tick) for tick in drawn])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.tick_params(axis="x", which="major", pad=style["xtick.major.pad"])
        ax.set_xlabel(panel.xlabel, labelpad=TITLE_PAD)
        # centred on the canvas, not on the axes, whose margins differ: a title wider than the axes then fits either side
        ax.xaxis.label.set_x((COLUMN / 2 - left) / width)
        top -= span + bottom + (spare / (len(panels) - 1) if len(panels) > 1 else 0.0)
    return fig


def note_lines(fig, pieces: list[str], first: float, rest: float) -> list[str]:
    """``pieces`` joined by spaces into lines, the first at most ``first`` inches wide and each later one ``rest``; a
    piece wider than its line starts a line of its own."""
    size = matplotlib.rcParams["font.size"]
    lines = [""]
    for piece in pieces:
        joined = f"{lines[-1]} {piece}" if lines[-1] else piece
        if lines[-1] and widest(fig, [joined], fontsize=size) > (first if len(lines) == 1 else rest):
            lines.append(piece)
        else:
            lines[-1] = joined
    return lines


def ends(panel: Panel, name: str, estimates: list[tuple[float, float, float]], width: float, low: float, high: float,
         half: float, star: float) -> tuple[list[float], float]:
    """Inches from the axis's left at which each of a row's marks ends, the farther of its interval (the axis's end for
    an interval past it, whose arrowhead ends there) and the mark itself; and at which the row ends, its STARs after
    them included."""
    place = (lambda value: math.log10(value / low) / math.log10(high / low)) if panel.log \
        else (lambda value: (value - low) / (high - low))
    each = [max(width * place(min(upper, high)), width * place(value) + half) for value, _, upper in estimates]
    past = max([end + (STAR_GAP + star if (name, index) in panel.starred else 0.0) for index, end in enumerate(each)],
               default=0.0)
    return each, past


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


def plot_width(fig, panels: list[Panel], clears: list[float], halves: list[float], room: float,
               ends_noted: bool) -> float:
    """Inches of the panels' shared x axis in ``room`` inches right of the names: as wide as leaves EDGE past the
    farthest of the axis's end, the half of each drawn tick label that sits past it, and each row's
    STARs past its marks, and its note too when ``ends_noted`` (``ends``; ``halves``: each panel's marks' half
    widths). Only drawn ticks count, so a
    decade outside a log axis takes no room. The widest axis that fits is found by bisection."""
    size = matplotlib.rcParams["xtick.labelsize"]
    star = widest(fig, [STAR], fontsize=matplotlib.rcParams["font.size"])

    def fits(width: float) -> bool:
        if width <= 2 * max(clears) + POSTER_PAD:
            return False  # no room for the marks inside their clearances
        reach = 0.0
        for panel, clear, half in zip(panels, clears, halves):
            low, high = limits(panel, width, clear)
            place = (lambda value: math.log10(value / low) / math.log10(high / low)) if panel.log \
                else (lambda value: (value - low) / (high - low))
            for tick in drawn_ticks(fig, panel, width, low, high):
                reach = max(reach, width * (place(tick) - 1) + widest(fig, [tick_text(panel)(tick)], fontsize=size) / 2)
            # the notes start at one x past the farthest end of their rows, so the widest note counts from there
            noted, note = 0.0, 0.0
            for name, estimates in panel.rows:
                if not estimates:
                    continue
                _, past = ends(panel, name, estimates, width, low, high, half, star)
                reach = max(reach, past - width)
                if ends_noted and name in panel.notes:
                    noted = max(noted, past)
                    note = max(note, NOTE_GAP + widest(fig, [" ".join(panel.notes[name])],
                                                       fontsize=matplotlib.rcParams["font.size"]))
            reach = max(reach, noted + note - width) if note else reach
        return width + reach + EDGE <= room

    low, high = 2 * max(clears) + 2 * POSTER_PAD, room - EDGE
    if fits(high):
        return high
    if not fits(low):
        raise SystemExit("the forest's x axis found no width whose tick labels, stars and notes fit the canvas")
    for _ in range(40):
        middle = (low + high) / 2
        low, high = (middle, high) if fits(middle) else (low, middle)
    return low


def drawn_ticks(fig, panel: Panel, width: float, low: float, high: float) -> list[float]:
    """The panel's ticks inside its axis (``inside``) over ``width`` inches, less an end tick while two neighbours'
    labels come closer than TICK_SPACE inches: of the two ends, the one farther outside the estimates goes first."""
    size = matplotlib.rcParams["xtick.labelsize"]
    place = (lambda value: math.log10(value / low) / math.log10(high / low)) if panel.log \
        else (lambda value: (value - low) / (high - low))
    values = [value for _, estimates in panel.rows for value, _, _ in estimates]
    scale = (lambda value: math.log10(value)) if panel.log else (lambda value: value)
    beyond = lambda tick: max(scale(min(values)) - scale(tick), scale(tick) - scale(max(values)), 0.0)
    kept = inside(panel, low, high)
    while len(kept) > 1 and any(width * (place(right) - place(left)) - (widest(fig, [tick_text(panel)(left)], fontsize=size)
                                                                       + widest(fig, [tick_text(panel)(right)], fontsize=size)) / 2
                                < TICK_SPACE for left, right in zip(kept, kept[1:])):
        kept = kept[1:] if beyond(kept[0]) >= beyond(kept[-1]) else kept[:-1]
    return kept


TICK_SPACE = 0.3
"""Inches at least between neighbouring tick labels."""


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


COMBINED = "Overall"
"""The name of each All of Us panel's pooled row, in bold; set here, whatever name the json files give the pooled
entries."""


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


def p_text(p: float) -> str:
    """A p value as the figure writes it: to one significant figure below 0.01, else to two decimals."""
    if 0 < p < 0.01:
        digits = -math.floor(math.log10(p))
        if round(p, digits) < 0.01:
            return f"{p:.{digits}f}"
    return f"{p:.2f}"


def signed(percent_value: float) -> str:
    """A percentage rounded to a whole percent with its sign, "+17%" or "−3%"."""
    return f"{round(percent_value):+d}%".replace("-", "−")


GROUPS = ("EUR", "AFR", "AMR")
"""The ancestry groups under each pooled row, in this order, from ANCESTRY's "groups"."""


def pooled_rows(overall: dict | None, groups: dict | None, test: str) -> tuple[list, dict]:
    """The pooled rows of a percent panel, COMBINED from ``overall`` (relative, relative_lower, relative_upper, p) and
    then each of GROUPS that ``groups`` (aou_ancestry.json's "groups") gives for ``test`` ("sv_gain" or "vs_ldpred2":
    percent, lower, upper, p_one), each mark a fraction for the percent ticks; and each row's one-sided p as its note."""
    rows, notes = [], {}
    if overall:
        rows.append((COMBINED, [tuple(overall[key] / 100 for key in ("relative", "relative_lower", "relative_upper"))]))
        notes[COMBINED] = [f"p = {p_text(overall['p'])}"]
    for group in GROUPS:
        values = ((groups or {}).get(group) or {}).get(test) or {}
        if all(key in values for key in ("percent", "lower", "upper", "p_one")):
            rows.append((group, [tuple(values[key] / 100 for key in ("percent", "lower", "upper"))]))
            notes[group] = [f"p = {p_text(values['p_one'])}"]
    return rows, notes


def gain_figure(matched: list[dict], common: dict | None = None, compared: list[dict] | None = None,
                pooled: dict | None = None, groups: dict | None = None) -> tuple:
    """Up to three panels, stacked. First, when ``compared`` (aou_vs_ldpred2.json's rows) gives the traits' held-out
    r², that r² in percent for LDpred2-auto and for SV-PGS with SVs on a log axis, a row per trait. Second, SV-PGS's
    pooled change in r² over LDpred2-auto's: COMBINED from ``pooled`` (aou_vs_ldpred2.json's "combined") and each of
    GROUPS from ``groups`` (aou_ancestry.json's "groups"), each with its 95% interval and its one-sided p at the row's
    end; pooled, so a relative axis. Third, each trait's change in held-out partial r² from adding SVs, relative to the
    r² without SVs (baseline_relative of the pre-specified g), with its 95% interval, then COMBINED, the traits' common
    g relative to the two arms' mean r² from ``common`` with its one-sided swap p (p_sharp), and each of GROUPS with its
    one-sided p. The two scales differ by about 0.1 percentage point at these gains, so one axis title serves both.
    The traits are in ``compared``'s order, then any other trait. Q values star the SV-PGS intervals whose q is below Q
    (``significant``). No figure on the poster uses a fit that did not converge, so a trait whose SV-PGS fit did not
    converge is in no panel."""
    keys = ("relative", "relative_lower", "relative_upper")
    dropped = {trait_name(row["trait"]) for row in matched if not converged(row)}
    panels = []
    measured = [row for row in compared or [] if "r2" in row and trait_name(row["trait"]) not in dropped]
    if measured:
        r2 = [(trait_name(row["trait"]), [tuple(100 * row["r2"][arm][key] for key in ("r2", "lower", "upper"))
                                          for arm in ("ldpred2_auto", "with_sv")]) for row in measured]
        panels.append(Panel(r2, METHODS, "Held-out r² (%)", log=True,
                            starred={(trait_name(row["trait"]), 1) for row in measured if significant(row, "with_sv")}))
    against = (pooled or {}).get("with_sv") or {}
    overall = ({key: against[key] for key in keys} | {"p": against["p_one"]}
               if all(key in against for key in (*keys, "p_one")) else None)
    versus, versus_notes = pooled_rows(overall, groups, "vs_ldpred2")
    if versus:
        panels.append(Panel(versus, [("", SV_PGS, "o", True)], "SV-PGS r² vs LDpred2-auto (%)", notes=versus_notes))
    rows = order([(trait_name(row["trait"]), [tuple(baseline_relative(row[key] / 100) for key in keys)])
                  for row in matched if converged(row)], compared)
    common_p = ({key: common[key] for key in keys} | {"p": common["p_sharp"]}
                if common and common.get("p_sharp") is not None else None)
    combined, gain_notes = pooled_rows(common_p, groups, "sv_gain")
    gained = {trait_name(row["trait"]) for row in compared or [] if significant(row, "sv_gain")}
    if rows:
        panels.append(Panel(rows + combined, [("", SKY, "o", False)], "Change in r² with SVs (%)",
                            starred={(name, 0) for name, _ in rows if name in gained}, notes=gain_notes))
    stars = any(panel.starred for panel in panels)
    fig = forest(panels, least=HEIGHT)
    return fig, [("Held-out partial r² in percent of LDpred2-auto and SV-PGS for each trait on a log axis; " if measured
                  else "")
                 + ("SV-PGS's pooled change in r² over LDpred2-auto, overall and by ancestry group, with one-sided p; "
                    if versus else "")
                 + "the change in held-out partial r² from adding SVs to SV-PGS for each trait, relative to the r² "
                 "without SVs, with 95% intervals, an arrowhead marking an interval that runs past the axis"
                 + (", and the pooled change overall and by ancestry group with one-sided p" if combined else "")
                 + (", an asterisk marking a Benjamini-Hochberg q below 0.05" if stars else "")]


def main() -> None:
    numbers = json.loads(NUMBERS.read_text())
    with plt.rc_context({**POSTER_STYLE, "axes.labelsize": POSTER_LABEL}):
        drawn = []
        if numbers.get("matched", {}).get("rows"):
            compared = json.loads(COMPARED.read_text()) if COMPARED.is_file() else {}
            ancestry = json.loads(ANCESTRY.read_text()) if ANCESTRY.is_file() else {}
            drawn.append(("aou_1_gain", gain_figure(numbers["matched"]["rows"], numbers["matched"].get("common"),
                                                    compared.get("rows"), compared.get("combined"),
                                                    ancestry.get("groups"))))
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

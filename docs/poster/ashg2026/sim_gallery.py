"""Figures of the simulation study run on MSI (benchmarks/power/grid.py on bench-sim v7's 1000 Genomes chr22 cohort,
Beagle-imputed arm), drawn for the poster and fit for print. Nothing here is from All of Us.

Every value is read from the replicate results copied from MSI (simulation.py's LOCAL tree: grid, grid-truth, window,
window-truth), and from simulation_summary.json for the AUC at full h² and the imputation accuracy bins.

    uv run --with matplotlib python docs/poster/ashg2026/sim_gallery.py [output directory]

Each figure is drawn at the width the poster places it (WIDE, the middle column of build_pptx.py's grid, or HALF of it)
in poster type: TYPE for ticks, legends and row labels, LABEL for axis titles and TITLE for panel titles, so a figure
reduced to a journal's 183 mm keeps 8-11 pt type. Every panel carries a title, every estimate its 95% interval, and
each method or genotype source one Okabe-Ito colour in every figure: SV-PGS blue, LDpred2-auto vermillion, imputed SV
genotypes sky blue. Each figure is written as a vector PDF and a 300 dpi PNG; gallery.pdf holds every figure on its own
page above its legend, and numbers.json the values the figures and legends show, which the poster's text quotes
(build_pptx.py). The poster also takes poster_accuracy_by_training_size.png (Figure 1 at the poster's height) and
poster_imputation_accuracy.png (Figure 5a alone, at an outer column's width).
"""
from __future__ import annotations

import glob
import json
import math
import re
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation, blended_transform_factory
from scipy import stats

HERE = Path(__file__).resolve().parent
LOCAL = Path.home() / "svpgs-bench" / "poster-grid2"
SUMMARY = HERE / "simulation_summary.json"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "figures" / "gallery"
SIZES = (5_000, 10_000, 20_000, 40_000)
PEOPLE = 50_000
"""People in the simulated cohort (bench-sim v7): the training people and the 10,000 held out."""

WIDE = 22.32
"""Inches: the poster's middle column (build_pptx.py's MIDDLE, 22.92) less its 0.3 in INSET each side, the width a
figure that spans it is placed at. build_pptx.py trims each figure to its ink, so a poster figure's ink spans nearly its
whole width, and its type prints at its own size."""
HALF = 10.46
"""Inches: half the middle column less its gutter and INSETs, the width each of two figures side by side is placed at."""
OUTER = 10.8
"""Inches: the poster's left column (build_pptx.py's LEFT, 11.4) less its INSETs, where Figure 5a alone goes."""
RING = 1.2
"""Points: the white ring round a filled mark, which parts overlapping marks without breaking a short interval's bar."""
TYPE, LABEL, TITLE = 28, 32, 36
"""Points at print size: tick labels, legends, row labels and notes; axis titles; panel titles. The poster's body text
is 38 pt and its glosses 30 pt (build_pptx.py)."""
LINE, BAR, MARK, DOT = 5.0, 4.0, 17.0, 10.0
"""Points: a data line's width, an interval's, a mean's marker and a replicate's dot."""
PITCH = 0.46
"""Inches between the rows of a forest plot: TYPE plus a sixth."""
PAIRED, SPLIT, PAIR_MARK = 0.42, 0.21, 14.0
"""A forest with two estimates a row (Figures 3 and 4, true and imputed SV genotypes): inches between rows, each
estimate's offset from its row's centre in rows, and its marker in points, so the pair's markers clear each other."""
INDENT = 0.4
"""Inches an architecture's label is indented under its group's heading."""

# Okabe-Ito, the colour-blind-safe palette.
BLUE = "#0072B2"
VERMILLION = "#D55E00"
ORANGE = "#E69F00"
SKY = "#56B4E9"
GREEN = "#009E73"
PURPLE = "#CC79A7"
INK = "#000000"
GREY = "#6E6E6E"
LIGHT = "#BDBDBD"
RULE = "#E4E4E4"
SV_PGS, LDPRED2, IMPUTED = BLUE, VERMILLION, SKY
"""The colours that name a method or a genotype source wherever it appears: SV-PGS (and any SV-PGS gain), LDpred2-auto,
and imputed SV genotypes beside true ones."""


def _font() -> str:
    names = {entry.name for entry in font_manager.fontManager.ttflist}
    return next((name for name in ("Helvetica", "Arial", "Helvetica Neue") if name in names), "DejaVu Sans")


plt.rcParams.update({
    "font.family": _font(),
    "font.size": TYPE,
    "axes.labelsize": LABEL,
    "axes.titlesize": TITLE,
    "xtick.labelsize": TYPE,
    "ytick.labelsize": TYPE,
    "legend.fontsize": TYPE,
    "legend.title_fontsize": TYPE,
    "axes.linewidth": 2.4,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelpad": 12,
    "xtick.major.width": 2.4,
    "ytick.major.width": 2.4,
    "xtick.minor.width": 1.2,
    "ytick.minor.width": 1.2,
    "xtick.major.size": 10,
    "ytick.major.size": 10,
    "xtick.minor.size": 5,
    "ytick.minor.size": 5,
    "xtick.major.pad": 8,
    "ytick.major.pad": 8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.color": INK,
    "ytick.color": INK,
    "lines.linewidth": LINE,
    "lines.markersize": MARK,
    "legend.frameon": False,
    "legend.handlelength": 1.4,
    "legend.borderaxespad": 0.2,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.default": "regular",
})

LABELS = {
    "null": "0% (no causal SVs)",
    "sv_5pct": "5%",
    "sv_20pct": "20%",
    "sv_5pct_fold3": "5% causal SVs, 3× SV effect",
    "sv_20pct_fold3": "20% causal SVs, 3× SV effect",
    "sv_heavy_tail": "Heavy-tailed SV effects",
    "sv_two_point": "Two-point SV effects",
    "length_positive": "Larger effects for longer SVs",
    "deletion_positive": "Deletion effects 3× insertions'",
    "multi_feature_positive": "Length, type and annotation",
    "tagged_sv": "Only well-imputed SVs",
    "tagged_sv_fold3": "Only well-imputed SVs, 3× effect",
    "common_sv": "Only common SVs",
    "sparse_sv": "50 causal variants",
    "polygenic_sv": "2,000 causal variants",
    "low_heritability": "h² = 0.01",
    "rare_disease": "Prevalence 1%",
}
"""Each architecture's name in numbers.json (whose keys build_pptx.py and its tests read) and the legends."""
PLAIN = {"sv_heavy_tail": "A few very large SV effects", "sv_two_point": "SV effects all one size"}
"""The legends' plain words for the architectures whose numbers.json names are technical, as the figures' rows read."""
SHORT = {
    "null": "0%",
    "sv_5pct": "5%",
    "sv_20pct": "20% (baseline)",
    "sv_5pct_fold3": "5% SVs, 3× effect",
    "sv_20pct_fold3": "20% SVs, 3× effect",
    "sv_heavy_tail": "A few very large",
    "sv_two_point": "All one size",
    "length_positive": "Longer SVs larger",
    "deletion_positive": "Deletions 3× larger",
    "multi_feature_positive": "Length, type, genes",
    "tagged_sv": "Well-imputed only",
    "tagged_sv_fold3": "Well-imputed, 3×",
    "common_sv": "Common only",
    "sparse_sv": "50 causal",
    "polygenic_sv": "2,000 causal",
    "low_heritability": "h² = 0.01",
    "rare_disease": "Prevalence 1%",
}
"""Each architecture's row label in a figure, under its group's heading, in plain words and short enough to leave the
plot half the width at poster type. The baseline is the architecture every other one changes one thing of."""
GROUPS = (
    ("Causal SVs", ("null", "sv_5pct", "sv_20pct")),
    ("SV effect sizes", ("sv_5pct_fold3", "sv_20pct_fold3", "sv_heavy_tail", "sv_two_point")),
    ("SV effects by feature", ("length_positive", "deletion_positive", "multi_feature_positive")),
    ("Which SVs are causal", ("tagged_sv", "tagged_sv_fold3", "common_sv")),
    ("Trait", ("sparse_sv", "polygenic_sv", "low_heritability", "rare_disease")),
)
NUMBERS: dict = {}
"""What each figure shows, by figure, for numbers.json: means with Student-t 95% intervals as [mean, lower, upper, n]."""
ARCHITECTURES = (
    "The baseline architecture has liability h² = 0.1, prevalence 10%, 200 causal variants and 20% causal SVs with SV "
    "effects drawn like small-variant effects, and every other changes one thing of it: '3×' triples the SV effect SD, "
    "'a few very large' draws SV effects from a Student t with 3 degrees of freedom, 'all one size' gives every causal "
    "SV the same effect size with a random sign, 'well-imputed' draws causal SVs only from those imputed with r² ≥ 0.8, "
    "and 'common' only from SVs of frequency ≥ 5%."
)


def thousands(count: int) -> str:
    return f"{count // 1000}"


def plain(value: float, digits: int = 2) -> str:
    """A tick ``value`` to ``digits`` places with a true minus and no plus sign, and zero as 0, as every difference axis
    shows it."""
    return "0" if round(value, digits) == 0 else f"{value:.{digits}f}".replace("-", "−")


def sentence(label: str) -> str:
    """An architecture's name inside a sentence: its first letter lower-cased unless its first word is an abbreviation."""
    first = label.split()[0]
    return label if len(first) > 1 and first.isupper() else label[0].lower() + label[1:]


def signed(value: float, digits: int = 1) -> str:
    """``value`` with a sign and a true minus, and no sign on a value that rounds to zero."""
    if round(value, digits) == 0:
        return f"{0:.{digits}f}"
    return f"{value:+.{digits}f}".replace("-", "−")


def interval(values) -> tuple[float, float, float, int]:
    """Mean and its Student-t 95% interval over replicates (no interval below two replicates)."""
    values = np.asarray([value for value in values if value is not None and math.isfinite(value)], float)
    n = values.size
    if n == 0:
        return math.nan, math.nan, math.nan, 0
    mean = float(values.mean())
    if n < 2:
        return mean, math.nan, math.nan, n
    half = float(stats.t.ppf(0.975, n - 1) * values.std(ddof=1) / math.sqrt(n))
    return mean, mean - half, mean + half, n


def median_interval(values) -> tuple[float, float, float, int]:
    """Median and its distribution-free 95% interval: the order statistics whose ranks a Binomial(n, 1/2) puts 2.5%
    beyond on each side (none below six values)."""
    values = np.sort(np.asarray([value for value in values if math.isfinite(value)], float))
    n = values.size
    if n < 6:
        return (float(np.median(values)) if n else math.nan), math.nan, math.nan, n
    rank = max(int(stats.binom.ppf(0.025, n, 0.5)), 1)
    return float(np.median(values)), float(values[rank - 1]), float(values[n - rank]), n


def rounded(value):
    """``value`` with every float to six decimals, for a readable numbers.json."""
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(item) for item in value]
    return value


# ------------------------------------------------------------------ layout in inches


def frame(fig, left: float, bottom: float, width: float, height: float):
    """An axes ``left`` and ``bottom`` inches in from the figure's lower left, ``width`` x ``height`` inches."""
    across, up = fig.get_size_inches()
    return fig.add_axes((left / across, bottom / up, width / across, height / up))


def widest(fig, texts, **style) -> float:
    """Inches the widest of ``texts`` takes when set in ``style`` (fontsize, fontweight)."""
    renderer = fig.canvas.get_renderer()
    widths = [0.0]
    for text in texts:
        artist = fig.text(0, 0, text, **style)
        widths.append(artist.get_window_extent(renderer).width / fig.dpi)
        artist.remove()
    return max(widths)


def shifted(ax, across: float, up: float = 0.0, data_rows: bool = True):
    """Axes x (and data y when ``data_rows``, else axes y) moved ``across`` and ``up`` inches."""
    base = blended_transform_factory(ax.transAxes, ax.transData) if data_rows else ax.transAxes
    return base + ScaledTranslation(across, up, ax.figure.dpi_scale_trans)


def heading(ax, title: str, letter: str = "", across: float = 0.0, up: float = 0.5) -> None:
    """A panel's bold title, starting ``across`` inches right of the axes' left edge (negative: over its row labels)
    and ``up`` inches above its top, after the panel's letter when a figure has several."""
    ax.text(0, 1, f"{letter}   {title}" if letter else title, transform=shifted(ax, across, up, data_rows=False),
            ha="left", va="baseline", fontsize=TITLE, fontweight="bold")


def label_width(fig, rows) -> float:
    """Inches the forest's labels take: the widest group heading, or the widest architecture after its indent."""
    headings = [value for kind, value in rows if kind == "heading"]
    names = [SHORT[value] for kind, value in rows if kind == "point"]
    return max(widest(fig, headings, fontsize=TYPE, fontweight="bold"), INDENT + widest(fig, names, fontsize=TYPE))


def row_labels(ax, rows, start: float) -> None:
    """Group headings in bold and architectures indented under them, left-aligned ``start`` inches left of the axes."""
    for y, (kind, value) in enumerate(rows):
        if kind == "heading":
            ax.text(0, y, value, transform=shifted(ax, -start), ha="left", va="center", fontweight="bold")
        else:
            ax.text(0, y, SHORT[value], transform=shifted(ax, -start + INDENT), ha="left", va="center")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_ylim(len(rows) - 0.5, -0.6)


def estimate(ax, y: float, mean: float, low: float, high: float, color: str, filled: bool = True,
             limits: tuple[float, float] | None = None, size: float = MARK) -> None:
    """A mean at row ``y`` and its 95% interval, the mean filled or open and ``size`` points across; with ``limits``,
    an interval running past them stops at the edge in an arrowhead."""
    if limits is not None:
        for end, edge, marker in ((low, limits[0], "<"), (high, limits[1], ">")):
            if (end < edge) if marker == "<" else (end > edge):
                ax.plot([edge], [y], marker=marker, markersize=size * 0.75, color=color, markeredgewidth=0, zorder=3,
                        clip_on=False)
        low, high = max(low, limits[0]), min(high, limits[1])
    ax.hlines(y, low, high, color=color, linewidth=BAR, zorder=3)
    ax.plot([mean], [y], "o", markersize=size, color=color, markerfacecolor=color if filled else "white",
            markeredgecolor="white" if filled else color, markeredgewidth=RING if filled else BAR, zorder=4)


def zero_line(ax, vertical: bool = True) -> None:
    (ax.axvline if vertical else ax.axhline)(0, color=INK, linewidth=2.4, zorder=1)


def value_grid(ax, axis: str = "x") -> None:
    ax.grid(axis=axis, color=RULE, linewidth=1.6, zorder=0)
    ax.set_axisbelow(True)


def load_results() -> dict[str, list[dict]]:
    grids: dict[str, list[dict]] = defaultdict(list)
    for grid in ("grid", "grid-truth", "window", "window-truth"):
        for path in glob.glob(str(LOCAL / grid / "tasks" / "*" / "n*" / "replicate*" / "result.json")):
            with open(path) as handle:
                grids[grid].append(json.load(handle))
    return grids


def cells(rows: list[dict]) -> dict[tuple[str, int], list[dict]]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["point"], int(row["train_count"]))].append(row)
    return grouped


def finite(rows: list[dict], key: str) -> list[float]:
    return [row[key] for row in rows if row.get(key) is not None and math.isfinite(row[key])]


def layout(present) -> list[tuple[str, str]]:
    """Rows top to bottom: each group's heading, then its architectures that ``present`` keeps."""
    rows = []
    for heading_text, points in GROUPS:
        kept = [point for point in points if present(point)]
        if kept:
            rows.append(("heading", heading_text))
            rows.extend(("point", point) for point in kept)
    return rows


# ------------------------------------------------------------------ Figure 1


SV_SHARES = (("null", "0%", INK, "o", "-"), ("sv_5pct", "5%", "#555555", "s", (0, (3, 1.5))),
             ("sv_20pct", "20%", "#8C8C8C", "D", (0, (1, 1.2))))
"""Figure 1b's settings: the share of causal variants that are SVs, drawn in greys darkest first, each its own marker
and line style (solid, dashed, dotted), so no two are told apart by grey alone."""


def log_size_axis(ax) -> None:
    """Training size on a true log2 axis, ticked at the grid's sizes only, in thousands."""
    ax.set_xscale("log", base=2)
    ax.set_xticks(SIZES, [thousands(size) for size in SIZES])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlim(SIZES[0] / 1.35, SIZES[-1] * 1.35)
    ax.set_xlabel("Training sample size (thousands)")


def accuracy_figure(grids, summary, height: float = 9.6):
    """(a) Test AUC against training size per method, pooled over 0%, 5% and 20% causal SVs; (b) the paired gap between
    the methods for each share of causal SVs, which shows the pooling hides nothing. ``height`` in inches."""
    grouped = cells(grids["grid"])
    fig = plt.figure(figsize=(WIDE, height))
    top, bottom = 1.25, 1.9
    ax = frame(fig, 1.95, bottom, 7.9, height - top - bottom)
    gap = frame(fig, 15.0, bottom, 7.2, height - top - bottom)
    log_size_axis(ax)
    ceiling = summary["points"]["null"]["true_value_auc"]
    setting = summary["points"]["null"]["setting"]
    ax.axhline(ceiling, color=GREY, linestyle=(0, (1, 1.4)), linewidth=3.2, zorder=1)
    ax.text(0.02, ceiling - 0.004, f"max AUC (h² = {setting['h2']:g})", transform=blended_transform_factory(ax.transAxes, ax.transData),
            color=GREY, va="top", ha="left")
    counts = {}
    NUMBERS["fig1"] = {"train_counts": list(SIZES), "test_people": summary["test_people"], "h2": setting["h2"],
                       "prevalence": setting["prevalence"], "causal": setting["causal"], "h2_auc": ceiling}
    for key, method, color in (("with_sv_auc", "SV-PGS", SV_PGS), ("ldpred2_auc", "LDpred2-auto", LDPRED2)):
        pooled = [interval([row[key] for point, *_ in SV_SHARES for row in grouped.get((point, size), [])])
                  for size in SIZES]
        NUMBERS["fig1"][method] = [list(cell) for cell in pooled]
        mean, low, high = (np.array([cell[i] for cell in pooled]) for i in range(3))
        counts[method] = [cell[3] for cell in pooled]
        ax.fill_between(SIZES, low, high, color=color, alpha=0.22, linewidth=0, zorder=2)
        ax.plot(SIZES, mean, color=color, zorder=3, marker="o", markeredgecolor="white", markeredgewidth=2.0)
        ax.text(1.03, mean[-1], method, transform=blended_transform_factory(ax.transAxes, ax.transData), color=color,
                fontweight="bold", va="center", ha="left")
    ax.set_ylim(0.5, 0.69)
    ax.set_yticks(np.arange(0.5, 0.651, 0.05), ["0.50", "0.55", "0.60", "0.65"])
    ax.text(0.98, 0.04, "0/5/20% SVs pooled\nbands: 95% CI", transform=ax.transAxes, ha="right", va="bottom", color=GREY,
            linespacing=1.2)
    ax.set_ylabel("Test AUC")
    value_grid(ax, "y")
    heading(ax, "Test AUC by training size", "a", across=-1.75)

    log_size_axis(gap)
    zero_line(gap, vertical=False)
    for index, (point, share, color, marker, style) in enumerate(SV_SHARES):
        nudge = 2 ** (0.15 * (index - 1))
        stats_by_size = [interval(finite(grouped.get((point, size), []), "delta_auc_vs_ldpred2")) for size in SIZES]
        xs = [size * nudge for size in SIZES]
        gap.vlines(xs, [cell[1] for cell in stats_by_size], [cell[2] for cell in stats_by_size], color=color,
                   linewidth=BAR, zorder=2)
        gap.plot(xs, [cell[0] for cell in stats_by_size], color=color, linewidth=3.4, linestyle=style, marker=marker,
                 markersize=MARK * (0.9 if marker == "D" else 1.0), markeredgecolor="white", markeredgewidth=2.0,
                 zorder=3, label=share)
    gap.set_ylim(-0.006, 0.08)
    gap.set_yticks(np.arange(0, 0.061, 0.02))
    gap.set_ylabel("ΔAUC, SV-PGS − LDpred2-auto")
    value_grid(gap, "y")
    gap.legend(title="Causal SVs; bars: 95% CI", loc="upper left", ncol=3, handlelength=2.2, columnspacing=1.0, handletextpad=0.4,
               alignment="left", borderaxespad=0.0)
    heading(gap, "Same gain with no causal SVs", "b", across=-1.95)
    caption = (
        "Figure 1 | SV-PGS is more accurate than LDpred2-auto at every training size, by the same margin whether 0%, "
        "5% or 20% of the causal variants are SVs, so the gain does not come from causal SVs. "
        "A simulated binary trait on chromosome 22 (liability h² = 0.1, "
        "prevalence 10%, 200 causal variants) scored in the same 10,000 held-out people; SV-PGS fits every SNV, indel "
        "and imputed SV, LDpred2-auto HapMap3-density SNVs of the same training people. a, Test AUC, mean and "
        "Student-t 95% interval over the three settings' replicates pooled ("
        + ", ".join(f"{n}" for n in counts["SV-PGS"])
        + " at 5k-40k); dotted, the largest AUC any score can reach, that of the true genetic value at h² = 0.1 (Wray "
        "et al. 2010). b, Paired difference in AUC on the same held-out people, per setting, mean and Student-t 95% "
        "interval."
    )
    return fig, caption


# ------------------------------------------------------------------ Figure 2


def ldpred2_figure(grids):
    size = SIZES[-1]
    grouped = cells(grids["grid"])
    values = {point: finite(grouped.get((point, size), []), "delta_auc_vs_ldpred2") for point in LABELS}
    rows = layout(lambda point: len(values[point]) >= 3)
    missing = [point for point in LABELS if len(values[point]) < 3]
    # no title: the poster's headline names the figure; the n column's head sits on the first heading's row
    top, bottom, right = 0.35, 1.75, 0.5
    height = top + bottom + PITCH * (len(rows) + 0.1)
    fig = plt.figure(figsize=(HALF, height))
    labels = label_width(fig, rows)
    left = labels + 0.55
    ax = frame(fig, left, bottom, HALF - left - right, height - top - bottom)
    zero_line(ax)
    value_grid(ax)
    limits = (-0.02, 0.11)
    wins = total = 0
    for y, (kind, point) in enumerate(rows):
        if kind != "point":
            continue
        mean, low, high, n = interval(values[point])
        estimate(ax, y, mean, low, high, SV_PGS, limits=limits)
        # each replicate over the interval's bar and under its mean, ringed so it shows on the bar
        ax.scatter(values[point], [y] * n, s=(DOT * 1.15) ** 2, color=GREY, edgecolors="white", linewidths=1.0,
                   zorder=3.5)
        ax.text(1, y, f"{n}", transform=shifted(ax, 0.2), ha="left", va="center", color=GREY)
        wins += sum(value > 0 for value in values[point])
        total += n
    # the column's head on the first group heading's row, which has no count
    ax.text(1, 0, "n", transform=shifted(ax, 0.2), ha="left", va="center", color=GREY, fontstyle="italic")
    NUMBERS["fig2"] = {"train_count": size, "wins": wins, "replicates": total,
                       "architectures": sum(kind == "point" for kind, _ in rows)}
    row_labels(ax, rows, left - 0.15)
    ax.set_xlim(*limits)
    ticks = np.arange(0, limits[1], 0.04)
    ax.set_xticks(ticks, [plain(tick) for tick in ticks])
    ax.set_xlabel("ΔAUC, SV-PGS − LDpred2-auto")
    caption = (
        f"Figure 2 | SV-PGS is more accurate than LDpred2-auto across simulated genetic architectures. Paired difference "
        f"in test AUC on the same 10,000 held-out people between SV-PGS (every SNV, indel and imputed SV) and "
        f"LDpred2-auto (HapMap3-density SNVs), both trained on the same {size:,} people: replicates (grey), mean and "
        f"Student-t 95% interval (blue; an arrowhead marks one that runs past {limits[1]:.2f}), n replicates at right. "
        f"SV-PGS had the higher AUC in {wins} of {total} replicates. {ARCHITECTURES} Not yet replicated three times at "
        f"{size:,}: {'; '.join(sentence(PLAIN.get(point, LABELS[point])) for point in missing)}."
    )
    return fig, caption


# ------------------------------------------------------------------ Figure 3


SOURCES = (("True SV genotypes", SV_PGS, True), ("Imputed SV genotypes", IMPUTED, False))
"""Figures 3 and 4's two genotype sources: their legend names, colours and whether their means are filled."""


def source_legend(ax, across: float, up: float) -> None:
    handles = [Line2D([0], [0], marker="o", color=color, markerfacecolor=color if filled else "white",
                      markeredgecolor="white" if filled else color, markeredgewidth=RING if filled else BAR,
                      markersize=PAIR_MARK, linewidth=BAR, label=label) for label, color, filled in SOURCES]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1), bbox_transform=shifted(ax, across, up, False),
              ncol=2, columnspacing=1.4, handlelength=1.6, handletextpad=0.5, borderaxespad=0.0)


def sv_gain_figure(grids):
    imputed, true = defaultdict(list), defaultdict(list)
    for row in grids["window"]:
        imputed[row["point"]].extend(finite([row], "delta_auc"))
    for row in grids["window-truth"]:
        true[row["point"]].extend(finite([row], "delta_auc"))
    rows = layout(lambda point: len(true[point]) >= 3 and len(imputed[point]) >= 3)
    # no title: the poster's headline names the figure; the legend over the rows, a note under them
    top, bottom, right = 1.0, 1.75, 0.15
    height = top + bottom + PAIRED * (len(rows) + 0.1)
    fig = plt.figure(figsize=(HALF, height))
    labels = label_width(fig, rows)
    left = labels + 0.55
    ax = frame(fig, left, bottom, HALF - left - right, height - top - bottom)
    zero_line(ax)
    value_grid(ax)
    limits = (-0.015, 0.035)
    clear = shown = 0
    NUMBERS["fig3"] = {"architectures": {LABELS[point]: {"true": list(interval(true[point])),
                                                         "imputed": list(interval(imputed[point]))}
                                         for kind, point in rows if kind == "point"}}
    counts = []
    for y, (kind, point) in enumerate(rows):
        if kind != "point":
            continue
        for offset, values, (_, color, filled) in ((-SPLIT, true[point], SOURCES[0]), (SPLIT, imputed[point], SOURCES[1])):
            mean, low, high, n = interval(values)
            estimate(ax, y + offset, mean, low, high, color, filled, limits=limits, size=PAIR_MARK)
            counts.append(n)
        shown += 1
        clear += interval(true[point])[1] > 0
    NUMBERS["fig3"].update(clear=clear, shown=shown)
    row_labels(ax, rows, left - 0.15)
    ax.set_xlim(*limits)
    ticks = np.arange(-0.01, 0.031, 0.01)
    ax.set_xticks(ticks, [plain(tick) for tick in ticks])
    ax.set_xlabel("ΔAUC, with SVs − without")
    source_legend(ax, -(left - 0.15), 0.3)
    sizes = sorted({int(row["train_count"]) for grid in ("window", "window-truth") for row in grids[grid]})
    fig.text(0.1 / HALF, 0.15 / height, f"Training {thousands(sizes[0])}–{thousands(sizes[-1])}k, pooled\n"
             f"{min(counts)}–{max(counts)} replicates each", ha="left", va="bottom", color=GREY, linespacing=1.2)
    # the widest true-genotype interval, said to be one replicate's doing when that replicate lies over 3.5 robust SDs
    # (1.4826 median absolute deviations) from the others' median
    widest_point = max((point for kind, point in rows if kind == "point"),
                       key=lambda point: np.subtract(*interval(true[point])[2:0:-1]))
    values = np.asarray(true[widest_point])
    farthest = values[np.argmax(np.abs(values - np.median(values)))]
    others = values[values != farthest]
    spread = 1.4826 * float(np.median(np.abs(others - np.median(others))))
    driven = (f" The widest true-genotype interval, {SHORT[widest_point]}, comes from one replicate at "
              f"{plain(float(farthest), 3)}." if spread > 0 and abs(farthest - np.median(others)) > 3.5 * spread else "")
    caption = (
        "Figure 3 | SVs add accuracy when their effects are large and they are measured well; imputation removes most of "
        "the gain. Paired difference in test AUC on the same held-out people between SV-PGS fitted with and without SVs "
        "on one 3 Mb chromosome 22 window (60,000 records), with every SV measured by its true genotype (filled) or "
        "imputed by Beagle from SNV haplotypes (open): mean and Student-t 95% interval over "
        f"{min(counts)}-{max(counts)} replicates, every training size from {sizes[0]:,} to {sizes[-1]:,} pooled (Figure 4 "
        f"shows them apart). The true-genotype interval lies above zero for {clear} of {shown} architectures. Same "
        f"simulated people, truth and fits; only the SV genotypes differ.{driven} {ARCHITECTURES}"
    )
    return fig, caption


# ------------------------------------------------------------------ Figure 4


def size_figure(grids):
    """The gain from adding SVs (Figure 3) at each training size apart, true and imputed SV genotypes in each row."""
    sizes = SIZES[:-1]
    summaries = {}
    for grid in ("window-truth", "window"):
        grouped = cells(grids[grid])
        summaries[grid] = {(point, size): interval(finite(grouped.get((point, size), []), "delta_auc"))
                           for point in LABELS for size in sizes}
    rows = layout(lambda point: True)
    limits = (-0.03, 0.05)
    top, bottom, panel, space = 2.45, 1.6, 4.95, 0.45
    height = top + bottom + PAIRED * (len(rows) + 0.1)
    fig = plt.figure(figsize=(WIDE, height))
    labels = label_width(fig, rows)
    left = labels + 0.55
    axes = [frame(fig, left + index * (panel + space), bottom, panel, height - top - bottom) for index in range(len(sizes))]
    for ax, size in zip(axes, sizes):
        zero_line(ax)
        value_grid(ax)
        for y, (kind, point) in enumerate(rows):
            if kind != "point":
                continue
            few = []
            for offset, grid, (_, color, filled) in ((-SPLIT, "window-truth", SOURCES[0]), (SPLIT, "window", SOURCES[1])):
                mean, low, high, n = summaries[grid][(point, size)]
                if n >= 3:
                    estimate(ax, y + offset, mean, low, high, color, filled, limits=limits, size=PAIR_MARK)
                else:
                    few.append(offset)
            if few:
                ax.text(0.97, y + (few[0] if len(few) == 1 else 0.0), "n < 3", transform=blended_transform_factory(ax.transAxes, ax.transData),
                        ha="right", va="center", color=GREY, fontstyle="italic")
        ax.set_xlim(*limits)
        ticks = np.arange(-0.02, 0.041, 0.02)
        ax.set_xticks(ticks, [plain(tick) for tick in ticks])
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_ylim(len(rows) - 0.5, -0.6)
        ax.text(0.5, 1, f"{size:,} training", transform=shifted(ax, 0, 0.2, False), ha="center", va="baseline",
                fontweight="bold")
    row_labels(axes[0], rows, left - 0.15)
    axes[1].set_xlabel("ΔAUC, with SVs − without")
    heading(axes[0], f"Gain from adding SVs at {sizes[0]:,}–{sizes[-1]:,} training", across=-(left - 0.15), up=1.75)
    source_legend(axes[0], -(left - 0.15), 0.8)
    # the largest size's window fits, too few per architecture to show apart; Figure 3 pools them with the rest
    largest = SIZES[-1]
    few = [len(finite(cells(grids[grid]).get((point, largest), []), "delta_auc")) for grid in ("window-truth", "window")
           for point in LABELS]
    caption = (
        f"Figure 4 | The gain from adding SVs at each training size from {sizes[0]:,} to {sizes[-1]:,}. Mean paired "
        "difference in test AUC between SV-PGS fitted with and without SVs on one chromosome 22 window, with SVs measured "
        f"by their true genotypes (filled) or imputed (open), and its Student-t 95% interval; arrowheads mark intervals "
        f"that run past {plain(limits[0])} or {plain(limits[1])}. A cell is drawn with at least three replicates (n < 3 "
        f"marks one with fewer). The {largest:,}-person window fits number {min(few)}-{max(few)} per architecture and "
        f"genotype source, too few to show apart; Figure 3 pools them with the other sizes. {ARCHITECTURES}"
    )
    return fig, caption


# ------------------------------------------------------------------ Figure 5


MEDIAN_Z = stats.norm.ppf(0.975) * math.sqrt(math.pi / 2) / (2 * stats.norm.ppf(0.75))
"""A median's 95% interval half-width per IQR / sqrt(n) under a normal shape (1.96 x 1.2533 / 1.349 = 1.82): the
summary keeps each frequency bin's quartiles and count, not its values."""


def imputation_panel(ax, accuracy) -> None:
    """The r² of each variant's Beagle-imputed dosage with its true genotype (benchmarks/power/imputation_accuracy.py,
    not Beagle's own DR2) by minor allele frequency, SNVs and SVs: each frequency bin's median with its approximate 95%
    interval (bars), over the bin's interquartile range (band) (Figure 5a)."""
    ends = []
    for key, color, label in (("SNV", INK, "SNVs"), ("SV", IMPUTED, "SVs")):
        bins = [b for b in accuracy[key]["bins"] if b["count"]]
        centers = np.array([math.sqrt(b["low"] * b["high"]) * 100 for b in bins])
        median = np.array([b["median"] for b in bins])
        half = np.array([MEDIAN_Z * (b["q75"] - b["q25"]) / math.sqrt(b["count"]) for b in bins])
        ax.fill_between(centers, [b["q25"] for b in bins], [b["q75"] for b in bins], color=color, alpha=0.2,
                        linewidth=0, zorder=1)
        ax.vlines(centers, np.clip(median - half, 0, 1), np.clip(median + half, 0, 1), color=color, linewidth=BAR,
                  zorder=2)
        ax.plot(centers, median, color=color, marker="o", markersize=MARK * 0.85, markeredgecolor="white",
                markeredgewidth=2.0, zorder=3)
        ends.append((centers[-1] * 1.25, median[-1], label))
    top = 1.12
    ax.set_xscale("log")
    ax.set_xlim(0.05, 100)
    ax.set_ylim(0, top)
    ax.spines["left"].set_bounds(0, 1)
    # each type named at its own line's end, the SNVs' name lifted over its line where the two ends meet
    height = ax.get_position().height * ax.figure.get_size_inches()[1]
    (x, snv, first), (_, sv, second) = ends
    ax.text(x, max(snv, sv + 0.62 * top / height), first, color=INK, fontweight="bold", va="center")
    ax.text(x, sv, second, color=INK, fontweight="bold", va="center")
    ax.set_xticks([0.1, 1, 10], ["0.1", "1", "10"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_yticks(np.arange(0, 1.01, 0.25), ["0", "0.25", "0.50", "0.75", "1"])
    ax.set_xlabel("Minor allele frequency (%)")
    ax.set_ylabel("r² with true genotype")
    ax.vlines(1, 0, 1, color=GREY, linewidth=2.0, linestyle=(0, (2, 2)), zorder=0)
    # under the SVs' band at common frequencies: the share of SVs left of the 1% line, pointing at it, and the key
    ax.annotate(f"{accuracy['SV']['below_1pct']:.0%} of SVs\nhave MAF < 1%", xy=(1.0, 0.36), xytext=(4.2, 0.36),
                color=INK, va="center", ha="left", linespacing=1.15,
                arrowprops={"arrowstyle": "-|>", "color": INK, "linewidth": 2.4, "shrinkA": 4, "shrinkB": 0,
                            "mutation_scale": 28})
    ax.text(4.2, 0.03, "bars: 95% CI\nof each median", color=GREY, va="bottom", ha="left", linespacing=1.15)
    # the band named where it is widest, inside it, rather than in a legend
    ax.text(0.068, 0.38, "middle 50%\nof SVs", color=INK, va="center", ha="left", linespacing=1.1)


SV_CLASSES = (("Deletion", ORANGE), ("Insertion", GREEN), ("Duplication", PURPLE))
"""Figure 5b's SV classes and their colours, away from the methods' blue and vermillion."""


def genotype_figure(summary):
    accuracy = summary["genotype_accuracy"]["r2_by_frequency"]
    records = np.load(LOCAL / "public" / "sv_records.npz")
    height = 9.6
    fig = plt.figure(figsize=(WIDE, height))
    top, bottom = 1.25, 1.9
    left = frame(fig, 2.0, bottom, 8.6, height - top - bottom)
    right = frame(fig, 13.9, bottom, 8.2, height - top - bottom)
    imputation_panel(left, accuracy)
    share = accuracy["SV"]["below_1pct"]
    heading(left, "Imputation accuracy by frequency", "a", across=-1.8)

    kind = records["svtype"][records["measured"]]
    change = records["len_change"][records["measured"]]
    length = np.abs(change).astype(float)
    masks = {"Deletion": (kind == "DEL") | ((kind == "") & (change < 0)),
             "Insertion": (kind == "INS") | ((kind == "") & (change > 0)), "Duplication": kind == "DUP"}
    groups = [(label, masks[label], color) for label, color in SV_CLASSES]
    edges = np.logspace(math.log10(50), 6, 37)
    bottom_counts = np.zeros(len(edges) - 1)
    stacked = {}
    for label, mask, color in groups:
        counts, _ = np.histogram(length[mask], bins=edges)
        right.bar(edges[:-1], counts, width=np.diff(edges), bottom=bottom_counts, align="edge", color=color,
                  edgecolor="white", linewidth=1.6, label=f"{label} ({int(mask.sum()):,})")
        bottom_counts += counts
        stacked[label] = counts
    # the insertions' peak near 300 bp, where Alu elements lie, named where the data show it: a bin there holding over
    # three times the insertions of either neighbour
    centres = np.sqrt(edges[:-1] * edges[1:])
    insertions = stacked["Insertion"]
    alu = [index for index in np.flatnonzero((centres > 250) & (centres < 400))
           if insertions[index] > 3 * max(insertions[index - 1], insertions[index + 1])]
    for index in alu:
        right.text(centres[index], bottom_counts[index] + 8, "Alu", ha="center", va="bottom", color=INK)
    # the long tail too thin to see at this scale, said in words
    long = int((length > 1e4).sum())
    if long:
        right.text(2e4, 0.08, f"{long} SVs > 10 kb,\nup to {length.max() / 1e3:,.0f} kb", color=GREY, va="bottom",
                   ha="left", linespacing=1.15, transform=blended_transform_factory(right.transData, right.transAxes))
    right.set_xscale("log")
    right.set_xlim(45, 1.1e6)
    right.set_xticks([1e2, 1e3, 1e4, 1e5, 1e6], ["10²", "10³", "10⁴", "10⁵", "10⁶"])
    right.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    right.set_xlabel("SV length (bp)")
    right.set_ylabel("Number of SVs")
    value_grid(right, "y")
    right.legend(loc="upper right", handlelength=0.9, handleheight=0.9, borderaxespad=0.0)
    middle = float(np.median(length[np.any([mask for _, mask, _ in groups], axis=0)]))
    right.axvline(middle, color=GREY, linewidth=2.0, linestyle=(0, (2, 2)), zorder=0)
    right.text(middle * 1.15, 0.45, f"median\n{middle:,.0f} bp", transform=blended_transform_factory(right.transData,
               right.transAxes), color=GREY, va="bottom", linespacing=1.15)
    heading(right, "Lengths of the simulated SVs", "b", across=-1.9)
    rare = {key: [b for b in accuracy[key]["bins"] if b["high"] <= 0.01 and b["count"]] for key in ("SV", "SNV")}
    NUMBERS["fig5"] = {"people": PEOPLE, "svs": accuracy["SV"]["records"], "snvs": accuracy["SNV"]["records"],
                       "svs_below_1pct": share,
                       # mean r^2 below 1% frequency, each bin's mean weighted by its variant count
                       "rare_r2": {key: sum(b["mean"] * b["count"] for b in bins) / sum(b["count"] for b in bins)
                                   for key, bins in rare.items()},
                       "classes": {label: int(mask.sum()) for label, mask, _ in groups}}
    caption = (
        f"Figure 5 | The simulated SVs are mostly rare and short, and rare SVs impute poorly. a, The r² of each "
        f"variant's Beagle-imputed dosage with its true genotype over all {PEOPLE:,} simulated people (not Beagle's DR2), "
        f"for {accuracy['SV']['records']:,} SVs and {accuracy['SNV']['records']:,} sampled SNVs on chromosome 22, by "
        f"minor allele frequency: each bin's median (points) with an approximate 95% interval ({MEDIAN_Z:.2f} IQR/√n; "
        f"bars), over the bin's middle 50% of variants (bands; SNVs grey, SVs blue). b, Lengths of the "
        f"{int(records['measured'].sum()):,} measured SVs, stacked by class (records without a type are classed by the "
        f"sign of their length change)" + ("; the insertions' peak near 300 bp is where Alu elements lie" if alu else "")
        + ". bench-sim v7 cohort from public 1000 Genomes haplotypes."
    )
    return fig, caption


# ------------------------------------------------------------------ the poster's own figures


POSTER_TYPE, POSTER_LABEL, POSTER_NAME = 40, 44, 48
"""Points at print size in the poster's figures: ticks and notes; axis titles; the bold names of methods and genotype
sources written beside their marks. A poster figure carries no legend and no title, since the poster's headline over
it states its message."""
POSTER_LINE, POSTER_MARK, POSTER_BAR = 8.0, 26.0, 6.0
"""Points: a data line's width, a mean's marker and an interval's line in the poster's figures."""
POSTER_HEIGHT = 7.6
"""Inches: the height of the two figures side by side at the top of the poster's middle column."""
POSTER_HALF = 11.06
"""Inches: the width each of those two figures is placed at (half of build_pptx.py's 22.92 in middle column less its
0.8 in gutter), so their type prints at its own size."""
POSTER_STYLE = {
    "font.size": POSTER_TYPE, "axes.labelsize": POSTER_LABEL, "xtick.labelsize": POSTER_TYPE,
    "ytick.labelsize": POSTER_TYPE, "axes.linewidth": 3.2, "xtick.major.width": 3.2, "ytick.major.width": 3.2,
    "xtick.major.size": 14, "ytick.major.size": 14, "xtick.major.pad": 10, "ytick.major.pad": 10,
    "axes.labelpad": 16, "lines.linewidth": POSTER_LINE, "lines.markersize": POSTER_MARK,
}


def swarm(values, across: float, diameter: float) -> list[float]:
    """Vertical offsets, in inches, that stack ``values`` (an axis ``across`` inches per data unit) into a beeswarm of
    dots ``diameter`` inches wide: each dot, smallest value first, goes at the offset nearest the centre line where it
    overlaps no dot already placed."""
    placed: list[tuple[float, float]] = []
    offsets = [0.0] * len(values)
    for index in np.argsort(values):
        x = values[index] * across
        candidates = sorted({0.0, *(y + sign * math.sqrt(max(diameter ** 2 - (x - px) ** 2, 0.0))
                                    for px, y in placed if abs(x - px) < diameter for sign in (1, -1))}, key=abs)
        offsets[index] = next(y for y in candidates
                              if all((x - px) ** 2 + (y - py) ** 2 >= diameter ** 2 * 0.999 for px, py in placed))
        placed.append((x, offsets[index]))
    return offsets


def poster_accuracy_figure(grids, summary):
    """Figure 1a for the poster: test AUC against training size for SV-PGS and LDpred2-auto, pooled over 0%, 5% and 20%
    causal SVs, each method named beside its line and its AUC at the largest size written after its last mark, under
    the AUC of the true genetic value."""
    grouped = cells(grids["grid"])
    fig = plt.figure(figsize=(POSTER_HALF, POSTER_HEIGHT))
    # the right margin holds each line's last AUC, in bold at tick size
    ends = widest(fig, ["0.000"], fontsize=POSTER_TYPE, fontweight="bold")
    top, bottom, left, right = 0.25, 2.05, 2.25, 0.3 + ends
    ax = frame(fig, left, bottom, POSTER_HALF - left - right, POSTER_HEIGHT - top - bottom)
    log_size_axis(ax)
    ax.set_xlabel("Training people (thousands)")
    ceiling = summary["points"]["null"]["true_value_auc"]
    ax.axhline(ceiling, color=INK, linestyle=(0, (1, 1.2)), linewidth=4.0, zorder=1)
    ax.text(SIZES[0] / 1.25, ceiling - 0.005, "Best possible", color=INK, va="top", ha="left")
    means, lows, highs = {}, {}, {}
    for key, method, color in (("with_sv_auc", "SV-PGS", SV_PGS), ("ldpred2_auc", "LDpred2-auto", LDPRED2)):
        pooled = [interval([row[key] for point, *_ in SV_SHARES for row in grouped.get((point, size), [])])
                  for size in SIZES]
        mean, low, high = (np.array([cell[i] for cell in pooled]) for i in range(3))
        means[method], lows[method], highs[method] = mean, low, high
        ax.fill_between(SIZES, low, high, color=color, alpha=0.2, linewidth=0, zorder=2)
        ax.plot(SIZES, mean, color=color, zorder=3, marker="o", markeredgewidth=0)
        ax.text(SIZES[-1] * 1.15, mean[-1], f"{mean[-1]:.3f}", color=color, fontweight="bold", ha="left", va="center")
    # the SV-PGS AUC at the smallest training size whose interval clears LDpred2-auto's at the largest (the size the
    # poster's text names), carried right to the largest size as a dashed level, over LDpred2-auto's end
    sooner = next((index for index in range(len(SIZES) - 1) if lows["SV-PGS"][index] > highs["LDpred2-auto"][-1]), None)
    if sooner is not None:
        ax.plot([SIZES[sooner], SIZES[-1]], [means["SV-PGS"][sooner]] * 2, color=SV_PGS, linewidth=4.0,
                linestyle=(0, (2, 1.5)), zorder=2.5)
    # each method's name on the empty side of its own line: SV-PGS above its left half, LDpred2-auto below its right
    ax.text(SIZES[0] / 1.2, means["SV-PGS"][1] + 0.022, "SV-PGS", color=SV_PGS, fontsize=POSTER_NAME,
            fontweight="bold", ha="left", va="bottom")
    ax.text(SIZES[-1] * 1.2, 0.506, "LDpred2-auto", color=LDPRED2, fontsize=POSTER_NAME, fontweight="bold",
            ha="right", va="bottom")
    ax.set_ylim(0.5, ceiling + 0.008)
    ax.set_yticks(np.arange(0.5, 0.651, 0.05), ["0.50", "0.55", "0.60", "0.65"])
    ax.set_ylabel("Test AUC")
    return fig


def poster_wins_figure(grids):
    """Every replicate's paired AUC difference between SV-PGS and LDpred2-auto at the largest training size, in every
    architecture, as one beeswarm right of a zero line; the replicates with no causal SVs are dark."""
    size = SIZES[-1]
    grouped = cells(grids["grid"])
    found = {point: finite(grouped.get((point, size), []), "delta_auc_vs_ldpred2") for point in LABELS}
    kept = [point for point in LABELS if found[point]]
    values = np.asarray([value for point in kept for value in found[point]])
    null = np.asarray([point == "null" for point in kept for _ in found[point]])
    NUMBERS["poster_wins"] = {"replicates": int(values.size), "wins": int((values > 0).sum()),
                              "null_replicates": int(null.sum()), "architectures": len(kept)}
    fig = plt.figure(figsize=(POSTER_HALF, POSTER_HEIGHT))
    top, bottom, left, right = 0.25, 2.05, 0.45, 0.75
    width, height = POSTER_HALF - left - right, POSTER_HEIGHT - top - bottom
    ax = frame(fig, left, bottom, width, height)
    limits = (-0.008, 0.082)
    diameter = 0.44
    offsets = np.asarray(swarm(values, width / (limits[1] - limits[0]), diameter))
    # one data unit on y is one inch, so the dots stay round; the words and arrow take the top 1.5 in, the swarm sits
    # centred in the rest
    ax.set_ylim(-height / 2, height / 2)
    upper, words = height / 2 - 0.1, 1.4
    centre = (upper - words - height / 2) / 2 - (offsets.max() + offsets.min()) / 2
    area = (diameter * 72) ** 2 * 0.9
    ax.scatter(values[~null], centre + offsets[~null], s=area, color=SV_PGS, linewidths=0, zorder=3)
    ax.scatter(values[null], centre + offsets[null], s=area, color=INK, linewidths=0, zorder=3)
    ax.axvline(0, color=INK, linewidth=5.0, zorder=2)
    ax.set_xlim(*limits)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ticks = np.arange(0, limits[1], 0.02)
    ax.set_xticks(ticks, [plain(tick) for tick in ticks])
    ax.set_xlabel("AUC gain over LDpred2-auto")
    ax.text(0.003, upper, "SV-PGS better", color=SV_PGS, fontsize=POSTER_NAME, fontweight="bold", ha="left", va="top")
    ax.text(0.081, upper, "Black: no\ncausal SVs", color=INK, fontweight="bold", ha="right", va="top", linespacing=1.05)
    ax.annotate("", xy=(0.052, upper - 1.05), xytext=(0.003, upper - 1.05), xycoords="data",
                arrowprops={"arrowstyle": "-|>,head_length=0.9,head_width=0.45", "color": SV_PGS, "linewidth": 5.0,
                            "shrinkA": 0, "shrinkB": 0})
    return fig


POSTER_ROWS = {
    "multi_feature_positive": "Effects depend on SV\nlength, type, annotation",
    "deletion_positive": "Deletions have 3×\nlarger effects",
    "sv_heavy_tail": "A few SVs with\nvery large effects",
    "sparse_sv": "50 causal\nvariants",
    "tagged_sv_fold3": "Causal SVs well\nimputed, 3× effects",
    "common_sv": "All causal\nSVs common",
    "sv_two_point": "All causal SVs\none effect size",
}
"""The poster's row labels for the architectures where adding true SV genotypes clearly raised AUC, in plain words."""
POSTER_GAIN, POSTER_GENOTYPES = 22.92, 11.4
"""Inches: the widths of the SV gain forest (the poster's middle column, build_pptx.py's MIDDLE) and the imputation
figure (the left column, LEFT), each placed at its column's full width."""


def poster_sv_gain_figure(grids):
    """The AUC gain from adding SVs with true and with imputed SV genotypes, for the architectures whose true-genotype
    gain has a 95% interval above zero, largest first: a row per architecture, its label at the left, the true gain's
    mean and interval over the imputed gain's."""
    imputed, true = defaultdict(list), defaultdict(list)
    for row in grids["window"]:
        imputed[row["point"]].extend(finite([row], "delta_auc"))
    for row in grids["window-truth"]:
        true[row["point"]].extend(finite([row], "delta_auc"))
    shown = [point for point in LABELS if len(true[point]) >= 3 and len(imputed[point]) >= 3]
    clear = sorted((point for point in shown if interval(true[point])[1] > 0), key=lambda point: -interval(true[point])[0])
    NUMBERS["poster_sv_gain"] = {
        "clear": len(clear), "shown": len(shown),
        "true_sum": sum(interval(true[point])[0] for point in clear),
        "imputed_sum": sum(interval(imputed[point])[0] for point in clear),
        "rows": {point: {"true": list(interval(true[point])), "imputed": list(interval(imputed[point]))} for point in clear},
    }
    pitch, top, bottom, right = 1.15, 1.05, 1.95, 0.5
    height = top + bottom + pitch * len(clear)
    fig = plt.figure(figsize=(POSTER_GAIN, height))
    left = widest(fig, [POSTER_ROWS[point] for point in clear], fontsize=POSTER_TYPE, linespacing=1.0) + 0.75
    ax = frame(fig, left, bottom, POSTER_GAIN - left - right, height - top - bottom)
    limits = (-0.006, 0.036)
    for y, point in enumerate(clear):
        for offset, values, (_, color, filled) in ((-0.2, true[point], SOURCES[0]), (0.2, imputed[point], SOURCES[1])):
            mean, low, high, _ = interval(values)
            ax.hlines(y + offset, max(low, limits[0]), min(high, limits[1]), color=color, linewidth=POSTER_BAR,
                      zorder=3, capstyle="round")
            ax.plot([mean], [y + offset], "o", markersize=POSTER_MARK, color=color,
                    markerfacecolor=color if filled else "white", markeredgecolor=color,
                    markeredgewidth=0 if filled else 6.0, zorder=4)
        ax.text(0, y, POSTER_ROWS[point], transform=shifted(ax, -0.3), ha="right", va="center", linespacing=1.0)
    ax.axvline(0, color=INK, linewidth=4.0, zorder=2)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_ylim(len(clear) - 0.5, -0.5)
    ax.set_xlim(*limits)
    ticks = np.arange(0, limits[1], 0.01)
    ax.set_xticks(ticks, [plain(tick) for tick in ticks])
    ax.set_xlabel("AUC gain from adding SVs")
    # the two genotype sources named once over the whole figure, labels and plot, each after a mark like its own
    for across, (label, color, filled) in ((left, SOURCES[0]), (left + 8.2, SOURCES[1])):
        fig.add_artist(Line2D([(across + 0.25) / POSTER_GAIN], [1 - 0.55 / height], marker="o", markersize=POSTER_MARK,
                              color=color, markerfacecolor=color if filled else "white",
                              markeredgecolor=color, markeredgewidth=0 if filled else 6.0))
        fig.text((across + 0.75) / POSTER_GAIN, 1 - 0.55 / height, label, color=color, fontsize=POSTER_NAME,
                 fontweight="bold", va="center", ha="left")
    return fig


def poster_imputation_figure(summary):
    """The imputation figure in the poster's left column, wide and short: each frequency bin's mean r² of the imputed
    dosage with the true genotype, SNVs and SVs, with frequencies below 1% shaded and the share of SVs there written
    in the shading. The mean, not the median, so the curve agrees with the mean r² below 1% the poster's text quotes."""
    accuracy = summary["genotype_accuracy"]["r2_by_frequency"]
    height = 5.2
    fig = plt.figure(figsize=(POSTER_GENOTYPES, height))
    # the y axis's title sits over its tick labels, left to right, which leaves the plot the figure's width
    top, bottom, left, right = 0.75, 1.75, 1.25, 0.4
    ax = frame(fig, left, bottom, POSTER_GENOTYPES - left - right, height - top - bottom)
    ax.axvspan(0.04, 1, color="#EBEBEB", zorder=0, linewidth=0)
    ends = {}
    for key, color in (("SNV", INK), ("SV", IMPUTED)):
        bins = [b for b in accuracy[key]["bins"] if b["count"]]
        centers = np.array([math.sqrt(b["low"] * b["high"]) * 100 for b in bins])
        mean = np.array([b["mean"] for b in bins])
        ax.plot(centers, mean, color=color, marker="o", markersize=POSTER_MARK * 0.9, markeredgewidth=0, zorder=3,
                clip_on=False)
        ends[key] = (centers[-1], mean[-1])
    # SNVs named over their line's end, SVs under theirs, so neither name sits between the lines
    ax.text(ends["SNV"][0], ends["SNV"][1] + 0.07, "SNVs", color=INK, fontsize=POSTER_NAME, fontweight="bold",
            ha="right", va="bottom")
    ax.text(ends["SV"][0], ends["SV"][1] - 0.16, "SVs", color=IMPUTED, fontsize=POSTER_NAME, fontweight="bold",
            ha="right", va="top")
    # the share of SVs in the shaded band, written in the band between the two curves
    ax.text(0.05, 0.64, f"{accuracy['SV']['below_1pct']:.0%} of SVs", ha="left", va="center", color=INK,
            fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(0.04, 50)
    ax.set_ylim(0, 1)
    ax.set_xticks([0.1, 1, 10], ["0.1", "1", "10"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.set_xlabel("Minor allele frequency (%)")
    ax.text(0, 1, "Imputation r²", transform=shifted(ax, -(left - 0.12), 0.25, False), ha="left", va="bottom",
            fontsize=POSTER_LABEL)
    return fig


# ------------------------------------------------------------------ Figure 6


def compute_figure(grids):
    grouped = cells(grids["grid"])
    height = 8.8
    fig = plt.figure(figsize=(HALF, height))
    top, bottom = 1.15, 1.9
    ax = frame(fig, 2.15, bottom, 5.6, height - top - bottom)
    # SV-PGS with and without SVs dodged apart and told apart by line as well as mark: their times nearly coincide
    methods = (("with_sv", "SV-PGS", SV_PGS, "o", 0.1, "-"), ("snv_indel", None, SV_PGS, "s", -0.1, (0, (2, 1.2))),
               ("ldpred2_auto", "LDpred2-auto", LDPRED2, "o", 0.0, "-"))
    xs = np.log2(np.array(SIZES) / SIZES[0])
    counts = []
    for key, label, color, marker, offset, style in methods:
        present, medians = [], []
        for step, size in enumerate(SIZES):
            minutes = [row["fit_seconds"][key] / 60 for (point, n), rows in grouped.items() if n == size
                       for row in rows if isinstance(row.get("fit_seconds"), dict)
                       and math.isfinite(row["fit_seconds"].get(key, math.nan))]
            median, low, high, n = median_interval(minutes)
            if not math.isfinite(low):
                continue
            counts.append(n)
            x = xs[step] + offset
            # capped, so an end that clears the mark reads as an interval's end, not a stray tick
            ax.errorbar([x], [median], yerr=[[median - low], [high - median]], fmt="none", ecolor=color,
                        elinewidth=BAR * 0.75, capsize=7, capthick=BAR * 0.75, zorder=2)
            present.append(x), medians.append(median)
        ax.plot(present, medians, color=color, linewidth=4.0 if label else 3.0, linestyle=style, marker=marker,
                markersize=MARK * (0.85 if marker == "s" else 1.0), markeredgecolor="white", markeredgewidth=RING,
                zorder=3 if label else 4)
        if label:
            ax.text(xs[-1] + 0.3, medians[-1], label, color=color, fontweight="bold", va="center")
    ax.set_yscale("log")
    ax.set_ylim(0.3, 2000)
    ax.set_yticks([1, 10, 100, 1000], ["1", "10", "100", "1,000"])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xticks(xs, [thousands(size) for size in SIZES])
    ax.set_xlim(xs[0] - 0.3, xs[-1] + 0.3)
    ax.set_xlabel("Training sample size (thousands)")
    ax.set_ylabel("Wall time per fit (min)")
    value_grid(ax, "y")
    handles = [Line2D([0], [0], marker=marker, color=SV_PGS, markeredgecolor="white", markeredgewidth=RING,
                      markersize=MARK * (0.85 if marker == "s" else 1.0), linewidth=3.0, linestyle=style, label=label)
               for label, marker, style in (("with SVs", "o", "-"), ("without SVs", "s", (0, (2, 1.2))))]
    # in the band between the methods: the key to the SV-PGS marks above the note on the intervals
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.02, 0.62), handlelength=2.0, handletextpad=0.4,
              borderaxespad=0.0, labelspacing=0.3)
    ax.text(0.98, 0.22, "bars: 95% CI;\nmost within the marks", transform=ax.transAxes, ha="right", va="bottom",
            color=GREY, linespacing=1.15)
    heading(ax, "Time to fit chromosome 22", across=-1.95)
    caption = (
        "Figure 6 | SV-PGS fits take hours and LDpred2-auto's minutes; SVs add no time. Wall time of one fit on "
        "chromosome 22 per method and training size, pooled over every architecture: median and its distribution-free "
        f"95% interval ({min(counts)}-{max(counts)} fits each; capped bars, most within the marks). MSI Slurm jobs, one "
        "fit per job; SV-PGS fits every variant by empirical Bayes (circles and solid line, with SVs; squares and dashed "
        "line, without), LDpred2-auto HapMap3-density SNVs. The results do not record each job's hardware, so the gap "
        "compares the methods as they were run, not on equal resources."
    )
    return fig, caption


# ------------------------------------------------------------------ main


def gallery_page(pdf: PdfPages, image: Path, caption: str) -> None:
    """One page: the figure's raster above its legend, for reading the set in a viewer."""
    picture = plt.imread(image)
    height, width = picture.shape[:2]
    page_width = 8.5
    figure_height = page_width * 0.86 * height / width
    page = plt.figure(figsize=(page_width, figure_height + 2.2))
    holder = page.add_axes([0.07, 2.2 / (figure_height + 2.2), 0.86, figure_height / (figure_height + 2.2)])
    holder.imshow(picture, interpolation="lanczos")
    holder.axis("off")
    page.text(0.07, 2.0 / (figure_height + 2.2), textwrap.fill(caption, 125), ha="left", va="top", fontsize=8.5,
              linespacing=1.35)
    pdf.savefig(page)
    plt.close(page)


COLUMN, EDGE = 11.4, 0.3
"""Inches: the poster's left column (build_pptx.py's LEFT), where its schematics are placed at full width so their type
prints at its own size, and the margin kept clear at either side."""
VARIANTS = (("SNV", INK, 0.09), ("indel", GREEN, 0.3), ("SV", IMPUTED, 1.3))
"""The variant classes as the left column draws them: name, colour, and the width in inches of the block that stands for
one variant on a chromosome (one base for an SNV, a few for an indel, many for an SV)."""
SCHEMATICS = {
    "poster_model": "An SNV, an indel and an SV on a chromosome, each class with its own effect-size prior (the SV's "
                    "narrower at low imputation r²), all fitted together into SV-PGS.",
    "poster_variants": "One stretch of chromosome: SV-PGS scores every SNV, indel and SV; LDpred2-auto scores SNVs at "
                       "HapMap3 density.",
}
"""Each schematic's one-sentence description, which build_pptx.py uses as its picture's alt text."""


def canvas(height: float):
    """A figure the left column's width and ``height`` inches tall, with one bare axes over all of it in inches."""
    fig = plt.figure(figsize=(COLUMN, height))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, COLUMN)
    ax.set_ylim(0, height)
    ax.axis("off")
    return fig, ax


def chromosome(ax, left: float, right: float, y: float, marks, tall: float = 0.56) -> None:
    """A stretch of chromosome from ``left`` to ``right`` at height ``y`` (a grey bar) with each variant of ``marks``,
    (centre, width, colour), a block ``tall`` inches high over it."""
    ax.add_patch(matplotlib.patches.FancyBboxPatch((left, y - 0.09), right - left, 0.18,
                                                   boxstyle="round,pad=0,rounding_size=0.09", facecolor=LIGHT,
                                                   edgecolor="none", zorder=1))
    for centre, width, colour in marks:
        ax.add_patch(matplotlib.patches.Rectangle((centre - width / 2, y - tall / 2), width, tall, facecolor=colour,
                                                  edgecolor="none", zorder=2))


def poster_model_figure():
    """The SV-PGS model for the poster's Methods: an SNV, an indel and an SV on a chromosome, each class with its own
    effect-size prior (the SV's also at a low imputation r², which narrows it), all fitted together into one score."""
    height = 4.4
    fig, ax = canvas(height)
    # the priors' centre line and half-width, and the score's box, in inches
    centre, reach = 5.75, 1.45
    box_left, box_middle, box_tall = 8.3, 2.05, 1.25
    bases = (2.85, 1.65, 0.25)
    # each class's prior SD in inches, a schematic (empirical Bayes learns the real ones); every prior encloses the
    # same area, so a narrower one stands taller, the SNV's 0.9 in
    spreads = (0.2, 0.28, 0.5)
    area = 0.9 * spreads[0] * math.sqrt(2 * math.pi)
    effect = np.linspace(-reach, reach, 400)
    ax.text(centre, height - 0.04, "Effect-size prior", ha="center", va="top", color=INK)
    for (name, colour, block), base, spread in zip(VARIANTS, bases, spreads):
        middle = base + 0.4
        ax.text(EDGE, middle, name, ha="left", va="center", color=colour, fontsize=POSTER_NAME, fontweight="bold")
        chromosome(ax, 2.1, 4.0, middle, [(3.05, block, colour)])
        ax.plot([centre - reach, centre + reach], [base, base], color=LIGHT, linewidth=3.2, zorder=1,
                solid_capstyle="round")
        # each curve drawn where it stands clear of its baseline, so the tails leave the baseline bare
        density = area * stats.norm.pdf(effect, scale=spread)
        shown = density > 0.03
        ax.fill_between(centre + effect, base, base + density, color=colour, alpha=0.22, linewidth=0, zorder=2)
        ax.plot(centre + effect[shown], base + density[shown], color=colour, linewidth=6.0, zorder=3,
                solid_capstyle="round")
        if name == "SV":
            # the same prior for an SV imputed with r^2 = 0.3: its variance scales by r^2
            low = area * stats.norm.pdf(effect, scale=spread * math.sqrt(0.3))
            shown = low > 0.03
            ax.plot(centre + effect[shown], base + low[shown], color=colour, linewidth=5.0, linestyle=(0, (1.4, 1.1)),
                    zorder=3)
            ax.text(centre, base + low.max() + 0.05, "low r²", ha="center", va="bottom", color=INK)
        # every class into the one model
        ax.annotate("", xy=(box_left - 0.08, box_middle + (middle - box_middle) * 0.4),
                    xytext=(centre + reach + 0.2, middle),
                    arrowprops={"arrowstyle": "-|>,head_length=0.55,head_width=0.28", "color": INK, "linewidth": 4.0,
                                "shrinkA": 0, "shrinkB": 0})
    ax.add_patch(matplotlib.patches.FancyBboxPatch((box_left, box_middle - box_tall / 2), COLUMN - EDGE - box_left,
                                                   box_tall, boxstyle="round,pad=0,rounding_size=0.2",
                                                   facecolor=SV_PGS, edgecolor="none"))
    ax.text((box_left + COLUMN - EDGE) / 2, box_middle, "SV-PGS", ha="center", va="center", color="white",
            fontsize=POSTER_NAME, fontweight="bold")
    return fig


def poster_variants_figure():
    """The variants each method scores on one stretch of chromosome: every SNV, indel and SV for SV-PGS, a sparse
    subset of the SNVs for LDpred2-auto (HapMap3 density)."""
    height = 2.25
    fig, ax = canvas(height)
    start, end = 4.9, COLUMN - EDGE
    # SVs, then indels clear of them, then SNVs every 0.155 in clear of both; LDpred2-auto keeps every fourth SNV
    svs = ((5.6, 0.75), (8.0, 1.0), (10.05, 0.6))
    indels = [x for x in (6.35, 7.0, 9.05, 10.65) if all(abs(x - centre) > width / 2 + 0.1 for centre, width in svs)]
    snvs = [x for x in np.arange(start + 0.1, end - 0.05, 0.155)
            if all(abs(x - centre) > width / 2 + 0.06 for centre, width in svs)
            and all(abs(x - indel) > 0.13 for indel in indels)]
    rows = ((height - 0.4, "SV-PGS", SV_PGS, [(c, w, IMPUTED) for c, w in svs] + [(c, 0.16, GREEN) for c in indels]
             + [(c, 0.05, INK) for c in snvs]),
            (height - 1.3, "LDpred2-auto", LDPRED2, [(c, 0.05, INK) for c in snvs[2::4]]))
    for y, name, colour, marks in rows:
        ax.text(EDGE, y, name, ha="left", va="center", color=colour, fontsize=POSTER_NAME, fontweight="bold")
        chromosome(ax, start, end, y, marks, tall=0.6)
    ax.text(start, 0.28, "HapMap3-density SNVs", ha="left", va="center", color=INK)
    return fig


def main() -> None:
    grids = load_results()
    with open(SUMMARY) as handle:
        summary = json.load(handle)
    OUT.mkdir(parents=True, exist_ok=True)
    for stale in OUT.glob("*"):
        stale.unlink()
    figures = [
        ("fig1_accuracy_by_training_size", *accuracy_figure(grids, summary)),
        ("fig2_gain_over_ldpred2", *ldpred2_figure(grids)),
        ("fig3_sv_gain_true_vs_imputed", *sv_gain_figure(grids)),
        ("fig4_sv_gain_by_training_size", *size_figure(grids)),
        ("fig5_simulated_svs", *genotype_figure(summary)),
        ("fig6_fit_time", *compute_figure(grids)),
    ]
    with PdfPages(OUT / "gallery.pdf") as gallery:
        for name, fig, caption in figures:
            fig.savefig(OUT / f"{name}.pdf")
            fig.savefig(OUT / f"{name}.png")
            plt.close(fig)
            (OUT / f"{name}.txt").write_text(caption + "\n")
            gallery_page(gallery, OUT / f"{name}.png", caption)
    # the poster's own versions, not gallery figures, in the poster's type and with no legends or gallery pages
    with plt.rc_context(POSTER_STYLE):
        for name, panel in (("poster_accuracy_by_training_size", poster_accuracy_figure(grids, summary)),
                            ("poster_wins", poster_wins_figure(grids)),
                            ("poster_sv_gain", poster_sv_gain_figure(grids)),
                            ("poster_imputation_accuracy", poster_imputation_figure(summary)),
                            ("poster_model", poster_model_figure()), ("poster_variants", poster_variants_figure())):
            # transparent, so the page shows through: an opaque white ground renders a shade off the page white in a PDF
            panel.savefig(OUT / f"{name}.png", transparent=True)
            plt.close(panel)
    for name, words in SCHEMATICS.items():
        (OUT / f"{name}.txt").write_text(words + "\n")
    text = json.dumps(rounded(NUMBERS), indent=1, ensure_ascii=False)
    # each [mean, lower, upper, n] on one line
    text = re.sub(r"\[\s*([^\[\]{}]*?)\s*\]", lambda match: "[" + " ".join(match.group(1).split()) + "]", text)
    (OUT / "numbers.json").write_text(text + "\n")
    print(OUT)


if __name__ == "__main__":
    main()

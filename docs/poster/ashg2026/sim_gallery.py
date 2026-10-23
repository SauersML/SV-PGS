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

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from benchmarks.power.truths import fieller, mean_test, truth_gains  # noqa: E402

HERE = Path(__file__).resolve().parent
LOCAL = Path.home() / "svpgs-bench" / "poster-grid2"
SUMMARY = HERE / "simulation_summary.json"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "figures" / "gallery"
SIZES = (5_000, 10_000, 20_000, 40_000)
PEOPLE = 50_000
"""Individuals in the simulated cohort (bench-sim v7): the training individuals and the 10,000 held out."""

WIDE = 22.32
"""Inches: the poster's middle column (build_pptx.py's MIDDLE, 22.92) less its 0.3 in INSET each side, the width a
figure that spans it is placed at. build_pptx.py trims each figure to its ink, so a poster figure's ink spans nearly its
whole width, and its type prints at its own size."""
HALF = 10.46
"""Inches: half the middle column less its gutter and INSETs, the width each of two figures side by side is placed at."""
OUTER = 10.8
"""Inches: the poster's left column (build_pptx.py's LEFT, 11.4) less its INSETs, where Figure 5a alone goes."""
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
# two beyond Okabe-Ito for the poster's three SV shares, which must stay apart from SV-PGS's blue and LDpred2-auto's
# vermillion as well as each other (validate_palette.js passes the five against white)
VIOLET = "#5E35B1"
WINE = "#882255"
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
plot half the width at poster type. The baseline is the architecture the others change settings of."""
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
    "effects drawn like small-variant effects. Each other architecture changes one or more of these settings or how "
    "SV effects are drawn: '3×' triples the SV effect SD, "
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


def difference(first, second) -> tuple[float, float, float]:
    """The difference of two groups' means with Welch's 95% interval."""
    first, second = (np.asarray(values, float) for values in (first, second))
    parts = first.var(ddof=1) / first.size, second.var(ddof=1) / second.size
    freedom = sum(parts) ** 2 / (parts[0] ** 2 / (first.size - 1) + parts[1] ** 2 / (second.size - 1))
    half = float(stats.t.ppf(0.975, freedom) * math.sqrt(sum(parts)))
    gap = float(first.mean() - second.mean())
    return gap, gap - half, gap + half


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


def tallest(fig, texts, **style) -> float:
    """Inches the tallest of ``texts`` takes when set in ``style``, its descenders included, as ``widest`` measures."""
    renderer = fig.canvas.get_renderer()
    heights = [0.0]
    for text in texts:
        artist = fig.text(0, 0, text, **style)
        heights.append(artist.get_window_extent(renderer).height / fig.dpi)
        artist.remove()
    return max(heights)


def overhang(fig) -> tuple[float, float, float, float]:
    """Inches the drawn figure reaches past its canvas at the left, bottom, right and top, negative where it stays
    inside: the tight box of everything drawn, axis titles, tick labels, legends and notes among it."""
    across, up = fig.get_size_inches()
    box = fig.get_tightbbox(fig.canvas.get_renderer())
    return -box.x0, -box.y0, box.x1 - across, box.y1 - up


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
            markeredgecolor=color, markeredgewidth=0 if filled else BAR, zorder=4)


def zero_line(ax, vertical: bool = True) -> None:
    (ax.axvline if vertical else ax.axhline)(0, color=INK, linewidth=2.4, zorder=1)


def load_results() -> dict[str, list[dict]]:
    """Each grid's simulations, every one of them, whether or not its SV-PGS fits carry a convergence certificate."""
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


SV_SHARES = (("null", "0%", INK, "o", "-"), ("sv_5pct", "5%", PURPLE, "s", (0, (3, 1.5))),
             ("sv_20pct", "20%", SV_PGS, "D", (0, (1, 1.2))))
"""Figure 1b's settings: the share of causal variants that are SVs, none in black as in the poster's scatter, each its
own colour, marker and line style (solid, dashed, dotted)."""


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
    ax.axhline(ceiling, color=INK, linestyle=(0, (1, 1.4)), linewidth=3.2, zorder=1)
    ax.text(0.02, ceiling - 0.004, f"Best possible (h² = {setting['h2']:g})",
            transform=blended_transform_factory(ax.transAxes, ax.transData), color=INK, va="top", ha="left")
    counts = {}
    NUMBERS["fig1"] = {"train_counts": list(SIZES), "test_people": summary["test_people"], "h2": setting["h2"],
                       "prevalence": setting["prevalence"], "causal": setting["causal"], "h2_auc": ceiling,
                       "baseline_sv_share": summary["points"]["sv_20pct"]["setting"]["sv_causal_fraction"]}
    for key, method, color in (("with_sv_auc", "SV-PGS", SV_PGS), ("ldpred2_auc", "LDpred2-auto", LDPRED2)):
        pooled = [interval([row[key] for point, *_ in SV_SHARES for row in grouped.get((point, size), [])])
                  for size in SIZES]
        NUMBERS["fig1"][method] = [list(cell) for cell in pooled]
        mean, low, high = (np.array([cell[i] for cell in pooled]) for i in range(3))
        counts[method] = [cell[3] for cell in pooled]
        ax.errorbar(SIZES, mean, yerr=(mean - low, high - mean), color=color, elinewidth=BAR, capsize=0, marker="o",
                    markeredgewidth=0, zorder=3)
        ax.text(1.03, mean[-1], method, transform=blended_transform_factory(ax.transAxes, ax.transData), color=color,
                fontweight="bold", va="center", ha="left")
    ax.set_ylim(0.5, 0.69)
    ax.set_yticks(np.arange(0.5, 0.651, 0.05), ["0.50", "0.55", "0.60", "0.65"])
    ax.text(0.98, 0.04, "0%, 5% and 20% causal SVs pooled", transform=ax.transAxes, ha="right", va="bottom",
            color=INK)
    ax.set_ylabel("Test AUC")
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
                 markersize=MARK * (0.9 if marker == "D" else 1.0), markeredgewidth=0, zorder=3, label=share)
    gap.set_ylim(-0.006, 0.08)
    gap.set_yticks(np.arange(0, 0.061, 0.02))
    gap.set_ylabel("ΔAUC, SV-PGS − LDpred2-auto")
    gap.legend(title="Causal SVs (bars: 95% CI)", loc="upper left", ncol=3, handlelength=2.2, columnspacing=1.0, handletextpad=0.4,
               alignment="left", borderaxespad=0.0)
    heading(gap, "Same gain with no causal SVs", "b", across=-1.95)
    caption = (
        "Figure 1 | SV-PGS is more accurate than LDpred2-auto at every training size, by a similar margin whether 0%, "
        "5% or 20% of the causal variants are SVs, so the gain does not come from causal SVs. "
        "A simulated binary trait on chromosome 22 (liability h² = 0.1, "
        "prevalence 10%, 200 causal variants) scored in the same 10,000 held-out individuals. SV-PGS fits every SNV, indel "
        "and imputed SV, and LDpred2-auto HapMap3-density SNVs of the same training individuals. a, Test AUC, mean and "
        "Student-t 95% interval over the three settings' simulations pooled ("
        + ", ".join(f"{n}" for n in counts["SV-PGS"])
        + " at 5k-40k). Dotted line: the largest AUC any score can reach, that of the true genetic value at h² = 0.1 (Wray "
        "et al. 2010). b, Paired difference in AUC on the same held-out individuals, per setting, mean and Student-t 95% "
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
    limits = (-0.02, 0.11)
    wins = total = 0
    for y, (kind, point) in enumerate(rows):
        if kind != "point":
            continue
        mean, low, high, n = interval(values[point])
        estimate(ax, y, mean, low, high, SV_PGS, limits=limits)
        # each simulation in a lighter blue under the interval's bar and its mean
        ax.scatter(values[point], [y] * n, s=(DOT * 1.15) ** 2, color=SV_PGS, alpha=0.35, linewidths=0, zorder=2)
        ax.text(1, y, f"{n}", transform=shifted(ax, 0.2), ha="left", va="center")
        wins += sum(value > 0 for value in values[point])
        total += n
    # the column's head on the first group heading's row, which has no count
    ax.text(1, 0, "n", transform=shifted(ax, 0.2), ha="left", va="center", fontstyle="italic")
    NUMBERS["fig2"] = {"train_count": size, "wins": wins, "replicates": total,
                       "architectures": sum(kind == "point" for kind, _ in rows)}
    row_labels(ax, rows, left - 0.15)
    ax.set_xlim(*limits)
    ticks = np.arange(0, limits[1], 0.04)
    ax.set_xticks(ticks, [plain(tick) for tick in ticks])
    ax.set_xlabel("ΔAUC, SV-PGS − LDpred2-auto")
    caption = (
        f"Figure 2 | SV-PGS is more accurate than LDpred2-auto across simulated genetic architectures. Paired difference "
        f"in test AUC on the same 10,000 held-out individuals between SV-PGS (every SNV, indel and imputed SV) and "
        f"LDpred2-auto (HapMap3-density SNVs), both trained on the same {size:,} individuals: each simulation (light blue), mean and "
        f"Student-t 95% interval (blue, with an arrowhead where one runs past {limits[1]:.2f}), n simulations at right. "
        f"SV-PGS had the higher AUC in {wins} of {total} simulations. {ARCHITECTURES} Fewer than three simulations at "
        f"{size:,}: {', '.join(sentence(PLAIN.get(point, LABELS[point])) for point in missing)}."
    )
    return fig, caption


# ------------------------------------------------------------------ Figure 3


SOURCES = (("True genotypes", SV_PGS, True), ("Called SNVs and imputed SVs", IMPUTED, False))
"""Figures 3 and 4's two genotype sources: their legend names, colours and whether their means are filled. The true
source (grid.py --measurement truth) has every record at its true genotype. The other (the beagle arm) has SNVs and
indels called from simulated reads and TRs and SVs imputed by Beagle, so the two differ in their SNVs too."""


def source_legend(ax, across: float, up: float) -> None:
    handles = [Line2D([0], [0], marker="o", color=color, markerfacecolor=color if filled else "white",
                      markeredgecolor=color, markeredgewidth=0 if filled else BAR, markersize=PAIR_MARK, linewidth=BAR,
                      label=label) for label, color, filled in SOURCES]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1), bbox_transform=shifted(ax, across, up, False),
              ncol=2, columnspacing=1.4, handlelength=1.6, handletextpad=0.5, borderaxespad=0.0)


def sv_gain_figure(grids):
    """The gain from adding SVs per architecture with true and with imputed SV genotypes, over simulations: each
    simulation's gain is its mean over the training sizes both sources finished (benchmarks/power/truths.py), since a
    simulation's sizes share its causal variants and test individuals and are not independent replicates."""
    gains = truth_gains(grids["window-truth"], grids["window"])
    true = {point: [gain for _, gain, _ in rows] for point, rows in gains.items()}
    imputed = {point: [gain for _, _, gain in rows] for point, rows in gains.items()}
    tests = {point: (mean_test(true[point]), mean_test(imputed[point])) for point in gains}
    rows = layout(lambda point: len(true.get(point, [])) >= 3)
    # no title: the poster's headline names the figure; the legend over the rows, and room right of the axis for half
    # its last tick's label
    top, bottom, right = 1.0, 1.75, 0.45
    height = top + bottom + PAIRED * (len(rows) + 0.1)
    fig = plt.figure(figsize=(HALF, height))
    labels = label_width(fig, rows)
    left = labels + 0.55
    ax = frame(fig, left, bottom, HALF - left - right, height - top - bottom)
    zero_line(ax)
    limits = (-0.02, 0.04)
    shown = [point for kind, point in rows if kind == "point"]
    clear = [point for point in shown if tests[point][0]["lower"] > 0]
    NUMBERS["fig3"] = {"architectures": {LABELS[point]: {source: [test["mean"], test["lower"], test["upper"],
                                                                  test["truths"]]
                                                         for source, test in zip(("true", "imputed"), tests[point])}
                                         for point in shown},
                       "clear": len(clear), "shown": len(shown)}
    for y, (kind, point) in enumerate(rows):
        if kind != "point":
            continue
        for offset, test, (_, color, filled) in zip((-SPLIT, SPLIT), tests[point], SOURCES):
            estimate(ax, y + offset, test["mean"], test["lower"], test["upper"], color, filled, limits=limits,
                     size=PAIR_MARK)
    row_labels(ax, rows, left - 0.15)
    ax.set_xlim(*limits)
    ticks = np.arange(-0.02, 0.041, 0.02)
    ax.set_xticks(ticks, [plain(tick) for tick in ticks])
    ax.set_xlabel("ΔAUC, with SVs − without")
    source_legend(ax, -(left - 0.15), 0.3)
    counts = [tests[point][0]["truths"] for point in shown]
    sizes = sorted({int(row["train_count"]) for grid in ("window", "window-truth") for row in grids[grid]})
    # the widest true-genotype interval, said to be one simulation's doing when that simulation lies over 3.5 robust
    # SDs (1.4826 median absolute deviations) from the others' median
    widest_point = max(shown, key=lambda point: tests[point][0]["upper"] - tests[point][0]["lower"])
    values = np.asarray(true[widest_point])
    farthest = values[np.argmax(np.abs(values - np.median(values)))]
    others = values[values != farthest]
    spread = 1.4826 * float(np.median(np.abs(others - np.median(others))))
    driven = (f" The widest true-genotype interval, '{SHORT[widest_point]}', comes from one simulation at "
              f"{plain(float(farthest), 3)}." if spread > 0 and abs(farthest - np.median(others)) > 3.5 * spread else "")
    named = ", ".join(f"'{SHORT[point]}'" for point in clear)
    caption = (
        f"Figure 3 | Adding SVs changed test AUC little in each architecture. With true genotypes the gain's interval "
        f"lies above zero for {len(clear)} of {len(shown)} architectures{': ' + named if clear else ''}. Paired "
        "difference in test AUC on the same held-out individuals between SV-PGS fitted with and without SVs on one 3 Mb "
        "chromosome 22 window (60,000 records), with true genotypes at every variant (filled) or with SNVs and indels "
        "called from simulated reads and SVs imputed by Beagle (open). Each simulation's difference is its mean over the training sizes both "
        f"genotype sources finished, from {sizes[0]:,} to {sizes[-1]:,} (Figure 4 shows the sizes apart). Mean and "
        f"Student-t 95% interval over {min(counts)}-{max(counts)} simulations per architecture, an arrowhead marking "
        "an interval that runs off the axis. Both sources use the same simulated individuals and causal variants."
        f"{driven} {ARCHITECTURES}"
    )
    return fig, caption


# ------------------------------------------------------------------ Figure 4


SIZE_PANEL, SIZE_TICKS, SIZE_GAP = 2.2, 1.65, 0.3
"""Figure 4's panels, in inches: each plot's height, the room left of it for its own y tick labels, and the gap between
one panel's plot and the next panel's tick labels."""
DODGE = 1.08
"""Figure 4 draws true SV genotypes at a training size / DODGE and imputed ones at x DODGE on its log axis, so the two
intervals at a size stand side by side."""


def gain_ticks(ax, low: float, high: float) -> None:
    """A gain axis from ``low`` to ``high`` padded by a tenth, ticked at three to five round values, 0 among them. A panel
    with nothing drawn (no size has three simulations) shows its zero line alone."""
    if high <= low:
        ax.set_ylim(-1, 1)
        ax.set_yticks([0], ["0"])
        return
    pad = 0.1 * (high - low)
    ax.set_ylim(low - pad, high + pad)
    ticks = [tick for tick in matplotlib.ticker.MaxNLocator(nbins=4, steps=[1, 2, 5, 10]).tick_values(low - pad, high + pad)
             if low - pad <= tick <= high + pad]
    digits = max(0, -math.floor(round(math.log10(ticks[1] - ticks[0]), 6)))
    ax.set_yticks(ticks, [plain(tick, digits) for tick in ticks])


def size_figure(grids):
    """The gain from adding SVs (Figure 3) against training size: a panel per architecture on its own y-scale, true and
    imputed SV genotypes as two series, and a row of panels per group of Figure 3's architectures."""
    sizes = SIZES[:-1]
    summaries = {}
    for grid in ("window-truth", "window"):
        grouped = cells(grids[grid])
        summaries[grid] = {(point, size): interval(finite(grouped.get((point, size), []), "delta_auc"))
                           for point in LABELS for size in sizes}
    NUMBERS["fig4"] = {LABELS[point]: {source: {f"{size}": list(summaries[grid][(point, size)]) for size in sizes}
                                       for source, grid in (("true", "window-truth"), ("imputed", "window"))}
                       for point in LABELS}
    left, right, top, bottom = 1.0, 0.15, 1.3, 1.05
    heading_room, title_room, tick_room = 0.65, 0.5, 0.75
    row = heading_room + title_room + SIZE_PANEL + tick_room
    height = top + bottom + row * len(GROUPS)
    fig = plt.figure(figsize=(WIDE, height))
    for index, (group, points) in enumerate(GROUPS):
        base = height - top - (index + 1) * row + tick_room
        fig.text(left / WIDE, (base + SIZE_PANEL + title_room + 0.12) / height, group, ha="left", va="baseline",
                 fontweight="bold")
        width = (WIDE - left - right - len(points) * SIZE_TICKS - (len(points) - 1) * SIZE_GAP) / len(points)
        for column, point in enumerate(points):
            ax = frame(fig, left + SIZE_TICKS + column * (SIZE_TICKS + width + SIZE_GAP), base, width, SIZE_PANEL)
            zero_line(ax, vertical=False)
            low = high = 0.0
            for (_, color, filled), grid, factor in zip(SOURCES, ("window-truth", "window"), (1 / DODGE, DODGE)):
                # a size is drawn with at least three simulations
                drawn = [(size * factor, *summaries[grid][(point, size)][:3]) for size in sizes
                         if summaries[grid][(point, size)][3] >= 3]
                if not drawn:
                    continue
                xs, means, lows, highs = (list(values) for values in zip(*drawn))
                ax.plot(xs, means, color=color, linewidth=3.0, zorder=3)
                ax.vlines(xs, lows, highs, color=color, linewidth=BAR, zorder=3)
                ax.plot(xs, means, "o", markersize=PAIR_MARK, color=color, markerfacecolor=color if filled else "white",
                        markeredgecolor=color, markeredgewidth=BAR if not filled else 0, zorder=4)
                low, high = min(low, *lows), max(high, *highs)
            ax.set_xscale("log", base=2)
            ax.set_xticks(sizes, [thousands(size) for size in sizes])
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.set_xlim(sizes[0] / 1.45, sizes[-1] * 1.45)
            gain_ticks(ax, low, high)
            ax.text(0.5, 1, SHORT[point], transform=shifted(ax, 0, 0.15, False), ha="center", va="baseline")
    fig.text(0.3 / WIDE, (bottom + row * len(GROUPS) / 2) / height, "ΔAUC, with SVs − without", rotation=90,
             ha="center", va="center", fontsize=LABEL)
    fig.text((left + (WIDE - left - right) / 2) / WIDE, 0.2 / height, "Training sample size (thousands)", ha="center",
             va="baseline", fontsize=LABEL)
    fig.text(left / WIDE, (height - 0.8) / height, "Gain from adding SVs, by training size", ha="left", va="baseline",
             fontsize=TITLE, fontweight="bold")
    handles = [Line2D([0], [0], marker="o", color=color, markerfacecolor=color if filled else "white",
                      markeredgecolor=color, markeredgewidth=BAR if not filled else 0, markersize=PAIR_MARK,
                      linewidth=3.0, label=label) for label, color, filled in SOURCES]
    fig.legend(handles=handles, loc="lower right", bbox_to_anchor=((WIDE - right) / WIDE, (height - 0.95) / height),
               ncol=2, columnspacing=1.4, handlelength=1.6, handletextpad=0.5, borderaxespad=0.0)
    # the largest size's window fits, too few per architecture to show apart; Figure 3 averages them in
    largest = SIZES[-1]
    few = [len(finite(cells(grids[grid]).get((point, largest), []), "delta_auc")) for grid in ("window-truth", "window")
           for point in LABELS]
    caption = (
        f"Figure 4 | The gain from adding SVs at each training size from {sizes[0]:,} to {sizes[-1]:,}, for each "
        "simulated architecture. Each panel shows the paired difference in test AUC between SV-PGS fitted with and "
        "without SVs on one 3 Mb chromosome 22 window, with true genotypes at every variant (filled) or with SNVs called "
        "from simulated reads and SVs imputed by Beagle (open): the mean over simulations and its Student-t 95% interval. Each panel has its own y-axis scale. "
        "A mean is drawn when its training size has at least three simulations. Each simulation uses the same causal "
        "variants and the same test individuals at every training size. We fitted "
        f"{min(few)}-{max(few)} simulations per architecture and genotype source at {largest:,} training individuals, too "
        f"few to show. Figure 3 averages them into each simulation's gain. {ARCHITECTURES}"
    )
    return fig, caption


# ------------------------------------------------------------------ Figure 5


MEDIAN_Z = stats.norm.ppf(0.975) * math.sqrt(math.pi / 2) / (2 * stats.norm.ppf(0.75))
"""A median's 95% interval half-width per IQR / sqrt(n) under a normal shape (1.96 x 1.2533 / 1.349 = 1.82): the
summary keeps each frequency bin's quartiles and count, not its values."""
AOU_IMPUTATION = HERE / "figures" / "imputation" / "imputation_numbers.json"
"""The All of Us SV imputation summary (git-ignored): each frequency bin's edges (%) and median SV r²."""
OPEN_LOW = 0.025
"""The lower edge (%) taken for that summary's open lowest bin, below 0.1%, so its point sits at 0.05%, as
figures/imputation/imputation_accuracy.py places it."""
AOU = ORANGE
"""The All of Us SV imputation curve's colour, apart from the SNVs' black and the simulated SVs' blue."""


def aou_imputation():
    """The All of Us SV imputation curve, each frequency bin's geometric midpoint (%) and median r², or None when the
    git-ignored summary is missing."""
    if not AOU_IMPUTATION.exists():
        return None
    bins = json.loads(AOU_IMPUTATION.read_text())["by_frequency"]["bins"]
    return (np.array([math.sqrt((b["low"] or OPEN_LOW) * b["high"]) for b in bins]),
            np.array([b["r2"] for b in bins]))


def runs(ax, x, y, parts, **style):
    """``parts``, each (words, font style), set on one line from (``x``, ``y``) in data, centred on ``y``."""
    piece = ax.text(x, y, parts[0][0], fontstyle=parts[0][1], va="center", **style)
    for words, slant in parts[1:]:
        piece = ax.annotate(words, xy=(1, 0), xycoords=piece, va="bottom", ha="left", fontstyle=slant,
                            annotation_clip=False, **style)


def imputation_panel(ax, accuracy) -> None:
    """The r² of each variant's measured dosage with its true genotype (benchmarks/power/imputation_accuracy.py, not
    Beagle's own DR2) by minor allele frequency, SNVs called from simulated reads and SVs imputed by Beagle: each
    frequency bin's median with its approximate 95% interval (Figure 5a). Beside them, when its summary is present, the
    median r² of SVs imputed in All of Us per frequency bin, with no interval."""
    ends = []
    for key, color, label in (("SNV", INK, [("SNVs", "normal")]), ("SV", IMPUTED, [("SVs, simulated", "normal")])):
        bins = [b for b in accuracy[key]["bins"] if b["count"]]
        centers = np.array([math.sqrt(b["low"] * b["high"]) * 100 for b in bins])
        median = np.array([b["median"] for b in bins])
        half = np.array([MEDIAN_Z * (b["q75"] - b["q25"]) / math.sqrt(b["count"]) for b in bins])
        ax.vlines(centers, np.clip(median - half, 0, 1), np.clip(median + half, 0, 1), color=color, linewidth=BAR,
                  zorder=2)
        ax.plot(centers, median, color=color, marker="o", markersize=MARK * 0.85, markeredgewidth=0, zorder=3)
        ends.append((centers[-1], median[-1], label))
    real = aou_imputation()
    if real is not None:
        ax.plot(*real, color=AOU, marker="o", markersize=MARK * 0.85, markeredgewidth=0, zorder=3, clip_on=False)
        ends.append((real[0][-1], real[1][-1], [("SVs, ", "normal"), ("All of Us", "italic")]))
    top = 1.12
    ax.set_xscale("log")
    ax.set_xlim(0.05, 100)
    ax.set_ylim(0, top)
    ax.spines["left"].set_bounds(0, 1)
    # each line named after the lines' ends, from the highest end down, each name at its line's end or a line under
    # the name above
    height = ax.get_position().height * ax.figure.get_size_inches()[1]
    x, previous = max(end for end, _, _ in ends) * 1.25, math.inf
    for _, y, parts in sorted(ends, key=lambda end: -end[1]):
        previous = min(y, previous - 0.62 * top / height)
        runs(ax, x, previous, parts, color=INK, fontweight="bold")
    ax.set_xticks([0.1, 1, 10], ["0.1", "1", "10"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_yticks(np.arange(0, 1.01, 0.25), ["0", "0.25", "0.50", "0.75", "1"])
    ax.set_xlabel("Minor allele frequency (%)")
    ax.set_ylabel("r² with true genotype")
    ax.vlines(1, 0, 1, color=INK, linewidth=2.0, linestyle=(0, (2, 2)), zorder=0)
    # under the SVs' line at common frequencies: the share of SVs left of the 1% line, pointing at it
    ax.annotate(f"{accuracy['SV']['below_1pct']:.0%} of SVs\nhave MAF < 1%", xy=(1.0, 0.36), xytext=(4.2, 0.36),
                color=INK, va="center", ha="left", linespacing=1.15,
                arrowprops={"arrowstyle": "-|>", "color": INK, "linewidth": 2.4, "shrinkA": 4, "shrinkB": 0,
                            "mutation_scale": 28})


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
                  linewidth=0, label=f"{label} ({int(mask.sum()):,})")
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
        right.text(2e4, 0.08, f"{long} SVs > 10 kb,\nup to {length.max() / 1e3:,.0f} kb", color=INK, va="bottom",
                   ha="left", linespacing=1.15, transform=blended_transform_factory(right.transData, right.transAxes))
    right.set_xscale("log")
    right.set_xlim(45, 1.1e6)
    right.set_xticks([1e2, 1e3, 1e4, 1e5, 1e6], ["10²", "10³", "10⁴", "10⁵", "10⁶"])
    right.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    right.set_xlabel("SV length (bp)")
    right.set_ylabel("Number of SVs")
    right.legend(loc="upper right", handlelength=0.9, handleheight=0.9, borderaxespad=0.0)
    middle = float(np.median(length[np.any([mask for _, mask, _ in groups], axis=0)]))
    right.axvline(middle, color=INK, linewidth=2.0, linestyle=(0, (2, 2)), zorder=0)
    right.text(middle * 1.15, 0.45, f"median\n{middle:,.0f} bp", transform=blended_transform_factory(right.transData,
               right.transAxes), color=INK, va="bottom", linespacing=1.15)
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
        f"variant's measured dosage with its true genotype over all {PEOPLE:,} simulated individuals (not Beagle's DR2), "
        f"for {accuracy['SV']['records']:,} SVs imputed by Beagle and {accuracy['SNV']['records']:,} sampled SNVs "
        f"called from simulated reads on chromosome 22, by "
        f"minor allele frequency: each bin's median (points) with an approximate 95% interval ({MEDIAN_Z:.2f} IQR/√n, "
        f"bars), SNVs black and SVs blue"
        + (", beside the median r² of SVs imputed in All of Us per frequency bin (orange, no interval)"
           if aou_imputation() is not None else "")
        + ". b, Lengths of the "
        f"{int(records['measured'].sum()):,} measured SVs, stacked by class (records without a type are classed by the "
        f"sign of their length change)" + (". The insertions' peak near 300 bp is where Alu elements lie" if alu else "")
        + ". bench-sim v7 cohort from public 1000 Genomes haplotypes."
    )
    return fig, caption


# ------------------------------------------------------------------ the poster's own figures


POSTER_TYPE, POSTER_LABEL, POSTER_NAME = 56, 60, 64
"""Points at print size in the poster's figures: ticks and notes; axis titles; the bold names of methods and genotype
sources written beside their marks. A poster figure carries no legend and no title: a headline over it, or the
figure's own labels, state its message."""
POSTER_LINE, POSTER_MARK, POSTER_BAR = 8.0, 26.0, 6.0
"""Points: a data line's width, a mean's marker and an interval's line in the poster's figures."""
POSTER_HEIGHT = 9.25
"""Inches: the height of the two figures side by side at the top of the poster's middle column."""
POSTER_HALF = 10.56
"""Inches: the width each of those two figures is placed at (half of build_pptx.py's 21.92 in middle column less its
0.8 in gutter), so their type prints at its own size."""
POSTER_STYLE = {
    "font.size": POSTER_TYPE, "axes.labelsize": POSTER_LABEL, "xtick.labelsize": POSTER_TYPE,
    "ytick.labelsize": POSTER_TYPE, "axes.linewidth": 3.2, "xtick.major.width": 3.2, "ytick.major.width": 3.2,
    "xtick.major.size": 14, "ytick.major.size": 14, "xtick.major.pad": 10, "ytick.major.pad": 10,
    "axes.labelpad": 16, "lines.linewidth": POSTER_LINE, "lines.markersize": POSTER_MARK,
}


POSTER_PAD = 0.1
"""Inches between a poster figure's outermost type and its canvas's edge, and between a name and the marks beside it."""
POSTER_X_TITLES = ("Training size (thousands)", "LDpred2-auto test AUC")
"""The x axes' titles of the two figures side by side at the top of the poster's middle column."""
POSTER_SHARES = (("null", "0%", GREEN), ("sv_5pct", "5%", VIOLET), ("sv_20pct", "20%", WINE))
"""The accuracy figure's architectures by the share of causal variants that are SVs, as its key writes each, and the
colour of their simulations' dots."""
POSTER_METHODS = (("with_sv_auc", "SV-PGS", SV_PGS, "o", -1), ("ldpred2_auc", "LDpred2-auto", LDPRED2, "s", 1))
"""Each method's AUC column, its name as the figure labels it, colour, marker, and the side of its training size its
simulations' dots take. No figure carries a citation mark; the poster's text cites LDpred2-auto (Privé 2020,
build_pptx.py's REFERENCES). SV-PGS fits every SNV, indel and SV with no frequency or r² floor
(stage2_wiring.fit_models); LDpred2-auto SNVs of training MAF >= 1% thinned to HapMap3's chr22 count, at most 16,000
(benchmarks/bench_sim/submissions/ldpred2_auto.py)."""
POSTER_AUC_FLOOR = 0.47
"""The AUC every poster AUC axis starts at, under the lowest simulation's AUC, so that every dot shows at its value; the
ticks start at 0.5. A dot under it stops the build."""
DOT_OFFSET, DOT_JITTER, DOT_ALPHA = 0.12, 0.05, 0.25
"""Doublings of training size a method's simulation dots sit to its side of the size, the most a dot moves at random
about that, and the dots' opacity."""


def under_axes(fig, titles=POSTER_X_TITLES) -> float:
    """Inches under a poster figure's x axis that its ticks, tick labels and the tallest of ``titles`` take in the
    poster's style, plus POSTER_PAD; by default the top row's titles, one bottom margin for both figures, so their x
    axes line up."""
    style = matplotlib.rcParams
    return ((style["xtick.major.size"] + style["xtick.major.pad"] + style["axes.labelpad"]) / 72
            + tallest(fig, ["0.50"], fontsize=style["xtick.labelsize"])
            + tallest(fig, titles, fontsize=style["axes.labelsize"]) + POSTER_PAD)


def poster_accuracy_figure(grids, summary):
    """Figure 1a for the poster: test AUC against training size for SV-PGS (circles) and LDpred2-auto (squares), pooled
    over the architectures with 0%, 5% and 20% causal SVs. Each mean has its 95% interval as an error bar, over every
    simulation's own AUC as a faint dot in its architecture's colour, left of its size for SV-PGS and right for
    LDpred2-auto. The y axis starts at POSTER_AUC_FLOOR and ticks every 0.1 from 0.5 up to the first tick over the AUC
    of the true genetic value, a dotted line labelled "Best possible". Each method is named beside its marks where its name fits: SV-PGS over the marks its name spans from the left,
    under the best possible AUC, and LDpred2-auto under the marks its name spans to the right end, over the x axis.
    When either does not fit, both stand on one line over the colours' key, ending at the plot's right edge. The key,
    "0%, 5% or 20% causal SVs", is one line over the plot, ending at its right edge: at tick size it is wider than the
    plot."""
    grouped = cells(grids["grid"])
    fig = plt.figure(figsize=(POSTER_HALF, POSTER_HEIGHT))
    # the key's words, each share in its dots' colour: "0%, 5% or 20% causal SVs"
    shares = [(label, share) for _, label, share in POSTER_SHARES]
    causal = [shares[0], *((part, tint) for index, (label, share) in enumerate(shares[1:], 1)
                           for part, tint in ((", " if index < len(shares) - 1 else " or ", INK), (label, share))),
              (" causal SVs", INK)]
    line = tallest(fig, ["Best possible"], fontsize=POSTER_TYPE)
    # the top margin holds the key over the plot; the left margin the ticks and the axis title
    top, bottom, left, right = 0.25 + line + POSTER_PAD, under_axes(fig), widest(fig, ["0.0"], fontsize=POSTER_TYPE) + 1.45, 0.3
    width, height = POSTER_HALF - left - right, POSTER_HEIGHT - top - bottom
    ax = frame(fig, left, bottom, width, height)
    log_size_axis(ax)
    # the title runs longer than the axis, so it ends at the axis's far end
    ax.set_xlabel(POSTER_X_TITLES[0], loc="right")
    first, last = ax.get_xlim()
    doubling = width / math.log2(last / first)
    random = np.random.default_rng(0)
    floor, step = POSTER_AUC_FLOOR, 0.1
    marks = []
    for key, _, color, marker, side in POSTER_METHODS:
        for point, _, share in POSTER_SHARES:
            for size in SIZES:
                values = finite(grouped.get((point, size), []), key)
                if values and min(values) < floor:
                    raise SystemExit(f"poster_accuracy_by_training_size: a {key} AUC of {min(values):.3f} is under "
                                     f"POSTER_AUC_FLOOR, {floor}")
                across = size * 2.0 ** (side * DOT_OFFSET + random.uniform(-DOT_JITTER, DOT_JITTER, len(values)))
                ax.plot(across, values, marker, color=share, alpha=DOT_ALPHA, markersize=POSTER_MARK * 0.5,
                        markeredgewidth=0, linestyle="none", zorder=2)
                marks.extend(zip(across, values))
        pooled = [interval(finite([row for point, *_ in POSTER_SHARES for row in grouped.get((point, size), [])], key))
                  for size in SIZES]
        mean, low, high = (np.array([cell[i] for cell in pooled]) for i in range(3))
        # marks smaller than the poster's usual, so each interval shows past its mark
        ax.errorbar(SIZES, mean, yerr=(mean - low, high - mean), color=color, elinewidth=POSTER_BAR, capsize=0,
                    marker=marker, markersize=POSTER_MARK * 0.7, markeredgewidth=0, zorder=3)
        marks.extend([*zip(SIZES, low), *zip(SIZES, high)])
    marks = np.asarray(marks)
    ceiling = summary["points"]["null"]["true_value_auc"]
    levels = np.arange(0.5, ceiling + step, step)
    style = {"fontsize": POSTER_NAME, "fontweight": "bold"}
    ours, theirs = (method for _, method, *_ in POSTER_METHODS)
    tall = {method: tallest(fig, [method], **style) for method in (ours, theirs)}
    # a name keeps POSTER_PAD from the edge of every mark, the means' marks reaching furthest from their centres, and
    # counts the marks within ``aside`` inches of its ends as under it, so that none stands at its side like a bullet
    reach = POSTER_MARK * 0.7 / 2 / 72
    clear, aside = POSTER_PAD + reach, 0.5
    start = SIZES[0] / 1.2

    def spanned(method, flush_right):
        """The AUCs of the marks a name spans, set flush left from ``start`` or flush right at the axis's end."""
        wide = 2.0 ** ((widest(fig, [method], **style) + aside) / doubling)
        low, high = (last / wide, last) if flush_right else (start / 2.0 ** (aside / doubling), start * wide)
        return marks[(marks[:, 0] >= low) & (marks[:, 0] <= high), 1]

    def scale(height):
        """AUC per inch on a plot ``height`` inches tall: room over the best possible AUC for its label and pads, and
        over the top tick for half its label."""
        return max((ceiling - floor) / (height - above), (levels[-1] - floor) / (height - half))

    above = POSTER_PAD + line + POSTER_PAD
    half = tallest(fig, ["0.5"], fontsize=POSTER_TYPE) / 2
    per_inch = scale(height)
    over = spanned(ours, False).max() + clear * per_inch
    under = spanned(theirs, True).min() - clear * per_inch
    if (over + (tall[ours] + POSTER_PAD) * per_inch <= ceiling
            and under - (tall[theirs] + POSTER_PAD) * per_inch >= floor):
        ax.text(start, over, ours, color=SV_PGS, ha="left", va="bottom", **style)
        ax.text(1, under, theirs, transform=shifted(ax, 0.0), color=LDPRED2, ha="right", va="top", **style)
        names = 0.0
    else:
        # both names on a line at the top, the key under it, and the plot shorter by the line
        names = max(tall.values()) + POSTER_PAD
        height -= names
        ax.set_position((left / POSTER_HALF, bottom / POSTER_HEIGHT, width / POSTER_HALF, height / POSTER_HEIGHT))
        per_inch = scale(height)
        gap = widest(fig, ["  "], **style)
        end = left + width
        for method, color in ((theirs, LDPRED2), (ours, SV_PGS)):
            fig.text(end / POSTER_HALF, 1 - 0.25 / POSTER_HEIGHT, method, color=color, ha="right", va="top", **style)
            end -= widest(fig, [method], **style) + gap
    across = (left + width - widest(fig, ["".join(words for words, _ in causal)], fontsize=POSTER_TYPE)) / POSTER_HALF
    piece = fig.text(across, 1 - (0.25 + names) / POSTER_HEIGHT, causal[0][0], color=causal[0][1], va="top", ha="left")
    for words, color in causal[1:]:
        piece = ax.annotate(words, xy=(1, 0), xycoords=piece, va="bottom", ha="left", color=color,
                            annotation_clip=False)
    ax.set_ylim(floor, floor + height * per_inch)
    ax.axhline(ceiling, color=INK, linestyle=(0, (1, 1.2)), linewidth=4.0, zorder=1)
    ax.text(SIZES[0] / 1.25, ceiling + POSTER_PAD * per_inch, "Best possible", color=INK, va="bottom", ha="left")
    ax.set_yticks(levels, [plain(level, 1) for level in levels])
    # the title runs longer than the axis, so it ends at the axis's top and runs on into the margin under it
    ax.set_ylabel("Test AUC", loc="top")
    return fig


WINS_ABOVE, WINS_INDENT, WINS_CLEAR = 0.2, 0.18, 0.4
"""Inches: how far the scatter's line at the top left reaches over its axes, how far right of the y axis it starts,
and how far under it the line of equal AUC stops."""


def poster_wins_figure(grids):
    """SV-PGS's test AUC against LDpred2-auto's in every simulation at the largest training size, over the line of equal
    AUC: one dot per simulation (black with no causal SVs), with the 95% interval of the paired AUC difference drawn
    through it, so a dot above the line is a simulation in which SV-PGS predicted better. Both axes start at
    POSTER_AUC_FLOOR and tick every 0.1 from 0.5."""
    size = SIZES[-1]
    grouped = cells(grids["grid"])
    rows = [row for point in LABELS for row in grouped.get((point, size), [])
            if row.get("delta_auc_vs_ldpred2") is not None and math.isfinite(row["delta_auc_vs_ldpred2"])]
    theirs = np.asarray([row["ldpred2_auc"] for row in rows])
    gain = np.asarray([row["delta_auc_vs_ldpred2"] for row in rows])
    ours = theirs + gain
    # the paired test's standard error is the gain over its z
    half = stats.norm.ppf(0.975) * np.abs(gain / np.asarray([row["z_vs_ldpred2"] for row in rows]))
    null = np.asarray([row["point"] == "null" for row in rows])
    # the same gain in the 3 Mb window, where SV-PGS and LDpred2-auto see far fewer variants
    window = [row["delta_auc_vs_ldpred2"] for point in LABELS for row in cells(grids["window"]).get((point, size), [])
              if row.get("delta_auc_vs_ldpred2") is not None]
    NUMBERS["poster_wins"] = {"train_count": size, "replicates": len(rows), "wins": int((gain > 0).sum()),
                              "clear": int((gain - half > 0).sum()), "null_replicates": int(null.sum()),
                              "architectures": len({row["point"] for row in rows}),
                              "gain": interval(gain), "null_gain": interval(gain[null]), "window_gain": interval(window)}
    fig = plt.figure(figsize=(POSTER_HALF, POSTER_HEIGHT))
    ticks = widest(fig, ["0.00"], fontsize=POSTER_TYPE)
    top, bottom, left = 0.25, under_axes(fig), ticks + 1.6
    width, height = POSTER_HALF - left - 0.3, POSTER_HEIGHT - top - bottom
    ax = frame(fig, left, bottom, width, height)
    # the training size over the dots at the top left, reaching WINS_ABOVE inches into the top margin
    note = f"Training N = {size:,}"
    need = tallest(fig, [note], fontsize=POSTER_TYPE) - WINS_ABOVE + POSTER_PAD
    # one scale on both axes, so the line of equal AUC runs at 45 degrees: up the y axis from POSTER_AUC_FLOOR to at
    # least 0.705, and far enough that the note clears the highest bar, and along the x axis at the same scale; an
    # interval may run under the floor, a dot may not
    low, reach = POSTER_AUC_FLOOR, (ours + half).max()
    if min(theirs.min(), ours.min()) < low:
        raise SystemExit(f"poster_wins: an AUC of {min(theirs.min(), ours.min()):.3f} is under POSTER_AUC_FLOOR, {low}")
    upper = max(0.705, (reach * height - low * need) / (height - need))
    right = low + (upper - low) / height * width
    # the line of equal AUC stops WINS_CLEAR inches under the note at the top left
    end = upper - (need - POSTER_PAD + WINS_CLEAR) * (upper - low) / height
    ax.plot([low, end], [low, end], color=INK, linewidth=4.0, zorder=1)
    for chosen, color in ((~null, SV_PGS), (null, INK)):
        ax.vlines(theirs[chosen], ours[chosen] - half[chosen], ours[chosen] + half[chosen], color=color,
                  linewidth=POSTER_BAR * 0.5, zorder=2)
        ax.plot(theirs[chosen], ours[chosen], "o", markersize=POSTER_MARK * 0.8, color=color, markeredgewidth=0,
                linestyle="none", zorder=3)
    ax.set_xlim(low, right)
    ax.set_ylim(low, upper)
    ax.set_aspect("equal")
    # a tick every 0.1, the x axis's last one far enough from its end that its label stays within the 0.3 in margin
    inches = height / (upper - low)
    for axis, levels in ((ax.xaxis, [level for level in np.arange(0.5, right, 0.1)
                                     if (right - level) * inches + 0.3 >= ticks / 2]),
                         (ax.yaxis, np.arange(0.5, upper, 0.1))):
        axis.set_ticks(levels, [plain(level, 1) for level in levels])
    # the x axis's title runs longer than its axis, so it ends at the axis's far end
    ax.set_xlabel(POSTER_X_TITLES[1], loc="right")
    ax.set_ylabel("SV-PGS test AUC")
    ax.text(0, 1, note, transform=shifted(ax, WINS_INDENT, WINS_ABOVE, data_rows=False), color=INK, ha="left",
            va="top")
    # under the line, where no dot falls: what a dot and its bar are at the foot, the key to the black dots just
    # over them
    foot = "Dot: simulation\nBar: 95% CI"
    ax.text(1, 0, foot, transform=shifted(ax, -POSTER_PAD, POSTER_PAD, data_rows=False), color=INK, ha="right",
            va="bottom", linespacing=1.0)
    over = POSTER_PAD + tallest(fig, [foot], fontsize=POSTER_TYPE, linespacing=1.0) + 0.15
    ax.text(1, 0, "Black: no\ncausal SVs", transform=shifted(ax, -POSTER_PAD, over, data_rows=False), color=INK,
            fontweight="bold", ha="right", va="bottom", linespacing=1.0)
    return fig


WELL_IMPUTED = ("tagged_sv", "tagged_sv_fold3")
"""The architectures whose causal SVs are drawn only from SVs that Beagle imputed with realized r² ≥ 0.8 (grid.py's
sv_min_realized_r2)."""
POSTER_GAIN, POSTER_GENOTYPES = POSTER_HALF, POSTER_HALF
"""Inches: the widths of the SV gain forest and the imputation figure, side by side in the lower row of the poster's
middle column (build_pptx.py's layout), each placed at that width so its type prints at its own size."""
POSTER_GAIN_HEIGHT = 8.52
"""Inches: the SV gain forest's height. Its rows share what the key and the x axis leave, and the build stops when that
is less than the rows need."""
POSTER_GENOTYPES_HEIGHT = POSTER_GAIN_HEIGHT
"""Inches: the imputation figure's height, the SV gain forest's, so the two sit flush side by side."""
GAIN_WEIGHT = 1.0
"""The SV gain forest's marks and intervals relative to the poster's (POSTER_MARK, POSTER_BAR)."""
GAIN_TITLE = "Test AUC gain from SVs"
"""The SV gain forest's x axis title, short enough for one line."""
FOREST_SIZES = SIZES[:3]
"""The training sizes the SV gain forest averages over, 5,000 to 20,000. Both genotype sources reached 40,000 in one
simulation only, a control one, and the gain grows with training size: with it, the control's gain with imputed SVs
was 0.00068 (95% CI 0.00002 to 0.00134), and without it 0.00022 (-0.00084 to 0.00128)."""


def gain_sets(grids) -> dict[str, dict]:
    """The SV gain with true genotypes and with called SNVs and imputed SVs over simulated truths
    (benchmarks/power/truths.py) at FOREST_SIZES, for every architecture with causal SVs, those whose causal SVs are
    well imputed, and the control with no causal SVs: each set's label, architecture count, mean gain per source with
    its t interval and sign-flip p, and the imputed gain's share of the true gain with Fieller's interval."""
    gains = truth_gains(*([row for row in grids[grid] if int(row["train_count"]) in FOREST_SIZES]
                          for grid in ("window-truth", "window")))
    causal = [point for point in LABELS if point != "null" and point in gains]
    sets = {}
    for key, label, points in (
        ("all", "With causal SVs", causal),
        ("well_imputed", "Causal SVs with\nr² ≥ 0.8", [point for point in causal if point in WELL_IMPUTED]),
        ("null", "No causal SVs\n(control)", ["null"]),
    ):
        true = [gain for point in points for _, gain, _ in gains.get(point, [])]
        imputed = [gain for point in points for _, _, gain in gains.get(point, [])]
        sets[key] = {"label": label, "architectures": len(points), "true": mean_test(true),
                     "imputed": mean_test(imputed), "share": fieller(imputed, true)}
    # the imputed gain where causal SVs are well imputed less where they are not, the two sets disjoint
    rest = [gain for point in causal if point not in WELL_IMPUTED for _, _, gain in gains.get(point, [])]
    well = [gain for point in causal if point in WELL_IMPUTED for _, _, gain in gains.get(point, [])]
    sets["well_imputed"]["versus_rest"] = difference(well, rest)
    return sets


def gain_forest(rows: list[dict], bold: dict):
    """The SV gain forest of ``poster_sv_gain_figure``, POSTER_GAIN wide and POSTER_GAIN_HEIGHT tall, ``bold``'s label
    in bold. Each row's true-genotype interval stands over its imputed one, their marks a pad apart, and the rows stand
    as close as their marks and labels allow, a pad apart. The key names the two genotype sources after marks like
    their own, stacked over the plot and ending at its right edge. A POSTER_GAIN_HEIGHT under what the key, the rows and
    the x axis need stops the build with that height; over it, the rows spread out."""
    top, right, height = POSTER_PAD, 0.5, POSTER_GAIN_HEIGHT
    mark, bar, ring = POSTER_MARK * GAIN_WEIGHT, POSTER_BAR * GAIN_WEIGHT, 6.0 * GAIN_WEIGHT
    fig = plt.figure(figsize=(POSTER_GAIN, height))
    style = dict(fontsize=POSTER_TYPE, fontweight="bold", linespacing=1.0)
    # inches: a mark's radius, its ring included, and from a key mark's centre to its name
    reach = (mark + ring) / 2 / 72
    gap = reach + 2 * POSTER_PAD
    # the key's names, wrapped after "and", each over a pad
    names = [label.replace(" and ", " and\n") for label, _, _ in SOURCES]
    heights = [tallest(fig, [name], **style) for name in names]
    key = sum(heights) + len(names) * POSTER_PAD
    bottom = under_axes(fig, [GAIN_TITLE])
    labels = [row["label"] for row in rows]
    left = widest(fig, labels, **style) + 0.75
    # inches down the plot to each row's centre: a pad over the first mark, a pad between the marks of neighbouring
    # rows and between their labels, and a pad under the last mark
    offset = reach + POSTER_PAD / 2
    tall = [tallest(fig, [label], **style) for label in labels]
    centres = [POSTER_PAD + reach + offset]
    for upper, lower in zip(tall, tall[1:]):
        centres.append(centres[-1] + max(2 * (offset + reach) + POSTER_PAD, (upper + lower) / 2 + POSTER_PAD))
    span = centres[-1] + offset + reach + POSTER_PAD
    need = top + key + span + bottom
    if height < need - 0.005:
        raise SystemExit(f"poster_sv_gain: POSTER_GAIN_HEIGHT is {height} in, under the {need:.2f} in its rows need")
    plot = height - top - key - bottom
    centres = [centre * plot / span for centre in centres]
    width = POSTER_GAIN - left - right
    ax = frame(fig, left, bottom, width, plot)
    ends = [row[source][end] for row in rows for source in ("true", "imputed") for end in ("lower", "upper")
            if math.isfinite(row[source][end])]
    low, high = min(0.0, *ends), max(ends)
    limits = (low - 0.05 * (high - low), high + 0.05 * (high - low))
    for y, row in zip(centres, rows):
        for shift, source, (_, color, filled) in ((-offset, "true", SOURCES[0]), (offset, "imputed", SOURCES[1])):
            test = row[source]
            if math.isfinite(test["lower"]):
                ax.hlines(y + shift, test["lower"], test["upper"], color=color, linewidth=bar, zorder=3,
                          capstyle="round")
            ax.plot([test["mean"]], [y + shift], "o", markersize=mark, color=color,
                    markerfacecolor=color if filled else "white", markeredgecolor=color,
                    markeredgewidth=0 if filled else ring, zorder=4)
        ax.text(0, y, row["label"], transform=shifted(ax, -0.3), ha="right", va="center", linespacing=1.0,
                fontweight="bold" if row is bold else "normal")
    ax.axvline(0, color=INK, linewidth=4.0, zorder=2)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    # the y axis in inches down the plot
    ax.set_ylim(plot, 0)
    ax.set_xlim(*limits)
    # the finest tick step whose labels stand 0.3 in apart
    wide = widest(fig, ["0.000"], fontsize=POSTER_TYPE) + 0.3
    step = next(step for step in (0.001, 0.002, 0.004, 0.005, 0.01, 0.02)
                if step * width / (limits[1] - limits[0]) >= wide)
    ticks = np.arange(math.ceil(limits[0] / step) * step, limits[1], step)
    ax.set_xticks(ticks, [plain(tick, 3) for tick in ticks])
    # the title runs longer than the axis, so it ends at the axis's far end
    ax.set_xlabel(GAIN_TITLE, loc="right")
    # the key over the plot, the names ending at its right edge, each mark before its name and level with its first
    # line
    line = tallest(fig, ["Called"], **style)
    across, up = POSTER_GAIN - right - widest(fig, names, **style), height - top
    for (label, color, filled), name, tall_name in zip(SOURCES, names, heights):
        ax.plot([across - gap], [up - line / 2], "o", transform=fig.dpi_scale_trans, markersize=mark, color=color,
                markerfacecolor=color if filled else "white", markeredgecolor=color,
                markeredgewidth=0 if filled else ring, clip_on=False, zorder=4)
        fig.text(across, up, name, transform=fig.dpi_scale_trans, color=color, va="top", ha="left", **style)
        up -= tall_name + POSTER_PAD
    return fig


def poster_sv_gain_figure(grids):
    """The AUC gain from adding SVs with true genotypes and with called SNVs and imputed SVs over simulated truths
    (``gain_sets``), a row per set with its label at the left and the true gain's mean and 95% interval over the imputed gain's; the label of
    the set with well-imputed causal SVs in bold. Returns the figure and its one-sentence description."""
    sets = gain_sets(grids)

    def entry(test):
        return [test["mean"], test["lower"], test["upper"], test["truths"]]

    NUMBERS["poster_sv_gain"] = {
        "shown": len({row["point"] for row in grids["window-truth"]} & {row["point"] for row in grids["window"]}),
        "rows": {key: {"architectures": row["architectures"], "true": entry(row["true"]), "p_true": row["true"]["p"],
                       "imputed": entry(row["imputed"]), "p_imputed": row["imputed"]["p"],
                       "share": [None if math.isinf(value) else value for value in row["share"]],
                       **({"versus_rest": list(row["versus_rest"])} if "versus_rest" in row else {})}
                 for key, row in sets.items()},
    }
    rows = list(sets.values())
    fig = gain_forest(rows, sets["well_imputed"])
    description = (
        "Test AUC gain from adding SVs to SV-PGS with true genotypes (filled) and with SNVs called from simulated reads "
        "and SVs imputed by Beagle (open), mean and 95% interval over simulations at training N 5,000 to 20,000, for "
        "every architecture with causal SVs, those whose causal SVs are well imputed, and the control with no causal "
        "SVs."
    )
    return fig, description


def poster_imputation_figure(summary):
    """The genotype accuracy figure beside the SV gain forest in the lower row of the poster's middle column: each frequency bin's mean r² of the
    measured genotype with the true genotype, for SNVs called from simulated reads and SVs imputed by Beagle, with a
    conservative 95% interval of the mean, ±1.96 √(mean (1 − mean) / n): an r² lies in [0, 1], so its variance is at
    most mean (1 − mean), and the summary keeps each bin's mean and count, not its values. Beside them, when its summary
    is present, the median r² of SVs imputed in All of Us per frequency bin, with no interval. The names stand in a
    column after the lines' last marks, each level with its line's end where the names above leave room. The margins are measured from the drawn type, so larger type widens them. The y axis's
    title is centred on the figure's height, and the x axis's, wider than the axes, ends at the figure's right edge."""
    accuracy = summary["genotype_accuracy"]["r2_by_frequency"]
    width, height, pad = POSTER_GENOTYPES, POSTER_GENOTYPES_HEIGHT, 0.08
    fig = plt.figure(figsize=(width, height))
    ax = frame(fig, 0, 0, width, height)
    mark = POSTER_MARK * 0.7
    ends = []
    for key, color in (("SNV", INK), ("SV", IMPUTED)):
        bins = [b for b in accuracy[key]["bins"] if b["count"]]
        centers = np.array([math.sqrt(b["low"] * b["high"]) * 100 for b in bins])
        mean = np.array([b["mean"] for b in bins])
        half = 1.96 * np.sqrt(mean * (1 - mean) / np.array([b["count"] for b in bins]))
        ax.vlines(centers, np.clip(mean - half, 0, 1), np.clip(mean + half, 0, 1), color=color, linewidth=POSTER_BAR,
                  zorder=2, clip_on=False)
        ax.plot(centers, mean, color=color, marker="o", markersize=mark, markeredgewidth=0, zorder=3, clip_on=False)
        ends.append(mean[-1])
    real = aou_imputation()
    if real is not None:
        ax.plot(*real, color=AOU, marker="o", markersize=mark, markeredgewidth=0, zorder=3, clip_on=False)
        ends.append(real[1][-1])
    # each line named in a column after the lines' last marks: a bold name over where its genotypes come from
    after = mark / 2 / 72 + 0.15
    labels = []
    for (name, below, slant), color, end in zip((("SNVs", None, "normal"), ("SVs", "simulated", "normal"),
                                                  ("SVs", "All of Us", "italic")), (INK, IMPUTED, AOU), ends):
        parts = [(ax.text(1, 1, name, color=color, fontsize=POSTER_NAME, fontweight="bold", ha="left", va="top"), 0.0)]
        first = tallest(fig, [name], fontsize=POSTER_NAME, fontweight="bold")
        total = first
        if below:
            parts.append((ax.text(1, 1, below, color=color, fontstyle=slant, ha="left", va="top"), first))
            total += tallest(fig, [below], fontsize=POSTER_TYPE, fontstyle=slant)
        labels.append((end, first, total, parts))

    def place():
        """From the highest line end down, each name's first line centred on its line's end, or a pad under the name
        above when that is lower."""
        tall, previous = ax.get_position().height * height, math.inf
        for end, first, total, parts in sorted(labels, key=lambda label: -label[0]):
            previous = min(end * tall + first / 2, previous)
            for artist, down in parts:
                artist.set_transform(shifted(ax, after, previous - down - tall, False))
            previous -= total + POSTER_PAD

    ax.set_xscale("log")
    ax.set_xlim(0.04, 50)
    ax.set_ylim(0, 1)
    ax.set_xticks([0.1, 1, 10], ["0.1", "1", "10"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.set_xlabel("Minor allele frequency (%)")
    ax.set_ylabel("Genotype r²")
    # each margin is what the type reaches past the axes, plus a pad, measured again until it settles
    renderer = fig.canvas.get_renderer()
    inches = fig.dpi_scale_trans.inverted()
    for _ in range(6):
        place()
        outer = ax.get_tightbbox(renderer).transformed(inches)
        inner = ax.get_window_extent(renderer).transformed(inches)
        left, bottom = inner.x0 - outer.x0 + pad, inner.y0 - outer.y0 + pad
        right, above = outer.x1 - inner.x1 + pad, outer.y1 - inner.y1 + pad
        across, up = width - left - right, height - bottom - above
        ax.set_position((left / width, bottom / height, across / width, up / height))
        ax.yaxis.label.set_y((height / 2 - bottom) / up)
        title = ax.xaxis.label.get_window_extent(renderer).transformed(inches).width
        ax.xaxis.label.set_x((width - pad - title / 2 - left) / across)
    place()
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
                markersize=MARK * (0.85 if marker == "s" else 1.0), markeredgewidth=0, zorder=3 if label else 4)
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
    handles = [Line2D([0], [0], marker=marker, color=SV_PGS, markeredgewidth=0,
                      markersize=MARK * (0.85 if marker == "s" else 1.0), linewidth=3.0, linestyle=style, label=label)
               for label, marker, style in (("with SVs", "o", "-"), ("without SVs", "s", (0, (2, 1.2))))]
    # in the band between the methods: the key to the SV-PGS marks
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.02, 0.62), handlelength=2.0, handletextpad=0.4,
              borderaxespad=0.0, labelspacing=0.3)
    heading(ax, "Time to fit chromosome 22", across=-1.95)
    caption = (
        "Figure 6 | SV-PGS fits take hours and LDpred2-auto's minutes, and SVs add no time. Wall time of one fit on "
        "chromosome 22 per method and training size, pooled over every architecture: median and its distribution-free "
        f"95% interval ({min(counts)}-{max(counts)} fits each, capped bars, most within the marks). MSI Slurm jobs, one "
        "fit per job. SV-PGS fits every variant by empirical Bayes (circles and solid line with SVs, squares and dashed "
        "line without), LDpred2-auto HapMap3-density SNVs. The results do not record each job's hardware, so the gap "
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
VARIANTS = (("SNV", INK, 0.09), ("indel", GREEN, 0.3), ("SV", IMPUTED, 1.1))
"""The variant classes as the left column draws them: name, colour, and the width in inches of the block that stands for
one variant on a chromosome (one base for an SNV, a few for an indel, many for an SV)."""
SCHEMATICS = {
    "poster_model": "An SNV, an indel and an SV on a chromosome, each class with its own effect size prior, all fitted "
                    "together into SV-PGS.",
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


def poster_model_figure():
    """The SV-PGS model for the poster's Methods: an SNV, an indel and an SV on a chromosome, each class with its own
    effect size prior, all fitted together into one score."""
    height = 4.15
    fig, ax = canvas(height)
    # left to right, in inches: the class names, a stretch of chromosome after the widest, the priors, the arrows, and
    # the score's box as wide as its name
    name_style = {"fontsize": POSTER_NAME, "fontweight": "bold"}
    stretch = EDGE + widest(fig, [name for name, *_ in VARIANTS], **name_style) + 0.2
    stretch = (stretch, stretch + 1.45)
    box_left = COLUMN - EDGE - widest(fig, ["SV-PGS"], **name_style) - 0.3
    box_middle, box_tall = 1.6, 1.25
    centre, reach = (stretch[1] + 0.2 + box_left - 1.1) / 2, (box_left - 1.1 - stretch[1] - 0.2) / 2
    bases = (2.3, 1.2, 0.1)
    # each class's prior SD in inches, a schematic (empirical Bayes learns the real ones); every prior encloses the
    # same area, so a narrower one stands taller, the SNV's 0.75 in
    spreads = (0.2, 0.28, 0.5)
    area = 0.75 * spreads[0] * math.sqrt(2 * math.pi)
    effect = np.linspace(-reach, reach, 400)
    ax.text(centre, height - 0.04, "Effect size prior", ha="center", va="top", color=INK)
    for (name, colour, block), base, spread in zip(VARIANTS, bases, spreads):
        middle = base + 0.4
        ax.text(EDGE, middle, name, ha="left", va="center", color=colour, **name_style)
        # the variant, a block as wide as the class is long, on a black line for the chromosome
        ax.plot(stretch, [middle, middle], color=INK, linewidth=3.2, solid_capstyle="round", zorder=1)
        ax.add_patch(matplotlib.patches.Rectangle((sum(stretch) / 2 - block / 2, middle - 0.28), block, 0.56,
                                                  facecolor=colour, edgecolor="none", zorder=2))
        ax.plot([centre - reach, centre + reach], [base, base], color=INK, linewidth=3.2, zorder=1,
                solid_capstyle="round")
        # each curve drawn where it stands clear of its baseline, so the tails leave the baseline bare
        density = area * stats.norm.pdf(effect, scale=spread)
        shown = density > 0.03
        ax.plot(centre + effect[shown], base + density[shown], color=colour, linewidth=6.0, zorder=3,
                solid_capstyle="round")
        # every class into the one model
        ax.annotate("", xy=(box_left - 0.08, box_middle + (middle - box_middle) * 0.4),
                    xytext=(centre + reach + 0.15, middle),
                    arrowprops={"arrowstyle": "-|>,head_length=0.55,head_width=0.28", "color": INK, "linewidth": 4.0,
                                "shrinkA": 0, "shrinkB": 0})
    ax.add_patch(matplotlib.patches.FancyBboxPatch((box_left, box_middle - box_tall / 2), COLUMN - EDGE - box_left,
                                                   box_tall, boxstyle="round,pad=0,rounding_size=0.2",
                                                   facecolor=SV_PGS, edgecolor="none"))
    ax.text((box_left + COLUMN - EDGE) / 2, box_middle, "SV-PGS", ha="center", va="center", color="white", **name_style)
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
        gain, gain_description = poster_sv_gain_figure(grids)
        for name, panel in (("poster_accuracy_by_training_size", poster_accuracy_figure(grids, summary)),
                            ("poster_wins", poster_wins_figure(grids)),
                            ("poster_sv_gain", gain),
                            ("poster_imputation_accuracy", poster_imputation_figure(summary)),
                            ("poster_model", poster_model_figure())):
            reach = dict(zip(("left", "bottom", "right", "top"), overhang(panel)))
            if max(reach.values()) > 0:
                raise SystemExit(f"{name}: its type runs past the canvas: "
                                 + ", ".join(f"{side} {inches:.2f} in" for side, inches in reach.items() if inches > 0))
            # transparent, so the page shows through: an opaque white ground renders a shade off the page white in a PDF
            panel.savefig(OUT / f"{name}.png", transparent=True)
            plt.close(panel)
    for name, words in (*SCHEMATICS.items(), ("poster_sv_gain", gain_description)):
        (OUT / f"{name}.txt").write_text(words + "\n")
    text = json.dumps(rounded(NUMBERS), indent=1, ensure_ascii=False)
    # each [mean, lower, upper, n] on one line
    text = re.sub(r"\[\s*([^\[\]{}]*?)\s*\]", lambda match: "[" + " ".join(match.group(1).split()) + "]", text)
    (OUT / "numbers.json").write_text(text + "\n")
    print(OUT)


if __name__ == "__main__":
    main()

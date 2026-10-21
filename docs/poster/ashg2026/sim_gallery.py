"""Journal-style figures of the simulation study run on MSI (benchmarks/power/grid.py on bench-sim v7's 1000 Genomes
chr22 cohort, Beagle-imputed arm). Nothing here is from All of Us.

Every value is read from the replicate results copied from MSI (simulation.py's LOCAL tree: grid, grid-truth, window,
window-truth), and from simulation_summary.json for the AUC at full h² and the imputation accuracy bins.

    uv run --with matplotlib python docs/poster/ashg2026/sim_gallery.py [output directory]

Figures are sized in millimetres for print (single column 89 mm, double 183 mm) with 11-12.5 pt type. Each is written as a
vector PDF and a 600 dpi PNG; gallery.pdf holds every figure on its own page above its legend.
"""
from __future__ import annotations

import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
from scipy import stats

HERE = Path(__file__).resolve().parent
LOCAL = Path.home() / "svpgs-bench" / "poster-grid2"
SUMMARY = HERE / "simulation_summary.json"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "figures" / "gallery"
SIZES = (5_000, 10_000, 20_000, 40_000)
MM = 1 / 25.4

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


def _font() -> str:
    names = {entry.name for entry in font_manager.fontManager.ttflist}
    return next((name for name in ("Helvetica", "Arial", "Helvetica Neue") if name in names), "DejaVu Sans")


plt.rcParams.update({
    "font.family": _font(),
    "font.size": 11.5,
    "axes.labelsize": 12.5,
    "axes.titlesize": 12.5,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.linewidth": 1.2,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelpad": 5,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 4.5,
    "ytick.major.size": 4.5,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.color": INK,
    "ytick.color": INK,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.borderaxespad": 0.2,
    "figure.dpi": 150,
    "savefig.dpi": 600,
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
GROUPS = (
    ("Causal variants that are SVs", ("null", "sv_5pct", "sv_20pct")),
    ("SV effect size and shape", ("sv_5pct_fold3", "sv_20pct_fold3", "sv_heavy_tail", "sv_two_point")),
    ("SV effect by SV feature", ("length_positive", "deletion_positive", "multi_feature_positive")),
    ("Which SVs are causal", ("tagged_sv", "tagged_sv_fold3", "common_sv")),
    ("Trait", ("sparse_sv", "polygenic_sv", "low_heritability", "rare_disease")),
)
ARCHITECTURES = (
    "Unless named, an architecture has liability h² = 0.1, prevalence 10%, 200 causal variants and 20% causal SVs, with "
    "SV effects drawn from the same distribution as small-variant effects; "
    "'3× SV effect' triples the SV effect SD, 'well-imputed' draws causal SVs only from those Beagle imputes with "
    "r² ≥ 0.8 and 'common' only from SVs of frequency ≥ 5%."
)


def shade(color: str, amount: float) -> tuple:
    """``color`` mixed with white by ``amount`` (0 keeps it, 1 is white)."""
    rgb = np.array(to_rgb(color))
    return tuple(rgb + (1 - rgb) * amount)


def thousands(count: int) -> str:
    return f"{count // 1000}k"


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


def letter(ax, text: str, x: float = -0.16, y: float = 1.04) -> None:
    ax.text(x, y, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=14, fontweight="bold")


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
    for heading, points in GROUPS:
        kept = [point for point in points if present(point)]
        if kept:
            rows.append(("heading", heading))
            rows.extend(("point", point) for point in kept)
    return rows


def row_labels(ax, rows, start: float) -> None:
    """Group headings in bold and architectures indented under them, left-aligned at axes x ``start``."""
    place = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    for y, (kind, value) in enumerate(rows):
        if kind == "heading":
            ax.text(start, y, value, transform=place, ha="left", va="center", fontweight="bold")
        else:
            ax.text(start + 0.04, y, LABELS[value], transform=place, ha="left", va="center")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_ylim(len(rows) - 0.5, -0.7)


# ------------------------------------------------------------------ Figure 1


SV_SHARES = (("null", "0%", INK), ("sv_5pct", "5%", "#767676"), ("sv_20pct", "20%", "#AAAAAA"))


def log_size_axis(ax) -> None:
    """Training size on a true log2 axis, ticked at the grid's sizes only."""
    ax.set_xscale("log", base=2)
    ax.set_xticks(SIZES, [thousands(size) for size in SIZES])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlim(SIZES[0] / 1.3, SIZES[-1] * 1.3)


def accuracy_figure(grids, summary):
    """(a) Test AUC against training size per method, pooled over 0%, 5% and 20% causal SVs; (b) the paired gap between
    the methods for each share of causal SVs, which shows the pooling hides nothing."""
    grouped = cells(grids["grid"])
    fig, (ax, gap) = plt.subplots(1, 2, figsize=(200 * MM, 86 * MM), gridspec_kw={"width_ratios": [1.25, 1.0]})
    fig.subplots_adjust(left=0.095, right=0.985, top=0.90, bottom=0.17, wspace=0.62)
    log_size_axis(ax)
    ceiling = summary["points"]["null"]["true_value_auc"]
    ax.hlines(ceiling, SIZES[0] / 1.3, SIZES[-1] * 1.06, color=GREY, linestyle=(0, (1, 1.5)), linewidth=1.6, zorder=4)
    ax.text(SIZES[-1] * 1.12, ceiling, "h²", color=GREY, fontsize=12.5, va="center", ha="left")
    counts = {}
    for key, method, color in (("with_sv_auc", "SV-PGS", BLUE), ("ldpred2_auc", "LDpred2-auto", VERMILLION)):
        pooled = [interval([row[key] for point, _, _ in SV_SHARES for row in grouped.get((point, size), [])])
                  for size in SIZES]
        mean, low, high = (np.array([cell[i] for cell in pooled]) for i in range(3))
        counts[method] = [cell[3] for cell in pooled]
        ax.fill_between(SIZES, low, high, color=color, alpha=0.18, linewidth=0, zorder=1)
        ax.plot(SIZES, mean, color=color, linewidth=3, zorder=3, marker="o", markersize=5.8,
                markeredgecolor="white", markeredgewidth=0.8)
        ax.text(SIZES[-1] * 1.12, mean[-1], method, color=color, fontsize=12.5, fontweight="bold", va="center",
                ha="left")
    ax.set_ylim(0.52, 0.68)
    ax.set_yticks(np.arange(0.52, 0.681, 0.04))
    ax.set_xlabel("Training sample size")
    ax.set_ylabel("Test AUC")
    letter(ax, "a", x=-0.2)

    log_size_axis(gap)
    gap.axhline(0, color=INK, linewidth=1.2, zorder=1)
    for index, (point, share, color) in enumerate(SV_SHARES):
        nudge = 2 ** (0.09 * (index - 1))
        stats_by_size = [interval(finite(grouped.get((point, size), []), "delta_auc_vs_ldpred2")) for size in SIZES]
        xs = [size * nudge for size in SIZES]
        means = [cell[0] for cell in stats_by_size]
        gap.vlines(xs, [cell[1] for cell in stats_by_size], [cell[2] for cell in stats_by_size], color=color,
                   linewidth=1.8, zorder=2)
        gap.plot(xs, means, color=color, linewidth=1.8, marker="o", markersize=5.8, markeredgecolor="white",
                 markeredgewidth=0.8, zorder=3, label=share)
    gap.set_ylim(-0.005, 0.075)
    gap.set_yticks(np.arange(0, 0.061, 0.02))
    gap.set_xlabel("Training sample size")
    gap.set_ylabel("ΔAUC, SV-PGS − LDpred2-auto")
    gap.legend(title="Causal variants that are SVs", loc="upper left", ncol=3, handlelength=1.2, columnspacing=1.0,
               handletextpad=0.4, title_fontsize=11, fontsize=11, alignment="left", borderaxespad=0.1)
    letter(gap, "b", x=-0.27)
    caption = (
        "Figure 1 | SV-PGS gains accuracy faster than LDpred2-auto as the training sample grows, by the same margin "
        "whether 0%, 5% or 20% of the causal variants are SVs. A simulated binary trait on chromosome 22 (liability "
        "h² = 0.1, prevalence 10%, 200 causal variants, SV effects drawn like small-variant effects), scored in the "
        "same 10,000 held-out people by SV-PGS fitted on every SNV, indel and imputed SV and by LDpred2-auto fitted "
        "on HapMap3-density SNVs of the same training people. a, Test AUC, mean and Student-t 95% interval over the "
        "replicates of all three settings pooled ("
        + ", ".join(f"{n}" for n in counts["SV-PGS"])
        + " replicates at 5k, 10k, 20k and 40k). Dotted: the AUC of a score that captures all of h² "
        "(Wray et al. 2010), the most any method can reach. b, The paired "
        "difference in AUC between the methods on the same held-out people for each setting separately, mean and "
        "Student-t 95% interval over its replicates; the three overlap at every size, which is why a pools them. "
        f"bench-sim v7 cohort from 1000 Genomes haplotypes, Beagle-imputed SVs; fitted on MSI."
    )
    return fig, caption


# ------------------------------------------------------------------ Figure 2


def ldpred2_figure(grids):
    size = SIZES[-1]
    grouped = cells(grids["grid"])
    values = {point: finite(grouped.get((point, size), []), "delta_auc_vs_ldpred2") for point in LABELS}
    rows = layout(lambda point: len(values[point]) >= 3)
    missing = [LABELS[point] for point in LABELS if len(values[point]) < 3]
    fig, ax = plt.subplots(figsize=(165 * MM, 140 * MM))
    fig.subplots_adjust(left=0.43, right=0.93, top=0.95, bottom=0.11)
    ax.axvline(0, color=INK, linewidth=1.2, zorder=1)
    wins = total = 0
    for y, (kind, point) in enumerate(rows):
        if kind != "point":
            continue
        mean, low, high, n = interval(values[point])
        ax.scatter(values[point], [y] * n, s=21, color=LIGHT, linewidths=0, zorder=2)
        ax.hlines(y, low, high, color=BLUE, linewidth=2.2, zorder=3)
        ax.scatter([mean], [y], s=60, color=BLUE, edgecolor="white", linewidth=1, zorder=4)
        ax.text(1.02, y, f"{n}", transform=matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData),
                ha="left", va="center", color=GREY)
        wins += sum(value > 0 for value in values[point])
        total += n
    ax.text(1.02, -1.0, "n", transform=matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData),
            ha="left", va="center", color=GREY, fontstyle="italic")
    row_labels(ax, rows, -0.78)
    ax.set_xlim(-0.015, 0.12)
    ax.set_xticks(np.arange(0, 0.121, 0.04))
    ax.set_xlabel("ΔAUC, SV-PGS − LDpred2-auto")
    caption = (
        f"Figure 2 | SV-PGS is more accurate than LDpred2-auto across simulated genetic architectures. Paired "
        f"difference in test AUC on the same 10,000 held-out people between SV-PGS (every SNV, indel and imputed SV) "
        f"and LDpred2-auto (HapMap3-density SNVs), both trained on the same {size:,} people: replicates (grey), mean "
        f"and Student-t 95% interval (blue); n replicates at right. SV-PGS had the higher AUC in {wins} of {total} "
        f"replicates. {ARCHITECTURES} Not yet replicated three times at {size:,}: "
        f"{', '.join(missing).lower()}."
    )
    return fig, caption


# ------------------------------------------------------------------ Figure 3


def sv_gain_figure(grids):
    imputed, true = defaultdict(list), defaultdict(list)
    for row in grids["window"]:
        imputed[row["point"]].extend(finite([row], "delta_auc"))
    for row in grids["window-truth"]:
        true[row["point"]].extend(finite([row], "delta_auc"))
    rows = layout(lambda point: len(true[point]) >= 3 and len(imputed[point]) >= 3)
    fig, ax = plt.subplots(figsize=(180 * MM, 152 * MM))
    fig.subplots_adjust(left=0.40, right=0.90, top=0.91, bottom=0.10)
    ax.axvline(0, color=INK, linewidth=1.2, zorder=1)
    place = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    clear = shown = 0
    for y, (kind, point) in enumerate(rows):
        if kind != "point":
            continue
        for column, offset, values, color, face in ((1.02, -0.17, true[point], BLUE, BLUE),
                                                    (1.09, 0.17, imputed[point], SKY, "white")):
            mean, low, high, n = interval(values)
            ax.hlines(y + offset, low, high, color=color, linewidth=2.2, zorder=3)
            ax.scatter([mean], [y + offset], s=54, facecolor=face, edgecolor=color if face == "white" else "white",
                       linewidth=1.8 if face == "white" else 0.5, zorder=4)
            ax.text(column, y, f"{n}", transform=place, ha="left", va="center", color=color)
        shown += 1
        clear += interval(true[point])[1] > 0
    ax.text(1.055, -1.0, "n", transform=place, ha="center", va="center", color=GREY, fontstyle="italic")
    row_labels(ax, rows, -0.75)
    ax.set_xlabel("ΔAUC, SV-PGS with SVs − SV-PGS without SVs")
    handles = [
        Line2D([0], [0], marker="o", color=BLUE, markeredgecolor="white", markeredgewidth=0.8, markersize=7.8,
               linewidth=2.2, label="True SV genotypes"),
        Line2D([0], [0], marker="o", color=SKY, markerfacecolor="white", markeredgewidth=1.4, markersize=7.1,
               linewidth=2.2, label="Imputed SV genotypes"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2, columnspacing=1.6)
    caption = (
        "Figure 3 | SVs add accuracy when they are measured well; imputation removes most of the gain. Paired "
        "difference in test AUC on the same held-out people between SV-PGS fitted with and without SVs, when every SV "
        "is measured by its true genotype (dark) or imputed by Beagle from SNV haplotypes (light), on one 3 Mb "
        "chromosome 22 window (60,000 records). Means with Student-t 95% intervals over replicates, training sizes "
        f"5,000-40,000 pooled; n replicates at right (true, imputed). The true-genotype interval lies above zero for {clear} of "
        f"{shown} architectures. Same simulated people, truth and fits; only the SV genotypes differ. "
        f"{ARCHITECTURES}"
    )
    return fig, caption


# ------------------------------------------------------------------ Figure 4


def heatmap_figure(grids):
    sizes = SIZES[:-1]
    panels = (("window-truth", "True SV genotypes"), ("window", "Imputed SV genotypes"))
    summaries = {}
    for grid, _ in panels:
        grouped = cells(grids[grid])
        summaries[grid] = {(point, size): interval(finite(grouped.get((point, size), []), "delta_auc"))
                           for point in LABELS for size in sizes}
    rows = layout(lambda point: True)
    clear_values = [abs(mean) for table in summaries.values() for mean, low, high, n in table.values()
                    if n >= 2 and (low > 0 or high < 0)]
    limit = max(0.005, math.ceil(1000 * max(clear_values, default=0.005)) / 1000)
    norm = matplotlib.colors.Normalize(-limit, limit)
    cmap = LinearSegmentedColormap.from_list("gain", [VERMILLION, "#FFFFFF", BLUE])
    fig, axes = plt.subplots(1, 2, figsize=(200 * MM, 168 * MM), sharey=True)
    fig.subplots_adjust(left=0.395, right=0.99, top=0.90, bottom=0.02, wspace=0.05)
    for panel, (ax, (grid, title)) in enumerate(zip(axes, panels)):
        colours = np.ones((len(rows), len(sizes), 3))
        for i, (kind, point) in enumerate(rows):
            if kind != "point":
                continue
            for j, size in enumerate(sizes):
                mean, low, high, n = summaries[grid][(point, size)]
                if n < 2:
                    colours[i, j] = to_rgb("#F2F2F2")
                    ax.text(j, i, "–", ha="center", va="center", fontsize=11, color=GREY)
                    continue
                clear = low > 0 or high < 0
                rgb = np.array(cmap(norm(mean))[:3])
                # an interval that covers zero is drawn pale, so colour carries only what the replicates support
                colours[i, j] = rgb if clear else rgb + (1 - rgb) * 0.75
                dark = colours[i, j] @ np.array([0.299, 0.587, 0.114]) < 0.55
                ax.text(j, i, "<0.001" if clear and abs(mean) < 0.0005 else signed(mean, 3), ha="center", va="center", fontsize=11,
                        color="white" if dark else (INK if clear else GREY),
                        fontweight="bold" if clear else "normal")
        ax.imshow(colours, aspect="auto", interpolation="nearest")
        ax.set_xticks(range(len(sizes)), [f"n = {thousands(size)}" for size in sizes])
        ax.xaxis.tick_top()
        ax.tick_params(axis="both", length=0)
        ax.set_xticks(np.arange(-0.5, len(sizes)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
        ax.grid(which="minor", color="white", linewidth=2.4)
        ax.tick_params(which="minor", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, fontweight="bold", pad=16)
        letter(ax, "ab"[panel], x=-0.04 if panel else -1.1, y=1.065)
    row_labels(axes[0], rows, -1.1)
    axes[0].set_ylim(len(rows) - 0.5, -0.5)
    caption = (
        "Figure 4 | The gain from adding SVs by architecture and training size. Mean paired difference in test AUC on "
        "the same held-out people between SV-PGS fitted with and without SVs on one chromosome 22 window, with SVs "
        "measured by their true genotypes (a) or imputed (b). Bold on full colour: the Student-t 95% interval over "
        f"replicates excludes zero ('<0.001': a gain below 0.001 whose interval excludes zero); grey on pale colour: it does not. Colour saturates at ±{limit:.3f}. Dashes: fewer "
        f"than two replicates. 40,000-person window fits are not yet replicated and are left out. {ARCHITECTURES}"
    )
    return fig, caption


# ------------------------------------------------------------------ Figure 5


def genotype_figure(summary):
    accuracy = summary["genotype_accuracy"]["r2_by_frequency"]
    records = np.load(LOCAL / "public" / "sv_records.npz")
    fig, (left, right) = plt.subplots(1, 2, figsize=(220 * MM, 88 * MM), gridspec_kw={"width_ratios": [1.15, 1.0]})
    fig.subplots_adjust(left=0.075, right=0.98, top=0.90, bottom=0.18, wspace=0.28)
    for key, color, label in (("SNV", INK, "SNVs"), ("SV", BLUE, "SVs")):
        bins = accuracy[key]["bins"]
        centers = np.array([math.sqrt(b["low"] * b["high"]) for b in bins])
        median = np.array([b["median"] for b in bins])
        left.fill_between(centers, [b["q25"] for b in bins], [b["q75"] for b in bins], color=color, alpha=0.15,
                          linewidth=0)
        left.plot(centers, median, color=color, linewidth=2, marker="o", markersize=5.1, markeredgecolor="white",
                  markeredgewidth=0.6)
        left.text(centers[-1] * 1.15, median[-1], label, color=color, fontsize=11, va="center", fontweight="bold")
    left.set_xscale("log")
    left.set_xlim(5e-4, 0.8)
    left.set_ylim(0, 1.02)
    left.set_xticks([0.001, 0.01, 0.1], ["0.1%", "1%", "10%"])
    left.set_xlabel("Minor allele frequency")
    left.set_ylabel("Imputation r²")
    share = accuracy["SV"]["below_1pct"]
    left.axvline(0.01, color=LIGHT, linewidth=1.2, linestyle=(0, (2, 2)), zorder=0)
    left.text(0.0108, 0.03, f"{share:.0%} of SVs\nhave MAF < 1%", fontsize=10.5, color=GREY, va="bottom")
    letter(left, "a", x=-0.13)

    kind = records["svtype"][records["measured"]]
    change = records["len_change"][records["measured"]]
    length = np.abs(change).astype(float)
    groups = (("Deletion", (kind == "DEL") | ((kind == "") & (change < 0)), VERMILLION),
              ("Insertion", (kind == "INS") | ((kind == "") & (change > 0)), BLUE),
              ("Duplication", kind == "DUP", GREEN))
    edges = np.logspace(math.log10(50), 6, 37)
    bottom = np.zeros(len(edges) - 1)
    for label, mask, color in groups:
        counts, _ = np.histogram(length[mask], bins=edges)
        right.bar(edges[:-1], counts, width=np.diff(edges), bottom=bottom, align="edge", color=color,
                  edgecolor="white", linewidth=0.6, label=f"{label} ({int(mask.sum()):,})")
        bottom += counts
    right.set_xscale("log")
    right.set_xlim(45, 1.1e6)
    right.set_xticks([1e2, 1e3, 1e4, 1e5, 1e6], ["100 bp", "1 kb", "10 kb", "100 kb", "1 Mb"])
    right.set_xlabel("SV length")
    right.set_ylabel("SVs")
    right.legend(loc="upper right", handlelength=0.9, handleheight=0.9)
    letter(right, "b", x=-0.14)
    caption = (
        f"Figure 5 | The simulated SVs are mostly rare and short, and rare SVs impute poorly. a, Dosage r² between "
        f"Beagle-imputed and true genotypes of {accuracy['SV']['records']:,} SVs and {accuracy['SNV']['records']:,} "
        f"sampled SNVs on chromosome 22 in 50,000 simulated people, by minor allele frequency: medians (points) and "
        f"interquartile ranges (bands) per frequency bin. b, Lengths of the {int(records['measured'].sum()):,} "
        f"measured SVs, stacked by class (sequence-resolved records without a type are classed by the sign of their "
        f"length change). bench-sim v7 cohort from public 1000 Genomes haplotypes."
    )
    return fig, caption


# ------------------------------------------------------------------ Figure 6


def compute_figure(grids):
    grouped = cells(grids["grid"])
    fig, ax = plt.subplots(figsize=(130 * MM, 95 * MM))
    fig.subplots_adjust(left=0.17, right=0.72, top=0.95, bottom=0.17)
    methods = (("with_sv", "SV-PGS with SVs", BLUE, 0.08, 0.80), ("snv_indel", "SV-PGS without SVs", SKY, -0.0, 1.25),
               ("ldpred2_auto", "LDpred2-auto", VERMILLION, -0.08, 1.0))
    xs = np.log2(np.array(SIZES) / SIZES[0])
    for key, label, color, offset, nudge in methods:
        present, medians = [], []
        for step, size in enumerate(SIZES):
            hours = [row["fit_seconds"][key] / 3600 for (point, n), rows in grouped.items() if n == size for row in rows
                     if isinstance(row.get("fit_seconds"), dict) and math.isfinite(row["fit_seconds"].get(key, math.nan))]
            if len(hours) < 3:
                continue
            q25, median, q75 = np.percentile(hours, [25, 50, 75])
            x = xs[step] + offset
            ax.vlines(x, q25, q75, color=color, linewidth=2)
            present.append(x), medians.append(median)
        ax.plot(present, medians, color=color, linewidth=2, marker="o", markersize=5.4, markeredgecolor="white",
                markeredgewidth=0.6)
        ax.text(xs[-1] + 0.2, medians[-1] * nudge, label, color=color, fontsize=10.5, va="center")
    ax.set_yscale("log")
    ax.set_yticks([1 / 60, 10 / 60, 1, 10], ["1 min", "10 min", "1 h", "10 h"])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xticks(xs, [thousands(size) for size in SIZES])
    ax.set_xlim(xs[0] - 0.25, xs[-1] + 0.25)
    ax.set_xlabel("Training sample size")
    ax.set_ylabel("Wall time per fit")
    caption = (
        "Figure 6 | Fit time. Wall time of one fit on chromosome 22 per method and training size, pooled over every "
        "architecture: median (points) and interquartile range (bars). MSI Slurm jobs, one fit per job; SV-PGS fits "
        "every variant by empirical Bayes, LDpred2-auto HapMap3-density SNVs."
    )
    return fig, caption


# ------------------------------------------------------------------ main


def gallery_page(pdf: PdfPages, image: Path, caption: str) -> None:
    """One page: the figure's 600 dpi raster above its legend, for reading the set in a viewer."""
    picture = plt.imread(image)
    height, width = picture.shape[:2]
    page_width = 8.5
    figure_height = page_width * 0.86 * height / width
    page = plt.figure(figsize=(page_width, figure_height + 2.2))
    holder = page.add_axes([0.07, 2.2 / (figure_height + 2.2), 0.86, figure_height / (figure_height + 2.2)])
    holder.imshow(picture, interpolation="lanczos")
    holder.axis("off")
    import textwrap
    page.text(0.07, (2.0) / (figure_height + 2.2), textwrap.fill(caption, 125), ha="left", va="top", fontsize=8.5,
              linespacing=1.35)
    pdf.savefig(page)
    plt.close(page)


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
        ("fig4_sv_gain_heatmap", *heatmap_figure(grids)),
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
    print(OUT)


if __name__ == "__main__":
    main()

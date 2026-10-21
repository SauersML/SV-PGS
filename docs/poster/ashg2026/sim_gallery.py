"""Journal-style figures of the simulation study run on MSI (benchmarks/power/grid.py on bench-sim v7's 1000 Genomes
chr22 cohort, Beagle-imputed arm). Nothing here is from All of Us.

Every value is read from the replicate results copied from MSI (simulation.py's LOCAL tree: grid, grid-truth, window,
window-truth), and from simulation_summary.json for the analytic expected AUC and the imputation accuracy bins.

    uv run --with matplotlib python docs/poster/ashg2026/sim_gallery.py [output directory]

Figures are sized in millimetres for print (single column 89 mm, double 183 mm) with 6-7 pt type. Each is written as a
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
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgb
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
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.6,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelpad": 3,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.color": INK,
    "ytick.color": INK,
    "lines.linewidth": 1.0,
    "lines.markersize": 3.5,
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
    "null": "No causal SVs",
    "sv_5pct": "5% of causal variants SVs",
    "sv_5pct_fold3": "5% SVs, 3× effect",
    "sv_20pct": "20% of causal variants SVs",
    "sv_20pct_fold3": "20% SVs, 3× effect",
    "length_positive": "Effect grows with SV length",
    "deletion_positive": "Deletions 3× variance",
    "multi_feature_positive": "Length, type and annotation",
    "sv_heavy_tail": "Heavy-tailed SV effects",
    "sv_two_point": "Two-point SV effects",
    "sparse_sv": "50 causal variants",
    "polygenic_sv": "2,000 causal variants",
    "rare_disease": "Prevalence 1%",
    "low_heritability": "h² = 0.01",
    "tagged_sv": "Only well-tagged SVs causal",
    "tagged_sv_fold3": "Well-tagged SVs, 3× effect",
    "common_sv": "Only common SVs causal",
}


def shade(color: str, amount: float) -> tuple:
    """``color`` mixed with white by ``amount`` (0 keeps it, 1 is white)."""
    rgb = np.array(to_rgb(color))
    return tuple(rgb + (1 - rgb) * amount)


def thousands(count: int) -> str:
    return f"{count // 1000}k"


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
    ax.text(x, y, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=8, fontweight="bold")


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


# ------------------------------------------------------------------ Figure 1


SV_SHARES = (("null", "0%", 0.62, "o"), ("sv_5pct", "5%", 0.32, "s"), ("sv_20pct", "20%", 0.0, "^"))


def accuracy_figure(grids, summary):
    """Test AUC against training size for SV-PGS and LDpred2-auto, at three shares of causal SVs."""
    grouped = cells(grids["grid"])
    fig, ax = plt.subplots(figsize=(110 * MM, 78 * MM))
    fig.subplots_adjust(left=0.13, right=0.80, top=0.95, bottom=0.15)
    xs = np.log2(np.array(SIZES) / SIZES[0])
    expected = summary["points"]["null"]["expected_auc"]
    ax.plot(xs, [expected[str(size)] for size in SIZES], color=GREY, linestyle=(0, (3, 2)), linewidth=0.8)
    ax.text(xs[-1] + 0.12, expected[str(SIZES[-1])], "Expected AUC,\nall h² captured", color=GREY, fontsize=6.5,
            va="center", ha="left", linespacing=1.1)
    methods = (("with_sv_auc", "SV-PGS", BLUE), ("ldpred2_auc", "LDpred2-auto", VERMILLION))
    ends = {}
    for key, method, color in methods:
        for index, (point, share, lightness, marker) in enumerate(SV_SHARES):
            tone = shade(color, lightness)
            dodge = (index - 1) * 0.07
            present, means, lows, highs = [], [], [], []
            for step, size in enumerate(SIZES):
                mean, low, high, n = interval([row[key] for row in grouped.get((point, size), [])])
                if n < 2:
                    continue
                present.append(xs[step] + dodge), means.append(mean), lows.append(low), highs.append(high)
            ax.plot(present, means, color=tone, linewidth=1.0, zorder=3)
            ax.vlines(present, lows, highs, color=tone, linewidth=0.8, zorder=3)
            ax.scatter(present, means, s=14, marker=marker, color=tone, edgecolor="white", linewidth=0.4, zorder=4)
            ends.setdefault(method, []).append(means[-1])
    for key, method, color in methods:
        ax.text(xs[-1] + 0.12, float(np.mean(ends[method])), method, color=color, fontsize=7, fontweight="bold",
                va="center", ha="left")
    ax.set_xticks(xs, [thousands(size) for size in SIZES])
    ax.set_xlim(xs[0] - 0.25, xs[-1] + 0.25)
    ax.set_ylim(0.50, 0.68)
    ax.set_yticks(np.arange(0.50, 0.681, 0.04))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.01))
    ax.set_xlabel("Training sample size")
    ax.set_ylabel("Test AUC")
    handles = [Line2D([0], [0], marker=marker, color=shade(GREY, lightness * 0.8), markeredgecolor="white",
                      markeredgewidth=0.4, markersize=4.2, linewidth=1.0, label=share)
               for _, share, lightness, marker in SV_SHARES]
    ax.legend(handles=handles, title="Causal variants that are SVs", title_fontsize=6.5, loc="upper left", ncol=3,
              columnspacing=1.0, handletextpad=0.3, alignment="left")
    caption = (
        "Figure 1 | SV-PGS gains accuracy faster than LDpred2-auto as the training sample grows, whatever the share "
        "of causal SVs. Test AUC in 10,000 held-out people for a simulated binary trait on chromosome 22 (liability "
        "h² = 0.1, prevalence 10%, 200 causal variants) in which 0%, 5% or 20% of the causal variants are SVs with the "
        "same per-variant effect distribution as small variants. SV-PGS fits every SNV, indel and imputed SV; "
        "LDpred2-auto fits HapMap3-density SNVs of the same training people. Points are means over simulation "
        "replicates (5-15 per point) with Student-t 95% intervals, offset horizontally for legibility; the dashed line "
        "is the expected AUC of a score capturing all of h² (Daetwyler et al. 2008; Wray et al. 2010). bench-sim v7 "
        "cohort from 1000 Genomes haplotypes, Beagle-imputed SVs; fitted on MSI."
    )
    return fig, caption


# ------------------------------------------------------------------ Figure 2


def ldpred2_figure(grids):
    grouped = cells(grids["grid"])
    rows = []
    for point in LABELS:
        for size in sorted(SIZES, reverse=True):
            values = [value for value in (row["delta_auc_vs_ldpred2"] for row in grouped.get((point, size), []))
                      if math.isfinite(value)]
            if len(values) >= 3:
                rows.append((point, size, values, *interval(values)))
                break
    rows.sort(key=lambda row: row[3])
    fig, ax = plt.subplots(figsize=(120 * MM, 92 * MM))
    fig.subplots_adjust(left=0.33, right=0.86, top=0.97, bottom=0.12)
    ax.axvline(0, color=INK, linewidth=0.6, zorder=1)
    rng = np.random.default_rng(1)
    for y, (point, size, values, mean, low, high, n) in enumerate(rows):
        ax.scatter(values, y + rng.uniform(-0.18, 0.18, len(values)), s=5, color=shade(BLUE, 0.55), linewidths=0,
                   zorder=2)
        ax.hlines(y, low, high, color=BLUE, linewidth=1.0, zorder=3)
        ax.scatter([mean], [y], s=16, color=BLUE, edgecolor="white", linewidth=0.4, zorder=4)
        wins = sum(value > 0 for value in values)
        ax.text(1.02, y, f"{wins}/{n}", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6.5)
        ax.text(1.12, y, thousands(size), transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6.5,
                color=GREY)
    top = len(rows) - 0.4
    ax.text(1.02, top, "Wins", transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=6.5,
            fontweight="bold")
    ax.text(1.12, top, "N", transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=6.5,
            color=GREY, style="italic")
    ax.set_yticks(range(len(rows)), [LABELS[row[0]] for row in rows])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.01))
    ax.set_xlabel("ΔAUC, SV-PGS − LDpred2-auto")
    wins = sum(sum(value > 0 for value in row[2]) for row in rows)
    total = sum(len(row[2]) for row in rows)
    caption = (
        f"Figure 2 | SV-PGS is more accurate than LDpred2-auto across simulated genetic architectures. Paired "
        f"difference in test AUC (same 10,000 held-out people) between SV-PGS with SVs and LDpred2-auto, per "
        f"replicate (small dots), with the mean and its Student-t 95% interval, at the largest training size with at "
        f"least three replicates (N, right). Wins: replicates in which SV-PGS had the higher AUC ({wins} of {total} "
        f"overall). Unless named, each architecture has liability h² = 0.1, prevalence 10%, 200 causal variants and "
        f"20% causal SVs; '3× effect' triples the SV effect variance, 'well-tagged' draws causal SVs only from those "
        f"imputed with r² ≥ 0.8, 'common' only from SVs of frequency ≥ 5%."
    )
    return fig, caption


# ------------------------------------------------------------------ Figure 3


def sv_gain_figure(grids):
    imputed, true = defaultdict(list), defaultdict(list)
    for row in grids["window"]:
        imputed[row["point"]].append(row["delta_auc"])
    for row in grids["window-truth"]:
        true[row["point"]].append(row["delta_auc"])
    points = [point for point in LABELS if len(true[point]) >= 3 and len(imputed[point]) >= 3]
    summary = {point: (interval(imputed[point]), interval(true[point])) for point in points}
    points.sort(key=lambda point: summary[point][1][0])
    fig, ax = plt.subplots(figsize=(120 * MM, 96 * MM))
    fig.subplots_adjust(left=0.33, right=0.97, top=0.90, bottom=0.11)
    ax.axvline(0, color=INK, linewidth=0.6, zorder=1)
    for y, point in enumerate(points):
        (mi, li, hi, ni), (mt, lt, ht, nt) = summary[point]
        ax.hlines(y + 0.16, lt, ht, color=BLUE, linewidth=1.0, zorder=3)
        ax.scatter([mt], [y + 0.16], s=16, color=BLUE, edgecolor="white", linewidth=0.4, zorder=4)
        ax.hlines(y - 0.16, li, hi, color=SKY, linewidth=1.0, zorder=3)
        ax.scatter([mi], [y - 0.16], s=14, facecolor="white", edgecolor=SKY, linewidth=0.9, zorder=4)
    ax.set_yticks(range(len(points)), [LABELS[point] for point in points])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.6, len(points) - 0.4)
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.005))
    ax.set_xlabel("ΔAUC from adding SVs, SV-PGS with − without SVs")
    handles = [
        Line2D([0], [0], marker="o", color=BLUE, markeredgecolor="white", markeredgewidth=0.4, markersize=4.2,
               linewidth=1.0, label="True SV genotypes"),
        Line2D([0], [0], marker="o", color=SKY, markerfacecolor="white", markeredgewidth=0.9, markersize=3.8,
               linewidth=1.0, label="Imputed SV genotypes"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, columnspacing=1.4)
    significant = [LABELS[point] for point in points if summary[point][1][1] > 0]
    caption = (
        "Figure 3 | SVs add accuracy when they are measured well; imputation removes most of the gain. Paired "
        "difference in test AUC between SV-PGS fitted with and without SVs, when every SV is measured by its true "
        "genotype (dark) or imputed by Beagle from SNV haplotypes (light), on one 3 Mb chromosome 22 window "
        "(60,000 records). Means with Student-t 95% intervals over replicates, training sizes 5,000-40,000 pooled. "
        f"The true-genotype interval excludes zero for {len(significant)} of {len(points)} architectures. Same "
        "simulated people, truth and fits; only the SV genotypes differ."
    )
    return fig, caption


# ------------------------------------------------------------------ Figure 4


def heatmap_figure(grids):
    true_cells = cells(grids["window-truth"])

    def order_key(point):
        means = [np.mean([r["delta_auc"] for r in true_cells[(point, size)]]) for size in SIZES
                 if len(true_cells.get((point, size), [])) >= 2]
        return -np.mean(means) if means else 0.0

    order = sorted(LABELS, key=order_key)
    cmap = LinearSegmentedColormap.from_list("gain", [shade(VERMILLION, 0.0), shade(VERMILLION, 0.85), "#FFFFFF",
                                                      shade(BLUE, 0.80), BLUE, "#003A5C"])
    norm = TwoSlopeNorm(vmin=-0.008, vcenter=0.0, vmax=0.024)
    fig, axes = plt.subplots(1, 2, figsize=(140 * MM, 100 * MM), sharey=True)
    fig.subplots_adjust(left=0.25, right=0.88, top=0.80, bottom=0.03, wspace=0.06)
    for panel, (ax, grid, title) in enumerate(((axes[0], "window-truth", "True SV genotypes"),
                                               (axes[1], "window", "Imputed SV genotypes"))):
        grouped = cells(grids[grid])
        matrix = np.full((len(order), len(SIZES)), np.nan)
        for i, point in enumerate(order):
            for j, size in enumerate(SIZES):
                values = [row["delta_auc"] for row in grouped.get((point, size), []) if math.isfinite(row["delta_auc"])]
                if len(values) >= 2:
                    matrix[i, j] = float(np.mean(values))
        image = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, norm=norm, aspect="auto")
        ax.set_facecolor("#F2F2F2")
        for i in range(len(order)):
            for j in range(len(SIZES)):
                if math.isnan(matrix[i, j]):
                    continue
                value = matrix[i, j] * 1000
                text = "0" if abs(value) < 0.05 else f"{value:+.1f}".replace("-", "−")
                ax.text(j, i, text, ha="center", va="center", fontsize=5.5,
                        color="white" if abs(matrix[i, j]) > 0.012 else INK)
        ax.set_xticks(range(len(SIZES)), [thousands(size) for size in SIZES])
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.set_xlabel("Training sample size", labelpad=4)
        ax.tick_params(axis="both", length=0)
        ax.set_xticks(np.arange(-0.5, len(SIZES)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(order)), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.0)
        ax.tick_params(which="minor", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(0.5, 1.115, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=7, fontweight="bold")
        letter(ax, "ab"[panel], x=-0.02 if panel else -0.62, y=1.115)
    axes[0].set_yticks(range(len(order)), [LABELS[point] for point in order])
    bar = fig.add_axes([0.905, 0.30, 0.012, 0.40])
    colorbar = fig.colorbar(image, cax=bar)
    colorbar.outline.set_linewidth(0.4)
    colorbar.set_ticks([-0.008, 0, 0.008, 0.016, 0.024], labels=["−8", "0", "8", "16", "24"])
    colorbar.ax.tick_params(labelsize=6, length=2, width=0.4)
    colorbar.set_label("ΔAUC × 10³", fontsize=6.5)
    caption = (
        "Figure 4 | Gain from adding SVs by architecture and training size. Mean paired ΔAUC (× 1,000) between "
        "SV-PGS fitted with and without SVs on one chromosome 22 window, with SVs measured by their true genotypes "
        "(a) or imputed (b). Grey cells had fewer than two replicates. Rows are ordered by the mean true-genotype gain."
    )
    return fig, caption


# ------------------------------------------------------------------ Figure 5


def genotype_figure(summary):
    accuracy = summary["genotype_accuracy"]["r2_by_frequency"]
    records = np.load(LOCAL / "public" / "sv_records.npz")
    fig, (left, right) = plt.subplots(1, 2, figsize=(183 * MM, 66 * MM), gridspec_kw={"width_ratios": [1.15, 1.0]})
    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.18, wspace=0.28)
    for key, color, label in (("SNV", INK, "SNVs"), ("SV", BLUE, "SVs")):
        bins = accuracy[key]["bins"]
        centers = np.array([math.sqrt(b["low"] * b["high"]) for b in bins])
        median = np.array([b["median"] for b in bins])
        left.fill_between(centers, [b["q25"] for b in bins], [b["q75"] for b in bins], color=color, alpha=0.15,
                          linewidth=0)
        left.plot(centers, median, color=color, linewidth=1.0, marker="o", markersize=3, markeredgecolor="white",
                  markeredgewidth=0.4)
        left.text(centers[-1] * 1.15, median[-1], label, color=color, fontsize=7, va="center", fontweight="bold")
    left.set_xscale("log")
    left.set_xlim(5e-4, 0.8)
    left.set_ylim(0, 1.02)
    left.set_xticks([0.001, 0.01, 0.1], ["0.1%", "1%", "10%"])
    left.set_xlabel("Minor allele frequency")
    left.set_ylabel("Imputation r² (dosage vs true genotype)")
    share = accuracy["SV"]["below_1pct"]
    left.axvline(0.01, color=LIGHT, linewidth=0.6, linestyle=(0, (2, 2)), zorder=0)
    left.text(0.0108, 0.03, f"{share:.0%} of SVs\nhave MAF < 1%", fontsize=6.5, color=GREY, va="bottom")
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
                  edgecolor="white", linewidth=0.3, label=f"{label} ({int(mask.sum()):,})")
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
    fig, ax = plt.subplots(figsize=(89 * MM, 68 * MM))
    fig.subplots_adjust(left=0.17, right=0.72, top=0.95, bottom=0.17)
    methods = (("with_sv", "SV-PGS, with SVs", BLUE, 0.08, 0.80), ("snv_indel", "SV-PGS, no SVs", SKY, -0.0, 1.25),
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
            ax.vlines(x, q25, q75, color=color, linewidth=1.0)
            present.append(x), medians.append(median)
        ax.plot(present, medians, color=color, linewidth=1.0, marker="o", markersize=3.2, markeredgecolor="white",
                markeredgewidth=0.4)
        ax.text(xs[-1] + 0.2, medians[-1] * nudge, label, color=color, fontsize=6.5, va="center")
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

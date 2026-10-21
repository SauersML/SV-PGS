"""Publication-style figures of the simulation study run on MSI (benchmarks/power/grid.py on bench-sim v7's 1000 Genomes
chr22 cohort, Beagle-imputed arm). Nothing here is from All of Us.

Every value is read from the replicate results copied from MSI (simulation.py's LOCAL tree: grid, grid-truth, window,
window-truth) and from simulation_summary.json for the analytic expected AUC and the imputation accuracy bins.

    uv run --with matplotlib python docs/poster/ashg2026/sim_gallery.py [output directory]

Writes one PNG (300 dpi) and one vector PDF per figure, and gallery.pdf with every figure as a page.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

HERE = Path(__file__).resolve().parent
LOCAL = Path.home() / "svpgs-bench" / "poster-grid2"
SUMMARY = HERE / "simulation_summary.json"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "figures" / "gallery"
SIZES = (5_000, 10_000, 20_000, 40_000)

# ------------------------------------------------------------------ style

WITH = "#0E7C9B"
WITHOUT = "#1E3550"
LDPRED = "#D97B29"
CEILING = "#9AA1AA"
INK = "#1B1F24"
MUTED = "#5F6670"
GRID = "#E9ECEF"
SPINE = "#BFC5CC"
PAPER = "#FFFFFF"
DEL = "#C2464C"
INS = "#0E7C9B"
DUP = "#7A5BA6"
OTHER = "#A9B0B8"


def _font() -> str:
    names = {entry.name for entry in font_manager.fontManager.ttflist}
    for name in ("Helvetica Neue", "Helvetica", "Arial"):
        if name in names:
            return name
    return "DejaVu Sans"


plt.rcParams.update({
    "font.family": _font(),
    "font.size": 10.5,
    "axes.edgecolor": SPINE,
    "axes.labelcolor": INK,
    "axes.labelsize": 10.5,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": MUTED,
    "ytick.labelcolor": MUTED,
    "xtick.major.size": 3.5,
    "ytick.major.size": 0,
    "xtick.major.width": 0.8,
    "legend.frameon": False,
    "legend.fontsize": 10,
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

LABELS = {
    "null": "No causal SVs",
    "sv_5pct": "5% of causal variants are SVs",
    "sv_5pct_fold3": "5% SVs, 3× effect",
    "sv_20pct": "20% of causal variants are SVs",
    "sv_20pct_fold3": "20% SVs, 3× effect",
    "length_positive": "Effect grows with SV length",
    "deletion_positive": "Deletions carry 3× variance",
    "multi_feature_positive": "Length + type + annotation",
    "sv_heavy_tail": "Heavy-tailed SV effects",
    "sv_two_point": "Two-point SV effects",
    "sparse_sv": "Sparse: 50 causal variants",
    "polygenic_sv": "Polygenic: 2,000 causal",
    "rare_disease": "Rare disease (1% prevalence)",
    "low_heritability": "Low heritability (h² = 0.01)",
    "tagged_sv": "Only well-tagged SVs causal",
    "tagged_sv_fold3": "Well-tagged SVs, 3× effect",
    "common_sv": "Only common SVs causal",
}
SHORT = {
    "sv_5pct": "5% SVs",
    "sv_20pct": "20% SVs",
}


def thousands(count: int) -> str:
    return f"{count // 1000}k"


def headline(fig, title: str, subtitle: str, top: float = 0.965) -> None:
    fig.text(0.012, top, title, ha="left", va="top", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.012, top - 0.062 * (6.0 / fig.get_figheight()), subtitle, ha="left", va="top", fontsize=10.5,
             color=MUTED)


def footnote(fig, text: str) -> None:
    fig.text(0.012, 0.012, text, ha="left", va="bottom", fontsize=8, color=MUTED)


def panel_letter(ax, letter: str) -> None:
    ax.text(-0.02, 1.0, letter, transform=ax.transAxes, ha="right", va="bottom", fontsize=13, fontweight="bold",
            color=INK)


def panel_title(ax, letter: str, title: str, detail: str) -> None:
    ax.text(-0.015, 1.10, letter, transform=ax.transAxes, ha="right", va="bottom", fontsize=13, fontweight="bold",
            color=INK)
    ax.text(0.0, 1.10, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=11.5, fontweight="bold",
            color=INK)
    ax.text(0.0, 1.035, detail, transform=ax.transAxes, ha="left", va="bottom", fontsize=9, color=MUTED)


def horizontal_grid(ax) -> None:
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


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


# ------------------------------------------------------------------ data


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


def jitter(n: int, width: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-width, width, n)


# ------------------------------------------------------------------ figure 1: accuracy against training size


def accuracy_figure(grids, summary) -> plt.Figure:
    grouped = cells(grids["grid"])
    points = ("null", "sv_5pct", "sv_20pct", "sv_20pct_fold3")
    fig, axes = plt.subplots(1, len(points), figsize=(13.0, 4.9), sharey=True)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.70, bottom=0.17, wspace=0.10)
    methods = (("with_sv_auc", "SV-PGS with SVs", WITH, 0.10), ("snv_indel_auc", "SV-PGS without SVs", WITHOUT, 0.0),
               ("ldpred2_auc", "LDpred2-auto", LDPRED, -0.10))
    for index, (ax, point) in enumerate(zip(axes, points)):
        horizontal_grid(ax)
        expected = summary["points"][point]["expected_auc"]
        xs = np.log2(np.array(SIZES) / 5000)
        ax.plot(xs, [expected[str(size)] for size in SIZES], color=CEILING, linestyle=(0, (4, 3)), linewidth=1.6,
                zorder=2)
        if index == 0:
            ax.text(xs[0], expected[str(SIZES[0])] + 0.007, "Heritability ceiling", color=MUTED, fontsize=9,
                    ha="left", va="bottom")
        for key, label, color, offset in methods:
            means, lows, highs, present = [], [], [], []
            for step, size in enumerate(SIZES):
                rows = grouped.get((point, size), [])
                values = [row[key] for row in rows]
                mean, low, high, n = interval(values)
                if n < 2:
                    continue
                x = step + offset
                ax.scatter(x + jitter(n, 0.035, step * 7 + len(key)), [v for v in values if math.isfinite(v)],
                           s=9, color=color, alpha=0.22, linewidths=0, zorder=3)
                means.append(mean), lows.append(low), highs.append(high), present.append(x)
            ax.plot(present, means, color=color, linewidth=2.2, zorder=4, solid_capstyle="round")
            ax.vlines(present, lows, highs, color=color, linewidth=1.6, zorder=4)
            ax.scatter(present, means, s=34, color=color, edgecolor="white", linewidth=1.1, zorder=5)
        ax.set_xticks(range(len(SIZES)), [thousands(size) for size in SIZES])
        ax.set_xlim(-0.4, len(SIZES) - 0.6)
        ax.set_title(LABELS[point].replace(" of causal variants are SVs", " causal SVs"), fontsize=11.5)
        ax.set_xlabel("Training samples" if index == 0 else "")
        ax.spines["left"].set_visible(index == 0)
    axes[0].set_ylabel("Test AUC (10,000 held-out people)")
    axes[0].set_ylim(0.49, 0.70)
    handles = [Line2D([0], [0], color=color, marker="o", markersize=6, markeredgecolor="white", linewidth=2.2,
                      label=label) for _, label, color, _ in methods]
    handles.append(Line2D([0], [0], color=CEILING, linestyle=(0, (4, 3)), linewidth=1.6,
                          label="Expected AUC capturing all h²"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.055, 0.835), ncol=4, handlelength=2.2,
               columnspacing=2.0)
    headline(fig, "SV-PGS gains accuracy faster than LDpred2-auto as training grows",
             "Simulated binary trait on chromosome 22 (h² = 0.1, prevalence 10%, 200 causal variants); "
             "means with 95% intervals, faint dots are replicates", top=0.975)
    footnote(fig, "bench-sim v7 cohort (1000 Genomes haplotypes), Beagle-imputed SVs; LDpred2-auto fitted on "
                  "HapMap3-density SNVs of the same training people. Run on MSI.")
    return fig


# ------------------------------------------------------------------ figure 2: paired gain over LDpred2-auto


def ldpred2_figure(grids) -> plt.Figure:
    grouped = cells(grids["grid"])
    rows = []
    for point in LABELS:
        for size in sorted(SIZES, reverse=True):
            values = [row["delta_auc_vs_ldpred2"] for row in grouped.get((point, size), [])]
            values = [value for value in values if math.isfinite(value)]
            if len(values) >= 3:
                mean, low, high, n = interval(values)
                rows.append((point, size, values, mean, low, high, n))
                break
    rows.sort(key=lambda row: row[3])
    fig, ax = plt.subplots(figsize=(10.5, 7.4))
    fig.subplots_adjust(left=0.29, right=0.86, top=0.855, bottom=0.10)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.axvline(0, color=MUTED, linewidth=1.0, zorder=1)
    for y, (point, size, values, mean, low, high, n) in enumerate(rows):
        ax.scatter(values, y + jitter(len(values), 0.16, y), s=16, color=WITH, alpha=0.30, linewidths=0, zorder=2)
        ax.hlines(y, low, high, color=WITH, linewidth=2.4, zorder=3)
        ax.scatter([mean], [y], s=62, color=WITH, edgecolor="white", linewidth=1.3, zorder=4)
        wins = sum(value > 0 for value in values)
        ax.text(1.015, y, f"{wins}/{n}", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=9.5,
                color=INK if wins == n else MUTED, fontweight="bold" if wins == n else "normal")
        ax.text(1.085, y, thousands(size), transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=9.5,
                color=MUTED)
    ax.text(1.015, len(rows) - 0.35, "wins", transform=ax.get_yaxis_transform(), ha="left", va="bottom",
            fontsize=9, color=MUTED, fontweight="bold")
    ax.text(1.085, len(rows) - 0.35, "train", transform=ax.get_yaxis_transform(), ha="left", va="bottom",
            fontsize=9, color=MUTED, fontweight="bold")
    ax.set_yticks(range(len(rows)), [LABELS[row[0]] for row in rows], color=INK, fontsize=10.5)
    ax.tick_params(axis="y", labelcolor=INK)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Test AUC of SV-PGS (with SVs) minus LDpred2-auto, same people")
    wins = sum(sum(value > 0 for value in row[2]) for row in rows)
    total = sum(len(row[2]) for row in rows)
    headline(fig, f"SV-PGS beats LDpred2-auto in {wins} of {total} paired replicates, across {len(rows)} architectures",
             "Paired difference in held-out AUC per replicate (dots), mean and 95% interval, at the largest "
             "training size with ≥ 3 replicates")
    footnote(fig, "Simulated binary traits on chromosome 22; 10,000 held-out people per replicate. "
                  "LDpred2-auto on HapMap3-density SNVs. Run on MSI.")
    return fig


# ------------------------------------------------------------------ figure 3: what SVs add, imputed vs true


def sv_gain_figure(grids) -> plt.Figure:
    imputed = defaultdict(list)
    true = defaultdict(list)
    for row in grids["window"]:
        imputed[row["point"]].append(row["delta_auc"])
    for row in grids["window-truth"]:
        true[row["point"]].append(row["delta_auc"])
    points = [point for point in LABELS if len(true[point]) >= 3 and len(imputed[point]) >= 3]
    summary = {point: (interval(imputed[point]), interval(true[point])) for point in points}
    points.sort(key=lambda point: summary[point][1][0])
    fig, ax = plt.subplots(figsize=(10.5, 7.6))
    fig.subplots_adjust(left=0.29, right=0.97, top=0.81, bottom=0.10)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.axvline(0, color=MUTED, linewidth=1.0, zorder=1)
    for y, point in enumerate(points):
        (mi, li, hi, ni), (mt, lt, ht, nt) = summary[point]
        ax.plot([mi, mt], [y, y], color="#CFE5EC", linewidth=5.0, solid_capstyle="round", zorder=2)
        ax.hlines(y + 0.17, lt, ht, color=WITH, linewidth=1.6, zorder=3)
        ax.hlines(y - 0.17, li, hi, color=WITH, linewidth=1.6, alpha=0.55, zorder=3)
        ax.scatter([mt], [y + 0.17], s=58, color=WITH, edgecolor="white", linewidth=1.2, zorder=4)
        ax.scatter([mi], [y - 0.17], s=50, facecolor="white", edgecolor=WITH, linewidth=1.8, zorder=4)
        significant = lt > 0
        if significant:
            ax.scatter([ht + 0.0014], [y + 0.17], marker="*", s=70, color=WITH, linewidths=0, zorder=4)
    ax.set_yticks(range(len(points)), [LABELS[point] for point in points], fontsize=10.5)
    ax.tick_params(axis="y", labelcolor=INK)
    ax.set_ylim(-0.7, len(points) - 0.3)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Test AUC gain from adding SVs to SV-PGS (with SVs minus without, same people)")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=WITH, markeredgecolor="white", markersize=8,
               label="SVs measured by their true genotypes"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=WITH,
               markeredgewidth=1.8, markersize=7.5, label="SVs imputed (Beagle, from SNV haplotypes)"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor=WITH, markeredgecolor="none", markersize=11,
               label="true-genotype 95% interval above zero"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, 0.885), ncol=3, columnspacing=1.8,
               handletextpad=0.4)
    headline(fig, "SVs add accuracy when they are measured well; imputation erases most of the gain",
             "One 3 Mb chr22 window (60,000 records), every architecture and training size (5k–40k) pooled; "
             "means with 95% intervals over replicates", top=0.975)
    footnote(fig, "Same simulated people, truth and fits; only the SV genotypes differ. Run on MSI.")
    return fig


# ------------------------------------------------------------------ figure 4: heatmap of the gain by size


def heatmap_figure(grids) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 7.8), sharey=True)
    fig.subplots_adjust(left=0.24, right=0.90, top=0.80, bottom=0.06, wspace=0.06)
    true_cells = cells(grids["window-truth"])
    order = sorted(LABELS, key=lambda point: -np.nanmean(
        [np.mean([r["delta_auc"] for r in true_cells.get((point, size), [])]) if true_cells.get((point, size))
         else np.nan for size in SIZES]))
    cmap = LinearSegmentedColormap.from_list("gain", ["#B4462F", "#F1DDD5", "#FFFFFF", "#CFE7EE", "#0E7C9B",
                                                      "#08445A"])
    norm = TwoSlopeNorm(vmin=-0.008, vcenter=0.0, vmax=0.024)
    for ax, grid, title in ((axes[0], "window-truth", "True SV genotypes"),
                            (axes[1], "window", "Imputed SV genotypes")):
        grouped = cells(grids[grid])
        matrix = np.full((len(order), len(SIZES)), np.nan)
        counts = np.zeros_like(matrix, dtype=int)
        for i, point in enumerate(order):
            for j, size in enumerate(SIZES):
                values = [row["delta_auc"] for row in grouped.get((point, size), []) if math.isfinite(row["delta_auc"])]
                counts[i, j] = len(values)
                if len(values) >= 2:
                    matrix[i, j] = float(np.mean(values))
        image = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
        for i in range(len(order)):
            for j in range(len(SIZES)):
                if math.isnan(matrix[i, j]):
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#F6F7F8", edgecolor="none"))
                    ax.text(j, i, "–", ha="center", va="center", fontsize=9, color=SPINE)
                    continue
                value = matrix[i, j] * 1000
                strong = abs(matrix[i, j]) > 0.012
                ax.text(j, i, "0.0" if abs(value) < 0.05 else f"{value:+.1f}".replace("-", "−"), ha="center", va="center", fontsize=8.6,
                        color="white" if strong else INK)
        ax.set_xticks(range(len(SIZES)), [thousands(size) for size in SIZES])
        ax.xaxis.tick_top()
        ax.tick_params(axis="x", length=0, labelcolor=INK)
        ax.set_title(title, pad=30, loc="center")
        ax.set_xticks(np.arange(-0.5, len(SIZES)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(order)), minor=True)
        ax.grid(which="minor", color="white", linewidth=2.0)
        ax.tick_params(which="minor", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_yticks(range(len(order)), [LABELS[point] for point in order], fontsize=10)
    axes[0].tick_params(axis="y", labelcolor=INK)
    bar = fig.add_axes([0.92, 0.25, 0.014, 0.45])
    colorbar = fig.colorbar(image, cax=bar)
    colorbar.outline.set_visible(False)
    colorbar.set_ticks([-0.008, 0, 0.008, 0.016, 0.024])
    colorbar.set_ticklabels(["−8", "0", "+8", "+16", "+24"])
    colorbar.ax.tick_params(labelsize=9, labelcolor=MUTED, length=0)
    colorbar.set_label("AUC gain × 1000", color=MUTED, fontsize=9.5)
    headline(fig, "Where adding SVs helps: architecture × training size",
             "Mean held-out AUC gain from adding SVs (×1000) by training samples (columns), one chr22 "
             "window; cells with fewer than two replicates are blank", top=0.975)
    footnote(fig, "Simulated binary traits; rows ordered by the true-genotype gain. Run on MSI.")
    return fig


# ------------------------------------------------------------------ figure 5: the SVs themselves


def genotype_figure(summary) -> plt.Figure:
    accuracy = summary["genotype_accuracy"]["r2_by_frequency"]
    records = np.load(LOCAL / "public" / "sv_records.npz")
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.6, 5.2), gridspec_kw={"width_ratios": [1.25, 1.0]})
    fig.subplots_adjust(left=0.065, right=0.985, top=0.70, bottom=0.14, wspace=0.24)

    horizontal_grid(left)
    for key, color, label in (("SNV", WITHOUT, "SNVs"), ("SV", WITH, "SVs")):
        bins = accuracy[key]["bins"]
        centers = np.array([math.sqrt(b["low"] * b["high"]) for b in bins])
        median = np.array([b["median"] for b in bins])
        q25 = np.array([b["q25"] for b in bins])
        q75 = np.array([b["q75"] for b in bins])
        left.fill_between(centers, q25, q75, color=color, alpha=0.14, linewidth=0, zorder=2)
        left.plot(centers, median, color=color, linewidth=2.4, zorder=3)
        left.scatter(centers, median, s=30, color=color, edgecolor="white", linewidth=1.0, zorder=4)
        left.text(centers[-1] * 1.12, median[-1], label, color=color, fontsize=11, fontweight="bold", va="center")
    left.set_xscale("log")
    left.set_xlim(5e-4, 0.65)
    left.set_ylim(-0.02, 1.04)
    left.set_xticks([0.001, 0.005, 0.01, 0.05, 0.1, 0.5], ["0.1%", "0.5%", "1%", "5%", "10%", "50%"])
    left.minorticks_off()
    left.set_xlabel("Minor allele frequency")
    left.set_ylabel("Imputed dosage r² with true genotype")
    left.axvline(0.01, color=SPINE, linewidth=0.9, linestyle=(0, (2, 2)), zorder=1)
    share = accuracy["SV"]["below_1pct"]
    left.text(0.0105, 0.04, f"{share:.0%} of SVs\nare below 1%", color=MUTED, fontsize=9, ha="left", va="bottom")
    panel_title(left, "A", "Rare SVs impute poorly", "median and interquartile range per frequency bin")

    svtype = records["svtype"]
    change = records["len_change"]
    measured = records["measured"]
    length = np.abs(change[measured]).astype(float)
    kind = svtype[measured]
    sign = change[measured]
    groups = (
        ("Deletions", (kind == "DEL") | ((kind == "") & (sign < 0)), DEL),
        ("Insertions", (kind == "INS") | ((kind == "") & (sign > 0)), INS),
        ("Duplications", kind == "DUP", DUP),
    )
    edges = np.logspace(math.log10(50), math.log10(1e6), 46)
    bottom = np.zeros(len(edges) - 1)
    horizontal_grid(right)
    for label, mask, color in groups:
        counts, _ = np.histogram(length[mask], bins=edges)
        right.bar(edges[:-1], counts, width=np.diff(edges), bottom=bottom, align="edge", color=color, alpha=0.9,
                  edgecolor="white", linewidth=0.4, label=f"{label} ({int(mask.sum()):,})", zorder=2)
        bottom += counts
    right.set_xscale("log")
    right.set_xlim(45, 1.2e6)
    right.set_xticks([50, 1e3, 1e4, 1e5, 1e6], ["50 bp", "1 kb", "10 kb", "100 kb", "1 Mb"])
    right.minorticks_off()
    right.set_xlabel("SV length")
    right.set_ylabel("SV records")
    right.legend(loc="upper right", handlelength=1.0, handleheight=1.0)
    panel_title(right, "B", "The chr22 SV catalogue", f"{int(measured.sum()):,} measured SV records, stacked by class")
    headline(fig, "The simulated SVs: mostly rare, short, and hard to impute",
             f"Beagle imputation of 50,000 people from 1000 Genomes haplotypes on chr22: "
             f"{accuracy['SV']['records']:,} SVs and {accuracy['SNV']['records']:,} sampled SNVs scored against "
             "their true genotypes", top=0.975)
    footnote(fig, "bench-sim v7 cohort (public 1000 Genomes data). Run on MSI.")
    return fig


# ------------------------------------------------------------------ figure 6: compute


def compute_figure(grids) -> plt.Figure:
    grouped = cells(grids["grid"])
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    fig.subplots_adjust(left=0.11, right=0.78, top=0.80, bottom=0.13)
    horizontal_grid(ax)
    methods = (("with_sv", "SV-PGS with SVs", WITH, 0.09, 0.78),
               ("snv_indel", "SV-PGS without SVs", WITHOUT, 0.0, 1.28), ("ldpred2_auto", "LDpred2-auto", LDPRED, -0.09, 1.0))
    for key, label, color, offset, nudge in methods:
        xs, medians = [], []
        for step, size in enumerate(SIZES):
            hours = [row["fit_seconds"][key] / 3600 for (point, n), rows in grouped.items() if n == size
                     for row in rows if isinstance(row.get("fit_seconds"), dict) and key in row["fit_seconds"]
                     and math.isfinite(row["fit_seconds"][key])]
            if len(hours) < 3:
                continue
            x = step + offset
            parts = ax.violinplot([hours], positions=[x], widths=0.16, showextrema=False)
            for body in parts["bodies"]:
                body.set_facecolor(color)
                body.set_alpha(0.18)
                body.set_edgecolor("none")
            q25, median, q75 = np.percentile(hours, [25, 50, 75])
            ax.vlines(x, q25, q75, color=color, linewidth=2.2, zorder=3)
            xs.append(x), medians.append(median)
        ax.plot(xs, medians, color=color, linewidth=2.0, zorder=3)
        ax.scatter(xs, medians, s=36, color=color, edgecolor="white", linewidth=1.1, zorder=4)
        ax.text(xs[-1] + 0.16, medians[-1] * nudge, label, color=color, fontsize=10.5, fontweight="bold", va="center")
    ax.set_yscale("log")
    ax.set_yticks([1 / 60, 5 / 60, 0.25, 1, 4, 12, 24], ["1 min", "5 min", "15 min", "1 h", "4 h", "12 h", "24 h"])
    ax.minorticks_off()
    ax.set_xticks(range(len(SIZES)), [thousands(size) for size in SIZES])
    ax.set_xlim(-0.4, len(SIZES) - 0.5)
    ax.set_xlabel("Training samples")
    ax.set_ylabel("Wall time per fit")
    headline(fig, "What the accuracy costs: fit time per replicate",
             "Whole chromosome 22; median, interquartile range and distribution over every "
             "architecture", top=0.975)
    footnote(fig, "MSI Slurm jobs, one fit per job. SV-PGS fits every variant by empirical Bayes; "
                  "LDpred2-auto fits HapMap3-density SNVs.")
    return fig


# ------------------------------------------------------------------ main


def main() -> None:
    grids = load_results()
    with open(SUMMARY) as handle:
        summary = json.load(handle)
    OUT.mkdir(parents=True, exist_ok=True)
    figures = [
        ("1_accuracy_by_training_size", accuracy_figure(grids, summary)),
        ("2_gain_over_ldpred2", ldpred2_figure(grids)),
        ("3_sv_gain_true_vs_imputed", sv_gain_figure(grids)),
        ("4_sv_gain_heatmap", heatmap_figure(grids)),
        ("5_simulated_svs", genotype_figure(summary)),
        ("6_fit_time", compute_figure(grids)),
    ]
    with PdfPages(OUT / "gallery.pdf") as gallery:
        for name, fig in figures:
            fig.savefig(OUT / f"{name}.png", dpi=300)
            fig.savefig(OUT / f"{name}.pdf")
            gallery.savefig(fig)
            plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()

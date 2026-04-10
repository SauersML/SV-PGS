"""Post-training analysis and visualization for SV-PGS model results.

Reads the coefficients and summary from a completed model run and produces
publication-quality figures with full structural variant type descriptions.

Usage:
    python scripts/analyze_model.py --model-dir ~/Downloads/hypertension_model
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Human-readable structural variant class names
# ---------------------------------------------------------------------------

SV_CLASS_LABELS: dict[str, str] = {
    "deletion_short": "Short Deletion\n(< 5 kb)",
    "deletion_long": "Long Deletion\n(\u2265 5 kb)",
    "duplication_short": "Short Duplication\n(< 5 kb)",
    "duplication_long": "Long Duplication\n(\u2265 5 kb)",
    "insertion_mei": "Mobile Element\nInsertion",
    "inversion_bnd_complex": "Inversion, Breakend,\nand Complex",
    "other_complex_sv": "Other Complex\nStructural Variant",
}

# Same but single-line for tight spaces (legends, bar charts)
SV_CLASS_LABELS_SHORT: dict[str, str] = {
    "deletion_short": "Short Deletion (< 5 kb)",
    "deletion_long": "Long Deletion (\u2265 5 kb)",
    "duplication_short": "Short Duplication (< 5 kb)",
    "duplication_long": "Long Duplication (\u2265 5 kb)",
    "insertion_mei": "Mobile Element Insertion",
    "inversion_bnd_complex": "Inversion / Breakend / Complex",
    "other_complex_sv": "Other Complex SV",
}

# Consistent colors per class
SV_CLASS_COLORS: dict[str, str] = {
    "deletion_short": "#2ca02c",
    "deletion_long": "#ff7f0e",
    "duplication_short": "#9467bd",
    "duplication_long": "#8c564b",
    "insertion_mei": "#1f77b4",
    "inversion_bnd_complex": "#d62728",
    "other_complex_sv": "#7f7f7f",
}

# Display order (largest effect types first, then by biology)
SV_CLASS_ORDER = [
    "inversion_bnd_complex",
    "insertion_mei",
    "deletion_short",
    "deletion_long",
    "duplication_short",
    "duplication_long",
    "other_complex_sv",
]


def _label(cls: str) -> str:
    return SV_CLASS_LABELS.get(cls, cls)


def _label_short(cls: str) -> str:
    return SV_CLASS_LABELS_SHORT.get(cls, cls)


def _color(cls: str) -> str:
    return SV_CLASS_COLORS.get(cls, "#333333")


def _extract_chromosome(variant_id: str) -> int | None:
    m = re.search(r"chr(\d+)", variant_id)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_coefficients(path: Path) -> list[dict]:
    rows = []
    with gzip.open(path, "rt") as f:
        reader = csv.DictReader(f, delimiter="\t")
        columns = reader.fieldnames or []
        has_length = "length" in columns
        has_af = "allele_frequency" in columns
        for row in reader:
            row["beta_float"] = float(row["beta"])
            row["abs_beta"] = abs(row["beta_float"])
            row["chromosome"] = _extract_chromosome(row["variant_id"])
            if has_length:
                row["length_float"] = float(row["length"])
            if has_af:
                row["af_float"] = float(row["allele_frequency"])
            rows.append(row)
    return rows


def load_summary(model_dir: Path) -> dict:
    for name in ("summary.json", "summary.json.gz"):
        p = model_dir / name
        if p.exists():
            if name.endswith(".gz"):
                with gzip.open(p, "rt") as f:
                    return json.load(f)
            else:
                with open(p) as f:
                    return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _save(fig, path: Path, dpi: int = 180):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path}")


def _styled_fig(*args, **kwargs):
    fig, ax = plt.subplots(*args, **kwargs)
    return fig, ax


# ---------------------------------------------------------------------------
# Individual plots
# ---------------------------------------------------------------------------

def plot_manhattan(rows: list[dict], out: Path):
    """Manhattan-style plot of absolute effect sizes across chromosomes."""
    fig, ax = plt.subplots(figsize=(16, 5))

    chrom_offsets: dict[int, int] = {}
    cumulative = 0
    by_chrom = defaultdict(list)
    for r in rows:
        c = r["chromosome"]
        if c is not None:
            by_chrom[c].append(r)

    x_all, y_all, c_all = [], [], []
    tick_positions = []
    tick_labels = []

    for chrom in sorted(by_chrom.keys()):
        items = by_chrom[chrom]
        chrom_offsets[chrom] = cumulative
        xs = [cumulative + i for i in range(len(items))]
        ys = [r["abs_beta"] for r in items]
        x_all.extend(xs)
        y_all.extend(ys)
        color = "#1f77b4" if chrom % 2 == 1 else "#ff7f0e"
        c_all.extend([color] * len(items))
        tick_positions.append(cumulative + len(items) / 2)
        tick_labels.append(str(chrom))
        cumulative += len(items)

    ax.scatter(x_all, y_all, c=c_all, s=1.5, alpha=0.4, rasterized=True, edgecolors="none")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_xlabel("Chromosome", fontsize=12)
    ax.set_ylabel("Absolute Effect Size", fontsize=12)
    ax.set_title(
        "Genome-Wide Distribution of Structural Variant Effect Sizes",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlim(-cumulative * 0.01, cumulative * 1.01)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, out / "manhattan_sv.png")


def plot_effect_size_distribution(rows: list[dict], out: Path):
    """Three-panel: raw effect size histogram, log10 absolute, and CDF."""
    betas = np.array([r["beta_float"] for r in rows])
    abs_betas = np.abs(betas)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: raw distribution
    ax = axes[0]
    ax.hist(betas, bins=200, color="#1f77b4", alpha=0.8, edgecolor="none")
    ax.axvline(0, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xlabel("Effect Size (model coefficient)", fontsize=11)
    ax.set_ylabel("Number of Variants", fontsize=11)
    ax.set_title(f"Effect Size Distribution\n({len(betas):,} active variants)", fontsize=12, fontweight="bold")

    # Panel 2: log10(|effect size|)
    ax = axes[1]
    nonzero = abs_betas[abs_betas > 0]
    ax.hist(np.log10(nonzero), bins=200, color="#ff7f0e", alpha=0.8, edgecolor="none")
    ax.set_xlabel("Log\u2081\u2080 Absolute Effect Size", fontsize=11)
    ax.set_ylabel("Number of Variants", fontsize=11)
    ax.set_title("Log-Scale Effect Size Distribution", fontsize=12, fontweight="bold")

    # Panel 3: CDF
    ax = axes[2]
    sorted_abs = np.sort(abs_betas)
    cdf = np.arange(1, len(sorted_abs) + 1) / len(sorted_abs)
    ax.plot(sorted_abs, cdf, color="#2ca02c", linewidth=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("Absolute Effect Size", fontsize=11)
    ax.set_ylabel("Cumulative Fraction of Variants", fontsize=11)
    ax.set_title("Cumulative Distribution of\nAbsolute Effect Sizes", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)

    for a in axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    fig.tight_layout()
    _save(fig, out / "effect_size_distribution.png")


def plot_effect_by_class_boxplot(rows: list[dict], out: Path):
    """Box plot of absolute effect sizes by structural variant type."""
    by_class: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_class[r["variant_class"]].append(r["abs_beta"])

    classes = [c for c in SV_CLASS_ORDER if c in by_class]
    data = [np.array(by_class[c]) for c in classes]
    labels = [f"{_label(c)}\n(n = {len(by_class[c]):,})" for c in classes]
    colors = [_color(c) for c in classes]

    fig, ax = plt.subplots(figsize=(14, 6))
    bp = ax.boxplot(
        data, tick_labels=labels, patch_artist=True, showfliers=False,
        medianprops=dict(color="black", linewidth=1.5),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Absolute Effect Size", fontsize=12)
    ax.set_title(
        "Effect Size Distribution by Structural Variant Type\n(outliers hidden for clarity)",
        fontsize=14, fontweight="bold",
    )
    ax.tick_params(axis="x", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, out / "effect_size_by_variant_type.png")


def plot_effect_by_class_violin(rows: list[dict], out: Path):
    """Violin plot showing full effect size distribution per SV type (signed)."""
    by_class: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_class[r["variant_class"]].append(r["beta_float"])

    classes = [c for c in SV_CLASS_ORDER if c in by_class]
    data = [np.array(by_class[c]) for c in classes]
    labels = [f"{_label(c)}\n(n = {len(by_class[c]):,})" for c in classes]
    colors = [_color(c) for c in classes]

    fig, ax = plt.subplots(figsize=(14, 6))
    parts = ax.violinplot(data, showmedians=True, showextrema=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("black")

    ax.axhline(0, color="red", linestyle="--", alpha=0.3, linewidth=1)
    ax.set_xticks(range(1, len(classes) + 1))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Effect Size (signed model coefficient)", fontsize=12)
    ax.set_title(
        "Effect Size Distribution by Structural Variant Type",
        fontsize=14, fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, out / "effect_size_violin_by_variant_type.png")


def plot_per_class_histograms(rows: list[dict], out: Path):
    """Stacked per-class histograms showing effect size shape for each SV type."""
    by_class: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_class[r["variant_class"]].append(r["beta_float"])

    classes = [c for c in SV_CLASS_ORDER if c in by_class]
    n = len(classes)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, cls in zip(axes, classes):
        vals = np.array(by_class[cls])
        ax.hist(vals, bins=150, color=_color(cls), alpha=0.8, edgecolor="none")
        ax.axvline(0, color="red", linestyle="--", alpha=0.4, linewidth=1)
        ax.set_ylabel("Number of\nVariants", fontsize=10)
        mean_abs = np.mean(np.abs(vals))
        ax.set_title(
            f"{_label_short(cls)}  (n = {len(vals):,},  mean absolute effect = {mean_abs:.5f})",
            fontsize=11, fontweight="bold", loc="left",
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Effect Size (model coefficient)", fontsize=11)
    fig.suptitle(
        "Effect Size Distributions by Structural Variant Type",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    _save(fig, out / "effect_size_per_variant_type.png")


def plot_class_per_chromosome(rows: list[dict], out: Path):
    """Stacked bar: variant type composition per chromosome."""
    counts: dict[int, Counter] = defaultdict(Counter)
    for r in rows:
        c = r["chromosome"]
        if c is not None:
            counts[c][r["variant_class"]] += 1

    chroms = sorted(counts.keys())
    classes = [c for c in SV_CLASS_ORDER if any(c in counts[ch] for ch in chroms)]

    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(chroms))
    bottoms = np.zeros(len(chroms))

    for cls in classes:
        vals = np.array([counts[ch].get(cls, 0) for ch in chroms])
        ax.bar(x, vals, bottom=bottoms, color=_color(cls), label=_label_short(cls), width=0.75)
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in chroms])
    ax.set_xlabel("Chromosome", fontsize=12)
    ax.set_ylabel("Number of Active Variants with Non-Zero Effects", fontsize=12)
    ax.set_title(
        "Structural Variant Type Composition per Chromosome",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, out / "variant_type_per_chromosome.png")


def plot_mean_effect_per_chromosome(rows: list[dict], out: Path):
    """Two-panel: mean absolute effect size and variant count per chromosome."""
    by_chrom: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        c = r["chromosome"]
        if c is not None:
            by_chrom[c].append(r["abs_beta"])

    chroms = sorted(by_chrom.keys())
    means = [np.mean(by_chrom[c]) for c in chroms]
    counts = [len(by_chrom[c]) for c in chroms]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    ax1.bar(range(len(chroms)), means, color="#1f77b4", width=0.7)
    ax1.set_ylabel("Mean Absolute Effect Size", fontsize=11)
    ax1.set_title("Mean Absolute Effect Size per Chromosome", fontsize=13, fontweight="bold")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2.bar(range(len(chroms)), counts, color="#ff7f0e", width=0.7)
    ax2.set_ylabel("Number of Active Variants", fontsize=11)
    ax2.set_title("Active Variant Count per Chromosome", fontsize=13, fontweight="bold")
    ax2.set_xticks(range(len(chroms)))
    ax2.set_xticklabels([str(c) for c in chroms])
    ax2.set_xlabel("Chromosome", fontsize=12)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    _save(fig, out / "effect_size_per_chromosome.png")


def plot_rank_curve(rows: list[dict], out: Path):
    """Rank plot: sorted absolute effect size on log-log, colored by SV type."""
    # Sort all by descending |beta|
    sorted_rows = sorted(rows, key=lambda r: r["abs_beta"], reverse=True)
    ranks = np.arange(1, len(sorted_rows) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each class separately for legend
    by_class_indices: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(sorted_rows):
        by_class_indices[r["variant_class"]].append(i)

    for cls in SV_CLASS_ORDER:
        if cls not in by_class_indices:
            continue
        idx = np.array(by_class_indices[cls])
        ax.scatter(
            ranks[idx],
            np.array([sorted_rows[i]["abs_beta"] for i in idx]),
            s=2, alpha=0.3, color=_color(cls),
            label=f"{_label_short(cls)} ({len(idx):,})",
            rasterized=True, edgecolors="none",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rank (variants sorted by decreasing absolute effect size)", fontsize=11)
    ax.set_ylabel("Absolute Effect Size", fontsize=11)
    ax.set_title(
        "Effect Size Rank Plot by Structural Variant Type",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=8, markerscale=4, loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, out / "effect_size_rank_by_type.png")


def plot_cumulative_variance(rows: list[dict], out: Path):
    """Cumulative contribution: fraction of total beta^2 vs number of variants."""
    abs_betas = np.array(sorted([r["abs_beta"] for r in rows], reverse=True))
    beta_sq = abs_betas ** 2
    cumsum = np.cumsum(beta_sq)
    total = cumsum[-1]
    frac = cumsum / total

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.arange(1, len(frac) + 1), frac, color="#2ca02c", linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("Number of Variants (ranked by decreasing absolute effect size)", fontsize=11)
    ax.set_ylabel("Cumulative Fraction of Total Squared Effect", fontsize=11)
    ax.set_title(
        "Cumulative Variance Contribution by Top Variants",
        fontsize=14, fontweight="bold",
    )
    ax.set_ylim(0, 1.05)

    # Annotate milestones
    for threshold, label_offset in [(0.5, (15, 10)), (0.9, (15, 10)), (0.99, (-80, -25))]:
        idx = np.searchsorted(frac, threshold)
        if idx < len(frac):
            ax.annotate(
                f"{int(threshold*100)}%: {idx+1:,} variants",
                xy=(idx + 1, frac[idx]),
                xytext=label_offset,
                textcoords="offset points",
                fontsize=9,
                arrowprops=dict(arrowstyle="->", color="gray"),
            )
            ax.plot(idx + 1, frac[idx], "ko", markersize=4)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, out / "cumulative_variance_contribution.png")


def plot_top_variants(rows: list[dict], out: Path, n: int = 50):
    """Horizontal bar chart of the top N variants by absolute effect size."""
    sorted_rows = sorted(rows, key=lambda r: r["abs_beta"], reverse=True)[:n]
    sorted_rows.reverse()  # bottom-to-top for horizontal bars

    fig, ax = plt.subplots(figsize=(12, max(6, n * 0.25)))
    y_pos = np.arange(len(sorted_rows))
    betas = [r["beta_float"] for r in sorted_rows]
    colors = [_color(r["variant_class"]) for r in sorted_rows]
    labels = [r["variant_id"].replace("AoUSVPhase2.", "") for r in sorted_rows]

    ax.barh(y_pos, betas, color=colors, height=0.7, edgecolor="none")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Effect Size (model coefficient)", fontsize=11)
    ax.set_title(
        f"Top {n} Structural Variants by Absolute Effect Size",
        fontsize=14, fontweight="bold",
    )
    ax.axvline(0, color="black", linewidth=0.5)

    # Legend for colors
    from matplotlib.patches import Patch
    legend_entries = []
    seen = set()
    for r in sorted_rows:
        cls = r["variant_class"]
        if cls not in seen:
            seen.add(cls)
            legend_entries.append(Patch(facecolor=_color(cls), label=_label_short(cls)))
    ax.legend(handles=legend_entries, fontsize=8, loc="lower right", framealpha=0.9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, out / "top_variants.png")


def plot_mean_effect_by_class(rows: list[dict], out: Path):
    """Bar chart: mean absolute effect size and count per structural variant type."""
    by_class: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_class[r["variant_class"]].append(r["abs_beta"])

    classes = [c for c in SV_CLASS_ORDER if c in by_class]
    means = [np.mean(by_class[c]) for c in classes]
    counts = [len(by_class[c]) for c in classes]
    colors = [_color(c) for c in classes]
    labels = [_label(c) for c in classes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Mean effect
    x = np.arange(len(classes))
    ax1.bar(x, means, color=colors, width=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8, ha="center")
    ax1.set_ylabel("Mean Absolute Effect Size", fontsize=11)
    ax1.set_title("Mean Absolute Effect Size\nby Structural Variant Type", fontsize=13, fontweight="bold")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Count
    ax2.bar(x, counts, color=colors, width=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8, ha="center")
    ax2.set_ylabel("Number of Active Variants", fontsize=11)
    ax2.set_title("Active Variant Count\nby Structural Variant Type", fontsize=13, fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    fig.tight_layout()
    _save(fig, out / "mean_effect_and_count_by_type.png")


def plot_length_distribution(rows: list[dict], out: Path):
    """Length distribution per structural variant type (log-scale histograms)."""
    if not any("length_float" in r for r in rows):
        print("  skipping length distribution (no length data in coefficients)")
        return

    by_class: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        length = r.get("length_float", None)
        if length is not None and length > 0:
            by_class[r["variant_class"]].append(length)

    classes = [c for c in SV_CLASS_ORDER if c in by_class and len(by_class[c]) > 10]
    if not classes:
        print("  skipping length distribution (insufficient length data)")
        return

    n = len(classes)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, cls in zip(axes, classes):
        lengths = np.array(by_class[cls])
        log_lengths = np.log10(lengths[lengths > 0])
        ax.hist(log_lengths, bins=100, color=_color(cls), alpha=0.8, edgecolor="none")
        median_len = np.median(lengths)
        ax.axvline(np.log10(median_len), color="black", linestyle="--", linewidth=1, alpha=0.7)

        # Format title with human-readable median
        if median_len >= 1_000_000:
            median_str = f"{median_len/1_000_000:.1f} Mb"
        elif median_len >= 1_000:
            median_str = f"{median_len/1_000:.1f} kb"
        else:
            median_str = f"{median_len:.0f} bp"

        ax.set_ylabel("Number of\nVariants", fontsize=10)
        ax.set_title(
            f"{_label_short(cls)}  (n = {len(lengths):,},  median length = {median_str})",
            fontsize=11, fontweight="bold", loc="left",
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Custom x-axis ticks in human units
    axes[-1].set_xlabel("Structural Variant Length", fontsize=12)
    tick_values = [0, 1, 2, 3, 4, 5, 6, 7]  # 1bp, 10bp, ..., 10Mb
    tick_labels_text = ["1 bp", "10 bp", "100 bp", "1 kb", "10 kb", "100 kb", "1 Mb", "10 Mb"]
    valid = [(v, l) for v, l in zip(tick_values, tick_labels_text) if v >= axes[-1].get_xlim()[0] and v <= axes[-1].get_xlim()[1]]
    if valid:
        axes[-1].set_xticks([v for v, _ in valid])
        axes[-1].set_xticklabels([l for _, l in valid], fontsize=10)

    fig.suptitle(
        "Length Distribution by Structural Variant Type",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    _save(fig, out / "length_distribution_by_type.png")


def plot_length_vs_effect(rows: list[dict], out: Path):
    """Scatter: SV length vs absolute effect size, per type."""
    if not any("length_float" in r for r in rows):
        print("  skipping length vs effect (no length data)")
        return

    by_class: dict[str, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
    for r in rows:
        length = r.get("length_float", None)
        if length is not None and length > 0:
            by_class[r["variant_class"]][0].append(length)
            by_class[r["variant_class"]][1].append(r["abs_beta"])

    classes = [c for c in SV_CLASS_ORDER if c in by_class and len(by_class[c][0]) > 10]
    if not classes:
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    for cls in classes:
        lengths, effects = by_class[cls]
        ax.scatter(
            lengths, effects, s=2, alpha=0.15, color=_color(cls),
            label=_label_short(cls), rasterized=True, edgecolors="none",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Structural Variant Length (base pairs)", fontsize=12)
    ax.set_ylabel("Absolute Effect Size", fontsize=12)
    ax.set_title(
        "Structural Variant Length vs. Effect Size",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=8, markerscale=5, loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, out / "length_vs_effect_size.png")


def plot_allele_frequency_spectrum(rows: list[dict], out: Path):
    """Site frequency spectrum per structural variant type."""
    if not any("af_float" in r for r in rows):
        print("  skipping allele frequency spectrum (no AF data in coefficients)")
        return

    by_class: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        af = r.get("af_float", None)
        if af is not None and af > 0:
            by_class[r["variant_class"]].append(af)

    classes = [c for c in SV_CLASS_ORDER if c in by_class and len(by_class[c]) > 10]
    if not classes:
        print("  skipping allele frequency spectrum (insufficient AF data)")
        return

    n = len(classes)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, cls in zip(axes, classes):
        afs = np.array(by_class[cls])
        # Use log-scale bins for better visualization of rare variants
        log_afs = np.log10(afs[afs > 0])
        ax.hist(log_afs, bins=100, color=_color(cls), alpha=0.8, edgecolor="none")
        median_af = np.median(afs)
        ax.axvline(np.log10(median_af), color="black", linestyle="--", linewidth=1, alpha=0.7)

        n_rare = np.sum(afs < 0.01)
        n_common = np.sum(afs >= 0.05)
        ax.set_ylabel("Number of\nVariants", fontsize=10)
        ax.set_title(
            f"{_label_short(cls)}  (n = {len(afs):,},  "
            f"rare [< 1%]: {n_rare:,},  common [\u2265 5%]: {n_common:,})",
            fontsize=11, fontweight="bold", loc="left",
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Custom x-axis ticks as percentages
    tick_values = [-4, -3, -2, -1, np.log10(0.5)]
    tick_labels_text = ["0.01%", "0.1%", "1%", "10%", "50%"]
    axes[-1].set_xticks(tick_values)
    axes[-1].set_xticklabels(tick_labels_text, fontsize=10)
    axes[-1].set_xlabel("Allele Frequency", fontsize=12)

    fig.suptitle(
        "Site Frequency Spectrum by Structural Variant Type",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    _save(fig, out / "site_frequency_spectrum_by_type.png")


def plot_af_vs_effect(rows: list[dict], out: Path):
    """Scatter: allele frequency vs absolute effect size."""
    if not any("af_float" in r for r in rows):
        print("  skipping AF vs effect (no AF data)")
        return

    by_class: dict[str, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
    for r in rows:
        af = r.get("af_float", None)
        if af is not None and af > 0:
            by_class[r["variant_class"]][0].append(af)
            by_class[r["variant_class"]][1].append(r["abs_beta"])

    classes = [c for c in SV_CLASS_ORDER if c in by_class and len(by_class[c][0]) > 10]
    if not classes:
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    for cls in classes:
        afs, effects = by_class[cls]
        ax.scatter(
            afs, effects, s=2, alpha=0.15, color=_color(cls),
            label=_label_short(cls), rasterized=True, edgecolors="none",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Allele Frequency", fontsize=12)
    ax.set_ylabel("Absolute Effect Size", fontsize=12)
    ax.set_title(
        "Allele Frequency vs. Effect Size by Structural Variant Type",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=8, markerscale=5, loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, out / "allele_frequency_vs_effect_size.png")


def plot_summary_dashboard(rows: list[dict], summary: dict, out: Path):
    """Summary dashboard with key model statistics."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # (0,0) Variant count by class - pie
    ax = axes[0, 0]
    by_class = Counter(r["variant_class"] for r in rows)
    classes = [c for c in SV_CLASS_ORDER if c in by_class]
    sizes = [by_class[c] for c in classes]
    colors = [_color(c) for c in classes]
    labels = [_label_short(c) for c in classes]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, colors=colors, autopct="%1.1f%%",
        pctdistance=0.8, startangle=90,
    )
    for t in autotexts:
        t.set_fontsize(8)
    ax.legend(
        wedges, [f"{l} ({s:,})" for l, s in zip(labels, sizes)],
        fontsize=7, loc="center left", bbox_to_anchor=(-0.3, 0.5),
    )
    ax.set_title("Active Variants by\nStructural Variant Type", fontsize=11, fontweight="bold")

    # (0,1) Effect size histogram
    ax = axes[0, 1]
    betas = np.array([r["beta_float"] for r in rows])
    ax.hist(betas, bins=150, color="#1f77b4", alpha=0.8, edgecolor="none")
    ax.axvline(0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Effect Size", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Effect Size Distribution", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (0,2) Key stats text
    ax = axes[0, 2]
    ax.axis("off")
    total_variants = summary.get("variant_count", "?")
    active = summary.get("active_variant_count", len(rows))
    samples = summary.get("sample_count", "?")
    auc = summary.get("training_auc", None)
    iters = summary.get("selected_iteration_count", "?")

    stats_text = (
        f"Model Summary\n"
        f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"Samples: {samples:,}\n"
        f"Total variants: {total_variants:,}\n"
        f"Active (non-zero): {active:,}\n"
        f"Sparsity: {100*(1 - active/total_variants):.1f}%\n"
        f"Training iterations: {iters}\n"
    )
    if auc is not None:
        stats_text += f"Training AUC: {auc:.6f}\n"
    ax.text(
        0.1, 0.5, stats_text, transform=ax.transAxes,
        fontsize=12, verticalalignment="center", fontfamily="monospace",
    )

    # (1,0) Per-chromosome count
    ax = axes[1, 0]
    by_chrom: dict[int, int] = Counter()
    for r in rows:
        c = r["chromosome"]
        if c is not None:
            by_chrom[c] += 1
    chroms = sorted(by_chrom.keys())
    ax.bar(range(len(chroms)), [by_chrom[c] for c in chroms], color="#ff7f0e", width=0.7)
    ax.set_xticks(range(len(chroms)))
    ax.set_xticklabels([str(c) for c in chroms], fontsize=8)
    ax.set_xlabel("Chromosome", fontsize=10)
    ax.set_ylabel("Active Variants", fontsize=10)
    ax.set_title("Active Variants per Chromosome", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (1,1) Mean |beta| per class bar
    ax = axes[1, 1]
    means = {c: np.mean([r["abs_beta"] for r in rows if r["variant_class"] == c]) for c in classes}
    ax.bar(range(len(classes)), [means[c] for c in classes], color=[_color(c) for c in classes], width=0.6)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([_label_short(c).split("(")[0].strip() for c in classes], fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("Mean Absolute Effect", fontsize=10)
    ax.set_title("Mean Effect by Variant Type", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (1,2) CDF
    ax = axes[1, 2]
    abs_betas = np.sort(np.array([r["abs_beta"] for r in rows]))
    cdf = np.arange(1, len(abs_betas) + 1) / len(abs_betas)
    ax.plot(abs_betas, cdf, color="#2ca02c", linewidth=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("Absolute Effect Size", fontsize=10)
    ax.set_ylabel("Cumulative Fraction", fontsize=10)
    ax.set_title("Cumulative Distribution", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Bayesian Polygenic Score Model for Hypertension\nStructural Variant Analysis Summary",
        fontsize=16, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    _save(fig, out / "model_summary_dashboard.png", dpi=200)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate analysis plots for SV-PGS model results.")
    parser.add_argument("--model-dir", required=True, help="Directory containing coefficients.tsv.gz and summary.json")
    parser.add_argument("--output-dir", default=None, help="Output directory for plots (default: same as model-dir)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.output_dir) if args.output_dir else model_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    coeff_path = model_dir / "coefficients.tsv.gz"
    if not coeff_path.exists():
        print(f"Error: {coeff_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Loading coefficients from {coeff_path}...")
    rows = load_coefficients(coeff_path)
    print(f"  {len(rows):,} active variants loaded")

    summary = load_summary(model_dir)
    if summary:
        print(f"  summary loaded: {summary.get('sample_count', '?')} samples, {summary.get('variant_count', '?')} total variants")

    print(f"\nGenerating plots in {out_dir}/...")
    plot_manhattan(rows, out_dir)
    plot_effect_size_distribution(rows, out_dir)
    plot_effect_by_class_boxplot(rows, out_dir)
    plot_effect_by_class_violin(rows, out_dir)
    plot_per_class_histograms(rows, out_dir)
    plot_class_per_chromosome(rows, out_dir)
    plot_mean_effect_per_chromosome(rows, out_dir)
    plot_rank_curve(rows, out_dir)
    plot_cumulative_variance(rows, out_dir)
    plot_top_variants(rows, out_dir)
    plot_mean_effect_by_class(rows, out_dir)
    plot_length_distribution(rows, out_dir)
    plot_length_vs_effect(rows, out_dir)
    plot_allele_frequency_spectrum(rows, out_dir)
    plot_af_vs_effect(rows, out_dir)
    plot_summary_dashboard(rows, summary, out_dir)

    print("\nDone! All plots generated.")


if __name__ == "__main__":
    main()

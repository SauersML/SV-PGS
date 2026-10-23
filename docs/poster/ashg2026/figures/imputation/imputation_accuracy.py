"""A candidate poster figure of imputed SV genotype accuracy, remade from the slides "SV imputation": 50 public 1000
Genomes samples (chr22, short reads) imputed with the All of Us long-read panel and scored against ONT long-read SV
genotypes. Panel a is each panel-frequency bin's median per-variant dosage r² (slide 1), drawn as a segment across the
bin; panel b each sequence context's dosage r² per SV allele (slide 4). Only values the slides print are drawn: slide
4's 95% intervals are drawn there but not printed, so they are left out. The values sit in imputation_numbers.json,
which stays out of git with the figures, as the panel is All of Us data.

Outputs, beside this file: sv_imputation_accuracy (both panels, one column wide) and each panel alone
(sv_imputation_by_frequency, sv_imputation_by_context), each as a PDF and a 300 dpi PNG.

    uv run --with matplotlib --with scipy python docs/poster/ashg2026/figures/imputation/imputation_accuracy.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import matplotlib  # noqa: E402

from sim_gallery import IMPUTED, INK, POSTER_LINE, POSTER_MARK, POSTER_PAD, POSTER_STYLE, plt  # noqa: E402

NUMBERS = json.loads((HERE / "imputation_numbers.json").read_text())
COLUMN = 11.4
"""Inches: one outer poster column (build_pptx.py's LEFT and RIGHT), the figure's width."""
PITCH = 0.74
"""Inches between panel b's rows: a line of poster type and a little air."""
FREQUENCY_HEIGHT = 4.2
"""Inches: panel a's axes."""
GAP = 0.9
"""Inches between panel a's x title and panel b's top row."""
LOWEST = 0.03
"""Panel a's axis starts at this frequency (%), where the open-ended lowest bin, below 0.1%, is drawn from."""


def frequency_panel(ax) -> None:
    for row in NUMBERS["by_frequency"]["bins"]:
        ax.plot([row["low"] or LOWEST, row["high"]], [row["r2"]] * 2, color=IMPUTED, linewidth=POSTER_LINE * 1.6,
                solid_capstyle="butt", clip_on=False)
    ax.set_xscale("log")
    ax.set_xlim(LOWEST, 50)
    ax.set_ylim(0, 1)
    ax.set_xticks([0.1, 1, 10], ["0.1", "1", "10"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.set_xlabel("SV frequency (%)")
    ax.set_ylabel("Median SV r²")


def context_panel(ax) -> None:
    rows = NUMBERS["by_context"]["rows"]
    for place, row in enumerate(rows):
        last = row["short"] == "All SVs"
        ax.plot(row["r2"], -place - (0.4 if last else 0), "o", color=INK if last else IMPUTED,
                markersize=POSTER_MARK * 0.8, markeredgewidth=0, clip_on=False)
    ax.set_yticks([-place - (0.4 if row["short"] == "All SVs" else 0) for place, row in enumerate(rows)],
                  [row["short"] for row in rows])
    for label in ax.get_yticklabels():
        if label.get_text() == "All SVs":
            label.set_fontweight("bold")
    ax.set_ylim(-len(rows) + 0.4 - 0.5, 0.5)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.set_xlabel("r² per SV allele")


def fit(fig, axes, heights, gaps, width, pad=POSTER_PAD) -> float:
    """Place ``axes`` stacked top to bottom at ``heights`` inches tall with ``gaps`` between, each as wide as its own
    type allows, every margin measured from the drawn type until it settles; returns the figure's height."""
    renderer = fig.canvas.get_renderer()
    inches = fig.dpi_scale_trans.inverted()
    total = sum(heights) + sum(gaps) + 2.5
    for _ in range(6):
        fig.set_size_inches(width, total)
        outer = [ax.get_tightbbox(renderer).transformed(inches) for ax in axes]
        inner = [ax.get_window_extent(renderer).transformed(inches) for ax in axes]
        lefts = [i.x0 - o.x0 + pad for i, o in zip(inner, outer)]
        rights = [o.x1 - i.x1 + pad for i, o in zip(inner, outer)]
        below = [i.y0 - o.y0 for i, o in zip(inner, outer)]
        above = [o.y1 - i.y1 for i, o in zip(inner, outer)]
        total = pad + sum(a + h + b for a, h, b in zip(above, heights, below)) + sum(gaps) + pad
        top = total - pad
        for ax, height, up, down, gap, left, right in zip(axes, heights, above, below, list(gaps) + [0], lefts, rights):
            top -= up
            ax.set_position((left / width, (top - height) / total, (width - left - right) / width, height / total))
            top -= height + down + gap
    return total


def draw(panels) -> plt.Figure:
    with plt.rc_context(POSTER_STYLE):
        fig = plt.figure(figsize=(COLUMN, 10))
        axes, heights = [], []
        for panel in panels:
            ax = fig.add_axes((0.2, 0.2, 0.6, 0.6))
            panel(ax)
            axes.append(ax)
            heights.append(FREQUENCY_HEIGHT if panel is frequency_panel
                           else PITCH * (len(NUMBERS["by_context"]["rows"]) + 0.4))
        fit(fig, axes, heights, [GAP] * (len(axes) - 1), COLUMN)
    return fig


def main() -> None:
    for name, panels in (("sv_imputation_accuracy", (frequency_panel, context_panel)),
                         ("sv_imputation_by_frequency", (frequency_panel,)),
                         ("sv_imputation_by_context", (context_panel,))):
        fig = draw(panels)
        for suffix in ("pdf", "png"):
            fig.savefig(HERE / f"{name}.{suffix}", transparent=suffix == "png")
        plt.close(fig)


if __name__ == "__main__":
    main()

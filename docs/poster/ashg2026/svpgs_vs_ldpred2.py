"""SV-PGS with SVs against LDpred2-auto on All of Us, converged fits only: each trait's change in held-out partial r²
relative to LDpred2-auto's, with its 95% interval, and a bold row for the converged traits' common change, SV-PGS
without SVs beside it. An arm's row is drawn only where that arm's SV-PGS fit converged (svgain's converged list,
svpgs_vs_ldpred2_app.py's SPEC). A trait's interval is the family bootstrap interval of the difference divided by
LDpred2-auto's r², since a ratio's own resamples are unstable where that r² is near 0. The values are svpgs_vs_ldpred2_app.py's aggregate line, saved as
figures/aou_review/svpgs_vs_ldpred2.json, which .gitignore keeps out of git with the figure until the results are
released. Not on the poster.

    uv run --with matplotlib --with scipy python docs/poster/ashg2026/svpgs_vs_ldpred2.py
"""
from __future__ import annotations

import json

import numpy as np

from aou_gallery import HERE, baseline_relative
from sim_gallery import BLUE, INK, ORANGE, plt

REVIEW = HERE / "figures" / "aou_review"
ARMS = (("with_sv", "SV-PGS with SVs", BLUE, -0.14), ("snv_indel", "SV-PGS without SVs", ORANGE, 0.14))
LIMIT = 1.5
"""The axis's half-width as a fraction of LDpred2-auto's r²; an interval past it ends in an arrowhead."""
COMBINED = "All converged traits"


def rows(found: dict) -> dict[str, tuple[float, float, float]]:
    """Per trait, (change, low, high) as fractions of LDpred2-auto's r², then the traits' common change."""
    out = {name: (difference / r2, low / r2, high / r2)
           for name, difference, low, high, r2 in zip(found["traits"], found["difference"], found["ci_low"], found["ci_high"], found["ldpred2_r2"])}
    common = found["common"]
    # the common change is a fraction of the two methods' mean r²; baseline_relative puts it over LDpred2-auto's
    out[COMBINED] = tuple(baseline_relative(common[key]) for key in ("estimate", "ci_low", "ci_high"))
    return out


def main() -> None:
    result = json.loads((REVIEW / "svpgs_vs_ldpred2.json").read_text(encoding="utf-8"))
    # one type size for every text (sim_gallery's import sets the poster's)
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
                         "axes.titlesize": 12, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "axes.edgecolor": INK,
                         "axes.linewidth": 1.0, "xtick.major.width": 1.0, "xtick.major.size": 4, "pdf.fonttype": 42})
    found = {arm: rows(result[arm]) for arm, *_ in ARMS}
    names = [name for name in dict.fromkeys(name for arm, *_ in ARMS for name in found[arm]) if name != COMBINED] + [COMBINED]
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(names) + 1.0))
    for arm, label, color, shift in ARMS:
        for index, name in enumerate(names):
            if name not in found[arm]:
                continue
            value, low, high = found[arm][name]
            y = index + shift
            ax.hlines(y, max(low, -LIMIT), min(high, LIMIT), color=color, linewidth=2.0)
            for end, edge, marker in ((low, -LIMIT, "<"), (high, LIMIT, ">")):
                if (end < edge) if marker == "<" else (end > edge):
                    ax.plot([edge], [y], marker=marker, markersize=6, color=color, markeredgewidth=0, clip_on=False)
            ax.plot([value], [y], "o", markersize=7, color=color, markeredgewidth=0, label=label if name == COMBINED else None)
    ax.axvline(0.0, color=INK, linewidth=1.0)
    ax.set_yticks(range(len(names)), names)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(names) - 0.5, -0.5)
    ax.set_xlim(-LIMIT, LIMIT)
    ticks = np.arange(-1.5, 1.51, 0.5)
    ax.set_xticks(ticks, [f"{tick * 100:.0f}".replace("-", "−") for tick in ticks])
    ax.set_xlabel("Held-out r² of SV-PGS relative to LDpred2-auto (% change), converged fits only")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, handletextpad=0.2, borderaxespad=0.2)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(REVIEW / f"svpgs_vs_ldpred2.{suffix}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()

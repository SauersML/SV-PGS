"""SV-PGS and LDpred2-auto on All of Us, converged fits only: each trait's held-out partial r² (the SV-gain test's) of
LDpred2-auto, SV-PGS without SVs and SV-PGS with SVs, with the 2.5% and 97.5% points of the joint family bootstrap, on
a log axis, since the traits' r² run from about 0.001 to 0.05. A method's mark is drawn only where its SV-PGS fit
converged (svgain's converged list, svpgs_vs_ldpred2_app.py's SPEC). The values are svpgs_vs_ldpred2_app.py's aggregate
line, saved as figures/aou_review/svpgs_vs_ldpred2.json, which .gitignore keeps out of git with the figure until the
results are released. Not on the poster.

    uv run --with matplotlib --with scipy python docs/poster/ashg2026/svpgs_vs_ldpred2.py
"""
from __future__ import annotations

import json

from aou_gallery import HERE
from sim_gallery import BLUE, INK, ORANGE, VERMILLION, plt

REVIEW = HERE / "figures" / "aou_review"
METHODS = (("ldpred2_auto", "LDpred2-auto", VERMILLION, "s", -0.2), ("snv_indel", "SV-PGS without SVs", ORANGE, "D", 0.0),
           ("with_sv", "SV-PGS with SVs", BLUE, "o", 0.2))


def main() -> None:
    found = json.loads((REVIEW / "svpgs_vs_ldpred2.json").read_text(encoding="utf-8"))["r2"]
    # one type size for every text (sim_gallery's import sets the poster's)
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
                         "axes.titlesize": 12, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "axes.edgecolor": INK,
                         "axes.linewidth": 1.0, "xtick.major.width": 1.0, "xtick.major.size": 4, "xtick.minor.width": 1.0,
                         "xtick.minor.size": 2.5, "pdf.fonttype": 42})
    names = list(found)
    fig, ax = plt.subplots(figsize=(7.0, 0.55 * len(names) + 1.1))
    ax.set_xscale("log")
    for method, label, color, marker, shift in METHODS:
        for index, name in enumerate(names):
            if method not in found[name]:
                continue
            value, low, high = found[name][method]
            ax.hlines(index + shift, max(low, 1e-5), high, color=color, linewidth=2.0)
            ax.plot([value], [index + shift], marker=marker, markersize=7, color=color, markeredgewidth=0, linestyle="none",
                    label=label if index == 0 else None)
    ax.set_yticks(range(len(names)), names)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(names) - 0.5, -0.5)
    ax.set_xlim(5e-5, 0.1)
    ax.set_xticks([0.0001, 0.001, 0.01, 0.1], ["0.0001", "0.001", "0.01", "0.1"])
    ax.set_xlabel("Held-out r²")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, frameon=False,
              handlelength=1.0, handletextpad=0.3, columnspacing=1.2, borderaxespad=0.2)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(REVIEW / f"svpgs_vs_ldpred2.{suffix}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()

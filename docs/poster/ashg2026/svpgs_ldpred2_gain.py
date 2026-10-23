"""SV-PGS (with SVs) against LDpred2-auto on All of Us held-out people, as two figures with the comparison on the y axis
and the traits along the x axis, then an "Overall" point: (1) the change in held-out partial r² as a percentage of
LDpred2-auto's r², (2) the absolute held-out r² gain. Per trait, the difference and its 2.5% and 97.5% joint family
bootstrap points are svpgs_vs_ldpred2_app.py's aggregate line (figures/aou_review/svpgs_vs_ldpred2.json, "with_sv"); the
percentage divides them by LDpred2-auto's r². Overall: (1) the poster's pooled relative change
(aou_vs_ldpred2.json "combined", sv_gain_meta's common g converted to % of LDpred2-auto's r²), (2) the fixed-effect
inverse-variance (GLS) pooled absolute difference with a normal 95% interval ("gls_absolute"). Both files are git-ignored
with the figures until the results are released. Not on the poster.

    uv run --with matplotlib --with scipy python docs/poster/ashg2026/svpgs_ldpred2_gain.py
"""
from __future__ import annotations

import json

from aou_gallery import HERE
from sim_gallery import BLUE, INK, plt

REVIEW = HERE / "figures" / "aou_review"
SHORT = {"LDL cholesterol": "LDL", "Hypothyroidism": "Hypothyroidism", "Atrial fibrillation": "Atrial\nfibrillation",
         "eGFR": "eGFR", "HDL cholesterol": "HDL"}


def points() -> tuple[list[str], list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    """Per trait then Overall: (estimate, lower, upper) as % of LDpred2-auto's r² and as absolute r²."""
    arm = json.loads((REVIEW / "svpgs_vs_ldpred2.json").read_text(encoding="utf-8"))["with_sv"]
    combined = json.loads((HERE / "aou_vs_ldpred2.json").read_text(encoding="utf-8"))["combined"]["with_sv"]
    names, percent, absolute = [], [], []
    for name, difference, low, high, base in zip(arm["traits"], arm["difference"], arm["ci_low"], arm["ci_high"],
                                                 arm["ldpred2_r2"]):
        names.append(SHORT.get(name, name))
        absolute.append((difference, low, high))
        percent.append((100 * difference / base, 100 * low / base, 100 * high / base))
    pooled = arm["gls_absolute"]
    names.append("Overall")
    percent.append((combined["relative"], combined["relative_lower"], combined["relative_upper"]))
    absolute.append((pooled["estimate"], pooled["estimate"] - 1.96 * pooled["standard_error"],
                     pooled["estimate"] + 1.96 * pooled["standard_error"]))
    return names, percent, absolute


def draw(names: list[str], values: list[tuple[float, float, float]], title: str, path: str) -> None:
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12,
                         "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                         "axes.edgecolor": INK, "axes.linewidth": 1.0, "pdf.fonttype": 42})
    # the traits, then a gap of half a slot before Overall
    places = list(range(len(names) - 1)) + [len(names) - 0.5]
    fig, ax = plt.subplots(figsize=(0.95 * len(names) + 1.6, 3.6))
    ax.axhline(0, color=INK, linewidth=1.0)
    for place, (value, low, high) in zip(places, values):
        ax.vlines(place, low, high, color=BLUE, linewidth=2.0)
        ax.plot([place], [value], marker="o", markersize=7, color=BLUE, markeredgewidth=0, linestyle="none")
    ax.set_xticks(places, names)
    ax.set_xlim(places[0] - 0.6, places[-1] + 0.6)
    ax.set_ylabel(title)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for suffix in (".pdf", ".png"):
        fig.savefig(REVIEW / f"{path}{suffix}", dpi=200)
    plt.close(fig)


def main() -> None:
    names, percent, absolute = points()
    draw(names, percent, "Change in held-out r² vs LDpred2-auto (%)", "svpgs_ldpred2_percent_gain")
    draw(names, absolute, "Held-out r² gain over LDpred2-auto", "svpgs_ldpred2_r2_gain")
    for name, (p, pl, ph), (a, al, ah) in zip(names, percent, absolute):
        print(f"{name.replace(chr(10), ' ')}: {p:+.1f}% [{pl:+.1f}, {ph:+.1f}]  {a:+.5f} [{al:+.5f}, {ah:+.5f}]")


if __name__ == "__main__":
    main()

"""The All of Us fits' learned prior and posterior effect variance per allele for each SV class relative to SNVs, from
aou_numbers.json's "prior_posterior" rows (aggregate values alone, typed in from the workspace like the gallery's other
numbers; the file, like the figures, stays out of git until the results are released). Neither figure is on the poster.

Each row is one (trait, class) of a fitted with_sv model: the class's learned prior variance per allele over the SNVs'
(``prior``, a point estimate: the fit saves no uncertainty for its hyperparameters), the class mean of the posterior
second moment E[beta^2 | data] per allele over the SNVs' (``posterior``), and its 95% interval (``posterior_lower``,
``posterior_upper``). That interval combines the saved posterior law's own spread (4,000 exact draws: a mixture
component, then every SV record's lattice node and normal; the SNVs' million-record mean normal per component) with a
200-block delete-one jackknife of the log ratio across the genome. ``fitted_effect`` is the same ratio of squared
posterior means, and ``posterior_variance_share`` the class's share of E[beta^2 | data] that is posterior variance.

Outputs, in figures/aou_review/ (each a .png and a .pdf), outside build_pptx.py's glob of figures/aou_*.png, so no
poster build places them, and inside .gitignore's figures/aou_*:
- prior_by_trait: a row per trait and class, the prior a black tick, the posterior a dot with its interval;
- prior_pooled: each class pooled across traits, over every trait and without gout, whose prior alone gives SV
  alleles larger variance than SNV alleles. A trait's estimate is its log posterior ratio with standard error
  (log upper - log lower) / (2 x 1.96); the traits are pooled by a random-effects model, tau^2 by REML and the mean's
  interval by Hartung-Knapp-Sidik-Jonkman (t on k - 1 degrees of freedom). The prior's tick is the geometric mean of
  the traits' priors. The pooled values, Cochran's Q p, I^2 and each leave-one-trait-out value are printed.

    uv run --with matplotlib --with scipy python docs/poster/ashg2026/aou_prior.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy import optimize, stats

HERE = Path(__file__).resolve().parent
NUMBERS = HERE / "aou_numbers.json"
OUT = HERE / "figures" / "aou_review"
TRAITS = {"ldl_cholesterol": "LDL cholesterol", "hypothyroidism": "Hypothyroidism",
          "atrial_fibrillation": "Atrial fibrillation", "egfr_ckd_epi_2021": "eGFR", "gout": "Gout"}
"""The traits in the order drawn, each with its name on the figures."""
GOUT = "gout"
INK = "#000000"
CLASSES = {"deletion": ("Deletions", "#E69F00"), "insertion": ("Insertions", "#009E73"),
           "str_vntr_repeat": ("Tandem repeats", "#8C6D1F")}
"""Each SV class's name and colour: the simulation figures' deletion orange and insertion green, and a brown no poster
figure uses for anything else."""
SIZE = 24
STYLE = {
    "font.family": "Arial", "font.size": SIZE, "axes.labelsize": SIZE, "xtick.labelsize": SIZE,
    "axes.linewidth": 2.5, "xtick.major.width": 2.5, "xtick.major.size": 9, "axes.spines.top": False,
    "axes.spines.right": False, "axes.spines.left": False, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "savefig.dpi": 200, "pdf.fonttype": 42,
}
Z = 1.959963984540054
MARK, BAR, TICK = 17, 5, 30
"""Points: a posterior's dot, its interval's line, and the prior's tick."""


def reml(y: np.ndarray, v: np.ndarray) -> float:
    """tau^2 maximizing the restricted likelihood of y_i ~ N(mu, v_i + tau^2); 0 where the likelihood is no higher
    inside than at the boundary."""
    def negative(log_tau2: float) -> float:
        total = v + math.exp(log_tau2)
        w = 1.0 / total
        mu = np.sum(w * y) / np.sum(w)
        return 0.5 * (np.sum(np.log(total)) + math.log(np.sum(w)) + np.sum(w * (y - mu) ** 2))

    best = optimize.minimize_scalar(negative, bounds=(-30.0, 10.0), method="bounded")
    return 0.0 if negative(-30.0) <= best.fun + 1e-9 else math.exp(best.x)


def pool(y: np.ndarray, v: np.ndarray) -> dict:
    """The REML random-effects mean of log ratios ``y`` (variances ``v``) with its HKSJ 95% interval, on the ratio
    scale, with tau, Cochran's Q p (inverse-variance weights) and I^2."""
    k = y.size
    tau2 = reml(y, v)
    w = 1.0 / (v + tau2)
    mu = float(np.sum(w * y) / np.sum(w))
    half = stats.t.ppf(0.975, k - 1) * math.sqrt(float(np.sum(w * (y - mu) ** 2)) / (k - 1) / np.sum(w))
    fixed_weights = 1.0 / v
    q = float(np.sum(fixed_weights * (y - np.sum(fixed_weights * y) / np.sum(fixed_weights)) ** 2))
    return {"ratio": math.exp(mu), "low": math.exp(mu - half), "high": math.exp(mu + half), "tau": math.sqrt(tau2),
            "q_p": float(stats.chi2.sf(q, k - 1)), "i2": max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0}


def pooled(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """Per (class, "all" or "without_gout"): the traits' pooled posterior ratio, the geometric mean of their priors,
    and the pooled ratio with each trait left out."""
    found = {}
    for name in CLASSES:
        for subset in ("all", "without_gout"):
            chosen = [row for trait in TRAITS for row in rows if row["trait"] == trait and row["class"] == name
                      and (subset == "all" or trait != GOUT)]
            y = np.array([math.log(row["posterior"]) for row in chosen])
            v = np.array([((math.log(row["posterior_upper"]) - math.log(row["posterior_lower"])) / (2 * Z)) ** 2 for row in chosen])
            entry = pool(y, v)
            entry["prior"] = float(np.exp(np.mean([math.log(row["prior"]) for row in chosen])))
            entry["leave_one_out"] = {TRAITS[row["trait"]]: pool(np.delete(y, i), np.delete(v, i))["ratio"]
                                      for i, row in enumerate(chosen)}
            found[(name, subset)] = entry
    return found


def log_axis(ax, limits: tuple[float, float], ticks: list[float]) -> None:
    ax.set_xscale("log")
    ax.set_xlim(*limits)
    ax.set_xticks(ticks, [f"{tick:g}×" for tick in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_yticks([])


def mark(ax, y: float, posterior: float, low: float, high: float, prior: float, colour: str, limits) -> None:
    """A posterior dot with its interval in ``colour``, and the prior's black tick on the same row."""
    ax.hlines(y, max(low, limits[0]), min(high, limits[1]), color=colour, linewidth=BAR, zorder=3)
    ax.plot([posterior], [y], "o", color=colour, markersize=MARK, markeredgewidth=0, zorder=4)
    ax.plot([prior], [y], "|", color=INK, markersize=TICK, markeredgewidth=4, zorder=5)


def by_trait_figure(rows: list[dict]):
    """A row per trait and class: the learned prior and the posterior with its 95% interval."""
    traits = [trait for trait in TRAITS if any(row["trait"] == trait for row in rows)]
    width, height, left, bottom, top = 16.0, 2.0 + 1.55 * len(traits), 3.7, 1.45, 0.35
    limits, offsets = (0.002, 100.0), (-0.26, 0.0, 0.26)
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes((left / width, bottom / height, (width - left - 0.5) / width, (height - bottom - top) / height))
    log_axis(ax, limits, [0.01, 0.1, 1, 10, 100])
    ax.set_ylim(len(traits) - 0.55, -0.45)
    for y, trait in enumerate(traits):
        ax.text(-0.03, y, TRAITS[trait], transform=ax.get_yaxis_transform(), ha="right", va="center")
        for offset, (name, (_, colour)) in zip(offsets, CLASSES.items()):
            row = next(row for row in rows if row["trait"] == trait and row["class"] == name)
            mark(ax, y + offset, row["posterior"], row["posterior_lower"], row["posterior_upper"], row["prior"], colour, limits)
    ax.set_xlabel("Effect variance per allele, relative to SNVs")
    # the key, in the empty left of the last row (gout's, whose intervals start above 3x): the classes' names on its
    # three rows, then the two marks
    last = len(traits) - 1
    for offset, (label, colour) in zip(offsets, CLASSES.values()):
        ax.text(0.0024, last + offset, label, color=colour, ha="left", va="center")
    ax.hlines(last - 0.15, 0.07, 0.13, color=INK, linewidth=BAR)
    ax.plot([0.095], [last - 0.15], "o", color=INK, markersize=MARK, markeredgewidth=0)
    ax.text(0.17, last - 0.15, "Posterior, 95% CI", ha="left", va="center")
    ax.plot([0.095], [last + 0.2], "|", color=INK, markersize=TICK, markeredgewidth=4)
    ax.text(0.17, last + 0.2, "Learned prior", ha="left", va="center")
    return fig


def pooled_figure(found: dict, traits: int):
    """A pair of rows per class, over every trait and without gout: the pooled posterior with its 95% interval and the
    traits' geometric-mean prior, with the SNVs' 1 drawn."""
    width, height, left, bottom, top = 16.0, 7.6, 5.6, 1.45, 0.3
    limits = (0.1, 20.0)
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes((left / width, bottom / height, (width - left - 0.5) / width, (height - bottom - top) / height))
    log_axis(ax, limits, [0.1, 0.2, 0.5, 1, 2, 5, 10, 20])
    ax.set_ylim(5.6, -0.6)
    ax.axvline(1.0, color=INK, linewidth=2.5, zorder=1)
    for index, (name, (label, colour)) in enumerate(CLASSES.items()):
        for offset, (subset, text) in enumerate((("all", f"All {traits} traits"), ("without_gout", "Without gout"))):
            entry = found[(name, subset)]
            y = 2 * index + offset
            mark(ax, y, entry["ratio"], entry["low"], entry["high"], entry["prior"], colour, limits)
            ax.text(-0.03, y, text, transform=ax.get_yaxis_transform(), ha="right", va="center")
        ax.text(-(left - 0.25) / (width - left - 0.5), 2 * index + 0.5, label, transform=ax.get_yaxis_transform(),
                ha="left", va="center")
    ax.set_xlabel("Posterior effect variance per allele, relative to SNVs")
    # the key, in the empty right of the deletions' rows
    ax.hlines(0.5, 2.5, 3.7, color=INK, linewidth=BAR)
    ax.plot([3.05], [0.5], "o", color=INK, markersize=MARK, markeredgewidth=0)
    ax.text(4.3, 0.5, "Pooled, 95% CI", ha="left", va="center")
    ax.plot([3.05], [1.25], "|", color=INK, markersize=TICK, markeredgewidth=4)
    ax.text(4.3, 1.25, "Learned prior", ha="left", va="center")
    return fig


def main() -> None:
    rows = json.loads(NUMBERS.read_text())["prior_posterior"]["rows"]
    found = pooled(rows)
    for (name, subset), entry in found.items():
        print(f"{name} {subset}: {entry['ratio']:.3g} ({entry['low']:.3g}, {entry['high']:.3g}), tau {entry['tau']:.3g}, "
              f"I2 {entry['i2']:.0%}, Q p {entry['q_p']:.2g}, prior {entry['prior']:.3g}, leaving one out "
              + ", ".join(f"{trait} {value:.3g}" for trait, value in entry["leave_one_out"].items()))
    traits = len({row["trait"] for row in rows})
    OUT.mkdir(exist_ok=True)
    with plt.rc_context(STYLE):
        for name, fig in (("prior_by_trait", by_trait_figure(rows)), ("prior_pooled", pooled_figure(found, traits))):
            fig.savefig(OUT / f"{name}.png")
            with PdfPages(OUT / f"{name}.pdf") as pdf:
                pdf.savefig(fig)
            plt.close(fig)
            print(OUT / f"{name}.pdf")


if __name__ == "__main__":
    main()

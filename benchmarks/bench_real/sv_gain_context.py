"""Post hoc context for the All of Us SV-gain results, from aggregates already in hand. Nothing here is a test of the
pre-specified families (SV_GAIN_PRESPEC.txt, FASTPGS_SCREEN_PRESPEC.txt); every output is labelled post hoc.

- ``screen``: the wide screen's one-sided per-trait p (LDpred2-auto, with_sv against snv_indel) against the uniform
  law they follow when no trait gains: a Q-Q plot, and the count below 0.05 against its binomial expectation. Input is
  a TSV of trait, gain text, one-sided p and BH q, typed from ~/fp/screen/screen_result.txt (aggregates only, no counts).
- ``cap``: how predictive the part of the with_sv score that the snv_indel score does not explain must be for a given
  gain. With both scores standardized after the covariates, write the with_sv score a = rho b + s e, s = sqrt(1 - rho^2),
  e uncorrelated with the snv_indel score b. Then r_a = rho r_b + s r_e exactly, so the relative gain in partial r^2 is
  f = (rho + s k)^2 - 1 with k = r_e / r_b, and a gain f needs k = (sqrt(1 + f) - rho) / s. rho is the arms' held-out
  score correlation (sv_gain_meta.py's "arm agreement"), which uses no outcome.
- ``simulation``: the simulations' mean gain from imputed SVs (poster_sv_gain, test AUC) on the relative partial r^2
  scale All of Us reports, through the binormal model AUC = Phi(d / sqrt 2) with r proportional to d, at the mean
  SV-PGS test AUC over the training sizes the gain averages (fig1). An approximation for scale only.

    uv run --with matplotlib --with scipy python benchmarks/bench_real/sv_gain_context.py screen|cap|simulation
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

POSTER = Path(__file__).resolve().parents[2] / "docs" / "poster" / "ashg2026"
REVIEW = POSTER / "figures" / "aou_review"
LABEL = "POST HOC"


def screen() -> None:
    rows = [line.split("\t") for line in (REVIEW / "screen_p.tsv").read_text().splitlines() if line.strip()]
    names = [row[0] for row in rows]
    p = np.array([float(row[2]) for row in rows])
    k = p.size
    below = int(np.sum(p < 0.05))
    print(f"{LABEL}: {k} screen traits; one-sided p < 0.05 for {below} (expected {0.05 * k:.2f} with no gain anywhere; "
          f"P(at least {below}) = {stats.binom.sf(below - 1, k, 0.05):.3g}); p < 0.5 for {int(np.sum(p < 0.5))} (expected {k / 2:.1f}); "
          f"Kolmogorov-Smirnov p against uniform {stats.kstest(p, 'uniform').pvalue:.3g}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "text.color": "black", "axes.labelcolor": "black", "xtick.color": "black",
                         "ytick.color": "black", "axes.edgecolor": "black", "pdf.fonttype": 42})
    order = np.argsort(p)
    expected = -np.log10((np.arange(1, k + 1) - 0.5) / k)
    observed = -np.log10(p[order])
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    top = max(expected.max(), observed.max()) * 1.08
    ax.plot([0, top], [0, top], color="black", linewidth=1)
    ax.plot(expected, observed, "o", color="#0072B2", markersize=5, markeredgewidth=0)
    for rank, index in enumerate(order[:2]):
        ax.annotate(names[index].replace("_", " "), (expected[rank], observed[rank]), xytext=(-6, 0), textcoords="offset points",
                    ha="right", va="center")
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel("Expected −log10 p")
    ax.set_ylabel("Observed −log10 p")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(REVIEW / f"screen_qq.{suffix}", dpi=200)
    print(f"wrote {REVIEW / 'screen_qq.pdf'}")


def needed_ratio(gain: float, one_minus_rho: float) -> float:
    rho = 1 - one_minus_rho
    return (math.sqrt(1 + gain) - rho) / math.sqrt(1 - rho * rho)


def cap(values: list[float]) -> None:
    print(f"{LABEL}: k = r_e / r_b needed for a relative gain f, by 1 - rho (r_e: the trait correlation of the with_sv score's part "
          f"the snv_indel score does not explain; r_b: the snv_indel score's)")
    gains = (0.005, 0.01, 0.02, 0.05)
    print("1-rho\tscore change (SD)\t" + "\t".join(f"f={gain:.1%}" for gain in gains))
    for value in values:
        print(f"{value:g}\t{math.sqrt(2 * value):.4f}\t" + "\t".join(f"{needed_ratio(gain, value):.3f}" for gain in gains))


def simulation() -> None:
    numbers = json.loads((POSTER / "figures" / "gallery" / "numbers.json").read_text())
    rows = numbers["poster_sv_gain"]["rows"]
    accuracy = numbers["fig1"]
    sizes = accuracy["train_counts"]
    base = float(np.mean([entry[0] for size, entry in zip(sizes, accuracy["SV-PGS"]) if size <= 20000]))
    d = math.sqrt(2) * stats.norm.ppf(base)

    def relative(auc_gain: float) -> float:
        return (math.sqrt(2) * stats.norm.ppf(base + auc_gain) / d) ** 2 - 1

    print(f"{LABEL}, approximate: simulated SV gain as a relative partial r^2 gain at mean SV-PGS test AUC {base:.3f} (training sizes <= 20,000)")
    for name, row in rows.items():
        for source in ("true", "imputed"):
            if source in row:
                mean, low, high = row[source][:3]
                print(f"  {name} {source}: AUC {mean:+.4f} [{low:+.4f}, {high:+.4f}] -> {relative(mean):+.1%} [{relative(low):+.1%}, {relative(high):+.1%}]")


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "screen":
        screen()
    elif command == "cap":
        cap([float(value) for value in sys.argv[2:]] or [0.0002, 0.0004, 0.002, 0.019])
    else:
        simulation()

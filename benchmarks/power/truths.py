"""The simulation grid's SV gain tested over simulated truths, the grid's unit of replication.

A grid replicate is one simulated truth: one draw of causal variants, effects and phenotypes
(``grid.scenario_parameters``), shared by every training size and by both genotype sources, and scored on the same
held-out people. A task's gain from adding SVs is ``grid.compare``'s: the with_sv arm's test AUC less the separately
fitted snv_indel arm's, on the same people, with its paired DeLong z. The gain varies far more between truths than the
DeLong standard error within one. In most truths it is near 0, and in the few whose causal SVs have large effects it is
0.01 to 0.06. So inference runs over truths:

- a truth's gain is its mean over the training sizes both genotype sources completed;
- a mean gain over truths has a Student-t 95% interval on k - 1 degrees of freedom and a one-sided sign-flip p, which
  is exact when, under the null, each truth's gain is symmetric about 0;
- the gain with imputed SV genotypes as a share of the gain with true ones is the ratio of their means over the same
  truths, with Fieller's 95% interval from the paired covariance.

Counting each training size of a truth as its own replicate makes intervals too narrow, since the sizes share the
truth's causal SVs and test phenotypes. Weighting truths by their DeLong variance pulls the mean toward 0, since the
DeLong standard error is near 0 wherever the with_sv fit barely uses its SVs.
"""
from __future__ import annotations

import itertools
import math
from collections import defaultdict
from collections.abc import Iterable

import numpy as np
from scipy import stats

EXACT_LIMIT = 16
"""Up to this many truths the sign-flip p enumerates every sign pattern; above it, it draws FLIP_DRAWS of them."""
FLIP_DRAWS = 1 << 17


def truth_gains(true_rows: Iterable[dict], imputed_rows: Iterable[dict]) -> dict[str, list[tuple[int, float, float]]]:
    """Each point's truths as (replicate, gain with true SV genotypes, gain with imputed ones): the mean ``delta_auc``
    over the training sizes where both sources have a finite gain. A truth with no such size is left out."""
    gains: dict[str, dict] = {"true": defaultdict(dict), "imputed": defaultdict(dict)}
    for source, rows in (("true", true_rows), ("imputed", imputed_rows)):
        for row in rows:
            value = row.get("delta_auc")
            if value is not None and math.isfinite(value):
                gains[source][(row["point"], row["replicate"])][row["train_count"]] = value
    truths: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    for (point, replicate), true in sorted(gains["true"].items()):
        imputed = gains["imputed"].get((point, replicate), {})
        sizes = sorted(set(true) & set(imputed))
        if sizes:
            truths[point].append((replicate, float(np.mean([true[size] for size in sizes])),
                                  float(np.mean([imputed[size] for size in sizes]))))
    return dict(truths)


def sign_flip_p(values, seed: int = 0) -> float:
    """One-sided p for a positive mean: the share of sign patterns whose mean is at least the observed mean, over every
    pattern for up to EXACT_LIMIT values and otherwise over FLIP_DRAWS random ones, as (hits + 1) / (draws + 1)."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return math.nan
    observed = values.mean() - 1e-12 * np.abs(values).max()
    if values.size <= EXACT_LIMIT:
        signs = np.array(list(itertools.product((1.0, -1.0), repeat=values.size)))
        return float(np.mean((signs * values).mean(axis=1) >= observed))
    generator, hits = np.random.default_rng(seed), 0
    for _ in range(FLIP_DRAWS // 4096):
        signs = generator.choice((1.0, -1.0), size=(4096, values.size))
        hits += int(np.sum((signs * values).mean(axis=1) >= observed))
    return float((hits + 1) / (FLIP_DRAWS + 1))


def mean_test(values) -> dict[str, float | int]:
    """The mean of independent truths' gains, its Student-t 95% interval and its one-sided sign-flip p (no interval
    below two truths)."""
    values = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    k = values.size
    mean = float(values.mean()) if k else math.nan
    if k < 2:
        return {"mean": mean, "lower": math.nan, "upper": math.nan, "truths": k, "p": math.nan}
    half = float(stats.t.ppf(0.975, k - 1) * values.std(ddof=1) / math.sqrt(k))
    return {"mean": mean, "lower": mean - half, "upper": mean + half, "truths": k, "p": sign_flip_p(values)}


def fieller(numerator, denominator, level: float = 0.95) -> tuple[float, float, float]:
    """The ratio of two means over the same truths and Fieller's interval for it, from the paired covariance of the
    truths' values on k - 1 degrees of freedom. The interval is (-inf, inf) unless the denominator's mean is clear of
    0 at ``level``."""
    numerator, denominator = np.asarray(numerator, dtype=np.float64), np.asarray(denominator, dtype=np.float64)
    k = numerator.size
    top, bottom = float(numerator.mean()), float(denominator.mean())
    ratio = top / bottom if bottom else math.nan
    if k < 3:
        return ratio, -math.inf, math.inf
    covariance = np.cov(np.vstack([numerator, denominator])) / k
    t = float(stats.t.ppf(0.5 + level / 2, k - 1))
    a = bottom ** 2 - t ** 2 * covariance[1, 1]
    b = top * bottom - t ** 2 * covariance[0, 1]
    c = top ** 2 - t ** 2 * covariance[0, 0]
    discriminant = b ** 2 - a * c
    if a <= 0 or discriminant < 0:
        return ratio, -math.inf, math.inf
    root = math.sqrt(discriminant)
    return ratio, (b - root) / a, (b + root) / a

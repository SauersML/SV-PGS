"""The simulation grid's SV gain tested over simulated truths (benchmarks/power/truths.py)."""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from benchmarks.power import truths

SIZES = (5_000, 10_000, 20_000)


def _rows(effects: np.ndarray, share: float, generator: np.random.Generator, noise: float = 0.001) -> tuple[list, list]:
    """One point's tasks: each truth's effect at every size and genotype source, plus each task's own test noise."""
    rows = {"true": [], "imputed": []}
    for replicate, effect in enumerate(effects):
        for size in SIZES:
            for source, scale in (("true", 1.0), ("imputed", share)):
                rows[source].append({"point": "p", "replicate": replicate, "train_count": size,
                                     "delta_auc": scale * effect + noise * generator.standard_normal()})
    return rows["true"], rows["imputed"]


def test_a_truth_is_its_mean_over_the_sizes_both_sources_completed():
    true = [{"point": "p", "replicate": 0, "train_count": 5_000, "delta_auc": 0.01},
            {"point": "p", "replicate": 0, "train_count": 10_000, "delta_auc": 0.03},
            {"point": "p", "replicate": 0, "train_count": 20_000, "delta_auc": 0.5},
            {"point": "p", "replicate": 1, "train_count": 5_000, "delta_auc": 0.02}]
    imputed = [{"point": "p", "replicate": 0, "train_count": 5_000, "delta_auc": 0.0},
               {"point": "p", "replicate": 0, "train_count": 10_000, "delta_auc": 0.02},
               {"point": "p", "replicate": 0, "train_count": 20_000, "delta_auc": math.nan},
               {"point": "p", "replicate": 2, "train_count": 5_000, "delta_auc": 0.02}]
    assert truths.truth_gains(true, imputed) == {"p": [(0, pytest.approx(0.02), pytest.approx(0.01))]}


def test_the_sign_flip_p_counts_every_pattern_at_least_as_large():
    assert truths.sign_flip_p([1.0, 2.0, 3.0]) == pytest.approx(1 / 8)
    assert truths.sign_flip_p([1.0, -2.0, 3.0]) == pytest.approx(3 / 8)
    assert truths.sign_flip_p(np.ones(20)) == pytest.approx(1 / (truths.FLIP_DRAWS + 1))


def test_tests_over_truths_hold_their_level_where_sizes_counted_as_replicates_do_not():
    """Five truths of a heavy-tailed gain symmetric about 0, each at three training sizes: the sign-flip test and the
    t interval over truths reject no more than their level, and a t interval over the 15 tasks rejects far more."""
    generator = np.random.default_rng(7)
    simulations = 3000
    flip = interval = pooled = 0
    for _ in range(simulations):
        true, imputed = _rows(0.01 * generator.standard_t(3, size=5), 0.4, generator)
        test = truths.mean_test([gain for _, gain, _ in truths.truth_gains(true, imputed)["p"]])
        flip += test["p"] <= 0.05
        interval += test["lower"] > 0
        tasks = np.array([row["delta_auc"] for row in true])
        pooled += tasks.mean() - stats.t.ppf(0.975, tasks.size - 1) * tasks.std(ddof=1) / math.sqrt(tasks.size) > 0
    allowance = 3 * math.sqrt(0.05 * 0.95 / simulations)
    assert flip / simulations <= 0.05 + allowance
    assert interval / simulations <= 0.025 + allowance
    assert pooled / simulations > 0.1


def test_a_planted_gain_is_found():
    generator = np.random.default_rng(3)
    true, imputed = _rows(0.01 + 0.002 * generator.standard_normal(20), 0.4, generator)
    test = truths.mean_test([gain for _, gain, _ in truths.truth_gains(true, imputed)["p"]])
    assert test["truths"] == 20 and test["lower"] > 0 and test["p"] < 0.001


def test_fieller_covers_a_planted_share():
    generator = np.random.default_rng(11)
    simulations, covered = 2000, 0
    for _ in range(simulations):
        true = generator.exponential(0.004, 40) + 0.002 * generator.standard_normal(40)
        imputed = 0.4 * true + 0.002 * generator.standard_normal(40)
        _, lower, upper = truths.fieller(imputed, true)
        covered += lower <= 0.4 <= upper
    assert abs(covered / simulations - 0.95) <= 3 * math.sqrt(0.05 * 0.95 / simulations)


def test_fieller_is_unbounded_when_the_denominator_mean_is_not_clear_of_zero():
    generator = np.random.default_rng(5)
    denominator = generator.standard_normal(10)
    ratio, lower, upper = truths.fieller(denominator + 1.0, denominator - denominator.mean())
    assert (lower, upper) == (-math.inf, math.inf)

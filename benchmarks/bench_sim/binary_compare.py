"""Paired comparison of two methods' probabilities on bench-sim's binary dev scenarios [sim].

For each scenario the test samples' labels (the dev truth's ``phenotype``) against each method's ``probability`` (and
AUC also on its ``linear`` predictor): AUC, log loss, Brier score, the logistic recalibration's slope and intercept and
calibration in the large (``sv_pgs.binary_likelihood``), each with a 95% percentile bootstrap interval over test
samples, and the paired differences (first method minus second) by the same resamples.

usage: python -m benchmarks.bench_sim.binary_compare <cohort> <dev dir> <results A> <results B> <scenario>...
(results X/scenario_NNN/prediction.npz), printing one JSON line per scenario.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from sv_pgs.binary_likelihood import auc, brier_score, calibration, log_loss

RESAMPLES = 2000
"""Bootstrap resamples per interval: the percentile interval's Monte Carlo error is small against its width at 2000."""
LEVEL = 0.95


def metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    fitted = calibration(labels, probability)
    return {
        "auc": auc(labels, probability), "log_loss": log_loss(labels, probability), "brier": brier_score(labels, probability),
        "calibration_slope": fitted.slope, "calibration_intercept": fitted.intercept, "calibration_in_the_large": fitted.in_the_large,
    }


def interval(values: np.ndarray) -> list[float]:
    tail = 100.0 * (1.0 - LEVEL) / 2.0
    return [float(np.percentile(values, tail)), float(np.percentile(values, 100.0 - tail))]


def compare(labels: np.ndarray, first: np.ndarray, second: np.ndarray, seed: int) -> dict:
    point_first, point_second = metrics(labels, first), metrics(labels, second)
    generator = np.random.default_rng(seed)
    draws_first = {name: [] for name in point_first}
    draws_second = {name: [] for name in point_first}
    for _ in range(RESAMPLES):
        rows = generator.integers(0, labels.shape[0], labels.shape[0])
        if labels[rows].min() == labels[rows].max():
            continue
        one, two = metrics(labels[rows], first[rows]), metrics(labels[rows], second[rows])
        for name in point_first:
            draws_first[name].append(one[name])
            draws_second[name].append(two[name])
    result = {}
    for name in point_first:
        a, b = np.array(draws_first[name]), np.array(draws_second[name])
        result[name] = {
            "first": point_first[name], "first_ci": interval(a), "second": point_second[name], "second_ci": interval(b),
            "difference": point_first[name] - point_second[name], "difference_ci": interval(a - b),
        }
    return result


def main() -> None:
    cohort, dev, first_root, second_root = (Path(argument) for argument in sys.argv[1:5])
    is_test = np.load(cohort / "samples.npz")["is_test"]
    for scenario in sys.argv[5:]:
        labels = np.load(dev / scenario / "truth.npz")["phenotype"][is_test]
        first = np.load(first_root / scenario / "prediction.npz")
        second = np.load(second_root / scenario / "prediction.npz")
        record = {"scenario": scenario, "test_samples": int(labels.shape[0]), "test_prevalence": float(labels.mean())}
        record["probability"] = compare(labels, first["probability"], second["probability"], seed=int(scenario.split("_")[-1]))
        record["auc_linear"] = {"first": auc(labels, first["linear"]), "second": auc(labels, second["linear"])}
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()

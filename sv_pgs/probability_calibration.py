"""Independent calibration and frozen proper scores. Never fit on evaluation labels."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.special import expit

from sv_pgs.held_out_comparison import (
    _binary_inputs, cluster_totals, comparison_from_estimate, logistic_recalibration_coefficients,
)


def calibration_mask(training, families, fraction, seed):
    """Reserve whole families from the eligible development set without reading outcomes."""
    training = np.asarray(training, dtype=bool)
    families = np.asarray(families)
    if families.shape != training.shape or not 0 <= fraction < 1:
        raise ValueError("Calibration needs aligned families and a fraction in [0, 1)")
    result = np.zeros_like(training)
    if not fraction:
        return result
    groups = np.unique(families[training])
    count = max(1, int(round(fraction * len(groups))))
    if count >= len(groups):
        raise ValueError("Not enough independent families for training and calibration")
    chosen = np.random.default_rng(seed).permutation(groups)[:count]
    result = training & np.isin(families, chosen)
    return result


@dataclass(frozen=True)
class FrozenCalibration:
    intercept: float
    slope: float

    @classmethod
    def fit(cls, scores, labels):
        intercept, slope = logistic_recalibration_coefficients(scores, labels)
        return cls(float(intercept), float(slope))

    def logits(self, scores):
        scores = np.asarray(scores, dtype=float)
        if not np.all(np.isfinite(scores)):
            raise ValueError("Calibration scores must be finite")
        return self.intercept + self.slope * scores


def proper_scores(labels, logits):
    labels, logits = _binary_inputs(labels, logits)
    return {"log_loss": np.logaddexp(0, logits) - labels * logits,
            "brier": (expit(logits) - labels) ** 2}


def paired_frozen_gain(labels, baseline_logits, candidate_logits, clusters=None, *, metric="log_loss"):
    """Positive favours candidate. Conditions on frozen fitted models and calibrators."""
    difference = proper_scores(labels, baseline_logits)[metric] - proper_scores(labels, candidate_logits)[metric]
    totals = cluster_totals(difference - difference.mean(), clusters)
    return comparison_from_estimate(float(difference.mean()), float(np.linalg.norm(totals) / difference.size))

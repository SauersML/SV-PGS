"""Frozen probability calibration of held-out scores, and the proper scores of its probabilities.

A disease model fitted on the Gaussian working likelihood has a linear predictor that is not a probability. A logistic
calibration maps it to one, and is fitted on labels other than the ones its probabilities are scored on: recalibrating
each score on the evaluated labels themselves credits a reversed score as fully as the score it reverses.
Primary evaluation reserves independent development families for ``FrozenCalibration.fit``. Other-fold
out-of-fold predictions depend on an outer fold's labels through their training; they are not independent
calibration data for that fold.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.special import expit

from sv_pgs._typing import F64Array, NDArray
from sv_pgs.held_out_comparison import (
    PairedComparison,
    RecalibrationSeparationError,
    _binary_inputs,
    cluster_totals,
    comparison_from_estimate,
    logistic_recalibration_coefficients,
)


@dataclass(frozen=True)
class FrozenCalibration:
    """A logistic map from a score to a logit, intercept + slope * score, fitted once and then applied unchanged."""

    intercept: float
    slope: float

    @classmethod
    def fit(cls, scores: NDArray, labels: NDArray) -> FrozenCalibration:
        """The maximum-likelihood map whose probability never falls as the score rises (slope >= 0).

        The log-likelihood is concave in (intercept, slope), so its profile in the slope is too, and the profile's
        derivative at slope 0 is n p (1 - p) times the cases' mean score less the controls' (p the prevalence). Where
        that difference is not positive the maximum over slope >= 0 is at slope 0, the prevalence for everyone: a score
        that ranks the calibration rows' cases below their controls gets no credit, where a free slope would reverse it
        and credit it in full. Elsewhere it is the unconstrained maximum (``logistic_recalibration_coefficients``),
        whose slope is then positive, or none where the score separates the labels (``RecalibrationSeparationError``).
        """
        labels, scores = _binary_inputs(labels, scores)
        positive = labels == 1.0
        if not positive.any() or positive.all():
            raise ValueError("a calibration needs both cases and controls")
        if np.mean(scores[positive]) <= np.mean(scores[~positive]):
            prevalence = float(np.mean(positive))
            return cls(math.log(prevalence / (1.0 - prevalence)), 0.0)
        intercept, slope = logistic_recalibration_coefficients(scores, labels)
        return cls(float(intercept), float(slope))

    def logits(self, scores: NDArray) -> F64Array:
        """The frozen logits of ``scores``."""
        scores = np.asarray(scores, dtype=np.float64)
        if not np.all(np.isfinite(scores)):
            raise ValueError("Calibration scores must be finite")
        return self.intercept + self.slope * scores


def proper_scores(labels: NDArray, logits: NDArray) -> dict[str, F64Array]:
    """Each row's log loss and Brier score of the probability expit(logit)."""
    labels, logits = _binary_inputs(labels, logits)
    return {"log_loss": np.logaddexp(0, logits) - labels * logits,
            "brier": (expit(logits) - labels) ** 2}


def paired_frozen_gain(
    labels: NDArray, baseline_logits: NDArray, candidate_logits: NDArray, clusters: NDArray | None = None, *, metric: str = "log_loss"
) -> PairedComparison:
    """The mean per-person reduction in ``metric`` (``proper_scores``) from the baseline's frozen probabilities to the
    candidate's, positive favouring the candidate (for log loss, the gain in log likelihood), with a family-clustered
    standard error conditional on the fitted models and their frozen calibrations."""
    difference = proper_scores(labels, baseline_logits)[metric] - proper_scores(labels, candidate_logits)[metric]
    totals = cluster_totals(difference - difference.mean(), clusters)
    return comparison_from_estimate(float(difference.mean()), float(np.linalg.norm(totals) / difference.size))


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

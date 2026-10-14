"""Held-out comparison of a baseline and a candidate polygenic score.

The primary SV-PGS claim is that the SV-aware score (candidate) predicts better
than the SNV-only score (baseline) fitted by the same method on the same folds.
Every test here returns a one-sided z statistic: positive values favour the
candidate. The hypotheses differ, so the tests are not interchangeable:

* equal accuracy (the primary claim): the candidate is no better than the
  baseline. Paired delta R^2 for quantitative traits; for binary traits the
  paired gain in log likelihood of probabilities from a frozen calibration
  (``probability_calibration``), never one fitted on the labels it scores, and
  paired delta AUC as a secondary check.
* encompassing (nested): the candidate adds any information at all. It rejects
  even when the two scores are equally accurate, so it never supports
  "predicts better" and is not implemented here.

Cross-fitting yields K different predictors, one per held-out fold, each with
its own fitted scale. Pooling their scores into one correlation adds a
between-fold component that no per-sample variance sees. With empirical-Bayes
and sparse fits under an equal-accuracy null it inflated the z SD to about 10.
Clusters (family ids, coded 0..C-1) make every variance family-block robust;
folds must keep each family inside one fold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import special, stats

from sv_pgs._typing import NDArray


@dataclass(frozen=True)
class PairedComparison:
    estimate: float
    standard_error: float
    z_score: float


def comparison_from_estimate(estimate: float, standard_error: float) -> PairedComparison:
    """A zero observed variance cannot establish infinite evidence of improvement.

    Identical scores have zero difference and z=0. A nonzero difference with no
    estimated sampling variance has undefined inference (for example, two AUCs
    with constant DeLong components); retain its estimate but return NaN for z.
    """
    if standard_error == 0.0:
        z_score = 0.0 if estimate == 0.0 else math.nan
    else:
        z_score = estimate / standard_error
    return PairedComparison(estimate, standard_error, z_score)


def _binary_inputs(labels: NDArray, score: NDArray) -> tuple[NDArray, NDArray]:
    labels = np.asarray(labels, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    if labels.ndim != 1 or score.shape != labels.shape or not labels.size:
        raise ValueError("labels and scores must be nonempty, aligned one-dimensional arrays")
    if not np.all(np.isfinite(score)) or not np.all((labels == 0.0) | (labels == 1.0)):
        raise ValueError("scores must be finite and labels must be binary (0 or 1)")
    return labels, score


def cluster_totals(values: NDArray, clusters: NDArray | None) -> NDArray:
    """Sum per-sample values within each cluster (family block)."""
    if clusters is None:
        return np.asarray(values, dtype=np.float64)
    return np.bincount(clusters, weights=np.asarray(values, dtype=np.float64))


def _standardized(values: NDArray) -> NDArray:
    centered = np.asarray(values, dtype=np.float64) - float(np.mean(values))
    spread = float(np.std(centered))
    return centered / spread if spread > 0.0 else np.full_like(centered, np.nan)


def _correlation_influence(outcome: NDArray, score: NDArray) -> tuple[float, NDArray]:
    standardized_outcome = _standardized(outcome)
    standardized_score = _standardized(score)
    correlation = float(np.mean(standardized_outcome * standardized_score))
    influence = standardized_outcome * standardized_score - 0.5 * correlation * (
        standardized_outcome**2 + standardized_score**2
    )
    return correlation, influence


def delta_r2_influence(outcome: NDArray, baseline: NDArray, candidate: NDArray) -> tuple[float, NDArray]:
    """Delta R^2 = corr(y, candidate)^2 - corr(y, baseline)^2 and its per-sample influence."""
    baseline_correlation, baseline_influence = _correlation_influence(outcome, baseline)
    candidate_correlation, candidate_influence = _correlation_influence(outcome, candidate)
    influence = 2.0 * candidate_correlation * candidate_influence - 2.0 * baseline_correlation * baseline_influence
    delta = candidate_correlation**2 - baseline_correlation**2
    return delta, influence


def paired_delta_r2(
    outcome: NDArray,
    baseline: NDArray,
    candidate: NDArray,
    clusters: NDArray | None = None,
) -> PairedComparison:
    """Paired delta R^2 on one held-out set that never trained either score,
    with a family-block influence-function standard error."""
    delta, influence = delta_r2_influence(outcome, baseline, candidate)
    totals = cluster_totals(influence, clusters)
    standard_error = math.sqrt(float(totals @ totals)) / outcome.shape[0]
    return comparison_from_estimate(delta, standard_error)


class RecalibrationSeparationError(ValueError):
    """A logistic recalibration has no finite maximum because its score separates the labels."""


def logistic_recalibration_coefficients(score: NDArray, labels: NDArray) -> NDArray:
    """Maximum-likelihood (a, b) of logit P(y=1) = a + b*score.

    The maximum is finite and unique unless the score separates the labels (Albert and Anderson 1984),
    which is refused. A constant score has an intercept-only fit with slope zero.
    Damped Newton ascends the strictly concave log-likelihood and stops once the
    Newton decrement, half of which estimates the remaining gain, is below the fp64 resolution of the
    log-likelihood. The maximum is equivariant under an affine map of the score, and the Newton decrement
    is invariant under it, so the fit runs on the standardized score, where the linear predictor carries
    no cancellation, and is mapped back. A step is taken only if it strictly raises the evaluated
    log-likelihood; when every trial length down to the coefficients' own resolution fails to, the
    remaining gain is below the rounding of the log-likelihood sum and the coefficients are returned.
    """
    labels, score = _binary_inputs(labels, score)
    positive = labels == 1.0
    if not positive.any() or positive.all():
        raise ValueError("recalibration needs both cases and controls")
    prevalence = float(np.mean(positive))
    if np.ptp(score) == 0.0:
        # Only the intercept is identifiable. The zero slope represents the
        # unique fitted probability, the observed prevalence.
        return np.array([math.log(prevalence / (1.0 - prevalence)), 0.0])
    positive_scores, negative_scores = score[positive], score[~positive]
    if positive_scores.min() >= negative_scores.max() or positive_scores.max() <= negative_scores.min():
        raise RecalibrationSeparationError("the score separates the labels, so its logistic recalibration has no finite maximum")
    location = float(np.mean(score))
    spread = float(np.std(score))
    design = np.column_stack([np.ones_like(score), (score - location) / spread])

    def log_likelihood(coefficients: NDArray) -> float:
        linear = design @ coefficients
        return float(np.sum(labels * linear - np.logaddexp(0.0, linear)))

    coefficients = np.array([math.log(prevalence / (1.0 - prevalence)), 0.0])
    current = log_likelihood(coefficients)
    ascending = True
    while ascending:
        fitted = special.expit(design @ coefficients)
        gradient = design.T @ (labels - fitted)
        step = np.linalg.solve(design.T @ (design * (fitted * (1.0 - fitted))[:, None]), gradient)
        if 0.5 * float(gradient @ step) <= np.finfo(np.float64).eps * abs(current):
            break
        length = 1.0
        while True:
            candidate = coefficients + length * step
            if np.array_equal(candidate, coefficients):
                ascending = False
                break
            candidate_value = log_likelihood(candidate)
            if candidate_value > current:
                coefficients, current = candidate, candidate_value
                break
            length *= 0.5
    return np.array([coefficients[0] - coefficients[1] * location / spread, coefficients[1] / spread])


@dataclass(frozen=True)
class AucComponents:
    """One score's AUC and its DeLong structural components (Sun and Xu 2014 midrank form): ``positive`` per case, the
    fraction of controls it outranks, and ``negative`` per control, the fraction of cases that outrank it."""

    auc: float
    positive: NDArray
    negative: NDArray


def auc_components(labels: NDArray, score: NDArray) -> AucComponents:
    """The AUC of ``score`` for the 0/1 ``labels`` and its DeLong components, ties counted half."""
    labels, score = _binary_inputs(labels, score)
    positive = labels == 1
    positive_count = int(positive.sum())
    negative_count = positive.shape[0] - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("an AUC needs both cases and controls")
    score = np.asarray(score, dtype=np.float64)
    positive_scores = score[positive]
    negative_scores = score[~positive]
    combined_ranks = stats.rankdata(np.concatenate([positive_scores, negative_scores]))
    positive_ranks = stats.rankdata(positive_scores)
    negative_ranks = stats.rankdata(negative_scores)
    return AucComponents(
        auc=float((combined_ranks[:positive_count].sum() / positive_count - (positive_count + 1.0) / 2.0) / negative_count),
        positive=(combined_ranks[:positive_count] - positive_ranks) / negative_count,
        negative=1.0 - (combined_ranks[positive_count:] - negative_ranks) / positive_count,
    )


def delta_auc_influence(labels: NDArray, baseline: NDArray, candidate: NDArray) -> tuple[float, NDArray]:
    """DeLong's delta AUC (candidate less baseline) and its per-sample influence: each case's and each control's
    centred placement difference over their count."""
    baseline_components = auc_components(labels, baseline)
    candidate_components = auc_components(labels, candidate)
    positive_difference = candidate_components.positive - baseline_components.positive
    negative_difference = candidate_components.negative - baseline_components.negative
    positive = np.asarray(labels) == 1
    influence = np.empty(positive.shape[0], dtype=np.float64)
    influence[positive] = (positive_difference - positive_difference.mean()) / positive_difference.size
    influence[~positive] = (negative_difference - negative_difference.mean()) / negative_difference.size
    return candidate_components.auc - baseline_components.auc, influence


def paired_delta_auc(
    labels: NDArray, baseline: NDArray, candidate: NDArray, clusters: NDArray | None = None,
) -> PairedComparison:
    """DeLong paired delta AUC, optionally with family-clustered influence variance.

    The clustered form sums the case/control influences within each family and
    applies the C/(C-1) cluster correction. It includes covariance between related
    cases and controls, which the independent-participant DeLong variance omits.
    """
    positive = np.asarray(labels) == 1
    positive_count, negative_count = int(positive.sum()), int((~positive).sum())
    if min(positive_count, negative_count) < 2:
        return comparison_from_estimate(auc_components(labels, candidate).auc - auc_components(labels, baseline).auc, math.nan)
    estimate, influence = delta_auc_influence(labels, baseline, candidate)
    if clusters is None:
        # DeLong's variance: each class's placement differences' sample variance over its count.
        variance = sum(
            float(np.var(influence[members], ddof=1)) * members.sum() for members in (positive, ~positive)
        )
    else:
        clusters = np.asarray(clusters)
        if clusters.shape != np.asarray(labels).shape:
            raise ValueError("clusters must be aligned with labels")
        _families, dense_clusters = np.unique(clusters, return_inverse=True)
        if _families.size < 2:
            return comparison_from_estimate(estimate, math.nan)
        totals = cluster_totals(influence, dense_clusters)
        variance = float(totals @ totals) * _families.size / (_families.size - 1)
    standard_error = math.sqrt(variance)
    return comparison_from_estimate(estimate, standard_error)


def power_weights(expected_z: NDArray) -> NDArray:
    """Pre-registered panel weights: proportional to each trait's expected z, never negative."""
    clipped = np.clip(np.asarray(expected_z, dtype=np.float64), 0.0, None)
    total = float(clipped.sum())
    if total <= 0.0:
        raise ValueError("every trait has non-positive expected z; the panel has no power")
    return clipped / total


def influence_correlation(per_trait_influence: Sequence[NDArray], clusters: NDArray | None) -> NDArray:
    """Correlation of per-trait statistics from their family-block summed influences.

    Every trait is scored on the same people (an unobserved person's influence is 0), so the joint family-block
    resampling distribution is captured exactly by the covariance of the cluster-summed influence functions, with no
    bootstrap needed."""
    totals = np.stack([cluster_totals(influence, clusters) for influence in per_trait_influence])
    covariance = totals @ totals.T
    scale = np.sqrt(np.diag(covariance))
    return covariance / np.outer(scale, scale)


def panel_z(per_trait_z: NDArray, weights: NDArray, correlation: NDArray) -> float:
    """Pre-registered weighted Stouffer statistic across traits, correlation-aware: w'z / sqrt(w'Rw)."""
    if np.any(weights < 0.0):
        raise ValueError("panel weights must be non-negative")
    spread = float(weights @ correlation @ weights)
    return float(weights @ per_trait_z) / math.sqrt(spread)


def hommel_adjusted(p_values: NDArray) -> NDArray:
    """Hommel family-wise adjusted p-values (valid under positive dependence)."""
    p_values = np.asarray(p_values, dtype=np.float64)
    count = p_values.shape[0]
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    adjusted = sorted_p.copy()
    for subset_size in range(count, 1, -1):
        simes_tail = float(np.min(subset_size * sorted_p[-subset_size:] / np.arange(1, subset_size + 1)))
        adjusted[-subset_size:] = np.maximum(adjusted[-subset_size:], simes_tail)
        head = count - subset_size
        adjusted[:head] = np.maximum(adjusted[:head], np.minimum(subset_size * sorted_p[:head], simes_tail))
    adjusted = np.maximum(adjusted, sorted_p)
    result = np.empty(count)
    result[order] = np.minimum(adjusted, 1.0)
    return result

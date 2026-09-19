"""Held-out comparison of a baseline and a candidate polygenic score.

The primary SV-PGS claim is that the SV-aware score (candidate) predicts better
than the SNV-only score (baseline) fitted by the same method on the same folds.
Every test here returns a one-sided z statistic: positive values favour the
candidate. The hypotheses differ, so the tests are not interchangeable:

* equal accuracy (the primary claim): the candidate is no better than the
  baseline. Paired delta R^2 for quantitative traits, paired delta log-loss for
  binary traits, paired delta AUC as a secondary check.
* encompassing (nested): the candidate adds any information at all. It rejects
  even when the two scores are equally accurate, so it never supports
  "predicts better" and is not implemented here.

Cross-fitting yields K different predictors, one per held-out fold, each with
its own fitted scale. Pooling their scores into one correlation adds a
between-fold component that no per-sample variance sees. With empirical-Bayes
and sparse fits under an equal-accuracy null it inflated the z SD to about 10.
`cross_fit_delta_r2` therefore evaluates each fold's predictor on its own fold
and averages with weights n_f / n. The fold statistics still share training
samples: every cross-fold pair of samples enters twice, because each sample's
outcome trains the other folds' predictors. Its variance adds the closed-form
pair term for linear (Gaussian empirical-Bayes posterior-mean) predictors.
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


@dataclass(frozen=True)
class CrossFitArm:
    """One arm's cross-fitted linear predictor, fold by fold.

    `fold_designs[fold]` holds the standardized design rows of that fold, and
    `training_inverses[fold]` is the posterior covariance, up to the noise
    scale, of the fit trained without that fold. The out-of-fold score of
    sample i is then x_i' M_fold(i) sum_{l outside fold(i)} x_l y_l.
    """

    fold_designs: Sequence[NDArray]
    training_inverses: Sequence[NDArray]


def cluster_totals(values: NDArray, clusters: NDArray | None) -> NDArray:
    """Sum per-sample values within each cluster (family block)."""
    if clusters is None:
        return np.asarray(values, dtype=np.float64)
    return np.bincount(clusters, weights=np.asarray(values, dtype=np.float64))


def _standardized(values: NDArray) -> NDArray:
    centered = np.asarray(values, dtype=np.float64) - float(np.mean(values))
    return centered / float(np.std(centered))


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
    return PairedComparison(delta, standard_error, delta / standard_error)


def cross_fit_delta_r2(
    outcome: NDArray,
    baseline: NDArray,
    candidate: NDArray,
    fold_of_sample: NDArray,
    baseline_arm: CrossFitArm,
    candidate_arm: CrossFitArm,
    clusters: NDArray | None = None,
) -> PairedComparison:
    """Fold-stratified paired delta R^2 of cross-fitted scores with the pair-corrected variance.

    The estimate is sum_f (n_f / n) * DeltaR^2_f, each fold's own predictor
    evaluated on its own fold. As a per-sample sum it is (1/n) sum_i psi_i,
    with psi_i the fold-standardized delta-R^2 influence. Its variance is
    (sum over family blocks of psi^2 + P) / n^2. The pair term
    P = sum over cross-fold pairs a_{il} a_{li} uses
    a_{il} = sum_arm w_{arm,i} H^{arm}_{il} (y_l - mean y), where
    H_{il} = x_i' M_fold(i) x_l is the arm's cross-fold smoother and w the
    derivative of psi_i with respect to sample i's own score. In closed form,
    P = sum_{f != g} sum_{arm, other} tr(M_{arm,f} C^{arm,other}_g M_{other,g} C^{other,arm}_f),
    with C^{arm,other}_g = X_{arm,g}' diag(w_{other,g} * e_g) X_{other,g}.
    Fold labels are 0..K-1, and each arm's `fold_designs[fold]` holds the rows
    of the samples with that label, in increasing sample index.
    """
    outcome = np.asarray(outcome, dtype=np.float64)
    sample_count = outcome.shape[0]
    fold_count = int(np.max(fold_of_sample)) + 1
    rows_by_fold = [np.flatnonzero(fold_of_sample == fold) for fold in range(fold_count)]
    influence = np.zeros(sample_count)
    weights = [np.zeros(sample_count), np.zeros(sample_count)]
    estimate = 0.0
    for rows in rows_by_fold:
        fold_outcome = outcome[rows]
        delta, fold_influence = delta_r2_influence(fold_outcome, baseline[rows], candidate[rows])
        estimate += rows.shape[0] / sample_count * delta
        influence[rows] = fold_influence
        standardized_outcome = _standardized(fold_outcome)
        for arm_index, (sign, score) in enumerate(((-1.0, baseline[rows]), (1.0, candidate[rows]))):
            correlation, _ = _correlation_influence(fold_outcome, score)
            weights[arm_index][rows] = (
                sign * 2.0 * correlation * (standardized_outcome - correlation * _standardized(score))
                / float(np.std(score))
            )
    deviation = outcome - float(np.mean(outcome))
    arms = (baseline_arm, candidate_arm)
    coupling = {}
    for fold, rows in enumerate(rows_by_fold):
        for arm_index, arm in enumerate(arms):
            for other_index, other in enumerate(arms):
                scaled = other.fold_designs[fold] * (weights[other_index][rows] * deviation[rows])[:, None]
                coupling[(fold, arm_index, other_index)] = arm.fold_designs[fold].T @ scaled
    pair = 0.0
    for fold in range(fold_count):
        for other_fold in range(fold_count):
            if other_fold == fold:
                continue
            for arm_index, arm in enumerate(arms):
                for other_index, other in enumerate(arms):
                    left = arm.training_inverses[fold] @ coupling[(other_fold, arm_index, other_index)]
                    right = other.training_inverses[other_fold] @ coupling[(fold, other_index, arm_index)]
                    pair += float(np.sum(left * right.T))
    totals = cluster_totals(influence, clusters)
    standard_error = math.sqrt(float(totals @ totals) + pair) / sample_count
    return PairedComparison(estimate, standard_error, estimate / standard_error)


def logistic_recalibration_coefficients(score: NDArray, labels: NDArray) -> NDArray:
    """Maximum-likelihood (a, b) of logit P(y=1) = a + b*score.

    The maximum is finite and unique unless the score separates the labels (Albert and Anderson 1984),
    which is refused. Damped Newton ascends the strictly concave log-likelihood and stops once the
    Newton decrement, half of which estimates the remaining gain, is below the fp64 resolution of the
    log-likelihood. The maximum is equivariant under an affine map of the score, and the Newton decrement
    is invariant under it, so the fit runs on the standardized score, where the linear predictor carries
    no cancellation, and is mapped back. A step is taken only if it strictly raises the evaluated
    log-likelihood; when every trial length down to the coefficients' own resolution fails to, the
    remaining gain is below the rounding of the log-likelihood sum and the coefficients are returned.
    """
    score = np.asarray(score, dtype=np.float64)
    positive = labels == 1.0
    if not positive.any() or positive.all():
        raise ValueError("recalibration needs both cases and controls")
    positive_scores, negative_scores = score[positive], score[~positive]
    if positive_scores.min() >= negative_scores.max() or positive_scores.max() <= negative_scores.min():
        raise ValueError("the score separates the labels, so its logistic recalibration has no finite maximum")
    location = float(np.mean(score))
    spread = float(np.std(score))
    design = np.column_stack([np.ones_like(score), (score - location) / spread])

    def log_likelihood(coefficients: NDArray) -> float:
        linear = design @ coefficients
        return float(np.sum(labels * linear - np.logaddexp(0.0, linear)))

    prevalence = float(np.mean(positive))
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


def _logistic_recalibration(score: NDArray, labels: NDArray) -> NDArray:
    """Per-sample log-likelihood of labels under the maximum-likelihood recalibration of the score."""
    coefficients = logistic_recalibration_coefficients(score, labels)
    linear = coefficients[0] + coefficients[1] * np.asarray(score, dtype=np.float64)
    return labels * linear - np.logaddexp(0.0, linear)


def paired_delta_log_loss(
    labels: NDArray,
    baseline: NDArray,
    candidate: NDArray,
    clusters: NDArray | None = None,
) -> PairedComparison:
    """Paired log-loss improvement of the recalibrated candidate over the recalibrated baseline.

    Each score is recalibrated by a two-parameter logistic fit on the held-out
    set, so the comparison is of discrimination and shape, not of intercepts.
    """
    labels = np.asarray(labels, dtype=np.float64)
    difference = _logistic_recalibration(candidate, labels) - _logistic_recalibration(baseline, labels)
    totals = cluster_totals(difference - float(np.mean(difference)), clusters)
    sample_count = labels.shape[0]
    standard_error = math.sqrt(float(totals @ totals)) / sample_count
    estimate = float(np.mean(difference))
    return PairedComparison(estimate, standard_error, estimate / standard_error)


def paired_delta_auc(labels: NDArray, baseline: NDArray, candidate: NDArray) -> PairedComparison:
    """DeLong paired delta AUC (Sun and Xu 2014 midrank form, positives ranked first)."""
    positive = np.asarray(labels) == 1
    positive_count = int(positive.sum())
    negative_count = positive.shape[0] - positive_count
    components = []
    aucs = []
    for score in (baseline, candidate):
        score = np.asarray(score, dtype=np.float64)
        positive_scores = score[positive]
        negative_scores = score[~positive]
        combined_ranks = stats.rankdata(np.concatenate([positive_scores, negative_scores]))
        positive_ranks = stats.rankdata(positive_scores)
        negative_ranks = stats.rankdata(negative_scores)
        auc = (combined_ranks[:positive_count].sum() / positive_count - (positive_count + 1.0) / 2.0) / negative_count
        positive_component = (combined_ranks[:positive_count] - positive_ranks) / negative_count
        negative_component = 1.0 - (combined_ranks[positive_count:] - negative_ranks) / positive_count
        aucs.append(auc)
        components.append((positive_component, negative_component))
    positive_difference = components[1][0] - components[0][0]
    negative_difference = components[1][1] - components[0][1]
    variance = float(np.var(positive_difference, ddof=1)) / positive_count + float(
        np.var(negative_difference, ddof=1)
    ) / negative_count
    estimate = aucs[1] - aucs[0]
    standard_error = math.sqrt(variance)
    return PairedComparison(estimate, standard_error, estimate / standard_error)


def permute_within_groups(group_labels: NDArray, rng: np.random.Generator) -> NDArray:
    """Sample permutation that shuffles only within each group (the A3null construction).

    Applying it to the SV columns keeps each ancestry group's SV frequencies and
    the SV class structure while breaking every SV-outcome and SV-SNV link.
    """
    permutation = np.arange(group_labels.shape[0])
    for group in np.unique(group_labels):
        members = np.flatnonzero(group_labels == group)
        permutation[members] = rng.permutation(members)
    return permutation


def power_weights(expected_z: NDArray) -> NDArray:
    """Pre-registered panel weights: proportional to each trait's expected z, never negative."""
    clipped = np.clip(np.asarray(expected_z, dtype=np.float64), 0.0, None)
    total = float(clipped.sum())
    if total <= 0.0:
        raise ValueError("every trait has non-positive expected z; the panel has no power")
    return clipped / total


def influence_correlation(per_trait_influence: Sequence[NDArray], clusters: NDArray | None) -> NDArray:
    """Correlation of per-trait statistics from their family-block summed influences.

    Every trait is scored on the same people, so the joint family-block
    resampling distribution is captured exactly by the covariance of the
    cluster-summed influence functions, with no bootstrap needed.
    """
    totals = np.stack([cluster_totals(influence, clusters) for influence in per_trait_influence])
    covariance = totals @ totals.T
    scale = np.sqrt(np.diag(covariance))
    return covariance / np.outer(scale, scale)


def panel_z(per_trait_z: NDArray, weights: NDArray, correlation: NDArray) -> float:
    """Pre-registered weighted Stouffer statistic across traits, correlation-aware."""
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


def size_gate(null_z: NDArray, alpha: float = 0.05) -> bool:
    """Gate G13: the one-sided test's rejections at `alpha` among the null replicates are not
    significantly more than an `alpha` share of them, by the exact one-sided binomial test at the
    same `alpha`."""
    null_z = np.asarray(null_z)
    rejections = int(np.sum(null_z > stats.norm.isf(alpha)))
    return float(stats.binom.sf(rejections - 1, null_z.shape[0], alpha)) > alpha


def null_cost_gate(r2_without: NDArray, r2_with: NDArray, alpha: float = 0.05) -> bool:
    """Gate G13c: adding columns with no effect does not make held-out R^2 significantly worse.

    Paired over null replicates by the exact one-sided sign test at `alpha`: it fails only when the
    replicates where R^2 drops are significantly more than half of the untied ones. The test needs
    no distributional assumption beyond independent replicates, and its result is the same on the
    absolute and the relative R^2 scale.
    """
    difference = np.asarray(r2_with, dtype=np.float64) - np.asarray(r2_without, dtype=np.float64)
    losses = int(np.sum(difference < 0.0))
    untied = int(np.sum(difference != 0.0))
    return float(stats.binom.sf(losses - 1, untied, 0.5)) > alpha

"""Held-out comparison tests (sv_pgs.held_out_comparison), with the size gate that checks their null calibration.

The null-calibration checks simulate frozen scores on a held-out set, the situation the primary SV-PGS claim meets.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import special, stats
from sklearn.metrics import roc_auc_score

from sv_pgs._typing import NDArray
from sv_pgs.held_out_comparison import (
    logistic_recalibration_coefficients,
    paired_delta_auc,
    paired_delta_r2,
)
from sv_pgs.probability_calibration import FrozenCalibration, paired_frozen_gain, proper_scores


def size_gate(null_z: NDArray, alpha: float = 0.05) -> bool:
    """Gate G13: the one-sided test's rejections at `alpha` among the null replicates are not
    significantly more than an `alpha` share of them, by the exact one-sided binomial test at the
    same `alpha`."""
    null_z = np.asarray(null_z)
    rejections = int(np.sum(null_z > stats.norm.isf(alpha)))
    return float(stats.binom.sf(rejections - 1, null_z.shape[0], alpha)) > alpha


def test_paired_delta_r2_estimate_is_the_difference_of_squared_correlations():
    rng = np.random.default_rng(0)
    outcome = rng.standard_normal(500)
    baseline = outcome + rng.standard_normal(500) * 2.0
    candidate = outcome + rng.standard_normal(500) * 1.5
    comparison = paired_delta_r2(outcome, baseline, candidate)
    expected = np.corrcoef(outcome, candidate)[0, 1] ** 2 - np.corrcoef(outcome, baseline)[0, 1] ** 2
    assert comparison.estimate == pytest.approx(expected, rel=1e-12)
    assert comparison.z_score > 0.0


def test_paired_delta_r2_holds_size_for_equally_accurate_frozen_scores():
    rng = np.random.default_rng(1)
    z_scores = []
    for _ in range(400):
        genetic = rng.standard_normal(2000) * 0.3
        outcome = genetic + rng.standard_normal(2000)
        baseline = genetic + rng.standard_normal(2000) * 0.3
        candidate = genetic + rng.standard_normal(2000) * 0.3
        z_scores.append(paired_delta_r2(outcome, baseline, candidate).z_score)
    z_scores = np.asarray(z_scores)
    assert 0.85 < float(np.std(z_scores)) < 1.15
    assert size_gate(z_scores)


def test_family_clusters_widen_the_standard_error_when_scores_share_family_noise():
    rng = np.random.default_rng(2)
    families = np.repeat(np.arange(500), 2)
    shared = np.repeat(rng.standard_normal(500), 2)
    outcome = shared + rng.standard_normal(1000)
    baseline = rng.standard_normal(1000)
    candidate = shared * 0.5 + rng.standard_normal(1000)
    unclustered = paired_delta_r2(outcome, baseline, candidate)
    clustered = paired_delta_r2(outcome, baseline, candidate, clusters=families)
    assert clustered.standard_error > unclustered.standard_error


def _held_out_disease(rng: np.random.Generator, people: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A liability, the disease it thresholds with noise, and which people are reserved to calibrate (the rest are
    evaluated), as the workspace reserves development families for its frozen calibration."""
    liability = rng.standard_normal(people)
    labels = (liability + rng.standard_normal(people) > 1.0).astype(float)
    return liability, labels, np.arange(people) % 2 == 0


def _frozen(labels: np.ndarray, score: np.ndarray, reserved: np.ndarray) -> np.ndarray:
    """The evaluated people's logits under a calibration fitted on the reserved people alone."""
    return FrozenCalibration.fit(score[reserved], labels[reserved]).logits(score[~reserved])


def test_the_frozen_log_likelihood_gain_favours_the_more_informative_score_and_holds_size():
    rng = np.random.default_rng(4)
    null_z = []
    for _ in range(200):
        liability, labels, reserved = _held_out_disease(rng, 6000)
        baseline = _frozen(labels, liability + rng.standard_normal(6000), reserved)
        candidate = _frozen(labels, liability + rng.standard_normal(6000), reserved)
        null_z.append(paired_frozen_gain(labels[~reserved], baseline, candidate).z_score)
    assert size_gate(np.asarray(null_z))
    liability, labels, reserved = _held_out_disease(rng, 10000)
    better = paired_frozen_gain(
        labels[~reserved], _frozen(labels, liability + rng.standard_normal(10000) * 2.0, reserved), _frozen(labels, liability, reserved)
    )
    assert better.z_score > 3.0


def test_a_negated_candidate_loses_log_likelihood_under_frozen_calibration():
    """Recalibrating each score on the labels it is scored on gave a candidate that negates the baseline exactly the
    baseline's log likelihood. Calibrated on reserved people, with a slope that cannot reverse a score, it keeps none of
    the baseline's information, and a candidate negated among the evaluated people alone loses there as well."""
    rng = np.random.default_rng(12)
    liability, labels, reserved = _held_out_disease(rng, 10000)
    baseline = liability + rng.standard_normal(10000)

    def recalibrated_on_its_own_labels(score: np.ndarray) -> float:
        intercept, slope = logistic_recalibration_coefficients(score, labels)
        return float(np.mean(proper_scores(labels, intercept + slope * score)["log_loss"]))

    assert recalibrated_on_its_own_labels(-baseline) == pytest.approx(recalibrated_on_its_own_labels(baseline), rel=1e-12)
    baseline_logits = _frozen(labels, baseline, reserved)
    negated = paired_frozen_gain(labels[~reserved], baseline_logits, _frozen(labels, -baseline, reserved))
    assert negated.estimate < 0.0 and negated.z_score < -5.0
    calibration = FrozenCalibration.fit(-baseline, labels)
    assert calibration.slope == 0.0 and special.expit(calibration.intercept) == pytest.approx(labels.mean())
    flipped = np.where(reserved, baseline, -baseline)
    in_evaluation = paired_frozen_gain(labels[~reserved], baseline_logits, _frozen(labels, flipped, reserved))
    assert in_evaluation.estimate < 0.0 and in_evaluation.z_score < -5.0


def _within_the_stopping_rule(score, labels, intercept, slope):
    """Half the Newton decrement at (intercept, slope) is within fp64 resolution of the log-likelihood."""
    design = np.column_stack([np.ones_like(score), score])
    linear = intercept + slope * score
    fitted = special.expit(linear)
    gradient = design.T @ (labels - fitted)
    information = design.T @ (design * (fitted * (1.0 - fitted))[:, None])
    log_likelihood = float(np.sum(labels * linear - np.logaddexp(0.0, linear)))
    return 0.5 * float(gradient @ np.linalg.solve(information, gradient)) <= np.finfo(np.float64).eps * abs(log_likelihood)


def test_logistic_recalibration_reaches_the_maximum_and_refuses_separation():
    rng = np.random.default_rng(10)
    score = rng.standard_normal(4000)
    labels = (rng.random(4000) < 1.0 / (1.0 + np.exp(-(0.3 + 1.2 * score)))).astype(float)
    intercept, slope = logistic_recalibration_coefficients(score, labels)
    assert _within_the_stopping_rule(score, labels, intercept, slope)
    with pytest.raises(ValueError, match="separates"):
        logistic_recalibration_coefficients(score, (score > 0.0).astype(float))
    with pytest.raises(ValueError, match="both cases and controls"):
        logistic_recalibration_coefficients(score, np.zeros_like(score))


def test_logistic_recalibration_returns_the_maximum_of_a_score_far_from_zero():
    # A raw score 10^6 from zero: fitted in its own coordinates the linear predictor cancelled, the
    # evaluated log-likelihood stopped rising while the Newton decrement stayed above its resolution,
    # and the line search accepted a zero step forever. Moving the fit to the centred score's
    # coordinates costs the intercept |a| eps of cancellation, far below the stopping rule's resolution.
    rng = np.random.default_rng([2000, 40, 300, 1, 1000000])
    centered = rng.standard_normal(2000)
    labels = (rng.random(2000) < special.expit(-4.0 + 3.0 * centered)).astype(float)
    intercept, slope = logistic_recalibration_coefficients(1e6 + centered, labels)
    assert _within_the_stopping_rule(centered, labels, intercept + slope * 1e6, slope)


def test_logistic_recalibration_terminates_at_the_same_maximum_for_small_offset_samples():
    # bug-recent's stall shape: n = 30, prevalence near 0.2, spread 0.1, far offsets. Each fit must
    # return, and its log-likelihood in the centred score's coordinates must reach the centred fit's
    # to within the rounding of evaluating it (gamma_{n+3} of the summed magnitudes).
    unit_roundoff = np.finfo(np.float64).eps / 2.0
    for seed in range(40):
        rng = np.random.default_rng([30, seed])
        centered = rng.standard_normal(30)
        labels = (rng.random(30) < special.expit(-1.4 + 1.5 * centered)).astype(float)
        positive = labels == 1.0
        if not positive.any() or positive.all() or centered[positive].min() >= centered[~positive].max():
            continue

        def centred_log_likelihood(intercept: float, slope: float) -> tuple[float, float]:
            linear = intercept + slope * centered
            terms = labels * linear - np.logaddexp(0.0, linear)
            return float(np.sum(terms)), float(np.sum(np.abs(terms)))

        reference, magnitude = centred_log_likelihood(*logistic_recalibration_coefficients(centered, labels))
        for offset in (1e3, -1e6):
            intercept, slope = logistic_recalibration_coefficients(offset + 0.1 * centered, labels)
            value, _ = centred_log_likelihood(intercept + slope * offset, 0.1 * slope)
            count = 30 + 3
            assert value >= reference - 2.0 * count * unit_roundoff / (1.0 - count * unit_roundoff) * magnitude


def test_size_gate_is_the_exact_binomial_test_at_alpha():
    null_count = 200
    critical = stats.norm.isf(0.05)
    # 15 of 200 rejections is not significantly above 5% (P = 0.078); 16 is (P = 0.044)
    for rejections, passes in ((15, True), (16, False)):
        null_z = np.where(np.arange(null_count) < rejections, critical + 1.0, 0.0)
        assert size_gate(null_z) is passes


def test_paired_delta_auc_matches_mann_whitney_and_ranks_positives_first():
    rng = np.random.default_rng(5)
    labels = (rng.random(800) < 0.2).astype(int)
    baseline = labels * 0.5 + rng.standard_normal(800)
    candidate = labels * 1.0 + rng.standard_normal(800)
    comparison = paired_delta_auc(labels, baseline, candidate)
    expected = roc_auc_score(labels, candidate) - roc_auc_score(labels, baseline)
    assert comparison.estimate == pytest.approx(expected, rel=1e-10)
    assert comparison.z_score > 0.0


def test_gates_accept_calibrated_nulls_and_reject_inflated_ones():
    rng = np.random.default_rng(9)
    assert size_gate(rng.standard_normal(2000))
    assert not size_gate(rng.standard_normal(2000) + 0.5)

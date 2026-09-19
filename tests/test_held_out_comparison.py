"""Held-out comparison tests and evaluation gates (sv_pgs.held_out_comparison).

The null-calibration checks simulate the two situations the primary SV-PGS
claim meets: frozen scores on a held-out set, and cross-fitted scores. There,
each fold's separately tuned predictor makes pooled scores miscalibrated, and
shared training samples make the naive fold-stratified variance too small.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import special, stats
from sklearn.metrics import roc_auc_score

from sv_pgs.held_out_comparison import (
    CrossFitArm,
    cross_fit_delta_r2,
    delta_r2_influence,
    hommel_adjusted,
    influence_correlation,
    logistic_recalibration_coefficients,
    null_cost_gate,
    paired_delta_auc,
    paired_delta_log_loss,
    paired_delta_r2,
    panel_z,
    permute_within_groups,
    power_weights,
    size_gate,
)


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


def _cross_fitted_ridge(designs, outcome, folds, penalties):
    fold_count = int(folds.max()) + 1
    scores = np.zeros(outcome.shape[0])
    fold_designs, inverses = [], []
    grams = [designs[folds == fold].T @ designs[folds == fold] for fold in range(fold_count)]
    cross = [designs[folds == fold].T @ outcome[folds == fold] for fold in range(fold_count)]
    total_gram, total_cross = sum(grams), sum(cross)
    for fold in range(fold_count):
        inverse = np.linalg.inv(total_gram - grams[fold] + penalties[fold] * np.eye(designs.shape[1]))
        rows = folds == fold
        scores[rows] = designs[rows] @ (inverse @ (total_cross - cross[fold]))
        fold_designs.append(designs[rows])
        inverses.append(inverse)
    return scores, CrossFitArm(fold_designs=fold_designs, training_inverses=inverses)


def test_cross_fit_delta_r2_is_calibrated_where_pooling_is_not():
    rng = np.random.default_rng(3)
    sample_count, feature_count, fold_count = 1000, 30, 5
    folds = np.arange(sample_count) % fold_count
    pooled, stratified = [], []
    for _ in range(300):
        baseline_features = rng.standard_normal((sample_count, feature_count))
        candidate_features = rng.standard_normal((sample_count, feature_count))
        effects = rng.standard_normal(feature_count) * np.sqrt(0.05 / feature_count)
        outcome = baseline_features @ effects + candidate_features @ effects + rng.standard_normal(sample_count)
        penalties = np.full(fold_count, 300.0)
        baseline, baseline_arm = _cross_fitted_ridge(baseline_features, outcome, folds, penalties)
        candidate, candidate_arm = _cross_fitted_ridge(candidate_features, outcome, folds, penalties)
        pooled.append(paired_delta_r2(outcome, baseline, candidate).z_score)
        stratified.append(
            cross_fit_delta_r2(outcome, baseline, candidate, folds, baseline_arm, candidate_arm).z_score
        )
    pooled, stratified = np.asarray(pooled), np.asarray(stratified)
    assert 0.85 < float(np.std(stratified)) < 1.15
    assert size_gate(stratified)
    assert float(np.std(pooled)) > float(np.std(stratified))


def test_paired_delta_log_loss_favours_the_more_informative_score_and_holds_size():
    rng = np.random.default_rng(4)
    null_z = []
    for _ in range(200):
        liability = rng.standard_normal(3000)
        labels = (liability + rng.standard_normal(3000) > 1.0).astype(float)
        baseline = liability + rng.standard_normal(3000)
        candidate = liability + rng.standard_normal(3000)
        null_z.append(paired_delta_log_loss(labels, baseline, candidate).z_score)
    assert size_gate(np.asarray(null_z))
    liability = rng.standard_normal(5000)
    labels = (liability + rng.standard_normal(5000) > 1.0).astype(float)
    better = paired_delta_log_loss(labels, liability + rng.standard_normal(5000) * 2.0, liability)
    assert better.z_score > 3.0


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


def test_permute_within_groups_never_moves_a_sample_across_groups():
    rng = np.random.default_rng(6)
    groups = rng.integers(0, 3, size=300)
    permutation = permute_within_groups(groups, rng)
    assert np.array_equal(groups[permutation], groups)
    assert sorted(permutation.tolist()) == list(range(300))
    assert not np.array_equal(permutation, np.arange(300))


def test_power_weights_are_non_negative_and_refuse_a_powerless_panel():
    weights = power_weights(np.array([2.0, -1.0, 0.5, 0.0]))
    assert np.all(weights >= 0.0)
    assert weights.sum() == pytest.approx(1.0)
    assert weights[1] == 0.0
    with pytest.raises(ValueError):
        power_weights(np.array([-1.0, 0.0]))


def test_panel_z_holds_size_across_correlated_traits():
    rng = np.random.default_rng(7)
    trait_count = 21
    correlation = np.full((trait_count, trait_count), 0.3) + np.eye(trait_count) * 0.7
    weights = power_weights(rng.uniform(0.2, 3.0, size=trait_count))
    draws = rng.multivariate_normal(np.zeros(trait_count), correlation, size=20000)
    panel = np.array([panel_z(draw, weights, correlation) for draw in draws])
    assert float(np.std(panel)) == pytest.approx(1.0, abs=0.03)
    assert abs(float(np.mean(panel > stats.norm.isf(0.05))) - 0.05) < 0.01
    with pytest.raises(ValueError):
        panel_z(draws[0], -weights, correlation)


def test_influence_correlation_matches_the_correlation_of_shared_scores():
    rng = np.random.default_rng(8)
    genetic = rng.standard_normal(4000)
    first_outcome = genetic + rng.standard_normal(4000)
    second_outcome = genetic + rng.standard_normal(4000)
    baseline = genetic + rng.standard_normal(4000) * 2.0
    candidate = genetic + rng.standard_normal(4000)
    first_influence = delta_r2_influence(first_outcome, baseline, candidate)[1]
    second_influence = delta_r2_influence(second_outcome, baseline, candidate)[1]
    correlation = influence_correlation([first_influence, second_influence], clusters=None)
    assert correlation[0, 0] == pytest.approx(1.0)
    assert 0.0 < correlation[0, 1] < 1.0


def test_hommel_adjusts_a_hand_worked_example_in_any_input_order():
    expected = np.array([0.03, 0.04, 0.04])
    assert np.allclose(hommel_adjusted(np.array([0.01, 0.02, 0.04])), expected)
    assert np.allclose(hommel_adjusted(np.array([0.04, 0.01, 0.02])), expected[[2, 0, 1]])


def test_gates_accept_calibrated_nulls_and_reject_inflated_ones():
    rng = np.random.default_rng(9)
    assert size_gate(rng.standard_normal(2000))
    assert not size_gate(rng.standard_normal(2000) + 0.5)


def test_null_cost_gate_is_the_exact_one_sided_sign_test_at_alpha():
    baseline = np.full(10, 0.2)
    # 8 of 10 replicates losing R^2 is not significant (P = 56/1024 = 0.055); 9 is (P = 11/1024 = 0.011)
    for losses, passes in ((8, True), (9, False)):
        with_columns = np.where(np.arange(10) < losses, 0.1999, 0.2001)
        assert null_cost_gate(baseline, with_columns) is passes
    # ties carry no evidence of a loss and leave the test to the untied replicates
    assert null_cost_gate(baseline, baseline)
    assert not null_cost_gate(np.full(12, 0.2), np.concatenate([np.full(9, 0.1999), np.full(3, 0.2)]))

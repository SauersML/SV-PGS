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
    cluster_bootstrap_counts,
    cross_fitted_stack,
    logistic_recalibration_coefficients,
    meta_analysis,
    paired_delta_auc,
    paired_delta_r2,
    weighted_partial_r2,
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
    # A shape that once stalled: n = 30, prevalence near 0.2, spread 0.1, far offsets. Each fit must
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


def _residual_r2(target: NDArray, score: NDArray, covariates: NDArray) -> float:
    """The partial r^2 by residualizing both on the covariates with least squares, row by row."""
    residuals = [values - covariates @ np.linalg.lstsq(covariates, values, rcond=None)[0] for values in (target, score)]
    return float(np.corrcoef(*residuals)[0, 1] ** 2)


def test_a_weighted_partial_r2_is_the_partial_r2_of_the_rows_repeated_by_their_weights():
    rng = np.random.default_rng(3)
    rows = 300
    # The intercept, an age-like column, its square (thousands) and two principal components (hundredths): scales far
    # apart, as the trait covariates are.
    age = rng.uniform(20.0, 80.0, rows)
    covariates = np.column_stack([np.ones(rows), age, age**2, rng.normal(0.0, 0.01, (rows, 2))])
    target = 0.02 * age + rng.normal(size=rows)
    scores = np.vstack([target * 0.3 + covariates[:, 3] * 40.0 + rng.normal(size=rows), rng.normal(size=rows)])
    unit = weighted_partial_r2(np.ones((1, rows)), covariates, target, scores)[0]
    np.testing.assert_allclose(unit, [_residual_r2(target, score, covariates) for score in scores], rtol=1e-9)
    weights = rng.integers(0, 4, size=(3, rows)).astype(np.float64)
    resampled = weighted_partial_r2(weights, covariates, target, scores)
    for weight, values in zip(weights, resampled):
        copies = np.repeat(np.arange(rows), weight.astype(np.int64))
        np.testing.assert_allclose(values, [_residual_r2(target[copies], score[copies], covariates[copies]) for score in scores], rtol=1e-8)


def test_a_weighted_partial_r2_projects_on_the_covariates_a_resample_keeps():
    rng = np.random.default_rng(4)
    rows = 200
    # A duplicated column, and an indicator of five rows that a resample drops whole.
    indicator = np.zeros(rows)
    indicator[:5] = 1.0
    shared = rng.normal(size=rows)
    covariates = np.column_stack([np.ones(rows), shared, shared, indicator])
    target, score = rng.normal(size=rows) + indicator * 3.0, rng.normal(size=rows)
    weights = np.ones((1, rows))
    weights[0, :5] = 0.0
    kept = np.arange(5, rows)
    expected = _residual_r2(target[kept], score[kept], covariates[kept])
    assert weighted_partial_r2(weights, covariates, target, score[None, :])[0, 0] == pytest.approx(expected, rel=1e-9)
    # A resample without rows has no partial r^2.
    assert np.isnan(weighted_partial_r2(np.zeros((1, rows)), covariates, target, score[None, :])[0, 0])


def _short_of_two_residual_dimensions(weights: NDArray, covariates: NDArray) -> NDArray:
    """Per resample, whether its drawn rows exceed the rank of their covariates by fewer than two."""
    return np.array([drawn.sum() - np.linalg.matrix_rank(covariates[drawn]) < 2 for drawn in np.atleast_2d(weights) > 0.0])


def test_a_weighted_partial_r2_is_the_least_squares_one_on_the_repeated_rows_wherever_two_residual_dimensions_remain():
    rng = np.random.default_rng(6)
    rows = 30
    # Nine columns on 30 rows (an age-like column and its square, principal-component-like columns, a 0/1 column of
    # two rows) and resamples from all rows drawn to only a few: some drop the 0/1 column's rows, some saturate.
    age = rng.uniform(20.0, 80.0, rows)
    rare = np.zeros(rows)
    rare[:2] = 1.0
    covariates = np.column_stack([np.ones(rows), age, age**2, rng.normal(0.0, 0.01, (rows, 5)), rare])
    target = 0.02 * age + rare + rng.normal(size=rows)
    scores = np.vstack([target * 0.4 + rng.normal(size=rows), rng.normal(size=rows)])
    weights = rng.integers(0, 4, size=(400, rows)) * (rng.random((400, rows)) < rng.uniform(0.2, 1.0, (400, 1)))
    values = weighted_partial_r2(weights, covariates, target, scores)
    short = _short_of_two_residual_dimensions(weights, covariates)
    assert short.any() and (~short).sum() > 200 and np.any(weights[~short, :2].sum(axis=1) == 0.0)
    assert np.all(np.isnan(values[short]))
    for weight, value in zip(weights[~short], values[~short]):
        copies = np.repeat(np.arange(rows), weight)
        np.testing.assert_allclose(value, [_residual_r2(target[copies], score[copies], covariates[copies]) for score in scores], rtol=0.0, atol=1e-10)


def test_a_resample_with_fewer_than_two_residual_dimensions_or_a_score_the_covariates_explain_has_no_partial_r2():
    rng = np.random.default_rng(8)
    rows = 40
    covariates = np.column_stack([np.ones(rows), rng.normal(size=(rows, 5))])
    target = rng.normal(size=rows)
    # The second score lies in the covariates' span: its residual variance is rounding.
    scores = np.vstack([rng.normal(size=rows), covariates @ rng.normal(size=6)])
    weights = np.zeros((4, rows))
    # Rows drawn: the covariates' rank (no residual), one more (a residual every score's is a multiple of), two more, all.
    for resample, drawn in enumerate((6, 7, 8, rows)):
        weights[resample, rng.choice(rows, drawn, replace=False)] = rng.integers(1, 4, drawn)
    values = weighted_partial_r2(weights, covariates, target, scores)
    assert np.all(np.isnan(values[:2])) and np.all(np.isfinite(values[2:, 0])) and np.all(np.isnan(values[:, 1]))


def test_a_small_cell_s_resampled_partial_r2_stays_in_the_unit_interval():
    # 43 people in 40 families with 21 covariates (intercept, age, age^2, age x female, female and 16 principal
    # components of one ancestry group), as the report's smallest shown ancestry group: a resample draws about 27 of
    # them, and 1.5-2% draw 22 or fewer, too few for the covariates.
    rng = np.random.default_rng(4)
    rows = 43
    age, female = rng.uniform(25.0, 80.0, rows), rng.integers(0, 2, rows).astype(np.float64)
    pcs = rng.normal(0.02, 0.003, (rows, 16)) * rng.uniform(0.2, 2.0, 16)
    covariates = np.column_stack([np.ones(rows), age, age**2, age * female, female, pcs])
    small, structural = rng.normal(size=rows), rng.normal(size=rows)
    target = 130.0 + 0.3 * age + 5.0 * female + 10.0 * (small + 0.5 * structural + 2.0 * rng.normal(size=rows))
    snv_indel = small + rng.normal(size=rows)
    scores = np.vstack([snv_indel, snv_indel + 0.5 * structural])
    families = np.r_[np.arange(6) // 2, 3 + np.arange(rows - 6)]
    weights = cluster_bootstrap_counts(np.zeros(families.max() + 1, dtype=np.int64), 2000, np.random.default_rng(104))[:, families]
    values = weighted_partial_r2(weights, covariates, target, scores)
    finite = np.isfinite(values)
    assert np.all((values[finite] >= 0.0) & (values[finite] <= 1.0 + rows * np.finfo(np.float64).eps))
    # A squared correlation's resampled standard deviation is at most 1/2.
    assert all(np.std(values[finite[:, index], index], ddof=1) <= 0.5 for index in range(scores.shape[0]))
    short = _short_of_two_residual_dimensions(weights, covariates)
    assert 0 < short.sum() < 100 and np.all(np.isnan(values[short])) and np.all(finite[~short])


def test_cluster_bootstrap_counts_draw_each_stratum_s_clusters_as_many_times_as_it_holds():
    strata = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 2])
    counts = cluster_bootstrap_counts(strata, 500, np.random.default_rng(5))
    assert counts.shape == (500, strata.shape[0]) and np.all(counts >= 0) and np.all(counts == np.round(counts))
    for stratum in range(3):
        np.testing.assert_array_equal(counts[:, strata == stratum].sum(axis=1), np.sum(strata == stratum))
    # Each cluster is drawn once per resample on average, and the same seed draws the same resamples.
    np.testing.assert_allclose(counts.mean(axis=0), 1.0, atol=0.15)
    np.testing.assert_array_equal(counts, cluster_bootstrap_counts(strata, 500, np.random.default_rng(5)))


def test_meta_analysis_is_inverse_variance_and_dersimonian_laird_on_independent_estimates():
    estimates = np.array([0.010, 0.002, 0.006, -0.001])
    variances = np.array([4e-6, 1e-6, 9e-6, 2e-6])
    result = meta_analysis(estimates, np.diag(variances))
    weights = 1.0 / variances
    fixed = float(weights @ estimates / weights.sum())
    assert result.fixed.estimate == pytest.approx(fixed, rel=1e-12)
    assert result.fixed.standard_error == pytest.approx(weights.sum() ** -0.5, rel=1e-12)
    q = float(weights @ (estimates - fixed) ** 2)
    tau = max(0.0, (q - 3) / (weights.sum() - (weights**2).sum() / weights.sum()))
    assert result.heterogeneity_q == pytest.approx(q, rel=1e-10) and result.tau_squared == pytest.approx(tau, rel=1e-10)
    assert result.heterogeneity_p == pytest.approx(stats.chi2.sf(q, 3), rel=1e-10) and result.i_squared == pytest.approx((q - 3) / q, rel=1e-10)
    random_weights = 1.0 / (variances + tau)
    assert result.random.estimate == pytest.approx(float(random_weights @ estimates / random_weights.sum()), rel=1e-10)
    assert result.random.standard_error == pytest.approx(random_weights.sum() ** -0.5, rel=1e-10)
    assert result.positive == 3


def test_correlated_estimates_count_less_than_independent_ones_in_the_meta_analysis():
    estimates = np.array([0.004, 0.006])
    independent = meta_analysis(estimates, np.diag([1e-6, 1e-6]))
    correlated = meta_analysis(estimates, np.array([[1e-6, 0.8e-6], [0.8e-6, 1e-6]]))
    assert correlated.fixed.estimate == pytest.approx(independent.fixed.estimate)
    # Var of the mean of two estimates with correlation rho is (1 + rho) / 2 of one's.
    assert correlated.fixed.standard_error == pytest.approx(np.sqrt(1.8e-6 / 2.0), rel=1e-10)
    assert independent.fixed.standard_error == pytest.approx(np.sqrt(1e-6 / 2.0), rel=1e-10)
    # Two estimates that move together exactly are one.
    identical = meta_analysis(np.array([0.004, 0.004]), np.full((2, 2), 1e-6))
    assert identical.fixed.standard_error == pytest.approx(1e-3, rel=1e-9)


def test_correlated_heterogeneity_is_the_moment_equation_under_the_traits_covariance():
    estimates = np.array([0.012, 0.001, 0.007])
    covariance = np.array([[4e-6, 1e-6, 0.5e-6], [1e-6, 2e-6, 0.8e-6], [0.5e-6, 0.8e-6, 3e-6]])
    result = meta_analysis(estimates, covariance)
    precision = np.linalg.inv(covariance)
    ones = np.ones(3)
    information = ones @ precision @ ones
    residual = precision - np.outer(precision @ ones, precision @ ones) / information
    q = estimates @ residual @ estimates
    tau = max(0.0, (q - 2) / np.trace(residual))
    assert tau > 0.0 and result.tau_squared == pytest.approx(tau, rel=1e-10) and result.heterogeneity_q == pytest.approx(q, rel=1e-10)
    widened = np.linalg.inv(covariance + tau * np.eye(3))
    assert result.random.estimate == pytest.approx(ones @ widened @ estimates / (ones @ widened @ ones), rel=1e-10)
    assert result.random.standard_error == pytest.approx((ones @ widened @ ones) ** -0.5, rel=1e-10)
    assert result.random.standard_error > result.fixed.standard_error


def test_a_meta_analysis_without_information_about_the_mean_is_undefined():
    # Two estimates that move exactly against each other say nothing about their mean, and a non-finite input nothing at all.
    opposed = meta_analysis(np.array([0.01, 0.02]), np.array([[1e-6, -1e-6], [-1e-6, 1e-6]]))
    missing = meta_analysis(np.array([0.01, np.nan]), np.diag([1e-6, 1e-6]))
    for result in (opposed, missing):
        assert np.isnan(result.fixed.estimate) and np.isnan(result.random.estimate) and np.isnan(result.heterogeneity_q)
    assert opposed.positive == 2 and missing.positive == 1


def test_cross_fitted_stacking_weights_never_learn_from_the_family_they_score():
    generator = np.random.default_rng(11)
    rows = 400
    families = np.repeat(np.arange(rows // 2), 2)
    covariates = np.column_stack([np.ones(rows), generator.normal(size=rows)])
    components = generator.normal(size=(2, rows))
    target = 0.3 * components[0] + 0.1 * components[1] + 0.2 * covariates[:, 1] + generator.normal(size=rows)
    stacked = cross_fitted_stack(target, covariates, components, families, 5, np.random.default_rng(3))
    # The same seed deals the same parts: each part's score is the components weighted by the fit on the other parts.
    order = np.random.default_rng(3).permutation(np.unique(families))
    part = {family: index % 5 for index, family in enumerate(order.tolist())}
    parts = np.array([part[family] for family in families.tolist()])
    for split in range(5):
        fitted = parts != split
        design = np.column_stack([covariates[fitted], components[:, fitted].T])
        weights = np.linalg.lstsq(design, target[fitted], rcond=None)[0][2:]
        np.testing.assert_allclose(stacked[parts == split], weights @ components[:, parts == split], rtol=1e-10)
    # A family's own outcome never moves its own score.
    changed = target.copy()
    changed[families == 7] += 50.0
    restacked = cross_fitted_stack(changed, covariates, components, families, 5, np.random.default_rng(3))
    np.testing.assert_allclose(restacked[families == 7], stacked[families == 7], rtol=1e-10)
    assert not np.allclose(restacked[parts != parts[14]], stacked[parts != parts[14]])
    # A row with no outcome is neither fitted nor scored.
    changed[0] = np.nan
    assert np.isnan(cross_fitted_stack(changed, covariates, components, families, 5, np.random.default_rng(3))[0])

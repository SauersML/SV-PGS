"""The measurement model: recalibration scales, their pooling, reliability offsets and the leakage map."""
from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.measurement_model import (
    LdBlock,
    apply_leakage_map,
    calibration_moments,
    calibration_pairs,
    concatenate_calibration_moments,
    fit_leakage_map,
    fit_measurement_model,
    leakage_transform,
    log_reliability_offsets,
    mapped_gram,
    merge_calibration_moments,
    pool_normal,
    pooled_calibration,
    pooled_log_reliability,
    residual_variances,
)
from sv_pgs.sample_ids import ResearchId
from tests.phenotype_bounds import rounding_gamma, sampling_bound


def _draw_type_column(genotype: np.ndarray, frequency: np.ndarray, keep: float, rng: np.random.Generator) -> np.ndarray:
    """D = G with probability ``keep``, else an independent draw from the record's genotype law.

    Cov(G, D) = keep Var(G) and Var(D) = Var(G), so kappa = keep and r^2 = keep^2: a confident draw.
    """
    redraw = rng.binomial(2, frequency[:, None], size=genotype.shape).astype(float)
    return np.where(rng.random(genotype.shape) < keep, genotype, redraw)


def _records(count: int, pairs: int, keep: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frequency = rng.uniform(0.05, 0.5, size=count)
    genotype = rng.binomial(2, frequency[:, None], size=(count, pairs)).astype(float)
    return frequency, genotype, _draw_type_column(genotype, frequency, keep, rng)


def test_the_pooled_scale_of_a_draw_type_column_is_its_keep_probability() -> None:
    rng = np.random.default_rng(20260919)
    records, pairs, keep = 300, 3000, 0.8
    _, genotype, dosage = _records(records, pairs, keep, rng)
    pooled = pooled_calibration(calibration_moments(dosage, genotype), dosage.var(axis=1), np.zeros(records, dtype=int))
    # Each record's slope has sampling variance Var(G - keep D) / S_DD = (1 - keep^2) / pairs.
    standard_error = np.sqrt((1 - keep**2) / (pairs * records))
    assert abs(float(np.mean(pooled.scales)) - keep) <= sampling_bound(float(standard_error))


def test_a_rare_record_whose_few_pairs_fit_exactly_is_pooled_to_its_stratum() -> None:
    rng = np.random.default_rng(37)
    _, genotype, dosage = _records(200, 200, 0.7, rng)
    rare_dosage = np.zeros((1, 200))
    rare_truth = np.zeros((1, 200))
    rare_dosage[0, 0], rare_truth[0, 0] = 1 / 127, 1.0
    moments = calibration_moments(np.vstack([dosage, rare_dosage]), np.vstack([genotype, rare_truth]))
    # The rare record's own least-squares slope is 127 with no residual.
    cohort_variance = np.append(dosage.var(axis=1), 2 * 0.005 * 0.995)
    pooled = pooled_calibration(moments, cohort_variance, np.zeros(201, dtype=int))
    assert pooled.scales[-1] < 1.0


def test_estimates_that_agree_within_their_error_are_pooled_with_no_between_variance() -> None:
    variances = np.linspace(0.01, 0.04, 50)
    pooled = pool_normal(np.full(50, 0.7), variances)
    assert pooled.between_variance == 0.0
    np.testing.assert_array_equal(pooled.shrunk, np.full(50, 0.7))


def test_the_between_variance_is_recovered_by_marginal_likelihood() -> None:
    rng = np.random.default_rng(7)
    count, between = 4000, 0.02
    variances = rng.uniform(0.005, 0.03, size=count)
    estimates = 0.6 + rng.normal(0.0, np.sqrt(between), count) + rng.normal(0.0, np.sqrt(variances))
    pooled = pool_normal(estimates, variances)
    information = np.sum(1.0 / (pooled.between_variance + variances) ** 2) / 2
    assert abs(pooled.between_variance - between) <= sampling_bound(float(np.sqrt(1.0 / information)))


def test_an_exact_estimate_is_kept_and_an_uninformative_one_gets_the_prior_mean() -> None:
    estimates = np.array([0.9, 0.5, 0.6, 0.55, 0.3])
    variances = np.array([0.0, 0.01, 0.01, 0.01, np.inf])
    pooled = pool_normal(estimates, variances)
    assert pooled.shrunk[0] == 0.9
    assert pooled.shrunk[4] == pytest.approx(pooled.coefficients[0])


def test_reliability_of_a_calibrated_draw_is_the_square_of_its_scale() -> None:
    rng = np.random.default_rng(11)
    records, keep = 200, 0.6
    _, genotype, dosage = _records(records, 4000, keep, rng)
    moments = calibration_moments(dosage, genotype)
    variance = dosage.var(axis=1)
    pooled = pooled_calibration(moments, variance, np.zeros(records, dtype=int))
    residual = residual_variances(variance, pooled.scales, pooled.variance_ratios)
    reliability = np.exp(log_reliability_offsets(variance, pooled.scales, residual))
    # About 14 rounded operations separate the two sides (a log or exp counts two).
    np.testing.assert_allclose(reliability, pooled.scales**2 / pooled.variance_ratios, rtol=rounding_gamma(16))
    # A draw keeps the genotype's variance, so lambda = 1 and r^2 = kappa^2.
    excess = moments.pair_counts * (moments.truth_variance - moments.dosage_variance)
    energy = np.sum(moments.pair_counts * moments.dosage_variance)
    assert abs(pooled.variance_ratios[0] - 1.0) <= sampling_bound(float(np.std(excess) * np.sqrt(records) / energy))


def test_the_residual_variance_is_the_genotype_variance_the_calibrated_column_misses() -> None:
    variance, scales, ratios = np.array([0.4, 0.3]), np.array([0.5, 1.2]), np.array([1.0, 1.0])
    residual = residual_variances(variance, scales, ratios)
    assert residual[0] == 0.4 * (1.0 - 0.5**2)
    assert residual[1] == 0.0


def test_the_cohort_reliability_pools_the_groups_models_exactly() -> None:
    rng = np.random.default_rng(41)
    sizes, scales, residuals = np.array([700, 300]), np.array([0.8, 0.6]), np.array([0.05, 0.2])
    columns = [mean + rng.normal(0.0, 0.5, size) for mean, size in zip((0.4, 1.1), sizes)]
    # The recalibration keeps each group's mean: D*_g = mu_g + kappa_g (D - mu_g).
    calibrated = np.concatenate([column.mean() + scale * (column - column.mean()) for scale, column in zip(scales, columns)])
    variances = np.array([column.var() for column in columns])
    group_means = np.array([column.mean() for column in columns])
    pooled = pooled_log_reliability(scales[None], residuals[None], group_means[None], variances[None], sizes)
    direct_residual = np.sum(sizes * residuals) / sizes.sum()
    direct = np.log(np.var(calibrated)) - np.log(np.var(calibrated) + direct_residual)
    assert abs(pooled[0] - direct) <= 2 * rounding_gamma(4 * int(sizes.sum()))
    unfitted = pooled_log_reliability(scales[None], np.array([[0.05, np.inf]]), group_means[None], variances[None], np.array([700, 0]))
    one_group = np.log(scales[0] ** 2 * variances[0]) - np.log(scales[0] ** 2 * variances[0] + residuals[0])
    assert abs(unfitted[0] - one_group) <= 2 * rounding_gamma(16)
    silent = pooled_log_reliability(np.zeros((1, 2)), residuals[None], np.zeros((1, 2)), variances[None], sizes)
    assert silent[0] == -np.inf


def test_moments_merged_across_chunks_are_the_moments_of_all_pairs() -> None:
    rng = np.random.default_rng(23)
    pairs = 900
    _, genotype, dosage = _records(40, pairs, 0.6, rng)
    dosage[rng.random(dosage.shape) < 0.1] = np.nan
    dosage[3] = np.nan
    whole = calibration_moments(dosage, genotype)
    merged = calibration_moments(dosage[:, :0], genotype[:, :0])
    for chunk in np.array_split(np.arange(pairs), 7):
        merged = merge_calibration_moments(merged, calibration_moments(dosage[:, chunk], genotype[:, chunk]))
    np.testing.assert_array_equal(merged.pair_counts, whole.pair_counts)
    assert merged.pair_counts[3] == 0
    joined = concatenate_calibration_moments([calibration_moments(dosage[:17], genotype[:17]), calibration_moments(dosage[17:], genotype[17:])])
    np.testing.assert_array_equal(joined.covariance, whole.covariance)
    bound = 2 * rounding_gamma(4 * pairs)
    for name, scale in (("dosage_mean", 2.0), ("truth_mean", 2.0), ("dosage_variance", 4.0), ("truth_variance", 4.0), ("covariance", 4.0)):
        assert np.all(np.abs(getattr(merged, name) - getattr(whole, name)) <= bound * scale), name


def _two_locus(rng: np.random.Generator, samples: int, frequency: float, linkage: float) -> tuple[np.ndarray, np.ndarray]:
    """Two-haplotype genotypes of an SV and a tag SNP whose alleles agree with probability ``linkage``."""
    sv_haplotypes = rng.random((samples, 2)) < frequency
    independent = rng.random((samples, 2)) < frequency
    snp_haplotypes = np.where(rng.random((samples, 2)) < linkage, sv_haplotypes, independent)
    return sv_haplotypes.sum(axis=1).astype(float), snp_haplotypes.sum(axis=1).astype(float)


def _regression(outcome: np.ndarray, columns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    design = np.column_stack([np.ones(outcome.size), columns])
    coefficients, *_ = np.linalg.lstsq(design, outcome, rcond=None)
    residual = outcome - design @ coefficients
    noise = residual @ residual / (outcome.size - design.shape[1])
    covariance = noise * np.linalg.inv(design.T @ design)
    return coefficients[1:], np.sqrt(np.diag(covariance))[1:]


def _shrinkage_gap(covariance: np.ndarray, ratio: float) -> float:
    """max_i 1 / (1 + t lambda_i): the largest relative shrinkage of the ridge posterior mean."""
    deviations = np.sqrt(np.diag(covariance))
    eigenvalues = np.linalg.eigvalsh(covariance / np.outer(deviations, deviations))
    return float(np.max(1.0 / (1.0 + ratio * eigenvalues)))


def test_the_leakage_map_moves_an_sv_effect_back_off_its_tag_snp() -> None:
    rng = np.random.default_rng(20260919)
    frequency, effect, keep = 0.3, 1.0, 0.5
    calibration_sv, calibration_snp = _two_locus(rng, 6000, frequency, 0.9)
    calibration_draw = _draw_type_column(calibration_sv[None], np.array([frequency]), keep, rng)[0]
    moments = calibration_moments(calibration_draw[None], calibration_sv[None])
    scale = pooled_calibration(moments, moments.dosage_variance, np.zeros(1, dtype=int)).scales[0]
    calibrated = moments.dosage_mean[0] + scale * (calibration_draw - moments.dosage_mean[0])

    cohort_sv, cohort_snp = _two_locus(rng, 40000, frequency, 0.9)
    cohort_draw = _draw_type_column(cohort_sv[None], np.array([frequency]), keep, rng)[0]
    cohort_calibrated = moments.dosage_mean[0] + scale * (cohort_draw - moments.dosage_mean[0])
    cohort_block = np.column_stack([cohort_calibrated, cohort_snp])
    covariance = np.cov(cohort_block.T, bias=True)
    leakage = fit_leakage_map(covariance, np.column_stack([calibrated, calibration_snp]), calibration_sv[:, None], np.array([0]))
    assert leakage.ridge_ratio > 0.0

    outcome = effect * cohort_sv + rng.normal(0.0, 1.0, cohort_sv.size)
    plain, plain_error = _regression(outcome, cohort_block)
    mapped, mapped_error = _regression(outcome, apply_leakage_map(cohort_block, leakage))
    # What leak remains is the effect times the map's own error on the SNP column:
    # its sampling error (that of the pairs' regression) and its ridge shrinkage.
    _, map_error = _regression(calibration_sv - calibrated, np.column_stack([calibrated, calibration_snp]))
    gap = _shrinkage_gap(covariance, leakage.ridge_ratio)
    shrinkage = gap / (1 - gap) * abs(leakage.coefficients[1, 0])
    assert plain[1] > sampling_bound(float(plain_error[1]))
    assert abs(mapped[1]) <= sampling_bound(float(np.hypot(mapped_error[1], effect * map_error[1]))) + effect * shrinkage
    assert abs(mapped[0] - effect) < abs(plain[0] - effect)


def test_a_joint_posterior_mean_column_gets_no_leakage_correction() -> None:
    rng = np.random.default_rng(5)
    frequency, linkage = 0.3, 0.8
    sv, snp = _two_locus(rng, 8000, frequency, linkage)
    agree = linkage + (1.0 - linkage) * frequency
    carrier_given_snp = agree
    carrier_given_no_snp = (1.0 - linkage) * frequency
    posterior_mean = snp * carrier_given_snp + (2.0 - snp) * carrier_given_no_snp
    block = np.column_stack([posterior_mean, snp])
    leakage = fit_leakage_map(np.cov(block.T, bias=True), block, sv[:, None], np.array([0]))
    residual = sv - posterior_mean
    bound = sampling_bound(float(np.std(residual) / (np.std(snp) * np.sqrt(sv.size))))
    assert np.all(np.abs(leakage.coefficients) <= bound)


def test_the_mapped_gram_is_the_gram_of_the_mapped_columns() -> None:
    rng = np.random.default_rng(9)
    block = rng.normal(size=(500, 6))
    truth = block[:, [1, 4]] + 0.3 * block[:, [0, 5]] + rng.normal(0.0, 0.5, (500, 2))
    leakage = fit_leakage_map(np.cov(block.T, bias=True), block, truth, np.array([1, 4]))
    centred = block - leakage.column_means
    mapped_centred = apply_leakage_map(block, leakage) - leakage.column_means
    transform = leakage_transform(leakage)
    product_bound = 2 * rounding_gamma(2 * block.shape[1]) * (np.abs(centred) @ np.abs(transform))
    assert np.all(np.abs(centred @ transform - mapped_centred) <= product_bound)
    gram_bound = 2 * rounding_gamma(4 * block.shape[0]) * (np.abs(transform.T) @ np.abs(centred.T) @ np.abs(centred) @ np.abs(transform))
    assert np.all(np.abs(mapped_gram(centred.T @ centred, leakage) - mapped_centred.T @ mapped_centred) <= gram_bound)


def test_the_map_recovers_a_known_leak_from_the_cohort_covariance() -> None:
    rng = np.random.default_rng(31)
    pairs, noise = 4000, 0.3
    leak = np.array([0.0, 0.4, -0.3, 0.2, 0.1])
    covariance = np.cov(rng.normal(size=(200000, 5)).T, bias=True)
    block = rng.normal(size=(pairs, 5))
    truth = block[:, [0]] + ((block - block.mean(axis=0)) @ leak)[:, None] + rng.normal(0.0, noise, (pairs, 1))
    leakage = fit_leakage_map(covariance, block, truth, np.array([0]))
    # Independent unit columns: s_j - c_j has variance (|c|^2 + c_j^2 + noise^2) / n from
    # the pairs' own correlation and noise, and the ridge shrinks by at most the gap.
    standard_errors = np.sqrt((leak @ leak + leak**2 + noise**2) / pairs)
    gap = _shrinkage_gap(covariance, leakage.ridge_ratio)
    bound = sampling_bound(1.0) * standard_errors + gap / (1 - gap) * np.abs(leakage.coefficients[:, 0])
    assert np.all(np.abs(leakage.coefficients[:, 0] - leak) <= bound)


def test_a_block_wider_than_its_pairs_still_gets_a_finite_map() -> None:
    rng = np.random.default_rng(13)
    covariance = np.cov(rng.normal(size=(2000, 40)).T, bias=True)
    block = rng.normal(size=(20, 40))
    leakage = fit_leakage_map(covariance, block, block[:, [3]] + rng.normal(size=(20, 1)), np.array([3]))
    assert leakage.ridge_ratio >= 0.0
    assert np.all(np.isfinite(leakage.coefficients))


def test_without_truth_the_model_degrades_loudly_and_records_it() -> None:
    variance = np.array([0.4, 0.3])
    with pytest.raises(ValueError, match="reported r"):
        fit_measurement_model(None, variance, np.zeros(2, dtype=int))
    reported = np.array([0.9, 0.5])
    model = fit_measurement_model(None, variance, np.zeros(2, dtype=int), reported_reliability=reported)
    np.testing.assert_array_equal(model.scales, np.ones(2))
    assert model.leakage_maps == ()
    assert model.certificate["calibrated_records"] == 0 and model.certificate["uncalibrated_records"] == 2
    assert "not applied to 2 records" in str(model.certificate["recalibration"])
    assert "not applied" in str(model.certificate["leakage_correction"])
    np.testing.assert_allclose(model.log_reliability, np.log(reported), rtol=rounding_gamma(2))
    np.testing.assert_allclose(model.residual_variance, variance * (1 - reported) / reported, rtol=rounding_gamma(3))


def test_calibration_pairs_need_typed_research_ids() -> None:
    with pytest.raises(TypeError):
        calibration_pairs(("1", "2"), np.zeros((1, 2)), np.zeros((1, 2)))
    with pytest.raises(ValueError):
        calibration_pairs((ResearchId("1"), ResearchId("1")), np.zeros((1, 2)), np.zeros((1, 2)))


def test_with_truth_the_model_recalibrates_and_maps_each_block() -> None:
    rng = np.random.default_rng(17)
    samples = 3000
    sv, snp = _two_locus(rng, samples, 0.3, 0.9)
    draw = _draw_type_column(sv[None], np.array([0.3]), 0.5, rng)[0]
    pairs = calibration_pairs(
        tuple(ResearchId(str(index)) for index in range(samples)),
        np.vstack([draw, snp]),
        np.vstack([sv, snp]),
        blocks=(LdBlock(np.array([0, 1]), np.array([0])),),
        block_covariances=(np.cov(np.vstack([draw, snp]), bias=True),),
    )
    model = fit_measurement_model(pairs, np.array([np.var(draw), np.var(snp)]), np.array([0, 1]))
    assert model.certificate["calibrated_records"] == 2 and model.certificate["uncalibrated_records"] == 0
    assert "applied to 1 LD blocks" in str(model.certificate["leakage_correction"])
    assert len(model.leakage_maps) == 1 and model.leakage_maps[0].ridge_ratio > 0.0
    # The SNP's truth is its own column: its slope is 1, kept exactly or pooled alone.
    assert abs(model.scales[1] - 1.0) <= rounding_gamma(2)


def test_records_without_pairs_fall_back_to_the_reported_reliability_and_are_counted() -> None:
    rng = np.random.default_rng(29)
    _, genotype, dosage = _records(6, 2000, 0.7, rng)
    dosage[4:] = np.nan
    pairs = calibration_pairs(tuple(ResearchId(str(index)) for index in range(2000)), dosage, genotype)
    variance = np.full(6, 0.3)
    variance[:4] = dosage[:4].var(axis=1)
    with pytest.raises(ValueError, match="2 records"):
        fit_measurement_model(pairs, variance, np.zeros(6, dtype=int))
    reported = np.full(6, 0.8)
    model = fit_measurement_model(pairs, variance, np.zeros(6, dtype=int), reported_reliability=reported)
    assert model.certificate["calibrated_records"] == 4 and model.certificate["uncalibrated_records"] == 2
    np.testing.assert_array_equal(model.scales[4:], np.ones(2))
    np.testing.assert_allclose(model.log_reliability[4:], np.log(reported[4:]), rtol=rounding_gamma(2))
    assert np.all(model.scales[:4] < 1.0)

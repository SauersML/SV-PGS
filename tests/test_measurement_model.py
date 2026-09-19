"""The measurement model: recalibration scales, their pooling, reliability offsets and the leakage map."""
from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.measurement_model import (
    LdBlock,
    apply_leakage_map,
    calibration_moments,
    calibration_pairs,
    fit_leakage_map,
    fit_measurement_model,
    leakage_transform,
    log_reliability_offsets,
    mapped_gram,
    merge_calibration_moments,
    pool_normal,
    record_scale_estimates,
    recalibration_scales,
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
    _, genotype, dosage = _records(300, 3000, 0.8, rng)
    slopes, variances = record_scale_estimates(calibration_moments(dosage, genotype))
    pooled = pool_normal(slopes, variances)
    weights = 1.0 / (pooled.between_variance + variances)
    assert abs(pooled.coefficients[0] - 0.8) <= sampling_bound(float(np.sqrt(1.0 / weights.sum())))


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
    keep = 0.6
    _, genotype, dosage = _records(200, 4000, keep, rng)
    moments = calibration_moments(dosage, genotype)
    scales = recalibration_scales(moments, np.zeros(200, dtype=int))
    residual = residual_variances(moments, scales)
    reliability = np.exp(log_reliability_offsets(moments.dosage_variance, scales, residual))
    # With Var(D) = Var(G) and Cov(G, D) = keep Var(G), a column scaled by kappa has
    # r^2 = kappa^2 / (kappa^2 + 1 + kappa^2 - 2 kappa keep), which is keep^2 at kappa = keep.
    kappa = float(np.mean(scales))
    expected = kappa**2 / (kappa**2 + 1.0 + kappa**2 - 2.0 * kappa * keep)
    assert abs(float(np.mean(reliability)) - expected) <= sampling_bound(float(np.std(reliability) / np.sqrt(reliability.size)))


def test_the_residual_variance_is_the_mean_square_of_the_calibrated_residual() -> None:
    rng = np.random.default_rng(3)
    pairs = 200
    _, genotype, dosage = _records(5, pairs, 0.7, rng)
    scales = np.array([0.7, 0.6, 0.8, 0.5, 0.9])
    residual = residual_variances(calibration_moments(dosage, genotype), scales)
    truth_centred = genotype - genotype.mean(axis=1, keepdims=True)
    dosage_centred = dosage - dosage.mean(axis=1, keepdims=True)
    direct = np.mean((truth_centred - scales[:, None] * dosage_centred) ** 2, axis=1)
    # The moment form sums three terms whose magnitudes bound its cancellation error.
    magnitude = np.mean(truth_centred**2 + 2 * np.abs(scales[:, None] * dosage_centred * truth_centred) + (scales[:, None] * dosage_centred) ** 2, axis=1)
    assert np.all(np.abs(residual - direct) <= 2 * rounding_gamma(4 * pairs) * magnitude)


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


def test_the_leakage_map_moves_an_sv_effect_back_off_its_tag_snp() -> None:
    rng = np.random.default_rng(20260919)
    frequency, effect, keep = 0.3, 1.0, 0.5
    calibration_sv, calibration_snp = _two_locus(rng, 6000, frequency, 0.9)
    calibration_draw = _draw_type_column(calibration_sv[None], np.array([frequency]), keep, rng)[0]
    moments = calibration_moments(calibration_draw[None], calibration_sv[None])
    scale = recalibration_scales(moments, np.zeros(1, dtype=int))[0]
    calibrated = moments.dosage_mean[0] + scale * (calibration_draw - moments.dosage_mean[0])
    leakage = fit_leakage_map(np.column_stack([calibrated, calibration_snp]), calibration_sv[:, None], np.array([0]))
    assert leakage.ridge_ratio > 0.0

    cohort_sv, cohort_snp = _two_locus(rng, 40000, frequency, 0.9)
    cohort_draw = _draw_type_column(cohort_sv[None], np.array([frequency]), keep, rng)[0]
    cohort_calibrated = moments.dosage_mean[0] + scale * (cohort_draw - moments.dosage_mean[0])
    outcome = effect * cohort_sv + rng.normal(0.0, 1.0, cohort_sv.size)
    plain, plain_error = _regression(outcome, np.column_stack([cohort_calibrated, cohort_snp]))
    mapped_block = apply_leakage_map(np.column_stack([cohort_calibrated, cohort_snp]), leakage)
    mapped, mapped_error = _regression(outcome, mapped_block)
    # The map is estimated from the calibration pairs, so what leak remains is the
    # effect times the map's own estimation error on the SNP column.
    _, map_error = _regression(calibration_sv - calibrated, np.column_stack([calibrated, calibration_snp]))
    assert plain[1] > sampling_bound(float(plain_error[1]))
    assert abs(mapped[1]) <= sampling_bound(float(np.hypot(mapped_error[1], effect * map_error[1])))
    assert abs(mapped[0] - effect) < abs(plain[0] - effect)


def test_a_joint_posterior_mean_column_gets_no_leakage_correction() -> None:
    rng = np.random.default_rng(5)
    frequency, linkage = 0.3, 0.8
    sv, snp = _two_locus(rng, 8000, frequency, linkage)
    agree = linkage + (1.0 - linkage) * frequency
    carrier_given_snp = agree
    carrier_given_no_snp = (1.0 - linkage) * frequency
    posterior_mean = snp * carrier_given_snp + (2.0 - snp) * carrier_given_no_snp
    leakage = fit_leakage_map(np.column_stack([posterior_mean, snp]), sv[:, None], np.array([0]))
    residual = sv - posterior_mean
    bound = sampling_bound(float(np.std(residual) / (np.std(snp) * np.sqrt(sv.size))))
    assert np.all(np.abs(leakage.coefficients) <= bound)


def test_the_mapped_gram_is_the_gram_of_the_mapped_columns() -> None:
    rng = np.random.default_rng(9)
    block = rng.normal(size=(500, 6))
    truth = block[:, [1, 4]] + 0.3 * block[:, [0, 5]] + rng.normal(0.0, 0.5, (500, 2))
    leakage = fit_leakage_map(block, truth, np.array([1, 4]))
    centred = block - leakage.column_means
    mapped_centred = apply_leakage_map(block, leakage) - leakage.column_means
    transform = leakage_transform(leakage)
    product_bound = 2 * rounding_gamma(2 * block.shape[1]) * (np.abs(centred) @ np.abs(transform))
    assert np.all(np.abs(centred @ transform - mapped_centred) <= product_bound)
    gram_bound = 2 * rounding_gamma(4 * block.shape[0]) * (np.abs(transform.T) @ np.abs(centred.T) @ np.abs(centred) @ np.abs(transform))
    assert np.all(np.abs(mapped_gram(centred.T @ centred, leakage) - mapped_centred.T @ mapped_centred) <= gram_bound)


def test_noise_free_leakage_is_recovered_exactly() -> None:
    rng = np.random.default_rng(31)
    block = rng.normal(size=(400, 5))
    leak = np.array([0.0, 0.4, -0.3, 0.2, 0.1])
    centred = block - block.mean(axis=0)
    truth = block[:, [0]] + (centred @ leak)[:, None]
    leakage = fit_leakage_map(block, truth, np.array([0]))
    # The residual lies in the columns' span, so the fit is the least-squares solution
    # up to rounding, amplified at most by the block's condition number.
    bound = rounding_gamma(block.size) * np.linalg.cond(centred / centred.std(axis=0)) * np.abs(leak).max()
    assert np.all(np.abs(leakage.coefficients[:, 0] - leak) <= bound)


def test_a_block_wider_than_its_pairs_still_gets_a_finite_map() -> None:
    rng = np.random.default_rng(13)
    block = rng.normal(size=(20, 40))
    leakage = fit_leakage_map(block, block[:, [3]] + rng.normal(size=(20, 1)), np.array([3]))
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

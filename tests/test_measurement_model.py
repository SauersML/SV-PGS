"""The measurement model: recalibration scales, their pooling, reliability offsets and the leakage map."""
from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.measurement_model import (
    CalibrationPairs,
    LdBlock,
    apply_leakage_map,
    calibration_moments,
    fit_leakage_map,
    fit_measurement_model,
    leakage_transform,
    log_reliability_offsets,
    mapped_gram,
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
    residual = residual_variances(dosage, genotype, scales)
    reliability = np.exp(log_reliability_offsets(moments.dosage_variance, scales, residual))
    # With Var(D) = Var(G) and Cov(G, D) = keep Var(G), a column scaled by kappa has
    # r^2 = kappa^2 / (kappa^2 + 1 + kappa^2 - 2 kappa keep), which is keep^2 at kappa = keep.
    kappa = float(np.mean(scales))
    expected = kappa**2 / (kappa**2 + 1.0 + kappa**2 - 2.0 * kappa * keep)
    assert abs(float(np.mean(reliability)) - expected) <= sampling_bound(float(np.std(reliability) / np.sqrt(reliability.size)))


def test_the_residual_variance_is_the_mean_square_of_the_calibrated_residual() -> None:
    rng = np.random.default_rng(3)
    _, genotype, dosage = _records(5, 200, 0.7, rng)
    scales = np.array([0.7, 0.6, 0.8, 0.5, 0.9])
    residual = residual_variances(dosage, genotype, scales)
    direct = [np.mean(((genotype[j] - genotype[j].mean()) - scales[j] * (dosage[j] - dosage[j].mean())) ** 2) for j in range(5)]
    np.testing.assert_allclose(residual, direct, rtol=2 * rounding_gamma(4 * 200))


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
    assert leakage.identified and leakage.ridge_ratio > 0.0

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


def test_a_block_the_pairs_can_interpolate_is_left_unmapped_and_says_so() -> None:
    rng = np.random.default_rng(13)
    block = rng.normal(size=(20, 40))
    truth = rng.normal(size=(20, 1))
    leakage = fit_leakage_map(block, truth, np.array([3]))
    assert not leakage.identified
    assert np.all(leakage.coefficients == 0.0)


def test_without_truth_the_model_degrades_loudly_and_records_it() -> None:
    variance = np.array([0.4, 0.3])
    with pytest.raises(ValueError, match="reported r"):
        fit_measurement_model(None, variance, np.zeros(2, dtype=int))
    model = fit_measurement_model(None, variance, np.zeros(2, dtype=int), reported_reliability=np.array([0.9, 0.5]))
    np.testing.assert_array_equal(model.scales, np.ones(2))
    assert model.leakage_maps == ()
    assert "not applied" in str(model.certificate["recalibration"])
    assert "not applied" in str(model.certificate["leakage_correction"])
    np.testing.assert_allclose(model.log_reliability, np.log([0.9, 0.5]))


def test_calibration_pairs_need_typed_research_ids() -> None:
    with pytest.raises(TypeError):
        CalibrationPairs(("1", "2"), np.zeros((1, 2)), np.zeros((1, 2)))
    with pytest.raises(ValueError):
        CalibrationPairs((ResearchId("1"), ResearchId("1")), np.zeros((1, 2)), np.zeros((1, 2)))


def test_with_truth_the_model_recalibrates_and_maps_each_block() -> None:
    rng = np.random.default_rng(17)
    samples = 3000
    sv, snp = _two_locus(rng, samples, 0.3, 0.9)
    draw = _draw_type_column(sv[None], np.array([0.3]), 0.5, rng)[0]
    pairs = CalibrationPairs(tuple(ResearchId(str(index)) for index in range(samples)), np.vstack([draw, snp]), np.vstack([sv, snp]))
    model = fit_measurement_model(
        pairs,
        np.array([np.var(draw), np.var(snp)]),
        np.array([0, 1]),
        blocks=(LdBlock(np.array([0, 1]), np.array([0])),),
    )
    assert model.certificate["reliability_source"] == "calibration pairs"
    assert len(model.leakage_maps) == 1 and model.leakage_maps[0].ridge_ratio > 0.0
    assert model.scales[1] == 1.0

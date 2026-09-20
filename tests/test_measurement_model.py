"""The measurement model: recalibration scales, their pooling, reliability offsets and the leakage map."""
from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.measurement_model import (
    BlockPairs,
    LdBlock,
    MeasurementModel,
    apply_leakage_map,
    calibration_moments,
    calibration_pairs,
    concatenate_calibration_moments,
    engine_blocks,
    fit_leakage_map,
    fit_measurement_model,
    leakage_transform,
    log_reliability_offsets,
    mapped_gram,
    pooled_calibration,
    pooled_log_reliability,
    pooled_measurement_model,
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


def test_records_that_agree_within_their_error_are_pooled_with_no_between_variance() -> None:
    rng = np.random.default_rng(43)
    _, genotype, dosage = _records(100, 500, 0.7, rng)
    moments = calibration_moments(dosage, genotype)
    pooled = pooled_calibration(moments, dosage.var(axis=1), np.zeros(100, dtype=int))
    # One kappa for every record: the moment estimate of tau^2 is at most its own
    # sampling error, from each record's robust S_DD s_j at the common kappa.
    keep = 0.7
    residual = (
        moments.dosage_squared_truth_squared - 2 * keep * moments.dosage_cubed_truth + keep**2 * moments.dosage_fourth
    ) / moments.dosage_variance
    bound = sampling_bound(float(np.sqrt(2 * np.sum(residual**2)) / np.sum(moments.pair_counts * moments.dosage_variance)))
    assert pooled.between_variances[0] <= bound


def test_the_between_record_variance_is_recovered() -> None:
    rng = np.random.default_rng(7)
    records, pairs, spread = 400, 2000, 0.05
    keeps = 0.7 + rng.normal(0.0, spread, records)
    frequency = rng.uniform(0.2, 0.5, size=records)
    genotype = rng.binomial(2, frequency[:, None], size=(records, pairs)).astype(float)
    redraw = rng.binomial(2, frequency[:, None], size=genotype.shape).astype(float)
    dosage = np.where(rng.random(genotype.shape) < keeps[:, None], genotype, redraw)
    pooled = pooled_calibration(calibration_moments(dosage, genotype), dosage.var(axis=1), np.zeros(records, dtype=int))
    # Each record's slope has sampling variance (1 - keep^2) / pairs, so the moment
    # estimate of tau^2 has standard error about sqrt(2 / records) (tau^2 + s).
    sampling = (1 - np.mean(keeps) ** 2) / pairs
    assert abs(pooled.between_variances[0] - spread**2) <= sampling_bound(float(np.sqrt(2 / records) * (spread**2 + sampling)))


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


def test_record_chunks_join_into_the_moments_of_all_records() -> None:
    rng = np.random.default_rng(23)
    _, genotype, dosage = _records(40, 900, 0.6, rng)
    dosage[rng.random(dosage.shape) < 0.1] = np.nan
    dosage[3] = np.nan
    whole = calibration_moments(dosage, genotype)
    assert whole.pair_counts[3] == 0 and whole.dosage_variance[3] == 0.0
    joined = concatenate_calibration_moments([calibration_moments(dosage[:17], genotype[:17]), calibration_moments(dosage[17:], genotype[17:])])
    for field in ("pair_counts", "covariance", "dosage_fourth", "dosage_cubed_truth", "dosage_squared_truth_squared"):
        np.testing.assert_array_equal(getattr(joined, field), getattr(whole, field))
    observed = np.isfinite(dosage[0])
    w = dosage[0, observed] - dosage[0, observed].mean()
    u = genotype[0, observed] - genotype[0, observed].mean()
    assert abs(whole.dosage_cubed_truth[0] - np.mean(w**3 * u)) <= rounding_gamma(4 * observed.sum()) * np.mean(np.abs(w**3 * u))


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
    assert 0.0 < leakage.ridge_ratio < np.inf and not leakage.fits_pairs_exactly
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
    with pytest.raises(ValueError, match="reported_reliability"):
        fit_measurement_model(None, variance, np.zeros(2, dtype=int), np.array([0.9, 1.5]))
    with pytest.raises(ValueError, match="undefined"):
        fit_measurement_model(None, variance, np.zeros(2, dtype=int), np.array([0.9, np.nan]))
    reported = np.array([0.9, 0.5])
    model = fit_measurement_model(None, variance, np.zeros(2, dtype=int), reported)
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
    model = fit_measurement_model(pairs, np.array([np.var(draw), np.var(snp)]), np.array([0, 1]), np.array([0.3, 1.0]))
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
    reported = np.full(6, 0.8)
    model = fit_measurement_model(pairs, variance, np.zeros(6, dtype=int), reported)
    assert model.certificate["calibrated_records"] == 4 and model.certificate["uncalibrated_records"] == 2
    np.testing.assert_array_equal(model.scales[4:], np.ones(2))
    np.testing.assert_allclose(model.log_reliability[4:], np.log(reported[4:]), rtol=rounding_gamma(2))
    assert np.all(model.scales[:4] < 1.0)


def test_a_copy_number_column_is_calibrated_in_copies() -> None:
    rng = np.random.default_rng(47)
    records, pairs, error = 300, 2000, 0.2
    copy_numbers = rng.binomial(5, 0.4, size=(records, pairs)).astype(float)
    # A miscall moves the measured copy number by one copy either way, independent of it.
    measured = copy_numbers + rng.choice([-1.0, 0.0, 1.0], size=copy_numbers.shape, p=[error / 2, 1 - error, error / 2])
    modal = 2.0
    moments = calibration_moments(measured - modal, copy_numbers - modal)
    pooled = pooled_calibration(moments, (measured - modal).var(axis=1), np.zeros(records, dtype=int))
    # Classical error: kappa = lambda = Var(CN) / (Var(CN) + error), so r^2 = kappa^2 / lambda = kappa.
    genotype_variance = 5 * 0.4 * 0.6
    kappa = genotype_variance / (genotype_variance + error)
    residual = genotype_variance - kappa**2 * (genotype_variance + error)
    kappa_error = np.sqrt(residual / ((genotype_variance + error) * pairs * records))
    assert abs(float(np.mean(pooled.scales)) - kappa) <= sampling_bound(float(kappa_error))
    ratios = moments.truth_variance / moments.dosage_variance
    assert abs(float(pooled.variance_ratios[0]) - kappa) <= sampling_bound(float(np.std(ratios) / np.sqrt(records)))


def test_a_direct_call_is_fused_into_its_imputed_record_and_leaves_the_fit() -> None:
    rng = np.random.default_rng(53)
    frequency, keep, pairs, cohort = 0.3, 0.5, 3000, 40000

    def sources(count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        genotype = rng.binomial(2, frequency, size=count).astype(float)
        imputed = _draw_type_column(genotype[None], np.array([frequency]), keep, rng)[0]
        # A read-depth call: the true copy change plus independent read-count noise.
        direct = genotype + rng.normal(0.0, 0.6, count)
        return genotype, imputed, direct

    genotype, imputed, direct = sources(pairs)
    cohort_genotype, cohort_imputed, cohort_direct = sources(cohort)
    cohort_block = np.vstack([cohort_imputed, cohort_direct])
    block = LdBlock(np.array([0, 1]), np.array([0]), np.array([1]))
    calibration = calibration_pairs(
        tuple(ResearchId(str(index)) for index in range(pairs)), np.vstack([imputed, direct]), np.vstack([genotype, genotype]),
        blocks=(block,), block_covariances=(np.cov(cohort_block, bias=True),),
    )
    model = fit_measurement_model(calibration, cohort_block.var(axis=1), np.array([0, 1]), np.array([0.3, 1.0]))
    assert model.log_reliability[1] == -np.inf and model.certificate["direct_calls_fused"] == 1

    centre = cohort_block.mean(axis=1, keepdims=True)
    calibrated = (centre + model.scales[:, None] * (cohort_block - centre)).T
    fused = apply_leakage_map(calibrated, model.leakage_maps[0])[:, 0]

    def r2(column: np.ndarray) -> float:
        return float(np.corrcoef(column, cohort_genotype)[0, 1] ** 2)

    assert r2(fused) > max(r2(calibrated[:, 0]), r2(cohort_direct))
    # The offset is the fused column's share of the genotype variance; its error is that of
    # the variance ratios estimated from the calibration pairs.
    assert abs(model.log_reliability[0] - np.log(r2(fused))) <= sampling_bound(float(np.sqrt(4 / pairs)))


def test_the_model_saves_and_loads_every_array_and_map(tmp_path) -> None:
    rng = np.random.default_rng(59)
    samples = 2000
    sv, snp = _two_locus(rng, samples, 0.3, 0.9)
    draw = _draw_type_column(sv[None], np.array([0.3]), 0.5, rng)[0]
    pairs = calibration_pairs(
        tuple(ResearchId(str(index)) for index in range(samples)), np.vstack([draw, snp]), np.vstack([sv, snp]),
        blocks=(LdBlock(np.array([0, 1]), np.array([0])),), block_covariances=(np.cov(np.vstack([draw, snp]), bias=True),),
    )
    model = fit_measurement_model(pairs, np.array([np.var(draw), np.var(snp)]), np.array([0, 1]), np.array([0.3, 1.0]))
    model.save(tmp_path / "measurement.npz")
    loaded = MeasurementModel.load(tmp_path / "measurement.npz")
    for name in ("scales", "residual_variance", "log_reliability"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(model, name))
    assert loaded.certificate == model.certificate and loaded.digest() == model.digest()
    for original, restored in zip(model.leakage_maps, loaded.leakage_maps, strict=True):
        for name in ("records", "targets", "column_means", "coefficients"):
            np.testing.assert_array_equal(getattr(restored, name), getattr(original, name))
        assert restored.ridge_ratio == original.ridge_ratio
    np.testing.assert_array_equal(loaded.leakage_maps[0].records, [0, 1])


def test_engine_blocks_fold_fused_pairs_into_their_block_and_refuse_a_split_pair() -> None:
    starts, stops = np.array([0, 10, 25]), np.array([10, 25, 40])
    blocks = engine_blocks(starts, stops, np.array([3, 12, 30]), np.array([4, 31]), np.array([3, 30]))
    assert [block.records[0] for block in blocks] == [0, 10, 25]
    assert blocks[0].targets.tolist() == [3] and blocks[0].absorbed.tolist() == [4]
    assert blocks[1].targets.tolist() == [2] and blocks[1].absorbed.tolist() == []
    assert blocks[2].targets.tolist() == [5] and blocks[2].absorbed.tolist() == [6]
    with pytest.raises(ValueError, match="straddles"):
        engine_blocks(starts, stops, np.array([9]), np.array([10]), np.array([9]))


def test_the_pooled_model_pools_the_groups_and_fits_the_maps_once() -> None:
    rng = np.random.default_rng(61)
    models, means, variances, counts = [], [], [], np.array([1500, 500])
    for size in counts:
        sv, snp = _two_locus(rng, int(size), 0.3, 0.9)
        draw = _draw_type_column(sv[None], np.array([0.3]), 0.5, rng)[0]
        pairs = calibration_pairs(tuple(ResearchId(str(index)) for index in range(size)), np.vstack([draw, snp]), np.vstack([sv, snp]))
        stored = np.vstack([draw, snp])
        models.append(fit_measurement_model(pairs, stored.var(axis=1), np.array([0, 1]), np.array([0.3, 1.0])))
        means.append(stored.mean(axis=1))
        variances.append(stored.var(axis=1))
    means, variances = np.column_stack(means), np.column_stack(variances)
    unmapped = pooled_measurement_model(models, means, variances, counts)
    scales = np.column_stack([model.scales for model in models])
    residuals = np.column_stack([model.residual_variance for model in models])
    np.testing.assert_array_equal(unmapped.log_reliability, pooled_log_reliability(scales, residuals, means, variances, counts))
    np.testing.assert_array_equal(unmapped.scales, np.ones(2))
    np.testing.assert_allclose(unmapped.residual_variance, residuals @ (counts / counts.sum()), rtol=rounding_gamma(4))

    sv, snp = _two_locus(rng, 3000, 0.3, 0.9)
    draw = 0.5 * _draw_type_column(sv[None], np.array([0.3]), 0.5, rng)[0]
    cohort = np.vstack([draw, snp])
    block = BlockPairs(LdBlock(np.array([0, 1]), np.array([0]), np.array([1])), cohort.T[:2000], sv[:2000, None], np.cov(cohort, bias=True))
    mapped = pooled_measurement_model(models, means, variances, counts, blocks=(block,))
    assert mapped.log_reliability[1] == -np.inf and mapped.certificate["direct_calls_fused"] == 1
    assert mapped.leakage_maps[0].ridge_ratio > 0.0
    assert mapped.log_reliability[0] > unmapped.log_reliability[0]

from dataclasses import replace

import numpy as np
import pytest

from sv_pgs.imputation_reliability import (
    CalibrationCurve,
    ReliabilityModel,
    calibrated_scale,
    fit_calibration_shape,
    triad_squared_correlation,
)


def _genotypes(generator, count, frequency=0.3):
    return generator.binomial(2, frequency, size=count).astype(float)


def test_triad_recovers_the_true_squared_correlation_that_one_noisy_truth_understates():
    generator = np.random.default_rng(11)
    genotype = _genotypes(generator, 400_000)
    dosage = genotype + generator.normal(0.0, 0.6, genotype.size)
    truth_a = genotype + generator.normal(0.0, 0.4, genotype.size)
    truth_b = genotype + generator.normal(0.0, 0.5, genotype.size)
    true_r2 = np.corrcoef(dosage, genotype)[0, 1] ** 2
    naive_r2 = np.corrcoef(dosage, truth_a)[0, 1] ** 2
    assert triad_squared_correlation(dosage, truth_a, truth_b) == pytest.approx(true_r2, abs=0.005)
    assert naive_r2 < true_r2 - 0.1


def test_triad_drops_missing_samples_and_rejects_unrelated_truths():
    generator = np.random.default_rng(12)
    genotype = _genotypes(generator, 20_000)
    dosage = genotype + generator.normal(0.0, 0.3, genotype.size)
    truth_a = genotype.copy()
    truth_a[::7] = np.nan
    unrelated = generator.normal(0.0, 1.0, genotype.size)
    assert np.isfinite(triad_squared_correlation(dosage, truth_a, genotype))
    with pytest.raises(ValueError, match="not positively correlated"):
        triad_squared_correlation(dosage, -genotype, genotype)
    with pytest.raises(ValueError, match="not positively correlated"):
        triad_squared_correlation(dosage, unrelated - 10 * genotype, genotype)


def test_the_triad_refuses_a_ratio_that_is_not_a_squared_correlation():
    generator = np.random.default_rng(15)
    genotype = _genotypes(generator, 2_000)
    truth_a = genotype + generator.normal(0.0, 0.4, genotype.size)
    truth_b = genotype + generator.normal(0.0, 0.4, genotype.size)
    with pytest.raises(ValueError, match="the dosage is constant"):
        triad_squared_correlation(np.full(genotype.size, 1.5), truth_a, truth_b)
    with pytest.raises(ValueError, match="the second truth is constant"):
        triad_squared_correlation(genotype, truth_a, np.zeros(genotype.size))
    # Three samples cannot resolve the ratio, which comes out at 1.5.
    with pytest.raises(ValueError, match="not a squared correlation"):
        triad_squared_correlation([0.0, 1.0, 2.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0])
    # A dosage driven by the truths' errors breaks the identity's independence: it correlates
    # positively with one truth and negatively with the other, so the ratio is negative.
    shared, error_a, error_b = generator.normal(size=(3, 5_000))
    with pytest.raises(ValueError, match="not a squared correlation"):
        triad_squared_correlation(error_a - error_b, shared + error_a, shared + error_b)


def test_calibration_shape_is_monotone_and_the_identity_for_a_calibrated_dosage():
    generator = np.random.default_rng(13)
    dosage = np.round(generator.uniform(0.0, 2.0, 200_000), 1)
    truth = generator.binomial(2, dosage / 2.0).astype(float)
    curve = fit_calibration_shape(dosage, truth, stratum="calibrated", version="test")
    assert np.all(np.diff(curve.knots_expectation) >= 0.0)
    assert np.max(np.abs(curve.shape(curve.knots_dosage) - curve.knots_dosage)) < 0.05
    assert curve.scale == 1.0


def test_shape_and_triad_scale_restore_calibration_of_an_overconfident_dosage():
    generator = np.random.default_rng(14)
    genotype = _genotypes(generator, 300_000, frequency=0.25)
    # An overconfident, nonlinear dosage: E[G | D] is not D (its OLS slope is about 0.69).
    dosage = np.round(2.0 * np.sqrt(genotype / 2.0) + generator.normal(0.0, 0.3, genotype.size), 2)
    truth_a = genotype + generator.normal(0.0, 0.45, genotype.size)
    truth_b = genotype + generator.normal(0.0, 0.35, genotype.size)
    curve = fit_calibration_shape(dosage, truth_a, stratum="overconfident", version="test")
    shaped = curve.shape(dosage)
    shape_truth_r = np.sqrt(triad_squared_correlation(shaped, truth_a, truth_b))
    genotype_sd = np.sqrt(np.cov(truth_a, truth_b)[0, 1])
    curve = CalibrationCurve(
        curve.stratum, curve.version, curve.knots_dosage, curve.knots_expectation,
        calibrated_scale(shape_truth_r, genotype_sd, float(np.std(shaped))),
    )
    # These truths are on the genotype's own scale, so the shape is the whole map: their noise
    # attenuates corr(D, T), not E[T | D]. The scale is 1 up to its sampling error.
    assert curve.scale == pytest.approx(1.0, abs=0.02)
    recalibrated = curve.apply(dosage)
    covariance = np.cov(genotype, recalibrated)
    assert covariance[0, 1] / covariance[1, 1] == pytest.approx(1.0, abs=0.03)
    raw_covariance = np.cov(genotype, dosage)
    assert abs(raw_covariance[0, 1] / raw_covariance[1, 1] - 1.0) > 0.2


def test_a_doubled_truth_calibrates_the_genotype_back_to_itself():
    """The audit's scaled-truth case: D = G and T = 2G must leave [0, 1, 2] where it is."""
    genotype = np.tile(np.arange(3, dtype=float), 10)
    curve = fit_calibration_shape(genotype, 2.0 * genotype, stratum="doubled", version="test")
    shape = curve.shape(genotype)
    curve = replace(curve, scale=calibrated_scale(1.0, float(np.std(genotype)), float(np.std(shape))))
    # Doubling scales every deviation by an exact power of two, so the scale is exactly 1/2.
    assert curve.scale == 0.5
    np.testing.assert_array_equal(curve.apply(np.arange(3, dtype=float)), np.arange(3, dtype=float))


def test_the_scale_calibrates_at_every_dosage_where_a_linear_map_only_matches_covariance():
    """A curved E[G | D] measured by a tripled truth: the shape's map is exact, the linear one is not."""
    genotype = np.concatenate([
        np.zeros(100),                                          # dosage 0: E[G | D] = 0.0
        np.repeat([0.0, 1.0], [90, 10]),                        # dosage 1: E[G | D] = 0.1
        np.repeat([0.0, 1.0], [60, 40]),                        # dosage 2: E[G | D] = 0.4
        np.full(100, 2.0),                                      # dosage 3: E[G | D] = 2.0
    ])
    dosage = np.repeat([0.0, 1.0, 2.0, 3.0], 100)
    conditional_mean = np.repeat([0.0, 0.1, 0.4, 2.0], 100)
    curve = fit_calibration_shape(dosage, 3.0 * genotype, stratum="tripled", version="test")
    shape = curve.shape(dosage)
    correlation = float(np.corrcoef(shape, genotype)[0, 1])
    curve = replace(curve, scale=calibrated_scale(correlation, float(np.std(genotype)), float(np.std(shape))))
    calibrated = curve.apply(dosage)
    assert curve.scale == pytest.approx(1.0 / 3.0, rel=1e-12)
    np.testing.assert_allclose(calibrated, conditional_mean, atol=1e-12)
    # The best linear recalibration of the same column matches the covariance and still misses
    # E[G | D] by 0.3 or more, and predicts a negative genotype at the lowest dosage.
    slope = float(np.cov(genotype, dosage)[0, 1] / np.var(dosage, ddof=1))
    linear = genotype.mean() + slope * (dosage - dosage.mean())
    assert np.cov(genotype, linear)[0, 1] / np.var(linear, ddof=1) == pytest.approx(1.0, rel=1e-12)
    assert np.max(np.abs(linear - conditional_mean)) > 0.3
    assert linear.min() < 0.0


def test_the_reliability_survives_an_affine_map_of_the_column_but_not_the_shape():
    """The offset is log of a squared correlation, so units and centring do not move it.

    Rescaling and shifting the stored column leaves the reliability, the fitted shape and the
    calibrated column where they were, which is what lets the codec's scale, the engine's centring
    and this recalibration all happen without the prior's offset following them. The monotone shape
    is not an affine map, so the reliability has to be measured on the shaped column
    (docs/design/math/scale_model.md section 1).
    """
    generator = np.random.default_rng(16)
    genotype = _genotypes(generator, 100_000, frequency=0.2)
    dosage = np.round(2.0 * np.sqrt(genotype / 2.0) + generator.normal(0.0, 0.3, genotype.size), 2)
    truth_a = genotype + generator.normal(0.0, 0.4, genotype.size)
    truth_b = genotype + generator.normal(0.0, 0.3, genotype.size)
    rescaled = 0.5 * dosage - 3.0

    stored = triad_squared_correlation(dosage, truth_a, truth_b)
    assert triad_squared_correlation(rescaled, truth_a, truth_b) == pytest.approx(stored, rel=1e-9)

    curve = fit_calibration_shape(dosage, truth_a, stratum="stored", version="test")
    affine = fit_calibration_shape(rescaled, truth_a, stratum="rescaled", version="test")
    np.testing.assert_array_equal(affine.shape(rescaled), curve.shape(dosage))

    shaped = triad_squared_correlation(curve.shape(dosage), truth_a, truth_b)
    assert abs(shaped - stored) > 0.01
    curve = replace(curve, scale=calibrated_scale(
        float(np.sqrt(shaped)), float(np.std(genotype)), float(np.std(curve.shape(dosage)))
    ))
    assert triad_squared_correlation(curve.apply(dosage), truth_a, truth_b) == pytest.approx(shaped, rel=1e-9)


def test_calibration_curve_round_trips_and_rejects_a_constant_dosage():
    curve = CalibrationCurve("VNTR", "v1", np.array([0.0, 1.0, 2.0]), np.array([0.1, 0.8, 1.7]), 0.9)
    restored = CalibrationCurve.from_dict(curve.to_dict())
    np.testing.assert_allclose(restored.apply([0.0, 0.5, 2.0]), curve.apply([0.0, 0.5, 2.0]))
    assert restored.stratum == "VNTR" and restored.version == "v1"
    with pytest.raises(ValueError, match="at least 2 distinct"):
        fit_calibration_shape(np.ones(10), np.ones(10), stratum="flat", version="v1")
    with pytest.raises(ValueError, match="calibrated_scale"):
        calibrated_scale(1.2, 1.0, 1.0)


def test_reliability_model_gives_a_finite_log_offset_for_every_prediction():
    model = ReliabilityModel(
        version="v1",
        feature_names=("info", "is_vntr"),
        intercept=-1.0,
        coefficients=np.array([3.0, -2.0]),
    )
    # The last record's linear predictor, -1 - 3e3 - 2, underflows exp(eta) to 0 in float64.
    features = np.array([[0.9, 0.0], [0.9, 1.0], [-1e3, 1.0]])
    linear_predictor = -1.0 + features @ np.array([3.0, -2.0])
    r2 = model.predict_r2(features)
    assert np.all((r2 >= 0.0) & (r2 < 1.0))
    assert r2[0] > r2[1]
    offset = model.log_reliability_offset(features)
    assert np.all(np.isfinite(offset))
    # Both forms round at most four times (exp, add, divide or log1p, log), each by eps / 2 at most.
    direct = np.log(1.0 / (1.0 + np.exp(-linear_predictor[:2])))
    np.testing.assert_allclose(offset[:2], direct, rtol=0.0, atol=4 * np.finfo(float).eps)
    assert offset[2] == linear_predictor[2]
    restored = ReliabilityModel.from_dict(model.to_dict())
    np.testing.assert_array_equal(restored.log_reliability_offset(features), offset)
    with pytest.raises(ValueError, match="expected features"):
        model.predict_r2(np.zeros((2, 3)))
    record = model.to_dict()
    record["coefficients"] = [1.0]
    with pytest.raises(ValueError, match="do not match"):
        ReliabilityModel.from_dict(record)

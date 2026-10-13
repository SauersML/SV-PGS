import numpy as np
import pytest

from sv_pgs.imputation_reliability import ColumnMeasurement


# ------------------------------------------------------------------ the unit contract (review F21)


def test_a_calibrated_dosage_is_not_shrunk_twice_the_audits_counterexample():
    """q = 0.25, Var(G) = 0.5, tau^2 = 2: a calibrated dosage has Var(D) = q Var(G) = 0.125, and the standardized
    coefficient's prior variance is Var(D) tau^2 = 0.25, not q Var(D) tau^2 = 0.0625. Against a record measured
    exactly with the same Var(G), its offset is lower by log q once, and both have the same log Var(G)."""
    measurement = ColumnMeasurement.build(2, log_reliability=np.log([1.0, 0.25]))
    code_scales = 127.0 * np.sqrt([0.5, 0.125])
    offset, log_genotype_variance = measurement.standardized_terms(np.arange(2), code_scales)
    tau_squared = 2.0
    variances = np.exp(offset - offset[0]) * 0.5 * tau_squared
    np.testing.assert_allclose(variances, [1.0, 0.25])
    np.testing.assert_allclose(log_genotype_variance, np.log([0.5, 0.5]))


def test_a_posterior_mean_dosage_has_covariance_with_its_genotype_equal_to_its_variance():
    """The identity the contract rests on, Cov(G, E[G | O]) = Var(E[G | O]), by exact enumeration of a genotype read
    through a noisy assay [own-sim: the algebra only]; the reliability is then Var(D) / Var(G)."""
    frequency = 0.3
    genotype_prior = np.array([(1 - frequency) ** 2, 2 * frequency * (1 - frequency), frequency**2])
    # O in {0, ..., 4}: a noisy read count whose law depends on G
    likelihood = np.array([[0.7, 0.2, 0.1, 0.0, 0.0], [0.1, 0.3, 0.3, 0.2, 0.1], [0.0, 0.0, 0.2, 0.3, 0.5]])
    joint = genotype_prior[:, None] * likelihood
    posterior_mean = (np.arange(3)[:, None] * joint).sum(axis=0) / joint.sum(axis=0)
    genotypes = np.arange(3.0)
    mean = genotype_prior @ genotypes
    covariance = sum(joint[g, o] * (g - mean) * (posterior_mean[o] - mean) for g in range(3) for o in range(5))
    dosage_variance = joint.sum(axis=0) @ (posterior_mean - mean) ** 2
    genotype_variance = genotype_prior @ (genotypes - mean) ** 2
    assert covariance == pytest.approx(dosage_variance, rel=1e-12)
    reliability = covariance**2 / (dosage_variance * genotype_variance)
    measurement = ColumnMeasurement.build(1, log_reliability=np.log([reliability]))
    _offset, log_genotype_variance = measurement.standardized_terms(np.arange(1), 127.0 * np.sqrt([dosage_variance]))
    assert np.exp(log_genotype_variance[0]) == pytest.approx(genotype_variance, rel=1e-12)


def test_an_uncalibrated_column_takes_its_slope_and_the_general_form_agrees():
    """kappa^2 Var(D) tau^2 = Corr(G, D)^2 Var(G) tau^2 for a column with a known calibration slope (a confident draw
    whose variance stays at Var(G)): the offset carries 2 log kappa, and Var(G) = kappa^2 Var(D) / r^2."""
    generator = np.random.default_rng(5)
    genotype = generator.binomial(2, 0.3, 400_000).astype(float)
    flip = generator.random(genotype.size) < 0.3
    draw = np.where(flip, generator.binomial(2, 0.3, genotype.size), genotype).astype(float)
    kappa = np.cov(genotype, draw)[0, 1] / np.var(draw, ddof=1)
    reliability = np.corrcoef(genotype, draw)[0, 1] ** 2
    measurement = ColumnMeasurement.build(2, log_reliability=np.log([1.0, reliability]), calibration_slope=np.array([1.0, kappa]))
    offset, log_genotype_variance = measurement.standardized_terms(np.arange(2), 127.0 * np.array([genotype.std(), draw.std()]))
    # Corr^2 Var(G) against 1 x Var(G): the offsets differ by log r^2, exactly the general form's
    np.testing.assert_allclose(offset[1] - offset[0], np.log(reliability), atol=0.01)
    np.testing.assert_allclose(log_genotype_variance, np.log(genotype.var()), atol=0.01)


def test_the_contract_refuses_what_is_not_a_measurement():
    with pytest.raises(ValueError, match="log reliability"):
        ColumnMeasurement.build(2, log_reliability=np.array([0.1, 0.0]))
    with pytest.raises(ValueError, match="calibration slope"):
        ColumnMeasurement.build(2, calibration_slope=np.array([1.0, -0.5]))
    with pytest.raises(ValueError, match="encoding scale"):
        ColumnMeasurement.build(2, codes_per_unit=np.array([127.0, 0.0]))
    with pytest.raises(ValueError, match="one value per record"):
        ColumnMeasurement.build(3, log_reliability=np.zeros(2))


def test_the_encoding_scale_cancels_in_the_raw_unit():
    """A copy-number column with 84 codes per copy and a dosage column with 127 per allele, both of raw-unit variance
    0.4: the same offset, whatever their code spreads."""
    measurement = ColumnMeasurement.build(2, codes_per_unit=np.array([127.0, 84.0]), raw_unit=np.array(["ALT allele", "copy"], dtype=object))
    offset, _log_genotype_variance = measurement.standardized_terms(np.arange(2), np.sqrt(0.4) * np.array([127.0, 84.0]))
    np.testing.assert_allclose(offset, [0.0, 0.0], atol=1e-14)

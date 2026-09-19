"""Sample-side logistic EP: certified tilted moments, sites, evidence and separation."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import integrate
from scipy.special import log_expit

from sv_pgs import logistic_ep

EPSILON = float(np.finfo(np.float64).eps)
ARITHMETIC_SLACK = 4.0 * EPSILON
"""Four correctly rounded operations follow the sums: a log, an add, a multiply and a subtract."""


def _reference_moments(mean: float, variance: float, label: float) -> tuple[float, float, float, float]:
    """log Z, tilted mean and variance by adaptive quadrature, and the quadrature's relative error."""
    sign = 2.0 * label - 1.0
    spread = math.sqrt(variance)
    reference = float(log_expit(sign * mean))

    def weight(z: float, power: int) -> float:
        return z**power * math.exp(float(log_expit(sign * (mean + spread * z))) - reference - 0.5 * z * z) / math.sqrt(2.0 * math.pi)

    values, errors = [], []
    for power in (0, 1, 2):
        value, error = integrate.quad(weight, -np.inf, np.inf, args=(power,), epsabs=0.0, epsrel=EPSILON**0.75, limit=400)
        values.append(value)
        errors.append(error)
    zeroth, first, second = values
    standardized_mean = first / zeroth
    tilted_variance = variance * (second / zeroth - standardized_mean**2)
    relative = max(error / zeroth for error in errors)
    return reference + math.log(zeroth), mean + spread * standardized_mean, tilted_variance, relative


@pytest.mark.parametrize("mean", [-30.0, -6.0, -0.4, 0.0, 1.5, 9.0, 32.0])
@pytest.mark.parametrize("variance", [1e-6, 0.03, 1.0, 7.0, 45.0])
@pytest.mark.parametrize("label", [0.0, 1.0])
def test_tilted_moments_match_adaptive_quadrature_within_the_certificate(mean: float, variance: float, label: float) -> None:
    moments = logistic_ep.tilted_moments(np.array([mean]), np.array([variance]), np.array([label]))
    log_normalizer, tilted_mean, tilted_variance, quadrature = _reference_moments(mean, variance, label)
    spread = math.sqrt(variance)
    log_bound = float(moments.relative_error[0]) + quadrature + ARITHMETIC_SLACK * abs(log_normalizer)
    assert abs(float(moments.log_normalizer[0]) - log_normalizer) <= log_bound
    moment_bound = float(moments.moment_error[0]) + quadrature * (1.0 + abs(tilted_mean - mean) / spread + tilted_variance / variance)
    assert abs(float(moments.mean[0]) - tilted_mean) <= spread * moment_bound + ARITHMETIC_SLACK * abs(tilted_mean)
    assert abs(float(moments.variance[0]) - tilted_variance) <= variance * 3.0 * moment_bound + ARITHMETIC_SLACK * tilted_variance


def test_sites_are_non_negative_and_reproduce_the_tilted_moments() -> None:
    grid_mean, grid_variance, grid_label = np.meshgrid([-20.0, -2.0, 0.0, 3.0, 25.0], [1e-4, 0.5, 4.0, 30.0], [0.0, 1.0])
    mean, variance, label = grid_mean.ravel(), grid_variance.ravel(), grid_label.ravel()
    moments = logistic_ep.tilted_moments(mean, variance, label)
    sites = logistic_ep.site_update(mean, variance, moments)
    assert np.all(sites.precision >= 0.0)
    # The product of cavity and site is the tilted Gaussian: its precision and shift add.
    product_variance = 1.0 / (1.0 / variance + sites.precision)
    product_mean = product_variance * (mean / variance + sites.shift)
    rounding = 8.0 * EPSILON
    assert np.allclose(product_variance, moments.variance, rtol=rounding, atol=0.0)
    assert np.allclose(product_mean, moments.mean, rtol=rounding, atol=rounding * np.sqrt(variance))
    weights, response = logistic_ep.working_model(sites)
    assert np.all(weights >= 0.0) and np.all(np.isfinite(response))


def test_one_sample_ep_is_the_exact_posterior_and_its_evidence_the_exact_likelihood() -> None:
    design = np.array([[1.3]])
    covariates = np.zeros((1, 0))
    labels = np.array([1.0])
    prior_precision = np.array([0.8])
    prior_shift = np.array([0.4])
    fit = logistic_ep.dense_fit(design, covariates, labels, prior_precision, prior_shift, tolerance=math.sqrt(EPSILON))
    prior_mean = prior_shift[0] / prior_precision[0]
    prior_sd = 1.0 / math.sqrt(prior_precision[0])

    def unnormalized(beta: float, power: int) -> float:
        density = math.exp(-0.5 * ((beta - prior_mean) / prior_sd) ** 2) / (prior_sd * math.sqrt(2.0 * math.pi))
        return beta**power * math.exp(float(log_expit(design[0, 0] * beta))) * density

    moments = [integrate.quad(unnormalized, -np.inf, np.inf, args=(power,), epsabs=0.0, epsrel=EPSILON**0.75)[0] for power in (0, 1, 2)]
    exact_mean = moments[1] / moments[0]
    exact_variance = moments[2] / moments[0] - exact_mean**2
    bound = EPSILON**0.75 + math.sqrt(EPSILON)
    assert fit.mean[0] == pytest.approx(exact_mean, rel=bound)
    assert fit.covariance[0, 0] == pytest.approx(exact_variance, rel=bound)
    assert fit.log_evidence == pytest.approx(math.log(moments[0]), abs=bound)


def test_dense_ep_reaches_a_moment_matched_fixed_point() -> None:
    generator = np.random.default_rng(20260919)
    sample_count, variant_count = 60, 4
    design = generator.standard_normal((sample_count, variant_count))
    covariates = np.column_stack([np.ones(sample_count), generator.standard_normal(sample_count)])
    truth = design @ generator.normal(0.0, 0.7, variant_count) + covariates @ np.array([-0.5, 0.3])
    labels = (generator.random(sample_count) < 1.0 / (1.0 + np.exp(-truth))).astype(np.float64)
    prior_precision = np.full(variant_count, 2.0)
    prior_shift = np.zeros(variant_count)
    tolerance = math.sqrt(EPSILON)
    fit = logistic_ep.dense_fit(design, covariates, labels, prior_precision, prior_shift, tolerance=tolerance)
    _, _, marginal_mean, marginal_variance = logistic_ep.dense_marginals(design, covariates, prior_precision, prior_shift, fit.sites)
    cavity_mean, cavity_variance = logistic_ep.cavities(marginal_mean, marginal_variance, fit.sites)
    moments = logistic_ep.tilted_moments(cavity_mean, cavity_variance, labels)
    # At a fixed point q's marginal is each site's tilted distribution, to the stopping tolerance.
    assert np.all(np.abs(moments.mean - marginal_mean) <= tolerance * np.sqrt(cavity_variance) * (1.0 + np.abs(marginal_mean)))
    assert np.all(np.abs(moments.variance - marginal_variance) <= tolerance * cavity_variance)
    assert np.isfinite(fit.log_evidence)


def test_separating_covariates_are_refused() -> None:
    generator = np.random.default_rng(7)
    sample_count = 50
    score = generator.standard_normal(sample_count)
    complete = np.column_stack([np.ones(sample_count), score])
    with pytest.raises(logistic_ep.SeparatedLabels):
        logistic_ep.assert_no_separation(complete, (score > 0.0).astype(np.float64))
    # Quasi-separation: every sample with the indicator is a case, as with a sex-limited disease.
    indicator = (np.arange(sample_count) % 2).astype(np.float64)
    labels = np.where(indicator == 1.0, 1.0, (generator.random(sample_count) < 0.3).astype(np.float64))
    with pytest.raises(logistic_ep.SeparatedLabels):
        logistic_ep.assert_no_separation(np.column_stack([np.ones(sample_count), indicator]), labels)
    mixed = (generator.random(sample_count) < 0.4).astype(np.float64)
    logistic_ep.assert_no_separation(complete, mixed)


def test_an_improper_cavity_raises() -> None:
    sites = logistic_ep.SampleSites(precision=np.array([2.0]), shift=np.array([0.0]))
    with pytest.raises(logistic_ep.ImproperCavity):
        logistic_ep.cavities(np.array([0.0]), np.array([1.0]), sites)

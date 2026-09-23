"""The Polya-Gamma Bernoulli bound (``binary_likelihood``): the bound against the exact logistic likelihood, the
weighted Gaussian step against dense algebra, the ascent's monotonicity, recovery and calibration.

Every simulation here is [own-sim]: it checks the mathematics, never accuracy."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate
from scipy.special import expit, logit

from sv_pgs.binary_likelihood import (
    BernoulliSites,
    BoundDecreased,
    CovariateStep,
    GaussianPriorStep,
    WeightedGaussianState,
    ascertainment_offset,
    auc,
    bernoulli_ascent,
    bernoulli_log_likelihood,
    bound_intercept,
    brier_score,
    calibrated_shift,
    calibration,
    covariate_evidence,
    fit_binary_covariates,
    log_loss,
    polya_gamma_mean,
    probability,
    site_bound,
)
from sv_pgs.logistic_ep import SeparatedLabels


def test_polya_gamma_mean_is_the_tanh_form_with_its_limit_at_zero() -> None:
    xi = np.array([1e-3, 0.1, 1.0, 5.0, 40.0])
    assert np.allclose(polya_gamma_mean(xi), np.tanh(xi / 2.0) / (2.0 * xi), rtol=1e-14, atol=0.0)
    assert polya_gamma_mean(np.array([0.0]))[0] == 0.25
    # Continuous at 0 to every digit (the series is 1/4 - xi^2 / 48 + ...).
    small = np.array([1e-9, 1e-6])
    assert np.allclose(polya_gamma_mean(small), 0.25 - small**2 / 48.0, rtol=1e-15, atol=0.0)
    assert np.array_equal(polya_gamma_mean(-xi), polya_gamma_mean(xi))


def test_the_bound_is_below_the_exact_log_likelihood_and_tight_at_xi_equal_to_eta() -> None:
    generator = np.random.default_rng(0)
    eta = generator.normal(scale=4.0, size=500)
    labels = (generator.random(500) < 0.5).astype(float)
    exact = bernoulli_log_likelihood(labels, eta)
    for xi in (np.zeros(500), np.abs(generator.normal(scale=3.0, size=500)), np.full(500, 30.0)):
        assert np.all(site_bound(labels, xi, eta, eta * eta) <= exact + 1e-13)
    tight = site_bound(labels, np.abs(eta), eta, eta * eta)
    assert np.allclose(tight, exact, rtol=0.0, atol=1e-12)
    # c(0) = -log 2: at eta = 0 the bound is the likelihood log sigmoid(0).
    assert bound_intercept(np.zeros(1))[0] == pytest.approx(-np.log(2.0), abs=1e-16)


@pytest.mark.parametrize("mean,variance,label", [(0.3, 0.5, 1.0), (-2.0, 4.0, 1.0), (1.5, 0.01, 0.0), (6.0, 9.0, 0.0)])
def test_the_bound_under_a_gaussian_q_is_below_the_expected_log_likelihood_at_the_optimal_xi(mean: float, variance: float, label: float) -> None:
    def integrand(value: float) -> float:
        density = np.exp(-0.5 * (value - mean) ** 2 / variance) / np.sqrt(2.0 * np.pi * variance)
        return float(bernoulli_log_likelihood(np.array([label]), np.array([value]))[0] * density)

    spread = np.sqrt(variance)
    expected, _error = integrate.quad(integrand, mean - 40.0 * spread, mean + 40.0 * spread, limit=400, epsabs=1e-13)
    second = mean * mean + variance
    optimal = site_bound(np.array([label]), np.array([np.sqrt(second)]), np.array([mean]), np.array([second]))[0]
    assert optimal <= expected + 1e-10
    # xi^2 = E eta^2 maximizes the bound over xi.
    for other in (0.5 * np.sqrt(second), 2.0 * np.sqrt(second), 0.0):
        assert site_bound(np.array([label]), np.array([other]), np.array([mean]), np.array([second]))[0] <= optimal + 1e-15


def test_the_xi_update_gain_is_exact_and_non_negative() -> None:
    generator = np.random.default_rng(1)
    labels = (generator.random(50) < 0.3).astype(float)
    sites = BernoulliSites(labels=labels, training=np.ones(50, dtype=bool), xi=np.abs(generator.normal(size=50)))
    mean = generator.normal(size=50)
    variance = generator.random(50)
    updated, gain = sites.updated(mean, variance)
    assert gain >= 0.0
    assert np.allclose(updated.xi, np.sqrt(mean * mean + variance))
    assert gain == pytest.approx(updated.value(mean, variance) - sites.value(mean, variance), rel=1e-12)
    # L's split: the sites' terms equal the weighted Gaussian quadratic plus the constant.
    weights, response = sites.weights, sites.response
    quadratic = -0.5 * float(np.sum(weights * ((response - mean) ** 2 + variance)))
    assert sites.value(mean, variance) == pytest.approx(quadratic + sites.constant(), rel=1e-12)


def test_rows_off_the_training_mask_carry_no_weight_and_no_term() -> None:
    labels = np.array([1.0, 0.0, np.nan, 1.0])
    training = np.array([True, True, False, True])
    sites = BernoulliSites(labels=labels, training=training, xi=np.array([0.5, 1.0, 3.0, 0.0]))
    assert sites.weights[2] == 0.0 and sites.response[2] == 0.0 and sites.xi[2] == 0.0
    full = BernoulliSites(labels=labels[training], training=np.ones(3, dtype=bool), xi=sites.xi[training])
    assert sites.constant() == pytest.approx(full.constant(), rel=1e-15)


def test_covariate_evidence_is_the_flat_prior_integral() -> None:
    generator = np.random.default_rng(2)
    covariates = np.column_stack([np.ones(40), generator.normal(size=(40, 2))])
    weights = generator.random(40) + 0.1
    gram = covariates.T @ (weights[:, None] * covariates)
    expected = 1.5 * np.log(2.0 * np.pi) - 0.5 * np.linalg.slogdet(gram)[1]
    assert covariate_evidence(weights, covariates) == pytest.approx(expected, rel=1e-12)
    # A dependent column adds no dimension (the rank rule).
    dependent = np.column_stack([covariates, covariates[:, 1]])
    singular = np.linalg.svd(np.sqrt(weights)[:, None] * dependent, compute_uv=False)
    assert covariate_evidence(weights, dependent) == pytest.approx(1.5 * np.log(2.0 * np.pi) - float(np.sum(np.log(singular[:3]))), rel=1e-12)


def _problem(seed: int, samples: int = 120, columns: int = 6, scale: float = 0.8) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    genotypes = generator.normal(size=(samples, columns))
    genotypes = (genotypes - genotypes.mean(axis=0)) / genotypes.std(axis=0)
    covariates = np.column_stack([np.ones(samples), generator.normal(size=samples)])
    effects = generator.normal(scale=scale, size=columns)
    eta = -0.5 + 0.4 * covariates[:, 1] + genotypes @ effects
    labels = (generator.random(samples) < expit(eta)).astype(float)
    return genotypes, covariates, labels, effects


def test_the_weighted_gaussian_step_matches_dense_joint_algebra() -> None:
    genotypes, covariates, labels, _effects = _problem(3)
    generator = np.random.default_rng(4)
    sites = BernoulliSites(labels=labels, training=np.ones(labels.shape[0], dtype=bool), xi=np.abs(generator.normal(size=labels.shape[0])))
    relative = generator.random(genotypes.shape[1]) + 0.5
    step = GaussianPriorStep(genotypes, covariates, relative)
    state = step(sites.weights, sites.response)
    scale = state.detail["scale"]
    # The joint Gaussian over (alpha, beta) with the flat prior on alpha, W = diag(omega), precision
    # [C'WC, C'WX; X'WC, X'WX + (t U)^-1] and shift [C'kappa; X'kappa].
    features = np.hstack([covariates, genotypes])
    weights = sites.weights
    precision = features.T @ (weights[:, None] * features)
    precision[covariates.shape[1]:, covariates.shape[1]:] += np.diag(1.0 / (scale * relative))
    covariance = np.linalg.inv(precision)
    joint_mean = covariance @ (features.T @ sites.kappa)
    assert np.allclose(state.detail["effects"], joint_mean[covariates.shape[1]:], rtol=1e-9, atol=1e-11)
    assert np.allclose(state.predictor_mean, features @ joint_mean, rtol=1e-9, atol=1e-10)
    assert np.allclose(state.predictor_variance, np.einsum("ij,jk,ik->i", features, covariance, features), rtol=1e-9, atol=1e-12)
    # The value is the weighted Gaussian integral over (alpha, beta), E_q[-||W^1/2 (z - eta)||^2 / 2] - KL + H: the
    # log of int exp(-||W^1/2 (z - C a - X b)||^2 / 2) N(b; 0, t U) da db, in closed form.
    response = sites.response
    quadratic = float(response @ (weights * response)) - float(joint_mean @ (features.T @ sites.kappa))
    prior_precision = np.zeros_like(precision)
    prior_precision[covariates.shape[1]:, covariates.shape[1]:] = np.diag(1.0 / (scale * relative))
    log_integral = (
        -0.5 * quadratic - 0.5 * np.linalg.slogdet(precision)[1] + 0.5 * features.shape[1] * np.log(2.0 * np.pi)
        - 0.5 * float(np.sum(np.log(2.0 * np.pi * scale * relative)))
    )
    assert state.value == pytest.approx(log_integral, rel=1e-10)


def test_the_ascent_is_monotone_and_the_bound_is_below_the_exact_evidence() -> None:
    genotypes, covariates, labels, _effects = _problem(5, samples=40, columns=1, scale=1.5)
    covariates = covariates[:, :1]
    step = GaussianPriorStep(genotypes, covariates, np.ones(1))
    ascent = bernoulli_ascent(step, BernoulliSites.start(labels), tolerance=1e-10)
    assert np.all(np.diff(ascent.history) >= -1e-10)
    assert ascent.remaining_gain <= 1e-10
    # The exact log evidence at the ascent's prior variance: int int prod_i p(y_i | a + x_i b) N(b; 0, t) da db, by
    # quadrature over (a, b); the bound is below it, and the Jaakkola-Jordan gap is small on a 40-sample problem.
    scale = ascent.state.detail["scale"]
    x = genotypes[:, 0]

    grid_a = np.linspace(-12.0, 12.0, 801)
    grid_b = np.linspace(-12.0, 12.0, 801)
    predictor = grid_a[:, None, None] + grid_b[None, :, None] * x[None, None, :]
    values = (
        np.sum(bernoulli_log_likelihood(labels[None, None, :], predictor), axis=2)
        - 0.5 * grid_b[None, :] ** 2 / scale - 0.5 * np.log(2.0 * np.pi * scale)
    )
    peak = values.max()
    log_evidence = peak + np.log(np.sum(np.exp(values - peak)) * (grid_a[1] - grid_a[0]) * (grid_b[1] - grid_b[0]))
    assert values[0, :].max() < peak - 40.0 and values[:, 0].max() < peak - 40.0 and values[-1, :].max() < peak - 40.0 and values[:, -1].max() < peak - 40.0
    assert ascent.bound <= log_evidence + 1e-6
    assert ascent.bound >= log_evidence - 1.0


def test_a_step_that_lowers_the_bound_is_refused() -> None:
    labels = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
    calls = []

    def step(weights: np.ndarray, response: np.ndarray) -> WeightedGaussianState:
        calls.append(1)
        value = 0.0 if len(calls) == 1 else -100.0
        return WeightedGaussianState(np.full(5, 2.0), np.ones(5), value)

    with pytest.raises(BoundDecreased):
        bernoulli_ascent(step, BernoulliSites.start(labels), tolerance=1e-12)


def test_recovery_on_a_small_simulation() -> None:
    """[own-sim] the effects' posterior means track the simulated ones."""
    genotypes, covariates, labels, effects = _problem(6, samples=3000, columns=5, scale=0.7)
    step = GaussianPriorStep(genotypes, covariates, np.ones(5))
    ascent = bernoulli_ascent(step, BernoulliSites.start(labels), tolerance=1e-8)
    estimate = ascent.state.detail["effects"]
    standard_error = 1.0 / np.sqrt(0.2 * genotypes.shape[0])
    assert np.all(np.abs(estimate - effects) < 5.0 * standard_error)


def test_the_null_genetic_model_is_the_covariate_logistic_fit() -> None:
    generator = np.random.default_rng(7)
    covariates = np.column_stack([np.ones(4000), generator.normal(size=4000)])
    labels = (generator.random(4000) < expit(-1.0 + 0.8 * covariates[:, 1])).astype(float)
    ascent, alpha, covariance = fit_binary_covariates(covariates, labels, tolerance=1e-9)
    assert np.all(np.diff(ascent.history) >= -1e-9)
    assert np.all(np.abs(alpha - np.array([-1.0, 0.8])) < 5.0 * np.sqrt(np.diag(covariance)))
    # The variational mean is near the maximum likelihood estimate when the posterior is concentrated.
    mle = calibration(labels, expit(covariates[:, 1]))
    assert alpha[1] == pytest.approx(mle.slope, abs=2.0 * np.sqrt(covariance[1, 1]))
    with pytest.raises(SeparatedLabels):
        fit_binary_covariates(covariates, (covariates[:, 1] > 0.0).astype(float), tolerance=1e-9)


def test_ascertainment_moves_the_intercept_by_the_prevalence_logits() -> None:
    """[own-sim] a case-control sample of a logistic population recovers the population intercept after the offset."""
    generator = np.random.default_rng(8)
    score = generator.normal(size=400_000)
    population = (generator.random(score.shape[0]) < expit(-3.0 + score)).astype(float)
    prevalence = float(population.mean())
    cases = np.flatnonzero(population == 1.0)
    controls = generator.choice(np.flatnonzero(population == 0.0), size=cases.shape[0], replace=False)
    rows = np.concatenate([cases, controls])
    fitted = calibration(population[rows], expit(score[rows]))
    standard_error = np.sqrt(4.0 / cases.shape[0])
    assert fitted.slope == pytest.approx(1.0, abs=4.0 * standard_error)
    assert fitted.intercept + ascertainment_offset(0.5, prevalence) == pytest.approx(-3.0, abs=4.0 * standard_error)
    assert ascertainment_offset(0.2, 0.2) == 0.0
    with pytest.raises(ValueError):
        ascertainment_offset(0.0, 0.1)


def test_calibration_of_the_predicted_probabilities() -> None:
    """[own-sim] labels drawn from the predictive are calibrated: slope 1, intercept 0, in the large 0, within their SEs;
    the calibrated shift puts the mean predictive at the training prevalence, and at a population prevalence after the
    ascertainment offset."""
    generator = np.random.default_rng(9)
    mean = generator.normal(scale=1.2, size=20_000) - 1.0
    variance = generator.random(20_000)
    chance = probability(mean, variance, 0.0)
    labels = (generator.random(20_000) < chance).astype(float)
    fitted = calibration(labels, chance)
    assert abs(fitted.slope - 1.0) < 4.0 * 0.03 and abs(fitted.intercept) < 4.0 * 0.03
    assert abs(fitted.in_the_large) < 4.0 * np.sqrt(0.25 / labels.shape[0])
    shift = calibrated_shift(mean, variance, labels)
    assert float(np.mean(probability(mean, variance, shift))) == pytest.approx(labels.mean(), abs=1e-12)
    deployed = calibrated_shift(mean, variance, labels, population_prevalence=0.02)
    assert deployed - shift == pytest.approx(logit(0.02) - logit(labels.mean()), rel=1e-12)


def test_evaluation_metrics_by_their_definitions() -> None:
    labels = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    chance = np.array([0.9, 0.2, 0.6, 0.6, 0.3])
    assert log_loss(labels, chance) == pytest.approx(-np.mean([np.log(0.9), np.log(0.8), np.log(0.6), np.log(0.4), np.log(0.3)]))
    assert brier_score(labels, chance) == pytest.approx(np.mean((chance - labels) ** 2))
    # Pairs (case, control): (0.9 > 0.2), (0.9 > 0.6), (0.6 > 0.2), (0.6 = 0.6 half), (0.3 > 0.2), (0.3 < 0.6).
    assert auc(labels, chance) == pytest.approx(4.5 / 6.0)
    assert log_loss(np.array([1.0]), np.array([0.0])) == np.inf


def test_the_covariate_step_is_the_exact_conditional() -> None:
    generator = np.random.default_rng(10)
    covariates = np.column_stack([np.ones(30), generator.normal(size=30)])
    labels = (generator.random(30) < 0.4).astype(float)
    sites = BernoulliSites(labels=labels, training=np.ones(30, dtype=bool), xi=np.abs(generator.normal(size=30)))
    step = CovariateStep(covariates)
    state = step(sites.weights, sites.response)
    alpha, covariance = step.coefficients(sites.weights, sites.response)
    assert np.allclose(state.predictor_mean, covariates @ alpha, rtol=1e-12, atol=1e-12)
    assert np.allclose(state.predictor_variance, np.einsum("ij,jk,ik->i", covariates, covariance, covariates), rtol=1e-10)

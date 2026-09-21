"""The small-n route's mean-field inference (sv_pgs/mean_field.py): each sweep is coordinate ascent on the exact
ELBO, its fixed point is the tilted-moment fixed point of the engine's own kernel terms, the linear response's sites
are the tilted precisions less the pseudo-likelihoods', and the whole fit certifies and scores. Synthetic data only:
machinery checks, never accuracy evidence."""

import numpy as np
import pytest

from sv_pgs.config import VariantClass
from sv_pgs.mean_field import MeanFieldFixedPoints
from sv_pgs.scale_mixture_ep import initial_hyperparameters, noise_gain, tilted_moments
from sv_pgs.small_n import dense_statistics, fit_small_n, small_n_prior, small_n_start

_WORKING_BYTES = 1 << 26


def _problem(seed: int, samples: int = 80, variants: int = 40, tied: bool = False):
    rng = np.random.default_rng(seed)
    dosage = rng.binomial(2, rng.uniform(0.05, 0.5, variants), size=(samples, variants))
    if tied:
        dosage[:, 3] = dosage[:, 1]
        dosage[:, 7] = 2 - dosage[:, 1]
    effects = np.zeros(variants)
    effects[[2, 11, 25]] = [0.9, -0.7, 0.6]
    standardized = (dosage - dosage.mean(axis=0)) / np.maximum(dosage.std(axis=0), 1e-9)
    covariate = rng.standard_normal(samples)
    target = standardized @ effects + 0.4 * covariate + rng.standard_normal(samples)
    codes = (dosage * 127).astype(np.uint8)
    classes = np.full(variants, list(VariantClass).index(VariantClass.SNV), dtype=np.uint8)
    classes[[11, 25]] = list(VariantClass).index(VariantClass.DELETION)
    covariates = np.column_stack([np.ones(samples), covariate])
    return codes, covariates, target, classes


def _oracle(seed: int, tied: bool = False):
    codes, covariates, target, classes = _problem(seed, tied=tied)
    statistics = dense_statistics(codes, covariates, target)
    prior = small_n_prior(statistics, classes, np.zeros(codes.shape[1]), 64)
    start, start_noise, _moment = small_n_start(statistics, prior)
    return statistics, prior, start, MeanFieldFixedPoints(statistics, prior, start_noise, 64, _WORKING_BYTES)


@pytest.mark.parametrize("tied", [False, True])
def test_every_sweep_raises_the_elbo_and_the_fixed_point_is_the_tilted_moments(tied):
    statistics, prior, start, oracle = _oracle(3, tied=tied)
    values = []
    for _sweep in range(6):
        divergence, weighted_variance, residual_square = oracle._sweep(start)
        values.append(oracle._elbo(divergence, weighted_variance, residual_square))
    gains = np.diff(values)
    assert np.all(gains >= -1e-9 * np.abs(values[1:]))
    # The residual is r = y_P - Xp m, exactly.
    expected = statistics.projected_target - statistics.design.image(oracle.mean)
    np.testing.assert_allclose(oracle.residual, expected, rtol=1e-10, atol=1e-12)
    (point,) = oracle([start])
    assert point is not None
    # Every q_j is the prior tilted by its pseudo-likelihood: the engine's tilted moments at the cavity (omega, h).
    moments = tilted_moments(prior, start, point.cavity, _WORKING_BYTES)
    np.testing.assert_allclose(point.mean, moments.mean, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(oracle.variance, moments.variance, rtol=1e-6, atol=1e-9)
    # The pseudo-likelihood precision is ||x_j||^2 / sigma^2 and its shift the residual without j's term.
    squares = statistics.design.squares
    np.testing.assert_allclose(point.cavity.precision, squares / oracle.noise, rtol=1e-12)
    without = statistics.design.back(oracle.residual) + squares * oracle.mean
    np.testing.assert_allclose(point.cavity.shift, without / oracle.noise, rtol=1e-6, atol=1e-9)
    # The linear response's sites are 1 / v_j - omega_j, and the response is that Gaussian's covariance.
    np.testing.assert_allclose(oracle.site_precision, 1.0 / oracle.variance - squares / oracle.noise, rtol=1e-10)
    design = statistics.projected
    covariance = oracle.noise * np.linalg.inv(design.T @ design + np.diag(oracle.noise * oracle.site_precision))
    right = np.random.default_rng(1).standard_normal((prior.variant_count, 3))
    np.testing.assert_allclose(point.posterior.solve(right, 0.0), covariance @ right, rtol=1e-8, atol=1e-10)
    direction = right[:, 0]
    np.testing.assert_allclose(point.precision_norm(direction), direction @ np.linalg.solve(covariance, direction), rtol=1e-8)


def test_the_noise_update_is_the_elbos_stationary_value():
    _statistics, _prior, start, oracle = _oracle(5)
    (point,) = oracle([start])
    assert point is not None
    # The returned state is the last sweep's, at its noise; the noise's stationary value there is pending, and its
    # exact gain is part of the certified remainder.
    residual = oracle.residual
    pending = (float(residual @ residual) + float(oracle.member_squares @ oracle.variance)) / oracle.residual_dimension
    np.testing.assert_allclose(oracle.noise_gain, noise_gain(pending, oracle.noise, oracle.sample_count, oracle.covariate_count), rtol=1e-10)
    assert oracle.noise_gain <= 0.5 / 64 and oracle.mean_move + oracle.noise_gain <= 0.5 / 64


def test_a_refused_call_restores_the_state():
    _statistics, prior, start, oracle = _oracle(8)
    oracle([start])
    before = oracle._snapshot()
    # Coefficients far past every effect's scale: the sweep overflows and the call refuses, leaving the state as it was.
    absurd = initial_hyperparameters(prior)
    absurd = type(absurd)(coefficients=absurd.coefficients + 1e4, log_smoothing=absurd.log_smoothing)
    (point,) = oracle([absurd])
    if point is None:
        assert oracle.refusals
        np.testing.assert_array_equal(oracle.mean, before["mean"])
        assert oracle.noise == before["noise"]


def test_the_draws_come_from_q():
    _statistics, prior, start, oracle = _oracle(13)
    (point,) = oracle([start])
    assert point is not None
    draws = oracle.draws(start, np.random.default_rng(2), 4000)
    assert draws.shape == (prior.variant_count, 4000) and np.all(np.isfinite(draws))
    # Each member's draws have q_j's mean and variance (Monte Carlo: within a few standard errors).
    error = 4.0 * np.sqrt(oracle.variance / 4000.0) + 1e-12
    assert np.all(np.abs(draws.mean(axis=1) - oracle.mean) <= error + 1e-6 * np.abs(oracle.mean))
    spread = draws.var(axis=1)
    assert np.all(np.abs(spread - oracle.variance) <= 0.2 * oracle.variance + 1e-12)


def _fit(inference: str):
    codes, covariates, target, classes = _problem(21, samples=160, variants=50)
    return fit_small_n(
        codes=codes, covariates=covariates, target=target, variant_class=classes, log_variance_offset=None,
        draw_count=64, working_bytes=_WORKING_BYTES, seed=0, inference=inference,
    )


def test_the_mean_field_fit_certifies_and_scores():
    """Machinery only (own simulation): the outer loop certifies on the mean-field oracle, and the scoring model
    carries the fit with the resolved effects on top."""
    fit = _fit("mean_field")
    assert fit.certificate.remaining_gain[0] <= 0.5 / 64
    assert fit.certificate.mean_move[0] <= fit.certificate.draw_tolerance[0]
    assert fit.certificate.noise_gain[0] <= 0.5 / 64
    assert np.all(np.isfinite(fit.scoring.coefficients)) and fit.scoring.posterior_draws.shape == (50, 64)
    assert np.all(np.isfinite(fit.scoring.posterior_draws)) and fit.noise_variance > 0.0
    assert fit.profile["sweeps"] > 0
    largest = np.argsort(np.abs(fit.scoring.coefficients))[-3:]
    assert set(largest.tolist()) == {2, 11, 25}


@pytest.mark.slow  # the EP outer loop on 50 columns
def test_the_mean_field_and_ep_fits_agree_on_the_resolved_effects():
    """Machinery only (own simulation): on a problem the data resolve, both inferences put the same effects on top
    and agree on their size."""
    fits = {inference: _fit(inference) for inference in ("mean_field", "ep")}
    strong = [2, 11, 25]
    for name, each in fits.items():
        assert set(np.argsort(np.abs(each.scoring.coefficients))[-3:].tolist()) == set(strong), name
    np.testing.assert_allclose(fits["mean_field"].scoring.coefficients[strong], fits["ep"].scoring.coefficients[strong], rtol=0.25)

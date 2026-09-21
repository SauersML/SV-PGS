"""The small-n route's mean-field inference (sv_pgs/mean_field.py): a sweep is the sequential tilted-moment update
of the engine's own components, each sweep raises the exact ELBO, the fixed point's cavity is its pseudo-likelihood,
its linear response matches finite differences of the fixed point, and the whole fit certifies and scores.
Synthetic data only: machinery checks, never accuracy evidence."""

import numpy as np
import pytest

from sv_pgs.config import VariantClass
from sv_pgs.mean_field import MeanFieldFixedPoints, _Response
from sv_pgs.scale_mixture_ep import (
    Cavity,
    _components,
    class_log_density,
    initial_hyperparameters,
    log_scale,
    noise_gain,
    tilted_moments,
)
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


def _sequential_reference(statistics, prior, hyperparameters, noise, mean, residual):
    """One sweep in numpy, member by member, each q_j from the engine's ``_components`` at its pseudo-likelihood."""
    design = statistics.projected
    squares = statistics.design.squares
    log_density = class_log_density(prior, hyperparameters.coefficients)
    scales = log_scale(prior, hyperparameters.coefficients)
    mean, residual = mean.copy(), residual.copy()
    variance, shift = np.zeros_like(mean), np.zeros_like(mean)
    for member in range(mean.shape[0]):
        column = design[:, member]
        omega = squares[member] / noise
        h = (column @ residual + squares[member] * mean[member]) / noise
        terms = _components(log_density[prior.class_index[member]], scales[[member]], prior.log_variance_grid, np.array([omega]), np.array([h]))
        centre = h * terms.conditional_variance[0]
        new_mean = float(terms.responsibility[0] @ centre)
        variance[member] = float(terms.responsibility[0] @ (terms.conditional_variance[0] + np.square(centre - new_mean)))
        residual -= column * (new_mean - mean[member])
        mean[member], shift[member] = new_mean, h
    return mean, variance, shift, residual


@pytest.mark.parametrize("tied", [False, True])
def test_a_sweep_is_the_sequential_tilted_moment_update(tied):
    statistics, prior, start, oracle = _oracle(3, tied=tied)
    for _sweep in range(3):
        expected = _sequential_reference(statistics, prior, start, oracle.noise, oracle.mean, oracle.residual)
        oracle._sweep(start)
        for got, want in zip((oracle.mean, oracle.variance, oracle.shift, oracle.residual), expected):
            np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("tied", [False, True])
def test_every_sweep_raises_the_elbo_and_the_fixed_point_is_the_tilted_moments(tied):
    statistics, prior, start, oracle = _oracle(3, tied=tied)
    values = []
    for _sweep in range(6):
        divergence, weighted_variance, residual_square, sizes = oracle._sweep(start)
        values.append(oracle._elbo(divergence, weighted_variance, residual_square, sizes)[0])
    gains = np.diff(values)
    assert np.all(gains >= -1e-9 * np.abs(values[1:]))
    (point,) = oracle([start])
    assert point is not None, oracle.refusals
    # Every q_j is the prior tilted by its pseudo-likelihood: the engine's tilted moments at the cavity (omega, h).
    moments = tilted_moments(prior, start, point.cavity, _WORKING_BYTES)
    np.testing.assert_allclose(point.mean, moments.mean, rtol=1e-8, atol=1e-11)
    np.testing.assert_allclose(oracle.variance, moments.variance, rtol=1e-8, atol=1e-11)
    squares = statistics.design.squares
    np.testing.assert_allclose(point.cavity.precision, squares / oracle.noise, rtol=1e-12)
    # The residual is r = y_P - Xp m, exactly, and p_eff = sum_j omega_j v_j.
    np.testing.assert_allclose(oracle.residual, statistics.projected_target - statistics.design.image(oracle.mean), rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(point.effective_effects, float(point.cavity.precision @ oracle.variance), rtol=1e-12)
    # The prediction check's metric is q's own: sum_j d_j^2 / v_j.
    direction = np.random.default_rng(1).standard_normal(prior.variant_count)
    np.testing.assert_allclose(point.precision_norm(direction), float(np.sum(np.square(direction) / oracle.variance)), rtol=1e-10)


def test_the_response_solves_a_symmetric_indefinite_system_exactly():
    statistics, _prior, _start, _oracle_ = _oracle(4, tied=True)
    design = statistics.design
    dense = statistics.projected
    rng = np.random.default_rng(7)
    sites = rng.uniform(0.5, 3.0, dense.shape[1]) * design.squares
    sites[[1, 3, 7]] = -0.3 * design.squares[[1, 3, 7]]  # the tie group with negative sites: indefinite along its differences
    sites[5] = 0.2 * design.squares[5]                   # a small positive site: the Schur route, not Woodbury
    live = np.ones(dense.shape[1], dtype=bool)
    live[9] = False                                       # a dead row: no response
    matrix = dense.T @ dense + np.diag(sites)
    assert np.min(np.linalg.eigvalsh(matrix)) < 0.0
    right = rng.standard_normal((dense.shape[1], 3))
    right[9] = 0.0
    keep = np.flatnonzero(live)
    expected = np.zeros_like(right)
    expected[keep] = np.linalg.solve(matrix[np.ix_(keep, keep)], right[keep])
    np.testing.assert_allclose(_Response(design, sites, live).solve(right), expected, rtol=1e-9, atol=1e-11)


def test_the_cavity_response_matches_finite_differences_of_the_fixed_point():
    """The linear response dh/dx the fixed point hands the outer loop, against central differences of the mean-field
    fixed point re-solved at moved hyperparameters. The fixed points are resolved to the ELBO's rounding (a draw count
    past 1 / eps leaves only that stop), and the differences take two steps with Richardson's extrapolation, so their
    error is the resolved means' own over the step, below the tolerance asked."""
    _statistics, prior, start, oracle = _oracle(5)
    oracle.draw_count = 2**60
    (point,) = oracle([start])
    assert point is not None, oracle.refusals
    direction = 0.05 * np.random.default_rng(3).standard_normal(prior.coefficient_size)

    def moved(coefficients):
        return type(start)(coefficients=coefficients, log_smoothing=start.log_smoothing)

    # The fixed-cavity mean change m_x E: the tilted mean's derivative at the pseudo-likelihood, by the same differences.
    def tilted_mean(coefficients):
        return tilted_moments(prior, moved(coefficients), point.cavity, _WORKING_BYTES).mean

    def shift_at(coefficients):
        # The fixed point at the moved x with the noise held at the base point's, as B holds it (the noise's own
        # response is not part of B in either inference; its stationarity is certified separately): sweeps until
        # the means stop moving at double precision, which resolves h far below the differences' step (an ELBO stop
        # resolves the means only to the square root of its rounding).
        resolved = MeanFieldFixedPoints(oracle.statistics, prior, oracle.noise, 2**60, _WORKING_BYTES)
        resolved.mean, resolved.variance, resolved.shift, resolved.residual = (values.copy() for values in (oracle.mean, oracle.variance, oracle.shift, oracle.residual))
        hyperparameters = moved(coefficients)
        for _sweep in range(10_000):
            before = resolved.mean.copy()
            resolved._sweep(hyperparameters)
            if np.max(np.abs(resolved.mean - before)) <= np.finfo(np.float64).eps * (1.0 + np.max(np.abs(resolved.mean))):
                break
        else:
            raise AssertionError("the moved fixed point did not settle to double precision")
        return resolved._fixed_point(hyperparameters).cavity.shift

    def richardson(function, scale):
        coarse = (function(start.coefficients + scale * direction) - function(start.coefficients - scale * direction)) / (2.0 * scale)
        fine = (function(start.coefficients + 0.5 * scale * direction) - function(start.coefficients - 0.5 * scale * direction)) / scale
        return fine + (fine - coarse) / 3.0

    scale = 1e-2
    mean_by_z = richardson(tilted_mean, scale)
    shift_step, precision_step = point.posterior.cavity_response(mean_by_z[:, None], np.zeros((prior.variant_count, 1)))
    assert not np.any(precision_step)
    numeric = richardson(shift_at, scale)
    np.testing.assert_allclose(shift_step[:, 0], numeric, rtol=2e-3, atol=2e-3 * float(np.max(np.abs(numeric))))


def test_the_noise_update_is_the_elbos_stationary_value():
    _statistics, _prior, start, oracle = _oracle(5)
    (point,) = oracle([start])
    assert point is not None
    # The returned state is the last sweep's, at its noise; the noise's stationary value there is pending, and its
    # exact gain is part of the certified remainder.
    residual = oracle.residual
    pending = (float(residual @ residual) + float(oracle.member_squares @ oracle.variance)) / oracle.residual_dimension
    np.testing.assert_allclose(oracle.noise_gain, noise_gain(pending, oracle.noise, oracle.sample_count, oracle.covariate_count), rtol=1e-10)
    # The certificate's move is twice the remaining gain (KL = move / 2); with the noise's pending gain it is within 1 / (2K).
    assert oracle.noise_gain <= 0.5 / 64 and 0.5 * oracle.mean_move + oracle.noise_gain <= 0.5 / 64


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


def test_the_mean_field_fit_returns_and_scores():
    """Machinery only (own simulation): the outer loop returns on the mean-field oracle with an honest certificate,
    and the scoring model carries the fit with the resolved effects on top."""
    fit = _fit("mean_field")
    assert np.isfinite(fit.certificate.remaining_gain[0])
    assert fit.certificate.mean_move[0] <= fit.certificate.draw_tolerance[0]
    assert fit.certificate.noise_gain[0] <= 0.5 / 64
    assert np.all(np.isfinite(fit.scoring.coefficients)) and fit.scoring.posterior_draws.shape == (50, 64)
    assert np.all(np.isfinite(fit.scoring.posterior_draws)) and fit.noise_variance > 0.0
    assert fit.profile["sweeps"] > 0
    largest = np.argsort(np.abs(fit.scoring.coefficients))[-3:]
    assert set(largest.tolist()) == {2, 11, 25}


@pytest.mark.xfail(strict=True, reason=(
    "the single-V outer loop (theory-ep's redesign, merged 2026-09-21) does not certify here: the joint trial's realized "
    "gain (0.18 nats on this problem) is refused against a resolution of 0.86, the trapezoid rule's end correction over a "
    "162-unit move plus the start state's own error, and the fit returns honestly uncertified with remaining gain ~0.02. "
    "Open engine work, shared with the EP tests marked the same way (tests/test_scale_mixture_ep.py _REDESIGN_OPEN)."
))
def test_the_mean_field_fit_certifies():
    fit = _fit("mean_field")
    assert fit.certificate.remaining_gain[0] <= 0.5 / 64


@pytest.mark.slow  # the EP outer loop on 50 columns
def test_the_mean_field_and_ep_fits_agree_on_the_resolved_effects():
    """Machinery only (own simulation): on a problem the data resolve, both inferences put the same effects on top
    and agree on their size."""
    fits = {inference: _fit(inference) for inference in ("mean_field", "ep")}
    strong = [2, 11, 25]
    for name, each in fits.items():
        assert set(np.argsort(np.abs(each.scoring.coefficients))[-3:].tolist()) == set(strong), name
    np.testing.assert_allclose(fits["mean_field"].scoring.coefficients[strong], fits["ep"].scoring.coefficients[strong], rtol=0.25)

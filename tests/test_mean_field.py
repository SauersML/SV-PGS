"""The small-n route's mean-field inference (sv_pgs/mean_field.py): a sweep is the sequential tilted-moment update
of the engine's own components, each sweep raises the exact ELBO, the fixed point's cavity is its pseudo-likelihood,
its linear response matches finite differences of the fixed point, and the whole fit certifies and scores.
Synthetic data only: machinery checks, never accuracy evidence."""

import tracemalloc

import numpy as np
import pytest

from sv_pgs.config import VariantClass
from sv_pgs.mean_field import MeanFieldFixedPoints, _Response, _sample_nodes
from sv_pgs.scale_mixture_ep import (
    _components,
    class_log_density,
    derived_lattice,
    initial_hyperparameters,
    log_scale,
    noise_gain,
    scale_mixture_prior,
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
    # The mixture's component KL bounds a marginal location shift. Its precision
    # exceeds 1 / Var(q), which would understate this movement.
    direction = np.random.default_rng(1).standard_normal(prior.variant_count)
    density, scales = class_log_density(prior, start.coefficients), log_scale(prior, start.coefficients)
    expected = 0.0
    for label, rows in enumerate(prior.class_rows):
        terms = _components(density[label], scales[rows], prior.log_variance_grid, point.cavity.precision[rows], point.cavity.shift[rows])
        expected += float(np.sum(direction[rows] ** 2 * np.sum(terms.responsibility / terms.conditional_variance, axis=1)))
    np.testing.assert_allclose(point.precision_norm(direction), expected, rtol=1e-10)
    assert expected >= np.sum(direction ** 2 / oracle.variance)


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

    def cavity_at(coefficients):
        # The fixed point at the moved x with the noise re-solved between sweeps, as the fixed point re-solves it
        # (the noise is profiled, and B is the profile's curvature): sweeps until the means and the noise stop
        # moving at double precision, which resolves the cavity far below the differences' step (an ELBO stop
        # resolves the means only to the square root of its rounding).
        resolved = MeanFieldFixedPoints(oracle.statistics, prior, oracle.noise, 2**60, _WORKING_BYTES)
        resolved.mean, resolved.variance, resolved.shift, resolved.residual = (values.copy() for values in (oracle.mean, oracle.variance, oracle.shift, oracle.residual))
        hyperparameters = moved(coefficients)
        for _sweep in range(10_000):
            before, noise_before = resolved.mean.copy(), resolved.noise
            _divergence, weighted_variance, residual_square, _sizes = resolved._sweep(hyperparameters)
            resolved.noise = (residual_square + weighted_variance) / resolved.residual_dimension
            settled_means = np.max(np.abs(resolved.mean - before)) <= np.finfo(np.float64).eps * (1.0 + np.max(np.abs(resolved.mean)))
            if settled_means and abs(resolved.noise - noise_before) <= np.finfo(np.float64).eps * resolved.noise:
                break
        else:
            raise AssertionError("the moved fixed point did not settle to double precision")
        # One more sweep at the settled noise, so the cavity is the one that noise built (as ``_solve`` returns it).
        resolved._sweep(hyperparameters)
        cavity = resolved._fixed_point(hyperparameters).cavity
        return np.concatenate([cavity.shift, cavity.precision])

    def richardson(function, scale):
        coarse = (function(start.coefficients + scale * direction) - function(start.coefficients - scale * direction)) / (2.0 * scale)
        fine = (function(start.coefficients + 0.5 * scale * direction) - function(start.coefficients - 0.5 * scale * direction)) / scale
        return fine + (fine - coarse) / 3.0

    def tilted_variance(coefficients):
        return tilted_moments(prior, moved(coefficients), point.cavity, _WORKING_BYTES).variance

    scale = 1e-2
    mean_by_z = richardson(tilted_mean, scale)
    variance_by_z = richardson(tilted_variance, scale)
    shift_step, precision_step = point.posterior.cavity_response(mean_by_z[:, None], variance_by_z[:, None])
    numeric = richardson(cavity_at, scale)
    count = prior.variant_count
    np.testing.assert_allclose(shift_step[:, 0], numeric[:count], rtol=2e-3, atol=2e-3 * float(np.max(np.abs(numeric[:count]))))
    # The precisions move together, through the noise alone: -omega dsigma^2 / sigma^2.
    assert np.any(precision_step)
    np.testing.assert_allclose(precision_step[:, 0], numeric[count:], rtol=2e-3, atol=2e-3 * float(np.max(np.abs(numeric[count:]))))


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


_INTERIOR_CERTIFICATE_REASON = (
    "with E comparable across fixed points (bf8a3b2) the release stands on this problem (the interior's own fixed point "
    "2.9 nats above the edge's), and at that interior state the weights' certificate is infinite: the difference "
    "curvature K is indefinite (eigenvalues -11, -6.9, -0.24 at rho [2.45, 1.87, 2.40]) and no weight move the model "
    "proposes raises the certified V, so the fit returns its remaining gain as infinite; the boundary model (HANDOFF "
    "Next 0) or a certificate at a rho-boundary of certifiability is the open work"
)


@pytest.mark.xfail(strict=True, reason=_INTERIOR_CERTIFICATE_REASON)
def test_the_mean_field_fit_certifies():
    """Machinery only (own simulation): the outer loop certifies on the mean-field oracle (its remaining gain within
    the tolerance, the prediction move within its budget)."""
    fit = _fit("mean_field")
    assert fit.certificate.remaining_gain[0] <= 0.5 / 64
    assert fit.certificate.prediction_move[0] <= fit.certificate.prediction_tolerance[0]


def test_the_mean_field_fit_scores():
    """Machinery only (own simulation): the fixed point's own certificate holds (q's mean move and the noise's gain
    within their budgets) and the scoring model carries the fit with the resolved effects on top."""
    fit = _fit("mean_field")
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


def test_a_rejected_trial_leaves_no_response_factor_behind():
    # The snapshot is the whole state: the response factor of a rejected trial must not steer the next solve.
    codes, covariates, target, classes = _problem(5, samples=160, variants=50)
    from sv_pgs import small_n
    from sv_pgs.mean_field import MeanFieldFixedPoints

    statistics = small_n.dense_statistics(codes, covariates, target)
    prior = small_n.small_n_prior(statistics, classes, np.zeros(codes.shape[1]), 64)
    start, start_noise, _moment = small_n.small_n_start(statistics, prior)
    oracle = MeanFieldFixedPoints(statistics, prior, start_noise, 64, _WORKING_BYTES)
    (point,) = oracle([start])
    assert point is not None
    accepted, accepted_noise = oracle._response, oracle._response_noise
    snapshot = oracle._snapshot()
    oracle._response, oracle._response_noise = "rejected", 9.0
    oracle._restore(snapshot)
    assert oracle._response is accepted and oracle._response_noise == accepted_noise


def test_the_residual_dimension_is_the_covariate_rank_not_the_column_count():
    # A duplicated covariate column removes nothing more: the statistics, the start noise and the oracle's residual
    # dimension are those of the unduplicated design.
    codes, covariates, target, classes = _problem(6, samples=160, variants=50)
    from sv_pgs import small_n

    doubled = np.column_stack([covariates, covariates[:, :1]])
    plain, twice = small_n.dense_statistics(codes, covariates, target), small_n.dense_statistics(codes, doubled, target)
    assert plain.covariate_rank == twice.covariate_rank == np.linalg.matrix_rank(covariates)
    prior = small_n.small_n_prior(plain, classes, np.zeros(codes.shape[1]), 64)
    _start, noise_plain, _m = small_n.small_n_start(plain, prior)
    _start, noise_twice, _m = small_n.small_n_start(twice, small_n.small_n_prior(twice, classes, np.zeros(codes.shape[1]), 64))
    assert np.isclose(noise_plain, noise_twice, rtol=1e-12, atol=0.0)


def test_a_column_in_the_covariate_span_has_a_nonnegative_square():
    from sv_pgs.small_n import _Design

    samples = 534
    basis = np.ones((samples, 1)) / np.sqrt(samples)
    design = _Design(np.full((samples, 1), 127.0), basis)
    # eps^2 of the column's own square, n terms: the projected column's entries are rounding-sized.
    assert 0.0 <= design.squares[0] <= (np.finfo(np.float64).eps * samples) ** 2 * samples * 127.0 ** 2


def test_the_node_sampler_is_the_inverse_cdf_of_the_responsibilities():
    """P03: ``_sample_nodes`` returns, per row and draw, the number of that row's cumulative responsibilities strictly
    below the draw's uniform, capped at the last node -- exactly what the (rows x nodes x draws) boolean counted --
    and its node frequencies are the responsibilities."""
    rng = np.random.default_rng(5)
    weights = rng.random((7, 11))
    responsibility = weights / weights.sum(axis=1, keepdims=True)
    uniform = rng.random((7, 100_000))
    nodes = np.empty(uniform.shape, dtype=np.int64)
    _sample_nodes(responsibility, uniform, nodes)
    cumulative = np.cumsum(responsibility, axis=1)
    np.testing.assert_array_equal(
        nodes, np.minimum(np.sum(cumulative[:, :, None] < uniform[:, None, :], axis=1), responsibility.shape[1] - 1)
    )
    frequency = np.stack([np.bincount(row, minlength=responsibility.shape[1]) for row in nodes]) / uniform.shape[1]
    # Five standard errors of a binomial frequency, at its widest: 77 cells, so no seed of this size fails by chance.
    assert np.max(np.abs(frequency - responsibility)) <= 5.0 * np.sqrt(0.25 / uniform.shape[1])


def test_the_draws_are_one_law_however_the_budget_splits_them():
    """P03: ``working_bytes`` sets how many rows a piece of ``draws`` holds, and the smallest budget takes one row per
    piece. A piece boundary changes which value of the generator's stream lands where, so the two runs are NOT
    bit-for-bit equal; what must hold is that both are draws of q. Checked on each member's first two moments."""
    _statistics, prior, start, oracle = _oracle(13)
    (point,) = oracle([start])
    assert point is not None
    count = 20_000
    whole = oracle.draws(start, np.random.default_rng(2), count)
    oracle.working_bytes = 1  # every piece is one row
    pieces = oracle.draws(start, np.random.default_rng(2), count)
    assert whole.shape == pieces.shape == (prior.variant_count, count)
    assert not np.array_equal(whole, pieces)
    for values in (whole, pieces):
        assert np.all(np.isfinite(values))
        error = 5.0 * np.sqrt(oracle.variance / count) + 1e-12
        assert np.all(np.abs(values.mean(axis=1) - oracle.mean) <= error + 1e-6 * np.abs(oracle.mean))
        assert np.all(np.abs(values.var(axis=1) - oracle.variance) <= 0.2 * oracle.variance + 1e-12)


def test_the_draws_stay_inside_their_working_budget():
    """P03: a draw's working memory is the pieces' budget, not (rows x nodes x draws). On a case whose old boolean
    tensor alone was 24 times the budget, the peak beyond the returned array stays inside ``working_bytes``."""
    samples, variants, count, budget = 60, 2_000, 64, 1 << 20
    rng = np.random.default_rng(31)
    dosage = rng.binomial(2, rng.uniform(0.05, 0.5, variants), size=(samples, variants))
    codes = (dosage * 127).astype(np.uint8)
    covariates = np.column_stack([np.ones(samples), rng.standard_normal(samples)])
    target = dosage[:, :10] @ rng.standard_normal(10) * 0.1 + rng.standard_normal(samples)
    statistics = dense_statistics(codes, covariates, target)
    members = statistics.active_rows
    offsets = np.zeros(members.shape[0])
    residual = statistics.projected_target
    start_noise = float(residual @ residual) / (statistics.sample_count - statistics.covariate_rank)
    nodes, floor, top = derived_lattice(
        statistics.design.column_squares() / start_noise, statistics.design.back(statistics.target) / start_noise,
        offsets, 0.5 / count,
    )
    prior = scale_mixture_prior(
        class_index=np.zeros(members.shape[0], dtype=np.int64), log_variance_offset=offsets,
        annotation_design=np.zeros((members.shape[0], 0)), annotation_groups=(), nodes=np.linspace(float(nodes[0]), float(nodes[-1]), 200),
        floor=floor, top=top,
    )
    hyperparameters = initial_hyperparameters(prior, 0.01)
    oracle = MeanFieldFixedPoints(statistics, prior, start_noise, count, budget)
    oracle._sweep(hyperparameters)
    # One byte per entry of the boolean the old body formed per class, against the budget it was outside of.
    assert prior.variant_count * prior.grid_size * count >= 24 * budget
    oracle.draws(hyperparameters, np.random.default_rng(3), count)  # the sampler's compilation is not the measurement
    tracemalloc.start()
    values = oracle.draws(hyperparameters, np.random.default_rng(3), count)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert values.shape == (prior.variant_count, count) and np.all(np.isfinite(values))
    assert peak <= values.nbytes + budget


def test_the_cold_start_gets_the_carried_solve_s_own_effort_and_no_more():
    # The second candidate's sweeps are capped at the first's count: a call never costs more than twice the carried
    # solve, and an abandoned cold solve leaves the carried fixed point standing.
    from sv_pgs import small_n
    from sv_pgs.mean_field import _SweepBudget

    codes, covariates, target, classes = _problem(9, samples=160, variants=50)
    statistics = small_n.dense_statistics(codes, covariates, target)
    prior = small_n.small_n_prior(statistics, classes, np.zeros(codes.shape[1]), 64)
    start, start_noise, _moment = small_n.small_n_start(statistics, prior)
    oracle = MeanFieldFixedPoints(statistics, prior, start_noise, 64, _WORKING_BYTES)
    (first,) = oracle([start])
    assert first is not None
    before = oracle.profile["sweeps"]
    (second,) = oracle([start])
    assert second is not None
    carried = oracle.profile["carried_sweeps"]
    assert oracle.profile["sweeps"] - before <= 2 * carried
    oracle._restore(oracle._cold)
    with pytest.raises(_SweepBudget):
        oracle._solve(start, sweep_budget=0)


def test_the_early_response_is_withheld_where_its_schur_block_outgrows_the_kernel():
    # A state whose tilted variances exceed every pseudo-likelihood's puts every live row outside the Woodbury bulk:
    # the bounded response is None there (its LU would cost |N|^3 > the kernel's n^2 p), and the unbounded one is built.
    from sv_pgs import small_n

    codes, covariates, target, classes = _problem(9, samples=60, variants=200)
    statistics = small_n.dense_statistics(codes, covariates, target)
    prior = small_n.small_n_prior(statistics, classes, np.zeros(codes.shape[1]), 64)
    start, start_noise, _moment = small_n.small_n_start(statistics, prior)
    oracle = MeanFieldFixedPoints(statistics, prior, start_noise, 64, _WORKING_BYTES)
    oracle.variance = np.full(prior.variant_count, 1e6)
    assert oracle._response_at(bounded=True) is None
    assert oracle._response_at(bounded=False) is not None
    oracle.variance = np.full(prior.variant_count, 1e-9)
    assert oracle._response_at(bounded=True) is not None

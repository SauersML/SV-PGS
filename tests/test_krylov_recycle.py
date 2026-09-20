"""krylov_recycle: block flexible GCRO-DR against dense solves, and B's linear response through it.

A math check on synthetic operators and the engine's dense EP fixture; never an accuracy claim.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

import sv_pgs.krylov_recycle as krylov
import sv_pgs.scale_mixture_ep as engine
from sv_pgs.krylov_recycle import RecycledSpace, block_gcro_dr
from sv_pgs.scale_mixture_ep import GaussianPosterior, MixtureHyperparameters, initial_hyperparameters, moment_matched_prior_sites, scale_mixture_prior
from tests.test_scale_mixture_ep import _WORKING_BYTES, _data, _dense_ep, _hyperparameters

EPS = np.finfo(np.float64).eps


def _operator(seed: int, size: int, spread: float):
    """A nonsymmetric, diagonalizable A = Q diag(lambda) Q^-1 with real eigenvalues spread over [1 / spread, 1]."""
    generator = np.random.default_rng(seed)
    basis = np.eye(size) + 0.3 * generator.standard_normal((size, size)) / np.sqrt(size)
    values = np.geomspace(1.0 / spread, 1.0, size)
    return basis @ np.diag(values) @ np.linalg.inv(basis)


def _solve(matrix, right, **keywords):
    keywords.setdefault("working_bytes", 64 * right.shape[0] * EPS.dtype.itemsize * right.shape[1])
    keywords.setdefault("application_limit", right.shape[0] * right.shape[1])
    keywords.setdefault("absolute_tolerance", 0.0)
    return block_gcro_dr(lambda values: matrix @ values, right, **keywords)


def test_every_column_meets_its_tolerance_on_a_true_residual() -> None:
    matrix = _operator(1, 80, 50.0)
    right = np.random.default_rng(2).standard_normal((80, 6))
    result = _solve(matrix, right, relative_tolerance=1e-10)
    assert result.residual_norm <= result.target * (1.0 + np.sqrt(EPS))
    np.testing.assert_allclose(result.residual_norm, np.linalg.norm(right - matrix @ result.solution), rtol=1e-12)
    exact = np.linalg.solve(matrix, right)
    assert np.linalg.norm(result.solution - exact) <= 1e-10 * np.linalg.cond(matrix) * np.linalg.norm(exact)


def test_one_block_space_serves_every_column_in_fewer_applications_than_flattened_gmres() -> None:
    size, width = 120, 8
    matrix = _operator(3, size, 200.0)
    right = np.random.default_rng(4).standard_normal((size, width))
    # The flattened GMRES that _total_curvature_columns replaced: one matvec applies A to every column.
    calls = []

    def matvec(vector):
        calls.append(1)
        return (matrix @ vector.reshape(size, width)).ravel()

    restart = 16
    flattened = LinearOperator((size * width, size * width), matvec=matvec, dtype=np.float64)
    _solution, information = gmres(flattened, right.ravel(), rtol=1e-8, atol=0.0, restart=restart, maxiter=size * width)
    assert information == 0
    # The same memory: restart + 1 flattened vectors of size p r.
    result = _solve(matrix, right, relative_tolerance=1e-8, working_bytes=(restart + 1) * size * width * EPS.dtype.itemsize)
    assert result.residual_norm <= 1e-8 * np.linalg.norm(right) * (1.0 + np.sqrt(EPS))
    assert result.applications < len(calls)


def test_a_recycled_space_saves_applications_across_a_sequence_of_nearby_operators() -> None:
    # A few outlying small eigenvalues are what stall a restarted cycle: the kept harmonic Ritz space captures them,
    # and a recycled one hands them to the next, nearby operator instead of rediscovering them.
    size, width = 100, 4
    generator = np.random.default_rng(5)
    basis = np.eye(size) + 0.3 * generator.standard_normal((size, size)) / np.sqrt(size)
    values = np.concatenate([np.geomspace(1e-4, 1e-3, width), np.linspace(0.5, 1.0, size - width)])
    base = basis @ np.diag(values) @ np.linalg.inv(basis)
    perturbation = generator.standard_normal((size, size)) / size
    rights = [generator.standard_normal((size, width)) for _step in range(5)]
    # Memory for short cycles, where restarting costs the most.
    memory = 24 * size * width * EPS.dtype.itemsize
    totals = {}
    for name, recycled in (("fresh", None), ("recycled", RecycledSpace())):
        totals[name] = 0
        for step, right in enumerate(rights):
            matrix = base + 1e-6 * step * perturbation
            result = _solve(matrix, right, relative_tolerance=1e-9, working_bytes=memory, recycled=recycled)
            assert result.residual_norm <= result.target * (1.0 + np.sqrt(EPS))
            np.testing.assert_allclose(result.residual_norm, np.linalg.norm(right - matrix @ result.solution), rtol=1e-8)
            totals[name] += result.applications
    assert totals["recycled"] < totals["fresh"]


def test_an_exact_preconditioner_converges_in_one_block_step() -> None:
    matrix = _operator(7, 60, 1e4)
    inverse = np.linalg.inv(matrix)
    right = np.random.default_rng(8).standard_normal((60, 3))
    result = _solve(matrix, right, relative_tolerance=1e-10, precondition=lambda values: inverse @ values)
    # One application in the cycle, one for the true residual.
    assert result.applications == 2
    assert result.residual_norm <= result.target * (1.0 + np.sqrt(EPS)) + np.linalg.cond(matrix) * 60 * EPS * np.linalg.norm(right)


def test_a_warm_start_is_kept_and_its_residual_recomputed() -> None:
    matrix = _operator(9, 50, 30.0)
    right = np.random.default_rng(10).standard_normal((50, 2))
    exact = np.linalg.solve(matrix, right)
    result = _solve(matrix, right, relative_tolerance=1e-6, start=exact)
    # The start already meets the tolerance: one application measures its true residual and the solve stops.
    assert result.applications == 1 and result.cycles == 0
    np.testing.assert_array_equal(result.solution, exact)


def test_the_local_response_inverts_each_block_of_its_local_matrix(monkeypatch) -> None:
    generator = np.random.default_rng(11)
    size = 24
    blocks = (np.arange(0, 10), np.arange(10, 17), np.arange(17, 24))
    root = generator.standard_normal((size, size))
    covariance = root @ root.T / size + np.eye(size)
    monkeypatch.setattr(krylov, "_prepare", lambda solve, grams: (None, None, None, None))
    monkeypatch.setattr(krylov, "_block_terms", lambda solve, grams, cross, variance, core, block: SimpleNamespace(covariance=covariance[np.ix_(blocks[block], blocks[block])]))
    solve = SimpleNamespace(site_precision=np.ones(size))
    grams = SimpleNamespace(blocks=blocks)
    left, right, diagonal, weight = (generator.uniform(0.5, 1.5, size) for _vector in range(4))
    inverse = krylov.local_response(solve, grams)(left, right, diagonal, weight)
    values = generator.standard_normal((size, 5))
    expected = np.empty_like(values)
    for members in blocks:
        local = covariance[np.ix_(members, members)]
        matrix = np.eye(members.size) - (np.eye(members.size) - weight[members, None] * np.square(local)) @ (
            left[members, None] * local * right[None, members] + np.diag(diagonal[members])
        )
        expected[members] = np.linalg.solve(matrix, values[members])
    np.testing.assert_allclose(inverse(values), expected, rtol=1e-10, atol=1e-12 * np.abs(expected).max())


def _dense_problem():
    """tests/test_scale_mixture_ep's dense EP fixture: 30 AR(1)-correlated variants, 400 samples."""
    generator = np.random.default_rng(45)
    variant_count, sample_count = 30, 400
    latent = generator.standard_normal((sample_count, variant_count))
    for column in range(1, variant_count):
        latent[:, column] = 0.6 * latent[:, column - 1] + 0.8 * latent[:, column]
    genotypes = (latent - latent.mean(axis=0)) / latent.std(axis=0)
    effects = np.where(generator.random(variant_count) < 0.3, generator.normal(0.0, 0.15, variant_count), 0.0)
    targets = genotypes @ effects + generator.standard_normal(sample_count)
    likelihood_precision, linear_term = genotypes.T @ genotypes, genotypes.T @ targets
    class_index, offset, design, groups, _cavity = _data(variant_count, 46)
    nodes = np.linspace(np.log(1e-4), np.log(0.2), 8)
    prior = scale_mixture_prior(
        class_index=class_index, log_variance_offset=offset, annotation_design=design, annotation_groups=groups, nodes=nodes, floor=nodes[0] - 1.0, top=nodes[-1],
    )
    coefficients = _hyperparameters(prior, 47).coefficients * 0.3 + initial_hyperparameters(prior).coefficients
    start = moment_matched_prior_sites(prior, MixtureHyperparameters(coefficients, np.zeros(len(prior.smoothing_blocks))))
    _sites, covariance, cavity = _dense_ep(prior, coefficients, likelihood_precision, linear_term, start)
    return prior, coefficients, cavity, covariance


def test_the_block_krylov_curvature_is_the_exact_linear_response_and_recycles(monkeypatch) -> None:
    prior, coefficients, cavity, covariance = _dense_problem()
    variant_count = covariance.shape[0]
    squared = np.square(covariance)
    posterior = GaussianPosterior(
        solve=lambda right, _relative_tolerance: covariance @ right, variance_jvp=lambda weights: -np.einsum("jk,kr,kj->jr", covariance, weights, covariance)
    )

    def linear_response(left, right, diagonal, weight, right_hand):
        matrix = np.eye(variant_count) - (np.eye(variant_count) - weight[:, None] * squared) @ (left[:, None] * covariance * right[None, :] + np.diag(diagonal))
        return np.linalg.solve(matrix, right_hand)

    exact = engine._total_curvature(prior, coefficients, cavity, replace(posterior, linear_response=linear_response), _WORKING_BYTES, 1e-13)
    blocks = (np.arange(0, 10), np.arange(10, 20), np.arange(20, 30))

    def local(left, right, diagonal, weight):
        factors = []
        for members in blocks:
            block = covariance[np.ix_(members, members)]
            matrix = np.eye(members.size) - (np.eye(members.size) - weight[members, None] * np.square(block)) @ (
                left[members, None] * block * right[None, members] + np.diag(diagonal[members])
            )
            factors.append((members, np.linalg.inv(matrix)))

        def inverse(values):
            result = np.empty_like(values)
            for members, factor in factors:
                result[members] = factor @ values[members]
            return result

        return inverse

    applications = []
    solver = engine.block_gcro_dr

    def counted(*arguments, **keywords):
        result = solver(*arguments, **keywords)
        applications.append(result.applications)
        return result

    monkeypatch.setattr(engine, "block_gcro_dr", counted)
    # Three directions, so the block is narrow against the 30 variants and the solve takes several steps.
    directions = prior.coefficient_map[:, :3]
    columns = lambda value: engine._total_curvature_columns(prior, coefficients, cavity, value, _WORKING_BYTES, 1e-13, directions)
    exact_columns = columns(replace(posterior, linear_response=linear_response))
    plain = columns(posterior)
    plain_cost = sum(applications)
    applications.clear()
    recycled = RecycledSpace()
    preconditioned = replace(posterior, local_response=local, recycled=recycled)
    first = columns(preconditioned)
    first_cost = sum(applications)
    applications.clear()
    second = columns(preconditioned)
    second_cost = sum(applications)
    scale = float(np.max(np.abs(exact_columns)))
    for value in (plain, first, second):
        np.testing.assert_allclose(value, exact_columns, rtol=1e-9, atol=1e-9 * scale)
    print({"plain": plain_cost, "preconditioned": first_cost, "recycled": second_cost})
    assert first_cost < plain_cost
    assert recycled.vectors is not None and second_cost <= first_cost

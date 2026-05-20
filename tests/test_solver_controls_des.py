"""Dembo-Eisenstat-Steihaug behavior tests for solver-control helpers."""

from __future__ import annotations

import numpy as np

from sv_pgs.genotype import as_raw_genotype_matrix
from sv_pgs.mixture_inference import (
    _binary_newton_solver_controls,
    _collapsed_posterior_solver_controls,
)


def _large_standardized():
    standardized = as_raw_genotype_matrix(np.zeros((1, 1), dtype=np.float32)).standardized(
        means=np.zeros(1, dtype=np.float32),
        scales=np.ones(1, dtype=np.float32),
    )
    standardized._n_samples = 20_000
    standardized.variant_indices = np.arange(40_000, dtype=np.int32)
    standardized.means = np.zeros(40_000, dtype=np.float32)
    standardized.scales = np.ones(40_000, dtype=np.float32)
    return standardized


def test_binary_newton_controls_relax_with_small_blend_weight():
    standardized = _large_standardized()
    small_blend_tolerance, _small_blend_iters = _binary_newton_solver_controls(
        standardized,
        solver_tolerance=1e-8,
        maximum_linear_solver_iterations=1024,
        update_blend_weight=0.05,
    )
    large_blend_tolerance, _large_blend_iters = _binary_newton_solver_controls(
        standardized,
        solver_tolerance=1e-8,
        maximum_linear_solver_iterations=1024,
        update_blend_weight=0.95,
    )
    assert small_blend_tolerance > large_blend_tolerance


def test_collapsed_controls_relax_with_small_blend_weight():
    standardized = _large_standardized()
    small_blend_tolerance, _small_blend_iters = _collapsed_posterior_solver_controls(
        standardized,
        solver_tolerance=1e-8,
        maximum_linear_solver_iterations=1024,
        compute_logdet=False,
        compute_beta_variance=False,
        update_blend_weight=0.05,
    )
    large_blend_tolerance, _large_blend_iters = _collapsed_posterior_solver_controls(
        standardized,
        solver_tolerance=1e-8,
        maximum_linear_solver_iterations=1024,
        compute_logdet=False,
        compute_beta_variance=False,
        update_blend_weight=0.95,
    )
    assert small_blend_tolerance > large_blend_tolerance


def test_collapsed_controls_monotone_in_em_iteration_index():
    standardized = _large_standardized()
    total_em_iterations = 10
    tolerances: list[float] = []
    for em_index in range(total_em_iterations):
        tolerance, _iterations = _collapsed_posterior_solver_controls(
            standardized,
            solver_tolerance=1e-8,
            maximum_linear_solver_iterations=1024,
            compute_logdet=False,
            compute_beta_variance=False,
            em_iteration_index=em_index,
            total_em_iterations=total_em_iterations,
        )
        tolerances.append(tolerance)
    for previous, current in zip(tolerances, tolerances[1:]):
        assert current <= previous + 1e-12, (previous, current)
    # The final-iteration tolerance should be strictly smaller than the
    # first-iteration tolerance (forcing schedule tightens with progress).
    assert tolerances[-1] < tolerances[0] - 1e-12, tolerances


def test_solver_controls_iteration_cap_in_bounds():
    standardized = _large_standardized()
    max_iterations = 1024
    blend_weights = [0.01, 0.1, 0.5, 0.9, 0.99]
    for blend in blend_weights:
        _newton_tol, newton_iters = _binary_newton_solver_controls(
            standardized,
            solver_tolerance=1e-8,
            maximum_linear_solver_iterations=max_iterations,
            update_blend_weight=blend,
        )
        assert 16 <= newton_iters <= max_iterations
        _collapsed_tol, collapsed_iters = _collapsed_posterior_solver_controls(
            standardized,
            solver_tolerance=1e-8,
            maximum_linear_solver_iterations=max_iterations,
            compute_logdet=False,
            compute_beta_variance=False,
            update_blend_weight=blend,
        )
        assert 16 <= collapsed_iters <= max_iterations

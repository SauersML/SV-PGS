"""Tests for Anderson(m) acceleration."""
from __future__ import annotations

import numpy as np

from sv_pgs.anderson import AndersonState, anderson_step


def _make_contraction(dimension: int, spectral_radius: float, seed: int):
    rng = np.random.default_rng(seed)
    random_matrix = rng.standard_normal((dimension, dimension))
    orthogonal, _ = np.linalg.qr(random_matrix)
    # Spread eigenvalues in [0.1, spectral_radius] so the matrix is a
    # genuine 2-norm contraction.
    eigenvalues = np.linspace(0.1, spectral_radius, dimension)
    contraction = orthogonal @ np.diag(eigenvalues) @ orthogonal.T
    offset = rng.standard_normal(dimension)
    fixed_point = np.linalg.solve(np.eye(dimension) - contraction, offset)
    return contraction, offset, fixed_point


def _iterate_to_fixed_point(fixed_point_map, initial_iterate, tolerance, max_iterations, memory_depth=None):
    """Map evaluations until the step is below tolerance; Anderson(m) when memory_depth is set."""
    state = None if memory_depth is None else AndersonState(memory_depth=memory_depth)
    current = np.asarray(initial_iterate, dtype=np.float64)
    for iteration in range(1, max_iterations + 1):
        mapped = fixed_point_map(current)
        if np.linalg.norm(mapped - current) < tolerance * max(1.0, float(np.linalg.norm(current))):
            return iteration, mapped
        current = mapped if state is None else anderson_step(state, x_current=current, map_value=mapped)
    return max_iterations, current


def test_affine_contraction_speedup():
    matrix, offset, fixed_point = _make_contraction(20, 0.97, seed=0)
    tolerance = 1e-8

    def fixed_point_map(vector):
        return matrix @ vector + offset

    plain_iterations, _ = _iterate_to_fixed_point(fixed_point_map, np.zeros(offset.shape), tolerance, 2000)
    anderson_iterations, result = _iterate_to_fixed_point(
        fixed_point_map, np.zeros(offset.shape), tolerance, plain_iterations, memory_depth=5
    )
    assert anderson_iterations < plain_iterations
    assert np.allclose(result, fixed_point, atol=1e-6)
    assert anderson_iterations * 3 <= plain_iterations, (
        f"Expected >=3x speedup, got Anderson={anderson_iterations} "
        f"vs plain={plain_iterations}"
    )


def test_quadratic_gradient_descent():
    rng = np.random.default_rng(1)
    dimension = 10
    sqrt_a = rng.standard_normal((dimension, dimension))
    hessian = sqrt_a.T @ sqrt_a + 0.1 * np.eye(dimension)
    linear_term = rng.standard_normal(dimension)
    minimum = np.linalg.solve(hessian, linear_term)
    eigenvalues = np.linalg.eigvalsh(hessian)
    step_size = 1.0 / float(eigenvalues.max())

    def gradient_step(vector):
        return vector - step_size * (hessian @ vector - linear_term)

    tolerance = 1e-8
    plain_iterations, _ = _iterate_to_fixed_point(gradient_step, np.zeros(dimension), tolerance, 5000)
    anderson_iterations, result = _iterate_to_fixed_point(
        gradient_step, np.zeros(dimension), tolerance, plain_iterations, memory_depth=5
    )
    assert anderson_iterations < plain_iterations
    assert np.allclose(result, minimum, atol=1e-5)


def test_first_call_returns_map_value():
    state = AndersonState(memory_depth=5)
    x_current = np.array([1.0, 2.0, 3.0])
    map_value = np.array([1.5, 2.5, 3.5])
    result = anderson_step(state, x_current=x_current, map_value=map_value)
    assert np.array_equal(result, map_value)
    assert len(state.residuals) == 1


def test_ill_conditioned_falls_back():
    state = AndersonState(memory_depth=5)
    x_a = np.array([0.0, 0.0])
    t_a = np.array([1.0, 1.0])
    anderson_step(state, x_current=x_a, map_value=t_a)

    # Construct a second step where the residual exactly matches the prior
    # one -> delta_residuals column is zero -> ill-conditioned.
    x_b = np.array([2.0, 2.0])
    t_b = np.array([3.0, 3.0])  # residual = [1, 1], same as before
    result = anderson_step(state, x_current=x_b, map_value=t_b)
    assert np.all(np.isfinite(result))
    assert np.array_equal(result, t_b)
    assert state.fallback_count == 1

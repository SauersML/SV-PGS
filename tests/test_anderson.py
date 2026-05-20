"""Tests for safeguarded Anderson(m) acceleration."""
from __future__ import annotations

import numpy as np

from sv_pgs.anderson import AndersonState, anderson_step, safeguarded_anderson


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


def _count_plain_iterations(matrix, offset, tolerance, max_iters=2000):
    x = np.zeros(offset.shape)
    for iteration in range(1, max_iters + 1):
        nxt = matrix @ x + offset
        if np.linalg.norm(nxt - x) < tolerance * max(1.0, np.linalg.norm(x)):
            return iteration
        x = nxt
    return max_iters


def test_affine_contraction_speedup():
    matrix, offset, fixed_point = _make_contraction(20, 0.97, seed=0)
    tolerance = 1e-8

    plain_iterations = _count_plain_iterations(matrix, offset, tolerance)

    def fixed_point_map(vector):
        return matrix @ vector + offset

    def objective(vector):
        # Negative residual norm: maximised at the fixed point.
        return -float(np.linalg.norm(matrix @ vector + offset - vector))

    # Generous safeguard slack: residual norm is not a Lyapunov function for
    # Anderson and we want to test the underlying acceleration, not the
    # safeguard's interaction with a non-monotone objective.
    result, history, converged = safeguarded_anderson(
        initial_iterate=np.zeros(offset.shape),
        fixed_point_map=fixed_point_map,
        objective=objective,
        memory_depth=5,
        tolerance=tolerance,
        max_iterations=plain_iterations,
        safeguard_slack=1e6,
    )
    assert converged
    assert np.allclose(result, fixed_point, atol=1e-6)
    anderson_iterations = len(history) - 1
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

    def objective(vector):
        return -0.5 * float(vector @ hessian @ vector) + float(linear_term @ vector)

    tolerance = 1e-8

    # Plain GD iteration count.
    x = np.zeros(dimension)
    plain_iters = 0
    for plain_iters in range(1, 5001):
        nxt = gradient_step(x)
        if np.linalg.norm(nxt - x) < tolerance * max(1.0, np.linalg.norm(x)):
            break
        x = nxt

    result, history, converged = safeguarded_anderson(
        initial_iterate=np.zeros(dimension),
        fixed_point_map=gradient_step,
        objective=objective,
        memory_depth=5,
        tolerance=tolerance,
        max_iterations=plain_iters,
    )
    assert converged
    assert np.allclose(result, minimum, atol=1e-5)
    assert len(history) - 1 < plain_iters


def test_safeguard_rejects_overshoot():
    matrix, offset, fixed_point = _make_contraction(8, 0.8, seed=2)
    rng = np.random.default_rng(3)
    perturbation_log: list[int] = []

    def fixed_point_map(vector):
        base = matrix @ vector + offset
        # Inject an occasional overshoot that violates monotone ascent.
        if rng.random() < 0.05:
            perturbation_log.append(1)
            return base + 50.0 * rng.standard_normal(vector.shape)
        return base

    def objective(vector):
        return -float(np.linalg.norm(vector - fixed_point))

    result, history, converged = safeguarded_anderson(
        initial_iterate=np.zeros(offset.shape),
        fixed_point_map=fixed_point_map,
        objective=objective,
        memory_depth=5,
        tolerance=1e-6,
        max_iterations=2000,
    )
    # Even with the noisy map we should approach the true fixed point
    # because the safeguard rejects bad steps. We allow loose tolerance
    # since the map itself is stochastic.
    assert np.linalg.norm(result - fixed_point) < 1.0
    assert isinstance(converged, bool)
    # Objective must be non-decreasing modulo numerical noise.
    for previous, current in zip(history, history[1:]):
        assert current >= previous - 1e-9


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

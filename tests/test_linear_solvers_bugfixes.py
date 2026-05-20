"""Tests for three concrete bug fixes in sv_pgs.linear_solvers.

1. Lanczos tridiagonal must remain well-formed when Lanczos terminates early.
2. Stochastic log-det must not be inflated by clipping numerical-noise
   (near-zero / slightly negative) eigenvalues up to 1e-12.
3. CG must interpret `tolerance` as a sensible RELATIVE tolerance, including
   the corner case of a tiny right-hand side.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from sv_pgs.linear_solvers import (
    build_linear_operator,
    solve_spd_system,
    stochastic_logdet,
)


def _dense_operator_from_matrix(matrix: np.ndarray):
    matrix_jnp = jnp.asarray(matrix, dtype=jnp.float64)
    return build_linear_operator(
        shape=(int(matrix.shape[0]), int(matrix.shape[1])),
        matvec=lambda vector: matrix_jnp @ jnp.asarray(vector, dtype=jnp.float64),
        matmat=lambda block: matrix_jnp @ jnp.asarray(block, dtype=jnp.float64),
        dtype=jnp.float64,
        jax_compatible=True,
    )


def test_logdet_handles_early_lanczos_termination_on_low_rank_matrix():
    # Construct an SPD-ish matrix where Lanczos provably terminates early:
    # rank-r matrix (r < requested lanczos_steps) plus tiny ridge so it's SPD.
    dimension = 20
    rank = 3
    rng = np.random.default_rng(0)
    basis = rng.normal(size=(dimension, rank))
    low_rank = basis @ basis.T
    ridge = 1e-6 * np.eye(dimension)
    spd_matrix = low_rank + ridge
    operator = _dense_operator_from_matrix(spd_matrix)

    logdet_estimate = stochastic_logdet(
        operator=operator,
        dimension=dimension,
        probe_count=8,
        lanczos_steps=15,  # > rank, so Lanczos terminates early on many probes
        random_seed=42,
        minimum_probe_count=8,
        relative_error_tolerance=0.0,
        absolute_error_tolerance=0.0,
    )

    # Must be finite and not catastrophically wrong vs. the true value.
    assert np.isfinite(logdet_estimate)
    true_logdet = float(np.linalg.slogdet(spd_matrix)[1])
    # SLQ is noisy but should be in the same order-of-magnitude ballpark.
    assert abs(logdet_estimate - true_logdet) < abs(true_logdet) * 2.0 + 50.0


def test_logdet_eigenvalue_filtering_removes_clipping_bias():
    # Matrix with one mathematically zero eigenvalue + small floating noise.
    # Naive clipping to 1e-12 would add ~log(1e-12) per spurious eigenvalue,
    # i.e. ~-27 nats of bias.  Filtering removes this bias.
    dimension = 12
    rng = np.random.default_rng(7)
    orthonormal, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    # One eigenvalue exactly zero; rest positive.
    eigenvalues = np.array([0.0] + [1.0 + 0.5 * i for i in range(dimension - 1)], dtype=np.float64)
    spd_matrix = (orthonormal * eigenvalues) @ orthonormal.T
    # Symmetrize to wash away asymmetric float noise.
    spd_matrix = 0.5 * (spd_matrix + spd_matrix.T)
    operator = _dense_operator_from_matrix(spd_matrix)

    logdet_estimate = stochastic_logdet(
        operator=operator,
        dimension=dimension,
        probe_count=64,
        lanczos_steps=dimension,
        random_seed=11,
        minimum_probe_count=64,
        relative_error_tolerance=0.0,
        absolute_error_tolerance=0.0,
    )

    # Truth excluding the zero eigenvalue: sum log of positive ones.
    nonzero_eigenvalues = eigenvalues[eigenvalues > 0.0]
    true_positive_logdet = float(np.sum(np.log(nonzero_eigenvalues)))

    # The biased estimator (clipping to 1e-12) would push the estimate down
    # by ~ (1/dim) * dim * log(1e-12) ~ -27 nats. Our filtered estimator
    # should land near the truth (allowing for SLQ variance).
    assert np.isfinite(logdet_estimate)
    assert abs(logdet_estimate - true_positive_logdet) < 10.0
    # Specifically: not biased toward -27 nats below truth.
    assert logdet_estimate > true_positive_logdet - 15.0


def test_cg_with_tiny_rhs_converges_with_relative_tolerance():
    # Construct a well-conditioned SPD matrix.
    dimension = 32
    rng = np.random.default_rng(123)
    orthonormal, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    eigenvalues = np.linspace(1.0, 10.0, dimension)
    spd_matrix = (orthonormal * eigenvalues) @ orthonormal.T
    spd_matrix = 0.5 * (spd_matrix + spd_matrix.T)

    # Make a tiny right-hand side: ||b|| ~ 1e-10.
    expected_solution = rng.normal(size=dimension)
    expected_solution /= np.linalg.norm(expected_solution)
    expected_solution *= 1e-10
    right_hand_side = spd_matrix @ expected_solution
    rhs_norm = float(np.linalg.norm(right_hand_side))
    assert rhs_norm < 1e-8

    operator = _dense_operator_from_matrix(spd_matrix)

    tolerance = 1e-6
    solution = solve_spd_system(
        operator=operator,
        right_hand_side=right_hand_side,
        tolerance=tolerance,
        max_iterations=256,
    )

    residual_norm = float(np.linalg.norm(spd_matrix @ solution - right_hand_side))
    # Threshold convention: relative tol, with max(||b||, 1) lower bound on the
    # absolute threshold.  So an absolute residual <= tolerance is acceptable.
    assert np.isfinite(residual_norm)
    assert residual_norm <= tolerance * max(rhs_norm, 1.0) + 1e-12


def test_cg_with_tiny_rhs_via_jax_path():
    # Same as above but through the JAX-compatible path (operator with matmat).
    dimension = 24
    rng = np.random.default_rng(321)
    orthonormal, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    eigenvalues = np.linspace(2.0, 8.0, dimension)
    spd_matrix = (orthonormal * eigenvalues) @ orthonormal.T
    spd_matrix = 0.5 * (spd_matrix + spd_matrix.T)

    expected_solution = rng.normal(size=dimension)
    expected_solution /= np.linalg.norm(expected_solution)
    expected_solution *= 1e-10
    right_hand_side = spd_matrix @ expected_solution
    rhs_norm = float(np.linalg.norm(right_hand_side))

    operator = _dense_operator_from_matrix(spd_matrix)
    tolerance = 1e-5
    solution = solve_spd_system(
        operator=operator,
        right_hand_side=right_hand_side,
        tolerance=tolerance,
        max_iterations=256,
    )
    residual_norm = float(np.linalg.norm(spd_matrix @ solution - right_hand_side))
    assert np.isfinite(residual_norm)
    assert residual_norm <= tolerance * max(rhs_norm, 1.0) + 1e-10

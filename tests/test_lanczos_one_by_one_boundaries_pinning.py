"""Pin: stochastic Lanczos logdet on a 1x1 SPD matrix == log(diagonal).

The smallest possible matrix is 1x1.  Stochastic Lanczos with a single
Lanczos step on a 1x1 operator must produce ``log(a_00)`` exactly (up to
floating-point), regardless of the probe.  Any deviation indicates the
tridiagonal-eigenvalue code path mis-handles the trivial case.
"""
from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

from sv_pgs.linear_solvers import build_linear_operator, stochastic_logdet


def _scalar_operator(value: float):
    def matvec(vector) -> jnp.ndarray:
        return jnp.asarray(
            value * np.asarray(vector, dtype=np.float64), dtype=jnp.float64
        )

    return build_linear_operator(shape=(1, 1), matvec=matvec)


def test_logdet_one_by_one_matches_log_diagonal():
    diagonal_value = 4.0
    operator = _scalar_operator(diagonal_value)
    estimate = stochastic_logdet(
        operator=operator,
        dimension=1,
        probe_count=4,
        lanczos_steps=1,
        random_seed=0,
        minimum_probe_count=1,
        relative_error_tolerance=0.0,
        absolute_error_tolerance=0.0,
        treat_as_rank_deficient=False,
    )
    expected = math.log(diagonal_value)
    np.testing.assert_allclose(estimate, expected, atol=1e-10)


def test_logdet_one_by_one_at_unity_is_zero():
    operator = _scalar_operator(1.0)
    estimate = stochastic_logdet(
        operator=operator,
        dimension=1,
        probe_count=2,
        lanczos_steps=1,
        random_seed=42,
        minimum_probe_count=1,
        relative_error_tolerance=0.0,
        absolute_error_tolerance=0.0,
        treat_as_rank_deficient=False,
    )
    np.testing.assert_allclose(estimate, 0.0, atol=1e-12)

"""Pin: TR-Newton with ``max_iterations=0`` returns the initial point.

The outer loop is ``for k in range(1, max_iterations + 1)`` — when
``max_iterations=0`` the loop body never executes, the gradient norm at
the initial point is computed but no step is taken, and the solver
returns the initial (β, α).  This pins:

* no crash,
* returned ``iterations == 0``,
* returned ``beta``/``alpha`` equal to the initial vectors element-wise,
* the linear predictor matches the initial point's η,
* ``converged`` is True only if the initial gradient is already below
  tolerance (here we set tolerance to ``inf`` to force the trivially-
  converged path).
"""
from __future__ import annotations

import numpy as np

from sv_pgs.tr_newton import trust_region_newton_logistic


def _identity_matvec(matrix: np.ndarray):
    def mv(v: np.ndarray) -> np.ndarray:
        return matrix @ v

    def mvt(u: np.ndarray) -> np.ndarray:
        return matrix.T @ u

    return mv, mvt


def test_max_iterations_zero_returns_initial_point():
    n_samples = 8
    p_dim = 3
    q_dim = 2
    rng = np.random.default_rng(0)
    design = rng.normal(size=(n_samples, p_dim)).astype(np.float64)
    covariates = rng.normal(size=(n_samples, q_dim)).astype(np.float64)
    targets = (rng.uniform(size=n_samples) > 0.5).astype(np.float64)
    beta_init = np.full(p_dim, 0.123, dtype=np.float64)
    alpha_init = np.full(q_dim, -0.456, dtype=np.float64)
    mv, mvt = _identity_matvec(design)
    prior_variances = np.full(p_dim, 0.5, dtype=np.float64)

    result = trust_region_newton_logistic(
        matvec_design=mv,
        matvec_design_transpose=mvt,
        covariate_matrix=covariates,
        targets=targets,
        prior_variances=prior_variances,
        predictor_offset=np.zeros(n_samples, dtype=np.float64),
        beta_init=beta_init,
        alpha_init=alpha_init,
        max_iterations=0,
        gradient_tolerance=float("inf"),  # force "trivially converged"
    )
    np.testing.assert_array_equal(result.beta, beta_init)
    np.testing.assert_array_equal(result.alpha, alpha_init)
    assert result.iterations == 0
    # gradient_tolerance=inf → grad_norm <= tol at exit → converged True.
    assert result.converged
    # Linear predictor matches the initial point.
    expected_eta = design @ beta_init + covariates @ alpha_init
    np.testing.assert_allclose(result.linear_predictor, expected_eta, atol=1e-12)


def test_max_iterations_zero_with_zero_unknowns_returns_trivially():
    """When p_dim == 0 and q_dim == 0 the solver short-circuits regardless
    of ``max_iterations``."""
    n_samples = 4
    targets = np.asarray([0.0, 1.0, 1.0, 0.0], dtype=np.float64)
    result = trust_region_newton_logistic(
        matvec_design=lambda v: np.zeros(n_samples, dtype=np.float64),
        matvec_design_transpose=lambda u: np.zeros(0, dtype=np.float64),
        covariate_matrix=np.zeros((n_samples, 0), dtype=np.float64),
        targets=targets,
        prior_variances=np.zeros(0, dtype=np.float64),
        predictor_offset=np.zeros(n_samples, dtype=np.float64),
        beta_init=np.zeros(0, dtype=np.float64),
        alpha_init=np.zeros(0, dtype=np.float64),
        max_iterations=0,
    )
    assert result.iterations == 0
    assert result.converged
    assert result.beta.shape == (0,)
    assert result.alpha.shape == (0,)

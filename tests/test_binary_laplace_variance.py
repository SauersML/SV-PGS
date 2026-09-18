"""Binary posterior variance is the Laplace covariance at the mode.

With a flat intercept prior and beta ~ N(0, diag(prior_variances)), the Laplace
approximation at the MAP is Cov = (Z' diag(mu (1 - mu)) Z + blockdiag(0, P))^{-1},
Z = [1 | X]; the reported beta variances are the beta block of its diagonal. The
Polya-Gamma weights tanh(|eta|/2) / (2 |eta|) majorize mu (1 - mu), so a variance
built from them understates this by about 2x at 10% prevalence.
"""

from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.genotype import as_raw_genotype_matrix
from sv_pgs.mixture_inference import _binary_posterior_state


def _low_prevalence_problem() -> dict:
    rng = np.random.default_rng(12)
    sample_count, variant_count = 600, 6
    raw = rng.binomial(2, 0.3, size=(sample_count, variant_count)).astype(np.float32)
    standardized = as_raw_genotype_matrix(raw).standardized(
        means=raw.mean(axis=0),
        scales=raw.std(axis=0),
    )
    design = np.asarray(standardized.materialize(), dtype=np.float64)
    beta_true = np.array([0.5, -0.4, 0.3, 0.0, 0.0, 0.2])
    linear_predictor = -2.2 + design @ beta_true
    targets = (rng.uniform(size=sample_count) < 1.0 / (1.0 + np.exp(-linear_predictor))).astype(np.float64)
    return {
        "standardized": standardized,
        "design": design,
        "targets": targets,
        "prior_variances": np.full(variant_count, 0.5),
    }


def _dense_laplace(problem: dict) -> tuple[np.ndarray, np.ndarray]:
    joint_design = np.column_stack([np.ones(problem["design"].shape[0]), problem["design"]])
    precision = np.diag(np.concatenate([[0.0], 1.0 / problem["prior_variances"]]))
    parameters = np.zeros(joint_design.shape[1])
    for _ in range(100):
        probabilities = 1.0 / (1.0 + np.exp(-(joint_design @ parameters)))
        gradient = joint_design.T @ (problem["targets"] - probabilities) - precision @ parameters
        hessian = joint_design.T @ (joint_design * (probabilities * (1.0 - probabilities))[:, None]) + precision
        parameters = parameters + np.linalg.solve(hessian, gradient)
    probabilities = 1.0 / (1.0 + np.exp(-(joint_design @ parameters)))
    hessian = joint_design.T @ (joint_design * (probabilities * (1.0 - probabilities))[:, None]) + precision
    return parameters[1:], np.diag(np.linalg.inv(hessian))[1:]


@pytest.mark.parametrize("use_tr_newton_binary", [False, True])
def test_binary_beta_variance_is_the_laplace_covariance(use_tr_newton_binary: bool) -> None:
    problem = _low_prevalence_problem()
    assert problem["targets"].mean() < 0.15
    map_beta, laplace_variance = _dense_laplace(problem)
    _alpha, beta, beta_variance, _linear_predictor, _objective, _iterations = _binary_posterior_state(
        genotype_matrix=problem["standardized"],
        covariate_matrix=np.ones((problem["design"].shape[0], 1)),
        targets=problem["targets"],
        prior_variances=problem["prior_variances"],
        alpha_init=np.zeros(1),
        beta_init=np.zeros(problem["design"].shape[1]),
        minimum_weight=1e-4,
        max_iterations=500,
        gradient_tolerance=1e-10,
        compute_logdet=False,
        compute_beta_variance=True,
        use_tr_newton_binary=use_tr_newton_binary,
    )
    np.testing.assert_allclose(beta, map_beta, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(beta_variance, laplace_variance, rtol=1e-4)

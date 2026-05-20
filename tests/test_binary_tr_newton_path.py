"""Wire-up tests for the automatic trust-region Newton-CG binary path."""
from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.config import ModelConfig, TraitType
from sv_pgs.mixture_inference import _binary_posterior_state


def _make_binary_problem(
    *,
    n: int = 300,
    p: int = 8,
    seed: int = 7,
) -> dict:
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n, p)).astype(np.float32)
    beta_star = np.zeros(p, dtype=np.float64)
    beta_star[0] = 1.2
    beta_star[1] = -0.8
    beta_star[2] = 0.6
    eta = raw @ beta_star
    probs = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.uniform(size=n) < probs).astype(np.float64)
    return {
        "raw": raw,
        "covariates": np.ones((n, 1), dtype=np.float64),
        "targets": y,
        "beta_star": beta_star,
    }


def _fit_once(problem: dict) -> tuple[np.ndarray, int]:
    config = ModelConfig(
        trait_type=TraitType.BINARY,
        max_outer_iterations=1,
        max_inner_newton_iterations=30,
        newton_gradient_tolerance=1e-6,
    )
    genotype_matrix = problem["raw"]
    p = int(genotype_matrix.shape[1])
    alpha_init = np.zeros(problem["covariates"].shape[1], dtype=np.float64)
    beta_init = np.zeros(p, dtype=np.float64)
    prior_variances = np.full(p, 10.0, dtype=np.float64)
    alpha, beta, _beta_variance, _linear_predictor, _objective, iters = _binary_posterior_state(
        genotype_matrix=genotype_matrix,
        covariate_matrix=problem["covariates"],
        targets=problem["targets"],
        prior_variances=prior_variances,
        alpha_init=alpha_init,
        beta_init=beta_init,
        minimum_weight=config.polya_gamma_minimum_weight,
        max_iterations=config.max_inner_newton_iterations,
        gradient_tolerance=config.newton_gradient_tolerance,
        compute_logdet=False,
        compute_beta_variance=False,
    )
    return np.asarray(beta, dtype=np.float64), int(iters)


def test_binary_fit_uses_tr_newton_automatically() -> None:
    problem = _make_binary_problem()

    beta_tr, iters_tr = _fit_once(problem)

    assert beta_tr.shape == problem["beta_star"].shape
    assert int(iters_tr) >= 1
    correlation = float(np.corrcoef(beta_tr, problem["beta_star"])[0, 1])
    assert correlation > 0.80


def test_tr_newton_surfaces_helper_failure(monkeypatch) -> None:
    """TR-Newton is the selected binary solver; helper failure is surfaced."""
    import sv_pgs.mixture_inference as mi

    monkeypatch.setattr(
        mi,
        "_binary_posterior_state_tr_newton",
        lambda **_kwargs: None,
    )

    problem = _make_binary_problem(n=200, p=6, seed=11)
    with pytest.raises(RuntimeError, match="TR-Newton returned no posterior state"):
        _fit_once(problem)

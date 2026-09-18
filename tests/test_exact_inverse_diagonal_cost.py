"""Posterior variances from the exact variant-space Cholesky factor, at p^3/3 cost.

diag(A^-1) = column sums of (L^-1)^2. Column block s:e of L^-1 is zero above row s
and its trailing part is L[s:, s:]^-1 applied to the identity, so triangular solves
against full-height identity blocks (p^3 flops in total) redo work whose answer is
known to be zero. Both routes must return the exact diagonal without that work.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import solve_triangular

from sv_pgs import mixture_inference
from sv_pgs.genotype import as_raw_genotype_matrix


def _spd_factor(dimension: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    samples = rng.standard_normal((dimension + 40, dimension))
    precision = samples.T @ samples / samples.shape[0] + np.eye(dimension)
    return precision, np.linalg.cholesky(precision)


def test_gpu_route_solves_only_trailing_triangles() -> None:
    dimension = 700  # three 256-column blocks with the mocked (numpy) device
    precision, factor = _spd_factor(dimension)
    solved_work: list[int] = []

    def recording_solve(matrix: np.ndarray, right_hand_side: np.ndarray, lower: bool) -> np.ndarray:
        solved_work.append(matrix.shape[0] * matrix.shape[0] * right_hand_side.shape[1])
        return solve_triangular(matrix, right_hand_side, lower=lower)

    diagonal = mixture_inference._gpu_exact_variant_inverse_diagonal(
        factor,
        solve_triangular_gpu=recording_solve,
        cupy=np,
    )
    np.testing.assert_allclose(diagonal, np.diag(np.linalg.inv(precision)), rtol=1e-10)
    assert sum(solved_work) <= 0.6 * dimension**3


def test_cpu_route_inverts_the_factor_without_an_identity_solve(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(5)
    sample_count, variant_count = 60, 25
    genotypes = rng.binomial(2, 0.4, size=(sample_count, variant_count)).astype(np.float32)
    standardized = as_raw_genotype_matrix(genotypes).standardized(
        means=genotypes.mean(axis=0),
        scales=genotypes.std(axis=0),
    )
    design = np.asarray(standardized.materialize(), dtype=np.float64)
    prior_precision = rng.uniform(0.5, 4.0, size=variant_count)
    inverse_noise = np.full(sample_count, 1.0 / 0.7)

    def forbid_identity_solve(matrix: np.ndarray, right_hand_side: np.ndarray, **kwargs: object) -> np.ndarray:
        if np.asarray(right_hand_side).shape == (variant_count, variant_count):
            raise AssertionError("variance route solved the factor against a full identity")
        return solve_triangular(matrix, right_hand_side, **kwargs)

    monkeypatch.setattr(mixture_inference, "solve_triangular", forbid_identity_solve)
    _beta, _predictor, beta_variance, _logdet = mixture_inference._solve_restricted_exact_variant_space(
        genotype_matrix=standardized,
        covariate_matrix=np.zeros((sample_count, 0)),
        targets=rng.standard_normal(sample_count),
        prior_precision=prior_precision,
        inverse_diagonal_noise=inverse_noise,
        covariate_precision_cholesky=np.zeros((0, 0)),
        posterior_variance_batch_size=16,
        compute_beta_variance=True,
        compute_logdet=False,
        warm_start=None,
    )
    precision = design.T @ (design * inverse_noise[:, None]) + np.diag(prior_precision + 1e-8)
    np.testing.assert_allclose(beta_variance, np.diag(np.linalg.inv(precision)), rtol=1e-8)

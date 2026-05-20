"""Tests for the SRHT preconditioner.

Covers:
1. Spectral bound of the sketched Gram matrix vs. the true Gram matrix.
2. PCG convergence rate on an ill-conditioned synthetic problem.
3. Round-trip exactness of the cached Cholesky factor.
4. Padding correctness when ``n`` is not a power of two.
5. Reproducibility for fixed seeds.
"""
from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.sketched_preconditioner import (
    SRHTPreconditioner,
    apply_srht_preconditioner,
    build_srht_preconditioner,
    srht_apply,
)


def _build_true_matrix(
    design: np.ndarray, weights: np.ndarray, prior_variances: np.ndarray
) -> np.ndarray:
    """A = X^T W X + diag(1/tau^2)."""
    weighted = design * weights[:, None]
    matrix = design.T @ weighted
    matrix.flat[:: matrix.shape[0] + 1] += 1.0 / prior_variances
    return 0.5 * (matrix + matrix.T)


def _symmetric_sqrt_inv(matrix: np.ndarray) -> np.ndarray:
    """Return M^{-1/2} via eigendecomposition (M is SPD)."""
    eigvals, eigvecs = np.linalg.eigh(matrix)
    return (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T


def test_spectral_bound_holds_across_seeds() -> None:
    rng = np.random.default_rng(20260520)
    n_samples, n_variants = 4096, 256
    design = rng.standard_normal((n_samples, n_variants))
    weights = rng.uniform(0.5, 1.5, size=n_samples)
    prior_variances = rng.uniform(0.5, 2.0, size=n_variants)

    true_matrix = _build_true_matrix(design, weights, prior_variances)

    epsilon = 0.2
    failures = 0
    for seed in range(5):
        pre = build_srht_preconditioner(
            design=design,
            weights=weights,
            prior_variances=prior_variances,
            sketch_rows=4096,  # full sketch dim for tight bounds at n=4096
            relative_error_target=epsilon,
            random_seed=seed,
        )
        sketched_gram = pre.cholesky_factor @ pre.cholesky_factor.T
        m_inv_half = _symmetric_sqrt_inv(sketched_gram)
        whitened = m_inv_half @ true_matrix @ m_inv_half
        eigvals = np.linalg.eigvalsh(whitened)
        if not (eigvals.min() >= 1.0 - epsilon and eigvals.max() <= 1.0 + epsilon):
            failures += 1

    assert failures <= 1, (
        f"Spectral bound exceeded for {failures}/5 seeds at eps={epsilon}"
    )


def _pcg(
    matvec,
    rhs: np.ndarray,
    preconditioner_solve,
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray, int, float]:
    """Tiny preconditioned conjugate gradient. Returns (x, iters, rel_res)."""
    x = np.zeros_like(rhs)
    r = rhs - matvec(x)
    z = preconditioner_solve(r)
    p = z.copy()
    rz = float(r @ z)
    rhs_norm = float(np.linalg.norm(rhs))
    if rhs_norm == 0.0:
        return x, 0, 0.0
    for iteration in range(1, max_iter + 1):
        ap = matvec(p)
        alpha = rz / float(p @ ap)
        x = x + alpha * p
        r = r - alpha * ap
        rel_res = float(np.linalg.norm(r)) / rhs_norm
        if rel_res < tol:
            return x, iteration, rel_res
        z = preconditioner_solve(r)
        rz_new = float(r @ z)
        beta = rz_new / rz
        p = z + beta * p
        rz = rz_new
    return x, max_iter, rel_res


def test_pcg_converges_faster_with_srht_preconditioner() -> None:
    rng = np.random.default_rng(7)
    n_samples, n_variants = 2048, 192
    design = rng.standard_normal((n_samples, n_variants))
    # Inject a wide spectrum to push cond(A) ~ 1e4.
    scale = np.geomspace(1.0, 1e2, num=n_variants)
    design = design * scale[None, :]
    weights = np.ones(n_samples)
    prior_variances = np.full(n_variants, 1e2)

    true_matrix = _build_true_matrix(design, weights, prior_variances)
    eigvals = np.linalg.eigvalsh(true_matrix)
    condition_number = eigvals.max() / eigvals.min()
    assert 1e2 < condition_number < 1e8, condition_number

    rhs = rng.standard_normal(n_variants)

    def matvec(vec: np.ndarray) -> np.ndarray:
        return true_matrix @ vec

    pre = build_srht_preconditioner(
        design=design,
        weights=weights,
        prior_variances=prior_variances,
        random_seed=11,
    )

    def precond(vec: np.ndarray) -> np.ndarray:
        return apply_srht_preconditioner(pre, vec)

    _, iters_pre, rel_pre = _pcg(matvec, rhs, precond, tol=1e-8, max_iter=200)
    _, iters_plain, rel_plain = _pcg(
        matvec, rhs, lambda v: v, tol=1e-8, max_iter=400
    )

    assert iters_pre <= 30, (iters_pre, rel_pre)
    assert iters_plain >= 100, (iters_plain, rel_plain)


def test_cholesky_factor_round_trip_is_identity() -> None:
    rng = np.random.default_rng(3)
    n_samples, n_variants = 512, 64
    design = rng.standard_normal((n_samples, n_variants))
    weights = rng.uniform(0.5, 1.5, size=n_samples)
    prior_variances = rng.uniform(0.5, 2.0, size=n_variants)

    pre = build_srht_preconditioner(
        design=design,
        weights=weights,
        prior_variances=prior_variances,
        random_seed=42,
    )

    sketched_gram = pre.cholesky_factor @ pre.cholesky_factor.T

    vector = rng.standard_normal(n_variants)
    # M v = (gram)^{-1} v ; then gram @ that = v.
    applied = apply_srht_preconditioner(pre, vector)
    reconstructed = sketched_gram @ applied
    np.testing.assert_allclose(reconstructed, vector, atol=1e-9, rtol=1e-9)


def test_handles_n_not_power_of_two() -> None:
    rng = np.random.default_rng(0)
    n_samples, n_variants = 1000, 48
    design = rng.standard_normal((n_samples, n_variants))
    weights = rng.uniform(0.5, 1.5, size=n_samples)
    prior_variances = rng.uniform(0.5, 2.0, size=n_variants)

    pre = build_srht_preconditioner(
        design=design,
        weights=weights,
        prior_variances=prior_variances,
        random_seed=5,
    )
    assert pre.cholesky_factor.shape == (n_variants, n_variants)
    vector = rng.standard_normal(n_variants)
    out = apply_srht_preconditioner(pre, vector)
    assert out.shape == (n_variants,)
    assert np.all(np.isfinite(out))

    # Bare srht_apply: ensure pad path also produces sensible output.
    sketched = srht_apply(np.random.default_rng(1), weights, design, sketch_rows=512)
    assert sketched.shape == (512, n_variants)
    assert np.all(np.isfinite(sketched))


def test_reproducibility_with_same_seed() -> None:
    rng = np.random.default_rng(0)
    n_samples, n_variants = 256, 32
    design = rng.standard_normal((n_samples, n_variants))
    weights = rng.uniform(0.5, 1.5, size=n_samples)
    prior_variances = rng.uniform(0.5, 2.0, size=n_variants)

    pre_a = build_srht_preconditioner(
        design=design,
        weights=weights,
        prior_variances=prior_variances,
        random_seed=123,
    )
    pre_b = build_srht_preconditioner(
        design=design,
        weights=weights,
        prior_variances=prior_variances,
        random_seed=123,
    )
    np.testing.assert_array_equal(pre_a.cholesky_factor, pre_b.cholesky_factor)
    assert pre_a.sketch_rows == pre_b.sketch_rows

    a = srht_apply(np.random.default_rng(9), weights, design, sketch_rows=128)
    b = srht_apply(np.random.default_rng(9), weights, design, sketch_rows=128)
    np.testing.assert_array_equal(a, b)


def test_srht_preconditioner_is_dataclass() -> None:
    pre = SRHTPreconditioner(
        cholesky_factor=np.eye(2),
        sketch_rows=4,
        relative_error=0.1,
    )
    assert pre.sketch_rows == 4
    assert pre.relative_error == pytest.approx(0.1)

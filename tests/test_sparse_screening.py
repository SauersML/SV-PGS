"""Parity tests for sparse marginal-z screening vs the dense formulation."""

from __future__ import annotations

import numpy as np

from sv_pgs.sparse_screening import compute_sparse_marginal_z


def _dense_marginal_z(
    G: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
    residual: np.ndarray,
) -> np.ndarray:
    """Reference dense implementation: z_j = (x_j^T r) / sqrt(x_j^T x_j * var(r))."""
    X = (G.astype(np.float64) - means[None, :]) / scales[None, :]
    var_r = residual.var()
    xtr = X.T @ residual
    xtx = np.einsum("ij,ij->j", X, X)
    denom = np.sqrt(xtx * var_r)
    z = np.zeros_like(xtr)
    nonzero = denom > 0
    z[nonzero] = xtr[nonzero] / denom[nonzero]
    return z


def _build_sparse_inputs(G: np.ndarray):
    n_samples, n_variants = G.shape
    idx_list = []
    gen_list = []
    for j in range(n_variants):
        carriers = np.nonzero(G[:, j])[0].astype(np.int32)
        idx_list.append(carriers)
        gen_list.append(G[carriers, j].astype(np.int8))
    return idx_list, gen_list


def test_sparse_matches_dense_small_synthetic():
    rng = np.random.default_rng(0)
    n_samples = 200
    n_variants = 25

    # Sparse rare-variant pattern: ~1-3% carriers per variant.
    G = np.zeros((n_samples, n_variants), dtype=np.int8)
    for j in range(n_variants):
        n_carriers = int(rng.integers(1, 6))
        carriers = rng.choice(n_samples, size=n_carriers, replace=False)
        G[carriers, j] = rng.integers(1, 3, size=n_carriers).astype(np.int8)

    means = G.mean(axis=0).astype(np.float64)
    # Use the raw-genotype sd as the scale (matches the standardized x_j
    # convention in the docstring). Guard against zero-variance columns.
    sds = G.std(axis=0).astype(np.float64)
    sds[sds == 0] = 1.0
    scales = sds

    residual = rng.standard_normal(n_samples)

    z_dense = _dense_marginal_z(G, means, scales, residual)

    idx_list, gen_list = _build_sparse_inputs(G)
    z_sparse = compute_sparse_marginal_z(
        idx_list, gen_list, means, scales, residual, n_samples=n_samples
    )

    np.testing.assert_allclose(z_sparse, z_dense, atol=1e-6, rtol=1e-6)


def test_sparse_handles_variant_with_no_carriers():
    n_samples = 50
    G = np.zeros((n_samples, 3), dtype=np.int8)
    G[5, 1] = 1
    G[10, 1] = 2
    G[20, 2] = 1

    means = G.mean(axis=0).astype(np.float64)
    sds = G.std(axis=0).astype(np.float64)
    # Variant 0 has zero variance; keep scale at 1 to avoid division by zero.
    sds[sds == 0] = 1.0
    scales = sds

    rng = np.random.default_rng(42)
    residual = rng.standard_normal(n_samples)

    z_dense = _dense_marginal_z(G, means, scales, residual)
    idx_list, gen_list = _build_sparse_inputs(G)
    z_sparse = compute_sparse_marginal_z(
        idx_list, gen_list, means, scales, residual, n_samples=n_samples
    )

    np.testing.assert_allclose(z_sparse, z_dense, atol=1e-6, rtol=1e-6)


def test_sparse_matches_dense_homozygous_carriers():
    """Carriers can be 1 or 2; verify x_j^T x_j accounts for g^2, not just g."""
    rng = np.random.default_rng(7)
    n_samples = 100
    n_variants = 10
    G = np.zeros((n_samples, n_variants), dtype=np.int8)
    for j in range(n_variants):
        # Force at least one homozygous carrier per variant.
        c1 = rng.choice(n_samples, size=3, replace=False)
        G[c1[0], j] = 2
        G[c1[1], j] = 1
        G[c1[2], j] = 2

    means = G.mean(axis=0).astype(np.float64)
    sds = G.std(axis=0).astype(np.float64)
    sds[sds == 0] = 1.0
    scales = sds
    residual = rng.standard_normal(n_samples)

    z_dense = _dense_marginal_z(G, means, scales, residual)
    idx_list, gen_list = _build_sparse_inputs(G)
    z_sparse = compute_sparse_marginal_z(
        idx_list, gen_list, means, scales, residual, n_samples=n_samples
    )

    np.testing.assert_allclose(z_sparse, z_dense, atol=1e-6, rtol=1e-6)

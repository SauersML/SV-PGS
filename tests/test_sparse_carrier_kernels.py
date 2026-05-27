"""Tests for the sparse standardized carrier matvec / rmatvec kernels.

These tests run on CPU using numpy arrays (the kernel module dispatches its
xp by the runtime array type, so numpy arrays exercise the exact same code
path as cupy arrays would, modulo the scatter-add backend).
"""

from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.sparse_carrier_kernels import sparse_matvec, sparse_rmatvec


def _build_synthetic(
    rng: np.random.Generator,
    n_samples: int,
    n_variants: int,
    carrier_fraction: float = 0.1,
    dtype: np.dtype = np.float64,
):
    raw = np.zeros((n_samples, n_variants), dtype=dtype)
    carriers: list[np.ndarray] = []
    carrier_geno: list[np.ndarray] = []
    for j in range(n_variants):
        k = max(0, int(rng.binomial(n_samples, carrier_fraction)))
        if k > 0:
            idx = rng.choice(n_samples, size=k, replace=False).astype(np.int64)
            idx.sort()
            geno = rng.choice([1.0, 2.0], size=k).astype(dtype)
            raw[idx, j] = geno
        else:
            idx = np.empty(0, dtype=np.int64)
            geno = np.empty(0, dtype=dtype)
        carriers.append(idx)
        carrier_geno.append(geno)

    means = raw.mean(axis=0).astype(dtype)
    # Use a non-trivial scale (sample std with floor), avoiding zeros.
    scales = raw.std(axis=0).astype(dtype)
    scales = np.where(scales < 1e-3, np.asarray(1.0, dtype=dtype), scales)
    standardized = (raw - means) / scales
    return raw, carriers, carrier_geno, means, scales, standardized


def test_sparse_matvec_matches_dense_reference():
    rng = np.random.default_rng(0)
    n_samples, n_variants = 64, 32
    raw, carriers, carrier_geno, means, scales, standardized = _build_synthetic(
        rng, n_samples, n_variants
    )
    x = rng.standard_normal(n_variants).astype(np.float64)

    y_dense = standardized @ x
    y_sparse = sparse_matvec(
        carriers, carrier_geno, means, scales, x, n_samples, n_variants
    )

    np.testing.assert_allclose(y_sparse, y_dense, rtol=1e-10, atol=1e-10)


def test_sparse_rmatvec_matches_dense_reference():
    rng = np.random.default_rng(1)
    n_samples, n_variants = 48, 24
    raw, carriers, carrier_geno, means, scales, standardized = _build_synthetic(
        rng, n_samples, n_variants
    )
    y = rng.standard_normal(n_samples).astype(np.float64)

    z_dense = standardized.T @ y
    z_sparse = sparse_rmatvec(
        carriers, carrier_geno, means, scales, y, n_samples, n_variants
    )

    np.testing.assert_allclose(z_sparse, z_dense, rtol=1e-10, atol=1e-10)


def test_sparse_matvec_zero_carrier_variant_gives_constant_column():
    """A variant with zero carriers contributes -mean/scale * x[j] to every sample."""

    n_samples, n_variants = 16, 3
    carriers = [np.empty(0, dtype=np.int64) for _ in range(n_variants)]
    carrier_geno = [np.empty(0, dtype=np.float64) for _ in range(n_variants)]
    means = np.array([0.4, 0.0, 0.7], dtype=np.float64)
    scales = np.array([1.0, 2.0, 0.5], dtype=np.float64)
    x = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    y = sparse_matvec(
        carriers, carrier_geno, means, scales, x, n_samples, n_variants
    )

    expected_const = -means[0] / scales[0] * x[0]
    expected = np.full(n_samples, expected_const, dtype=np.float64)
    np.testing.assert_allclose(y, expected, rtol=1e-12, atol=1e-12)


def test_sparse_matvec_linearity():
    """matvec is linear in x; sparse impl must respect that."""

    rng = np.random.default_rng(2)
    n_samples, n_variants = 20, 10
    _, carriers, carrier_geno, means, scales, _ = _build_synthetic(
        rng, n_samples, n_variants
    )
    x1 = rng.standard_normal(n_variants)
    x2 = rng.standard_normal(n_variants)

    y1 = sparse_matvec(carriers, carrier_geno, means, scales, x1, n_samples, n_variants)
    y2 = sparse_matvec(carriers, carrier_geno, means, scales, x2, n_samples, n_variants)
    y_sum = sparse_matvec(
        carriers, carrier_geno, means, scales, x1 + x2, n_samples, n_variants
    )
    np.testing.assert_allclose(y1 + y2, y_sum, rtol=1e-12, atol=1e-12)


def test_sparse_matvec_rejects_length_mismatch():
    with pytest.raises(ValueError):
        sparse_matvec(
            [np.empty(0, np.int64)],
            [np.empty(0, np.float64)],
            np.zeros(2),
            np.ones(2),
            np.zeros(2),
            4,
            2,
        )


def test_sparse_rmatvec_rejects_length_mismatch():
    with pytest.raises(ValueError):
        sparse_rmatvec(
            [np.empty(0, np.int64)],
            [np.empty(0, np.float64)],
            np.zeros(2),
            np.ones(2),
            np.zeros(4),
            4,
            2,
        )

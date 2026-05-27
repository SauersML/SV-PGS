"""Numpy-backend correctness tests for the hybrid genotype matrix scaffold.

No GPU required: ``GpuSparseCarrierMatrix`` and ``BioHybridGenotypeMatrix``
accept an ``xp_backend`` parameter so the same code paths run against NumPy.
"""

from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.hybrid_matrix import (
    BioHybridGenotypeMatrix,
    GpuSparseCarrierMatrix,
    default_carrier_threshold,
)


class _NumpyDenseStub:
    """Minimal stand-in for ``BitpackedDeviceMatrix`` (numpy-backed).

    Exposes the same ``shape`` / ``matvec(x)`` / ``rmatvec(y)`` surface the
    hybrid matrix needs. Operates on a pre-standardized dense matrix so the
    test math is direct.
    """

    def __init__(self, standardized_dense: np.ndarray) -> None:
        self._z = np.asarray(standardized_dense, dtype=np.float32)

    @property
    def shape(self) -> tuple[int, int]:
        return int(self._z.shape[0]), int(self._z.shape[1])

    def matvec(self, x: np.ndarray) -> np.ndarray:
        return (self._z @ np.asarray(x, dtype=np.float32)).astype(np.float32)

    def rmatvec(self, y: np.ndarray) -> np.ndarray:
        return (self._z.T @ np.asarray(y, dtype=np.float32)).astype(np.float32)


def _standardize_columns(raw: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((raw.astype(np.float32) - mean.astype(np.float32)) / scale.astype(np.float32)).astype(
        np.float32
    )


def test_default_carrier_threshold_divisor() -> None:
    assert default_carrier_threshold(640) == 10
    assert default_carrier_threshold(1) == 1  # floor
    assert default_carrier_threshold(256000) == 4000


def test_sparse_matvec_matches_dense_reference() -> None:
    rng = np.random.default_rng(0)
    n_samples = 200
    n_variants = 8
    # Rare-variant pattern: 3-7 carriers per variant.
    raw = np.zeros((n_samples, n_variants), dtype=np.int8)
    for v in range(n_variants):
        n_carriers = int(rng.integers(3, 8))
        idx = rng.choice(n_samples, size=n_carriers, replace=False)
        raw[idx, v] = rng.integers(1, 3, size=n_carriers, dtype=np.int8)
    mean = raw.astype(np.float32).mean(axis=0)
    scale = raw.astype(np.float32).std(axis=0) + 1e-3  # avoid /0 for emptyish cols

    z = _standardize_columns(raw, mean, scale)

    sparse = GpuSparseCarrierMatrix.from_dense(raw, mean, scale, xp_backend=np)

    x = rng.standard_normal(n_variants).astype(np.float32)
    y = rng.standard_normal(n_samples).astype(np.float32)

    matvec_got = np.asarray(sparse.matvec(x))
    rmatvec_got = np.asarray(sparse.rmatvec(y))

    np.testing.assert_allclose(matvec_got, z @ x, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(rmatvec_got, z.T @ y, rtol=1e-4, atol=1e-4)


def test_sparse_materialize_matches_reference() -> None:
    raw = np.array(
        [
            [0, 0, 1],
            [1, 0, 0],
            [0, 2, 1],
            [0, 0, 0],
        ],
        dtype=np.int8,
    )
    mean = np.array([0.25, 0.5, 0.5], dtype=np.float32)
    scale = np.array([0.5, 1.0, 0.5], dtype=np.float32)
    sparse = GpuSparseCarrierMatrix.from_dense(raw, mean, scale, xp_backend=np)
    got = sparse.materialize()
    expected = _standardize_columns(raw, mean, scale)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


def test_hybrid_matvec_rmatvec_match_full_dense_reference() -> None:
    """The key correctness invariant: routing variants by carrier count must
    leave matvec / rmatvec identical to a single dense reference."""
    rng = np.random.default_rng(7)
    n_samples = 300
    n_global = 20
    threshold = default_carrier_threshold(n_samples)  # = 4

    # Construct a global raw matrix with a mix of common + rare variants.
    raw_global = np.zeros((n_samples, n_global), dtype=np.int8)
    for v in range(n_global):
        if v % 3 == 0:
            # Rare: few carriers (well below threshold).
            n_carriers = int(rng.integers(1, threshold + 1))
        else:
            # Common: many carriers.
            n_carriers = int(rng.integers(n_samples // 4, n_samples // 2))
        idx = rng.choice(n_samples, size=n_carriers, replace=False)
        raw_global[idx, v] = rng.integers(1, 3, size=n_carriers, dtype=np.int8)

    mean_global = raw_global.astype(np.float32).mean(axis=0)
    scale_global = raw_global.astype(np.float32).std(axis=0) + 1e-3
    z_global = _standardize_columns(raw_global, mean_global, scale_global)

    # Route by carrier count.
    carrier_counts = (raw_global != 0).sum(axis=0)
    dense_mask = carrier_counts > threshold
    sparse_mask = ~dense_mask
    dense_to_global = np.flatnonzero(dense_mask).astype(np.int64)
    sparse_to_global = np.flatnonzero(sparse_mask).astype(np.int64)

    # Verify the partition was meaningful (both sides non-empty).
    assert dense_to_global.size > 0
    assert sparse_to_global.size > 0

    # Build the dense side as a pre-standardized stub.
    dense_stub = _NumpyDenseStub(z_global[:, dense_to_global])
    # Build the sparse side from the raw (unstandardized) rare-variant block.
    sparse = GpuSparseCarrierMatrix.from_dense(
        raw_global[:, sparse_to_global],
        mean_global[sparse_to_global],
        scale_global[sparse_to_global],
        xp_backend=np,
    )

    hybrid = BioHybridGenotypeMatrix(
        dense=dense_stub,
        sparse=sparse,
        dense_to_global=dense_to_global,
        sparse_to_global=sparse_to_global,
        xp_backend=np,
    )

    assert hybrid.shape == (n_samples, n_global)

    # matvec invariance: X @ x identical to the full dense reference.
    x = rng.standard_normal(n_global).astype(np.float32)
    y = rng.standard_normal(n_samples).astype(np.float32)

    got_matvec = np.asarray(hybrid.matvec(x))
    np.testing.assert_allclose(got_matvec, z_global @ x, rtol=1e-4, atol=1e-4)

    # rmatvec invariance: X.T @ y identical to the full dense reference,
    # with entries placed back at *global* variant positions.
    got_rmatvec = np.asarray(hybrid.rmatvec(y))
    np.testing.assert_allclose(got_rmatvec, z_global.T @ y, rtol=1e-4, atol=1e-4)


def test_hybrid_round_trip_matvec_rmatvec_identity() -> None:
    """The split is invariant under matvec/rmatvec: ``X.T @ X @ x`` equals
    the corresponding dense-reference quadratic form."""
    rng = np.random.default_rng(11)
    n_samples = 120
    n_global = 12
    threshold = default_carrier_threshold(n_samples)

    raw = np.zeros((n_samples, n_global), dtype=np.int8)
    for v in range(n_global):
        n_carriers = int(rng.integers(1, n_samples // 2))
        idx = rng.choice(n_samples, size=n_carriers, replace=False)
        raw[idx, v] = rng.integers(1, 3, size=n_carriers, dtype=np.int8)
    mean = raw.astype(np.float32).mean(axis=0)
    scale = raw.astype(np.float32).std(axis=0) + 1e-3
    z = _standardize_columns(raw, mean, scale)

    counts = (raw != 0).sum(axis=0)
    d2g = np.flatnonzero(counts > threshold).astype(np.int64)
    s2g = np.flatnonzero(counts <= threshold).astype(np.int64)

    hybrid = BioHybridGenotypeMatrix(
        dense=_NumpyDenseStub(z[:, d2g]),
        sparse=GpuSparseCarrierMatrix.from_dense(
            raw[:, s2g], mean[s2g], scale[s2g], xp_backend=np
        ),
        dense_to_global=d2g,
        sparse_to_global=s2g,
        xp_backend=np,
    )

    x = rng.standard_normal(n_global).astype(np.float32)
    got = np.asarray(hybrid.rmatvec(hybrid.matvec(x)))
    expected = z.T @ (z @ x)
    np.testing.assert_allclose(got, expected, rtol=1e-3, atol=1e-3)


def test_hybrid_overlap_detection() -> None:
    raw = np.zeros((10, 4), dtype=np.int8)
    raw[0, 0] = 1
    raw[1, 1] = 1
    raw[2, 2] = 1
    raw[3, 3] = 1
    mean = raw.astype(np.float32).mean(axis=0)
    scale = raw.astype(np.float32).std(axis=0) + 1e-3
    z = _standardize_columns(raw, mean, scale)
    with pytest.raises(ValueError, match="overlap"):
        BioHybridGenotypeMatrix(
            dense=_NumpyDenseStub(z[:, :2]),
            sparse=GpuSparseCarrierMatrix.from_dense(
                raw[:, [1, 3]], mean[[1, 3]], scale[[1, 3]], xp_backend=np
            ),
            dense_to_global=np.array([0, 1], dtype=np.int64),
            sparse_to_global=np.array([1, 3], dtype=np.int64),  # overlaps at 1
            xp_backend=np,
        )


def test_empty_sparse_side_is_valid() -> None:
    rng = np.random.default_rng(3)
    raw = rng.integers(0, 3, size=(50, 5), dtype=np.int8)
    mean = raw.astype(np.float32).mean(axis=0)
    scale = raw.astype(np.float32).std(axis=0) + 1e-3
    z = _standardize_columns(raw, mean, scale)
    sparse_empty = GpuSparseCarrierMatrix(
        n_samples=50,
        carrier_indices=[],
        carrier_genotypes=[],
        mean=np.empty((0,), dtype=np.float32),
        scale=np.empty((0,), dtype=np.float32),
        xp_backend=np,
    )
    hybrid = BioHybridGenotypeMatrix(
        dense=_NumpyDenseStub(z),
        sparse=sparse_empty,
        dense_to_global=np.arange(5, dtype=np.int64),
        sparse_to_global=np.empty((0,), dtype=np.int64),
        xp_backend=np,
    )
    x = rng.standard_normal(5).astype(np.float32)
    np.testing.assert_allclose(np.asarray(hybrid.matvec(x)), z @ x, rtol=1e-4, atol=1e-4)

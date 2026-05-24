"""Phase 5 RED regression tests: block-decomposed matvec equivalence.

These tests pin the mathematical contract that a block-decomposed matvec is
EXACTLY equivalent (to fp32 noise) to the global matvec:

    X @ beta = sum over blocks b of X_b @ beta_b
    X.T @ y  = concat-rows(X_b.T @ y for b in blocks)

The block matvec API lives in ``sv_pgs.block_matvec`` (phase 2). Until that
module lands these tests skip; once phase 2 implements the API they must pass.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    from sv_pgs.block_matvec import block_matvec, block_transpose_matvec

    _HAS_BLOCK_MATVEC = True
except ImportError:
    _HAS_BLOCK_MATVEC = False

try:
    from sv_pgs.ld_blocks import assign_ld_blocks, block_partition  # noqa: F401

    _HAS_LD_BLOCKS = True
except ImportError:
    _HAS_LD_BLOCKS = False

pytestmark = pytest.mark.skipif(
    not _HAS_BLOCK_MATVEC, reason="phase-2 block_matvec not landed yet"
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _rand_X(n_samples: int, n_variants: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_samples, n_variants), dtype=np.float32)


def _contiguous_block_ids(n_variants: int, n_blocks: int) -> np.ndarray:
    assert n_variants % n_blocks == 0
    per = n_variants // n_blocks
    return np.repeat(np.arange(n_blocks, dtype=np.int32), per)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_block_matvec_matches_global_matvec_small():
    """Identity: X @ beta == sum_b X_b @ beta_b over a contiguous partition."""
    X = _rand_X(100, 500, seed=1)
    rng = np.random.default_rng(2)
    beta = rng.standard_normal(500, dtype=np.float32)
    block_ids = _contiguous_block_ids(500, 5)

    got = block_matvec(X, beta, block_ids)
    expected = X @ beta
    assert np.allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_block_transpose_matvec_matches_global():
    """Identity: X.T @ y == concat-rows(X_b.T @ y) over a contiguous partition."""
    X = _rand_X(100, 500, seed=3)
    rng = np.random.default_rng(4)
    y = rng.standard_normal(100, dtype=np.float32)
    block_ids = _contiguous_block_ids(500, 5)

    got = block_transpose_matvec(X, y, block_ids)
    expected = X.T @ y
    assert np.allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_block_matvec_with_non_contiguous_blocks():
    """Interleaved block_ids must still satisfy the global-equivalence identity.

    Catches implementations that assume block ids form contiguous variant ranges.
    """
    X = _rand_X(64, 200, seed=5)
    rng = np.random.default_rng(6)
    beta = rng.standard_normal(200, dtype=np.float32)
    # Interleave: variant i -> block (i % 4)
    block_ids = (np.arange(200, dtype=np.int32) % 4).astype(np.int32)

    got = block_matvec(X, beta, block_ids)
    expected = X @ beta
    assert np.allclose(got, expected, rtol=1e-6, atol=1e-6)

    got_t = block_transpose_matvec(X, rng.standard_normal(64, dtype=np.float32), block_ids)
    # transpose path: must still equal X.T @ y regardless of block assignment
    # recompute with the same y used above
    # (re-roll y for determinism)
    y = np.random.default_rng(7).standard_normal(64, dtype=np.float32)
    got_t = block_transpose_matvec(X, y, block_ids)
    assert np.allclose(got_t, X.T @ y, rtol=1e-6, atol=1e-6)


def test_block_matvec_with_singleton_blocks():
    """One singleton block (1 variant) mixed with normal-sized blocks."""
    n_blocks = 10
    sizes = [1] + [50] * (n_blocks - 1)  # 1 + 9*50 = 451
    n_variants = sum(sizes)
    X = _rand_X(40, n_variants, seed=8)
    rng = np.random.default_rng(9)
    beta = rng.standard_normal(n_variants, dtype=np.float32)
    block_ids = np.concatenate(
        [np.full(sz, b, dtype=np.int32) for b, sz in enumerate(sizes)]
    )

    got = block_matvec(X, beta, block_ids)
    assert np.allclose(got, X @ beta, rtol=1e-6, atol=1e-6)


def test_block_matvec_with_one_block_covering_all():
    """Degenerate partition: a single block over all variants == X @ beta."""
    X = _rand_X(80, 500, seed=10)
    rng = np.random.default_rng(11)
    beta = rng.standard_normal(500, dtype=np.float32)
    block_ids = np.zeros(500, dtype=np.int32)

    got = block_matvec(X, beta, block_ids)
    assert np.allclose(got, X @ beta, rtol=1e-6, atol=1e-6)


def test_block_matvec_with_zero_variants_in_block():
    """Block id present in the id space but assigned to no variants must not crash.

    The contract: block ids referenced by no variant contribute zero. Simulated
    here by leaving a gap in observed ids (e.g. ids 0,1,3 with no 2). The result
    must still equal X @ beta.
    """
    X = _rand_X(32, 100, seed=12)
    rng = np.random.default_rng(13)
    beta = rng.standard_normal(100, dtype=np.float32)
    # ids 0,1,3 — block 2 has zero variants
    block_ids = np.concatenate(
        [
            np.full(40, 0, dtype=np.int32),
            np.full(30, 1, dtype=np.int32),
            np.full(30, 3, dtype=np.int32),
        ]
    )

    got = block_matvec(X, beta, block_ids)
    assert np.allclose(got, X @ beta, rtol=1e-6, atol=1e-6)


def test_block_matvec_dtype_preservation():
    """fp32 in -> fp32 out (no silent fp64 upcasts)."""
    X = _rand_X(50, 200, seed=14)
    rng = np.random.default_rng(15)
    beta = rng.standard_normal(200, dtype=np.float32)
    block_ids = _contiguous_block_ids(200, 4)

    out = block_matvec(X, beta, block_ids)
    assert out.dtype == np.float32

    y = rng.standard_normal(50, dtype=np.float32)
    out_t = block_transpose_matvec(X, y, block_ids)
    assert out_t.dtype == np.float32


def test_block_matvec_with_standardization():
    """Per-block standardized matvec matches the global standardized matvec.

    The genotype matrix in sv_pgs is standardized: X_std = (X - mean) / scale.
    The block API accepts ``means`` and ``scales`` arrays of length n_variants
    and computes (X_std) @ beta block-by-block. Equivalence must hold.
    """
    X = _rand_X(64, 300, seed=16)
    rng = np.random.default_rng(17)
    beta = rng.standard_normal(300, dtype=np.float32)
    means = rng.standard_normal(300, dtype=np.float32)
    scales = (rng.uniform(0.5, 2.0, size=300)).astype(np.float32)
    block_ids = _contiguous_block_ids(300, 6)

    X_std = (X - means[np.newaxis, :]) / scales[np.newaxis, :]
    expected = X_std @ beta

    got = block_matvec(X, beta, block_ids, means=means, scales=scales)
    assert np.allclose(got, expected, rtol=1e-5, atol=1e-5)

    # Transpose path under standardization: X_std.T @ y
    y = rng.standard_normal(64, dtype=np.float32)
    expected_t = X_std.T @ y
    got_t = block_transpose_matvec(X, y, block_ids, means=means, scales=scales)
    assert np.allclose(got_t, expected_t, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not _HAS_LD_BLOCKS, reason="phase-1 sv_pgs.ld_blocks not landed yet")
def test_block_matvec_with_ld_block_partition():
    """End-to-end: use the real LD-block assigner from phase 1, verify identity."""
    from sv_pgs.ld_blocks import assign_ld_blocks  # type: ignore

    rng = np.random.default_rng(18)
    n_variants = 400
    # half on chr1, half on chr2; random positions within 0..1e7
    chroms = np.array(["1"] * (n_variants // 2) + ["2"] * (n_variants // 2))
    positions = np.sort(rng.integers(0, 10_000_000, size=n_variants)).astype(np.int64)
    # re-sort within each chrom to be realistic
    half = n_variants // 2
    positions[:half] = np.sort(positions[:half])
    positions[half:] = np.sort(positions[half:])

    block_ids = np.asarray(assign_ld_blocks(chroms, positions), dtype=np.int32)
    assert block_ids.shape == (n_variants,)

    X = _rand_X(50, n_variants, seed=19)
    beta = rng.standard_normal(n_variants, dtype=np.float32)

    got = block_matvec(X, beta, block_ids)
    assert np.allclose(got, X @ beta, rtol=1e-5, atol=1e-5)


def test_block_matvec_large_random():
    """Larger stress test: 1000x10000 with 20 non-uniform random blocks."""
    rng = np.random.default_rng(20)
    n_samples, n_variants, n_blocks = 1000, 10_000, 20

    # Non-uniform sizes summing to n_variants (mean 500 each).
    raw = rng.uniform(0.5, 1.5, size=n_blocks)
    sizes = np.maximum(1, np.round(raw / raw.sum() * n_variants).astype(int))
    # fix rounding drift
    diff = n_variants - sizes.sum()
    sizes[0] += diff
    assert sizes.sum() == n_variants and (sizes > 0).all()

    block_ids = np.concatenate(
        [np.full(sz, b, dtype=np.int32) for b, sz in enumerate(sizes)]
    )

    X = _rand_X(n_samples, n_variants, seed=21)
    beta = rng.standard_normal(n_variants, dtype=np.float32)

    got = block_matvec(X, beta, block_ids)
    expected = X @ beta
    max_abs = float(np.max(np.abs(got - expected)))
    assert max_abs < 1e-4, f"max abs diff {max_abs} exceeded 1e-4"

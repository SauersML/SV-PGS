"""Tests for GPU memory hygiene in mixture_inference.

Covers GPU memory hygiene cases:

1. ``_SampleSpacePreconditionerCacheEntry.clear_gpu_arrays()`` nulls both
   ``nystrom_basis_gpu`` and ``nystrom_factor_gpu`` so the cupy memory pool
   can reclaim their backing storage.
2. After calling ``cupy.get_default_memory_pool().free_all_blocks()`` the
   pool's reported size should decrease (skipped when cupy is unavailable).
3. The GPU low-rank preconditioner does not cache a second full weighted
   factor matrix.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from sv_pgs.mixture_inference import (
    _SampleSpacePreconditionerCacheEntry,
    _apply_sample_space_low_rank_preconditioner_gpu,
    _build_sample_space_low_rank_bundle_gpu,
)


def _make_entry_with_gpu_arrays() -> _SampleSpacePreconditionerCacheEntry:
    # We mock "GPU" arrays with numpy ndarrays since clear_gpu_arrays() only
    # nulls the references and never inspects array contents.
    nystrom_basis = np.zeros((4, 2), dtype=np.float64)
    nystrom_factor = np.zeros((4, 2), dtype=np.float64)
    return _SampleSpacePreconditionerCacheEntry(
        batch_size=2,
        rank=2,
        random_seed=0,
        prior_variances=np.ones(4, dtype=np.float64),
        diagonal_noise=np.ones(4, dtype=np.float64),
        diagonal_preconditioner=np.ones(4, dtype=np.float64),
        preconditioner=None,
        nystrom_basis_gpu=nystrom_basis,
        nystrom_factor_gpu=nystrom_factor,
    )


def test_clear_gpu_arrays_nulls_both_fields() -> None:
    entry = _make_entry_with_gpu_arrays()
    assert entry.nystrom_basis_gpu is not None
    assert entry.nystrom_factor_gpu is not None

    entry.clear_gpu_arrays()

    assert entry.nystrom_basis_gpu is None
    assert entry.nystrom_factor_gpu is None


def test_clear_gpu_arrays_safe_when_already_none() -> None:
    # Entry never held GPU arrays — calling clear must be a no-op.
    entry = _SampleSpacePreconditionerCacheEntry(
        batch_size=1,
        rank=1,
        random_seed=0,
        prior_variances=np.ones(2, dtype=np.float64),
        diagonal_noise=np.ones(2, dtype=np.float64),
        diagonal_preconditioner=np.ones(2, dtype=np.float64),
        preconditioner=None,
    )
    entry.clear_gpu_arrays()
    assert entry.nystrom_basis_gpu is None
    assert entry.nystrom_factor_gpu is None


def test_gpu_low_rank_preconditioner_avoids_cached_weighted_factor(monkeypatch, random_generator) -> None:
    fake_cupy = types.SimpleNamespace(
        asarray=np.asarray,
        dtype=np.dtype,
        einsum=np.einsum,
        eye=np.eye,
        float32=np.float32,
        float64=np.float64,
        asfortranarray=np.asfortranarray,
        linalg=np.linalg,
        maximum=np.maximum,
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)

    low_rank_factor = random_generator.normal(size=(10, 3)).astype(np.float64) * 0.1
    selected_diagonal = np.einsum("ij,ij->i", low_rank_factor, low_rank_factor)
    base_diagonal = random_generator.uniform(1.0, 2.0, size=10).astype(np.float64)
    diagonal_preconditioner = base_diagonal + selected_diagonal
    rhs = random_generator.normal(size=(10, 2)).astype(np.float64)

    bundle = _build_sample_space_low_rank_bundle_gpu(
        low_rank_factor,
        diagonal_preconditioner,
        cp=fake_cupy,
        bundle_dtype=np.float64,
    )

    assert len(bundle) == 3

    def solve_triangular_numpy(matrix, right_hand_side, *, lower, trans=0, check_finite=True):
        triangular = np.tril(matrix) if lower else np.triu(matrix)
        if trans in {"T", "C", 1, 2}:
            triangular = triangular.T
        return np.linalg.solve(triangular, right_hand_side)

    actual = _apply_sample_space_low_rank_preconditioner_gpu(
        rhs,
        bundle,
        cp=fake_cupy,
        solve_triangular_gpu=solve_triangular_numpy,
    )
    expected = np.linalg.solve(np.diag(base_diagonal) + low_rank_factor @ low_rank_factor.T, rhs)

    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)


def test_free_all_blocks_returns_memory_to_driver() -> None:
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:  # pragma: no cover - no CUDA device available
        pytest.skip("CUDA device not available")

    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    baseline = pool.total_bytes()

    array = cp.zeros(1 << 20, dtype=cp.float32)  # ~4 MB allocation
    assert pool.total_bytes() >= baseline + array.nbytes
    del array

    # Freed memory sits in the pool until free_all_blocks() releases it.
    pool.free_all_blocks()
    assert pool.total_bytes() <= baseline

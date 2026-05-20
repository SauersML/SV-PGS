"""Tests for GPU memory hygiene in mixture_inference.

Covers two audit findings:

1. ``_SampleSpacePreconditionerCacheEntry.clear_gpu_arrays()`` nulls both
   ``nystrom_basis_gpu`` and ``nystrom_factor_gpu`` so the cupy memory pool
   can reclaim their backing storage.
2. After calling ``cupy.get_default_memory_pool().free_all_blocks()`` the
   pool's reported size should decrease (skipped when cupy is unavailable).
"""

from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.mixture_inference import _SampleSpacePreconditionerCacheEntry


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

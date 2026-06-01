"""Focused regression tests for two genotype.py bug fixes.

1. Async upload sync gap: ``_upload_*_tiles_overlapped`` must wait on the
   final in-flight H2D event before returning so callers reading
   ``self._cupy_cache`` immediately after ``try_materialize_gpu()`` see
   fully-materialized data.

2. Subset GPU cache view safety: ``StandardizedGenotypeMatrix.subset()``
   must keep a strong reference to the parent so a subset whose
   ``_cupy_cache`` is a view into the parent's GPU buffer cannot become
   a dangling pointer if the caller drops their reference to the parent.

Both tests use ``pytest.importorskip('cupy')`` so they are skipped (not
failed) on machines without a working CuPy runtime.
"""
from __future__ import annotations

import gc

import numpy as np
import pytest

from sv_pgs.genotype import as_raw_genotype_matrix


def test_try_materialize_gpu_no_async_race_on_last_tile():
    """After ``try_materialize_gpu()`` returns, the GPU cache must contain the
    fully-uploaded matrix — there must be no in-flight async H2D copy left over
    from the last tile.
    """
    pytest.importorskip("cupy")

    # Build a small int8 genotype matrix. Use multiple variants so the upload
    # loop issues more than one tile and the "last tile event" actually exists.
    rng = np.random.default_rng(0)
    sample_count = 64
    variant_count = 32
    raw_values = rng.integers(0, 3, size=(sample_count, variant_count), dtype=np.int8)
    means = raw_values.astype(np.float32).mean(axis=0)
    scales = raw_values.astype(np.float32).std(axis=0) + 1e-6

    standardized = as_raw_genotype_matrix(raw_values.astype(np.float32)).standardized(
        means, scales
    )

    # If CuPy initialization or GPU upload fails (no device, OOM), skip rather
    # than fail — the test's purpose is to validate post-condition correctness
    # when GPU materialization actually succeeds.
    if not standardized.try_materialize_gpu():
        pytest.skip("GPU materialization unavailable in this environment.")

    expected = as_raw_genotype_matrix(raw_values.astype(np.float32)).standardized(
        means, scales
    ).materialize()

    # Read the cache immediately — no extra synchronize() here. If the last
    # tile's async H2D copy were still in flight, this would return stale
    # (zeroed) data and the assertion would fail.
    # ``_cupy_cache`` is a CuPy device array; copy it to host explicitly
    # (CuPy 13 rejects the implicit ``np.asarray`` device→host conversion).
    materialized_from_cache = standardized._cupy_cache.get()
    np.testing.assert_allclose(materialized_from_cache, expected, atol=1e-5)


def test_subset_gpu_cache_survives_parent_gc():
    """A GPU subset cache must remain valid even after the caller drops their
    reference to the parent ``StandardizedGenotypeMatrix``. The subset holds
    a view into the parent's GPU buffer; the parent-lifetime invariant on
    ``_parent_genotype_matrix`` must keep that buffer alive.
    """
    pytest.importorskip("cupy")

    rng = np.random.default_rng(1)
    sample_count = 48
    variant_count = 16
    raw_values = rng.integers(0, 3, size=(sample_count, variant_count), dtype=np.int8)
    means = raw_values.astype(np.float32).mean(axis=0)
    scales = raw_values.astype(np.float32).std(axis=0) + 1e-6

    parent = as_raw_genotype_matrix(raw_values.astype(np.float32)).standardized(
        means, scales
    )

    if not parent.try_materialize_gpu():
        pytest.skip("GPU materialization unavailable in this environment.")

    local_indices = np.array([0, 2, 5, 9, 11], dtype=np.int32)
    subset = parent.subset(local_indices)
    assert subset._cupy_cache is not None
    # The parent-lifetime invariant: subset must retain a reference to parent.
    assert subset._parent_genotype_matrix is parent

    expected = as_raw_genotype_matrix(raw_values.astype(np.float32)).standardized(
        means, scales
    ).subset(local_indices).materialize()

    # Drop the caller's parent reference and force GC. The subset's stored
    # reference to ``parent`` must keep the underlying GPU buffer alive.
    del parent
    gc.collect()

    # ``_cupy_cache`` is a CuPy device array; copy it to host explicitly
    # (CuPy 13 rejects the implicit ``np.asarray`` device→host conversion).
    materialized_from_subset = subset._cupy_cache.get()
    np.testing.assert_allclose(materialized_from_subset, expected, atol=1e-5)

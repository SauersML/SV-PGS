from __future__ import annotations

import numpy as np
import pytest

import sv_pgs.genotype as genotype_module
from sv_pgs.genotype import RawGenotypeBatch, RawGenotypeMatrix, as_raw_genotype_matrix


class _StreamingRawGenotypeMatrix(RawGenotypeMatrix):
    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = np.asarray(matrix, dtype=np.float32)

    @property
    def shape(self) -> tuple[int, int]:
        return self.matrix.shape

    def iter_column_batches(
        self,
        variant_indices=None,
        batch_size: int = 1024,
    ):
        resolved_indices = (
            np.arange(self.matrix.shape[1], dtype=np.int32)
            if variant_indices is None
            else np.asarray(variant_indices, dtype=np.int32)
        )
        safe_batch_size = max(int(batch_size), 1)
        for start_index in range(0, resolved_indices.shape[0], safe_batch_size):
            batch_indices = resolved_indices[start_index : start_index + safe_batch_size]
            yield RawGenotypeBatch(
                variant_indices=batch_indices,
                values=np.asarray(self.matrix[:, batch_indices], dtype=np.float32),
            )

    def materialize(self, variant_indices=None) -> np.ndarray:
        resolved_indices = (
            np.arange(self.matrix.shape[1], dtype=np.int32)
            if variant_indices is None
            else np.asarray(variant_indices, dtype=np.int32)
        )
        return np.asarray(self.matrix[:, resolved_indices], dtype=np.float32)


class _SpyStreamingRawGenotypeMatrix(_StreamingRawGenotypeMatrix):
    def __init__(self, matrix: np.ndarray) -> None:
        super().__init__(matrix)
        self.iter_requests: list[list[int]] = []
        self.materialize_requests: list[list[int]] = []

    def iter_column_batches(
        self,
        variant_indices=None,
        batch_size: int = 1024,
    ):
        resolved_indices = (
            np.arange(self.matrix.shape[1], dtype=np.int32)
            if variant_indices is None
            else np.asarray(variant_indices, dtype=np.int32)
        )
        self.iter_requests.append(resolved_indices.tolist())
        yield from super().iter_column_batches(resolved_indices, batch_size=batch_size)

    def materialize(self, variant_indices=None) -> np.ndarray:
        resolved_indices = (
            np.arange(self.matrix.shape[1], dtype=np.int32)
            if variant_indices is None
            else np.asarray(variant_indices, dtype=np.int32)
        )
        self.materialize_requests.append(resolved_indices.tolist())
        return super().materialize(resolved_indices)


def test_streaming_standardized_linear_algebra_matches_dense_path():
    raw_matrix = np.array(
        [
            [0.0, 1.0, np.nan, 0.0],
            [1.0, 0.0, 2.0, 1.0],
            [2.0, 1.0, 1.0, 0.0],
            [np.nan, 2.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    means = np.array([1.0, 1.0, 1.0, 0.5], dtype=np.float32)
    scales = np.array([0.5, 1.0, 1.0, 0.5], dtype=np.float32)
    coefficients = np.array([0.2, -0.1, 0.5, 0.3], dtype=np.float64)
    sample_vector = np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float64)
    sample_matrix = np.column_stack([sample_vector, sample_vector * -0.5]).astype(np.float64)

    dense = as_raw_genotype_matrix(raw_matrix).standardized(means, scales)
    streaming = _StreamingRawGenotypeMatrix(raw_matrix).standardized(means, scales)

    np.testing.assert_allclose(
        np.asarray(streaming.matvec(coefficients, batch_size=2), dtype=np.float64),
        np.asarray(dense.matvec(coefficients, batch_size=2), dtype=np.float64),
    )
    np.testing.assert_allclose(
        np.asarray(streaming.transpose_matvec(sample_vector, batch_size=2), dtype=np.float64),
        np.asarray(dense.transpose_matvec(sample_vector, batch_size=2), dtype=np.float64),
    )
    np.testing.assert_allclose(
        np.asarray(streaming.transpose_matmat(sample_matrix, batch_size=2), dtype=np.float64),
        np.asarray(dense.transpose_matmat(sample_matrix, batch_size=2), dtype=np.float64),
    )


def test_materialized_linear_algebra_matches_dense_reference():
    random_generator = np.random.default_rng(0)
    raw_matrix = random_generator.normal(size=(12, 130)).astype(np.float32)
    means = raw_matrix.mean(axis=0).astype(np.float32)
    scales = np.maximum(raw_matrix.std(axis=0), 0.25).astype(np.float32)
    coefficients = random_generator.normal(size=raw_matrix.shape[1]).astype(np.float64)
    variant_matrix = random_generator.normal(size=(raw_matrix.shape[1], 11)).astype(np.float64)
    sample_vector = random_generator.normal(size=raw_matrix.shape[0]).astype(np.float64)
    sample_matrix = random_generator.normal(size=(raw_matrix.shape[0], 11)).astype(np.float64)

    dense = as_raw_genotype_matrix(raw_matrix).standardized(means, scales)
    gpu_cached = as_raw_genotype_matrix(raw_matrix).standardized(means, scales)
    gpu_cached._dense_cache = gpu_cached.materialize()

    np.testing.assert_allclose(
        np.asarray(gpu_cached.matvec(coefficients), dtype=np.float64),
        np.asarray(dense.matvec(coefficients), dtype=np.float64),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(gpu_cached.matmat(variant_matrix), dtype=np.float64),
        np.asarray(dense.matmat(variant_matrix), dtype=np.float64),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(gpu_cached.transpose_matvec(sample_vector), dtype=np.float64),
        np.asarray(dense.transpose_matvec(sample_vector), dtype=np.float64),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(gpu_cached.transpose_matmat(sample_matrix), dtype=np.float64),
        np.asarray(dense.transpose_matmat(sample_matrix), dtype=np.float64),
        rtol=1e-5,
        atol=1e-5,
    )


def test_standardized_streaming_batches_delegate_to_raw_batch_iterator():
    raw_matrix = np.arange(24, dtype=np.float32).reshape(4, 6)
    raw = _SpyStreamingRawGenotypeMatrix(raw_matrix)
    standardized = raw.standardized(
        means=np.zeros(raw_matrix.shape[1], dtype=np.float32),
        scales=np.ones(raw_matrix.shape[1], dtype=np.float32),
    ).subset(np.array([1, 3, 4], dtype=np.int32))

    batches = list(standardized.iter_column_batches(batch_size=2))

    assert [batch.variant_indices.tolist() for batch in batches] == [[0, 1], [2]]
    assert raw.iter_requests == [[1, 3, 4]]
    assert raw.materialize_requests == []


def test_try_materialize_gpu_does_not_force_cpu_dense_fallback(monkeypatch: pytest.MonkeyPatch):
    raw_matrix = np.arange(24, dtype=np.float32).reshape(4, 6)
    standardized = as_raw_genotype_matrix(raw_matrix).standardized(
        means=np.zeros(raw_matrix.shape[1], dtype=np.float32),
        scales=np.ones(raw_matrix.shape[1], dtype=np.float32),
    )

    monkeypatch.setattr(genotype_module, "_try_import_cupy", lambda: None)

    assert standardized.try_materialize_gpu() is False
    assert standardized._cupy_cache is None
    assert standardized._dense_cache is None


def test_subset_after_releasing_raw_storage_keeps_materialized_shape():
    raw_matrix = np.arange(12, dtype=np.float32).reshape(3, 4)
    standardized = as_raw_genotype_matrix(raw_matrix).standardized(
        means=np.zeros(raw_matrix.shape[1], dtype=np.float32),
        scales=np.ones(raw_matrix.shape[1], dtype=np.float32),
    )
    standardized._dense_cache = standardized.materialize()
    standardized.release_raw_storage()

    subset = standardized.subset(np.array([1, 3], dtype=np.int32))

    assert subset.shape == (3, 2)
    np.testing.assert_allclose(subset.materialize(), raw_matrix[:, [1, 3]])

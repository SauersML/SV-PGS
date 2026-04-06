from __future__ import annotations

from typing import Any, cast

import jax.numpy as jnp
import numpy as np
import pytest

import sv_pgs.genotype as genotype_module
from sv_pgs.genotype import ConcatenatedRawGenotypeMatrix, Int8RawGenotypeMatrix, RawGenotypeBatch, RawGenotypeMatrix, as_raw_genotype_matrix


class _StreamingRawGenotypeMatrix(RawGenotypeMatrix):
    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = np.asarray(matrix, dtype=np.float32)

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.matrix.shape[0]), int(self.matrix.shape[1])

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


class _SpyInt8StreamingRawGenotypeMatrix(RawGenotypeMatrix):
    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = np.asarray(matrix, dtype=np.int8)
        self.i8_requests: list[list[int]] = []
        self.float_requests: list[list[int]] = []

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.matrix.shape[0]), int(self.matrix.shape[1])

    def iter_column_batches_i8(
        self,
        variant_indices=None,
        batch_size: int = 1024,
    ):
        resolved_indices = (
            np.arange(self.matrix.shape[1], dtype=np.int32)
            if variant_indices is None
            else np.asarray(variant_indices, dtype=np.int32)
        )
        self.i8_requests.append(resolved_indices.tolist())
        safe_batch_size = max(int(batch_size), 1)
        for start_index in range(0, resolved_indices.shape[0], safe_batch_size):
            batch_indices = resolved_indices[start_index : start_index + safe_batch_size]
            yield RawGenotypeBatch(
                variant_indices=batch_indices,
                values=np.asarray(self.matrix[:, batch_indices], dtype=np.int8),
            )

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
        self.float_requests.append(resolved_indices.tolist())
        raise AssertionError("GPU materialization should prefer int8 batches when available.")

    def materialize(self, variant_indices=None) -> np.ndarray:
        resolved_indices = (
            np.arange(self.matrix.shape[1], dtype=np.int32)
            if variant_indices is None
            else np.asarray(variant_indices, dtype=np.int32)
        )
        values = self.matrix[:, resolved_indices].astype(np.float32)
        values[self.matrix[:, resolved_indices] == -127] = np.nan
        return values


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


def test_matvec_skips_streaming_when_coefficients_are_zero():
    raw_matrix = np.arange(24, dtype=np.float32).reshape(4, 6)
    raw = _SpyStreamingRawGenotypeMatrix(raw_matrix)
    standardized = raw.standardized(
        means=np.zeros(raw_matrix.shape[1], dtype=np.float32),
        scales=np.ones(raw_matrix.shape[1], dtype=np.float32),
    )

    result = standardized.matvec(np.zeros(raw_matrix.shape[1], dtype=np.float32), batch_size=2)

    np.testing.assert_allclose(np.asarray(result), np.zeros(raw_matrix.shape[0], dtype=np.float32))
    assert raw.iter_requests == []


def test_try_cache_locally_rebases_to_local_int8_cache():
    raw_i8 = np.array(
        [
            [0, 1, -127, 2],
            [1, -127, 2, 0],
            [2, 1, 0, 1],
        ],
        dtype=np.int8,
    )
    raw = _SpyInt8StreamingRawGenotypeMatrix(raw_i8)
    means = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    scales = np.array([0.5, 2.0, 1.0, 0.5], dtype=np.float32)
    standardized = raw.standardized(means, scales).subset(np.array([1, 3], dtype=np.int32))

    assert standardized.try_cache_locally() is True
    assert raw.i8_requests == [[1, 3]]
    assert isinstance(standardized.raw, Int8RawGenotypeMatrix)

    raw.i8_requests.clear()
    np.testing.assert_allclose(
        standardized.materialize(),
        np.array(
            [
                [0.0, 2.0],
                [0.0, -2.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    assert raw.i8_requests == []


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


def test_concatenated_raw_genotype_matrix_preserves_column_order_and_values():
    left = _StreamingRawGenotypeMatrix(
        np.array(
            [
                [0.0, 1.0],
                [2.0, 3.0],
            ],
            dtype=np.float32,
        )
    )
    right = _StreamingRawGenotypeMatrix(
        np.array(
            [
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
            dtype=np.float32,
        )
    )
    concatenated = ConcatenatedRawGenotypeMatrix((left, right))

    np.testing.assert_allclose(
        concatenated.materialize(),
        np.array(
            [
                [0.0, 1.0, 4.0, 5.0, 6.0],
                [2.0, 3.0, 7.0, 8.0, 9.0],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(
        concatenated.materialize(np.array([4, 1, 3], dtype=np.int32)),
        np.array(
            [
                [6.0, 1.0, 5.0],
                [9.0, 3.0, 8.0],
            ],
            dtype=np.float32,
        ),
    )

    batches = list(concatenated.iter_column_batches(np.array([4, 1, 3], dtype=np.int32), batch_size=2))
    assert [batch.variant_indices.tolist() for batch in batches] == [[4, 1], [3]]
    np.testing.assert_allclose(
        batches[0].values,
        np.array(
            [
                [6.0, 1.0],
                [9.0, 3.0],
            ],
            dtype=np.float32,
        ),
    )


def test_concatenated_raw_genotype_matrix_supports_int8_batch_iteration():
    left = Int8RawGenotypeMatrix(
        np.array(
            [
                [0, 1],
                [2, -127],
            ],
            dtype=np.int8,
        )
    )
    right = Int8RawGenotypeMatrix(
        np.array(
            [
                [2, 0, 1],
                [1, 2, -127],
            ],
            dtype=np.int8,
        )
    )
    concatenated = ConcatenatedRawGenotypeMatrix((left, right))

    batches = list(concatenated.iter_column_batches_i8(np.array([4, 1, 3], dtype=np.int32), batch_size=2))

    assert [batch.variant_indices.tolist() for batch in batches] == [[4, 1], [3]]
    np.testing.assert_array_equal(
        batches[0].values,
        np.array(
            [
                [1, 1],
                [-127, -127],
            ],
            dtype=np.int8,
        ),
    )
    np.testing.assert_array_equal(
        batches[1].values,
        np.array(
            [
                [0],
                [2],
            ],
            dtype=np.int8,
        ),
    )


def test_try_materialize_gpu_skips_when_matrix_exceeds_budget(monkeypatch: pytest.MonkeyPatch):
    class _FakeCudaRuntime:
        @staticmethod
        def memGetInfo():
            return (1_000_000_000, 16_000_000_000)

    class _FakeDevice:
        def synchronize(self) -> None:
            return None

    class _FakeCuda:
        runtime = _FakeCudaRuntime()

        @staticmethod
        def Device():
            return _FakeDevice()

    class _FakeCupy:
        float32 = np.float32
        cuda = _FakeCuda()

        @staticmethod
        def asarray(array, dtype=None):
            raise AssertionError("GPU upload should be skipped before asarray.")

        @staticmethod
        def empty(shape, dtype=None, order=None):
            raise AssertionError("GPU allocation should be skipped before empty.")

    raw_matrix = np.zeros((4, 6), dtype=np.int8)
    standardized = as_raw_genotype_matrix(raw_matrix).standardized(
        means=np.zeros(raw_matrix.shape[1], dtype=np.float32),
        scales=np.ones(raw_matrix.shape[1], dtype=np.float32),
    )

    monkeypatch.setattr(genotype_module, "_try_import_cupy", lambda: _FakeCupy())
    monkeypatch.setattr(genotype_module, "_gpu_materialization_budget_bytes", lambda cupy: 1)

    assert standardized.try_materialize_gpu() is False
    assert standardized._cupy_cache is None


def test_to_cupy_float32_uses_dlpack_for_jax_arrays(monkeypatch: pytest.MonkeyPatch):
    dlpack_inputs: list[object] = []

    class _FakeCupyArray:
        def astype(self, dtype, copy=False):
            return ("astype", dtype, copy)

    fake_cupy_array = _FakeCupyArray()

    class _FakeCupy:
        float32 = np.float32

        @staticmethod
        def from_dlpack(array):
            dlpack_inputs.append(array)
            return fake_cupy_array

        @staticmethod
        def asarray(array, dtype=None):
            raise AssertionError("DLPack path should bypass host asarray.")

    monkeypatch.setattr(genotype_module, "_try_import_cupy", lambda: _FakeCupy())

    result = genotype_module._to_cupy_float32(jnp.asarray([1.0, 2.0], dtype=jnp.float32))

    assert len(dlpack_inputs) == 1
    assert result == ("astype", np.float32, False)


def test_cupy_to_jax_uses_dlpack_when_available(monkeypatch: pytest.MonkeyPatch):
    dlpack_inputs: list[object] = []

    class _FakeCupyArray:
        def __dlpack__(self, _stream=None):
            return "dlpack-capsule"

        def get(self):
            raise AssertionError("DLPack path should bypass host get().")

    fake_array = _FakeCupyArray()
    expected = jnp.asarray([3.0, 4.0], dtype=genotype_module.gpu_compute_jax_dtype())

    def _record_from_dlpack(array: object):
        dlpack_inputs.append(array)
        return expected

    monkeypatch.setattr(genotype_module.jax_dlpack, "from_dlpack", _record_from_dlpack)

    result = genotype_module._cupy_to_jax(fake_array)

    assert dlpack_inputs == [fake_array]
    np.testing.assert_allclose(np.asarray(result), np.asarray(expected))


def test_try_materialize_gpu_standardizes_batches_directly_on_gpu(monkeypatch: pytest.MonkeyPatch):
    allocation_orders: list[str | None] = []

    class _FakeCudaRuntime:
        @staticmethod
        def memGetInfo():
            return (8_000_000_000, 16_000_000_000)

    class _FakeDevice:
        def synchronize(self) -> None:
            return None

    class _FakeCuda:
        runtime = _FakeCudaRuntime()

        @staticmethod
        def Device():
            return _FakeDevice()

    class _FakeCupy:
        float32 = np.float32
        cuda = _FakeCuda()

        @staticmethod
        def asarray(array, dtype=None):
            return np.asarray(array, dtype=dtype)

        @staticmethod
        def empty(shape, dtype=None, order=None):
            allocation_orders.append(order)
            return np.empty(shape, dtype=dtype, order="C" if order is None else order)

        @staticmethod
        def isnan(array):
            return np.isnan(array)

    raw_matrix = np.array(
        [
            [0.0, 1.0, np.nan],
            [1.0, np.nan, 2.0],
            [2.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    means = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    scales = np.array([0.5, 2.0, 1.0], dtype=np.float32)
    expected = as_raw_genotype_matrix(raw_matrix).standardized(means, scales).materialize()
    standardized = as_raw_genotype_matrix(raw_matrix).standardized(means, scales)

    monkeypatch.setattr(genotype_module, "_try_import_cupy", lambda: _FakeCupy())

    assert standardized.try_materialize_gpu() is True
    assert allocation_orders == ["F"]
    np.testing.assert_allclose(np.asarray(cast(Any, standardized._cupy_cache)), expected)


def test_try_materialize_gpu_prefers_int8_batches_when_available(monkeypatch: pytest.MonkeyPatch):
    class _FakeCudaRuntime:
        @staticmethod
        def memGetInfo():
            return (8_000_000_000, 16_000_000_000)

    class _FakeDevice:
        def synchronize(self) -> None:
            return None

    class _FakeCuda:
        runtime = _FakeCudaRuntime()

        @staticmethod
        def Device():
            return _FakeDevice()

    class _FakeCupy:
        float32 = np.float32
        cuda = _FakeCuda()

        @staticmethod
        def asarray(array, dtype=None):
            return np.asarray(array, dtype=dtype)

        @staticmethod
        def empty(shape, dtype=None, order=None):
            return np.empty(shape, dtype=dtype, order="C" if order is None else order)

        @staticmethod
        def isnan(array):
            return np.isnan(array)

    raw_i8 = np.array(
        [
            [0, 1, -127, 2],
            [1, -127, 2, 0],
            [2, 1, 0, 1],
        ],
        dtype=np.int8,
    )
    raw = _SpyInt8StreamingRawGenotypeMatrix(raw_i8)
    means = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    scales = np.array([0.5, 2.0, 1.0, 0.5], dtype=np.float32)
    standardized = raw.standardized(means, scales).subset(np.array([1, 3], dtype=np.int32))
    expected = np.array(
        [
            [0.0, 2.0],
            [0.0, -2.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )

    monkeypatch.setattr(genotype_module, "_try_import_cupy", lambda: _FakeCupy())

    assert standardized.try_materialize_gpu() is True
    assert raw.i8_requests == [[1, 3]]
    assert raw.float_requests == []
    np.testing.assert_allclose(np.asarray(cast(Any, standardized._cupy_cache)), expected)

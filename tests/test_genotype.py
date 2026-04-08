from __future__ import annotations

from typing import Any, cast

import jax
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


def test_try_cache_persistently_rebases_to_persistent_int8_cache(tmp_path):
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
    cache_path = tmp_path / "reduced_raw_i8.npy"

    assert standardized.try_cache_persistently(cache_path) is True
    assert raw.i8_requests == [[1, 3]]
    assert isinstance(standardized.raw, Int8RawGenotypeMatrix)
    assert cache_path.exists()

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


def test_try_cache_persistently_skips_when_disk_space_is_insufficient(tmp_path, monkeypatch: pytest.MonkeyPatch):
    raw_i8 = np.array(
        [
            [0, 1, -127, 2],
            [1, -127, 2, 0],
            [2, 1, 0, 1],
        ],
        dtype=np.int8,
    )
    raw = _SpyInt8StreamingRawGenotypeMatrix(raw_i8)
    standardized = raw.standardized(
        means=np.ones(raw_i8.shape[1], dtype=np.float32),
        scales=np.ones(raw_i8.shape[1], dtype=np.float32),
    )
    cache_path = tmp_path / "reduced_raw_i8.npy"

    monkeypatch.setattr(
        genotype_module,
        "_has_sufficient_free_space_for_int8_npy",
        lambda path, shape, fortran_order: (False, 123, 45),
    )

    assert standardized.try_cache_persistently(cache_path) is False
    assert not cache_path.exists()
    assert raw.i8_requests == []


def test_try_cache_locally_skips_when_disk_space_is_insufficient(monkeypatch: pytest.MonkeyPatch):
    raw_i8 = np.array(
        [
            [0, 1, -127, 2],
            [1, -127, 2, 0],
            [2, 1, 0, 1],
        ],
        dtype=np.int8,
    )
    raw = _SpyInt8StreamingRawGenotypeMatrix(raw_i8)
    standardized = raw.standardized(
        means=np.ones(raw_i8.shape[1], dtype=np.float32),
        scales=np.ones(raw_i8.shape[1], dtype=np.float32),
    )

    monkeypatch.setattr(
        genotype_module,
        "_has_sufficient_free_space_for_int8_npy",
        lambda path, shape, fortran_order: (False, 123, 45),
    )

    assert standardized.try_cache_locally() is False
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


def test_int8_gpu_cache_supports_linear_algebra(monkeypatch: pytest.MonkeyPatch):
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
        float64 = np.float64
        cuda = _FakeCuda()

        @staticmethod
        def asarray(array, dtype=None):
            return np.asarray(array, dtype=dtype)

        @staticmethod
        def empty(shape, dtype=None, order=None):
            return np.empty(shape, dtype=dtype, order="C" if order is None else order)

        @staticmethod
        def zeros(shape, dtype=None):
            return np.zeros(shape, dtype=dtype)

        @staticmethod
        def where(condition, x, y):
            return np.where(condition, x, y)

    raw_i8 = np.array(
        [
            [0, 1, -127, 2],
            [1, -127, 2, 0],
            [2, 1, 0, 1],
            [-127, 2, 1, 1],
        ],
        dtype=np.int8,
    )
    means = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    scales = np.array([0.5, 1.0, 1.0, 0.5], dtype=np.float32)
    coefficients = np.array([0.25, -0.75, 0.5, 1.25], dtype=np.float64)
    sample_matrix = np.column_stack(
        [
            np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float64),
            np.array([-1.5, 0.0, 2.0, 1.0], dtype=np.float64),
        ]
    )
    standardized = as_raw_genotype_matrix(raw_i8).standardized(means, scales)
    dense_matrix = standardized.materialize().astype(np.float64, copy=False)
    standardized._dense_cache = None

    monkeypatch.setattr(genotype_module, "_try_import_cupy", lambda: _FakeCupy())
    monkeypatch.setattr(genotype_module, "gpu_compute_numpy_dtype", lambda: np.dtype(np.float64))
    monkeypatch.setattr(genotype_module, "gpu_compute_jax_dtype", lambda: jnp.float64)
    monkeypatch.setattr(
        genotype_module,
        "_cupy_to_jax",
        lambda array: jnp.asarray(np.asarray(array), dtype=jnp.float64),
    )

    assert standardized.try_materialize_gpu() is True

    np.testing.assert_allclose(np.asarray(standardized.matvec(coefficients)), dense_matrix @ coefficients)
    np.testing.assert_allclose(np.asarray(standardized.matmat(sample_matrix)), dense_matrix @ sample_matrix)
    transpose_vector = np.array([1.5, -0.25, 0.75, -1.0], dtype=np.float64)
    np.testing.assert_allclose(np.asarray(standardized.transpose_matvec(transpose_vector)), dense_matrix.T @ transpose_vector)
    np.testing.assert_allclose(np.asarray(standardized.transpose_matmat(sample_matrix)), dense_matrix.T @ sample_matrix)


def test_streaming_linear_algebra_uses_gpu_batches_when_available(monkeypatch: pytest.MonkeyPatch):
    gpu_batch_calls = 0
    original_iter_standardized_gpu_batches = genotype_module._iter_standardized_gpu_batches

    class _FakeCupy:
        float32 = np.float32
        float64 = np.float64

        @staticmethod
        def asarray(array, dtype=None):
            return np.asarray(array, dtype=dtype)

        @staticmethod
        def zeros(shape, dtype=None):
            return np.zeros(shape, dtype=dtype)

        @staticmethod
        def empty(shape, dtype=None, order=None):
            return np.empty(shape, dtype=dtype, order="C" if order is None else order)

        @staticmethod
        def isnan(array):
            return np.isnan(array)

    def _record_gpu_batches(*args, **kwargs):
        nonlocal gpu_batch_calls
        gpu_batch_calls += 1
        yield from original_iter_standardized_gpu_batches(*args, **kwargs)

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
    sample_matrix = np.column_stack(
        [
            np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float32),
            np.array([-1.5, 0.0, 2.0, 1.0], dtype=np.float32),
        ]
    )

    dense = as_raw_genotype_matrix(raw_matrix).standardized(means, scales)
    streaming = _StreamingRawGenotypeMatrix(raw_matrix).standardized(means, scales)

    monkeypatch.setattr(genotype_module, "_try_import_cupy", lambda: _FakeCupy())
    monkeypatch.setattr(
        genotype_module,
        "_cupy_to_jax",
        lambda array: jnp.asarray(np.asarray(array), dtype=genotype_module.gpu_compute_jax_dtype()),
    )
    monkeypatch.setattr(genotype_module, "_iter_standardized_gpu_batches", _record_gpu_batches)

    result = streaming.transpose_matmat(sample_matrix, batch_size=2)

    assert gpu_batch_calls > 0
    np.testing.assert_allclose(
        np.asarray(result, dtype=np.float64),
        np.asarray(dense.transpose_matmat(sample_matrix, batch_size=2), dtype=np.float64),
        rtol=1e-5,
        atol=1e-5,
    )


def test_streaming_linear_algebra_uses_float32_gpu_batches_on_t4(monkeypatch: pytest.MonkeyPatch):
    gpu_batch_dtypes: list[object] = []
    original_iter_standardized_gpu_batches = genotype_module._iter_standardized_gpu_batches

    class _FakeCupy:
        float32 = np.float32
        float64 = np.float64

        @staticmethod
        def asarray(array, dtype=None):
            return np.asarray(array, dtype=dtype)

        @staticmethod
        def zeros(shape, dtype=None):
            return np.zeros(shape, dtype=dtype)

        @staticmethod
        def empty(shape, dtype=None, order=None):
            return np.empty(shape, dtype=dtype, order="C" if order is None else order)

        @staticmethod
        def isnan(array):
            return np.isnan(array)

    def _record_gpu_batches(*args, **kwargs):
        gpu_batch_dtypes.append(kwargs.get("dtype"))
        yield from original_iter_standardized_gpu_batches(*args, **kwargs)

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
    sample_matrix = np.column_stack(
        [
            np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float32),
            np.array([-1.5, 0.0, 2.0, 1.0], dtype=np.float32),
        ]
    )

    streaming = _StreamingRawGenotypeMatrix(raw_matrix).standardized(means, scales)

    monkeypatch.setattr(genotype_module, "_try_import_cupy", lambda: _FakeCupy())
    monkeypatch.setattr(genotype_module, "gpu_compute_numpy_dtype", lambda: np.dtype(np.float32))
    monkeypatch.setattr(genotype_module, "gpu_compute_jax_dtype", lambda: jnp.float32)
    monkeypatch.setattr(
        genotype_module,
        "_cupy_to_jax",
        lambda array: jnp.asarray(np.asarray(array), dtype=jnp.float32),
    )
    monkeypatch.setattr(genotype_module, "_iter_standardized_gpu_batches", _record_gpu_batches)

    _ = streaming.transpose_matmat(sample_matrix, batch_size=2)

    assert gpu_batch_dtypes
    assert set(gpu_batch_dtypes) == {np.float32}


def test_dense_gpu_cache_linear_algebra_respects_gpu_compute_dtype(monkeypatch: pytest.MonkeyPatch):
    class _FakeCupy:
        float32 = np.float32
        float64 = np.float64

        @staticmethod
        def asarray(array, dtype=None):
            return np.asarray(array, dtype=dtype)

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
    coefficients = np.array([0.25, -0.75, 0.5, 1.25], dtype=np.float64)
    sample_matrix = np.column_stack(
        [
            np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float64),
            np.array([-1.5, 0.0, 2.0, 1.0], dtype=np.float64),
        ]
    )
    standardized = as_raw_genotype_matrix(raw_matrix).standardized(means, scales)
    dense_matrix = standardized.materialize().astype(np.float64, copy=False)
    standardized._cupy_cache = standardized.materialize().astype(np.float32, copy=False)
    standardized._dense_cache = None

    monkeypatch.setattr(genotype_module, "_try_import_cupy", lambda: _FakeCupy())
    monkeypatch.setattr(genotype_module, "gpu_compute_numpy_dtype", lambda: np.dtype(np.float64))
    monkeypatch.setattr(genotype_module, "gpu_compute_jax_dtype", lambda: jnp.float64)
    monkeypatch.setattr(
        genotype_module,
        "_cupy_to_jax",
        lambda array: jnp.asarray(np.asarray(array), dtype=jnp.float64),
    )

    matvec_result = standardized.matvec(coefficients)
    matmat_result = standardized.matmat(sample_matrix)
    transpose_matvec_result = standardized.transpose_matvec(np.array([1.5, -0.25, 0.75, -1.0], dtype=np.float64))
    transpose_matmat_result = standardized.transpose_matmat(
        np.column_stack(
            [
                np.array([1.0, -0.5, 0.25, 2.0], dtype=np.float64),
                np.array([-1.5, 0.0, 1.0, 0.5], dtype=np.float64),
            ]
        )
    )

    assert np.asarray(matvec_result).dtype == np.float64
    assert np.asarray(matmat_result).dtype == np.float64
    assert np.asarray(transpose_matvec_result).dtype == np.float64
    assert np.asarray(transpose_matmat_result).dtype == np.float64
    np.testing.assert_allclose(np.asarray(matvec_result), dense_matrix @ coefficients)
    np.testing.assert_allclose(np.asarray(matmat_result), dense_matrix @ sample_matrix)
    np.testing.assert_allclose(
        np.asarray(transpose_matvec_result),
        dense_matrix.T @ np.array([1.5, -0.25, 0.75, -1.0], dtype=np.float64),
    )
    np.testing.assert_allclose(
        np.asarray(transpose_matmat_result),
        dense_matrix.T @ np.column_stack(
            [
                np.array([1.0, -0.5, 0.25, 2.0], dtype=np.float64),
                np.array([-1.5, 0.0, 1.0, 0.5], dtype=np.float64),
            ]
        ),
    )


def test_dense_gpu_cache_prefers_jax_dense_ops_without_cupy_bridge(monkeypatch: pytest.MonkeyPatch):
    raw_matrix = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
        ],
        dtype=np.float32,
    )
    standardized = as_raw_genotype_matrix(raw_matrix).standardized(
        means=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        scales=np.ones(4, dtype=np.float32),
    )
    dense_matrix = standardized.materialize().astype(np.float64, copy=False)
    standardized._cupy_cache = standardized.materialize().astype(np.float32, copy=False)
    standardized._dense_cache = None
    coefficients = np.array([0.25, -0.75, 0.5, 1.25], dtype=np.float64)
    sample_vector = np.array([1.5, -0.25, 0.75], dtype=np.float64)
    sample_matrix = np.column_stack(
        [
            np.array([1.0, -0.5, 0.25], dtype=np.float64),
            np.array([-1.5, 0.0, 1.0], dtype=np.float64),
        ]
    )
    variant_matrix = np.column_stack(
        [
            np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float64),
            np.array([-1.5, 0.0, 2.0, 1.0], dtype=np.float64),
        ]
    )

    monkeypatch.setattr(genotype_module, "jax_dense_linear_algebra_preferred", lambda: True)
    monkeypatch.setattr(
        genotype_module,
        "_cupy_to_jax",
        lambda array: (_ for _ in ()).throw(AssertionError("dense JAX path should not use CuPy->JAX conversion")),
    )

    np.testing.assert_allclose(np.asarray(standardized.matvec(coefficients), dtype=np.float64), dense_matrix @ coefficients)
    np.testing.assert_allclose(np.asarray(standardized.matmat(variant_matrix), dtype=np.float64), dense_matrix @ variant_matrix)
    np.testing.assert_allclose(
        np.asarray(standardized.transpose_matvec(sample_vector), dtype=np.float64),
        dense_matrix.T @ sample_vector,
    )
    np.testing.assert_allclose(
        np.asarray(standardized.transpose_matmat(sample_matrix), dtype=np.float64),
        dense_matrix.T @ sample_matrix,
    )


def test_try_materialize_gpu_subset_streams_only_selected_columns(monkeypatch: pytest.MonkeyPatch):
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

    raw = _SpyStreamingRawGenotypeMatrix(
        np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 3.0, 4.0, 5.0],
            ],
            dtype=np.float32,
        )
    )
    means = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    scales = np.ones(4, dtype=np.float32)
    standardized = raw.standardized(means, scales)

    monkeypatch.setattr(genotype_module, "_try_import_cupy", lambda: _FakeCupy())

    gpu_subset = standardized.try_materialize_gpu_subset(np.array([1, 3], dtype=np.int32))

    assert gpu_subset is not None
    assert raw.iter_requests == [[1, 3]]
    np.testing.assert_allclose(
        np.asarray(cast(Any, gpu_subset)),
        np.array(
            [
                [-1.0, -1.0],
                [0.0, 0.0],
                [1.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )


def test_dense_materialized_genotype_ops_are_jax_traceable():
    raw_matrix = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
        ],
        dtype=np.float32,
    )
    standardized = as_raw_genotype_matrix(raw_matrix).standardized(
        means=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        scales=np.ones(3, dtype=np.float32),
    )
    standardized.try_materialize()

    coefficient_vector = jnp.asarray([1.0, -2.0, 0.5], dtype=jnp.float64)
    sample_vector = jnp.asarray([0.5, -1.0, 2.0], dtype=jnp.float64)

    jit_matvec = jax.jit(lambda beta: standardized.matvec(beta))
    jit_transpose_matvec = jax.jit(lambda vector: standardized.transpose_matvec(vector))

    np.testing.assert_allclose(
        np.asarray(jit_matvec(coefficient_vector), dtype=np.float64),
        np.asarray(standardized.materialize(), dtype=np.float64) @ np.asarray(coefficient_vector, dtype=np.float64),
    )
    np.testing.assert_allclose(
        np.asarray(jit_transpose_matvec(sample_vector), dtype=np.float64),
        np.asarray(standardized.materialize(), dtype=np.float64).T @ np.asarray(sample_vector, dtype=np.float64),
    )


def test_hybrid_standardized_operator_matches_dense_reference(monkeypatch: pytest.MonkeyPatch):
    raw_i8 = np.array(
        [
            [0, 1, genotype_module.PLINK_MISSING_INT8, 2, 0, 1],
            [1, 1, 0, 1, 2, 0],
            [0, 2, 1, 0, 1, genotype_module.PLINK_MISSING_INT8],
            [0, 0, 0, 2, 1, 0],
        ],
        dtype=np.int8,
    )
    raw_float = raw_i8.astype(np.float32)
    raw_float[raw_i8 == genotype_module.PLINK_MISSING_INT8] = np.nan
    means = np.nanmean(raw_float, axis=0).astype(np.float32)
    scales = np.maximum(np.nanstd(raw_float, axis=0), 0.25).astype(np.float32)
    support_counts = np.array([1, 3, 1, 3, 3, 1], dtype=np.int32)
    coefficients = np.array([0.5, -0.25, 1.0, 0.75, -0.5, 0.125], dtype=np.float64)
    variant_matrix = np.array(
        [
            [0.5, -1.0],
            [1.0, 0.25],
            [-0.5, 0.75],
            [0.25, -0.5],
            [1.5, 0.5],
            [-1.25, 1.0],
        ],
        dtype=np.float64,
    )
    sample_vector = np.array([1.0, -0.75, 0.5, 1.25], dtype=np.float64)
    sample_matrix = np.column_stack([sample_vector, sample_vector * -0.5]).astype(np.float64)

    monkeypatch.setattr(genotype_module, "HYBRID_SPARSE_SUPPORT_THRESHOLD", 1)
    monkeypatch.setattr(genotype_module, "HYBRID_SPARSE_MIN_VARIANT_COUNT", 1)

    raw = _SpyInt8StreamingRawGenotypeMatrix(raw_i8)
    hybrid = raw.standardized(means, scales, support_counts=support_counts)
    dense = as_raw_genotype_matrix(raw_float).standardized(means, scales)

    assert hybrid._uses_hybrid_backend() is True
    assert hybrid._sparse_backend is not None
    assert hybrid._dense_backend is not None
    assert raw.float_requests == []

    np.testing.assert_allclose(
        np.asarray(hybrid.matvec(coefficients, batch_size=2), dtype=np.float64),
        np.asarray(dense.matvec(coefficients, batch_size=2), dtype=np.float64),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(hybrid.matmat(variant_matrix, batch_size=2), dtype=np.float64),
        np.asarray(dense.matmat(variant_matrix, batch_size=2), dtype=np.float64),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(hybrid.transpose_matvec(sample_vector, batch_size=2), dtype=np.float64),
        np.asarray(dense.transpose_matvec(sample_vector, batch_size=2), dtype=np.float64),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(hybrid.transpose_matmat(sample_matrix, batch_size=2), dtype=np.float64),
        np.asarray(dense.transpose_matmat(sample_matrix, batch_size=2), dtype=np.float64),
        rtol=1e-5,
        atol=1e-5,
    )

    raw.i8_requests.clear()
    subset = hybrid.subset(np.array([5, 1, 0, 3], dtype=np.int32))
    assert raw.i8_requests == []
    batches = list(subset.iter_column_batches(batch_size=2))

    assert [batch.variant_indices.tolist() for batch in batches] == [[0, 1], [2, 3]]
    np.testing.assert_allclose(
        np.concatenate([np.asarray(batch.values, dtype=np.float32) for batch in batches], axis=1),
        dense.subset(np.array([5, 1, 0, 3], dtype=np.int32)).materialize(batch_size=2),
        rtol=1e-5,
        atol=1e-5,
    )
    assert raw.float_requests == []

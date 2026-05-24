from __future__ import annotations

from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import sv_pgs.genotype as genotype_module
from sv_pgs.genotype import ConcatenatedRawGenotypeMatrix, Int8RawGenotypeMatrix, RawGenotypeBatch, RawGenotypeMatrix, as_raw_genotype_matrix


class _FakeCupyEvent:
    def __init__(self) -> None:
        self.synchronized = False

    def synchronize(self) -> None:
        self.synchronized = True


class _FakeCupyStream:
    def __init__(self) -> None:
        self.recorded_events: list[_FakeCupyEvent] = []
        self.enter_count = 0

    def __enter__(self) -> "_FakeCupyStream":
        self.enter_count += 1
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        return None

    def record(self) -> _FakeCupyEvent:
        event = _FakeCupyEvent()
        self.recorded_events.append(event)
        return event


def _install_fake_pinned_and_streams(fake_cupy_cls, *, stream_factory=None) -> None:
    """Attach a pinned-memory allocator and ``Stream`` factory onto a fake CuPy class."""
    if not hasattr(fake_cupy_cls, "cuda"):
        raise AssertionError("fake cupy class must expose a cuda attribute before installing CUDA shims")

    def _alloc_pinned_memory(nbytes: int) -> Any:
        return bytearray(int(nbytes))

    def _default_stream_factory(*_args, **_kwargs) -> _FakeCupyStream:
        return _FakeCupyStream()

    setattr(fake_cupy_cls.cuda, "alloc_pinned_memory", staticmethod(_alloc_pinned_memory))
    resolved_factory = stream_factory if stream_factory is not None else _default_stream_factory
    setattr(fake_cupy_cls.cuda, "Stream", staticmethod(resolved_factory))


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
        *,
        num_threads: int | None = None,
    ):
        del num_threads  # spy ignores decode-thread budgeting
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
        streaming.matvec_numpy(coefficients, batch_size=2),
        dense.matvec_numpy(coefficients, batch_size=2),
    )
    np.testing.assert_allclose(
        np.asarray(streaming.transpose_matvec(sample_vector, batch_size=2), dtype=np.float64),
        np.asarray(dense.transpose_matvec(sample_vector, batch_size=2), dtype=np.float64),
    )
    np.testing.assert_allclose(
        streaming.transpose_matvec_numpy(sample_vector, batch_size=2),
        dense.transpose_matvec_numpy(sample_vector, batch_size=2),
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


def test_matvec_streams_only_active_variant_columns():
    raw_matrix = np.arange(24, dtype=np.float32).reshape(4, 6)
    raw = _SpyStreamingRawGenotypeMatrix(raw_matrix)
    standardized = raw.standardized(
        means=np.zeros(raw_matrix.shape[1], dtype=np.float32),
        scales=np.ones(raw_matrix.shape[1], dtype=np.float32),
    )

    result = standardized.matvec(np.array([0.0, 2.0, 0.0, 0.0, -1.5, 0.0], dtype=np.float32), batch_size=2)

    np.testing.assert_allclose(
        np.asarray(result),
        raw_matrix[:, [1, 4]] @ np.array([2.0, -1.5], dtype=np.float32),
    )
    assert raw.iter_requests == [[1, 4]]


def test_matvec_rejects_nonfinite_coefficients():
    raw_matrix = np.arange(12, dtype=np.float32).reshape(3, 4)
    standardized = as_raw_genotype_matrix(raw_matrix).standardized(
        means=np.zeros(raw_matrix.shape[1], dtype=np.float32),
        scales=np.ones(raw_matrix.shape[1], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="coefficient vector must contain only finite values"):
        standardized.matvec(np.array([0.5, np.nan, 0.0, 1.0], dtype=np.float32))


def test_transpose_matvec_rejects_nonfinite_sample_vector():
    raw_matrix = np.arange(12, dtype=np.float32).reshape(3, 4)
    standardized = as_raw_genotype_matrix(raw_matrix).standardized(
        means=np.zeros(raw_matrix.shape[1], dtype=np.float32),
        scales=np.ones(raw_matrix.shape[1], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="sample vector must contain only finite values"):
        standardized.transpose_matvec(np.array([1.0, np.inf, -0.5], dtype=np.float32))


def test_matmat_streams_only_active_variant_rows():
    raw_matrix = np.arange(24, dtype=np.float32).reshape(4, 6)
    raw = _SpyStreamingRawGenotypeMatrix(raw_matrix)
    standardized = raw.standardized(
        means=np.zeros(raw_matrix.shape[1], dtype=np.float32),
        scales=np.ones(raw_matrix.shape[1], dtype=np.float32),
    )
    variant_matrix = np.array(
        [
            [0.0, 0.0],
            [2.0, -1.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [-1.5, 0.25],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )

    result = standardized.matmat(variant_matrix, batch_size=2)

    np.testing.assert_allclose(
        np.asarray(result),
        raw_matrix[:, [1, 4]] @ variant_matrix[[1, 4], :],
    )
    assert raw.iter_requests == [[1, 4]]


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

    _install_fake_pinned_and_streams(_FakeCupy)
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

    _install_fake_pinned_and_streams(_FakeCupy)
    monkeypatch.setattr(genotype_module, "_try_import_cupy", lambda: _FakeCupy())

    assert standardized.try_materialize_gpu() is True
    assert raw.i8_requests == [[1, 3]]
    assert raw.float_requests == []
    np.testing.assert_allclose(np.asarray(cast(Any, standardized._cupy_cache)), expected)


def test_streaming_gpu_context_uses_int8_budget_for_plink_like_backends(monkeypatch: pytest.MonkeyPatch):
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

    raw = _SpyInt8StreamingRawGenotypeMatrix(np.zeros((1_000, 16), dtype=np.int8))
    standardized = raw.standardized(
        np.zeros(16, dtype=np.float32),
        np.ones(16, dtype=np.float32),
    )

    monkeypatch.setattr(genotype_module, "_try_import_cupy", lambda: _FakeCupy())

    _, streaming_batch_size = standardized._streaming_gpu_context(batch_size=32)

    assert streaming_batch_size == genotype_module.auto_batch_size_i8(standardized.shape[0])


def test_streaming_gpu_context_caps_large_int8_standardized_tiles(monkeypatch: pytest.MonkeyPatch):
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

    class _ShapeOnlyInt8Raw(RawGenotypeMatrix):
        @property
        def shape(self) -> tuple[int, int]:
            return 77_689, 1_106_883

        def iter_column_batches_i8(self, variant_indices=None, batch_size: int = 1024):
            raise AssertionError("shape-only test must not stream genotypes")

        def iter_column_batches(self, variant_indices=None, batch_size: int = 1024):
            raise AssertionError("shape-only test must not stream genotypes")

        def materialize(self, variant_indices=None) -> np.ndarray:
            raise AssertionError("shape-only test must not materialize genotypes")

    monkeypatch.setattr(genotype_module, "_try_import_cupy", lambda: _FakeCupy())
    raw = _ShapeOnlyInt8Raw()
    standardized = raw.standardized(
        np.zeros(raw.shape[1], dtype=np.float32),
        np.ones(raw.shape[1], dtype=np.float32),
    )

    _, streaming_batch_size = standardized._streaming_gpu_context(batch_size=12_428)

    expected = genotype_module.GPU_STANDARDIZED_STREAMING_TARGET_BATCH_BYTES // (raw.shape[0] * 4)
    assert streaming_batch_size == expected
    assert streaming_batch_size < genotype_module.auto_batch_size_i8(raw.shape[0])


def test_int8_cupy_standardization_uses_elementwise_kernel():
    calls: list[str] = []

    class _FakeCupy:
        float32 = np.float32
        int8 = np.int8

        @staticmethod
        def asarray(array, dtype=None):
            return np.asarray(array, dtype=dtype)

        @staticmethod
        def ElementwiseKernel(_inputs, _outputs, _operation, name):
            calls.append(name)

            def _kernel(raw, means, scales, missing):
                raw_array = np.asarray(raw, dtype=np.int8)
                return np.where(
                    raw_array == np.int8(missing),
                    np.float32(0.0),
                    (raw_array.astype(np.float32) - means) / scales,
                ).astype(np.float32)

            return _kernel

    raw_values = np.array([[0, -127], [2, 1]], dtype=np.int8)
    result = genotype_module._standardize_batch_cupy(
        raw_values,
        np.array([1.0, 1.0], dtype=np.float32),
        np.array([2.0, 1.0], dtype=np.float32),
        _FakeCupy,
        missing_sentinel=-127,
        dtype=np.float32,
    )

    assert calls == ["sv_pgs_standardize_int8_missing_zero"]
    np.testing.assert_allclose(
        result,
        np.array([[-0.5, 0.0], [0.5, 0.0]], dtype=np.float32),
    )


def test_cupy_int8_cache_batches_shrink_when_gpu_free_memory_is_low():
    class _FakeCudaRuntime:
        @staticmethod
        def memGetInfo():
            return (1_000_000, 16_000_000_000)

    class _FakeCuda:
        runtime = _FakeCudaRuntime()

    class _FakeCupy:
        float32 = np.float32
        cuda = _FakeCuda()

        @staticmethod
        def asarray(array, dtype=None):
            return np.asarray(array, dtype=dtype)

    sample_count = 1_000
    cache = genotype_module._CupyInt8StandardizedCache(
        raw_values=np.zeros((sample_count, 12), dtype=np.int8),
        means=np.zeros(12, dtype=np.float32),
        scales=np.ones(12, dtype=np.float32),
        cupy=_FakeCupy,
    )

    batches = list(
        genotype_module._iter_cupy_cache_standardized_batches(
            cache,
            sample_count=sample_count,
            batch_size=12,
            cupy=_FakeCupy,
            dtype=np.float32,
        )
    )

    widths = [batch_slice.stop - batch_slice.start for batch_slice, _batch in batches]
    assert widths == [5, 5, 2]


def test_cupy_int8_cache_subset_is_zero_copy_view():
    """``subset`` must never duplicate the (large) raw_values buffer on the device.

    This is the OOM-prevention invariant on memory-constrained GPUs: a
    non-contiguous selection that touches most of the cache used to allocate a
    second full-size copy via fancy indexing, OOMing the device. The fix routes
    fancy selections through a view that shares ``raw_values`` with the parent;
    contiguous selections take a zero-copy slice. Both paths must leave the
    parent buffer object untouched.
    """

    class _FakeCupy:
        float32 = np.float32
        int8 = np.int8

        @staticmethod
        def asarray(array, dtype=None):
            return np.asarray(array, dtype=dtype)

    raw_values = np.arange(8 * 6, dtype=np.int8).reshape(8, 6)
    means = np.zeros(6, dtype=np.float32)
    scales = np.ones(6, dtype=np.float32)
    cache = genotype_module._CupyInt8StandardizedCache(
        raw_values=raw_values,
        means=means,
        scales=scales,
        cupy=_FakeCupy,
    )

    # Non-contiguous fancy selection: deferred-gather view.
    fancy_view = cache.subset(np.array([5, 1, 3], dtype=np.int32))
    assert fancy_view.column_indices is not None
    assert fancy_view.raw_values is raw_values, "fancy subset must share raw_values"
    assert fancy_view.shape == (8, 3)
    np.testing.assert_array_equal(np.asarray(fancy_view.column_indices), np.array([5, 1, 3]))

    # Contiguous selection on a root cache: zero-copy slice view.
    contig_view = cache.subset(np.array([2, 3, 4], dtype=np.int32))
    assert contig_view.column_indices is None
    assert np.shares_memory(contig_view.raw_values, raw_values), "contiguous subset must be a slice view"
    assert contig_view.shape == (8, 3)

    # Identity selection: returns the same cache instance.
    assert cache.subset(np.arange(6, dtype=np.int32)) is cache

    # View-of-view contiguous subset: slices the parent's index array, still no copy.
    composed = fancy_view.subset(np.array([0, 1], dtype=np.int32))
    assert composed.column_indices is not None
    assert composed.raw_values is raw_values
    np.testing.assert_array_equal(np.asarray(composed.column_indices), np.array([5, 1]))

    # nbytes reports the logical view footprint, not the shared parent buffer.
    assert fancy_view.nbytes < cache.nbytes
    assert fancy_view.nbytes >= 8 * 3 * raw_values.dtype.itemsize


def test_gpu_cached_matvec_with_all_active_coefficients_reuses_cache(monkeypatch: pytest.MonkeyPatch):
    class _FakeCudaRuntime:
        @staticmethod
        def memGetInfo():
            return (8_000_000_000, 16_000_000_000)

    class _FakeCuda:
        runtime = _FakeCudaRuntime()

    class _FakeCupy:
        float32 = np.float32
        float64 = np.float64
        cuda = _FakeCuda()

        @staticmethod
        def asarray(array, dtype=None):
            return np.asarray(array, dtype=dtype)

        @staticmethod
        def any(array):
            return np.any(array)

        @staticmethod
        def zeros(shape, dtype=None):
            return np.zeros(shape, dtype=dtype)

        @staticmethod
        def multiply(left, right, out=None):
            return np.multiply(left, right, out=out)

    raw_values = np.array(
        [
            [0, 1, 2],
            [1, 2, 0],
            [2, 0, 1],
        ],
        dtype=np.int8,
    )
    means = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    scales = np.array([0.5, 1.0, 2.0], dtype=np.float32)
    coefficients = np.array([0.25, -1.5, 2.0], dtype=np.float32)
    standardized = Int8RawGenotypeMatrix(raw_values).standardized(means, scales)
    standardized._cupy_cache = genotype_module._CupyInt8StandardizedCache(
        raw_values=raw_values,
        means=means,
        scales=scales,
        cupy=_FakeCupy,
    )

    def _unexpected_subset(*_args, **_kwargs):
        raise AssertionError("full active matvec must not allocate a subset cache")

    monkeypatch.setattr(genotype_module, "_try_import_cupy", lambda: _FakeCupy())
    monkeypatch.setattr(genotype_module, "_cupy_cache_subset_columns", _unexpected_subset)

    result = standardized.matvec_numpy(coefficients, batch_size=2)

    np.testing.assert_allclose(result, standardized.materialize() @ coefficients, rtol=1e-6, atol=1e-6)


def test_gpu_cached_partial_matvec_streams_selected_columns(monkeypatch: pytest.MonkeyPatch):
    class _FakeCudaRuntime:
        @staticmethod
        def memGetInfo():
            return (8_000_000_000, 16_000_000_000)

    class _FakeCuda:
        runtime = _FakeCudaRuntime()

    class _FakeCupy:
        float32 = np.float32
        float64 = np.float64
        cuda = _FakeCuda()

        @staticmethod
        def asarray(array, dtype=None):
            return np.asarray(array, dtype=dtype)

        @staticmethod
        def any(array):
            return np.any(array)

        @staticmethod
        def zeros(shape, dtype=None):
            return np.zeros(shape, dtype=dtype)

        @staticmethod
        def multiply(left, right, out=None):
            return np.multiply(left, right, out=out)

    raw_values = np.array(
        [
            [0, 1, 2, 0],
            [1, 2, 0, 1],
            [2, 0, 1, 2],
        ],
        dtype=np.int8,
    )
    means = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    scales = np.array([0.5, 1.0, 2.0, 0.25], dtype=np.float32)
    coefficients = np.array([0.0, -1.5, 0.0, 2.0], dtype=np.float32)
    standardized = Int8RawGenotypeMatrix(raw_values).standardized(means, scales)
    standardized._cupy_cache = genotype_module._CupyInt8StandardizedCache(
        raw_values=raw_values,
        means=means,
        scales=scales,
        cupy=_FakeCupy,
    )

    def _unexpected_subset(*_args, **_kwargs):
        raise AssertionError("partial active matvec must stream selected cache columns")

    monkeypatch.setattr(genotype_module, "_try_import_cupy", lambda: _FakeCupy())
    monkeypatch.setattr(genotype_module, "_cupy_cache_subset_columns", _unexpected_subset)

    result = standardized.matvec_numpy(coefficients, batch_size=1)

    np.testing.assert_allclose(result, standardized.materialize() @ coefficients, rtol=1e-6, atol=1e-6)


def test_gpu_int8_transpose_matvec_matches_standardized_cpu_path(monkeypatch: pytest.MonkeyPatch):
    class _FakeCupy:
        float32 = np.float32
        float64 = np.float64
        int8 = np.int8

        @staticmethod
        def asarray(array, dtype=None):
            return np.asarray(array, dtype=dtype)

        @staticmethod
        def empty(shape, dtype=None):
            return np.empty(shape, dtype=dtype)

        @staticmethod
        def sum(array, axis=None):
            return np.sum(array, axis=axis)

    raw_matrix = np.array(
        [
            [0, 1, -127, 2],
            [1, 2, 0, -127],
            [2, 0, 1, 1],
            [0, -127, 2, 2],
        ],
        dtype=np.int8,
    )
    means = np.array([0.75, 1.0, 1.0, 1.25], dtype=np.float32)
    scales = np.array([0.8, 0.9, 1.1, 1.2], dtype=np.float32)
    vector = np.array([0.5, -1.0, 2.0, 0.25], dtype=np.float32)
    standardized = Int8RawGenotypeMatrix(raw_matrix).standardized(means, scales)
    expected = standardized.materialize().T @ vector

    monkeypatch.setattr(genotype_module, "_try_import_cupy", lambda: _FakeCupy())

    result = standardized.transpose_matvec_numpy(vector, batch_size=2)

    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)


def test_try_materialize_gpu_uses_int8_batch_size_for_plink_like_backends(monkeypatch: pytest.MonkeyPatch):
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
        int8 = np.int8
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

    class _RecordingInt8Raw(_SpyInt8StreamingRawGenotypeMatrix):
        def __init__(self, matrix: np.ndarray) -> None:
            super().__init__(matrix)
            self.i8_batch_sizes: list[int] = []

        def iter_column_batches_i8(
            self,
            variant_indices=None,
            batch_size: int = 1024,
            *,
            num_threads: int | None = None,
        ):
            self.i8_batch_sizes.append(int(batch_size))
            yield from super().iter_column_batches_i8(
                variant_indices=variant_indices,
                batch_size=batch_size,
                num_threads=num_threads,
            )

    raw = _RecordingInt8Raw(np.zeros((32, 8), dtype=np.int8))
    standardized = raw.standardized(
        np.zeros(8, dtype=np.float32),
        np.ones(8, dtype=np.float32),
    )

    _install_fake_pinned_and_streams(_FakeCupy)
    monkeypatch.setattr(genotype_module, "_try_import_cupy", lambda: _FakeCupy())
    monkeypatch.setattr(genotype_module, "auto_batch_size", lambda sample_count: 3)
    monkeypatch.setattr(genotype_module, "auto_batch_size_i8", lambda sample_count: 7)

    assert standardized.try_materialize_gpu() is True
    assert raw.i8_batch_sizes == [7]


def test_try_materialize_gpu_uses_one_shot_int8_upload_for_contiguous_subset(monkeypatch: pytest.MonkeyPatch):
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
        int8 = np.int8
        cuda = _FakeCuda()

        @staticmethod
        def asarray(array, dtype=None):
            return np.asarray(array, dtype=dtype)

        @staticmethod
        def asfortranarray(array):
            return np.asfortranarray(array)

        @staticmethod
        def empty(shape, dtype=None, order=None):
            raise AssertionError("contiguous one-shot int8 upload should not allocate batched GPU staging")

        @staticmethod
        def isnan(array):
            return np.isnan(array)

    raw_i8 = np.arange(48, dtype=np.int8).reshape(6, 8, order="F")
    standardized = Int8RawGenotypeMatrix(raw_i8).standardized(
        means=np.zeros(raw_i8.shape[1], dtype=np.float32),
        scales=np.ones(raw_i8.shape[1], dtype=np.float32),
    ).subset(np.array([2, 3, 4, 5], dtype=np.int32))

    monkeypatch.setattr(genotype_module, "_try_import_cupy", lambda: _FakeCupy())
    monkeypatch.setattr(genotype_module, "_gpu_materialization_budget_bytes", lambda cupy: 1_000_000_000)
    monkeypatch.setattr(
        Int8RawGenotypeMatrix,
        "iter_column_batches_i8",
        lambda self, variant_indices=None, batch_size=1024: (_ for _ in ()).throw(
            AssertionError("contiguous one-shot int8 upload should bypass iter_column_batches_i8")
        ),
    )

    assert standardized.try_materialize_gpu() is True
    assert standardized._cupy_cache is not None
    np.testing.assert_array_equal(
        np.asarray(cast(Any, standardized._cupy_cache.raw_values)),
        raw_i8[:, 2:6],
    )


def test_try_materialize_gpu_uses_one_shot_int8_upload_near_budget(monkeypatch: pytest.MonkeyPatch):
    class _FakeDevice:
        def synchronize(self) -> None:
            return None

    class _FakeCuda:
        @staticmethod
        def Device():
            return _FakeDevice()

    class _FakeCupy:
        float32 = np.float32
        int8 = np.int8
        cuda = _FakeCuda()

        @staticmethod
        def asarray(array, dtype=None):
            return np.asarray(array, dtype=dtype)

        @staticmethod
        def asfortranarray(array):
            return np.asfortranarray(array)

        @staticmethod
        def empty(shape, dtype=None, order=None):
            raise AssertionError("near-budget one-shot int8 upload should not allocate batched GPU staging")

        @staticmethod
        def isnan(array):
            return np.isnan(array)

    raw_i8 = np.arange(10_000, dtype=np.int16).reshape(100, 100, order="F").astype(np.int8, copy=False)
    standardized = Int8RawGenotypeMatrix(raw_i8).standardized(
        means=np.zeros(raw_i8.shape[1], dtype=np.float32),
        scales=np.ones(raw_i8.shape[1], dtype=np.float32),
    )

    monkeypatch.setattr(genotype_module, "_try_import_cupy", lambda: _FakeCupy())
    monkeypatch.setattr(genotype_module, "_gpu_materialization_budget_bytes", lambda cupy: 12_000)
    monkeypatch.setattr(
        Int8RawGenotypeMatrix,
        "iter_column_batches_i8",
        lambda self, variant_indices=None, batch_size=1024: (_ for _ in ()).throw(
            AssertionError("near-budget one-shot int8 upload should bypass iter_column_batches_i8")
        ),
    )

    assert standardized.try_materialize_gpu() is True
    assert standardized._cupy_cache is not None
    np.testing.assert_array_equal(
        np.asarray(cast(Any, standardized._cupy_cache.raw_values)),
        raw_i8,
    )


def test_read_int8_columns_one_shot_handles_wrapped_raw_matrices():
    left = np.arange(24, dtype=np.int8).reshape(4, 6, order="F")
    right = (100 + np.arange(20, dtype=np.int8)).reshape(4, 5, order="F")
    concatenated = ConcatenatedRawGenotypeMatrix(
        (
            Int8RawGenotypeMatrix(left),
            Int8RawGenotypeMatrix(right),
        )
    )
    row_subset = genotype_module.RowSubsetRawGenotypeMatrix(
        concatenated,
        np.array([3, 1], dtype=np.int32),
    )
    indexed = genotype_module.IndexedRawGenotypeMatrix(
        row_subset,
        np.array([0, 5, 6, 8, 10], dtype=np.int32),
    )

    actual = genotype_module._read_int8_columns_one_shot(
        indexed,
        np.array([1, 2, 4], dtype=np.int32),
    )
    expected_full = np.concatenate([left, right], axis=1)
    expected = expected_full[np.array([3, 1])[:, None], np.array([5, 6, 10])[None, :]]
    assert actual is not None
    assert actual.flags.f_contiguous
    np.testing.assert_array_equal(actual, expected)


def test_read_int8_columns_one_shot_rejects_excessive_row_subset_expansion():
    raw_i8 = np.arange(200, dtype=np.int16).reshape(20, 10, order="F").astype(np.int8, copy=False)
    row_subset = genotype_module.RowSubsetRawGenotypeMatrix(
        Int8RawGenotypeMatrix(raw_i8),
        np.array([0, 2], dtype=np.int32),
    )

    actual = genotype_module._read_int8_columns_one_shot(
        row_subset,
        np.array([1, 2, 3], dtype=np.int32),
    )
    assert actual is None


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


def test_streaming_linear_algebra_uses_float32_gpu_batches(monkeypatch: pytest.MonkeyPatch):
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


def test_gpu_streaming_batch_size_uses_live_free_memory(monkeypatch: pytest.MonkeyPatch):
    class _FakeCupy:
        pass

    raw_matrix = _StreamingRawGenotypeMatrix(np.zeros((1000, 500), dtype=np.float32))

    monkeypatch.setattr(genotype_module, "_gpu_free_bytes", lambda _cupy: 20_000_000)

    batch_size = genotype_module._gpu_streaming_batch_size(
        raw_matrix,
        sample_count=1000,
        requested_batch_size=1024,
        cupy=_FakeCupy(),
        dtype=np.float32,
    )

    assert batch_size == 100


def test_gpu_materialization_budget_respects_live_free_memory(monkeypatch: pytest.MonkeyPatch):
    class _FakeCupy:
        pass

    # With a generous total/free pair (well above the 1.5 GB TR-Newton/HVP
    # safety reservation introduced after the live-memory budget audit) the
    # budget should be capped by free memory minus the safety margin and
    # any per-call solver headroom. For n_rows=n_cols=0 the headroom is
    # just the safety margin.
    monkeypatch.setattr(genotype_module, "_gpu_total_bytes", lambda _cupy: 16_000_000_000)
    monkeypatch.setattr(genotype_module, "_gpu_free_bytes", lambda _cupy: 4_000_000_000)

    expected = 4_000_000_000 - genotype_module._GPU_RESERVED_OVERHEAD_BYTES
    assert genotype_module._gpu_materialization_budget_bytes(_FakeCupy()) == expected


def test_gpu_standardized_batches_split_after_cupy_oom(monkeypatch: pytest.MonkeyPatch):
    class _FakeCupyOutOfMemoryError(RuntimeError):
        pass

    class _FakeCupy:
        float32 = np.float32

        @staticmethod
        def asarray(array, dtype=None):
            return np.asarray(array, dtype=dtype)

    def _flaky_standardize(values, means, scales, cupy, *, missing_sentinel=None, dtype=None):
        if values.shape[1] > 2:
            raise _FakeCupyOutOfMemoryError("Out of memory allocating test batch")
        standardized = np.asarray(values, dtype=dtype)
        standardized -= np.asarray(means, dtype=dtype)[None, :]
        standardized /= np.asarray(scales, dtype=dtype)[None, :]
        return standardized

    raw_matrix = _StreamingRawGenotypeMatrix(
        np.arange(20, dtype=np.float32).reshape(5, 4)
    )
    means = np.zeros(4, dtype=np.float32)
    scales = np.ones(4, dtype=np.float32)
    monkeypatch.setattr(genotype_module, "_standardize_batch_cupy", _flaky_standardize)

    batches = list(
        genotype_module._iter_standardized_gpu_batches(
            raw_matrix,
            np.arange(4, dtype=np.int32),
            means,
            scales,
            batch_size=4,
            cupy=_FakeCupy(),
            dtype=np.float32,
        )
    )

    assert [batch_slice for batch_slice, _ in batches] == [slice(0, 2), slice(2, 4)]
    np.testing.assert_allclose(
        np.concatenate([np.asarray(batch) for _, batch in batches], axis=1),
        raw_matrix.matrix,
    )


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


def test_gpu_native_linear_algebra_matches_dense_reference(monkeypatch: pytest.MonkeyPatch):
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
    variant_matrix = np.column_stack(
        [
            np.array([0.25, -0.75, 0.5, 1.25], dtype=np.float64),
            np.array([1.0, -0.5, 0.25, 2.0], dtype=np.float64),
        ]
    )
    sample_matrix = np.column_stack(
        [
            np.array([1.0, -0.5, 0.25, 2.0], dtype=np.float64),
            np.array([-1.5, 0.0, 1.0, 0.5], dtype=np.float64),
        ]
    )
    standardized = as_raw_genotype_matrix(raw_matrix).standardized(means, scales)
    dense_matrix = standardized.materialize().astype(np.float64, copy=False)
    standardized._cupy_cache = standardized.materialize().astype(np.float32, copy=False)
    standardized._dense_cache = None

    monkeypatch.setattr(genotype_module, "_try_import_cupy", lambda: _FakeCupy())

    gpu_matmat_result = standardized.gpu_matmat(variant_matrix, cupy=_FakeCupy(), dtype=np.float64)
    gpu_transpose_result = standardized.gpu_transpose_matmat(sample_matrix, cupy=_FakeCupy(), dtype=np.float64)

    np.testing.assert_allclose(np.asarray(gpu_matmat_result, dtype=np.float64), dense_matrix @ variant_matrix)
    np.testing.assert_allclose(np.asarray(gpu_transpose_result, dtype=np.float64), dense_matrix.T @ sample_matrix)


def test_multi_gpu_sharded_cache_linear_algebra_matches_dense_reference():
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
        def concatenate(arrays, axis=0):
            return np.concatenate(arrays, axis=axis)

        @staticmethod
        def any(array):
            return np.any(array)

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
    standardized = as_raw_genotype_matrix(raw_matrix).standardized(means, scales)
    dense_matrix = standardized.materialize().astype(np.float64, copy=False)
    fake_cupy = _FakeCupy()
    standardized._cupy_cache = genotype_module._CupyShardedStandardizedCache(
        (
            genotype_module._CupyDeviceCacheShard(
                device_id=0,
                column_start=0,
                cache=dense_matrix[:, :2].astype(np.float32, copy=False),
            ),
            genotype_module._CupyDeviceCacheShard(
                device_id=1,
                column_start=2,
                cache=dense_matrix[:, 2:].astype(np.float32, copy=False),
            ),
        ),
        cupy=fake_cupy,
    )
    standardized._dense_cache = None

    variant_matrix = np.column_stack(
        [
            np.array([0.25, 0.0, 0.0, 1.25], dtype=np.float64),
            np.array([1.0, -0.5, 0.25, 2.0], dtype=np.float64),
        ]
    )
    sample_matrix = np.column_stack(
        [
            np.array([1.0, -0.5, 0.25, 2.0], dtype=np.float64),
            np.array([-1.5, 0.0, 1.0, 0.5], dtype=np.float64),
        ]
    )

    gpu_matmat_result = standardized.gpu_matmat(variant_matrix, cupy=fake_cupy, dtype=np.float64)
    gpu_transpose_result = standardized.gpu_transpose_matmat(sample_matrix, cupy=fake_cupy, dtype=np.float64)

    np.testing.assert_allclose(np.asarray(gpu_matmat_result, dtype=np.float64), dense_matrix @ variant_matrix)
    np.testing.assert_allclose(np.asarray(gpu_transpose_result, dtype=np.float64), dense_matrix.T @ sample_matrix)


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

    jit_matvec = jax.jit(lambda beta: standardized.matvec_jax(beta))
    jit_transpose_matvec = jax.jit(lambda vector: standardized.transpose_matvec_jax(vector))

    np.testing.assert_allclose(
        np.asarray(jit_matvec(coefficient_vector), dtype=np.float64),
        np.asarray(standardized.materialize(), dtype=np.float64) @ np.asarray(coefficient_vector, dtype=np.float64),
    )
    np.testing.assert_allclose(
        np.asarray(jit_transpose_matvec(sample_vector), dtype=np.float64),
        np.asarray(standardized.materialize(), dtype=np.float64).T @ np.asarray(sample_vector, dtype=np.float64),
    )


def test_explicit_numpy_backend_ops_return_numpy_arrays():
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
    coefficients = np.array([1.0, -2.0, 0.5], dtype=np.float64)
    sample_vector = np.array([0.5, -1.0, 2.0], dtype=np.float64)

    matvec_result = standardized.matvec_numpy(coefficients)
    transpose_result = standardized.transpose_matvec_numpy(sample_vector)

    assert isinstance(matvec_result, np.ndarray)
    assert isinstance(transpose_result, np.ndarray)
    np.testing.assert_allclose(matvec_result, standardized.materialize() @ coefficients)
    np.testing.assert_allclose(transpose_result, standardized.materialize().T @ sample_vector)


def test_require_gpu_allows_cpu_only_runtime(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(genotype_module, "_gpu_verified", False)
    monkeypatch.setattr(genotype_module, "_cupy_module", None)
    monkeypatch.setattr(genotype_module, "_try_import_cupy", lambda: None)
    monkeypatch.setattr(genotype_module, "_cupy_runtime_diagnostic", lambda: "cupy unavailable")
    monkeypatch.setattr(genotype_module, "_nvidia_driver_diagnostic", lambda: "nvidia unavailable")

    assert genotype_module.require_gpu() is None
    assert genotype_module._gpu_verified is True


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


def test_try_upload_int8_parallel_memmap_bit_identical_to_source(tmp_path, monkeypatch):
    """Parallel-pread fast path must reproduce source bytes exactly.

    Builds an F-order int8 numpy memmap on disk, wraps it as an
    Int8RawGenotypeMatrix leaf, invokes ``_try_upload_int8_parallel_memmap``
    under a fake CuPy module, and verifies the resulting GPU destination
    buffer matches the source matrix byte-for-byte. The path is also
    asserted to *not* fall back to the iterator on memmap inputs.
    """
    rng = np.random.default_rng(2026)
    n_samples = 128
    n_variants_total = 64
    src = rng.integers(-1, 3, size=(n_samples, n_variants_total), dtype=np.int8)
    src_fortran = np.asfortranarray(src)
    cache_path = tmp_path / "raw_i8.npy"
    np.save(cache_path, src_fortran)
    mmap_array = np.load(cache_path, mmap_mode="r")
    assert isinstance(mmap_array, np.memmap)
    assert mmap_array.flags.f_contiguous

    raw = Int8RawGenotypeMatrix(mmap_array)

    class _FakeCuda:
        @staticmethod
        def alloc_pinned_memory(nbytes: int) -> Any:
            return bytearray(int(nbytes))

        @staticmethod
        def Device():
            class _Dev:
                def synchronize(self) -> None:
                    return None

            return _Dev()

        @staticmethod
        def Stream(non_blocking: bool = False) -> _FakeCupyStream:
            return _FakeCupyStream()

    class _FakeCupy:
        int8 = np.int8
        cuda = _FakeCuda()

        @staticmethod
        def asarray(array, dtype=None):
            return np.asarray(array, dtype=dtype)

        @staticmethod
        def empty(shape, dtype=None, order=None):
            return np.empty(shape, dtype=dtype, order="F" if order is None else order)

    # Fail loudly if the iterator chain is invoked — proves the parallel path
    # is the one doing the work, not a silent fallback.
    monkeypatch.setattr(
        Int8RawGenotypeMatrix,
        "iter_column_batches_i8",
        lambda self, variant_indices=None, batch_size=1024: (_ for _ in ()).throw(
            AssertionError("parallel-memmap path must not fall back to iter_column_batches_i8")
        ),
    )

    variant_indices = np.arange(n_variants_total, dtype=np.int32)
    gpu_dst = _FakeCupy.empty((n_samples, n_variants_total), dtype=np.int8, order="F")
    assert genotype_module._try_upload_int8_parallel_memmap(
        cupy=_FakeCupy(),
        raw=raw,
        variant_indices=variant_indices,
        gpu_destination=gpu_dst,
        sample_count=n_samples,
        n_workers=4,
    ) is True
    np.testing.assert_array_equal(np.asarray(gpu_dst), src_fortran)

    # Contiguous subset (not anchored at 0) must also work and stay bit-identical.
    sub_indices = np.arange(8, 8 + 24, dtype=np.int32)
    sub_dst = _FakeCupy.empty((n_samples, sub_indices.shape[0]), dtype=np.int8, order="F")
    assert genotype_module._try_upload_int8_parallel_memmap(
        cupy=_FakeCupy(),
        raw=raw,
        variant_indices=sub_indices,
        gpu_destination=sub_dst,
        sample_count=n_samples,
        n_workers=8,
    ) is True
    np.testing.assert_array_equal(np.asarray(sub_dst), src_fortran[:, 8:32])


def test_try_upload_int8_parallel_memmap_rejects_non_memmap_and_noncontiguous(tmp_path):
    """The fast path must return False (no upload) for inputs it can't handle.

    Eligibility gates: leaf must be an F-order int8 numpy memmap, variant
    indices must form a contiguous range, sample count must match. Anything
    else returns False so the caller falls back to the tiled iterator path.
    """

    class _FakeCuda:
        @staticmethod
        def alloc_pinned_memory(nbytes: int) -> Any:
            return bytearray(int(nbytes))

        @staticmethod
        def Stream(non_blocking: bool = False) -> _FakeCupyStream:
            return _FakeCupyStream()

    class _FakeCupy:
        int8 = np.int8
        cuda = _FakeCuda()

        @staticmethod
        def asarray(array, dtype=None):
            return np.asarray(array, dtype=dtype)

        @staticmethod
        def empty(shape, dtype=None, order=None):
            return np.empty(shape, dtype=dtype, order="F" if order is None else order)

    # In-memory (non-memmap) int8 matrix → must skip.
    in_memory_raw = Int8RawGenotypeMatrix(
        np.asfortranarray(np.zeros((16, 8), dtype=np.int8))
    )
    dst = _FakeCupy.empty((16, 8), dtype=np.int8, order="F")
    assert genotype_module._try_upload_int8_parallel_memmap(
        cupy=_FakeCupy(),
        raw=in_memory_raw,
        variant_indices=np.arange(8, dtype=np.int32),
        gpu_destination=dst,
        sample_count=16,
    ) is False

    # F-order memmap but non-contiguous variant indices → must skip.
    src = np.asfortranarray(np.arange(16 * 8, dtype=np.int8).reshape(16, 8))
    cache_path = tmp_path / "mm.npy"
    np.save(cache_path, src)
    mmap_array = np.load(cache_path, mmap_mode="r")
    raw = Int8RawGenotypeMatrix(mmap_array)
    dst2 = _FakeCupy.empty((16, 4), dtype=np.int8, order="F")
    assert genotype_module._try_upload_int8_parallel_memmap(
        cupy=_FakeCupy(),
        raw=raw,
        variant_indices=np.array([0, 2, 4, 6], dtype=np.int32),
        gpu_destination=dst2,
        sample_count=16,
    ) is False


def test_try_upload_int8_parallel_memmap_uses_direct_h2d_no_staging(tmp_path):
    """Verify the OOM-avoiding direct H2D path.

    When ``cupy.cuda.runtime.memcpyAsync`` is available, the fast upload must
    write straight into the pre-allocated GPU destination slice instead of
    allocating an intermediate ``staged_tile = cupy.asarray(pinned)`` per
    worker. The naïve staging path peaks at ``n_workers × stripe_bytes`` of
    GPU memory on top of the destination — at AoU sizes that's ~12 GB extra,
    which OOMs a T4. This test fails loudly (via an explicit raise) if the
    worker ever takes the ``cupy.asarray(pinned)`` fallback when memcpyAsync
    is available.
    """
    rng = np.random.default_rng(31337)
    n_samples = 96
    n_variants = 48
    src = np.asfortranarray(rng.integers(-1, 3, size=(n_samples, n_variants), dtype=np.int8))
    cache_path = tmp_path / "raw_i8.npy"
    np.save(cache_path, src)
    mmap_array = np.load(cache_path, mmap_mode="r")
    raw = Int8RawGenotypeMatrix(mmap_array)

    class _FakeGpuArray:
        """Minimal cupy-like ndarray backed by numpy bytes + a pointer to them."""
        def __init__(self, np_array: np.ndarray) -> None:
            self._arr = np.ascontiguousarray(np_array, dtype=np.int8) if not np_array.flags.f_contiguous else np_array
            class _Data:
                def __init__(self, arr: np.ndarray) -> None:
                    self.ptr = arr.ctypes.data
            self.data = _Data(self._arr)
            self.shape = self._arr.shape
            self.dtype = self._arr.dtype
            self.flags = self._arr.flags

        def __getitem__(self, key):
            view = self._arr[key]
            wrapped = _FakeGpuArray.__new__(_FakeGpuArray)
            wrapped._arr = view
            class _Data:
                def __init__(self, arr: np.ndarray) -> None:
                    self.ptr = arr.ctypes.data
            wrapped.data = _Data(view)
            wrapped.shape = view.shape
            wrapped.dtype = view.dtype
            wrapped.flags = view.flags
            return wrapped

    fake_gpu_dst_np = np.zeros((n_samples, n_variants), dtype=np.int8, order="F")
    fake_gpu_dst = _FakeGpuArray(fake_gpu_dst_np)

    memcpy_calls: list[tuple[int, int]] = []

    def _memcpy_async(dst_ptr: int, src_ptr: int, size: int, kind: int, stream_ptr: int) -> None:
        memcpy_calls.append((int(dst_ptr), int(size)))
        # Emulate the device copy: copy ``size`` bytes from src to dst host pointers.
        import ctypes
        ctypes.memmove(int(dst_ptr), int(src_ptr), int(size))

    class _FakeRuntime:
        memcpyHostToDevice = 1
        memcpyAsync = staticmethod(_memcpy_async)

    class _FakeStream:
        def __init__(self) -> None:
            self.ptr = 0
        def __enter__(self) -> "_FakeStream":
            return self
        def __exit__(self, *args) -> None:
            return None
        def record(self):
            class _Evt:
                def synchronize(self) -> None:
                    return None
            return _Evt()

    class _FakeCuda:
        runtime = _FakeRuntime()

        @staticmethod
        def alloc_pinned_memory(nbytes: int) -> Any:
            return bytearray(int(nbytes))

        @staticmethod
        def Device():
            class _Dev:
                def synchronize(self) -> None:
                    return None
            return _Dev()

        @staticmethod
        def Stream(non_blocking: bool = False) -> _FakeStream:
            return _FakeStream()

    class _FakeCupy:
        int8 = np.int8
        cuda = _FakeCuda()

        @staticmethod
        def asarray(array, dtype=None):
            raise AssertionError(
                "direct memcpyAsync path must not call cupy.asarray(pinned_slice) — "
                "the staging-buffer fallback would peak at n_workers × stripe_bytes "
                "of extra GPU memory and OOM the device."
            )

        @staticmethod
        def empty(shape, dtype=None, order=None):
            return _FakeGpuArray(np.zeros(shape, dtype=dtype, order="F" if order is None else order))

    assert genotype_module._try_upload_int8_parallel_memmap(
        cupy=_FakeCupy(),
        raw=raw,
        variant_indices=np.arange(n_variants, dtype=np.int32),
        gpu_destination=fake_gpu_dst,
        sample_count=n_samples,
        n_workers=4,
    ) is True
    # Direct H2D path executed exactly once per worker, with the full stripe
    # bytes going straight into the destination slice — no staging buffer.
    assert len(memcpy_calls) == 4
    assert sum(size for _, size in memcpy_calls) == n_samples * n_variants
    np.testing.assert_array_equal(fake_gpu_dst_np, src)

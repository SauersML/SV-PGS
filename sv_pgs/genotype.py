from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import io
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterator, Protocol, Sequence, TypeGuard, cast

import sv_pgs._jax  # noqa: F401
import jax
from jax import core as jax_core
import jax.numpy as jnp
from jax import dlpack as jax_dlpack
import numpy as np
from sv_pgs._jax import gpu_compute_jax_dtype, gpu_compute_numpy_dtype, jax_dense_linear_algebra_preferred
from sv_pgs.plink import PLINK_MISSING_INT8, open_bed
from sv_pgs.progress import log, mem

DEFAULT_GENOTYPE_BATCH_SIZE = 1024  # fallback when sample count is unknown

# Memory cap per PLINK batch. PLINK .bed files store genotypes on disk
# as 2 bits per sample, but we expand to int8 or float32 in memory. JAX
# batch statistics and solver setup also create float32 intermediates.
# This budget ensures each batch fits comfortably in GPU/CPU memory:
#   500 MB / (447k samples * 4 bytes) ≈ 279 variants per batch
BED_READER_TARGET_BATCH_BYTES = 500_000_000
MIN_BED_READER_BATCH_SIZE = 32  # always read at least this many variants
STANDARDIZED_STREAMING_TARGET_BATCH_BYTES = 128_000_000
LOCAL_INT8_STANDARDIZED_STREAMING_TARGET_BATCH_BYTES = 512_000_000

# If the reduced genotype matrix (after tie-group dedup) is smaller than 4 GB,
# cache it in RAM.  This avoids re-reading from disk on every EM iteration
# (typically 10-30 iterations), giving a huge speedup.
MATERIALIZE_THRESHOLD_BYTES = 4_000_000_000  # 4 GB
T4_SAFE_GPU_CACHE_BYTES = 4_500_000_000
HYBRID_SPARSE_SUPPORT_THRESHOLD = 4_096
HYBRID_SPARSE_MIN_VARIANT_COUNT = 64
REDUCED_INT8_CACHE_FREE_SPACE_RESERVE_BYTES = 1_000_000_000


def as_raw_genotype_matrix(genotypes: RawGenotypeMatrix | np.ndarray) -> RawGenotypeMatrix:
    if isinstance(genotypes, RawGenotypeMatrix):
        return genotypes
    array = np.asanyarray(genotypes)
    if array.dtype == np.int8:
        return Int8RawGenotypeMatrix(array)
    return DenseRawGenotypeMatrix(np.asarray(array, dtype=np.float32))


def _int8_npy_header_bytes(shape: tuple[int, int], *, fortran_order: bool) -> bytes:
    header_buffer = io.BytesIO()
    np.lib.format.write_array_header_2_0(
        header_buffer,
        {
            "descr": np.lib.format.dtype_to_descr(np.dtype(np.int8)),
            "fortran_order": bool(fortran_order),
            "shape": tuple(int(dimension) for dimension in shape),
        },
    )
    return header_buffer.getvalue()


def _int8_npy_expected_size(shape: tuple[int, int], *, fortran_order: bool) -> int:
    return len(_int8_npy_header_bytes(shape, fortran_order=fortran_order)) + int(np.prod(shape, dtype=np.int64))


def _has_sufficient_free_space_for_int8_npy(path: Path, shape: tuple[int, int], *, fortran_order: bool) -> tuple[bool, int, int]:
    required_bytes = _int8_npy_expected_size(shape, fortran_order=fortran_order)
    available_bytes = shutil.disk_usage(path).free
    reserve_bytes = max(REDUCED_INT8_CACHE_FREE_SPACE_RESERVE_BYTES, required_bytes // 20)
    return available_bytes >= required_bytes + reserve_bytes, required_bytes, available_bytes


def _stream_write_int8_npy(
    path: Path,
    *,
    shape: tuple[int, int],
    column_batches: Iterator[np.ndarray],
    fortran_order: bool,
) -> None:
    expected_sample_count = int(shape[0])
    expected_variant_count = int(shape[1])
    written_variant_count = 0
    header_bytes = _int8_npy_header_bytes(shape, fortran_order=fortran_order)
    with path.open("wb") as handle:
        handle.write(header_bytes)
        for batch_values in column_batches:
            batch_array = np.asarray(batch_values, dtype=np.int8)
            if batch_array.ndim != 2:
                raise ValueError("int8 cache batches must be two-dimensional.")
            if batch_array.shape[0] != expected_sample_count:
                raise ValueError(
                    f"int8 cache batch sample count mismatch: {batch_array.shape[0]} != {expected_sample_count}"
                )
            handle.write(np.asfortranarray(batch_array).tobytes(order="F"))
            written_variant_count += int(batch_array.shape[1])
        handle.flush()
        os.fsync(handle.fileno())
    if written_variant_count != expected_variant_count:
        raise ValueError(
            f"int8 cache variant count mismatch after streaming write: {written_variant_count} != {expected_variant_count}"
        )


@dataclass(slots=True)
class RawGenotypeBatch:
    variant_indices: np.ndarray
    values: np.ndarray


class RawGenotypeMatrix(ABC):
    """Abstract base for genotype matrices (samples x variants).

    Values are dosages: 0 = homozygous reference, 1 = heterozygous,
    2 = homozygous alternate, NaN = missing.  Subclasses handle different
    storage backends (in-memory numpy array vs on-disk PLINK .bed file).

    All access is through streaming iterators (iter_column_batches) that
    read a few hundred variants at a time, keeping memory bounded even for
    biobank-scale data (e.g. 447k samples x 900k variants).
    """
    @property
    @abstractmethod
    def shape(self) -> tuple[int, int]:
        raise NotImplementedError

    @abstractmethod
    def iter_column_batches(
        self,
        variant_indices: Sequence[int] | np.ndarray | None = None,
        batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE,
    ) -> Iterator[RawGenotypeBatch]:
        raise NotImplementedError

    @abstractmethod
    def materialize(
        self,
        variant_indices: Sequence[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        raise NotImplementedError

    def standardized(
        self,
        means: np.ndarray,
        scales: np.ndarray,
        support_counts: np.ndarray | None = None,
    ) -> StandardizedGenotypeMatrix:
        return StandardizedGenotypeMatrix(
            raw=self,
            means=np.asarray(means, dtype=np.float32),
            scales=np.asarray(scales, dtype=np.float32),
            variant_indices=np.arange(self.shape[1], dtype=np.int32),
            support_counts=None if support_counts is None else np.asarray(support_counts, dtype=np.int32),
        )

    def __array__(self, dtype: np.dtype | type | None = None) -> np.ndarray:
        matrix = self.materialize()
        if dtype is None:
            return matrix
        return np.asarray(matrix, dtype=dtype)


class Int8BatchCapable(Protocol):
    def iter_column_batches_i8(
        self,
        variant_indices: Sequence[int] | np.ndarray | None = None,
        batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE,
    ) -> Iterator[RawGenotypeBatch]: ...


def _supports_int8_batches(matrix: object) -> TypeGuard[Int8BatchCapable]:
    return hasattr(matrix, "iter_column_batches_i8")


@dataclass(slots=True)
class DenseRawGenotypeMatrix(RawGenotypeMatrix):
    matrix: np.ndarray

    def __post_init__(self) -> None:
        matrix_array = np.asanyarray(self.matrix)
        if matrix_array.dtype == np.int8:
            self.matrix = matrix_array  # preserve memmap-backed int8 arrays
        else:
            self.matrix = np.asarray(matrix_array, dtype=np.float32)
        if self.matrix.ndim != 2:
            raise ValueError("genotypes must be 2D.")

    def _to_float32(self, batch: np.ndarray) -> np.ndarray:
        """Convert a column slice to float32, replacing missing sentinels with NaN."""
        if self.matrix.dtype == np.int8:
            result = batch.astype(np.float32)
            result[batch == PLINK_MISSING_INT8] = np.nan
            return result
        return np.asarray(batch, dtype=np.float32)

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.matrix.shape[0]), int(self.matrix.shape[1])

    def iter_column_batches(
        self,
        variant_indices: Sequence[int] | np.ndarray | None = None,
        batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE,
    ) -> Iterator[RawGenotypeBatch]:
        resolved_indices = _resolve_variant_indices(self.shape[1], variant_indices)
        safe_batch_size = max(int(batch_size), 1)
        for start_index in range(0, resolved_indices.shape[0], safe_batch_size):
            batch_indices = resolved_indices[start_index : start_index + safe_batch_size]
            yield RawGenotypeBatch(
                variant_indices=batch_indices,
                values=self._to_float32(self.matrix[:, batch_indices]),
            )

    def materialize(
        self,
        variant_indices: Sequence[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        resolved_indices = _resolve_variant_indices(self.shape[1], variant_indices)
        return self._to_float32(self.matrix[:, resolved_indices])


@dataclass(slots=True)
class Int8RawGenotypeMatrix(RawGenotypeMatrix):
    matrix: np.ndarray

    def __post_init__(self) -> None:
        matrix_array = np.asanyarray(self.matrix)
        self.matrix = matrix_array if matrix_array.dtype == np.int8 else np.asarray(matrix_array, dtype=np.int8)
        if self.matrix.ndim != 2:
            raise ValueError("genotypes must be 2D.")

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.matrix.shape[0]), int(self.matrix.shape[1])

    def iter_column_batches_i8(
        self,
        variant_indices: Sequence[int] | np.ndarray | None = None,
        batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE,
    ) -> Iterator[RawGenotypeBatch]:
        resolved_indices = _resolve_variant_indices(self.shape[1], variant_indices)
        safe_batch_size = max(int(batch_size), 1)
        for start_index in range(0, resolved_indices.shape[0], safe_batch_size):
            batch_indices = resolved_indices[start_index : start_index + safe_batch_size]
            yield RawGenotypeBatch(
                variant_indices=batch_indices,
                values=np.asarray(self.matrix[:, batch_indices], dtype=np.int8),
            )

    def iter_column_batches(
        self,
        variant_indices: Sequence[int] | np.ndarray | None = None,
        batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE,
    ) -> Iterator[RawGenotypeBatch]:
        for batch in self.iter_column_batches_i8(variant_indices, batch_size=batch_size):
            yield RawGenotypeBatch(
                variant_indices=batch.variant_indices,
                values=_int8_batch_to_float32(batch.values),
            )

    def materialize(
        self,
        variant_indices: Sequence[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        resolved_indices = _resolve_variant_indices(self.shape[1], variant_indices)
        return _int8_batch_to_float32(self.matrix[:, resolved_indices])


@dataclass(slots=True)
class PlinkRawGenotypeMatrix(RawGenotypeMatrix):
    bed_path: Path
    sample_indices: np.ndarray
    variant_count: int
    total_sample_count: int
    batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE
    _reader: Any | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.sample_indices = np.asarray(self.sample_indices, dtype=np.intp)
        if self.sample_indices.ndim != 1:
            raise ValueError("sample_indices must be 1D.")

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.sample_indices.shape[0]), int(self.variant_count)

    def _read_batch(self, reader: Any, batch_indices: np.ndarray) -> np.ndarray:
        """Read one batch as int8, convert to float32 with NaN for missing."""
        raw_i8 = self._read_batch_i8(reader, batch_indices)
        result = np.asarray(raw_i8, dtype=np.float32)
        result[raw_i8 == PLINK_MISSING_INT8] = np.nan
        return result

    def _read_batch_i8(self, reader: Any, batch_indices: np.ndarray) -> np.ndarray:
        """Read one batch as raw int8 (0/1/2/PLINK_MISSING_INT8). No float conversion."""
        sample_index = _contiguous_index_or_slice(self.sample_indices)
        col_index = _contiguous_index_or_slice(batch_indices)
        return np.asarray(
            reader.read(index=(sample_index, col_index), dtype="int8", order="F", num_threads=None),
            dtype=np.int8,
        )

    def iter_column_batches_i8(
        self,
        variant_indices: Sequence[int] | np.ndarray | None = None,
        batch_size: int | None = None,
    ) -> Iterator[RawGenotypeBatch]:
        """Iterate as int8 batches (4x less memory, no float conversion).

        Values are 0/1/2/PLINK_MISSING_INT8 (missing). Callers must handle the shared sentinel.
        """
        resolved_indices = _resolve_variant_indices(self.variant_count, variant_indices)
        # int8 reads are 4x smaller than float32, but JAX kernels still expand to
        # float32 intermediates (~10 bytes/element peak), so do NOT inflate batch size.
        requested = max(int(self.batch_size if batch_size is None else batch_size), 1)
        bytes_per_variant = self.shape[0]  # 1 byte per sample for int8
        max_variants = max(BED_READER_TARGET_BATCH_BYTES // max(bytes_per_variant, 1), MIN_BED_READER_BATCH_SIZE)
        safe_batch_size = min(requested, max_variants)
        reader = self._bed_reader()
        total = resolved_indices.shape[0]
        batch_mb = self.shape[0] * safe_batch_size / (1024 * 1024)
        n_batches = (total + safe_batch_size - 1) // safe_batch_size
        log(f"    int8 batch: {safe_batch_size} variants x {self.shape[0]} samples = {batch_mb:.0f} MB/batch, {n_batches} batches  mem={mem()}")

        if total <= safe_batch_size:
            values = self._read_batch_i8(reader, resolved_indices)
            yield RawGenotypeBatch(variant_indices=resolved_indices, values=values)
            return

        # Prefetch with background thread
        batch_index_list = [
            resolved_indices[s : s + safe_batch_size]
            for s in range(0, total, safe_batch_size)
        ]
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._read_batch_i8, reader, batch_index_list[0])
            for i in range(len(batch_index_list)):
                values = future.result()
                if i + 1 < len(batch_index_list):
                    future = executor.submit(self._read_batch_i8, reader, batch_index_list[i + 1])
                yield RawGenotypeBatch(variant_indices=batch_index_list[i], values=values)

    def iter_column_batches(
        self,
        variant_indices: Sequence[int] | np.ndarray | None = None,
        batch_size: int | None = None,
    ) -> Iterator[RawGenotypeBatch]:
        resolved_indices = _resolve_variant_indices(self.variant_count, variant_indices)
        requested_batch_size = max(int(self.batch_size if batch_size is None else batch_size), 1)
        safe_batch_size = _effective_bed_reader_batch_size(
            sample_count=self.shape[0],
            requested_batch_size=requested_batch_size,
        )
        reader = self._bed_reader()
        total = resolved_indices.shape[0]

        # Build list of batch index arrays
        batch_index_list: list[np.ndarray] = []
        for start_index in range(0, total, safe_batch_size):
            batch_index_list.append(resolved_indices[start_index : start_index + safe_batch_size])

        if len(batch_index_list) <= 1:
            # Single batch — no prefetch overhead needed
            for batch_indices in batch_index_list:
                values = self._read_batch(reader, batch_indices)
                yield RawGenotypeBatch(variant_indices=batch_indices, values=values)
            return

        # Prefetch: read batch N+1 in background thread while caller processes batch N
        with ThreadPoolExecutor(max_workers=1) as executor:
            # Submit first read
            future = executor.submit(self._read_batch, reader, batch_index_list[0])
            for i in range(len(batch_index_list)):
                # Wait for current batch
                values = future.result()
                # Submit next batch read (if any) before yielding
                if i + 1 < len(batch_index_list):
                    future = executor.submit(self._read_batch, reader, batch_index_list[i + 1])
                yield RawGenotypeBatch(variant_indices=batch_index_list[i], values=values)

    def materialize(
        self,
        variant_indices: Sequence[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        resolved_indices = _resolve_variant_indices(self.variant_count, variant_indices)
        reader = self._bed_reader()
        return self._read_batch(reader, resolved_indices)

    def _bed_reader(self) -> Any:
        if self._reader is None:
            log(f"    opening PLINK reader (lazy, no metadata): iid_count={self.total_sample_count} sid_count={self.variant_count}  mem={mem()}")
            self._reader = open_bed(
                self.bed_path,
                iid_count=self.total_sample_count,
                sid_count=self.variant_count,
                properties={},
                num_threads=None,
            )
            log(f"    PLINK reader opened  mem={mem()}")
        return self._reader


@dataclass(slots=True)
class ConcatenatedRawGenotypeMatrix(RawGenotypeMatrix):
    children: tuple[RawGenotypeMatrix, ...]
    _sample_count: int = field(init=False, repr=False)
    _variant_offsets: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.children:
            raise ValueError("children cannot be empty.")
        self._sample_count = int(self.children[0].shape[0])
        variant_offsets = [0]
        for child in self.children:
            if int(child.shape[0]) != self._sample_count:
                raise ValueError("all concatenated genotype matrices must have the same sample count.")
            variant_offsets.append(variant_offsets[-1] + int(child.shape[1]))
        self._variant_offsets = np.asarray(variant_offsets, dtype=np.int64)

    @property
    def shape(self) -> tuple[int, int]:
        return self._sample_count, int(self._variant_offsets[-1])

    def iter_column_batches(
        self,
        variant_indices: Sequence[int] | np.ndarray | None = None,
        batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE,
    ) -> Iterator[RawGenotypeBatch]:
        resolved_indices = _resolve_variant_indices(self.shape[1], variant_indices)
        safe_batch_size = max(int(batch_size), 1)
        for start_index in range(0, resolved_indices.shape[0], safe_batch_size):
            batch_indices = resolved_indices[start_index : start_index + safe_batch_size]
            child_ids = np.searchsorted(self._variant_offsets[1:], batch_indices, side="right")
            batch_values = np.empty((self.shape[0], batch_indices.shape[0]), dtype=np.float32)
            for child_index in np.unique(child_ids):
                child_positions = np.nonzero(child_ids == child_index)[0]
                child_variant_indices = batch_indices[child_positions] - int(self._variant_offsets[child_index])
                batch_values[:, child_positions] = self.children[int(child_index)].materialize(child_variant_indices)
            yield RawGenotypeBatch(
                variant_indices=batch_indices,
                values=batch_values,
            )

    def iter_column_batches_i8(
        self,
        variant_indices: Sequence[int] | np.ndarray | None = None,
        batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE,
    ) -> Iterator[RawGenotypeBatch]:
        if not all(_supports_int8_batches(child) for child in self.children):
            raise RuntimeError("int8 batch iteration requires every concatenated child to support iter_column_batches_i8.")
        resolved_indices = _resolve_variant_indices(self.shape[1], variant_indices)
        safe_batch_size = max(int(batch_size), 1)
        for start_index in range(0, resolved_indices.shape[0], safe_batch_size):
            batch_indices = resolved_indices[start_index : start_index + safe_batch_size]
            child_ids = np.searchsorted(self._variant_offsets[1:], batch_indices, side="right")
            batch_values = np.empty((self.shape[0], batch_indices.shape[0]), dtype=np.int8)
            for child_index in np.unique(child_ids):
                child_positions = np.nonzero(child_ids == child_index)[0]
                child_variant_indices = batch_indices[child_positions] - int(self._variant_offsets[child_index])
                child = cast(Int8BatchCapable, self.children[int(child_index)])
                child_batch_iter = child.iter_column_batches_i8(child_variant_indices, batch_size=max(child_variant_indices.shape[0], 1))
                child_batch = next(child_batch_iter)
                batch_values[:, child_positions] = np.asarray(child_batch.values, dtype=np.int8)
            yield RawGenotypeBatch(
                variant_indices=batch_indices,
                values=batch_values,
            )

    def materialize(
        self,
        variant_indices: Sequence[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        resolved_indices = _resolve_variant_indices(self.shape[1], variant_indices)
        child_ids = np.searchsorted(self._variant_offsets[1:], resolved_indices, side="right")
        matrix = np.empty((self.shape[0], resolved_indices.shape[0]), dtype=np.float32)
        for child_index in np.unique(child_ids):
            child_positions = np.nonzero(child_ids == child_index)[0]
            child_variant_indices = resolved_indices[child_positions] - int(self._variant_offsets[child_index])
            matrix[:, child_positions] = self.children[int(child_index)].materialize(child_variant_indices)
        return matrix


def _try_import_cupy() -> Any | None:
    """Import CuPy if available (GPU matmul via cuBLAS, bypassing JAX/XLA)."""
    try:
        import cupy
        if cupy.cuda.runtime.getDeviceCount() > 0:
            return cupy
    except Exception:
        pass
    return None


def _as_gpu_compute_jax(array) -> jnp.ndarray:
    return jnp.asarray(array, dtype=gpu_compute_jax_dtype())


def _cupy_to_jax(array) -> jnp.ndarray:
    """Convert CuPy result to JAX, preferring zero-copy DLPack interop."""
    if hasattr(array, "__dlpack__"):
        return jax_dlpack.from_dlpack(array).astype(gpu_compute_jax_dtype())
    return jnp.asarray(array.get(), dtype=gpu_compute_jax_dtype())


def _to_cupy_float32(array):
    """Convert JAX/numpy array to CuPy float32 for CuPy matmul."""
    cupy = _try_import_cupy()
    if cupy is None:
        raise RuntimeError("CuPy is not available.")
    if type(array).__module__.startswith("cupy"):
        return array.astype(cupy.float32, copy=False)
    if isinstance(array, jax.Array) and hasattr(cupy, "from_dlpack"):
        return cupy.from_dlpack(array).astype(cupy.float32, copy=False)
    return cupy.asarray(np.asarray(array, dtype=np.float32))


def _to_cupy_float64(array):
    """Convert JAX/numpy array to CuPy float64 for numerically sensitive GPU solves."""
    cupy = _try_import_cupy()
    if cupy is None:
        raise RuntimeError("CuPy is not available.")
    if type(array).__module__.startswith("cupy"):
        return array.astype(cupy.float64, copy=False)
    if isinstance(array, jax.Array) and hasattr(cupy, "from_dlpack"):
        return cupy.from_dlpack(array).astype(cupy.float64, copy=False)
    return cupy.asarray(np.asarray(array, dtype=np.float64))


def _cupy_compute_dtype(cupy):
    return cupy.float32 if gpu_compute_numpy_dtype() == np.dtype(np.float32) else cupy.float64


def _to_cupy_compute(array):
    cupy = _try_import_cupy()
    if cupy is None:
        raise RuntimeError("CuPy is not available.")
    compute_dtype = _cupy_compute_dtype(cupy)
    if compute_dtype == cupy.float32:
        return _to_cupy_float32(array)
    return _to_cupy_float64(array)


def _standardize_batch_cupy(
    batch_values: np.ndarray,
    means,
    scales,
    cupy,
    *,
    missing_sentinel: int | None = None,
    dtype=None,
):
    """Standardize a raw batch directly on GPU."""
    resolved_dtype = cupy.float32 if dtype is None else dtype
    standardized = cupy.asarray(batch_values, dtype=resolved_dtype)
    if missing_sentinel is None:
        missing_mask = (
            cupy.isnan(standardized)
            if hasattr(cupy, "isnan")
            else np.isnan(np.asarray(standardized))
        )
    else:
        missing_mask = standardized == float(missing_sentinel)
    standardized -= means[None, :]
    standardized /= scales[None, :]
    standardized[missing_mask] = 0.0
    return standardized


def _iter_standardized_gpu_batches(
    raw: RawGenotypeMatrix,
    variant_indices: np.ndarray,
    means,
    scales,
    *,
    batch_size: int,
    cupy,
    dtype=None,
):
    resolved_dtype = cupy.float32 if dtype is None else dtype
    selected_means = cupy.asarray(means[variant_indices], dtype=resolved_dtype)
    selected_scales = cupy.asarray(scales[variant_indices], dtype=resolved_dtype)
    local_start = 0
    if _supports_int8_batches(raw):
        batch_iter = raw.iter_column_batches_i8(variant_indices, batch_size=batch_size)
        missing_sentinel: int | None = int(PLINK_MISSING_INT8)
    else:
        batch_iter = raw.iter_column_batches(variant_indices, batch_size=batch_size)
        missing_sentinel = None
    for raw_batch in batch_iter:
        batch_width = raw_batch.values.shape[1]
        local_stop = local_start + batch_width
        batch_slice = slice(local_start, local_stop)
        yield batch_slice, _standardize_batch_cupy(
            raw_batch.values,
            selected_means[batch_slice],
            selected_scales[batch_slice],
            cupy,
            missing_sentinel=missing_sentinel,
            dtype=resolved_dtype,
        )
        local_start = local_stop


def _gpu_materialization_budget_bytes(cupy) -> int:
    """Conservative GPU cache budget for a single-device training run.

    The T4 path needs room for the cached genotype matrix plus iterative solver
    workspace. Cap the cache at a T4-safe fixed ceiling and at most 60% of the
    currently free device memory.
    """
    free_bytes, _ = cupy.cuda.runtime.memGetInfo()
    return min(int(free_bytes * 0.6), T4_SAFE_GPU_CACHE_BYTES)


@dataclass(slots=True)
class _SparseCarrierBackend:
    sample_count: int
    means: np.ndarray
    scales: np.ndarray
    variant_ptr: np.ndarray
    variant_sample_indices: np.ndarray
    variant_dosages: np.ndarray
    missing_variant_ptr: np.ndarray
    missing_variant_sample_indices: np.ndarray
    sample_ptr: np.ndarray
    sample_variant_indices: np.ndarray
    sample_dosages: np.ndarray
    sample_missing_ptr: np.ndarray
    sample_missing_variant_indices: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.sample_count, int(self.means.shape[0])

    def materialize_columns(self, local_variant_indices: np.ndarray) -> np.ndarray:
        resolved_local_indices = np.asarray(local_variant_indices, dtype=np.int32)
        if resolved_local_indices.ndim != 1:
            raise ValueError("local_variant_indices must be 1D.")
        output = np.empty((self.sample_count, resolved_local_indices.shape[0]), dtype=np.float32)
        for output_column, local_variant_index in enumerate(resolved_local_indices.tolist()):
            baseline = -float(self.means[local_variant_index] / self.scales[local_variant_index])
            column = np.full(self.sample_count, baseline, dtype=np.float32)
            start = int(self.variant_ptr[local_variant_index])
            stop = int(self.variant_ptr[local_variant_index + 1])
            if stop > start:
                sample_indices = self.variant_sample_indices[start:stop]
                dosages = self.variant_dosages[start:stop]
                column[sample_indices] += dosages / self.scales[local_variant_index]
            missing_start = int(self.missing_variant_ptr[local_variant_index])
            missing_stop = int(self.missing_variant_ptr[local_variant_index + 1])
            if missing_stop > missing_start:
                column[self.missing_variant_sample_indices[missing_start:missing_stop]] = 0.0
            output[:, output_column] = column
        return output

    def matvec(self, coefficients: np.ndarray) -> np.ndarray:
        coefficient_array = np.asarray(coefficients, dtype=gpu_compute_numpy_dtype()).reshape(-1)
        if coefficient_array.shape[0] != self.shape[1]:
            raise ValueError("coefficient vector must match sparse variant count.")
        if not np.any(coefficient_array):
            return np.zeros(self.sample_count, dtype=coefficient_array.dtype)
        scaled_coefficients = coefficient_array / np.asarray(self.scales, dtype=coefficient_array.dtype)
        result = np.full(
            self.sample_count,
            -float(np.dot(np.asarray(self.means, dtype=coefficient_array.dtype), scaled_coefficients)),
            dtype=coefficient_array.dtype,
        )
        if self.sample_variant_indices.size > 0:
            carrier_weights = self.sample_dosages.astype(coefficient_array.dtype, copy=False) * scaled_coefficients[self.sample_variant_indices]
            nonempty_samples = self.sample_ptr[1:] > self.sample_ptr[:-1]
            if np.any(nonempty_samples):
                starts = self.sample_ptr[:-1][nonempty_samples]
                result[nonempty_samples] += np.add.reduceat(carrier_weights, starts)
        if self.sample_missing_variant_indices.size > 0:
            missing_weights = np.asarray(
                self.means[self.sample_missing_variant_indices] * scaled_coefficients[self.sample_missing_variant_indices],
                dtype=coefficient_array.dtype,
            )
            nonempty_missing = self.sample_missing_ptr[1:] > self.sample_missing_ptr[:-1]
            if np.any(nonempty_missing):
                starts = self.sample_missing_ptr[:-1][nonempty_missing]
                result[nonempty_missing] += np.add.reduceat(missing_weights, starts)
        return result

    def matmat(self, matrix: np.ndarray) -> np.ndarray:
        matrix_array = np.asarray(matrix, dtype=gpu_compute_numpy_dtype())
        if matrix_array.ndim != 2 or matrix_array.shape[0] != self.shape[1]:
            raise ValueError("variant matrix must match sparse variant count.")
        if not np.any(matrix_array):
            return np.zeros((self.sample_count, matrix_array.shape[1]), dtype=matrix_array.dtype)
        scaled_matrix = matrix_array / np.asarray(self.scales, dtype=matrix_array.dtype)[:, None]
        output = np.broadcast_to(
            -np.sum(np.asarray(self.means, dtype=matrix_array.dtype)[:, None] * scaled_matrix, axis=0, dtype=matrix_array.dtype),
            (self.sample_count, matrix_array.shape[1]),
        ).copy()
        if self.sample_variant_indices.size > 0:
            carrier_weights = self.sample_dosages.astype(matrix_array.dtype, copy=False)[:, None] * scaled_matrix[self.sample_variant_indices, :]
            nonempty_samples = self.sample_ptr[1:] > self.sample_ptr[:-1]
            if np.any(nonempty_samples):
                starts = self.sample_ptr[:-1][nonempty_samples]
                output[nonempty_samples, :] += np.add.reduceat(carrier_weights, starts, axis=0)
        if self.sample_missing_variant_indices.size > 0:
            missing_weights = np.asarray(
                self.means[self.sample_missing_variant_indices, None] * scaled_matrix[self.sample_missing_variant_indices, :],
                dtype=matrix_array.dtype,
            )
            nonempty_missing = self.sample_missing_ptr[1:] > self.sample_missing_ptr[:-1]
            if np.any(nonempty_missing):
                starts = self.sample_missing_ptr[:-1][nonempty_missing]
                output[nonempty_missing, :] += np.add.reduceat(missing_weights, starts, axis=0)
        return output

    def transpose_matvec(self, vector: np.ndarray) -> np.ndarray:
        vector_array = np.asarray(vector, dtype=gpu_compute_numpy_dtype()).reshape(-1)
        if vector_array.shape[0] != self.sample_count:
            raise ValueError("sample vector must match sparse sample count.")
        if not np.any(vector_array):
            return np.zeros(self.shape[1], dtype=vector_array.dtype)
        output = np.zeros(self.shape[1], dtype=vector_array.dtype)
        global_sum = float(np.sum(vector_array, dtype=vector_array.dtype))
        if self.variant_sample_indices.size > 0:
            carrier_weights = self.variant_dosages.astype(vector_array.dtype, copy=False) * vector_array[self.variant_sample_indices]
            nonempty_variants = self.variant_ptr[1:] > self.variant_ptr[:-1]
            if np.any(nonempty_variants):
                starts = self.variant_ptr[:-1][nonempty_variants]
                output[nonempty_variants] += np.add.reduceat(carrier_weights, starts)
        observed_sums = np.full(self.shape[1], global_sum, dtype=vector_array.dtype)
        if self.missing_variant_sample_indices.size > 0:
            missing_weights = vector_array[self.missing_variant_sample_indices]
            nonempty_missing = self.missing_variant_ptr[1:] > self.missing_variant_ptr[:-1]
            if np.any(nonempty_missing):
                starts = self.missing_variant_ptr[:-1][nonempty_missing]
                observed_sums[nonempty_missing] -= np.add.reduceat(missing_weights, starts)
        output -= np.asarray(self.means, dtype=vector_array.dtype) * observed_sums
        output /= np.asarray(self.scales, dtype=vector_array.dtype)
        return output

    def transpose_matmat(self, matrix: np.ndarray) -> np.ndarray:
        matrix_array = np.asarray(matrix, dtype=gpu_compute_numpy_dtype())
        if matrix_array.ndim != 2 or matrix_array.shape[0] != self.sample_count:
            raise ValueError("sample matrix must match sparse sample count.")
        if not np.any(matrix_array):
            return np.zeros((self.shape[1], matrix_array.shape[1]), dtype=matrix_array.dtype)
        output = np.zeros((self.shape[1], matrix_array.shape[1]), dtype=matrix_array.dtype)
        global_sum = np.sum(matrix_array, axis=0, dtype=matrix_array.dtype)
        if self.variant_sample_indices.size > 0:
            carrier_weights = self.variant_dosages.astype(matrix_array.dtype, copy=False)[:, None] * matrix_array[self.variant_sample_indices, :]
            nonempty_variants = self.variant_ptr[1:] > self.variant_ptr[:-1]
            if np.any(nonempty_variants):
                starts = self.variant_ptr[:-1][nonempty_variants]
                output[nonempty_variants, :] += np.add.reduceat(carrier_weights, starts, axis=0)
        observed_sums = np.broadcast_to(global_sum, output.shape).copy()
        if self.missing_variant_sample_indices.size > 0:
            missing_weights = matrix_array[self.missing_variant_sample_indices, :]
            nonempty_missing = self.missing_variant_ptr[1:] > self.missing_variant_ptr[:-1]
            if np.any(nonempty_missing):
                starts = self.missing_variant_ptr[:-1][nonempty_missing]
                observed_sums[nonempty_missing, :] -= np.add.reduceat(missing_weights, starts, axis=0)
        output -= np.asarray(self.means, dtype=matrix_array.dtype)[:, None] * observed_sums
        output /= np.asarray(self.scales, dtype=matrix_array.dtype)[:, None]
        return output

    def subset(self, local_variant_indices: np.ndarray) -> _SparseCarrierBackend:
        resolved_local_indices = np.asarray(local_variant_indices, dtype=np.int32)
        if resolved_local_indices.ndim != 1:
            raise ValueError("local_variant_indices must be 1D.")
        if resolved_local_indices.size == 0:
            empty_int = np.zeros(0, dtype=np.int32)
            empty_ptr = np.zeros(1, dtype=np.int64)
            return _SparseCarrierBackend(
                sample_count=self.sample_count,
                means=np.zeros(0, dtype=np.float32),
                scales=np.zeros(0, dtype=np.float32),
                variant_ptr=empty_ptr,
                variant_sample_indices=empty_int,
                variant_dosages=np.zeros(0, dtype=np.float32),
                missing_variant_ptr=empty_ptr.copy(),
                missing_variant_sample_indices=empty_int.copy(),
                sample_ptr=np.zeros(self.sample_count + 1, dtype=np.int64),
                sample_variant_indices=empty_int.copy(),
                sample_dosages=np.zeros(0, dtype=np.float32),
                sample_missing_ptr=np.zeros(self.sample_count + 1, dtype=np.int64),
                sample_missing_variant_indices=empty_int.copy(),
            )
        old_to_new = np.full(self.shape[1], -1, dtype=np.int32)
        old_to_new[resolved_local_indices] = np.arange(resolved_local_indices.shape[0], dtype=np.int32)
        selected_variant_sample_indices: list[np.ndarray] = []
        selected_variant_dosages: list[np.ndarray] = []
        variant_counts = np.zeros(resolved_local_indices.shape[0], dtype=np.int64)
        selected_missing_sample_indices: list[np.ndarray] = []
        missing_counts = np.zeros(resolved_local_indices.shape[0], dtype=np.int64)
        for new_variant_index, old_variant_index in enumerate(resolved_local_indices.tolist()):
            carrier_start = int(self.variant_ptr[old_variant_index])
            carrier_stop = int(self.variant_ptr[old_variant_index + 1])
            selected_variant_sample_indices.append(self.variant_sample_indices[carrier_start:carrier_stop].astype(np.int32, copy=False))
            selected_variant_dosages.append(self.variant_dosages[carrier_start:carrier_stop].astype(np.float32, copy=False))
            variant_counts[new_variant_index] = carrier_stop - carrier_start
            missing_start = int(self.missing_variant_ptr[old_variant_index])
            missing_stop = int(self.missing_variant_ptr[old_variant_index + 1])
            selected_missing_sample_indices.append(self.missing_variant_sample_indices[missing_start:missing_stop].astype(np.int32, copy=False))
            missing_counts[new_variant_index] = missing_stop - missing_start
        variant_sample_indices = (
            np.concatenate(selected_variant_sample_indices).astype(np.int32, copy=False)
            if selected_variant_sample_indices else np.zeros(0, dtype=np.int32)
        )
        variant_dosages = (
            np.concatenate(selected_variant_dosages).astype(np.float32, copy=False)
            if selected_variant_dosages else np.zeros(0, dtype=np.float32)
        )
        missing_variant_sample_indices = (
            np.concatenate(selected_missing_sample_indices).astype(np.int32, copy=False)
            if selected_missing_sample_indices else np.zeros(0, dtype=np.int32)
        )
        sample_mask = old_to_new[self.sample_variant_indices] >= 0
        sample_variant_indices = old_to_new[self.sample_variant_indices[sample_mask]].astype(np.int32, copy=False)
        sample_dosages = self.sample_dosages[sample_mask].astype(np.float32, copy=False)
        if sample_mask.any():
            kept_sample_ids = np.repeat(
                np.arange(self.sample_count, dtype=np.int32),
                np.diff(self.sample_ptr).astype(np.int32, copy=False),
            )[sample_mask]
            sample_counts = np.bincount(kept_sample_ids, minlength=self.sample_count).astype(np.int64, copy=False)
        else:
            sample_counts = np.zeros(self.sample_count, dtype=np.int64)
        sample_missing_mask = old_to_new[self.sample_missing_variant_indices] >= 0
        sample_missing_variant_indices = old_to_new[self.sample_missing_variant_indices[sample_missing_mask]].astype(np.int32, copy=False)
        if sample_missing_mask.any():
            kept_missing_sample_ids = np.repeat(
                np.arange(self.sample_count, dtype=np.int32),
                np.diff(self.sample_missing_ptr).astype(np.int32, copy=False),
            )[sample_missing_mask]
            sample_missing_counts = np.bincount(kept_missing_sample_ids, minlength=self.sample_count).astype(np.int64, copy=False)
        else:
            sample_missing_counts = np.zeros(self.sample_count, dtype=np.int64)
        variant_ptr = np.zeros(resolved_local_indices.shape[0] + 1, dtype=np.int64)
        variant_ptr[1:] = np.cumsum(variant_counts, dtype=np.int64)
        missing_variant_ptr = np.zeros(resolved_local_indices.shape[0] + 1, dtype=np.int64)
        missing_variant_ptr[1:] = np.cumsum(missing_counts, dtype=np.int64)
        sample_ptr = np.zeros(self.sample_count + 1, dtype=np.int64)
        sample_ptr[1:] = np.cumsum(sample_counts, dtype=np.int64)
        sample_missing_ptr = np.zeros(self.sample_count + 1, dtype=np.int64)
        sample_missing_ptr[1:] = np.cumsum(sample_missing_counts, dtype=np.int64)
        return _SparseCarrierBackend(
            sample_count=self.sample_count,
            means=np.asarray(self.means[resolved_local_indices], dtype=np.float32),
            scales=np.asarray(self.scales[resolved_local_indices], dtype=np.float32),
            variant_ptr=variant_ptr,
            variant_sample_indices=variant_sample_indices,
            variant_dosages=variant_dosages,
            missing_variant_ptr=missing_variant_ptr,
            missing_variant_sample_indices=missing_variant_sample_indices,
            sample_ptr=sample_ptr,
            sample_variant_indices=sample_variant_indices,
            sample_dosages=sample_dosages,
            sample_missing_ptr=sample_missing_ptr,
            sample_missing_variant_indices=sample_missing_variant_indices,
        )


def _build_sparse_backend(
    raw: Int8BatchCapable,
    raw_variant_indices: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
    sample_count: int,
) -> _SparseCarrierBackend:
    resolved_variant_indices = np.asarray(raw_variant_indices, dtype=np.int32)
    carrier_sample_chunks: list[np.ndarray] = []
    carrier_dosage_chunks: list[np.ndarray] = []
    carrier_counts = np.zeros(resolved_variant_indices.shape[0], dtype=np.int64)
    missing_sample_chunks: list[np.ndarray] = []
    missing_counts = np.zeros(resolved_variant_indices.shape[0], dtype=np.int64)
    local_start = 0
    batch_size = max(auto_batch_size_i8(sample_count), 1)
    for raw_batch in raw.iter_column_batches_i8(resolved_variant_indices, batch_size=batch_size):
        batch_values = np.asarray(raw_batch.values, dtype=np.int8)
        for local_batch_index in range(batch_values.shape[1]):
            local_variant_index = local_start + local_batch_index
            column = batch_values[:, local_batch_index]
            carrier_sample_indices = np.flatnonzero((column != PLINK_MISSING_INT8) & (column > 0)).astype(np.int32)
            missing_sample_indices = np.flatnonzero(column == PLINK_MISSING_INT8).astype(np.int32)
            carrier_sample_chunks.append(carrier_sample_indices)
            carrier_dosage_chunks.append(column[carrier_sample_indices].astype(np.float32, copy=False))
            missing_sample_chunks.append(missing_sample_indices)
            carrier_counts[local_variant_index] = carrier_sample_indices.shape[0]
            missing_counts[local_variant_index] = missing_sample_indices.shape[0]
        local_start += batch_values.shape[1]
    variant_sample_indices = (
        np.concatenate(carrier_sample_chunks).astype(np.int32, copy=False)
        if carrier_sample_chunks else np.zeros(0, dtype=np.int32)
    )
    variant_dosages = (
        np.concatenate(carrier_dosage_chunks).astype(np.float32, copy=False)
        if carrier_dosage_chunks else np.zeros(0, dtype=np.float32)
    )
    missing_variant_sample_indices = (
        np.concatenate(missing_sample_chunks).astype(np.int32, copy=False)
        if missing_sample_chunks else np.zeros(0, dtype=np.int32)
    )
    variant_ptr = np.zeros(resolved_variant_indices.shape[0] + 1, dtype=np.int64)
    variant_ptr[1:] = np.cumsum(carrier_counts, dtype=np.int64)
    missing_variant_ptr = np.zeros(resolved_variant_indices.shape[0] + 1, dtype=np.int64)
    missing_variant_ptr[1:] = np.cumsum(missing_counts, dtype=np.int64)
    carrier_variant_indices = np.repeat(
        np.arange(resolved_variant_indices.shape[0], dtype=np.int32),
        carrier_counts.astype(np.int32, copy=False),
    )
    if variant_sample_indices.size > 0:
        sample_order = np.argsort(variant_sample_indices, kind="stable")
        sample_variant_indices = carrier_variant_indices[sample_order].astype(np.int32, copy=False)
        sample_dosages = variant_dosages[sample_order].astype(np.float32, copy=False)
        sorted_sample_indices = variant_sample_indices[sample_order]
        sample_counts = np.bincount(sorted_sample_indices, minlength=sample_count).astype(np.int64, copy=False)
    else:
        sample_variant_indices = np.zeros(0, dtype=np.int32)
        sample_dosages = np.zeros(0, dtype=np.float32)
        sample_counts = np.zeros(sample_count, dtype=np.int64)
    sample_ptr = np.zeros(sample_count + 1, dtype=np.int64)
    sample_ptr[1:] = np.cumsum(sample_counts, dtype=np.int64)
    missing_variant_indices = np.repeat(
        np.arange(resolved_variant_indices.shape[0], dtype=np.int32),
        missing_counts.astype(np.int32, copy=False),
    )
    if missing_variant_sample_indices.size > 0:
        missing_sample_order = np.argsort(missing_variant_sample_indices, kind="stable")
        sample_missing_variant_indices = missing_variant_indices[missing_sample_order].astype(np.int32, copy=False)
        sorted_missing_sample_indices = missing_variant_sample_indices[missing_sample_order]
        sample_missing_counts = np.bincount(sorted_missing_sample_indices, minlength=sample_count).astype(np.int64, copy=False)
    else:
        sample_missing_variant_indices = np.zeros(0, dtype=np.int32)
        sample_missing_counts = np.zeros(sample_count, dtype=np.int64)
    sample_missing_ptr = np.zeros(sample_count + 1, dtype=np.int64)
    sample_missing_ptr[1:] = np.cumsum(sample_missing_counts, dtype=np.int64)
    return _SparseCarrierBackend(
        sample_count=sample_count,
        means=np.asarray(means[resolved_variant_indices], dtype=np.float32),
        scales=np.asarray(scales[resolved_variant_indices], dtype=np.float32),
        variant_ptr=variant_ptr,
        variant_sample_indices=variant_sample_indices,
        variant_dosages=variant_dosages,
        missing_variant_ptr=missing_variant_ptr,
        missing_variant_sample_indices=missing_variant_sample_indices,
        sample_ptr=sample_ptr,
        sample_variant_indices=sample_variant_indices,
        sample_dosages=sample_dosages,
        sample_missing_ptr=sample_missing_ptr,
        sample_missing_variant_indices=sample_missing_variant_indices,
    )


@dataclass(slots=True)
class StandardizedGenotypeMatrix:
    """A genotype matrix that applies z-score standardization on the fly.

    For each variant j: standardized_value = (raw_dosage - mean_j) / scale_j
    Missing values (NaN) are imputed to the mean (producing 0 after centering).

    GPU acceleration uses CuPy (cuBLAS) for matmul, bypassing JAX/XLA which
    has known segfault bugs on Turing GPUs.  Falls back to numpy BLAS on CPU.
    """
    raw: RawGenotypeMatrix | None
    means: np.ndarray       # per-variant mean from training data
    scales: np.ndarray      # per-variant std dev from training data
    variant_indices: np.ndarray  # which columns of raw to use (for subsetting)
    support_counts: np.ndarray | None = None  # non-zero dosage count per source variant
    sample_count: int | None = field(default=None, repr=False)
    _enable_hybrid_backend: bool = field(default=True, repr=False)
    _dense_cache: np.ndarray | None = field(init=False, default=None, repr=False)
    _cupy_cache: Any | None = field(init=False, default=None, repr=False)  # cupy.ndarray
    _jax_cache: jax.Array | None = field(init=False, default=None, repr=False)
    _cupy_subset_cache: Any | None = field(init=False, default=None, repr=False)
    _cupy_subset_cache_local_indices: np.ndarray | None = field(init=False, default=None, repr=False)
    _local_cache_directory: tempfile.TemporaryDirectory[str] | None = field(init=False, default=None, repr=False)
    _dense_backend: StandardizedGenotypeMatrix | None = field(init=False, default=None, repr=False)
    _sparse_backend: _SparseCarrierBackend | None = field(init=False, default=None, repr=False)
    _dense_local_lookup: np.ndarray | None = field(init=False, default=None, repr=False)
    _sparse_local_lookup: np.ndarray | None = field(init=False, default=None, repr=False)
    _sample_space_nystrom_basis_cpu_cache: dict[tuple[int, int], np.ndarray] = field(init=False, default_factory=dict, repr=False)
    _sample_space_nystrom_basis_gpu_cache: dict[tuple[int, int], Any] = field(init=False, default_factory=dict, repr=False)
    _sample_space_probe_projection_cache: dict[tuple[int, int], np.ndarray] = field(init=False, default_factory=dict, repr=False)
    _sample_space_cpu_preconditioner_cache: Any | None = field(init=False, default=None, repr=False)
    _sample_space_gpu_preconditioner_cache: Any | None = field(init=False, default=None, repr=False)
    _n_samples: int = field(init=False, default=0, repr=False)

    def __post_init__(self) -> None:
        self.means = np.asarray(self.means, dtype=np.float32)
        self.scales = np.asarray(self.scales, dtype=np.float32)
        if self.means.ndim != 1 or self.scales.ndim != 1 or self.means.shape != self.scales.shape:
            raise ValueError("means and scales must be matching 1D arrays.")
        if self.support_counts is not None:
            self.support_counts = np.asarray(self.support_counts, dtype=np.int32)
            if self.support_counts.ndim != 1 or self.support_counts.shape != self.means.shape:
                raise ValueError("support_counts must match means/scales shape.")
        self.variant_indices = np.asarray(self.variant_indices, dtype=np.int32)
        source_variant_count = int(self.means.shape[0])
        if self.raw is not None:
            self._n_samples = int(self.raw.shape[0])
            if int(self.raw.shape[1]) != source_variant_count:
                raise ValueError("raw genotype matrix width must match means/scales.")
        elif self.sample_count is not None:
            self._n_samples = int(self.sample_count)
        else:
            self._n_samples = 0
        if self.variant_indices.ndim != 1:
            raise ValueError("variant_indices must be 1D.")
        if np.any(self.variant_indices < 0) or np.any(self.variant_indices >= source_variant_count):
            raise ValueError("variant_indices out of bounds.")
        self._configure_operator_backend()

    def _configure_operator_backend(self) -> None:
        self._dense_backend = None
        self._sparse_backend = None
        self._dense_local_lookup = None
        self._sparse_local_lookup = None
        if (
            not self._enable_hybrid_backend
            or self.raw is None
            or self.support_counts is None
            or not _supports_int8_batches(self.raw)
            or self.shape[1] == 0
        ):
            return
        selected_support_counts = np.asarray(self.support_counts[self.variant_indices], dtype=np.int32)
        sparse_local_indices = np.flatnonzero(selected_support_counts <= HYBRID_SPARSE_SUPPORT_THRESHOLD).astype(np.int32)
        if sparse_local_indices.shape[0] < HYBRID_SPARSE_MIN_VARIANT_COUNT:
            return
        dense_mask = np.ones(self.shape[1], dtype=bool)
        dense_mask[sparse_local_indices] = False
        dense_local_indices = np.flatnonzero(dense_mask).astype(np.int32)
        self._sparse_backend = _build_sparse_backend(
            raw=cast(Int8BatchCapable, self.raw),
            raw_variant_indices=self.variant_indices[sparse_local_indices],
            means=self.means,
            scales=self.scales,
            sample_count=self.shape[0],
        )
        self._sparse_local_lookup = np.full(self.shape[1], -1, dtype=np.int32)
        self._sparse_local_lookup[sparse_local_indices] = np.arange(sparse_local_indices.shape[0], dtype=np.int32)
        if dense_local_indices.shape[0] > 0:
            self._dense_backend = StandardizedGenotypeMatrix(
                raw=self.raw,
                means=self.means,
                scales=self.scales,
                variant_indices=self.variant_indices[dense_local_indices],
                support_counts=self.support_counts,
                sample_count=self.shape[0],
                _enable_hybrid_backend=False,
            )
            self._dense_local_lookup = np.full(self.shape[1], -1, dtype=np.int32)
            self._dense_local_lookup[dense_local_indices] = np.arange(dense_local_indices.shape[0], dtype=np.int32)
        log(
            "    hybrid standardized operator: "
            + f"{sparse_local_indices.shape[0]} sparse variants + {dense_local_indices.shape[0]} dense variants  mem={mem()}"
        )

    def _uses_hybrid_backend(self) -> bool:
        return self._sparse_backend is not None

    def _hybrid_local_components(
        self,
        local_variant_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        resolved_local_indices = np.asarray(local_variant_indices, dtype=np.int32)
        if resolved_local_indices.ndim != 1:
            raise ValueError("local_variant_indices must be 1D.")
        if not self._uses_hybrid_backend():
            empty = np.zeros(0, dtype=np.int32)
            empty_mask = np.zeros(resolved_local_indices.shape[0], dtype=bool)
            return empty_mask, empty, empty_mask.copy(), empty
        sparse_lookup = (
            np.full(self.shape[1], -1, dtype=np.int32)
            if self._sparse_local_lookup is None
            else self._sparse_local_lookup
        )
        dense_lookup = (
            np.full(self.shape[1], -1, dtype=np.int32)
            if self._dense_local_lookup is None
            else self._dense_local_lookup
        )
        sparse_child_local_indices = sparse_lookup[resolved_local_indices]
        dense_child_local_indices = dense_lookup[resolved_local_indices]
        sparse_mask = sparse_child_local_indices >= 0
        dense_mask = dense_child_local_indices >= 0
        if np.any(~(sparse_mask | dense_mask)):
            raise RuntimeError("hybrid standardized operator lost local column mapping.")
        return (
            sparse_mask,
            sparse_child_local_indices[sparse_mask].astype(np.int32, copy=False),
            dense_mask,
            dense_child_local_indices[dense_mask].astype(np.int32, copy=False),
        )

    def _hybrid_parent_local_indices(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._uses_hybrid_backend():
            return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
        sparse_parent_local_indices = (
            np.flatnonzero(self._sparse_local_lookup >= 0).astype(np.int32)
            if self._sparse_local_lookup is not None
            else np.zeros(0, dtype=np.int32)
        )
        dense_parent_local_indices = (
            np.flatnonzero(self._dense_local_lookup >= 0).astype(np.int32)
            if self._dense_local_lookup is not None
            else np.zeros(0, dtype=np.int32)
        )
        return sparse_parent_local_indices, dense_parent_local_indices

    def _materialize_hybrid_columns(
        self,
        local_variant_indices: np.ndarray,
        *,
        batch_size: int,
    ) -> np.ndarray:
        resolved_local_indices = np.asarray(local_variant_indices, dtype=np.int32)
        if resolved_local_indices.ndim != 1:
            raise ValueError("local_variant_indices must be 1D.")
        output = np.empty((self.shape[0], resolved_local_indices.shape[0]), dtype=np.float32)
        if resolved_local_indices.size == 0:
            return output
        sparse_mask, sparse_child_local_indices, dense_mask, dense_child_local_indices = self._hybrid_local_components(
            resolved_local_indices
        )
        if np.any(sparse_mask):
            if self._sparse_backend is None:
                raise RuntimeError("hybrid sparse backend is not configured.")
            output[:, sparse_mask] = self._sparse_backend.materialize_columns(sparse_child_local_indices)
        if np.any(dense_mask):
            if self._dense_backend is None:
                raise RuntimeError("hybrid dense backend is not configured.")
            output[:, dense_mask] = self._dense_backend.subset(dense_child_local_indices).materialize(batch_size=batch_size)
        return output

    @property
    def _gpu_cache(self) -> Any | None:
        return self._cupy_cache

    @_gpu_cache.setter
    def _gpu_cache(self, value: Any | None) -> None:
        self._cupy_cache = value

    @property
    def shape(self) -> tuple[int, int]:
        return self._n_samples, int(self.variant_indices.shape[0])

    def dense_bytes(self) -> int:
        """Estimated bytes if materialized as float32."""
        return int(self.shape[0]) * int(self.shape[1]) * 4

    def release_raw_storage(self) -> None:
        """Drop the backing raw matrix once a dense or GPU cache exists."""
        if self._dense_cache is None and self._cupy_cache is None:
            raise RuntimeError("cannot release raw storage before materializing genotype data.")
        self.raw = None
        self._local_cache_directory = None
        self._cupy_subset_cache = None
        self._cupy_subset_cache_local_indices = None
        if self._dense_backend is not None:
            self._dense_backend.raw = None
            self._dense_backend._local_cache_directory = None

    def clear_sample_space_nystrom_cache(self) -> None:
        self._sample_space_nystrom_basis_cpu_cache.clear()
        self._sample_space_nystrom_basis_gpu_cache.clear()
        self._sample_space_probe_projection_cache.clear()
        self._sample_space_cpu_preconditioner_cache = None
        self._sample_space_gpu_preconditioner_cache = None
        self._cupy_subset_cache = None
        self._cupy_subset_cache_local_indices = None

    def try_materialize_gpu(self) -> bool:
        """Materialize the standardized matrix onto GPU memory when possible."""
        if self._cupy_cache is not None:
            return True
        cupy = _try_import_cupy()
        if cupy is None:
            return False
        try:
            nbytes = self.dense_bytes()
            budget_bytes = _gpu_materialization_budget_bytes(cupy)
            if nbytes > budget_bytes:
                log(
                    f"    skipping GPU materialization: need {nbytes / 1e9:.1f} GB, "
                    f"budget is {budget_bytes / 1e9:.1f} GB  mem={mem()}"
                )
                return False
            if self._dense_cache is not None:
                log(f"    uploading RAM-resident matrix to GPU ({nbytes / 1e9:.1f} GB)  mem={mem()}")
                self._cupy_cache = cupy.asarray(self._dense_cache)
                dense_ref = self._dense_cache
                self._dense_cache = None
                del dense_ref
            else:
                log(f"    streaming {self.shape[1]} variants x {self.shape[0]} samples ({nbytes / 1e9:.1f} GB) directly to GPU  mem={mem()}")
                gpu_matrix = cupy.empty(self.shape, dtype=cupy.float32, order="F")
                if self.raw is None:
                    raise RuntimeError("GPU materialization requires raw backing storage or a dense cache.")
                # For GPU materialization, maximize batch size to minimize Python
                # overhead and kernel launch latency. The working memory per batch is
                # ~5 bytes/element (int8 + float32), so we can fit larger batches than
                # the CPU-oriented BED_READER_TARGET_BATCH_BYTES assumes.
                gpu_working_bytes = int(self.shape[0]) * int(self.shape[1]) * 5
                if gpu_working_bytes < _gpu_materialization_budget_bytes(cupy) * 0.5:
                    upload_batch_size = self.shape[1]  # entire block in one batch
                elif isinstance(self.raw, Int8RawGenotypeMatrix):
                    upload_batch_size = auto_batch_size_i8(self.shape[0])
                else:
                    upload_batch_size = auto_batch_size(self.shape[0])
                for batch_slice, standardized_batch in _iter_standardized_gpu_batches(
                    self.raw,
                    self.variant_indices,
                    self.means,
                    self.scales,
                    batch_size=upload_batch_size,
                    cupy=cupy,
                ):
                    gpu_matrix[:, batch_slice] = standardized_batch
                self._cupy_cache = gpu_matrix
            cupy.cuda.Device().synchronize()
            import gc
            gc.collect()
            log(f"    CuPy GPU matrix ready ({self._cupy_cache.nbytes / 1e9:.1f} GB)  mem={mem()}")
            return True
        except Exception as exc:
            log(f"    CuPy GPU upload failed ({exc})  mem={mem()}")
            self._cupy_cache = None
            return False

    def try_materialize_gpu_subset(
        self,
        local_variant_indices: Sequence[int] | np.ndarray,
    ) -> Any | None:
        """Materialize a selected standardized column subset onto GPU memory when possible."""
        resolved_local_indices = np.asarray(local_variant_indices, dtype=np.int32)
        if resolved_local_indices.ndim != 1:
            raise ValueError("local_variant_indices must be 1D.")
        if resolved_local_indices.size == 0:
            return None
        if self._cupy_cache is not None:
            return self._cupy_cache[:, resolved_local_indices]
        if (
            self._cupy_subset_cache is not None
            and self._cupy_subset_cache_local_indices is not None
            and np.array_equal(self._cupy_subset_cache_local_indices, resolved_local_indices)
        ):
            return self._cupy_subset_cache

        cupy = _try_import_cupy()
        if cupy is None:
            return None
        nbytes = int(self.shape[0]) * int(resolved_local_indices.shape[0]) * 4
        budget_bytes = _gpu_materialization_budget_bytes(cupy)
        if nbytes > budget_bytes:
            log(
                f"    skipping GPU subset materialization: need {nbytes / 1e9:.1f} GB, "
                f"budget is {budget_bytes / 1e9:.1f} GB  mem={mem()}"
            )
            return None
        try:
            if self._dense_cache is not None:
                gpu_subset = cupy.asarray(self._dense_cache[:, resolved_local_indices], dtype=cupy.float32)
            else:
                if self.raw is None:
                    raise RuntimeError("GPU subset materialization requires raw backing storage or a dense cache.")
                gpu_subset = cupy.empty((self.shape[0], resolved_local_indices.shape[0]), dtype=cupy.float32, order="F")
                selected_variant_indices = self.variant_indices[resolved_local_indices]
                for batch_slice, standardized_batch in _iter_standardized_gpu_batches(
                    self.raw,
                    selected_variant_indices,
                    self.means,
                    self.scales,
                    batch_size=auto_batch_size(self.shape[0]),
                    cupy=cupy,
                ):
                    gpu_subset[:, batch_slice] = standardized_batch
            cupy.cuda.Device().synchronize()
            self._cupy_subset_cache = gpu_subset
            self._cupy_subset_cache_local_indices = resolved_local_indices.copy()
            log(
                "    CuPy GPU subset ready "
                + f"({resolved_local_indices.shape[0]} variants, {nbytes / 1e9:.1f} GB)  mem={mem()}"
            )
            return self._cupy_subset_cache
        except Exception as exc:
            log(f"    CuPy GPU subset upload failed ({exc})  mem={mem()}")
            self._cupy_subset_cache = None
            self._cupy_subset_cache_local_indices = None
            return None

    def try_materialize(self) -> bool:
        """Materialize into RAM if below the auto-materialize threshold.

        Returns True if now cached in memory, False if still streaming from disk.
        """
        if self._cupy_cache is not None or self._dense_cache is not None:
            return True
        nbytes = self.dense_bytes()
        if nbytes > MATERIALIZE_THRESHOLD_BYTES:
            return False
        log(f"    auto-materializing {self.shape[1]} variants x {self.shape[0]} samples ({nbytes / 1e9:.1f} GB) into RAM  mem={mem()}")
        self._dense_cache = self.materialize()
        self._jax_cache = None
        log(f"    materialized  mem={mem()}")
        return True

    def supports_jax_dense_ops(self) -> bool:
        return jax_dense_linear_algebra_preferred() and (self._dense_cache is not None or self._jax_cache is not None)

    def _ensure_jax_cache(self) -> jax.Array:
        if self._jax_cache is not None:
            return self._jax_cache
        if self._dense_cache is None:
            raise RuntimeError("JAX cache requires a dense materialized genotype matrix.")
        jax_cache = jnp.asarray(self._dense_cache, dtype=gpu_compute_jax_dtype())
        if isinstance(jax_cache, jax_core.Tracer):
            return jax_cache
        self._jax_cache = jax_cache
        return self._jax_cache

    def try_cache_locally(self) -> bool:
        """Rebase onto a local int8 memmap to avoid repeated upstream streaming passes."""
        if self.raw is None or self._dense_cache is not None or self._cupy_cache is not None:
            return False
        if not _supports_int8_batches(self.raw):
            return False
        if isinstance(self.raw, Int8RawGenotypeMatrix):
            return True
        batch_size = auto_batch_size_i8(self.shape[0])
        selected_variant_count = int(self.variant_indices.shape[0])
        log(
            "    caching reduced raw genotypes locally as int8 "
            + f"({selected_variant_count} variants x {self.shape[0]} samples)  mem={mem()}"
        )
        cache_directory = tempfile.TemporaryDirectory(prefix="svpgs-genotype-")
        cache_path = Path(cache_directory.name) / "reduced_raw_i8.npy"
        try:
            has_space, required_bytes, available_bytes = _has_sufficient_free_space_for_int8_npy(
                Path(cache_directory.name),
                self.shape,
                fortran_order=True,
            )
            if not has_space:
                log(
                    "    local int8 cache skipped: insufficient free space "
                    + f"(need~{required_bytes / 1e9:.1f} GB, free~{available_bytes / 1e9:.1f} GB)  mem={mem()}"
                )
                cache_directory.cleanup()
                return False
            raw_int8 = cast(Int8BatchCapable, self.raw)
            def _column_batches() -> Iterator[np.ndarray]:
                for raw_batch in raw_int8.iter_column_batches_i8(self.variant_indices, batch_size=batch_size):
                    yield raw_batch.values
            _stream_write_int8_npy(
                cache_path,
                shape=self.shape,
                column_batches=_column_batches(),
                fortran_order=True,
            )
            rebased_raw = Int8RawGenotypeMatrix(np.load(cache_path, mmap_mode="r"))
            self.raw = rebased_raw
            self.means = np.asarray(self.means[self.variant_indices], dtype=np.float32)
            self.scales = np.asarray(self.scales[self.variant_indices], dtype=np.float32)
            if self.support_counts is not None:
                self.support_counts = np.asarray(self.support_counts[self.variant_indices], dtype=np.int32)
            self.variant_indices = np.arange(selected_variant_count, dtype=np.int32)
            self.clear_sample_space_nystrom_cache()
            self._local_cache_directory = cache_directory
            self._cupy_subset_cache = None
            self._cupy_subset_cache_local_indices = None
            self._enable_hybrid_backend = False  # GPU streaming handles everything
            log("    local int8 cache ready  mem=" + mem())
            return True
        except Exception as exc:
            cache_directory.cleanup()
            log(f"    local int8 cache failed ({exc})  mem={mem()}")
            return False

    def try_cache_persistently(self, cache_path: Path) -> bool:
        """Persist a reduced raw int8 cache to a stable path for reuse across runs."""
        if self.raw is None or self._dense_cache is not None or self._cupy_cache is not None:
            return False
        if not _supports_int8_batches(self.raw):
            return False
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        batch_size = auto_batch_size_i8(self.shape[0])
        selected_variant_count = int(self.variant_indices.shape[0])
        temp_directory = Path(tempfile.mkdtemp(prefix=f"{cache_path.name}.tmp.", dir=cache_path.parent))
        temp_path = temp_directory / cache_path.name
        log(
            "    persisting reduced raw genotypes as int8 "
            + f"({selected_variant_count} variants x {self.shape[0]} samples) → {cache_path}  mem={mem()}"
        )
        try:
            has_space, required_bytes, available_bytes = _has_sufficient_free_space_for_int8_npy(
                temp_directory,
                self.shape,
                fortran_order=True,
            )
            if not has_space:
                log(
                    "    persistent int8 cache skipped: insufficient free space "
                    + f"(need~{required_bytes / 1e9:.1f} GB, free~{available_bytes / 1e9:.1f} GB)  mem={mem()}"
                )
                return False
            raw_int8 = cast(Int8BatchCapable, self.raw)
            def _column_batches() -> Iterator[np.ndarray]:
                for raw_batch in raw_int8.iter_column_batches_i8(self.variant_indices, batch_size=batch_size):
                    yield raw_batch.values
            _stream_write_int8_npy(
                temp_path,
                shape=self.shape,
                column_batches=_column_batches(),
                fortran_order=True,
            )
            temp_path.replace(cache_path)
            persisted_raw = Int8RawGenotypeMatrix(np.load(cache_path, mmap_mode="r"))
            self.raw = persisted_raw
            self.means = np.asarray(self.means[self.variant_indices], dtype=np.float32)
            self.scales = np.asarray(self.scales[self.variant_indices], dtype=np.float32)
            if self.support_counts is not None:
                self.support_counts = np.asarray(self.support_counts[self.variant_indices], dtype=np.int32)
            self.variant_indices = np.arange(selected_variant_count, dtype=np.int32)
            self.clear_sample_space_nystrom_cache()
            self._local_cache_directory = None
            self._cupy_subset_cache = None
            self._cupy_subset_cache_local_indices = None
            # Don't rebuild hybrid backend — GPU streaming handles everything
            self._enable_hybrid_backend = False
            log("    persistent int8 cache ready  mem=" + mem())
            return True
        except Exception as exc:
            log(f"    persistent int8 cache failed ({exc})  mem={mem()}")
            return False
        finally:
            if temp_directory.exists():
                for child in temp_directory.iterdir():
                    child.unlink(missing_ok=True)
                temp_directory.rmdir()

    def subset(self, local_variant_indices: Sequence[int] | np.ndarray) -> StandardizedGenotypeMatrix:
        resolved_local_indices = np.asarray(local_variant_indices, dtype=np.int32)
        preserve_hybrid = (
            self._uses_hybrid_backend()
            and self._dense_cache is None
            and self._cupy_cache is None
            and self._jax_cache is None
        )
        subset = StandardizedGenotypeMatrix(
            raw=self.raw,
            means=self.means,
            scales=self.scales,
            variant_indices=self.variant_indices[resolved_local_indices],
            support_counts=self.support_counts,
            sample_count=self.shape[0],
            _enable_hybrid_backend=False if preserve_hybrid else self._enable_hybrid_backend,
        )
        if preserve_hybrid:
            sparse_mask, sparse_child_local_indices, dense_mask, dense_child_local_indices = self._hybrid_local_components(
                resolved_local_indices
            )
            if np.any(sparse_mask):
                if self._sparse_backend is None:
                    raise RuntimeError("hybrid sparse backend is not configured.")
                subset._sparse_backend = self._sparse_backend.subset(sparse_child_local_indices)
                subset._sparse_local_lookup = np.full(resolved_local_indices.shape[0], -1, dtype=np.int32)
                subset._sparse_local_lookup[sparse_mask] = np.arange(sparse_child_local_indices.shape[0], dtype=np.int32)
            if np.any(dense_mask):
                if self._dense_backend is None:
                    raise RuntimeError("hybrid dense backend is not configured.")
                subset._dense_backend = self._dense_backend.subset(dense_child_local_indices)
                subset._dense_local_lookup = np.full(resolved_local_indices.shape[0], -1, dtype=np.int32)
                subset._dense_local_lookup[dense_mask] = np.arange(dense_child_local_indices.shape[0], dtype=np.int32)
        if self._cupy_cache is not None:
            subset._cupy_cache = self._cupy_cache[:, resolved_local_indices]
        if self._jax_cache is not None:
            subset._jax_cache = self._jax_cache[:, resolved_local_indices]
        elif self._dense_cache is not None:
            subset._dense_cache = np.asarray(self._dense_cache[:, resolved_local_indices], dtype=np.float32)
        subset._local_cache_directory = self._local_cache_directory
        return subset

    def iter_column_batches(
        self,
        batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE,
    ) -> Iterator[RawGenotypeBatch]:
        if self._dense_cache is not None:
            safe_batch_size = max(int(batch_size), 1)
            for start_index in range(0, self._dense_cache.shape[1], safe_batch_size):
                local_indices = np.arange(start_index, min(start_index + safe_batch_size, self._dense_cache.shape[1]), dtype=np.int32)
                yield RawGenotypeBatch(
                    variant_indices=local_indices,
                    values=np.asarray(self._dense_cache[:, local_indices], dtype=np.float32),
                )
            return
        if self._uses_hybrid_backend():
            safe_batch_size = _effective_standardized_streaming_batch_size(
                self.shape[0],
                max(int(batch_size), 1),
                target_batch_bytes=STANDARDIZED_STREAMING_TARGET_BATCH_BYTES,
            )
            for start_index in range(0, self.shape[1], safe_batch_size):
                local_indices = np.arange(start_index, min(start_index + safe_batch_size, self.shape[1]), dtype=np.int32)
                yield RawGenotypeBatch(
                    variant_indices=local_indices,
                    values=self._materialize_hybrid_columns(local_indices, batch_size=safe_batch_size),
                )
            return
        if self.raw is None:
            raise RuntimeError("streaming genotype batches require raw backing storage or a materialized cache.")
        target_batch_bytes = (
            LOCAL_INT8_STANDARDIZED_STREAMING_TARGET_BATCH_BYTES
            if isinstance(self.raw, Int8RawGenotypeMatrix)
            else STANDARDIZED_STREAMING_TARGET_BATCH_BYTES
        )
        safe_batch_size = _effective_standardized_streaming_batch_size(
            self.shape[0],
            batch_size,
            target_batch_bytes=target_batch_bytes,
        )
        local_start = 0
        if _supports_int8_batches(self.raw):
            raw_int8 = cast(Int8BatchCapable, self.raw)
            for raw_batch in raw_int8.iter_column_batches_i8(self.variant_indices, batch_size=safe_batch_size):
                local_stop = local_start + raw_batch.variant_indices.shape[0]
                local_indices = np.arange(local_start, local_stop, dtype=np.int32)
                yield RawGenotypeBatch(
                    variant_indices=local_indices,
                    values=_standardize_batch_i8(
                        raw_batch.values,
                        self.means[raw_batch.variant_indices],
                        self.scales[raw_batch.variant_indices],
                    ),
                )
                local_start = local_stop
            return
        for raw_batch in self.raw.iter_column_batches(self.variant_indices, batch_size=safe_batch_size):
            local_stop = local_start + raw_batch.variant_indices.shape[0]
            local_indices = np.arange(local_start, local_stop, dtype=np.int32)
            yield RawGenotypeBatch(
                variant_indices=local_indices,
                values=_standardize_batch(
                    raw_batch.values,
                    self.means[raw_batch.variant_indices],
                    self.scales[raw_batch.variant_indices],
                ),
            )
            local_start = local_stop

    def materialize(self, batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE) -> np.ndarray:
        if self._dense_cache is not None:
            return self._dense_cache
        if self._cupy_cache is not None:
            self._dense_cache = self._cupy_cache.get()  # cupy -> numpy
            self._jax_cache = None
            return self._dense_cache
        matrix = np.empty(self.shape, dtype=np.float32)
        for batch in self.iter_column_batches(batch_size=batch_size):
            matrix[:, batch.variant_indices] = batch.values
        self._dense_cache = matrix
        self._jax_cache = None
        return matrix

    def _streaming_gpu_context(self, batch_size: int):
        if self._cupy_cache is not None or self.supports_jax_dense_ops() or self.raw is None or self._uses_hybrid_backend():
            return None, None
        cupy = _try_import_cupy()
        if cupy is None:
            return None, None
        target_batch_bytes = (
            LOCAL_INT8_STANDARDIZED_STREAMING_TARGET_BATCH_BYTES
            if isinstance(self.raw, Int8RawGenotypeMatrix)
            else STANDARDIZED_STREAMING_TARGET_BATCH_BYTES
        )
        return cupy, _effective_standardized_streaming_batch_size(
            self.shape[0],
            batch_size,
            target_batch_bytes=target_batch_bytes,
        )

    def matvec(self, coefficients: np.ndarray | jnp.ndarray, batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE) -> jnp.ndarray:
        """Compute X_std @ beta (genotype matrix times coefficient vector).

        When CuPy GPU-cached: cuBLAS matmul. Otherwise: batched numpy on CPU.
        """
        if self._cupy_cache is not None:
            cupy = _try_import_cupy()
            if cupy is None:
                raise RuntimeError("CuPy cache requires CuPy runtime.")
            compute_np_dtype = gpu_compute_numpy_dtype()
            compute_cp_dtype = _cupy_compute_dtype(cupy)
            coeff_jax = jnp.ravel(jnp.asarray(coefficients, dtype=gpu_compute_jax_dtype()))
            if coeff_jax.shape[0] != self.shape[1]:
                raise ValueError("coefficient vector must match genotype column count.")
            if bool(jnp.all(coeff_jax == 0.0)):
                return _as_gpu_compute_jax(np.zeros(self.shape[0], dtype=compute_np_dtype))
            coeff_cupy = _to_cupy_compute(coeff_jax)
            result = self._cupy_cache.astype(compute_cp_dtype, copy=False) @ coeff_cupy
            return _cupy_to_jax(result)
        cupy, streaming_batch_size = self._streaming_gpu_context(batch_size)
        if cupy is not None and streaming_batch_size is not None:
            compute_np_dtype = gpu_compute_numpy_dtype()
            compute_cp_dtype = _cupy_compute_dtype(cupy)
            coeff_np = np.asarray(coefficients, dtype=compute_np_dtype).ravel()
            if coeff_np.shape[0] != self.shape[1]:
                raise ValueError("coefficient vector must match genotype column count.")
            if not np.any(coeff_np):
                return _as_gpu_compute_jax(np.zeros(self.shape[0], dtype=compute_np_dtype))
            coeff_gpu = _to_cupy_compute(coeff_np)
            result_gpu = cupy.zeros(self.shape[0], dtype=compute_cp_dtype)
            for batch_slice, standardized_batch in _iter_standardized_gpu_batches(
                self.raw,
                self.variant_indices,
                self.means,
                self.scales,
                batch_size=streaming_batch_size,
                cupy=cupy,
                dtype=compute_cp_dtype,
            ):
                result_gpu += standardized_batch @ coeff_gpu[batch_slice]
            return _cupy_to_jax(result_gpu)
        if self.supports_jax_dense_ops():
            coeff_jax = jnp.ravel(jnp.asarray(coefficients, dtype=gpu_compute_jax_dtype()))
            if coeff_jax.shape[0] != self.shape[1]:
                raise ValueError("coefficient vector must match genotype column count.")
            return self._ensure_jax_cache() @ coeff_jax
        coeff_np = np.asarray(coefficients, dtype=gpu_compute_numpy_dtype()).ravel()
        if coeff_np.shape[0] != self.shape[1]:
            raise ValueError("coefficient vector must match genotype column count.")
        if not np.any(coeff_np):
            return _as_gpu_compute_jax(np.zeros(self.shape[0], dtype=coeff_np.dtype))
        if self._uses_hybrid_backend():
            sparse_parent_local_indices, dense_parent_local_indices = self._hybrid_parent_local_indices()
            result = np.zeros(self.shape[0], dtype=coeff_np.dtype)
            if sparse_parent_local_indices.size > 0:
                if self._sparse_backend is None:
                    raise RuntimeError("hybrid sparse backend is not configured.")
                result += self._sparse_backend.matvec(coeff_np[sparse_parent_local_indices])
            if dense_parent_local_indices.size > 0:
                if self._dense_backend is None:
                    raise RuntimeError("hybrid dense backend is not configured.")
                result += np.asarray(
                    self._dense_backend.matvec(coeff_np[dense_parent_local_indices], batch_size=batch_size),
                    dtype=coeff_np.dtype,
                )
            return _as_gpu_compute_jax(result)
        result = np.zeros(self.shape[0], dtype=gpu_compute_numpy_dtype())
        for batch in self.iter_column_batches(batch_size=batch_size):
            batch_values = np.asarray(batch.values, dtype=result.dtype)
            result += batch_values @ coeff_np[batch.variant_indices]
        return _as_gpu_compute_jax(result)

    def matmat(self, matrix: np.ndarray | jnp.ndarray, batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE) -> jnp.ndarray:
        """Compute X_std @ M. When CuPy GPU-cached: cuBLAS matmul. Otherwise: batched numpy."""
        if self._cupy_cache is not None:
            cupy = _try_import_cupy()
            if cupy is None:
                raise RuntimeError("CuPy cache requires CuPy runtime.")
            compute_np_dtype = gpu_compute_numpy_dtype()
            compute_cp_dtype = _cupy_compute_dtype(cupy)
            matrix_jax = jnp.asarray(matrix, dtype=gpu_compute_jax_dtype())
            if matrix_jax.ndim != 2 or matrix_jax.shape[0] != self.shape[1]:
                raise ValueError("variant matrix must match genotype column count.")
            if bool(jnp.all(matrix_jax == 0.0)):
                return _as_gpu_compute_jax(np.zeros((self.shape[0], matrix_jax.shape[1]), dtype=compute_np_dtype))
            result = self._cupy_cache.astype(compute_cp_dtype, copy=False) @ _to_cupy_compute(matrix_jax)
            return _cupy_to_jax(result)
        cupy, streaming_batch_size = self._streaming_gpu_context(batch_size)
        if cupy is not None and streaming_batch_size is not None:
            compute_np_dtype = gpu_compute_numpy_dtype()
            compute_cp_dtype = _cupy_compute_dtype(cupy)
            matrix_np = np.asarray(matrix, dtype=compute_np_dtype)
            if matrix_np.ndim != 2 or matrix_np.shape[0] != self.shape[1]:
                raise ValueError("variant matrix must match genotype column count.")
            if not np.any(matrix_np):
                return _as_gpu_compute_jax(np.zeros((self.shape[0], matrix_np.shape[1]), dtype=compute_np_dtype))
            matrix_gpu = _to_cupy_compute(matrix_np)
            result_gpu = cupy.zeros((self.shape[0], matrix_np.shape[1]), dtype=compute_cp_dtype)
            for batch_slice, standardized_batch in _iter_standardized_gpu_batches(
                self.raw,
                self.variant_indices,
                self.means,
                self.scales,
                batch_size=streaming_batch_size,
                cupy=cupy,
                dtype=compute_cp_dtype,
            ):
                result_gpu += standardized_batch @ matrix_gpu[batch_slice, :]
            return _cupy_to_jax(result_gpu)
        if self.supports_jax_dense_ops():
            matrix_jax = jnp.asarray(matrix, dtype=gpu_compute_jax_dtype())
            if matrix_jax.ndim != 2 or matrix_jax.shape[0] != self.shape[1]:
                raise ValueError("variant matrix must match genotype column count.")
            return self._ensure_jax_cache() @ matrix_jax
        m_np = np.asarray(matrix, dtype=gpu_compute_numpy_dtype())
        if m_np.ndim != 2 or m_np.shape[0] != self.shape[1]:
            raise ValueError("variant matrix must match genotype column count.")
        if not np.any(m_np):
            return _as_gpu_compute_jax(np.zeros((self.shape[0], m_np.shape[1]), dtype=m_np.dtype))
        if self._uses_hybrid_backend():
            sparse_parent_local_indices, dense_parent_local_indices = self._hybrid_parent_local_indices()
            output = np.zeros((self.shape[0], m_np.shape[1]), dtype=m_np.dtype)
            if sparse_parent_local_indices.size > 0:
                if self._sparse_backend is None:
                    raise RuntimeError("hybrid sparse backend is not configured.")
                output += self._sparse_backend.matmat(m_np[sparse_parent_local_indices, :])
            if dense_parent_local_indices.size > 0:
                if self._dense_backend is None:
                    raise RuntimeError("hybrid dense backend is not configured.")
                output += np.asarray(
                    self._dense_backend.matmat(m_np[dense_parent_local_indices, :], batch_size=batch_size),
                    dtype=m_np.dtype,
                )
            return _as_gpu_compute_jax(output)
        output = np.zeros((self.shape[0], m_np.shape[1]), dtype=m_np.dtype)
        for batch in self.iter_column_batches(batch_size=batch_size):
            batch_values = np.asarray(batch.values, dtype=output.dtype)
            output += batch_values @ m_np[batch.variant_indices, :]
        return _as_gpu_compute_jax(output)

    def transpose_matvec(self, vector: np.ndarray | jnp.ndarray, batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE) -> jnp.ndarray:
        """Compute X_std^T @ v. When CuPy GPU-cached: cuBLAS matmul. Otherwise: batched numpy."""
        if self._cupy_cache is not None:
            cupy = _try_import_cupy()
            if cupy is None:
                raise RuntimeError("CuPy cache requires CuPy runtime.")
            compute_np_dtype = gpu_compute_numpy_dtype()
            compute_cp_dtype = _cupy_compute_dtype(cupy)
            vector_jax = jnp.ravel(jnp.asarray(vector, dtype=gpu_compute_jax_dtype()))
            if vector_jax.shape[0] != self.shape[0]:
                raise ValueError("sample vector must match genotype row count.")
            if bool(jnp.all(vector_jax == 0.0)):
                return _as_gpu_compute_jax(np.zeros(self.shape[1], dtype=compute_np_dtype))
            result = self._cupy_cache.astype(compute_cp_dtype, copy=False).T @ _to_cupy_compute(vector_jax)
            return _cupy_to_jax(result)
        cupy, streaming_batch_size = self._streaming_gpu_context(batch_size)
        if cupy is not None and streaming_batch_size is not None:
            compute_np_dtype = gpu_compute_numpy_dtype()
            compute_cp_dtype = _cupy_compute_dtype(cupy)
            vector_np = np.asarray(vector, dtype=compute_np_dtype).ravel()
            if vector_np.shape[0] != self.shape[0]:
                raise ValueError("sample vector must match genotype row count.")
            if not np.any(vector_np):
                return _as_gpu_compute_jax(np.zeros(self.shape[1], dtype=compute_np_dtype))
            vector_gpu = _to_cupy_compute(vector_np)
            output_gpu = cupy.empty(self.shape[1], dtype=compute_cp_dtype)
            for batch_slice, standardized_batch in _iter_standardized_gpu_batches(
                self.raw,
                self.variant_indices,
                self.means,
                self.scales,
                batch_size=streaming_batch_size,
                cupy=cupy,
                dtype=compute_cp_dtype,
            ):
                output_gpu[batch_slice] = standardized_batch.T @ vector_gpu
            return _cupy_to_jax(output_gpu)
        if self.supports_jax_dense_ops():
            vector_jax = jnp.ravel(jnp.asarray(vector, dtype=gpu_compute_jax_dtype()))
            if vector_jax.shape[0] != self.shape[0]:
                raise ValueError("sample vector must match genotype row count.")
            return self._ensure_jax_cache().T @ vector_jax
        v_np = np.asarray(vector, dtype=gpu_compute_numpy_dtype()).ravel()
        if v_np.shape[0] != self.shape[0]:
            raise ValueError("sample vector must match genotype row count.")
        if not np.any(v_np):
            return _as_gpu_compute_jax(np.zeros(self.shape[1], dtype=v_np.dtype))
        if self._uses_hybrid_backend():
            sparse_parent_local_indices, dense_parent_local_indices = self._hybrid_parent_local_indices()
            output = np.empty(self.shape[1], dtype=v_np.dtype)
            if sparse_parent_local_indices.size > 0:
                if self._sparse_backend is None:
                    raise RuntimeError("hybrid sparse backend is not configured.")
                output[sparse_parent_local_indices] = self._sparse_backend.transpose_matvec(v_np)
            if dense_parent_local_indices.size > 0:
                if self._dense_backend is None:
                    raise RuntimeError("hybrid dense backend is not configured.")
                output[dense_parent_local_indices] = np.asarray(
                    self._dense_backend.transpose_matvec(v_np, batch_size=batch_size),
                    dtype=v_np.dtype,
                )
            return _as_gpu_compute_jax(output)
        output = np.empty(self.shape[1], dtype=v_np.dtype)
        for batch in self.iter_column_batches(batch_size=batch_size):
            output[batch.variant_indices] = np.asarray(batch.values, dtype=output.dtype).T @ v_np
        return _as_gpu_compute_jax(output)

    def transpose_matmat(self, matrix: np.ndarray | jnp.ndarray, batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE) -> jnp.ndarray:
        """Compute X_std^T @ M. When CuPy GPU-cached: cuBLAS matmul. Otherwise: batched numpy."""
        if self._cupy_cache is not None:
            cupy = _try_import_cupy()
            if cupy is None:
                raise RuntimeError("CuPy cache requires CuPy runtime.")
            compute_np_dtype = gpu_compute_numpy_dtype()
            compute_cp_dtype = _cupy_compute_dtype(cupy)
            matrix_jax = jnp.asarray(matrix, dtype=gpu_compute_jax_dtype())
            if matrix_jax.ndim != 2 or matrix_jax.shape[0] != self.shape[0]:
                raise ValueError("sample matrix must match genotype row count.")
            if bool(jnp.all(matrix_jax == 0.0)):
                return _as_gpu_compute_jax(np.zeros((self.shape[1], matrix_jax.shape[1]), dtype=compute_np_dtype))
            result = self._cupy_cache.astype(compute_cp_dtype, copy=False).T @ _to_cupy_compute(matrix_jax)
            return _cupy_to_jax(result)
        cupy, streaming_batch_size = self._streaming_gpu_context(batch_size)
        if cupy is not None and streaming_batch_size is not None:
            compute_np_dtype = gpu_compute_numpy_dtype()
            compute_cp_dtype = _cupy_compute_dtype(cupy)
            matrix_np = np.asarray(matrix, dtype=compute_np_dtype)
            if matrix_np.ndim != 2 or matrix_np.shape[0] != self.shape[0]:
                raise ValueError("sample matrix must match genotype row count.")
            if not np.any(matrix_np):
                return _as_gpu_compute_jax(np.zeros((self.shape[1], matrix_np.shape[1]), dtype=compute_np_dtype))
            matrix_gpu = _to_cupy_compute(matrix_np)
            output_gpu = cupy.empty((self.shape[1], matrix_np.shape[1]), dtype=compute_cp_dtype)
            for batch_slice, standardized_batch in _iter_standardized_gpu_batches(
                self.raw,
                self.variant_indices,
                self.means,
                self.scales,
                batch_size=streaming_batch_size,
                cupy=cupy,
                dtype=compute_cp_dtype,
            ):
                output_gpu[batch_slice, :] = standardized_batch.T @ matrix_gpu
            return _cupy_to_jax(output_gpu)
        if self.supports_jax_dense_ops():
            matrix_jax = jnp.asarray(matrix, dtype=gpu_compute_jax_dtype())
            if matrix_jax.ndim != 2 or matrix_jax.shape[0] != self.shape[0]:
                raise ValueError("sample matrix must match genotype row count.")
            return self._ensure_jax_cache().T @ matrix_jax
        m_np = np.asarray(matrix, dtype=gpu_compute_numpy_dtype())
        if m_np.ndim != 2 or m_np.shape[0] != self.shape[0]:
            raise ValueError("sample matrix must match genotype row count.")
        if not np.any(m_np):
            return _as_gpu_compute_jax(np.zeros((self.shape[1], m_np.shape[1]), dtype=m_np.dtype))
        if self._uses_hybrid_backend():
            sparse_parent_local_indices, dense_parent_local_indices = self._hybrid_parent_local_indices()
            output = np.empty((self.shape[1], m_np.shape[1]), dtype=m_np.dtype)
            if sparse_parent_local_indices.size > 0:
                if self._sparse_backend is None:
                    raise RuntimeError("hybrid sparse backend is not configured.")
                output[sparse_parent_local_indices, :] = self._sparse_backend.transpose_matmat(m_np)
            if dense_parent_local_indices.size > 0:
                if self._dense_backend is None:
                    raise RuntimeError("hybrid dense backend is not configured.")
                output[dense_parent_local_indices, :] = np.asarray(
                    self._dense_backend.transpose_matmat(m_np, batch_size=batch_size),
                    dtype=m_np.dtype,
                )
            return _as_gpu_compute_jax(output)
        output = np.empty((self.shape[1], m_np.shape[1]), dtype=m_np.dtype)
        for batch in self.iter_column_batches(batch_size=batch_size):
            output[batch.variant_indices, :] = np.asarray(batch.values, dtype=output.dtype).T @ m_np
        return _as_gpu_compute_jax(output)


def _resolve_variant_indices(
    variant_count: int,
    variant_indices: Sequence[int] | np.ndarray | None,
) -> np.ndarray:
    if variant_indices is None:
        return np.arange(variant_count, dtype=np.int32)
    resolved_indices = np.asarray(variant_indices, dtype=np.int32)
    if resolved_indices.ndim != 1:
        raise ValueError("variant_indices must be 1D.")
    return resolved_indices


# Auto-choose a good batch size based on sample count and memory budget.
# Larger batches = fewer I/O round-trips = faster, but each batch must fit
# in memory.  With 447k samples at 4 bytes each, one variant = ~1.7 MB,
# so 500 MB budget => ~279 variants per batch.
def auto_batch_size(sample_count: int) -> int:
    """Pick a genotype batch size that fits within the memory budget."""
    if sample_count < 1:
        return DEFAULT_GENOTYPE_BATCH_SIZE
    bytes_per_variant = sample_count * 4  # float32
    memory_capped = max(BED_READER_TARGET_BATCH_BYTES // max(bytes_per_variant, 1), 1)
    return max(MIN_BED_READER_BATCH_SIZE, min(DEFAULT_GENOTYPE_BATCH_SIZE, memory_capped))


def auto_batch_size_i8(sample_count: int) -> int:
    """Pick an int8-native batch size that fits the same IO budget."""
    if sample_count < 1:
        return DEFAULT_GENOTYPE_BATCH_SIZE
    bytes_per_variant = sample_count  # int8
    memory_capped = max(BED_READER_TARGET_BATCH_BYTES // max(bytes_per_variant, 1), 1)
    return max(MIN_BED_READER_BATCH_SIZE, min(int(memory_capped), 8 * DEFAULT_GENOTYPE_BATCH_SIZE))


def _effective_standardized_streaming_batch_size(
    sample_count: int,
    requested_batch_size: int,
    target_batch_bytes: int = STANDARDIZED_STREAMING_TARGET_BATCH_BYTES,
) -> int:
    if sample_count < 1:
        raise ValueError("sample_count must be positive.")
    if requested_batch_size < 1:
        raise ValueError("requested_batch_size must be positive.")
    if target_batch_bytes < 1:
        raise ValueError("target_batch_bytes must be positive.")
    bytes_per_variant = sample_count * np.dtype(np.float32).itemsize
    memory_capped_batch_size = max(target_batch_bytes // max(bytes_per_variant, 1), 1)
    return max(1, min(requested_batch_size, max(memory_capped_batch_size, MIN_BED_READER_BATCH_SIZE)))


# Cap the batch size so each batch doesn't exceed the memory budget.
# With 447k samples at 4 bytes (float32) each, one variant = ~1.7 MB.
# At the 500 MB budget that's ~279 variants per batch.  We also enforce
# a minimum of 32 variants per batch to avoid excessive I/O overhead.
def _effective_bed_reader_batch_size(
    sample_count: int,
    requested_batch_size: int,
) -> int:
    if sample_count < 1:
        raise ValueError("sample_count must be positive.")
    if requested_batch_size < 1:
        raise ValueError("requested_batch_size must be positive.")
    bytes_per_variant = sample_count * np.dtype(np.float32).itemsize
    memory_capped_batch_size = max(BED_READER_TARGET_BATCH_BYTES // max(bytes_per_variant, 1), 1)
    return max(1, min(requested_batch_size, max(memory_capped_batch_size, MIN_BED_READER_BATCH_SIZE)))


# Optimization: if the requested variant indices are consecutive (e.g. [5,6,7,8]),
# convert to a slice (5:9) which the PLINK reader can read much faster (sequential disk I/O)
# than random-access indexing.  Falls back to an index array for non-contiguous indices.
def _contiguous_index_or_slice(indices: np.ndarray) -> slice | np.ndarray:
    resolved_indices = np.asarray(indices, dtype=np.intp)
    if resolved_indices.ndim != 1:
        raise ValueError("indices must be 1D.")
    if resolved_indices.size == 0:
        return slice(0, 0, 1)
    if resolved_indices.size == 1:
        start = int(resolved_indices[0])
        return slice(start, start + 1, 1)
    deltas = np.diff(resolved_indices)
    if np.all(deltas == 1):
        return slice(int(resolved_indices[0]), int(resolved_indices[-1]) + 1, 1)
    return np.ascontiguousarray(resolved_indices, dtype=np.intp)


def _standardize_batch(batch: np.ndarray, means: np.ndarray, scales: np.ndarray) -> np.ndarray:
    batch_f32 = np.asarray(batch, dtype=np.float32)
    means_f32 = np.asarray(means, dtype=np.float32)
    scales_f32 = np.asarray(scales, dtype=np.float32)
    standardized = batch_f32 - means_f32[None, :]
    standardized[np.isnan(batch_f32)] = 0.0
    standardized /= scales_f32[None, :]
    return standardized.astype(np.float32, copy=False)


def _int8_batch_to_float32(batch: np.ndarray) -> np.ndarray:
    batch_i8 = np.asarray(batch, dtype=np.int8)
    batch_f32 = batch_i8.astype(np.float32)
    batch_f32[batch_i8 == PLINK_MISSING_INT8] = np.nan
    return batch_f32


def _standardize_batch_i8(batch: np.ndarray, means: np.ndarray, scales: np.ndarray) -> np.ndarray:
    batch_i8 = np.asarray(batch, dtype=np.int8)
    means_f32 = np.asarray(means, dtype=np.float32)
    scales_f32 = np.asarray(scales, dtype=np.float32)
    standardized = batch_i8.astype(np.float32)
    missing_mask = batch_i8 == PLINK_MISSING_INT8
    standardized -= means_f32[None, :]
    standardized[missing_mask] = 0.0
    standardized /= scales_f32[None, :]
    return standardized.astype(np.float32, copy=False)

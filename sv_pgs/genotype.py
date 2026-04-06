from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
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


def as_raw_genotype_matrix(genotypes: RawGenotypeMatrix | np.ndarray) -> RawGenotypeMatrix:
    if isinstance(genotypes, RawGenotypeMatrix):
        return genotypes
    array = np.asanyarray(genotypes)
    if array.dtype == np.int8:
        return Int8RawGenotypeMatrix(array)
    return DenseRawGenotypeMatrix(np.asarray(array, dtype=np.float32))


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

    def standardized(self, means: np.ndarray, scales: np.ndarray) -> StandardizedGenotypeMatrix:
        return StandardizedGenotypeMatrix(
            raw=self,
            means=np.asarray(means, dtype=np.float32),
            scales=np.asarray(scales, dtype=np.float32),
            variant_indices=np.arange(self.shape[1], dtype=np.int32),
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
        missing_mask = cupy.isnan(standardized)
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
    sample_count: int | None = field(default=None, repr=False)
    _dense_cache: np.ndarray | None = field(init=False, default=None, repr=False)
    _cupy_cache: Any | None = field(init=False, default=None, repr=False)  # cupy.ndarray
    _jax_cache: jax.Array | None = field(init=False, default=None, repr=False)
    _cupy_subset_cache: Any | None = field(init=False, default=None, repr=False)
    _cupy_subset_cache_local_indices: np.ndarray | None = field(init=False, default=None, repr=False)
    _local_cache_directory: tempfile.TemporaryDirectory[str] | None = field(init=False, default=None, repr=False)
    _n_samples: int = field(init=False, default=0, repr=False)

    def __post_init__(self) -> None:
        self.means = np.asarray(self.means, dtype=np.float32)
        self.scales = np.asarray(self.scales, dtype=np.float32)
        if self.means.ndim != 1 or self.scales.ndim != 1 or self.means.shape != self.scales.shape:
            raise ValueError("means and scales must be matching 1D arrays.")
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
                for batch_slice, standardized_batch in _iter_standardized_gpu_batches(
                    self.raw,
                    self.variant_indices,
                    self.means,
                    self.scales,
                    batch_size=auto_batch_size(self.shape[0]),
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
        batch_size = auto_batch_size(self.shape[0])
        selected_variant_count = int(self.variant_indices.shape[0])
        log(
            "    caching reduced raw genotypes locally as int8 "
            + f"({selected_variant_count} variants x {self.shape[0]} samples)  mem={mem()}"
        )
        cache_directory = tempfile.TemporaryDirectory(prefix="svpgs-genotype-")
        cache_path = Path(cache_directory.name) / "reduced_raw_i8.npy"
        try:
            raw_cache = np.lib.format.open_memmap(
                cache_path,
                mode="w+",
                dtype=np.int8,
                shape=self.shape,
                fortran_order=True,
            )
            local_start = 0
            raw_int8 = cast(Int8BatchCapable, self.raw)
            for raw_batch in raw_int8.iter_column_batches_i8(self.variant_indices, batch_size=batch_size):
                batch_width = raw_batch.values.shape[1]
                local_stop = local_start + batch_width
                raw_cache[:, local_start:local_stop] = raw_batch.values
                local_start = local_stop
            raw_cache.flush()
            del raw_cache
            rebased_raw = Int8RawGenotypeMatrix(np.load(cache_path, mmap_mode="r"))
            self.raw = rebased_raw
            self.means = np.asarray(self.means[self.variant_indices], dtype=np.float32)
            self.scales = np.asarray(self.scales[self.variant_indices], dtype=np.float32)
            self.variant_indices = np.arange(selected_variant_count, dtype=np.int32)
            self._local_cache_directory = cache_directory
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
        batch_size = auto_batch_size(self.shape[0])
        selected_variant_count = int(self.variant_indices.shape[0])
        temp_directory = Path(tempfile.mkdtemp(prefix=f"{cache_path.name}.tmp.", dir=cache_path.parent))
        temp_path = temp_directory / cache_path.name
        log(
            "    persisting reduced raw genotypes as int8 "
            + f"({selected_variant_count} variants x {self.shape[0]} samples) → {cache_path}  mem={mem()}"
        )
        try:
            raw_cache = np.lib.format.open_memmap(
                temp_path,
                mode="w+",
                dtype=np.int8,
                shape=self.shape,
                fortran_order=True,
            )
            local_start = 0
            raw_int8 = cast(Int8BatchCapable, self.raw)
            for raw_batch in raw_int8.iter_column_batches_i8(self.variant_indices, batch_size=batch_size):
                batch_width = raw_batch.values.shape[1]
                local_stop = local_start + batch_width
                raw_cache[:, local_start:local_stop] = raw_batch.values
                local_start = local_stop
            raw_cache.flush()
            del raw_cache
            temp_path.replace(cache_path)
            persisted_raw = Int8RawGenotypeMatrix(np.load(cache_path, mmap_mode="r"))
            self.raw = persisted_raw
            self.means = np.asarray(self.means[self.variant_indices], dtype=np.float32)
            self.scales = np.asarray(self.scales[self.variant_indices], dtype=np.float32)
            self.variant_indices = np.arange(selected_variant_count, dtype=np.int32)
            self._local_cache_directory = None
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
        subset = StandardizedGenotypeMatrix(
            raw=self.raw,
            means=self.means,
            scales=self.scales,
            variant_indices=self.variant_indices[resolved_local_indices],
            sample_count=self.shape[0],
        )
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
        if self._cupy_cache is not None or self.supports_jax_dense_ops() or self.raw is None:
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
            coeff_jax = jnp.ravel(jnp.asarray(coefficients, dtype=jnp.float32))
            if coeff_jax.shape[0] != self.shape[1]:
                raise ValueError("coefficient vector must match genotype column count.")
            if bool(jnp.all(coeff_jax == 0.0)):
                return _as_gpu_compute_jax(np.zeros(self.shape[0], dtype=np.float32))
            coeff_cupy = _to_cupy_float32(coeff_jax)
            result = self._cupy_cache @ coeff_cupy
            return _cupy_to_jax(result)
        cupy, streaming_batch_size = self._streaming_gpu_context(batch_size)
        if cupy is not None and streaming_batch_size is not None:
            coeff_np = np.asarray(coefficients, dtype=np.float64).ravel()
            if coeff_np.shape[0] != self.shape[1]:
                raise ValueError("coefficient vector must match genotype column count.")
            if not np.any(coeff_np):
                return _as_gpu_compute_jax(np.zeros(self.shape[0], dtype=np.float64))
            coeff_gpu = _to_cupy_float64(coeff_np)
            result_gpu = cupy.zeros(self.shape[0], dtype=cupy.float64)
            for batch_slice, standardized_batch in _iter_standardized_gpu_batches(
                self.raw,
                self.variant_indices,
                self.means,
                self.scales,
                batch_size=streaming_batch_size,
                cupy=cupy,
                dtype=cupy.float64,
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
        result = np.zeros(self.shape[0], dtype=gpu_compute_numpy_dtype())
        for batch in self.iter_column_batches(batch_size=batch_size):
            batch_values = np.asarray(batch.values, dtype=result.dtype)
            result += batch_values @ coeff_np[batch.variant_indices]
        return _as_gpu_compute_jax(result)

    def matmat(self, matrix: np.ndarray | jnp.ndarray, batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE) -> jnp.ndarray:
        """Compute X_std @ M. When CuPy GPU-cached: cuBLAS matmul. Otherwise: batched numpy."""
        if self._cupy_cache is not None:
            matrix_jax = jnp.asarray(matrix, dtype=jnp.float32)
            if matrix_jax.ndim != 2 or matrix_jax.shape[0] != self.shape[1]:
                raise ValueError("variant matrix must match genotype column count.")
            if bool(jnp.all(matrix_jax == 0.0)):
                return _as_gpu_compute_jax(np.zeros((self.shape[0], matrix_jax.shape[1]), dtype=np.float32))
            result = self._cupy_cache @ _to_cupy_float32(matrix_jax)
            return _cupy_to_jax(result)
        cupy, streaming_batch_size = self._streaming_gpu_context(batch_size)
        if cupy is not None and streaming_batch_size is not None:
            matrix_np = np.asarray(matrix, dtype=np.float64)
            if matrix_np.ndim != 2 or matrix_np.shape[0] != self.shape[1]:
                raise ValueError("variant matrix must match genotype column count.")
            if not np.any(matrix_np):
                return _as_gpu_compute_jax(np.zeros((self.shape[0], matrix_np.shape[1]), dtype=np.float64))
            matrix_gpu = _to_cupy_float64(matrix_np)
            result_gpu = cupy.zeros((self.shape[0], matrix_np.shape[1]), dtype=cupy.float64)
            for batch_slice, standardized_batch in _iter_standardized_gpu_batches(
                self.raw,
                self.variant_indices,
                self.means,
                self.scales,
                batch_size=streaming_batch_size,
                cupy=cupy,
                dtype=cupy.float64,
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
        output = np.zeros((self.shape[0], m_np.shape[1]), dtype=m_np.dtype)
        for batch in self.iter_column_batches(batch_size=batch_size):
            batch_values = np.asarray(batch.values, dtype=output.dtype)
            output += batch_values @ m_np[batch.variant_indices, :]
        return _as_gpu_compute_jax(output)

    def transpose_matvec(self, vector: np.ndarray | jnp.ndarray, batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE) -> jnp.ndarray:
        """Compute X_std^T @ v. When CuPy GPU-cached: cuBLAS matmul. Otherwise: batched numpy."""
        if self._cupy_cache is not None:
            vector_jax = jnp.ravel(jnp.asarray(vector, dtype=jnp.float32))
            if vector_jax.shape[0] != self.shape[0]:
                raise ValueError("sample vector must match genotype row count.")
            if bool(jnp.all(vector_jax == 0.0)):
                return _as_gpu_compute_jax(np.zeros(self.shape[1], dtype=np.float32))
            result = self._cupy_cache.T @ _to_cupy_float32(vector_jax)
            return _cupy_to_jax(result)
        cupy, streaming_batch_size = self._streaming_gpu_context(batch_size)
        if cupy is not None and streaming_batch_size is not None:
            vector_np = np.asarray(vector, dtype=np.float64).ravel()
            if vector_np.shape[0] != self.shape[0]:
                raise ValueError("sample vector must match genotype row count.")
            if not np.any(vector_np):
                return _as_gpu_compute_jax(np.zeros(self.shape[1], dtype=np.float64))
            vector_gpu = _to_cupy_float64(vector_np)
            output_gpu = cupy.empty(self.shape[1], dtype=cupy.float64)
            for batch_slice, standardized_batch in _iter_standardized_gpu_batches(
                self.raw,
                self.variant_indices,
                self.means,
                self.scales,
                batch_size=streaming_batch_size,
                cupy=cupy,
                dtype=cupy.float64,
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
        output = np.empty(self.shape[1], dtype=v_np.dtype)
        for batch in self.iter_column_batches(batch_size=batch_size):
            output[batch.variant_indices] = np.asarray(batch.values, dtype=output.dtype).T @ v_np
        return _as_gpu_compute_jax(output)

    def transpose_matmat(self, matrix: np.ndarray | jnp.ndarray, batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE) -> jnp.ndarray:
        """Compute X_std^T @ M. When CuPy GPU-cached: cuBLAS matmul. Otherwise: batched numpy."""
        if self._cupy_cache is not None:
            matrix_jax = jnp.asarray(matrix, dtype=jnp.float32)
            if matrix_jax.ndim != 2 or matrix_jax.shape[0] != self.shape[0]:
                raise ValueError("sample matrix must match genotype row count.")
            if bool(jnp.all(matrix_jax == 0.0)):
                return _as_gpu_compute_jax(np.zeros((self.shape[1], matrix_jax.shape[1]), dtype=np.float32))
            result = self._cupy_cache.T @ _to_cupy_float32(matrix_jax)
            return _cupy_to_jax(result)
        cupy, streaming_batch_size = self._streaming_gpu_context(batch_size)
        if cupy is not None and streaming_batch_size is not None:
            matrix_np = np.asarray(matrix, dtype=np.float64)
            if matrix_np.ndim != 2 or matrix_np.shape[0] != self.shape[0]:
                raise ValueError("sample matrix must match genotype row count.")
            if not np.any(matrix_np):
                return _as_gpu_compute_jax(np.zeros((self.shape[1], matrix_np.shape[1]), dtype=np.float64))
            matrix_gpu = _to_cupy_float64(matrix_np)
            output_gpu = cupy.empty((self.shape[1], matrix_np.shape[1]), dtype=cupy.float64)
            for batch_slice, standardized_batch in _iter_standardized_gpu_batches(
                self.raw,
                self.variant_indices,
                self.means,
                self.scales,
                batch_size=streaming_batch_size,
                cupy=cupy,
                dtype=cupy.float64,
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

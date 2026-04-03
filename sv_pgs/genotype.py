from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import sv_pgs._jax  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np
from bed_reader import open_bed

DEFAULT_GENOTYPE_BATCH_SIZE = 1024  # fallback when sample count is unknown

# Memory cap per bed_reader batch.  PLINK .bed files store genotypes on disk
# as 2 bits per sample, but we expand to int8 or float32 in memory.  The JAX
# screening kernels also create float32 intermediates (~10 bytes/element peak).
# This budget ensures each batch fits comfortably in GPU/CPU memory:
#   500 MB / (447k samples * 4 bytes) ≈ 279 variants per batch
BED_READER_TARGET_BATCH_BYTES = 500_000_000
MIN_BED_READER_BATCH_SIZE = 32  # always read at least this many variants

# If the reduced genotype matrix (after tie-group dedup) is smaller than 4 GB,
# cache it in RAM.  This avoids re-reading from disk on every EM iteration
# (typically 10-30 iterations), giving a huge speedup.
MATERIALIZE_THRESHOLD_BYTES = 4_000_000_000  # 4 GB


def as_raw_genotype_matrix(genotypes: RawGenotypeMatrix | np.ndarray) -> RawGenotypeMatrix:
    if isinstance(genotypes, RawGenotypeMatrix):
        return genotypes
    return DenseRawGenotypeMatrix(np.asarray(genotypes, dtype=np.float32))


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


@dataclass(slots=True)
class DenseRawGenotypeMatrix(RawGenotypeMatrix):
    matrix: np.ndarray

    def __post_init__(self) -> None:
        if self.matrix.dtype == np.int8:
            pass  # int8 with -1 sentinel for missing — 4x smaller than float32
        else:
            self.matrix = np.asarray(self.matrix, dtype=np.float32)
        if self.matrix.ndim != 2:
            raise ValueError("genotypes must be 2D.")

    def _to_float32(self, batch: np.ndarray) -> np.ndarray:
        """Convert a column slice to float32, replacing missing sentinels with NaN."""
        if self.matrix.dtype == np.int8:
            result = batch.astype(np.float32)
            result[batch == -1] = np.nan
            return result
        return np.asarray(batch, dtype=np.float32)

    @property
    def shape(self) -> tuple[int, int]:
        return self.matrix.shape

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
class PlinkRawGenotypeMatrix(RawGenotypeMatrix):
    bed_path: Path
    sample_indices: np.ndarray
    variant_count: int
    total_sample_count: int
    batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE
    _reader: open_bed | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.sample_indices = np.asarray(self.sample_indices, dtype=np.intp)
        if self.sample_indices.ndim != 1:
            raise ValueError("sample_indices must be 1D.")

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.sample_indices.shape[0]), int(self.variant_count)

    def _read_batch(self, reader: open_bed, batch_indices: np.ndarray) -> np.ndarray:
        """Read one batch as int8, convert to float32 with NaN for missing."""
        raw_i8 = self._read_batch_i8(reader, batch_indices)
        result = np.asarray(raw_i8, dtype=np.float32)
        result[raw_i8 == -127] = np.nan
        return result

    def _read_batch_i8(self, reader: open_bed, batch_indices: np.ndarray) -> np.ndarray:
        """Read one batch as raw int8 (0/1/2/-127). No float conversion."""
        sample_index = _contiguous_index_or_slice(self.sample_indices)
        col_index = _contiguous_index_or_slice(batch_indices)
        return np.asarray(
            reader.read(index=(sample_index, col_index), dtype="int8", order="F", num_threads=None),
            dtype=np.int8,
        )

    def iter_column_batches_i8(
        self,
        batch_size: int | None = None,
    ) -> Iterator[RawGenotypeBatch]:
        """Iterate as int8 batches (4x less memory, no float conversion).

        Values are 0/1/2/-127(missing). Callers must handle -127 sentinel.
        """
        resolved_indices = _resolve_variant_indices(self.variant_count, None)
        # int8 reads are 4x smaller than float32, but JAX kernels still expand to
        # float32 intermediates (~10 bytes/element peak), so do NOT inflate batch size.
        requested = max(int(self.batch_size if batch_size is None else batch_size), 1)
        bytes_per_variant = self.shape[0]  # 1 byte per sample for int8
        max_variants = max(BED_READER_TARGET_BATCH_BYTES // max(bytes_per_variant, 1), MIN_BED_READER_BATCH_SIZE)
        safe_batch_size = min(requested, max_variants)
        reader = self._bed_reader()
        total = resolved_indices.shape[0]
        from sv_pgs.progress import log, mem
        batch_mb = self.shape[0] * safe_batch_size / (1024 * 1024)
        n_batches = (total + safe_batch_size - 1) // safe_batch_size
        log(f"    int8 batch: {safe_batch_size} variants x {self.shape[0]} samples = {batch_mb:.0f} MB/batch, {n_batches} batches  mem={mem()}")

        if total <= safe_batch_size:
            values = self._read_batch_i8(reader, resolved_indices)
            yield RawGenotypeBatch(variant_indices=resolved_indices, values=values)
            return

        # Prefetch with background thread
        from concurrent.futures import ThreadPoolExecutor
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

    def _bed_reader(self) -> open_bed:
        if self._reader is None:
            from sv_pgs.progress import log, mem
            log(f"    opening bed_reader (lazy, no metadata): iid_count={self.total_sample_count} sid_count={self.variant_count}  mem={mem()}")
            self._reader = open_bed(
                self.bed_path,
                iid_count=self.total_sample_count,
                sid_count=self.variant_count,
                properties={},
                skip_format_check=True,
                num_threads=None,
            )
            log(f"    bed_reader opened  mem={mem()}")
        return self._reader


@dataclass(slots=True)
class StandardizedGenotypeMatrix:
    """A genotype matrix that applies z-score standardization on the fly.

    For each variant j: standardized_value = (raw_dosage - mean_j) / scale_j
    Missing values (NaN) are imputed to the mean (producing 0 after centering).

    This wraps a RawGenotypeMatrix and applies the transformation lazily during
    iteration, so we never need to store the full standardized matrix unless
    we choose to cache it (try_materialize).  Supports matrix-vector products
    (matvec, transpose_matvec) needed by the Bayesian inference engine.

    When the matrix fits in GPU VRAM (try_materialize_gpu), it is uploaded once
    and kept resident — matvec/transpose_matvec then become single GPU matmul
    calls with zero CPU→GPU transfer overhead per iteration.
    """
    raw: RawGenotypeMatrix
    means: np.ndarray       # per-variant mean from training data
    scales: np.ndarray      # per-variant std dev from training data
    variant_indices: np.ndarray  # which columns of raw to use (for subsetting)
    _dense_cache: np.ndarray | None = field(init=False, default=None, repr=False)
    _gpu_cache: jnp.ndarray | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.means = np.asarray(self.means, dtype=np.float32)
        self.scales = np.asarray(self.scales, dtype=np.float32)
        self.variant_indices = np.asarray(self.variant_indices, dtype=np.int32)
        if self.variant_indices.ndim != 1:
            raise ValueError("variant_indices must be 1D.")
        if np.any(self.variant_indices < 0) or np.any(self.variant_indices >= self.raw.shape[1]):
            raise ValueError("variant_indices out of bounds.")

    @property
    def shape(self) -> tuple[int, int]:
        return self.raw.shape[0], int(self.variant_indices.shape[0])

    def dense_bytes(self) -> int:
        """Estimated bytes if materialized as float32."""
        return int(self.shape[0]) * int(self.shape[1]) * 4

    def try_materialize_gpu(self) -> bool:
        """Materialize the standardized matrix directly into GPU VRAM.

        If the matrix fits (float32), uploads it once so that matvec/transpose
        become single GPU matmul calls with zero per-iteration transfer overhead.
        Returns True if now resident on GPU, False on CPU-only or OOM.
        """
        if self._gpu_cache is not None:
            return True
        try:
            devices = jax.devices()
            if not devices or devices[0].platform == "cpu":
                return False
        except Exception:
            return False
        nbytes_f32 = self.dense_bytes()
        from sv_pgs.progress import log, mem
        log(f"    materializing {self.shape[1]} variants x {self.shape[0]} samples ({nbytes_f32 / 1e9:.1f} GB) onto GPU  mem={mem()}")
        try:
            # Build standardized matrix on CPU in batches, then upload to GPU
            cpu_matrix = np.empty(self.shape, dtype=np.float32)
            for batch in self.iter_column_batches(batch_size=auto_batch_size(self.shape[0])):
                cpu_matrix[:, batch.variant_indices] = batch.values
            self._gpu_cache = jnp.asarray(cpu_matrix, dtype=jnp.float32)
            del cpu_matrix
            # Force the transfer to complete and verify it landed on GPU
            self._gpu_cache.block_until_ready()
            log(f"    GPU-resident matrix ready ({self._gpu_cache.dtype}, {self._gpu_cache.device()})  mem={mem()}")
            return True
        except Exception as e:
            log(f"    GPU materialization failed ({e}), falling back to streaming  mem={mem()}")
            self._gpu_cache = None
            return False

    def try_materialize(self) -> bool:
        """Materialize into RAM if below the auto-materialize threshold.

        Returns True if now cached in memory, False if still streaming from disk.
        """
        if self._gpu_cache is not None or self._dense_cache is not None:
            return True
        nbytes = self.dense_bytes()
        if nbytes > MATERIALIZE_THRESHOLD_BYTES:
            return False
        from sv_pgs.progress import log, mem
        log(f"    auto-materializing {self.shape[1]} variants x {self.shape[0]} samples ({nbytes / 1e9:.1f} GB) into RAM  mem={mem()}")
        self._dense_cache = self.materialize()
        log(f"    materialized  mem={mem()}")
        return True

    def subset(self, local_variant_indices: Sequence[int] | np.ndarray) -> StandardizedGenotypeMatrix:
        resolved_local_indices = np.asarray(local_variant_indices, dtype=np.int32)
        subset = StandardizedGenotypeMatrix(
            raw=self.raw,
            means=self.means,
            scales=self.scales,
            variant_indices=self.variant_indices[resolved_local_indices],
        )
        if self._gpu_cache is not None:
            subset._gpu_cache = self._gpu_cache[:, resolved_local_indices]
        elif self._dense_cache is not None:
            subset._dense_cache = np.asarray(self._dense_cache[:, resolved_local_indices], dtype=np.float32)
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
        safe_batch_size = max(int(batch_size), 1)
        local_start = 0
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
        if self._gpu_cache is not None:
            self._dense_cache = np.asarray(self._gpu_cache, dtype=np.float32)
            return self._dense_cache
        matrix = np.empty(self.shape, dtype=np.float32)
        for batch in self.iter_column_batches(batch_size=batch_size):
            matrix[:, batch.variant_indices] = batch.values
        self._dense_cache = matrix
        return matrix

    def matvec(self, coefficients: np.ndarray | jnp.ndarray, batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE) -> jnp.ndarray:
        """Compute X_std @ beta (genotype matrix times coefficient vector).

        When GPU-cached: single GPU matmul. Otherwise: batched numpy on CPU.
        """
        if self._gpu_cache is not None:
            coeff = jnp.asarray(coefficients, dtype=jnp.float32)
            if coeff.ndim != 1 or coeff.shape[0] != self.shape[1]:
                raise ValueError("coefficient vector must match genotype column count.")
            return (self._gpu_cache @ coeff).astype(jnp.float64)
        coeff_np = np.asarray(coefficients, dtype=np.float64).ravel()
        if coeff_np.shape[0] != self.shape[1]:
            raise ValueError("coefficient vector must match genotype column count.")
        result = np.zeros(self.shape[0], dtype=np.float64)
        for batch in self.iter_column_batches(batch_size=batch_size):
            batch_f64 = np.asarray(batch.values, dtype=np.float64)
            result += batch_f64 @ coeff_np[batch.variant_indices]
        return jnp.asarray(result, dtype=jnp.float64)

    def transpose_matvec(self, vector: np.ndarray | jnp.ndarray, batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE) -> jnp.ndarray:
        """Compute X_std^T @ v. When GPU-cached: single GPU matmul. Otherwise: batched numpy."""
        if self._gpu_cache is not None:
            v = jnp.asarray(vector, dtype=jnp.float32)
            if v.ndim != 1 or v.shape[0] != self.shape[0]:
                raise ValueError("sample vector must match genotype row count.")
            return (self._gpu_cache.T @ v).astype(jnp.float64)
        v_np = np.asarray(vector, dtype=np.float64).ravel()
        if v_np.shape[0] != self.shape[0]:
            raise ValueError("sample vector must match genotype row count.")
        output = np.empty(self.shape[1], dtype=np.float64)
        for batch in self.iter_column_batches(batch_size=batch_size):
            output[batch.variant_indices] = np.asarray(batch.values, dtype=np.float64).T @ v_np
        return jnp.asarray(output, dtype=jnp.float64)

    def transpose_matmat(self, matrix: np.ndarray | jnp.ndarray, batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE) -> jnp.ndarray:
        if self._gpu_cache is not None:
            m = jnp.asarray(matrix, dtype=jnp.float32)
            if m.ndim != 2 or m.shape[0] != self.shape[0]:
                raise ValueError("sample matrix must match genotype row count.")
            return (self._gpu_cache.T @ m).astype(jnp.float64)
        m_np = np.asarray(matrix, dtype=np.float64)
        if m_np.ndim != 2 or m_np.shape[0] != self.shape[0]:
            raise ValueError("sample matrix must match genotype row count.")
        output = np.empty((self.shape[1], m_np.shape[1]), dtype=np.float64)
        for batch in self.iter_column_batches(batch_size=batch_size):
            output[batch.variant_indices, :] = np.asarray(batch.values, dtype=np.float64).T @ m_np
        return jnp.asarray(output, dtype=jnp.float64)


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
# convert to a slice (5:9) which bed_reader can read much faster (sequential disk I/O)
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


# Z-score standardize a batch of genotype columns:
#   1. Replace NaN (missing) with the variant's mean (mean imputation)
#   2. Subtract the mean (centering)
#   3. Divide by the standard deviation (scaling)
# After this, each variant column has mean ~0 and std ~1.
# JIT-compiled for GPU acceleration — first call compiles, subsequent calls are fast.
@jax.jit
def _standardize_batch_jit(batch_jax: jnp.ndarray, mean_jax: jnp.ndarray, scale_jax: jnp.ndarray) -> jnp.ndarray:
    nan_mask = jnp.isnan(batch_jax)
    centered = batch_jax - mean_jax[None, :]
    return jnp.where(nan_mask, 0.0, centered) / scale_jax[None, :]


def _standardize_batch(batch: np.ndarray, means: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return np.asarray(_standardize_batch_jit(
        jnp.asarray(batch, dtype=jnp.float32),
        jnp.asarray(means, dtype=jnp.float32),
        jnp.asarray(scales, dtype=jnp.float32),
    ), dtype=np.float32)

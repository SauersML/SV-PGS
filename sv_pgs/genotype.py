from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import io
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterator, Protocol, Sequence, TypeGuard, cast

import sv_pgs._jax as _jax_side_effects  # side-effect: configures JAX/XLA env
del _jax_side_effects
import jax
from jax import core as jax_core
import jax.numpy as jnp
from jax import dlpack as jax_dlpack
import numpy as np
from sv_pgs._jax import gpu_compute_jax_dtype, gpu_compute_numpy_dtype, jax_dense_linear_algebra_preferred
from sv_pgs._typing import JaxArray, NDArray
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
STANDARDIZED_STREAMING_TARGET_BATCH_BYTES = 1_024_000_000
LOCAL_INT8_STANDARDIZED_STREAMING_TARGET_BATCH_BYTES = 4_096_000_000
GPU_STANDARDIZED_STREAMING_TARGET_BATCH_BYTES = 512_000_000
GPU_STANDARDIZED_PREFETCH_TARGET_BYTES = 1_024_000_000
GPU_STANDARDIZED_DYNAMIC_FREE_FRACTION = 0.20
GPU_STANDARDIZED_DYNAMIC_RESERVE_BYTES = 512_000_000
PLINK_INT8_TARGET_BATCH_BYTES = 1_024_000_000
PLINK_INT8_MAX_PREFETCH_DEPTH = 1
PLINK_BED_READER_NUM_THREADS = min(16, max(1, os.cpu_count() or 1))

# If the reduced genotype matrix (after tie-group dedup) is smaller than 4 GB,
# cache it in RAM.  This avoids re-reading from disk on every EM iteration
# (typically 10-30 iterations), giving a huge speedup.
MATERIALIZE_THRESHOLD_BYTES = 4_000_000_000  # 4 GB
HYBRID_SPARSE_SUPPORT_THRESHOLD = 4_096
HYBRID_SPARSE_MIN_VARIANT_COUNT = 64
REDUCED_INT8_CACHE_FREE_SPACE_RESERVE_BYTES = 64 * 1024 * 1024
INT8_ONE_SHOT_GPU_BUDGET_FRACTION = 0.90
ROW_SUBSET_ONE_SHOT_MAX_SAMPLE_RATIO = 8.0


def _madvise_willneed_array(array: NDArray) -> None:
    """Best-effort posix_madvise(WILLNEED) on the mmap backing `array`.

    Tells the kernel we'll touch the whole int8 cache soon and to retain
    those pages. Without this, per-block "uploading raw int8 genotypes to
    GPU" runs at disk speed instead of RAM speed when pages get evicted
    between training blocks under memory pressure.
    """
    posix_madvise = getattr(os, "posix_madvise", None)
    willneed = getattr(os, "POSIX_MADV_WILLNEED", None)
    if posix_madvise is None or willneed is None:
        return
    try:
        import mmap as _mmap_module
    except ImportError:
        return
    base: object = array
    while getattr(base, "base", None) is not None:
        next_base = getattr(base, "base", None)
        if next_base is None:
            break
        base = next_base
        if isinstance(base, _mmap_module.mmap):
            break
    if not isinstance(base, _mmap_module.mmap):
        return
    try:
        posix_madvise(base, 0, len(base), willneed)
    except (OSError, ValueError):
        pass


def as_raw_genotype_matrix(genotypes: RawGenotypeMatrix | NDArray) -> RawGenotypeMatrix:
    if isinstance(genotypes, RawGenotypeMatrix):
        return genotypes
    array = np.asanyarray(genotypes)
    if array.dtype == np.int8:
        return Int8RawGenotypeMatrix(array)
    return DenseRawGenotypeMatrix(np.asarray(array, dtype=np.float32))


def _int8_npy_header_bytes(shape: tuple[int, int], *, fortran_order: bool) -> bytes:
    header_buffer = io.BytesIO()
    write_header: Any = np.lib.format.write_array_header_2_0
    dtype_to_descr: Any = np.lib.format.dtype_to_descr
    write_header(
        header_buffer,
        {
            "descr": dtype_to_descr(np.dtype(np.int8)),
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
    column_batches: Iterator[NDArray],
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
    variant_indices: NDArray
    values: NDArray


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
        variant_indices: Sequence[int] | NDArray | None = None,
        batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE,
    ) -> Iterator[RawGenotypeBatch]:
        raise NotImplementedError

    @abstractmethod
    def materialize(
        self,
        variant_indices: Sequence[int] | NDArray | None = None,
    ) -> NDArray:
        raise NotImplementedError

    def standardized(
        self,
        means: NDArray,
        scales: NDArray,
        support_counts: NDArray | None = None,
    ) -> StandardizedGenotypeMatrix:
        return StandardizedGenotypeMatrix(
            raw=self,
            means=np.asarray(means, dtype=np.float32),
            scales=np.asarray(scales, dtype=np.float32),
            variant_indices=np.arange(self.shape[1], dtype=np.int32),
            support_counts=None if support_counts is None else np.asarray(support_counts, dtype=np.int32),
        )

    def __array__(self, dtype: np.dtype[Any] | type | None = None) -> NDArray:
        matrix = self.materialize()
        if dtype is None:
            return matrix
        return np.asarray(matrix, dtype=dtype)


class Int8BatchCapable(Protocol):
    @property
    def shape(self) -> tuple[int, int]: ...
    def iter_column_batches(
        self,
        variant_indices: Sequence[int] | NDArray | None = None,
        batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE,
    ) -> Iterator[RawGenotypeBatch]: ...
    def iter_column_batches_i8(
        self,
        variant_indices: Sequence[int] | NDArray | None = None,
        batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE,
    ) -> Iterator[RawGenotypeBatch]: ...


def _supports_int8_batches(matrix: object) -> TypeGuard[Int8BatchCapable]:
    return hasattr(matrix, "iter_column_batches_i8")


@dataclass(slots=True)
class DenseRawGenotypeMatrix(RawGenotypeMatrix):
    matrix: NDArray

    def __post_init__(self) -> None:
        matrix_array = np.asanyarray(self.matrix)
        if matrix_array.dtype == np.int8:
            self.matrix = matrix_array  # preserve memmap-backed int8 arrays
        else:
            self.matrix = np.asarray(matrix_array, dtype=np.float32)
        if self.matrix.ndim != 2:
            raise ValueError("genotypes must be 2D.")

    def _to_float32(self, batch: NDArray) -> NDArray:
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
        variant_indices: Sequence[int] | NDArray | None = None,
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
        variant_indices: Sequence[int] | NDArray | None = None,
    ) -> NDArray:
        resolved_indices = _resolve_variant_indices(self.shape[1], variant_indices)
        return self._to_float32(self.matrix[:, resolved_indices])


@dataclass(slots=True)
class Int8RawGenotypeMatrix(RawGenotypeMatrix):
    matrix: NDArray

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
        variant_indices: Sequence[int] | NDArray | None = None,
        batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE,
    ) -> Iterator[RawGenotypeBatch]:
        resolved_indices = _resolve_variant_indices(self.shape[1], variant_indices)
        safe_batch_size = max(int(batch_size), 1)
        for start_index in range(0, resolved_indices.shape[0], safe_batch_size):
            batch_indices = resolved_indices[start_index : start_index + safe_batch_size]
            column_index = _contiguous_index_or_slice(batch_indices)
            yield RawGenotypeBatch(
                variant_indices=batch_indices,
                values=np.asarray(self.matrix[:, column_index], dtype=np.int8),
            )

    def iter_column_batches(
        self,
        variant_indices: Sequence[int] | NDArray | None = None,
        batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE,
    ) -> Iterator[RawGenotypeBatch]:
        for batch in self.iter_column_batches_i8(variant_indices, batch_size=batch_size):
            yield RawGenotypeBatch(
                variant_indices=batch.variant_indices,
                values=_int8_batch_to_float32(batch.values),
            )

    def materialize(
        self,
        variant_indices: Sequence[int] | NDArray | None = None,
    ) -> NDArray:
        resolved_indices = _resolve_variant_indices(self.shape[1], variant_indices)
        column_index = _contiguous_index_or_slice(resolved_indices)
        return _int8_batch_to_float32(self.matrix[:, column_index])


@dataclass(slots=True)
class IndexedRawGenotypeMatrix(RawGenotypeMatrix):
    """Expose only a selected subset of a child matrix's columns.

    Used by the multi-VCF dataset loader to drop cross-source duplicate
    variants without rewriting on-disk caches: the wrapper advertises
    shape (n_samples, len(selected_columns)) and routes column i to the
    child's column selected_columns[i]. Yielded batch.variant_indices stay
    in the wrapper's local coordinate space so downstream consumers can
    index back into us directly.
    """
    child: RawGenotypeMatrix
    selected_columns: NDArray

    def __post_init__(self) -> None:
        indices = np.asarray(self.selected_columns, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError("selected_columns must be 1D.")
        child_variant_count = int(self.child.shape[1])
        if indices.size and (indices.min() < 0 or indices.max() >= child_variant_count):
            raise ValueError("selected_columns contains an out-of-range index.")
        self.selected_columns = indices

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.child.shape[0]), int(self.selected_columns.shape[0])

    def _child_columns(self, local_indices: NDArray) -> NDArray:
        return self.selected_columns[local_indices].astype(np.int32, copy=False)

    def iter_column_batches(
        self,
        variant_indices: Sequence[int] | NDArray | None = None,
        batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE,
    ) -> Iterator[RawGenotypeBatch]:
        resolved_indices = _resolve_variant_indices(self.shape[1], variant_indices)
        safe_batch_size = max(int(batch_size), 1)
        for start_index in range(0, resolved_indices.shape[0], safe_batch_size):
            local_batch = resolved_indices[start_index : start_index + safe_batch_size]
            yield RawGenotypeBatch(
                variant_indices=local_batch,
                values=self.child.materialize(self._child_columns(local_batch)),
            )

    def iter_column_batches_i8(
        self,
        variant_indices: Sequence[int] | NDArray | None = None,
        batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE,
    ) -> Iterator[RawGenotypeBatch]:
        if not _supports_int8_batches(self.child):
            raise RuntimeError("int8 batch iteration requires the wrapped child to support iter_column_batches_i8.")
        resolved_indices = _resolve_variant_indices(self.shape[1], variant_indices)
        safe_batch_size = max(int(batch_size), 1)
        child = self.child
        for start_index in range(0, resolved_indices.shape[0], safe_batch_size):
            local_batch = resolved_indices[start_index : start_index + safe_batch_size]
            child_batch_indices = self._child_columns(local_batch)
            # Read the entire local batch as one child request so columns come
            # back in our coordinate order — no concat or reordering needed.
            child_batch = next(child.iter_column_batches_i8(
                child_batch_indices, batch_size=max(child_batch_indices.shape[0], 1),
            ))
            yield RawGenotypeBatch(
                variant_indices=local_batch,
                values=np.asarray(child_batch.values, dtype=np.int8),
            )

    def materialize(
        self,
        variant_indices: Sequence[int] | NDArray | None = None,
    ) -> NDArray:
        resolved_indices = _resolve_variant_indices(self.shape[1], variant_indices)
        return self.child.materialize(self._child_columns(resolved_indices))


@dataclass(slots=True)
class RowSubsetRawGenotypeMatrix(RawGenotypeMatrix):
    """Lazily expose a row-subset (sample-subset) of a child matrix.

    Replaces the upfront write-temp-mmap reindex path that the multi-VCF /
    multi-source dataset loader previously used. The earlier approach
    materialized a brand-new (n_kept_samples, n_variants) int8 mmap on disk
    per chromosome — for the 80/20 split on the 97k-sample AoU SV cohort
    this writes ~11 GB to /tmp per chromosome × 22+ chromosomes, dominating
    end-to-end wall time (>1 hour just on row reindex) and creating page-
    cache pressure visible as a 25 GB host-RSS spike.

    This wrapper instead holds the original full-sample child + a row index
    array and applies the row selection at iteration time. Downstream uses
    `iter_column_batches[_i8]` exclusively — each batch fetches its
    columns from the child once (memory-mapped) and applies the row
    indexing in a single fancy-indexed numpy slice on the resulting
    contiguous block. No intermediate disk write, no page-cache pressure.

    For repeated passes the kernel page cache covers the hot columns
    naturally, so the lazy form is as fast or faster than the eager form
    on any storage tier where the temp mmap is not faster than the source
    cache. On AoU/GCP they live on the same disk, so this is always a win.
    """
    child: RawGenotypeMatrix
    row_indices: NDArray

    def __post_init__(self) -> None:
        indices = np.asarray(self.row_indices, dtype=np.intp)
        if indices.ndim != 1:
            raise ValueError("row_indices must be 1D.")
        child_row_count = int(self.child.shape[0])
        if indices.size and (indices.min() < 0 or indices.max() >= child_row_count):
            raise ValueError("row_indices contains an out-of-range row.")
        self.row_indices = indices

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.row_indices.shape[0]), int(self.child.shape[1])

    def iter_column_batches(
        self,
        variant_indices: Sequence[int] | NDArray | None = None,
        batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE,
    ) -> Iterator[RawGenotypeBatch]:
        for batch in self.child.iter_column_batches(variant_indices=variant_indices, batch_size=batch_size):
            yield RawGenotypeBatch(
                variant_indices=batch.variant_indices,
                values=np.ascontiguousarray(batch.values[self.row_indices, :]),
            )

    def iter_column_batches_i8(
        self,
        variant_indices: Sequence[int] | NDArray | None = None,
        batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE,
    ) -> Iterator[RawGenotypeBatch]:
        if not _supports_int8_batches(self.child):
            raise RuntimeError("int8 batch iteration requires the wrapped child to support iter_column_batches_i8.")
        child = self.child
        for batch in child.iter_column_batches_i8(variant_indices=variant_indices, batch_size=batch_size):
            yield RawGenotypeBatch(
                variant_indices=batch.variant_indices,
                values=np.ascontiguousarray(batch.values[self.row_indices, :]),
            )

    def materialize(
        self,
        variant_indices: Sequence[int] | NDArray | None = None,
    ) -> NDArray:
        full = self.child.materialize(variant_indices)
        return np.ascontiguousarray(full[self.row_indices, :])


@dataclass(slots=True)
class PlinkRawGenotypeMatrix(RawGenotypeMatrix):
    bed_path: Path
    sample_indices: NDArray
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

    def _read_batch(self, reader: Any, batch_indices: NDArray) -> NDArray:
        """Read one batch as int8, convert to float32 with NaN for missing."""
        raw_i8 = self._read_batch_i8(reader, batch_indices)
        result = np.asarray(raw_i8, dtype=np.float32)
        result[raw_i8 == PLINK_MISSING_INT8] = np.nan
        return result

    def _read_batch_i8(self, reader: Any, batch_indices: NDArray) -> NDArray:
        """Read one batch as raw int8 (0/1/2/PLINK_MISSING_INT8). No float conversion."""
        import time as _time
        sample_index = _contiguous_index_or_slice(self.sample_indices)
        col_index = _contiguous_index_or_slice(batch_indices)
        t0 = _time.monotonic()
        result = np.asarray(
            reader.read(
                index=(sample_index, col_index),
                dtype="int8",
                order="F",
                num_threads=PLINK_BED_READER_NUM_THREADS,
            ),
            dtype=np.int8,
        )
        elapsed = _time.monotonic() - t0
        # Per-call timing: lets variant-stats logs distinguish bed-decode
        # time from JAX-compute time. Log only when noticeably non-trivial
        # so per-variant gather calls (variant_count == 1) don't spam.
        if elapsed >= 0.5 or int(batch_indices.shape[0]) >= 256:
            log(
                f"      bed read: variants={int(batch_indices.shape[0])} "
                f"samples={int(self.sample_indices.shape[0])}/{int(self.total_sample_count)}  "
                f"elapsed={elapsed:.2f}s"
            )
        return result

    def iter_column_batches_i8(
        self,
        variant_indices: Sequence[int] | NDArray | None = None,
        batch_size: int | None = None,
    ) -> Iterator[RawGenotypeBatch]:
        """Iterate as int8 batches (4x less memory, no float conversion).

        Values are 0/1/2/PLINK_MISSING_INT8 (missing). Callers must handle the shared sentinel.
        """
        resolved_indices = _resolve_variant_indices(self.variant_count, variant_indices)
        # int8 reads are 4x smaller than float32, but JAX kernels still expand to
        # float32 intermediates (~10 bytes/element peak), so keep decoded batches
        # near 1 GB rather than using the generic float32 reader cap.
        requested = max(int(self.batch_size if batch_size is None else batch_size), 1)
        bytes_per_variant = self.shape[0]  # 1 byte per sample for int8
        max_variants = max(
            PLINK_INT8_TARGET_BATCH_BYTES // max(bytes_per_variant, 1),
            MIN_BED_READER_BATCH_SIZE,
        )
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

        # Keep one read in flight so CPU decode can overlap with JAX stats.
        # Deeper prefetch would launch multiple large PLINK decodes at once,
        # which competes with the reader's own worker threads and doubles
        # decoded-batch memory pressure.
        batch_index_list = [
            resolved_indices[s : s + safe_batch_size]
            for s in range(0, total, safe_batch_size)
        ]
        prefetch_depth = min(PLINK_INT8_MAX_PREFETCH_DEPTH, len(batch_index_list))
        log(f"    int8 prefetch depth={prefetch_depth} (~{batch_mb * prefetch_depth:.0f} MB queued)  mem={mem()}")
        with ThreadPoolExecutor(max_workers=prefetch_depth) as executor:
            in_flight: deque[Future[NDArray]] = deque(
                executor.submit(self._read_batch_i8, reader, batch_index_list[i])
                for i in range(prefetch_depth)
            )
            next_to_submit = prefetch_depth
            for i in range(len(batch_index_list)):
                values = in_flight.popleft().result()
                if next_to_submit < len(batch_index_list):
                    in_flight.append(
                        executor.submit(
                            self._read_batch_i8, reader, batch_index_list[next_to_submit]
                        )
                    )
                    next_to_submit += 1
                yield RawGenotypeBatch(variant_indices=batch_index_list[i], values=values)

    def iter_column_batches(
        self,
        variant_indices: Sequence[int] | NDArray | None = None,
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
        batch_index_list: list[NDArray] = []
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
        variant_indices: Sequence[int] | NDArray | None = None,
    ) -> NDArray:
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
    _variant_offsets: NDArray = field(init=False, repr=False)

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
        variant_indices: Sequence[int] | NDArray | None = None,
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
        variant_indices: Sequence[int] | NDArray | None = None,
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
        variant_indices: Sequence[int] | NDArray | None = None,
    ) -> NDArray:
        resolved_indices = _resolve_variant_indices(self.shape[1], variant_indices)
        child_ids = np.searchsorted(self._variant_offsets[1:], resolved_indices, side="right")
        matrix = np.empty((self.shape[0], resolved_indices.shape[0]), dtype=np.float32)
        for child_index in np.unique(child_ids):
            child_positions = np.nonzero(child_ids == child_index)[0]
            child_variant_indices = resolved_indices[child_positions] - int(self._variant_offsets[child_index])
            matrix[:, child_positions] = self.children[int(child_index)].materialize(child_variant_indices)
        return matrix


_cupy_module = None
_cupy_checked = False


def _try_import_cupy() -> Any | None:
    """Import CuPy, caching the result. Returns None only during tests."""
    global _cupy_module, _cupy_checked
    if _cupy_checked:
        return _cupy_module
    _cupy_checked = True
    try:
        import cupy
        if cupy.cuda.runtime.getDeviceCount() > 0:
            _cupy_module = cupy
            return cupy
    except (ImportError, OSError, RuntimeError):
        pass
    _cupy_module = None
    return None


# Cache of (stream_a, stream_b) keyed by the id of the cupy module so that
# tests with mocked cupy instances each get their own pair, while real runs
# create the two non-default streams exactly once per process.
_cuda_upload_streams_cache: dict[int, tuple[Any, Any]] = {}


def _cuda_upload_stream_pair(cupy: Any) -> tuple[Any, Any]:
    """Return two non-default CUDA streams used to overlap H2D copies with CPU work.

    Created once per (cupy module) and reused across every materialization.
    """
    cache_key = id(cupy)
    cached = _cuda_upload_streams_cache.get(cache_key)
    if cached is not None:
        return cached
    stream_a = cupy.cuda.Stream(non_blocking=True)
    stream_b = cupy.cuda.Stream(non_blocking=True)
    pair = (stream_a, stream_b)
    _cuda_upload_streams_cache[cache_key] = pair
    return pair


def _pinned_int8_host_buffer(
    cupy: Any,
    sample_count: int,
    max_tile_variants: int,
) -> tuple[NDArray, NDArray, Any]:
    """Allocate a pair of reusable page-locked int8 host buffers (one per stream).

    Returns ``(buffer_a, buffer_b, pinned_memory_owner)``; each buffer is shaped
    ``(sample_count, max_tile_variants)``. The two buffers share a single
    ``cupy.cuda.alloc_pinned_memory`` allocation so they live as long as the
    returned owner reference. Raises ``MemoryError``/``RuntimeError`` loudly if
    the CUDA driver cannot pin host memory — there is no silent fallback.
    """
    if int(sample_count) <= 0 or int(max_tile_variants) <= 0:
        raise ValueError(
            "pinned int8 host buffer requires positive sample_count and max_tile_variants; "
            f"got sample_count={sample_count}, max_tile_variants={max_tile_variants}."
        )
    slot_shape = (int(sample_count), int(max_tile_variants))
    slot_element_count = slot_shape[0] * slot_shape[1]
    total_nbytes = slot_element_count * 2 * np.dtype(np.int8).itemsize
    pinned_memory = cupy.cuda.alloc_pinned_memory(total_nbytes)
    flat_view = np.frombuffer(pinned_memory, dtype=np.int8, count=slot_element_count * 2)
    buffer_a = flat_view[:slot_element_count].reshape(slot_shape)
    buffer_b = flat_view[slot_element_count:].reshape(slot_shape)
    return buffer_a, buffer_b, pinned_memory


def _upload_standardized_int8_tiles_overlapped(
    *,
    cupy: Any,
    raw_int8: "Int8BatchCapable",
    variant_indices: NDArray,
    means: NDArray,
    scales: NDArray,
    gpu_destination: Any,
    sample_count: int,
    upload_batch_size: int,
    standardized_dtype: Any,
) -> None:
    """Upload + standardize int8 tiles into a float GPU strip with overlapped H2D.

    Mirrors ``_upload_int8_tiles_overlapped`` but additionally casts each tile to
    ``standardized_dtype`` on the GPU and applies ``(x - mean) / scale`` with the
    PLINK missing-int8 sentinel zeroed out, matching ``_standardize_batch_cupy``
    bit-for-bit while keeping the work on the upload stream so it overlaps the
    next tile's BED-reader pass.
    """
    stream_pair = _cuda_upload_stream_pair(cupy)
    pinned_buffer_a, pinned_buffer_b, _pinned_owner = _pinned_int8_host_buffer(
        cupy, sample_count=sample_count, max_tile_variants=int(upload_batch_size)
    )
    pinned_slots = (pinned_buffer_a, pinned_buffer_b)
    in_flight_events: list[Any] = [None, None]
    selected_means_gpu = cupy.asarray(means[variant_indices], dtype=standardized_dtype)
    selected_scales_gpu = cupy.asarray(scales[variant_indices], dtype=standardized_dtype)
    local_start = 0
    for tile_index, raw_batch in enumerate(
        raw_int8.iter_column_batches_i8(variant_indices, batch_size=upload_batch_size)
    ):
        slot_index = tile_index % 2
        previous_event = in_flight_events[slot_index]
        if previous_event is not None:
            previous_event.synchronize()
        batch_width = raw_batch.values.shape[1]
        pinned_slot = pinned_slots[slot_index]
        pinned_slot[:, :batch_width] = raw_batch.values
        local_stop = local_start + batch_width
        stream = stream_pair[slot_index]
        with stream:
            staged_int8 = cupy.asarray(pinned_slot[:, :batch_width])
            standardized_tile = _standardize_int8_cupy(
                staged_int8,
                selected_means_gpu[local_start:local_stop],
                selected_scales_gpu[local_start:local_stop],
                cupy,
                dtype=standardized_dtype,
            )
            gpu_destination[:, local_start:local_stop] = standardized_tile
        in_flight_events[slot_index] = stream.record()
        local_start = local_stop
    # Synchronization invariant: after this function returns, the caller assigns
    # ``gpu_destination`` to ``self._cupy_cache`` and downstream consumers will
    # immediately read from it. We must therefore wait on every in-flight async
    # H2D event — in particular the LAST tile's event, which would otherwise
    # still be in flight when the loop body exits.
    for in_flight_event in in_flight_events:
        if in_flight_event is not None:
            in_flight_event.synchronize()


def _try_upload_int8_parallel_memmap(
    *,
    cupy: Any,
    raw: object,
    variant_indices: NDArray,
    gpu_destination: Any,
    sample_count: int,
    n_workers: int = 8,
) -> bool:
    """Fast parallel-memmap upload path for an F-order int8 numpy memmap leaf.

    Bypasses the per-batch iterator chain: allocates one big pinned host buffer,
    splits variant_indices into ``n_workers`` column stripes, then for each stripe
    a worker thread (a) reads the memmap slice into its stripe of the pinned
    buffer — letting the kernel issue parallel disk I/O across worker threads —
    and (b) issues an async H2D copy on its own non-blocking CUDA stream into the
    matching GPU column range. Returns True on success; False if the source isn't
    eligible (no memmap leaf, wrong dtype/order, or non-contiguous indices).

    Bit-identical to ``_upload_int8_tiles_overlapped`` byte-for-byte: same raw
    int8 bytes get DMA'd to the same GPU offsets.
    """
    if not isinstance(raw, Int8RawGenotypeMatrix):
        return False
    matrix = raw.matrix
    if not isinstance(matrix, np.memmap):
        return False
    if matrix.dtype != np.int8:
        return False
    if matrix.ndim != 2:
        return False
    if not matrix.flags.f_contiguous:
        return False
    if int(matrix.shape[0]) != int(sample_count):
        return False
    n_variants = int(variant_indices.shape[0])
    if n_variants == 0:
        return True
    vi = np.asarray(variant_indices, dtype=np.int64)
    if vi.size > 1:
        if not np.all(np.diff(vi) == 1):
            return False
    src_col_start = int(vi[0])
    src_col_end = src_col_start + n_variants
    if src_col_start < 0 or src_col_end > int(matrix.shape[1]):
        return False

    total_bytes = int(sample_count) * n_variants
    try:
        pinned_owner = cupy.cuda.alloc_pinned_memory(total_bytes)
    except (MemoryError, RuntimeError) as exc:
        log(f"    parallel-pread upload: pinned alloc failed ({exc}); falling back")
        return False
    flat = np.frombuffer(pinned_owner, dtype=np.int8, count=total_bytes)
    pinned_buf = flat.reshape((n_variants, sample_count)).T
    if not pinned_buf.flags.f_contiguous:
        return False

    try:
        if matrix.filename:
            with open(matrix.filename, "rb") as fadvise_handle:
                file_offset = int(getattr(matrix, "offset", 0)) + src_col_start * int(sample_count)
                # posix_fadvise is Linux-only; fall through cleanly on macOS/Windows.
                posix_fadvise = getattr(os, "posix_fadvise", None)
                willneed = getattr(os, "POSIX_FADV_WILLNEED", None)
                if posix_fadvise is not None and willneed is not None:
                    try:
                        posix_fadvise(fadvise_handle.fileno(), file_offset, total_bytes, willneed)
                    except OSError:
                        pass
    except (AttributeError, OSError):
        pass

    effective_workers = max(1, min(int(n_workers), n_variants))
    stripe_size = (n_variants + effective_workers - 1) // effective_workers
    stripes: list[tuple[int, int]] = []
    for worker_idx in range(effective_workers):
        s = worker_idx * stripe_size
        e = min(s + stripe_size, n_variants)
        if s < e:
            stripes.append((s, e))
    effective_workers = len(stripes)

    streams = [cupy.cuda.Stream(non_blocking=True) for _ in range(effective_workers)]
    events: list[Any] = [None] * effective_workers
    errors: list[BaseException] = []

    runtime = getattr(getattr(cupy, "cuda", None), "runtime", None)
    memcpy_async = getattr(runtime, "memcpyAsync", None) if runtime is not None else None
    memcpy_h2d_kind = getattr(runtime, "memcpyHostToDevice", None) if runtime is not None else None
    direct_h2d_supported = (
        memcpy_async is not None
        and memcpy_h2d_kind is not None
        and hasattr(gpu_destination, "data")
        and hasattr(getattr(gpu_destination, "data", None), "ptr")
    )

    def worker(worker_idx: int, col_start: int, col_end: int) -> None:
        try:
            src_view = matrix[:, src_col_start + col_start : src_col_start + col_end]
            # Forces parallel page faults + memcpy from page cache → pinned buffer.
            # numpy releases the GIL for this memcpy, so workers fan out across cores.
            pinned_buf[:, col_start:col_end] = src_view
            stripe_bytes = int(sample_count) * (col_end - col_start)
            stream = streams[worker_idx]
            if direct_h2d_supported:
                # Direct async H2D into the pre-allocated GPU destination slice.
                # The naïve ``cupy.asarray(pinned_slice)`` route would allocate
                # ``stripe_bytes`` of GPU staging per worker — at AoU sizes that
                # is ~1.5 GB × 8 workers ≈ 12 GB on top of the 11.8 GB
                # destination, OOM'ing a 15 GB T4. ``memcpyAsync`` writes
                # straight into the F-order destination view: zero extra GPU
                # allocation, identical bytes on-device.
                assert memcpy_async is not None and memcpy_h2d_kind is not None
                gpu_view = gpu_destination[:, col_start:col_end]
                host_slice = pinned_buf[:, col_start:col_end]
                memcpy_async(
                    int(gpu_view.data.ptr),
                    int(host_slice.ctypes.data),
                    int(stripe_bytes),
                    memcpy_h2d_kind,
                    int(stream.ptr),
                )
            else:
                with stream:
                    gpu_int8_dtype = getattr(cupy, "int8", np.int8)
                    staged_tile = cupy.asarray(pinned_buf[:, col_start:col_end], dtype=gpu_int8_dtype)
                    gpu_destination[:, col_start:col_end] = staged_tile
            events[worker_idx] = stream.record()
        except BaseException as exc:  # noqa: BLE001 - reraised on main thread
            errors.append(exc)

    log(
        "    parallel-pread upload: "
        + f"{n_variants} variants x {sample_count} samples ({total_bytes / 1e9:.1f} GB) "
        + f"across {effective_workers} workers/streams  mem={mem()}"
    )
    with ThreadPoolExecutor(max_workers=effective_workers, thread_name_prefix="i8-pread") as executor:
        futures = [executor.submit(worker, i, s, e) for i, (s, e) in enumerate(stripes)]
        for future in futures:
            future.result()

    if errors:
        raise errors[0]

    for event in events:
        if event is not None:
            event.synchronize()
    return True


def _upload_int8_tiles_overlapped(
    *,
    cupy: Any,
    raw_int8: "Int8BatchCapable",
    variant_indices: NDArray,
    gpu_destination: Any,
    sample_count: int,
    upload_batch_size: int,
    gpu_int8_dtype: Any,
) -> None:
    """Upload int8 column tiles into ``gpu_destination`` with overlapped H2D copies.

    Issues each tile's async H2D copy on one of two CUDA streams, double-buffered
    through a pair of pinned int8 host buffers so the next tile's BED-reader
    CPU work runs concurrently with the previous tile's H2D transfer. The math
    is bit identical to a serial pageable ``cupy.asarray`` upload — this is a
    pure scheduling change.
    """
    stream_pair = _cuda_upload_stream_pair(cupy)
    pinned_buffer_a, pinned_buffer_b, _pinned_owner = _pinned_int8_host_buffer(
        cupy, sample_count=sample_count, max_tile_variants=int(upload_batch_size)
    )
    pinned_slots = (pinned_buffer_a, pinned_buffer_b)
    in_flight_events: list[Any] = [None, None]
    local_start = 0
    for tile_index, raw_batch in enumerate(
        raw_int8.iter_column_batches_i8(variant_indices, batch_size=upload_batch_size)
    ):
        slot_index = tile_index % 2
        previous_event = in_flight_events[slot_index]
        if previous_event is not None:
            previous_event.synchronize()
        batch_width = raw_batch.values.shape[1]
        pinned_slot = pinned_slots[slot_index]
        pinned_slot[:, :batch_width] = raw_batch.values
        local_stop = local_start + batch_width
        stream = stream_pair[slot_index]
        with stream:
            staged_tile = cupy.asarray(pinned_slot[:, :batch_width], dtype=gpu_int8_dtype)
            gpu_destination[:, local_start:local_stop] = staged_tile
        in_flight_events[slot_index] = stream.record()
        local_start = local_stop
    # Synchronization invariant: after this function returns, the caller assigns
    # ``gpu_destination`` to ``self._cupy_cache`` and downstream consumers will
    # immediately read from it. We must therefore wait on every in-flight async
    # H2D event — in particular the LAST tile's event, which would otherwise
    # still be in flight when the loop body exits.
    for in_flight_event in in_flight_events:
        if in_flight_event is not None:
            in_flight_event.synchronize()


_gpu_verified = False


def require_gpu() -> Any:
    """Validate GPU+CuPy at pipeline entry. Crash loudly if GPU exists but CuPy fails.

    Returns the CuPy module if available, None if no GPU hardware at all.
    Skips re-verification after first successful call.
    """
    global _gpu_verified
    if _gpu_verified:
        return _cupy_module
    from sv_pgs.progress import log
    import shutil
    import subprocess
    nvidia_smi = shutil.which("nvidia-smi")
    has_gpu_hardware = False
    if nvidia_smi is not None:
        try:
            result = subprocess.run(
                [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5.0, check=False,
            )
            has_gpu_hardware = result.returncode == 0 and bool(result.stdout.strip())
        except (OSError, subprocess.SubprocessError):
            pass
    if not has_gpu_hardware:
        log("  no NVIDIA GPU detected — running CPU-only (this will be slow)")
        return None
    # GPU exists — CuPy MUST work
    cupy = _try_import_cupy()
    if cupy is None:
        raise RuntimeError(
            "NVIDIA GPU detected but CuPy is not working. "
            "Install with: uv pip install cupy-cuda12x  "
            "(or cupy-cuda11x for CUDA 11). "
            "Verify: python -c 'import cupy; print(cupy.cuda.runtime.memGetInfo())'"
        )
    free_bytes, total_bytes = cupy.cuda.runtime.memGetInfo()
    log(f"  GPU verified: {total_bytes / 1e9:.1f} GB total, {free_bytes / 1e9:.1f} GB free")
    # If another process is pinning most of the device, the training pipeline
    # will silently fall back to slow streaming paths and may still OOM on the
    # remaining residual. Surface it once, at entry, so the user can kill stale
    # kernels before burning hours on a degraded run.
    if total_bytes > 0 and free_bytes < total_bytes * 0.5:
        held_bytes = total_bytes - free_bytes
        log(
            f"  WARNING: only {free_bytes / 1e9:.1f} GB of {total_bytes / 1e9:.1f} GB free "
            f"({held_bytes / 1e9:.1f} GB held by other processes). "
            f"Run `nvidia-smi` and kill stale GPU processes to avoid OOM / slow streaming fallbacks."
        )
    _gpu_verified = True
    return cupy


def _as_gpu_compute_jax(array: Any) -> JaxArray:
    return jnp.asarray(array, dtype=gpu_compute_jax_dtype())


def _cupy_to_jax(array: Any) -> JaxArray:
    """Convert CuPy result to JAX, preferring zero-copy DLPack interop."""
    if hasattr(array, "__dlpack__"):
        return jax_dlpack.from_dlpack(array).astype(gpu_compute_jax_dtype())
    return jnp.asarray(array.get(), dtype=gpu_compute_jax_dtype())


def _to_cupy_float32(array: Any) -> Any:
    """Convert JAX/numpy array to CuPy float32 for CuPy matmul."""
    cupy = _try_import_cupy()
    if cupy is None:
        raise RuntimeError("CuPy is not available.")
    if type(array).__module__.startswith("cupy"):
        return array.astype(cupy.float32, copy=False)
    if isinstance(array, jax.Array) and hasattr(cupy, "from_dlpack"):
        return cupy.from_dlpack(array).astype(cupy.float32, copy=False)
    return cupy.asarray(np.asarray(array, dtype=np.float32))


def _to_cupy_float64(array: Any) -> Any:
    """Convert JAX/numpy array to CuPy float64 for numerically sensitive GPU solves."""
    cupy = _try_import_cupy()
    if cupy is None:
        raise RuntimeError("CuPy is not available.")
    if type(array).__module__.startswith("cupy"):
        return array.astype(cupy.float64, copy=False)
    if isinstance(array, jax.Array) and hasattr(cupy, "from_dlpack"):
        return cupy.from_dlpack(array).astype(cupy.float64, copy=False)
    return cupy.asarray(np.asarray(array, dtype=np.float64))


def _cupy_compute_dtype(cupy: Any) -> Any:
    return cupy.float32 if gpu_compute_numpy_dtype() == np.dtype(np.float32) else cupy.float64


def _cupy_dtype_to_numpy_dtype(dtype: Any) -> np.dtype[Any]:
    try:
        return np.dtype(dtype)
    except TypeError:
        return np.dtype(np.float32)


def _standardize_int8_cupy(raw_values: Any, means: Any, scales: Any, cupy: Any, *, dtype: Any) -> Any:
    """Standardize int8 genotypes on GPU without materializing a mask buffer."""
    resolved_dtype = cupy.float32 if dtype is None else dtype
    elementwise_kernel = getattr(cupy, "ElementwiseKernel", None)
    if elementwise_kernel is not None:
        kernel_cache = getattr(_standardize_int8_cupy, "_kernel_cache", None)
        if kernel_cache is None:
            kernel_cache = {}
            setattr(_standardize_int8_cupy, "_kernel_cache", kernel_cache)
        cache_key = id(elementwise_kernel)
        kernel = kernel_cache.get(cache_key)
        if kernel is None:
            kernel = elementwise_kernel(
                "int8 raw, T means, T scales, int8 missing",
                "T out",
                "out = (raw == missing) ? (T)0 : (((T)raw - means) / scales)",
                "sv_pgs_standardize_int8_missing_zero",
            )
            kernel_cache[cache_key] = kernel
        return kernel(raw_values, means[None, :], scales[None, :], np.int8(PLINK_MISSING_INT8))

    standardized = raw_values.astype(resolved_dtype, copy=False)
    valid_mask = standardized != float(PLINK_MISSING_INT8)
    standardized -= means[None, :]
    standardized /= scales[None, :]
    multiply = getattr(cupy, "multiply", np.multiply)
    multiply(standardized, valid_mask, out=standardized)
    return standardized


def _to_cupy_compute(array: Any) -> Any:
    cupy = _try_import_cupy()
    if cupy is None:
        raise RuntimeError("CuPy is not available.")
    compute_dtype = _cupy_compute_dtype(cupy)
    if compute_dtype == cupy.float32:
        return _to_cupy_float32(array)
    return _to_cupy_float64(array)


def _cupy_to_numpy(array: Any, *, dtype: Any) -> NDArray:
    host_array = array.get() if hasattr(array, "get") else array
    return np.asarray(host_array, dtype=dtype)


def _standardize_batch_cupy(
    batch_values: NDArray,
    means: Any,
    scales: Any,
    cupy: Any,
    *,
    missing_sentinel: int | None = None,
    dtype: Any = None,
) -> Any:
    """Standardize a raw batch directly on GPU.

    Memory-sensitive: boolean-mask scatter (``standardized[missing] = 0``) is
    implemented in CuPy via a prefix-sum scan that allocates a batch-sized
    scratch buffer, and ``cupy.where`` allocates a second batch-sized float
    buffer; either OOMs on wide AoU batches. The int8 path uses a fused
    elementwise kernel; the float path uses in-place NaN scrubbing.
    """
    resolved_dtype = cupy.float32 if dtype is None else dtype
    if missing_sentinel is not None:
        raw_gpu = cupy.asarray(batch_values, dtype=getattr(cupy, "int8", np.int8))
        return _standardize_int8_cupy(raw_gpu, means, scales, cupy, dtype=resolved_dtype)

    standardized = cupy.asarray(batch_values, dtype=resolved_dtype)
    # Raw float source: missing entries arrive as NaN and propagate as NaN
    # through center/scale. ``NaN * 0 == NaN`` (IEEE 754), so a multiply-by-
    # mask would not clear them — use in-place nan_to_num instead, which
    # also scrubs ±inf from zero-scale columns at no extra cost.
    standardized -= means[None, :]
    standardized /= scales[None, :]
    nan_to_num = getattr(cupy, "nan_to_num", None)
    if nan_to_num is not None:
        nan_to_num(standardized, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        host = np.asarray(standardized)
        np.nan_to_num(host, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        standardized = cupy.asarray(host, dtype=resolved_dtype)
    return standardized


def _iter_standardized_gpu_batches(
    raw: RawGenotypeMatrix | Int8BatchCapable,
    variant_indices: NDArray,
    means: Any,
    scales: Any,
    *,
    batch_size: int,
    cupy: Any,
    dtype: Any = None,
) -> Iterator[tuple[slice, Any]]:
    resolved_dtype = cupy.float32 if dtype is None else dtype
    selected_means = cupy.asarray(means[variant_indices], dtype=resolved_dtype)
    selected_scales = cupy.asarray(scales[variant_indices], dtype=resolved_dtype)
    for batch_slice, values in _iter_prefetched_raw_batches(raw, variant_indices, batch_size=batch_size):
        for relative_slice, standardized_batch in _iter_standardized_gpu_subbatches(
            values,
            selected_means[batch_slice],
            selected_scales[batch_slice],
            cupy,
            missing_sentinel=int(PLINK_MISSING_INT8) if _supports_int8_batches(raw) else None,
            dtype=resolved_dtype,
        ):
            yield (
                slice(
                    int(batch_slice.start or 0) + int(relative_slice.start or 0),
                    int(batch_slice.start or 0) + int(relative_slice.stop or 0),
                ),
                standardized_batch,
            )


def _iter_standardized_gpu_subbatches(
    values: NDArray,
    means: Any,
    scales: Any,
    cupy: Any,
    *,
    missing_sentinel: int | None,
    dtype: Any,
) -> Iterator[tuple[slice, Any]]:
    try:
        yield slice(0, int(values.shape[1])), _standardize_batch_cupy(
            values,
            means,
            scales,
            cupy,
            missing_sentinel=missing_sentinel,
            dtype=dtype,
        )
        return
    except Exception as exc:
        if not _is_cupy_out_of_memory(exc) or int(values.shape[1]) <= 1:
            raise
        _release_cupy_cached_memory(cupy)

    split_at = max(int(values.shape[1]) // 2, 1)
    log(
        "        CuPy OOM while standardizing "
        + f"{int(values.shape[1]):,} variants; retrying as {split_at:,}"
        + f" + {int(values.shape[1]) - split_at:,} variants  mem={mem()}"
    )
    for child_slice, standardized_batch in _iter_standardized_gpu_subbatches(
        values[:, :split_at],
        means[:split_at],
        scales[:split_at],
        cupy,
        missing_sentinel=missing_sentinel,
        dtype=dtype,
    ):
        yield child_slice, standardized_batch
    for child_slice, standardized_batch in _iter_standardized_gpu_subbatches(
        values[:, split_at:],
        means[split_at:],
        scales[split_at:],
        cupy,
        missing_sentinel=missing_sentinel,
        dtype=dtype,
    ):
        yield (
            slice(
                split_at + int(child_slice.start or 0),
                split_at + int(child_slice.stop or 0),
            ),
            standardized_batch,
        )


def _iter_prefetched_raw_batches(
    raw: RawGenotypeMatrix | Int8BatchCapable,
    variant_indices: NDArray,
    *,
    batch_size: int,
) -> Iterator[tuple[slice, NDArray]]:
    variant_indices_arr = np.asarray(variant_indices)
    n = int(variant_indices_arr.shape[0])
    if n == 0:
        return
    safe_batch_size = max(int(batch_size), 1)
    sample_count = int(raw.shape[0])

    if _supports_int8_batches(raw):
        i8_raw = raw
        def _read_chunk(chunk_indices: NDArray) -> NDArray:
            for batch in i8_raw.iter_column_batches_i8(chunk_indices, batch_size=chunk_indices.shape[0]):
                return np.asarray(batch.values)
            return np.empty((sample_count, 0), dtype=np.int8)
    else:
        float_raw = raw
        def _read_chunk(chunk_indices: NDArray) -> NDArray:
            for batch in float_raw.iter_column_batches(chunk_indices, batch_size=chunk_indices.shape[0]):
                return np.asarray(batch.values)
            return np.empty((sample_count, 0), dtype=np.float32)

    chunk_starts = list(range(0, n, safe_batch_size))
    chunks = [variant_indices_arr[s : s + safe_batch_size] for s in chunk_starts]

    # Parallel disk reads help on cloud SSDs, but large AoU row subsets make
    # each chunk hundreds of MB. Bound both workers and queued chunks by bytes
    # so prefetch does not evict the pages/GPU buffers the current matmul needs.
    bytes_per_raw_value = np.dtype(np.int8 if _supports_int8_batches(raw) else np.float32).itemsize
    chunk_bytes = max(sample_count * safe_batch_size * bytes_per_raw_value, 1)
    memory_capped_in_flight = max(1, GPU_STANDARDIZED_PREFETCH_TARGET_BYTES // chunk_bytes)
    num_io_workers = min(4, int(memory_capped_in_flight), max(1, len(chunks)))
    in_flight_limit = min(num_io_workers * 2, int(memory_capped_in_flight), len(chunks))
    if num_io_workers <= 1 or len(chunks) <= 1:
        local_start = 0
        for chunk in chunks:
            values = _read_chunk(chunk)
            batch_width = values.shape[1]
            local_stop = local_start + batch_width
            yield slice(local_start, local_stop), values
            local_start = local_stop
        return

    executor = ThreadPoolExecutor(
        max_workers=num_io_workers,
        thread_name_prefix="standardized-gpu-prefetch",
    )
    futures: deque[Future[Any]] = deque()
    next_to_submit = 0

    def _submit_more() -> None:
        nonlocal next_to_submit
        while next_to_submit < len(chunks) and len(futures) < in_flight_limit:
            futures.append(executor.submit(_read_chunk, chunks[next_to_submit]))
            next_to_submit += 1

    try:
        _submit_more()
        local_start = 0
        while futures:
            future = futures.popleft()
            values = future.result()
            _submit_more()
            batch_width = values.shape[1]
            local_stop = local_start + batch_width
            yield slice(local_start, local_stop), values
            local_start = local_stop
    finally:
        for pending in futures:
            pending.cancel()
        executor.shutdown(wait=False)


def _gpu_streaming_batch_size(
    raw: RawGenotypeMatrix | Int8BatchCapable,
    *,
    sample_count: int,
    requested_batch_size: int,
    cupy: Any = None,
    dtype: Any = None,
) -> int:
    if _supports_int8_batches(raw):
        requested_batch_size = max(int(requested_batch_size), auto_batch_size_i8(sample_count))
    resolved_cupy = _try_import_cupy() if cupy is None else cupy
    if resolved_cupy is not None:
        target_batch_bytes = _gpu_dynamic_standardized_target_batch_bytes(
            resolved_cupy,
            static_target_batch_bytes=GPU_STANDARDIZED_STREAMING_TARGET_BATCH_BYTES,
        )
    else:
        target_batch_bytes = GPU_STANDARDIZED_STREAMING_TARGET_BATCH_BYTES
    return _effective_gpu_standardized_streaming_batch_size(
        sample_count=sample_count,
        requested_batch_size=requested_batch_size,
        target_batch_bytes=target_batch_bytes,
        dtype=dtype,
    )


def _gpu_int8_transpose_matmul(
    *,
    raw_int8: Int8BatchCapable,
    variant_indices: NDArray,
    means: NDArray,
    scales: NDArray,
    matrix_gpu: Any,
    batch_size: int,
    cupy: Any,
    dtype: Any,
    progress_label: str | None,
) -> Any:
    selected_variant_indices = np.asarray(variant_indices, dtype=np.int32)
    selected_means_gpu = cupy.asarray(means[selected_variant_indices], dtype=dtype)
    selected_scales_gpu = cupy.asarray(scales[selected_variant_indices], dtype=dtype)
    result_gpu = cupy.empty((selected_variant_indices.shape[0], matrix_gpu.shape[1]), dtype=dtype)
    total_variants = int(selected_variant_indices.shape[0])
    completed_variants = 0
    last_logged_variants = 0
    log_interval = max(total_variants // 50, 1)
    import time as _time
    t_start = _time.monotonic()
    if progress_label is not None:
        log(f"        {progress_label}: start streaming {total_variants:,} variants (batch_size={batch_size})  mem={mem()}")
    for batch_slice, host_values in _iter_prefetched_raw_batches(
        raw_int8,
        selected_variant_indices,
        batch_size=batch_size,
    ):
        int8_gpu = cupy.asarray(host_values, dtype=getattr(cupy, "int8", np.int8))
        means_gpu = selected_means_gpu[batch_slice]
        scales_gpu = selected_scales_gpu[batch_slice]
        standardized_gpu = _standardize_int8_cupy(
            int8_gpu,
            means_gpu,
            scales_gpu,
            cupy,
            dtype=dtype,
        )
        result_gpu[batch_slice, :] = standardized_gpu.T @ matrix_gpu
        if progress_label is not None:
            completed_variants = batch_slice.stop
            if completed_variants - last_logged_variants >= log_interval:
                last_logged_variants = completed_variants
                elapsed_seconds = _time.monotonic() - t_start
                rate = completed_variants / max(elapsed_seconds, 1e-6)
                eta = (total_variants - completed_variants) / max(rate, 1e-6)
                log(
                    f"        {progress_label}: {completed_variants:,}/{total_variants:,} "
                    f"({100*completed_variants/total_variants:.1f}%) "
                    f"elapsed={elapsed_seconds:.0f}s rate={rate:,.0f}v/s eta={eta:.0f}s  mem={mem()}"
                )
    if progress_label is not None:
        elapsed_seconds = _time.monotonic() - t_start
        log(f"        {progress_label}: done {total_variants:,} variants in {elapsed_seconds:.1f}s  mem={mem()}")
    return result_gpu


def _gpu_free_bytes(cupy: Any) -> int:
    """Return free GPU device memory in bytes, or 0 if unavailable."""
    try:
        free, _ = cupy.cuda.runtime.memGetInfo()
        return int(free)
    except (AttributeError, OSError, RuntimeError):
        return 0


def _gpu_total_bytes(cupy: Any) -> int:
    """Return total GPU device memory in bytes, or 0 if unavailable.

    Returns 0 on mocked / partial CuPy stand-ins (used by unit tests that
    don't have a real CUDA runtime) so callers fall back to their static
    defaults instead of crashing.
    """
    try:
        _, total = cupy.cuda.runtime.memGetInfo()
        return int(total)
    except (AttributeError, OSError, RuntimeError):
        return 0


def _gpu_dynamic_standardized_target_batch_bytes(cupy: Any, *, static_target_batch_bytes: int) -> int:
    free_bytes = _gpu_free_bytes(cupy)
    if free_bytes <= 0:
        return int(static_target_batch_bytes)
    usable_bytes = max(
        free_bytes - GPU_STANDARDIZED_DYNAMIC_RESERVE_BYTES,
        int(free_bytes * 0.10),
    )
    dynamic_target = max(int(usable_bytes * GPU_STANDARDIZED_DYNAMIC_FREE_FRACTION), 1)
    return max(1, min(int(static_target_batch_bytes), dynamic_target))


def _is_cupy_out_of_memory(exc: BaseException) -> bool:
    exc_name = exc.__class__.__name__.lower()
    exc_message = str(exc).lower()
    return (
        "outofmemory" in exc_name
        or "out of memory" in exc_message
        or "cuda_error_out_of_memory" in exc_message
    )


def _release_cupy_cached_memory(cupy: Any) -> None:
    """Best-effort release of CuPy pool blocks after a failed allocation."""
    try:
        pool = cupy.get_default_memory_pool()
        pool.free_all_blocks()
    except (AttributeError, OSError, RuntimeError):
        pass
    try:
        pinned_pool = cupy.get_default_pinned_memory_pool()
        pinned_pool.free_all_blocks()
    except (AttributeError, OSError, RuntimeError):
        pass


def _effective_gpu_standardized_streaming_batch_size(
    *,
    sample_count: int,
    requested_batch_size: int,
    target_batch_bytes: int,
    dtype: Any,
) -> int:
    if sample_count < 1:
        raise ValueError("sample_count must be positive.")
    if requested_batch_size < 1:
        raise ValueError("requested_batch_size must be positive.")
    if target_batch_bytes < 1:
        raise ValueError("target_batch_bytes must be positive.")
    resolved_dtype = _cupy_dtype_to_numpy_dtype(np.float32 if dtype is None else dtype)
    bytes_per_variant = sample_count * resolved_dtype.itemsize
    memory_capped_batch_size = max(target_batch_bytes // max(bytes_per_variant, 1), 1)
    return max(1, min(int(requested_batch_size), int(memory_capped_batch_size)))


_GPU_RESERVED_OVERHEAD_BYTES = 1_500_000_000  # 1.5 GB
_GPU_BUDGET_TOTAL_FRACTION_CEILING = 0.90


def _gpu_materialization_budget_bytes(cupy: Any) -> int:
    """GPU cache budget: total device memory minus a fixed overhead margin.

    Uses total memory (not free) to avoid dependence on JAX/XLA pre-allocation
    state — JAX is already constrained via XLA_PYTHON_CLIENT_MEM_FRACTION in
    _jax.py, so its peak footprint is bounded independently of this budget.

    The 1.5 GB reserve covers the CUDA context (~500 MB), cuSOLVER workspace
    for typical Cholesky/Gram operations (~256 MB peak), the JAX/XLA reserved
    fraction (~10% of device memory), and a safety margin. The 90% ceiling
    prevents the budget from ever consuming the entire device on very large
    GPUs where 1.5 GB reserve would still leave too thin a margin.

    The earlier "40% of total" heuristic was a conservative defensive value
    chosen before JAX preallocation was constrained. It is too tight on
    common 16 GB GPUs: a 14 GB standardized matrix is rejected, falling back
    to a per-block streaming SVI path that is ~25× slower and does not
    converge under the existing Robbins-Monro schedule.
    """
    total = _gpu_total_bytes(cupy)
    if total <= 0:
        return 0
    reserved = min(_GPU_RESERVED_OVERHEAD_BYTES, int(total * (1.0 - _GPU_BUDGET_TOTAL_FRACTION_CEILING)))
    budget = max(total - _GPU_RESERVED_OVERHEAD_BYTES, int(total * 0.5))
    budget = min(budget, int(total * _GPU_BUDGET_TOTAL_FRACTION_CEILING), total - reserved)
    # Cap by actual *free* memory so co-tenant processes holding the bulk of
    # the device don't lead us to size allocations as if the GPU were empty.
    # Without this, runtime policy reports e.g. 14.1 GB on a 16 GB T4 even
    # when only ~200 MB is free, then downstream allocations instantly OOM.
    free = _gpu_free_bytes(cupy)
    if free > 0:
        free_budget = max(free - _GPU_RESERVED_OVERHEAD_BYTES, int(free * 0.5))
        budget = min(budget, free_budget)
    return max(budget, 0)


@dataclass(slots=True)
class _CupyInt8StandardizedCache:
    raw_values: Any
    means: Any
    scales: Any
    cupy: Any = field(repr=False)
    # When set, this cache is a view over ``raw_values``: the i-th logical
    # column maps to ``raw_values[:, column_indices[i]]``. ``means`` / ``scales``
    # are always pre-subset to logical column order, so they are indexed
    # directly with logical positions.
    column_indices: Any | None = None

    @property
    def shape(self) -> tuple[int, int]:
        rows = int(self.raw_values.shape[0])
        if self.column_indices is None:
            return rows, int(self.raw_values.shape[1])
        return rows, int(self.column_indices.shape[0])

    @property
    def nbytes(self) -> int:
        # Report the *logical* footprint so callers (budget logs, materialization
        # accounting) see the view's effective size rather than the shared parent
        # buffer. Root caches (column_indices is None) report the true allocation.
        means_bytes = int(self.means.nbytes)
        scales_bytes = int(self.scales.nbytes)
        if self.column_indices is None:
            return int(self.raw_values.nbytes) + means_bytes + scales_bytes
        sample_count = int(self.raw_values.shape[0])
        view_columns = int(self.column_indices.shape[0])
        itemsize = int(self.raw_values.dtype.itemsize)
        return sample_count * view_columns * itemsize + means_bytes + scales_bytes + int(self.column_indices.nbytes)

    def subset(self, local_variant_indices: NDArray) -> _CupyInt8StandardizedCache:
        resolved_local_indices = np.asarray(local_variant_indices, dtype=np.int32)
        if _local_indices_select_all(resolved_local_indices, self.shape[1]):
            return self
        cp = self.cupy
        local_slice = _local_indices_as_slice(resolved_local_indices)
        if local_slice is not None:
            # Contiguous selection: subset means/scales by slice (cheap) and
            # either keep a zero-copy slice view of raw_values (root cache) or
            # slice the parent's column-index array (view-of-view). Both paths
            # avoid materializing a fresh genotype copy on the device.
            means = self.means[local_slice]
            scales = self.scales[local_slice]
            if self.column_indices is None:
                return _CupyInt8StandardizedCache(
                    raw_values=self.raw_values[:, local_slice],
                    means=means,
                    scales=scales,
                    cupy=cp,
                )
            return _CupyInt8StandardizedCache(
                raw_values=self.raw_values,
                means=means,
                scales=scales,
                cupy=cp,
                column_indices=self.column_indices[local_slice],
            )
        # Non-contiguous fancy selection: defer the raw_values gather. Keep
        # the parent buffer shared and store the resolved column ids on the
        # device, turning ``subset`` from O(samples x selected) device bytes
        # into O(selected) device bytes; critical when ``selected`` covers
        # most of the cache and would otherwise OOM by duplicating it.
        # ``standardized_columns`` gathers each working batch on demand.
        means = self.means[resolved_local_indices]
        scales = self.scales[resolved_local_indices]
        device_indices = cp.asarray(resolved_local_indices)
        composed_indices = (
            device_indices if self.column_indices is None else self.column_indices[device_indices]
        )
        return _CupyInt8StandardizedCache(
            raw_values=self.raw_values,
            means=means,
            scales=scales,
            cupy=cp,
            column_indices=composed_indices,
        )

    def _resolve_raw_selector(self, sel: Any) -> Any:
        """Map a logical column selector onto the underlying ``raw_values``."""
        if self.column_indices is None:
            return sel
        return self.column_indices[sel]

    def standardized_columns(
        self,
        local_variant_indices: NDArray | slice,
        *,
        dtype: Any = None,
    ) -> Any:
        cp = self.cupy
        resolved_dtype = _cupy_compute_dtype(cp) if dtype is None else dtype
        raw_chunk = self.raw_values[:, self._resolve_raw_selector(local_variant_indices)]
        means = self.means[local_variant_indices].astype(resolved_dtype, copy=False)
        scales = self.scales[local_variant_indices].astype(resolved_dtype, copy=False)
        standardized = _standardize_int8_cupy(
            raw_chunk,
            means,
            scales,
            cp,
            dtype=resolved_dtype,
        )
        if hasattr(cp, "asarray"):
            return cp.asarray(standardized, dtype=resolved_dtype)
        standardized_np = np.asarray(standardized)
        return standardized_np

    def __array__(self, dtype: np.dtype[Any] | type | None = None) -> NDArray:
        standardized = self.standardized_columns(slice(None), dtype=np.float32)
        host = standardized.get() if hasattr(standardized, "get") else standardized
        return np.asarray(host, dtype=np.float32 if dtype is None else dtype)


def _cupy_cache_is_int8_standardized(cache: Any | None) -> TypeGuard[_CupyInt8StandardizedCache]:
    return isinstance(cache, _CupyInt8StandardizedCache)


def _cupy_cache_shape(cache: Any) -> tuple[int, int]:
    if _cupy_cache_is_int8_standardized(cache):
        return cache.shape
    return int(cache.shape[0]), int(cache.shape[1])


def _cupy_cache_nbytes(cache: Any) -> int:
    if _cupy_cache_is_int8_standardized(cache):
        return cache.nbytes
    return int(cache.nbytes)


def _cupy_cache_subset_columns(cache: Any, local_variant_indices: NDArray) -> Any:
    resolved_local_indices = np.asarray(local_variant_indices, dtype=np.int32)
    if _cupy_cache_is_int8_standardized(cache):
        return cache.subset(resolved_local_indices)
    if _local_indices_select_all(resolved_local_indices, int(cache.shape[1])):
        return cache
    local_slice = _local_indices_as_slice(resolved_local_indices)
    if local_slice is not None:
        return cache[:, local_slice]
    return cache[:, resolved_local_indices]


def _cupy_cache_standardized_columns(
    cache: Any,
    local_variant_indices: NDArray | slice,
    *,
    cupy: Any,
    dtype: Any = None,
) -> Any:
    if _cupy_cache_is_int8_standardized(cache):
        return cache.standardized_columns(local_variant_indices, dtype=dtype)
    resolved_dtype = _cupy_compute_dtype(cupy) if dtype is None else dtype
    return cache[:, local_variant_indices].astype(resolved_dtype, copy=False)


def _dense_array_cache_available(cache: Any | None) -> bool:
    return cache is not None and not _cupy_cache_is_int8_standardized(cache)


def _iter_cupy_cache_standardized_batches(
    cache: Any,
    *,
    sample_count: int,
    batch_size: int,
    cupy: Any,
    dtype: Any = None,
) -> Iterator[tuple[slice, Any]]:
    variant_count = _cupy_cache_shape(cache)[1]
    resolved_dtype = _cupy_dtype_to_numpy_dtype(np.float32 if dtype is None else dtype)
    static_target_batch_bytes = (
        LOCAL_INT8_STANDARDIZED_STREAMING_TARGET_BATCH_BYTES
        if _cupy_cache_is_int8_standardized(cache)
        else STANDARDIZED_STREAMING_TARGET_BATCH_BYTES
    )
    target_batch_bytes = _gpu_dynamic_standardized_target_batch_bytes(
        cupy,
        static_target_batch_bytes=static_target_batch_bytes,
    )
    bytes_per_variant = sample_count * resolved_dtype.itemsize
    memory_capped_batch_size = max(target_batch_bytes // max(bytes_per_variant, 1), 1)
    safe_batch_size = max(1, min(max(int(batch_size), 1), memory_capped_batch_size))
    for start_index in range(0, variant_count, safe_batch_size):
        stop_index = min(start_index + safe_batch_size, variant_count)
        batch_slice = slice(start_index, stop_index)
        yield batch_slice, _cupy_cache_standardized_columns(
            cache,
            batch_slice,
            cupy=cupy,
            dtype=dtype,
        )


def _iter_selected_cupy_cache_standardized_batches(
    cache: Any,
    local_variant_indices: NDArray,
    *,
    sample_count: int,
    batch_size: int,
    cupy: Any,
    dtype: Any = None,
) -> Iterator[tuple[slice, Any]]:
    selected_variant_count = int(local_variant_indices.shape[0])
    resolved_dtype = _cupy_dtype_to_numpy_dtype(np.float32 if dtype is None else dtype)
    static_target_batch_bytes = (
        LOCAL_INT8_STANDARDIZED_STREAMING_TARGET_BATCH_BYTES
        if _cupy_cache_is_int8_standardized(cache)
        else STANDARDIZED_STREAMING_TARGET_BATCH_BYTES
    )
    target_batch_bytes = _gpu_dynamic_standardized_target_batch_bytes(
        cupy,
        static_target_batch_bytes=static_target_batch_bytes,
    )
    bytes_per_variant = sample_count * resolved_dtype.itemsize
    memory_capped_batch_size = max(target_batch_bytes // max(bytes_per_variant, 1), 1)
    safe_batch_size = max(1, min(max(int(batch_size), 1), memory_capped_batch_size))
    for start_index in range(0, selected_variant_count, safe_batch_size):
        stop_index = min(start_index + safe_batch_size, selected_variant_count)
        operand_slice = slice(start_index, stop_index)
        selected_indices = local_variant_indices[operand_slice]
        maybe_slice = _local_indices_as_slice(selected_indices)
        local_selection: NDArray | slice = maybe_slice if maybe_slice is not None else selected_indices
        yield operand_slice, _cupy_cache_standardized_columns(
            cache,
            local_selection,
            cupy=cupy,
            dtype=dtype,
        )


def _normalize_numpy_vector_operand(
    operand: NDArray | JaxArray,
    *,
    expected_length: int,
    shape_error: str,
    finite_name: str,
) -> NDArray:
    vector = np.asarray(operand, dtype=gpu_compute_numpy_dtype()).reshape(-1)
    if vector.shape[0] != expected_length:
        raise ValueError(shape_error)
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{finite_name} must contain only finite values.")
    return vector


def _normalize_numpy_matrix_operand(
    operand: NDArray | JaxArray,
    *,
    expected_rows: int,
    shape_error: str,
    finite_name: str,
) -> NDArray:
    matrix = np.asarray(operand, dtype=gpu_compute_numpy_dtype())
    if matrix.ndim != 2 or matrix.shape[0] != expected_rows:
        raise ValueError(shape_error)
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{finite_name} must contain only finite values.")
    return matrix


def _normalize_jax_vector_operand(
    operand: NDArray | JaxArray,
    *,
    expected_length: int,
    shape_error: str,
    finite_name: str,
) -> jax.Array:
    vector = jnp.ravel(jnp.asarray(operand, dtype=gpu_compute_jax_dtype()))
    if vector.shape[0] != expected_length:
        raise ValueError(shape_error)
    if not isinstance(vector, jax_core.Tracer):
        if not np.all(np.isfinite(np.asarray(vector))):
            raise ValueError(f"{finite_name} must contain only finite values.")
    return vector


def _normalize_jax_matrix_operand(
    operand: NDArray | JaxArray,
    *,
    expected_rows: int,
    shape_error: str,
    finite_name: str,
) -> jax.Array:
    matrix = jnp.asarray(operand, dtype=gpu_compute_jax_dtype())
    if matrix.ndim != 2 or matrix.shape[0] != expected_rows:
        raise ValueError(shape_error)
    if not isinstance(matrix, jax_core.Tracer):
        if not np.all(np.isfinite(np.asarray(matrix))):
            raise ValueError(f"{finite_name} must contain only finite values.")
    return matrix


def _active_vector_local_indices(vector: NDArray) -> NDArray:
    return np.flatnonzero(vector != 0).astype(np.int32, copy=False)


def _active_matrix_row_local_indices(matrix: NDArray) -> NDArray:
    return np.flatnonzero(np.any(matrix != 0, axis=1)).astype(np.int32, copy=False)


def _local_indices_select_all(local_variant_indices: NDArray, variant_count: int) -> bool:
    if local_variant_indices.shape[0] != variant_count:
        return False
    if variant_count == 0:
        return True
    return bool(
        local_variant_indices[0] == 0
        and local_variant_indices[-1] == variant_count - 1
        and np.all(local_variant_indices == np.arange(variant_count, dtype=local_variant_indices.dtype))
    )


def _local_indices_as_slice(local_variant_indices: NDArray) -> slice | None:
    if local_variant_indices.size == 0:
        return slice(0, 0)
    start = int(local_variant_indices[0])
    stop = int(local_variant_indices[-1]) + 1
    if stop < start:
        return None
    if stop - start != int(local_variant_indices.size):
        return None
    if bool(np.all(local_variant_indices == np.arange(start, stop, dtype=local_variant_indices.dtype))):
        return slice(start, stop)
    return None


def _selected_or_all_local_indices(local_variant_indices: NDArray, variant_count: int) -> NDArray | None:
    return None if _local_indices_select_all(local_variant_indices, variant_count) else local_variant_indices


@dataclass(slots=True)
class _SparseCarrierBackend:
    sample_count: int
    means: NDArray
    scales: NDArray
    variant_ptr: NDArray
    variant_sample_indices: NDArray
    variant_dosages: NDArray
    missing_variant_ptr: NDArray
    missing_variant_sample_indices: NDArray
    sample_ptr: NDArray
    sample_variant_indices: NDArray
    sample_dosages: NDArray
    sample_missing_ptr: NDArray
    sample_missing_variant_indices: NDArray

    @property
    def shape(self) -> tuple[int, int]:
        return self.sample_count, int(self.means.shape[0])

    def materialize_columns(self, local_variant_indices: NDArray) -> NDArray:
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

    def matvec(self, coefficients: NDArray) -> NDArray:
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

    def matmat(self, matrix: NDArray) -> NDArray:
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

    def transpose_matvec(self, vector: NDArray) -> NDArray:
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

    def transpose_matmat(self, matrix: NDArray) -> NDArray:
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

    def subset(self, local_variant_indices: NDArray) -> _SparseCarrierBackend:
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
        selected_variant_sample_indices: list[NDArray] = []
        selected_variant_dosages: list[NDArray] = []
        variant_counts = np.zeros(resolved_local_indices.shape[0], dtype=np.int64)
        selected_missing_sample_indices: list[NDArray] = []
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
    raw_variant_indices: NDArray,
    means: NDArray,
    scales: NDArray,
    sample_count: int,
) -> _SparseCarrierBackend:
    resolved_variant_indices = np.asarray(raw_variant_indices, dtype=np.int32)
    carrier_sample_chunks: list[NDArray] = []
    carrier_dosage_chunks: list[NDArray] = []
    carrier_counts = np.zeros(resolved_variant_indices.shape[0], dtype=np.int64)
    missing_sample_chunks: list[NDArray] = []
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
    has known compilation bugs on some GPU architectures.  Falls back to numpy BLAS on CPU.
    """
    raw: RawGenotypeMatrix | None
    means: NDArray       # per-variant mean from training data
    scales: NDArray      # per-variant std dev from training data
    variant_indices: NDArray  # which columns of raw to use (for subsetting)
    support_counts: NDArray | None = None  # non-zero dosage count per source variant
    sample_count: int | None = field(default=None, repr=False)
    _enable_hybrid_backend: bool = field(default=True, repr=False)
    _dense_cache: NDArray | None = field(init=False, default=None, repr=False)
    _cupy_cache: Any | None = field(init=False, default=None, repr=False)  # cupy.ndarray
    _jax_cache: jax.Array | None = field(init=False, default=None, repr=False)
    _cupy_subset_cache: Any | None = field(init=False, default=None, repr=False)
    _cupy_subset_cache_local_indices: NDArray | None = field(init=False, default=None, repr=False)
    _local_cache_directory: tempfile.TemporaryDirectory[str] | None = field(init=False, default=None, repr=False)
    _dense_backend: StandardizedGenotypeMatrix | None = field(init=False, default=None, repr=False)
    _sparse_backend: _SparseCarrierBackend | None = field(init=False, default=None, repr=False)
    _dense_local_lookup: NDArray | None = field(init=False, default=None, repr=False)
    _sparse_local_lookup: NDArray | None = field(init=False, default=None, repr=False)
    _sample_space_nystrom_basis_cpu_cache: dict[tuple[int, int], NDArray] = field(init=False, default_factory=dict, repr=False)
    _sample_space_nystrom_basis_gpu_cache: dict[tuple[int, int], Any] = field(init=False, default_factory=dict, repr=False)
    _sample_space_probe_projection_cache: dict[tuple[int, int], NDArray] = field(init=False, default_factory=dict, repr=False)
    _sample_space_probe_projection_gpu_cache: dict[tuple[int, int], Any] = field(init=False, default_factory=dict, repr=False)
    _sample_space_cpu_preconditioner_cache: Any | None = field(init=False, default=None, repr=False)
    _sample_space_gpu_preconditioner_cache: Any | None = field(init=False, default=None, repr=False)
    # Parent-lifetime invariant: a GPU subset cache produced by ``subset()`` is a
    # view/slice into the parent's underlying CuPy buffer (see
    # ``_cupy_cache_subset_columns``). If the parent ``StandardizedGenotypeMatrix``
    # is garbage-collected before the subset is consumed, that buffer is freed
    # and the subset becomes a dangling GPU pointer. Holding a strong reference
    # to the parent here keeps it alive for the subset's full lifetime.
    _parent_genotype_matrix: StandardizedGenotypeMatrix | None = field(init=False, default=None, repr=False)
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
            raw=self.raw,
            raw_variant_indices=self.variant_indices[sparse_local_indices],
            means=self.means,
            scales=self.scales,
            sample_count=self.shape[0],
        )
        sparse_lookup = np.full(self.shape[1], -1, dtype=np.int32)
        sparse_lookup[sparse_local_indices] = np.arange(sparse_local_indices.shape[0], dtype=np.int32)
        self._sparse_local_lookup = sparse_lookup
        if dense_local_indices.shape[0] > 0:
            dense_raw = cast(RawGenotypeMatrix, self.raw)
            self._dense_backend = StandardizedGenotypeMatrix(
                raw=dense_raw,
                means=self.means,
                scales=self.scales,
                variant_indices=self.variant_indices[dense_local_indices],
                support_counts=self.support_counts,
                sample_count=self.shape[0],
                _enable_hybrid_backend=False,
            )
            dense_lookup = np.full(self.shape[1], -1, dtype=np.int32)
            dense_lookup[dense_local_indices] = np.arange(dense_local_indices.shape[0], dtype=np.int32)
            self._dense_local_lookup = dense_lookup
        log(
            "    hybrid standardized operator: "
            + f"{sparse_local_indices.shape[0]} sparse variants + {dense_local_indices.shape[0]} dense variants  mem={mem()}"
        )

    def _uses_hybrid_backend(self) -> bool:
        return self._sparse_backend is not None

    def _hybrid_local_components(
        self,
        local_variant_indices: NDArray,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
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

    def _hybrid_parent_local_indices(self) -> tuple[NDArray, NDArray]:
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
        local_variant_indices: NDArray,
        *,
        batch_size: int,
    ) -> NDArray:
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
        self._sample_space_probe_projection_gpu_cache.clear()
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
            use_int8_gpu_cache = self._dense_cache is None and self.raw is not None and _supports_int8_batches(self.raw)
            nbytes = (
                int(self.shape[0]) * int(self.shape[1])
                + int(self.shape[1]) * np.dtype(np.float32).itemsize * 2
                if use_int8_gpu_cache
                else self.dense_bytes()
            )
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
            elif use_int8_gpu_cache:
                log(f"    uploading raw int8 genotypes to GPU ({nbytes / 1e9:.1f} GB incl. scales)  mem={mem()}")
                raw_matrix = self.raw
                if raw_matrix is None:
                    raise RuntimeError("raw int8 GPU cache requires raw backing storage.")
                raw_int8 = cast(Int8BatchCapable, raw_matrix)
                gpu_int8_dtype = cupy.int8 if hasattr(cupy, "int8") else np.int8
                if isinstance(raw_matrix, Int8RawGenotypeMatrix):
                    _madvise_willneed_array(raw_matrix.matrix)
                # Fastest path: if the leaf is a single F-order int8 numpy memmap
                # (the typical state after ``try_cache_persistently``), upload it
                # via N parallel pinned-buffer reads + N async H2D streams. This
                # bypasses the per-batch iterator and saturates disk queue depth
                # + PCIe simultaneously. Bit-identical to the tiled path.
                if isinstance(raw_matrix, Int8RawGenotypeMatrix) and isinstance(raw_matrix.matrix, np.memmap):
                    parallel_dst = cupy.empty(self.shape, dtype=gpu_int8_dtype, order="F")
                    if _try_upload_int8_parallel_memmap(
                        cupy=cupy,
                        raw=raw_matrix,
                        variant_indices=self.variant_indices,
                        gpu_destination=parallel_dst,
                        sample_count=int(self.shape[0]),
                    ):
                        gpu_matrix = parallel_dst
                        cupy.cuda.Device().synchronize()
                        self._cupy_cache = _CupyInt8StandardizedCache(
                            raw_values=gpu_matrix,
                            means=cupy.asarray(self.means[self.variant_indices], dtype=cupy.float32),
                            scales=cupy.asarray(self.scales[self.variant_indices], dtype=cupy.float32),
                            cupy=cupy,
                        )
                        cupy.cuda.Device().synchronize()
                        import gc
                        gc.collect()
                        log(f"    CuPy GPU matrix ready ({_cupy_cache_nbytes(self._cupy_cache) / 1e9:.1f} GB)  mem={mem()}")
                        return True
                    del parallel_dst
                # Fast path: if the entire int8 block fits inside the GPU cache
                # budget, prefer one contiguous raw read plus one H2D copy over
                # Python tile iteration. AoU fit-stage caches are Fortran-order
                # int8 mmaps, so this turns a minutes-long tiled upload into one
                # sequential mmap pass and one device allocation.
                int8_total_bytes = int(self.shape[0]) * int(self.shape[1])
                if int8_total_bytes <= int(budget_bytes * INT8_ONE_SHOT_GPU_BUDGET_FRACTION):
                    full_int8_block = _read_int8_columns_one_shot(raw_matrix, self.variant_indices)
                    if full_int8_block is not None:
                        log(
                            "    raw int8 block fits one-shot upload budget; "
                            + f"uploading {self.shape[1]} variants in one H2D copy  mem={mem()}"
                        )
                        gpu_matrix = cupy.asarray(full_int8_block, dtype=gpu_int8_dtype)
                        if not gpu_matrix.flags.f_contiguous:
                            if hasattr(cupy, "asfortranarray"):
                                gpu_matrix = cupy.asfortranarray(gpu_matrix)
                            else:
                                gpu_matrix = np.asfortranarray(np.asarray(gpu_matrix, dtype=gpu_int8_dtype))
                    else:
                        gpu_matrix = cupy.empty(self.shape, dtype=gpu_int8_dtype, order="F")
                        _upload_int8_tiles_overlapped(
                            cupy=cupy,
                            raw_int8=raw_int8,
                            variant_indices=self.variant_indices,
                            gpu_destination=gpu_matrix,
                            sample_count=int(self.shape[0]),
                            upload_batch_size=auto_batch_size_i8(self.shape[0]),
                            gpu_int8_dtype=gpu_int8_dtype,
                        )
                else:
                    gpu_matrix = cupy.empty(self.shape, dtype=gpu_int8_dtype, order="F")
                    _upload_int8_tiles_overlapped(
                        cupy=cupy,
                        raw_int8=raw_int8,
                        variant_indices=self.variant_indices,
                        gpu_destination=gpu_matrix,
                        sample_count=int(self.shape[0]),
                        upload_batch_size=auto_batch_size_i8(self.shape[0]),
                        gpu_int8_dtype=gpu_int8_dtype,
                    )
                # Synchronization invariant: callers may read ``self._cupy_cache``
                # immediately after ``try_materialize_gpu()`` returns, so make sure
                # every async H2D copy issued above has completed before we publish
                # the cache reference.
                cupy.cuda.Device().synchronize()
                self._cupy_cache = _CupyInt8StandardizedCache(
                    raw_values=gpu_matrix,
                    means=cupy.asarray(self.means[self.variant_indices], dtype=cupy.float32),
                    scales=cupy.asarray(self.scales[self.variant_indices], dtype=cupy.float32),
                    cupy=cupy,
                )
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
                elif _supports_int8_batches(self.raw):
                    upload_batch_size = auto_batch_size_i8(self.shape[0])
                else:
                    upload_batch_size = auto_batch_size(self.shape[0])
                if _supports_int8_batches(self.raw):
                    _upload_standardized_int8_tiles_overlapped(
                        cupy=cupy,
                        raw_int8=self.raw,
                        variant_indices=self.variant_indices,
                        means=self.means,
                        scales=self.scales,
                        gpu_destination=gpu_matrix,
                        sample_count=int(self.shape[0]),
                        upload_batch_size=int(upload_batch_size),
                        standardized_dtype=cupy.float32,
                    )
                else:
                    raw_matrix = cast(RawGenotypeMatrix, self.raw)
                    for batch_slice, standardized_batch in _iter_standardized_gpu_batches(
                        raw_matrix,
                        self.variant_indices,
                        self.means,
                        self.scales,
                        batch_size=upload_batch_size,
                        cupy=cupy,
                    ):
                        gpu_matrix[:, batch_slice] = standardized_batch
                # Synchronization invariant: callers may read ``self._cupy_cache``
                # immediately after ``try_materialize_gpu()`` returns, so make sure
                # every async H2D copy issued above has completed before we publish
                # the cache reference.
                cupy.cuda.Device().synchronize()
                self._cupy_cache = gpu_matrix
            cupy.cuda.Device().synchronize()
            import gc
            gc.collect()
            log(f"    CuPy GPU matrix ready ({_cupy_cache_nbytes(self._cupy_cache) / 1e9:.1f} GB)  mem={mem()}")
            return True
        except (MemoryError, OSError, RuntimeError) as exc:
            log(f"    CuPy GPU upload failed ({exc})  mem={mem()}")
            self._cupy_cache = None
            return False

    def try_materialize_gpu_subset(
        self,
        local_variant_indices: Sequence[int] | NDArray,
    ) -> Any | None:
        """Materialize a selected standardized column subset onto GPU memory when possible."""
        resolved_local_indices = np.asarray(local_variant_indices, dtype=np.int32)
        if resolved_local_indices.ndim != 1:
            raise ValueError("local_variant_indices must be 1D.")
        if resolved_local_indices.size == 0:
            return None
        if self._cupy_cache is not None:
            cupy = _try_import_cupy()
            if cupy is None:
                raise RuntimeError("CuPy cache requires CuPy runtime.")
            if _cupy_cache_is_int8_standardized(self._cupy_cache):
                return _cupy_cache_standardized_columns(
                    self._cupy_cache,
                    resolved_local_indices,
                    cupy=cupy,
                    dtype=cupy.float32,
                )
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
                subset_batch_size = (
                    auto_batch_size_i8(self.shape[0])
                    if _supports_int8_batches(self.raw)
                    else auto_batch_size(self.shape[0])
                )
                if _supports_int8_batches(self.raw):
                    _upload_standardized_int8_tiles_overlapped(
                        cupy=cupy,
                        raw_int8=self.raw,
                        variant_indices=selected_variant_indices,
                        means=self.means,
                        scales=self.scales,
                        gpu_destination=gpu_subset,
                        sample_count=int(self.shape[0]),
                        upload_batch_size=int(subset_batch_size),
                        standardized_dtype=cupy.float32,
                    )
                else:
                    for batch_slice, standardized_batch in _iter_standardized_gpu_batches(
                        self.raw,
                        selected_variant_indices,
                        self.means,
                        self.scales,
                        batch_size=subset_batch_size,
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
        except (MemoryError, OSError, RuntimeError) as exc:
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
        return jax_dense_linear_algebra_preferred() and (
            self._dense_cache is not None
            or self._jax_cache is not None
            or _dense_array_cache_available(self._cupy_cache)
        )

    def _ensure_jax_cache(self) -> jax.Array:
        if self._jax_cache is not None:
            return self._jax_cache
        if self._dense_cache is not None:
            cache_source = self._dense_cache
            jax_cache = jnp.asarray(cache_source, dtype=gpu_compute_jax_dtype())
        elif _dense_array_cache_available(self._cupy_cache):
            cupy_cache_source = self._cupy_cache
            if hasattr(cupy_cache_source, "__dlpack__"):
                jax_cache = jax_dlpack.from_dlpack(cupy_cache_source).astype(gpu_compute_jax_dtype())
            else:
                jax_cache = jnp.asarray(np.asarray(cupy_cache_source), dtype=gpu_compute_jax_dtype())
        else:
            raise RuntimeError("JAX cache requires a dense materialized genotype matrix.")
        if isinstance(jax_cache, jax_core.Tracer):
            return jax_cache
        self._jax_cache = jax_cache
        return jax_cache

    def try_cache_locally(self) -> bool:
        """Rebase onto a local int8 memmap to avoid repeated upstream streaming passes."""
        if self.raw is None or self._dense_cache is not None or self._cupy_cache is not None:
            return False
        if not _supports_int8_batches(self.raw):
            return False
        if isinstance(self.raw, Int8RawGenotypeMatrix):
            # Already mmap-backed int8 (e.g. swapped in by io.py from a
            # persisted PLINK int8 cache). Hint the kernel to keep pages
            # resident so per-block GPU uploads run at RAM speed.
            _madvise_willneed_array(self.raw.matrix)
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
            raw_int8 = self.raw
            def _column_batches() -> Iterator[NDArray]:
                for raw_batch in raw_int8.iter_column_batches_i8(self.variant_indices, batch_size=batch_size):
                    yield raw_batch.values
            _stream_write_int8_npy(
                cache_path,
                shape=self.shape,
                column_batches=_column_batches(),
                fortran_order=True,
            )
            cache_mmap = np.load(cache_path, mmap_mode="r")
            _madvise_willneed_array(cache_mmap)
            rebased_raw = Int8RawGenotypeMatrix(cache_mmap)
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
        except (OSError, RuntimeError, ValueError) as exc:
            cache_directory.cleanup()
            log(f"    local int8 cache failed ({exc})  mem={mem()}")
            return False

    def try_cache_persistently(self, cache_path: Path) -> bool:
        """Persist a reduced raw int8 cache to a stable path for reuse across runs."""
        if self.raw is None:
            return False
        if not _supports_int8_batches(self.raw):
            return False
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        selected_variant_count = int(self.variant_indices.shape[0])
        # Short-circuit if a previous run already wrote this exact cache. Just rebase
        # onto the existing F-order int8 memmap; no rewrite, no double-build.
        if cache_path.exists():
            try:
                existing_mmap = np.load(cache_path, mmap_mode="r")
                if existing_mmap.shape == self.shape and existing_mmap.dtype == np.int8:
                    _madvise_willneed_array(existing_mmap)
                    self.raw = Int8RawGenotypeMatrix(existing_mmap)
                    self.means = np.asarray(self.means[self.variant_indices], dtype=np.float32)
                    self.scales = np.asarray(self.scales[self.variant_indices], dtype=np.float32)
                    if self.support_counts is not None:
                        self.support_counts = np.asarray(self.support_counts[self.variant_indices], dtype=np.int32)
                    self.variant_indices = np.arange(selected_variant_count, dtype=np.int32)
                    self.clear_sample_space_nystrom_cache()
                    self._local_cache_directory = None
                    self._cupy_subset_cache = None
                    self._cupy_subset_cache_local_indices = None
                    self._enable_hybrid_backend = False
                    log(f"    persistent int8 cache reused from {cache_path}  mem={mem()}")
                    return True
            except (OSError, ValueError) as exc:
                log(f"    existing persistent int8 cache unreadable ({exc}); rebuilding")
        batch_size = auto_batch_size_i8(self.shape[0])
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
            raw_int8 = self.raw
            def _column_batches() -> Iterator[NDArray]:
                for raw_batch in raw_int8.iter_column_batches_i8(self.variant_indices, batch_size=batch_size):
                    yield raw_batch.values
            _stream_write_int8_npy(
                temp_path,
                shape=self.shape,
                column_batches=_column_batches(),
                fortran_order=True,
            )
            temp_path.replace(cache_path)
            persisted_mmap = np.load(cache_path, mmap_mode="r")
            _madvise_willneed_array(persisted_mmap)
            persisted_raw = Int8RawGenotypeMatrix(persisted_mmap)
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
        except (OSError, RuntimeError, ValueError) as exc:
            log(f"    persistent int8 cache failed ({exc})  mem={mem()}")
            return False
        finally:
            if temp_directory.exists():
                for child in temp_directory.iterdir():
                    child.unlink(missing_ok=True)
                temp_directory.rmdir()

    def subset(self, local_variant_indices: Sequence[int] | NDArray) -> StandardizedGenotypeMatrix:
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
            subset._cupy_cache = _cupy_cache_subset_columns(self._cupy_cache, resolved_local_indices)
            # Parent-lifetime invariant: ``_cupy_cache_subset_columns`` returns a
            # view/slice into ``self._cupy_cache``'s GPU buffer (no copy). If
            # ``self`` is garbage-collected before ``subset`` is consumed, the
            # underlying GPU memory is freed and ``subset._cupy_cache`` becomes a
            # dangling pointer. Retain a strong reference to ``self`` so the
            # parent's buffer outlives the subset.
            subset._parent_genotype_matrix = self
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
            if _supports_int8_batches(self.raw)
            else STANDARDIZED_STREAMING_TARGET_BATCH_BYTES
        )
        safe_batch_size = _effective_standardized_streaming_batch_size(
            self.shape[0],
            batch_size,
            target_batch_bytes=target_batch_bytes,
        )
        local_start = 0
        if _supports_int8_batches(self.raw):
            raw_int8 = self.raw
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

    def materialize(self, batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE) -> NDArray:
        if self._dense_cache is not None:
            return self._dense_cache
        if self._cupy_cache is not None:
            if _cupy_cache_is_int8_standardized(self._cupy_cache):
                dense_from_cupy = np.asarray(self._cupy_cache, dtype=np.float32)
            else:
                dense_from_cupy = self._cupy_cache.get()  # cupy -> numpy
            self._dense_cache = dense_from_cupy
            self._jax_cache = None
            return dense_from_cupy
        matrix = np.empty(self.shape, dtype=np.float32)
        for batch in self.iter_column_batches(batch_size=batch_size):
            matrix[:, batch.variant_indices] = batch.values
        self._dense_cache = matrix
        self._jax_cache = None
        return matrix

    def _streaming_gpu_context(
        self, batch_size: int, *, cupy: Any = None, dtype: Any = None
    ) -> tuple[Any, int | None]:
        if self._cupy_cache is not None or self.supports_jax_dense_ops() or self.raw is None or self._uses_hybrid_backend():
            return None, None
        resolved_cupy = _try_import_cupy() if cupy is None else cupy
        if resolved_cupy is None:
            return None, None
        return resolved_cupy, _gpu_streaming_batch_size(
            self.raw,
            sample_count=self.shape[0],
            requested_batch_size=batch_size,
            cupy=resolved_cupy,
            dtype=dtype,
        )

    def _gpu_variant_matmul(
        self,
        matrix: Any,
        *,
        batch_size: int,
        cupy: Any,
        dtype: Any = None,
        local_variant_indices: NDArray | None = None,
        progress_label: str | None = None,
    ) -> Any:
        resolved_dtype = _cupy_compute_dtype(cupy) if dtype is None else dtype
        matrix_gpu = cupy.asarray(matrix, dtype=resolved_dtype)
        if matrix_gpu.ndim == 1:
            matrix_gpu = matrix_gpu[:, None]
            vector_input = True
        elif matrix_gpu.ndim == 2:
            vector_input = False
        else:
            raise ValueError("GPU genotype matmul expects a vector or matrix right-hand side.")
        any_fn = getattr(cupy, "any", np.any)
        if not bool(any_fn(matrix_gpu)):
            result_gpu = cupy.zeros((self.shape[0], matrix_gpu.shape[1]), dtype=resolved_dtype)
            return result_gpu[:, 0] if vector_input else result_gpu
        resolved_local_indices = (
            np.arange(self.shape[1], dtype=np.int32)
            if local_variant_indices is None
            else np.asarray(local_variant_indices, dtype=np.int32)
        )
        if resolved_local_indices.ndim != 1:
            raise ValueError("local_variant_indices must be 1D.")
        if matrix_gpu.shape[0] != resolved_local_indices.shape[0]:
            raise ValueError("GPU genotype matmul right-hand side row count must match the selected variant count.")
        if resolved_local_indices.size == 0:
            result_gpu = cupy.zeros((self.shape[0], matrix_gpu.shape[1]), dtype=resolved_dtype)
            return result_gpu[:, 0] if vector_input else result_gpu
        if self._cupy_cache is not None:
            if local_variant_indices is None:
                cache = self._cupy_cache
                if _cupy_cache_is_int8_standardized(cache):
                    result_gpu = cupy.zeros((self.shape[0], matrix_gpu.shape[1]), dtype=resolved_dtype)
                    for batch_slice, standardized_batch in _iter_cupy_cache_standardized_batches(
                        cache,
                        sample_count=self.shape[0],
                        batch_size=batch_size,
                        cupy=cupy,
                        dtype=resolved_dtype,
                    ):
                        result_gpu += standardized_batch @ matrix_gpu[batch_slice, :]
                else:
                    result_gpu = cache.astype(resolved_dtype, copy=False) @ matrix_gpu
            else:
                result_gpu = cupy.zeros((self.shape[0], matrix_gpu.shape[1]), dtype=resolved_dtype)
                for operand_slice, standardized_batch in _iter_selected_cupy_cache_standardized_batches(
                    self._cupy_cache,
                    resolved_local_indices,
                    sample_count=self.shape[0],
                    batch_size=batch_size,
                    cupy=cupy,
                    dtype=resolved_dtype,
                ):
                    result_gpu += standardized_batch @ matrix_gpu[operand_slice, :]
            return result_gpu[:, 0] if vector_input else result_gpu
        streaming_cupy, streaming_batch_size = self._streaming_gpu_context(
            batch_size,
            cupy=cupy,
            dtype=resolved_dtype,
        )
        if streaming_cupy is None or streaming_batch_size is None or self.raw is None:
            raise RuntimeError("GPU genotype matmul requires a GPU cache or a streaming raw backend.")
        active_variant_indices = self.variant_indices[resolved_local_indices]
        result_gpu = cupy.zeros((self.shape[0], matrix_gpu.shape[1]), dtype=resolved_dtype)
        raw_matrix = self.raw
        total_variants = int(resolved_local_indices.shape[0])
        completed_variants = 0
        last_logged_variants = 0
        log_interval = max(total_variants // 50, 1)
        import time as _time
        _t_start = _time.monotonic()
        if progress_label is not None:
            log(f"        {progress_label}: start streaming {total_variants:,} variants (batch_size={streaming_batch_size})  mem={mem()}")
        for batch_slice, standardized_batch in _iter_standardized_gpu_batches(
            raw_matrix,
            active_variant_indices,
            self.means,
            self.scales,
            batch_size=streaming_batch_size,
            cupy=cupy,
            dtype=resolved_dtype,
        ):
            result_gpu += standardized_batch @ matrix_gpu[batch_slice, :]
            if progress_label is not None:
                completed_variants = (
                    batch_slice.stop
                    if isinstance(batch_slice, slice)
                    else completed_variants + standardized_batch.shape[1]
                )
                if completed_variants - last_logged_variants >= log_interval:
                    last_logged_variants = completed_variants
                    _elapsed = _time.monotonic() - _t_start
                    _rate = completed_variants / max(_elapsed, 1e-6)
                    _eta = (total_variants - completed_variants) / max(_rate, 1e-6)
                    log(
                        f"        {progress_label}: {completed_variants:,}/{total_variants:,} "
                        f"({100*completed_variants/total_variants:.1f}%) "
                        f"elapsed={_elapsed:.0f}s rate={_rate:,.0f}v/s eta={_eta:.0f}s  mem={mem()}"
                    )
        if progress_label is not None:
            _elapsed = _time.monotonic() - _t_start
            log(f"        {progress_label}: done {total_variants:,} variants in {_elapsed:.1f}s  mem={mem()}")
        return result_gpu[:, 0] if vector_input else result_gpu

    def _gpu_transpose_matmul(
        self,
        matrix: Any,
        *,
        batch_size: int,
        cupy: Any,
        dtype: Any = None,
        progress_label: str | None = None,
    ) -> Any:
        resolved_dtype = _cupy_compute_dtype(cupy) if dtype is None else dtype
        matrix_gpu = cupy.asarray(matrix, dtype=resolved_dtype)
        if matrix_gpu.ndim == 1:
            matrix_gpu = matrix_gpu[:, None]
            vector_input = True
        elif matrix_gpu.ndim == 2:
            vector_input = False
        else:
            raise ValueError("GPU genotype transpose matmul expects a vector or matrix right-hand side.")
        if matrix_gpu.shape[0] != self.shape[0]:
            raise ValueError("GPU genotype transpose matmul right-hand side row count must match the sample count.")
        if self._cupy_cache is not None:
            if _cupy_cache_is_int8_standardized(self._cupy_cache):
                result_gpu = cupy.empty((self.shape[1], matrix_gpu.shape[1]), dtype=resolved_dtype)
                for batch_slice, standardized_batch in _iter_cupy_cache_standardized_batches(
                    self._cupy_cache,
                    sample_count=self.shape[0],
                    batch_size=batch_size,
                    cupy=cupy,
                    dtype=resolved_dtype,
                ):
                    result_gpu[batch_slice, :] = standardized_batch.T @ matrix_gpu
            else:
                result_gpu = self._cupy_cache.astype(resolved_dtype, copy=False).T @ matrix_gpu
            return result_gpu[:, 0] if vector_input else result_gpu
        streaming_cupy, streaming_batch_size = self._streaming_gpu_context(
            batch_size,
            cupy=cupy,
            dtype=resolved_dtype,
        )
        if streaming_cupy is None or streaming_batch_size is None or self.raw is None:
            raise RuntimeError("GPU genotype transpose matmul requires a GPU cache or a streaming raw backend.")
        raw_matrix = self.raw
        if _supports_int8_batches(raw_matrix):
            result_gpu = _gpu_int8_transpose_matmul(
                raw_int8=raw_matrix,
                variant_indices=self.variant_indices,
                means=self.means,
                scales=self.scales,
                matrix_gpu=matrix_gpu,
                batch_size=streaming_batch_size,
                cupy=cupy,
                dtype=resolved_dtype,
                progress_label=progress_label,
            )
            return result_gpu[:, 0] if vector_input else result_gpu
        result_gpu = cupy.empty((self.shape[1], matrix_gpu.shape[1]), dtype=resolved_dtype)
        total_variants = self.shape[1]
        completed_variants = 0
        last_logged_variants = 0
        log_interval = max(total_variants // 50, 1)
        import time as _time
        _t_start = _time.monotonic()
        if progress_label is not None:
            log(f"        {progress_label}: start streaming {total_variants:,} variants (batch_size={streaming_batch_size})  mem={mem()}")
        for batch_slice, standardized_batch in _iter_standardized_gpu_batches(
            raw_matrix,
            self.variant_indices,
            self.means,
            self.scales,
            batch_size=streaming_batch_size,
            cupy=cupy,
            dtype=resolved_dtype,
        ):
            result_gpu[batch_slice, :] = standardized_batch.T @ matrix_gpu
            if progress_label is not None:
                completed_variants = (
                    batch_slice.stop
                    if isinstance(batch_slice, slice)
                    else completed_variants + standardized_batch.shape[1]
                )
                if completed_variants - last_logged_variants >= log_interval:
                    last_logged_variants = completed_variants
                    _elapsed = _time.monotonic() - _t_start
                    _rate = completed_variants / max(_elapsed, 1e-6)
                    _eta = (total_variants - completed_variants) / max(_rate, 1e-6)
                    log(
                        f"        {progress_label}: {completed_variants:,}/{total_variants:,} "
                        f"({100*completed_variants/total_variants:.1f}%) "
                        f"elapsed={_elapsed:.0f}s rate={_rate:,.0f}v/s eta={_eta:.0f}s  mem={mem()}"
                    )
        if progress_label is not None:
            _elapsed = _time.monotonic() - _t_start
            log(f"        {progress_label}: done {total_variants:,} variants in {_elapsed:.1f}s  mem={mem()}")
        return result_gpu[:, 0] if vector_input else result_gpu

    def gpu_matmat(
        self,
        matrix: Any,
        *,
        batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE,
        cupy: Any = None,
        dtype: Any = None,
    ) -> Any:
        resolved_cupy = _try_import_cupy() if cupy is None else cupy
        if resolved_cupy is None:
            raise RuntimeError("GPU genotype matmul requires CuPy.")
        return self._gpu_variant_matmul(
            matrix,
            batch_size=batch_size,
            cupy=resolved_cupy,
            dtype=dtype,
        )

    def gpu_transpose_matmat(
        self,
        matrix: Any,
        *,
        batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE,
        cupy: Any = None,
        dtype: Any = None,
    ) -> Any:
        resolved_cupy = _try_import_cupy() if cupy is None else cupy
        if resolved_cupy is None:
            raise RuntimeError("GPU genotype transpose matmul requires CuPy.")
        return self._gpu_transpose_matmul(
            matrix,
            batch_size=batch_size,
            cupy=resolved_cupy,
            dtype=dtype,
        )

    def _iter_selected_column_batches(
        self,
        local_variant_indices: NDArray,
        *,
        batch_size: int,
    ) -> Iterator[RawGenotypeBatch]:
        resolved_local_indices = np.asarray(local_variant_indices, dtype=np.int32)
        if resolved_local_indices.ndim != 1:
            raise ValueError("local_variant_indices must be 1D.")
        if resolved_local_indices.size == 0:
            return
        if self._dense_cache is not None:
            safe_batch_size = max(int(batch_size), 1)
            for start_index in range(0, resolved_local_indices.shape[0], safe_batch_size):
                batch_local_indices = resolved_local_indices[start_index : start_index + safe_batch_size]
                yield RawGenotypeBatch(
                    variant_indices=batch_local_indices,
                    values=np.asarray(self._dense_cache[:, batch_local_indices], dtype=np.float32),
                )
            return
        if self._uses_hybrid_backend():
            safe_batch_size = _effective_standardized_streaming_batch_size(
                self.shape[0],
                max(int(batch_size), 1),
                target_batch_bytes=STANDARDIZED_STREAMING_TARGET_BATCH_BYTES,
            )
            for start_index in range(0, resolved_local_indices.shape[0], safe_batch_size):
                batch_local_indices = resolved_local_indices[start_index : start_index + safe_batch_size]
                yield RawGenotypeBatch(
                    variant_indices=batch_local_indices,
                    values=self._materialize_hybrid_columns(batch_local_indices, batch_size=safe_batch_size),
                )
            return
        if self.raw is None:
            raise RuntimeError("streaming genotype batches require raw backing storage or a materialized cache.")
        target_batch_bytes = (
            LOCAL_INT8_STANDARDIZED_STREAMING_TARGET_BATCH_BYTES
            if _supports_int8_batches(self.raw)
            else STANDARDIZED_STREAMING_TARGET_BATCH_BYTES
        )
        safe_batch_size = _effective_standardized_streaming_batch_size(
            self.shape[0],
            batch_size,
            target_batch_bytes=target_batch_bytes,
        )
        selected_variant_indices = self.variant_indices[resolved_local_indices]
        selected_means = np.asarray(self.means[selected_variant_indices], dtype=np.float32)
        selected_scales = np.asarray(self.scales[selected_variant_indices], dtype=np.float32)
        local_start = 0
        if _supports_int8_batches(self.raw):
            raw_int8 = self.raw
            for raw_batch in raw_int8.iter_column_batches_i8(selected_variant_indices, batch_size=safe_batch_size):
                batch_width = raw_batch.variant_indices.shape[0]
                local_stop = local_start + batch_width
                batch_slice = slice(local_start, local_stop)
                yield RawGenotypeBatch(
                    variant_indices=resolved_local_indices[batch_slice],
                    values=_standardize_batch_i8(
                        raw_batch.values,
                        selected_means[batch_slice],
                        selected_scales[batch_slice],
                    ),
                )
                local_start = local_stop
            return
        for raw_batch in self.raw.iter_column_batches(selected_variant_indices, batch_size=safe_batch_size):
            batch_width = raw_batch.variant_indices.shape[0]
            local_stop = local_start + batch_width
            batch_slice = slice(local_start, local_stop)
            yield RawGenotypeBatch(
                variant_indices=resolved_local_indices[batch_slice],
                values=_standardize_batch(
                    raw_batch.values,
                    selected_means[batch_slice],
                    selected_scales[batch_slice],
                ),
            )
            local_start = local_stop

    def matvec_numpy(self, coefficients: NDArray | JaxArray, batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE) -> NDArray:
        coeff_np = _normalize_numpy_vector_operand(
            coefficients,
            expected_length=self.shape[1],
            shape_error="coefficient vector must match genotype column count.",
            finite_name="coefficient vector",
        )
        active_local_indices = _active_vector_local_indices(coeff_np)
        if active_local_indices.size == 0:
            return np.zeros(self.shape[0], dtype=coeff_np.dtype)
        selected_local_indices = _selected_or_all_local_indices(active_local_indices, self.shape[1])
        selected_coefficients = coeff_np if selected_local_indices is None else coeff_np[active_local_indices]
        if self._cupy_cache is not None:
            cupy = _try_import_cupy()
            if cupy is None:
                if _cupy_cache_is_int8_standardized(self._cupy_cache):
                    raise RuntimeError("CuPy int8 cache requires CuPy runtime.")
                dense_cache = (
                    np.asarray(self._cupy_cache, dtype=coeff_np.dtype)
                    if selected_local_indices is None
                    else np.asarray(self._cupy_cache[:, selected_local_indices], dtype=coeff_np.dtype)
                )
                return np.asarray(dense_cache @ selected_coefficients, dtype=coeff_np.dtype)
            return _cupy_to_numpy(
                self._gpu_variant_matmul(
                    selected_coefficients,
                    batch_size=batch_size,
                    cupy=cupy,
                    dtype=_cupy_compute_dtype(cupy),
                    local_variant_indices=selected_local_indices,
                ),
                dtype=coeff_np.dtype,
            )
        cupy = _try_import_cupy()
        streaming_dtype = None if cupy is None else _cupy_compute_dtype(cupy)
        cupy, streaming_batch_size = self._streaming_gpu_context(
            batch_size,
            cupy=cupy,
            dtype=streaming_dtype,
        )
        if cupy is not None and streaming_batch_size is not None:
            return _cupy_to_numpy(
                self._gpu_variant_matmul(
                    selected_coefficients,
                    batch_size=streaming_batch_size,
                    cupy=cupy,
                    dtype=streaming_dtype,
                    local_variant_indices=selected_local_indices,
                    progress_label="GPU matvec",
                ),
                dtype=coeff_np.dtype,
            )
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
                result += self._dense_backend.matvec_numpy(coeff_np[dense_parent_local_indices], batch_size=batch_size)
            return np.asarray(result, dtype=coeff_np.dtype)
        result = np.zeros(self.shape[0], dtype=coeff_np.dtype)
        _total_variants = int(active_local_indices.shape[0])
        _completed_variants = 0
        _last_log_variants = 0
        _log_interval = max(_total_variants // 10, 1)
        for batch in self._iter_selected_column_batches(active_local_indices, batch_size=batch_size):
            batch_values = np.asarray(batch.values, dtype=result.dtype)
            result += batch_values @ coeff_np[batch.variant_indices]
            _completed_variants += len(batch.variant_indices)
            if _completed_variants - _last_log_variants >= _log_interval:
                _last_log_variants = _completed_variants
                log(f"        matvec: {_completed_variants:,}/{_total_variants:,} ({100*_completed_variants/_total_variants:.0f}%)  mem={mem()}")
        return result

    def matvec_jax(self, coefficients: NDArray | JaxArray, batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE) -> JaxArray:
        if self.supports_jax_dense_ops():
            coeff_jax = _normalize_jax_vector_operand(
                coefficients,
                expected_length=self.shape[1],
                shape_error="coefficient vector must match genotype column count.",
                finite_name="coefficient vector",
            )
            return self._ensure_jax_cache() @ coeff_jax
        return _as_gpu_compute_jax(self.matvec_numpy(coefficients, batch_size=batch_size))

    def matvec(self, coefficients: NDArray | JaxArray, batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE) -> JaxArray:
        """Compute X_std @ beta with JAX as the public return type."""
        return self.matvec_jax(coefficients, batch_size=batch_size)

    def matmat_numpy(self, matrix: NDArray | JaxArray, batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE) -> NDArray:
        m_np = _normalize_numpy_matrix_operand(
            matrix,
            expected_rows=self.shape[1],
            shape_error="variant matrix must match genotype column count.",
            finite_name="variant matrix",
        )
        active_local_indices = _active_matrix_row_local_indices(m_np)
        if active_local_indices.size == 0:
            return np.zeros((self.shape[0], m_np.shape[1]), dtype=m_np.dtype)
        selected_local_indices = _selected_or_all_local_indices(active_local_indices, self.shape[1])
        selected_matrix = m_np if selected_local_indices is None else m_np[active_local_indices, :]
        if self._cupy_cache is not None:
            cupy = _try_import_cupy()
            if cupy is None:
                if _cupy_cache_is_int8_standardized(self._cupy_cache):
                    raise RuntimeError("CuPy int8 cache requires CuPy runtime.")
                dense_cache = (
                    np.asarray(self._cupy_cache, dtype=m_np.dtype)
                    if selected_local_indices is None
                    else np.asarray(self._cupy_cache[:, selected_local_indices], dtype=m_np.dtype)
                )
                return np.asarray(dense_cache @ selected_matrix, dtype=m_np.dtype)
            return _cupy_to_numpy(
                self._gpu_variant_matmul(
                    selected_matrix,
                    batch_size=batch_size,
                    cupy=cupy,
                    dtype=_cupy_compute_dtype(cupy),
                    local_variant_indices=selected_local_indices,
                ),
                dtype=m_np.dtype,
            )
        cupy = _try_import_cupy()
        streaming_dtype = None if cupy is None else _cupy_compute_dtype(cupy)
        cupy, streaming_batch_size = self._streaming_gpu_context(
            batch_size,
            cupy=cupy,
            dtype=streaming_dtype,
        )
        if cupy is not None and streaming_batch_size is not None:
            return _cupy_to_numpy(
                self._gpu_variant_matmul(
                    selected_matrix,
                    batch_size=streaming_batch_size,
                    cupy=cupy,
                    dtype=streaming_dtype,
                    local_variant_indices=selected_local_indices,
                ),
                dtype=m_np.dtype,
            )
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
                output += self._dense_backend.matmat_numpy(m_np[dense_parent_local_indices, :], batch_size=batch_size)
            return output
        output = np.zeros((self.shape[0], m_np.shape[1]), dtype=m_np.dtype)
        for batch in self._iter_selected_column_batches(active_local_indices, batch_size=batch_size):
            batch_values = np.asarray(batch.values, dtype=output.dtype)
            output += batch_values @ m_np[batch.variant_indices, :]
        return output

    def matmat_jax(self, matrix: NDArray | JaxArray, batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE) -> JaxArray:
        if self.supports_jax_dense_ops():
            matrix_jax = _normalize_jax_matrix_operand(
                matrix,
                expected_rows=self.shape[1],
                shape_error="variant matrix must match genotype column count.",
                finite_name="variant matrix",
            )
            return self._ensure_jax_cache() @ matrix_jax
        return _as_gpu_compute_jax(self.matmat_numpy(matrix, batch_size=batch_size))

    def matmat(self, matrix: NDArray | JaxArray, batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE) -> JaxArray:
        """Compute X_std @ M with JAX as the public return type."""
        return self.matmat_jax(matrix, batch_size=batch_size)

    def transpose_matvec_numpy(self, vector: NDArray | JaxArray, batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE) -> NDArray:
        v_np = _normalize_numpy_vector_operand(
            vector,
            expected_length=self.shape[0],
            shape_error="sample vector must match genotype row count.",
            finite_name="sample vector",
        )
        if not np.any(v_np):
            return np.zeros(self.shape[1], dtype=v_np.dtype)
        if self._cupy_cache is not None:
            cupy = _try_import_cupy()
            if cupy is None:
                if _cupy_cache_is_int8_standardized(self._cupy_cache):
                    raise RuntimeError("CuPy int8 cache requires CuPy runtime.")
                return np.asarray(np.asarray(self._cupy_cache, dtype=v_np.dtype).T @ v_np, dtype=v_np.dtype)
            return _cupy_to_numpy(
                self._gpu_transpose_matmul(
                    v_np,
                    batch_size=batch_size,
                    cupy=cupy,
                    dtype=_cupy_compute_dtype(cupy),
                ),
                dtype=v_np.dtype,
            )
        cupy = _try_import_cupy()
        streaming_dtype = None if cupy is None else _cupy_compute_dtype(cupy)
        cupy, streaming_batch_size = self._streaming_gpu_context(
            batch_size,
            cupy=cupy,
            dtype=streaming_dtype,
        )
        if cupy is not None and streaming_batch_size is not None:
            return _cupy_to_numpy(
                self._gpu_transpose_matmul(
                    v_np,
                    batch_size=streaming_batch_size,
                    cupy=cupy,
                    dtype=streaming_dtype,
                    progress_label="GPU transpose_matvec",
                ),
                dtype=v_np.dtype,
            )
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
                output[dense_parent_local_indices] = self._dense_backend.transpose_matvec_numpy(v_np, batch_size=batch_size)
            return output
        output = np.empty(self.shape[1], dtype=v_np.dtype)
        _total_variants = self.shape[1]
        _completed_variants = 0
        _last_log_variants = 0
        _log_interval = max(_total_variants // 10, 1)
        for batch in self.iter_column_batches(batch_size=batch_size):
            output[batch.variant_indices] = np.asarray(batch.values, dtype=output.dtype).T @ v_np
            _completed_variants += len(batch.variant_indices)
            if _completed_variants - _last_log_variants >= _log_interval:
                _last_log_variants = _completed_variants
                log(f"        transpose_matvec: {_completed_variants:,}/{_total_variants:,} ({100*_completed_variants/_total_variants:.0f}%)  mem={mem()}")
        return output

    def transpose_matvec_jax(self, vector: NDArray | JaxArray, batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE) -> JaxArray:
        if self.supports_jax_dense_ops():
            vector_jax = _normalize_jax_vector_operand(
                vector,
                expected_length=self.shape[0],
                shape_error="sample vector must match genotype row count.",
                finite_name="sample vector",
            )
            return self._ensure_jax_cache().T @ vector_jax
        return _as_gpu_compute_jax(self.transpose_matvec_numpy(vector, batch_size=batch_size))

    def transpose_matvec(self, vector: NDArray | JaxArray, batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE) -> JaxArray:
        """Compute X_std^T @ v with JAX as the public return type."""
        return self.transpose_matvec_jax(vector, batch_size=batch_size)

    def transpose_matmat_numpy(self, matrix: NDArray | JaxArray, batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE) -> NDArray:
        m_np = _normalize_numpy_matrix_operand(
            matrix,
            expected_rows=self.shape[0],
            shape_error="sample matrix must match genotype row count.",
            finite_name="sample matrix",
        )
        if not np.any(m_np):
            return np.zeros((self.shape[1], m_np.shape[1]), dtype=m_np.dtype)
        if self._cupy_cache is not None:
            cupy = _try_import_cupy()
            if cupy is None:
                if _cupy_cache_is_int8_standardized(self._cupy_cache):
                    raise RuntimeError("CuPy int8 cache requires CuPy runtime.")
                return np.asarray(np.asarray(self._cupy_cache, dtype=m_np.dtype).T @ m_np, dtype=m_np.dtype)
            return _cupy_to_numpy(
                self._gpu_transpose_matmul(
                    m_np,
                    batch_size=batch_size,
                    cupy=cupy,
                    dtype=_cupy_compute_dtype(cupy),
                ),
                dtype=m_np.dtype,
            )
        cupy = _try_import_cupy()
        streaming_dtype = None if cupy is None else _cupy_compute_dtype(cupy)
        cupy, streaming_batch_size = self._streaming_gpu_context(
            batch_size,
            cupy=cupy,
            dtype=streaming_dtype,
        )
        if cupy is not None and streaming_batch_size is not None:
            return _cupy_to_numpy(
                self._gpu_transpose_matmul(
                    m_np,
                    batch_size=streaming_batch_size,
                    cupy=cupy,
                    dtype=streaming_dtype,
                ),
                dtype=m_np.dtype,
            )
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
                output[dense_parent_local_indices, :] = self._dense_backend.transpose_matmat_numpy(m_np, batch_size=batch_size)
            return output
        output = np.empty((self.shape[1], m_np.shape[1]), dtype=m_np.dtype)
        for batch in self.iter_column_batches(batch_size=batch_size):
            output[batch.variant_indices, :] = np.asarray(batch.values, dtype=output.dtype).T @ m_np
        return output

    def transpose_matmat_jax(self, matrix: NDArray | JaxArray, batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE) -> JaxArray:
        if self.supports_jax_dense_ops():
            matrix_jax = _normalize_jax_matrix_operand(
                matrix,
                expected_rows=self.shape[0],
                shape_error="sample matrix must match genotype row count.",
                finite_name="sample matrix",
            )
            return self._ensure_jax_cache().T @ matrix_jax
        return _as_gpu_compute_jax(self.transpose_matmat_numpy(matrix, batch_size=batch_size))

    def transpose_matmat(self, matrix: NDArray | JaxArray, batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE) -> JaxArray:
        """Compute X_std^T @ M with JAX as the public return type."""
        return self.transpose_matmat_jax(matrix, batch_size=batch_size)


def _resolve_variant_indices(
    variant_count: int,
    variant_indices: Sequence[int] | NDArray | None,
) -> NDArray:
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
    """Pick an int8-native batch size that fits the int8 decode budget."""
    if sample_count < 1:
        return DEFAULT_GENOTYPE_BATCH_SIZE
    bytes_per_variant = sample_count  # int8
    memory_capped = max(PLINK_INT8_TARGET_BATCH_BYTES // max(bytes_per_variant, 1), 1)
    return max(
        MIN_BED_READER_BATCH_SIZE,
        min(int(memory_capped), 16 * DEFAULT_GENOTYPE_BATCH_SIZE),
    )


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
def _contiguous_index_or_slice(indices: NDArray) -> slice | NDArray:
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


def _read_int8_columns_one_shot(
    raw: RawGenotypeMatrix | Int8BatchCapable,
    variant_indices: NDArray,
) -> NDArray | None:
    resolved_indices = np.asarray(variant_indices, dtype=np.int32)
    if resolved_indices.ndim != 1:
        raise ValueError("variant_indices must be 1D.")
    if resolved_indices.size == 0:
        return np.empty((raw.shape[0], 0), dtype=np.int8, order="F")
    if isinstance(raw, Int8RawGenotypeMatrix):
        column_index = _contiguous_index_or_slice(resolved_indices)
        return np.asfortranarray(raw.matrix[:, column_index], dtype=np.int8)
    if isinstance(raw, PlinkRawGenotypeMatrix):
        reader = raw._bed_reader()
        return np.asfortranarray(raw._read_batch_i8(reader, resolved_indices), dtype=np.int8)
    if isinstance(raw, IndexedRawGenotypeMatrix) and _supports_int8_batches(raw.child):
        child_block = _read_int8_columns_one_shot(
            raw.child,
            raw._child_columns(resolved_indices),
        )
        return None if child_block is None else np.asfortranarray(child_block, dtype=np.int8)
    if isinstance(raw, RowSubsetRawGenotypeMatrix) and _supports_int8_batches(raw.child):
        child_sample_count = max(int(raw.child.shape[0]), 1)
        subset_sample_count = max(int(raw.shape[0]), 1)
        if child_sample_count > int(subset_sample_count * ROW_SUBSET_ONE_SHOT_MAX_SAMPLE_RATIO):
            return None
        child_block = _read_int8_columns_one_shot(
            raw.child,
            resolved_indices,
        )
        if child_block is None:
            return None
        return np.asfortranarray(child_block[raw.row_indices, :], dtype=np.int8)
    if isinstance(raw, ConcatenatedRawGenotypeMatrix) and all(_supports_int8_batches(child) for child in raw.children):
        child_ids = np.searchsorted(raw._variant_offsets[1:], resolved_indices, side="right")
        result = np.empty((raw.shape[0], resolved_indices.shape[0]), dtype=np.int8, order="F")
        for child_index in np.unique(child_ids):
            child_positions = np.nonzero(child_ids == child_index)[0]
            child_variant_indices = resolved_indices[child_positions] - int(raw._variant_offsets[child_index])
            child_block = _read_int8_columns_one_shot(
                cast(Int8BatchCapable, raw.children[int(child_index)]),
                child_variant_indices,
            )
            if child_block is None:
                return None
            result[:, child_positions] = child_block
        return result
    return None


def _standardize_batch(batch: NDArray, means: NDArray, scales: NDArray) -> NDArray:
    batch_f32 = np.asarray(batch, dtype=np.float32)
    means_f32 = np.asarray(means, dtype=np.float32)
    scales_f32 = np.asarray(scales, dtype=np.float32)
    standardized = batch_f32 - means_f32[None, :]
    standardized[np.isnan(batch_f32)] = 0.0
    standardized /= scales_f32[None, :]
    return standardized.astype(np.float32, copy=False)


def _int8_batch_to_float32(batch: NDArray) -> NDArray:
    batch_i8 = np.asarray(batch, dtype=np.int8)
    batch_f32 = batch_i8.astype(np.float32)
    batch_f32[batch_i8 == PLINK_MISSING_INT8] = np.nan
    return batch_f32


def _standardize_batch_i8(batch: NDArray, means: NDArray, scales: NDArray) -> NDArray:
    batch_i8 = np.asarray(batch, dtype=np.int8)
    means_f32 = np.asarray(means, dtype=np.float32)
    scales_f32 = np.asarray(scales, dtype=np.float32)
    standardized = batch_i8.astype(np.float32)
    missing_mask = batch_i8 == PLINK_MISSING_INT8
    standardized -= means_f32[None, :]
    standardized[missing_mask] = 0.0
    standardized /= scales_f32[None, :]
    return standardized.astype(np.float32, copy=False)

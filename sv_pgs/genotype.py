from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import sys
from typing import Iterator, Sequence

import sv_pgs._jax  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np
from bed_reader import open_bed
from sv_pgs.progress import gpu_memory_snapshot, jax_runtime_snapshot, log, mem, nvidia_smi_snapshot

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

# Keep individual dense GPU kernels small enough that XLA/CUDA compilation is
# stable even for ~100k-sample matrices.  The chunk width is chosen
# dynamically from these caps.
_GPU_MATMUL_TARGET_ELEMENTS = 32_000_000
_GPU_MATMUL_MIN_CHUNK = 16
_GPU_MATMUL_MAX_CHUNK = 512
_GPU_MATMAT_MAX_RHS_COLUMNS = 8
_GPU_PROBE_TIMEOUT_SECONDS = 45.0
_GPU_DIAGNOSTIC_LOG_LIMIT = 4


def _floor_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value.bit_length() - 1)


def _power_of_two_candidates(maximum_chunk: int) -> list[int]:
    candidates: list[int] = []
    current = _floor_power_of_two(max(maximum_chunk, _GPU_MATMUL_MIN_CHUNK))
    while current >= _GPU_MATMUL_MIN_CHUNK:
        candidates.append(current)
        current //= 2
    if _GPU_MATMUL_MIN_CHUNK not in candidates:
        candidates.append(_GPU_MATMUL_MIN_CHUNK)
    return candidates


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

    def _bed_reader(self) -> open_bed:
        if self._reader is None:
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
    _gpu_chunk_override: int | None = field(init=False, default=None, repr=False)
    _gpu_op_counts: dict[str, int] = field(init=False, default_factory=dict, repr=False)

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

    def _gpu_chunk_description(self, rhs_columns: int = 1) -> str:
        auto_chunk = max(
            _GPU_MATMUL_MIN_CHUNK,
            min(_GPU_MATMUL_MAX_CHUNK, _floor_power_of_two(max(_GPU_MATMUL_TARGET_ELEMENTS // max(self.shape[0] * max(int(rhs_columns), 1), 1), 1))),
        )
        chunk = self._gpu_column_chunk(rhs_columns=rhs_columns)
        return (
            f"chunk={chunk} auto_chunk={auto_chunk} override={self._gpu_chunk_override} "
            f"shape={self.shape} rhs_columns={rhs_columns}"
        )

    def _log_gpu_operation(self, operation: str, detail: str, *, limit: int = _GPU_DIAGNOSTIC_LOG_LIMIT) -> None:
        current_count = self._gpu_op_counts.get(operation, 0) + 1
        self._gpu_op_counts[operation] = current_count
        if current_count > limit:
            return
        log(f"    GPU op {operation} call={current_count}: {detail}")
        log(f"    GPU op {operation} device snapshot: {gpu_memory_snapshot()}")
        if current_count == 1:
            log(f"    GPU op {operation} nvidia-smi: {nvidia_smi_snapshot()}")

    def _probe_safe_gpu_chunk(self, candidate_chunk: int) -> tuple[bool, str]:
        probe_code = """
import sys
import jax
import jax.numpy as jnp
n_samples = int(sys.argv[1])
chunk = int(sys.argv[2])
matrix = jnp.zeros((n_samples, chunk), dtype=jnp.float32)
vector = jnp.ones((chunk,), dtype=jnp.float32)
result = matrix @ vector
result.block_until_ready()
print(f"backend={jax.default_backend()} devices={jax.devices()} shape={(n_samples, chunk)}")
"""
        try:
            completed = subprocess.run(
                [sys.executable, "-c", probe_code, str(self.shape[0]), str(candidate_chunk)],
                capture_output=True,
                text=True,
                timeout=_GPU_PROBE_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            return False, f"timeout after {error.timeout}s"
        except Exception as error:
            return False, f"probe_error={error}"
        stdout = completed.stdout.strip().replace("\n", " | ")
        stderr = completed.stderr.strip().replace("\n", " | ")
        if completed.returncode == 0:
            return True, stdout or "probe_ok"
        if completed.returncode < 0:
            return False, f"signal={-completed.returncode} stdout={stdout} stderr={stderr}"
        return False, f"rc={completed.returncode} stdout={stdout} stderr={stderr}"

    def try_materialize_gpu(self) -> bool:
        """Materialize the standardized matrix into RAM and optionally onto GPU.

        GPU GEMM is disabled by default because JAX/XLA has known segfault bugs
        on Turing GPUs (T4) during matmul execution (jax-ml/jax#17349).  Set
        SV_PGS_GPU_MATMUL=1 to force GPU matmul if you know your GPU works.
        """
        if self._gpu_cache is not None:
            return True
        import os
        gpu_matmul_enabled = os.environ.get("SV_PGS_GPU_MATMUL", "0") == "1"
        if self._dense_cache is None:
            nbytes = self.dense_bytes()
            log(f"    materializing {self.shape[1]} variants x {self.shape[0]} samples ({nbytes / 1e9:.1f} GB) into RAM  mem={mem()}")
            self._dense_cache = np.empty(self.shape, dtype=np.float32)
            for batch in self.iter_column_batches(batch_size=auto_batch_size(self.shape[0])):
                self._dense_cache[:, batch.variant_indices] = batch.values
            log(f"    RAM-resident matrix ready  mem={mem()}")
        if not gpu_matmul_enabled:
            log("    GPU GEMM disabled (T4 segfault workaround; set SV_PGS_GPU_MATMUL=1 to override)")
            return True  # materialized in RAM, skip GPU upload
        try:
            devices = jax.devices()
            if not devices or devices[0].platform == "cpu":
                log("    no GPU backend detected; leaving matrix in RAM")
                return False
            log(f"    GPU materialization runtime: {jax_runtime_snapshot()}")
            log(f"    GPU materialization pre-upload: {gpu_memory_snapshot()}")
            log(f"    GPU materialization nvidia-smi: {nvidia_smi_snapshot()}")
            test_slice = jnp.asarray(self._dense_cache[:64, :64], dtype=jnp.float32)
            test_vec = jnp.ones(64, dtype=jnp.float32)
            test_result = test_slice @ test_vec
            test_result.block_until_ready()
            del test_slice, test_vec, test_result
            auto_chunk = self._gpu_column_chunk(rhs_columns=1)
            probe_candidates = _power_of_two_candidates(auto_chunk)
            log(
                "    GPU matmul probe candidates: "
                + ", ".join(str(candidate) for candidate in probe_candidates)
                + f"  sample_count={self.shape[0]}"
            )
            for candidate in probe_candidates:
                probe_ok, probe_detail = self._probe_safe_gpu_chunk(candidate)
                log(f"    GPU matmul probe chunk={candidate}: {'PASS' if probe_ok else 'FAIL'}  {probe_detail}")
                if probe_ok:
                    self._gpu_chunk_override = candidate
                    break
            if self._gpu_chunk_override is None:
                log("    GPU probe found no safe dense matmul chunk; using auto chunk for live diagnostics")
            else:
                log(f"    GPU probe selected safe chunk={self._gpu_chunk_override}  auto_chunk={auto_chunk}")
            log(
                f"    uploading full matrix to GPU  {self._gpu_chunk_description(rhs_columns=1)}  mem={mem()}"
            )
            self._gpu_cache = jnp.asarray(self._dense_cache, dtype=jnp.float32)
            self._gpu_cache.block_until_ready()
            log(f"    GPU-resident matrix ready ({self._gpu_cache.dtype})  mem={mem()}")
            log(f"    GPU materialization post-upload: {gpu_memory_snapshot()}")
            log(f"    GPU materialization post-upload nvidia-smi: {nvidia_smi_snapshot()}")
            return True
        except Exception as error:
            log(f"    GPU upload failed ({error}), leaving matrix in RAM  mem={mem()}")
            log(f"    GPU materialization failure snapshot: {gpu_memory_snapshot()}")
            log(f"    GPU materialization failure nvidia-smi: {nvidia_smi_snapshot()}")
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
            subset._gpu_chunk_override = self._gpu_chunk_override
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

    def _gpu_column_chunk(self, rhs_columns: int = 1) -> int:
        safe_rhs_columns = max(int(rhs_columns), 1)
        target_chunk = max(
            _GPU_MATMUL_TARGET_ELEMENTS // max(self.shape[0] * safe_rhs_columns, 1),
            1,
        )
        chunk = _floor_power_of_two(target_chunk)
        bounded_chunk = max(_GPU_MATMUL_MIN_CHUNK, min(_GPU_MATMUL_MAX_CHUNK, chunk, self.shape[1]))
        if self._gpu_chunk_override is not None:
            bounded_chunk = max(_GPU_MATMUL_MIN_CHUNK, min(bounded_chunk, self._gpu_chunk_override, self.shape[1]))
        return bounded_chunk

    def _gpu_rhs_chunk(self, rhs_columns: int) -> int:
        return max(1, min(int(rhs_columns), _GPU_MATMAT_MAX_RHS_COLUMNS))

    def _gpu_chunked_matvec(self, coeff: jnp.ndarray) -> jnp.ndarray:
        """GPU matmul in bounded chunks to avoid large-shape XLA crashes."""
        chunk = self._gpu_column_chunk(rhs_columns=1)
        self._log_gpu_operation(
            "matvec",
            f"{self._gpu_chunk_description(rhs_columns=1)} coeff_shape={tuple(coeff.shape)} coeff_dtype={coeff.dtype}",
        )
        result = jnp.zeros(self.shape[0], dtype=jnp.float32)
        for start in range(0, self.shape[1], chunk):
            end = min(start + chunk, self.shape[1])
            self._log_gpu_operation(
                "matvec_chunk",
                f"range=[{start}:{end}) matrix_shape=({self.shape[0]}, {end - start}) vector_shape=({end - start},)",
            )
            chunk_term = self._gpu_cache[:, start:end] @ coeff[start:end]
            result = result + chunk_term
        return result

    def _gpu_chunked_transpose_matvec(self, v: jnp.ndarray) -> jnp.ndarray:
        """GPU transpose matmul in bounded chunks."""
        chunk = self._gpu_column_chunk(rhs_columns=1)
        self._log_gpu_operation(
            "transpose_matvec",
            f"{self._gpu_chunk_description(rhs_columns=1)} vector_shape={tuple(v.shape)} vector_dtype={v.dtype}",
        )
        parts = []
        for start in range(0, self.shape[1], chunk):
            end = min(start + chunk, self.shape[1])
            self._log_gpu_operation(
                "transpose_matvec_chunk",
                f"range=[{start}:{end}) matrix_shape=({self.shape[0]}, {end - start}) vector_shape=({self.shape[0]},)",
            )
            parts.append(self._gpu_cache[:, start:end].T @ v)
        return jnp.concatenate(parts)

    def _gpu_chunked_matmat(self, matrix: jnp.ndarray) -> jnp.ndarray:
        """GPU matrix-matrix multiply with chunking over both variants and RHS."""
        variant_chunk = self._gpu_column_chunk(rhs_columns=matrix.shape[1])
        rhs_chunk = self._gpu_rhs_chunk(matrix.shape[1])
        self._log_gpu_operation(
            "matmat",
            f"{self._gpu_chunk_description(rhs_columns=matrix.shape[1])} rhs_shape={tuple(matrix.shape)} rhs_chunk={rhs_chunk}",
        )
        result_blocks = []
        for rhs_start in range(0, matrix.shape[1], rhs_chunk):
            rhs_end = min(rhs_start + rhs_chunk, matrix.shape[1])
            rhs_block = matrix[:, rhs_start:rhs_end]
            rhs_result = jnp.zeros((self.shape[0], rhs_end - rhs_start), dtype=jnp.float32)
            for start in range(0, self.shape[1], variant_chunk):
                end = min(start + variant_chunk, self.shape[1])
                self._log_gpu_operation(
                    "matmat_chunk",
                    f"col_range=[{start}:{end}) rhs_range=[{rhs_start}:{rhs_end}) matrix_shape=({self.shape[0]}, {end - start}) rhs_block_shape=({end - start}, {rhs_end - rhs_start})",
                )
                chunk_term = self._gpu_cache[:, start:end] @ rhs_block[start:end, :]
                rhs_result = rhs_result + chunk_term
            result_blocks.append(rhs_result)
        return jnp.concatenate(result_blocks, axis=1)

    def matvec(self, coefficients: np.ndarray | jnp.ndarray, batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE) -> jnp.ndarray:
        """Compute X_std @ beta (genotype matrix times coefficient vector).

        When GPU-cached: chunked GPU matmul. Otherwise: batched numpy on CPU.
        """
        if self._gpu_cache is not None:
            coeff = jnp.asarray(coefficients, dtype=jnp.float32)
            if coeff.ndim != 1 or coeff.shape[0] != self.shape[1]:
                raise ValueError("coefficient vector must match genotype column count.")
            return self._gpu_chunked_matvec(coeff).astype(jnp.float64)
        coeff_np = np.asarray(coefficients, dtype=np.float64).ravel()
        if coeff_np.shape[0] != self.shape[1]:
            raise ValueError("coefficient vector must match genotype column count.")
        result = np.zeros(self.shape[0], dtype=np.float64)
        for batch in self.iter_column_batches(batch_size=batch_size):
            batch_f64 = np.asarray(batch.values, dtype=np.float64)
            result += batch_f64 @ coeff_np[batch.variant_indices]
        return jnp.asarray(result, dtype=jnp.float64)

    def matmat(self, matrix: np.ndarray | jnp.ndarray, batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE) -> jnp.ndarray:
        """Compute X_std @ M. When GPU-cached: chunked GPU matmul. Otherwise: batched numpy."""
        if self._gpu_cache is not None:
            m = jnp.asarray(matrix, dtype=jnp.float32)
            if m.ndim != 2 or m.shape[0] != self.shape[1]:
                raise ValueError("variant matrix must match genotype column count.")
            return self._gpu_chunked_matmat(m).astype(jnp.float64)
        m_np = np.asarray(matrix, dtype=np.float64)
        if m_np.ndim != 2 or m_np.shape[0] != self.shape[1]:
            raise ValueError("variant matrix must match genotype column count.")
        output = np.zeros((self.shape[0], m_np.shape[1]), dtype=np.float64)
        for batch in self.iter_column_batches(batch_size=batch_size):
            batch_f64 = np.asarray(batch.values, dtype=np.float64)
            output += batch_f64 @ m_np[batch.variant_indices, :]
        return jnp.asarray(output, dtype=jnp.float64)

    def transpose_matvec(self, vector: np.ndarray | jnp.ndarray, batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE) -> jnp.ndarray:
        """Compute X_std^T @ v. When GPU-cached: chunked GPU matmul. Otherwise: batched numpy."""
        if self._gpu_cache is not None:
            v = jnp.asarray(vector, dtype=jnp.float32)
            if v.ndim != 1 or v.shape[0] != self.shape[0]:
                raise ValueError("sample vector must match genotype row count.")
            return self._gpu_chunked_transpose_matvec(v).astype(jnp.float64)
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
            chunk = self._gpu_column_chunk(rhs_columns=m.shape[1])
            rhs_chunk = self._gpu_rhs_chunk(m.shape[1])
            self._log_gpu_operation(
                "transpose_matmat",
                f"{self._gpu_chunk_description(rhs_columns=m.shape[1])} matrix_shape={tuple(m.shape)} rhs_chunk={rhs_chunk}",
            )
            parts = []
            for start in range(0, self.shape[1], chunk):
                end = min(start + chunk, self.shape[1])
                rhs_blocks = []
                for rhs_start in range(0, m.shape[1], rhs_chunk):
                    rhs_end = min(rhs_start + rhs_chunk, m.shape[1])
                    self._log_gpu_operation(
                        "transpose_matmat_chunk",
                        f"col_range=[{start}:{end}) rhs_range=[{rhs_start}:{rhs_end}) matrix_shape=({self.shape[0]}, {end - start}) sample_block_shape=({self.shape[0]}, {rhs_end - rhs_start})",
                    )
                    rhs_blocks.append(self._gpu_cache[:, start:end].T @ m[:, rhs_start:rhs_end])
                parts.append(jnp.concatenate(rhs_blocks, axis=1))
            return jnp.concatenate(parts).astype(jnp.float64)
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

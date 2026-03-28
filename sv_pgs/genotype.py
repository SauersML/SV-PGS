from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import sv_pgs._jax  # noqa: F401
import jax.numpy as jnp
import numpy as np
from bed_reader import open_bed

DEFAULT_GENOTYPE_BATCH_SIZE = 4096


def as_raw_genotype_matrix(genotypes: RawGenotypeMatrix | np.ndarray) -> RawGenotypeMatrix:
    if isinstance(genotypes, RawGenotypeMatrix):
        return genotypes
    return DenseRawGenotypeMatrix(np.asarray(genotypes, dtype=np.float32))


@dataclass(slots=True)
class RawGenotypeBatch:
    variant_indices: np.ndarray
    values: np.ndarray


class RawGenotypeMatrix(ABC):
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
        array = np.asarray(self.matrix, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError("genotypes must be 2D.")
        self.matrix = array

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
                values=np.asarray(self.matrix[:, batch_indices], dtype=np.float32),
            )

    def materialize(
        self,
        variant_indices: Sequence[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        resolved_indices = _resolve_variant_indices(self.shape[1], variant_indices)
        return np.asarray(self.matrix[:, resolved_indices], dtype=np.float32)


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

    def iter_column_batches(
        self,
        variant_indices: Sequence[int] | np.ndarray | None = None,
        batch_size: int | None = None,
    ) -> Iterator[RawGenotypeBatch]:
        resolved_indices = _resolve_variant_indices(self.variant_count, variant_indices)
        safe_batch_size = max(int(self.batch_size if batch_size is None else batch_size), 1)
        reader = self._bed_reader()
        total = resolved_indices.shape[0]
        for start_index in range(0, total, safe_batch_size):
            batch_indices = resolved_indices[start_index : start_index + safe_batch_size]
            col_index = np.ascontiguousarray(batch_indices, dtype=np.intp)
            values = np.asarray(
                reader.read(index=(self.sample_indices, col_index), dtype="float32", num_threads=4),
                dtype=np.float32,
            )
            yield RawGenotypeBatch(
                variant_indices=batch_indices,
                values=values,
            )

    def materialize(
        self,
        variant_indices: Sequence[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        resolved_indices = _resolve_variant_indices(self.variant_count, variant_indices)
        reader = self._bed_reader()
        col_index = np.ascontiguousarray(resolved_indices, dtype=np.intp)
        return np.asarray(
            reader.read(index=(self.sample_indices, col_index), dtype="float32", num_threads=4),
            dtype=np.float32,
        )

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
                num_threads=4,
            )
            log(f"    bed_reader opened  mem={mem()}")
        return self._reader


@dataclass(slots=True)
class StandardizedGenotypeMatrix:
    raw: RawGenotypeMatrix
    means: np.ndarray
    scales: np.ndarray
    variant_indices: np.ndarray
    _dense_cache: np.ndarray | None = field(init=False, default=None, repr=False)

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

    def subset(self, local_variant_indices: Sequence[int] | np.ndarray) -> StandardizedGenotypeMatrix:
        resolved_local_indices = np.asarray(local_variant_indices, dtype=np.int32)
        subset = StandardizedGenotypeMatrix(
            raw=self.raw,
            means=self.means,
            scales=self.scales,
            variant_indices=self.variant_indices[resolved_local_indices],
        )
        if self._dense_cache is not None:
            subset._dense_cache = np.asarray(self._dense_cache[:, resolved_local_indices], dtype=np.float32)
        return subset

    def iter_column_batches(
        self,
        batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE,
    ) -> Iterator[RawGenotypeBatch]:
        if isinstance(self.raw, DenseRawGenotypeMatrix):
            dense_matrix = self.materialize()
            safe_batch_size = max(int(batch_size), 1)
            for start_index in range(0, dense_matrix.shape[1], safe_batch_size):
                local_indices = np.arange(start_index, min(start_index + safe_batch_size, dense_matrix.shape[1]), dtype=np.int32)
                yield RawGenotypeBatch(
                    variant_indices=local_indices,
                    values=np.asarray(dense_matrix[:, local_indices], dtype=np.float32),
                )
            return
        safe_batch_size = max(int(batch_size), 1)
        for start_index in range(0, self.variant_indices.shape[0], safe_batch_size):
            local_indices = np.arange(
                start_index,
                min(start_index + safe_batch_size, self.variant_indices.shape[0]),
                dtype=np.int32,
            )
            raw_indices = self.variant_indices[local_indices]
            raw_batch = self.raw.materialize(raw_indices)
            yield RawGenotypeBatch(
                variant_indices=local_indices,
                values=_standardize_batch(raw_batch, self.means[raw_indices], self.scales[raw_indices]),
            )

    def materialize(self, batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE) -> np.ndarray:
        if self._dense_cache is not None:
            return self._dense_cache
        if isinstance(self.raw, DenseRawGenotypeMatrix):
            raw_values = np.asarray(self.raw.matrix[:, self.variant_indices], dtype=np.float32)
            self._dense_cache = _standardize_batch(raw_values, self.means[self.variant_indices], self.scales[self.variant_indices])
            return self._dense_cache
        matrix = np.empty(self.shape, dtype=np.float32)
        for batch in self.iter_column_batches(batch_size=batch_size):
            matrix[:, batch.variant_indices] = batch.values
        return matrix

    def matvec(self, coefficients: np.ndarray | jnp.ndarray, batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE) -> jnp.ndarray:
        coefficient_vector = jnp.asarray(coefficients, dtype=jnp.float64)
        if coefficient_vector.ndim != 1 or coefficient_vector.shape[0] != self.shape[1]:
            raise ValueError("coefficient vector must match genotype column count.")
        if isinstance(self.raw, DenseRawGenotypeMatrix):
            return jnp.asarray(self.materialize(), dtype=jnp.float64) @ coefficient_vector
        result = jnp.zeros(self.shape[0], dtype=jnp.float64)
        for batch in self.iter_column_batches(batch_size=batch_size):
            batch_values = jnp.asarray(batch.values, dtype=jnp.float64)
            batch_coefficients = coefficient_vector[batch.variant_indices]
            result = result + batch_values @ batch_coefficients
        return result

    def transpose_matvec(self, vector: np.ndarray | jnp.ndarray, batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE) -> jnp.ndarray:
        sample_vector = jnp.asarray(vector, dtype=jnp.float64)
        if sample_vector.ndim != 1 or sample_vector.shape[0] != self.shape[0]:
            raise ValueError("sample vector must match genotype row count.")
        if isinstance(self.raw, DenseRawGenotypeMatrix):
            return jnp.asarray(self.materialize(), dtype=jnp.float64).T @ sample_vector
        outputs: list[jnp.ndarray] = []
        for batch in self.iter_column_batches(batch_size=batch_size):
            batch_values = jnp.asarray(batch.values, dtype=jnp.float64)
            outputs.append(batch_values.T @ sample_vector)
        if not outputs:
            return jnp.zeros(self.shape[1], dtype=jnp.float64)
        return jnp.concatenate(outputs, axis=0)

    def transpose_matmat(self, matrix: np.ndarray | jnp.ndarray, batch_size: int = DEFAULT_GENOTYPE_BATCH_SIZE) -> jnp.ndarray:
        sample_matrix = jnp.asarray(matrix, dtype=jnp.float64)
        if sample_matrix.ndim != 2 or sample_matrix.shape[0] != self.shape[0]:
            raise ValueError("sample matrix must match genotype row count.")
        if isinstance(self.raw, DenseRawGenotypeMatrix):
            return jnp.asarray(self.materialize(), dtype=jnp.float64).T @ sample_matrix
        outputs: list[jnp.ndarray] = []
        for batch in self.iter_column_batches(batch_size=batch_size):
            batch_values = jnp.asarray(batch.values, dtype=jnp.float64)
            outputs.append(batch_values.T @ sample_matrix)
        if not outputs:
            return jnp.zeros((self.shape[1], sample_matrix.shape[1]), dtype=jnp.float64)
        return jnp.concatenate(outputs, axis=0)


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


def _standardize_batch(batch: np.ndarray, means: np.ndarray, scales: np.ndarray) -> np.ndarray:
    mean_vector = np.asarray(means, dtype=np.float32)
    scale_vector = np.asarray(scales, dtype=np.float32)
    imputed = np.where(np.isnan(batch), mean_vector[None, :], batch)
    standardized = (imputed - mean_vector[None, :]) / scale_vector[None, :]
    return np.asarray(standardized, dtype=np.float32)

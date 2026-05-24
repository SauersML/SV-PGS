from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence, cast

import numpy as np

from sv_pgs._typing import F32Array, I32Array, NDArray

# Importing sv_pgs._jax has the side-effect of configuring XLA env vars
# before any other module imports jax. It must come before sv_pgs.artifact,
# sv_pgs.genotype, sv_pgs.inference, etc., which transitively import jax.
# Sorted alphabetically it naturally lands first among sv_pgs.* imports.
from sv_pgs import _jax as _jax_side_effects
from sv_pgs.artifact import ModelArtifact, load_artifact, save_artifact
from sv_pgs.config import ModelConfig, TraitType
from sv_pgs.data import PreparedArrays, TieGroup, TieMap, VariantRecord, VariantStatistics, normalize_variant_record
from sv_pgs.genotype import (
    ConcatenatedRawGenotypeMatrix,
    DenseRawGenotypeMatrix,
    IndexedRawGenotypeMatrix,
    Int8RawGenotypeMatrix,
    PlinkRawGenotypeMatrix,
    RawGenotypeMatrix,
    RowSubsetRawGenotypeMatrix,
    StandardizedGenotypeMatrix,
    _standardize_batch,
    as_raw_genotype_matrix,
    auto_batch_size,
)
from sv_pgs.inference import VariationalFitCheckpoint, VariationalFitResult, fit_variational_em
from sv_pgs.numeric import stable_sigmoid
from sv_pgs.preprocessing import (
    Preprocessor,
    build_tie_map,
    compute_marginal_z_scores,
    compute_variant_statistics,
    fit_preprocessor_from_stats,
)
from sv_pgs.progress import log, mem
from sv_pgs.runtime_policy import runtime_training_policy_for_fit, runtime_training_policy_summary

del _jax_side_effects


def _release_gpu_memory_pools() -> None:
    """Free CuPy memory pools after a fit completes.

    Reduced genotype matrices and validation matrices may live on device
    during fitting. After we copy fit results back to host (numpy), the device
    buffers are unreferenced but pooled — repeated fits in one process would
    otherwise gradually starve the GPU. Guarded because cupy may not import in
    CPU-only environments.
    """
    try:
        import cupy as cp
    except (ImportError, OSError, RuntimeError):
        return
    try:
        cp.get_default_memory_pool().free_all_blocks()
    except (AttributeError, RuntimeError):
        pass
    try:
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except (AttributeError, RuntimeError):
        pass


def _select_active_variant_indices_fast(
    allele_frequencies: NDArray,
    minimum_minor_allele_frequency: float,
) -> I32Array:
    """Vectorized MAF filter — avoids creating 1.68M VariantRecord objects."""
    af = np.asarray(allele_frequencies, dtype=np.float32)
    maf = np.minimum(af, 1.0 - af)
    return np.flatnonzero(maf >= minimum_minor_allele_frequency).astype(np.int32)


@dataclass(slots=True)
class FittedState:
    variant_records: list[VariantRecord]
    full_variant_records: list[VariantRecord]
    active_variant_indices: I32Array
    preprocessor: Preprocessor
    tie_map: TieMap
    fit_result: VariationalFitResult
    full_coefficients: F32Array
    nonzero_coefficient_indices: I32Array
    nonzero_coefficients: F32Array
    nonzero_means: F32Array
    nonzero_scales: F32Array
    training_genetic_score: F32Array | None = None
    training_covariate_score: F32Array | None = None
    training_linear_predictor: F32Array | None = None

    def __post_init__(self) -> None:
        variant_count = len(self.full_variant_records)
        if self.full_coefficients.shape != (variant_count,):
            raise ValueError("full_coefficients must align with full_variant_records.")
        if self.preprocessor.means.shape != (variant_count,):
            raise ValueError("Preprocessor means must align with full_variant_records.")
        if self.preprocessor.scales.shape != (variant_count,):
            raise ValueError("Preprocessor scales must align with full_variant_records.")
        if self.tie_map.original_to_reduced.shape != (variant_count,):
            raise ValueError("tie_map must align with full_variant_records.")
        training_arrays = (
            self.training_genetic_score,
            self.training_covariate_score,
            self.training_linear_predictor,
        )
        provided_training_arrays = [array for array in training_arrays if array is not None]
        if provided_training_arrays and len(provided_training_arrays) != len(training_arrays):
            raise ValueError("Training score cache must provide genetic, covariate, and linear arrays together.")
        if self.training_linear_predictor is not None:
            sample_count = self.training_linear_predictor.shape[0]
            if self.training_linear_predictor.ndim != 1:
                raise ValueError("training_linear_predictor must be 1D.")
            if self.training_genetic_score is None or self.training_genetic_score.shape != (sample_count,):
                raise ValueError("training_genetic_score must align with training_linear_predictor.")
            if self.training_covariate_score is None or self.training_covariate_score.shape != (sample_count,):
                raise ValueError("training_covariate_score must align with training_linear_predictor.")


_FIT_STAGE_CACHE_DIRNAME = ".sv_pgs_cache"
_FIT_STAGE_CACHE_SUBDIR = "fit_stage"
_FIT_STAGE_CACHE_VERSION = 4
_FIT_CHECKPOINT_VERSION = 1
# Refuse to silently fall back to mmap streaming for binary TR-Newton fits when
# no GPU/RAM/local int8 cache is available. The streaming path is catastrophically
# slow inside Newton-CG and was responsible for the overfit run that motivated
# this guard. Flip to False (or set config.allow_mmap_streaming_for_binary=True)
# to override.
_REFUSE_BINARY_TR_NEWTON_NO_CACHE = False
_REGISTERED_FIT_CHECKPOINT_PATHS: dict[int, Path] = {}


def register_fit_checkpoint_path(config: ModelConfig, checkpoint_path: str | Path) -> None:
    _REGISTERED_FIT_CHECKPOINT_PATHS[id(config)] = Path(checkpoint_path)


def _pop_registered_fit_checkpoint_path(config: ModelConfig) -> Path | None:
    return _REGISTERED_FIT_CHECKPOINT_PATHS.pop(id(config), None)


@dataclass(slots=True)
class _FitStageCachePaths:
    key: str
    cache_dir: Path
    manifest_path: Path
    active_indices_path: Path
    tie_map_path: Path
    reduced_raw_i8_path: Path
    em_checkpoint_path: Path
    fit_key: str | None = None

    def __post_init__(self) -> None:
        if self.fit_key is None:
            self.fit_key = self.key


def _fit_stage_cache_dir() -> Path:
    return Path.cwd() / _FIT_STAGE_CACHE_DIRNAME / _FIT_STAGE_CACHE_SUBDIR


def _fit_stage_variant_stats_cache_path(
    genotype_matrix: RawGenotypeMatrix,
    config: ModelConfig,
) -> Path | None:
    raw_signature = _persistent_raw_signature(genotype_matrix)
    if raw_signature is None:
        return None
    stats_hasher = hashlib.sha256()
    stats_hasher.update(f"fit-stage-stats-v{_FIT_STAGE_CACHE_VERSION}".encode("utf-8"))
    stats_hasher.update(raw_signature.encode("utf-8"))
    stats_hasher.update(np.asarray([config.minimum_scale], dtype=np.float64).tobytes())
    stats_key = stats_hasher.hexdigest()[:24]
    return _fit_stage_cache_dir() / f"{stats_key}.variant_stats.npz"


def _validate_fit_inputs(
    genotype_matrix: RawGenotypeMatrix,
    covariates: NDArray,
    targets: NDArray,
    variant_records: Sequence[VariantRecord | dict[str, Any]],
    variant_stats: VariantStatistics | None,
) -> None:
    sample_count, variant_count = genotype_matrix.shape
    covariate_matrix = np.asarray(covariates)
    target_array = np.asarray(targets).reshape(-1)
    if covariate_matrix.ndim != 2:
        raise ValueError("covariates must be 2D.")
    if covariate_matrix.shape[0] != sample_count:
        raise ValueError("covariates sample count must match genotypes.")
    if not np.all(np.isfinite(covariate_matrix)):
        raise ValueError("covariates must be finite.")
    if target_array.shape[0] != sample_count:
        raise ValueError("targets sample count must match genotypes.")
    if not np.all(np.isfinite(target_array)):
        raise ValueError("targets must be finite.")
    if len(variant_records) != variant_count:
        raise ValueError("variant_records length must match genotype column count.")
    if variant_stats is None:
        return
    if variant_stats.means.shape != (variant_count,):
        raise ValueError("variant_stats.means must match genotype column count.")
    if variant_stats.scales.shape != (variant_count,):
        raise ValueError("variant_stats.scales must match genotype column count.")
    if variant_stats.allele_frequencies.shape != (variant_count,):
        raise ValueError("variant_stats.allele_frequencies must match genotype column count.")
    if variant_stats.support_counts.shape != (variant_count,):
        raise ValueError("variant_stats.support_counts must match genotype column count.")
    if not np.all(np.isfinite(variant_stats.means)):
        raise ValueError("variant_stats.means must be finite.")
    if not np.all(np.isfinite(variant_stats.scales)) or np.any(variant_stats.scales <= 0.0):
        raise ValueError("variant_stats.scales must be finite and positive.")
    if not np.all(np.isfinite(variant_stats.allele_frequencies)) or np.any(variant_stats.allele_frequencies < 0.0) or np.any(variant_stats.allele_frequencies > 1.0):
        raise ValueError("variant_stats.allele_frequencies must be finite and lie in [0.0, 1.0].")
    if np.any(np.asarray(variant_stats.support_counts) < 0):
        raise ValueError("variant_stats.support_counts must be non-negative.")

def _fit_stage_cache_paths(
    genotype_matrix: RawGenotypeMatrix,
    allele_frequencies: NDArray,
    means: NDArray,
    scales: NDArray,
    covariates: NDArray,
    targets: NDArray,
    config: ModelConfig,
) -> _FitStageCachePaths | None:
    raw_signature = _persistent_raw_signature(genotype_matrix)
    if raw_signature is None:
        return None
    structure_hasher = hashlib.sha256()
    structure_hasher.update(f"fit-stage-structure-v{_FIT_STAGE_CACHE_VERSION}".encode("utf-8"))
    structure_hasher.update(raw_signature.encode("utf-8"))
    structure_hasher.update(np.asarray([config.minimum_minor_allele_frequency], dtype=np.float64).tobytes())
    _update_hash_with_array_bytes(structure_hasher, np.asarray(allele_frequencies, dtype=np.float32))
    _update_hash_with_array_bytes(structure_hasher, np.asarray(means, dtype=np.float32))
    _update_hash_with_array_bytes(structure_hasher, np.asarray(scales, dtype=np.float32))
    marginal_screen_min_abs_z = float(config.marginal_screen_min_abs_z)
    structure_hasher.update(np.asarray([marginal_screen_min_abs_z], dtype=np.float64).tobytes())
    if marginal_screen_min_abs_z > 0.0:
        _update_hash_with_array_bytes(structure_hasher, np.asarray(covariates, dtype=np.float32))
        _update_hash_with_array_bytes(structure_hasher, np.asarray(targets, dtype=np.float32))
    structure_key = structure_hasher.hexdigest()[:24]
    fit_hasher = hashlib.sha256()
    fit_hasher.update(f"fit-stage-fit-v{_FIT_STAGE_CACHE_VERSION}".encode("utf-8"))
    fit_hasher.update(structure_key.encode("utf-8"))
    fit_hasher.update(config.trait_type.value.encode("utf-8"))
    _update_hash_with_array_bytes(fit_hasher, np.asarray(covariates, dtype=np.float32))
    _update_hash_with_array_bytes(fit_hasher, np.asarray(targets, dtype=np.float32))
    fit_key = fit_hasher.hexdigest()[:24]
    cache_dir = _fit_stage_cache_dir()
    return _FitStageCachePaths(
        key=structure_key,
        cache_dir=cache_dir,
        manifest_path=cache_dir / f"{structure_key}.manifest.json",
        active_indices_path=cache_dir / f"{structure_key}.active.npy",
        tie_map_path=cache_dir / f"{structure_key}.tie.pkl",
        reduced_raw_i8_path=cache_dir / f"{structure_key}.reduced_raw_i8.npy",
        em_checkpoint_path=cache_dir / f"{fit_key}.em.pkl",
        fit_key=fit_key,
    )


def _update_hash_with_array_bytes(hasher: "hashlib._Hash", array: NDArray) -> None:
    contiguous = np.ascontiguousarray(array)
    hasher.update(str(contiguous.dtype).encode("utf-8"))
    hasher.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    hasher.update(contiguous.view(np.uint8).tobytes())


def _persistent_raw_signature(genotype_matrix: RawGenotypeMatrix) -> str | None:
    if isinstance(genotype_matrix, PlinkRawGenotypeMatrix):
        stat = genotype_matrix.bed_path.stat()
        sample_hash = hashlib.sha256(np.asarray(genotype_matrix.sample_indices, dtype=np.int64).tobytes()).hexdigest()[:16]
        return (
            f"plink:{genotype_matrix.bed_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:"
            + f"{genotype_matrix.variant_count}:{sample_hash}"
        )
    if isinstance(genotype_matrix, ConcatenatedRawGenotypeMatrix):
        child_signatures = [_persistent_raw_signature(child) for child in genotype_matrix.children]
        if any(signature is None for signature in child_signatures):
            return None
        return "concat:" + "|".join(str(signature) for signature in child_signatures)
    if isinstance(genotype_matrix, RowSubsetRawGenotypeMatrix):
        child_signature = _persistent_raw_signature(genotype_matrix.child)
        if child_signature is None:
            return None
        row_hash = hashlib.sha256(
            np.asarray(genotype_matrix.row_indices, dtype=np.int64).tobytes()
        ).hexdigest()[:16]
        return f"rowsubset:{child_signature}:n={int(genotype_matrix.row_indices.shape[0])}:{row_hash}"
    if isinstance(genotype_matrix, IndexedRawGenotypeMatrix):
        child_signature = _persistent_raw_signature(genotype_matrix.child)
        if child_signature is None:
            return None
        col_hash = hashlib.sha256(
            np.asarray(genotype_matrix.selected_columns, dtype=np.int64).tobytes()
        ).hexdigest()[:16]
        return f"indexed:{child_signature}:m={int(genotype_matrix.selected_columns.shape[0])}:{col_hash}"
    if isinstance(genotype_matrix, (Int8RawGenotypeMatrix, DenseRawGenotypeMatrix)):
        backing_matrix = np.asanyarray(genotype_matrix.matrix)
        if isinstance(backing_matrix, np.memmap):
            backing_path = Path(str(backing_matrix.filename)).resolve()
            stat = backing_path.stat()
            path_parts = set(backing_path.parts)
            if _FIT_STAGE_CACHE_DIRNAME in path_parts:
                # Persisted genotype caches already use content-derived filenames.
                # Avoid invalidating downstream fit/tie caches when the memmap file
                # is rewritten or touched without any semantic data change.
                return f"memmap-cache:{backing_path}:{stat.st_size}:{backing_matrix.shape}:{backing_matrix.dtype}"
            return f"memmap:{backing_path}:{stat.st_size}:{backing_matrix.shape}:{backing_matrix.dtype}"
    return None


def _save_fit_stage_variant_stats_cache(
    cache_path: Path,
    variant_stats: VariantStatistics,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.parent / f"{cache_path.name}.tmp.npz"
    try:
        np.savez_compressed(
            tmp_path,
            means=np.asarray(variant_stats.means, dtype=np.float32),
            scales=np.asarray(variant_stats.scales, dtype=np.float32),
            allele_frequencies=np.asarray(variant_stats.allele_frequencies, dtype=np.float32),
            support_counts=np.asarray(variant_stats.support_counts, dtype=np.int32),
        )
        tmp_path.replace(cache_path)
        log(
            "fit-stage variant stats cache saved: "
            + f"{cache_path.parent.name}/{cache_path.name} "
            + f"({variant_stats.means.shape[0]} variants)"
        )
    except (OSError, ValueError) as exc:
        tmp_path.unlink(missing_ok=True)
        log(f"fit-stage variant stats cache save failed ({exc}); continuing without durable cohort stats")


def _try_load_fit_stage_variant_stats_cache(
    cache_path: Path,
    *,
    variant_count: int,
) -> VariantStatistics | None:
    if not cache_path.exists():
        return None
    try:
        cached_arrays = np.load(cache_path, allow_pickle=False)
        means = np.asarray(cached_arrays["means"], dtype=np.float32)
        scales = np.asarray(cached_arrays["scales"], dtype=np.float32)
        allele_frequencies = np.asarray(cached_arrays["allele_frequencies"], dtype=np.float32)
        support_counts = np.asarray(cached_arrays["support_counts"], dtype=np.int32)
        expected_shape = (int(variant_count),)
        if (
            means.shape != expected_shape
            or scales.shape != expected_shape
            or allele_frequencies.shape != expected_shape
            or support_counts.shape != expected_shape
        ):
            raise ValueError("cached variant stats shape mismatch")
        if not np.all(np.isfinite(means)):
            raise ValueError("cached variant means must be finite")
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
            raise ValueError("cached variant scales must be finite and positive")
        if (
            not np.all(np.isfinite(allele_frequencies))
            or np.any(allele_frequencies < 0.0)
            or np.any(allele_frequencies > 1.0)
        ):
            raise ValueError("cached allele frequencies must lie in [0.0, 1.0]")
        if np.any(support_counts < 0):
            raise ValueError("cached support counts must be non-negative")
        log(
            "fit-stage variant stats cache hit — loading from "
            + f"{cache_path.parent.name}/{cache_path.name}"
        )
        return VariantStatistics(
            means=means,
            scales=scales,
            allele_frequencies=allele_frequencies,
            support_counts=support_counts,
        )
    except (OSError, ValueError, EOFError, KeyError) as exc:
        log(f"fit-stage variant stats cache load failed ({exc}); recomputing cohort stats")
        cache_path.unlink(missing_ok=True)
        return None


def _save_fit_stage_structure_cache(
    cache_paths: _FitStageCachePaths,
    active_variant_indices: NDArray,
    reduced_tie_map: TieMap,
) -> None:
    cache_paths.cache_dir.mkdir(parents=True, exist_ok=True)
    active_tmp = cache_paths.cache_dir / f"{cache_paths.key}.active.tmp.npy"
    tie_tmp = cache_paths.cache_dir / f"{cache_paths.key}.tie.tmp.pkl"
    np.save(active_tmp, np.asarray(active_variant_indices, dtype=np.int32))
    with tie_tmp.open("wb") as handle:
        pickle.dump(reduced_tie_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
    active_tmp.replace(cache_paths.active_indices_path)
    tie_tmp.replace(cache_paths.tie_map_path)
    _write_fit_stage_cache_manifest(
        cache_paths=cache_paths,
        active_variant_count=int(np.asarray(active_variant_indices).shape[0]),
        reduced_variant_count=int(np.asarray(reduced_tie_map.kept_indices).shape[0]),
        has_reduced_raw_i8=cache_paths.reduced_raw_i8_path.exists(),
    )
    log(
        "fit-stage cache saved: "
        + f"{cache_paths.cache_dir.name}/{cache_paths.key}.* "
        + f"({int(np.asarray(active_variant_indices).shape[0])} active -> {int(np.asarray(reduced_tie_map.kept_indices).shape[0])} unique)"
    )


def _write_fit_stage_cache_manifest(
    cache_paths: _FitStageCachePaths,
    *,
    active_variant_count: int,
    reduced_variant_count: int,
    has_reduced_raw_i8: bool,
) -> None:
    manifest_tmp = cache_paths.cache_dir / f"{cache_paths.key}.manifest.tmp.json"
    manifest_payload = {
        "version": _FIT_STAGE_CACHE_VERSION,
        "active_variant_count": int(active_variant_count),
        "reduced_variant_count": int(reduced_variant_count),
        "has_reduced_raw_i8": bool(has_reduced_raw_i8),
    }
    manifest_tmp.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    manifest_tmp.replace(cache_paths.manifest_path)


def _sample_space_basis_cache_path(
    cache_paths: _FitStageCachePaths,
    *,
    rank: int,
    random_seed: int,
) -> Path:
    return cache_paths.cache_dir / f"{cache_paths.key}.sample_space_basis.r{int(rank)}.seed{int(random_seed)}.npy"


def _try_restore_sample_space_basis_cache(
    cache_paths: _FitStageCachePaths,
    genotype_matrix: StandardizedGenotypeMatrix,
    *,
    rank: int,
    random_seed: int,
) -> bool:
    basis_path = _sample_space_basis_cache_path(cache_paths, rank=rank, random_seed=random_seed)
    if not basis_path.exists():
        return False
    try:
        basis_matrix = np.load(basis_path, mmap_mode="r")
        basis_array = np.asarray(basis_matrix)
        if basis_array.dtype != np.float64:
            basis_array = basis_array.astype(np.float64, copy=False)
        if basis_array.ndim != 2 or basis_array.shape[0] != genotype_matrix.shape[0]:
            raise ValueError("sample-space basis cache shape mismatch.")
        genotype_matrix._sample_space_nystrom_basis_cpu_cache[(basis_array.shape[1], int(random_seed))] = basis_array
        log(
            "sample-space basis cache restored: "
            + f"{cache_paths.cache_dir.name}/{basis_path.name} "
            + f"(rank={basis_array.shape[1]})"
        )
        return True
    except (OSError, ValueError, EOFError) as exc:
        log(f"sample-space basis cache load failed ({exc}); rebuilding")
        basis_path.unlink(missing_ok=True)
        return False


def _save_sample_space_basis_cache(
    cache_paths: _FitStageCachePaths,
    genotype_matrix: StandardizedGenotypeMatrix,
    *,
    rank: int,
    random_seed: int,
) -> bool:
    basis_path = _sample_space_basis_cache_path(cache_paths, rank=rank, random_seed=random_seed)
    cached_basis = genotype_matrix._sample_space_nystrom_basis_cpu_cache.get((int(rank), int(random_seed)))
    if cached_basis is None:
        return False
    basis_array = np.asarray(cached_basis, dtype=np.float64)
    if basis_array.ndim != 2 or basis_array.shape[0] != genotype_matrix.shape[0]:
        return False
    cache_paths.cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_paths.cache_dir / f"{basis_path.name}.tmp"
    try:
        np.save(tmp_path, basis_array)
        tmp_npy = tmp_path if tmp_path.suffix == ".npy" else tmp_path.with_suffix(tmp_path.suffix + ".npy")
        tmp_npy.replace(basis_path)
        log(
            "sample-space basis cache saved: "
            + f"{cache_paths.cache_dir.name}/{basis_path.name} "
            + f"(rank={basis_array.shape[1]})"
        )
        return True
    except (OSError, ValueError) as exc:
        tmp_path.unlink(missing_ok=True)
        tmp_path.with_suffix(tmp_path.suffix + ".npy").unlink(missing_ok=True)
        log(f"sample-space basis cache save failed ({exc}); continuing without durable solver cache")
        return False


def _try_load_fit_stage_cache(
    cache_paths: _FitStageCachePaths,
    prepared_arrays: PreparedArrays,
    standardized_genotypes: StandardizedGenotypeMatrix,
) -> tuple[I32Array, TieMap, StandardizedGenotypeMatrix, bool] | None:
    if not cache_paths.manifest_path.exists():
        log(f"fit-stage cache miss (key={cache_paths.key})")
        return None
    if not cache_paths.active_indices_path.exists() or not cache_paths.tie_map_path.exists():
        log(f"fit-stage cache incomplete (key={cache_paths.key})")
        return None
    try:
        manifest_payload = json.loads(cache_paths.manifest_path.read_text(encoding="utf-8"))
        if int(manifest_payload.get("version", -1)) != _FIT_STAGE_CACHE_VERSION:
            log(f"fit-stage cache version mismatch (key={cache_paths.key}), rebuilding")
            return None
        # Materialize to RAM (no mmap_mode). Downstream code fancy-indexes this
        # array repeatedly (incl. inside the per-tie-group Python loop), and a
        # memmap-backed array can stall for minutes under disk pressure.
        active_variant_indices = np.ascontiguousarray(
            np.load(cache_paths.active_indices_path),
            dtype=np.int32,
        )
        with cache_paths.tie_map_path.open("rb") as handle:
            reduced_tie_map = pickle.load(handle)
        expected_active_variant_count = int(manifest_payload.get("active_variant_count", -1))
        expected_reduced_variant_count = int(manifest_payload.get("reduced_variant_count", -1))
        if active_variant_indices.ndim != 1:
            raise ValueError("active variant cache must be 1D.")
        if expected_active_variant_count != active_variant_indices.shape[0]:
            raise ValueError("active variant cache count does not match manifest.")
        if np.any(active_variant_indices < 0) or np.any(active_variant_indices >= standardized_genotypes.shape[1]):
            raise ValueError("active variant cache indices are out of bounds.")
        if reduced_tie_map.kept_indices.ndim != 1:
            raise ValueError("tie-map kept indices cache must be 1D.")
        if expected_reduced_variant_count != reduced_tie_map.kept_indices.shape[0]:
            raise ValueError("tie-map kept-index count does not match manifest.")
        if np.any(reduced_tie_map.kept_indices < 0) or np.any(reduced_tie_map.kept_indices >= active_variant_indices.shape[0]):
            raise ValueError("tie-map kept indices are out of bounds for cached active variants.")
        if reduced_tie_map.original_to_reduced.shape != (active_variant_indices.shape[0],):
            raise ValueError("cached tie-map original_to_reduced shape does not match cached active variants.")
        if (
            reduced_tie_map.reduced_to_group
            and len(reduced_tie_map.reduced_to_group) != reduced_tie_map.kept_indices.shape[0]
        ):
            raise ValueError("cached tie-map group count does not match kept indices.")
        combined_indices = np.asarray(active_variant_indices[reduced_tie_map.kept_indices], dtype=np.int32)
        if cache_paths.reduced_raw_i8_path.exists():
            reduced_raw = Int8RawGenotypeMatrix(np.load(cache_paths.reduced_raw_i8_path, mmap_mode="r"))
            if reduced_raw.shape != (standardized_genotypes.shape[0], combined_indices.shape[0]):
                raise ValueError(
                    f"reduced raw cache shape mismatch: {reduced_raw.shape} != {(standardized_genotypes.shape[0], combined_indices.shape[0])}"
                )
            reduced_genotypes = StandardizedGenotypeMatrix(
                raw=reduced_raw,
                means=np.asarray(prepared_arrays.means[combined_indices], dtype=np.float32),
                scales=np.asarray(prepared_arrays.scales[combined_indices], dtype=np.float32),
                variant_indices=np.arange(combined_indices.shape[0], dtype=np.int32),
                support_counts=np.asarray(prepared_arrays.support_counts[combined_indices], dtype=np.int32),
                _enable_hybrid_backend=False,  # skip sparse backend on mmap — GPU streaming handles it
            )
            log(f"fit-stage cache hit — loading from {cache_paths.cache_dir.name}/{cache_paths.key}.*")
            return active_variant_indices, reduced_tie_map, reduced_genotypes, True
        log(f"fit-stage structure cache hit — loading from {cache_paths.cache_dir.name}/{cache_paths.key}.*")
        return active_variant_indices, reduced_tie_map, standardized_genotypes.subset(combined_indices), False
    except (OSError, ValueError, EOFError, KeyError) as exc:
        log(f"fit-stage cache load failed ({exc}), rebuilding")
        _invalidate_fit_stage_cache(cache_paths)
        return None


def _invalidate_fit_stage_cache(cache_paths: _FitStageCachePaths) -> None:
    for path in (
        cache_paths.manifest_path,
        cache_paths.active_indices_path,
        cache_paths.tie_map_path,
        cache_paths.reduced_raw_i8_path,
        cache_paths.em_checkpoint_path,
    ):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
    try:
        for basis_path in cache_paths.cache_dir.glob(f"{cache_paths.key}.sample_space_basis.r*.seed*.npy"):
            basis_path.unlink(missing_ok=True)
    except OSError:
        pass


def _save_variational_checkpoint(
    cache_paths: _FitStageCachePaths,
    checkpoint: VariationalFitCheckpoint,
) -> None:
    # Refuse to persist a no-progress checkpoint (see _save_fit_checkpoint
    # for rationale): an iter=0 file has nothing to resume from and risks
    # poisoning a subsequent run with a zero-beta "resume" state.
    try:
        completed_iterations = int(checkpoint.completed_iterations)
    except (TypeError, ValueError):
        completed_iterations = 0
    if completed_iterations <= 0:
        log(
            "variational checkpoint skipped: "
            + f"{cache_paths.cache_dir.name}/{cache_paths.key}.em.pkl "
            + f"(completed_iterations={completed_iterations}; nothing to resume)"
        )
        return
    checkpoint_tmp = cache_paths.cache_dir / f"{cache_paths.key}.em.tmp.pkl"
    try:
        cache_paths.cache_dir.mkdir(parents=True, exist_ok=True)
        with checkpoint_tmp.open("wb") as handle:
            pickle.dump(checkpoint, handle, protocol=pickle.HIGHEST_PROTOCOL)
        checkpoint_tmp.replace(cache_paths.em_checkpoint_path)
        log(
            "variational checkpoint saved: "
            + f"{cache_paths.cache_dir.name}/{cache_paths.key}.em.pkl "
            + f"(completed_iterations={checkpoint.completed_iterations})"
        )
    except (OSError, pickle.PicklingError) as exc:
        if checkpoint_tmp.exists():
            checkpoint_tmp.unlink()
        log(f"variational checkpoint save failed ({exc}); continuing without durable resume state")


def _try_load_variational_checkpoint(
    cache_paths: _FitStageCachePaths,
) -> VariationalFitCheckpoint | None:
    if not cache_paths.em_checkpoint_path.exists():
        return None
    try:
        with cache_paths.em_checkpoint_path.open("rb") as handle:
            checkpoint = pickle.load(handle)
        if not isinstance(checkpoint, VariationalFitCheckpoint):
            raise ValueError("variational checkpoint has unexpected type.")
        if int(getattr(checkpoint, "completed_iterations", 0)) <= 0:
            log(
                "variational checkpoint ignored: stale iter=0 at "
                + f"{cache_paths.cache_dir.name}/{cache_paths.key}.em.pkl; "
                + "treating as absent and starting fresh"
            )
            _clear_variational_checkpoint(cache_paths)
            return None
        log(
            "variational checkpoint restored: "
            + f"{cache_paths.cache_dir.name}/{cache_paths.key}.em.pkl "
            + f"(completed_iterations={checkpoint.completed_iterations})"
        )
        return checkpoint
    except (OSError, pickle.UnpicklingError, ValueError, EOFError, AttributeError) as exc:
        log(f"variational checkpoint load failed ({exc}); discarding stale checkpoint")
        _clear_variational_checkpoint(cache_paths)
        return None


def _clear_variational_checkpoint(cache_paths: _FitStageCachePaths) -> None:
    if cache_paths.em_checkpoint_path.exists():
        cache_paths.em_checkpoint_path.unlink()


def _fit_checkpoint_tmp_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name(f"{checkpoint_path.stem}.tmp{checkpoint_path.suffix}")


def _fit_checkpoint_config_hash(
    *,
    genotype_matrix: RawGenotypeMatrix,
    covariates: NDArray,
    targets: NDArray,
    variant_records: Sequence[VariantRecord],
    config: ModelConfig,
) -> str:
    hasher = hashlib.sha256()
    hasher.update(f"fit-checkpoint-v{_FIT_CHECKPOINT_VERSION}".encode("utf-8"))
    hasher.update(np.asarray(genotype_matrix.shape, dtype=np.int64).tobytes())
    variant_hasher = hashlib.sha256()
    for record in variant_records:
        variant_payload = {
            "id": record.variant_id,
            "class": record.variant_class.value,
            "chromosome": record.chromosome,
            "position": int(record.position),
            "length": float(record.length),
            "allele_frequency": float(record.allele_frequency),
            "quality": float(record.quality),
            "training_support": None if record.training_support is None else int(record.training_support),
            "is_repeat": bool(record.is_repeat),
            "is_copy_number": bool(record.is_copy_number),
            "prior_binary_features": record.prior_binary_features,
            "prior_continuous_features": record.prior_continuous_features,
            "prior_categorical_features": record.prior_categorical_features,
            "prior_membership_features": record.prior_membership_features,
            "prior_nested_features": record.prior_nested_features,
            "prior_nested_membership_features": record.prior_nested_membership_features,
            "prior_class_members": [member.value for member in record.prior_class_members],
            "prior_class_membership": [float(weight) for weight in record.prior_class_membership],
        }
        variant_hasher.update(json.dumps(variant_payload, sort_keys=True).encode("utf-8"))
        variant_hasher.update(b"\n")
    hasher.update(variant_hasher.hexdigest().encode("utf-8"))
    _update_hash_with_array_bytes(hasher, np.asarray(covariates, dtype=np.float32))
    _update_hash_with_array_bytes(hasher, np.asarray(targets, dtype=np.float32).reshape(-1))
    config_payload = {
        "trait_type": config.trait_type.value,
        "max_outer_iterations": int(config.max_outer_iterations),
        "convergence_tolerance": float(config.convergence_tolerance),
        "max_inner_newton_iterations": int(config.max_inner_newton_iterations),
        "newton_gradient_tolerance": float(config.newton_gradient_tolerance),
        "linear_solver_tolerance": float(config.linear_solver_tolerance),
        "maximum_linear_solver_iterations": int(config.maximum_linear_solver_iterations),
        "beta_variance_update_interval": int(config.beta_variance_update_interval),
        "minimum_minor_allele_frequency": float(config.minimum_minor_allele_frequency),
        "marginal_screen_min_abs_z": float(config.marginal_screen_min_abs_z),
        "stochastic_variational_updates": bool(config.stochastic_variational_updates),
        "stochastic_min_variant_count": int(config.stochastic_min_variant_count),
        "stochastic_variant_batch_size": int(config.stochastic_variant_batch_size),
        "posterior_working_set_min_variants": int(config.posterior_working_set_min_variants),
        "posterior_working_set_initial_size": int(config.posterior_working_set_initial_size),
        "posterior_working_set_growth": int(config.posterior_working_set_growth),
        "posterior_working_set_max_passes": int(config.posterior_working_set_max_passes),
        "posterior_working_set_coefficient_tolerance": float(config.posterior_working_set_coefficient_tolerance),
        "sample_space_preconditioner_rank": int(config.sample_space_preconditioner_rank),
        "random_seed": int(config.random_seed),
    }
    hasher.update(json.dumps(config_payload, sort_keys=True).encode("utf-8"))
    return hasher.hexdigest()


def _save_fit_checkpoint(
    checkpoint_path: Path,
    checkpoint: VariationalFitCheckpoint,
    *,
    config_hash: str,
) -> None:
    # Refuse to persist a no-progress checkpoint: an iter=0 file has nothing
    # to resume from and risks poisoning a subsequent run with a zero-beta
    # "resume" state. See _save_variational_checkpoint for the same guard.
    try:
        completed_iterations = int(checkpoint.completed_iterations)
    except (TypeError, ValueError):
        completed_iterations = 0
    if completed_iterations <= 0:
        log(
            "fit checkpoint skipped: "
            + f"{checkpoint_path} "
            + f"(completed_iterations={completed_iterations}; nothing to resume)"
        )
        return
    checkpoint_tmp = _fit_checkpoint_tmp_path(checkpoint_path)
    try:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_bytes = np.frombuffer(
            pickle.dumps(checkpoint, protocol=pickle.HIGHEST_PROTOCOL),
            dtype=np.uint8,
        )
        np.savez(
            checkpoint_tmp,
            version=np.asarray([_FIT_CHECKPOINT_VERSION], dtype=np.int32),
            iter=np.asarray([int(checkpoint.completed_iterations)], dtype=np.int32),
            beta=np.asarray(checkpoint.beta_state, dtype=np.float64),
            checkpoint_pickle=checkpoint_bytes,
            config_hash=np.asarray(config_hash),
        )
        os.rename(checkpoint_tmp, checkpoint_path)
        log(
            "fit checkpoint saved: "
            + f"{checkpoint_path} (completed_iterations={checkpoint.completed_iterations})"
        )
    except (OSError, pickle.PicklingError, ValueError) as exc:
        try:
            checkpoint_tmp.unlink(missing_ok=True)
        except OSError:
            pass
        log(f"fit checkpoint save failed ({exc}); continuing without durable resume state")


def _try_load_fit_checkpoint(
    checkpoint_path: Path,
    *,
    config_hash: str,
) -> VariationalFitCheckpoint | None:
    if not checkpoint_path.exists():
        return None
    try:
        with np.load(checkpoint_path, allow_pickle=False) as payload:
            stored_hash = str(payload["config_hash"].item())
            if stored_hash != config_hash:
                log(f"fit checkpoint ignored: config hash mismatch at {checkpoint_path}")
                return None
            # Cheap pre-flight: if the file records iter=0 there is nothing
            # to resume from. Treat it as absent and clear it so we start
            # fresh rather than replaying a zero-beta "resume" state.
            try:
                stored_iter = int(np.asarray(payload["iter"]).reshape(-1)[0])
            except (KeyError, ValueError, IndexError):
                stored_iter = 0
            stale_iter_zero = stored_iter <= 0
            if stale_iter_zero:
                checkpoint_bytes = None
            else:
                checkpoint_bytes = np.asarray(payload["checkpoint_pickle"], dtype=np.uint8).tobytes()
        if stale_iter_zero:
            log(
                "fit checkpoint ignored: stale iter=0 at "
                + f"{checkpoint_path}; treating as absent and starting fresh"
            )
            _clear_fit_checkpoint(checkpoint_path)
            return None
        checkpoint = pickle.loads(checkpoint_bytes)
        if not isinstance(checkpoint, VariationalFitCheckpoint):
            raise ValueError("fit checkpoint has unexpected type.")
        if int(getattr(checkpoint, "completed_iterations", 0)) <= 0:
            log(
                "fit checkpoint ignored: stale iter=0 at "
                + f"{checkpoint_path}; treating as absent and starting fresh"
            )
            _clear_fit_checkpoint(checkpoint_path)
            return None
        log(
            "fit checkpoint restored: "
            + f"{checkpoint_path} (completed_iterations={checkpoint.completed_iterations})"
        )
        return checkpoint
    except (OSError, pickle.UnpicklingError, ValueError, EOFError, AttributeError, KeyError) as exc:
        log(f"fit checkpoint load failed ({exc}); discarding stale checkpoint")
        try:
            checkpoint_path.unlink(missing_ok=True)
        except OSError:
            pass
        return None


def _clear_fit_checkpoint(checkpoint_path: Path | None) -> None:
    if checkpoint_path is None:
        return
    try:
        checkpoint_path.unlink(missing_ok=True)
    except OSError:
        pass


class BayesianPGS:
    """Main entry point for fitting and applying a Bayesian Polygenic Score.

    A PGS predicts a trait (disease risk or continuous measurement) from an
    individual's genotypes across many variants.  This implementation uses a
    Bayesian approach that:

      1. Automatically learns which variants matter (most get shrunk to ~zero)
      2. Gives structural variants (deletions, duplications, etc.) wider priors
         than SNVs, reflecting their potentially larger per-variant effects
      3. Handles hundreds of thousands of samples and variants via streaming
         genotype I/O, GPU-accelerated statistics, and efficient linear algebra

    Typical workflow:
        model = BayesianPGS(config)
        model.fit(genotypes, covariates, targets, variant_records)
        predictions = model.predict(new_genotypes, new_covariates)
        model.export("model.npz")
    """
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.state: FittedState | None = None

    def fit(
        self,
        genotypes: RawGenotypeMatrix | NDArray,
        covariates: NDArray,
        targets: NDArray,
        variant_records: Sequence[VariantRecord | dict[str, Any]],
        validation_data: tuple[RawGenotypeMatrix | NDArray, NDArray, NDArray] | None = None,
        variant_stats: VariantStatistics | None = None,
        per_epoch_eval_callback: Callable[[dict[str, Any]], None] | None = None,
        validation_is_holdout_only: bool = False,
        fit_checkpoint_path: str | Path | None = None,
    ) -> BayesianPGS:
        from sv_pgs.genotype import require_gpu
        require_gpu()
        registered_fit_checkpoint_path = _pop_registered_fit_checkpoint_path(self.config)
        if fit_checkpoint_path is None:
            fit_checkpoint_path = registered_fit_checkpoint_path
        durable_fit_checkpoint_path = None if fit_checkpoint_path is None else Path(fit_checkpoint_path)
        log(f"=== MODEL FIT START ===  genotypes={genotypes.shape}  covariates={covariates.shape}  targets={targets.shape}  pre_computed_stats={'YES' if variant_stats else 'NO'}")
        raw_genotype_matrix = as_raw_genotype_matrix(genotypes)
        _validate_fit_inputs(
            genotype_matrix=raw_genotype_matrix,
            covariates=covariates,
            targets=targets,
            variant_records=variant_records,
            variant_stats=variant_stats,
        )
        full_variant_records = _normalize_variant_records(variant_records)
        tuned_config, tuning_summary = _runtime_tuned_config_for_fit(
            config=self.config,
            genotype_matrix=raw_genotype_matrix,
        )
        self.config = tuned_config
        if tuning_summary is not None:
            log(tuning_summary)
        covariate_matrix = self._with_intercept(covariates)
        total_variant_count = len(variant_records)
        fit_checkpoint_config_hash = None
        if durable_fit_checkpoint_path is not None:
            fit_checkpoint_config_hash = _fit_checkpoint_config_hash(
                genotype_matrix=raw_genotype_matrix,
                covariates=covariate_matrix,
                targets=np.asarray(targets, dtype=np.float32).reshape(-1),
                variant_records=full_variant_records,
                config=self.config,
            )
        log(f"normalized full variant records: {total_variant_count:,}  mem={mem()}")
        fit_stage_variant_stats_cache_path = _fit_stage_variant_stats_cache_path(
            raw_genotype_matrix,
            self.config,
        )

        # Use pre-computed stats if available (saves 3 full data passes)
        if variant_stats is not None:
            log("using pre-computed variant statistics (means, scales, support) [NO DATA PASSES]")
            if fit_stage_variant_stats_cache_path is not None and not fit_stage_variant_stats_cache_path.exists():
                _save_fit_stage_variant_stats_cache(fit_stage_variant_stats_cache_path, variant_stats)
            prepared_arrays = fit_preprocessor_from_stats(variant_stats, covariate_matrix, targets)
        else:
            if fit_stage_variant_stats_cache_path is not None:
                variant_stats = _try_load_fit_stage_variant_stats_cache(
                    fit_stage_variant_stats_cache_path,
                    variant_count=raw_genotype_matrix.shape[1],
                )
            if variant_stats is None:
                log("computing variant statistics in-fit so support/standardization share one pass...")
                variant_stats = compute_variant_statistics(
                    raw_genotypes=raw_genotype_matrix,
                    config=self.config,
                )
                if fit_stage_variant_stats_cache_path is not None:
                    _save_fit_stage_variant_stats_cache(fit_stage_variant_stats_cache_path, variant_stats)
            prepared_arrays = fit_preprocessor_from_stats(variant_stats, covariate_matrix, targets)
        preprocessor = Preprocessor(means=prepared_arrays.means, scales=prepared_arrays.scales)
        log(f"preprocessor ready  {total_variant_count} variants  mem={mem()}")

        log("creating standardized genotype view...")
        standardized_genotypes = StandardizedGenotypeMatrix(
            raw=raw_genotype_matrix,
            means=prepared_arrays.means,
            scales=prepared_arrays.scales,
            variant_indices=np.arange(raw_genotype_matrix.shape[1], dtype=np.int32),
            support_counts=prepared_arrays.support_counts,
            _enable_hybrid_backend=False,  # skip sparse backend build on 1.68M mmap'd variants
        )
        log(f"  standardized view created  mem={mem()}")
        log("  computing fit-stage cache key...")
        fit_stage_cache_paths = _fit_stage_cache_paths(
            genotype_matrix=raw_genotype_matrix,
            allele_frequencies=variant_stats.allele_frequencies,
            means=prepared_arrays.means,
            scales=prepared_arrays.scales,
            covariates=prepared_arrays.covariates,
            targets=prepared_arrays.targets,
            config=self.config,
        )
        log(f"  fit-stage cache key: {fit_stage_cache_paths.key if fit_stage_cache_paths else 'NONE'}  mem={mem()}")
        log("  checking fit-stage cache...")
        cached_fit_stage = (
            None
            if fit_stage_cache_paths is None
            else _try_load_fit_stage_cache(
                cache_paths=fit_stage_cache_paths,
                prepared_arrays=prepared_arrays,
                standardized_genotypes=standardized_genotypes,
            )
        )
        if cached_fit_stage is None:
            log("selecting active variant indices (vectorized MAF filter)...")
            active_variant_indices = _select_active_variant_indices_fast(
                allele_frequencies=variant_stats.allele_frequencies,
                minimum_minor_allele_frequency=self.config.minimum_minor_allele_frequency,
            )
            log(f"active variants: {len(active_variant_indices)} / {total_variant_count} ({100.0*len(active_variant_indices)/max(total_variant_count,1):.1f}%)")
            if self.config.marginal_screen_min_abs_z > 0.0 and active_variant_indices.size > 0:
                log(
                    f"applying marginal pre-screen at |z| >= {self.config.marginal_screen_min_abs_z:.2f}  "
                    f"(residualized on {prepared_arrays.covariates.shape[1]} covariates)..."
                )
                pre_screen_count = int(active_variant_indices.size)
                marginal_z_scores = compute_marginal_z_scores(
                    standardized_genotypes=standardized_genotypes,
                    active_variant_indices=active_variant_indices,
                    covariate_matrix=prepared_arrays.covariates,
                    target_vector=prepared_arrays.targets,
                )
                z_pass_mask = np.abs(marginal_z_scores) >= self.config.marginal_screen_min_abs_z
                kept_count = int(np.sum(z_pass_mask))
                if kept_count == 0:
                    log(
                        "  marginal pre-screen kept 0 variants — disabling screen to avoid "
                        "fitting a covariates-only model on a poorly calibrated z-score."
                    )
                else:
                    active_variant_indices = np.asarray(
                        active_variant_indices[z_pass_mask], dtype=np.int32
                    )
                    log(
                        f"  marginal pre-screen kept {kept_count} of {pre_screen_count} variants  "
                        f"({100.0 * kept_count / pre_screen_count:.1f}% of post-MAF)"
                    )
        else:
            active_variant_indices, reduced_tie_map, reduced_genotypes, local_cache = cached_fit_stage
            log(
                "active/tie cache restored: "
                + f"{len(active_variant_indices)} active -> {len(reduced_tie_map.kept_indices)} unique  mem={mem()}"
            )
        if active_variant_indices.size == 0:
            log("no active variants remain after filtering; fitting covariates-only model...")
            reduced_validation_dense: tuple[F32Array, F32Array, F32Array] | None = None
            if validation_data is not None:
                _, validation_covariates, validation_targets = validation_data
                reduced_validation_dense = (
                    np.empty((len(validation_targets), 0), dtype=np.float32),
                    self._with_intercept(np.asarray(validation_covariates, dtype=np.float32)),
                    np.asarray(validation_targets, dtype=np.float32),
                )
            fit_result = _fit_without_active_variants(
                covariates=prepared_arrays.covariates,
                targets=prepared_arrays.targets,
                config=self.config,
                validation_data=reduced_validation_dense,
                predictor_offset=None,
                validation_offset=None,
            )
            full_coefficients = np.zeros(total_variant_count, dtype=np.float32)
            nonzero_coefficient_indices, nonzero_coefficients = _nonzero_coefficient_cache(full_coefficients)
            training_linear_predictor = _training_linear_predictor_cache(
                fit_result=fit_result,
                covariate_matrix=prepared_arrays.covariates,
                reduced_genotypes=None,
            )
            self.state = FittedState(
                variant_records=[],
                full_variant_records=full_variant_records,
                active_variant_indices=np.zeros(0, dtype=np.int32),
                preprocessor=preprocessor,
                tie_map=_empty_tie_map(total_variant_count),
                fit_result=fit_result,
                full_coefficients=full_coefficients,
                nonzero_coefficient_indices=nonzero_coefficient_indices,
                nonzero_coefficients=nonzero_coefficients,
                nonzero_means=np.zeros(0, dtype=np.float32),
                nonzero_scales=np.zeros(0, dtype=np.float32),
                training_genetic_score=np.zeros(training_linear_predictor.shape[0], dtype=np.float32),
                training_covariate_score=training_linear_predictor.copy(),
                training_linear_predictor=training_linear_predictor,
            )
            if fit_stage_cache_paths is not None:
                _clear_variational_checkpoint(fit_stage_cache_paths)
            _clear_fit_checkpoint(durable_fit_checkpoint_path)
            log(f"coefficients: 0 non-zero out of {total_variant_count} total")
            _release_gpu_memory_pools()
            log(f"=== MODEL FIT DONE ===  mem={mem()}")
            return self

        # Create VariantRecord objects ONLY for active variants (not all 1.68M)
        # This saves ~840 MB of Python object overhead
        log(f"creating training records for {len(active_variant_indices)} active variants...")
        active_records = _training_records_from_stats(
            full_variant_records,
            variant_stats,
            active_variant_indices,
        )
        import gc
        gc.collect()
        log(f"active training records created: {len(active_records)}  mem={mem()}")
        if cached_fit_stage is None:
            active_genotypes = standardized_genotypes.subset(active_variant_indices)
            log("building tie map (detecting identical/negated genotype columns)...")
            reduced_tie_map = build_tie_map(active_genotypes, active_records, self.config)
            if fit_stage_cache_paths is not None:
                _save_fit_stage_structure_cache(
                    cache_paths=fit_stage_cache_paths,
                    active_variant_indices=active_variant_indices,
                    reduced_tie_map=reduced_tie_map,
                )
            if _tie_map_keeps_all_active_variants(reduced_tie_map, active_variant_indices.shape[0]):
                reduced_genotypes = active_genotypes
            else:
                combined_indices = active_variant_indices[reduced_tie_map.kept_indices]
                reduced_genotypes = standardized_genotypes.subset(combined_indices)
            local_cache = False
        else:
            active_genotypes = None
        log("  projecting tie map to original variant space...")
        original_space_tie_map = _project_tie_map_to_original_space(
            reduced_tie_map=reduced_tie_map,
            active_variant_indices=active_variant_indices,
            original_variant_count=total_variant_count,
        )
        log(f"  tie map projected  mem={mem()}")
        log(f"tie map: {len(active_variant_indices)} active -> {len(reduced_tie_map.kept_indices)} unique ({len(reduced_tie_map.reduced_to_group)} groups)  mem={mem()}")

        reduced_validation = None
        if validation_data is not None:
            log("  building combined validation indices...")
            validation_genotypes, validation_covariates, validation_targets = validation_data
            combined_validation_indices = active_variant_indices[reduced_tie_map.kept_indices]
            log(f"  combined validation indices ready: {combined_validation_indices.shape[0]} variants  mem={mem()}")
            validation_t0 = time.monotonic()
            log(
                "preparing reduced validation genotype view "
                + f"({len(validation_targets)} samples x {combined_validation_indices.shape[0]} variants)  mem={mem()}"
            )
            validation_raw_genotypes = as_raw_genotype_matrix(validation_genotypes)
            standardized_validation = StandardizedGenotypeMatrix(
                raw=validation_raw_genotypes,
                means=prepared_arrays.means,
                scales=prepared_arrays.scales,
                variant_indices=np.asarray(combined_validation_indices, dtype=np.int32),
                support_counts=prepared_arrays.support_counts,
                sample_count=validation_raw_genotypes.shape[0],
                _enable_hybrid_backend=False,
            )
            reduced_validation = (
                standardized_validation,
                self._with_intercept(np.asarray(validation_covariates, dtype=np.float32)),
                np.asarray(validation_targets, dtype=np.float32),
            )
            log(
                "  reduced validation view ready "
                + f"in {time.monotonic() - validation_t0:.1f}s  mem={mem()}"
            )

        # Free original mmap'd chromosome data BEFORE materialization/persistence.
        # The tie map is done — we only need reduced_genotypes going forward.
        # Freeing early reduces memory pressure and prevents Bus errors from
        # mmap page faults on full disk during int8 persistence.
        log("  freeing original genotype data before materialization...")
        del raw_genotype_matrix, standardized_genotypes, active_genotypes
        import gc
        gc.collect()
        log(f"  freed. mem={mem()}")

        # Materialize the reduced genotype matrix (RAM or GPU via CuPy).
        log(f"materializing reduced genotype matrix ({reduced_genotypes.shape})...")
        in_memory = reduced_genotypes.try_materialize_gpu()
        if in_memory:
            log(f"  GPU materialization succeeded  mem={mem()}")
        if not in_memory:
            in_memory = reduced_genotypes.try_materialize()
            if in_memory:
                log(f"  RAM materialization succeeded  mem={mem()}")
        persistent_reduced_cache = (
            fit_stage_cache_paths is not None
            and fit_stage_cache_paths.reduced_raw_i8_path.exists()
        )
        if fit_stage_cache_paths is not None and not persistent_reduced_cache:
            log("  persisting reduced int8 genotypes for cohort cache reuse...")
            persistent_reduced_cache = reduced_genotypes.try_cache_persistently(
                fit_stage_cache_paths.reduced_raw_i8_path,
            )
            if persistent_reduced_cache:
                log(f"  persistent int8 cohort cache saved  mem={mem()}")
                _write_fit_stage_cache_manifest(
                    cache_paths=fit_stage_cache_paths,
                    active_variant_count=int(active_variant_indices.shape[0]),
                    reduced_variant_count=int(reduced_tie_map.kept_indices.shape[0]),
                    has_reduced_raw_i8=True,
                )
        if in_memory:
            reduced_genotypes.release_raw_storage()
        else:
            local_cache = local_cache or persistent_reduced_cache
            if not local_cache:
                local_cache = reduced_genotypes.try_cache_locally()
                if local_cache:
                    log(f"  local tmpdir cache ready  mem={mem()}")
            if not local_cache:
                if (
                    _REFUSE_BINARY_TR_NEWTON_NO_CACHE
                    and self.config.trait_type == TraitType.BINARY
                    and not bool(getattr(self.config, "allow_mmap_streaming_for_binary", False))
                ):
                    raise RuntimeError(
                        "No GPU/RAM/local int8 cache available for binary TR-Newton fit. "
                        "Free disk or reduce block size; refusing full mmap streaming inside Newton-CG."
                    )
                log(f"  no materialization possible — streaming from mmap  mem={mem()}")
        if reduced_validation is not None:
            validation_genotype_matrix = reduced_validation[0]
            if isinstance(validation_genotype_matrix, StandardizedGenotypeMatrix):
                # Persist a validation-side int8 cache analogous to the training
                # reduced_raw_i8 cache. Without this, every run re-reads sparse
                # columns through the per-chromosome Concatenated→Indexed→Int8
                # wrapper chain — minutes of wall-clock for ~3 GB of data. With
                # the cache present, the validation upload reuses the fast
                # parallel-pread GPU materialization path.
                if (
                    fit_stage_cache_paths is not None
                    and isinstance(validation_genotype_matrix.raw, RawGenotypeMatrix)
                ):
                    validation_raw_signature = _persistent_raw_signature(
                        validation_genotype_matrix.raw
                    )
                    if validation_raw_signature is not None:
                        val_hasher = hashlib.sha256()
                        val_hasher.update(
                            f"fit-stage-val-i8-v{_FIT_STAGE_CACHE_VERSION}".encode("utf-8")
                        )
                        val_hasher.update(validation_raw_signature.encode("utf-8"))
                        _update_hash_with_array_bytes(
                            val_hasher,
                            np.asarray(validation_genotype_matrix.variant_indices, dtype=np.int64),
                        )
                        val_subkey = val_hasher.hexdigest()[:24]
                        validation_int8_cache_path = (
                            fit_stage_cache_paths.cache_dir
                            / f"{fit_stage_cache_paths.key}.val_{val_subkey}.reduced_raw_i8.npy"
                        )
                        if validation_int8_cache_path.exists():
                            log(
                                "  validation int8 cache hit — rebinding to "
                                + f"{validation_int8_cache_path.name}  mem={mem()}"
                            )
                        else:
                            log("  persisting validation int8 cache (one-time cost; future runs reuse it)...")
                        validation_genotype_matrix.try_cache_persistently(validation_int8_cache_path)
                materialize_validation = (
                    not validation_is_holdout_only
                    or bool(getattr(self.config, "validate_first_iteration", True))
                    or int(self.config.validation_interval) < int(self.config.max_outer_iterations)
                )
                if materialize_validation:
                    log(f"materializing validation genotype matrix ({validation_genotype_matrix.shape})...")
                    validation_in_memory = validation_genotype_matrix.try_materialize_gpu()
                    if validation_in_memory:
                        log(f"  validation GPU materialization succeeded  mem={mem()}")
                    if not validation_in_memory:
                        validation_in_memory = validation_genotype_matrix.try_materialize()
                        if validation_in_memory:
                            log(f"  validation RAM materialization succeeded  mem={mem()}")
                    if validation_in_memory:
                        validation_genotype_matrix.release_raw_storage()
                    else:
                        log(f"  validation matrix remains streaming  mem={mem()}")
                else:
                    log("  held-out validation matrix remains streaming until final evaluation  mem=" + mem())
        if fit_stage_cache_paths is not None and int(self.config.sample_space_preconditioner_rank) > 0:
            log("  restoring preconditioner basis cache...")
            _try_restore_sample_space_basis_cache(
                cache_paths=fit_stage_cache_paths,
                genotype_matrix=reduced_genotypes,
                rank=int(self.config.sample_space_preconditioner_rank),
                random_seed=int(self.config.random_seed),
            )
            log(f"  preconditioner cache restored  mem={mem()}")
        log(
            f"starting variational EM  max_iterations={self.config.max_outer_iterations}  "
            f"validation_interval={self.config.validation_interval}  "
            f"reduced_matrix={reduced_genotypes.shape}  in_memory={in_memory}  "
            f"local_cache={local_cache or persistent_reduced_cache}  "
            f"on_gpu={reduced_genotypes._cupy_cache is not None}  "
            f"mem={mem()}"
        )
        em_validation_data = reduced_validation
        em_per_epoch_eval_callback = per_epoch_eval_callback
        em_validation_is_holdout_only = validation_is_holdout_only
        if durable_fit_checkpoint_path is not None and reduced_validation is not None:
            if validation_is_holdout_only:
                log(
                    "  fit checkpoint enabled; held-out validation is used for monitoring only "
                    "(not model selection) so checkpoint/resume remains active"
                )
                # Decouple monitoring from model selection: keep the per-epoch
                # callback so training_history.tsv receives holdout metrics each
                # epoch, but clear em_validation_data so EM does not use holdout
                # for best-epoch selection.
                em_validation_data = None
                em_per_epoch_eval_callback = per_epoch_eval_callback
                em_validation_is_holdout_only = True
            else:
                log("  fit checkpoint disabled because validation data drives EM model selection")
                durable_fit_checkpoint_path = None
                fit_checkpoint_config_hash = None
        log("  checking for EM checkpoint to resume from...")
        resume_checkpoint = None
        if durable_fit_checkpoint_path is not None and fit_checkpoint_config_hash is not None:
            resume_checkpoint = _try_load_fit_checkpoint(
                durable_fit_checkpoint_path,
                config_hash=fit_checkpoint_config_hash,
            )
        if resume_checkpoint is None and durable_fit_checkpoint_path is None:
            resume_checkpoint = (
                None
                if fit_stage_cache_paths is None
                else _try_load_variational_checkpoint(fit_stage_cache_paths)
            )
        log(f"  EM checkpoint: {'found — resuming' if resume_checkpoint else 'none — starting fresh'}")
        checkpoint_callback: Callable[[VariationalFitCheckpoint], None] | None
        if fit_stage_cache_paths is not None or (
            durable_fit_checkpoint_path is not None and fit_checkpoint_config_hash is not None
        ):
            cache_paths_for_callback = fit_stage_cache_paths
            checkpoint_path_for_callback = durable_fit_checkpoint_path
            checkpoint_hash_for_callback = fit_checkpoint_config_hash
            # Bind a non-None alias for the closure: `reduced_genotypes` is later
            # reassigned to None for GC, but at this point it is guaranteed
            # populated (validated by the early-exit covariate-only branch above).
            reduced_genotypes_for_callback: StandardizedGenotypeMatrix = reduced_genotypes

            def _checkpoint_callback(checkpoint: VariationalFitCheckpoint) -> None:
                if checkpoint_path_for_callback is not None and checkpoint_hash_for_callback is not None:
                    _save_fit_checkpoint(
                        checkpoint_path_for_callback,
                        checkpoint,
                        config_hash=checkpoint_hash_for_callback,
                    )
                if cache_paths_for_callback is not None:
                    _save_variational_checkpoint(cache_paths_for_callback, checkpoint)
                if int(self.config.sample_space_preconditioner_rank) <= 0:
                    return
                if cache_paths_for_callback is None:
                    return
                basis_path = _sample_space_basis_cache_path(
                    cache_paths_for_callback,
                    rank=int(self.config.sample_space_preconditioner_rank),
                    random_seed=int(self.config.random_seed),
                )
                if not basis_path.exists():
                    _save_sample_space_basis_cache(
                        cache_paths=cache_paths_for_callback,
                        genotype_matrix=reduced_genotypes_for_callback,
                        rank=int(self.config.sample_space_preconditioner_rank),
                        random_seed=int(self.config.random_seed),
                    )

            checkpoint_callback = _checkpoint_callback
        else:
            checkpoint_callback = None

        fit_result = fit_variational_em(
            genotypes=reduced_genotypes,
            covariates=prepared_arrays.covariates,
            targets=prepared_arrays.targets,
            records=active_records,
            tie_map=reduced_tie_map,
            config=self.config,
            validation_data=em_validation_data,
            resume_checkpoint=resume_checkpoint,
            checkpoint_callback=checkpoint_callback,
            predictor_offset=None,
            validation_offset=None,
            per_epoch_eval_callback=em_per_epoch_eval_callback,
            validation_is_holdout_only=em_validation_is_holdout_only,
        )
        if fit_stage_cache_paths is not None:
            _clear_variational_checkpoint(fit_stage_cache_paths)
            if int(self.config.sample_space_preconditioner_rank) > 0:
                _save_sample_space_basis_cache(
                    cache_paths=fit_stage_cache_paths,
                    genotype_matrix=reduced_genotypes,
                    rank=int(self.config.sample_space_preconditioner_rank),
                    random_seed=int(self.config.random_seed),
                )
        # Final artifacts are the durable completion marker; remove iteration
        # state so the next identical run starts from completed outputs.
        _clear_fit_checkpoint(durable_fit_checkpoint_path)
        final_obj = fit_result.objective_history[-1]
        if fit_result.converged:
            log(
                f"variational EM converged in {len(fit_result.objective_history)} iterations  "
                f"final_obj={final_obj:.4f}  mem={mem()}"
            )
        else:
            delta_str = (
                "N/A"
                if fit_result.final_parameter_change is None
                else f"{fit_result.final_parameter_change:.3e}"
            )
            predictor_delta_str = (
                "N/A"
                if fit_result.final_predictor_change is None
                else f"{fit_result.final_predictor_change:.3e}"
            )
            objective_delta_str = (
                "N/A"
                if fit_result.final_objective_change is None
                else f"{fit_result.final_objective_change:.3e}"
            )
            hyper_delta_str = (
                "N/A"
                if fit_result.final_hyperparameter_change is None
                else f"{fit_result.final_hyperparameter_change:.3e}"
            )
            log(
                f"variational EM stopped after {len(fit_result.objective_history)} iterations "
                f"without convergence  final_obj={final_obj:.4f}  "
                f"delta={delta_str}  pred_delta={predictor_delta_str}  "
                f"obj_delta={objective_delta_str}  hyper_delta={hyper_delta_str}  "
                f"tol={self.config.convergence_tolerance:.3e}  mem={mem()}"
            )

        log("expanding coefficients from reduced to full space...")
        tie_group_weights = _tie_group_export_weights(
            tie_map=reduced_tie_map,
            fit_result=fit_result,
        )
        active_coefficients = reduced_tie_map.expand_coefficients(
            fit_result.beta_reduced,
            group_weights=tie_group_weights,
        )
        full_coefficients = np.zeros(total_variant_count, dtype=np.float32)
        full_coefficients[active_variant_indices] = active_coefficients
        nonzero_coefficient_indices, nonzero_coefficients = _nonzero_coefficient_cache(full_coefficients)
        nonzero_means = np.asarray(prepared_arrays.means[nonzero_coefficient_indices], dtype=np.float32)
        nonzero_scales = np.asarray(prepared_arrays.scales[nonzero_coefficient_indices], dtype=np.float32)
        training_linear_predictor = _training_linear_predictor_cache(
            fit_result=fit_result,
            covariate_matrix=prepared_arrays.covariates,
            reduced_genotypes=reduced_genotypes,
        )
        training_covariate_score = np.asarray(prepared_arrays.covariates @ fit_result.alpha, dtype=np.float32)
        training_genetic_score = np.asarray(training_linear_predictor - training_covariate_score, dtype=np.float32)
        nonzero_count = int(np.count_nonzero(full_coefficients))
        log(f"coefficients: {nonzero_count} non-zero out of {total_variant_count} total")

        self.state = FittedState(
            variant_records=active_records,
            full_variant_records=full_variant_records,
            active_variant_indices=active_variant_indices,
            preprocessor=preprocessor,
            tie_map=original_space_tie_map,
            fit_result=fit_result,
            full_coefficients=full_coefficients,
            nonzero_coefficient_indices=nonzero_coefficient_indices,
            nonzero_coefficients=nonzero_coefficients,
            nonzero_means=nonzero_means,
            nonzero_scales=nonzero_scales,
            training_genetic_score=training_genetic_score,
            training_covariate_score=training_covariate_score,
            training_linear_predictor=training_linear_predictor,
        )
        # Drop device-backed references before flushing CuPy pools so the next
        # fit in this process starts with a clean GPU. All FittedState arrays
        # above are forced through np.asarray(...) to host-side float32.
        reduced_genotypes = None  # type: ignore[assignment]
        if "validation_genotype_matrix" in locals():
            validation_genotype_matrix = None  # type: ignore[assignment]
        _release_gpu_memory_pools()
        log(f"=== MODEL FIT DONE ===  mem={mem()}")
        return self

    def decision_function(self, genotypes: RawGenotypeMatrix | NDArray, covariates: NDArray) -> F32Array:
        """Compute the raw linear predictor (before sigmoid for binary traits).

        For each individual: score = sum_j(beta_j * standardized_genotype_j) + covariates @ alpha.
        Only reads variants with non-zero coefficients (typically <1% of all variants),
        making prediction fast even on huge genotype files.
        """
        genotype_component, covariate_component = self.decision_components(genotypes, covariates)
        return np.asarray(genotype_component + covariate_component, dtype=np.float32)

    def decision_components(
        self,
        genotypes: RawGenotypeMatrix | NDArray,
        covariates: NDArray,
    ) -> tuple[F32Array, F32Array]:
        """Return the separate genetic and covariate contributions to the predictor."""
        fitted_state = self._require_state()
        covariate_matrix = self._with_intercept(np.asarray(covariates, dtype=np.float32))
        covariate_component = np.asarray(covariate_matrix @ fitted_state.fit_result.alpha, dtype=np.float32)

        nonzero_indices = fitted_state.nonzero_coefficient_indices
        nonzero_coefficients = fitted_state.nonzero_coefficients
        log(f"decision_function: {genotypes.shape[0]} samples, {len(nonzero_indices)} non-zero coefficients (of {len(fitted_state.full_coefficients)} total)  mem={mem()}")

        if len(nonzero_indices) == 0:
            return np.zeros(genotypes.shape[0], dtype=np.float32), covariate_component

        raw_genotypes = as_raw_genotype_matrix(genotypes)
        genotype_component = _raw_standardized_subset_matvec(
            raw_genotypes=raw_genotypes,
            variant_indices=nonzero_indices,
            means=fitted_state.nonzero_means,
            scales=fitted_state.nonzero_scales,
            coefficients=nonzero_coefficients,
            batch_size=auto_batch_size(raw_genotypes.shape[0]),
        )
        log(f"  decision_function done  mem={mem()}")
        return np.asarray(genotype_component, dtype=np.float32), covariate_component

    def predict_proba(self, genotypes: RawGenotypeMatrix | NDArray, covariates: NDArray) -> F32Array:
        if self.config.trait_type != TraitType.BINARY:
            raise ValueError("predict_proba is only available for binary traits.")
        linear_predictor = self.decision_function(genotypes, covariates)
        positive_probability = np.asarray(stable_sigmoid(linear_predictor), dtype=np.float32)
        return np.column_stack([1.0 - positive_probability, positive_probability])

    def predict(self, genotypes: RawGenotypeMatrix | NDArray, covariates: NDArray) -> NDArray:
        linear_predictor = self.decision_function(genotypes, covariates)
        if self.config.trait_type == TraitType.BINARY:
            return (np.asarray(stable_sigmoid(linear_predictor), dtype=np.float32) >= 0.5).astype(np.int32)
        return linear_predictor

    def export(self, path: str | Path, *, fit_fingerprint: str = "") -> None:
        fitted_state = self._require_state()
        fit_result = fitted_state.fit_result
        if not fit_result.converged and not bool(
            getattr(self.config, "allow_nonconverged_export", False)
        ):
            param_delta = (
                "N/A"
                if fit_result.final_parameter_change is None
                else f"{fit_result.final_parameter_change:.3e}"
            )
            predictor_delta = (
                "N/A"
                if fit_result.final_predictor_change is None
                else f"{fit_result.final_predictor_change:.3e}"
            )
            objective_delta = (
                "N/A"
                if fit_result.final_objective_change is None
                else f"{fit_result.final_objective_change:.3e}"
            )
            hyper_delta = (
                "N/A"
                if fit_result.final_hyperparameter_change is None
                else f"{fit_result.final_hyperparameter_change:.3e}"
            )
            raise RuntimeError(
                "Fit did not converge (final params/objective changes: "
                f"param={param_delta}, predictor={predictor_delta}, "
                f"objective={objective_delta}, hyperparameter={hyper_delta}); "
                "refusing final artifact export. "
                "Set config.allow_nonconverged_export=True to override."
            )
        artifact = ModelArtifact(
            config=self.config,
            records=fitted_state.full_variant_records,
            means=fitted_state.preprocessor.means,
            scales=fitted_state.preprocessor.scales,
            alpha=fit_result.alpha,
            beta_reduced=fit_result.beta_reduced,
            beta_full=fitted_state.full_coefficients,
            beta_variance=fit_result.beta_variance,
            tie_map=fitted_state.tie_map,
            sigma_e2=fit_result.sigma_error2,
            prior_scales=fit_result.prior_scales,
            global_scale=fit_result.global_scale,
            class_tpb_shape_a=fit_result.class_tpb_shape_a,
            class_tpb_shape_b=fit_result.class_tpb_shape_b,
            scale_model_coefficients=fit_result.scale_model_coefficients,
            scale_model_feature_names=fit_result.scale_model_feature_names,
            objective_history=fit_result.objective_history,
            validation_history=fit_result.validation_history,
            fit_fingerprint=fit_fingerprint,
            converged=bool(fit_result.converged),
            selected_iteration_count=int(
                getattr(fit_result, "selected_iteration_count", len(fit_result.objective_history))
                or len(fit_result.objective_history)
            ),
            final_parameter_change=fit_result.final_parameter_change,
            final_predictor_change=fit_result.final_predictor_change,
            final_objective_change=fit_result.final_objective_change,
            final_hyperparameter_change=fit_result.final_hyperparameter_change,
        )
        save_artifact(path, artifact)

    @classmethod
    def load(cls, path: str | Path) -> BayesianPGS:
        artifact = load_artifact(path)
        loaded_model = cls(config=artifact.config)
        nonzero_coefficient_indices, nonzero_coefficients = _nonzero_coefficient_cache(artifact.beta_full)
        nonzero_means = np.asarray(artifact.means[nonzero_coefficient_indices], dtype=np.float32)
        nonzero_scales = np.asarray(artifact.scales[nonzero_coefficient_indices], dtype=np.float32)
        loaded_model.state = FittedState(
            variant_records=[],
            full_variant_records=artifact.records,
            active_variant_indices=np.where(artifact.tie_map.original_to_reduced >= 0)[0].astype(np.int32),
            preprocessor=Preprocessor(means=artifact.means, scales=artifact.scales),
            tie_map=artifact.tie_map,
            fit_result=VariationalFitResult(
                alpha=artifact.alpha,
                beta_reduced=artifact.beta_reduced,
                beta_variance=artifact.beta_variance,
                prior_scales=artifact.prior_scales,
                global_scale=artifact.global_scale,
                class_tpb_shape_a=artifact.class_tpb_shape_a,
                class_tpb_shape_b=artifact.class_tpb_shape_b,
                scale_model_coefficients=artifact.scale_model_coefficients,
                scale_model_feature_names=artifact.scale_model_feature_names,
                sigma_error2=artifact.sigma_e2,
                objective_history=artifact.objective_history,
                validation_history=artifact.validation_history,
                member_prior_variances=artifact.prior_scales,
                linear_predictor=None,
                selected_iteration_count=(
                    int(getattr(artifact, "selected_iteration_count", 0) or 0)
                    or len(artifact.objective_history)
                ),
                # Use the saved converged flag instead of hardcoding False
                # (older bug). Default to False only when reading legacy
                # artifacts that predate the convergence metadata fields.
                converged=bool(getattr(artifact, "converged", False)),
                final_parameter_change=getattr(artifact, "final_parameter_change", None),
                final_predictor_change=getattr(artifact, "final_predictor_change", None),
                final_objective_change=getattr(artifact, "final_objective_change", None),
                final_hyperparameter_change=getattr(
                    artifact, "final_hyperparameter_change", None
                ),
            ),
            full_coefficients=artifact.beta_full,
            nonzero_coefficient_indices=nonzero_coefficient_indices,
            nonzero_coefficients=nonzero_coefficients,
            nonzero_means=nonzero_means,
            nonzero_scales=nonzero_scales,
        )
        return loaded_model

    def training_decision_components(self) -> tuple[F32Array, F32Array] | None:
        fitted_state = self._require_state()
        if (
            fitted_state.training_genetic_score is None
            or fitted_state.training_covariate_score is None
        ):
            return None
        return fitted_state.training_genetic_score, fitted_state.training_covariate_score

    def coefficient_table(
        self,
        *,
        nonzero_only: bool = False,
        minimum_abs_beta: float = 0.0,
    ) -> list[dict[str, object]]:
        fitted_state = self._require_state()
        if minimum_abs_beta < 0.0:
            raise ValueError("minimum_abs_beta must be non-negative.")
        if len(fitted_state.full_variant_records) != len(fitted_state.full_coefficients):
            raise ValueError("Full variant records must align with full_coefficients.")
        if nonzero_only:
            if minimum_abs_beta == 0.0:
                row_indices = fitted_state.nonzero_coefficient_indices
            else:
                row_indices = np.flatnonzero(
                    np.abs(fitted_state.full_coefficients) > minimum_abs_beta
                ).astype(np.int32)
        elif minimum_abs_beta > 0.0:
            row_indices = np.flatnonzero(
                np.abs(fitted_state.full_coefficients) > minimum_abs_beta
            ).astype(np.int32)
        else:
            row_indices = np.arange(len(fitted_state.full_coefficients), dtype=np.int32)
        return [
            {
                "variant_id": fitted_state.full_variant_records[int(row_index)].variant_id,
                "variant_class": fitted_state.full_variant_records[int(row_index)].variant_class.value,
                "beta": float(fitted_state.full_coefficients[int(row_index)]),
                "chromosome": fitted_state.full_variant_records[int(row_index)].chromosome,
                "position": int(fitted_state.full_variant_records[int(row_index)].position),
                "length": float(fitted_state.full_variant_records[int(row_index)].length),
                "allele_frequency": float(fitted_state.full_variant_records[int(row_index)].allele_frequency),
            }
            for row_index in row_indices
        ]

    def _require_state(self) -> FittedState:
        if self.state is None:
            raise ValueError("Model is not fitted.")
        return self.state

    @staticmethod
    def _with_intercept(covariates: NDArray) -> F32Array:
        covariate_matrix = np.asarray(covariates, dtype=np.float32)
        if covariate_matrix.ndim != 2:
            raise ValueError("covariates must be 2D.")
        output = np.empty((covariate_matrix.shape[0], covariate_matrix.shape[1] + 1), dtype=np.float32)
        output[:, 0] = 1.0
        output[:, 1:] = covariate_matrix
        return output


def _runtime_tuned_config_for_fit(
    config: ModelConfig,
    genotype_matrix: RawGenotypeMatrix,
) -> tuple[ModelConfig, str | None]:
    policy = runtime_training_policy_for_fit(
        config=config,
        genotype_matrix=genotype_matrix,
    )
    return policy.tuned_config, runtime_training_policy_summary(policy, config)


def _nonzero_coefficient_cache(coefficients: NDArray) -> tuple[I32Array, F32Array]:
    coefficient_array = np.asarray(coefficients, dtype=np.float32)
    nonzero_indices = np.flatnonzero(np.abs(coefficient_array) > 0.0).astype(np.int32)
    return nonzero_indices, np.asarray(coefficient_array[nonzero_indices], dtype=np.float32)


def _training_linear_predictor_cache(
    fit_result: VariationalFitResult,
    covariate_matrix: NDArray,
    reduced_genotypes: StandardizedGenotypeMatrix | NDArray | None,
) -> F32Array:
    if fit_result.linear_predictor is not None:
        return np.asarray(fit_result.linear_predictor, dtype=np.float32)
    training_linear_predictor = np.asarray(covariate_matrix @ fit_result.alpha, dtype=np.float32)
    if reduced_genotypes is not None and fit_result.beta_reduced.shape[0] > 0:
        if isinstance(reduced_genotypes, StandardizedGenotypeMatrix):
            genetic_score = reduced_genotypes.matvec(
                fit_result.beta_reduced,
                batch_size=auto_batch_size(covariate_matrix.shape[0]),
            )
        else:
            genetic_score = np.asarray(reduced_genotypes, dtype=np.float32) @ fit_result.beta_reduced
        training_linear_predictor = training_linear_predictor + np.asarray(genetic_score, dtype=np.float32)
    return np.asarray(training_linear_predictor, dtype=np.float32)


def _empty_tie_map(original_variant_count: int) -> TieMap:
    return TieMap(
        kept_indices=np.zeros(0, dtype=np.int32),
        original_to_reduced=np.full(original_variant_count, -1, dtype=np.int32),
        reduced_to_group=[],
    )


def _fit_without_active_variants(
    covariates: NDArray,
    targets: NDArray,
    config: ModelConfig,
    validation_data: tuple[NDArray, NDArray, NDArray] | None,
    predictor_offset: NDArray | None = None,
    validation_offset: NDArray | None = None,
) -> VariationalFitResult:
    return fit_variational_em(
        genotypes=np.empty((covariates.shape[0], 0), dtype=np.float32),
        covariates=covariates,
        targets=targets,
        records=[],
        config=config,
        tie_map=_empty_tie_map(0),
        validation_data=validation_data,
        predictor_offset=predictor_offset,
        validation_offset=validation_offset,
    )


# Compute genotypes @ coefficients for a subset of variants, standardizing
# on the fly.  This avoids materializing the full standardized genotype
# matrix — instead we read raw genotypes in batches, standardize each batch,
# multiply by the corresponding coefficients, and accumulate the result.
# This is the inner loop of prediction and is I/O-bound for PLINK files.
def _raw_standardized_subset_matvec(
    raw_genotypes: RawGenotypeMatrix,
    variant_indices: NDArray,
    means: NDArray,
    scales: NDArray,
    coefficients: NDArray,
    batch_size: int,
) -> F32Array:
    if variant_indices.shape[0] == 0:
        return np.zeros(raw_genotypes.shape[0], dtype=np.float32)
    result = np.zeros(raw_genotypes.shape[0], dtype=np.float32)
    offset = 0
    for batch in raw_genotypes.iter_column_batches(variant_indices, batch_size=batch_size):
        batch_width = batch.variant_indices.shape[0]
        batch_slice = slice(offset, offset + batch_width)
        standardized_batch = _standardize_batch(batch.values, means[batch_slice], scales[batch_slice])
        result += standardized_batch @ coefficients[batch_slice]
        offset += batch_width
    return result


# The tie map was built in "active variant space" (indices 0..n_active-1).
# This function translates those indices back to the original variant numbering
# so the exported model can be applied to new genotype files that use the
# original variant ordering.
def _project_tie_map_to_original_space(
    reduced_tie_map: TieMap,
    active_variant_indices: NDArray,
    original_variant_count: int,
) -> TieMap:
    if _active_indices_cover_original(active_variant_indices, original_variant_count):
        return reduced_tie_map
    # Force the inputs into contiguous in-memory int32 arrays. The caller may
    # pass a numpy memmap (cache load path), and random fancy-indexing into a
    # memmap inside the per-group Python loop below can stall for minutes
    # under disk pressure. One bulk read here is fast.
    active_indices_array = np.ascontiguousarray(active_variant_indices, dtype=np.int32)
    kept_indices = active_indices_array[np.asarray(reduced_tie_map.kept_indices, dtype=np.int64)]
    original_to_reduced = np.full(original_variant_count, -1, dtype=np.int32)
    original_to_reduced[active_indices_array] = reduced_tie_map.original_to_reduced
    original_groups: list[TieGroup] = []
    if reduced_tie_map.reduced_to_group:
        # Hoist the per-group Python-level work out of the hot loop. For each
        # tie group we just want active_indices_array[group.member_indices]
        # and active_indices_array[group.representative_index]. Bulk-resolve
        # representatives across all groups in one vectorized gather, and only
        # the (small, variable-length) member arrays remain inside the loop.
        representative_indices = np.fromiter(
            (int(tie_group.representative_index) for tie_group in reduced_tie_map.reduced_to_group),
            dtype=np.int64,
            count=len(reduced_tie_map.reduced_to_group),
        )
        representative_originals = active_indices_array[representative_indices]
        for tie_group, representative_original in zip(
            reduced_tie_map.reduced_to_group,
            representative_originals,
        ):
            member_indices_local = np.asarray(tie_group.member_indices, dtype=np.int64)
            original_groups.append(
                TieGroup(
                    representative_index=int(representative_original),
                    member_indices=active_indices_array[member_indices_local].astype(np.int32, copy=False),
                    signs=np.asarray(tie_group.signs, dtype=np.float32),
                )
            )
    return TieMap(
        kept_indices=kept_indices.astype(np.int32, copy=False),
        original_to_reduced=original_to_reduced,
        reduced_to_group=original_groups,
    )


def _active_indices_cover_original(active_variant_indices: NDArray, original_variant_count: int) -> bool:
    return (
        active_variant_indices.shape == (int(original_variant_count),)
        and np.array_equal(active_variant_indices, np.arange(int(original_variant_count), dtype=np.int32))
    )


# When expanding a single group effect back to individual members, how much
# weight does each member get?  We split proportional to each member's prior
# variance — variants the model trusted more a priori get a larger share of
# the group's fitted coefficient.  This is fairer than equal splitting when
# group members differ in type (e.g. a deletion and a duplication tied
# together).
def _tie_group_export_weights(
    tie_map: TieMap,
    fit_result: VariationalFitResult,
) -> list[F32Array]:
    if not tie_map.reduced_to_group:
        return []
    member_prior_variances = np.asarray(fit_result.member_prior_variances, dtype=np.float32)
    group_weights: list[F32Array] = []
    for tie_group in tie_map.reduced_to_group:
        member_variances = np.asarray(member_prior_variances[tie_group.member_indices], dtype=np.float32)
        normalized_weights = member_variances / np.maximum(np.sum(member_variances), 1e-12)
        group_weights.append(normalized_weights.astype(np.float32))
    return group_weights


def _training_records_from_stats(
    records: Sequence[VariantRecord],
    variant_stats: VariantStatistics,
    variant_indices: NDArray,
) -> list[VariantRecord]:
    """Build training records using cohort-derived training statistics."""
    structural_variant_classes = ModelConfig.structural_variant_classes()
    allele_frequencies = variant_stats.allele_frequencies
    support_counts = variant_stats.support_counts
    training_records = cast(list[VariantRecord], [None] * int(len(variant_indices)))
    for output_index, variant_index in enumerate(variant_indices):
        variant_index_int = int(variant_index)
        record = records[variant_index_int]
        support = record.training_support
        if support is None and record.variant_class in structural_variant_classes:
            support = int(support_counts[variant_index_int])
        training_records[output_index] = (
            _training_record_with_stats(
                record,
                allele_frequency=float(allele_frequencies[variant_index_int]),
                training_support=support,
            )
        )
    log(f"  training records from stats: {len(training_records)} records [NO DATA PASS]")
    return training_records


def _training_record_with_stats(
    record: VariantRecord,
    *,
    allele_frequency: float,
    training_support: int | None,
) -> VariantRecord:
    training_record = object.__new__(VariantRecord)
    training_record.variant_id = record.variant_id
    training_record.variant_class = record.variant_class
    training_record.chromosome = record.chromosome
    training_record.position = record.position
    training_record.length = record.length
    training_record.allele_frequency = allele_frequency
    training_record.quality = record.quality
    training_record.training_support = training_support
    training_record.is_repeat = record.is_repeat
    training_record.is_copy_number = record.is_copy_number
    training_record.prior_binary_features = record.prior_binary_features
    training_record.prior_continuous_features = record.prior_continuous_features
    training_record.prior_categorical_features = record.prior_categorical_features
    training_record.prior_membership_features = record.prior_membership_features
    training_record.prior_nested_features = record.prior_nested_features
    training_record.prior_nested_membership_features = record.prior_nested_membership_features
    training_record.prior_class_members = record.prior_class_members
    training_record.prior_class_membership = record.prior_class_membership
    return training_record


def _normalize_variant_records(records: Sequence[VariantRecord | dict[str, Any]]) -> list[VariantRecord]:
    if isinstance(records, list) and (not records or isinstance(records[0], VariantRecord)):
        return cast(list[VariantRecord], records)
    return [normalize_variant_record(record) for record in records]


def _tie_map_keeps_all_active_variants(tie_map: TieMap, active_variant_count: int) -> bool:
    return (
        tie_map.kept_indices.shape == (int(active_variant_count),)
        and tie_map.original_to_reduced.shape == (int(active_variant_count),)
        and np.array_equal(tie_map.kept_indices, np.arange(int(active_variant_count), dtype=np.int32))
        and np.array_equal(tie_map.original_to_reduced, np.arange(int(active_variant_count), dtype=np.int32))
    )

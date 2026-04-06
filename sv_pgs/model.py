from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import pickle
from typing import Sequence

import numpy as np

import sv_pgs._jax  # noqa: F401

from sv_pgs.artifact import ModelArtifact, load_artifact, save_artifact
from sv_pgs.config import ModelConfig, TraitType
from sv_pgs.data import TieGroup, TieMap, VariantRecord, VariantStatistics, normalize_variant_records
from sv_pgs.genotype import (
    ConcatenatedRawGenotypeMatrix,
    DenseRawGenotypeMatrix,
    Int8RawGenotypeMatrix,
    PlinkRawGenotypeMatrix,
    RawGenotypeMatrix,
    StandardizedGenotypeMatrix,
    _gpu_materialization_budget_bytes,
    _try_import_cupy,
    _standardize_batch,
    as_raw_genotype_matrix,
    auto_batch_size,
)
from sv_pgs.inference import VariationalFitCheckpoint, VariationalFitResult, fit_variational_em
from sv_pgs.numeric import stable_sigmoid
from sv_pgs.preprocessing import (
    Preprocessor,
    build_tie_map,
    compute_variant_statistics,
    fit_preprocessor_from_stats,
    select_active_variant_indices,
)
from sv_pgs.progress import log, mem

@dataclass(slots=True)
class FittedState:
    variant_records: list[VariantRecord]
    active_variant_indices: np.ndarray
    preprocessor: Preprocessor
    tie_map: TieMap
    fit_result: VariationalFitResult
    full_coefficients: np.ndarray
    nonzero_coefficient_indices: np.ndarray
    nonzero_coefficients: np.ndarray
    nonzero_means: np.ndarray
    nonzero_scales: np.ndarray


GPU_EXACT_SOLVER_LIMIT = 1_024
GPU_PRECONDITIONER_RANK_LIMIT = 1_024
_FIT_STAGE_CACHE_DIRNAME = ".sv_pgs_cache"
_FIT_STAGE_CACHE_SUBDIR = "fit_stage"
_FIT_STAGE_CACHE_VERSION = 2


@dataclass(slots=True)
class _FitStageCachePaths:
    key: str
    cache_dir: Path
    manifest_path: Path
    active_indices_path: Path
    tie_map_path: Path
    reduced_raw_i8_path: Path
    em_checkpoint_path: Path


def _validate_fit_inputs(
    genotype_matrix: RawGenotypeMatrix,
    covariates: np.ndarray,
    targets: np.ndarray,
    variant_records: Sequence[VariantRecord | dict],
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
    allele_frequencies: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
    covariates: np.ndarray,
    targets: np.ndarray,
    config: ModelConfig,
) -> _FitStageCachePaths | None:
    raw_signature = _persistent_raw_signature(genotype_matrix)
    if raw_signature is None:
        return None
    key_hasher = hashlib.sha256()
    key_hasher.update(f"fit-stage-cache-v{_FIT_STAGE_CACHE_VERSION}".encode("utf-8"))
    key_hasher.update(raw_signature.encode("utf-8"))
    key_hasher.update(config.trait_type.value.encode("utf-8"))
    key_hasher.update(np.asarray([config.minimum_minor_allele_frequency], dtype=np.float64).tobytes())
    _update_hash_with_array_bytes(key_hasher, np.asarray(allele_frequencies, dtype=np.float32))
    _update_hash_with_array_bytes(key_hasher, np.asarray(means, dtype=np.float32))
    _update_hash_with_array_bytes(key_hasher, np.asarray(scales, dtype=np.float32))
    _update_hash_with_array_bytes(key_hasher, np.asarray(covariates, dtype=np.float32))
    _update_hash_with_array_bytes(key_hasher, np.asarray(targets, dtype=np.float32))
    key = key_hasher.hexdigest()[:24]
    cache_dir = Path.cwd() / _FIT_STAGE_CACHE_DIRNAME / _FIT_STAGE_CACHE_SUBDIR
    return _FitStageCachePaths(
        key=key,
        cache_dir=cache_dir,
        manifest_path=cache_dir / f"{key}.manifest.json",
        active_indices_path=cache_dir / f"{key}.active.npy",
        tie_map_path=cache_dir / f"{key}.tie.pkl",
        reduced_raw_i8_path=cache_dir / f"{key}.reduced_raw_i8.npy",
        em_checkpoint_path=cache_dir / f"{key}.em.pkl",
    )


def _update_hash_with_array_bytes(hasher, array: np.ndarray) -> None:
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
    if isinstance(genotype_matrix, (Int8RawGenotypeMatrix, DenseRawGenotypeMatrix)):
        backing_matrix = np.asanyarray(genotype_matrix.matrix)
        if isinstance(backing_matrix, np.memmap):
            backing_path = Path(str(backing_matrix.filename)).resolve()
            stat = backing_path.stat()
            return f"memmap:{backing_path}:{stat.st_size}:{stat.st_mtime_ns}:{backing_matrix.shape}:{backing_matrix.dtype}"
    return None


def _save_fit_stage_structure_cache(
    cache_paths: _FitStageCachePaths,
    active_variant_indices: np.ndarray,
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


def _try_load_fit_stage_cache(
    cache_paths: _FitStageCachePaths,
    prepared_arrays,
    standardized_genotypes: StandardizedGenotypeMatrix,
) -> tuple[np.ndarray, TieMap, StandardizedGenotypeMatrix, bool] | None:
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
        active_variant_indices = np.load(cache_paths.active_indices_path, mmap_mode="r").astype(np.int32, copy=False)
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
        if len(reduced_tie_map.reduced_to_group) != reduced_tie_map.kept_indices.shape[0]:
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
            )
            log(f"fit-stage cache hit — loading from {cache_paths.cache_dir.name}/{cache_paths.key}.*")
            return active_variant_indices, reduced_tie_map, reduced_genotypes, True
        log(f"fit-stage structure cache hit — loading from {cache_paths.cache_dir.name}/{cache_paths.key}.*")
        return active_variant_indices, reduced_tie_map, standardized_genotypes.subset(combined_indices), False
    except Exception as exc:
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
        except Exception:
            pass


def _save_variational_checkpoint(
    cache_paths: _FitStageCachePaths,
    checkpoint: VariationalFitCheckpoint,
) -> None:
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
    except Exception as exc:
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
        log(
            "variational checkpoint restored: "
            + f"{cache_paths.cache_dir.name}/{cache_paths.key}.em.pkl "
            + f"(completed_iterations={checkpoint.completed_iterations})"
        )
        return checkpoint
    except Exception as exc:
        log(f"variational checkpoint load failed ({exc}); discarding stale checkpoint")
        _clear_variational_checkpoint(cache_paths)
        return None


def _clear_variational_checkpoint(cache_paths: _FitStageCachePaths) -> None:
    if cache_paths.em_checkpoint_path.exists():
        cache_paths.em_checkpoint_path.unlink()


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
        genotypes: RawGenotypeMatrix | np.ndarray,
        covariates: np.ndarray,
        targets: np.ndarray,
        variant_records: Sequence[VariantRecord | dict],
        validation_data: tuple[RawGenotypeMatrix | np.ndarray, np.ndarray, np.ndarray] | None = None,
        variant_stats: VariantStatistics | None = None,
    ) -> BayesianPGS:
        log(f"=== MODEL FIT START ===  genotypes={genotypes.shape}  covariates={covariates.shape}  targets={targets.shape}  pre_computed_stats={'YES' if variant_stats else 'NO'}")
        raw_genotype_matrix = as_raw_genotype_matrix(genotypes)
        _validate_fit_inputs(
            genotype_matrix=raw_genotype_matrix,
            covariates=covariates,
            targets=targets,
            variant_records=variant_records,
            variant_stats=variant_stats,
        )
        tuned_config, tuning_summary = _runtime_tuned_config_for_fit(
            config=self.config,
            genotype_matrix=raw_genotype_matrix,
        )
        self.config = tuned_config
        if tuning_summary is not None:
            log(tuning_summary)
        covariate_matrix = self._with_intercept(covariates)
        selection_records = normalize_variant_records(variant_records)

        # Use pre-computed stats if available (saves 3 full data passes)
        if variant_stats is not None:
            log("using pre-computed variant statistics (means, scales, support) [NO DATA PASSES]")
            normalized_records = _training_records_from_stats(selection_records, variant_stats)
            prepared_arrays = fit_preprocessor_from_stats(variant_stats, covariate_matrix, targets)
        else:
            log("computing variant statistics in-fit so support/standardization share one pass...")
            variant_stats = compute_variant_statistics(
                raw_genotypes=raw_genotype_matrix,
                config=self.config,
            )
            normalized_records = _training_records_from_stats(selection_records, variant_stats)
            prepared_arrays = fit_preprocessor_from_stats(variant_stats, covariate_matrix, targets)
        preprocessor = Preprocessor(means=prepared_arrays.means, scales=prepared_arrays.scales)
        log(f"preprocessor ready  {len(normalized_records)} variant records  mem={mem()}")

        log("creating standardized genotype view...")
        standardized_genotypes = raw_genotype_matrix.standardized(prepared_arrays.means, prepared_arrays.scales)
        fit_stage_cache_paths = _fit_stage_cache_paths(
            genotype_matrix=raw_genotype_matrix,
            allele_frequencies=np.asarray([record.allele_frequency for record in normalized_records], dtype=np.float32),
            means=prepared_arrays.means,
            scales=prepared_arrays.scales,
            covariates=prepared_arrays.covariates,
            targets=prepared_arrays.targets,
            config=self.config,
        )
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
            log("selecting active variant indices...")
            active_variant_indices = select_active_variant_indices(
                variant_records=normalized_records,
                config=self.config,
                standardized_genotypes=standardized_genotypes,
                covariates=prepared_arrays.covariates,
                targets=prepared_arrays.targets,
                trait_type=self.config.trait_type,
            )
            log(f"active variants: {len(active_variant_indices)} / {len(normalized_records)} ({100.0*len(active_variant_indices)/max(len(normalized_records),1):.1f}%)")
        else:
            active_variant_indices, reduced_tie_map, reduced_genotypes, local_cache = cached_fit_stage
            log(
                "active/tie cache restored: "
                + f"{len(active_variant_indices)} active -> {len(reduced_tie_map.kept_indices)} unique  mem={mem()}"
            )
        if active_variant_indices.size == 0:
            log("no active variants remain after filtering; fitting covariates-only model...")
            reduced_validation_dense: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
            if validation_data is not None:
                _validation_genotypes, validation_covariates, validation_targets = validation_data
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
            )
            full_coefficients = np.zeros(len(normalized_records), dtype=np.float32)
            nonzero_coefficient_indices, nonzero_coefficients = _nonzero_coefficient_cache(full_coefficients)
            self.state = FittedState(
                variant_records=normalized_records,
                active_variant_indices=np.zeros(0, dtype=np.int32),
                preprocessor=preprocessor,
                tie_map=_empty_tie_map(len(normalized_records)),
                fit_result=fit_result,
                full_coefficients=full_coefficients,
                nonzero_coefficient_indices=nonzero_coefficient_indices,
                nonzero_coefficients=nonzero_coefficients,
                nonzero_means=np.zeros(0, dtype=np.float32),
                nonzero_scales=np.zeros(0, dtype=np.float32),
            )
            if fit_stage_cache_paths is not None:
                _clear_variational_checkpoint(fit_stage_cache_paths)
            log(f"coefficients: 0 non-zero out of {len(normalized_records)} total")
            log(f"=== MODEL FIT DONE ===  mem={mem()}")
            return self

        active_records = [normalized_records[int(variant_index)] for variant_index in active_variant_indices]
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
            combined_indices = active_variant_indices[reduced_tie_map.kept_indices]
            reduced_genotypes = standardized_genotypes.subset(combined_indices)
            local_cache = False
        else:
            active_genotypes = None
        original_space_tie_map = _project_tie_map_to_original_space(
            reduced_tie_map=reduced_tie_map,
            active_variant_indices=active_variant_indices,
            original_variant_count=len(normalized_records),
        )
        log(f"tie map: {len(active_variant_indices)} active -> {len(reduced_tie_map.kept_indices)} unique ({len(reduced_tie_map.reduced_to_group)} groups)  mem={mem()}")

        reduced_validation = None
        if validation_data is not None:
            validation_genotypes, validation_covariates, validation_targets = validation_data
            standardized_validation = as_raw_genotype_matrix(validation_genotypes).standardized(
                prepared_arrays.means,
                prepared_arrays.scales,
            )
            combined_validation_indices = active_variant_indices[reduced_tie_map.kept_indices]
            reduced_validation = (
                standardized_validation.subset(combined_validation_indices),
                self._with_intercept(np.asarray(validation_covariates, dtype=np.float32)),
                np.asarray(validation_targets, dtype=np.float32),
            )

        # Materialize the reduced genotype matrix (RAM or GPU via CuPy).
        in_memory = reduced_genotypes.try_materialize_gpu()
        if not in_memory:
            in_memory = reduced_genotypes.try_materialize()
        if in_memory:
            # After materialization, reduced_genotypes no longer needs raw.
            reduced_genotypes.release_raw_storage()
        else:
            if not local_cache and fit_stage_cache_paths is not None:
                local_cache = reduced_genotypes.try_cache_persistently(fit_stage_cache_paths.reduced_raw_i8_path)
                if local_cache:
                    _write_fit_stage_cache_manifest(
                        cache_paths=fit_stage_cache_paths,
                        active_variant_count=int(active_variant_indices.shape[0]),
                        reduced_variant_count=int(reduced_tie_map.kept_indices.shape[0]),
                        has_reduced_raw_i8=True,
                    )
            if not local_cache:
                local_cache = reduced_genotypes.try_cache_locally()
            if not local_cache:
                log("keeping reduced genotype matrix streaming (no RAM/GPU/local cache)  mem=" + mem())
        del raw_genotype_matrix, standardized_genotypes, active_genotypes
        import gc
        gc.collect()
        log(f"memory freed after materialization  mem={mem()}")
        log(
            f"starting variational EM  max_iterations={self.config.max_outer_iterations}  "
            f"reduced_matrix={reduced_genotypes.shape}  in_memory={in_memory}  "
            f"local_cache={local_cache}  "
            f"on_gpu={reduced_genotypes._cupy_cache is not None}  "
            f"mem={mem()}"
        )
        resume_checkpoint = (
            None
            if fit_stage_cache_paths is None
            else _try_load_variational_checkpoint(fit_stage_cache_paths)
        )
        fit_result = fit_variational_em(
            genotypes=reduced_genotypes,
            covariates=prepared_arrays.covariates,
            targets=prepared_arrays.targets,
            records=active_records,
            tie_map=reduced_tie_map,
            config=self.config,
            validation_data=reduced_validation,
            resume_checkpoint=resume_checkpoint,
            checkpoint_callback=(
                None
                if fit_stage_cache_paths is None
                else lambda checkpoint: _save_variational_checkpoint(fit_stage_cache_paths, checkpoint)
            ),
        )
        if fit_stage_cache_paths is not None:
            _clear_variational_checkpoint(fit_stage_cache_paths)
        log(f"variational EM converged in {len(fit_result.objective_history)} iterations  final_obj={fit_result.objective_history[-1]:.4f}  mem={mem()}")

        log("expanding coefficients from reduced to full space...")
        tie_group_weights = _tie_group_export_weights(
            tie_map=reduced_tie_map,
            fit_result=fit_result,
        )
        active_coefficients = reduced_tie_map.expand_coefficients(
            fit_result.beta_reduced,
            group_weights=tie_group_weights,
        )
        full_coefficients = np.zeros(len(normalized_records), dtype=np.float32)
        full_coefficients[active_variant_indices] = active_coefficients
        nonzero_coefficient_indices, nonzero_coefficients = _nonzero_coefficient_cache(full_coefficients)
        nonzero_means = np.asarray(prepared_arrays.means[nonzero_coefficient_indices], dtype=np.float32)
        nonzero_scales = np.asarray(prepared_arrays.scales[nonzero_coefficient_indices], dtype=np.float32)
        nonzero_count = int(np.count_nonzero(full_coefficients))
        log(f"coefficients: {nonzero_count} non-zero out of {len(normalized_records)} total")

        self.state = FittedState(
            variant_records=normalized_records,
            active_variant_indices=active_variant_indices,
            preprocessor=preprocessor,
            tie_map=original_space_tie_map,
            fit_result=fit_result,
            full_coefficients=full_coefficients,
            nonzero_coefficient_indices=nonzero_coefficient_indices,
            nonzero_coefficients=nonzero_coefficients,
            nonzero_means=nonzero_means,
            nonzero_scales=nonzero_scales,
        )
        log(f"=== MODEL FIT DONE ===  mem={mem()}")
        return self

    def decision_function(self, genotypes: RawGenotypeMatrix | np.ndarray, covariates: np.ndarray) -> np.ndarray:
        """Compute the raw linear predictor (before sigmoid for binary traits).

        For each individual: score = sum_j(beta_j * standardized_genotype_j) + covariates @ alpha.
        Only reads variants with non-zero coefficients (typically <1% of all variants),
        making prediction fast even on huge genotype files.
        """
        genotype_component, covariate_component = self.decision_components(genotypes, covariates)
        return genotype_component + covariate_component

    def decision_components(
        self,
        genotypes: RawGenotypeMatrix | np.ndarray,
        covariates: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
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

    def predict_proba(self, genotypes: RawGenotypeMatrix | np.ndarray, covariates: np.ndarray) -> np.ndarray:
        if self.config.trait_type != TraitType.BINARY:
            raise ValueError("predict_proba is only available for binary traits.")
        linear_predictor = self.decision_function(genotypes, covariates)
        positive_probability = np.asarray(stable_sigmoid(linear_predictor), dtype=np.float32)
        return np.column_stack([1.0 - positive_probability, positive_probability])

    def predict(self, genotypes: RawGenotypeMatrix | np.ndarray, covariates: np.ndarray) -> np.ndarray:
        linear_predictor = self.decision_function(genotypes, covariates)
        if self.config.trait_type == TraitType.BINARY:
            return (np.asarray(stable_sigmoid(linear_predictor), dtype=np.float32) >= 0.5).astype(np.int32)
        return linear_predictor

    def export(self, path: str | Path) -> None:
        fitted_state = self._require_state()
        artifact = ModelArtifact(
            config=self.config,
            records=fitted_state.variant_records,
            means=fitted_state.preprocessor.means,
            scales=fitted_state.preprocessor.scales,
            alpha=fitted_state.fit_result.alpha,
            beta_reduced=fitted_state.fit_result.beta_reduced,
            beta_full=fitted_state.full_coefficients,
            beta_variance=fitted_state.fit_result.beta_variance,
            tie_map=fitted_state.tie_map,
            sigma_e2=fitted_state.fit_result.sigma_error2,
            prior_scales=fitted_state.fit_result.prior_scales,
            global_scale=fitted_state.fit_result.global_scale,
            class_tpb_shape_a=fitted_state.fit_result.class_tpb_shape_a,
            class_tpb_shape_b=fitted_state.fit_result.class_tpb_shape_b,
            scale_model_coefficients=fitted_state.fit_result.scale_model_coefficients,
            scale_model_feature_names=fitted_state.fit_result.scale_model_feature_names,
            objective_history=fitted_state.fit_result.objective_history,
            validation_history=fitted_state.fit_result.validation_history,
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
            variant_records=artifact.records,
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
            ),
            full_coefficients=artifact.beta_full,
            nonzero_coefficient_indices=nonzero_coefficient_indices,
            nonzero_coefficients=nonzero_coefficients,
            nonzero_means=nonzero_means,
            nonzero_scales=nonzero_scales,
        )
        return loaded_model

    def coefficient_table(self) -> list[dict[str, object]]:
        fitted_state = self._require_state()
        return [
            {
                "variant_id": variant_record.variant_id,
                "variant_class": variant_record.variant_class.value,
                "beta": float(coefficient),
            }
            for variant_record, coefficient in zip(
                fitted_state.variant_records,
                fitted_state.full_coefficients,
                strict=True,
            )
        ]

    def _require_state(self) -> FittedState:
        if self.state is None:
            raise ValueError("Model is not fitted.")
        return self.state

    @staticmethod
    def _with_intercept(covariates: np.ndarray) -> np.ndarray:
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
    cupy = _try_import_cupy()
    if cupy is None:
        return config, None
    sample_count = int(genotype_matrix.shape[0])
    if sample_count < 1:
        return config, None
    gpu_budget_bytes = _gpu_materialization_budget_bytes(cupy)
    cacheable_dense_variants = max(int(gpu_budget_bytes // max(sample_count * 4, 1)), 1)
    tuned_exact_solver_limit = min(
        int(config.exact_solver_matrix_limit),
        max(int(cacheable_dense_variants * 0.9), 1),
        GPU_EXACT_SOLVER_LIMIT,
    )
    max_gpu_preconditioner_rank = max(1, min(cacheable_dense_variants, GPU_PRECONDITIONER_RANK_LIMIT))
    target_preconditioner_rank = int(config.sample_space_preconditioner_rank)
    if sample_count >= 32_768 and genotype_matrix.shape[1] >= 65_536:
        target_preconditioner_rank = max(target_preconditioner_rank, GPU_PRECONDITIONER_RANK_LIMIT)
    tuned_preconditioner_rank = min(target_preconditioner_rank, max_gpu_preconditioner_rank)
    if (
        tuned_exact_solver_limit == int(config.exact_solver_matrix_limit)
        and tuned_preconditioner_rank == int(config.sample_space_preconditioner_rank)
    ):
        return config, (
            "GPU runtime profile active: "
            + f"gpu_budget={gpu_budget_bytes / 1e9:.1f} GB "
            + f"cacheable_dense_variants~{cacheable_dense_variants} "
            + "(user config already fits GPU profile)"
        )
    tuned_config = replace(
        config,
        exact_solver_matrix_limit=tuned_exact_solver_limit,
        sample_space_preconditioner_rank=tuned_preconditioner_rank,
    )
    return tuned_config, (
        "GPU runtime profile active: "
        + f"gpu_budget={gpu_budget_bytes / 1e9:.1f} GB "
        + f"cacheable_dense_variants~{cacheable_dense_variants} "
        + f"exact_solver_matrix_limit={config.exact_solver_matrix_limit}->{tuned_exact_solver_limit} "
        + f"sample_space_preconditioner_rank={config.sample_space_preconditioner_rank}->{tuned_preconditioner_rank}"
    )


def _nonzero_coefficient_cache(coefficients: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coefficient_array = np.asarray(coefficients, dtype=np.float32)
    nonzero_indices = np.flatnonzero(np.abs(coefficient_array) > 0.0).astype(np.int32)
    return nonzero_indices, np.asarray(coefficient_array[nonzero_indices], dtype=np.float32)


def _empty_tie_map(original_variant_count: int) -> TieMap:
    return TieMap(
        kept_indices=np.zeros(0, dtype=np.int32),
        original_to_reduced=np.full(original_variant_count, -1, dtype=np.int32),
        reduced_to_group=[],
    )


def _fit_without_active_variants(
    covariates: np.ndarray,
    targets: np.ndarray,
    config: ModelConfig,
    validation_data: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
) -> VariationalFitResult:
    return fit_variational_em(
        genotypes=np.empty((covariates.shape[0], 0), dtype=np.float32),
        covariates=covariates,
        targets=targets,
        records=[],
        config=config,
        tie_map=_empty_tie_map(0),
        validation_data=validation_data,
    )


# Compute genotypes @ coefficients for a subset of variants, standardizing
# on the fly.  This avoids materializing the full standardized genotype
# matrix — instead we read raw genotypes in batches, standardize each batch,
# multiply by the corresponding coefficients, and accumulate the result.
# This is the inner loop of prediction and is I/O-bound for PLINK files.
def _raw_standardized_subset_matvec(
    raw_genotypes: RawGenotypeMatrix,
    variant_indices: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
    coefficients: np.ndarray,
    batch_size: int,
) -> np.ndarray:
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
    active_variant_indices: np.ndarray,
    original_variant_count: int,
) -> TieMap:
    kept_indices = active_variant_indices[reduced_tie_map.kept_indices]
    original_to_reduced = np.full(original_variant_count, -1, dtype=np.int32)
    original_to_reduced[active_variant_indices] = reduced_tie_map.original_to_reduced
    original_groups: list[TieGroup] = []
    for tie_group in reduced_tie_map.reduced_to_group:
        original_groups.append(
            TieGroup(
                representative_index=int(active_variant_indices[tie_group.representative_index]),
                member_indices=active_variant_indices[tie_group.member_indices].astype(np.int32),
                signs=tie_group.signs.astype(np.float32),
            )
        )
    return TieMap(
        kept_indices=kept_indices.astype(np.int32),
        original_to_reduced=original_to_reduced,
        reduced_to_group=original_groups,
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
) -> list[np.ndarray]:
    member_prior_variances = np.asarray(fit_result.member_prior_variances, dtype=np.float32)
    group_weights: list[np.ndarray] = []
    for tie_group in tie_map.reduced_to_group:
        member_variances = np.asarray(member_prior_variances[tie_group.member_indices], dtype=np.float32)
        normalized_weights = member_variances / np.maximum(np.sum(member_variances), 1e-12)
        group_weights.append(normalized_weights.astype(np.float32))
    return group_weights


def _training_records_from_stats(
    records: Sequence[VariantRecord],
    variant_stats: VariantStatistics,
) -> list[VariantRecord]:
    """Build training records using cohort-derived training statistics."""
    training_records: list[VariantRecord] = []
    for variant_index, record in enumerate(records):
        support = record.training_support
        if support is None and record.variant_class in ModelConfig.structural_variant_classes():
            support = int(variant_stats.support_counts[variant_index])
        training_records.append(
            VariantRecord(
                variant_id=record.variant_id,
                variant_class=record.variant_class,
                chromosome=record.chromosome,
                position=record.position,
                length=record.length,
                allele_frequency=float(variant_stats.allele_frequencies[variant_index]),
                quality=record.quality,
                training_support=support,
                is_repeat=record.is_repeat,
                is_copy_number=record.is_copy_number,
                prior_continuous_features=dict(record.prior_continuous_features),
                prior_class_members=record.prior_class_members,
                prior_class_membership=record.prior_class_membership,
            )
        )
    log(f"  training records from stats: {len(training_records)} records [NO DATA PASS]")
    return training_records

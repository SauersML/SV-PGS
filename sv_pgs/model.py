from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from sv_pgs.artifact import ModelArtifact, load_artifact, save_artifact
from sv_pgs.config import ModelConfig, TraitType
from sv_pgs.data import TieGroup, TieMap, VariantRecord, VariantStatistics, normalize_variant_records
from sv_pgs.genotype import RawGenotypeMatrix, StandardizedGenotypeMatrix, as_raw_genotype_matrix
from sv_pgs.inference import VariationalFitResult, fit_variational_em
from sv_pgs.numeric import stable_sigmoid
from sv_pgs.preprocessing import (
    Preprocessor,
    _infer_support_count_from_raw_genotypes,
    build_tie_map,
    compute_variant_statistics,
    fit_preprocessor,
    fit_preprocessor_from_stats,
    select_active_variant_indices,
)

STRUCTURAL_VARIANT_CLASSES = set(ModelConfig.structural_variant_classes())


@dataclass(slots=True)
class FittedState:
    variant_records: list[VariantRecord]
    active_variant_indices: np.ndarray
    preprocessor: Preprocessor
    tie_map: TieMap
    fit_result: VariationalFitResult
    full_coefficients: np.ndarray


class BayesianPGS:
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.state: FittedState | None = None

    def fit(
        self,
        genotypes: RawGenotypeMatrix | np.ndarray,
        covariates: np.ndarray,
        targets: np.ndarray,
        variant_records: Sequence[VariantRecord | dict],
        validation_data: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
        variant_stats: VariantStatistics | None = None,
    ) -> BayesianPGS:
        from sv_pgs.progress import log, mem
        log(f"=== MODEL FIT START ===  genotypes={genotypes.shape}  covariates={covariates.shape}  targets={targets.shape}  pre_computed_stats={'YES' if variant_stats else 'NO'}")
        raw_genotype_matrix = as_raw_genotype_matrix(genotypes)

        # Use pre-computed stats if available (saves 3 full data passes)
        if variant_stats is not None:
            log("using pre-computed variant statistics (means, scales, support) [NO DATA PASSES]")
            normalized_records = _training_records_from_stats(
                normalize_variant_records(variant_records), variant_stats,
            )
            covariate_matrix = self._with_intercept(covariates)
            prepared_arrays = fit_preprocessor_from_stats(variant_stats, covariate_matrix, targets)
        else:
            log("normalizing variant records and computing training support...")
            normalized_records = _training_records(raw_genotype_matrix, normalize_variant_records(variant_records), self.config)
            covariate_matrix = self._with_intercept(covariates)
            log(f"fitting preprocessor (streaming mean/scale over {raw_genotype_matrix.shape[1]} variants)...")
            prepared_arrays = fit_preprocessor(raw_genotype_matrix, covariate_matrix, targets, self.config)
        preprocessor = Preprocessor(means=prepared_arrays.means, scales=prepared_arrays.scales)
        log(f"preprocessor ready  {len(normalized_records)} variant records  mem={mem()}")

        log("creating standardized genotype view...")
        standardized_genotypes = raw_genotype_matrix.standardized(prepared_arrays.means, prepared_arrays.scales)

        log("selecting active variant indices (filtering low-carrier SVs)...")
        active_variant_indices = select_active_variant_indices(
            variant_records=normalized_records,
            support_counts=variant_stats.support_counts if variant_stats is not None else _compute_support_counts(raw_genotype_matrix, normalized_records, self.config),
            config=self.config,
        )
        if active_variant_indices.shape[0] > self.config.maximum_active_variants:
            log(
                f"screening active variants down to top {self.config.maximum_active_variants} "
                f"from {active_variant_indices.shape[0]} using marginal scores..."
            )
            active_variant_indices = _screen_active_variant_indices(
                raw_genotypes=raw_genotype_matrix,
                active_variant_indices=active_variant_indices,
                covariates=prepared_arrays.covariates,
                targets=prepared_arrays.targets,
                means=prepared_arrays.means,
                scales=prepared_arrays.scales,
                maximum_active_variants=self.config.maximum_active_variants,
                batch_size=self.config.genotype_batch_size,
            )
        log(f"active variants: {len(active_variant_indices)} / {len(normalized_records)} ({100.0*len(active_variant_indices)/max(len(normalized_records),1):.1f}%)")
        active_records = [normalized_records[int(variant_index)] for variant_index in active_variant_indices]
        active_genotypes = standardized_genotypes.subset(active_variant_indices)

        if active_variant_indices.shape[0] > self.config.maximum_tie_map_variants:
            log(
                f"skipping tie map for {active_variant_indices.shape[0]} active variants "
                f"(limit={self.config.maximum_tie_map_variants}); using identity mapping"
            )
            reduced_tie_map = _identity_tie_map(active_variant_indices.shape[0])
        else:
            log("building tie map (detecting identical/negated genotype columns)...")
            reduced_tie_map = build_tie_map(active_genotypes, active_records, self.config)
        original_space_tie_map = _project_tie_map_to_original_space(
            reduced_tie_map=reduced_tie_map,
            active_variant_indices=active_variant_indices,
            original_variant_count=len(normalized_records),
        )
        reduced_genotypes = active_genotypes.subset(reduced_tie_map.kept_indices)
        log(f"tie map: {len(active_variant_indices)} active -> {len(reduced_tie_map.kept_indices)} unique ({len(reduced_tie_map.reduced_to_group)} groups)  mem={mem()}")

        reduced_validation = None
        if validation_data is not None:
            validation_genotypes, validation_covariates, validation_targets = validation_data
            standardized_validation = preprocessor.transform(np.asarray(validation_genotypes, dtype=np.float32))
            if not isinstance(standardized_validation, np.ndarray):
                raise RuntimeError("validation_data must produce an in-memory standardized matrix.")
            reduced_validation = (
                standardized_validation[:, active_variant_indices][:, reduced_tie_map.kept_indices],
                self._with_intercept(np.asarray(validation_covariates, dtype=np.float32)),
                np.asarray(validation_targets, dtype=np.float32),
            )

        log(f"starting variational EM  max_iterations={self.config.max_outer_iterations}  reduced_matrix={reduced_genotypes.shape}  mem={mem()}")
        fit_result = fit_variational_em(
            genotypes=reduced_genotypes,
            covariates=prepared_arrays.covariates,
            targets=prepared_arrays.targets,
            records=active_records,
            tie_map=reduced_tie_map,
            config=self.config,
            validation_data=reduced_validation,
        )
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
        nonzero_count = int(np.count_nonzero(full_coefficients))
        log(f"coefficients: {nonzero_count} non-zero out of {len(normalized_records)} total")

        self.state = FittedState(
            variant_records=normalized_records,
            active_variant_indices=active_variant_indices,
            preprocessor=preprocessor,
            tie_map=original_space_tie_map,
            fit_result=fit_result,
            full_coefficients=full_coefficients,
        )
        log(f"=== MODEL FIT DONE ===  mem={mem()}")
        return self

    def decision_function(self, genotypes: RawGenotypeMatrix | np.ndarray, covariates: np.ndarray) -> np.ndarray:
        from sv_pgs.progress import log, mem
        log(f"decision_function: computing predictions for {genotypes.shape[0]} samples...  mem={mem()}")
        fitted_state = self._require_state()
        standardized_genotypes = fitted_state.preprocessor.transform(genotypes)
        covariate_matrix = self._with_intercept(np.asarray(covariates, dtype=np.float32))
        if isinstance(standardized_genotypes, StandardizedGenotypeMatrix):
            log(f"  streaming matvec over {standardized_genotypes.shape[1]} standardized variants...")
            genotype_component = np.asarray(
                standardized_genotypes.matvec(fitted_state.full_coefficients, batch_size=self.config.genotype_batch_size),
                dtype=np.float32,
            )
        else:
            log(f"  dense matvec: {standardized_genotypes.shape}")
            genotype_component = np.asarray(standardized_genotypes @ fitted_state.full_coefficients, dtype=np.float32)
        log(f"  decision_function done  mem={mem()}")
        return genotype_component + covariate_matrix @ fitted_state.fit_result.alpha

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
        intercept_column = np.ones((covariate_matrix.shape[0], 1), dtype=np.float32)
        return np.concatenate([intercept_column, covariate_matrix], axis=1)


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


def _training_records(
    raw_genotypes: RawGenotypeMatrix,
    records: Sequence[VariantRecord],
    config: ModelConfig,
) -> list[VariantRecord]:
    from sv_pgs.progress import log, mem
    training_supports = [record.training_support for record in records]
    unresolved_variant_indices = [
        variant_index
        for variant_index, record in enumerate(records)
        if record.variant_class in STRUCTURAL_VARIANT_CLASSES and record.training_support is None
    ]
    log(f"  training records: {len(records)} total, {len(unresolved_variant_indices)} SVs need support counts  mem={mem()}")
    if unresolved_variant_indices:
        unresolved_lookup = {variant_index: offset for offset, variant_index in enumerate(unresolved_variant_indices)}
        unresolved_supports = [0] * len(unresolved_variant_indices)
        for batch in raw_genotypes.iter_column_batches(unresolved_variant_indices, batch_size=config.genotype_batch_size):
            for local_index, variant_index in enumerate(batch.variant_indices):
                unresolved_supports[unresolved_lookup[int(variant_index)]] = _infer_support_count_from_raw_genotypes(
                    batch.values[:, local_index],
                    records[int(variant_index)],
                )
        for variant_index, support in zip(unresolved_variant_indices, unresolved_supports, strict=True):
            training_supports[variant_index] = support

    training_records: list[VariantRecord] = []
    for variant_index, record in enumerate(records):
        training_records.append(
            VariantRecord(
                variant_id=record.variant_id,
                variant_class=record.variant_class,
                chromosome=record.chromosome,
                position=record.position,
                length=record.length,
                allele_frequency=record.allele_frequency,
                quality=record.quality,
                training_support=training_supports[variant_index],
                is_repeat=record.is_repeat,
                is_copy_number=record.is_copy_number,
                prior_class_members=record.prior_class_members,
                prior_class_membership=record.prior_class_membership,
            )
        )
    return training_records


def _training_records_from_stats(
    records: Sequence[VariantRecord],
    variant_stats: VariantStatistics,
) -> list[VariantRecord]:
    """Build training records using pre-computed support counts (no data pass)."""
    from sv_pgs.progress import log
    training_records: list[VariantRecord] = []
    for variant_index, record in enumerate(records):
        support = record.training_support
        if support is None and record.variant_class in STRUCTURAL_VARIANT_CLASSES:
            support = int(variant_stats.support_counts[variant_index])
        training_records.append(
            VariantRecord(
                variant_id=record.variant_id,
                variant_class=record.variant_class,
                chromosome=record.chromosome,
                position=record.position,
                length=record.length,
                allele_frequency=record.allele_frequency,
                quality=record.quality,
                training_support=support,
                is_repeat=record.is_repeat,
                is_copy_number=record.is_copy_number,
                prior_class_members=record.prior_class_members,
                prior_class_membership=record.prior_class_membership,
            )
        )
    log(f"  training records from stats: {len(training_records)} records [NO DATA PASS]")
    return training_records


def _compute_support_counts(
    raw_genotypes: RawGenotypeMatrix,
    records: Sequence[VariantRecord],
    config: ModelConfig,
) -> np.ndarray:
    """Compute support counts by streaming (fallback when no pre-computed stats)."""
    support_counts = np.zeros(len(records), dtype=np.int32)
    for batch in raw_genotypes.iter_column_batches(batch_size=config.genotype_batch_size):
        for local_index, variant_index in enumerate(batch.variant_indices):
            col = batch.values[:, local_index]
            non_missing = col[~np.isnan(col)]
            support_counts[int(variant_index)] = int(np.count_nonzero(np.abs(non_missing) > 0.5))
    return support_counts


def _screen_active_variant_indices(
    raw_genotypes: RawGenotypeMatrix,
    active_variant_indices: np.ndarray,
    covariates: np.ndarray,
    targets: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
    maximum_active_variants: int,
    batch_size: int,
) -> np.ndarray:
    from sv_pgs.progress import log, mem
    if active_variant_indices.shape[0] <= maximum_active_variants:
        return np.asarray(active_variant_indices, dtype=np.int32)

    covariate_matrix = np.asarray(covariates, dtype=np.float64)
    target_vector = np.asarray(targets, dtype=np.float64).reshape(-1)
    screening_coefficients, *_ = np.linalg.lstsq(covariate_matrix, target_vector, rcond=None)
    residual = target_vector - covariate_matrix @ screening_coefficients
    score_by_variant_index = np.full(raw_genotypes.shape[1], -np.inf, dtype=np.float32)

    log(
        f"  marginal screening: scoring {active_variant_indices.shape[0]} active variants "
        f"with batch_size={batch_size}  mem={mem()}"
    )
    variants_done = 0
    total_active = int(active_variant_indices.shape[0])
    for batch in raw_genotypes.iter_column_batches(active_variant_indices, batch_size=batch_size):
        batch_means = means[batch.variant_indices]
        batch_scales = scales[batch.variant_indices]
        standardized_batch = _standardize_batch(batch.values, batch_means, batch_scales).astype(np.float64, copy=False)
        batch_scores = np.abs(standardized_batch.T @ residual).astype(np.float32, copy=False)
        score_by_variant_index[batch.variant_indices] = batch_scores
        variants_done += len(batch.variant_indices)
        if variants_done == len(batch.variant_indices) or variants_done % max(total_active // 10, 1) < len(batch.variant_indices) or variants_done == total_active:
            log(
                f"  marginal screening: {variants_done}/{total_active} "
                f"({100 * variants_done // total_active}%)  mem={mem()}"
            )

    local_scores = score_by_variant_index[active_variant_indices]
    top_local_indices = np.argpartition(local_scores, -maximum_active_variants)[-maximum_active_variants:]
    selected_variant_indices = np.sort(active_variant_indices[top_local_indices].astype(np.int32))
    log(
        f"  marginal screening done: kept {selected_variant_indices.shape[0]} / {active_variant_indices.shape[0]} "
        f"active variants"
    )
    return selected_variant_indices


def _identity_tie_map(variant_count: int) -> TieMap:
    kept_indices = np.arange(variant_count, dtype=np.int32)
    return TieMap(
        kept_indices=kept_indices,
        original_to_reduced=kept_indices.copy(),
        reduced_to_group=[
            TieGroup(
                representative_index=int(variant_index),
                member_indices=np.asarray([variant_index], dtype=np.int32),
                signs=np.asarray([1.0], dtype=np.float32),
            )
            for variant_index in range(variant_count)
        ],
    )


def _standardize_batch(
    batch: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    mean_vector = np.asarray(means, dtype=np.float32)
    scale_vector = np.asarray(scales, dtype=np.float32)
    imputed = np.where(np.isnan(batch), mean_vector[None, :], batch)
    standardized = (imputed - mean_vector[None, :]) / scale_vector[None, :]
    return np.asarray(standardized, dtype=np.float32)

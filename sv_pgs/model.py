from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from sv_pgs.artifact import ModelArtifact, load_artifact, save_artifact
from sv_pgs.config import ModelConfig, TraitType
from sv_pgs.data import TieGroup, TieMap, VariantRecord, normalize_variant_records
from sv_pgs.inference import VariationalFitResult, fit_variational_em
from sv_pgs.numeric import stable_sigmoid
from sv_pgs.preprocessing import (
    Preprocessor,
    _infer_support_count_from_raw_genotypes,
    build_tie_map,
    fit_preprocessor,
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
        genotypes: np.ndarray,
        covariates: np.ndarray,
        targets: np.ndarray,
        variant_records: Sequence[VariantRecord | dict],
        validation_data: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> "BayesianPGS":
        print(f"[fit] START  genotypes={genotypes.shape}  covariates={covariates.shape}  targets={targets.shape}", flush=True)
        raw_genotype_matrix = np.asarray(genotypes, dtype=np.float32)
        normalized_records = _training_records(
            raw_genotype_matrix,
            normalize_variant_records(variant_records),
        )
        print(f"[fit] normalized {len(normalized_records)} variant records", flush=True)
        covariate_matrix = self._with_intercept(covariates)
        print(f"[fit] fitting preprocessor...", flush=True)
        prepared_arrays = fit_preprocessor(raw_genotype_matrix, covariate_matrix, targets, self.config)
        preprocessor = Preprocessor(means=prepared_arrays.means, scales=prepared_arrays.scales)
        print(f"[fit] preprocessor done", flush=True)

        active_variant_indices = select_active_variant_indices(
            genotype_matrix=raw_genotype_matrix,
            variant_records=normalized_records,
            config=self.config,
        )
        print(f"[fit] active variants: {len(active_variant_indices)} / {len(normalized_records)}", flush=True)
        active_genotypes = prepared_arrays.genotypes[:, active_variant_indices]
        active_records = [normalized_records[int(variant_index)] for variant_index in active_variant_indices]

        print(f"[fit] building tie map...", flush=True)
        reduced_tie_map = build_tie_map(active_genotypes, active_records, self.config)
        original_space_tie_map = _project_tie_map_to_original_space(
            reduced_tie_map=reduced_tie_map,
            active_variant_indices=active_variant_indices,
            original_variant_count=len(normalized_records),
        )
        reduced_genotypes = active_genotypes[:, reduced_tie_map.kept_indices]
        print(f"[fit] reduced genotypes shape={reduced_genotypes.shape}  kept={len(reduced_tie_map.kept_indices)} groups={len(reduced_tie_map.reduced_to_group)}", flush=True)

        reduced_validation = None
        if validation_data is not None:
            validation_genotypes, validation_covariates, validation_targets = validation_data
            standardized_validation = preprocessor.transform(np.asarray(validation_genotypes, dtype=np.float32))
            reduced_validation = (
                standardized_validation[:, active_variant_indices][:, reduced_tie_map.kept_indices],
                self._with_intercept(np.asarray(validation_covariates, dtype=np.float32)),
                np.asarray(validation_targets, dtype=np.float32),
            )

        print(f"[fit] starting variational EM (max_iter={self.config.max_outer_iterations})...", flush=True)
        fit_result = fit_variational_em(
            genotypes=reduced_genotypes,
            covariates=prepared_arrays.covariates,
            targets=prepared_arrays.targets,
            records=active_records,
            tie_map=reduced_tie_map,
            config=self.config,
            validation_data=reduced_validation,
        )
        print(f"[fit] variational EM done  iterations={len(fit_result.objective_history)}", flush=True)
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

        self.state = FittedState(
            variant_records=normalized_records,
            active_variant_indices=active_variant_indices,
            preprocessor=preprocessor,
            tie_map=original_space_tie_map,
            fit_result=fit_result,
            full_coefficients=full_coefficients,
        )
        return self

    def decision_function(self, genotypes: np.ndarray, covariates: np.ndarray) -> np.ndarray:
        fitted_state = self._require_state()
        standardized_genotypes = fitted_state.preprocessor.transform(np.asarray(genotypes, dtype=np.float32))
        covariate_matrix = self._with_intercept(np.asarray(covariates, dtype=np.float32))
        return standardized_genotypes @ fitted_state.full_coefficients + covariate_matrix @ fitted_state.fit_result.alpha

    def predict_proba(self, genotypes: np.ndarray, covariates: np.ndarray) -> np.ndarray:
        if self.config.trait_type != TraitType.BINARY:
            raise ValueError("predict_proba is only available for binary traits.")
        linear_predictor = self.decision_function(genotypes, covariates)
        positive_probability = stable_sigmoid(linear_predictor)
        return np.column_stack([1.0 - positive_probability, positive_probability])

    def predict(self, genotypes: np.ndarray, covariates: np.ndarray) -> np.ndarray:
        linear_predictor = self.decision_function(genotypes, covariates)
        if self.config.trait_type == TraitType.BINARY:
            return (stable_sigmoid(linear_predictor) >= 0.5).astype(np.int32)
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
    def load(cls, path: str | Path) -> "BayesianPGS":
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
    raw_genotypes: np.ndarray,
    records: Sequence[VariantRecord],
) -> list[VariantRecord]:
    training_records: list[VariantRecord] = []
    for variant_index, record in enumerate(records):
        raw_variant_values = np.asarray(raw_genotypes[:, variant_index], dtype=np.float32)
        training_support = _training_support(raw_variant_values, record)
        training_records.append(
            VariantRecord(
                variant_id=record.variant_id,
                variant_class=record.variant_class,
                chromosome=record.chromosome,
                position=record.position,
                length=record.length,
                allele_frequency=record.allele_frequency,
                quality=record.quality,
                training_support=training_support,
                is_repeat=record.is_repeat,
                is_copy_number=record.is_copy_number,
                prior_class_members=record.prior_class_members,
                prior_class_membership=record.prior_class_membership,
            )
        )
    return training_records


def _training_support(raw_variant_values: np.ndarray, record: VariantRecord) -> int | None:
    if record.variant_class not in STRUCTURAL_VARIANT_CLASSES:
        return record.training_support
    if record.training_support is not None:
        return record.training_support
    return _infer_support_count_from_raw_genotypes(raw_variant_values, record)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from sv_pgs.artifact import ModelArtifact, load_artifact, save_artifact
from sv_pgs.config import ModelConfig, TraitType
from sv_pgs.data import TieGroup, TieMap, VariantRecord, normalize_variant_records
from sv_pgs.inference import VariationalFitResult, compute_export_baseline_variances, fit_variational_em
from sv_pgs.numeric import stable_sigmoid
from sv_pgs.preprocessing import Preprocessor, build_tie_map, fit_preprocessor, select_active_variant_indices


@dataclass(slots=True)
class FittedState:
    variant_records: list[VariantRecord]
    active_variant_indices: np.ndarray
    preprocessor: Preprocessor
    tie_map: TieMap
    fit_result: VariationalFitResult
    full_coefficients: np.ndarray


class BayesianPGS:
    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self.state: FittedState | None = None

    def fit(
        self,
        genotypes: np.ndarray,
        covariates: np.ndarray,
        targets: np.ndarray,
        variant_records: Sequence[VariantRecord | dict],
        validation_data: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> "BayesianPGS":
        normalized_records = normalize_variant_records(variant_records)
        covariate_matrix = self._with_intercept(covariates)
        prepared_arrays = fit_preprocessor(genotypes, covariate_matrix, targets, self.config)
        preprocessor = Preprocessor(means=prepared_arrays.means, scales=prepared_arrays.scales)

        active_variant_indices = select_active_variant_indices(
            genotype_matrix=np.asarray(genotypes, dtype=np.float32),
            variant_records=normalized_records,
            config=self.config,
        )
        active_genotypes = prepared_arrays.genotypes[:, active_variant_indices]
        active_records = [normalized_records[int(variant_index)] for variant_index in active_variant_indices]

        reduced_tie_map = build_tie_map(active_genotypes, active_records, self.config)
        original_space_tie_map = _project_tie_map_to_original_space(
            reduced_tie_map=reduced_tie_map,
            active_variant_indices=active_variant_indices,
            original_variant_count=len(normalized_records),
        )
        reduced_genotypes = active_genotypes[:, reduced_tie_map.kept_indices]

        reduced_validation = None
        if validation_data is not None:
            validation_genotypes, validation_covariates, validation_targets = validation_data
            standardized_validation = preprocessor.transform(np.asarray(validation_genotypes, dtype=np.float32))
            reduced_validation = (
                standardized_validation[:, active_variant_indices][:, reduced_tie_map.kept_indices],
                self._with_intercept(np.asarray(validation_covariates, dtype=np.float32)),
                np.asarray(validation_targets, dtype=np.float32),
            )

        fit_result = fit_variational_em(
            genotypes=reduced_genotypes,
            covariates=prepared_arrays.covariates,
            targets=prepared_arrays.targets,
            records=active_records,
            tie_map=reduced_tie_map,
            config=self.config,
            validation_data=reduced_validation,
        )
        tie_group_weights = _tie_group_export_weights(
            records=active_records,
            tie_map=reduced_tie_map,
            fit_result=fit_result,
            config=self.config,
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

    def artifact_metadata(self) -> dict[str, object]:
        fitted_state = self._require_state()
        return {
            "prior_version": self.config.prior_version,
            "transform_version": self.config.transform_version,
            "variant_count": len(fitted_state.variant_records),
            "active_variant_count": int(fitted_state.active_variant_indices.shape[0]),
            "reduced_variant_count": int(fitted_state.fit_result.beta_reduced.shape[0]),
        }

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
    records: Sequence[VariantRecord],
    tie_map: TieMap,
    fit_result: VariationalFitResult,
    config: ModelConfig,
) -> list[np.ndarray]:
    if fit_result.member_prior_variances is None:
        baseline_prior_variances = compute_export_baseline_variances(
            records=records,
            scale_model_coefficients=fit_result.scale_model_coefficients,
            global_scale=fit_result.global_scale,
            config=config,
        )
    else:
        baseline_prior_variances = np.asarray(fit_result.member_prior_variances, dtype=np.float32)
    group_weights: list[np.ndarray] = []
    for tie_group in tie_map.reduced_to_group:
        member_variances = _regularize_tie_group_member_variances(
            baseline_prior_variances[tie_group.member_indices]
        )
        normalized_weights = member_variances / np.maximum(np.sum(member_variances), 1e-12)
        group_weights.append(normalized_weights.astype(np.float32))
    return group_weights


def _regularize_tie_group_member_variances(member_variances: np.ndarray) -> np.ndarray:
    member_variance_array = np.asarray(member_variances, dtype=np.float64)
    if member_variance_array.shape[0] <= 1:
        return member_variance_array.astype(np.float32)
    group_mean_variance = float(np.mean(member_variance_array))
    return np.sqrt(member_variance_array * max(group_mean_variance, 1e-12)).astype(np.float32)

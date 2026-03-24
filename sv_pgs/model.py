from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.special import expit

from sv_pgs.artifact import ModelArtifact, load_artifact, save_artifact
from sv_pgs.config import ModelConfig, TraitType
from sv_pgs.data import GraphEdges, TieGroup, TieMap, VariantRecord, normalize_variant_records
from sv_pgs.graph import build_correlation_graph
from sv_pgs.inference import VariationalFitResult, fit_variational_em
from sv_pgs.preprocessing import Preprocessor, build_tie_map, fit_preprocessor


@dataclass(slots=True)
class FittedState:
    variant_records: list[VariantRecord]
    active_variant_indices: np.ndarray
    preprocessor: Preprocessor
    tie_map: TieMap
    graph: GraphEdges
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

        active_variant_indices = _select_active_variant_indices(
            genotype_matrix=np.asarray(genotypes, dtype=np.float32),
            variant_records=normalized_records,
            config=self.config,
        )
        active_genotypes = prepared_arrays.genotypes[:, active_variant_indices]
        active_records = [normalized_records[int(variant_index)] for variant_index in active_variant_indices]

        reduced_tie_map = build_tie_map(active_genotypes, active_records)
        original_space_tie_map = _project_tie_map_to_original_space(
            reduced_tie_map=reduced_tie_map,
            active_variant_indices=active_variant_indices,
            original_variant_count=len(normalized_records),
        )
        correlation_graph = build_correlation_graph(active_genotypes, active_records, reduced_tie_map, self.config)

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
            genotypes=active_genotypes,
            covariates=prepared_arrays.covariates,
            targets=prepared_arrays.targets,
            records=active_records,
            tie_map=reduced_tie_map,
            graph=correlation_graph,
            config=self.config,
            validation_data=reduced_validation,
        )
        active_coefficients = reduced_tie_map.expand_coefficients(fit_result.beta_reduced)
        full_coefficients = np.zeros(len(normalized_records), dtype=np.float32)
        full_coefficients[active_variant_indices] = active_coefficients

        self.state = FittedState(
            variant_records=normalized_records,
            active_variant_indices=active_variant_indices,
            preprocessor=preprocessor,
            tie_map=original_space_tie_map,
            graph=correlation_graph,
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
        positive_probability = expit(linear_predictor)
        return np.column_stack([1.0 - positive_probability, positive_probability])

    def predict(self, genotypes: np.ndarray, covariates: np.ndarray) -> np.ndarray:
        linear_predictor = self.decision_function(genotypes, covariates)
        if self.config.trait_type == TraitType.BINARY:
            return (expit(linear_predictor) >= 0.5).astype(np.int32)
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
            tie_map=fitted_state.tie_map,
            graph=fitted_state.graph,
            sigma_e2=fitted_state.fit_result.sigma_e2,
            class_mixture_weights=fitted_state.fit_result.class_mixture_weights,
            class_variances=fitted_state.fit_result.class_variances,
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
            graph=artifact.graph,
            fit_result=VariationalFitResult(
                alpha=artifact.alpha,
                beta_reduced=artifact.beta_reduced,
                beta_variance=np.zeros_like(artifact.beta_reduced),
                responsibilities=np.zeros(
                    (artifact.beta_reduced.shape[0], artifact.config.mixture_components),
                    dtype=np.float32,
                ),
                class_mixture_weights=artifact.class_mixture_weights,
                class_variances=artifact.class_variances,
                sigma_e2=artifact.sigma_e2,
                objective_history=artifact.objective_history,
                validation_history=artifact.validation_history,
                block_posteriors=[],
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
            "graph_metadata_version": self.config.graph_metadata_version,
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


def _select_active_variant_indices(
    genotype_matrix: np.ndarray,
    variant_records: Sequence[VariantRecord],
    config: ModelConfig,
) -> np.ndarray:
    active_flags = np.ones(len(variant_records), dtype=bool)
    structural_variant_classes = set(config.structural_variant_classes())
    for variant_index, variant_record in enumerate(variant_records):
        if variant_record.variant_class not in structural_variant_classes:
            continue
        carrier_count = _carrier_count(genotype_matrix[:, variant_index])
        if carrier_count < config.minimum_structural_variant_carriers:
            active_flags[variant_index] = False
    return np.where(active_flags)[0].astype(np.int32)


def _carrier_count(variant_values: np.ndarray) -> int:
    non_missing_values = variant_values[~np.isnan(variant_values)]
    return int(np.count_nonzero(np.abs(non_missing_values) > 0.0))


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

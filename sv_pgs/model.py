from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.special import expit

from sv_pgs.artifact import ModelArtifact, load_artifact, save_artifact
from sv_pgs.config import ModelConfig, TraitType
from sv_pgs.data import GraphEdges, TieMap, VariantRecord, normalize_variant_records
from sv_pgs.graph import build_correlation_graph
from sv_pgs.inference import VariationalFitResult, fit_variational_em
from sv_pgs.preprocessing import Preprocessor, build_tie_map, fit_preprocessor


@dataclass(slots=True)
class FittedState:
    records: list[VariantRecord]
    preprocessor: Preprocessor
    tie_map: TieMap
    graph: GraphEdges
    fit_result: VariationalFitResult
    beta_full: np.ndarray


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
        records = normalize_variant_records(variant_records)
        prepared = fit_preprocessor(genotypes, self._with_intercept(covariates), targets, self.config)
        preprocessor = Preprocessor(means=prepared.means, scales=prepared.scales)
        tie_map = build_tie_map(prepared.genotypes, records)
        graph = build_correlation_graph(prepared.genotypes, records, tie_map, self.config)

        reduced_validation = None
        if validation_data is not None:
            val_x, val_c, val_y = validation_data
            transformed_val = preprocessor.transform(np.asarray(val_x, dtype=np.float32))
            reduced_validation = (
                transformed_val[:, tie_map.kept_indices],
                self._with_intercept(np.asarray(val_c, dtype=np.float32)),
                np.asarray(val_y, dtype=np.float32),
            )

        fit_result = fit_variational_em(
            genotypes=prepared.genotypes,
            covariates=prepared.covariates,
            targets=prepared.targets,
            records=records,
            tie_map=tie_map,
            graph=graph,
            config=self.config,
            validation_data=reduced_validation,
        )
        beta_full = tie_map.expand_coefficients(fit_result.beta_reduced)
        self.state = FittedState(
            records=records,
            preprocessor=preprocessor,
            tie_map=tie_map,
            graph=graph,
            fit_result=fit_result,
            beta_full=beta_full,
        )
        return self

    def decision_function(self, genotypes: np.ndarray, covariates: np.ndarray) -> np.ndarray:
        state = self._require_state()
        standardized = state.preprocessor.transform(np.asarray(genotypes, dtype=np.float32))
        covariates = self._with_intercept(np.asarray(covariates, dtype=np.float32))
        return standardized @ state.beta_full + covariates @ state.fit_result.alpha

    def predict_proba(self, genotypes: np.ndarray, covariates: np.ndarray) -> np.ndarray:
        if self.config.trait_type != TraitType.BINARY:
            raise ValueError("predict_proba is only available for binary traits.")
        linear = self.decision_function(genotypes, covariates)
        positive = expit(linear)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, genotypes: np.ndarray, covariates: np.ndarray) -> np.ndarray:
        linear = self.decision_function(genotypes, covariates)
        if self.config.trait_type == TraitType.BINARY:
            return (expit(linear) >= 0.5).astype(np.int32)
        return linear

    def export(self, path: str | Path) -> None:
        state = self._require_state()
        artifact = ModelArtifact(
            config=self.config,
            records=state.records,
            means=state.preprocessor.means,
            scales=state.preprocessor.scales,
            alpha=state.fit_result.alpha,
            beta_reduced=state.fit_result.beta_reduced,
            beta_full=state.beta_full,
            tie_map=state.tie_map,
            graph=state.graph,
            sigma_e2=state.fit_result.sigma_e2,
            class_mixture_weights=state.fit_result.class_mixture_weights,
            class_variances=state.fit_result.class_variances,
            objective_history=state.fit_result.objective_history,
            validation_history=state.fit_result.validation_history,
        )
        save_artifact(path, artifact)

    @classmethod
    def load(cls, path: str | Path) -> "BayesianPGS":
        artifact = load_artifact(path)
        model = cls(config=artifact.config)
        model.state = FittedState(
            records=artifact.records,
            preprocessor=Preprocessor(means=artifact.means, scales=artifact.scales),
            tie_map=artifact.tie_map,
            graph=artifact.graph,
            fit_result=VariationalFitResult(
                alpha=artifact.alpha,
                beta_reduced=artifact.beta_reduced,
                beta_variance=np.zeros_like(artifact.beta_reduced),
                responsibilities=np.zeros((artifact.beta_reduced.shape[0], artifact.config.mixture_components), dtype=np.float32),
                class_mixture_weights=artifact.class_mixture_weights,
                class_variances=artifact.class_variances,
                sigma_e2=artifact.sigma_e2,
                objective_history=artifact.objective_history,
                validation_history=artifact.validation_history,
                block_posteriors=[],
            ),
            beta_full=artifact.beta_full,
        )
        return model

    def coefficient_table(self) -> list[dict[str, object]]:
        state = self._require_state()
        return [
            {
                "variant_id": record.variant_id,
                "variant_class": record.variant_class.value,
                "beta": float(beta),
            }
            for record, beta in zip(state.records, state.beta_full, strict=True)
        ]

    def artifact_metadata(self) -> dict[str, object]:
        state = self._require_state()
        return {
            "graph_metadata_version": self.config.graph_metadata_version,
            "transform_version": self.config.transform_version,
            "variant_count": len(state.records),
            "reduced_variant_count": int(state.fit_result.beta_reduced.shape[0]),
        }

    def _require_state(self) -> FittedState:
        if self.state is None:
            raise ValueError("Model is not fitted.")
        return self.state

    @staticmethod
    def _with_intercept(covariates: np.ndarray) -> np.ndarray:
        covariates = np.asarray(covariates, dtype=np.float32)
        if covariates.ndim != 2:
            raise ValueError("covariates must be 2D.")
        intercept = np.ones((covariates.shape[0], 1), dtype=np.float32)
        return np.concatenate([intercept, covariates], axis=1)

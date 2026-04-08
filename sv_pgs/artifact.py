from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from sv_pgs.config import ModelConfig, TraitType, VariantClass
from sv_pgs.data import VariantRecord, TieGroup, TieMap, normalize_variant_records


@dataclass(slots=True)
class VariantMetadataTable:
    variant_ids: list[str]
    variant_classes: list[VariantClass]

    def __post_init__(self) -> None:
        self.variant_ids = [str(variant_id) for variant_id in self.variant_ids]
        self.variant_classes = [
            variant_class if isinstance(variant_class, VariantClass) else VariantClass(str(variant_class))
            for variant_class in self.variant_classes
        ]
        if len(self.variant_ids) != len(self.variant_classes):
            raise ValueError("variant_ids and variant_classes must have the same length.")

    def __len__(self) -> int:
        return len(self.variant_ids)


@dataclass(slots=True)
class ModelArtifact:
    config: ModelConfig
    records: list[VariantRecord]
    means: np.ndarray
    scales: np.ndarray
    alpha: np.ndarray
    beta_reduced: np.ndarray
    beta_full: np.ndarray
    beta_variance: np.ndarray
    tie_map: TieMap
    sigma_e2: float
    prior_scales: np.ndarray
    global_scale: float
    class_tpb_shape_a: dict[VariantClass, float]
    class_tpb_shape_b: dict[VariantClass, float]
    scale_model_coefficients: np.ndarray
    scale_model_feature_names: list[str]
    objective_history: list[float]
    validation_history: list[float]
    variant_metadata: VariantMetadataTable = field(init=False)

    def __post_init__(self) -> None:
        self.records = normalize_variant_records(self.records)
        self.variant_metadata = VariantMetadataTable(
            variant_ids=[record.variant_id for record in self.records],
            variant_classes=[record.variant_class for record in self.records],
        )
        variant_count = len(self.variant_metadata)
        if self.means.shape != (variant_count,):
            raise ValueError("Artifact means must align with full variant metadata.")
        if self.scales.shape != (variant_count,):
            raise ValueError("Artifact scales must align with full variant metadata.")
        if self.beta_full.shape != (variant_count,):
            raise ValueError("Artifact beta_full must align with full variant metadata.")
        if self.tie_map.original_to_reduced.shape != (variant_count,):
            raise ValueError("Artifact tie_map must align with full variant metadata.")


def save_artifact(path: str | Path, artifact: ModelArtifact) -> None:
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        root / "arrays.npz",
        means=artifact.means,
        scales=artifact.scales,
        alpha=artifact.alpha,
        beta_reduced=artifact.beta_reduced,
        beta_full=artifact.beta_full,
        beta_variance=artifact.beta_variance,
        tie_kept_indices=artifact.tie_map.kept_indices,
        tie_original_to_reduced=artifact.tie_map.original_to_reduced,
        prior_scales=artifact.prior_scales,
        scale_model_coefficients=artifact.scale_model_coefficients,
    )

    payload = {
        "config": _config_to_json(artifact.config),
        "records": [
            {
                "variant_id": record.variant_id,
                "variant_class": record.variant_class.value,
                "chromosome": record.chromosome,
                "position": record.position,
                "length": record.length,
                "allele_frequency": record.allele_frequency,
                "quality": record.quality,
                "training_support": record.training_support,
                "is_repeat": record.is_repeat,
                "is_copy_number": record.is_copy_number,
                "prior_binary_features": dict(record.prior_binary_features),
                "prior_continuous_features": dict(record.prior_continuous_features),
                "prior_categorical_features": dict(record.prior_categorical_features),
                "prior_membership_features": {
                    feature_name: dict(feature_memberships)
                    for feature_name, feature_memberships in record.prior_membership_features.items()
                },
                "prior_nested_features": {
                    feature_name: list(feature_path)
                    for feature_name, feature_path in record.prior_nested_features.items()
                },
                "prior_nested_membership_features": {
                    feature_name: dict(feature_memberships)
                    for feature_name, feature_memberships in record.prior_nested_membership_features.items()
                },
                "prior_class_members": [
                    variant_class.value for variant_class in record.prior_class_members
                ],
                "prior_class_membership": list(record.prior_class_membership),
            }
            for record in artifact.records
        ],
        "tie_groups": [
            {
                "representative_index": group.representative_index,
                "member_indices": group.member_indices.tolist(),
                "signs": group.signs.tolist(),
            }
            for group in artifact.tie_map.reduced_to_group
        ],
        "sigma_e2": artifact.sigma_e2,
        "global_scale": float(artifact.global_scale),
        "class_tpb_shape_a": {
            variant_class.value: float(value) for variant_class, value in artifact.class_tpb_shape_a.items()
        },
        "class_tpb_shape_b": {
            variant_class.value: float(value) for variant_class, value in artifact.class_tpb_shape_b.items()
        },
        "scale_model_feature_names": artifact.scale_model_feature_names,
        "objective_history": artifact.objective_history,
        "validation_history": artifact.validation_history,
    }
    (root / "metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_artifact(path: str | Path) -> ModelArtifact:
    root = Path(path)
    arrays = np.load(root / "arrays.npz", allow_pickle=False)
    payload = json.loads((root / "metadata.json").read_text(encoding="utf-8"))

    tie_map = TieMap(
        kept_indices=arrays["tie_kept_indices"].astype(np.int32),
        original_to_reduced=arrays["tie_original_to_reduced"].astype(np.int32),
        reduced_to_group=[
            TieGroup(
                representative_index=int(group["representative_index"]),
                member_indices=np.asarray(group["member_indices"], dtype=np.int32),
                signs=np.asarray(group["signs"], dtype=np.float32),
            )
            for group in payload["tie_groups"]
        ],
    )
    return ModelArtifact(
        config=_config_from_json(payload["config"]),
        records=normalize_variant_records(payload["records"]),
        means=arrays["means"].astype(np.float32),
        scales=arrays["scales"].astype(np.float32),
        alpha=arrays["alpha"].astype(np.float32),
        beta_reduced=arrays["beta_reduced"].astype(np.float32),
        beta_full=arrays["beta_full"].astype(np.float32),
        beta_variance=arrays["beta_variance"].astype(np.float32),
        tie_map=tie_map,
        sigma_e2=float(payload["sigma_e2"]),
        prior_scales=arrays["prior_scales"].astype(np.float32),
        global_scale=float(payload["global_scale"]),
        class_tpb_shape_a={
            VariantClass(key): float(value)
            for key, value in payload["class_tpb_shape_a"].items()
        },
        class_tpb_shape_b={
            VariantClass(key): float(value)
            for key, value in payload["class_tpb_shape_b"].items()
        },
        scale_model_coefficients=arrays["scale_model_coefficients"].astype(np.float32),
        scale_model_feature_names=[str(feature_name) for feature_name in payload["scale_model_feature_names"]],
        objective_history=[float(value) for value in payload["objective_history"]],
        validation_history=[float(value) for value in payload["validation_history"]],
    )


def _config_to_json(config: ModelConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["trait_type"] = config.trait_type.value
    return payload


def _config_from_json(payload: dict[str, Any]) -> ModelConfig:
    restored_payload = dict(payload)
    restored_payload["trait_type"] = TraitType(payload["trait_type"])
    return ModelConfig(**restored_payload)

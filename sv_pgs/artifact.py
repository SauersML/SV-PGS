from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sv_pgs.config import ModelConfig, TraitType, VariantClass
from sv_pgs.data import TieGroup, TieMap


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
    variant_metadata: VariantMetadataTable
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

    def __post_init__(self) -> None:
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
        "variant_metadata": {
            "variant_ids": list(artifact.variant_metadata.variant_ids),
            "variant_classes": [
                variant_class.value
                for variant_class in artifact.variant_metadata.variant_classes
            ],
        },
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
        variant_metadata=VariantMetadataTable(
            variant_ids=[str(variant_id) for variant_id in payload["variant_metadata"]["variant_ids"]],
            variant_classes=[
                VariantClass(variant_class)
                for variant_class in payload["variant_metadata"]["variant_classes"]
            ],
        ),
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

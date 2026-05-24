from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _fsync_parent_dir(path: Path) -> None:
    """Best-effort fsync of `path.parent` so an atomic rename survives a crash.

    Atomic rename guarantees the file's new name is visible OR the old one
    is, but durability of the directory entry update itself requires fsync
    on the parent directory FD. Silent no-op on platforms that don't
    support directory fsync (e.g. Windows).
    """
    try:
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        try:
            os.close(dir_fd)
        except OSError:
            pass

_LEGACY_DIAGNOSTICS_LOGGED = False
_logger = logging.getLogger(__name__)

from sv_pgs._typing import F32Array
from sv_pgs.config import ModelConfig, TraitType, VariantClass
from sv_pgs.data import VariantRecord, TieGroup, TieMap, normalize_variant_records


@dataclass(slots=True)
class ModelArtifact:
    config: ModelConfig
    records: list[VariantRecord]
    means: F32Array
    scales: F32Array
    alpha: F32Array
    beta_reduced: F32Array
    beta_full: F32Array
    beta_variance: F32Array
    tie_map: TieMap
    sigma_e2: float
    prior_scales: F32Array
    global_scale: float
    class_tpb_shape_a: dict[VariantClass, float]
    class_tpb_shape_b: dict[VariantClass, float]
    scale_model_coefficients: F32Array
    scale_model_feature_names: list[str]
    objective_history: list[float]
    validation_history: list[float]
    # SHA-256 of (genotype shape, variant records, covariates, targets, config)
    # captured at fit time. Empty string means "unknown / legacy artifact" —
    # auto-reuse paths must treat that as a miss. Populated via
    # ``BayesianPGS.export(..., fit_fingerprint=...)``.
    fit_fingerprint: str = ""
    # Convergence diagnostics persisted from the variational fit. Older
    # artifacts predate these fields; load_artifact() back-fills them with
    # defaults (converged=False, others None) and emits a one-shot warning.
    converged: bool = False
    selected_iteration_count: int = 0
    final_parameter_change: float | None = None
    final_predictor_change: float | None = None
    final_objective_change: float | None = None
    final_hyperparameter_change: float | None = None

    def __post_init__(self) -> None:
        self.records = normalize_variant_records(self.records)
        variant_count = len(self.records)
        if self.means.shape != (variant_count,):
            raise ValueError("Artifact means must align with records.")
        if self.scales.shape != (variant_count,):
            raise ValueError("Artifact scales must align with records.")
        if self.beta_full.shape != (variant_count,):
            raise ValueError("Artifact beta_full must align with records.")
        if self.tie_map.original_to_reduced.shape != (variant_count,):
            raise ValueError("Artifact tie_map must align with records.")


def save_artifact(path: str | Path, artifact: ModelArtifact) -> None:
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    arrays_final = root / "arrays.npz"
    metadata_final = root / "metadata.json"
    # Per-process unique temp names so two concurrent save_artifact() calls
    # targeting the same output dir cannot clobber each other's staging
    # files mid-write. The replace at the end is still atomic per file.
    unique_tag = f"{os.getpid()}.{uuid.uuid4().hex}"
    arrays_tmp = root / f"arrays.tmp.{unique_tag}.npz"
    metadata_tmp = root / f"metadata.json.tmp.{unique_tag}"

    # Write arrays to a staging file and fsync before publishing either output,
    # so a crash between the two replaces cannot leave new arrays paired with
    # old metadata (or vice versa).
    try:
        np.savez_compressed(
            arrays_tmp,
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
        # Reopen read-write so the fsync flushes dirty pages of THIS handle's
        # cache (read-mode fsync on a fresh fd was a no-op in practice).
        with open(arrays_tmp, "r+b") as arrays_handle:
            os.fsync(arrays_handle.fileno())
    except BaseException:
        # Don't leak per-pid staging files when serialization fails.
        try:
            Path(arrays_tmp).unlink(missing_ok=True)
        except OSError:
            pass
        raise

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
        "fit_fingerprint": artifact.fit_fingerprint,
        "converged": bool(artifact.converged),
        "selected_iteration_count": int(artifact.selected_iteration_count),
        "final_parameter_change": (
            None if artifact.final_parameter_change is None else float(artifact.final_parameter_change)
        ),
        "final_predictor_change": (
            None if artifact.final_predictor_change is None else float(artifact.final_predictor_change)
        ),
        "final_objective_change": (
            None if artifact.final_objective_change is None else float(artifact.final_objective_change)
        ),
        "final_hyperparameter_change": (
            None
            if artifact.final_hyperparameter_change is None
            else float(artifact.final_hyperparameter_change)
        ),
    }
    metadata_bytes = json.dumps(payload, indent=2).encode("utf-8")
    try:
        with open(metadata_tmp, "wb") as metadata_handle:
            metadata_handle.write(metadata_bytes)
            metadata_handle.flush()
            os.fsync(metadata_handle.fileno())
    except BaseException:
        try:
            Path(metadata_tmp).unlink(missing_ok=True)
        except OSError:
            pass
        try:
            Path(arrays_tmp).unlink(missing_ok=True)
        except OSError:
            pass
        raise

    # Both staging files are durable on disk; publish via atomic per-file
    # replaces. A crash before the first replace leaves the prior pair intact.
    # The window between the two replaces is two metadata operations — orders
    # of magnitude smaller than the prior write window — and the metadata
    # replace is treated as the commit point.
    os.replace(arrays_tmp, arrays_final)
    os.replace(metadata_tmp, metadata_final)
    # fsync the directory so the rename itself is durable across a crash;
    # without this the rename can be lost even though the files survive.
    _fsync_parent_dir(arrays_final)


def load_artifact(path: str | Path) -> ModelArtifact:
    root = Path(path)
    # ``np.load`` on an .npz returns an NpzFile that keeps the underlying
    # ZipFile open until ``close()``. Without an explicit close, the file
    # descriptor lingers until GC — on long-running pipelines that load many
    # artifacts (e.g. sweeping across diseases) this can exhaust the per-process
    # fd limit. Use ``with`` so the zip handle is released deterministically
    # even when key lookups or ``astype`` raise mid-construction.
    with np.load(root / "arrays.npz", allow_pickle=False) as arrays:
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
            records=payload["records"],
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
            fit_fingerprint=str(payload.get("fit_fingerprint", "")),
            **_load_diagnostics(payload),
        )


def _load_diagnostics(payload: dict[str, Any]) -> dict[str, Any]:
    global _LEGACY_DIAGNOSTICS_LOGGED
    if "converged" not in payload and not _LEGACY_DIAGNOSTICS_LOGGED:
        _logger.warning(
            "Loading legacy artifact without convergence diagnostics; "
            "defaulting converged=False and final_*_change=None."
        )
        _LEGACY_DIAGNOSTICS_LOGGED = True

    def _opt_float(key: str) -> float | None:
        value = payload.get(key)
        return None if value is None else float(value)

    return {
        "converged": bool(payload.get("converged", False)),
        "selected_iteration_count": int(payload.get("selected_iteration_count", 0)),
        "final_parameter_change": _opt_float("final_parameter_change"),
        "final_predictor_change": _opt_float("final_predictor_change"),
        "final_objective_change": _opt_float("final_objective_change"),
        "final_hyperparameter_change": _opt_float("final_hyperparameter_change"),
    }


def try_load_artifact_if_fingerprint_matches(
    path: str | Path,
    expected_fingerprint: str,
) -> ModelArtifact | None:
    """Return the artifact at ``path`` iff its ``fit_fingerprint`` matches.

    Returns ``None`` when the artifact is absent, malformed, missing a
    fingerprint, or the fingerprint differs. Callers use this to decide
    whether to skip a refit and reuse a prior run's outputs.
    """
    root = Path(path)
    if not (root / "arrays.npz").exists() or not (root / "metadata.json").exists():
        return None
    if not expected_fingerprint:
        return None
    try:
        artifact = load_artifact(root)
    except (OSError, ValueError, KeyError):
        return None
    if not artifact.fit_fingerprint or artifact.fit_fingerprint != expected_fingerprint:
        return None
    return artifact


def _config_to_json(config: ModelConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["trait_type"] = config.trait_type.value
    return payload


def _config_from_json(payload: dict[str, Any]) -> ModelConfig:
    restored_payload = dict(payload)
    restored_payload["trait_type"] = TraitType(payload["trait_type"])
    return ModelConfig(**restored_payload)

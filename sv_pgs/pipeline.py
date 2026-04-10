from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.metrics import log_loss, r2_score, roc_auc_score

from sv_pgs.config import ModelConfig, TraitType
from sv_pgs.io import LoadedDataset, _coerce_float, _format_float, _open_text_file
from sv_pgs.model import BayesianPGS
from sv_pgs.numeric import stable_sigmoid
from sv_pgs.progress import log, mem


@dataclass(slots=True)
class PipelineOutputs:
    artifact_dir: Path
    summary_path: Path
    predictions_path: Path
    coefficients_path: Path


def run_training_pipeline(
    dataset: LoadedDataset,
    config: ModelConfig,
    output_dir: str | Path,
) -> PipelineOutputs:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    from sv_pgs.progress import _log_file, set_log_file

    if _log_file is None:
        set_log_file(destination / f"training.{time.strftime('%Y%m%d_%H%M%S')}.log")
    log(
        f"=== TRAINING PIPELINE START ===  samples={len(dataset.sample_ids)}  "
        + f"variants={dataset.genotypes.shape[1]}  trait={config.trait_type.value}  mem={mem()}"
    )

    if dataset.variant_stats is not None:
        if dataset.variant_stats_minimum_scale is None:
            raise ValueError("dataset.variant_stats_minimum_scale must be set when variant_stats are provided.")
        if float(dataset.variant_stats_minimum_scale) != float(config.minimum_scale):
            raise ValueError(
                "dataset.variant_stats were computed with minimum_scale="
                + f"{dataset.variant_stats_minimum_scale:.6g}, but run_training_pipeline received minimum_scale="
                + f"{config.minimum_scale:.6g}. Reload the dataset with the same config."
            )

    summary_payload: dict[str, Any] = {
        "validation_enabled": False,
        "tuning_sample_count": int(dataset.genotypes.shape[0]),
        "validation_sample_count": 0,
        "validation_history": [],
    }
    log("fitting Bayesian PGS model...")
    model = BayesianPGS(config).fit(
        dataset.genotypes,
        dataset.covariates,
        dataset.targets,
        dataset.variant_records,
        variant_stats=dataset.variant_stats,
    )
    log(f"model fitted  mem={mem()}")

    log("exporting model artifacts...")
    artifact_dir = destination / "artifact"
    model.export(artifact_dir)
    log(f"artifacts written to {artifact_dir}")

    log("writing coefficients table...")
    coefficients_path = destination / "coefficients.tsv.gz"
    coefficient_rows = model.coefficient_table(nonzero_only=True)
    _write_delimited_rows(
        coefficients_path,
        header=("variant_id", "variant_class", "beta", "chromosome", "position", "length", "allele_frequency"),
        rows=(
            (
                str(coefficient_row["variant_id"]),
                str(coefficient_row["variant_class"]),
                _format_float(_coerce_float(coefficient_row["beta"])),
                str(coefficient_row["chromosome"]),
                str(int(coefficient_row["position"])),
                _format_float(float(coefficient_row["length"])),
                _format_float(float(coefficient_row["allele_frequency"])),
            )
            for coefficient_row in coefficient_rows
        ),
    )
    log(f"wrote {len(coefficient_rows)} non-zero coefficient rows to {coefficients_path}")

    log("writing predictions...")
    predictions_path = destination / "predictions.tsv.gz"
    summary_payload.update(
        _write_predictions_and_summary(
            predictions_path=predictions_path,
            dataset=dataset,
            model=model,
        )
    )
    fitted_state = model.state
    if fitted_state is None:
        raise RuntimeError("trained model is missing fitted state.")
    active_count = int(fitted_state.active_variant_indices.shape[0])
    selected_iteration_count = getattr(
        fitted_state.fit_result,
        "selected_iteration_count",
        model.config.max_outer_iterations,
    )
    if selected_iteration_count is None:
        selected_iteration_count = model.config.max_outer_iterations
    summary_payload.update(
        {
            "sample_count": int(dataset.genotypes.shape[0]),
            "variant_count": int(dataset.genotypes.shape[1]),
            "active_variant_count": active_count,
            "trait_type": config.trait_type.value,
            "fit_max_outer_iterations": int(model.config.max_outer_iterations),
            "selected_iteration_count": int(selected_iteration_count),
        }
    )
    log(f"predictions written: {active_count} active variants out of {dataset.genotypes.shape[1]}")

    log("writing summary JSON...")
    summary_path = destination / "summary.json.gz"
    with _open_text_file(summary_path, "wt") as handle:
        handle.write(json.dumps(summary_payload, indent=2))
    log(f"=== TRAINING PIPELINE DONE ===  mem={mem()}")
    return PipelineOutputs(
        artifact_dir=artifact_dir,
        summary_path=summary_path,
        predictions_path=predictions_path,
        coefficients_path=coefficients_path,
    )


def _write_predictions_and_summary(
    predictions_path: Path,
    dataset: LoadedDataset,
    model: BayesianPGS,
) -> dict[str, Any]:
    log(f"computing predictions for {len(dataset.sample_ids)} samples, trait={model.config.trait_type.value}  mem={mem()}")
    training_components_getter = getattr(model, "training_decision_components", None)
    cached_components = None if training_components_getter is None else training_components_getter()
    if (
        cached_components is not None
        and cached_components[0].shape == (len(dataset.sample_ids),)
        and cached_components[1].shape == (len(dataset.sample_ids),)
    ):
        genetic_score, covariate_score = cached_components
    else:
        genetic_score, covariate_score = model.decision_components(dataset.genotypes, dataset.covariates)
    linear_predictor = np.asarray(genetic_score + covariate_score, dtype=np.float32)
    if model.config.trait_type == TraitType.BINARY:
        probabilities = np.asarray(stable_sigmoid(linear_predictor), dtype=np.float32)
        predicted_labels = (probabilities >= 0.5).astype(np.int32)
        log(
            f"binary predictions: mean_prob={float(np.mean(probabilities)):.4f}  "
            + f"pred_positive={int(np.sum(predicted_labels))}  pred_negative={int(np.sum(1-predicted_labels))}"
        )
        _write_delimited_rows(
            predictions_path,
            header=("sample_id", "target", "genetic_score", "covariate_score", "linear_predictor", "probability", "predicted_label"),
            rows=(
                (
                    sample_id,
                    _format_float(float(target)),
                    _format_float(float(genetic_component)),
                    _format_float(float(covariate_component)),
                    _format_float(float(raw_score)),
                    _format_float(float(probability)),
                    str(int(predicted_label)),
                )
                for sample_id, target, genetic_component, covariate_component, raw_score, probability, predicted_label in zip(
                    dataset.sample_ids,
                    dataset.targets,
                    genetic_score,
                    covariate_score,
                    linear_predictor,
                    probabilities,
                    predicted_labels,
                    strict=True,
                )
            ),
        )
        unique_targets = np.unique(dataset.targets)
        training_auc = None if unique_targets.shape[0] < 2 else float(roc_auc_score(dataset.targets, probabilities))
        training_accuracy = float(np.mean(predicted_labels == dataset.targets))
        training_log_loss_val = float(log_loss(dataset.targets, probabilities, labels=[0.0, 1.0]))
        log(
            f"training metrics: AUC={training_auc}  log_loss={training_log_loss_val:.4f}  "
            + f"accuracy={training_accuracy:.4f}  mem={mem()}"
        )
        return {
            "training_auc": training_auc,
            "training_log_loss": training_log_loss_val,
            "training_accuracy": training_accuracy,
        }

    predictions = linear_predictor
    _write_delimited_rows(
        predictions_path,
        header=("sample_id", "target", "genetic_score", "covariate_score", "prediction"),
        rows=(
            (
                sample_id,
                _format_float(float(target)),
                _format_float(float(genetic_component)),
                _format_float(float(covariate_component)),
                _format_float(float(prediction)),
            )
            for sample_id, target, genetic_component, covariate_component, prediction in zip(
                dataset.sample_ids,
                dataset.targets,
                genetic_score,
                covariate_score,
                predictions,
                strict=True,
            )
        ),
    )
    residuals = dataset.targets - predictions
    return {
        "training_r2": float(r2_score(dataset.targets, predictions)),
        "training_rmse": float(np.sqrt(np.mean(residuals * residuals))),
    }


def _write_delimited_rows(
    path: Path,
    header: Sequence[str],
    rows: Iterable[Sequence[str]],
) -> None:
    with _open_text_file(path, "wt", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)

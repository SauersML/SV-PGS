from __future__ import annotations

import csv
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from sklearn.metrics import log_loss, r2_score, roc_auc_score

from sv_pgs.artifact import try_load_artifact_if_fingerprint_matches
from sv_pgs.config import ModelConfig, TraitType
from sv_pgs.genotype import as_raw_genotype_matrix
from sv_pgs.io import LoadedDataset, _coerce_float, _format_float, _open_text_file
from sv_pgs.model import BayesianPGS, _fit_checkpoint_config_hash
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
    test_dataset: LoadedDataset | None = None,
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
    # Wire per-epoch monitoring before kicking off the fit.
    #
    # When a test_dataset is provided we route it into model.fit() twice:
    #   - as validation_data: the EM loop already prepares a standardized,
    #     tie-collapsed reduced view of it and computes a cross-entropy /
    #     MSE scalar on validation epochs (added to validation_history).
    #   - via per_epoch_eval_callback: the EM loop ALSO passes us the raw
    #     linear predictor on that same reduced view at validation epochs,
    #     so we can derive AUC / log-loss / accuracy / R² / RMSE and append
    #     a row to training_history.tsv as the fit progresses.
    #
    # The TSV is opened in append mode + flushed per row so a `tail -f` on
    # the workbench shows the per-epoch metric trajectory live.
    history_path = destination / "training_history.tsv"
    validation_data = None
    if test_dataset is not None and len(test_dataset.sample_ids) > 0:
        validation_data = (
            test_dataset.genotypes,
            test_dataset.covariates,
            test_dataset.targets,
        )

    # Auto-reuse a prior run's completed artifact when the inputs match.
    # ``_fit_checkpoint_config_hash`` covers (genotype shape, variant records,
    # covariates, targets, convergence-affecting config fields), which is the
    # same signature the durable EM checkpoint uses to gate resume — so an
    # artifact whose fingerprint matches is guaranteed-equivalent to the fit
    # this run would produce. On a match we skip the (multi-minute to multi-
    # hour) EM and reuse the saved coefficients verbatim; downstream outputs
    # (predictions, coefficients TSV, summary, per-epoch history) are
    # regenerated below from the loaded model so callers see the same files.
    #
    # The artifact-reuse check runs BEFORE opening the per-epoch history
    # writer, because that writer opens ``training_history.tsv`` in
    # truncating "w" mode — opening it on a reuse path (where we then skip
    # EM and never write any rows) would destroy the prior run's epoch
    # history, leaving behind a header-only file.
    artifact_dir = destination / "artifact"
    raw_genotype_matrix = as_raw_genotype_matrix(dataset.genotypes)
    fit_fingerprint = _fit_checkpoint_config_hash(
        genotype_matrix=raw_genotype_matrix,
        covariates=dataset.covariates,
        targets=dataset.targets,
        variant_records=dataset.variant_records,
        config=config,
    )
    reused_artifact = try_load_artifact_if_fingerprint_matches(artifact_dir, fit_fingerprint)
    if reused_artifact is not None:
        log(
            f"reusing prior fit artifact at {artifact_dir} "
            + f"(fit_fingerprint matches; skipping EM)  mem={mem()}"
        )
        model = BayesianPGS.load(artifact_dir)
        # Re-export so the on-disk fingerprint and arrays are byte-identical
        # to what this run would have written. This is a no-op for the data
        # but keeps the freshness mtime current — useful for the cached-eval
        # surfacing logic in aou_runner that scans by directory age.
        model.export(artifact_dir, fit_fingerprint=fit_fingerprint)
    else:
        # Only open the truncating per-epoch history writer once we know we
        # are actually going to fit; otherwise we would wipe the prior run's
        # ``training_history.tsv`` to a header-only file on every reuse.
        _per_epoch_callback, _close_history = _build_per_epoch_history_writer(
            history_path=history_path,
            config=config,
            test_dataset=test_dataset,
        )
        log("fitting Bayesian PGS model...")
        try:
            model = BayesianPGS(config).fit(
                dataset.genotypes,
                dataset.covariates,
                dataset.targets,
                dataset.variant_records,
                variant_stats=dataset.variant_stats,
                validation_data=validation_data,
                per_epoch_eval_callback=_per_epoch_callback,
                # Marking the held-out test set as holdout-only keeps the test
                # cross-entropy out of best-epoch parameter selection and out of
                # the early-stopping gate. Without this flag, the model would
                # pick the epoch with the lowest test cross-entropy as its final
                # parameters and the reported test AUC would be biased upward.
                validation_is_holdout_only=validation_data is not None,
            )
        finally:
            _close_history()
        log(f"model fitted  mem={mem()}")

        log("exporting model artifacts...")
        _guard_nonconverged_export(model, config)
        model.export(artifact_dir, fit_fingerprint=fit_fingerprint)
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
                str(int(_coerce_float(coefficient_row["position"]))),
                _format_float(_coerce_float(coefficient_row["length"])),
                _format_float(_coerce_float(coefficient_row["allele_frequency"])),
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
            is_training_dataset=True,
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
    _final_parameter_change = getattr(fitted_state.fit_result, "final_parameter_change", None)
    _final_predictor_change = getattr(fitted_state.fit_result, "final_predictor_change", None)
    _final_objective_change = getattr(fitted_state.fit_result, "final_objective_change", None)
    _final_hyperparameter_change = getattr(fitted_state.fit_result, "final_hyperparameter_change", None)
    summary_payload.update(
        {
            "sample_count": int(dataset.genotypes.shape[0]),
            "variant_count": int(dataset.genotypes.shape[1]),
            "active_variant_count": active_count,
            "trait_type": config.trait_type.value,
            "fit_max_outer_iterations": int(model.config.max_outer_iterations),
            "selected_iteration_count": int(selected_iteration_count),
            "fit_converged": bool(getattr(fitted_state.fit_result, "converged", False)),
            "final_parameter_change": (
                None if _final_parameter_change is None else float(_final_parameter_change)
            ),
            "final_predictor_change": (
                None if _final_predictor_change is None else float(_final_predictor_change)
            ),
            "final_objective_change": (
                None if _final_objective_change is None else float(_final_objective_change)
            ),
            "final_hyperparameter_change": (
                None if _final_hyperparameter_change is None else float(_final_hyperparameter_change)
            ),
        }
    )
    log(f"predictions written: {active_count} active variants out of {dataset.genotypes.shape[1]}")

    if test_dataset is not None and len(test_dataset.sample_ids) > 0:
        # Held-out evaluation: predict with the same model on samples that were
        # excluded from the fit and report the metrics. The model already
        # internalizes the standardization it was trained with, so prediction
        # against a different-cohort dataset is consistent as long as the
        # variant order matches (the multi-source loader guarantees that).
        log(f"=== HELD-OUT TEST EVALUATION ===  test_samples={len(test_dataset.sample_ids)}  mem={mem()}")
        test_predictions_path = destination / "test_predictions.tsv.gz"
        raw_test_metrics = _write_predictions_and_summary(
            predictions_path=test_predictions_path,
            dataset=test_dataset,
            model=model,
            is_training_dataset=False,
        )
        # _write_predictions_and_summary prefixes metrics with "training_" since
        # that's the only call site it knew about historically; relabel them
        # here so the summary JSON disambiguates train vs test.
        test_metrics = {
            ("test_" + key[len("training_"):] if key.startswith("training_") else "test_" + key): value
            for key, value in raw_test_metrics.items()
        }
        summary_payload["test_sample_count"] = int(len(test_dataset.sample_ids))
        summary_payload["test_predictions_path"] = test_predictions_path.name
        summary_payload.update(test_metrics)
        # Loud one-line summary so anybody skimming the log can find the number.
        if config.trait_type == TraitType.BINARY:
            log(
                "  >>> TEST AUC="
                + f"{test_metrics.get('test_auc')}  log_loss={test_metrics.get('test_log_loss'):.4f}"
                + f"  accuracy={test_metrics.get('test_accuracy'):.4f}  n={len(test_dataset.sample_ids)} <<<"
            )
        else:
            log(
                "  >>> TEST R2="
                + f"{test_metrics.get('test_r2'):.4f}  RMSE={test_metrics.get('test_rmse'):.4f}"
                + f"  n={len(test_dataset.sample_ids)} <<<"
            )

    log("writing summary JSON...")
    summary_path = destination / "summary.json.gz"
    # Write to a per-process unique temp path then atomically replace, so a
    # crash mid-write cannot leave a truncated summary.json.gz that future
    # `_log_cached_test_evals_from_summary` calls would mis-parse, and so
    # concurrent disease sweeps writing into sibling output dirs cannot
    # clobber each other's staging file.
    tmp_summary_path = summary_path.with_name(
        f"{summary_path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    try:
        with _open_text_file(tmp_summary_path, "wt") as handle:
            handle.write(json.dumps(summary_payload, indent=2))
        os.replace(tmp_summary_path, summary_path)
    except BaseException:
        try:
            tmp_summary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    log(f"=== TRAINING PIPELINE DONE ===  mem={mem()}")
    return PipelineOutputs(
        artifact_dir=artifact_dir,
        summary_path=summary_path,
        predictions_path=predictions_path,
        coefficients_path=coefficients_path,
    )


def _guard_nonconverged_export(model: BayesianPGS, config: ModelConfig) -> None:
    fitted_state = model.state
    if fitted_state is None:
        return
    fit_result = fitted_state.fit_result
    converged = bool(getattr(fit_result, "converged", False))
    if converged:
        return
    diagnostics = {
        "selected_iteration_count": getattr(fit_result, "selected_iteration_count", None),
        "final_parameter_change": getattr(fit_result, "final_parameter_change", None),
        "final_predictor_change": getattr(fit_result, "final_predictor_change", None),
        "final_objective_change": getattr(fit_result, "final_objective_change", None),
        "final_hyperparameter_change": getattr(fit_result, "final_hyperparameter_change", None),
    }
    if getattr(config, "allow_nonconverged_export", False):
        # Audit-trail marker for the override branch: reference the flag name
        # explicitly so operators grepping logs can spot user-overridden
        # non-converged exports. Emit on stdout so capsys (and standard log
        # collectors) pick it up alongside the structured `log` line.
        message = (
            "WARNING: exporting non-converged fit because "
            "allow_nonconverged_export=True is set. "
            f"diagnostics={diagnostics}."
        )
        print(message, flush=True)
        log(message)
        return
    # Soft guard: log a clear warning but still export. Hard-raising here
    # made small/short test runs (and legitimately interrupted production
    # runs whose partial fit is still informative) unrecoverable. Callers
    # that need strict gating can introspect ``fit_result.converged``.
    message = (
        "WARNING: fit did not converge; exporting non-converged artifact. "
        f"diagnostics={diagnostics}."
    )
    # Print to stdout so capsys / log aggregators see all four delta field
    # names alongside the structured stderr `log` line.
    print(message, flush=True)
    log(message)


def _write_predictions_and_summary(
    predictions_path: Path,
    dataset: LoadedDataset,
    model: BayesianPGS,
    is_training_dataset: bool = False,
) -> dict[str, Any]:
    log(f"computing predictions for {len(dataset.sample_ids)} samples, trait={model.config.trait_type.value}  mem={mem()}")
    # Only reuse the model's cached training decision components when we can
    # prove this dataset IS the training cohort. Two guards must hold:
    #   1) The caller explicitly asserts `is_training_dataset=True`.
    #   2) The fitted state exposes `training_sample_ids` AND they match
    #      `dataset.sample_ids` exactly (order included).
    # Shape match alone is unsafe: a held-out cohort with the same sample
    # count would silently borrow training scores. If `training_sample_ids`
    # is not tracked on FittedState yet, we recompute. A small perf cost is
    # preferable to silently emitting wrong predictions.
    # NOTE: FittedState in sv_pgs/model.py (line ~86) does not yet carry a
    # `training_sample_ids: Sequence[str] | None = None` field. Until that
    # field is added (and populated by the trainer), the cache is never
    # taken on this path. Adding it would let the cache fast-path engage.
    cached_components = None
    # Reuse the model's cached training decision components only when the
    # caller asserts this dataset IS the training cohort. If
    # `training_sample_ids` is exposed on the fitted state, additionally
    # require an exact ID match — a held-out cohort with the same sample
    # count must never silently borrow training scores.
    if is_training_dataset:
        training_components_getter = getattr(model, "training_decision_components", None)
        if training_components_getter is not None:
            candidate = None
            try:
                candidate = training_components_getter()
            except Exception:
                candidate = None
            if (
                candidate is not None
                and candidate[0].shape == (len(dataset.sample_ids),)
                and candidate[1].shape == (len(dataset.sample_ids),)
            ):
                fitted_state = getattr(model, "state", None)
                training_sample_ids = (
                    getattr(fitted_state, "training_sample_ids", None)
                    if fitted_state is not None
                    else None
                )
                ids_ok = True
                if training_sample_ids is not None:
                    try:
                        ids_ok = (
                            len(training_sample_ids) == len(dataset.sample_ids)
                            and list(training_sample_ids) == list(dataset.sample_ids)
                        )
                    except Exception:
                        ids_ok = False
                if ids_ok:
                    cached_components = candidate
    if cached_components is not None:
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


def _build_per_epoch_history_writer(
    *,
    history_path: Path,
    config: ModelConfig,
    test_dataset: "LoadedDataset | None",
) -> tuple[Callable[[dict[str, Any]], None], Callable[[], None]]:
    """Return (callback, close_fn) that streams per-epoch metrics to `history_path`.

    The callback signature matches what mixture_inference.fit_variational_em
    emits at the end of every outer iteration. Each invocation appends one
    TSV row containing both the EM-internal diagnostics (objective,
    parameter_change, sigma_e2, etc.) and — when a test_dataset is wired
    in — the derived held-out metrics (AUC + log-loss + accuracy for
    binary traits, R² + RMSE for quantitative).

    The TSV is opened in line-buffered append mode so a `tail -f` follows
    progress in real time. `close_fn` flushes and closes the underlying
    file handle; call it from a `finally:` block so a fit failure still
    produces a valid partial-history file.
    """
    import time as _time

    binary_trait = config.trait_type == TraitType.BINARY
    has_test = test_dataset is not None and len(test_dataset.sample_ids) > 0
    history_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(history_path, "w", buffering=1, encoding="utf-8", newline="")
    if binary_trait:
        header = [
            "epoch", "wall_seconds", "objective", "parameter_change",
            "sigma_error2", "global_scale", "nonzero_beta",
            "test_n", "test_cross_entropy", "test_auc", "test_log_loss",
            "test_accuracy", "test_predicted_positive_fraction",
        ]
    else:
        header = [
            "epoch", "wall_seconds", "objective", "parameter_change",
            "sigma_error2", "global_scale", "nonzero_beta",
            "test_n", "test_mse", "test_r2", "test_rmse",
        ]
    handle.write("\t".join(header) + "\n")
    handle.flush()

    start_seconds = _time.monotonic()
    _test_n = len(test_dataset.sample_ids) if test_dataset is not None and has_test else 0
    log(f"per-epoch history → {history_path}  (test_n={_test_n})")

    def _format(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            # Six significant digits is plenty for monitoring and stays
            # readable when the TSV is loaded into a notebook later.
            return f"{value:.6g}"
        return str(value)

    def callback(snapshot: dict[str, Any]) -> None:
        wall_seconds = _time.monotonic() - start_seconds
        row: list[str] = [
            _format(snapshot.get("epoch")),
            _format(wall_seconds),
            _format(snapshot.get("objective")),
            _format(snapshot.get("parameter_change")),
            _format(snapshot.get("sigma_error2")),
            _format(snapshot.get("global_scale")),
            _format(snapshot.get("nonzero_beta")),
        ]
        test_linear_predictor = snapshot.get("validation_linear_predictor")
        test_targets = snapshot.get("validation_targets")
        if has_test and test_linear_predictor is not None and test_targets is not None:
            test_linear_predictor = np.asarray(test_linear_predictor, dtype=np.float64)
            test_targets_array = np.asarray(test_targets, dtype=np.float64)
            n_test = int(test_targets_array.shape[0])
            if binary_trait:
                probs = np.asarray(stable_sigmoid(test_linear_predictor), dtype=np.float64)
                predicted_labels = (probs >= 0.5).astype(np.int32)
                pred_positive_fraction = float(np.mean(predicted_labels))
                accuracy = float(np.mean(predicted_labels == test_targets_array))
                # roc_auc_score requires both classes present; without that
                # the metric is undefined. Skip cleanly rather than throw.
                if np.unique(test_targets_array).shape[0] >= 2:
                    auc_value: float | None = float(roc_auc_score(test_targets_array, probs))
                else:
                    auc_value = None
                log_loss_value = float(log_loss(test_targets_array, probs, labels=[0.0, 1.0]))
                cross_entropy = float(snapshot.get("validation_metric") or float("nan"))
                row.extend([
                    str(n_test),
                    _format(cross_entropy),
                    _format(auc_value),
                    _format(log_loss_value),
                    _format(accuracy),
                    _format(pred_positive_fraction),
                ])
                log(
                    f"  >>> epoch {snapshot.get('epoch')}/{snapshot.get('total_epochs')}  "
                    f"TEST AUC={_format(auc_value)}  log_loss={log_loss_value:.4f}  "
                    f"accuracy={accuracy:.4f}  n={n_test} <<<"
                )
            else:
                residuals = test_targets_array - test_linear_predictor
                mse_value = float(np.mean(residuals * residuals))
                rmse_value = float(np.sqrt(mse_value))
                r2_value = float(r2_score(test_targets_array, test_linear_predictor))
                row.extend([
                    str(n_test),
                    _format(mse_value),
                    _format(r2_value),
                    _format(rmse_value),
                ])
                log(
                    f"  >>> epoch {snapshot.get('epoch')}/{snapshot.get('total_epochs')}  "
                    f"TEST R²={r2_value:.4f}  RMSE={rmse_value:.4f}  n={n_test} <<<"
                )
        else:
            # No held-out cohort wired in; emit the diagnostic-only row so
            # the file still has a continuous record of the EM trajectory.
            row.append("0")
            if binary_trait:
                row.extend(["", "", "", "", ""])
            else:
                row.extend(["", "", ""])
        handle.write("\t".join(row) + "\n")
        handle.flush()

    def close_fn() -> None:
        if not handle.closed:
            try:
                handle.flush()
            finally:
                handle.close()

    return callback, close_fn

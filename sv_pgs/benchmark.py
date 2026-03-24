from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm
from sklearn.metrics import average_precision_score, log_loss, r2_score, roc_auc_score

from sv_pgs.config import BenchmarkConfig, ModelConfig, TraitType
from sv_pgs.data import VariantRecord, normalize_variant_records
from sv_pgs.model import BayesianPGS


@dataclass(slots=True)
class BenchmarkMetrics:
    auc: float | None
    log_loss: float | None
    pr_auc: float | None
    calibration_intercept: float | None
    calibration_slope: float | None
    liability_r2: float | None
    r2: float | None
    top_tail_enrichment: float


def run_benchmark_suite(
    train_genotypes: np.ndarray,
    train_covariates: np.ndarray,
    train_targets: np.ndarray,
    test_genotypes: np.ndarray,
    test_covariates: np.ndarray,
    test_targets: np.ndarray,
    records: Sequence[VariantRecord | dict],
    benchmark_config: BenchmarkConfig | None = None,
) -> dict[str, BenchmarkMetrics]:
    benchmark_config = benchmark_config or BenchmarkConfig()
    shared_config = benchmark_config.shared_config

    normalized_records = normalize_variant_records(records)
    snv_mask = np.asarray(
        [record.variant_class in benchmark_config.snv_classes for record in normalized_records],
        dtype=bool,
    )
    snv_records = [
        record for record, is_snv in zip(normalized_records, snv_mask, strict=True) if is_snv
    ]

    current_snv_config = _copy_config(shared_config, update_hyperparameters=False)
    snv_only_config = _copy_config(shared_config)
    joint_config = _copy_config(shared_config)

    model_specs: list[tuple[str, BayesianPGS, np.ndarray]] = [
        (
            "current_snv_score",
            _fit_model(current_snv_config, train_genotypes[:, snv_mask], train_covariates, train_targets, snv_records),
            test_genotypes[:, snv_mask],
        ),
        (
            "snv_only_continuous",
            _fit_model(snv_only_config, train_genotypes[:, snv_mask], train_covariates, train_targets, snv_records),
            test_genotypes[:, snv_mask],
        ),
        (
            "joint_snv_sv_continuous",
            _fit_model(joint_config, train_genotypes, train_covariates, train_targets, normalized_records),
            test_genotypes,
        ),
    ]

    return {
        name: _compute_metrics(
            model=model,
            genotypes=model_test_genotypes,
            covariates=test_covariates,
            targets=test_targets,
            benchmark_config=benchmark_config,
        )
        for name, model, model_test_genotypes in model_specs
    }


def _copy_config(config: ModelConfig, **overrides) -> ModelConfig:
    payload = {config_field.name: getattr(config, config_field.name) for config_field in fields(config)}
    payload.update(overrides)
    return ModelConfig(**payload)


def _fit_model(
    config: ModelConfig,
    genotypes: np.ndarray,
    covariates: np.ndarray,
    targets: np.ndarray,
    records: Sequence[VariantRecord],
) -> BayesianPGS:
    model = BayesianPGS(config)
    model.fit(genotypes=genotypes, covariates=covariates, targets=targets, variant_records=records)
    return model


def _compute_metrics(
    model: BayesianPGS,
    genotypes: np.ndarray,
    covariates: np.ndarray,
    targets: np.ndarray,
    benchmark_config: BenchmarkConfig,
) -> BenchmarkMetrics:
    scores = model.decision_function(genotypes, covariates)
    trait_type = model.config.trait_type
    if trait_type == TraitType.BINARY:
        probabilities = expit(scores)
        auc = float(roc_auc_score(targets, probabilities))
        pr_auc = float(average_precision_score(targets, probabilities))
        loss = float(log_loss(targets, probabilities))
        intercept, slope = _calibration(probabilities, targets)
        liability_value = None
        if benchmark_config.prevalence is not None:
            liability_value = _liability_r2(probabilities, targets, benchmark_config.prevalence)
        return BenchmarkMetrics(
            auc=auc,
            log_loss=loss,
            pr_auc=pr_auc,
            calibration_intercept=intercept,
            calibration_slope=slope,
            liability_r2=liability_value,
            r2=None,
            top_tail_enrichment=_top_tail_enrichment(probabilities, targets, benchmark_config.top_tail_fraction),
        )

    return BenchmarkMetrics(
        auc=None,
        log_loss=None,
        pr_auc=None,
        calibration_intercept=None,
        calibration_slope=None,
        liability_r2=None,
        r2=float(r2_score(targets, scores)),
        top_tail_enrichment=_top_tail_enrichment(scores, targets, benchmark_config.top_tail_fraction),
    )


def _calibration(probabilities: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    logits = np.log(
        np.clip(probabilities, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - probabilities, 1e-6, 1.0 - 1e-6)
    )

    def objective(parameters: np.ndarray) -> float:
        intercept, slope = parameters
        calibrated_probabilities = expit(intercept + slope * logits)
        return -float(
            np.mean(
                targets * np.log(calibrated_probabilities + 1e-8)
                + (1.0 - targets) * np.log(1.0 - calibrated_probabilities + 1e-8)
            )
        )

    optimization = minimize(objective, x0=np.array([0.0, 1.0], dtype=np.float64), method="BFGS")
    return float(optimization.x[0]), float(optimization.x[1])


def _liability_r2(probabilities: np.ndarray, targets: np.ndarray, prevalence: float) -> float:
    observed_r2 = max(0.0, float(r2_score(targets, probabilities)))
    sample_prevalence = float(np.mean(targets))
    threshold = norm.ppf(1.0 - prevalence)
    density = norm.pdf(threshold)
    scale_value = ((prevalence * (1.0 - prevalence)) / density) ** 2 / max(
        sample_prevalence * (1.0 - sample_prevalence),
        1e-8,
    )
    return observed_r2 * scale_value


def _top_tail_enrichment(scores: np.ndarray, targets: np.ndarray, fraction: float) -> float:
    cutoff = max(1, int(np.ceil(scores.shape[0] * fraction)))
    top_indices = np.argsort(scores)[-cutoff:]
    baseline = float(np.mean(targets))
    if abs(baseline) < 1e-8:
        return 0.0
    return float(np.mean(targets[top_indices]) / baseline)

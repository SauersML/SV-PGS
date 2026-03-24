"""BayesR-style blockwise variational EM for the joint SNP+SV model.

Prior: K=5 class-adaptive Gaussian mixture (BayesR-style)
  beta_j ~ sum_k pi_{g_j,k} N(0, sigma_{g_j,k}^2)

Inference: blockwise eigenspace posterior updates with global hyperparameter learning.
Binary traits use Polya-Gamma working-response outer iterations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.special import logsumexp

from sv_pgs.blocks import BlockDecomposition, build_block_decomposition, compute_block_posterior
from sv_pgs.config import MIXTURE_COMPONENT_COUNT, ModelConfig, TraitType, VariantClass
from sv_pgs.data import TieMap, VariantRecord


@dataclass(slots=True)
class VariationalFitResult:
    alpha: np.ndarray
    beta_reduced: np.ndarray
    beta_variance: np.ndarray
    responsibilities: np.ndarray
    class_mixture_weights: dict[VariantClass, np.ndarray]
    component_variances: np.ndarray
    sigma_error2: float
    objective_history: list[float]
    validation_history: list[float]


def fit_variational_em(
    genotypes: np.ndarray,
    covariates: np.ndarray,
    targets: np.ndarray,
    records: Sequence[VariantRecord],
    tie_map: TieMap,
    config: ModelConfig,
    validation_data: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> VariationalFitResult:
    reduced_genotypes = np.asarray(genotypes[:, tie_map.kept_indices], dtype=np.float32)
    reduced_records = [records[int(idx)] for idx in tie_map.kept_indices]
    variant_count = reduced_genotypes.shape[1]
    covariate_matrix = np.asarray(covariates, dtype=np.float32)
    target_vector = np.asarray(targets, dtype=np.float32)

    block_decomposition = build_block_decomposition(
        reduced_genotypes.astype(np.float64), reduced_records, config,
    )

    class_lookup, inverse_class_lookup = _build_class_lookup(reduced_records)
    class_indices = np.array(
        [class_lookup[rec.variant_class] for rec in reduced_records], dtype=np.int32,
    )
    class_count = len(class_lookup)
    component_variances = config.component_variances()
    class_mixture_weights = _initialize_class_weights(inverse_class_lookup, class_count, config)

    coefficient_mean = np.zeros(variant_count, dtype=np.float32)
    coefficient_variance = np.full(variant_count, component_variances[0], dtype=np.float32)
    covariate_coefficients = np.zeros(covariate_matrix.shape[1], dtype=np.float32)
    responsibilities = _initialize_responsibilities(class_indices, class_mixture_weights)
    residual_variance = 1.0

    validation_payload = _prepare_validation(validation_data)
    objective_history: list[float] = []
    validation_history: list[float] = []

    outer_iteration = 0
    while outer_iteration < config.max_outer_iterations:
        linear_predictor = reduced_genotypes @ coefficient_mean + covariate_matrix @ covariate_coefficients
        sample_weights, pseudo_response, residual_variance = _likelihood_update(
            config.trait_type, target_vector, linear_predictor,
            config.sigma_error_floor, config.polya_gamma_minimum_weight,
        )

        genetic_prediction = reduced_genotypes @ coefficient_mean
        covariate_coefficients = _solve_covariates(
            covariate_matrix, pseudo_response, genetic_prediction, sample_weights,
        )

        prior_precision = _effective_prior_precision(responsibilities, component_variances)
        weighted_residual = (
            sample_weights * (pseudo_response - covariate_matrix @ covariate_coefficients)
        ).astype(np.float64)

        coefficient_mean, coefficient_variance = _blockwise_posterior_update(
            block_decomposition, reduced_genotypes.astype(np.float64),
            sample_weights.astype(np.float64), weighted_residual,
            prior_precision.astype(np.float64), coefficient_mean,
        )

        updated_genetic = reduced_genotypes @ coefficient_mean
        covariate_coefficients = _solve_covariates(
            covariate_matrix, pseudo_response, updated_genetic, sample_weights,
        )

        expected_beta_squared = coefficient_mean.astype(np.float64) ** 2 + coefficient_variance.astype(np.float64)
        responsibilities = _update_responsibilities(
            expected_beta_squared, component_variances, class_indices, class_mixture_weights,
        )

        if config.update_hyperparameters:
            class_mixture_weights = _update_class_weights(
                class_indices, responsibilities, class_mixture_weights,
                class_count, config.dirichlet_concentration,
            )
            component_variances = _update_component_variances(
                responsibilities, expected_beta_squared, component_variances,
            )

        updated_predictor = reduced_genotypes @ coefficient_mean + covariate_matrix @ covariate_coefficients
        objective_history.append(_compute_objective(
            config.trait_type, target_vector, updated_predictor,
            coefficient_mean, prior_precision, residual_variance,
        ))

        if validation_payload is not None:
            val_geno, val_cov, val_targ = validation_payload
            val_pred = val_geno @ coefficient_mean + val_cov @ covariate_coefficients
            validation_history.append(_validation_metric(config.trait_type, val_targ, val_pred))

        outer_iteration += 1
        if len(objective_history) >= 2:
            if abs(objective_history[-1] - objective_history[-2]) < config.convergence_tolerance:
                break

    return VariationalFitResult(
        alpha=covariate_coefficients,
        beta_reduced=coefficient_mean,
        beta_variance=coefficient_variance,
        responsibilities=responsibilities.astype(np.float32),
        class_mixture_weights={
            inverse_class_lookup[class_idx]: class_mixture_weights[class_idx].astype(np.float32)
            for class_idx in range(class_count)
        },
        component_variances=component_variances.astype(np.float32),
        sigma_error2=float(residual_variance),
        objective_history=objective_history,
        validation_history=validation_history,
    )


def _build_class_lookup(
    records: Sequence[VariantRecord],
) -> tuple[dict[VariantClass, int], dict[int, VariantClass]]:
    unique_classes = sorted({rec.variant_class for rec in records}, key=lambda vc: vc.value)
    class_lookup = {vc: idx for idx, vc in enumerate(unique_classes)}
    inverse_lookup = {idx: vc for vc, idx in class_lookup.items()}
    return class_lookup, inverse_lookup


def _initialize_class_weights(
    inverse_class_lookup: dict[int, VariantClass],
    class_count: int,
    config: ModelConfig,
) -> np.ndarray:
    default_weights = config.class_mixture_weights()
    weights = np.zeros((class_count, MIXTURE_COMPONENT_COUNT), dtype=np.float64)
    for class_idx in range(class_count):
        variant_class = inverse_class_lookup[class_idx]
        class_weights = default_weights.get(variant_class, None)
        if class_weights is not None:
            weights[class_idx] = class_weights
        else:
            weights[class_idx] = 1.0 / MIXTURE_COMPONENT_COUNT
    return weights


def _initialize_responsibilities(
    class_indices: np.ndarray,
    class_mixture_weights: np.ndarray,
) -> np.ndarray:
    return class_mixture_weights[class_indices].copy()


def _prepare_validation(
    validation_data: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if validation_data is None:
        return None
    val_geno, val_cov, val_targ = validation_data
    return (
        np.asarray(val_geno, dtype=np.float32),
        np.asarray(val_cov, dtype=np.float32),
        np.asarray(val_targ, dtype=np.float32),
    )


def _likelihood_update(
    trait_type: TraitType,
    targets: np.ndarray,
    linear_predictor: np.ndarray,
    sigma_error_floor: float,
    min_pg_weight: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    if trait_type == TraitType.BINARY:
        sample_weights = _polya_gamma_expectation(linear_predictor, min_pg_weight)
        pseudo_response = (targets - 0.5) / np.maximum(sample_weights, 1e-10)
        return sample_weights, pseudo_response, 1.0
    residual = targets - linear_predictor
    sigma_e2 = float(np.mean(residual * residual) + sigma_error_floor)
    sample_weights = np.full(targets.shape[0], 1.0 / sigma_e2, dtype=np.float32)
    return sample_weights, targets, sigma_e2


def _polya_gamma_expectation(
    linear_predictor: np.ndarray,
    min_weight: float,
) -> np.ndarray:
    abs_eta = np.abs(linear_predictor)
    safe_eta = np.where(abs_eta < 1e-6, 1.0, abs_eta)
    weights = 0.5 * np.tanh(safe_eta / 2.0) / safe_eta
    weights = np.where(abs_eta < 1e-6, 0.25, weights)
    return np.maximum(weights, min_weight).astype(np.float32)


def _solve_covariates(
    covariate_matrix: np.ndarray,
    pseudo_response: np.ndarray,
    genetic_prediction: np.ndarray,
    sample_weights: np.ndarray,
) -> np.ndarray:
    weighted_cov = covariate_matrix.T * sample_weights[None, :]
    normal_matrix = weighted_cov @ covariate_matrix
    rhs = weighted_cov @ (pseudo_response - genetic_prediction)
    jitter = 1e-6 * np.eye(covariate_matrix.shape[1], dtype=np.float32)
    return np.linalg.solve(normal_matrix + jitter, rhs).astype(np.float32)


def _effective_prior_precision(
    responsibilities: np.ndarray,
    component_variances: np.ndarray,
) -> np.ndarray:
    safe_variances = np.maximum(component_variances, 1e-12)
    component_precisions = 1.0 / safe_variances
    return np.sum(responsibilities * component_precisions[None, :], axis=1)


def _blockwise_posterior_update(
    block_decomposition: BlockDecomposition,
    genotypes: np.ndarray,
    sample_weights: np.ndarray,
    weighted_residual: np.ndarray,
    prior_precision: np.ndarray,
    current_mean: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    updated_mean = current_mean.copy().astype(np.float32)
    updated_variance = np.zeros_like(current_mean, dtype=np.float32)

    for ld_block in block_decomposition.blocks:
        block_indices = ld_block.variant_indices
        block_genetic = genotypes[:, block_indices] @ current_mean[block_indices].astype(np.float64)
        block_residual = weighted_residual + sample_weights * block_genetic

        block_mean, block_var = compute_block_posterior(
            ld_block, genotypes, sample_weights, block_residual, prior_precision,
        )
        updated_mean[block_indices] = block_mean
        updated_variance[block_indices] = block_var

    return updated_mean, updated_variance


def _update_responsibilities(
    expected_beta_squared: np.ndarray,
    component_variances: np.ndarray,
    class_indices: np.ndarray,
    class_mixture_weights: np.ndarray,
) -> np.ndarray:
    safe_variances = np.maximum(component_variances, 1e-12)
    log_weights = np.log(class_mixture_weights[class_indices] + 1e-12)
    log_normals = (
        -0.5 * np.log(safe_variances)[None, :]
        - 0.5 * expected_beta_squared[:, None] / safe_variances[None, :]
    )
    log_resp = log_weights + log_normals
    log_resp -= logsumexp(log_resp, axis=1, keepdims=True)
    return np.exp(log_resp)


def _update_class_weights(
    class_indices: np.ndarray,
    responsibilities: np.ndarray,
    current_weights: np.ndarray,
    class_count: int,
    dirichlet_concentration: float,
) -> np.ndarray:
    updated = np.zeros_like(current_weights)
    for class_idx in range(class_count):
        class_mask = class_indices == class_idx
        prior_counts = dirichlet_concentration * current_weights[class_idx]
        posterior_counts = responsibilities[class_mask].sum(axis=0) + prior_counts
        updated[class_idx] = posterior_counts / posterior_counts.sum()
    return updated


def _update_component_variances(
    responsibilities: np.ndarray,
    expected_beta_squared: np.ndarray,
    current_variances: np.ndarray,
) -> np.ndarray:
    component_count = current_variances.shape[0]
    updated = np.zeros(component_count, dtype=np.float64)
    for comp_idx in range(component_count):
        total_weight = responsibilities[:, comp_idx].sum() + 1e-8
        weighted_sq = (responsibilities[:, comp_idx] * expected_beta_squared).sum()
        updated[comp_idx] = weighted_sq / total_weight
    updated = np.sort(updated)
    updated[0] = max(updated[0], current_variances[0])
    return np.maximum(updated, 1e-12)


def _compute_objective(
    trait_type: TraitType,
    targets: np.ndarray,
    linear_predictor: np.ndarray,
    coefficient_mean: np.ndarray,
    prior_precision: np.ndarray,
    residual_variance: float,
) -> float:
    if trait_type == TraitType.BINARY:
        log_lik = float(np.sum(
            targets * (-np.logaddexp(0.0, -linear_predictor))
            + (1.0 - targets) * (-np.logaddexp(0.0, linear_predictor))
        ))
    else:
        residual = targets - linear_predictor
        log_lik = float(-0.5 * np.sum(residual * residual / max(residual_variance, 1e-8)))
    prior_term = float(-0.5 * np.sum(prior_precision * coefficient_mean.astype(np.float64) ** 2))
    return log_lik + prior_term


def _validation_metric(
    trait_type: TraitType,
    targets: np.ndarray,
    linear_predictor: np.ndarray,
) -> float:
    if trait_type == TraitType.BINARY:
        prob = 1.0 / (1.0 + np.exp(-linear_predictor))
        return float(-np.mean(
            targets * np.log(prob + 1e-8)
            + (1.0 - targets) * np.log(1.0 - prob + 1e-8)
        ))
    residual = targets - linear_predictor
    return float(np.mean(residual * residual))

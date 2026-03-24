"""Blockwise low-rank variational EM for a metadata-conditioned BayesR-style mixture prior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax.numpy as jnp
import numpy as np

from sv_pgs.blocks import build_block_decomposition, compute_block_posterior
from sv_pgs.config import ModelConfig, TraitType, VariantClass
from sv_pgs.data import VariantRecord
from sv_pgs.operator import GenotypeOperator, matvec


@dataclass(slots=True)
class PriorDesign:
    design_matrix: np.ndarray
    feature_names: list[str]
    class_membership_matrix: np.ndarray
    inverse_class_lookup: dict[int, VariantClass]


@dataclass(slots=True)
class VariationalFitResult:
    alpha: np.ndarray
    beta_reduced: np.ndarray
    beta_variance: np.ndarray
    prior_scales: np.ndarray
    class_mixture_weights: dict[VariantClass, np.ndarray]
    scale_model_coefficients: np.ndarray
    scale_model_feature_names: list[str]
    sigma_error2: float
    objective_history: list[float]
    validation_history: list[float]


def fit_variational_em(
    genotypes: np.ndarray,
    covariates: np.ndarray,
    targets: np.ndarray,
    records: Sequence[VariantRecord],
    config: ModelConfig,
    validation_data: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> VariationalFitResult:
    genotype_matrix = np.asarray(genotypes, dtype=np.float32)
    covariate_host = np.asarray(covariates, dtype=np.float32)
    target_host = np.asarray(targets, dtype=np.float32)
    prior_design = _build_prior_design(records)
    genotype_operator = GenotypeOperator.from_numpy(genotype_matrix, tile_size=config.tile_size)
    block_decomposition = build_block_decomposition(genotype_matrix.astype(np.float64), records, config)
    validation_payload = _prepare_validation(validation_data)

    scale_model_coefficients = _initialize_scale_model_coefficients(prior_design, config)
    prior_scales = _prior_scales_from_coefficients(scale_model_coefficients, prior_design.design_matrix, config)
    component_variances = np.asarray(config.mixture_variance_multipliers, dtype=np.float64)
    inverse_component_variances = 1.0 / component_variances
    class_weight_matrix = _initialize_class_weight_matrix(prior_design, config)
    responsibilities = _local_class_weights(prior_design.class_membership_matrix, class_weight_matrix)

    coefficient_mean = np.zeros(genotype_matrix.shape[1], dtype=np.float32)
    coefficient_variance = np.asarray(prior_scales * component_variances[0], dtype=np.float32)
    covariate_coefficients = np.zeros(covariate_host.shape[1], dtype=np.float32)
    residual_variance = 1.0

    objective_history: list[float] = []
    validation_history: list[float] = []

    outer_iteration = 0
    while outer_iteration < config.max_outer_iterations:
        linear_predictor = np.asarray(matvec(genotype_operator, jnp.asarray(coefficient_mean))) + covariate_host @ covariate_coefficients
        sample_weights, pseudo_response, residual_variance = _likelihood_update(
            trait_type=config.trait_type,
            targets=target_host,
            linear_predictor=linear_predictor,
            sigma_error_floor=config.sigma_error_floor,
            minimum_polya_gamma_weight=config.polya_gamma_minimum_weight,
        )
        current_genetic_prediction = np.asarray(matvec(genotype_operator, jnp.asarray(coefficient_mean)))
        covariate_coefficients = _solve_covariates(
            covariate_matrix=covariate_host,
            pseudo_response=pseudo_response,
            current_genetic_prediction=current_genetic_prediction,
            sample_weights=sample_weights,
        )

        expected_inverse_mixture = responsibilities @ inverse_component_variances
        prior_precision = expected_inverse_mixture / np.maximum(prior_scales, 1e-12)
        block_weighted_residual = sample_weights * (pseudo_response - covariate_host @ covariate_coefficients - current_genetic_prediction)
        coefficient_mean, coefficient_variance = _update_block_posterior(
            genotype_matrix=genotype_matrix,
            block_decomposition=block_decomposition,
            sample_weights=sample_weights,
            weighted_residual=block_weighted_residual,
            prior_precision=prior_precision,
            previous_mean=coefficient_mean,
        )
        updated_genetic_prediction = np.asarray(matvec(genotype_operator, jnp.asarray(coefficient_mean)))
        covariate_coefficients = _solve_covariates(
            covariate_matrix=covariate_host,
            pseudo_response=pseudo_response,
            current_genetic_prediction=updated_genetic_prediction,
            sample_weights=sample_weights,
        )

        coefficient_second_moment = coefficient_mean.astype(np.float64) ** 2 + coefficient_variance.astype(np.float64)
        responsibilities = _update_responsibilities(
            coefficient_second_moment=coefficient_second_moment,
            prior_scales=prior_scales,
            class_membership_matrix=prior_design.class_membership_matrix,
            class_weight_matrix=class_weight_matrix,
            component_variances=component_variances,
        )
        class_weight_matrix = _update_class_weight_matrix(
            responsibilities=responsibilities,
            class_membership_matrix=prior_design.class_membership_matrix,
            prior_design=prior_design,
            config=config,
        )
        if config.update_hyperparameters:
            scale_model_coefficients = _update_scale_model(
                design_matrix=prior_design.design_matrix,
                feature_names=prior_design.feature_names,
                coefficient_second_moment=coefficient_second_moment,
                expected_inverse_mixture=responsibilities @ inverse_component_variances,
                current_coefficients=scale_model_coefficients,
                config=config,
            )
            prior_scales = _prior_scales_from_coefficients(scale_model_coefficients, prior_design.design_matrix, config)

        updated_linear_predictor = updated_genetic_prediction + covariate_host @ covariate_coefficients
        objective_history.append(
            _surrogate_objective(
                trait_type=config.trait_type,
                targets=target_host,
                linear_predictor=updated_linear_predictor,
                coefficient_second_moment=coefficient_second_moment,
                prior_scales=prior_scales,
                class_membership_matrix=prior_design.class_membership_matrix,
                class_weight_matrix=class_weight_matrix,
                responsibilities=responsibilities,
                component_variances=component_variances,
                residual_variance=residual_variance,
            )
        )

        if validation_payload is not None:
            validation_genotypes, validation_covariates, validation_targets = validation_payload
            validation_linear_predictor = validation_genotypes @ coefficient_mean + validation_covariates @ covariate_coefficients
            validation_history.append(
                _validation_metric(
                    trait_type=config.trait_type,
                    targets=validation_targets,
                    linear_predictor=validation_linear_predictor,
                )
            )

        outer_iteration += 1
        if len(objective_history) >= 2:
            objective_delta = abs(objective_history[-1] - objective_history[-2])
            if objective_delta < config.convergence_tolerance:
                break

    return VariationalFitResult(
        alpha=covariate_coefficients.astype(np.float32),
        beta_reduced=coefficient_mean.astype(np.float32),
        beta_variance=coefficient_variance.astype(np.float32),
        prior_scales=prior_scales.astype(np.float32),
        class_mixture_weights={
            prior_design.inverse_class_lookup[class_index]: class_weight_matrix[class_index].astype(np.float32)
            for class_index in range(class_weight_matrix.shape[0])
        },
        scale_model_coefficients=scale_model_coefficients.astype(np.float32),
        scale_model_feature_names=list(prior_design.feature_names),
        sigma_error2=float(residual_variance),
        objective_history=objective_history,
        validation_history=validation_history,
    )


def _prepare_validation(
    validation_data: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if validation_data is None:
        return None
    validation_genotypes, validation_covariates, validation_targets = validation_data
    return (
        np.asarray(validation_genotypes, dtype=np.float32),
        np.asarray(validation_covariates, dtype=np.float32),
        np.asarray(validation_targets, dtype=np.float32),
    )


def _build_prior_design(records: Sequence[VariantRecord]) -> PriorDesign:
    unique_classes = sorted(
        {
            prior_class
            for record in records
            for prior_class in _prior_class_members(record)
        },
        key=lambda variant_class: variant_class.value,
    )
    class_lookup = {variant_class: class_index for class_index, variant_class in enumerate(unique_classes)}
    inverse_class_lookup = {
        class_index: variant_class for variant_class, class_index in class_lookup.items()
    }
    class_membership_matrix = np.zeros((len(records), len(unique_classes)), dtype=np.float64)
    for record_index, record in enumerate(records):
        prior_classes = _prior_class_members(record)
        prior_membership = _prior_class_membership(record)
        for prior_class, prior_weight in zip(prior_classes, prior_membership, strict=True):
            class_membership_matrix[record_index, class_lookup[prior_class]] = prior_weight

    log_length = np.log(np.maximum(np.asarray([record.length for record in records], dtype=np.float64), 1.0))
    allele_frequency = np.clip(np.asarray([record.allele_frequency for record in records], dtype=np.float64), 1e-6, 0.5)
    quality = np.clip(np.asarray([record.quality for record in records], dtype=np.float64), 1e-6, 1.0)
    repeat_indicator = np.asarray([float(record.is_repeat) for record in records], dtype=np.float64)
    copy_number_indicator = np.asarray([float(record.is_copy_number) for record in records], dtype=np.float64)

    standardized_log_length = _standardize_metadata(log_length)
    standardized_allele_frequency = _standardize_metadata(allele_frequency)
    standardized_quality = _standardize_metadata(quality)

    design_columns: list[np.ndarray] = [np.ones(len(records), dtype=np.float64)]
    feature_names = ["intercept"]
    for class_index, variant_class in enumerate(unique_classes):
        class_mask = class_membership_matrix[:, class_index]
        design_columns.append(class_mask)
        feature_names.append(f"type_offset::{variant_class.value}")
        design_columns.append(class_mask * standardized_log_length)
        feature_names.append(f"log_length_linear::{variant_class.value}")
        design_columns.append(class_mask * standardized_log_length * standardized_log_length)
        feature_names.append(f"log_length_quadratic::{variant_class.value}")
        design_columns.append(class_mask * standardized_allele_frequency)
        feature_names.append(f"allele_frequency_linear::{variant_class.value}")
        design_columns.append(class_mask * standardized_allele_frequency * standardized_allele_frequency)
        feature_names.append(f"allele_frequency_quadratic::{variant_class.value}")

    design_columns.append(standardized_quality)
    feature_names.append("quality_linear")
    design_columns.append(standardized_quality * standardized_quality)
    feature_names.append("quality_quadratic")
    design_columns.append(repeat_indicator)
    feature_names.append("repeat_indicator")
    design_columns.append(copy_number_indicator)
    feature_names.append("copy_number_indicator")

    return PriorDesign(
        design_matrix=np.column_stack(design_columns).astype(np.float64),
        feature_names=feature_names,
        class_membership_matrix=class_membership_matrix,
        inverse_class_lookup=inverse_class_lookup,
    )


def _prior_class_members(record: VariantRecord) -> tuple[VariantClass, ...]:
    if record.prior_class_members:
        return record.prior_class_members
    return (record.variant_class,)


def _prior_class_membership(record: VariantRecord) -> tuple[float, ...]:
    if record.prior_class_membership:
        return record.prior_class_membership
    return (1.0,)


def _standardize_metadata(values: np.ndarray) -> np.ndarray:
    centered_values = values - float(np.mean(values))
    scale_value = float(np.std(values))
    if scale_value < 1e-8:
        return np.zeros_like(values)
    return centered_values / scale_value


def _initialize_scale_model_coefficients(prior_design: PriorDesign, config: ModelConfig) -> np.ndarray:
    default_log_scales = config.class_log_prior_scales()
    class_baselines = np.asarray(
        [default_log_scales[prior_design.inverse_class_lookup[class_index]] for class_index in range(len(prior_design.inverse_class_lookup))],
        dtype=np.float64,
    )
    average_baseline = float(np.mean(class_baselines))
    initialized_coefficients = np.zeros(prior_design.design_matrix.shape[1], dtype=np.float64)
    initialized_coefficients[0] = average_baseline
    for feature_index, feature_name in enumerate(prior_design.feature_names):
        if not feature_name.startswith("type_offset::"):
            continue
        variant_class = VariantClass(feature_name.split("::", maxsplit=1)[1])
        initialized_coefficients[feature_index] = default_log_scales[variant_class] - average_baseline
    return initialized_coefficients


def _initialize_class_weight_matrix(prior_design: PriorDesign, config: ModelConfig) -> np.ndarray:
    default_class_weights = config.class_mixture_weights()
    return np.vstack(
        [default_class_weights[prior_design.inverse_class_lookup[class_index]] for class_index in range(len(prior_design.inverse_class_lookup))]
    ).astype(np.float64)


def _local_class_weights(
    class_membership_matrix: np.ndarray,
    class_weight_matrix: np.ndarray,
) -> np.ndarray:
    local_class_weights = class_membership_matrix @ class_weight_matrix
    local_class_weights = np.maximum(local_class_weights, 1e-12)
    return local_class_weights / np.sum(local_class_weights, axis=1, keepdims=True)


def _prior_scales_from_coefficients(
    scale_model_coefficients: np.ndarray,
    design_matrix: np.ndarray,
    config: ModelConfig,
) -> np.ndarray:
    linear_prediction = design_matrix @ scale_model_coefficients
    bounded_log_scales = np.clip(
        linear_prediction,
        np.log(config.prior_scale_floor),
        np.log(config.prior_scale_ceiling),
    )
    return np.exp(bounded_log_scales).astype(np.float64)


def _likelihood_update(
    trait_type: TraitType,
    targets: np.ndarray,
    linear_predictor: np.ndarray,
    sigma_error_floor: float,
    minimum_polya_gamma_weight: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    if trait_type == TraitType.BINARY:
        sample_weights = _polya_gamma_expectation(linear_predictor, minimum_polya_gamma_weight)
        pseudo_response = (targets - 0.5) / sample_weights
        return sample_weights, pseudo_response, 1.0

    residual_vector = targets - linear_predictor
    residual_variance = float(np.mean(residual_vector * residual_vector) + sigma_error_floor)
    sample_weights = np.full(targets.shape[0], 1.0 / residual_variance, dtype=np.float32)
    return sample_weights, targets, residual_variance


def _polya_gamma_expectation(
    linear_predictor: np.ndarray,
    minimum_polya_gamma_weight: float,
) -> np.ndarray:
    absolute_linear_predictor = np.abs(linear_predictor)
    safe_linear_predictor = np.where(absolute_linear_predictor < 1e-6, 1.0, absolute_linear_predictor)
    expected_weight = 0.5 * np.tanh(safe_linear_predictor / 2.0) / safe_linear_predictor
    expected_weight = np.where(absolute_linear_predictor < 1e-6, 0.25, expected_weight)
    return np.maximum(expected_weight, minimum_polya_gamma_weight).astype(np.float32)


def _solve_covariates(
    covariate_matrix: np.ndarray,
    pseudo_response: np.ndarray,
    current_genetic_prediction: np.ndarray,
    sample_weights: np.ndarray,
) -> np.ndarray:
    weighted_covariates = np.transpose(covariate_matrix) * sample_weights[None, :]
    normal_matrix = weighted_covariates @ covariate_matrix
    right_hand_side = weighted_covariates @ (pseudo_response - current_genetic_prediction)
    ridge_jitter = 1e-6 * np.eye(covariate_matrix.shape[1], dtype=np.float32)
    return np.linalg.solve(normal_matrix + ridge_jitter, right_hand_side).astype(np.float32)


def _update_block_posterior(
    genotype_matrix: np.ndarray,
    block_decomposition,
    sample_weights: np.ndarray,
    weighted_residual: np.ndarray,
    prior_precision: np.ndarray,
    previous_mean: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    updated_mean = previous_mean.astype(np.float32).copy()
    updated_variance = np.zeros_like(updated_mean, dtype=np.float32)
    current_weighted_residual = weighted_residual.astype(np.float64).copy()

    for ld_block in block_decomposition.blocks:
        block_indices = ld_block.variant_indices
        block_genotypes = genotype_matrix[:, block_indices].astype(np.float64)
        old_block_mean = updated_mean[block_indices].astype(np.float64)
        current_weighted_residual += sample_weights.astype(np.float64) * (block_genotypes @ old_block_mean)
        block_mean, block_variance = compute_block_posterior(
            ld_block,
            genotype_matrix.astype(np.float64),
            sample_weights.astype(np.float64),
            current_weighted_residual,
            prior_precision.astype(np.float64),
        )
        updated_mean[block_indices] = block_mean
        updated_variance[block_indices] = block_variance
        current_weighted_residual -= sample_weights.astype(np.float64) * (block_genotypes @ block_mean.astype(np.float64))

    return updated_mean, updated_variance


def _update_responsibilities(
    coefficient_second_moment: np.ndarray,
    prior_scales: np.ndarray,
    class_membership_matrix: np.ndarray,
    class_weight_matrix: np.ndarray,
    component_variances: np.ndarray,
) -> np.ndarray:
    local_class_weights = _local_class_weights(class_membership_matrix, class_weight_matrix)
    component_variance_matrix = prior_scales[:, None] * component_variances[None, :]
    log_responsibilities = np.log(np.maximum(local_class_weights, 1e-12))
    log_responsibilities -= 0.5 * np.log(component_variance_matrix)
    log_responsibilities -= 0.5 * coefficient_second_moment[:, None] / np.maximum(component_variance_matrix, 1e-12)
    log_responsibilities -= np.max(log_responsibilities, axis=1, keepdims=True)
    unnormalized_weights = np.exp(log_responsibilities)
    return unnormalized_weights / np.sum(unnormalized_weights, axis=1, keepdims=True)


def _update_class_weight_matrix(
    responsibilities: np.ndarray,
    class_membership_matrix: np.ndarray,
    prior_design: PriorDesign,
    config: ModelConfig,
) -> np.ndarray:
    default_class_weights = config.class_mixture_weights()
    component_count = responsibilities.shape[1]
    updated_weight_matrix = np.zeros((len(prior_design.inverse_class_lookup), component_count), dtype=np.float64)
    for class_index in range(updated_weight_matrix.shape[0]):
        membership_weight = class_membership_matrix[:, class_index][:, None]
        class_responsibility_sum = np.sum(membership_weight * responsibilities, axis=0)
        default_weights = default_class_weights[prior_design.inverse_class_lookup[class_index]]
        posterior_weights = class_responsibility_sum + config.dirichlet_strength * default_weights
        updated_weight_matrix[class_index] = posterior_weights / np.sum(posterior_weights)
    return updated_weight_matrix


def _update_scale_model(
    design_matrix: np.ndarray,
    feature_names: Sequence[str],
    coefficient_second_moment: np.ndarray,
    expected_inverse_mixture: np.ndarray,
    current_coefficients: np.ndarray,
    config: ModelConfig,
) -> np.ndarray:
    penalty_diagonal = _scale_model_penalty(feature_names, config)
    updated_coefficients = current_coefficients.copy()
    sufficient_statistic = np.maximum(coefficient_second_moment * expected_inverse_mixture, 1e-12)

    iteration_index = 0
    while iteration_index < config.maximum_scale_model_iterations:
        linear_prediction = design_matrix @ updated_coefficients
        bounded_linear_prediction = np.clip(
            linear_prediction,
            np.log(config.prior_scale_floor),
            np.log(config.prior_scale_ceiling),
        )
        working_weight = 0.5 * sufficient_statistic * np.exp(-bounded_linear_prediction)
        gradient_vector = design_matrix.T @ (-0.5 + working_weight) - penalty_diagonal * updated_coefficients
        weighted_design = design_matrix * working_weight[:, None]
        hessian_matrix = -(design_matrix.T @ weighted_design) - np.diag(penalty_diagonal)
        coefficient_step = np.linalg.solve(hessian_matrix, gradient_vector)
        updated_coefficients -= coefficient_step
        if float(np.max(np.abs(coefficient_step))) < 1e-5:
            break
        iteration_index += 1

    return updated_coefficients


def _scale_model_penalty(feature_names: Sequence[str], config: ModelConfig) -> np.ndarray:
    penalty_values = np.full(len(feature_names), config.scale_model_ridge_penalty, dtype=np.float64)
    penalty_values[0] = 1e-6
    for feature_index, feature_name in enumerate(feature_names):
        if feature_name.startswith("type_offset::"):
            penalty_values[feature_index] = config.type_offset_penalty
    return penalty_values


def _surrogate_objective(
    trait_type: TraitType,
    targets: np.ndarray,
    linear_predictor: np.ndarray,
    coefficient_second_moment: np.ndarray,
    prior_scales: np.ndarray,
    class_membership_matrix: np.ndarray,
    class_weight_matrix: np.ndarray,
    responsibilities: np.ndarray,
    component_variances: np.ndarray,
    residual_variance: float,
) -> float:
    if trait_type == TraitType.BINARY:
        log_likelihood = float(
            np.sum(
                targets * (-np.logaddexp(0.0, -linear_predictor))
                + (1.0 - targets) * (-np.logaddexp(0.0, linear_predictor))
            )
        )
    else:
        residual_vector = targets - linear_predictor
        log_likelihood = float(
            -0.5
            * np.sum(
                np.log(2.0 * np.pi * residual_variance)
                + (residual_vector * residual_vector) / max(residual_variance, 1e-8)
            )
        )

    local_class_weights = _local_class_weights(class_membership_matrix, class_weight_matrix)
    component_variance_matrix = prior_scales[:, None] * component_variances[None, :]
    prior_term = float(
        np.sum(
            responsibilities
            * (
                np.log(np.maximum(local_class_weights, 1e-12))
                - 0.5 * np.log(np.maximum(component_variance_matrix, 1e-12))
                - 0.5 * coefficient_second_moment[:, None] / np.maximum(component_variance_matrix, 1e-12)
            )
        )
    )
    entropy_term = float(-np.sum(responsibilities * np.log(np.maximum(responsibilities, 1e-12))))
    return log_likelihood + prior_term + entropy_term


def _validation_metric(
    trait_type: TraitType,
    targets: np.ndarray,
    linear_predictor: np.ndarray,
) -> float:
    if trait_type == TraitType.BINARY:
        positive_probability = 1.0 / (1.0 + np.exp(-linear_predictor))
        return float(
            -np.mean(
                targets * np.log(positive_probability + 1e-8)
                + (1.0 - targets) * np.log(1.0 - positive_probability + 1e-8)
            )
        )
    residual_vector = targets - linear_predictor
    return float(np.mean(residual_vector * residual_vector))

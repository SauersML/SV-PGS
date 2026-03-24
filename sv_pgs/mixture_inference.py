"""Metadata-adaptive TPB / gamma-gamma shrinkage with a global JAX PCG solve."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammaln

from sv_pgs.blocks import (
    BlockDecomposition,
    apply_block_preconditioner,
    build_block_decomposition,
    estimate_variance_from_blocks,
)
from sv_pgs.config import ModelConfig, TraitType, VariantClass
from sv_pgs.data import VariantRecord
from sv_pgs.operator import (
    GenotypeOperator,
    apply_precision_matrix,
    matvec,
    weighted_rmatvec,
)


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
    global_scale: float
    class_tpb_shape_a: dict[VariantClass, float]
    class_tpb_shape_b: dict[VariantClass, float]
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
    covariate_matrix = np.asarray(covariates, dtype=np.float32)
    target_vector = np.asarray(targets, dtype=np.float32)

    genotype_operator = GenotypeOperator.from_numpy(
        genotype_matrix,
        tile_size=config.operator_tile_size,
    )
    covariate_matrix_device = jnp.asarray(covariate_matrix, dtype=jnp.float32)
    target_vector_device = jnp.asarray(target_vector, dtype=jnp.float32)
    block_decomposition = build_block_decomposition(genotype_matrix, records, config)
    prior_design = _build_prior_design(records)
    validation_payload = _prepare_validation(validation_data)

    scale_model_coefficients = _initialize_scale_model_coefficients(prior_design, config)
    metadata_baseline_scales = _metadata_baseline_scales_from_coefficients(
        scale_model_coefficients,
        prior_design.design_matrix,
        config,
    )
    global_scale = 1.0
    baseline_prior_variances = (global_scale * metadata_baseline_scales) ** 2

    tpb_shape_a_vector = _initialize_tpb_shape_a_vector(prior_design, config)
    tpb_shape_b_vector = _initialize_tpb_shape_b_vector(prior_design, config)
    local_shape_a = prior_design.class_membership_matrix @ tpb_shape_a_vector
    local_shape_b = prior_design.class_membership_matrix @ tpb_shape_b_vector

    coefficient_mean = jnp.zeros(genotype_matrix.shape[1], dtype=jnp.float32)
    covariate_coefficients = jnp.zeros(covariate_matrix.shape[1], dtype=jnp.float32)
    local_scale = np.ones(genotype_matrix.shape[1], dtype=np.float64)
    auxiliary_delta = local_shape_b.copy()
    expected_inverse_local_scale = 1.0 / np.maximum(local_scale, config.local_scale_floor)
    coefficient_variance = baseline_prior_variances.astype(np.float32)
    residual_variance = 1.0

    objective_history: list[float] = []
    validation_history: list[float] = []

    outer_iteration = 0
    while outer_iteration < config.max_outer_iterations:
        current_genetic_prediction = matvec(genotype_operator, coefficient_mean)
        linear_predictor = current_genetic_prediction + covariate_matrix_device @ covariate_coefficients

        sample_weights, pseudo_response, residual_variance = _likelihood_update(
            trait_type=config.trait_type,
            targets=target_vector_device,
            linear_predictor=linear_predictor,
            sigma_error_floor=config.sigma_error_floor,
            minimum_polya_gamma_weight=config.polya_gamma_minimum_weight,
        )

        covariate_coefficients = _solve_covariates(
            covariate_matrix=covariate_matrix_device,
            pseudo_response=pseudo_response,
            current_genetic_prediction=current_genetic_prediction,
            sample_weights=sample_weights,
        )

        prior_precision = expected_inverse_local_scale / np.maximum(baseline_prior_variances, 1e-12)
        prior_precision_device = jnp.asarray(prior_precision, dtype=jnp.float32)
        right_hand_side = _compute_right_hand_side(
            genotype_operator=genotype_operator,
            sample_weights=sample_weights,
            pseudo_response=pseudo_response,
            covariate_matrix=covariate_matrix_device,
            covariate_coefficients=covariate_coefficients,
        )

        coefficient_mean = _solve_global_posterior_mean(
            genotype_operator=genotype_operator,
            block_decomposition=block_decomposition,
            sample_weights=sample_weights,
            prior_precision=prior_precision_device,
            right_hand_side=right_hand_side,
            initial_mean=coefficient_mean,
            config=config,
        )

        updated_genetic_prediction = matvec(genotype_operator, coefficient_mean)
        covariate_coefficients = _solve_covariates(
            covariate_matrix=covariate_matrix_device,
            pseudo_response=pseudo_response,
            current_genetic_prediction=updated_genetic_prediction,
            sample_weights=sample_weights,
        )

        coefficient_variance = _estimate_block_variance(
            block_decomposition=block_decomposition,
            prior_precision=prior_precision_device,
            sample_weights=sample_weights,
        )

        coefficient_mean_host = np.asarray(coefficient_mean, dtype=np.float32)
        covariate_coefficients_host = np.asarray(covariate_coefficients, dtype=np.float32)
        updated_linear_predictor_host = np.asarray(
            updated_genetic_prediction + covariate_matrix_device @ covariate_coefficients,
            dtype=np.float32,
        )
        coefficient_second_moment = (
            coefficient_mean_host.astype(np.float64) ** 2
            + coefficient_variance.astype(np.float64)
        )

        local_scale, auxiliary_delta, expected_inverse_local_scale = _update_local_scales(
            coefficient_second_moment=coefficient_second_moment,
            baseline_prior_variances=baseline_prior_variances,
            local_shape_a=local_shape_a,
            local_shape_b=local_shape_b,
            auxiliary_delta=auxiliary_delta,
            config=config,
        )

        if config.update_hyperparameters:
            global_scale = _update_global_scale(
                coefficient_second_moment=coefficient_second_moment,
                metadata_baseline_scales=metadata_baseline_scales,
                local_scale=local_scale,
                config=config,
            )
            scale_model_coefficients = _update_scale_model(
                design_matrix=prior_design.design_matrix,
                feature_names=prior_design.feature_names,
                coefficient_second_moment=coefficient_second_moment,
                global_scale=global_scale,
                local_scale=local_scale,
                current_coefficients=scale_model_coefficients,
                config=config,
            )
            metadata_baseline_scales = _metadata_baseline_scales_from_coefficients(
                scale_model_coefficients,
                prior_design.design_matrix,
                config,
            )
            global_scale = _update_global_scale(
                coefficient_second_moment=coefficient_second_moment,
                metadata_baseline_scales=metadata_baseline_scales,
                local_scale=local_scale,
                config=config,
            )

        baseline_prior_variances = (global_scale * metadata_baseline_scales) ** 2

        objective_history.append(
            _surrogate_objective(
                trait_type=config.trait_type,
                targets=target_vector,
                linear_predictor=updated_linear_predictor_host,
                coefficient_variance=coefficient_variance.astype(np.float64),
                coefficient_second_moment=coefficient_second_moment,
                baseline_prior_variances=baseline_prior_variances,
                local_scale=local_scale,
                auxiliary_delta=auxiliary_delta,
                local_shape_a=local_shape_a,
                local_shape_b=local_shape_b,
                residual_variance=residual_variance,
            )
        )

        if validation_payload is not None:
            validation_genotypes, validation_covariates, validation_targets = validation_payload
            validation_linear_predictor = (
                validation_genotypes @ coefficient_mean_host
                + validation_covariates @ covariate_coefficients_host
            )
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
        alpha=np.asarray(covariate_coefficients, dtype=np.float32),
        beta_reduced=np.asarray(coefficient_mean, dtype=np.float32),
        beta_variance=coefficient_variance.astype(np.float32),
        prior_scales=baseline_prior_variances.astype(np.float32),
        global_scale=float(global_scale),
        class_tpb_shape_a={
            prior_design.inverse_class_lookup[class_index]: float(tpb_shape_a_vector[class_index])
            for class_index in range(tpb_shape_a_vector.shape[0])
        },
        class_tpb_shape_b={
            prior_design.inverse_class_lookup[class_index]: float(tpb_shape_b_vector[class_index])
            for class_index in range(tpb_shape_b_vector.shape[0])
        },
        scale_model_coefficients=scale_model_coefficients.astype(np.float32),
        scale_model_feature_names=list(prior_design.feature_names),
        sigma_error2=float(residual_variance),
        objective_history=objective_history,
        validation_history=validation_history,
    )


def compute_export_baseline_variances(
    records: Sequence[VariantRecord],
    scale_model_coefficients: np.ndarray,
    global_scale: float,
    config: ModelConfig,
) -> np.ndarray:
    prior_design = _build_prior_design(records)
    metadata_baseline_scales = _metadata_baseline_scales_from_coefficients(
        np.asarray(scale_model_coefficients, dtype=np.float64),
        prior_design.design_matrix,
        config,
    )
    return ((float(global_scale) * metadata_baseline_scales) ** 2).astype(np.float32)


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
    class_lookup = {
        variant_class: class_index
        for class_index, variant_class in enumerate(unique_classes)
    }
    inverse_class_lookup = {
        class_index: variant_class
        for variant_class, class_index in class_lookup.items()
    }

    class_membership_matrix = np.zeros((len(records), len(unique_classes)), dtype=np.float64)
    for record_index, record in enumerate(records):
        prior_classes = _prior_class_members(record)
        prior_membership = _prior_class_membership(record)
        for prior_class, prior_weight in zip(prior_classes, prior_membership, strict=True):
            class_membership_matrix[record_index, class_lookup[prior_class]] = prior_weight

    log_length = np.log(
        np.maximum(np.asarray([record.length for record in records], dtype=np.float64), 1.0)
    )
    allele_frequency = np.clip(
        np.asarray([record.allele_frequency for record in records], dtype=np.float64),
        1e-6,
        0.5,
    )
    quality = np.clip(
        np.asarray([record.quality for record in records], dtype=np.float64),
        1e-6,
        1.0,
    )
    repeat_indicator = np.asarray(
        [float(record.is_repeat) for record in records],
        dtype=np.float64,
    )
    copy_number_indicator = np.asarray(
        [float(record.is_copy_number) for record in records],
        dtype=np.float64,
    )

    standardized_log_length = _standardize_metadata(log_length)
    standardized_allele_frequency = _standardize_metadata(allele_frequency)
    standardized_quality = _standardize_metadata(quality)

    design_columns: list[np.ndarray] = [np.ones(len(records), dtype=np.float64)]
    feature_names = ["intercept"]

    for class_index, variant_class in enumerate(unique_classes):
        class_membership = class_membership_matrix[:, class_index]
        design_columns.append(class_membership)
        feature_names.append("type_offset::" + variant_class.value)

        design_columns.append(class_membership * standardized_log_length)
        feature_names.append("log_length_linear::" + variant_class.value)
        design_columns.append(class_membership * standardized_log_length * standardized_log_length)
        feature_names.append("log_length_quadratic::" + variant_class.value)

        design_columns.append(class_membership * standardized_allele_frequency)
        feature_names.append("allele_frequency_linear::" + variant_class.value)
        design_columns.append(
            class_membership
            * standardized_allele_frequency
            * standardized_allele_frequency
        )
        feature_names.append("allele_frequency_quadratic::" + variant_class.value)

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


def _initialize_scale_model_coefficients(
    prior_design: PriorDesign,
    config: ModelConfig,
) -> np.ndarray:
    default_log_scales = config.class_log_baseline_scales()
    class_baselines = np.asarray(
        [
            default_log_scales[prior_design.inverse_class_lookup[class_index]]
            for class_index in range(len(prior_design.inverse_class_lookup))
        ],
        dtype=np.float64,
    )
    average_baseline = float(np.mean(class_baselines))
    initialized_coefficients = np.zeros(prior_design.design_matrix.shape[1], dtype=np.float64)
    initialized_coefficients[0] = average_baseline

    for feature_index, feature_name in enumerate(prior_design.feature_names):
        if not feature_name.startswith("type_offset::"):
            continue
        variant_class = VariantClass(feature_name.split("::", maxsplit=1)[1])
        initialized_coefficients[feature_index] = (
            default_log_scales[variant_class] - average_baseline
        )
    return initialized_coefficients


def _initialize_tpb_shape_a_vector(
    prior_design: PriorDesign,
    config: ModelConfig,
) -> np.ndarray:
    default_shape_a = config.class_tpb_shape_a()
    return np.asarray(
        [
            default_shape_a[prior_design.inverse_class_lookup[class_index]]
            for class_index in range(len(prior_design.inverse_class_lookup))
        ],
        dtype=np.float64,
    )


def _initialize_tpb_shape_b_vector(
    prior_design: PriorDesign,
    config: ModelConfig,
) -> np.ndarray:
    default_shape_b = config.class_tpb_shape_b()
    return np.asarray(
        [
            default_shape_b[prior_design.inverse_class_lookup[class_index]]
            for class_index in range(len(prior_design.inverse_class_lookup))
        ],
        dtype=np.float64,
    )


def _metadata_baseline_scales_from_coefficients(
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
    targets: jnp.ndarray,
    linear_predictor: jnp.ndarray,
    sigma_error_floor: float,
    minimum_polya_gamma_weight: float,
) -> tuple[jnp.ndarray, jnp.ndarray, float]:
    if trait_type == TraitType.BINARY:
        sample_weights = _polya_gamma_expectation(
            linear_predictor,
            minimum_polya_gamma_weight,
        )
        pseudo_response = linear_predictor + (targets - 0.5) / sample_weights
        return sample_weights, pseudo_response, 1.0

    residual_vector = targets - linear_predictor
    residual_variance = float(jnp.mean(residual_vector * residual_vector) + sigma_error_floor)
    sample_weights = jnp.full(targets.shape[0], 1.0 / residual_variance, dtype=jnp.float32)
    return sample_weights, targets, residual_variance


@jax.jit
def _polya_gamma_expectation(
    linear_predictor: jnp.ndarray,
    minimum_polya_gamma_weight: float,
) -> jnp.ndarray:
    absolute_linear_predictor = jnp.abs(linear_predictor)
    safe_linear_predictor = jnp.where(
        absolute_linear_predictor < 1e-6,
        1.0,
        absolute_linear_predictor,
    )
    expected_weight = 0.5 * jnp.tanh(safe_linear_predictor / 2.0) / safe_linear_predictor
    expected_weight = jnp.where(absolute_linear_predictor < 1e-6, 0.25, expected_weight)
    return jnp.maximum(expected_weight, minimum_polya_gamma_weight)


@jax.jit
def _solve_covariates(
    covariate_matrix: jnp.ndarray,
    pseudo_response: jnp.ndarray,
    current_genetic_prediction: jnp.ndarray,
    sample_weights: jnp.ndarray,
) -> jnp.ndarray:
    response_vector = pseudo_response - current_genetic_prediction
    weighted_covariates = jnp.transpose(covariate_matrix) * sample_weights[None, :]
    normal_matrix = weighted_covariates @ covariate_matrix
    right_hand_side = weighted_covariates @ response_vector
    ridge_jitter = 1e-6 * jnp.eye(covariate_matrix.shape[1], dtype=jnp.float32)
    return jnp.linalg.solve(normal_matrix + ridge_jitter, right_hand_side)


@jax.jit
def _compute_right_hand_side(
    genotype_operator: GenotypeOperator,
    sample_weights: jnp.ndarray,
    pseudo_response: jnp.ndarray,
    covariate_matrix: jnp.ndarray,
    covariate_coefficients: jnp.ndarray,
) -> jnp.ndarray:
    response_vector = pseudo_response - covariate_matrix @ covariate_coefficients
    return weighted_rmatvec(genotype_operator, sample_weights, response_vector)


@jax.jit
def _apply_global_precision(
    genotype_operator: GenotypeOperator,
    sample_weights: jnp.ndarray,
    prior_precision: jnp.ndarray,
    vector: jnp.ndarray,
) -> jnp.ndarray:
    return apply_precision_matrix(
        genotype_operator,
        sample_weights,
        prior_precision,
        vector,
    )


def _solve_global_posterior_mean(
    genotype_operator: GenotypeOperator,
    block_decomposition: BlockDecomposition,
    sample_weights: jnp.ndarray,
    prior_precision: jnp.ndarray,
    right_hand_side: jnp.ndarray,
    initial_mean: jnp.ndarray,
    config: ModelConfig,
) -> jnp.ndarray:
    coefficient_mean = jnp.asarray(initial_mean, dtype=jnp.float32)
    residual_vector = right_hand_side - _apply_global_precision(
        genotype_operator=genotype_operator,
        sample_weights=sample_weights,
        prior_precision=prior_precision,
        vector=coefficient_mean,
    )
    sample_weight_sum = float(jnp.sum(sample_weights))
    preconditioned_residual = apply_block_preconditioner(
        block_decomposition=block_decomposition,
        vector=residual_vector,
        prior_precision=prior_precision,
        sample_weight_sum=sample_weight_sum,
    )
    search_direction = preconditioned_residual
    previous_inner_product = float(jnp.vdot(residual_vector, preconditioned_residual))

    iteration_index = 0
    while iteration_index < config.max_pcg_iterations:
        projected_direction = _apply_global_precision(
            genotype_operator=genotype_operator,
            sample_weights=sample_weights,
            prior_precision=prior_precision,
            vector=search_direction,
        )
        step_size = previous_inner_product / max(
            float(jnp.vdot(search_direction, projected_direction)),
            1e-12,
        )
        coefficient_mean = coefficient_mean + step_size * search_direction
        residual_vector = residual_vector - step_size * projected_direction
        residual_norm = float(jnp.linalg.norm(residual_vector))
        if residual_norm <= config.pcg_tolerance:
            break

        preconditioned_residual = apply_block_preconditioner(
            block_decomposition=block_decomposition,
            vector=residual_vector,
            prior_precision=prior_precision,
            sample_weight_sum=sample_weight_sum,
        )
        current_inner_product = float(jnp.vdot(residual_vector, preconditioned_residual))
        conjugate_weight = current_inner_product / max(previous_inner_product, 1e-12)
        search_direction = preconditioned_residual + conjugate_weight * search_direction
        previous_inner_product = current_inner_product
        iteration_index += 1

    return coefficient_mean


def _estimate_block_variance(
    block_decomposition: BlockDecomposition,
    prior_precision: jnp.ndarray,
    sample_weights: jnp.ndarray,
) -> np.ndarray:
    return np.asarray(
        estimate_variance_from_blocks(
            block_decomposition=block_decomposition,
            prior_precision=prior_precision,
            sample_weight_sum=float(jnp.sum(sample_weights)),
        ),
        dtype=np.float32,
    )


def _update_local_scales(
    coefficient_second_moment: np.ndarray,
    baseline_prior_variances: np.ndarray,
    local_shape_a: np.ndarray,
    local_shape_b: np.ndarray,
    auxiliary_delta: np.ndarray,
    config: ModelConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalized_second_moment = coefficient_second_moment / np.maximum(
        baseline_prior_variances,
        1e-12,
    )
    shape_offset = local_shape_a - 1.5
    discriminant = np.maximum(
        shape_offset * shape_offset + 2.0 * auxiliary_delta * normalized_second_moment,
        1e-12,
    )
    updated_local_scale = (
        shape_offset + np.sqrt(discriminant)
    ) / np.maximum(2.0 * auxiliary_delta, 1e-12)
    updated_local_scale = np.maximum(updated_local_scale, config.local_scale_floor)
    updated_auxiliary_delta = (local_shape_a + local_shape_b) / np.maximum(
        1.0 + updated_local_scale,
        config.local_scale_floor,
    )
    expected_inverse_local_scale = 1.0 / np.maximum(
        updated_local_scale,
        config.local_scale_floor,
    )
    return updated_local_scale, updated_auxiliary_delta, expected_inverse_local_scale


def _update_global_scale(
    coefficient_second_moment: np.ndarray,
    metadata_baseline_scales: np.ndarray,
    local_scale: np.ndarray,
    config: ModelConfig,
) -> float:
    denominator = np.maximum(
        metadata_baseline_scales * metadata_baseline_scales * local_scale,
        config.local_scale_floor,
    )
    updated_global_scale = float(np.sqrt(np.mean(coefficient_second_moment / denominator)))
    return float(np.clip(
        updated_global_scale,
        config.global_scale_floor,
        config.global_scale_ceiling,
    ))


def _update_scale_model(
    design_matrix: np.ndarray,
    feature_names: Sequence[str],
    coefficient_second_moment: np.ndarray,
    global_scale: float,
    local_scale: np.ndarray,
    current_coefficients: np.ndarray,
    config: ModelConfig,
) -> np.ndarray:
    penalty_diagonal = _scale_model_penalty(feature_names, config)
    updated_coefficients = jnp.asarray(current_coefficients, dtype=jnp.float32)
    design_matrix_device = jnp.asarray(design_matrix, dtype=jnp.float32)
    target_scale_ratio = jnp.asarray(
        coefficient_second_moment / np.maximum(global_scale * global_scale * local_scale, 1e-12),
        dtype=jnp.float32,
    )
    penalty_diagonal_device = jnp.asarray(penalty_diagonal, dtype=jnp.float32)

    iteration_index = 0
    while iteration_index < config.maximum_scale_model_iterations:
        linear_prediction = design_matrix_device @ updated_coefficients
        bounded_linear_prediction = jnp.clip(
            linear_prediction,
            jnp.log(config.prior_scale_floor),
            jnp.log(config.prior_scale_ceiling),
        )
        working_weight = jnp.maximum(
            target_scale_ratio * jnp.exp(-2.0 * bounded_linear_prediction),
            1e-8,
        )
        gradient_vector = (
            design_matrix_device.T @ (-1.0 + working_weight)
            - penalty_diagonal_device * updated_coefficients
        )
        weighted_design = design_matrix_device * (2.0 * working_weight[:, None])
        hessian_matrix = -(design_matrix_device.T @ weighted_design) - jnp.diag(penalty_diagonal_device)
        coefficient_step = jnp.linalg.solve(hessian_matrix, gradient_vector)
        updated_coefficients = updated_coefficients - coefficient_step
        if float(jnp.max(jnp.abs(coefficient_step))) < 1e-5:
            break
        iteration_index += 1

    return np.asarray(updated_coefficients, dtype=np.float64)


def _scale_model_penalty(
    feature_names: Sequence[str],
    config: ModelConfig,
) -> np.ndarray:
    penalty_values = np.full(
        len(feature_names),
        config.scale_model_ridge_penalty,
        dtype=np.float64,
    )
    penalty_values[0] = 1e-6
    for feature_index, feature_name in enumerate(feature_names):
        if feature_name.startswith("type_offset::"):
            penalty_values[feature_index] = config.type_offset_penalty
    return penalty_values


def _surrogate_objective(
    trait_type: TraitType,
    targets: np.ndarray,
    linear_predictor: np.ndarray,
    coefficient_variance: np.ndarray,
    coefficient_second_moment: np.ndarray,
    baseline_prior_variances: np.ndarray,
    local_scale: np.ndarray,
    auxiliary_delta: np.ndarray,
    local_shape_a: np.ndarray,
    local_shape_b: np.ndarray,
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

    total_prior_variances = np.maximum(
        baseline_prior_variances * local_scale,
        1e-12,
    )
    beta_prior_term = float(
        np.sum(
            -0.5 * np.log(2.0 * np.pi * total_prior_variances)
            - 0.5 * coefficient_second_moment / total_prior_variances
        )
    )
    local_scale_prior_term = float(
        np.sum(
            local_shape_a * np.log(np.maximum(auxiliary_delta, 1e-12))
            - np.asarray(gammaln(jnp.asarray(local_shape_a, dtype=jnp.float32)), dtype=np.float64)
            + (local_shape_a - 1.0) * np.log(np.maximum(local_scale, 1e-12))
            - auxiliary_delta * local_scale
        )
    )
    auxiliary_prior_term = float(
        np.sum(
            -np.asarray(gammaln(jnp.asarray(local_shape_b, dtype=jnp.float32)), dtype=np.float64)
            + (local_shape_b - 1.0) * np.log(np.maximum(auxiliary_delta, 1e-12))
            - auxiliary_delta
        )
    )
    beta_entropy = float(
        0.5 * np.sum(np.log(2.0 * np.pi * np.e * np.maximum(coefficient_variance, 1e-12)))
    )
    return log_likelihood + beta_prior_term + local_scale_prior_term + auxiliary_prior_term + beta_entropy


def _validation_metric(
    trait_type: TraitType,
    targets: np.ndarray,
    linear_predictor: np.ndarray,
) -> float:
    if trait_type == TraitType.BINARY:
        positive_probability = _sigmoid(linear_predictor)
        return float(
            -np.mean(
                targets * np.log(positive_probability + 1e-8)
                + (1.0 - targets) * np.log(1.0 - positive_probability + 1e-8)
            )
        )

    residual_vector = targets - linear_predictor
    return float(np.mean(residual_vector * residual_vector))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped_values = np.asarray(np.clip(values, -80.0, 80.0), dtype=np.float64)
    positive_mask = clipped_values >= 0.0
    negative_mask = ~positive_mask
    output = np.empty_like(clipped_values, dtype=np.float64)
    output[positive_mask] = 1.0 / (1.0 + np.exp(-clipped_values[positive_mask]))
    exp_values = np.exp(clipped_values[negative_mask])
    output[negative_mask] = exp_values / (1.0 + exp_values)
    return output.astype(np.float64)

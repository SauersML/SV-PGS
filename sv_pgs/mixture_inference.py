"""Continuous metadata-conditioned shrinkage with a global PCG mean solve."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.special import digamma, gammaln, polygamma

from sv_pgs.blocks import BlockDecomposition, build_block_decomposition, compute_block_posterior
from sv_pgs.config import ModelConfig, TraitType, VariantClass
from sv_pgs.data import VariantRecord


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
    class_tail_shapes: dict[VariantClass, float]
    scale_model_coefficients: np.ndarray
    scale_model_feature_names: list[str]
    sigma_error2: float
    objective_history: list[float]
    validation_history: list[float]


@dataclass(slots=True)
class BlockPreconditioner:
    variant_indices: np.ndarray
    cholesky_factor: np.ndarray


def fit_variational_em(
    genotypes: np.ndarray,
    covariates: np.ndarray,
    targets: np.ndarray,
    records: Sequence[VariantRecord],
    config: ModelConfig,
    validation_data: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> VariationalFitResult:
    genotype_matrix = np.asarray(genotypes, dtype=np.float32)
    genotype_matrix_float64 = np.asarray(genotype_matrix, dtype=np.float64)
    covariate_matrix = np.asarray(covariates, dtype=np.float32)
    target_vector = np.asarray(targets, dtype=np.float32)
    block_decomposition = build_block_decomposition(genotype_matrix_float64, records, config)
    prior_design = _build_prior_design(records)
    validation_payload = _prepare_validation(validation_data)

    scale_model_coefficients = _initialize_scale_model_coefficients(prior_design, config)
    prior_scales = _prior_scales_from_coefficients(scale_model_coefficients, prior_design.design_matrix, config)
    class_tail_shape_vector = _initialize_class_tail_shape_vector(prior_design, config)
    local_tail_shape = prior_design.class_membership_matrix @ class_tail_shape_vector

    coefficient_mean = np.zeros(genotype_matrix.shape[1], dtype=np.float32)
    coefficient_variance = prior_scales.astype(np.float32)
    expected_local_precision = np.ones(genotype_matrix.shape[1], dtype=np.float64)
    expected_log_local_precision = np.zeros(genotype_matrix.shape[1], dtype=np.float64)
    covariate_coefficients = np.zeros(covariate_matrix.shape[1], dtype=np.float32)
    residual_variance = 1.0

    objective_history: list[float] = []
    validation_history: list[float] = []

    outer_iteration = 0
    while outer_iteration < config.max_outer_iterations:
        current_genetic_prediction = genotype_matrix @ coefficient_mean
        linear_predictor = current_genetic_prediction + covariate_matrix @ covariate_coefficients
        sample_weights, pseudo_response, residual_variance = _likelihood_update(
            trait_type=config.trait_type,
            targets=target_vector,
            linear_predictor=linear_predictor,
            sigma_error_floor=config.sigma_error_floor,
            minimum_polya_gamma_weight=config.polya_gamma_minimum_weight,
        )
        covariate_coefficients = _solve_covariates(
            covariate_matrix=covariate_matrix,
            pseudo_response=pseudo_response,
            current_genetic_prediction=current_genetic_prediction,
            sample_weights=sample_weights,
        )

        prior_precision = expected_local_precision / np.maximum(prior_scales, 1e-12)
        right_hand_side = _compute_right_hand_side(
            genotype_matrix=genotype_matrix_float64,
            sample_weights=sample_weights,
            pseudo_response=pseudo_response,
            covariate_matrix=covariate_matrix,
            covariate_coefficients=covariate_coefficients,
        )
        block_preconditioner = _build_block_preconditioner(
            block_decomposition=block_decomposition,
            genotype_matrix=genotype_matrix_float64,
            sample_weights=sample_weights.astype(np.float64),
            prior_precision=prior_precision.astype(np.float64),
        )
        coefficient_mean = _solve_global_posterior_mean(
            genotype_matrix=genotype_matrix_float64,
            sample_weights=sample_weights,
            prior_precision=prior_precision.astype(np.float32),
            right_hand_side=right_hand_side,
            block_preconditioner=block_preconditioner,
            initial_mean=coefficient_mean,
            config=config,
        )

        updated_genetic_prediction = genotype_matrix @ coefficient_mean
        covariate_coefficients = _solve_covariates(
            covariate_matrix=covariate_matrix,
            pseudo_response=pseudo_response,
            current_genetic_prediction=updated_genetic_prediction,
            sample_weights=sample_weights,
        )

        coefficient_variance = _estimate_block_variance(
            genotype_matrix=genotype_matrix_float64,
            block_decomposition=block_decomposition,
            sample_weights=sample_weights.astype(np.float64),
            pseudo_response=pseudo_response.astype(np.float64),
            covariate_matrix=covariate_matrix.astype(np.float64),
            covariate_coefficients=covariate_coefficients.astype(np.float64),
            current_genetic_prediction=updated_genetic_prediction.astype(np.float64),
            coefficient_mean=coefficient_mean.astype(np.float64),
            prior_precision=prior_precision.astype(np.float64),
        )
        coefficient_second_moment = coefficient_mean.astype(np.float64) ** 2 + coefficient_variance.astype(np.float64)

        posterior_precision_shape = local_tail_shape + 0.5
        posterior_precision_rate = local_tail_shape + 0.5 * coefficient_second_moment / np.maximum(prior_scales, 1e-12)
        expected_local_precision = posterior_precision_shape / np.maximum(posterior_precision_rate, 1e-12)
        expected_log_local_precision = digamma(posterior_precision_shape) - np.log(np.maximum(posterior_precision_rate, 1e-12))

        if config.update_hyperparameters:
            class_tail_shape_vector = _update_class_tail_shapes(
                prior_design=prior_design,
                expected_local_precision=expected_local_precision,
                expected_log_local_precision=expected_log_local_precision,
                current_class_tail_shape_vector=class_tail_shape_vector,
                config=config,
            )
            local_tail_shape = prior_design.class_membership_matrix @ class_tail_shape_vector
            scale_model_coefficients = _update_scale_model(
                design_matrix=prior_design.design_matrix,
                feature_names=prior_design.feature_names,
                coefficient_second_moment=coefficient_second_moment,
                expected_local_precision=expected_local_precision,
                current_coefficients=scale_model_coefficients,
                config=config,
            )
            prior_scales = _prior_scales_from_coefficients(scale_model_coefficients, prior_design.design_matrix, config)

        updated_linear_predictor = updated_genetic_prediction + covariate_matrix @ covariate_coefficients
        objective_history.append(
            _surrogate_objective(
                trait_type=config.trait_type,
                targets=target_vector,
                linear_predictor=updated_linear_predictor,
                coefficient_variance=coefficient_variance.astype(np.float64),
                coefficient_second_moment=coefficient_second_moment,
                prior_scales=prior_scales,
                local_tail_shape=local_tail_shape,
                expected_local_precision=expected_local_precision,
                expected_log_local_precision=expected_log_local_precision,
                posterior_precision_shape=posterior_precision_shape,
                posterior_precision_rate=posterior_precision_rate,
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
        class_tail_shapes={
            prior_design.inverse_class_lookup[class_index]: float(class_tail_shape_vector[class_index])
            for class_index in range(class_tail_shape_vector.shape[0])
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
        class_membership = class_membership_matrix[:, class_index]
        design_columns.append(class_membership)
        feature_names.append("type_offset::" + variant_class.value)
        design_columns.append(class_membership * standardized_log_length)
        feature_names.append("log_length_linear::" + variant_class.value)
        design_columns.append(class_membership * standardized_log_length * standardized_log_length)
        feature_names.append("log_length_quadratic::" + variant_class.value)
        design_columns.append(class_membership * standardized_allele_frequency)
        feature_names.append("allele_frequency_linear::" + variant_class.value)
        design_columns.append(class_membership * standardized_allele_frequency * standardized_allele_frequency)
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


def _initialize_scale_model_coefficients(prior_design: PriorDesign, config: ModelConfig) -> np.ndarray:
    default_log_scales = config.class_log_prior_scales()
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
        initialized_coefficients[feature_index] = default_log_scales[variant_class] - average_baseline
    return initialized_coefficients


def _initialize_class_tail_shape_vector(prior_design: PriorDesign, config: ModelConfig) -> np.ndarray:
    default_tail_shapes = config.class_tail_shapes()
    return np.asarray(
        [
            default_tail_shapes[prior_design.inverse_class_lookup[class_index]]
            for class_index in range(len(prior_design.inverse_class_lookup))
        ],
        dtype=np.float64,
    )


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
        pseudo_response = linear_predictor + (targets - 0.5) / sample_weights
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


def _compute_right_hand_side(
    genotype_matrix: np.ndarray,
    sample_weights: np.ndarray,
    pseudo_response: np.ndarray,
    covariate_matrix: np.ndarray,
    covariate_coefficients: np.ndarray,
) -> np.ndarray:
    weighted_response = sample_weights * (pseudo_response - covariate_matrix @ covariate_coefficients)
    return genotype_matrix.T @ weighted_response


def _build_block_preconditioner(
    block_decomposition: BlockDecomposition,
    genotype_matrix: np.ndarray,
    sample_weights: np.ndarray,
    prior_precision: np.ndarray,
) -> list[BlockPreconditioner]:
    block_preconditioner: list[BlockPreconditioner] = []
    for block in block_decomposition.blocks:
        block_indices = block.variant_indices
        block_genotypes = genotype_matrix[:, block_indices]
        local_precision_matrix = block_genotypes.T @ (sample_weights[:, None] * block_genotypes)
        local_precision_matrix += np.diag(prior_precision[block_indices])
        local_precision_matrix += np.eye(block_indices.shape[0], dtype=np.float64) * block.jitter
        cholesky_factor = np.linalg.cholesky(local_precision_matrix)
        block_preconditioner.append(
            BlockPreconditioner(
                variant_indices=block_indices.astype(np.int32),
                cholesky_factor=cholesky_factor.astype(np.float64),
            )
        )
    return block_preconditioner


def _apply_preconditioner(
    block_preconditioner: Sequence[BlockPreconditioner],
    vector: np.ndarray,
) -> np.ndarray:
    preconditioned_vector = np.zeros_like(vector, dtype=np.float64)
    for block in block_preconditioner:
        block_vector = vector[block.variant_indices]
        block_solution = np.linalg.solve(
            block.cholesky_factor.T,
            np.linalg.solve(block.cholesky_factor, block_vector),
        )
        preconditioned_vector[block.variant_indices] = block_solution
    return preconditioned_vector


def _apply_global_precision(
    genotype_matrix: np.ndarray,
    sample_weights: np.ndarray,
    prior_precision: np.ndarray,
    vector: np.ndarray,
) -> np.ndarray:
    sample_projection = genotype_matrix @ vector
    weighted_projection = sample_weights * sample_projection
    transpose_projection = genotype_matrix.T @ weighted_projection
    return transpose_projection + prior_precision.astype(np.float64) * vector.astype(np.float64)


def _solve_global_posterior_mean(
    genotype_matrix: np.ndarray,
    sample_weights: np.ndarray,
    prior_precision: np.ndarray,
    right_hand_side: np.ndarray,
    block_preconditioner: Sequence[BlockPreconditioner],
    initial_mean: np.ndarray,
    config: ModelConfig,
) -> np.ndarray:
    coefficient_mean = initial_mean.astype(np.float64).copy()
    residual_vector = right_hand_side.astype(np.float64) - _apply_global_precision(
        genotype_matrix=genotype_matrix,
        sample_weights=sample_weights,
        prior_precision=prior_precision,
        vector=coefficient_mean,
    )
    preconditioned_residual = _apply_preconditioner(block_preconditioner, residual_vector)
    search_direction = preconditioned_residual.copy()
    previous_inner_product = float(residual_vector @ preconditioned_residual)

    iteration_index = 0
    while iteration_index < config.max_pcg_iterations:
        projected_direction = _apply_global_precision(
            genotype_matrix=genotype_matrix,
            sample_weights=sample_weights,
            prior_precision=prior_precision,
            vector=search_direction,
        )
        step_size = previous_inner_product / max(float(search_direction @ projected_direction), 1e-12)
        coefficient_mean += step_size * search_direction
        residual_vector -= step_size * projected_direction
        residual_norm = float(np.linalg.norm(residual_vector))
        if residual_norm <= config.pcg_tolerance:
            break

        preconditioned_residual = _apply_preconditioner(block_preconditioner, residual_vector)
        current_inner_product = float(residual_vector @ preconditioned_residual)
        conjugate_weight = current_inner_product / max(previous_inner_product, 1e-12)
        search_direction = preconditioned_residual + conjugate_weight * search_direction
        previous_inner_product = current_inner_product
        iteration_index += 1

    return coefficient_mean.astype(np.float32)


def _estimate_block_variance(
    genotype_matrix: np.ndarray,
    block_decomposition: BlockDecomposition,
    sample_weights: np.ndarray,
    pseudo_response: np.ndarray,
    covariate_matrix: np.ndarray,
    covariate_coefficients: np.ndarray,
    current_genetic_prediction: np.ndarray,
    coefficient_mean: np.ndarray,
    prior_precision: np.ndarray,
) -> np.ndarray:
    weighted_residual = sample_weights * (
        pseudo_response - covariate_matrix @ covariate_coefficients - current_genetic_prediction
    )
    coefficient_variance = np.zeros(coefficient_mean.shape[0], dtype=np.float64)

    for block in block_decomposition.blocks:
        block_indices = block.variant_indices
        block_genotypes = genotype_matrix[:, block_indices]
        block_residual = weighted_residual + sample_weights * (block_genotypes @ coefficient_mean[block_indices])
        _, block_variance = compute_block_posterior(
            block,
            genotype_matrix,
            sample_weights,
            block_residual,
            prior_precision,
        )
        coefficient_variance[block_indices] = block_variance.astype(np.float64)

    return np.maximum(coefficient_variance, 1e-8).astype(np.float32)


def _update_class_tail_shapes(
    prior_design: PriorDesign,
    expected_local_precision: np.ndarray,
    expected_log_local_precision: np.ndarray,
    current_class_tail_shape_vector: np.ndarray,
    config: ModelConfig,
) -> np.ndarray:
    updated_class_tail_shape_vector = current_class_tail_shape_vector.copy()
    for class_index in range(updated_class_tail_shape_vector.shape[0]):
        class_weights = prior_design.class_membership_matrix[:, class_index]
        effective_count = float(np.sum(class_weights))
        if effective_count <= 1e-8:
            continue
        weighted_expected_precision = float(np.sum(class_weights * expected_local_precision))
        weighted_expected_log_precision = float(np.sum(class_weights * expected_log_local_precision))
        tail_shape_value = updated_class_tail_shape_vector[class_index]
        iteration_index = 0
        while iteration_index < config.maximum_tail_shape_iterations:
            gradient_value = (
                effective_count * (np.log(tail_shape_value) + 1.0 - digamma(tail_shape_value))
                + weighted_expected_log_precision
                - weighted_expected_precision
            )
            hessian_value = effective_count * ((1.0 / tail_shape_value) - polygamma(1, tail_shape_value))
            if abs(hessian_value) < 1e-8:
                break
            tail_shape_step = gradient_value / hessian_value
            tail_shape_value = np.clip(
                tail_shape_value - tail_shape_step,
                config.minimum_tail_shape,
                config.maximum_tail_shape,
            )
            if abs(tail_shape_step) < 1e-5:
                break
            iteration_index += 1
        updated_class_tail_shape_vector[class_index] = tail_shape_value
    return updated_class_tail_shape_vector


def _update_scale_model(
    design_matrix: np.ndarray,
    feature_names: Sequence[str],
    coefficient_second_moment: np.ndarray,
    expected_local_precision: np.ndarray,
    current_coefficients: np.ndarray,
    config: ModelConfig,
) -> np.ndarray:
    penalty_diagonal = _scale_model_penalty(feature_names, config)
    updated_coefficients = current_coefficients.copy()
    sufficient_statistic = np.maximum(coefficient_second_moment * expected_local_precision, 1e-12)

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
    coefficient_variance: np.ndarray,
    coefficient_second_moment: np.ndarray,
    prior_scales: np.ndarray,
    local_tail_shape: np.ndarray,
    expected_local_precision: np.ndarray,
    expected_log_local_precision: np.ndarray,
    posterior_precision_shape: np.ndarray,
    posterior_precision_rate: np.ndarray,
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

    beta_prior_term = float(
        np.sum(
            0.5 * expected_log_local_precision
            - 0.5 * np.log(2.0 * np.pi * np.maximum(prior_scales, 1e-12))
            - 0.5 * expected_local_precision * coefficient_second_moment / np.maximum(prior_scales, 1e-12)
        )
    )
    precision_prior_term = float(
        np.sum(
            local_tail_shape * np.log(np.maximum(local_tail_shape, 1e-12))
            - gammaln(local_tail_shape)
            + (local_tail_shape - 1.0) * expected_log_local_precision
            - local_tail_shape * expected_local_precision
        )
    )
    beta_entropy = float(0.5 * np.sum(np.log(2.0 * np.pi * np.e * np.maximum(coefficient_variance, 1e-12))))
    precision_entropy = float(
        np.sum(
            posterior_precision_shape
            - np.log(np.maximum(posterior_precision_rate, 1e-12))
            + gammaln(posterior_precision_shape)
            + (1.0 - posterior_precision_shape) * digamma(posterior_precision_shape)
        )
    )
    return log_likelihood + beta_prior_term + precision_prior_term + beta_entropy + precision_entropy


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

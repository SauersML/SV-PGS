"""Collapsed evidence-centric TPB / gamma-gamma inference built around JAX."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import gammaln as scipy_gammaln
from scipy.special import kve as scipy_kve
from scipy.special import polygamma as scipy_polygamma

from sv_pgs.config import ModelConfig, TraitType, VariantClass
from sv_pgs.data import TieGroup, TieMap, VariantRecord
from sv_pgs.jax_backend import resolve_single_device
from sv_pgs.numeric import stable_sigmoid
from sv_pgs.preprocessing import collapse_tie_groups


@dataclass(slots=True)
class PriorDesign:
    design_matrix: np.ndarray
    feature_names: list[str]
    class_membership_matrix: np.ndarray
    inverse_class_lookup: dict[int, VariantClass]


@dataclass(slots=True)
class TieLayout:
    member_to_reduced: np.ndarray
    reduced_member_indices: np.ndarray
    reduced_member_mask: np.ndarray


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
    member_prior_variances: np.ndarray | None = None


@dataclass(slots=True)
class PosteriorState:
    alpha: np.ndarray
    beta: np.ndarray
    beta_variance: np.ndarray
    linear_predictor: np.ndarray
    collapsed_objective: float
    sigma_error2: float


def fit_variational_em(
    genotypes: np.ndarray,
    covariates: np.ndarray,
    targets: np.ndarray,
    records: Sequence[VariantRecord],
    config: ModelConfig,
    tie_map: TieMap | None = None,
    validation_data: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> VariationalFitResult:
    genotype_matrix = np.asarray(genotypes, dtype=np.float32)
    covariate_matrix = np.asarray(covariates, dtype=np.float32)
    target_vector = np.asarray(targets, dtype=np.float32)
    member_records = list(records)
    if tie_map is None:
        tie_map = _identity_tie_map(len(member_records))
    if genotype_matrix.shape[1] != len(tie_map.reduced_to_group):
        raise ValueError("Reduced genotype columns must match tie-map group count.")
    if len(member_records) != tie_map.original_to_reduced.shape[0]:
        raise ValueError("records must align with tie_map member space.")

    reduced_records = collapse_tie_groups(member_records, tie_map)
    if len(reduced_records) != genotype_matrix.shape[1]:
        raise ValueError("Collapsed records must align with reduced genotype matrix.")

    device = resolve_single_device(config)
    with jax.default_device(device):
        genotype_matrix_device = jax.device_put(jnp.asarray(genotype_matrix, dtype=jnp.float32), device=device)
        covariate_matrix_device = jax.device_put(jnp.asarray(covariate_matrix, dtype=jnp.float32), device=device)
        target_vector_device = jax.device_put(jnp.asarray(target_vector, dtype=jnp.float32), device=device)
        validation_payload = _prepare_validation(validation_data, device)

        prior_design = _build_prior_design(member_records)
        tie_layout = _build_tie_layout(tie_map)
        scale_penalty = _scale_model_penalty(prior_design.feature_names, config)

        global_scale, scale_model_coefficients = _initialize_scale_model(prior_design, config)
        tpb_shape_a_vector = _initialize_tpb_shape_a_vector(prior_design, config)
        tpb_shape_b_vector = _initialize_tpb_shape_b_vector(prior_design, config)
        local_shape_a = prior_design.class_membership_matrix @ tpb_shape_a_vector
        local_shape_b = prior_design.class_membership_matrix @ tpb_shape_b_vector

        local_scale = np.ones(len(member_records), dtype=np.float64)
        auxiliary_delta = np.asarray(local_shape_b, dtype=np.float64)
        sigma_error2 = 1.0
        alpha_state = np.zeros(covariate_matrix.shape[1], dtype=np.float32)
        beta_state = np.zeros(genotype_matrix.shape[1], dtype=np.float32)

        objective_history: list[float] = []
        validation_history: list[float] = []

        previous_alpha: np.ndarray | None = None
        previous_beta: np.ndarray | None = None
        previous_local_scale: np.ndarray | None = None
        previous_theta: np.ndarray | None = None

        outer_iteration = 0
        while outer_iteration < config.max_outer_iterations:
            current_theta = _pack_theta(
                global_scale=float(global_scale),
                scale_model_coefficients=np.asarray(scale_model_coefficients, dtype=np.float64),
            )
            if config.update_hyperparameters:
                optimized_theta = _optimize_theta_laml(
                    theta=current_theta,
                    genotype_matrix=genotype_matrix_device,
                    covariate_matrix=covariate_matrix_device,
                    targets=target_vector_device,
                    design_matrix=prior_design.design_matrix,
                    scale_penalty=scale_penalty,
                    local_scale=local_scale,
                    tie_layout=tie_layout,
                    sigma_error2=sigma_error2,
                    alpha_init=alpha_state,
                    beta_init=beta_state,
                    trait_type=config.trait_type,
                    config=config,
                )
                global_scale, scale_model_coefficients = _unpack_theta(optimized_theta)
            metadata_baseline_scales = _metadata_baseline_scales_from_coefficients(
                np.asarray(scale_model_coefficients, dtype=np.float64),
                prior_design.design_matrix,
                config,
            )
            baseline_member_prior_variances = (float(global_scale) * metadata_baseline_scales) ** 2
            member_prior_variances = _effective_prior_variances(
                baseline_prior_variances=baseline_member_prior_variances,
                local_scale=local_scale,
                trait_type=config.trait_type,
                config=config,
            )
            reduced_prior_variances = _reduce_member_values(member_prior_variances, tie_layout)

            posterior_state = _fit_collapsed_posterior(
                genotype_matrix=genotype_matrix_device,
                covariate_matrix=covariate_matrix_device,
                targets=target_vector_device,
                reduced_prior_variances=reduced_prior_variances,
                sigma_error2=sigma_error2,
                alpha_init=alpha_state,
                beta_init=beta_state,
                trait_type=config.trait_type,
                config=config,
            )
            alpha_state = posterior_state.alpha
            beta_state = posterior_state.beta
            sigma_error2 = posterior_state.sigma_error2

            reduced_second_moment = (
                np.asarray(beta_state, dtype=np.float64) ** 2
                + np.asarray(posterior_state.beta_variance, dtype=np.float64)
            )
            member_second_moment = _project_reduced_second_moment_to_members(
                reduced_second_moment=reduced_second_moment,
                member_prior_variances=member_prior_variances,
                tie_layout=tie_layout,
            )

            full_objective = posterior_state.collapsed_objective + _local_scale_prior_objective(
                local_scale=local_scale,
                auxiliary_delta=auxiliary_delta,
                local_shape_a=local_shape_a,
                local_shape_b=local_shape_b,
            ) + _scale_penalty_objective(
                scale_model_coefficients=np.asarray(scale_model_coefficients, dtype=np.float64),
                scale_penalty=scale_penalty,
            )
            objective_history.append(float(full_objective))

            if validation_payload is not None:
                validation_history.append(
                    _validation_metric(
                        trait_type=config.trait_type,
                        genotype_matrix=validation_payload[0],
                        covariate_matrix=validation_payload[1],
                        targets=validation_payload[2],
                        alpha=alpha_state,
                        beta=beta_state,
                    )
                )

            updated_local_scale, updated_auxiliary_delta = _update_local_scales(
                coefficient_second_moment=member_second_moment,
                baseline_prior_variances=baseline_member_prior_variances,
                local_shape_a=local_shape_a,
                local_shape_b=local_shape_b,
                auxiliary_delta=auxiliary_delta,
                config=config,
            )
            local_scale = _share_local_scale_within_tie_groups(
                local_scale=updated_local_scale,
                baseline_prior_variances=baseline_member_prior_variances,
                tie_layout=tie_layout,
            )
            auxiliary_delta = (local_shape_a + local_shape_b) / np.maximum(
                1.0 + local_scale,
                config.local_scale_floor,
            )

            parameter_change = _relative_parameter_change(
                current_beta=beta_state,
                previous_beta=previous_beta,
                current_alpha=alpha_state,
                previous_alpha=previous_alpha,
                current_local_scale=local_scale,
                previous_local_scale=previous_local_scale,
                current_theta=_pack_theta(
                    global_scale=float(global_scale),
                    scale_model_coefficients=np.asarray(scale_model_coefficients, dtype=np.float64),
                ),
                previous_theta=previous_theta,
            )
            previous_alpha = alpha_state.copy()
            previous_beta = beta_state.copy()
            previous_local_scale = local_scale.copy()
            previous_theta = _pack_theta(
                global_scale=float(global_scale),
                scale_model_coefficients=np.asarray(scale_model_coefficients, dtype=np.float64),
            )

            outer_iteration += 1
            if validation_history:
                if len(validation_history) >= 2:
                    validation_delta = abs(validation_history[-1] - validation_history[-2])
                    if parameter_change < config.convergence_tolerance and validation_delta < config.convergence_tolerance:
                        break
            elif parameter_change < config.convergence_tolerance:
                break

        final_metadata_baseline_scales = _metadata_baseline_scales_from_coefficients(
            np.asarray(scale_model_coefficients, dtype=np.float64),
            prior_design.design_matrix,
            config,
        )
        final_baseline_member_prior_variances = (float(global_scale) * final_metadata_baseline_scales) ** 2
        final_member_prior_variances = _effective_prior_variances(
            baseline_prior_variances=final_baseline_member_prior_variances,
            local_scale=local_scale,
            trait_type=config.trait_type,
            config=config,
        )
        final_reduced_prior_variances = _reduce_member_values(final_member_prior_variances, tie_layout)
        final_state = _fit_collapsed_posterior(
            genotype_matrix=genotype_matrix_device,
            covariate_matrix=covariate_matrix_device,
            targets=target_vector_device,
            reduced_prior_variances=final_reduced_prior_variances,
            sigma_error2=sigma_error2,
            alpha_init=alpha_state,
            beta_init=beta_state,
            trait_type=config.trait_type,
            config=config,
        )
        if config.trait_type == TraitType.BINARY:
            calibrated_alpha = final_state.alpha.copy()
            calibrated_alpha[0] += np.float32(
                _calibrate_binary_intercept(
                    linear_predictor=final_state.linear_predictor,
                    targets=target_vector,
                )
            )
            final_state = PosteriorState(
                alpha=calibrated_alpha,
                beta=final_state.beta,
                beta_variance=final_state.beta_variance,
                linear_predictor=genotype_matrix @ final_state.beta + covariate_matrix @ calibrated_alpha,
                collapsed_objective=final_state.collapsed_objective,
                sigma_error2=final_state.sigma_error2,
            )

        return VariationalFitResult(
            alpha=np.asarray(final_state.alpha, dtype=np.float32),
            beta_reduced=np.asarray(final_state.beta, dtype=np.float32),
            beta_variance=np.asarray(final_state.beta_variance, dtype=np.float32),
            prior_scales=final_member_prior_variances.astype(np.float32),
            global_scale=float(global_scale),
            class_tpb_shape_a={
                prior_design.inverse_class_lookup[class_index]: float(tpb_shape_a_vector[class_index])
                for class_index in range(tpb_shape_a_vector.shape[0])
            },
            class_tpb_shape_b={
                prior_design.inverse_class_lookup[class_index]: float(tpb_shape_b_vector[class_index])
                for class_index in range(tpb_shape_b_vector.shape[0])
            },
            scale_model_coefficients=np.asarray(scale_model_coefficients, dtype=np.float32),
            scale_model_feature_names=list(prior_design.feature_names),
            sigma_error2=float(final_state.sigma_error2),
            objective_history=objective_history,
            validation_history=validation_history,
            member_prior_variances=final_member_prior_variances.astype(np.float32),
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
    device: jax.Device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if validation_data is None:
        return None
    validation_genotypes, validation_covariates, validation_targets = validation_data
    return (
        np.asarray(validation_genotypes, dtype=np.float32),
        np.asarray(validation_covariates, dtype=np.float32),
        np.asarray(validation_targets, dtype=np.float32),
    )


def _fit_collapsed_posterior(
    genotype_matrix: jnp.ndarray,
    covariate_matrix: jnp.ndarray,
    targets: jnp.ndarray,
    reduced_prior_variances: np.ndarray,
    sigma_error2: float,
    alpha_init: np.ndarray,
    beta_init: np.ndarray,
    trait_type: TraitType,
    config: ModelConfig,
) -> PosteriorState:
    prior_variance_device = jnp.asarray(np.maximum(reduced_prior_variances, 1e-8), dtype=jnp.float32)
    if trait_type == TraitType.QUANTITATIVE:
        alpha, beta, beta_variance, linear_predictor, collapsed_objective, sigma_error2_new = _quantitative_posterior_state(
            genotype_matrix=genotype_matrix,
            covariate_matrix=covariate_matrix,
            targets=targets,
            prior_variances=prior_variance_device,
            sigma_error2=max(float(sigma_error2), config.sigma_error_floor),
            sigma_error_floor=config.sigma_error_floor,
        )
    else:
        alpha, beta, beta_variance, linear_predictor, collapsed_objective = _binary_posterior_state(
            genotype_matrix=genotype_matrix,
            covariate_matrix=covariate_matrix,
            targets=targets,
            prior_variances=prior_variance_device,
            alpha_init=jnp.asarray(alpha_init, dtype=jnp.float32),
            beta_init=jnp.asarray(beta_init, dtype=jnp.float32),
            minimum_weight=config.polya_gamma_minimum_weight,
            max_iterations=max(12, min(config.max_outer_iterations * 3, 64)),
        )
        sigma_error2_new = 1.0
    return PosteriorState(
        alpha=np.asarray(jax.device_get(alpha), dtype=np.float32),
        beta=np.asarray(jax.device_get(beta), dtype=np.float32),
        beta_variance=np.asarray(jax.device_get(beta_variance), dtype=np.float32),
        linear_predictor=np.asarray(jax.device_get(linear_predictor), dtype=np.float32),
        collapsed_objective=float(jax.device_get(collapsed_objective)),
        sigma_error2=float(jax.device_get(sigma_error2_new)),
    )


def _optimize_theta_laml(
    theta: np.ndarray,
    genotype_matrix: jnp.ndarray,
    covariate_matrix: jnp.ndarray,
    targets: jnp.ndarray,
    design_matrix: np.ndarray,
    scale_penalty: np.ndarray,
    local_scale: np.ndarray,
    tie_layout: TieLayout,
    sigma_error2: float,
    alpha_init: np.ndarray,
    beta_init: np.ndarray,
    trait_type: TraitType,
    config: ModelConfig,
) -> np.ndarray:
    theta_device = jnp.asarray(theta, dtype=jnp.float32)
    design_matrix_device = jnp.asarray(design_matrix, dtype=jnp.float32)
    scale_penalty_device = jnp.asarray(scale_penalty, dtype=jnp.float32)
    local_scale_device = jnp.asarray(local_scale, dtype=jnp.float32)
    reduced_member_indices = jnp.asarray(tie_layout.reduced_member_indices, dtype=jnp.int32)
    reduced_member_mask = jnp.asarray(tie_layout.reduced_member_mask, dtype=jnp.float32)
    alpha_init_device = jnp.asarray(alpha_init, dtype=jnp.float32)
    beta_init_device = jnp.asarray(beta_init, dtype=jnp.float32)

    def objective_fn(theta_value: jnp.ndarray) -> jnp.ndarray:
        global_scale_value = jnp.exp(theta_value[0])
        scale_model_coefficients = theta_value[1:]
        metadata_baseline_scales = _metadata_baseline_scales_from_coefficients_jax(
            scale_model_coefficients=scale_model_coefficients,
            design_matrix=design_matrix_device,
            config=config,
        )
        member_prior_variances = _effective_prior_variances_jax(
            baseline_prior_variances=(global_scale_value * metadata_baseline_scales) ** 2,
            local_scale=local_scale_device,
            trait_type=trait_type,
            config=config,
        )
        reduced_prior_variances = _reduce_member_values_jax(
            member_values=member_prior_variances,
            reduced_member_indices=reduced_member_indices,
            reduced_member_mask=reduced_member_mask,
        )
        if trait_type == TraitType.QUANTITATIVE:
            _alpha, _beta, _beta_variance, _linear_predictor, collapsed_objective, _sigma_error2 = _quantitative_posterior_state(
                genotype_matrix=genotype_matrix,
                covariate_matrix=covariate_matrix,
                targets=targets,
                prior_variances=reduced_prior_variances,
                sigma_error2=max(float(sigma_error2), config.sigma_error_floor),
                sigma_error_floor=config.sigma_error_floor,
            )
        else:
            _alpha, _beta, _beta_variance, _linear_predictor, collapsed_objective = _binary_posterior_state(
                genotype_matrix=genotype_matrix,
                covariate_matrix=covariate_matrix,
                targets=targets,
                prior_variances=reduced_prior_variances,
                alpha_init=alpha_init_device,
                beta_init=beta_init_device,
                minimum_weight=config.polya_gamma_minimum_weight,
                max_iterations=max(12, min(config.max_outer_iterations * 3, 64)),
            )
        return collapsed_objective - 0.5 * jnp.sum(scale_penalty_device * scale_model_coefficients * scale_model_coefficients)

    value_and_grad = jax.value_and_grad(objective_fn)
    hessian_fn = jax.hessian(objective_fn)

    best_theta = np.asarray(theta, dtype=np.float64)
    best_objective = float(jax.device_get(objective_fn(jnp.asarray(best_theta, dtype=jnp.float32))))
    damping = 1.0
    for _iteration_index in range(4):
        theta_eval = jnp.asarray(best_theta, dtype=jnp.float32)
        objective_value, gradient_value = value_and_grad(theta_eval)
        hessian_value = hessian_fn(theta_eval)
        objective_host = float(jax.device_get(objective_value))
        gradient_host = np.asarray(jax.device_get(gradient_value), dtype=np.float64)
        hessian_host = np.asarray(jax.device_get(hessian_value), dtype=np.float64)
        metric = np.diag(np.maximum(np.abs(np.diag(hessian_host)), 1e-4))
        improved = False
        for _attempt_index in range(6):
            system = -hessian_host + damping * metric + np.eye(hessian_host.shape[0], dtype=np.float64) * 1e-4
            step = np.linalg.solve(system, gradient_host)
            step = np.clip(step, -1.0, 1.0)
            candidate_theta = best_theta + step
            candidate_theta[0] = np.clip(
                candidate_theta[0],
                np.log(config.global_scale_floor),
                np.log(config.global_scale_ceiling),
            )
            candidate_objective = float(
                jax.device_get(objective_fn(jnp.asarray(candidate_theta, dtype=jnp.float32)))
            )
            if np.isfinite(candidate_objective) and candidate_objective > objective_host:
                best_theta = candidate_theta
                best_objective = candidate_objective
                damping = max(damping * 0.5, 1e-3)
                improved = True
                break
            damping = min(damping * 4.0, 1e6)
        if not improved:
            break
        if np.linalg.norm(step) / max(np.linalg.norm(best_theta), 1e-8) < config.convergence_tolerance:
            break
    if not np.isfinite(best_objective):
        return np.asarray(theta, dtype=np.float64)
    return best_theta


def _metadata_baseline_scales_from_coefficients_jax(
    scale_model_coefficients: jnp.ndarray,
    design_matrix: jnp.ndarray,
    config: ModelConfig,
) -> jnp.ndarray:
    if design_matrix.shape[1] == 0:
        return jnp.ones(design_matrix.shape[0], dtype=jnp.float32)
    linear_prediction = design_matrix @ scale_model_coefficients
    bounded_log_scales = jnp.clip(
        linear_prediction,
        jnp.log(config.prior_scale_floor),
        jnp.log(config.prior_scale_ceiling),
    )
    return jnp.exp(bounded_log_scales)


def _effective_prior_variances_jax(
    baseline_prior_variances: jnp.ndarray,
    local_scale: jnp.ndarray,
    trait_type: TraitType,
    config: ModelConfig,
) -> jnp.ndarray:
    total_prior_variances = baseline_prior_variances * jnp.maximum(local_scale, config.local_scale_floor)
    if trait_type != TraitType.BINARY or not config.enable_horseshoe_slab:
        return jnp.maximum(total_prior_variances, 1e-8)
    slab_variance = config.regularized_horseshoe_slab_scale ** 2
    regularized_variances = slab_variance * total_prior_variances / jnp.maximum(
        slab_variance + total_prior_variances,
        1e-8,
    )
    return jnp.maximum(regularized_variances, 1e-8)


@jax.jit
def _reduce_member_values_jax(
    member_values: jnp.ndarray,
    reduced_member_indices: jnp.ndarray,
    reduced_member_mask: jnp.ndarray,
) -> jnp.ndarray:
    return jnp.sum(member_values[reduced_member_indices] * reduced_member_mask, axis=1)


@partial(jax.jit, static_argnames=("sigma_error_floor",))
def _quantitative_posterior_state(
    genotype_matrix: jnp.ndarray,
    covariate_matrix: jnp.ndarray,
    targets: jnp.ndarray,
    prior_variances: jnp.ndarray,
    sigma_error2: float,
    sigma_error_floor: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    sample_count = genotype_matrix.shape[0]
    jitter = 1e-5
    covariance_matrix = sigma_error2 * jnp.eye(sample_count, dtype=jnp.float32) + (
        genotype_matrix * prior_variances[None, :]
    ) @ jnp.transpose(genotype_matrix)
    covariance_matrix = covariance_matrix + jitter * jnp.eye(sample_count, dtype=jnp.float32)
    cholesky_factor = jnp.linalg.cholesky(covariance_matrix)

    inverse_covariance_targets = jax.scipy.linalg.cho_solve((cholesky_factor, True), targets)
    inverse_covariance_covariates = jax.scipy.linalg.cho_solve((cholesky_factor, True), covariate_matrix)
    gls_normal_matrix = (
        jnp.transpose(covariate_matrix) @ inverse_covariance_covariates
        + jitter * jnp.eye(covariate_matrix.shape[1], dtype=jnp.float32)
    )
    alpha = jnp.linalg.solve(
        gls_normal_matrix,
        jnp.transpose(covariate_matrix) @ inverse_covariance_targets,
    )

    residual = targets - covariate_matrix @ alpha
    inverse_covariance_residual = jax.scipy.linalg.cho_solve((cholesky_factor, True), residual)
    beta = prior_variances * (jnp.transpose(genotype_matrix) @ inverse_covariance_residual)

    inverse_covariance_genotypes = jax.scipy.linalg.cho_solve((cholesky_factor, True), genotype_matrix)
    leverage_diagonal = jnp.sum(genotype_matrix * inverse_covariance_genotypes, axis=0)
    beta_variance = jnp.maximum(prior_variances - (prior_variances * prior_variances) * leverage_diagonal, 1e-8)

    linear_predictor = covariate_matrix @ alpha + genotype_matrix @ beta
    residual_vector = targets - linear_predictor
    sigma_error2_new = jnp.maximum(jnp.mean(residual_vector * residual_vector), sigma_error_floor)
    collapsed_objective = -0.5 * (
        residual @ inverse_covariance_residual
        + 2.0 * jnp.sum(jnp.log(jnp.diag(cholesky_factor)))
        + sample_count * jnp.log(2.0 * jnp.pi)
    )
    return alpha, beta, beta_variance, linear_predictor, collapsed_objective, sigma_error2_new


@partial(jax.jit, static_argnames=("max_iterations",))
def _binary_posterior_state(
    genotype_matrix: jnp.ndarray,
    covariate_matrix: jnp.ndarray,
    targets: jnp.ndarray,
    prior_variances: jnp.ndarray,
    alpha_init: jnp.ndarray,
    beta_init: jnp.ndarray,
    minimum_weight: float,
    max_iterations: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    prior_precision = 1.0 / jnp.maximum(prior_variances, 1e-8)
    parameter_count = covariate_matrix.shape[1] + genotype_matrix.shape[1]
    initial_parameters = jnp.concatenate([alpha_init, beta_init], axis=0)

    def penalized_terms(parameters: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        alpha = parameters[: covariate_matrix.shape[1]]
        beta = parameters[covariate_matrix.shape[1] :]
        linear_predictor = covariate_matrix @ alpha + genotype_matrix @ beta
        probabilities = jax.nn.sigmoid(linear_predictor)
        weights = jnp.maximum(probabilities * (1.0 - probabilities), minimum_weight)
        residual = targets - probabilities

        gradient_alpha = jnp.transpose(covariate_matrix) @ residual
        gradient_beta = jnp.transpose(genotype_matrix) @ residual - prior_precision * beta
        gradient = jnp.concatenate([gradient_alpha, gradient_beta], axis=0)

        weighted_covariates = covariate_matrix * weights[:, None]
        weighted_genotypes = genotype_matrix * weights[:, None]
        precision_aa = jnp.transpose(covariate_matrix) @ weighted_covariates
        precision_ab = jnp.transpose(covariate_matrix) @ weighted_genotypes
        precision_bb = (
            jnp.transpose(genotype_matrix) @ weighted_genotypes
            + jnp.diag(prior_precision)
        )
        precision_matrix = jnp.block([
            [precision_aa, precision_ab],
            [jnp.transpose(precision_ab), precision_bb],
        ])

        penalized_log_posterior = jnp.sum(
            targets * jax.nn.log_sigmoid(linear_predictor)
            + (1.0 - targets) * jax.nn.log_sigmoid(-linear_predictor)
        ) - 0.5 * jnp.sum(prior_precision * beta * beta)
        return penalized_log_posterior, gradient, precision_matrix, linear_predictor

    def body_fun(_iteration_index: int, state: tuple[jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
        parameters, damping = state
        current_objective, gradient, precision_matrix, _linear_predictor = penalized_terms(parameters)
        damping_metric = jnp.diag(jnp.maximum(jnp.diag(precision_matrix), 1e-3))
        step = jnp.linalg.solve(
            precision_matrix + damping * damping_metric + 1e-4 * jnp.eye(parameter_count, dtype=jnp.float32),
            gradient,
        )
        candidate_parameters = parameters + step
        candidate_objective, _candidate_gradient, _candidate_precision, _candidate_linear_predictor = penalized_terms(candidate_parameters)
        accept = candidate_objective > current_objective
        next_parameters = jnp.where(accept, candidate_parameters, parameters)
        next_damping = jnp.where(accept, jnp.maximum(damping * 0.5, 1e-4), jnp.minimum(damping * 4.0, 1e6))
        return next_parameters, next_damping

    final_parameters, _final_damping = jax.lax.fori_loop(
        0,
        max_iterations,
        body_fun,
        (initial_parameters, jnp.float32(1.0)),
    )
    final_objective, _gradient, final_precision, linear_predictor = penalized_terms(final_parameters)
    final_precision = final_precision + 1e-4 * jnp.eye(parameter_count, dtype=jnp.float32)
    cholesky_factor = jnp.linalg.cholesky(final_precision)
    inverse_precision = jax.scipy.linalg.cho_solve((cholesky_factor, True), jnp.eye(parameter_count, dtype=jnp.float32))
    beta_variance = jnp.maximum(
        jnp.diag(inverse_precision)[covariate_matrix.shape[1] :],
        1e-8,
    )
    laplace_objective = (
        final_objective
        + 0.5 * jnp.sum(jnp.log(prior_precision))
        - jnp.sum(jnp.log(jnp.diag(cholesky_factor)))
    )
    return (
        final_parameters[: covariate_matrix.shape[1]],
        final_parameters[covariate_matrix.shape[1] :],
        beta_variance,
        linear_predictor,
        laplace_objective,
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

    log_length = np.log(np.maximum(np.asarray([record.length for record in records], dtype=np.float64), 1.0))
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
    repeat_indicator = np.asarray([float(record.is_repeat) for record in records], dtype=np.float64)
    copy_number_indicator = np.asarray([float(record.is_copy_number) for record in records], dtype=np.float64)

    standardized_log_length = _standardize_metadata(log_length)
    standardized_allele_frequency = _standardize_metadata(allele_frequency)
    standardized_quality = _standardize_metadata(quality)

    design_columns: list[np.ndarray] = []
    feature_names: list[str] = []
    for class_index, variant_class in enumerate(unique_classes):
        class_membership = class_membership_matrix[:, class_index]
        design_columns.append(_center_design_column(class_membership))
        feature_names.append("type_offset::" + variant_class.value)

        design_columns.append(_center_design_column(class_membership * standardized_log_length))
        feature_names.append("log_length_linear::" + variant_class.value)
        design_columns.append(_center_design_column(class_membership * standardized_log_length * standardized_log_length))
        feature_names.append("log_length_quadratic::" + variant_class.value)

        design_columns.append(_center_design_column(class_membership * standardized_allele_frequency))
        feature_names.append("allele_frequency_linear::" + variant_class.value)
        design_columns.append(
            _center_design_column(class_membership * standardized_allele_frequency * standardized_allele_frequency)
        )
        feature_names.append("allele_frequency_quadratic::" + variant_class.value)

    design_columns.append(_center_design_column(standardized_quality))
    feature_names.append("quality_linear")
    design_columns.append(_center_design_column(standardized_quality * standardized_quality))
    feature_names.append("quality_quadratic")
    design_columns.append(_center_design_column(repeat_indicator))
    feature_names.append("repeat_indicator")
    design_columns.append(_center_design_column(copy_number_indicator))
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


def _center_design_column(values: np.ndarray) -> np.ndarray:
    centered = np.asarray(values, dtype=np.float64) - float(np.mean(values))
    if np.max(np.abs(centered)) < 1e-10:
        return np.zeros_like(centered)
    return centered


def _initialize_scale_model(
    prior_design: PriorDesign,
    config: ModelConfig,
) -> tuple[float, np.ndarray]:
    default_log_scales = config.class_log_baseline_scales()
    class_log_scale_vector = np.asarray(
        [
            default_log_scales[prior_design.inverse_class_lookup[class_index]]
            for class_index in range(len(prior_design.inverse_class_lookup))
        ],
        dtype=np.float64,
    )
    default_log_scale_by_variant = prior_design.class_membership_matrix @ class_log_scale_vector
    mean_log_scale = float(np.mean(default_log_scale_by_variant))
    initialized_global_scale = float(
        np.clip(
            np.exp(mean_log_scale),
            config.global_scale_floor,
            config.global_scale_ceiling,
        )
    )
    if prior_design.design_matrix.shape[1] == 0:
        return initialized_global_scale, np.zeros(0, dtype=np.float64)

    target_offsets = default_log_scale_by_variant - mean_log_scale
    penalty = _scale_model_penalty(prior_design.feature_names, config)
    normal_matrix = prior_design.design_matrix.T @ prior_design.design_matrix + np.diag(np.maximum(penalty, 1e-8))
    right_hand_side = prior_design.design_matrix.T @ target_offsets
    coefficients = np.linalg.solve(normal_matrix, right_hand_side)
    return initialized_global_scale, coefficients.astype(np.float64)


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
    if design_matrix.shape[1] == 0:
        return np.ones(design_matrix.shape[0], dtype=np.float64)
    linear_prediction = design_matrix @ scale_model_coefficients
    bounded_log_scales = np.clip(
        linear_prediction,
        np.log(config.prior_scale_floor),
        np.log(config.prior_scale_ceiling),
    )
    return np.exp(bounded_log_scales).astype(np.float64)


def _effective_prior_variances(
    baseline_prior_variances: np.ndarray,
    local_scale: np.ndarray,
    trait_type: TraitType,
    config: ModelConfig,
) -> np.ndarray:
    total_prior_variances = baseline_prior_variances * np.maximum(local_scale, config.local_scale_floor)
    if trait_type != TraitType.BINARY or not config.enable_horseshoe_slab:
        return np.maximum(total_prior_variances, 1e-8)
    slab_variance = config.regularized_horseshoe_slab_scale ** 2
    regularized_variances = slab_variance * total_prior_variances / np.maximum(
        slab_variance + total_prior_variances,
        1e-8,
    )
    return np.maximum(regularized_variances, 1e-8)


def _scale_model_penalty(
    feature_names: Sequence[str],
    config: ModelConfig,
) -> np.ndarray:
    penalty_values = np.full(len(feature_names), config.scale_model_ridge_penalty, dtype=np.float64)
    for feature_index, feature_name in enumerate(feature_names):
        if feature_name.startswith("type_offset::"):
            penalty_values[feature_index] = config.type_offset_penalty
    return penalty_values


def _scale_penalty_objective(
    scale_model_coefficients: np.ndarray,
    scale_penalty: np.ndarray,
) -> float:
    return float(-0.5 * np.sum(scale_penalty * scale_model_coefficients * scale_model_coefficients))


def _pack_theta(global_scale: float, scale_model_coefficients: np.ndarray) -> np.ndarray:
    return np.concatenate([
        np.asarray([np.log(np.maximum(global_scale, 1e-8))], dtype=np.float64),
        np.asarray(scale_model_coefficients, dtype=np.float64),
    ])


def _unpack_theta(theta: np.ndarray) -> tuple[float, np.ndarray]:
    return float(np.exp(theta[0])), np.asarray(theta[1:], dtype=np.float64)


def _local_scale_prior_objective(
    local_scale: np.ndarray,
    auxiliary_delta: np.ndarray,
    local_shape_a: np.ndarray,
    local_shape_b: np.ndarray,
) -> float:
    return float(
        np.sum(
            local_shape_a * np.log(np.maximum(auxiliary_delta, 1e-12))
            - scipy_gammaln(local_shape_a)
            + (local_shape_a - 1.0) * np.log(np.maximum(local_scale, 1e-12))
            - auxiliary_delta * local_scale
        )
        + np.sum(
            -scipy_gammaln(local_shape_b)
            + (local_shape_b - 1.0) * np.log(np.maximum(auxiliary_delta, 1e-12))
            - auxiliary_delta
        )
    )


def _update_local_scales(
    coefficient_second_moment: np.ndarray,
    baseline_prior_variances: np.ndarray,
    local_shape_a: np.ndarray,
    local_shape_b: np.ndarray,
    auxiliary_delta: np.ndarray,
    config: ModelConfig,
) -> tuple[np.ndarray, np.ndarray]:
    chi = np.maximum(
        coefficient_second_moment / np.maximum(baseline_prior_variances, 1e-12),
        1e-12,
    )
    psi = np.maximum(2.0 * auxiliary_delta, 1e-12)
    p_parameter = np.asarray(local_shape_a, dtype=np.float64) - 0.5
    expected_local_scale = _gig_moment(
        p_parameter=p_parameter,
        chi=chi,
        psi=psi,
        moment_power=1.0,
    )
    updated_local_scale = np.maximum(expected_local_scale, config.local_scale_floor)
    updated_auxiliary_delta = (local_shape_a + local_shape_b) / np.maximum(
        1.0 + updated_local_scale,
        config.local_scale_floor,
    )
    return updated_local_scale, updated_auxiliary_delta


def _gig_moment(
    p_parameter: np.ndarray,
    chi: np.ndarray,
    psi: np.ndarray,
    moment_power: float,
) -> np.ndarray:
    z_value = np.sqrt(np.maximum(chi * psi, 1e-12))
    numerator = scipy_kve(p_parameter + moment_power, z_value)
    denominator = np.maximum(scipy_kve(p_parameter, z_value), 1e-300)
    moment_ratio = numerator / denominator
    return np.asarray(
        np.power(np.maximum(chi / psi, 1e-12), 0.5 * moment_power) * moment_ratio,
        dtype=np.float64,
    )


def _identity_tie_map(variant_count: int) -> TieMap:
    return TieMap(
        kept_indices=np.arange(variant_count, dtype=np.int32),
        original_to_reduced=np.arange(variant_count, dtype=np.int32),
        reduced_to_group=[
            TieGroup(
                representative_index=variant_index,
                member_indices=np.asarray([variant_index], dtype=np.int32),
                signs=np.asarray([1.0], dtype=np.float32),
            )
            for variant_index in range(variant_count)
        ],
    )


def _build_tie_layout(tie_map: TieMap) -> TieLayout:
    member_to_reduced = np.asarray(tie_map.original_to_reduced, dtype=np.int32)
    if np.any(member_to_reduced < 0):
        raise ValueError("tie_map for inference cannot contain inactive members.")
    if not tie_map.reduced_to_group:
        return TieLayout(
            member_to_reduced=member_to_reduced,
            reduced_member_indices=np.zeros((0, 0), dtype=np.int32),
            reduced_member_mask=np.zeros((0, 0), dtype=np.float64),
        )
    max_group_size = max(int(group.member_indices.shape[0]) for group in tie_map.reduced_to_group)
    reduced_count = len(tie_map.reduced_to_group)
    reduced_member_indices = np.zeros((reduced_count, max_group_size), dtype=np.int32)
    reduced_member_mask = np.zeros((reduced_count, max_group_size), dtype=np.float64)
    for reduced_index, tie_group in enumerate(tie_map.reduced_to_group):
        group_size = int(tie_group.member_indices.shape[0])
        reduced_member_indices[reduced_index, :group_size] = tie_group.member_indices.astype(np.int32)
        reduced_member_mask[reduced_index, :group_size] = 1.0
    return TieLayout(
        member_to_reduced=member_to_reduced,
        reduced_member_indices=reduced_member_indices,
        reduced_member_mask=reduced_member_mask,
    )


def _reduce_member_values(
    member_values: np.ndarray,
    tie_layout: TieLayout,
) -> np.ndarray:
    gathered_values = member_values[tie_layout.reduced_member_indices]
    return np.sum(gathered_values * tie_layout.reduced_member_mask, axis=1)


def _project_reduced_second_moment_to_members(
    reduced_second_moment: np.ndarray,
    member_prior_variances: np.ndarray,
    tie_layout: TieLayout,
) -> np.ndarray:
    reduced_prior_variances = np.maximum(_reduce_member_values(member_prior_variances, tie_layout), 1e-12)
    reduced_group_second_moment = reduced_second_moment[tie_layout.member_to_reduced]
    conditional_mean_weight = member_prior_variances / reduced_prior_variances[tie_layout.member_to_reduced]
    conditional_residual_variance = member_prior_variances * (1.0 - conditional_mean_weight)
    return conditional_residual_variance + (conditional_mean_weight * conditional_mean_weight) * reduced_group_second_moment


def _share_local_scale_within_tie_groups(
    local_scale: np.ndarray,
    baseline_prior_variances: np.ndarray,
    tie_layout: TieLayout,
) -> np.ndarray:
    if tie_layout.reduced_member_indices.size == 0:
        return local_scale

    shared_local_scale = np.asarray(local_scale, dtype=np.float64).copy()
    for group_indices, group_mask in zip(
        tie_layout.reduced_member_indices,
        tie_layout.reduced_member_mask,
        strict=True,
    ):
        active_indices = group_indices[group_mask.astype(bool)]
        if active_indices.shape[0] <= 1:
            continue
        group_baseline_variances = np.asarray(baseline_prior_variances[active_indices], dtype=np.float64)
        normalized_weights = group_baseline_variances / np.maximum(np.sum(group_baseline_variances), 1e-12)
        group_local_scale = float(np.sum(normalized_weights * shared_local_scale[active_indices]))
        shared_local_scale[active_indices] = group_local_scale
    return shared_local_scale


def _relative_parameter_change(
    current_beta: np.ndarray,
    previous_beta: np.ndarray | None,
    current_alpha: np.ndarray,
    previous_alpha: np.ndarray | None,
    current_local_scale: np.ndarray,
    previous_local_scale: np.ndarray | None,
    current_theta: np.ndarray,
    previous_theta: np.ndarray | None,
) -> float:
    if previous_beta is None:
        return float("inf")
    changes = [
        _relative_change(current_beta, previous_beta),
        _relative_change(current_alpha, previous_alpha),
        _relative_change(current_local_scale, previous_local_scale),
        _relative_change(current_theta, previous_theta),
    ]
    return float(max(changes))


def _relative_change(current_values: np.ndarray, previous_values: np.ndarray | None) -> float:
    if previous_values is None:
        return 0.0
    denominator = max(float(np.linalg.norm(previous_values)), 1e-8)
    return float(np.linalg.norm(current_values - previous_values) / denominator)


def _validation_metric(
    trait_type: TraitType,
    genotype_matrix: np.ndarray,
    covariate_matrix: np.ndarray,
    targets: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> float:
    linear_predictor = genotype_matrix @ beta + covariate_matrix @ alpha
    if trait_type == TraitType.BINARY:
        positive_probability = stable_sigmoid(linear_predictor)
        return float(
            -np.mean(
                targets * np.log(positive_probability + 1e-8)
                + (1.0 - targets) * np.log(1.0 - positive_probability + 1e-8)
            )
        )
    residual_vector = targets - linear_predictor
    return float(np.mean(residual_vector * residual_vector))


def _calibrate_binary_intercept(
    linear_predictor: np.ndarray,
    targets: np.ndarray,
) -> float:
    target_array = np.asarray(targets, dtype=np.float64)
    base_linear_predictor = np.asarray(linear_predictor, dtype=np.float64)
    target_prevalence = float(np.clip(np.mean(target_array), 1e-6, 1.0 - 1e-6))
    intercept_shift = float(np.log(target_prevalence / (1.0 - target_prevalence)) - np.mean(base_linear_predictor))
    for _iteration_index in range(25):
        shifted_predictor = base_linear_predictor + intercept_shift
        probabilities = np.asarray(stable_sigmoid(shifted_predictor), dtype=np.float64)
        gradient = float(np.sum(probabilities - target_array))
        hessian = float(np.sum(probabilities * (1.0 - probabilities)))
        if hessian <= 1e-8:
            break
        step = gradient / hessian
        intercept_shift -= step
        if abs(step) < 1e-6:
            break
    return float(intercept_shift)


def _trigamma(value: float) -> float:
    return float(scipy_polygamma(1, np.float64(value)))

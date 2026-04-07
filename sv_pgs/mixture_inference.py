"""Variational EM inference for Bayesian polygenic scores.

High-level idea
---------------
We want to estimate how much each genetic variant contributes to a trait
(e.g. type-2 diabetes risk).  Most variants have near-zero effect, but a
few may be large — especially structural variants (deletions, duplications,
etc.).  This module fits a Bayesian model that automatically learns which
variants matter and how much to trust each one.

Model
-----
Each variant j gets an effect size beta_j drawn from a normal distribution
whose variance tau_j^2 encodes our prior belief about how big that effect
could be:

    beta_j ~ Normal(0, tau_j^2)

The prior variance tau_j^2 is built from three pieces:
  - sigma_g      : a single global scale (shared across all variants)
  - s_j          : a metadata-driven baseline scale (depends on variant
                   type, length, repeat status — bigger for SVs than SNVs)
  - lambda_j     : a per-variant local shrinkage factor that lets the model
                   pull unimportant variants toward zero while letting
                   important ones stay large

Together:  tau_j^2 = (sigma_g * s_j)^2 * lambda_j

The local shrinkage lambda_j uses a "three-parameter Beta" (TPB) prior
(a type of heavy-tailed distribution), controlled by class-specific shape
parameters (a_j, b_j).  Smaller shapes = heavier tails = more tolerance
for large effects.

Inference loop (variational EM)
-------------------------------
The algorithm iterates three steps until convergence:

  1. **E-step (posterior)**: Given current prior variances, solve for the
     best-fit effect sizes beta_j.  For continuous traits this uses REML
     (restricted maximum likelihood); for binary traits it uses a
     Newton trust-region optimizer with Polya-Gamma augmentation.

  2. **Local scale update**: Given the fitted betas, update each variant's
     local shrinkage lambda_j using Generalized Inverse Gaussian (GIG)
     moments — think of this as re-calibrating how much each variant
     should be shrunk toward zero.

  3. **Hyperparameter update** (every 4th iteration): Re-estimate the
     global scale sigma_g, the metadata scale-model coefficients, and
     the TPB shape parameters (a_j, b_j) to better match the data.
"""

from __future__ import annotations

from dataclasses import dataclass, fields as dataclass_fields
import hashlib
import json
from typing import Callable, Sequence, cast

import sv_pgs._jax  # noqa: F401
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular as jax_solve_triangular
from jax.scipy.special import digamma as jax_digamma
from jax.scipy.special import gammaln as jax_gammaln
import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import kve as scipy_bessel_kve

from sv_pgs._jax import gpu_compute_jax_dtype, gpu_compute_numpy_dtype
from sv_pgs.config import ModelConfig, TraitType, VariantClass
from sv_pgs.data import TieMap, VariantRecord
from sv_pgs.genotype import (
    DenseRawGenotypeMatrix,
    StandardizedGenotypeMatrix,
    _cupy_compute_dtype,
    _iter_standardized_gpu_batches,
    _try_import_cupy,
    _cupy_to_jax,
    _to_cupy_compute,
    _to_cupy_float64,
)
from sv_pgs.linear_solvers import build_linear_operator, solve_spd_system, stochastic_logdet
from sv_pgs.numeric import stable_sigmoid
from sv_pgs.preprocessing import collapse_tie_groups
from sv_pgs.progress import log, mem

# GPU exact variant solve needs ~3× the matrix memory for intermediates
# (X + weighted_X + projected_X). On 16 GB T4 with 4 GB matrix, only ~6 GB
# free → can handle up to ~2000 variants before OOM. Use PCG for larger.
GPU_EXACT_VARIANT_SOLVE_LIMIT = 2_000


@dataclass(slots=True)
class PriorDesign:
    """Describes what we know about each variant *before* seeing the outcome data.

    Each variant's "allowed effect size range" depends on its metadata:
    variant type (SNV vs deletion vs duplication ...), length, whether it
    overlaps a repeat region, etc.  This dataclass holds the matrices that
    encode those relationships so the model can learn how metadata
    predicts effect magnitude.
    """
    design_matrix: np.ndarray           # each row = one variant's metadata features
    feature_names: list[str]            # human-readable names for each feature column
    feature_specs: tuple[ScaleModelFeatureSpec, ...]  # compiled feature descriptors for fast reuse
    class_membership_matrix: np.ndarray # which variant class(es) each variant belongs to
    inverse_class_lookup: dict[int, VariantClass]  # column index -> VariantClass enum


@dataclass(slots=True)
class ScaleModelFeatureSpec:
    kind: str
    variant_class: VariantClass | None = None
    continuous_feature_name: str | None = None


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
    member_prior_variances: np.ndarray


@dataclass(slots=True)
class VariationalFitCheckpoint:
    config_signature: str
    prior_design_signature: str
    validation_enabled: bool
    completed_iterations: int
    alpha_state: np.ndarray
    beta_state: np.ndarray
    local_scale: np.ndarray
    auxiliary_delta: np.ndarray
    sigma_error2: float
    global_scale: float
    scale_model_coefficients: np.ndarray
    tpb_shape_a_vector: np.ndarray
    tpb_shape_b_vector: np.ndarray
    objective_history: list[float]
    validation_history: list[float]
    previous_alpha: np.ndarray | None
    previous_beta: np.ndarray | None
    previous_local_scale: np.ndarray | None
    previous_theta: np.ndarray | None
    previous_tpb_shape_a_vector: np.ndarray | None
    previous_tpb_shape_b_vector: np.ndarray | None
    best_validation_metric: float | None
    best_alpha: np.ndarray | None
    best_beta: np.ndarray | None
    best_local_scale: np.ndarray | None
    best_theta: np.ndarray | None
    best_sigma_error2: float | None
    best_tpb_shape_a_vector: np.ndarray | None
    best_tpb_shape_b_vector: np.ndarray | None


@dataclass(slots=True)
class PosteriorState:
    """Result of fitting effect sizes for one EM iteration.

    The prediction model is:
        predicted_trait = covariates_contribution + genotype_contribution
                        = (covariates @ alpha)   + (genotypes @ beta)

    alpha captures non-genetic effects (age, sex, PCs, intercept).
    beta captures each variant's estimated effect on the trait.
    """
    alpha: np.ndarray          # covariate coefficients (intercept, age, sex, PCs ...)
    beta: np.ndarray           # estimated effect size per variant (reduced/unique set)
    beta_variance: np.ndarray  # uncertainty in each beta (larger = less confident)
    linear_predictor: np.ndarray  # full predicted values for each sample
    collapsed_objective: float    # model quality score (higher = better fit)
    sigma_error2: float           # unexplained noise variance (fixed at 1.0 for binary)


def _checkpoint_config_signature(config: ModelConfig) -> str:
    payload: dict[str, object] = {}
    for field in dataclass_fields(config):
        value = getattr(config, field.name)
        payload[field.name] = value.value if hasattr(value, "value") else value
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _checkpoint_prior_design_signature(prior_design: PriorDesign) -> str:
    hasher = hashlib.sha256()
    hasher.update(np.ascontiguousarray(prior_design.design_matrix, dtype=np.float64).view(np.uint8).tobytes())
    hasher.update(np.ascontiguousarray(prior_design.class_membership_matrix, dtype=np.float64).view(np.uint8).tobytes())
    hasher.update(json.dumps(list(prior_design.feature_names), sort_keys=False).encode("utf-8"))
    hasher.update(
        json.dumps(
            {int(index): variant_class.value for index, variant_class in sorted(prior_design.inverse_class_lookup.items())},
            sort_keys=True,
        ).encode("utf-8")
    )
    return hasher.hexdigest()


def _initialize_alpha_state(
    covariate_matrix: np.ndarray,
    targets: np.ndarray,
    trait_type: TraitType,
) -> np.ndarray:
    covariates = np.asarray(covariate_matrix, dtype=np.float64)
    target_array = np.asarray(targets, dtype=np.float64)
    if covariates.ndim != 2:
        raise ValueError("covariate_matrix must be 2D.")
    alpha_state = np.zeros(covariates.shape[1], dtype=np.float64)
    if covariates.shape[1] == 0:
        return alpha_state
    if trait_type == TraitType.BINARY:
        prevalence = float(np.clip(np.mean(target_array), 1e-6, 1.0 - 1e-6))
        alpha_state[0] = float(np.log(prevalence / (1.0 - prevalence)))
        return alpha_state
    normal_matrix = covariates.T @ covariates + np.eye(covariates.shape[1], dtype=np.float64) * 1e-8
    right_hand_side = covariates.T @ target_array
    return np.linalg.solve(normal_matrix, right_hand_side).astype(np.float64, copy=False)


def _apply_binary_intercept_calibration(
    posterior_state: PosteriorState,
    targets: np.ndarray,
) -> PosteriorState:
    intercept_shift = np.float64(
        _calibrate_binary_intercept(
            linear_predictor=posterior_state.linear_predictor,
            targets=targets,
        )
    )
    calibrated_alpha = np.asarray(posterior_state.alpha, dtype=np.float64).copy()
    calibrated_alpha[0] += intercept_shift
    return PosteriorState(
        alpha=calibrated_alpha,
        beta=np.asarray(posterior_state.beta, dtype=np.float64),
        beta_variance=np.asarray(posterior_state.beta_variance, dtype=np.float64),
        linear_predictor=np.asarray(posterior_state.linear_predictor, dtype=np.float64) + intercept_shift,
        collapsed_objective=float(posterior_state.collapsed_objective),
        sigma_error2=float(posterior_state.sigma_error2),
    )


def _gpu_exact_variant_linear_predictor(X_gpu, beta: np.ndarray) -> np.ndarray:
    import cupy as cp

    return np.asarray((X_gpu.astype(cp.float64, copy=False) @ cp.asarray(beta, dtype=cp.float64)).get(), dtype=np.float64)


def _gpu_cholesky_solve(right_hand_side, cholesky_factor_gpu, solve_triangular_gpu):
    import cupy as cp

    rhs_gpu = cp.asarray(right_hand_side, dtype=cholesky_factor_gpu.dtype)
    lower_solution = solve_triangular_gpu(cholesky_factor_gpu, rhs_gpu, lower=True)
    return solve_triangular_gpu(cholesky_factor_gpu.T, lower_solution, lower=False)


def _covariates_only_fit_result(
    covariate_matrix: np.ndarray,
    targets: np.ndarray,
    trait_type: TraitType,
    config: ModelConfig,
    validation_data: tuple[StandardizedGenotypeMatrix | np.ndarray, np.ndarray, np.ndarray] | None,
) -> VariationalFitResult:
    alpha = (
        _fit_covariates_only_binary(
            covariate_matrix=covariate_matrix,
            targets=targets,
            minimum_weight=config.polya_gamma_minimum_weight,
            max_iterations=config.max_inner_newton_iterations,
            gradient_tolerance=config.newton_gradient_tolerance,
        )
        if trait_type == TraitType.BINARY
        else _initialize_alpha_state(
            covariate_matrix=covariate_matrix,
            targets=targets,
            trait_type=trait_type,
        )
    )
    beta = np.zeros(0, dtype=np.float64)
    linear_predictor = np.asarray(covariate_matrix @ alpha, dtype=np.float64)
    sigma_error2 = 1.0
    if trait_type == TraitType.BINARY:
        if config.binary_intercept_calibration and alpha.size > 0:
            intercept_shift = _calibrate_binary_intercept(linear_predictor=linear_predictor, targets=targets)
            alpha = alpha.copy()
            alpha[0] += intercept_shift
            linear_predictor = linear_predictor + intercept_shift
        positive_probability = np.asarray(stable_sigmoid(linear_predictor), dtype=np.float64)
        objective_value = float(
            np.mean(
                targets * np.log(positive_probability + 1e-8)
                + (1.0 - targets) * np.log(1.0 - positive_probability + 1e-8)
            )
        )
    else:
        residual_vector = np.asarray(targets - linear_predictor, dtype=np.float64)
        sigma_error2 = float(max(np.mean(residual_vector * residual_vector), config.sigma_error_floor))
        objective_value = float(-0.5 * np.mean(residual_vector * residual_vector))

    validation_history: list[float] = []
    if validation_data is not None:
        validation_history.append(
            _validation_metric(
                trait_type=trait_type,
                genotype_matrix=validation_data[0],
                covariate_matrix=validation_data[1],
                targets=validation_data[2],
                alpha=alpha,
                beta=beta,
            )
        )

    return VariationalFitResult(
        alpha=np.asarray(alpha, dtype=np.float32),
        beta_reduced=np.zeros(0, dtype=np.float32),
        beta_variance=np.zeros(0, dtype=np.float32),
        prior_scales=np.zeros(0, dtype=np.float32),
        global_scale=float(np.clip(1.0, config.global_scale_floor, config.global_scale_ceiling)),
        class_tpb_shape_a={},
        class_tpb_shape_b={},
        scale_model_coefficients=np.zeros(0, dtype=np.float32),
        scale_model_feature_names=[],
        sigma_error2=float(sigma_error2),
        objective_history=[objective_value],
        validation_history=validation_history,
        member_prior_variances=np.zeros(0, dtype=np.float32),
    )


def _fit_covariates_only_binary(
    covariate_matrix: np.ndarray,
    targets: np.ndarray,
    minimum_weight: float,
    max_iterations: int,
    gradient_tolerance: float,
) -> np.ndarray:
    covariates = np.asarray(covariate_matrix, dtype=np.float64)
    target_array = np.asarray(targets, dtype=np.float64)
    alpha = _initialize_alpha_state(covariates, target_array, TraitType.BINARY)
    if covariates.shape[1] == 0:
        return alpha
    for _ in range(max_iterations):
        linear_predictor = np.asarray(covariates @ alpha, dtype=np.float64)
        probabilities = np.asarray(stable_sigmoid(linear_predictor), dtype=np.float64)
        weights = np.maximum(probabilities * (1.0 - probabilities), minimum_weight)
        gradient = covariates.T @ (target_array - probabilities)
        if float(np.linalg.norm(gradient)) <= gradient_tolerance:
            break
        hessian = covariates.T @ (weights[:, None] * covariates)
        step = np.linalg.solve(
            hessian + np.eye(hessian.shape[0], dtype=np.float64) * 1e-8,
            gradient,
        )
        alpha += step
        if float(np.linalg.norm(step)) <= gradient_tolerance:
            break
    return np.asarray(alpha, dtype=np.float64)


def _fit_binary_alpha_with_offset(
    covariate_matrix: np.ndarray,
    targets: np.ndarray,
    predictor_offset: np.ndarray,
    minimum_weight: float,
    max_iterations: int,
    gradient_tolerance: float,
    alpha_init: np.ndarray | None = None,
) -> np.ndarray:
    covariates = np.asarray(covariate_matrix, dtype=np.float64)
    target_array = np.asarray(targets, dtype=np.float64)
    offset = np.asarray(predictor_offset, dtype=np.float64).reshape(-1)
    if covariates.shape[0] != offset.shape[0]:
        raise ValueError("predictor_offset sample count must match covariates.")
    if covariates.shape[1] == 0:
        return np.zeros(0, dtype=np.float64)
    alpha = (
        np.asarray(alpha_init, dtype=np.float64).copy()
        if alpha_init is not None
        else _initialize_alpha_state(covariates, target_array, TraitType.BINARY)
    )
    for _ in range(max_iterations):
        linear_predictor = np.asarray(offset + covariates @ alpha, dtype=np.float64)
        probabilities = np.asarray(stable_sigmoid(linear_predictor), dtype=np.float64)
        weights = np.maximum(probabilities * (1.0 - probabilities), minimum_weight)
        gradient = covariates.T @ (target_array - probabilities)
        if float(np.linalg.norm(gradient)) <= gradient_tolerance:
            break
        hessian = covariates.T @ (weights[:, None] * covariates)
        step = np.linalg.solve(
            hessian + np.eye(hessian.shape[0], dtype=np.float64) * 1e-8,
            gradient,
        )
        alpha += step
        if float(np.linalg.norm(step)) <= gradient_tolerance:
            break
    return np.asarray(alpha, dtype=np.float64)


def _stochastic_step_size(config: ModelConfig, step_index: int) -> float:
    if step_index < 1:
        raise ValueError("step_index must be positive.")
    return float((config.stochastic_step_offset + float(step_index)) ** (-config.stochastic_step_exponent))


def _stochastic_variant_blocks(
    variant_count: int,
    block_size: int,
    random_generator: np.random.Generator,
) -> list[np.ndarray]:
    if variant_count < 1:
        return []
    safe_block_size = max(int(block_size), 1)
    block_starts = np.arange(0, variant_count, safe_block_size, dtype=np.int32)
    random_generator.shuffle(block_starts)
    return [
        np.arange(
            int(block_start),
            min(int(block_start) + safe_block_size, variant_count),
            dtype=np.int32,
        )
        for block_start in block_starts
    ]


def _should_use_stochastic_variational_updates(
    genotype_matrix: StandardizedGenotypeMatrix,
    config: ModelConfig,
) -> bool:
    variant_count = int(genotype_matrix.shape[1])
    if not config.stochastic_variational_updates:
        return False
    if variant_count < max(int(config.stochastic_min_variant_count), 1):
        return False
    return variant_count > int(config.stochastic_variant_batch_size)


def _stochastic_epoch_objective(
    trait_type: TraitType,
    targets: np.ndarray,
    linear_predictor: np.ndarray,
    beta: np.ndarray,
    reduced_prior_variances: np.ndarray,
    local_scale: np.ndarray,
    auxiliary_delta: np.ndarray,
    local_shape_a: np.ndarray,
    local_shape_b: np.ndarray,
    scale_model_coefficients: np.ndarray,
    scale_penalty: np.ndarray,
) -> float:
    predictor = np.asarray(linear_predictor, dtype=np.float64)
    target_array = np.asarray(targets, dtype=np.float64)
    if trait_type == TraitType.BINARY:
        probabilities = np.asarray(stable_sigmoid(predictor), dtype=np.float64)
        data_term = float(
            np.mean(
                target_array * np.log(probabilities + 1e-12)
                + (1.0 - target_array) * np.log(1.0 - probabilities + 1e-12)
            )
        )
    else:
        residual = np.asarray(target_array - predictor, dtype=np.float64)
        data_term = float(-0.5 * np.mean(residual * residual))
    beta_array = np.asarray(beta, dtype=np.float64)
    prior_term = float(
        -0.5 * np.mean(beta_array * beta_array / np.maximum(np.asarray(reduced_prior_variances, dtype=np.float64), 1e-8))
    )
    local_scale_term = _local_scale_prior_objective(
        local_scale=local_scale,
        auxiliary_delta=auxiliary_delta,
        local_shape_a=local_shape_a,
        local_shape_b=local_shape_b,
    ) / max(int(beta_array.shape[0]), 1)
    scale_penalty_term = _scale_penalty_objective(
        scale_model_coefficients=scale_model_coefficients,
        scale_penalty=scale_penalty,
    ) / max(int(beta_array.shape[0]), 1)
    return float(data_term + prior_term + local_scale_term + scale_penalty_term)


def fit_variational_em(
    genotypes: StandardizedGenotypeMatrix | np.ndarray,
    covariates: np.ndarray,
    targets: np.ndarray,
    records: Sequence[VariantRecord],
    config: ModelConfig,
    tie_map: TieMap,
    validation_data: tuple[StandardizedGenotypeMatrix | np.ndarray, np.ndarray, np.ndarray] | None = None,
    resume_checkpoint: VariationalFitCheckpoint | None = None,
    checkpoint_callback: Callable[[VariationalFitCheckpoint], None] | None = None,
) -> VariationalFitResult:
    genotype_matrix = _as_standardized_genotype_matrix(genotypes)
    covariate_matrix = np.asarray(covariates, dtype=np.float64)
    target_vector = np.asarray(targets, dtype=np.float64)
    member_records = list(records)
    if genotype_matrix.shape[1] != len(tie_map.reduced_to_group):
        raise ValueError("Reduced genotype columns must match tie-map group count.")
    if len(member_records) != tie_map.original_to_reduced.shape[0]:
        raise ValueError("records must align with tie_map member space.")

    reduced_records = collapse_tie_groups(member_records, tie_map)
    if len(reduced_records) != genotype_matrix.shape[1]:
        raise ValueError("Collapsed records must align with reduced genotype matrix.")

    validation_payload = _prepare_validation(validation_data)
    if validation_payload is not None:
        if resume_checkpoint is not None:
            log("  variational EM: ignoring resume checkpoint because validation data is present")
            resume_checkpoint = None
        checkpoint_callback = None
    if genotype_matrix.shape[1] == 0:
        log(
            f"  variational EM: covariates-only mode, 0 reduced variants, "
            f"{covariate_matrix.shape[1]} covariates, {target_vector.shape[0]} samples"
        )
        return _covariates_only_fit_result(
            covariate_matrix=covariate_matrix,
            targets=target_vector,
            trait_type=config.trait_type,
            config=config,
            validation_data=validation_payload,
        )
    prior_design = _build_prior_design(reduced_records)
    config_signature = _checkpoint_config_signature(config)
    prior_design_signature = _checkpoint_prior_design_signature(prior_design)
    scale_penalty = _scale_model_penalty(prior_design.feature_names, config)

    def _copy_optional(array: np.ndarray | None) -> np.ndarray | None:
        return None if array is None else np.asarray(array, dtype=np.float64).copy()

    def _build_checkpoint(completed_iterations: int) -> VariationalFitCheckpoint:
        return VariationalFitCheckpoint(
            config_signature=config_signature,
            prior_design_signature=prior_design_signature,
            validation_enabled=validation_payload is not None,
            completed_iterations=int(completed_iterations),
            alpha_state=np.asarray(alpha_state, dtype=np.float64).copy(),
            beta_state=np.asarray(beta_state, dtype=np.float64).copy(),
            local_scale=np.asarray(local_scale, dtype=np.float64).copy(),
            auxiliary_delta=np.asarray(auxiliary_delta, dtype=np.float64).copy(),
            sigma_error2=float(sigma_error2),
            global_scale=float(global_scale),
            scale_model_coefficients=np.asarray(scale_model_coefficients, dtype=np.float64).copy(),
            tpb_shape_a_vector=np.asarray(tpb_shape_a_vector, dtype=np.float64).copy(),
            tpb_shape_b_vector=np.asarray(tpb_shape_b_vector, dtype=np.float64).copy(),
            objective_history=[float(value) for value in objective_history],
            validation_history=[float(value) for value in validation_history],
            previous_alpha=_copy_optional(previous_alpha),
            previous_beta=_copy_optional(previous_beta),
            previous_local_scale=_copy_optional(previous_local_scale),
            previous_theta=_copy_optional(previous_theta),
            previous_tpb_shape_a_vector=_copy_optional(previous_tpb_shape_a_vector),
            previous_tpb_shape_b_vector=_copy_optional(previous_tpb_shape_b_vector),
            best_validation_metric=None if best_validation_metric is None else float(best_validation_metric),
            best_alpha=_copy_optional(best_alpha),
            best_beta=_copy_optional(best_beta),
            best_local_scale=_copy_optional(best_local_scale),
            best_theta=_copy_optional(best_theta),
            best_sigma_error2=None if best_sigma_error2 is None else float(best_sigma_error2),
            best_tpb_shape_a_vector=_copy_optional(best_tpb_shape_a_vector),
            best_tpb_shape_b_vector=_copy_optional(best_tpb_shape_b_vector),
        )

    def _initialize_em_state() -> None:
        nonlocal global_scale, scale_model_coefficients, tpb_shape_a_vector, tpb_shape_b_vector
        nonlocal local_shape_a, local_shape_b, local_scale, auxiliary_delta, sigma_error2
        nonlocal alpha_state, beta_state, objective_history, validation_history
        nonlocal previous_alpha, previous_beta, previous_local_scale, previous_theta
        nonlocal previous_tpb_shape_a_vector, previous_tpb_shape_b_vector
        nonlocal best_validation_metric, best_alpha, best_beta, best_local_scale, best_theta
        nonlocal best_sigma_error2, best_tpb_shape_a_vector, best_tpb_shape_b_vector, start_iteration
        global_scale, scale_model_coefficients = _initialize_scale_model(prior_design, config)
        tpb_shape_a_vector = _initialize_tpb_shape_a_vector(prior_design, config)
        tpb_shape_b_vector = _initialize_tpb_shape_b_vector(prior_design, config)
        local_scale = np.ones(len(reduced_records), dtype=np.float64)
        local_shape_a = prior_design.class_membership_matrix @ tpb_shape_a_vector
        local_shape_b = prior_design.class_membership_matrix @ tpb_shape_b_vector
        auxiliary_delta = np.asarray(local_shape_b, dtype=np.float64)
        sigma_error2 = 1.0
        alpha_state = _initialize_alpha_state(
            covariate_matrix=covariate_matrix,
            targets=target_vector,
            trait_type=config.trait_type,
        )
        beta_state = np.zeros(genotype_matrix.shape[1], dtype=np.float64)
        objective_history = []
        validation_history = []
        previous_alpha = None
        previous_beta = None
        previous_local_scale = None
        previous_theta = None
        previous_tpb_shape_a_vector = None
        previous_tpb_shape_b_vector = None
        best_validation_metric = None
        best_alpha = None
        best_beta = None
        best_local_scale = None
        best_theta = None
        best_sigma_error2 = None
        best_tpb_shape_a_vector = None
        best_tpb_shape_b_vector = None
        start_iteration = 0

    if resume_checkpoint is None:
        _initialize_em_state()
    else:
        if (
            resume_checkpoint.config_signature != config_signature
            or resume_checkpoint.prior_design_signature != prior_design_signature
            or bool(resume_checkpoint.validation_enabled) != bool(validation_payload is not None)
        ):
            log("  variational EM: checkpoint incompatible with current fit inputs; starting from scratch")
            resume_checkpoint = None
            _initialize_em_state()
        else:
            if resume_checkpoint.alpha_state.shape != (covariate_matrix.shape[1],):
                raise ValueError("resume checkpoint alpha_state shape does not match covariates.")
            if resume_checkpoint.beta_state.shape != (genotype_matrix.shape[1],):
                raise ValueError("resume checkpoint beta_state shape does not match reduced genotypes.")
            if resume_checkpoint.local_scale.shape != (len(reduced_records),):
                raise ValueError("resume checkpoint local_scale shape does not match reduced records.")
            if resume_checkpoint.auxiliary_delta.shape != (len(reduced_records),):
                raise ValueError("resume checkpoint auxiliary_delta shape does not match reduced records.")
            if resume_checkpoint.scale_model_coefficients.shape != (prior_design.design_matrix.shape[1],):
                raise ValueError("resume checkpoint scale_model_coefficients shape does not match prior design.")
            if resume_checkpoint.tpb_shape_a_vector.shape != (prior_design.class_membership_matrix.shape[1],):
                raise ValueError("resume checkpoint TPB shape-a size does not match prior classes.")
            if resume_checkpoint.tpb_shape_b_vector.shape != (prior_design.class_membership_matrix.shape[1],):
                raise ValueError("resume checkpoint TPB shape-b size does not match prior classes.")
            if resume_checkpoint.completed_iterations < 0 or resume_checkpoint.completed_iterations > config.max_outer_iterations:
                raise ValueError("resume checkpoint completed_iterations is out of range.")
            if len(resume_checkpoint.objective_history) != resume_checkpoint.completed_iterations:
                raise ValueError("resume checkpoint objective history length does not match completed iterations.")
            global_scale = float(resume_checkpoint.global_scale)
            scale_model_coefficients = np.asarray(resume_checkpoint.scale_model_coefficients, dtype=np.float64).copy()
            tpb_shape_a_vector = np.asarray(resume_checkpoint.tpb_shape_a_vector, dtype=np.float64).copy()
            tpb_shape_b_vector = np.asarray(resume_checkpoint.tpb_shape_b_vector, dtype=np.float64).copy()
            local_shape_a = prior_design.class_membership_matrix @ tpb_shape_a_vector
            local_shape_b = prior_design.class_membership_matrix @ tpb_shape_b_vector
            local_scale = np.asarray(resume_checkpoint.local_scale, dtype=np.float64).copy()
            auxiliary_delta = np.asarray(resume_checkpoint.auxiliary_delta, dtype=np.float64).copy()
            sigma_error2 = float(resume_checkpoint.sigma_error2)
            alpha_state = np.asarray(resume_checkpoint.alpha_state, dtype=np.float64).copy()
            beta_state = np.asarray(resume_checkpoint.beta_state, dtype=np.float64).copy()
            objective_history = [float(value) for value in resume_checkpoint.objective_history]
            validation_history = [float(value) for value in resume_checkpoint.validation_history]
            previous_alpha = _copy_optional(resume_checkpoint.previous_alpha)
            previous_beta = _copy_optional(resume_checkpoint.previous_beta)
            previous_local_scale = _copy_optional(resume_checkpoint.previous_local_scale)
            previous_theta = _copy_optional(resume_checkpoint.previous_theta)
            previous_tpb_shape_a_vector = _copy_optional(resume_checkpoint.previous_tpb_shape_a_vector)
            previous_tpb_shape_b_vector = _copy_optional(resume_checkpoint.previous_tpb_shape_b_vector)
            best_validation_metric = None if resume_checkpoint.best_validation_metric is None else float(resume_checkpoint.best_validation_metric)
            best_alpha = _copy_optional(resume_checkpoint.best_alpha)
            best_beta = _copy_optional(resume_checkpoint.best_beta)
            best_local_scale = _copy_optional(resume_checkpoint.best_local_scale)
            best_theta = _copy_optional(resume_checkpoint.best_theta)
            best_sigma_error2 = None if resume_checkpoint.best_sigma_error2 is None else float(resume_checkpoint.best_sigma_error2)
            best_tpb_shape_a_vector = _copy_optional(resume_checkpoint.best_tpb_shape_a_vector)
            best_tpb_shape_b_vector = _copy_optional(resume_checkpoint.best_tpb_shape_b_vector)
            start_iteration = int(resume_checkpoint.completed_iterations)
            log(
                "  variational EM: resuming from checkpoint "
                + f"after {start_iteration}/{config.max_outer_iterations} iterations  mem={mem()}"
            )

    log(f"  variational EM: {genotype_matrix.shape[1]} reduced variants, {covariate_matrix.shape[1]} covariates, {target_vector.shape[0]} samples, max_iter={config.max_outer_iterations}")
    use_stochastic_updates = _should_use_stochastic_variational_updates(genotype_matrix, config)
    if use_stochastic_updates:
        block_size = min(int(config.stochastic_variant_batch_size), int(genotype_matrix.shape[1]))
        block_count = max((int(genotype_matrix.shape[1]) + block_size - 1) // block_size, 1)
        step_index = start_iteration * block_count
        empty_covariates = np.zeros((target_vector.shape[0], 0), dtype=np.float64)
        metadata_baseline_scales = _metadata_baseline_scales_from_coefficients(
            scale_model_coefficients,
            prior_design.design_matrix,
            config,
        )
        baseline_reduced_prior_variances = (float(global_scale) * metadata_baseline_scales) ** 2
        reduced_prior_variances = _effective_prior_variances(
            baseline_prior_variances=baseline_reduced_prior_variances,
            local_scale=local_scale,
            config=config,
        )
        reduced_second_moment = np.maximum(
            np.asarray(beta_state * beta_state, dtype=np.float64),
            np.asarray(reduced_prior_variances, dtype=np.float64),
        )
        log(
            "  variational inference mode: stochastic variant-block updates "
            + f"(block_size={block_size}, blocks_per_epoch={block_count})"
        )
        for outer_iteration in range(start_iteration, config.max_outer_iterations):
            log(f"  variational EM epoch {outer_iteration + 1}/{config.max_outer_iterations} start  sigma_e2={sigma_error2:.6f}  global_scale={global_scale:.6f}  mem={mem()}")
            posterior_theta = _pack_theta(
                global_scale=float(global_scale),
                scale_model_coefficients=scale_model_coefficients,
            )
            epoch_rng = np.random.default_rng(config.random_seed + outer_iteration)
            local_shape_a = prior_design.class_membership_matrix @ tpb_shape_a_vector
            local_shape_b = prior_design.class_membership_matrix @ tpb_shape_b_vector
            metadata_baseline_scales = _metadata_baseline_scales_from_coefficients(
                scale_model_coefficients,
                prior_design.design_matrix,
                config,
            )
            baseline_reduced_prior_variances = (float(global_scale) * metadata_baseline_scales) ** 2
            reduced_prior_variances = _effective_prior_variances(
                baseline_prior_variances=baseline_reduced_prior_variances,
                local_scale=local_scale,
                config=config,
            )
            genetic_linear_predictor = np.array(
                genotype_matrix.matvec(beta_state, batch_size=config.posterior_variance_batch_size),
                dtype=np.float64,
                copy=True,
            )
            if config.trait_type == TraitType.BINARY:
                alpha_state = _fit_binary_alpha_with_offset(
                    covariate_matrix=covariate_matrix,
                    targets=target_vector,
                    predictor_offset=genetic_linear_predictor,
                    minimum_weight=config.polya_gamma_minimum_weight,
                    max_iterations=config.max_inner_newton_iterations,
                    gradient_tolerance=config.newton_gradient_tolerance,
                    alpha_init=alpha_state,
                )
                sigma_error2 = 1.0
            else:
                alpha_state = _initialize_alpha_state(
                    covariate_matrix=covariate_matrix,
                    targets=target_vector - genetic_linear_predictor,
                    trait_type=TraitType.QUANTITATIVE,
                )
            covariate_linear_predictor = np.asarray(covariate_matrix @ alpha_state, dtype=np.float64)

            for block_indices in _stochastic_variant_blocks(genotype_matrix.shape[1], block_size, epoch_rng):
                step_index += 1
                step_size = _stochastic_step_size(config, step_index)
                block_genotypes = genotype_matrix.subset(block_indices)
                block_prior_variances = np.asarray(reduced_prior_variances[block_indices], dtype=np.float64)
                block_beta_previous = np.asarray(beta_state[block_indices], dtype=np.float64).copy()
                block_linear_predictor_previous = np.asarray(
                    block_genotypes.matvec(block_beta_previous, batch_size=config.posterior_variance_batch_size),
                    dtype=np.float64,
                )
                predictor_offset = covariate_linear_predictor + genetic_linear_predictor - block_linear_predictor_previous
                if config.trait_type == TraitType.BINARY:
                    block_state = _binary_posterior_state(
                        genotype_matrix=block_genotypes,
                        covariate_matrix=empty_covariates,
                        targets=target_vector,
                        prior_variances=block_prior_variances,
                        alpha_init=np.zeros(0, dtype=np.float64),
                        beta_init=block_beta_previous,
                        minimum_weight=config.polya_gamma_minimum_weight,
                        max_iterations=config.max_inner_newton_iterations,
                        gradient_tolerance=config.newton_gradient_tolerance,
                        initial_damping=config.trust_region_initial_damping,
                        damping_increase_factor=config.trust_region_damping_increase_factor,
                        damping_decrease_factor=config.trust_region_damping_decrease_factor,
                        success_threshold=config.trust_region_success_threshold,
                        minimum_damping=config.trust_region_minimum_damping,
                        solver_tolerance=config.linear_solver_tolerance,
                        maximum_linear_solver_iterations=config.maximum_linear_solver_iterations,
                        logdet_probe_count=config.logdet_probe_count,
                        logdet_lanczos_steps=config.logdet_lanczos_steps,
                        exact_solver_matrix_limit=config.exact_solver_matrix_limit,
                        posterior_variance_batch_size=config.posterior_variance_batch_size,
                        posterior_variance_probe_count=config.posterior_variance_probe_count,
                        random_seed=config.random_seed + step_index,
                        compute_logdet=False,
                        compute_beta_variance=True,
                        sample_space_preconditioner_rank=config.sample_space_preconditioner_rank,
                        predictor_offset=predictor_offset,
                    )
                    block_beta_candidate = np.asarray(block_state[1], dtype=np.float64)
                    block_beta_variance = np.asarray(block_state[2], dtype=np.float64)
                else:
                    block_state = _fit_collapsed_posterior(
                        genotype_matrix=block_genotypes,
                        covariate_matrix=empty_covariates,
                        targets=target_vector - predictor_offset,
                        reduced_prior_variances=block_prior_variances,
                        sigma_error2=sigma_error2,
                        alpha_init=np.zeros(0, dtype=np.float64),
                        beta_init=block_beta_previous,
                        trait_type=TraitType.QUANTITATIVE,
                        config=config,
                        compute_logdet=False,
                        compute_beta_variance=True,
                    )
                    block_beta_candidate = np.asarray(block_state.beta, dtype=np.float64)
                    block_beta_variance = np.asarray(block_state.beta_variance, dtype=np.float64)
                block_beta_updated = block_beta_previous + step_size * (block_beta_candidate - block_beta_previous)
                beta_delta = block_beta_updated - block_beta_previous
                if np.any(beta_delta):
                    genetic_linear_predictor += np.asarray(
                        block_genotypes.matvec(beta_delta, batch_size=config.posterior_variance_batch_size),
                        dtype=np.float64,
                    )
                    beta_state[block_indices] = block_beta_updated
                block_second_moment = np.asarray(
                    block_beta_candidate * block_beta_candidate + block_beta_variance,
                    dtype=np.float64,
                )
                reduced_second_moment[block_indices] = (
                    (1.0 - step_size) * reduced_second_moment[block_indices]
                    + step_size * block_second_moment
                )
                updated_local_scale_block, updated_auxiliary_delta_block = _update_local_scales(
                    coefficient_second_moment=reduced_second_moment[block_indices],
                    baseline_prior_variances=baseline_reduced_prior_variances[block_indices],
                    local_shape_a=local_shape_a[block_indices],
                    local_shape_b=local_shape_b[block_indices],
                    auxiliary_delta=auxiliary_delta[block_indices],
                    config=config,
                )
                local_scale[block_indices] = (
                    (1.0 - step_size) * local_scale[block_indices]
                    + step_size * updated_local_scale_block
                )
                auxiliary_delta[block_indices] = (
                    (1.0 - step_size) * auxiliary_delta[block_indices]
                    + step_size * updated_auxiliary_delta_block
                )

            if config.trait_type == TraitType.BINARY:
                alpha_state = _fit_binary_alpha_with_offset(
                    covariate_matrix=covariate_matrix,
                    targets=target_vector,
                    predictor_offset=genetic_linear_predictor,
                    minimum_weight=config.polya_gamma_minimum_weight,
                    max_iterations=config.max_inner_newton_iterations,
                    gradient_tolerance=config.newton_gradient_tolerance,
                    alpha_init=alpha_state,
                )
                sigma_error2 = 1.0
            else:
                alpha_state = _initialize_alpha_state(
                    covariate_matrix=covariate_matrix,
                    targets=target_vector - genetic_linear_predictor,
                    trait_type=TraitType.QUANTITATIVE,
                )
            should_update_hyperparameters = (
                config.update_hyperparameters
                and (outer_iteration + 1 >= 4)
                and ((outer_iteration + 1) % 4 == 0 or outer_iteration + 1 == config.max_outer_iterations)
            )
            if should_update_hyperparameters:
                global_scale, scale_model_coefficients = _update_scale_model(
                    reduced_second_moment=reduced_second_moment,
                    local_scale=local_scale,
                    prior_design=prior_design,
                    scale_penalty=scale_penalty,
                    current_global_scale=float(global_scale),
                    current_scale_model_coefficients=scale_model_coefficients,
                    config=config,
                )
                tpb_shape_a_vector, tpb_shape_b_vector = _update_tpb_shape_vectors(
                    class_membership_matrix=prior_design.class_membership_matrix,
                    current_shape_a_vector=tpb_shape_a_vector,
                    current_shape_b_vector=tpb_shape_b_vector,
                    local_scale=local_scale,
                    auxiliary_delta=auxiliary_delta,
                    config=config,
                )
                local_shape_a = prior_design.class_membership_matrix @ tpb_shape_a_vector
                local_shape_b = prior_design.class_membership_matrix @ tpb_shape_b_vector
            auxiliary_delta = (local_shape_a + local_shape_b) / np.maximum(1.0 + local_scale, config.local_scale_floor)
            reduced_prior_variances = _effective_prior_variances(
                baseline_prior_variances=(float(global_scale) * _metadata_baseline_scales_from_coefficients(
                    scale_model_coefficients,
                    prior_design.design_matrix,
                    config,
                )) ** 2,
                local_scale=local_scale,
                config=config,
            )
            covariate_linear_predictor = np.asarray(covariate_matrix @ alpha_state, dtype=np.float64)
            linear_predictor = covariate_linear_predictor + genetic_linear_predictor
            beta_variance_state = np.maximum(reduced_second_moment - beta_state * beta_state, 1e-8)
            if config.trait_type == TraitType.QUANTITATIVE:
                leverage_weight = np.maximum(reduced_prior_variances - beta_variance_state, 0.0) / np.maximum(reduced_prior_variances, 1e-12)
                residual_vector = np.asarray(target_vector - linear_predictor, dtype=np.float64)
                effective_dof = max(float(target_vector.shape[0]) - float(np.sum(leverage_weight)), 1.0)
                sigma_error2 = max(float(np.dot(residual_vector, residual_vector)) / effective_dof, config.sigma_error_floor)
            objective_history.append(
                _stochastic_epoch_objective(
                    trait_type=config.trait_type,
                    targets=target_vector,
                    linear_predictor=linear_predictor,
                    beta=beta_state,
                    reduced_prior_variances=reduced_prior_variances,
                    local_scale=local_scale,
                    auxiliary_delta=auxiliary_delta,
                    local_shape_a=local_shape_a,
                    local_shape_b=local_shape_b,
                    scale_model_coefficients=scale_model_coefficients,
                    scale_penalty=scale_penalty,
                )
            )
            if validation_payload is not None:
                should_validate = (
                    outer_iteration == 0
                    or ((outer_iteration + 1) % config.validation_interval == 0)
                    or outer_iteration + 1 == config.max_outer_iterations
                )
                if should_validate:
                    validation_metric = _validation_metric(
                        trait_type=config.trait_type,
                        genotype_matrix=validation_payload[0],
                        covariate_matrix=validation_payload[1],
                        targets=validation_payload[2],
                        alpha=alpha_state,
                        beta=beta_state,
                    )
                    validation_history.append(validation_metric)
                    if best_validation_metric is None or validation_metric < best_validation_metric:
                        best_validation_metric = validation_metric
                        best_alpha = alpha_state.copy()
                        best_beta = beta_state.copy()
                        best_local_scale = local_scale.copy()
                        best_theta = _pack_theta(global_scale, scale_model_coefficients)
                        best_sigma_error2 = float(sigma_error2)
                        best_tpb_shape_a_vector = tpb_shape_a_vector.copy()
                        best_tpb_shape_b_vector = tpb_shape_b_vector.copy()
                    log(f"  variational EM epoch {outer_iteration + 1}: validation_metric={validation_metric:.6f}")
            parameter_change = _relative_parameter_change(
                current_beta=beta_state,
                previous_beta=previous_beta,
                current_alpha=alpha_state,
                previous_alpha=previous_alpha,
                current_local_scale=local_scale,
                previous_local_scale=previous_local_scale,
                current_theta=_pack_theta(global_scale, scale_model_coefficients),
                previous_theta=previous_theta,
                current_tpb_shape_a_vector=tpb_shape_a_vector,
                previous_tpb_shape_a_vector=previous_tpb_shape_a_vector,
                current_tpb_shape_b_vector=tpb_shape_b_vector,
                previous_tpb_shape_b_vector=previous_tpb_shape_b_vector,
            )
            previous_alpha = alpha_state.copy()
            previous_beta = beta_state.copy()
            previous_local_scale = local_scale.copy()
            previous_theta = _pack_theta(global_scale, scale_model_coefficients)
            previous_tpb_shape_a_vector = tpb_shape_a_vector.copy()
            previous_tpb_shape_b_vector = tpb_shape_b_vector.copy()
            iter_num = outer_iteration + 1
            obj_str = f"{objective_history[-1]:.6f}" if objective_history else "N/A"
            val_str = f"  val={validation_history[-1]:.6f}" if validation_history else ""
            hyper_str = "  [+hyper]" if should_update_hyperparameters else ""
            nonzero_beta = int(np.count_nonzero(np.abs(beta_state) > 1e-8))
            log(f"  SVI epoch {iter_num}/{config.max_outer_iterations}  obj={obj_str}  delta={parameter_change:.2e}  sigma_e2={sigma_error2:.4f}  g_scale={float(global_scale):.4f}  nz_beta={nonzero_beta}{val_str}{hyper_str}  mem={mem()}")
            if checkpoint_callback is not None:
                checkpoint_callback(_build_checkpoint(iter_num))
            if validation_history:
                if len(validation_history) >= 2:
                    validation_delta = abs(validation_history[-1] - validation_history[-2])
                    if parameter_change < config.convergence_tolerance and validation_delta < config.convergence_tolerance:
                        log(f"  stochastic variational updates converged on epoch {outer_iteration + 1} with parameter_change={parameter_change:.3e} validation_delta={validation_delta:.3e}")
                        break
            elif parameter_change < config.convergence_tolerance:
                log(f"  stochastic variational updates converged on epoch {outer_iteration + 1} with parameter_change={parameter_change:.3e}")
                break
    else:
        for outer_iteration in range(start_iteration, config.max_outer_iterations):
            log(f"  variational EM iteration {outer_iteration + 1}/{config.max_outer_iterations} start  sigma_e2={sigma_error2:.6f}  global_scale={global_scale:.6f}  mem={mem()}")
            posterior_theta = _pack_theta(
                global_scale=float(global_scale),
                scale_model_coefficients=scale_model_coefficients,
            )
            posterior_local_scale = local_scale.copy()
            posterior_tpb_shape_a_vector = tpb_shape_a_vector.copy()
            posterior_tpb_shape_b_vector = tpb_shape_b_vector.copy()
            metadata_baseline_scales = _metadata_baseline_scales_from_coefficients(
                scale_model_coefficients,
                prior_design.design_matrix,
                config,
            )
            baseline_reduced_prior_variances = (float(global_scale) * metadata_baseline_scales) ** 2
            reduced_prior_variances = _effective_prior_variances(
                baseline_prior_variances=baseline_reduced_prior_variances,
                local_scale=local_scale,
                config=config,
            )

            posterior_state = _fit_collapsed_posterior(
                genotype_matrix=genotype_matrix,
                covariate_matrix=covariate_matrix,
                targets=target_vector,
                reduced_prior_variances=reduced_prior_variances,
                sigma_error2=sigma_error2,
                alpha_init=alpha_state,
                beta_init=beta_state,
                trait_type=config.trait_type,
                config=config,
                compute_logdet=False,
                compute_beta_variance=True,
            )
            if config.trait_type == TraitType.BINARY and config.binary_intercept_calibration:
                posterior_state = _apply_binary_intercept_calibration(
                    posterior_state=posterior_state,
                    targets=target_vector,
                )
            alpha_state = posterior_state.alpha
            beta_state = posterior_state.beta
            sigma_error2 = posterior_state.sigma_error2

            reduced_second_moment = np.asarray(beta_state * beta_state + posterior_state.beta_variance, dtype=np.float64)
            full_objective = posterior_state.collapsed_objective + _local_scale_prior_objective(
                local_scale=local_scale,
                auxiliary_delta=auxiliary_delta,
                local_shape_a=local_shape_a,
                local_shape_b=local_shape_b,
            ) + _scale_penalty_objective(
                scale_model_coefficients=scale_model_coefficients,
                scale_penalty=scale_penalty,
            )
            objective_history.append(float(full_objective))
            log(f"  variational EM iteration {outer_iteration + 1}: objective={full_objective:.6f} sigma_e2={sigma_error2:.6f}")

            if validation_payload is not None:
                should_validate = (
                    outer_iteration == 0
                    or ((outer_iteration + 1) % config.validation_interval == 0)
                    or outer_iteration + 1 == config.max_outer_iterations
                )
                if should_validate:
                    validation_metric = _validation_metric(
                        trait_type=config.trait_type,
                        genotype_matrix=validation_payload[0],
                        covariate_matrix=validation_payload[1],
                        targets=validation_payload[2],
                        alpha=alpha_state,
                        beta=beta_state,
                    )
                    validation_history.append(validation_metric)
                    if best_validation_metric is None or validation_metric < best_validation_metric:
                        best_validation_metric = validation_metric
                        best_alpha = alpha_state.copy()
                        best_beta = beta_state.copy()
                        best_local_scale = posterior_local_scale.copy()
                        best_theta = posterior_theta.copy()
                        best_sigma_error2 = float(sigma_error2)
                        best_tpb_shape_a_vector = posterior_tpb_shape_a_vector.copy()
                        best_tpb_shape_b_vector = posterior_tpb_shape_b_vector.copy()
                    log(f"  variational EM iteration {outer_iteration + 1}: validation_metric={validation_metric:.6f}")

            updated_local_scale, updated_auxiliary_delta = _update_local_scales(
                coefficient_second_moment=reduced_second_moment,
                baseline_prior_variances=baseline_reduced_prior_variances,
                local_shape_a=local_shape_a,
                local_shape_b=local_shape_b,
                auxiliary_delta=auxiliary_delta,
                config=config,
            )
            should_update_hyperparameters = (
                config.update_hyperparameters
                and (outer_iteration + 1 >= 4)
                and ((outer_iteration + 1) % 4 == 0 or outer_iteration + 1 == config.max_outer_iterations)
            )
            if should_update_hyperparameters:
                global_scale, scale_model_coefficients = _update_scale_model(
                    reduced_second_moment=reduced_second_moment,
                    local_scale=updated_local_scale,
                    prior_design=prior_design,
                    scale_penalty=scale_penalty,
                    current_global_scale=float(global_scale),
                    current_scale_model_coefficients=scale_model_coefficients,
                    config=config,
                )
                tpb_shape_a_vector, tpb_shape_b_vector = _update_tpb_shape_vectors(
                    class_membership_matrix=prior_design.class_membership_matrix,
                    current_shape_a_vector=tpb_shape_a_vector,
                    current_shape_b_vector=tpb_shape_b_vector,
                    local_scale=updated_local_scale,
                    auxiliary_delta=updated_auxiliary_delta,
                    config=config,
                )
                local_shape_a = prior_design.class_membership_matrix @ tpb_shape_a_vector
                local_shape_b = prior_design.class_membership_matrix @ tpb_shape_b_vector
            local_scale = updated_local_scale
            auxiliary_delta = (local_shape_a + local_shape_b) / np.maximum(1.0 + local_scale, config.local_scale_floor)

            parameter_change = _relative_parameter_change(
                current_beta=beta_state,
                previous_beta=previous_beta,
                current_alpha=alpha_state,
                previous_alpha=previous_alpha,
                current_local_scale=local_scale,
                previous_local_scale=previous_local_scale,
                current_theta=_pack_theta(global_scale, scale_model_coefficients),
                previous_theta=previous_theta,
                current_tpb_shape_a_vector=tpb_shape_a_vector,
                previous_tpb_shape_a_vector=previous_tpb_shape_a_vector,
                current_tpb_shape_b_vector=tpb_shape_b_vector,
                previous_tpb_shape_b_vector=previous_tpb_shape_b_vector,
            )
            previous_alpha = alpha_state.copy()
            previous_beta = beta_state.copy()
            previous_local_scale = local_scale.copy()
            previous_theta = _pack_theta(global_scale, scale_model_coefficients)
            previous_tpb_shape_a_vector = tpb_shape_a_vector.copy()
            previous_tpb_shape_b_vector = tpb_shape_b_vector.copy()

            iter_num = outer_iteration + 1
            obj_str = f"{objective_history[-1]:.6f}" if objective_history else "N/A"
            val_str = f"  val={validation_history[-1]:.6f}" if validation_history else ""
            hyper_str = "  [+hyper]" if should_update_hyperparameters else ""
            nonzero_beta = int(np.count_nonzero(np.abs(beta_state) > 1e-8))
            log(f"  EM iter {iter_num}/{config.max_outer_iterations}  obj={obj_str}  delta={parameter_change:.2e}  sigma_e2={sigma_error2:.4f}  g_scale={float(global_scale):.4f}  nz_beta={nonzero_beta}{val_str}{hyper_str}  mem={mem()}")
            if checkpoint_callback is not None:
                checkpoint_callback(_build_checkpoint(iter_num))

            if validation_history:
                if len(validation_history) >= 2:
                    validation_delta = abs(validation_history[-1] - validation_history[-2])
                    if parameter_change < config.convergence_tolerance and validation_delta < config.convergence_tolerance:
                        log(f"  variational EM converged on iteration {outer_iteration + 1} with parameter_change={parameter_change:.3e} validation_delta={validation_delta:.3e}")
                        break
            elif parameter_change < config.convergence_tolerance:
                log(f"  variational EM converged on iteration {outer_iteration + 1} with parameter_change={parameter_change:.3e}")
                break

    if best_validation_metric is not None:
        if (
            best_alpha is None
            or best_beta is None
            or best_local_scale is None
            or best_theta is None
            or best_sigma_error2 is None
            or best_tpb_shape_a_vector is None
            or best_tpb_shape_b_vector is None
        ):
            raise RuntimeError("best validation snapshot is incomplete")
        alpha_state = best_alpha
        beta_state = best_beta
        local_scale = best_local_scale
        sigma_error2 = best_sigma_error2
        global_scale, scale_model_coefficients = _unpack_theta(best_theta)
        tpb_shape_a_vector = best_tpb_shape_a_vector
        tpb_shape_b_vector = best_tpb_shape_b_vector

    log(f"  EM loop done after {len(objective_history)} iterations, computing final posterior...  mem={mem()}")
    final_metadata_baseline_scales = _metadata_baseline_scales_from_coefficients(
        scale_model_coefficients,
        prior_design.design_matrix,
        config,
    )
    final_baseline_reduced_prior_variances = (float(global_scale) * final_metadata_baseline_scales) ** 2
    final_reduced_prior_variances = _effective_prior_variances(
        baseline_prior_variances=final_baseline_reduced_prior_variances,
        local_scale=local_scale,
        config=config,
    )
    final_state = _fit_collapsed_posterior(
        genotype_matrix=genotype_matrix,
        covariate_matrix=covariate_matrix,
        targets=target_vector,
        reduced_prior_variances=final_reduced_prior_variances,
        sigma_error2=sigma_error2,
        alpha_init=alpha_state,
        beta_init=beta_state,
        trait_type=config.trait_type,
        config=config,
        compute_logdet=True,
        compute_beta_variance=True,
    )
    if config.trait_type == TraitType.BINARY and config.binary_intercept_calibration:
        final_state = _apply_binary_intercept_calibration(
            posterior_state=final_state,
            targets=target_vector,
        )

    log(f"  final posterior computed  obj={final_state.collapsed_objective:.6f}  sigma_e2={final_state.sigma_error2:.4f}  mem={mem()}")
    final_member_prior_variances = _member_prior_variances_from_reduced_state(
        member_records=member_records,
        tie_map=tie_map,
        scale_model_coefficients=scale_model_coefficients,
        scale_model_feature_specs=prior_design.feature_specs,
        global_scale=float(global_scale),
        local_scale=local_scale,
        config=config,
    )
    local_shape_a = prior_design.class_membership_matrix @ tpb_shape_a_vector
    local_shape_b = prior_design.class_membership_matrix @ tpb_shape_b_vector
    log(f"  variational EM returning results  mem={mem()}")
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


def _prepare_validation(
    validation_data: tuple[StandardizedGenotypeMatrix | np.ndarray, np.ndarray, np.ndarray] | None,
) -> tuple[StandardizedGenotypeMatrix | np.ndarray, np.ndarray, np.ndarray] | None:
    if validation_data is None:
        return None
    validation_genotypes, validation_covariates, validation_targets = validation_data
    return (
        validation_genotypes if isinstance(validation_genotypes, StandardizedGenotypeMatrix) else np.asarray(validation_genotypes, dtype=np.float64),
        np.asarray(validation_covariates, dtype=np.float64),
        np.asarray(validation_targets, dtype=np.float64),
    )


def _as_standardized_genotype_matrix(
    genotypes: StandardizedGenotypeMatrix | np.ndarray,
) -> StandardizedGenotypeMatrix:
    if isinstance(genotypes, StandardizedGenotypeMatrix):
        return genotypes
    dense_raw = DenseRawGenotypeMatrix(np.asarray(genotypes, dtype=np.float32))
    return dense_raw.standardized(
        means=np.zeros(dense_raw.shape[1], dtype=np.float32),
        scales=np.ones(dense_raw.shape[1], dtype=np.float32),
    )


def _fit_collapsed_posterior(
    genotype_matrix: StandardizedGenotypeMatrix,
    covariate_matrix: np.ndarray,
    targets: np.ndarray,
    reduced_prior_variances: np.ndarray,
    sigma_error2: float,
    alpha_init: np.ndarray,
    beta_init: np.ndarray,
    trait_type: TraitType,
    config: ModelConfig,
    compute_logdet: bool = True,
    compute_beta_variance: bool = True,
) -> PosteriorState:
    log(f"    collapsed posterior: trait={trait_type.value}  n_variants={genotype_matrix.shape[1]}  n_samples={genotype_matrix.shape[0]}  sigma_e2={sigma_error2:.6f}  mem={mem()}")
    prior_variances = np.maximum(np.asarray(reduced_prior_variances, dtype=np.float64), 1e-8)
    if trait_type == TraitType.QUANTITATIVE:
        alpha, beta, beta_variance, linear_predictor, collapsed_objective, sigma_error2_new = _quantitative_posterior_state(
            genotype_matrix=genotype_matrix,
            covariate_matrix=covariate_matrix,
            targets=targets,
            prior_variances=prior_variances,
            sigma_error2=max(float(sigma_error2), config.sigma_error_floor),
            sigma_error_floor=config.sigma_error_floor,
            solver_tolerance=config.linear_solver_tolerance,
            maximum_linear_solver_iterations=config.maximum_linear_solver_iterations,
            logdet_probe_count=config.logdet_probe_count,
            logdet_lanczos_steps=config.logdet_lanczos_steps,
            exact_solver_matrix_limit=config.exact_solver_matrix_limit,
            posterior_variance_batch_size=config.posterior_variance_batch_size,
            posterior_variance_probe_count=config.posterior_variance_probe_count,
            random_seed=config.random_seed,
            compute_logdet=compute_logdet,
            compute_beta_variance=compute_beta_variance,
            sample_space_preconditioner_rank=config.sample_space_preconditioner_rank,
        )
    else:
        alpha, beta, beta_variance, linear_predictor, collapsed_objective = _binary_posterior_state(
            genotype_matrix=genotype_matrix,
            covariate_matrix=covariate_matrix,
            targets=targets,
            prior_variances=prior_variances,
            alpha_init=np.asarray(alpha_init, dtype=np.float64),
            beta_init=np.asarray(beta_init, dtype=np.float64),
            minimum_weight=config.polya_gamma_minimum_weight,
            max_iterations=config.max_inner_newton_iterations,
            gradient_tolerance=config.newton_gradient_tolerance,
            initial_damping=config.trust_region_initial_damping,
            damping_increase_factor=config.trust_region_damping_increase_factor,
            damping_decrease_factor=config.trust_region_damping_decrease_factor,
            success_threshold=config.trust_region_success_threshold,
            minimum_damping=config.trust_region_minimum_damping,
            solver_tolerance=config.linear_solver_tolerance,
            maximum_linear_solver_iterations=config.maximum_linear_solver_iterations,
            logdet_probe_count=config.logdet_probe_count,
            logdet_lanczos_steps=config.logdet_lanczos_steps,
            exact_solver_matrix_limit=config.exact_solver_matrix_limit,
            posterior_variance_batch_size=config.posterior_variance_batch_size,
            posterior_variance_probe_count=config.posterior_variance_probe_count,
            random_seed=config.random_seed,
            compute_logdet=compute_logdet,
            compute_beta_variance=compute_beta_variance,
            sample_space_preconditioner_rank=config.sample_space_preconditioner_rank,
        )
        sigma_error2_new = 1.0
    return PosteriorState(
        alpha=np.asarray(alpha, dtype=np.float64),
        beta=np.asarray(beta, dtype=np.float64),
        beta_variance=np.asarray(beta_variance, dtype=np.float64),
        linear_predictor=np.asarray(linear_predictor, dtype=np.float64),
        collapsed_objective=float(collapsed_objective),
        sigma_error2=float(sigma_error2_new),
    )


# Fit effect sizes for a continuous trait (e.g. blood pressure, BMI).
#
# Strategy: build a covariance matrix that accounts for both random noise
# (sigma_e^2) and the genotype-explained signal (via prior variances tau^2).
# Then solve a weighted least-squares problem to get alpha (covariate effects)
# and beta (variant effects).  Also re-estimate the noise variance sigma_e^2
# by comparing predictions to actual values, correcting for model complexity
# (leverage — variants the model is very confident about get less weight in
# the noise estimate, to avoid double-counting).
def _quantitative_posterior_state(
    genotype_matrix: StandardizedGenotypeMatrix | np.ndarray,
    covariate_matrix: np.ndarray,
    targets: np.ndarray,
    prior_variances: np.ndarray,
    sigma_error2: float,
    sigma_error_floor: float,
    solver_tolerance: float = 1e-6,
    maximum_linear_solver_iterations: int = 256,
    logdet_probe_count: int = 6,
    logdet_lanczos_steps: int = 12,
    exact_solver_matrix_limit: int = 512,
    posterior_variance_batch_size: int = 64,
    posterior_variance_probe_count: int = 12,
    random_seed: int = 0,
    compute_logdet: bool = True,
    compute_beta_variance: bool = True,
    sample_space_preconditioner_rank: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    standardized_genotypes = _as_standardized_genotype_matrix(genotype_matrix)
    sample_count = standardized_genotypes.shape[0]
    alpha, beta, beta_variance, _projected_targets, linear_predictor, restricted_quadratic, logdet_covariance, logdet_gls = (
        _restricted_posterior_state(
            genotype_matrix=standardized_genotypes,
            covariate_matrix=covariate_matrix,
            targets=targets,
            prior_variances=prior_variances,
            diagonal_noise=np.full(sample_count, sigma_error2, dtype=np.float64),
            solver_tolerance=solver_tolerance,
            maximum_linear_solver_iterations=maximum_linear_solver_iterations,
            logdet_probe_count=logdet_probe_count,
            logdet_lanczos_steps=logdet_lanczos_steps,
            exact_solver_matrix_limit=exact_solver_matrix_limit,
            posterior_variance_batch_size=posterior_variance_batch_size,
            posterior_variance_probe_count=posterior_variance_probe_count,
            random_seed=random_seed,
            compute_logdet=compute_logdet,
            compute_beta_variance=compute_beta_variance,
            sample_space_preconditioner_rank=sample_space_preconditioner_rank,
        )
    )
    # Re-estimate noise variance.  Naive approach (just use residuals) would
    # underestimate noise because the model "overfits" a little to noise.
    # Leverage correction accounts for this: variants the model is very
    # confident about (low variance relative to prior) contribute to the
    # correction term, preventing the noise estimate from shrinking too fast.
    leverage_weight = np.maximum(prior_variances - beta_variance, 0.0) / np.maximum(prior_variances, 1e-12)
    residual_vector = targets - linear_predictor
    residual_sum_squares = float(np.dot(residual_vector, residual_vector))
    posterior_fit_uncertainty = sigma_error2 * float(np.sum(leverage_weight))
    sigma_error2_new = max((residual_sum_squares + posterior_fit_uncertainty) / sample_count, sigma_error_floor)
    # Restricted log-likelihood: measures how well the model explains the data
    # after accounting for model complexity (via log-determinant terms).
    # Higher (less negative) = better fit with appropriate complexity.
    collapsed_objective = -0.5 * (
        restricted_quadratic
        + (logdet_covariance + logdet_gls if compute_logdet else 0.0)
        + (sample_count - covariate_matrix.shape[1]) * np.log(2.0 * np.pi)
    )
    return alpha, beta, beta_variance, linear_predictor, collapsed_objective, sigma_error2_new


def _binary_newton_solver_controls(
    genotype_matrix: StandardizedGenotypeMatrix,
    *,
    solver_tolerance: float,
    maximum_linear_solver_iterations: int,
) -> tuple[float, int]:
    if genotype_matrix.shape[0] < 16_384 and genotype_matrix.shape[1] < 16_384:
        return solver_tolerance, maximum_linear_solver_iterations
    relaxed_tolerance = max(float(solver_tolerance), 1e-3)
    relaxed_maximum_iterations = min(int(maximum_linear_solver_iterations), 128)
    if genotype_matrix.shape[1] > genotype_matrix.shape[0]:
        relaxed_tolerance = max(relaxed_tolerance, 5e-3)
        relaxed_maximum_iterations = min(relaxed_maximum_iterations, 96)
    return relaxed_tolerance, max(relaxed_maximum_iterations, 16)


# Fit effect sizes for a binary trait (e.g. disease case/control).
#
# Binary traits can't use the simple least-squares approach above because
# the outcome is 0/1, not continuous.  Instead we use logistic regression
# with a Bayesian twist:
#   - Convert the linear predictor to a probability via the sigmoid function
#   - Use Newton's method with a "trust region" to find the best betas
#   - The trust region (controlled by a damping parameter) prevents the
#     optimizer from taking steps that are too large and overshooting
#   - At each Newton step, we build a local quadratic approximation
#     (iteratively reweighted least squares / IRLS) and solve it using
#     the same linear algebra as the quantitative case
#
# The final objective is a Laplace approximation: the log-posterior at
# its peak, corrected for the curvature (how "peaked" the posterior is).
def _binary_posterior_state(
    genotype_matrix: StandardizedGenotypeMatrix | np.ndarray,
    covariate_matrix: np.ndarray,
    targets: np.ndarray,
    prior_variances: np.ndarray,
    alpha_init: np.ndarray,
    beta_init: np.ndarray,
    minimum_weight: float,
    max_iterations: int,
    gradient_tolerance: float,
    initial_damping: float,
    damping_increase_factor: float,
    damping_decrease_factor: float,
    success_threshold: float,
    minimum_damping: float,
    solver_tolerance: float = 1e-6,
    maximum_linear_solver_iterations: int = 256,
    logdet_probe_count: int = 6,
    logdet_lanczos_steps: int = 12,
    exact_solver_matrix_limit: int = 512,
    posterior_variance_batch_size: int = 64,
    posterior_variance_probe_count: int = 12,
    random_seed: int = 0,
    compute_logdet: bool = True,
    compute_beta_variance: bool = True,
    sample_space_preconditioner_rank: int = 256,
    predictor_offset: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    standardized_genotypes = _as_standardized_genotype_matrix(genotype_matrix)
    compute_np_dtype = gpu_compute_numpy_dtype()
    compute_jax_dtype = gpu_compute_jax_dtype()
    prior_precision = (1.0 / np.maximum(prior_variances, 1e-8)).astype(compute_np_dtype, copy=False)
    covariate_count = covariate_matrix.shape[1]
    parameters = np.concatenate([alpha_init, beta_init], axis=0).astype(compute_np_dtype, copy=True)
    damping = float(initial_damping)
    predictor_offset_array = (
        np.zeros(standardized_genotypes.shape[0], dtype=compute_np_dtype)
        if predictor_offset is None
        else np.asarray(predictor_offset, dtype=compute_np_dtype).reshape(-1)
    )
    if predictor_offset_array.shape != (standardized_genotypes.shape[0],):
        raise ValueError("predictor_offset must match genotype sample count.")

    # Pre-convert the covariate matrix to JAX once so that matmuls stay on GPU
    # throughout the Newton loop.  The covariate matrix is small
    # (n_samples x ~5-10 covariates) so this is cheap.
    covariate_matrix_jax = jnp.asarray(covariate_matrix, dtype=compute_jax_dtype)
    targets_jax = jnp.asarray(targets, dtype=compute_jax_dtype)
    prior_precision_jax = jnp.asarray(prior_precision, dtype=compute_jax_dtype)
    predictor_offset_jax = jnp.asarray(predictor_offset_array, dtype=compute_jax_dtype)
    newton_solver_tolerance, newton_maximum_linear_solver_iterations = _binary_newton_solver_controls(
        standardized_genotypes,
        solver_tolerance=solver_tolerance,
        maximum_linear_solver_iterations=maximum_linear_solver_iterations,
    )

    penalized_term_calls = 0

    def penalized_terms(current_parameters: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute the penalized log-posterior, its gradient, IRLS weights, and predictions.

        The objective balances two things:
          - Data fit: how well do the predicted probabilities match actual 0/1 outcomes
          - Prior penalty: large betas are penalized (more so for small prior variance)
        """
        nonlocal penalized_term_calls
        penalized_term_calls += 1
        alpha = current_parameters[:covariate_count]
        beta = current_parameters[covariate_count:]
        if penalized_term_calls <= 2:
            log(
                "      binary penalized_terms diagnostics: "
                + f"call={penalized_term_calls} alpha_shape={alpha.shape} beta_shape={beta.shape} "
                + f"covariates_shape={covariate_matrix_jax.shape} targets_shape={targets_jax.shape} "
                + f"genotype_shape={standardized_genotypes.shape} gpu_cached={standardized_genotypes._cupy_cache is not None} "
            )
        # Compute linear predictor on GPU: covariate part via JAX matmul,
        # genotype part via StandardizedGenotypeMatrix.matvec (already JAX).
        alpha_jax = jnp.asarray(alpha, dtype=compute_jax_dtype)
        beta_jax = jnp.asarray(beta, dtype=compute_jax_dtype)
        linear_predictor_jax = (
            predictor_offset_jax
            + covariate_matrix_jax @ alpha_jax
            + standardized_genotypes.matvec(beta, batch_size=posterior_variance_batch_size)
        )
        probabilities_jax = stable_sigmoid(linear_predictor_jax)
        # IRLS weights: p*(1-p).  Large near p=0.5 (uncertain samples contribute
        # more to the update), small near 0 or 1 (confident predictions are stable).
        weights_jax = jnp.maximum(probabilities_jax * (1.0 - probabilities_jax), minimum_weight)
        residual_jax = targets_jax - probabilities_jax
        # Gradient computations stay on GPU: covariate part via JAX matmul,
        # genotype part via transpose_matvec (already JAX).
        gradient_alpha_jax = covariate_matrix_jax.T @ residual_jax
        gradient_beta_jax = (
            standardized_genotypes.transpose_matvec(residual_jax, batch_size=posterior_variance_batch_size)
            - prior_precision_jax * beta_jax
        )
        gradient = np.concatenate(
            [np.asarray(gradient_alpha_jax, dtype=compute_np_dtype),
             np.asarray(gradient_beta_jax, dtype=compute_np_dtype)],
            axis=0,
        )
        penalized_log_posterior = float(
            jnp.sum(targets_jax * jnp.log(probabilities_jax + 1e-12)
                     + (1.0 - targets_jax) * jnp.log(1.0 - probabilities_jax + 1e-12))
            - 0.5 * jnp.sum(prior_precision_jax * beta_jax * beta_jax)
        )
        # Convert outputs to numpy for the outer Newton loop
        linear_predictor = np.asarray(linear_predictor_jax, dtype=compute_np_dtype)
        weights = np.asarray(weights_jax, dtype=compute_np_dtype)
        probabilities = np.asarray(probabilities_jax, dtype=compute_np_dtype)
        return penalized_log_posterior, gradient, weights, linear_predictor, probabilities

    import time as _time
    stalled_objective_relative_tolerance = 1e-12
    cached_terms_parameters: np.ndarray | None = None
    cached_terms: tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
    cached_proposal_base_parameters: np.ndarray | None = None
    cached_proposed_parameters: np.ndarray | None = None
    cached_newton_direction: np.ndarray | None = None
    log(f"      Newton trust-region: {standardized_genotypes.shape[1]} variants, max_iter={max_iterations}, damping={damping:.2e}  mem={mem()}")
    for _iteration_index in range(max_iterations):
        _newton_t0 = _time.monotonic()
        if (
            cached_terms_parameters is not None
            and cached_terms is not None
            and np.array_equal(parameters, cached_terms_parameters)
        ):
            current_objective, gradient, weights, linear_predictor, probabilities = cached_terms
            _t_penalized = 0.0
        else:
            current_objective, gradient, weights, linear_predictor, probabilities = penalized_terms(parameters)
            cached_terms_parameters = parameters.copy()
            cached_terms = (current_objective, gradient, weights, linear_predictor, probabilities)
            _t_penalized = _time.monotonic() - _newton_t0
        gradient_norm = float(np.linalg.norm(gradient))
        if gradient_norm <= gradient_tolerance:
            log(f"      Newton converged at iter {_iteration_index+1}: grad_norm={gradient_norm:.2e} <= tol={gradient_tolerance:.2e}")
            break
        working_response = (
            linear_predictor
            + (targets.astype(compute_np_dtype, copy=False) - probabilities) / weights
            - predictor_offset_array
        )
        if (
            cached_proposal_base_parameters is not None
            and np.array_equal(parameters, cached_proposal_base_parameters)
            and cached_proposed_parameters is not None
            and cached_newton_direction is not None
        ):
            proposed_parameters = cached_proposed_parameters
            newton_direction = cached_newton_direction
            _t_solve = 0.0
        else:
            _t_solve_start = _time.monotonic()
            proposed_alpha, proposed_beta, _, _projected_targets, _fitted_response, _restricted_quadratic, _logdet_covariance, _logdet_gls = (
                _restricted_posterior_state(
                        genotype_matrix=standardized_genotypes,
                    covariate_matrix=covariate_matrix,
                    targets=working_response,
                    prior_variances=prior_variances,
                    diagonal_noise=1.0 / weights,
                    solver_tolerance=newton_solver_tolerance,
                    maximum_linear_solver_iterations=newton_maximum_linear_solver_iterations,
                    logdet_probe_count=logdet_probe_count,
                    logdet_lanczos_steps=logdet_lanczos_steps,
                    exact_solver_matrix_limit=exact_solver_matrix_limit,
                    posterior_variance_batch_size=posterior_variance_batch_size,
                    posterior_variance_probe_count=posterior_variance_probe_count,
                    random_seed=random_seed + _iteration_index,
                    compute_logdet=False,
                    compute_beta_variance=False,
                    initial_beta_guess=parameters[covariate_count:],
                    sample_space_preconditioner_rank=sample_space_preconditioner_rank,
                )
            )
            proposed_parameters = np.concatenate([proposed_alpha, proposed_beta], axis=0)
            newton_direction = proposed_parameters - parameters
            cached_proposal_base_parameters = parameters.copy()
            cached_proposed_parameters = proposed_parameters
            cached_newton_direction = newton_direction
            _t_solve = _time.monotonic() - _t_solve_start
        step_scale = 1.0 / (1.0 + damping)
        candidate_parameters = parameters + step_scale * newton_direction
        candidate_objective, _candidate_gradient, _candidate_weights, _candidate_linear_predictor, _candidate_probabilities = penalized_terms(candidate_parameters)
        _t_total = _time.monotonic() - _newton_t0
        actual_gain = candidate_objective - current_objective
        newton_curvature = float(np.dot(gradient, newton_direction))
        predicted_gain = step_scale * (1.0 - 0.5 * step_scale) * max(newton_curvature, 1e-12)
        gain_ratio = actual_gain / max(predicted_gain, 1e-8)
        relative_step_size = float(np.linalg.norm(step_scale * newton_direction)) / max(
            float(np.linalg.norm(parameters)),
            1e-8,
        )
        accept = np.isfinite(candidate_objective) and actual_gain > 0.0 and gain_ratio >= success_threshold
        if accept:
            parameters = candidate_parameters
            cached_terms_parameters = candidate_parameters.copy()
            cached_terms = (
                candidate_objective,
                _candidate_gradient,
                _candidate_weights,
                _candidate_linear_predictor,
                _candidate_probabilities,
            )
            cached_proposal_base_parameters = None
            cached_proposed_parameters = None
            cached_newton_direction = None
            damping = max(damping * damping_decrease_factor, minimum_damping)
            log(f"      Newton iter {_iteration_index+1}/{max_iterations}: ACCEPT  obj={candidate_objective:.4f}  gain={actual_gain:.2e}  ratio={gain_ratio:.3f}  grad={gradient_norm:.2e}  step={relative_step_size:.2e}  damping={damping:.2e}  [penalized={_t_penalized:.1f}s solve={_t_solve:.1f}s total={_t_total:.1f}s]  mem={mem()}")
            if relative_step_size <= gradient_tolerance:
                log(f"      Newton converged at iter {_iteration_index+1}: step_size={relative_step_size:.2e} <= tol={gradient_tolerance:.2e}")
                break
        else:
            stalled_objective_tolerance = stalled_objective_relative_tolerance * max(abs(current_objective), 1.0)
            if (
                np.isfinite(candidate_objective)
                and abs(actual_gain) <= stalled_objective_tolerance
            ):
                log(
                    f"      Newton converged at iter {_iteration_index+1}: stalled gain={actual_gain:.2e} "
                    + f"with step={relative_step_size:.2e}"
                )
                break
            damping *= damping_increase_factor
            log(f"      Newton iter {_iteration_index+1}/{max_iterations}: REJECT  obj={current_objective:.4f}  cand_obj={candidate_objective:.4f}  gain={actual_gain:.2e}  ratio={gain_ratio:.3f}  grad={gradient_norm:.2e}  step={relative_step_size:.2e}  damping={damping:.2e}  [penalized={_t_penalized:.1f}s solve={_t_solve:.1f}s total={_t_total:.1f}s]  mem={mem()}")

    final_objective, _final_gradient, final_weights, linear_predictor, probabilities = penalized_terms(parameters)
    target_values = targets.astype(compute_np_dtype, copy=False)
    final_working_response = linear_predictor + (target_values - probabilities) / final_weights - predictor_offset_array
    working_alpha, working_beta, _working_variance, _working_projected_targets, _working_fitted_response, _working_quadratic, _working_logdet_covariance, _working_logdet_gls = (
        _restricted_posterior_state(
            genotype_matrix=standardized_genotypes,
            covariate_matrix=covariate_matrix,
            targets=final_working_response,
            prior_variances=prior_variances,
            diagonal_noise=1.0 / final_weights,
            solver_tolerance=newton_solver_tolerance,
            maximum_linear_solver_iterations=newton_maximum_linear_solver_iterations,
            logdet_probe_count=logdet_probe_count,
            logdet_lanczos_steps=logdet_lanczos_steps,
            exact_solver_matrix_limit=exact_solver_matrix_limit,
            posterior_variance_batch_size=posterior_variance_batch_size,
            posterior_variance_probe_count=posterior_variance_probe_count,
            random_seed=random_seed + 2 * max_iterations,
            compute_logdet=False,
            compute_beta_variance=False,
            initial_beta_guess=parameters[covariate_count:],
            sample_space_preconditioner_rank=sample_space_preconditioner_rank,
        )
    )
    working_parameters = np.concatenate([working_alpha, working_beta], axis=0)
    working_objective, _working_gradient, _working_weights, _working_linear_predictor, _working_probabilities = penalized_terms(working_parameters)
    if working_objective >= final_objective:
        parameters = working_parameters
        final_objective = working_objective
        final_weights = _working_weights
        linear_predictor = _working_linear_predictor
        probabilities = _working_probabilities

    final_working_response = linear_predictor + (target_values - probabilities) / final_weights - predictor_offset_array
    final_alpha, final_beta, beta_variance, _projected_targets, _fitted_response, _restricted_quadratic, logdet_covariance, logdet_gls = (
        _restricted_posterior_state(
                genotype_matrix=standardized_genotypes,
            covariate_matrix=covariate_matrix,
            targets=final_working_response,
            prior_variances=prior_variances,
            diagonal_noise=1.0 / final_weights,
            solver_tolerance=solver_tolerance,
            maximum_linear_solver_iterations=maximum_linear_solver_iterations,
            logdet_probe_count=logdet_probe_count,
            logdet_lanczos_steps=logdet_lanczos_steps,
            exact_solver_matrix_limit=exact_solver_matrix_limit,
            posterior_variance_batch_size=posterior_variance_batch_size,
            posterior_variance_probe_count=posterior_variance_probe_count,
            random_seed=random_seed + 2 * max_iterations + 17,
            compute_logdet=compute_logdet,
            compute_beta_variance=compute_beta_variance,
            initial_beta_guess=parameters[covariate_count:],
            sample_space_preconditioner_rank=sample_space_preconditioner_rank,
        )
    )
    laplace_weights = np.asarray(final_weights, dtype=compute_np_dtype)
    final_parameters = np.concatenate([final_alpha, final_beta], axis=0)
    final_objective, _final_gradient, _final_penalty_weights, linear_predictor, _final_probabilities = penalized_terms(final_parameters)
    logdet_hessian = (
        float(np.sum(np.log(np.maximum(prior_precision, 1e-12))))
        + float(np.sum(np.log(np.maximum(laplace_weights, 1e-12))))
        + (logdet_covariance + logdet_gls if compute_logdet else 0.0)
    )
    laplace_objective = final_objective - 0.5 * logdet_hessian
    log(f"      binary posterior done: laplace_obj={laplace_objective:.4f}  final_obj={final_objective:.4f}  logdet_hessian={logdet_hessian:.4f}  mem={mem()}")
    return (
        final_parameters[:covariate_count],
        final_parameters[covariate_count:],
        beta_variance,
        linear_predictor,
        float(laplace_objective),
    )


# Defines the covariance matrix V = D + X @ diag(tau^2) @ X^T as a
# "matrix-free" operator — we never build V explicitly (it would be
# n_samples x n_samples, way too big), instead we define how to multiply
# V times a vector.  This lets us solve V^{-1} @ b using iterative methods
# (conjugate gradient) without ever storing the full matrix.
def _sample_space_operator(
    genotype_matrix: StandardizedGenotypeMatrix,
    prior_variances: np.ndarray,
    diagonal_noise: np.ndarray,
    batch_size: int = 1024,
):
    compute_dtype = gpu_compute_jax_dtype()
    diag_noise_jax = jnp.asarray(diagonal_noise, dtype=compute_dtype)
    prior_var_jax = jnp.asarray(prior_variances, dtype=compute_dtype)
    streaming_dtype = np.float64
    cupy = None
    streaming_gpu_enabled = False
    if genotype_matrix._cupy_cache is None and not genotype_matrix.supports_jax_dense_ops() and genotype_matrix.raw is not None:
        cupy = _try_import_cupy()
        streaming_gpu_enabled = cupy is not None
    if streaming_gpu_enabled:
        assert cupy is not None
        compute_cp_dtype = _cupy_compute_dtype(cupy)
        diag_noise_gpu = cupy.asarray(diagonal_noise, dtype=compute_cp_dtype)
        prior_var_gpu = cupy.asarray(prior_variances, dtype=compute_cp_dtype)

    def matvec(vector) -> jnp.ndarray:
        v = jnp.asarray(vector, dtype=compute_dtype)
        if streaming_gpu_enabled:
            assert cupy is not None
            compute_cp_dtype = _cupy_compute_dtype(cupy)
            vector_gpu = _to_cupy_compute(v)
            result_gpu = diag_noise_gpu * vector_gpu
            for batch_slice, standardized_batch in _iter_standardized_gpu_batches(
                genotype_matrix.raw,
                genotype_matrix.variant_indices,
                genotype_matrix.means,
                genotype_matrix.scales,
                batch_size=batch_size,
                cupy=cupy,
                dtype=compute_cp_dtype,
            ):
                scaled_projection = prior_var_gpu[batch_slice] * (standardized_batch.T @ vector_gpu)
                result_gpu += standardized_batch @ scaled_projection
            return _cupy_to_jax(result_gpu)
        if genotype_matrix._cupy_cache is None and not genotype_matrix.supports_jax_dense_ops():
            vector_np = np.asarray(v, dtype=streaming_dtype)
            genotype_term = np.zeros(genotype_matrix.shape[0], dtype=streaming_dtype)
            prior_var_stream = np.asarray(prior_variances, dtype=streaming_dtype)
            diag_noise_stream = np.asarray(diagonal_noise, dtype=streaming_dtype)
            for batch in genotype_matrix.iter_column_batches(batch_size=batch_size):
                genotype_batch = np.asarray(batch.values, dtype=streaming_dtype)
                scaled_projection = prior_var_stream[batch.variant_indices] * (genotype_batch.T @ vector_np)
                genotype_term += genotype_batch @ scaled_projection
            return jnp.asarray(diag_noise_stream * vector_np + genotype_term, dtype=compute_dtype)
        projected = genotype_matrix.transpose_matvec(v)
        return diag_noise_jax * v + genotype_matrix.matvec(prior_var_jax * projected)

    def matmat(matrix) -> jnp.ndarray:
        matrix_jax = jnp.asarray(matrix, dtype=compute_dtype)
        if streaming_gpu_enabled:
            assert cupy is not None
            compute_cp_dtype = _cupy_compute_dtype(cupy)
            matrix_gpu = _to_cupy_compute(matrix_jax)
            result_gpu = diag_noise_gpu[:, None] * matrix_gpu
            for batch_slice, standardized_batch in _iter_standardized_gpu_batches(
                genotype_matrix.raw,
                genotype_matrix.variant_indices,
                genotype_matrix.means,
                genotype_matrix.scales,
                batch_size=batch_size,
                cupy=cupy,
                dtype=compute_cp_dtype,
            ):
                scaled_projection = prior_var_gpu[batch_slice, None] * (standardized_batch.T @ matrix_gpu)
                result_gpu += standardized_batch @ scaled_projection
            return _cupy_to_jax(result_gpu)
        if genotype_matrix._cupy_cache is None and not genotype_matrix.supports_jax_dense_ops():
            matrix_np = np.asarray(matrix_jax, dtype=streaming_dtype)
            genotype_term = np.zeros((genotype_matrix.shape[0], matrix_np.shape[1]), dtype=streaming_dtype)
            prior_var_stream = np.asarray(prior_variances, dtype=streaming_dtype)
            diag_noise_stream = np.asarray(diagonal_noise, dtype=streaming_dtype)
            for batch in genotype_matrix.iter_column_batches(batch_size=batch_size):
                genotype_batch = np.asarray(batch.values, dtype=streaming_dtype)
                scaled_projection = prior_var_stream[batch.variant_indices, None] * (genotype_batch.T @ matrix_np)
                genotype_term += genotype_batch @ scaled_projection
            return jnp.asarray(diag_noise_stream[:, None] * matrix_np + genotype_term, dtype=compute_dtype)
        # X^T @ M gives (p, k), scale by prior variance, then X @ result gives (n, k)
        projected = genotype_matrix.transpose_matmat(matrix_jax)  # (p, k)
        scaled = prior_var_jax[:, None] * projected  # (p, k)
        genotype_term_jax = genotype_matrix.matmat(scaled)
        return diag_noise_jax[:, None] * matrix_jax + genotype_term_jax

    return build_linear_operator(
        shape=(genotype_matrix.shape[0], genotype_matrix.shape[0]),
        matvec=matvec,
        matmat=matmat,
        dtype=compute_dtype,
        jax_compatible=genotype_matrix.supports_jax_dense_ops(),
    )


def _sample_space_diagonal_preconditioner(
    genotype_matrix: StandardizedGenotypeMatrix,
    prior_variances: np.ndarray,
    diagonal_noise: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    if genotype_matrix._cupy_cache is not None:
        import cupy as cp
        compute_cp_dtype = _cupy_compute_dtype(cp)
        X = genotype_matrix._cupy_cache  # (n, p) float32 on GPU
        pv = cp.asarray(prior_variances, dtype=compute_cp_dtype)
        # diag(X @ diag(pv) @ X^T) = sum(X^2 * pv, axis=1)
        # Chunked to avoid allocating a full (n, p) intermediate on GPU.
        diag_gpu = cp.zeros(X.shape[0], dtype=compute_cp_dtype)
        chunk = max(1, min(512, X.shape[1]))
        for start in range(0, X.shape[1], chunk):
            end = min(start + chunk, X.shape[1])
            x_chunk = X[:, start:end].astype(compute_cp_dtype, copy=False)
            diag_gpu += cp.sum(x_chunk * x_chunk * pv[start:end], axis=1)
        result = np.asarray(diagonal_noise, dtype=np.float64) + diag_gpu.get().astype(np.float64)
        return np.maximum(result, 1e-8)
    if genotype_matrix.raw is not None and not genotype_matrix.supports_jax_dense_ops():
        cupy = _try_import_cupy()
        if cupy is not None:
            compute_cp_dtype = _cupy_compute_dtype(cupy)
            diagonal_gpu = cupy.asarray(diagonal_noise, dtype=compute_cp_dtype)
            prior_variances_gpu = cupy.asarray(prior_variances, dtype=compute_cp_dtype)
            for batch_slice, standardized_batch in _iter_standardized_gpu_batches(
                genotype_matrix.raw,
                genotype_matrix.variant_indices,
                genotype_matrix.means,
                genotype_matrix.scales,
                batch_size=batch_size,
                cupy=cupy,
                dtype=compute_cp_dtype,
            ):
                diagonal_gpu += cupy.sum(
                    standardized_batch * standardized_batch * prior_variances_gpu[batch_slice][None, :],
                    axis=1,
                )
            return np.maximum(np.asarray(diagonal_gpu.get(), dtype=np.float64), 1e-8)
    diagonal = np.asarray(diagonal_noise, dtype=np.float64).copy()
    prior_variances_f64 = np.asarray(prior_variances, dtype=np.float64)
    for batch in genotype_matrix.iter_column_batches(batch_size=batch_size):
        genotype_batch = np.asarray(batch.values, dtype=np.float64)
        diagonal += np.sum(
            genotype_batch * genotype_batch * prior_variances_f64[batch.variant_indices][None, :],
            axis=1,
        )
    return np.maximum(diagonal, 1e-8)


def _sample_space_nystrom_rank(rank: int, sample_count: int) -> int:
    resolved_rank = min(max(int(rank), 0), int(sample_count))
    if resolved_rank <= 0:
        return 0
    oversampling = min(16, max(4, resolved_rank // 8))
    return min(sample_count, resolved_rank + oversampling)


def _positive_semidefinite_factor(matrix: np.ndarray) -> np.ndarray | None:
    symmetric_matrix = np.asarray(matrix, dtype=np.float64)
    symmetric_matrix = (symmetric_matrix + symmetric_matrix.T) * 0.5
    if symmetric_matrix.size == 0:
        return None
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric_matrix)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    tolerance = max(np.finfo(np.float64).eps * symmetric_matrix.shape[0] * scale * 16.0, 1e-12)
    keep = eigenvalues > tolerance
    if not np.any(keep):
        return None
    return np.asarray(
        eigenvectors[:, keep] * np.sqrt(eigenvalues[keep])[None, :],
        dtype=np.float64,
    )


def _sample_space_kernel_matmat_cpu(
    genotype_matrix: StandardizedGenotypeMatrix,
    prior_variances: np.ndarray,
    matrix: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    rhs = np.asarray(matrix, dtype=np.float64)
    vector_input = rhs.ndim == 1
    if vector_input:
        rhs = rhs[:, None]
    elif rhs.ndim != 2:
        raise ValueError("sample-space kernel expects a vector or matrix right-hand side.")
    kernel_term = np.zeros((genotype_matrix.shape[0], rhs.shape[1]), dtype=np.float64)
    prior_variances_f64 = np.asarray(prior_variances, dtype=np.float64)
    for batch in genotype_matrix.iter_column_batches(batch_size=batch_size):
        genotype_batch = np.asarray(batch.values, dtype=np.float64)
        scaled_projection = prior_variances_f64[batch.variant_indices, None] * (genotype_batch.T @ rhs)
        kernel_term += genotype_batch @ scaled_projection
    return kernel_term[:, 0] if vector_input else kernel_term


def _sample_space_nystrom_basis_cpu(
    genotype_matrix: StandardizedGenotypeMatrix,
    prior_variances: np.ndarray,
    batch_size: int,
    rank: int,
    random_seed: int,
) -> np.ndarray | None:
    sketch_rank = _sample_space_nystrom_rank(rank, genotype_matrix.shape[0])
    if sketch_rank <= 0:
        return None
    sketch_probes = _orthogonal_probe_matrix(
        dimension=genotype_matrix.shape[0],
        probe_count=sketch_rank,
        random_seed=random_seed,
    )
    sketch_response = _sample_space_kernel_matmat_cpu(
        genotype_matrix=genotype_matrix,
        prior_variances=prior_variances,
        matrix=sketch_probes,
        batch_size=batch_size,
    )
    basis_matrix, triangular_matrix = np.linalg.qr(np.asarray(sketch_response, dtype=np.float64), mode="reduced")
    diagonal = np.abs(np.diag(triangular_matrix))
    effective_rank = min(int(rank), int(np.sum(diagonal > 1e-10)))
    if effective_rank <= 0:
        return None
    return np.asarray(basis_matrix[:, :effective_rank], dtype=np.float64)


def _sample_space_nystrom_factor_cpu(
    genotype_matrix: StandardizedGenotypeMatrix,
    prior_variances: np.ndarray,
    batch_size: int,
    rank: int,
    random_seed: int,
) -> np.ndarray | None:
    basis_matrix = _sample_space_nystrom_basis_cpu(
        genotype_matrix=genotype_matrix,
        prior_variances=prior_variances,
        batch_size=batch_size,
        rank=rank,
        random_seed=random_seed,
    )
    if basis_matrix is None:
        return None
    projected_kernel = _sample_space_kernel_matmat_cpu(
        genotype_matrix=genotype_matrix,
        prior_variances=prior_variances,
        matrix=basis_matrix,
        batch_size=batch_size,
    )
    gram_factor = _positive_semidefinite_factor(basis_matrix.T @ projected_kernel)
    if gram_factor is None:
        return None
    return np.asarray(basis_matrix @ gram_factor, dtype=np.float64)


def _sample_space_nystrom_factor_gpu(
    genotype_matrix: StandardizedGenotypeMatrix,
    prior_variances: np.ndarray,
    batch_size: int,
    rank: int,
    random_seed: int,
):
    cupy = _try_import_cupy()
    if cupy is None:
        return None
    compute_cp_dtype = cupy.float64
    sketch_rank = _sample_space_nystrom_rank(rank, genotype_matrix.shape[0])
    if sketch_rank <= 0:
        return None
    if not hasattr(cupy.linalg, "qr") or not hasattr(cupy, "abs") or not hasattr(cupy, "diag"):
        return None
    sketch_probes_gpu = cupy.asarray(
        _orthogonal_probe_matrix(
            dimension=genotype_matrix.shape[0],
            probe_count=sketch_rank,
            random_seed=random_seed,
        ),
        dtype=compute_cp_dtype,
    )
    zero_diagonal_noise = np.zeros(genotype_matrix.shape[0], dtype=np.float64)
    sketch_response_gpu = _apply_sample_space_operator_gpu(
        genotype_matrix=genotype_matrix,
        prior_variances=prior_variances,
        diagonal_noise=zero_diagonal_noise,
        matrix_gpu=sketch_probes_gpu,
        batch_size=batch_size,
        cp=cupy,
        dtype=compute_cp_dtype,
    )
    try:
        basis_gpu, triangular_gpu = cupy.linalg.qr(sketch_response_gpu, mode="reduced")
    except TypeError:
        basis_gpu, triangular_gpu = cupy.linalg.qr(sketch_response_gpu)
    diagonal = np.asarray(cupy.abs(cupy.diag(triangular_gpu)), dtype=np.float64)
    effective_rank = min(int(rank), int(np.sum(diagonal > 1e-10)))
    if effective_rank <= 0:
        return None
    basis_gpu = cupy.asarray(basis_gpu[:, :effective_rank], dtype=compute_cp_dtype)
    projected_kernel_gpu = _apply_sample_space_operator_gpu(
        genotype_matrix=genotype_matrix,
        prior_variances=prior_variances,
        diagonal_noise=zero_diagonal_noise,
        matrix_gpu=basis_gpu,
        batch_size=batch_size,
        cp=cupy,
        dtype=compute_cp_dtype,
    )
    gram_factor = _positive_semidefinite_factor(np.asarray(basis_gpu.T @ projected_kernel_gpu, dtype=np.float64))
    if gram_factor is None:
        return None
    return basis_gpu @ cupy.asarray(gram_factor, dtype=compute_cp_dtype)


def _sample_space_preconditioner(
    genotype_matrix: StandardizedGenotypeMatrix,
    prior_variances: np.ndarray,
    diagonal_noise: np.ndarray,
    batch_size: int,
    rank: int,
    random_seed: int = 0,
) -> Callable[[jnp.ndarray], jnp.ndarray] | np.ndarray | jnp.ndarray:
    diagonal_preconditioner = _sample_space_diagonal_preconditioner(
        genotype_matrix=genotype_matrix,
        prior_variances=prior_variances,
        diagonal_noise=diagonal_noise,
        batch_size=batch_size,
    )
    if rank <= 0:
        return diagonal_preconditioner
    selected_rank = min(int(rank), int(genotype_matrix.shape[0]))
    if selected_rank <= 0:
        return diagonal_preconditioner
    low_rank_factor_gpu = _sample_space_nystrom_factor_gpu(
        genotype_matrix=genotype_matrix,
        prior_variances=prior_variances,
        batch_size=batch_size,
        rank=selected_rank,
        random_seed=random_seed,
    )
    if low_rank_factor_gpu is not None:
        import cupy as cp
        from cupyx.scipy.linalg import solve_triangular as cp_solve_triangular

        gpu_cache_source = "full" if genotype_matrix._cupy_cache is not None else "streaming"
        effective_rank = int(low_rank_factor_gpu.shape[1])
        log(f"      sample-space preconditioner: GPU Nyström-Woodbury rank={effective_rank} source={gpu_cache_source}")
        weighted_selected_genotypes = low_rank_factor_gpu.astype(cp.float64, copy=False)
        base_diagonal = cp.maximum(
            cp.asarray(diagonal_preconditioner, dtype=cp.float64)
            - cp.sum(weighted_selected_genotypes * weighted_selected_genotypes, axis=1),
            cp.float64(1e-8),
        )
        inverse_base_diagonal = cp.float64(1.0) / base_diagonal
        weighted_inverse_selected = inverse_base_diagonal[:, None] * weighted_selected_genotypes
        low_rank_precision = cp.eye(effective_rank, dtype=cp.float64) + (
            weighted_selected_genotypes.T @ weighted_inverse_selected
        )
        low_rank_cholesky = cp.linalg.cholesky(
            low_rank_precision + cp.eye(effective_rank, dtype=cp.float64) * cp.float64(1e-8)
        )

        def apply_preconditioner_gpu(right_hand_side: jnp.ndarray) -> jnp.ndarray:
            right_hand_side_gpu = _to_cupy_float64(right_hand_side)
            if right_hand_side_gpu.ndim not in (1, 2):
                raise ValueError("sample-space preconditioner expects a vector or matrix right-hand side.")
            weighted_rhs = (
                inverse_base_diagonal[:, None] * right_hand_side_gpu
                if right_hand_side_gpu.ndim == 2
                else inverse_base_diagonal * right_hand_side_gpu
            )
            correction_rhs = weighted_selected_genotypes.T @ weighted_rhs
            correction = weighted_inverse_selected @ _gpu_cholesky_solve(
                correction_rhs,
                low_rank_cholesky,
                cp_solve_triangular,
            )
            return _cupy_to_jax(weighted_rhs - correction)

        return apply_preconditioner_gpu
    low_rank_factor = _sample_space_nystrom_factor_cpu(
        genotype_matrix=genotype_matrix,
        prior_variances=prior_variances,
        batch_size=batch_size,
        rank=selected_rank,
        random_seed=random_seed,
    )
    if low_rank_factor is None:
        return diagonal_preconditioner
    effective_rank = int(low_rank_factor.shape[1])
    log(f"      sample-space preconditioner: CPU Nyström-Woodbury rank={effective_rank}")
    weighted_selected_genotypes = np.asarray(low_rank_factor, dtype=np.float64)
    selected_diagonal = np.sum(weighted_selected_genotypes * weighted_selected_genotypes, axis=1)
    base_diagonal = np.maximum(diagonal_preconditioner - selected_diagonal, 1e-8)
    inverse_base_diagonal = 1.0 / base_diagonal
    weighted_inverse_selected = inverse_base_diagonal[:, None] * weighted_selected_genotypes
    low_rank_precision = np.eye(effective_rank, dtype=np.float64) + weighted_selected_genotypes.T @ weighted_inverse_selected
    low_rank_cholesky = np.linalg.cholesky(low_rank_precision + np.eye(effective_rank, dtype=np.float64) * 1e-8)

    def apply_preconditioner_cpu(right_hand_side: jnp.ndarray) -> jnp.ndarray:
        right_hand_side_array = np.asarray(right_hand_side, dtype=np.float64)
        weighted_rhs = (
            inverse_base_diagonal[:, None] * right_hand_side_array
            if right_hand_side_array.ndim == 2
            else inverse_base_diagonal * right_hand_side_array
        )
        correction_rhs = weighted_selected_genotypes.T @ weighted_rhs
        correction = weighted_inverse_selected @ _cholesky_solve(
            low_rank_cholesky,
            correction_rhs,
        )
        return jnp.asarray(weighted_rhs - correction, dtype=gpu_compute_jax_dtype())

    return apply_preconditioner_cpu


def _sample_space_gpu_preconditioner(
    genotype_matrix: StandardizedGenotypeMatrix,
    prior_variances: np.ndarray,
    diagonal_noise: np.ndarray,
    batch_size: int,
    rank: int,
    random_seed: int = 0,
):
    import cupy as cp
    from cupyx.scipy.linalg import solve_triangular as cp_solve_triangular

    diagonal_preconditioner = _sample_space_diagonal_preconditioner(
        genotype_matrix=genotype_matrix,
        prior_variances=prior_variances,
        diagonal_noise=diagonal_noise,
        batch_size=batch_size,
    )
    diagonal_preconditioner_gpu = cp.asarray(diagonal_preconditioner, dtype=cp.float64)

    def apply_diagonal(right_hand_side_gpu):
        right_hand_side_gpu = cp.asarray(right_hand_side_gpu, dtype=cp.float64)
        if right_hand_side_gpu.ndim == 2:
            return right_hand_side_gpu / diagonal_preconditioner_gpu[:, None]
        return right_hand_side_gpu / diagonal_preconditioner_gpu

    if rank <= 0:
        return apply_diagonal
    selected_rank = min(int(rank), int(genotype_matrix.shape[0]))
    if selected_rank <= 0:
        return apply_diagonal
    low_rank_factor_gpu = _sample_space_nystrom_factor_gpu(
        genotype_matrix=genotype_matrix,
        prior_variances=prior_variances,
        batch_size=batch_size,
        rank=selected_rank,
        random_seed=random_seed,
    )
    if low_rank_factor_gpu is None:
        return apply_diagonal

    gpu_cache_source = "full" if genotype_matrix._cupy_cache is not None else "streaming"
    effective_rank = int(low_rank_factor_gpu.shape[1])
    log(f"      sample-space preconditioner: GPU Nyström-Woodbury rank={effective_rank} source={gpu_cache_source}")
    def build_low_rank_bundle():
        weighted_selected_genotypes = low_rank_factor_gpu.astype(cp.float64, copy=False)
        base_diagonal = cp.maximum(
            diagonal_preconditioner_gpu
            - cp.sum(weighted_selected_genotypes * weighted_selected_genotypes, axis=1),
            cp.float64(1e-8),
        )
        inverse_base_diagonal = cp.float64(1.0) / base_diagonal
        weighted_inverse_selected = inverse_base_diagonal[:, None] * weighted_selected_genotypes
        low_rank_precision = cp.eye(effective_rank, dtype=cp.float64) + (
            weighted_selected_genotypes.T @ weighted_inverse_selected
        )
        low_rank_cholesky = cp.linalg.cholesky(
            low_rank_precision + cp.eye(effective_rank, dtype=cp.float64) * cp.float64(1e-8)
        )
        return weighted_selected_genotypes, inverse_base_diagonal, weighted_inverse_selected, low_rank_cholesky

    weighted_selected_genotypes, inverse_base_diagonal, weighted_inverse_selected, low_rank_cholesky = build_low_rank_bundle()

    def apply_low_rank(right_hand_side_gpu):
        if right_hand_side_gpu.ndim not in (1, 2):
            raise ValueError("sample-space preconditioner expects a vector or matrix right-hand side.")
        right_hand_side_gpu = cp.asarray(right_hand_side_gpu, dtype=cp.float64)
        weighted_rhs = (
            inverse_base_diagonal[:, None] * right_hand_side_gpu
            if right_hand_side_gpu.ndim == 2
            else inverse_base_diagonal * right_hand_side_gpu
        )
        correction_rhs = weighted_selected_genotypes.T @ weighted_rhs
        correction = weighted_inverse_selected @ _gpu_cholesky_solve(
            correction_rhs,
            low_rank_cholesky,
            cp_solve_triangular,
        )
        return weighted_rhs - correction

    return apply_low_rank


def _apply_sample_space_operator_gpu(
    genotype_matrix: StandardizedGenotypeMatrix,
    prior_variances: np.ndarray,
    diagonal_noise: np.ndarray,
    matrix_gpu,
    *,
    batch_size: int,
    cp,
    dtype,
):
    input_gpu = cp.asarray(matrix_gpu, dtype=dtype)
    diagonal_noise_gpu = cp.asarray(diagonal_noise, dtype=dtype)
    prior_variances_gpu = cp.asarray(prior_variances, dtype=dtype)
    if input_gpu.ndim == 1:
        input_gpu = input_gpu[:, None]
        vector_input = True
    elif input_gpu.ndim == 2:
        vector_input = False
    else:
        raise ValueError("sample-space GPU operator expects a vector or matrix right-hand side.")
    result_gpu = diagonal_noise_gpu[:, None] * input_gpu
    if genotype_matrix._cupy_cache is not None:
        x_gpu = genotype_matrix._cupy_cache
        column_chunk = max(1, min(batch_size, x_gpu.shape[1]))
        for start in range(0, x_gpu.shape[1], column_chunk):
            stop = min(start + column_chunk, x_gpu.shape[1])
            x_chunk = x_gpu[:, start:stop].astype(dtype, copy=False)
            scaled_projection = prior_variances_gpu[start:stop, None] * (x_chunk.T @ input_gpu)
            result_gpu += x_chunk @ scaled_projection
    else:
        for batch_slice, standardized_batch in _iter_standardized_gpu_batches(
            genotype_matrix.raw,
            genotype_matrix.variant_indices,
            genotype_matrix.means,
            genotype_matrix.scales,
            batch_size=batch_size,
            cupy=cp,
            dtype=dtype,
        ):
            scaled_projection = prior_variances_gpu[batch_slice, None] * (standardized_batch.T @ input_gpu)
            result_gpu += standardized_batch @ scaled_projection
    return result_gpu[:, 0] if vector_input else result_gpu


def _solve_sample_space_rhs_gpu_inner(
    genotype_matrix: StandardizedGenotypeMatrix,
    prior_variances: np.ndarray,
    diagonal_noise: np.ndarray,
    right_hand_side_gpu,
    tolerance: float,
    max_iterations: int,
    preconditioner,
    batch_size: int,
    cp,
    compute_cp_dtype,
):
    rhs_gpu = cp.asarray(right_hand_side_gpu, dtype=compute_cp_dtype)
    vector_input = rhs_gpu.ndim == 1
    if vector_input:
        rhs_gpu = rhs_gpu[:, None]
    elif rhs_gpu.ndim != 2:
        raise ValueError("GPU sample-space solve expects a vector or matrix right-hand side.")

    def apply_operator(matrix_gpu):
        return _apply_sample_space_operator_gpu(
            genotype_matrix=genotype_matrix,
            prior_variances=prior_variances,
            diagonal_noise=diagonal_noise,
            matrix_gpu=matrix_gpu,
            batch_size=batch_size,
            cp=cp,
            dtype=compute_cp_dtype,
        )

    residual_refresh_interval = 32
    tol_sq = float(tolerance) * float(tolerance)
    solution_gpu = preconditioner(rhs_gpu)
    residual_gpu = rhs_gpu - apply_operator(solution_gpu)
    residual_norm_sq = np.asarray(cp.sum(residual_gpu * residual_gpu, axis=0, dtype=cp.float64), dtype=np.float64)
    rhs_norm_sq = np.asarray(cp.sum(rhs_gpu * rhs_gpu, axis=0, dtype=cp.float64), dtype=np.float64)
    convergence_threshold_sq = np.maximum(tol_sq, tol_sq * np.maximum(residual_norm_sq, rhs_norm_sq))
    converged = residual_norm_sq <= convergence_threshold_sq
    if np.all(converged):
        solution = cp.asarray(solution_gpu, dtype=cp.float64)
        return solution[:, 0] if vector_input else solution

    preconditioned_residual_gpu = preconditioner(residual_gpu)
    search_direction_gpu = preconditioned_residual_gpu
    residual_dot = np.asarray(cp.sum(residual_gpu * preconditioned_residual_gpu, axis=0, dtype=cp.float64), dtype=np.float64)
    for iteration_index in range(max_iterations):
        active_columns = np.flatnonzero(~converged).astype(np.int32, copy=False)
        if active_columns.size == 0:
            break
        masked_search_gpu = search_direction_gpu[:, active_columns]
        operator_search_gpu = apply_operator(masked_search_gpu)
        step_denom = np.asarray(cp.sum(masked_search_gpu * operator_search_gpu, axis=0, dtype=cp.float64), dtype=np.float64)
        if np.any(~np.isfinite(step_denom) | (step_denom <= 0.0)):
            raise RuntimeError("GPU conjugate-gradient operator is not positive definite.")
        step_scale = residual_dot[active_columns] / step_denom
        step_scale_gpu = cp.asarray(step_scale, dtype=compute_cp_dtype)
        solution_gpu[:, active_columns] += masked_search_gpu * step_scale_gpu[None, :]
        residual_gpu[:, active_columns] -= operator_search_gpu * step_scale_gpu[None, :]
        if (iteration_index + 1) % residual_refresh_interval == 0:
            residual_gpu[:, active_columns] = rhs_gpu[:, active_columns] - apply_operator(solution_gpu[:, active_columns])
        residual_norm_sq[active_columns] = np.asarray(
            cp.sum(residual_gpu[:, active_columns] * residual_gpu[:, active_columns], axis=0, dtype=cp.float64),
            dtype=np.float64,
        )
        converged = residual_norm_sq <= convergence_threshold_sq
        if np.all(converged):
            break
        refreshed_residual_gpu = residual_gpu[:, active_columns]
        refreshed_preconditioned_gpu = preconditioner(refreshed_residual_gpu)
        updated_residual_dot_active = np.asarray(
            cp.sum(refreshed_residual_gpu * refreshed_preconditioned_gpu, axis=0, dtype=cp.float64),
            dtype=np.float64,
        )
        beta_active = updated_residual_dot_active / np.maximum(residual_dot[active_columns], 1e-30)
        if np.any(~np.isfinite(beta_active) | (beta_active < 0.0)):
            raise RuntimeError("GPU conjugate-gradient preconditioner produced an invalid update.")
        beta_active_gpu = cp.asarray(beta_active, dtype=compute_cp_dtype)
        search_direction_gpu[:, active_columns] = refreshed_preconditioned_gpu + (
            search_direction_gpu[:, active_columns] * beta_active_gpu[None, :]
        )
        residual_dot[active_columns] = updated_residual_dot_active
    solution = cp.asarray(solution_gpu, dtype=cp.float64)
    return solution[:, 0] if vector_input else solution


def _effective_sample_space_preconditioner_rank(
    genotype_matrix: StandardizedGenotypeMatrix,
    sample_count: int,
    variant_count: int,
    requested_rank: int,
) -> int:
    resolved_rank = max(int(requested_rank), 0)
    if resolved_rank == 0:
        return 0
    target_rank = resolved_rank
    if sample_count >= 32_768 and variant_count >= 65_536:
        target_rank = max(target_rank, 1_024)
    return min(target_rank, variant_count)


def _prefer_iterative_variant_space(
    genotype_matrix: StandardizedGenotypeMatrix,
    sample_count: int,
    variant_count: int,
    *,
    compute_beta_variance: bool,
    compute_logdet: bool,
    initial_beta_guess: np.ndarray | None,
) -> bool:
    return (
        not compute_beta_variance
        and
        not compute_logdet
        and initial_beta_guess is not None
        and genotype_matrix._cupy_cache is None
        and variant_count > sample_count
    )


def _solve_sample_space_rhs_gpu(
    genotype_matrix: StandardizedGenotypeMatrix,
    prior_variances: np.ndarray,
    diagonal_noise: np.ndarray,
    right_hand_side: np.ndarray,
    tolerance: float,
    max_iterations: int,
    preconditioner,
    batch_size: int,
) -> np.ndarray:
    import cupy as cp

    cupy = _try_import_cupy()
    if cupy is None:
        raise RuntimeError("GPU sample-space solve requires CuPy.")
    streaming_gpu_enabled = genotype_matrix._cupy_cache is None and genotype_matrix.raw is not None and not genotype_matrix.supports_jax_dense_ops()
    if genotype_matrix._cupy_cache is None and not streaming_gpu_enabled:
        raise RuntimeError("GPU sample-space solve requires a CuPy-resident matrix or GPU-streamable raw storage.")
    compute_cp_dtype = _cupy_compute_dtype(cp)
    right_hand_side_gpu64 = cp.asarray(right_hand_side, dtype=cp.float64)
    vector_input = right_hand_side_gpu64.ndim == 1
    if vector_input:
        right_hand_side_gpu64 = right_hand_side_gpu64[:, None]
    elif right_hand_side_gpu64.ndim != 2:
        raise ValueError("GPU sample-space solve expects a vector or matrix right-hand side.")

    rhs_norm_sq = np.asarray(cp.sum(right_hand_side_gpu64 * right_hand_side_gpu64, axis=0, dtype=cp.float64), dtype=np.float64)
    convergence_threshold_sq = np.maximum(float(tolerance) * float(tolerance), float(tolerance) * float(tolerance) * rhs_norm_sq)
    mixed_precision_enabled = compute_cp_dtype == cp.float32
    mixed_precision_failure: RuntimeError | None = None

    def true_residual(solution_gpu):
        residual_gpu64 = right_hand_side_gpu64 - _apply_sample_space_operator_gpu(
            genotype_matrix=genotype_matrix,
            prior_variances=prior_variances,
            diagonal_noise=diagonal_noise,
            matrix_gpu=solution_gpu,
            batch_size=batch_size,
            cp=cp,
            dtype=cp.float64,
        )
        residual_norm_sq = np.asarray(cp.sum(residual_gpu64 * residual_gpu64, axis=0, dtype=cp.float64), dtype=np.float64)
        return residual_gpu64, residual_norm_sq

    if not mixed_precision_enabled:
        solution_gpu64 = _solve_sample_space_rhs_gpu_inner(
            genotype_matrix=genotype_matrix,
            prior_variances=prior_variances,
            diagonal_noise=diagonal_noise,
            right_hand_side_gpu=right_hand_side_gpu64,
            tolerance=float(tolerance),
            max_iterations=max_iterations,
            preconditioner=preconditioner,
            batch_size=batch_size,
            cp=cp,
            compute_cp_dtype=cp.float64,
        )
        if solution_gpu64.ndim == 1:
            solution_gpu64 = solution_gpu64[:, None]
        _, residual_norm_sq = true_residual(solution_gpu64)
        final_residual = float(np.max(residual_norm_sq))
        final_threshold = float(np.max(convergence_threshold_sq))
        if final_residual > final_threshold:
            raise RuntimeError(
                "GPU conjugate-gradient solve failed to converge: "
                + f"residual={final_residual:.2e} threshold={final_threshold:.2e} "
                + f"iterations={max_iterations}"
            )
        solution = np.asarray(solution_gpu64.get() if hasattr(solution_gpu64, "get") else solution_gpu64, dtype=np.float64)
        return solution[:, 0] if vector_input else solution

    solution_gpu64 = cp.zeros(right_hand_side_gpu64.shape, dtype=cp.float64)
    inner_tolerance = max(float(tolerance), 1e-4)
    max_refinement_steps = 4
    for refinement_index in range(max_refinement_steps):
        residual_gpu64, residual_norm_sq = true_residual(solution_gpu64)
        if np.all(residual_norm_sq <= convergence_threshold_sq):
            solution = np.asarray(solution_gpu64.get() if hasattr(solution_gpu64, "get") else solution_gpu64, dtype=np.float64)
            return solution[:, 0] if vector_input else solution
        try:
            correction_gpu64 = _solve_sample_space_rhs_gpu_inner(
                genotype_matrix=genotype_matrix,
                prior_variances=prior_variances,
                diagonal_noise=diagonal_noise,
                right_hand_side_gpu=residual_gpu64,
                tolerance=inner_tolerance,
                max_iterations=max_iterations,
                preconditioner=preconditioner,
                batch_size=batch_size,
                cp=cp,
                compute_cp_dtype=compute_cp_dtype,
            )
        except RuntimeError as exc:
            mixed_precision_failure = exc
            break
        if correction_gpu64.ndim == 1:
            correction_gpu64 = correction_gpu64[:, None]
        solution_gpu64 += correction_gpu64
    residual_gpu64, residual_norm_sq = true_residual(solution_gpu64)
    if np.any(residual_norm_sq > convergence_threshold_sq):
        fallback_reason = (
            str(mixed_precision_failure)
            if mixed_precision_failure is not None
            else "mixed-precision iterative refinement did not hit the float64 residual target"
        )
        log(f"      sample-space solve: retrying GPU CG in float64 ({fallback_reason})")
        correction_gpu64 = _solve_sample_space_rhs_gpu_inner(
            genotype_matrix=genotype_matrix,
            prior_variances=prior_variances,
            diagonal_noise=diagonal_noise,
            right_hand_side_gpu=residual_gpu64,
            tolerance=float(tolerance),
            max_iterations=max_iterations,
            preconditioner=preconditioner,
            batch_size=batch_size,
            cp=cp,
            compute_cp_dtype=cp.float64,
        )
        if correction_gpu64.ndim == 1:
            correction_gpu64 = correction_gpu64[:, None]
        solution_gpu64 += correction_gpu64
        _, residual_norm_sq = true_residual(solution_gpu64)
    final_residual = float(np.max(residual_norm_sq))
    final_threshold = float(np.max(convergence_threshold_sq))
    if final_residual > final_threshold:
        failure_suffix = (
            f" last_mixed_precision_error={mixed_precision_failure}"
            if mixed_precision_failure is not None
            else ""
        )
        raise RuntimeError(
            "GPU conjugate-gradient solve failed to converge after iterative refinement: "
            + f"residual={final_residual:.2e} threshold={final_threshold:.2e} "
            + f"iterations={max_iterations} refinement_steps={max_refinement_steps}"
            + failure_suffix
        )
    solution = np.asarray(solution_gpu64.get() if hasattr(solution_gpu64, "get") else solution_gpu64, dtype=np.float64)
    return solution[:, 0] if vector_input else solution


def _solve_sample_space_rhs_cpu(
    genotype_matrix: StandardizedGenotypeMatrix,
    prior_variances: np.ndarray,
    diagonal_noise: np.ndarray,
    right_hand_side: np.ndarray,
    tolerance: float,
    max_iterations: int,
    preconditioner: Callable[[np.ndarray], np.ndarray | jnp.ndarray] | Callable[[jnp.ndarray], np.ndarray | jnp.ndarray] | np.ndarray | jnp.ndarray,
    batch_size: int,
) -> np.ndarray:
    import time

    rhs = np.asarray(right_hand_side, dtype=np.float64)
    vector_input = rhs.ndim == 1
    if vector_input:
        rhs = rhs[:, None]
    elif rhs.ndim != 2:
        raise ValueError("CPU sample-space solve expects a vector or matrix right-hand side.")

    diagonal_noise_stream = np.asarray(diagonal_noise, dtype=np.float32)
    prior_variances_stream = np.asarray(prior_variances, dtype=np.float32)

    def apply_operator(matrix: np.ndarray) -> np.ndarray:
        matrix_array = np.asarray(matrix, dtype=np.float32)
        if matrix_array.ndim != 2:
            raise ValueError("CPU sample-space operator expects a matrix right-hand side.")
        result = diagonal_noise_stream[:, None] * matrix_array
        for batch in genotype_matrix.iter_column_batches(batch_size=batch_size):
            genotype_batch = np.asarray(batch.values, dtype=np.float32)
            scaled_projection = (
                prior_variances_stream[batch.variant_indices, None]
                * (genotype_batch.T @ matrix_array)
            )
            result += genotype_batch @ scaled_projection
        return np.asarray(result, dtype=np.float64)

    residual_refresh_interval = 32
    tol_sq = float(tolerance) * float(tolerance)
    apply_preconditioner = cast(Callable[[np.ndarray], np.ndarray | jnp.ndarray], preconditioner) if callable(preconditioner) else None
    solution = (
        np.asarray(apply_preconditioner(rhs), dtype=np.float64)
        if apply_preconditioner is not None
        else rhs / np.maximum(np.asarray(preconditioner, dtype=np.float64)[:, None], 1e-12)
    )
    residual = rhs - apply_operator(solution)
    residual_norm_sq = np.sum(residual * residual, axis=0, dtype=np.float64)
    initial_residual_norm_sq = residual_norm_sq.copy()
    rhs_norm_sq = np.sum(rhs * rhs, axis=0, dtype=np.float64)
    convergence_threshold_sq = np.maximum(tol_sq, tol_sq * np.maximum(residual_norm_sq, rhs_norm_sq))
    converged = residual_norm_sq <= convergence_threshold_sq
    if np.all(converged):
        return solution[:, 0] if vector_input else solution

    preconditioned_residual = (
        residual
        if preconditioner is None
        else (
            np.asarray(preconditioner(residual), dtype=np.float64)
            if callable(preconditioner)
            else residual / np.maximum(np.asarray(preconditioner, dtype=np.float64)[:, None], 1e-12)
        )
    )
    search_direction = preconditioned_residual.copy()
    residual_dot = np.sum(residual * preconditioned_residual, axis=0, dtype=np.float64)
    t_start = time.monotonic()
    last_log = t_start

    for iteration_index in range(max_iterations):
        active_columns = np.flatnonzero(~converged).astype(np.int32, copy=False)
        if active_columns.size == 0:
            break
        masked_search = search_direction[:, active_columns]
        operator_search = apply_operator(masked_search)
        step_denom = np.sum(masked_search * operator_search, axis=0, dtype=np.float64)
        if np.any(~np.isfinite(step_denom) | (step_denom <= 0.0)):
            raise RuntimeError("CPU conjugate-gradient operator is not positive definite.")
        step_scale = residual_dot[active_columns] / step_denom
        solution[:, active_columns] += masked_search * step_scale[None, :]
        residual[:, active_columns] -= operator_search * step_scale[None, :]
        if (iteration_index + 1) % residual_refresh_interval == 0:
            residual[:, active_columns] = rhs[:, active_columns] - apply_operator(solution[:, active_columns])
        residual_norm_sq[active_columns] = np.sum(
            residual[:, active_columns] * residual[:, active_columns],
            axis=0,
            dtype=np.float64,
        )
        converged = residual_norm_sq <= convergence_threshold_sq
        if np.all(converged):
            break
        refreshed_residual = residual[:, active_columns]
        refreshed_preconditioned = (
            refreshed_residual
            if apply_preconditioner is None and preconditioner is None
            else (
                np.asarray(apply_preconditioner(refreshed_residual), dtype=np.float64)
                if apply_preconditioner is not None
                else refreshed_residual / np.maximum(np.asarray(preconditioner, dtype=np.float64)[:, None], 1e-12)
            )
        )
        updated_residual_dot_active = np.sum(
            refreshed_residual * refreshed_preconditioned,
            axis=0,
            dtype=np.float64,
        )
        beta_active = updated_residual_dot_active / np.maximum(residual_dot[active_columns], 1e-30)
        if np.any(~np.isfinite(beta_active) | (beta_active < 0.0)):
            raise RuntimeError("CPU conjugate-gradient preconditioner produced an invalid update.")
        search_direction[:, active_columns] = refreshed_preconditioned + (
            search_direction[:, active_columns] * beta_active[None, :]
        )
        residual_dot[active_columns] = updated_residual_dot_active

        now = time.monotonic()
        if now - last_log >= 5.0:
            progress = np.zeros_like(residual_norm_sq)
            progress[converged] = 100.0
            unconverged = ~converged
            progress[unconverged] = np.clip(
                100.0
                * (
                    np.log10(np.maximum(initial_residual_norm_sq[unconverged], 1e-30))
                    - np.log10(np.maximum(residual_norm_sq[unconverged], 1e-30))
                )
                / np.maximum(
                    np.log10(np.maximum(initial_residual_norm_sq[unconverged], 1e-30))
                    - np.log10(np.maximum(convergence_threshold_sq[unconverged], 1e-30)),
                    1e-6,
                ),
                0.0,
                100.0,
            )
            log(
                f"      CG iter {iteration_index+1}/{max_iterations}: {float(np.mean(progress)):.0f}% converged  "
                + f"active={int(np.sum(~converged))}/{rhs.shape[1]}  residual={float(np.max(residual_norm_sq)):.2e}  "
                + f"({now - t_start:.1f}s)"
            )
            last_log = now

    final_residual = float(np.max(residual_norm_sq))
    final_threshold = float(np.max(convergence_threshold_sq))
    if final_residual > final_threshold:
        raise RuntimeError(
            "CPU conjugate-gradient solve failed to converge: "
            + f"residual={final_residual:.2e} threshold={final_threshold:.2e} "
            + f"iterations={max_iterations}"
        )
    return solution[:, 0] if vector_input else solution


def _restricted_precision_projector(
    covariate_matrix: np.ndarray,
    diagonal_noise: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, Callable[[np.ndarray], np.ndarray]]:
    compute_np_dtype = gpu_compute_numpy_dtype()
    inverse_diagonal_noise = 1.0 / np.maximum(np.asarray(diagonal_noise, dtype=compute_np_dtype), 1e-12)
    weighted_covariates = inverse_diagonal_noise[:, None] * covariate_matrix
    covariate_precision = covariate_matrix.T @ weighted_covariates + np.eye(covariate_matrix.shape[1], dtype=compute_np_dtype) * 1e-8
    covariate_precision_cholesky = np.linalg.cholesky(covariate_precision)
    covariate_precision_logdet = 2.0 * float(np.sum(np.log(np.diag(covariate_precision_cholesky))))

    def apply_projector(right_hand_side: np.ndarray) -> np.ndarray:
        rhs = np.asarray(right_hand_side, dtype=compute_np_dtype)
        weighted_rhs = inverse_diagonal_noise[:, None] * rhs if rhs.ndim == 2 else inverse_diagonal_noise * rhs
        correction = weighted_covariates @ _cholesky_solve(
            covariate_precision_cholesky,
            covariate_matrix.T @ weighted_rhs,
        )
        return weighted_rhs - correction

    return inverse_diagonal_noise, covariate_precision_cholesky, covariate_precision_logdet, apply_projector


def _restricted_variant_space_operator(
    genotype_matrix: StandardizedGenotypeMatrix,
    prior_precision: np.ndarray,
    inverse_diagonal_noise: np.ndarray,
    covariate_matrix: np.ndarray,
    covariate_precision_cholesky: np.ndarray,
    batch_size: int,
):
    compute_dtype = gpu_compute_jax_dtype()
    prior_precision_jax = jnp.asarray(prior_precision, dtype=compute_dtype)
    apply_projector = _build_restricted_projector_jax(
        inverse_diagonal_noise=inverse_diagonal_noise,
        covariate_matrix=covariate_matrix,
        covariate_precision_cholesky=covariate_precision_cholesky,
        compute_dtype=compute_dtype,
    )

    def matvec(vector) -> jnp.ndarray:
        coefficients = jnp.asarray(vector, dtype=compute_dtype)
        genotype_projection = genotype_matrix.matvec(coefficients, batch_size=batch_size)
        restricted_projection = apply_projector(genotype_projection)
        return prior_precision_jax * coefficients + genotype_matrix.transpose_matvec(
            restricted_projection,
            batch_size=batch_size,
        )

    def matmat(matrix) -> jnp.ndarray:
        coefficients = jnp.asarray(matrix, dtype=compute_dtype)
        genotype_projection = genotype_matrix.matmat(coefficients, batch_size=batch_size)
        restricted_projection = apply_projector(genotype_projection)
        return prior_precision_jax[:, None] * coefficients + genotype_matrix.transpose_matmat(
            restricted_projection,
            batch_size=batch_size,
        )

    return build_linear_operator(
        shape=(genotype_matrix.shape[1], genotype_matrix.shape[1]),
        matvec=matvec,
        matmat=matmat,
        dtype=compute_dtype,
        jax_compatible=genotype_matrix.supports_jax_dense_ops(),
    )


def _build_restricted_projector_jax(
    inverse_diagonal_noise: np.ndarray,
    covariate_matrix: np.ndarray,
    covariate_precision_cholesky: np.ndarray,
    compute_dtype,
):
    inverse_diagonal_noise_jax = jnp.asarray(inverse_diagonal_noise, dtype=compute_dtype)
    covariate_matrix_jax = jnp.asarray(covariate_matrix, dtype=compute_dtype)
    covariate_precision_cholesky_jax = jnp.asarray(covariate_precision_cholesky, dtype=compute_dtype)
    weighted_covariates_jax = inverse_diagonal_noise_jax[:, None] * covariate_matrix_jax

    def apply_projector(right_hand_side: np.ndarray | jnp.ndarray) -> jnp.ndarray:
        rhs = jnp.asarray(right_hand_side, dtype=compute_dtype)
        if rhs.ndim not in (1, 2):
            raise ValueError("restricted projector expects a vector or matrix right-hand side.")
        weighted_rhs = (
            inverse_diagonal_noise_jax[:, None] * rhs
            if rhs.ndim == 2
            else inverse_diagonal_noise_jax * rhs
        )
        correction_rhs = covariate_matrix_jax.T @ weighted_rhs
        lower_solution = jax_solve_triangular(
            covariate_precision_cholesky_jax,
            correction_rhs,
            lower=True,
        )
        upper_solution = jax_solve_triangular(
            covariate_precision_cholesky_jax.T,
            lower_solution,
            lower=False,
        )
        return weighted_rhs - weighted_covariates_jax @ upper_solution

    return apply_projector


def _restricted_variant_space_diagonal_preconditioner(
    genotype_matrix: StandardizedGenotypeMatrix,
    covariate_matrix: np.ndarray,
    inverse_diagonal_noise: np.ndarray,
    covariate_precision_cholesky: np.ndarray,
    prior_precision: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    compute_np_dtype = gpu_compute_numpy_dtype()
    if genotype_matrix._cupy_cache is not None:
        import cupy as cp
        X = genotype_matrix._cupy_cache  # (n, p) float32 on GPU
        inv_d = cp.asarray(inverse_diagonal_noise, dtype=cp.float64)
        cov_gpu = cp.asarray(covariate_matrix, dtype=cp.float64)  # (n, k) — tiny
        # Compute diag(X^T D^{-1} X) and X^T D^{-1} C entirely on GPU.
        # Only download small results: (p,) diagonal and (p, k) cross-term.
        raw_diag = cp.zeros(X.shape[1], dtype=cp.float64)
        weighted_Xt_gpu = cp.empty((X.shape[1], cov_gpu.shape[1]), dtype=cp.float64)
        chunk = max(1, min(512, X.shape[1]))
        for start in range(0, X.shape[1], chunk):
            end = min(start + chunk, X.shape[1])
            weighted_chunk = X[:, start:end] * inv_d[:, None]  # (n, chunk) on GPU
            raw_diag[start:end] = cp.sum(X[:, start:end] * weighted_chunk, axis=0)
            weighted_Xt_gpu[start:end, :] = weighted_chunk.T @ cov_gpu  # (chunk, k) on GPU
        # Download only the small results: (p,) + (p, k) ≈ 120 KB
        raw_diag_np = raw_diag.get().astype(compute_np_dtype)
        cross = weighted_Xt_gpu.get().astype(compute_np_dtype).T  # (k, p)
        correction = _cholesky_solve(covariate_precision_cholesky, cross)
        diag_correction = np.sum(cross * correction, axis=0)
        return np.maximum(prior_precision + raw_diag_np - diag_correction, 1e-8)
    if genotype_matrix.raw is not None and not genotype_matrix.supports_jax_dense_ops():
        cupy = _try_import_cupy()
        if cupy is not None:
            inv_d_gpu = cupy.asarray(inverse_diagonal_noise, dtype=cupy.float64)
            cov_gpu = cupy.asarray(covariate_matrix, dtype=cupy.float64)
            raw_diag_gpu = cupy.empty(genotype_matrix.shape[1], dtype=cupy.float64)
            weighted_xt_gpu = cupy.empty((genotype_matrix.shape[1], cov_gpu.shape[1]), dtype=cupy.float64)
            for batch_slice, standardized_batch in _iter_standardized_gpu_batches(
                genotype_matrix.raw,
                genotype_matrix.variant_indices,
                genotype_matrix.means,
                genotype_matrix.scales,
                batch_size=batch_size,
                cupy=cupy,
                dtype=cupy.float64,
            ):
                weighted_batch = inv_d_gpu[:, None] * standardized_batch
                raw_diag_gpu[batch_slice] = cupy.sum(standardized_batch * weighted_batch, axis=0)
                weighted_xt_gpu[batch_slice, :] = weighted_batch.T @ cov_gpu
            raw_diag_np = raw_diag_gpu.get().astype(compute_np_dtype)
            cross = weighted_xt_gpu.get().astype(compute_np_dtype).T
            correction = _cholesky_solve(covariate_precision_cholesky, cross)
            diag_correction = np.sum(cross * correction, axis=0)
            return np.maximum(prior_precision + raw_diag_np - diag_correction, 1e-8)
    diagonal = np.asarray(prior_precision, dtype=compute_np_dtype).copy()
    for batch in genotype_matrix.iter_column_batches(batch_size=batch_size):
        genotype_batch = np.asarray(batch.values, dtype=compute_np_dtype)
        weighted_batch = inverse_diagonal_noise[:, None] * genotype_batch
        cross_terms = covariate_matrix.T @ weighted_batch
        correction = _cholesky_solve(covariate_precision_cholesky, cross_terms)
        diagonal[batch.variant_indices] += np.sum(genotype_batch * weighted_batch, axis=0) - np.sum(
            cross_terms * correction,
            axis=0,
        )
    return np.maximum(diagonal, 1e-8)


def _posterior_variance_hutchinson_diagonal(
    solve_variant_rhs: Callable[[np.ndarray], np.ndarray],
    dimension: int,
    probe_count: int,
    random_seed: int,
) -> np.ndarray:
    if probe_count < 1:
        raise ValueError("probe_count must be positive.")
    if probe_count == 1:
        probes = _orthogonal_probe_matrix(
            dimension=dimension,
            probe_count=probe_count,
            random_seed=random_seed,
        )
        solutions = np.asarray(solve_variant_rhs(probes), dtype=np.float64)
        return np.maximum(np.mean(probes * solutions, axis=1), 1e-8)

    sketch_probe_count = min(max(1, probe_count // 2), dimension)
    residual_probe_count = max(probe_count - sketch_probe_count, 0)
    sketch_probes = _orthogonal_probe_matrix(
        dimension=dimension,
        probe_count=sketch_probe_count,
        random_seed=random_seed,
    )
    residual_probes = (
        _orthogonal_probe_matrix(
            dimension=dimension,
            probe_count=residual_probe_count,
            random_seed=random_seed + 1,
        )
        if residual_probe_count > 0
        else None
    )
    all_probes = (
        np.concatenate([sketch_probes, residual_probes], axis=1)
        if residual_probes is not None
        else sketch_probes
    )
    all_solutions = np.asarray(solve_variant_rhs(all_probes), dtype=np.float64)
    sketch_solutions = np.asarray(all_solutions[:, :sketch_probe_count], dtype=np.float64)
    sketch_gram = np.asarray(sketch_probes.T @ sketch_solutions, dtype=np.float64)
    sketch_gram = 0.5 * (sketch_gram + sketch_gram.T)
    sketch_gram += np.eye(sketch_gram.shape[0], dtype=np.float64) * 1e-8
    try:
        sketch_gram_inverse = np.linalg.inv(sketch_gram)
    except np.linalg.LinAlgError:
        sketch_gram_inverse = np.linalg.pinv(sketch_gram, rcond=1e-10)
    low_rank_weighted_solutions = sketch_solutions @ sketch_gram_inverse
    low_rank_diagonal = np.sum(low_rank_weighted_solutions * sketch_solutions, axis=1)
    if residual_probe_count == 0 or residual_probes is None:
        return np.maximum(low_rank_diagonal, 1e-8)
    residual_solutions = np.asarray(all_solutions[:, sketch_probe_count:], dtype=np.float64)
    low_rank_residual_projection = sketch_solutions @ (
        sketch_gram_inverse @ (sketch_solutions.T @ residual_probes)
    )
    residual_diagonal = np.mean(
        residual_probes * (residual_solutions - low_rank_residual_projection),
        axis=1,
    )
    return np.maximum(low_rank_diagonal + residual_diagonal, 1e-8)


def _orthogonal_probe_matrix(
    dimension: int,
    probe_count: int,
    random_seed: int,
) -> np.ndarray:
    if dimension < 1:
        raise ValueError("dimension must be positive.")
    if probe_count < 1:
        raise ValueError("probe_count must be positive.")
    random_generator = np.random.default_rng(random_seed)
    probe_blocks: list[np.ndarray] = []
    probes_remaining = int(probe_count)
    while probes_remaining > 0:
        block_probe_count = min(probes_remaining, dimension)
        gaussian_block = random_generator.standard_normal((dimension, block_probe_count)).astype(np.float64)
        orthogonal_block, triangular_block = np.linalg.qr(gaussian_block, mode="reduced")
        diagonal_signs = np.sign(np.diag(triangular_block))
        diagonal_signs[diagonal_signs == 0.0] = 1.0
        orthogonal_block *= diagonal_signs[None, :]
        probe_blocks.append(np.sqrt(float(dimension)) * orthogonal_block[:, :block_probe_count])
        probes_remaining -= block_probe_count
    return np.ascontiguousarray(np.column_stack(probe_blocks), dtype=np.float64)


def _use_gpu_exact_variant_solve(
    genotype_matrix: StandardizedGenotypeMatrix,
    variant_count: int,
    exact_solver_matrix_limit: int,
) -> bool:
    return (
        genotype_matrix._cupy_cache is not None
        and variant_count > exact_solver_matrix_limit
        and variant_count <= GPU_EXACT_VARIANT_SOLVE_LIMIT
    )


def _restricted_posterior_state(
    genotype_matrix: StandardizedGenotypeMatrix,
    covariate_matrix: np.ndarray,
    targets: np.ndarray,
    prior_variances: np.ndarray,
    diagonal_noise: np.ndarray,
    solver_tolerance: float,
    maximum_linear_solver_iterations: int,
    logdet_probe_count: int,
    logdet_lanczos_steps: int,
    exact_solver_matrix_limit: int,
    posterior_variance_batch_size: int,
    posterior_variance_probe_count: int,
    random_seed: int,
    compute_logdet: bool,
    compute_beta_variance: bool = True,
    initial_beta_guess: np.ndarray | None = None,
    sample_space_preconditioner_rank: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    from sv_pgs.progress import log, mem
    compute_jax_dtype = gpu_compute_jax_dtype()
    compute_np_dtype = gpu_compute_numpy_dtype()
    sample_count = genotype_matrix.shape[0]
    diagonal_noise = np.asarray(diagonal_noise, dtype=np.float64)
    if diagonal_noise.shape != (sample_count,):
        raise ValueError("diagonal_noise must have one entry per sample.")

    prior_variances = np.maximum(np.asarray(prior_variances, dtype=np.float64), 1e-8)
    prior_precision = 1.0 / prior_variances
    variant_count = genotype_matrix.shape[1]
    use_exact_sample = sample_count <= exact_solver_matrix_limit
    use_exact_variant = variant_count <= exact_solver_matrix_limit
    use_gpu_exact_variant = _use_gpu_exact_variant_solve(
        genotype_matrix=genotype_matrix,
        variant_count=variant_count,
        exact_solver_matrix_limit=exact_solver_matrix_limit,
    )
    prefer_iterative_variant_space = _prefer_iterative_variant_space(
        genotype_matrix=genotype_matrix,
        sample_count=sample_count,
        variant_count=variant_count,
        compute_beta_variance=compute_beta_variance,
        compute_logdet=compute_logdet,
        initial_beta_guess=initial_beta_guess,
    )
    use_variant_space = (not use_exact_sample) and (
        use_exact_variant
        or use_gpu_exact_variant
        or variant_count <= sample_count
        or prefer_iterative_variant_space
    )

    if use_exact_sample:
        log(f"    restricted posterior: exact sample-space Cholesky for n={sample_count}")
        covariance_matrix = np.diag(diagonal_noise)
        for batch in genotype_matrix.iter_column_batches(batch_size=posterior_variance_batch_size):
            genotype_batch = np.asarray(batch.values, dtype=np.float64)
            covariance_matrix += (genotype_batch * prior_variances[batch.variant_indices][None, :]) @ genotype_batch.T
        covariance_matrix += np.eye(sample_count, dtype=np.float64) * 1e-8
        cholesky_factor = np.linalg.cholesky(covariance_matrix)

        def solve_rhs(right_hand_side: np.ndarray) -> np.ndarray:
            return _cholesky_solve(cholesky_factor, np.asarray(right_hand_side, dtype=np.float64))

        inverse_covariance_rhs = solve_rhs(
            np.concatenate([targets[:, None], covariate_matrix], axis=1),
        )
        inverse_covariance_targets = np.asarray(inverse_covariance_rhs[:, 0], dtype=np.float64)
        inverse_covariance_covariates = np.asarray(inverse_covariance_rhs[:, 1:], dtype=np.float64)
        logdet_covariance = 2.0 * float(np.sum(np.log(np.diag(cholesky_factor)))) if compute_logdet else 0.0
        gls_normal_matrix = covariate_matrix.T @ inverse_covariance_covariates + np.eye(covariate_matrix.shape[1]) * 1e-8
        gls_cholesky = np.linalg.cholesky(gls_normal_matrix)
        alpha = np.asarray(
            _cholesky_solve(gls_cholesky, covariate_matrix.T @ inverse_covariance_targets),
            dtype=np.float64,
        )
        projected_targets = np.asarray(
            inverse_covariance_targets - inverse_covariance_covariates @ alpha,
            dtype=np.float64,
        )
        beta = np.asarray(
            prior_variances * np.asarray(genotype_matrix.transpose_matvec(projected_targets), dtype=np.float64),
            dtype=np.float64,
        )
        if compute_beta_variance:
            log(f"    computing leverage diagonal for beta variance ({variant_count} variants)...")
            leverage_diagonal = _restricted_cross_leverage_diagonal(
                genotype_matrix=genotype_matrix,
                covariate_matrix=covariate_matrix,
                solve_rhs=solve_rhs,
                inverse_covariance_covariates=inverse_covariance_covariates,
                gls_cholesky=gls_cholesky,
                batch_size=posterior_variance_batch_size,
            )
            beta_variance = np.maximum(prior_variances - (prior_variances * prior_variances) * leverage_diagonal, 1e-8)
        else:
            beta_variance = np.zeros_like(prior_variances, dtype=np.float64)
        linear_predictor = covariate_matrix @ alpha + np.asarray(
            genotype_matrix.matvec(beta, batch_size=posterior_variance_batch_size),
            dtype=np.float64,
        )
        sign_gls, logdet_gls = np.linalg.slogdet(gls_normal_matrix)
        if sign_gls <= 0.0:
            raise RuntimeError("Restricted GLS normal matrix is not positive definite.")
        restricted_quadratic = float(np.dot(targets, projected_targets))
        log(f"    posterior beta computed: max|beta|={float(np.max(np.abs(beta))):.4f}  mean|beta|={float(np.mean(np.abs(beta))):.6f}  logdet_cov={float(logdet_covariance):.4f}  mem={mem()}")
        return (
            np.asarray(alpha, dtype=np.float64),
            np.asarray(beta, dtype=np.float64),
            np.asarray(beta_variance, dtype=np.float64),
            np.asarray(projected_targets, dtype=np.float64),
            np.asarray(linear_predictor, dtype=np.float64),
            restricted_quadratic,
            float(logdet_covariance),
            float(logdet_gls),
        )

    inverse_diagonal_noise, covariate_precision_cholesky, covariate_precision_logdet, apply_projector = (
        _restricted_precision_projector(covariate_matrix, diagonal_noise)
    )
    apply_projector_jax = _build_restricted_projector_jax(
        inverse_diagonal_noise=inverse_diagonal_noise,
        covariate_matrix=covariate_matrix,
        covariate_precision_cholesky=covariate_precision_cholesky,
        compute_dtype=compute_jax_dtype,
    )

    if use_variant_space:
        if prefer_iterative_variant_space and not (use_exact_variant or use_gpu_exact_variant):
            log(
                "    restricted posterior: forcing iterative variant-space solve "
                + f"for warm-started binary update (p={variant_count}, n={sample_count})  mem={mem()}"
            )
        if use_exact_variant or use_gpu_exact_variant:
            log(f"    restricted posterior: exact variant-space Cholesky (p={variant_count}, n={sample_count})  mem={mem()}")
            # Use CuPy GPU path if available to avoid downloading 4 GB from GPU
            if genotype_matrix._cupy_cache is not None:
                import cupy as cp
                from cupyx.scipy.linalg import solve_triangular as cp_solve_triangular
                X_gpu = genotype_matrix._cupy_cache  # (n, p) float32 on GPU
                # Apply projector on GPU: P = D^{-1} - D^{-1} C (C^T D^{-1} C)^{-1} C^T D^{-1}
                # Do not materialize projected_X (another n x p matrix); on a T4 that
                # blows VRAM. Use:
                #   X^T P X = X^T D^{-1} X - (X^T D^{-1} C) A^{-1} (C^T D^{-1} X)
                # where A = C^T D^{-1} C.
                inv_d_cp = cp.asarray(inverse_diagonal_noise, dtype=cp.float64)
                cov_cp = cp.asarray(covariate_matrix, dtype=cp.float64)
                chunk = 256
                xtdxc_gpu = cp.empty((variant_count, cov_cp.shape[1]), dtype=cp.float64)
                xtdx_gpu = cp.empty((variant_count, variant_count), dtype=cp.float64)
                for start in range(0, variant_count, chunk):
                    end = min(start + chunk, variant_count)
                    weighted_chunk = inv_d_cp[:, None] * X_gpu[:, start:end].astype(cp.float64, copy=False)
                    xtdxc_gpu[start:end, :] = weighted_chunk.T @ cov_cp
                    xtdx_gpu[:, start:end] = X_gpu.astype(cp.float64, copy=False).T @ weighted_chunk
                CtWX = xtdxc_gpu.T.get().astype(np.float64)  # (k, p)
                correction_coeff = _cholesky_solve(covariate_precision_cholesky, CtWX)
                projected_targets_np = apply_projector(targets)
                projected_targets_cp = cp.asarray(projected_targets_np, dtype=cp.float64)
                variant_rhs_cp = X_gpu.astype(cp.float64, copy=False).T @ projected_targets_cp
                if use_gpu_exact_variant:
                    correction_gpu = xtdxc_gpu @ cp.asarray(correction_coeff, dtype=cp.float64)
                    prior_precision_cp = cp.asarray(prior_precision, dtype=cp.float64)
                    variant_precision_gpu = xtdx_gpu - correction_gpu
                    variant_precision_gpu += cp.diag(prior_precision_cp)
                    variant_precision_gpu = (variant_precision_gpu + variant_precision_gpu.T) * 0.5
                    variant_precision_gpu += cp.eye(variant_count, dtype=cp.float64) * 1e-8
                    variant_precision_cholesky_gpu = cp.linalg.cholesky(variant_precision_gpu)

                    def solve_variant_rhs(right_hand_side: np.ndarray) -> np.ndarray:
                        return np.asarray(
                            _gpu_cholesky_solve(
                                right_hand_side,
                                variant_precision_cholesky_gpu,
                                cp_solve_triangular,
                            ).get(),
                            dtype=np.float64,
                        )

                    beta_cp = _gpu_cholesky_solve(
                        variant_rhs_cp,
                        variant_precision_cholesky_gpu,
                        cp_solve_triangular,
                    )
                    beta = np.asarray(beta_cp.get(), dtype=np.float64)
                    beta_variance = (
                        _posterior_variance_hutchinson_diagonal(
                            solve_variant_rhs=solve_variant_rhs,
                            dimension=variant_count,
                            probe_count=posterior_variance_probe_count,
                            random_seed=random_seed,
                        )
                        if compute_beta_variance
                        else np.zeros(variant_count, dtype=np.float64)
                    )
                    logdet_A = (
                        2.0 * float(cp.sum(cp.log(cp.diag(variant_precision_cholesky_gpu))).get())
                        if compute_logdet
                        else 0.0
                    )
                    genetic_linear_predictor = _gpu_exact_variant_linear_predictor(X_gpu, beta)
                    del (
                        beta_cp,
                        correction_gpu,
                        prior_precision_cp,
                        projected_targets_cp,
                        xtdxc_gpu,
                        xtdx_gpu,
                        variant_precision_gpu,
                        variant_precision_cholesky_gpu,
                        variant_rhs_cp,
                    )
                else:
                    correction_cpu = CtWX.T @ correction_coeff
                    XtPX = np.asarray(xtdx_gpu.get(), dtype=np.float64) - correction_cpu
                    variant_rhs = np.asarray(variant_rhs_cp.get(), dtype=np.float64)
                    del projected_targets_cp, xtdxc_gpu, xtdx_gpu, variant_rhs_cp
            else:
                dense_genotypes = np.asarray(genotype_matrix.materialize(batch_size=posterior_variance_batch_size), dtype=np.float64)
                projected_genotypes = apply_projector(dense_genotypes)
                XtPX = dense_genotypes.T @ projected_genotypes
                variant_rhs = dense_genotypes.T @ apply_projector(targets)
            if not use_gpu_exact_variant:
                variant_precision_matrix = (
                    np.diag(prior_precision) + XtPX + np.eye(variant_count, dtype=np.float64) * 1e-8
                )
                variant_precision_cholesky = np.linalg.cholesky(variant_precision_matrix)

                def solve_variant_rhs(right_hand_side: np.ndarray) -> np.ndarray:
                    return _cholesky_solve(variant_precision_cholesky, np.asarray(right_hand_side, dtype=np.float64))

                beta = np.asarray(solve_variant_rhs(variant_rhs), dtype=np.float64)
                beta_variance = (
                    np.maximum(np.diag(solve_variant_rhs(np.eye(variant_count, dtype=np.float64))), 1e-8)
                    if compute_beta_variance and variant_count <= exact_solver_matrix_limit
                    else (
                        _posterior_variance_hutchinson_diagonal(
                            solve_variant_rhs=solve_variant_rhs,
                            dimension=variant_count,
                            probe_count=posterior_variance_probe_count,
                            random_seed=random_seed,
                        )
                        if compute_beta_variance
                        else np.zeros(variant_count, dtype=np.float64)
                    )
                )
                logdet_A = 2.0 * float(np.sum(np.log(np.diag(variant_precision_cholesky)))) if compute_logdet else 0.0
                if genotype_matrix._cupy_cache is not None:
                    genetic_linear_predictor = _gpu_exact_variant_linear_predictor(X_gpu, beta)
                else:
                    genetic_linear_predictor = np.asarray(genotype_matrix.matvec(beta), dtype=np.float64)
        else:
            import time as _time
            log(f"    restricted posterior: PCG variant-space solve (p={variant_count}, n={sample_count})  mem={mem()}")
            _t0 = _time.monotonic()
            variant_operator = _restricted_variant_space_operator(
                genotype_matrix=genotype_matrix,
                prior_precision=prior_precision,
                inverse_diagonal_noise=inverse_diagonal_noise,
                covariate_matrix=covariate_matrix,
                covariate_precision_cholesky=covariate_precision_cholesky,
                batch_size=posterior_variance_batch_size,
            )
            log(f"      operator setup: {_time.monotonic() - _t0:.1f}s  mem={mem()}")
            _t0 = _time.monotonic()
            variant_preconditioner = _restricted_variant_space_diagonal_preconditioner(
                genotype_matrix=genotype_matrix,
                covariate_matrix=covariate_matrix,
                inverse_diagonal_noise=inverse_diagonal_noise,
                covariate_precision_cholesky=covariate_precision_cholesky,
                prior_precision=prior_precision,
                batch_size=posterior_variance_batch_size,
            )
            log(f"      preconditioner: {_time.monotonic() - _t0:.1f}s  mem={mem()}")
            _t0 = _time.monotonic()
            restricted_targets = apply_projector_jax(targets)
            variant_rhs = np.asarray(
                genotype_matrix.transpose_matvec(
                    restricted_targets,
                    batch_size=posterior_variance_batch_size,
                ),
                dtype=np.float64,
            )
            log(f"      rhs: {_time.monotonic() - _t0:.1f}s  mem={mem()}")

            def solve_variant_rhs(right_hand_side: np.ndarray) -> np.ndarray:
                rhs_array = np.asarray(right_hand_side, dtype=np.float64)
                return solve_spd_system(
                    variant_operator,
                    rhs_array,
                    tolerance=solver_tolerance,
                    max_iterations=maximum_linear_solver_iterations,
                    initial_guess=initial_beta_guess if rhs_array.ndim == 1 else None,
                    preconditioner=variant_preconditioner,
                )

            beta = np.asarray(solve_variant_rhs(variant_rhs), dtype=np.float64)
            if compute_beta_variance:
                beta_variance = _posterior_variance_hutchinson_diagonal(
                    solve_variant_rhs=solve_variant_rhs,
                    dimension=variant_count,
                    probe_count=posterior_variance_probe_count,
                    random_seed=random_seed,
                )
            else:
                beta_variance = np.zeros(variant_count, dtype=compute_np_dtype)
            logdet_A = (
                stochastic_logdet(
                    variant_operator,
                    dimension=variant_count,
                    probe_count=logdet_probe_count,
                    lanczos_steps=logdet_lanczos_steps,
                    random_seed=random_seed,
                )
                if compute_logdet
                else 0.0
            )
            genetic_linear_predictor = np.asarray(
                genotype_matrix.matvec(beta, batch_size=posterior_variance_batch_size),
                dtype=compute_np_dtype,
            )

        alpha = np.asarray(
            _cholesky_solve(
                covariate_precision_cholesky,
                covariate_matrix.T @ (inverse_diagonal_noise * (targets - genetic_linear_predictor)),
            ),
            dtype=np.float64,
        )
        projected_targets = np.asarray(apply_projector(targets - genetic_linear_predictor), dtype=np.float64)
        linear_predictor = covariate_matrix @ alpha + genetic_linear_predictor
        restricted_quadratic = float(np.dot(targets, projected_targets))
        logdet_covariance = (
            float(np.sum(np.log(np.maximum(diagonal_noise, 1e-12))))
            - float(np.sum(np.log(np.maximum(prior_precision, 1e-12))))
            + logdet_A
            if compute_logdet
            else 0.0
        )
        logdet_gls = covariate_precision_logdet if compute_logdet else 0.0
        log(f"    posterior beta computed: max|beta|={float(np.max(np.abs(beta))):.4f}  mean|beta|={float(np.mean(np.abs(beta))):.6f}  logdet_cov={float(logdet_covariance):.4f}  mem={mem()}")
        return (
            np.asarray(alpha, dtype=np.float64),
            np.asarray(beta, dtype=np.float64),
            np.asarray(beta_variance, dtype=np.float64),
            np.asarray(projected_targets, dtype=np.float64),
            np.asarray(linear_predictor, dtype=np.float64),
            restricted_quadratic,
            float(logdet_covariance),
            float(logdet_gls),
        )

    effective_sample_space_preconditioner_rank = _effective_sample_space_preconditioner_rank(
        genotype_matrix=genotype_matrix,
        sample_count=sample_count,
        variant_count=variant_count,
        requested_rank=sample_space_preconditioner_rank,
    )
    sample_space_gpu_enabled = genotype_matrix._cupy_cache is not None or (
        genotype_matrix.raw is not None
        and not genotype_matrix.supports_jax_dense_ops()
        and _try_import_cupy() is not None
    )
    if sample_space_gpu_enabled:
        gpu_source = "full-cache" if genotype_matrix._cupy_cache is not None else "streaming"
        log(
            "    restricted posterior: GPU block-CG sample-space solve "
            + f"(p={variant_count}, n={sample_count}, source={gpu_source})"
        )
        sample_space_preconditioner_gpu = _sample_space_gpu_preconditioner(
            genotype_matrix=genotype_matrix,
            prior_variances=prior_variances,
            diagonal_noise=diagonal_noise,
            batch_size=posterior_variance_batch_size,
            rank=effective_sample_space_preconditioner_rank,
            random_seed=random_seed,
        )

        def solve_rhs_iterative(right_hand_side: np.ndarray) -> np.ndarray:
            return _solve_sample_space_rhs_gpu(
                genotype_matrix=genotype_matrix,
                prior_variances=prior_variances,
                diagonal_noise=diagonal_noise,
                right_hand_side=right_hand_side,
                tolerance=solver_tolerance,
                max_iterations=maximum_linear_solver_iterations,
                preconditioner=sample_space_preconditioner_gpu,
                batch_size=posterior_variance_batch_size,
            )
    else:
        log(
            "    restricted posterior: CPU block-PCG sample-space solve "
            + f"(p={variant_count}, n={sample_count}, preconditioner_rank={effective_sample_space_preconditioner_rank})"
        )
        sample_space_preconditioner = _sample_space_preconditioner(
            genotype_matrix=genotype_matrix,
            prior_variances=prior_variances,
            diagonal_noise=diagonal_noise,
            batch_size=posterior_variance_batch_size,
            rank=effective_sample_space_preconditioner_rank,
            random_seed=random_seed,
        )

        def solve_rhs_iterative(right_hand_side: np.ndarray) -> np.ndarray:
            return _solve_sample_space_rhs_cpu(
                genotype_matrix=genotype_matrix,
                prior_variances=prior_variances,
                diagonal_noise=diagonal_noise,
                right_hand_side=right_hand_side,
                tolerance=solver_tolerance,
                max_iterations=maximum_linear_solver_iterations,
                preconditioner=sample_space_preconditioner,
                batch_size=posterior_variance_batch_size,
            )

    sample_probes = (
        _orthogonal_probe_matrix(
            dimension=sample_count,
            probe_count=posterior_variance_probe_count,
            random_seed=random_seed,
        )
        if compute_beta_variance
        else None
    )
    solve_rhs_blocks = [targets[:, None], covariate_matrix]
    if sample_probes is not None:
        solve_rhs_blocks.append(sample_probes)
    inverse_covariance_rhs = solve_rhs_iterative(np.concatenate(solve_rhs_blocks, axis=1))
    inverse_covariance_targets = np.asarray(inverse_covariance_rhs[:, 0], dtype=np.float64)
    inverse_covariance_covariates = np.asarray(
        inverse_covariance_rhs[:, 1 : 1 + covariate_matrix.shape[1]],
        dtype=np.float64,
    )
    inverse_covariance_probe_matrix = (
        np.asarray(inverse_covariance_rhs[:, 1 + covariate_matrix.shape[1] :], dtype=np.float64)
        if sample_probes is not None
        else None
    )
    logdet_covariance = (
        stochastic_logdet(
            _sample_space_operator(
                genotype_matrix,
                prior_variances,
                diagonal_noise,
                batch_size=posterior_variance_batch_size,
            ),
            dimension=sample_count,
            probe_count=logdet_probe_count,
            lanczos_steps=logdet_lanczos_steps,
            random_seed=random_seed,
        )
        if compute_logdet
        else 0.0
    )
    gls_normal_matrix = covariate_matrix.T @ inverse_covariance_covariates + np.eye(covariate_matrix.shape[1]) * 1e-8
    gls_cholesky = np.linalg.cholesky(gls_normal_matrix)
    alpha = np.asarray(
        _cholesky_solve(gls_cholesky, covariate_matrix.T @ inverse_covariance_targets),
        dtype=np.float64,
    )
    projected_targets = np.asarray(
        inverse_covariance_targets - inverse_covariance_covariates @ alpha,
        dtype=np.float64,
    )
    beta = np.asarray(
        prior_variances * np.asarray(genotype_matrix.transpose_matvec(projected_targets), dtype=compute_np_dtype),
        dtype=compute_np_dtype,
    )
    if compute_beta_variance:
        log(f"    computing stochastic leverage diagonal for beta variance ({variant_count} variants, {posterior_variance_probe_count} probes)...")
        leverage_diagonal = _stochastic_restricted_cross_leverage_diagonal(
            genotype_matrix=genotype_matrix,
            covariate_matrix=covariate_matrix,
            solve_rhs=solve_rhs_iterative,
            inverse_covariance_covariates=inverse_covariance_covariates,
            gls_cholesky=gls_cholesky,
            batch_size=posterior_variance_batch_size,
            probe_count=posterior_variance_probe_count,
            random_seed=random_seed,
            sample_probes=sample_probes,
            inverse_covariance_probe_matrix=inverse_covariance_probe_matrix,
        )
        beta_variance = np.maximum(prior_variances - (prior_variances * prior_variances) * leverage_diagonal, 1e-8)
    else:
        beta_variance = np.zeros_like(prior_variances, dtype=np.float64)
    linear_predictor = covariate_matrix @ alpha + np.asarray(
        genotype_matrix.matvec(beta, batch_size=posterior_variance_batch_size),
        dtype=compute_np_dtype,
    )
    sign_gls, logdet_gls = np.linalg.slogdet(gls_normal_matrix)
    if sign_gls <= 0.0:
        raise RuntimeError("Restricted GLS normal matrix is not positive definite.")
    restricted_quadratic = float(np.dot(targets, projected_targets))
    log(f"    posterior beta computed: max|beta|={float(np.max(np.abs(beta))):.4f}  mean|beta|={float(np.mean(np.abs(beta))):.6f}  logdet_cov={float(logdet_covariance):.4f}  mem={mem()}")
    return (
        np.asarray(alpha, dtype=np.float64),
        np.asarray(beta, dtype=np.float64),
        np.asarray(beta_variance, dtype=np.float64),
        np.asarray(projected_targets, dtype=np.float64),
        np.asarray(linear_predictor, dtype=np.float64),
        restricted_quadratic,
        float(logdet_covariance),
        float(logdet_gls),
    )


# Compute the "leverage" of each variant — how much influence it has on
# its own predicted value.  High leverage = the model is very confident
# about this variant's effect.  We need this to compute posterior variance
# (uncertainty) for each beta: Var(beta_j) = tau_j^2 - tau_j^4 * h_j
# where h_j is the leverage.  Computed in batches to limit memory usage.
def _restricted_cross_leverage_diagonal(
    genotype_matrix: StandardizedGenotypeMatrix,
    covariate_matrix: np.ndarray,
    solve_rhs,
    inverse_covariance_covariates: np.ndarray,
    gls_cholesky: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    from sv_pgs.progress import log, mem
    variant_count = genotype_matrix.shape[1]
    leverage_diagonal = np.zeros(variant_count, dtype=np.float64)
    variants_done = 0
    for batch in genotype_matrix.iter_column_batches(batch_size=batch_size):
        genotype_batch = np.asarray(batch.values, dtype=np.float64)
        inverse_covariance_genotype_batch = solve_rhs(genotype_batch)
        restricted_batch = inverse_covariance_genotype_batch - inverse_covariance_covariates @ _cholesky_solve(
            gls_cholesky,
            covariate_matrix.T @ inverse_covariance_genotype_batch,
        )
        leverage_diagonal[batch.variant_indices] = np.asarray(
            jnp.sum(jnp.asarray(genotype_batch) * jnp.asarray(restricted_batch), axis=0),
            dtype=np.float64,
        )
        variants_done += len(batch.variant_indices)
        if variants_done == len(batch.variant_indices) or variants_done % max(variant_count // 5, 1) < len(batch.variant_indices) or variants_done == variant_count:
            log(f"      leverage diagonal: {variants_done}/{variant_count} ({100*variants_done//max(variant_count,1)}%)  mem={mem()}")
    return leverage_diagonal


def _stochastic_restricted_cross_leverage_diagonal(
    genotype_matrix: StandardizedGenotypeMatrix,
    covariate_matrix: np.ndarray,
    solve_rhs,
    inverse_covariance_covariates: np.ndarray,
    gls_cholesky: np.ndarray,
    batch_size: int,
    probe_count: int,
    random_seed: int,
    sample_probes: np.ndarray | None = None,
    inverse_covariance_probe_matrix: np.ndarray | None = None,
) -> np.ndarray:
    if sample_probes is None:
        sample_probes = _orthogonal_probe_matrix(
            dimension=genotype_matrix.shape[0],
            probe_count=probe_count,
            random_seed=random_seed,
        )
    else:
        sample_probes = np.asarray(sample_probes, dtype=np.float64)
        if sample_probes.shape != (genotype_matrix.shape[0], probe_count):
            raise ValueError("sample_probes shape does not match sample_count/probe_count.")
    if inverse_covariance_probe_matrix is None:
        inverse_covariance_probe_matrix = np.asarray(solve_rhs(sample_probes), dtype=np.float64)
    else:
        inverse_covariance_probe_matrix = np.asarray(inverse_covariance_probe_matrix, dtype=np.float64)
        if inverse_covariance_probe_matrix.shape != sample_probes.shape:
            raise ValueError("inverse_covariance_probe_matrix shape does not match sample_probes.")
    restricted_probe_matrix = inverse_covariance_probe_matrix - inverse_covariance_covariates @ _cholesky_solve(
        gls_cholesky,
        covariate_matrix.T @ inverse_covariance_probe_matrix,
    )
    probe_projection_matrix = np.asarray(
        genotype_matrix.transpose_matmat(sample_probes, batch_size=batch_size),
        dtype=np.float64,
    )
    restricted_probe_projection_matrix = np.asarray(
        genotype_matrix.transpose_matmat(restricted_probe_matrix, batch_size=batch_size),
        dtype=np.float64,
    )
    return np.maximum(
        np.mean(probe_projection_matrix * restricted_probe_projection_matrix, axis=1),
        1e-8,
    )


# Build the metadata design matrix for the prior scale model.
#
# Each variant gets a row of features describing its properties:
#   - Type indicators (one per variant class: SNV, deletion, duplication, ...)
#   - log(length) polynomial (linear + quadratic) per class — longer SVs
#     may have different effect size distributions
#   - Repeat region indicator, copy number indicator
#
# These features let the model learn, e.g., "long deletions in repeat
# regions tend to have larger effects" and set prior variances accordingly.
def _build_prior_design(records: Sequence[VariantRecord]) -> PriorDesign:
    unique_classes = sorted(
        {
            prior_class
            for record in records
            for prior_class in record.prior_class_members
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
        prior_classes = record.prior_class_members
        prior_membership = record.prior_class_membership
        for prior_class, prior_weight in zip(prior_classes, prior_membership, strict=True):
            class_membership_matrix[record_index, class_lookup[prior_class]] = prior_weight
    class_membership_totals = np.sum(class_membership_matrix, axis=0)

    continuous_feature_names, continuous_feature_matrix = _continuous_prior_design_matrix(records)
    repeat_indicator = np.asarray([float(record.is_repeat) for record in records], dtype=np.float64)
    copy_number_indicator = np.asarray([float(record.is_copy_number) for record in records], dtype=np.float64)

    design_columns: list[np.ndarray] = []
    feature_names: list[str] = []
    feature_specs: list[ScaleModelFeatureSpec] = []
    for class_index, variant_class in enumerate(unique_classes):
        class_membership = class_membership_matrix[:, class_index]
        design_columns.append(_center_design_column(class_membership))
        feature_names.append("type_offset::" + variant_class.value)
        feature_specs.append(ScaleModelFeatureSpec(kind="type_offset", variant_class=variant_class))

        if class_membership_totals[class_index] < 3.0:
            continue

        for continuous_feature_index, continuous_feature_name in enumerate(continuous_feature_names):
            standardized_values = continuous_feature_matrix[:, continuous_feature_index]
            design_columns.append(_center_design_column(class_membership * standardized_values))
            feature_names.append(f"continuous_linear::{continuous_feature_name}::{variant_class.value}")
            feature_specs.append(
                ScaleModelFeatureSpec(
                    kind="continuous_linear",
                    variant_class=variant_class,
                    continuous_feature_name=continuous_feature_name,
                )
            )
            design_columns.append(_center_design_column(class_membership * standardized_values * standardized_values))
            feature_names.append(f"continuous_quadratic::{continuous_feature_name}::{variant_class.value}")
            feature_specs.append(
                ScaleModelFeatureSpec(
                    kind="continuous_quadratic",
                    variant_class=variant_class,
                    continuous_feature_name=continuous_feature_name,
                )
            )
    design_columns.append(_center_design_column(repeat_indicator))
    feature_names.append("repeat_indicator")
    feature_specs.append(ScaleModelFeatureSpec(kind="repeat_indicator"))
    design_columns.append(_center_design_column(copy_number_indicator))
    feature_names.append("copy_number_indicator")
    feature_specs.append(ScaleModelFeatureSpec(kind="copy_number_indicator"))

    return PriorDesign(
        design_matrix=np.column_stack(design_columns).astype(np.float64),
        feature_names=feature_names,
        feature_specs=tuple(feature_specs),
        class_membership_matrix=class_membership_matrix,
        inverse_class_lookup=inverse_class_lookup,
    )
def _design_matrix_for_feature_specs(
    records: Sequence[VariantRecord],
    feature_specs: Sequence[ScaleModelFeatureSpec],
) -> np.ndarray:
    if len(feature_specs) == 0:
        return np.zeros((len(records), 0), dtype=np.float64)
    class_membership_by_class: dict[VariantClass, np.ndarray] = {}
    for variant_class in VariantClass:
        class_membership_by_class[variant_class] = np.asarray(
            [
                _class_membership_weight(record, variant_class)
                for record in records
            ],
            dtype=np.float64,
        )

    continuous_feature_names, continuous_feature_matrix = _continuous_prior_design_matrix(records)
    continuous_feature_index_by_name = {
        feature_name: feature_index
        for feature_index, feature_name in enumerate(continuous_feature_names)
    }
    repeat_indicator = np.asarray([float(record.is_repeat) for record in records], dtype=np.float64)
    copy_number_indicator = np.asarray([float(record.is_copy_number) for record in records], dtype=np.float64)

    design_columns: list[np.ndarray] = []
    for feature_spec in feature_specs:
        if feature_spec.kind == "repeat_indicator":
            design_columns.append(_center_design_column(repeat_indicator))
            continue
        if feature_spec.kind == "copy_number_indicator":
            design_columns.append(_center_design_column(copy_number_indicator))
            continue
        if feature_spec.variant_class is None:
            raise ValueError("Scale-model feature spec is missing variant_class.")
        variant_class = feature_spec.variant_class
        class_membership = class_membership_by_class[variant_class]
        if feature_spec.kind == "type_offset":
            design_columns.append(_center_design_column(class_membership))
            continue
        if feature_spec.continuous_feature_name is None:
            raise ValueError("Scale-model feature spec is missing continuous_feature_name.")
        if feature_spec.continuous_feature_name not in continuous_feature_index_by_name:
            raise ValueError("Unknown continuous scale-model feature: " + str(feature_spec.continuous_feature_name))
        standardized_values = continuous_feature_matrix[
            :,
            continuous_feature_index_by_name[feature_spec.continuous_feature_name],
        ]
        if feature_spec.kind == "continuous_linear":
            design_columns.append(_center_design_column(class_membership * standardized_values))
            continue
        if feature_spec.kind == "continuous_quadratic":
            design_columns.append(
                _center_design_column(
                    class_membership * standardized_values * standardized_values
                )
            )
            continue
        raise ValueError("Unsupported scale-model feature kind: " + feature_spec.kind)

    return np.column_stack(design_columns).astype(np.float64)


def _class_membership_weight(record: VariantRecord, variant_class: VariantClass) -> float:
    for member_class, member_weight in zip(record.prior_class_members, record.prior_class_membership, strict=True):
        if member_class == variant_class:
            return float(member_weight)
    return 0.0


def _parse_scale_model_feature_names(
    feature_names: Sequence[str],
) -> tuple[ScaleModelFeatureSpec, ...]:
    feature_specs: list[ScaleModelFeatureSpec] = []
    for feature_name in feature_names:
        if feature_name == "repeat_indicator":
            feature_specs.append(ScaleModelFeatureSpec(kind="repeat_indicator"))
            continue
        if feature_name == "copy_number_indicator":
            feature_specs.append(ScaleModelFeatureSpec(kind="copy_number_indicator"))
            continue
        feature_parts = feature_name.split("::")
        if len(feature_parts) == 2:
            prefix, class_name = feature_parts
            continuous_feature_name = None
        elif len(feature_parts) == 3:
            prefix, continuous_feature_name, class_name = feature_parts
        else:
            raise ValueError("Unsupported scale-model feature: " + feature_name)
        feature_specs.append(
            ScaleModelFeatureSpec(
                kind=prefix,
                variant_class=VariantClass(class_name),
                continuous_feature_name=continuous_feature_name,
            )
        )
    return tuple(feature_specs)


def _continuous_prior_design_matrix(
    records: Sequence[VariantRecord],
) -> tuple[tuple[str, ...], np.ndarray]:
    allele_frequencies = np.clip(
        np.nan_to_num(
            np.asarray([record.allele_frequency for record in records], dtype=np.float64),
            nan=0.5,
            posinf=1.0,
            neginf=0.0,
        ),
        1e-6,
        1.0 - 1e-6,
    )
    training_support = np.maximum(
        np.nan_to_num(
            np.asarray(
                [
                    0.0 if record.training_support is None else float(record.training_support)
                    for record in records
                ],
                dtype=np.float64,
            ),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
        0.0,
    )
    feature_names = [
        "log_length",
        "logit_allele_frequency",
        "quality",
        "log_training_support",
    ]
    feature_columns = [
        np.log(np.maximum(np.asarray([record.length for record in records], dtype=np.float64), 1.0)),
        np.log(allele_frequencies) - np.log1p(-allele_frequencies),
        np.nan_to_num(
            np.asarray([record.quality for record in records], dtype=np.float64),
            nan=1.0,
            posinf=1.0,
            neginf=0.0,
        ),
        np.log1p(training_support),
    ]
    custom_feature_names = sorted(
        {
            feature_name
            for record in records
            for feature_name in record.prior_continuous_features
        }
    )
    for feature_name in custom_feature_names:
        feature_names.append(feature_name)
        feature_columns.append(
            [
                record.prior_continuous_features.get(feature_name, 0.0)
                for record in records
            ]
        )
    feature_matrix = np.column_stack(feature_columns).astype(np.float64)
    for feature_index in range(feature_matrix.shape[1]):
        feature_matrix[:, feature_index] = _standardize_metadata(feature_matrix[:, feature_index])
    return tuple(feature_names), feature_matrix


def _standardize_metadata(values: np.ndarray) -> np.ndarray:
    centered_values = values - float(np.mean(values))
    scale_value = float(np.std(values))
    if scale_value < 1e-8:
        return np.zeros_like(values)
    return centered_values / scale_value


def _cholesky_solve(cholesky_factor: np.ndarray, right_hand_side: np.ndarray) -> np.ndarray:
    lower_solution = solve_triangular(
        cholesky_factor,
        right_hand_side,
        lower=True,
        check_finite=False,
    )
    return solve_triangular(
        cholesky_factor,
        lower_solution,
        lower=True,
        trans="T",
        check_finite=False,
    )


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


# Re-estimate the global scale and metadata coefficients with the exact
# convex Newton update for the expected Gaussian prior term.
def _update_scale_model(
    reduced_second_moment: np.ndarray,
    local_scale: np.ndarray,
    prior_design: PriorDesign,
    scale_penalty: np.ndarray,
    current_global_scale: float,
    current_scale_model_coefficients: np.ndarray,
    config: ModelConfig,
) -> tuple[float, np.ndarray]:
    expected_scale = np.maximum(
        np.asarray(reduced_second_moment, dtype=np.float64) / np.maximum(local_scale, config.local_scale_floor),
        config.prior_scale_floor**2,
    )
    global_log_floor = float(np.log(config.global_scale_floor))
    global_log_ceiling = float(np.log(config.global_scale_ceiling))
    global_log_scale = float(
        np.clip(
            np.log(np.maximum(current_global_scale, config.global_scale_floor)),
            global_log_floor,
            global_log_ceiling,
        )
    )
    design_matrix = np.asarray(prior_design.design_matrix, dtype=np.float64)
    coefficients = np.asarray(current_scale_model_coefficients, dtype=np.float64).copy()
    penalty = np.maximum(np.asarray(scale_penalty, dtype=np.float64), 1e-8)
    if design_matrix.shape[1] == 0:
        updated_global_log_scale = float(
            np.clip(
                0.5 * np.log(float(np.mean(expected_scale))),
                global_log_floor,
                global_log_ceiling,
            )
        )
        return float(np.exp(updated_global_log_scale)), np.zeros(0, dtype=np.float64)

    augmented_design = np.column_stack(
        [np.ones(design_matrix.shape[0], dtype=np.float64), design_matrix]
    )
    penalty_vector = np.concatenate([np.zeros(1, dtype=np.float64), penalty])
    theta = np.concatenate([[global_log_scale], coefficients]).astype(np.float64, copy=False)

    def objective(theta_value: np.ndarray) -> float:
        linear_predictor = augmented_design @ theta_value
        return float(
            0.5 * np.sum(expected_scale * np.exp(-2.0 * linear_predictor) + 2.0 * linear_predictor)
            + 0.5 * np.sum(penalty_vector[1:] * theta_value[1:] * theta_value[1:])
        )

    for _iteration_index in range(config.maximum_scale_model_iterations):
        linear_predictor = augmented_design @ theta
        weights = expected_scale * np.exp(-2.0 * linear_predictor)
        residual = 1.0 - weights
        gradient = augmented_design.T @ residual + penalty_vector * theta
        if float(np.linalg.norm(gradient)) <= config.convergence_tolerance:
            break
        hessian = 2.0 * augmented_design.T @ (weights[:, None] * augmented_design) + np.diag(penalty_vector)
        newton_direction = -np.linalg.solve(hessian + np.eye(hessian.shape[0], dtype=np.float64) * 1e-10, gradient)
        step_size = 1.0
        current_objective = objective(theta)
        candidate_theta = theta.copy()
        while step_size >= 1e-4:
            candidate_theta = theta + step_size * newton_direction
            candidate_theta[0] = float(np.clip(candidate_theta[0], global_log_floor, global_log_ceiling))
            if objective(candidate_theta) <= current_objective:
                break
            step_size *= 0.5
        scale_change = float(np.linalg.norm(candidate_theta - theta)) / max(float(np.linalg.norm(theta)), 1e-8)
        theta = candidate_theta
        if scale_change < config.convergence_tolerance:
            break

    return float(np.exp(theta[0])), np.asarray(theta[1:], dtype=np.float64)


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


# Update the TPB shape parameters (a, b) for each variant class.
#
# These shapes control the "tail weight" of the shrinkage prior:
#   - Smaller a,b = heavier tails = more tolerance for large effects
#   - Larger a,b  = lighter tails = more shrinkage toward zero
#
# SVs (deletions, duplications) get smaller shapes than SNVs because they
# tend to have larger individual effects.  We optimize (a, b) by gradient
# ascent on the marginal likelihood of the local scales, with a
# hierarchical penalty that keeps classes from diverging too much.
def _update_tpb_shape_vectors(
    class_membership_matrix: np.ndarray,
    current_shape_a_vector: np.ndarray,
    current_shape_b_vector: np.ndarray,
    local_scale: np.ndarray,
    auxiliary_delta: np.ndarray,
    config: ModelConfig,
) -> tuple[np.ndarray, np.ndarray]:
    class_count = current_shape_a_vector.shape[0]
    if class_count == 0:
        return current_shape_a_vector, current_shape_b_vector

    log_local_scale = np.log(np.maximum(np.asarray(local_scale, dtype=np.float64), config.local_scale_floor))
    log_auxiliary_delta = np.log(np.maximum(np.asarray(auxiliary_delta, dtype=np.float64), config.local_scale_floor))
    lower_bound = np.log(config.minimum_tpb_shape)
    upper_bound = np.log(config.maximum_tpb_shape)
    initial_log_shape = np.concatenate(
        [
            np.log(np.clip(current_shape_a_vector, config.minimum_tpb_shape, config.maximum_tpb_shape)),
            np.log(np.clip(current_shape_b_vector, config.minimum_tpb_shape, config.maximum_tpb_shape)),
        ]
    )

    def objective_and_gradient(log_shape_vector: np.ndarray) -> tuple[float, np.ndarray]:
        log_shape_a = log_shape_vector[:class_count]
        log_shape_b = log_shape_vector[class_count:]
        shape_a_vector = np.exp(log_shape_a)
        shape_b_vector = np.exp(log_shape_b)
        local_shape_a = class_membership_matrix @ shape_a_vector
        local_shape_b = class_membership_matrix @ shape_b_vector
        centered_log_shape_a = log_shape_a - np.mean(log_shape_a)
        centered_log_shape_b = log_shape_b - np.mean(log_shape_b)
        hierarchical_penalty = -0.5 * (
            np.sum(centered_log_shape_a * centered_log_shape_a)
            + np.sum(centered_log_shape_b * centered_log_shape_b)
        ) / config.tpb_hierarchical_prior_variance
        # Batch all JAX special function calls into one GPU round-trip
        a_jax = jnp.asarray(local_shape_a, dtype=jnp.float64)
        b_jax = jnp.asarray(local_shape_b, dtype=jnp.float64)
        gammaln_a = np.asarray(jax_gammaln(a_jax), dtype=np.float64)
        gammaln_b = np.asarray(jax_gammaln(b_jax), dtype=np.float64)
        digamma_a = np.asarray(jax_digamma(a_jax), dtype=np.float64)
        digamma_b = np.asarray(jax_digamma(b_jax), dtype=np.float64)
        objective_value = float(
            np.sum(
                local_shape_a * log_auxiliary_delta
                - gammaln_a
                + (local_shape_a - 1.0) * log_local_scale
            )
            + np.sum(
                (local_shape_b - 1.0) * log_auxiliary_delta
                - gammaln_b
            )
            + hierarchical_penalty
        )
        score_a = log_auxiliary_delta - digamma_a + log_local_scale
        score_b = log_auxiliary_delta - digamma_b
        gradient_a = shape_a_vector * (class_membership_matrix.T @ score_a)
        gradient_b = shape_b_vector * (class_membership_matrix.T @ score_b)
        gradient_a -= centered_log_shape_a / config.tpb_hierarchical_prior_variance
        gradient_b -= centered_log_shape_b / config.tpb_hierarchical_prior_variance
        gradient = np.concatenate([gradient_a, gradient_b]).astype(np.float64)
        return objective_value, gradient

    optimized_log_shape = initial_log_shape.copy()
    for _iteration_index in range(config.maximum_tpb_shape_iterations):
        _objective_value, gradient = objective_and_gradient(optimized_log_shape)
        step = np.clip(config.tpb_shape_learning_rate * gradient, -0.25, 0.25)
        updated_log_shape = np.clip(optimized_log_shape + step, lower_bound, upper_bound)
        shape_relative_change = float(np.linalg.norm(updated_log_shape - optimized_log_shape)) / max(
            float(np.linalg.norm(optimized_log_shape)),
            1e-8,
        )
        if shape_relative_change < config.convergence_tolerance:
            optimized_log_shape = updated_log_shape
            break
        optimized_log_shape = updated_log_shape
    return (
        np.exp(optimized_log_shape[:class_count]).astype(np.float64),
        np.exp(optimized_log_shape[class_count:]).astype(np.float64),
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
    config: ModelConfig,
) -> np.ndarray:
    return np.maximum(
        baseline_prior_variances * np.maximum(local_scale, config.local_scale_floor),
        1e-8,
    )


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


# Log-probability of the local shrinkage factors under the TPB prior.
# This is the "regularization" contribution to the overall objective:
# it penalizes implausibly large or small local scales, with the penalty
# shape determined by the class-specific (a, b) parameters.
def _local_scale_prior_objective(
    local_scale: np.ndarray,
    auxiliary_delta: np.ndarray,
    local_shape_a: np.ndarray,
    local_shape_b: np.ndarray,
) -> float:
    gammaln_a = np.asarray(jax_gammaln(jnp.asarray(local_shape_a, dtype=jnp.float64)), dtype=np.float64)
    gammaln_b = np.asarray(jax_gammaln(jnp.asarray(local_shape_b, dtype=jnp.float64)), dtype=np.float64)
    log_delta = np.log(np.maximum(auxiliary_delta, 1e-12))
    log_scale = np.log(np.maximum(local_scale, 1e-12))
    return float(
        np.sum(local_shape_a * log_delta - gammaln_a + (local_shape_a - 1.0) * log_scale - auxiliary_delta * local_scale)
        + np.sum(-gammaln_b + (local_shape_b - 1.0) * log_delta - auxiliary_delta)
    )


# Update the per-variant local shrinkage factors (lambda_j).
#
# Each variant's lambda controls how much its effect gets pulled toward
# zero.  A variant with strong evidence (large beta^2 relative to its
# baseline prior) will get a large lambda ("let it through"), while a
# variant with weak evidence gets a small lambda ("shrink it to zero").
#
# The optimal lambda comes from a Generalized Inverse Gaussian (GIG)
# distribution — a family that naturally arises when you combine a
# Gaussian likelihood with a Gamma prior.  We compute its expected value
# using modified Bessel functions (via TensorFlow Probability).
#
# We alternate between updating lambda and its auxiliary rate delta in a
# fixed-point loop (typically converges in 2-3 iterations).
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
    p_parameter = np.asarray(local_shape_a, dtype=np.float64) - 0.5
    current_auxiliary_delta = np.maximum(np.asarray(auxiliary_delta, dtype=np.float64), config.local_scale_floor)
    current_local_scale = np.maximum(np.ones_like(current_auxiliary_delta), config.local_scale_floor)
    updated_local_scale = current_local_scale.copy()
    for _iteration_index in range(6):
        psi = np.maximum(2.0 * current_auxiliary_delta, 1e-12)
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
        fixed_point_change = max(
            float(np.linalg.norm(updated_auxiliary_delta - current_auxiliary_delta)) / max(
                float(np.linalg.norm(current_auxiliary_delta)),
                1e-8,
            ),
            float(np.linalg.norm(updated_local_scale - current_local_scale)) / max(
                float(np.linalg.norm(current_local_scale)),
                1e-8,
            ),
        )
        current_auxiliary_delta = np.maximum(updated_auxiliary_delta, config.local_scale_floor)
        current_local_scale = updated_local_scale
        if fixed_point_change < config.convergence_tolerance:
            break
    return updated_local_scale, current_auxiliary_delta


# Compute the expected value of X^r where X ~ GIG(p, chi, psi).
#
# The Generalized Inverse Gaussian is the conjugate distribution that
# appears when you combine a Gaussian likelihood (the data) with a Gamma
# prior (the shrinkage).  Its moments involve ratios of modified Bessel
# functions K_v(z).  We use the exponentially-scaled version (kve) for
# numerical stability — the exponential scaling cancels in the ratio.
#
# In our context: p = shape_a - 0.5, chi = beta^2 / baseline_variance,
# psi = 2 * delta.  The result E[lambda] tells us how much to shrink
# each variant.
def _gig_moment(
    p_parameter: np.ndarray,
    chi: np.ndarray,
    psi: np.ndarray,
    moment_power: float,
) -> np.ndarray:
    chi_array = np.asarray(chi, dtype=np.float64)
    psi_array = np.asarray(psi, dtype=np.float64)
    p_array = np.asarray(p_parameter, dtype=np.float64)
    z_value = np.sqrt(np.maximum(chi_array * psi_array, 1e-12))
    numerator = scipy_bessel_kve(np.abs(p_array + moment_power), z_value)
    denominator = np.maximum(scipy_bessel_kve(np.abs(p_array), z_value), 1e-300)
    moment_ratio = numerator / denominator
    return np.asarray(
        np.power(np.maximum(chi_array / psi_array, 1e-12), 0.5 * moment_power) * moment_ratio,
        dtype=np.float64,
    )


def _member_prior_variances_from_reduced_state(
    member_records: Sequence[VariantRecord],
    tie_map: TieMap,
    scale_model_coefficients: np.ndarray,
    scale_model_feature_specs: Sequence[ScaleModelFeatureSpec],
    global_scale: float,
    local_scale: np.ndarray,
    config: ModelConfig,
) -> np.ndarray:
    member_design_matrix = _design_matrix_for_feature_specs(
        records=member_records,
        feature_specs=scale_model_feature_specs,
    )
    member_baseline_scales = _metadata_baseline_scales_from_coefficients(
        scale_model_coefficients=np.asarray(scale_model_coefficients, dtype=np.float64),
        design_matrix=member_design_matrix,
        config=config,
    )
    member_baseline_prior_variances = (float(global_scale) * member_baseline_scales) ** 2
    member_local_scale = _expand_group_values_to_members(
        reduced_values=np.asarray(local_scale, dtype=np.float64),
        tie_map=tie_map,
    )
    return _effective_prior_variances(
        baseline_prior_variances=member_baseline_prior_variances,
        local_scale=member_local_scale,
        config=config,
    )


def _expand_group_values_to_members(
    reduced_values: np.ndarray,
    tie_map: TieMap,
) -> np.ndarray:
    member_values = np.zeros(tie_map.original_to_reduced.shape[0], dtype=np.float64)
    for reduced_index, tie_group in enumerate(tie_map.reduced_to_group):
        member_values[tie_group.member_indices] = float(reduced_values[reduced_index])
    return member_values


def _relative_parameter_change(
    current_beta: np.ndarray,
    previous_beta: np.ndarray | None,
    current_alpha: np.ndarray,
    previous_alpha: np.ndarray | None,
    current_local_scale: np.ndarray,
    previous_local_scale: np.ndarray | None,
    current_theta: np.ndarray,
    previous_theta: np.ndarray | None,
    current_tpb_shape_a_vector: np.ndarray,
    previous_tpb_shape_a_vector: np.ndarray | None,
    current_tpb_shape_b_vector: np.ndarray,
    previous_tpb_shape_b_vector: np.ndarray | None,
) -> float:
    if previous_beta is None:
        return float("inf")
    changes = [
        _relative_change(current_beta, previous_beta),
        _relative_change(current_alpha, previous_alpha),
        _relative_change(current_local_scale, previous_local_scale),
        _relative_change(current_theta, previous_theta),
        _relative_change(current_tpb_shape_a_vector, previous_tpb_shape_a_vector),
        _relative_change(current_tpb_shape_b_vector, previous_tpb_shape_b_vector),
    ]
    return float(max(changes))


def _relative_change(current_values: np.ndarray, previous_values: np.ndarray | None) -> float:
    if previous_values is None:
        return 0.0
    denominator = max(float(np.linalg.norm(previous_values)), 1e-8)
    return float(np.linalg.norm(current_values - previous_values) / denominator)


# Evaluate model performance on held-out validation data.
# For binary traits: mean cross-entropy loss (lower = better predictions).
# For quantitative traits: mean squared error (lower = better predictions).
def _validation_metric(
    trait_type: TraitType,
    genotype_matrix: StandardizedGenotypeMatrix | np.ndarray,
    covariate_matrix: np.ndarray,
    targets: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> float:
    if isinstance(genotype_matrix, StandardizedGenotypeMatrix):
        genotype_component = np.asarray(genotype_matrix.matvec(beta), dtype=np.float64)
    else:
        genotype_component = np.asarray(genotype_matrix @ beta, dtype=np.float64)
    linear_predictor = genotype_component + covariate_matrix @ alpha
    if trait_type == TraitType.BINARY:
        positive_probability = np.asarray(stable_sigmoid(linear_predictor), dtype=np.float64)
        return float(
            -np.mean(
                targets * np.log(positive_probability + 1e-8)
                + (1.0 - targets) * np.log(1.0 - positive_probability + 1e-8)
            )
        )
    residual_vector = targets - linear_predictor
    return float(np.mean(residual_vector * residual_vector))


# Adjust the intercept so that the model's average predicted prevalence
# matches the observed prevalence in the training data.  Uses a few
# Newton-Raphson steps on the logistic likelihood with respect to the
# intercept only — fast because it's a 1D optimization.
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

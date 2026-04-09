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
     Newton-style Polya-Gamma updates.

  2. **Local scale update**: Given the fitted betas, update each variant's
     local shrinkage lambda_j using Generalized Inverse Gaussian (GIG)
     moments — think of this as re-calibrating how much each variant
     should be shrunk toward zero.

  3. **Hyperparameter update** (every 4th iteration): Re-estimate the
     global scale sigma_g, the metadata scale-model coefficients, and
     the TPB shape parameters (a_j, b_j) to better match the data.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields as dataclass_fields, replace as dataclass_replace
import gc
import hashlib
import json
import time
from typing import Any, Callable, Sequence, cast

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
from sv_pgs.data import NESTED_PATH_DELIMITER, TieMap, VariantRecord
from sv_pgs.genotype import (
    DenseRawGenotypeMatrix,
    RawGenotypeMatrix,
    StandardizedGenotypeMatrix,
    _cupy_cache_is_int8_standardized,
    _cupy_cache_standardized_columns,
    _iter_cupy_cache_standardized_batches,
    _cupy_compute_dtype,
    _gpu_free_bytes,
    _gpu_total_bytes,
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

# GPU exact variant-space Cholesky: form X^T W X via cuBLAS + Cholesky.
# Cost: O(n p²) matmul + O(p³) Cholesky. This is 10-20x faster than
# sample-space CG and gives exact solutions with no convergence issues.
# The implementation now uses either a full standardized matrix or a tiled
# exact build depending on the live GPU working set.
_GPU_EXACT_VARIANT_MEMORY_UTILIZATION = 0.8
_GPU_EXACT_VARIANT_TILE_MAX_VARIANTS = 1_024
_GPU_EXACT_VARIANT_TILE_MIN_VARIANTS = 256
_GPU_EXACT_VARIANT_ALWAYS_FULL_MAX_VARIANTS = 64
_GPU_EXACT_VARIANT_FULL_MATRIX_HEADROOM = 2.0


def _gpu_exact_variant_base_bytes(variant_count: int, covariate_count: int) -> int:
    precision_matrix_bytes = 8 * int(variant_count) * int(variant_count)
    covariate_correction_bytes = precision_matrix_bytes if covariate_count > 0 else 0
    rhs_bytes = 8 * int(variant_count)
    inverse_diagonal_workspace_bytes = 8 * int(variant_count) * min(
        int(variant_count),
        _GPU_EXACT_VARIANT_TILE_MIN_VARIANTS,
    )
    return int(
        precision_matrix_bytes
        + covariate_correction_bytes
        + rhs_bytes
        + inverse_diagonal_workspace_bytes
    )


def _gpu_exact_variant_full_matrix_required_bytes(
    *,
    sample_count: int,
    variant_count: int,
    covariate_count: int,
    cache_is_int8_standardized: bool,
) -> int:
    expanded_matrix_bytes = 8 * int(sample_count) * int(variant_count)
    resident_cache_bytes = int(sample_count) * int(variant_count) if cache_is_int8_standardized else (
        4 * int(sample_count) * int(variant_count)
    )
    weighted_chunk_bytes = 8 * int(sample_count) * min(int(variant_count), _GPU_EXACT_VARIANT_TILE_MAX_VARIANTS)
    precision_workspace_bytes = 8 * int(variant_count) * int(variant_count)
    return int(
        resident_cache_bytes
        + expanded_matrix_bytes
        + weighted_chunk_bytes
        + _gpu_exact_variant_base_bytes(int(variant_count), int(covariate_count))
        + precision_workspace_bytes
    )


def _gpu_exact_variant_full_matrix_fits(
    cupy,
    *,
    sample_count: int,
    variant_count: int,
    covariate_count: int,
    cache_is_int8_standardized: bool,
) -> bool:
    if int(variant_count) <= _GPU_EXACT_VARIANT_ALWAYS_FULL_MAX_VARIANTS:
        return True
    # Use total GPU memory minus a fixed reserve, not free memory.
    # Free memory depends on JAX pre-allocation state which varies.
    total_bytes = _gpu_total_bytes(cupy)
    usable_bytes = int(total_bytes * _GPU_EXACT_VARIANT_MEMORY_UTILIZATION)
    required_bytes = _gpu_exact_variant_full_matrix_required_bytes(
        sample_count=int(sample_count),
        variant_count=int(variant_count),
        covariate_count=int(covariate_count),
        cache_is_int8_standardized=cache_is_int8_standardized,
    )
    return int(required_bytes * _GPU_EXACT_VARIANT_FULL_MATRIX_HEADROOM) <= usable_bytes


def _gpu_exact_variant_tile_size(
    cupy,
    *,
    sample_count: int,
    variant_count: int,
    covariate_count: int,
) -> int:
    total_bytes = _gpu_total_bytes(cupy)
    usable_bytes = int(total_bytes * _GPU_EXACT_VARIANT_MEMORY_UTILIZATION)
    fixed_bytes = _gpu_exact_variant_base_bytes(int(variant_count), int(covariate_count))
    if usable_bytes <= fixed_bytes:
        return 0
    # One float64 row tile stays resident while a second float64 tile is
    # materialized and weighted for the current Gram block.
    bytes_per_variant = int(sample_count) * (2 * 8 + 1)
    if bytes_per_variant <= 0:
        return 0
    tile_variants = min(
        int(variant_count),
        _GPU_EXACT_VARIANT_TILE_MAX_VARIANTS,
        int((usable_bytes - fixed_bytes) // bytes_per_variant),
    )
    minimum_required = min(int(variant_count), _GPU_EXACT_VARIANT_TILE_MIN_VARIANTS)
    if tile_variants < minimum_required:
        return 0
    return max(tile_variants, 0)


def _gpu_exact_variant_inverse_diagonal(
    cholesky_factor_gpu,
    *,
    solve_triangular_gpu,
    cupy,
) -> Any:
    dimension = int(cholesky_factor_gpu.shape[0])
    if dimension < 1:
        return cupy.zeros(0, dtype=cupy.float64)
    block_size = min(dimension, _GPU_EXACT_VARIANT_TILE_MIN_VARIANTS)
    inverse_diagonal_gpu = cupy.empty(dimension, dtype=cupy.float64)
    for start in range(0, dimension, block_size):
        stop = min(start + block_size, dimension)
        block_width = stop - start
        rhs_gpu = cupy.zeros((dimension, block_width), dtype=cupy.float64)
        rhs_gpu[start:stop, :] = cupy.eye(block_width, dtype=cupy.float64)
        inverse_cholesky_block_gpu = solve_triangular_gpu(cholesky_factor_gpu, rhs_gpu, lower=True)
        inverse_diagonal_gpu[start:stop] = (inverse_cholesky_block_gpu * inverse_cholesky_block_gpu).sum(
            axis=0,
            dtype=cupy.float64,
        )
    return inverse_diagonal_gpu


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
class _SampleSpacePreconditionerCacheEntry:
    batch_size: int
    rank: int
    random_seed: int
    prior_variances: np.ndarray
    diagonal_noise: np.ndarray
    diagonal_preconditioner: np.ndarray
    preconditioner: Any
    previous_iterations: int | None = None
    last_iterations: int | None = None
    nystrom_basis_cpu: np.ndarray | None = None
    nystrom_basis_gpu: Any = None  # cached GPU Nyström basis (cupy array)
    nystrom_factor_cpu: np.ndarray | None = None
    nystrom_factor_gpu: Any = None  # cached GPU Nyström factor (cupy array)
    global_background: bool = False


@dataclass(slots=True)
class ScaleModelFeatureSpec:
    kind: str
    variant_class: VariantClass | None = None
    source_name: str | None = None
    level_name: str | None = None
    nested_depth: int | None = None
    basis_index: int | None = None
    basis_kind: str | None = None
    standardize_mean: float | None = None
    standardize_scale: float | None = None
    knot_values: tuple[float, ...] = ()


@dataclass(slots=True)
class _PriorAnnotationTables:
    continuous_values_by_source: dict[str, np.ndarray]
    factor_weights_by_source: dict[str, dict[str, np.ndarray]]
    nested_weights_by_source: dict[str, dict[int, dict[str, np.ndarray]]]


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
    linear_predictor: np.ndarray | None = None
    selected_iteration_count: int | None = None


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
    best_beta_variance: np.ndarray | None
    best_local_scale: np.ndarray | None
    best_theta: np.ndarray | None
    best_sigma_error2: float | None
    best_tpb_shape_a_vector: np.ndarray | None
    best_tpb_shape_b_vector: np.ndarray | None
    best_validation_iteration: int | None = None
    completed_blocks_in_iteration: int = 0
    beta_variance_state: np.ndarray | None = None
    reduced_second_moment: np.ndarray | None = None
    epoch_reduced_prior_variances: np.ndarray | None = None
    binary_block_resume_state: dict[str, object] | None = None

    def __getstate__(self) -> dict[str, object]:
        return {
            field.name: getattr(self, field.name)
            for field in dataclass_fields(type(self))
        }

    def __setstate__(self, state: object) -> None:
        if isinstance(state, tuple):
            state_dict: dict[str, object] = {}
            for item in state:
                if isinstance(item, dict):
                    state_dict.update(item)
        elif isinstance(state, dict):
            state_dict = dict(state)
        else:
            raise TypeError(f"Unsupported VariationalFitCheckpoint pickle state: {type(state)!r}")
        for field in dataclass_fields(type(self)):
            if field.name in state_dict:
                value = state_dict[field.name]
            elif field.default is not MISSING:
                value = field.default
            elif field.default_factory is not MISSING:
                value = field.default_factory()
            else:
                raise TypeError(f"Missing required checkpoint field: {field.name}")
            object.__setattr__(self, field.name, value)


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


@dataclass(slots=True)
class _RestrictedPosteriorWarmStart:
    sample_space_inverse_covariance_rhs: Any = None
    sample_space_inverse_covariance_rhs_matrix_token: int | None = None
    sample_space_background_owner_token: int | None = None
    sample_space_background_variant_count: int | None = None
    sample_space_background_sample_count: int | None = None
    sample_space_background_gpu_preconditioner: _SampleSpacePreconditionerCacheEntry | None = None
    sample_space_background_cpu_preconditioner: _SampleSpacePreconditionerCacheEntry | None = None
    posterior_working_set_variant_count: int | None = None
    posterior_working_set_matrix_token: int | None = None
    posterior_working_set_ever_active: np.ndarray | None = None
    posterior_working_set_screening_score: np.ndarray | None = None
    posterior_working_set_target_size: int | None = None
    weighted_covariate_projection_matrix_token: int | None = None
    weighted_covariate_projection_signature: str | None = None
    weighted_covariate_projection: np.ndarray | None = None
    weighted_covariate_projection_gpu: Any = None


@dataclass(slots=True)
class _SampleSpaceCGLanczosRecorder:
    monitored_columns: np.ndarray
    maximum_steps: int
    alpha_history: np.ndarray
    beta_history: np.ndarray
    step_lengths: np.ndarray
    column_to_slot: np.ndarray


# Fields that affect convergence speed but not the mathematical model result.
# Changing these should NOT invalidate EM checkpoints.
_CHECKPOINT_EXCLUDED_CONFIG_FIELDS = frozenset({
    "stochastic_variant_batch_size",
    "stochastic_step_offset",
    "stochastic_step_exponent",
    "stochastic_min_variant_count",
    "linear_solver_tolerance",
    "maximum_linear_solver_iterations",
    "logdet_probe_count",
    "logdet_lanczos_steps",
    "exact_solver_matrix_limit",
    "posterior_variance_batch_size",
    "posterior_variance_probe_count",
    "beta_variance_update_interval",
    "sample_space_preconditioner_rank",
    "max_inner_newton_iterations",
    "newton_gradient_tolerance",
    "posterior_working_set_initial_size",
    "posterior_working_set_growth",
    "posterior_working_set_max_passes",
    "posterior_working_set_coefficient_tolerance",
    "validation_interval",
    "random_seed",
    "final_posterior_refinement",
})


def _checkpoint_config_signature(config: ModelConfig) -> str:
    payload: dict[str, object] = {}
    for field in dataclass_fields(config):
        if field.name in _CHECKPOINT_EXCLUDED_CONFIG_FIELDS:
            continue
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

def _gpu_cholesky_solve(right_hand_side, cholesky_factor_gpu, solve_triangular_gpu):
    import cupy as cp

    rhs_gpu = cp.asarray(right_hand_side, dtype=cholesky_factor_gpu.dtype)
    lower_solution = solve_triangular_gpu(cholesky_factor_gpu, rhs_gpu, lower=True)
    return solve_triangular_gpu(cholesky_factor_gpu.T, lower_solution, lower=False)


def _resolve_gpu_solve_triangular():
    try:
        from cupyx.scipy.linalg import solve_triangular as cp_solve_triangular
    except ModuleNotFoundError:
        cp_solve_triangular = solve_triangular
    return cp_solve_triangular


def _covariates_only_fit_result(
    covariate_matrix: np.ndarray,
    targets: np.ndarray,
    trait_type: TraitType,
    config: ModelConfig,
    validation_data: tuple[StandardizedGenotypeMatrix | np.ndarray, np.ndarray, np.ndarray] | None,
    predictor_offset: np.ndarray | None = None,
    validation_offset: np.ndarray | None = None,
) -> VariationalFitResult:
    offset = (
        np.zeros(len(targets), dtype=np.float64)
        if predictor_offset is None
        else np.asarray(predictor_offset, dtype=np.float64).reshape(-1)
    )
    alpha = (
        _fit_binary_alpha_with_offset(
            covariate_matrix=covariate_matrix,
            targets=targets,
            predictor_offset=offset,
            minimum_weight=config.polya_gamma_minimum_weight,
            max_iterations=config.max_inner_newton_iterations,
            gradient_tolerance=config.newton_gradient_tolerance,
        )
        if trait_type == TraitType.BINARY
        else _initialize_alpha_state(
            covariate_matrix=covariate_matrix,
            targets=np.asarray(targets, dtype=np.float64) - offset,
            trait_type=trait_type,
        )
    )
    beta = np.zeros(0, dtype=np.float64)
    linear_predictor = np.asarray(offset + covariate_matrix @ alpha, dtype=np.float64)
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
                predictor_offset=validation_offset,
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
        linear_predictor=np.asarray(linear_predictor, dtype=np.float32),
        selected_iteration_count=1,
    )

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


def _stochastic_binary_newton_iterations(
    *,
    maximum_iterations: int,
    step_size: float,
) -> int:
    if maximum_iterations < 1:
        raise ValueError("maximum_iterations must be positive.")
    # The stochastic outer update only applies a blend-weighted fraction of the
    # block solution. One Newton step already gives the Fisher-scoring direction;
    # additional refinements should therefore scale with the outer blend weight.
    reference_step = 0.27
    scheduled_iterations = max(1, int(np.floor(6.0 * max(float(step_size), 0.0) / reference_step + 0.5)))
    return min(int(maximum_iterations), scheduled_iterations)


def _stochastic_sample_space_preconditioner_rank(
    *,
    requested_rank: int,
    step_size: float,
) -> int:
    if requested_rank < 0:
        raise ValueError("requested_rank must be non-negative.")
    if requested_rank == 0:
        return 0
    reference_step = 0.27
    scaled_fraction = min(max(float(step_size), 0.0) / reference_step, 1.0)
    # The outer stochastic blend damps preconditioner error linearly in step_size,
    # while CG iteration counts depend roughly on the square root of the residual
    # conditioning. Scale the retained low-rank spectrum sublinearly so late,
    # low-weight blocks use cheaper preconditioners without changing the solved
    # linear system.
    rank_fraction = scaled_fraction ** 0.75
    minimum_rank = min(int(requested_rank), 96)
    scheduled_rank = int(np.ceil(int(requested_rank) * rank_fraction))
    return max(minimum_rank, min(int(requested_rank), scheduled_rank))



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
    # Working-set path (collapsed posterior + KKT certification) is better when
    # the full matrix is GPU-resident — each matvec is ~0.1s instead of ~2.5 min.
    # When the matrix streams from mmap, each full gradient costs 2.5 min and the
    # working-set path needs 3-6 per EM iteration.  Stochastic blocks need only 1
    # full matvec per epoch and each block solve runs on GPU.
    #
    # Route: use working-set only if matrix is on GPU.  Otherwise stochastic blocks.
    if genotype_matrix._cupy_cache is not None:
        # Matrix fits on GPU — working-set path is optimal
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
    predictor_offset: np.ndarray | None = None,
    validation_offset: np.ndarray | None = None,
) -> VariationalFitResult:
    genotype_matrix = _as_standardized_genotype_matrix(genotypes)
    covariate_matrix = np.asarray(covariates, dtype=np.float64)
    target_vector = np.asarray(targets, dtype=np.float64)
    predictor_offset_array = (
        np.zeros(target_vector.shape[0], dtype=np.float64)
        if predictor_offset is None
        else np.asarray(predictor_offset, dtype=np.float64).reshape(-1)
    )
    if predictor_offset_array.shape != target_vector.shape:
        raise ValueError("predictor_offset must match training target shape.")
    validation_offset_array = (
        None
        if validation_offset is None
        else np.asarray(validation_offset, dtype=np.float64).reshape(-1)
    )
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
            predictor_offset=predictor_offset_array,
            validation_offset=validation_offset_array,
        )
    prior_design = _build_prior_design(reduced_records)
    config_signature = _checkpoint_config_signature(config)
    prior_design_signature = _checkpoint_prior_design_signature(prior_design)
    scale_penalty = _scale_model_penalty(prior_design.feature_names, config)
    best_validation_iteration: int | None = None
    global_scale = 0.0
    scale_model_coefficients = np.zeros(prior_design.design_matrix.shape[1], dtype=np.float64)
    tpb_shape_a_vector = np.zeros(prior_design.class_membership_matrix.shape[1], dtype=np.float64)
    tpb_shape_b_vector = np.zeros(prior_design.class_membership_matrix.shape[1], dtype=np.float64)
    local_shape_a = np.zeros(len(reduced_records), dtype=np.float64)
    local_shape_b = np.zeros(len(reduced_records), dtype=np.float64)
    local_scale = np.ones(len(reduced_records), dtype=np.float64)
    auxiliary_delta = np.zeros(len(reduced_records), dtype=np.float64)
    sigma_error2 = 1.0
    alpha_state = np.zeros(covariate_matrix.shape[1], dtype=np.float64)
    beta_state = np.zeros(genotype_matrix.shape[1], dtype=np.float64)
    objective_history: list[float] = []
    validation_history: list[float] = []
    previous_alpha: np.ndarray | None = None
    previous_beta: np.ndarray | None = None
    previous_local_scale: np.ndarray | None = None
    previous_theta: np.ndarray | None = None
    previous_tpb_shape_a_vector: np.ndarray | None = None
    previous_tpb_shape_b_vector: np.ndarray | None = None
    best_validation_metric: float | None = None
    best_alpha: np.ndarray | None = None
    best_beta: np.ndarray | None = None
    best_beta_variance: np.ndarray | None = None
    best_local_scale: np.ndarray | None = None
    best_theta: np.ndarray | None = None
    best_sigma_error2: float | None = None
    best_tpb_shape_a_vector: np.ndarray | None = None
    best_tpb_shape_b_vector: np.ndarray | None = None
    start_iteration = 0

    def _copy_optional(array: np.ndarray | None) -> np.ndarray | None:
        return None if array is None else np.asarray(array, dtype=np.float64).copy()

    def _copy_resume_state_value(value: object) -> object:
        if isinstance(value, np.ndarray):
            return value.copy()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {
                str(key): _copy_resume_state_value(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [_copy_resume_state_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(_copy_resume_state_value(item) for item in value)
        return value

    def _copy_binary_block_resume_state(
        state: dict[str, object] | None,
    ) -> dict[str, object] | None:
        return None if state is None else cast(dict[str, object], _copy_resume_state_value(state))

    def _build_checkpoint(
        completed_iterations: int,
        *,
        completed_blocks_in_iteration: int = 0,
        beta_variance_state_override: np.ndarray | None = None,
        reduced_second_moment_override: np.ndarray | None = None,
        epoch_reduced_prior_variances_override: np.ndarray | None = None,
        binary_block_resume_state_override: dict[str, object] | None = None,
    ) -> VariationalFitCheckpoint:
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
            best_beta_variance=_copy_optional(best_beta_variance),
            best_local_scale=_copy_optional(best_local_scale),
            best_theta=_copy_optional(best_theta),
            best_sigma_error2=None if best_sigma_error2 is None else float(best_sigma_error2),
            best_tpb_shape_a_vector=_copy_optional(best_tpb_shape_a_vector),
            best_tpb_shape_b_vector=_copy_optional(best_tpb_shape_b_vector),
            best_validation_iteration=None if best_validation_iteration is None else int(best_validation_iteration),
            completed_blocks_in_iteration=int(completed_blocks_in_iteration),
            beta_variance_state=_copy_optional(beta_variance_state_override),
            reduced_second_moment=_copy_optional(reduced_second_moment_override),
            epoch_reduced_prior_variances=_copy_optional(epoch_reduced_prior_variances_override),
            binary_block_resume_state=_copy_binary_block_resume_state(binary_block_resume_state_override),
        )

    def _initialize_em_state() -> None:
        nonlocal global_scale, scale_model_coefficients, tpb_shape_a_vector, tpb_shape_b_vector
        nonlocal local_shape_a, local_shape_b, local_scale, auxiliary_delta, sigma_error2
        nonlocal alpha_state, beta_state, objective_history, validation_history
        nonlocal previous_alpha, previous_beta, previous_local_scale, previous_theta
        nonlocal previous_tpb_shape_a_vector, previous_tpb_shape_b_vector
        nonlocal best_validation_metric, best_alpha, best_beta, best_beta_variance, best_local_scale, best_theta
        nonlocal best_sigma_error2, best_tpb_shape_a_vector, best_tpb_shape_b_vector, best_validation_iteration, start_iteration
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
        best_beta_variance = None
        best_local_scale = None
        best_theta = None
        best_sigma_error2 = None
        best_tpb_shape_a_vector = None
        best_tpb_shape_b_vector = None
        best_validation_iteration = None
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
            best_beta_variance = _copy_optional(resume_checkpoint.best_beta_variance)
            best_local_scale = _copy_optional(resume_checkpoint.best_local_scale)
            best_theta = _copy_optional(resume_checkpoint.best_theta)
            best_sigma_error2 = None if resume_checkpoint.best_sigma_error2 is None else float(resume_checkpoint.best_sigma_error2)
            best_tpb_shape_a_vector = _copy_optional(resume_checkpoint.best_tpb_shape_a_vector)
            best_tpb_shape_b_vector = _copy_optional(resume_checkpoint.best_tpb_shape_b_vector)
            best_validation_iteration = (
                None
                if resume_checkpoint.best_validation_iteration is None
                else int(resume_checkpoint.best_validation_iteration)
            )
            start_iteration = int(resume_checkpoint.completed_iterations)
            log(
                "  variational EM: resuming from checkpoint "
                + f"after {start_iteration}/{config.max_outer_iterations} iterations  mem={mem()}"
            )

    log(f"  variational EM: {genotype_matrix.shape[1]} reduced variants, {covariate_matrix.shape[1]} covariates, {target_vector.shape[0]} samples, max_iter={config.max_outer_iterations}")
    use_stochastic_updates = _should_use_stochastic_variational_updates(genotype_matrix, config)
    gpu_resident = genotype_matrix._cupy_cache is not None
    matrix_bytes = int(genotype_matrix.shape[0]) * int(genotype_matrix.shape[1])
    log(
        f"  solver routing: gpu_resident={gpu_resident}  "
        f"matrix={matrix_bytes/1e9:.1f} GB int8  "
        f"stochastic={'yes' if use_stochastic_updates else 'no'}  "
        f"reason={'matrix on GPU → working-set' if gpu_resident else 'streaming from mmap → stochastic blocks' if use_stochastic_updates else 'small variant count → collapsed'}"
    )
    beta_variance_state: np.ndarray | None = None
    restricted_posterior_warm_start = _RestrictedPosteriorWarmStart()
    if use_stochastic_updates:
        block_size = min(int(config.stochastic_variant_batch_size), int(genotype_matrix.shape[1]))
        block_count = max((int(genotype_matrix.shape[1]) + block_size - 1) // block_size, 1)
        resume_completed_blocks_in_iteration = (
            0
            if resume_checkpoint is None
            else int(resume_checkpoint.completed_blocks_in_iteration)
        )
        if resume_completed_blocks_in_iteration < 0 or resume_completed_blocks_in_iteration >= block_count:
            if not (resume_completed_blocks_in_iteration == 0 and block_count == 1):
                raise ValueError("resume checkpoint completed_blocks_in_iteration is out of range.")
        resume_beta_variance_state = None
        resume_reduced_second_moment = None
        resume_epoch_reduced_prior_variances = None
        resume_binary_block_state = (
            None
            if resume_checkpoint is None
            else _copy_binary_block_resume_state(resume_checkpoint.binary_block_resume_state)
        )
        if resume_completed_blocks_in_iteration > 0 or resume_binary_block_state is not None:
            if resume_checkpoint is None:
                raise ValueError("resume checkpoint block progress requires checkpoint state.")
            if resume_checkpoint.beta_variance_state is None or resume_checkpoint.reduced_second_moment is None:
                raise ValueError("resume checkpoint missing stochastic state for mid-epoch resume.")
            if resume_checkpoint.epoch_reduced_prior_variances is None:
                raise ValueError("resume checkpoint missing frozen epoch prior variances for mid-epoch resume.")
            if resume_checkpoint.beta_variance_state.shape != (genotype_matrix.shape[1],):
                raise ValueError("resume checkpoint beta_variance_state shape does not match reduced genotypes.")
            if resume_checkpoint.reduced_second_moment.shape != (genotype_matrix.shape[1],):
                raise ValueError("resume checkpoint reduced_second_moment shape does not match reduced genotypes.")
            if resume_checkpoint.epoch_reduced_prior_variances.shape != (genotype_matrix.shape[1],):
                raise ValueError("resume checkpoint epoch_reduced_prior_variances shape does not match reduced genotypes.")
            resume_beta_variance_state = np.asarray(resume_checkpoint.beta_variance_state, dtype=np.float64).copy()
            resume_reduced_second_moment = np.asarray(resume_checkpoint.reduced_second_moment, dtype=np.float64).copy()
            resume_epoch_reduced_prior_variances = np.asarray(
                resume_checkpoint.epoch_reduced_prior_variances,
                dtype=np.float64,
            ).copy()
        step_index = start_iteration * block_count + resume_completed_blocks_in_iteration
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
        beta_variance_state = (
            np.maximum(reduced_prior_variances.copy(), 1e-8)
            if resume_beta_variance_state is None
            else resume_beta_variance_state
        )
        reduced_second_moment = (
            np.maximum(
                np.asarray(beta_state * beta_state, dtype=np.float64),
                np.asarray(reduced_prior_variances, dtype=np.float64),
            )
            if resume_reduced_second_moment is None
            else resume_reduced_second_moment
        )
        stochastic_epoch_cupy = None
        stochastic_epoch_compute_cp_dtype = None
        predictor_offset_gpu = None
        target_vector_gpu = None
        use_gpu_epoch_predictor_state = (
            genotype_matrix._cupy_cache is not None
            or _streaming_cupy_backend_available(genotype_matrix)
        )
        if use_gpu_epoch_predictor_state:
            stochastic_epoch_cupy = _try_import_cupy()
            use_gpu_epoch_predictor_state = stochastic_epoch_cupy is not None
        if use_gpu_epoch_predictor_state:
            assert stochastic_epoch_cupy is not None
            stochastic_epoch_compute_cp_dtype = _cupy_compute_dtype(stochastic_epoch_cupy)
            predictor_offset_gpu = stochastic_epoch_cupy.asarray(
                predictor_offset_array,
                dtype=stochastic_epoch_compute_cp_dtype,
            )
            target_vector_gpu = stochastic_epoch_cupy.asarray(
                target_vector,
                dtype=stochastic_epoch_compute_cp_dtype,
            )
        log(
            "  variational inference mode: stochastic variant-block updates "
            + f"(block_size={block_size}, blocks_per_epoch={block_count})"
        )
        for outer_iteration in range(start_iteration, config.max_outer_iterations):
            log(f"  variational EM epoch {outer_iteration + 1}/{config.max_outer_iterations} start  sigma_e2={sigma_error2:.6f}  global_scale={global_scale:.6f}  mem={mem()}")
            epoch_wall_t0 = time.monotonic()
            epoch_total_newton_iters = 0
            resuming_mid_epoch = (
                outer_iteration == start_iteration
                and (
                    resume_completed_blocks_in_iteration > 0
                    or resume_binary_block_state is not None
                )
            )
            refresh_beta_variance = _should_refresh_beta_variance(
                outer_iteration,
                refresh_interval=config.beta_variance_update_interval,
                total_iterations=config.max_outer_iterations,
                force_final_refresh=not config.final_posterior_refinement,
                trait_type=config.trait_type,
                sample_count=genotype_matrix.shape[0],
                variant_count=genotype_matrix.shape[1],
                exact_solver_matrix_limit=config.exact_solver_matrix_limit,
            )
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
            reduced_prior_variances = (
                np.asarray(resume_epoch_reduced_prior_variances, dtype=np.float64).copy()
                if resuming_mid_epoch and resume_epoch_reduced_prior_variances is not None
                else _effective_prior_variances(
                    baseline_prior_variances=baseline_reduced_prior_variances,
                    local_scale=local_scale,
                    config=config,
                )
            )
            genetic_linear_predictor_gpu = None
            if use_gpu_epoch_predictor_state:
                assert stochastic_epoch_cupy is not None
                assert stochastic_epoch_compute_cp_dtype is not None
                genetic_linear_predictor_gpu = genotype_matrix.gpu_matmat(
                    stochastic_epoch_cupy.asarray(beta_state, dtype=stochastic_epoch_compute_cp_dtype),
                    batch_size=config.posterior_variance_batch_size,
                    cupy=stochastic_epoch_cupy,
                    dtype=stochastic_epoch_compute_cp_dtype,
                )
                genetic_linear_predictor = _cupy_array_to_numpy(genetic_linear_predictor_gpu, dtype=np.float64)
            else:
                genetic_linear_predictor = np.array(
                    genotype_matrix.matvec_numpy(beta_state, batch_size=config.posterior_variance_batch_size),
                    dtype=np.float64,
                    copy=True,
                )
            if not resuming_mid_epoch:
                if config.trait_type == TraitType.BINARY:
                    alpha_state = _fit_binary_alpha_with_offset(
                        covariate_matrix=covariate_matrix,
                        targets=target_vector,
                        predictor_offset=predictor_offset_array + genetic_linear_predictor,
                        minimum_weight=config.polya_gamma_minimum_weight,
                        max_iterations=config.max_inner_newton_iterations,
                        gradient_tolerance=config.newton_gradient_tolerance,
                        alpha_init=alpha_state,
                    )
                    sigma_error2 = 1.0
                else:
                    alpha_state = _initialize_alpha_state(
                        covariate_matrix=covariate_matrix,
                        targets=target_vector - predictor_offset_array - genetic_linear_predictor,
                        trait_type=TraitType.QUANTITATIVE,
                    )
            covariate_linear_predictor = np.asarray(covariate_matrix @ alpha_state, dtype=np.float64)
            covariate_linear_predictor_gpu = (
                None
                if not use_gpu_epoch_predictor_state
                else stochastic_epoch_cupy.asarray(
                    covariate_linear_predictor,
                    dtype=stochastic_epoch_compute_cp_dtype,
                )
            )
            background_preconditioner_rank = _effective_sample_space_preconditioner_rank(
                genotype_matrix=genotype_matrix,
                sample_count=genotype_matrix.shape[0],
                variant_count=genotype_matrix.shape[1],
                requested_rank=config.sample_space_preconditioner_rank,
            )
            if background_preconditioner_rank > 0:
                if config.trait_type == TraitType.BINARY:
                    if use_gpu_epoch_predictor_state:
                        assert stochastic_epoch_cupy is not None
                        assert stochastic_epoch_compute_cp_dtype is not None
                        assert predictor_offset_gpu is not None
                        assert covariate_linear_predictor_gpu is not None
                        assert genetic_linear_predictor_gpu is not None
                        epoch_linear_predictor_gpu = (
                            predictor_offset_gpu
                            + covariate_linear_predictor_gpu
                            + genetic_linear_predictor_gpu
                        )
                        epoch_weights = _cupy_array_to_numpy(
                            _binary_expected_polya_gamma_weights_cupy(
                                stochastic_epoch_cupy,
                                epoch_linear_predictor_gpu,
                                config.polya_gamma_minimum_weight,
                                dtype=stochastic_epoch_compute_cp_dtype,
                            ),
                            dtype=np.float64,
                        )
                    else:
                        epoch_linear_predictor = (
                            predictor_offset_array
                            + covariate_linear_predictor
                            + genetic_linear_predictor
                        )
                        epoch_weights = _binary_expected_polya_gamma_weights(
                            linear_predictor=epoch_linear_predictor,
                            minimum_weight=config.polya_gamma_minimum_weight,
                        )
                    background_diagonal_noise = 1.0 / np.maximum(epoch_weights, 1e-12)
                else:
                    background_diagonal_noise = np.full(
                        genotype_matrix.shape[0],
                        sigma_error2,
                        dtype=np.float64,
                    )
                if genotype_matrix._cupy_cache is not None or _streaming_cupy_backend_available(genotype_matrix):
                    _get_or_build_background_sample_space_gpu_preconditioner(
                        genotype_matrix=genotype_matrix,
                        prior_variances=reduced_prior_variances,
                        diagonal_noise=background_diagonal_noise,
                        batch_size=config.posterior_variance_batch_size,
                        rank=background_preconditioner_rank,
                        random_seed=config.random_seed + outer_iteration,
                        warm_start=restricted_posterior_warm_start,
                    )
                else:
                    _get_or_build_background_sample_space_cpu_preconditioner(
                        genotype_matrix=genotype_matrix,
                        prior_variances=reduced_prior_variances,
                        diagonal_noise=background_diagonal_noise,
                        batch_size=config.posterior_variance_batch_size,
                        rank=background_preconditioner_rank,
                        random_seed=config.random_seed + outer_iteration,
                        warm_start=restricted_posterior_warm_start,
                    )

            epoch_blocks = _stochastic_variant_blocks(genotype_matrix.shape[1], block_size, epoch_rng)
            n_blocks = len(epoch_blocks)
            block_count = 0 if not resuming_mid_epoch else resume_completed_blocks_in_iteration
            if resuming_mid_epoch:
                log(
                    "  variational EM: resuming stochastic epoch "
                    + f"{outer_iteration + 1}/{config.max_outer_iterations} at block {resume_completed_blocks_in_iteration + 1}/{n_blocks}  mem={mem()}"
                )
            for block_indices in epoch_blocks[resume_completed_blocks_in_iteration:]:
                step_index += 1
                block_count += 1
                step_size = _stochastic_step_size(config, step_index)
                active_binary_resume_state: dict[str, object] | None = None
                if resume_binary_block_state is not None:
                    expected_block_indices = np.asarray(
                        resume_binary_block_state.get("block_indices"),
                        dtype=np.int32,
                    ).reshape(-1)
                    if expected_block_indices.shape == block_indices.shape and np.array_equal(expected_block_indices, block_indices):
                        solver_state = resume_binary_block_state.get("solver_state")
                        if not isinstance(solver_state, dict):
                            raise ValueError("resume checkpoint binary block state is missing solver_state.")
                        active_binary_resume_state = cast(dict[str, object], _copy_resume_state_value(solver_state))
                        log(
                            "    resuming binary block from cached Newton state "
                            + f"at block {block_count}/{n_blocks}  mem={mem()}"
                        )
                    else:
                        log(
                            "    discarding stale binary block resume state "
                            + f"(expected block size {expected_block_indices.size}, got {block_indices.size})"
                        )
                    resume_binary_block_state = None
                block_genotypes = genotype_matrix.subset(block_indices)
                # Upload block to GPU — fits easily in GPU budget.
                # Without this, every CG iteration streams from mmap (40s vs 2s).
                block_genotypes.try_materialize_gpu()
                if block_count <= 3 or block_count % max(n_blocks // 10, 1) == 0 or block_count == n_blocks:
                    log(f"    block {block_count}/{n_blocks}  variants={len(block_indices)}  step_size={step_size:.4f}  gpu={'yes' if block_genotypes._cupy_cache is not None else 'no'}  mem={mem()}")
                block_prior_variances = np.asarray(reduced_prior_variances[block_indices], dtype=np.float64)
                # Let the restricted posterior choose between full-matrix exact,
                # tiled exact, and sample-space CG from the live GPU working set.
                allow_gpu_exact_variant_for_block = True
                block_sample_space_preconditioner_rank = _stochastic_sample_space_preconditioner_rank(
                    requested_rank=int(config.sample_space_preconditioner_rank),
                    step_size=step_size,
                )
                block_config = dataclass_replace(
                    config,
                    sample_space_preconditioner_rank=block_sample_space_preconditioner_rank,
                )
                block_beta_previous = np.asarray(beta_state[block_indices], dtype=np.float64).copy()
                predictor_offset_gpu_block = None
                if use_gpu_epoch_predictor_state:
                    assert stochastic_epoch_cupy is not None
                    assert stochastic_epoch_compute_cp_dtype is not None
                    assert predictor_offset_gpu is not None
                    assert covariate_linear_predictor_gpu is not None
                    assert genetic_linear_predictor_gpu is not None
                    block_linear_predictor_previous_gpu = block_genotypes.gpu_matmat(
                        stochastic_epoch_cupy.asarray(block_beta_previous, dtype=stochastic_epoch_compute_cp_dtype),
                        batch_size=config.posterior_variance_batch_size,
                        cupy=stochastic_epoch_cupy,
                        dtype=stochastic_epoch_compute_cp_dtype,
                    )
                    predictor_offset_gpu_block = (
                        predictor_offset_gpu
                        + covariate_linear_predictor_gpu
                        + genetic_linear_predictor_gpu
                        - block_linear_predictor_previous_gpu
                    )
                    predictor_offset = _cupy_array_to_numpy(
                        predictor_offset_gpu_block,
                        dtype=np.float64,
                    )
                else:
                    block_linear_predictor_previous = np.asarray(
                        block_genotypes.matvec_numpy(block_beta_previous, batch_size=config.posterior_variance_batch_size),
                        dtype=np.float64,
                    )
                    predictor_offset = (
                        predictor_offset_array
                        + covariate_linear_predictor
                        + genetic_linear_predictor
                        - block_linear_predictor_previous
                    )
                if config.trait_type == TraitType.BINARY:
                    def _save_partial_binary_block(binary_state: dict[str, object]) -> None:
                        if checkpoint_callback is None:
                            return
                        checkpoint_callback(
                            _build_checkpoint(
                                outer_iteration,
                                completed_blocks_in_iteration=block_count - 1,
                                beta_variance_state_override=beta_variance_state,
                                reduced_second_moment_override=reduced_second_moment,
                                epoch_reduced_prior_variances_override=reduced_prior_variances,
                                binary_block_resume_state_override={
                                    "block_indices": np.asarray(block_indices, dtype=np.int32).copy(),
                                    "solver_state": binary_state,
                                },
                            )
                        )

                    block_state = _binary_posterior_state(
                        genotype_matrix=block_genotypes,
                        covariate_matrix=empty_covariates,
                        targets=target_vector,
                        prior_variances=block_prior_variances,
                        alpha_init=np.zeros(0, dtype=np.float64),
                        beta_init=block_beta_previous,
                        minimum_weight=config.polya_gamma_minimum_weight,
                        # Inner Newton work should scale with the stochastic blend
                        # weight: later low-step updates only need a coarse local
                        # optimum because the global state receives a damped move.
                        max_iterations=_stochastic_binary_newton_iterations(
                            maximum_iterations=config.max_inner_newton_iterations,
                            step_size=step_size,
                        ),
                        gradient_tolerance=max(config.newton_gradient_tolerance, 1e-4),
                        solver_tolerance=config.linear_solver_tolerance,
                        maximum_linear_solver_iterations=config.maximum_linear_solver_iterations,
                        logdet_probe_count=config.logdet_probe_count,
                        logdet_lanczos_steps=config.logdet_lanczos_steps,
                        exact_solver_matrix_limit=config.exact_solver_matrix_limit,
                        posterior_variance_batch_size=config.posterior_variance_batch_size,
                        posterior_variance_probe_count=config.posterior_variance_probe_count,
                        random_seed=config.random_seed + step_index,
                        compute_logdet=False,
                        compute_beta_variance=refresh_beta_variance,
                        sample_space_preconditioner_rank=block_sample_space_preconditioner_rank,
                        predictor_offset=predictor_offset,
                        update_blend_weight=step_size,
                        resume_state=active_binary_resume_state,
                        progress_callback=_save_partial_binary_block if checkpoint_callback is not None else None,
                        restricted_posterior_warm_start=restricted_posterior_warm_start,
                        allow_gpu_exact_variant=allow_gpu_exact_variant_for_block,
                    )
                    block_beta_candidate = np.asarray(block_state[1], dtype=np.float64)
                    block_beta_variance = (
                        np.asarray(block_state[2], dtype=np.float64)
                        if refresh_beta_variance
                        else np.asarray(beta_variance_state[block_indices], dtype=np.float64)
                    )
                    epoch_total_newton_iters += int(block_state[5])
                else:
                    collapsed_block_state = _fit_collapsed_posterior(
                        genotype_matrix=block_genotypes,
                        covariate_matrix=empty_covariates,
                        targets=(
                            _cupy_array_to_numpy(target_vector_gpu - predictor_offset_gpu_block, dtype=np.float64)
                            if use_gpu_epoch_predictor_state
                            else target_vector - predictor_offset
                        ),
                        reduced_prior_variances=block_prior_variances,
                        sigma_error2=sigma_error2,
                        alpha_init=np.zeros(0, dtype=np.float64),
                        beta_init=block_beta_previous,
                        trait_type=TraitType.QUANTITATIVE,
                        config=block_config,
                        compute_logdet=False,
                        compute_beta_variance=refresh_beta_variance,
                        stale_beta_variance=None if beta_variance_state is None else beta_variance_state[block_indices],
                        restricted_posterior_warm_start=restricted_posterior_warm_start,
                        em_iteration_index=outer_iteration,
                        total_em_iterations=config.max_outer_iterations,
                        update_blend_weight=step_size,
                        allow_gpu_exact_variant=allow_gpu_exact_variant_for_block,
                    )
                    block_beta_candidate = np.asarray(collapsed_block_state.beta, dtype=np.float64)
                    block_beta_variance = np.asarray(collapsed_block_state.beta_variance, dtype=np.float64)
                    epoch_total_newton_iters += 1
                block_beta_updated = block_beta_previous + step_size * (block_beta_candidate - block_beta_previous)
                beta_delta = block_beta_updated - block_beta_previous
                if np.any(beta_delta):
                    if use_gpu_epoch_predictor_state:
                        assert stochastic_epoch_cupy is not None
                        assert stochastic_epoch_compute_cp_dtype is not None
                        assert genetic_linear_predictor_gpu is not None
                        genetic_linear_predictor_gpu = genetic_linear_predictor_gpu + block_genotypes.gpu_matmat(
                            stochastic_epoch_cupy.asarray(beta_delta, dtype=stochastic_epoch_compute_cp_dtype),
                            batch_size=config.posterior_variance_batch_size,
                            cupy=stochastic_epoch_cupy,
                            dtype=stochastic_epoch_compute_cp_dtype,
                        )
                    else:
                        genetic_linear_predictor += np.asarray(
                            block_genotypes.matvec_numpy(beta_delta, batch_size=config.posterior_variance_batch_size),
                            dtype=np.float64,
                        )
                    beta_state[block_indices] = block_beta_updated
                block_second_moment = np.asarray(
                    block_beta_candidate * block_beta_candidate + block_beta_variance,
                    dtype=np.float64,
                )
                beta_variance_state[block_indices] = (
                    (1.0 - step_size) * beta_variance_state[block_indices]
                    + step_size * block_beta_variance
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
                # Free GPU memory for this block before next iteration
                block_genotypes._cupy_cache = None
                del block_genotypes
                # Periodic GC to reclaim any cyclic garbage built up over blocks.
                if block_count % 5 == 0:
                    gc.collect()
                log(f"    block {block_count}/{n_blocks} done  mem={mem()}")
                if checkpoint_callback is not None:
                    checkpoint_callback(
                        _build_checkpoint(
                            outer_iteration,
                            completed_blocks_in_iteration=block_count,
                            beta_variance_state_override=beta_variance_state,
                            reduced_second_moment_override=reduced_second_moment,
                            epoch_reduced_prior_variances_override=reduced_prior_variances,
                        )
                    )
            resume_completed_blocks_in_iteration = 0
            resume_beta_variance_state = None
            resume_reduced_second_moment = None
            resume_epoch_reduced_prior_variances = None
            resume_binary_block_state = None

            epoch_wall_seconds = time.monotonic() - epoch_wall_t0
            n_nonzero_beta = int(np.sum(beta_state != 0.0))
            avg_newton = epoch_total_newton_iters / max(block_count, 1)
            log(f"  variational EM epoch {outer_iteration + 1} done: blocks={block_count}  wall={epoch_wall_seconds:.1f}s  nonzero_beta={n_nonzero_beta}  avg_newton_iters={avg_newton:.1f}")
            if use_gpu_epoch_predictor_state:
                assert genetic_linear_predictor_gpu is not None
                genetic_linear_predictor = _cupy_array_to_numpy(genetic_linear_predictor_gpu, dtype=np.float64)

            if config.trait_type == TraitType.BINARY:
                alpha_state = _fit_binary_alpha_with_offset(
                    covariate_matrix=covariate_matrix,
                        targets=target_vector,
                        predictor_offset=predictor_offset_array + genetic_linear_predictor,
                    minimum_weight=config.polya_gamma_minimum_weight,
                    max_iterations=config.max_inner_newton_iterations,
                    gradient_tolerance=config.newton_gradient_tolerance,
                    alpha_init=alpha_state,
                )
                sigma_error2 = 1.0
            else:
                alpha_state = _initialize_alpha_state(
                    covariate_matrix=covariate_matrix,
                        targets=target_vector - predictor_offset_array - genetic_linear_predictor,
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
            linear_predictor = predictor_offset_array + covariate_linear_predictor + genetic_linear_predictor
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
                        predictor_offset=validation_offset_array,
                    )
                    validation_history.append(validation_metric)
                    if best_validation_metric is None or validation_metric < best_validation_metric:
                        best_validation_metric = validation_metric
                        best_validation_iteration = outer_iteration + 1
                        best_alpha = alpha_state.copy()
                        best_beta = beta_state.copy()
                        best_beta_variance = beta_variance_state.copy()
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
            variance_str = "  [beta_var]" if refresh_beta_variance else "  [beta_var=reuse]"
            nonzero_beta = int(np.count_nonzero(np.abs(beta_state) > 1e-8))
            log(f"  SVI epoch {iter_num}/{config.max_outer_iterations}  obj={obj_str}  delta={parameter_change:.2e}  sigma_e2={sigma_error2:.4f}  g_scale={float(global_scale):.4f}  nz_beta={nonzero_beta}{val_str}{hyper_str}{variance_str}  mem={mem()}")
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
        if (
            config.posterior_working_sets
            and genotype_matrix.shape[1] >= config.posterior_working_set_min_variants
        ):
            log(
                "  variational inference mode: posterior working sets "
                + f"(total_variants={genotype_matrix.shape[1]}, "
                + f"initial={config.posterior_working_set_initial_size}, "
                + f"max_passes={config.posterior_working_set_max_passes})"
            )
        for outer_iteration in range(start_iteration, config.max_outer_iterations):
            iter_wall_t0 = time.monotonic()
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
            if beta_variance_state is None:
                beta_variance_state = np.maximum(reduced_prior_variances.copy(), 1e-8)
            refresh_beta_variance = _should_refresh_beta_variance(
                outer_iteration,
                refresh_interval=config.beta_variance_update_interval,
                total_iterations=config.max_outer_iterations,
                force_final_refresh=not config.final_posterior_refinement,
                trait_type=config.trait_type,
                sample_count=genotype_matrix.shape[0],
                variant_count=genotype_matrix.shape[1],
                exact_solver_matrix_limit=config.exact_solver_matrix_limit,
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
                compute_beta_variance=refresh_beta_variance,
                predictor_offset=predictor_offset_array,
                stale_beta_variance=beta_variance_state,
                restricted_posterior_warm_start=restricted_posterior_warm_start,
                em_iteration_index=outer_iteration,
                total_em_iterations=config.max_outer_iterations,
            )
            if config.trait_type == TraitType.BINARY and config.binary_intercept_calibration:
                posterior_state = _apply_binary_intercept_calibration(
                    posterior_state=posterior_state,
                    targets=target_vector,
                )
            alpha_state = posterior_state.alpha
            beta_state = posterior_state.beta
            sigma_error2 = posterior_state.sigma_error2
            beta_variance_state = np.asarray(posterior_state.beta_variance, dtype=np.float64)

            reduced_second_moment = np.asarray(beta_state * beta_state + beta_variance_state, dtype=np.float64)
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
                        predictor_offset=validation_offset_array,
                    )
                    validation_history.append(validation_metric)
                    if best_validation_metric is None or validation_metric < best_validation_metric:
                        best_validation_metric = validation_metric
                        best_validation_iteration = outer_iteration + 1
                        best_alpha = alpha_state.copy()
                        best_beta = beta_state.copy()
                        best_beta_variance = beta_variance_state.copy()
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
            variance_str = "  [beta_var]" if refresh_beta_variance else "  [beta_var=reuse]"
            nonzero_beta = int(np.count_nonzero(np.abs(beta_state) > 1e-8))
            iter_wall_seconds = time.monotonic() - iter_wall_t0
            log(f"  EM iter {iter_num}/{config.max_outer_iterations}  obj={obj_str}  delta={parameter_change:.2e}  sigma_e2={sigma_error2:.4f}  g_scale={float(global_scale):.4f}  nz_beta={nonzero_beta}  wall={iter_wall_seconds:.1f}s{val_str}{hyper_str}{variance_str}  mem={mem()}")
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
            or best_beta_variance is None
            or best_local_scale is None
            or best_theta is None
            or best_sigma_error2 is None
            or best_tpb_shape_a_vector is None
            or best_tpb_shape_b_vector is None
        ):
            raise RuntimeError("best validation snapshot is incomplete")
        alpha_state = best_alpha
        beta_state = best_beta
        beta_variance_state = best_beta_variance
        local_scale = best_local_scale
        sigma_error2 = best_sigma_error2
        global_scale, scale_model_coefficients = _unpack_theta(best_theta)
        tpb_shape_a_vector = best_tpb_shape_a_vector
        tpb_shape_b_vector = best_tpb_shape_b_vector

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
    if config.final_posterior_refinement:
        log(f"  EM loop done after {len(objective_history)} iterations, computing final posterior...  mem={mem()}")
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
            predictor_offset=predictor_offset_array,
        )
        if config.trait_type == TraitType.BINARY and config.binary_intercept_calibration:
            final_state = _apply_binary_intercept_calibration(
                posterior_state=final_state,
                targets=target_vector,
            )
        log(f"  final posterior computed  obj={final_state.collapsed_objective:.6f}  sigma_e2={final_state.sigma_error2:.4f}  mem={mem()}")
    else:
        if beta_variance_state is None:
            beta_variance_state = np.maximum(final_reduced_prior_variances, 1e-8)
        final_linear_predictor = predictor_offset_array + np.asarray(covariate_matrix @ alpha_state, dtype=np.float64)
        if genotype_matrix.shape[1] > 0:
            final_linear_predictor = final_linear_predictor + np.asarray(
                genotype_matrix.matvec_numpy(beta_state, batch_size=config.posterior_variance_batch_size),
                dtype=np.float64,
            )
        final_state = PosteriorState(
            alpha=np.asarray(alpha_state, dtype=np.float64).copy(),
            beta=np.asarray(beta_state, dtype=np.float64).copy(),
            beta_variance=np.asarray(beta_variance_state, dtype=np.float64).copy(),
            linear_predictor=np.asarray(final_linear_predictor, dtype=np.float64),
            collapsed_objective=float(objective_history[-1]) if objective_history else 0.0,
            sigma_error2=float(sigma_error2),
        )
        if config.trait_type == TraitType.BINARY and config.binary_intercept_calibration:
            final_state = _apply_binary_intercept_calibration(
                posterior_state=final_state,
                targets=target_vector,
            )
        log("  final posterior refinement skipped; returning current variational state  mem=" + mem())
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
        linear_predictor=np.asarray(final_state.linear_predictor, dtype=np.float32),
        selected_iteration_count=(
            int(best_validation_iteration)
            if best_validation_iteration is not None
            else int(len(objective_history))
        ),
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


def _should_refresh_beta_variance(
    iteration_index: int,
    *,
    refresh_interval: int,
    total_iterations: int,
    force_final_refresh: bool,
    trait_type: TraitType | None = None,
    sample_count: int | None = None,
    variant_count: int | None = None,
    exact_solver_matrix_limit: int | None = None,
) -> bool:
    # In the exact small-n quantitative regime, stale beta variances destabilize
    # the sigma_e^2 update because the leverage correction lags by several EM
    # steps. Refresh every iteration there so the single Bayesian path remains
    # well-behaved in p >> n settings.
    if (
        trait_type == TraitType.QUANTITATIVE
        and sample_count is not None
        and variant_count is not None
        and exact_solver_matrix_limit is not None
        and int(variant_count) > int(sample_count)
        and int(sample_count) <= int(exact_solver_matrix_limit)
    ):
        return True
    if force_final_refresh and iteration_index + 1 == max(int(total_iterations), 1):
        return True
    return ((iteration_index + 1) % max(int(refresh_interval), 1)) == 0


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


def _effective_beta_variance_state(
    *,
    compute_beta_variance: bool,
    beta_variance: np.ndarray,
    stale_beta_variance: np.ndarray | None,
    prior_variances: np.ndarray,
) -> np.ndarray:
    if compute_beta_variance:
        resolved = np.asarray(beta_variance, dtype=np.float64)
    elif stale_beta_variance is not None:
        resolved = np.asarray(stale_beta_variance, dtype=np.float64)
    else:
        resolved = np.asarray(prior_variances, dtype=np.float64)
    if resolved.shape != np.asarray(prior_variances, dtype=np.float64).shape:
        raise ValueError("beta_variance state must match prior_variances shape.")
    return np.maximum(resolved, 1e-8)


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
    predictor_offset: np.ndarray | None = None,
    stale_beta_variance: np.ndarray | None = None,
    restricted_posterior_warm_start: _RestrictedPosteriorWarmStart | None = None,
    em_iteration_index: int | None = None,
    total_em_iterations: int | None = None,
    update_blend_weight: float | None = None,
    allow_gpu_exact_variant: bool = True,
) -> PosteriorState:
    log(f"    collapsed posterior: trait={trait_type.value}  n_variants={genotype_matrix.shape[1]}  n_samples={genotype_matrix.shape[0]}  sigma_e2={sigma_error2:.6f}  mem={mem()}")
    prior_variances = np.maximum(np.asarray(reduced_prior_variances, dtype=np.float64), 1e-8)
    # Mirror the gpu_available check in _solve_restricted_full so the
    # solver-controls function knows whether GPU CG (cheap ~30ms/iter) will be used.
    _collapsed_gpu_available = (
        genotype_matrix._cupy_cache is not None
        or _streaming_cupy_backend_available(genotype_matrix)
    )
    posterior_solver_tolerance, posterior_maximum_linear_solver_iterations = _collapsed_posterior_solver_controls(
        genotype_matrix=genotype_matrix,
        solver_tolerance=config.linear_solver_tolerance,
        maximum_linear_solver_iterations=config.maximum_linear_solver_iterations,
        compute_logdet=compute_logdet,
        compute_beta_variance=compute_beta_variance,
        gpu_enabled=_collapsed_gpu_available,
        em_iteration_index=em_iteration_index,
        total_em_iterations=total_em_iterations,
        update_blend_weight=update_blend_weight,
    )
    predictor_offset_array = (
        np.zeros(genotype_matrix.shape[0], dtype=np.float64)
        if predictor_offset is None
        else np.asarray(predictor_offset, dtype=np.float64).reshape(-1)
    )
    if predictor_offset_array.shape != (genotype_matrix.shape[0],):
        raise ValueError("predictor_offset must match genotype sample count.")
    if trait_type == TraitType.QUANTITATIVE:
        alpha, beta, beta_variance, linear_predictor, collapsed_objective, sigma_error2_new = _quantitative_posterior_state(
            genotype_matrix=genotype_matrix,
            covariate_matrix=covariate_matrix,
            targets=np.asarray(targets, dtype=np.float64) - predictor_offset_array,
            prior_variances=prior_variances,
            sigma_error2=max(float(sigma_error2), config.sigma_error_floor),
            sigma_error_floor=config.sigma_error_floor,
            solver_tolerance=posterior_solver_tolerance,
            maximum_linear_solver_iterations=posterior_maximum_linear_solver_iterations,
            logdet_probe_count=config.logdet_probe_count,
            logdet_lanczos_steps=config.logdet_lanczos_steps,
            exact_solver_matrix_limit=config.exact_solver_matrix_limit,
            posterior_variance_batch_size=config.posterior_variance_batch_size,
            posterior_variance_probe_count=config.posterior_variance_probe_count,
            random_seed=config.random_seed,
            compute_logdet=compute_logdet,
            compute_beta_variance=compute_beta_variance,
            sample_space_preconditioner_rank=config.sample_space_preconditioner_rank,
            posterior_working_sets=config.posterior_working_sets,
            posterior_working_set_min_variants=config.posterior_working_set_min_variants,
            posterior_working_set_initial_size=config.posterior_working_set_initial_size,
            posterior_working_set_growth=config.posterior_working_set_growth,
            posterior_working_set_max_passes=config.posterior_working_set_max_passes,
            posterior_working_set_coefficient_tolerance=config.posterior_working_set_coefficient_tolerance,
            stale_beta_variance=stale_beta_variance,
            restricted_posterior_warm_start=restricted_posterior_warm_start,
            update_blend_weight=update_blend_weight,
            allow_gpu_exact_variant=allow_gpu_exact_variant,
        )
        beta_variance = _effective_beta_variance_state(
            compute_beta_variance=compute_beta_variance,
            beta_variance=np.asarray(beta_variance, dtype=np.float64),
            stale_beta_variance=stale_beta_variance,
            prior_variances=prior_variances,
        )
        linear_predictor = np.asarray(linear_predictor, dtype=np.float64) + predictor_offset_array
    else:
        alpha, beta, beta_variance, linear_predictor, collapsed_objective, _ = _binary_posterior_state(
            genotype_matrix=genotype_matrix,
            covariate_matrix=covariate_matrix,
            targets=targets,
            prior_variances=prior_variances,
            alpha_init=np.asarray(alpha_init, dtype=np.float64),
            beta_init=np.asarray(beta_init, dtype=np.float64),
            minimum_weight=config.polya_gamma_minimum_weight,
            max_iterations=config.max_inner_newton_iterations,
            gradient_tolerance=config.newton_gradient_tolerance,
            solver_tolerance=posterior_solver_tolerance,
            maximum_linear_solver_iterations=posterior_maximum_linear_solver_iterations,
            logdet_probe_count=config.logdet_probe_count,
            logdet_lanczos_steps=config.logdet_lanczos_steps,
            exact_solver_matrix_limit=config.exact_solver_matrix_limit,
            posterior_variance_batch_size=config.posterior_variance_batch_size,
            posterior_variance_probe_count=config.posterior_variance_probe_count,
            random_seed=config.random_seed,
            compute_logdet=compute_logdet,
            compute_beta_variance=compute_beta_variance,
            sample_space_preconditioner_rank=config.sample_space_preconditioner_rank,
            predictor_offset=predictor_offset_array,
            posterior_working_sets=config.posterior_working_sets,
            posterior_working_set_min_variants=config.posterior_working_set_min_variants,
            posterior_working_set_initial_size=config.posterior_working_set_initial_size,
            posterior_working_set_growth=config.posterior_working_set_growth,
            posterior_working_set_max_passes=config.posterior_working_set_max_passes,
            posterior_working_set_coefficient_tolerance=config.posterior_working_set_coefficient_tolerance,
            restricted_posterior_warm_start=restricted_posterior_warm_start,
            allow_gpu_exact_variant=allow_gpu_exact_variant,
        )
        beta_variance = _effective_beta_variance_state(
            compute_beta_variance=compute_beta_variance,
            beta_variance=np.asarray(beta_variance, dtype=np.float64),
            stale_beta_variance=stale_beta_variance,
            prior_variances=prior_variances,
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
    posterior_working_sets: bool = True,
    posterior_working_set_min_variants: int = 65_536,
    posterior_working_set_initial_size: int = 16_384,
    posterior_working_set_growth: int = 16_384,
    posterior_working_set_max_passes: int = 6,
    posterior_working_set_coefficient_tolerance: float = 1e-4,
    stale_beta_variance: np.ndarray | None = None,
    restricted_posterior_warm_start: _RestrictedPosteriorWarmStart | None = None,
    update_blend_weight: float | None = None,
    allow_gpu_exact_variant: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    standardized_genotypes = _as_standardized_genotype_matrix(genotype_matrix)
    # Quantitative block updates only use blend weight to relax upstream solver
    # controls; the restricted solve itself still computes a full posterior.
    _ = update_blend_weight
    sample_count = standardized_genotypes.shape[0]
    if not compute_logdet and not compute_beta_variance:
        alpha, beta, _projected_targets, linear_predictor, restricted_quadratic = _solve_restricted_mean_only(
            genotype_matrix=standardized_genotypes,
            covariate_matrix=covariate_matrix,
            targets=targets,
            prior_variances=prior_variances,
            diagonal_noise=np.full(sample_count, sigma_error2, dtype=np.float64),
            solver_tolerance=solver_tolerance,
            maximum_linear_solver_iterations=maximum_linear_solver_iterations,
            exact_solver_matrix_limit=exact_solver_matrix_limit,
            posterior_variance_batch_size=posterior_variance_batch_size,
            random_seed=random_seed,
            sample_space_preconditioner_rank=sample_space_preconditioner_rank,
            posterior_working_sets=posterior_working_sets,
            posterior_working_set_min_variants=posterior_working_set_min_variants,
            posterior_working_set_initial_size=posterior_working_set_initial_size,
            posterior_working_set_growth=posterior_working_set_growth,
            posterior_working_set_max_passes=posterior_working_set_max_passes,
            posterior_working_set_coefficient_tolerance=posterior_working_set_coefficient_tolerance,
            warm_start=restricted_posterior_warm_start,
            allow_gpu_exact_variant=allow_gpu_exact_variant,
        )
        beta_variance = np.zeros_like(np.asarray(prior_variances, dtype=np.float64), dtype=np.float64)
        logdet_covariance = 0.0
        logdet_gls = 0.0
    else:
        alpha, beta, beta_variance, _projected_targets, linear_predictor, restricted_quadratic, logdet_covariance, logdet_gls = (
            _solve_restricted_full(
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
                posterior_working_sets=posterior_working_sets,
                posterior_working_set_min_variants=posterior_working_set_min_variants,
                posterior_working_set_initial_size=posterior_working_set_initial_size,
                posterior_working_set_growth=posterior_working_set_growth,
                posterior_working_set_max_passes=posterior_working_set_max_passes,
                posterior_working_set_coefficient_tolerance=posterior_working_set_coefficient_tolerance,
                warm_start=restricted_posterior_warm_start,
                allow_gpu_exact_variant=allow_gpu_exact_variant,
            )
        )
    # Re-estimate noise variance.  Naive approach (just use residuals) would
    # underestimate noise because the model "overfits" a little to noise.
    # Leverage correction accounts for this: variants the model is very
    # confident about (low variance relative to prior) contribute to the
    # correction term, preventing the noise estimate from shrinking too fast.
    effective_beta_variance = (
        np.asarray(beta_variance, dtype=np.float64)
        if compute_beta_variance or stale_beta_variance is None
        else np.asarray(stale_beta_variance, dtype=np.float64)
    )
    leverage_weight = np.maximum(prior_variances - effective_beta_variance, 0.0) / np.maximum(prior_variances, 1e-12)
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
    return alpha, beta, effective_beta_variance, linear_predictor, collapsed_objective, sigma_error2_new


def _binary_newton_solver_controls(
    genotype_matrix: StandardizedGenotypeMatrix,
    *,
    solver_tolerance: float,
    maximum_linear_solver_iterations: int,
    gpu_enabled: bool = False,
    update_blend_weight: float | None = None,
) -> tuple[float, int]:
    if genotype_matrix.shape[0] < 16_384 and genotype_matrix.shape[1] < 16_384:
        return solver_tolerance, maximum_linear_solver_iterations
    if update_blend_weight is not None:
        blend_weight = float(np.clip(update_blend_weight, 0.0, 1.0))
        if blend_weight < 0.08:
            relaxed_tolerance = max(float(solver_tolerance), 1e-2)
            iteration_cap = 128 if gpu_enabled else 64
        elif blend_weight < 0.20:
            relaxed_tolerance = max(float(solver_tolerance), 5e-3)
            iteration_cap = 192 if gpu_enabled else 96
        else:
            relaxed_tolerance = max(float(solver_tolerance), 2e-3)
            iteration_cap = 256 if gpu_enabled else 128
        if genotype_matrix.shape[1] > genotype_matrix.shape[0]:
            relaxed_tolerance = max(relaxed_tolerance, 5e-3)
            iteration_cap = min(iteration_cap, 160 if gpu_enabled else 80)
        return relaxed_tolerance, max(min(int(maximum_linear_solver_iterations), iteration_cap), 16)
    relaxed_tolerance = max(float(solver_tolerance), 1e-3)
    # GPU CG iterations are cheap (~30ms each with cached data), so allow more
    # iterations before declaring convergence failure on large blocks.
    iteration_cap = 512 if gpu_enabled else 128
    relaxed_maximum_iterations = min(int(maximum_linear_solver_iterations), iteration_cap)
    if genotype_matrix.shape[1] > genotype_matrix.shape[0]:
        relaxed_tolerance = max(relaxed_tolerance, 5e-3)
        wide_cap = 384 if gpu_enabled else 96
        relaxed_maximum_iterations = min(relaxed_maximum_iterations, wide_cap)
    return relaxed_tolerance, max(relaxed_maximum_iterations, 16)


def _collapsed_posterior_solver_controls(
    genotype_matrix: StandardizedGenotypeMatrix,
    *,
    solver_tolerance: float,
    maximum_linear_solver_iterations: int,
    compute_logdet: bool,
    compute_beta_variance: bool,
    gpu_enabled: bool = False,
    em_iteration_index: int | None = None,
    total_em_iterations: int | None = None,
    update_blend_weight: float | None = None,
) -> tuple[float, int]:
    if compute_logdet and compute_beta_variance:
        return float(solver_tolerance), int(maximum_linear_solver_iterations)
    resolved_tolerance = float(solver_tolerance)
    resolved_maximum_iterations = int(maximum_linear_solver_iterations)
    is_large_problem = genotype_matrix.shape[0] >= 16_384 or genotype_matrix.shape[1] >= 16_384

    # For stochastic quantitative block updates, the candidate solve is blended
    # back into the global state. Small step sizes do not justify a tight inner
    # CG solve even when the block itself is modest in size.
    if update_blend_weight is not None:
        blend_weight = float(np.clip(update_blend_weight, 0.0, 1.0))
        if blend_weight < 0.08:
            relaxed_tolerance = max(resolved_tolerance, 5e-3)
            iteration_cap = 128 if gpu_enabled else 64
        elif blend_weight < 0.20:
            relaxed_tolerance = max(resolved_tolerance, 2e-3)
            iteration_cap = 192 if gpu_enabled else 96
        else:
            relaxed_tolerance = max(resolved_tolerance, 1e-3)
            iteration_cap = 256 if gpu_enabled else 128
        relaxed_maximum_iterations = min(resolved_maximum_iterations, iteration_cap)
    elif is_large_problem:
        progress_fraction = 1.0
        if total_em_iterations is not None and total_em_iterations > 1 and em_iteration_index is not None:
            progress_fraction = float(np.clip(em_iteration_index / (total_em_iterations - 1), 0.0, 1.0))
        if progress_fraction < 1.0 / 3.0:
            relaxed_tolerance = max(resolved_tolerance, 1e-3)
            iteration_cap = 256 if gpu_enabled else 128
        elif progress_fraction < 2.0 / 3.0:
            relaxed_tolerance = max(resolved_tolerance, 3e-4)
            iteration_cap = 512 if gpu_enabled else 256
        else:
            relaxed_tolerance = max(resolved_tolerance, 1e-4)
            iteration_cap = 768 if gpu_enabled else 384
        relaxed_maximum_iterations = min(resolved_maximum_iterations, iteration_cap)
    else:
        return resolved_tolerance, resolved_maximum_iterations

    if genotype_matrix.shape[1] > genotype_matrix.shape[0]:
        wide_tolerance_floor = 1e-4 if compute_beta_variance else 1e-3
        relaxed_tolerance = max(relaxed_tolerance, wide_tolerance_floor)
        wide_cap = 384 if gpu_enabled else 192
        relaxed_maximum_iterations = min(relaxed_maximum_iterations, wide_cap)
    if not compute_beta_variance:
        relaxed_tolerance = max(relaxed_tolerance, 1e-3)
        no_variance_cap = 512 if gpu_enabled else 128
        relaxed_maximum_iterations = min(relaxed_maximum_iterations, no_variance_cap)
    return relaxed_tolerance, max(relaxed_maximum_iterations, 32)


def _binary_expected_polya_gamma_weights(
    linear_predictor: np.ndarray,
    minimum_weight: float,
) -> np.ndarray:
    absolute_predictor = np.abs(np.asarray(linear_predictor, dtype=np.float64))
    weights = np.empty_like(absolute_predictor, dtype=np.float64)
    tiny_mask = absolute_predictor < 1e-6
    weights[tiny_mask] = 0.25
    nonzero_predictor = absolute_predictor[~tiny_mask]
    if nonzero_predictor.size > 0:
        weights[~tiny_mask] = np.tanh(nonzero_predictor * 0.5) / (2.0 * nonzero_predictor)
    return np.maximum(weights, float(minimum_weight))


def _binary_penalized_log_posterior(
    linear_predictor: np.ndarray,
    targets: np.ndarray,
    prior_precision: np.ndarray,
    beta: np.ndarray,
) -> float:
    probabilities = np.asarray(stable_sigmoid(np.asarray(linear_predictor, dtype=np.float64)), dtype=np.float64)
    target_array = np.asarray(targets, dtype=np.float64)
    beta_array = np.asarray(beta, dtype=np.float64)
    prior_precision_array = np.asarray(prior_precision, dtype=np.float64)
    return float(
        np.sum(
            target_array * np.log(probabilities + 1e-12)
            + (1.0 - target_array) * np.log(1.0 - probabilities + 1e-12)
        )
        - 0.5 * np.sum(prior_precision_array * beta_array * beta_array)
    )


def _cupy_array_to_numpy(array, *, dtype: np.dtype | type[np.floating[Any]]) -> np.ndarray:
    host_array = array.get() if hasattr(array, "get") else array
    return np.asarray(host_array, dtype=dtype)


def _softplus_cupy(cp, values_gpu, *, dtype):
    positive_branch = values_gpu >= dtype(0.0)
    return cp.where(
        positive_branch,
        values_gpu + cp.log1p(cp.exp(-values_gpu)),
        cp.log1p(cp.exp(values_gpu)),
    )


def _binary_expected_polya_gamma_weights_cupy(
    cp,
    linear_predictor_gpu,
    minimum_weight: float,
    *,
    dtype,
):
    absolute_predictor = cp.abs(linear_predictor_gpu)
    tiny_mask = absolute_predictor < dtype(1e-6)
    safe_predictor = cp.where(tiny_mask, dtype(1.0), absolute_predictor)
    nonzero_weights = cp.tanh(safe_predictor * dtype(0.5)) / (dtype(2.0) * safe_predictor)
    weights = cp.where(tiny_mask, dtype(0.25), nonzero_weights)
    return cp.maximum(weights, dtype(minimum_weight))


def _binary_penalized_log_posterior_cupy(
    cp,
    linear_predictor_gpu,
    targets_gpu,
    prior_precision_gpu,
    beta_gpu,
    *,
    dtype,
) -> float:
    return float(
        cp.sum(
            targets_gpu * linear_predictor_gpu - _softplus_cupy(cp, linear_predictor_gpu, dtype=dtype),
            dtype=cp.float64,
        )
        - dtype(0.5) * cp.sum(prior_precision_gpu * beta_gpu * beta_gpu, dtype=cp.float64)
    )


# Fit effect sizes for a binary trait (e.g. disease case/control) by
# alternating between:
#   1. updating expected Polya-Gamma latent weights from the current
#      logistic linear predictor, and
#   2. solving the resulting Gaussian subproblem with the same restricted
#      posterior machinery used by the quantitative path.
#
# This avoids trust-region accept/reject loops and turns each binary
# iteration into one posterior solve plus one weight refresh.
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
    posterior_working_sets: bool = True,
    posterior_working_set_min_variants: int = 65_536,
    posterior_working_set_initial_size: int = 16_384,
    posterior_working_set_growth: int = 16_384,
    posterior_working_set_max_passes: int = 6,
    posterior_working_set_coefficient_tolerance: float = 1e-4,
    restricted_posterior_warm_start: _RestrictedPosteriorWarmStart | None = None,
    update_blend_weight: float | None = None,
    resume_state: dict[str, object] | None = None,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    allow_gpu_exact_variant: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int]:
    standardized_genotypes = _as_standardized_genotype_matrix(genotype_matrix)
    prior_precision = np.asarray(1.0 / np.maximum(prior_variances, 1e-8), dtype=np.float64)
    covariate_count = covariate_matrix.shape[1]
    parameters = np.concatenate([alpha_init, beta_init], axis=0).astype(np.float64, copy=True)
    predictor_offset_array = (
        np.zeros(standardized_genotypes.shape[0], dtype=np.float64)
        if predictor_offset is None
        else np.asarray(predictor_offset, dtype=np.float64).reshape(-1)
    )
    if predictor_offset_array.shape != (standardized_genotypes.shape[0],):
        raise ValueError("predictor_offset must match genotype sample count.")
    covariate_matrix_f64 = np.asarray(covariate_matrix, dtype=np.float64)
    target_array = np.asarray(targets, dtype=np.float64).reshape(-1)
    kappa = target_array - 0.5
    warm_start = (
        _RestrictedPosteriorWarmStart()
        if restricted_posterior_warm_start is None
        else restricted_posterior_warm_start
    )
    current_linear_predictor_gpu = None
    best_linear_predictor_gpu = None
    log(f"      binary setup: {standardized_genotypes.shape[1]} variants, {standardized_genotypes.shape[0]} samples  mem={mem()}")
    cupy = None
    gpu_binary_backend = standardized_genotypes._cupy_cache is not None or (
        standardized_genotypes.raw is not None
        and not standardized_genotypes.supports_jax_dense_ops()
    )
    if gpu_binary_backend:
        cupy = _try_import_cupy()
        gpu_binary_backend = cupy is not None
    if gpu_binary_backend:
        assert cupy is not None
        log(f"      binary GPU backend active  mem={mem()}")
        compute_cp_dtype = _cupy_compute_dtype(cupy)
        covariate_matrix_gpu = cupy.asarray(covariate_matrix_f64, dtype=compute_cp_dtype)
        target_array_gpu = cupy.asarray(target_array, dtype=compute_cp_dtype)
        prior_precision_gpu = cupy.asarray(prior_precision, dtype=compute_cp_dtype)
        predictor_offset_gpu = cupy.asarray(predictor_offset_array, dtype=compute_cp_dtype)
    # Mirror the gpu_available check in _solve_restricted_full so the
    # solver-controls function knows whether GPU CG (cheap ~30ms/iter) will be used.
    _newton_gpu_available = (
        standardized_genotypes._cupy_cache is not None
        or _streaming_cupy_backend_available(standardized_genotypes)
    )
    inexact_solver_tolerance, inexact_maximum_linear_solver_iterations = _binary_newton_solver_controls(
        standardized_genotypes,
        solver_tolerance=solver_tolerance,
        maximum_linear_solver_iterations=maximum_linear_solver_iterations,
        gpu_enabled=_newton_gpu_available,
        update_blend_weight=update_blend_weight,
    )
    def _build_resume_snapshot(
        *,
        completed_iterations: int,
        parameters_state: np.ndarray,
        current_linear_predictor_state: np.ndarray,
        current_objective_state: float,
        best_parameters_state: np.ndarray,
        best_linear_predictor_state: np.ndarray,
        best_objective_state: float,
        stall_count_state: int,
    ) -> dict[str, object]:
        return {
            "completed_iterations": int(completed_iterations),
            "parameters": np.asarray(parameters_state, dtype=np.float64).copy(),
            "current_linear_predictor": np.asarray(current_linear_predictor_state, dtype=np.float64).copy(),
            "current_objective": float(current_objective_state),
            "best_parameters": np.asarray(best_parameters_state, dtype=np.float64).copy(),
            "best_linear_predictor": np.asarray(best_linear_predictor_state, dtype=np.float64).copy(),
            "best_objective": float(best_objective_state),
            "stall_count": int(stall_count_state),
        }

    resume_completed_iterations = 0
    if resume_state is not None:
        resume_completed_iterations = int(resume_state.get("completed_iterations", 0))
        if resume_completed_iterations < 0 or resume_completed_iterations > max_iterations:
            raise ValueError("binary resume_state completed_iterations is out of range.")
        parameters = np.asarray(resume_state["parameters"], dtype=np.float64).copy()
        current_linear_predictor = np.asarray(
            resume_state["current_linear_predictor"],
            dtype=np.float64,
        ).copy()
        current_objective = float(resume_state["current_objective"])
        best_parameters = np.asarray(
            resume_state.get("best_parameters", parameters),
            dtype=np.float64,
        ).copy()
        best_linear_predictor = np.asarray(
            resume_state.get("best_linear_predictor", current_linear_predictor),
            dtype=np.float64,
        ).copy()
        best_objective = float(resume_state.get("best_objective", current_objective))
        stall_count = int(resume_state.get("stall_count", 0))
        if parameters.shape != (covariate_count + standardized_genotypes.shape[1],):
            raise ValueError("binary resume_state parameters shape does not match the block.")
        if current_linear_predictor.shape != (standardized_genotypes.shape[0],):
            raise ValueError("binary resume_state current_linear_predictor shape does not match samples.")
        if best_parameters.shape != parameters.shape:
            raise ValueError("binary resume_state best_parameters shape does not match parameters.")
        if best_linear_predictor.shape != current_linear_predictor.shape:
            raise ValueError("binary resume_state best_linear_predictor shape does not match samples.")
        if gpu_binary_backend:
            assert cupy is not None
            current_linear_predictor_gpu = cupy.asarray(current_linear_predictor, dtype=compute_cp_dtype)
            best_linear_predictor_gpu = cupy.asarray(best_linear_predictor, dtype=compute_cp_dtype)
        log(
            "      resuming binary PG updates "
            + f"from iter {resume_completed_iterations + 1}/{max_iterations}  mem={mem()}"
        )
    else:
        log(f"      computing initial linear predictor...  mem={mem()}")
        _init_pred_t0 = time.monotonic()
        if gpu_binary_backend:
            assert cupy is not None
            current_linear_predictor_gpu = (
                predictor_offset_gpu
                + covariate_matrix_gpu @ cupy.asarray(parameters[:covariate_count], dtype=compute_cp_dtype)
                + standardized_genotypes.gpu_matmat(
                    cupy.asarray(parameters[covariate_count:], dtype=compute_cp_dtype),
                    batch_size=posterior_variance_batch_size,
                    cupy=cupy,
                    dtype=compute_cp_dtype,
                )
            )
            current_linear_predictor = _cupy_array_to_numpy(current_linear_predictor_gpu, dtype=np.float64)
            current_objective = _binary_penalized_log_posterior_cupy(
                cupy,
                current_linear_predictor_gpu,
                target_array_gpu,
                prior_precision_gpu,
                cupy.asarray(parameters[covariate_count:], dtype=compute_cp_dtype),
                dtype=compute_cp_dtype,
            )
        else:
            current_linear_predictor = predictor_offset_array + np.asarray(
                covariate_matrix_f64 @ parameters[:covariate_count]
                + standardized_genotypes.matvec_numpy(
                    parameters[covariate_count:],
                    batch_size=posterior_variance_batch_size,
                ),
                dtype=np.float64,
            )
            current_objective = _binary_penalized_log_posterior(
                linear_predictor=current_linear_predictor,
                targets=target_array,
                prior_precision=prior_precision,
                beta=parameters[covariate_count:],
            )
        _init_pred_seconds = time.monotonic() - _init_pred_t0
        log(f"      initial predictor computed in {_init_pred_seconds:.1f}s  obj={current_objective:.4f}  mem={mem()}")
        best_objective = current_objective
        best_parameters = parameters.copy()
        best_linear_predictor = current_linear_predictor.copy()
        best_linear_predictor_gpu = current_linear_predictor_gpu if gpu_binary_backend else None
        stall_count = 0
    log(
        f"      binary PG updates: {standardized_genotypes.shape[1]} variants, "
        + f"max_iter={max_iterations}  mem={mem()}"
    )
    final_weights = _binary_expected_polya_gamma_weights(current_linear_predictor, minimum_weight)
    best_weights = _binary_expected_polya_gamma_weights(best_linear_predictor, minimum_weight)
    _binary_newton_iters_used = resume_completed_iterations
    for iteration_index in range(resume_completed_iterations, max_iterations):
        _binary_newton_iters_used = iteration_index + 1
        iteration_start = time.monotonic()
        if gpu_binary_backend:
            assert cupy is not None
            current_weights_gpu = _binary_expected_polya_gamma_weights_cupy(
                cupy,
                current_linear_predictor_gpu,
                minimum_weight,
                dtype=compute_cp_dtype,
            )
            current_weights = _cupy_array_to_numpy(current_weights_gpu, dtype=np.float64)
        else:
            current_weights = _binary_expected_polya_gamma_weights(current_linear_predictor, minimum_weight)
        pseudo_response = kappa / current_weights - predictor_offset_array
        solve_start = time.monotonic()
        updated_alpha, updated_beta, _projected_targets, updated_fitted_response, _restricted_quadratic = (
            _solve_restricted_mean_only(
                genotype_matrix=standardized_genotypes,
                covariate_matrix=covariate_matrix,
                targets=pseudo_response,
                prior_variances=prior_variances,
                diagonal_noise=1.0 / current_weights,
                solver_tolerance=inexact_solver_tolerance,
                maximum_linear_solver_iterations=inexact_maximum_linear_solver_iterations,
                exact_solver_matrix_limit=exact_solver_matrix_limit,
                posterior_variance_batch_size=posterior_variance_batch_size,
                random_seed=random_seed + iteration_index,
                initial_beta_guess=parameters[covariate_count:],
                sample_space_preconditioner_rank=sample_space_preconditioner_rank,
                posterior_working_sets=posterior_working_sets,
                posterior_working_set_min_variants=posterior_working_set_min_variants,
                posterior_working_set_initial_size=posterior_working_set_initial_size,
                posterior_working_set_growth=posterior_working_set_growth,
                posterior_working_set_max_passes=posterior_working_set_max_passes,
                posterior_working_set_coefficient_tolerance=posterior_working_set_coefficient_tolerance,
                warm_start=warm_start,
                allow_gpu_exact_variant=allow_gpu_exact_variant,
            )
        )
        solve_seconds = time.monotonic() - solve_start
        updated_parameters = np.concatenate([updated_alpha, updated_beta], axis=0).astype(np.float64, copy=False)
        updated_linear_predictor = predictor_offset_array + np.asarray(updated_fitted_response, dtype=np.float64)
        if gpu_binary_backend:
            assert cupy is not None
            updated_linear_predictor_gpu = cupy.asarray(updated_linear_predictor, dtype=compute_cp_dtype)
            updated_objective = _binary_penalized_log_posterior_cupy(
                cupy,
                updated_linear_predictor_gpu,
                target_array_gpu,
                prior_precision_gpu,
                cupy.asarray(updated_beta, dtype=compute_cp_dtype),
                dtype=compute_cp_dtype,
            )
        else:
            updated_objective = _binary_penalized_log_posterior(
                linear_predictor=updated_linear_predictor,
                targets=target_array,
                prior_precision=prior_precision,
                beta=updated_beta,
            )
        relative_parameter_step = float(np.linalg.norm(updated_parameters - parameters)) / max(
            float(np.linalg.norm(parameters)),
            1e-8,
        )
        relative_predictor_step = float(np.linalg.norm(updated_linear_predictor - current_linear_predictor)) / max(
            float(np.linalg.norm(current_linear_predictor)),
            max(np.sqrt(float(updated_linear_predictor.shape[0])), 1.0),
        )
        effective_update_scale = (
            float(np.clip(update_blend_weight, 0.0, 1.0))
            if update_blend_weight is not None
            else 1.0
        )
        effective_parameter_step = effective_update_scale * relative_parameter_step
        effective_predictor_step = effective_update_scale * relative_predictor_step
        objective_gain = updated_objective - current_objective
        total_seconds = time.monotonic() - iteration_start
        log(
            f"      binary iter {iteration_index + 1}/{max_iterations}: "
            + f"obj={updated_objective:.4f} gain={objective_gain:.2e} "
            + f"param_step={relative_parameter_step:.2e} predictor_step={relative_predictor_step:.2e} "
            + f"applied_predictor_step={effective_predictor_step:.2e} "
            + f"mean_omega={float(np.mean(current_weights)):.3e} "
            + f"[solve={solve_seconds:.1f}s total={total_seconds:.1f}s]  mem={mem()}"
        )
        parameters = updated_parameters
        current_linear_predictor = updated_linear_predictor
        if gpu_binary_backend:
            current_linear_predictor_gpu = updated_linear_predictor_gpu
        current_objective = updated_objective
        final_weights = current_weights
        # Stall detection: track whether the objective has improved.
        if updated_objective > best_objective:
            best_objective = updated_objective
            best_parameters = parameters.copy()
            best_linear_predictor = current_linear_predictor.copy()
            best_linear_predictor_gpu = current_linear_predictor_gpu if gpu_binary_backend else None
            best_weights = final_weights.copy()
            stall_count = 0
        else:
            stall_count += 1
            if stall_count >= 3:
                log(
                    f"      binary stall-break at iter {iteration_index + 1}: "
                    f"objective has not improved for {stall_count} consecutive iterations "
                    f"(best={best_objective:.4f}, current={updated_objective:.4f}), "
                    f"reverting to best-seen state"
                )
                parameters = best_parameters
                current_linear_predictor = best_linear_predictor
                if gpu_binary_backend:
                    current_linear_predictor_gpu = best_linear_predictor_gpu
                current_objective = best_objective
                final_weights = best_weights
                break
        if progress_callback is not None:
            progress_callback(
                _build_resume_snapshot(
                    completed_iterations=iteration_index + 1,
                    parameters_state=parameters,
                    current_linear_predictor_state=current_linear_predictor,
                    current_objective_state=current_objective,
                    best_parameters_state=best_parameters,
                    best_linear_predictor_state=best_linear_predictor,
                    best_objective_state=best_objective,
                    stall_count_state=stall_count,
                )
            )
        if (
            effective_predictor_step <= gradient_tolerance
            or max(effective_parameter_step, effective_predictor_step) <= gradient_tolerance
        ):
            log(
                f"      binary converged at iter {iteration_index + 1}: "
                + f"param_step={relative_parameter_step:.2e} predictor_step={relative_predictor_step:.2e} "
                + f"applied_predictor_step={effective_predictor_step:.2e} "
                + f"tol={gradient_tolerance:.2e}"
            )
            break

    # Skip the expensive final re-solve when neither logdet nor beta_variance is
    # needed. The Newton loop already converged to the same beta — the final solve
    # only adds tighter tolerance. For stochastic blocks (compute_logdet=False,
    # compute_beta_variance=False), this saves ~10s per block.
    if not compute_logdet and not compute_beta_variance:
        final_alpha = parameters[:covariate_count]
        final_beta = parameters[covariate_count:]
        beta_variance = np.zeros_like(np.asarray(prior_variances, dtype=np.float64), dtype=np.float64)
        _fitted_response = current_linear_predictor - predictor_offset_array
        logdet_covariance = 0.0
        logdet_gls = 0.0
    else:
        final_pseudo_response = kappa / final_weights - predictor_offset_array
        try:
            final_alpha, final_beta, beta_variance, _projected_targets, _fitted_response, _restricted_quadratic, logdet_covariance, logdet_gls = (
                _solve_restricted_full(
                    genotype_matrix=standardized_genotypes,
                    covariate_matrix=covariate_matrix,
                    targets=final_pseudo_response,
                    prior_variances=prior_variances,
                    diagonal_noise=1.0 / final_weights,
                    solver_tolerance=solver_tolerance,
                    maximum_linear_solver_iterations=maximum_linear_solver_iterations,
                    logdet_probe_count=logdet_probe_count,
                    logdet_lanczos_steps=logdet_lanczos_steps,
                    exact_solver_matrix_limit=exact_solver_matrix_limit,
                    posterior_variance_batch_size=posterior_variance_batch_size,
                    posterior_variance_probe_count=posterior_variance_probe_count,
                    random_seed=random_seed + max_iterations + 17,
                    compute_logdet=compute_logdet,
                    compute_beta_variance=compute_beta_variance,
                    initial_beta_guess=parameters[covariate_count:],
                    sample_space_preconditioner_rank=sample_space_preconditioner_rank,
                    posterior_working_sets=posterior_working_sets,
                    posterior_working_set_min_variants=posterior_working_set_min_variants,
                    posterior_working_set_initial_size=posterior_working_set_initial_size,
                    posterior_working_set_growth=posterior_working_set_growth,
                    posterior_working_set_max_passes=posterior_working_set_max_passes,
                    posterior_working_set_coefficient_tolerance=posterior_working_set_coefficient_tolerance,
                    warm_start=warm_start,
                    allow_gpu_exact_variant=allow_gpu_exact_variant,
                )
            )
        except RuntimeError as exc:
            log(f"      final posterior solve failed ({exc}), using last Newton iteration result")
            final_alpha = parameters[:covariate_count]
            final_beta = parameters[covariate_count:]
            beta_variance = np.zeros_like(np.asarray(prior_variances, dtype=np.float64), dtype=np.float64)
            _fitted_response = current_linear_predictor - predictor_offset_array
            logdet_covariance = 0.0
            logdet_gls = 0.0
    final_linear_predictor = predictor_offset_array + np.asarray(_fitted_response, dtype=np.float64)
    if gpu_binary_backend:
        assert cupy is not None
        final_linear_predictor_gpu = cupy.asarray(final_linear_predictor, dtype=compute_cp_dtype)
        final_weights = _cupy_array_to_numpy(
            _binary_expected_polya_gamma_weights_cupy(
                cupy,
                final_linear_predictor_gpu,
                minimum_weight,
                dtype=compute_cp_dtype,
            ),
            dtype=np.float64,
        )
    else:
        final_weights = _binary_expected_polya_gamma_weights(final_linear_predictor, minimum_weight)
    final_parameters = np.concatenate([final_alpha, final_beta], axis=0).astype(np.float64, copy=False)
    if gpu_binary_backend:
        assert cupy is not None
        final_objective = _binary_penalized_log_posterior_cupy(
            cupy,
            final_linear_predictor_gpu,
            target_array_gpu,
            prior_precision_gpu,
            cupy.asarray(final_beta, dtype=compute_cp_dtype),
            dtype=compute_cp_dtype,
        )
    else:
        final_objective = _binary_penalized_log_posterior(
            linear_predictor=final_linear_predictor,
            targets=target_array,
            prior_precision=prior_precision,
            beta=final_beta,
        )
    logdet_hessian = (
        float(np.sum(np.log(np.maximum(prior_precision, 1e-12))))
        + float(np.sum(np.log(np.maximum(final_weights, 1e-12))))
        + (logdet_covariance + logdet_gls if compute_logdet else 0.0)
    )
    laplace_objective = final_objective - 0.5 * logdet_hessian
    log(f"      binary posterior done: laplace_obj={laplace_objective:.4f}  final_obj={final_objective:.4f}  logdet_hessian={logdet_hessian:.4f}  mem={mem()}")
    return (
        final_parameters[:covariate_count],
        final_parameters[covariate_count:],
        beta_variance,
        final_linear_predictor,
        float(laplace_objective),
        _binary_newton_iters_used,
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
    streaming_dtype = gpu_compute_numpy_dtype()
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
            raw_matrix = cast(RawGenotypeMatrix, genotype_matrix.raw)
            compute_cp_dtype = _cupy_compute_dtype(cupy)
            vector_gpu = _to_cupy_compute(v)
            result_gpu = diag_noise_gpu * vector_gpu
            for batch_slice, standardized_batch in _iter_standardized_gpu_batches(
                raw_matrix,
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
            raw_matrix = cast(RawGenotypeMatrix, genotype_matrix.raw)
            compute_cp_dtype = _cupy_compute_dtype(cupy)
            matrix_gpu = _to_cupy_compute(matrix_jax)
            result_gpu = diag_noise_gpu[:, None] * matrix_gpu
            for batch_slice, standardized_batch in _iter_standardized_gpu_batches(
                raw_matrix,
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
        pv = cp.asarray(prior_variances, dtype=compute_cp_dtype)
        # diag(X @ diag(pv) @ X^T) = sum(X^2 * pv, axis=1)
        # Chunked to avoid allocating a full (n, p) intermediate on GPU.
        diag_gpu = cp.zeros(genotype_matrix.shape[0], dtype=compute_cp_dtype)
        for batch_slice, x_chunk in _iter_cupy_cache_standardized_batches(
            genotype_matrix._cupy_cache,
            sample_count=genotype_matrix.shape[0],
            batch_size=512,
            cupy=cp,
            dtype=compute_cp_dtype,
        ):
            diag_gpu += cp.sum(x_chunk * x_chunk * pv[batch_slice], axis=1)
        result = np.asarray(diagonal_noise, dtype=np.float64) + _cupy_array_to_numpy(diag_gpu, dtype=np.float64)
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


def _sample_space_genotype_gram_matmat_cpu(
    genotype_matrix: StandardizedGenotypeMatrix,
    matrix: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    rhs = np.asarray(matrix, dtype=np.float64)
    vector_input = rhs.ndim == 1
    if vector_input:
        rhs = rhs[:, None]
    elif rhs.ndim != 2:
        raise ValueError("sample-space genotype Gram expects a vector or matrix right-hand side.")
    gram_term = np.zeros((genotype_matrix.shape[0], rhs.shape[1]), dtype=np.float64)
    for batch in genotype_matrix.iter_column_batches(batch_size=batch_size):
        genotype_batch = np.asarray(batch.values, dtype=np.float64)
        gram_term += genotype_batch @ (genotype_batch.T @ rhs)
    return gram_term[:, 0] if vector_input else gram_term


def _cached_sample_space_basis(
    basis_cache: dict[tuple[int, int], Any],
    *,
    requested_rank: int,
    random_seed: int,
):
    matching_basis: Any | None = None
    matching_width: int | None = None
    for (_cached_rank, cached_seed), cached_basis in basis_cache.items():
        if cached_seed != random_seed:
            continue
        basis_width = int(cached_basis.shape[1])
        if basis_width < requested_rank:
            continue
        if matching_width is None or basis_width < matching_width:
            matching_basis = cached_basis
            matching_width = basis_width
    return matching_basis


def _sample_space_nystrom_basis_cpu(
    genotype_matrix: StandardizedGenotypeMatrix,
    batch_size: int,
    rank: int,
    random_seed: int,
) -> np.ndarray | None:
    requested_rank = min(max(int(rank), 0), int(genotype_matrix.shape[0]))
    if requested_rank <= 0:
        return None
    cached_basis = _cached_sample_space_basis(
        genotype_matrix._sample_space_nystrom_basis_cpu_cache,
        requested_rank=requested_rank,
        random_seed=random_seed,
    )
    if cached_basis is not None:
        return np.asarray(cached_basis[:, :requested_rank], dtype=np.float64)
    sketch_rank = _sample_space_nystrom_rank(requested_rank, genotype_matrix.shape[0])
    if sketch_rank <= 0:
        return None
    sketch_probes = _orthogonal_probe_matrix(
        dimension=genotype_matrix.shape[0],
        probe_count=sketch_rank,
        random_seed=random_seed,
    )
    sketch_response = _sample_space_genotype_gram_matmat_cpu(
        genotype_matrix=genotype_matrix,
        matrix=sketch_probes,
        batch_size=batch_size,
    )
    basis_matrix, triangular_matrix = np.linalg.qr(np.asarray(sketch_response, dtype=np.float64), mode="reduced")
    diagonal = np.abs(np.diag(triangular_matrix))
    effective_rank = min(requested_rank, int(np.sum(diagonal > 1e-10)))
    if effective_rank <= 0:
        return None
    basis_matrix = np.asarray(basis_matrix[:, :effective_rank], dtype=np.float64)
    genotype_matrix._sample_space_nystrom_basis_cpu_cache[(effective_rank, int(random_seed))] = basis_matrix
    return basis_matrix

def _sample_space_nystrom_basis_gpu(
    genotype_matrix: StandardizedGenotypeMatrix,
    batch_size: int,
    rank: int,
    random_seed: int,
):
    cupy = _try_import_cupy()
    if cupy is None:
        return None
    compute_cp_dtype = _cupy_compute_dtype(cupy)
    requested_rank = min(max(int(rank), 0), int(genotype_matrix.shape[0]))
    if requested_rank <= 0:
        return None
    cached_basis_gpu = _cached_sample_space_basis(
        genotype_matrix._sample_space_nystrom_basis_gpu_cache,
        requested_rank=requested_rank,
        random_seed=random_seed,
    )
    if cached_basis_gpu is not None:
        return cached_basis_gpu[:, :requested_rank]
    cached_basis_cpu = _cached_sample_space_basis(
        genotype_matrix._sample_space_nystrom_basis_cpu_cache,
        requested_rank=requested_rank,
        random_seed=random_seed,
    )
    if cached_basis_cpu is not None:
        basis_gpu = cupy.asarray(np.asarray(cached_basis_cpu[:, :requested_rank], dtype=np.float64), dtype=compute_cp_dtype)
        genotype_matrix._sample_space_nystrom_basis_gpu_cache[(basis_gpu.shape[1], int(random_seed))] = basis_gpu
        return basis_gpu
    sketch_rank = _sample_space_nystrom_rank(requested_rank, genotype_matrix.shape[0])
    if sketch_rank <= 0 or not hasattr(cupy.linalg, "qr") or not hasattr(cupy, "abs") or not hasattr(cupy, "diag"):
        return None
    sketch_probes_gpu = cupy.asarray(
        _orthogonal_probe_matrix(
            dimension=genotype_matrix.shape[0],
            probe_count=sketch_rank,
            random_seed=random_seed,
        ),
        dtype=compute_cp_dtype,
    )
    sketch_response_gpu = _apply_sample_space_operator_gpu(
        genotype_matrix=genotype_matrix,
        prior_variances=np.ones(genotype_matrix.shape[1], dtype=np.float64),
        diagonal_noise=np.zeros(genotype_matrix.shape[0], dtype=np.float64),
        matrix_gpu=sketch_probes_gpu,
        batch_size=batch_size,
        cp=cupy,
        dtype=compute_cp_dtype,
    )
    try:
        basis_gpu, triangular_gpu = cupy.linalg.qr(sketch_response_gpu, mode="reduced")
    except TypeError:
        basis_gpu, triangular_gpu = cupy.linalg.qr(sketch_response_gpu)
    diagonal = _cupy_array_to_numpy(cupy.abs(cupy.diag(triangular_gpu)), dtype=np.float64)
    effective_rank = min(requested_rank, int(np.sum(diagonal > 1e-10)))
    if effective_rank <= 0:
        return None
    basis_gpu = cupy.asarray(basis_gpu[:, :effective_rank], dtype=compute_cp_dtype)
    basis_cpu = np.asarray(basis_gpu.get() if hasattr(basis_gpu, "get") else basis_gpu, dtype=np.float64)
    genotype_matrix._sample_space_nystrom_basis_cpu_cache[(effective_rank, int(random_seed))] = basis_cpu
    genotype_matrix._sample_space_nystrom_basis_gpu_cache[(effective_rank, int(random_seed))] = basis_gpu
    return basis_gpu


def _sample_space_nystrom_factor_cpu(
    genotype_matrix: StandardizedGenotypeMatrix,
    prior_variances: np.ndarray,
    batch_size: int,
    rank: int,
    random_seed: int,
) -> np.ndarray | None:
    basis_matrix = _sample_space_nystrom_basis_cpu(
        genotype_matrix=genotype_matrix,
        batch_size=batch_size,
        rank=rank,
        random_seed=random_seed,
    )
    return _sample_space_nystrom_factor_cpu_from_basis(
        genotype_matrix=genotype_matrix,
        prior_variances=prior_variances,
        basis_matrix=basis_matrix,
        batch_size=batch_size,
    )


def _sample_space_nystrom_factor_cpu_from_basis(
    genotype_matrix: StandardizedGenotypeMatrix,
    prior_variances: np.ndarray,
    basis_matrix: np.ndarray | None,
    *,
    batch_size: int,
) -> np.ndarray | None:
    if basis_matrix is None:
        return None
    basis_matrix = np.asarray(basis_matrix, dtype=np.float64)
    if basis_matrix.ndim != 2 or basis_matrix.shape[1] == 0:
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
    compute_cp_dtype = _cupy_compute_dtype(cupy)
    basis_gpu = _sample_space_nystrom_basis_gpu(
        genotype_matrix=genotype_matrix,
        batch_size=batch_size,
        rank=rank,
        random_seed=random_seed,
    )
    return _sample_space_nystrom_factor_gpu_from_basis(
        genotype_matrix=genotype_matrix,
        prior_variances=prior_variances,
        basis_gpu=basis_gpu,
        batch_size=batch_size,
    )


def _sample_space_nystrom_factor_gpu_from_basis(
    genotype_matrix: StandardizedGenotypeMatrix,
    prior_variances: np.ndarray,
    basis_gpu,
    *,
    batch_size: int,
):
    cupy = _try_import_cupy()
    if cupy is None:
        return None
    compute_cp_dtype = _cupy_compute_dtype(cupy)
    if basis_gpu is None:
        return None
    basis_gpu = cupy.asarray(basis_gpu, dtype=compute_cp_dtype)
    if basis_gpu.ndim != 2 or int(basis_gpu.shape[1]) == 0:
        return None
    projected_kernel_gpu = _apply_sample_space_operator_gpu(
        genotype_matrix=genotype_matrix,
        prior_variances=prior_variances,
        diagonal_noise=np.zeros(genotype_matrix.shape[0], dtype=np.float64),
        matrix_gpu=basis_gpu,
        batch_size=batch_size,
        cp=cupy,
        dtype=compute_cp_dtype,
    )
    gram_factor = _positive_semidefinite_factor(_cupy_array_to_numpy(basis_gpu.T @ projected_kernel_gpu, dtype=np.float64))
    if gram_factor is None:
        return None
    return basis_gpu @ cupy.asarray(gram_factor, dtype=compute_cp_dtype)


def _build_sample_space_low_rank_bundle_gpu(
    low_rank_factor_gpu,
    diagonal_preconditioner_gpu,
    *,
    cp,
    bundle_dtype,
):
    effective_rank = int(low_rank_factor_gpu.shape[1])
    weighted_selected_genotypes = cp.asarray(low_rank_factor_gpu, dtype=bundle_dtype)
    diagonal_vector = cp.asarray(diagonal_preconditioner_gpu, dtype=bundle_dtype)
    base_diagonal = cp.maximum(
        diagonal_vector - cp.sum(weighted_selected_genotypes * weighted_selected_genotypes, axis=1),
        bundle_dtype(1e-8),
    )
    inverse_base_diagonal = bundle_dtype(1.0) / base_diagonal
    weighted_inverse_selected = inverse_base_diagonal[:, None] * weighted_selected_genotypes
    low_rank_precision = cp.eye(effective_rank, dtype=bundle_dtype) + (
        weighted_selected_genotypes.T @ weighted_inverse_selected
    )
    jitter_scale = 1e-6 if bundle_dtype == cp.float32 else 1e-8
    low_rank_cholesky = cp.linalg.cholesky(
        low_rank_precision + cp.eye(effective_rank, dtype=bundle_dtype) * bundle_dtype(jitter_scale)
    )
    return weighted_selected_genotypes, inverse_base_diagonal, weighted_inverse_selected, low_rank_cholesky


def _apply_sample_space_low_rank_preconditioner_gpu(
    right_hand_side_gpu,
    bundle,
    *,
    cp,
    solve_triangular_gpu,
):
    weighted_selected_genotypes, inverse_base_diagonal, weighted_inverse_selected, low_rank_cholesky = bundle
    right_hand_side_gpu = cp.asarray(right_hand_side_gpu, dtype=low_rank_cholesky.dtype)
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
        solve_triangular_gpu,
    )
    return weighted_rhs - correction


def _relative_array_change(current: np.ndarray, previous: np.ndarray) -> float:
    current_array = np.asarray(current, dtype=np.float64)
    previous_array = np.asarray(previous, dtype=np.float64)
    if current_array.shape != previous_array.shape:
        return np.inf
    denominator = np.maximum(np.abs(previous_array), 1e-8)
    relative_change = np.abs(current_array - previous_array) / denominator
    # Preconditioner quality depends on the overall operator drift, not on the
    # single most volatile entry. Polya-Gamma weights routinely produce a few
    # outliers that should not force a full rebuild when the bulk of the system
    # is still well approximated by the cached preconditioner.
    return float(np.quantile(relative_change, 0.95))


def _sample_space_preconditioner_stale(cache_entry: _SampleSpacePreconditionerCacheEntry) -> bool:
    if cache_entry.global_background:
        return False
    if cache_entry.previous_iterations is None or cache_entry.last_iterations is None:
        return False
    return cache_entry.last_iterations > max(int(cache_entry.previous_iterations * 1.5), cache_entry.previous_iterations + 8)


def _can_reuse_sample_space_preconditioner(
    cache_entry: _SampleSpacePreconditionerCacheEntry | None,
    *,
    batch_size: int,
    rank: int,
    random_seed: int,
    prior_variances: np.ndarray,
    diagonal_noise: np.ndarray,
) -> bool:
    if cache_entry is None:
        return False
    if cache_entry.batch_size != int(batch_size) or cache_entry.rank != int(rank):
        return False
    # Don't check random_seed — the seed only determines the initial random
    # projection for the Nyström basis. Once built, the preconditioner is valid
    # regardless of what seed was used. Checking seed defeats the cache across
    # Newton iterations (which pass random_seed + iteration_index).
    if _sample_space_preconditioner_stale(cache_entry):
        return False
    # Relaxed threshold (50%) — the preconditioner only needs to be a reasonable
    # approximation. Within Newton iterations for binary traits, the Polya-Gamma
    # weights change but the preconditioner from the previous iteration is still
    # effective. Rebuilding costs ~5s; reusing saves ~30s per block.
    return (
        _relative_array_change(prior_variances, cache_entry.prior_variances) <= 0.50
        and _relative_array_change(diagonal_noise, cache_entry.diagonal_noise) <= 0.50
    )


def _can_reuse_nystrom_factor(
    cache_entry: _SampleSpacePreconditionerCacheEntry | None,
    *,
    batch_size: int,
    rank: int,
    prior_variances: np.ndarray,
) -> bool:
    """Check if the cached Nyström factor can be reused.

    The Nyström factor depends only on X (genotype_matrix) and prior_variances,
    NOT on diagonal_noise. So when only diagonal_noise changes (e.g. Polya-Gamma
    weight updates during Newton iterations), we can skip the expensive Nyström
    computation (~3s) and only rebuild the cheap diagonal + Woodbury bundle (~0.5s).
    """
    if cache_entry is None:
        return False
    if cache_entry.nystrom_factor_gpu is None and cache_entry.nystrom_factor_cpu is None:
        return False
    if cache_entry.batch_size != int(batch_size) or cache_entry.rank != int(rank):
        return False
    if _sample_space_preconditioner_stale(cache_entry):
        return False
    return _relative_array_change(prior_variances, cache_entry.prior_variances) <= 0.50


def _update_sample_space_preconditioner_iterations(
    cache_entry: _SampleSpacePreconditionerCacheEntry | None,
    iterations: int,
) -> None:
    if cache_entry is None:
        return
    if cache_entry.global_background:
        return
    cache_entry.previous_iterations = cache_entry.last_iterations
    cache_entry.last_iterations = int(iterations)


def _sample_space_background_owner_token(genotype_matrix: StandardizedGenotypeMatrix) -> int:
    return id(genotype_matrix.raw) if genotype_matrix.raw is not None else id(genotype_matrix)


def _sample_space_rhs_matrix_token(genotype_matrix: StandardizedGenotypeMatrix) -> int:
    return id(genotype_matrix)


def _reset_sample_space_background_preconditioner(
    warm_start: _RestrictedPosteriorWarmStart | None,
    genotype_matrix: StandardizedGenotypeMatrix,
) -> None:
    if warm_start is None:
        return
    warm_start.sample_space_background_owner_token = _sample_space_background_owner_token(genotype_matrix)
    warm_start.sample_space_background_variant_count = int(genotype_matrix.shape[1])
    warm_start.sample_space_background_sample_count = int(genotype_matrix.shape[0])
    warm_start.sample_space_background_gpu_preconditioner = None
    warm_start.sample_space_background_cpu_preconditioner = None


def _background_sample_space_preconditioner_entry_for_subset(
    warm_start: _RestrictedPosteriorWarmStart | None,
    genotype_matrix: StandardizedGenotypeMatrix,
    diagonal_noise: np.ndarray,
    *,
    use_gpu: bool,
) -> _SampleSpacePreconditionerCacheEntry | None:
    if warm_start is None:
        return None
    owner_token = _sample_space_background_owner_token(genotype_matrix)
    background_variant_count = warm_start.sample_space_background_variant_count
    background_sample_count = warm_start.sample_space_background_sample_count
    if (
        warm_start.sample_space_background_owner_token != owner_token
        or background_variant_count is None
        or background_sample_count is None
        or int(genotype_matrix.shape[1]) >= int(background_variant_count)
        or int(genotype_matrix.shape[0]) != int(background_sample_count)
    ):
        return None
    background_entry = (
        warm_start.sample_space_background_gpu_preconditioner
        if use_gpu
        else warm_start.sample_space_background_cpu_preconditioner
    )
    if background_entry is None:
        return None
    if np.asarray(diagonal_noise, dtype=np.float64).shape != background_entry.diagonal_noise.shape:
        return None
    return background_entry


def _get_or_build_background_sample_space_cpu_preconditioner(
    genotype_matrix: StandardizedGenotypeMatrix,
    *,
    prior_variances: np.ndarray,
    diagonal_noise: np.ndarray,
    batch_size: int,
    rank: int,
    random_seed: int,
    warm_start: _RestrictedPosteriorWarmStart | None,
) -> _SampleSpacePreconditionerCacheEntry | None:
    if warm_start is None:
        return None
    owner_token = _sample_space_background_owner_token(genotype_matrix)
    if (
        warm_start.sample_space_background_owner_token != owner_token
        or warm_start.sample_space_background_variant_count != int(genotype_matrix.shape[1])
        or warm_start.sample_space_background_sample_count != int(genotype_matrix.shape[0])
    ):
        _reset_sample_space_background_preconditioner(warm_start, genotype_matrix)
    cache_entry = warm_start.sample_space_background_cpu_preconditioner
    if _can_reuse_sample_space_preconditioner(
        cache_entry,
        batch_size=batch_size,
        rank=rank,
        random_seed=random_seed,
        prior_variances=prior_variances,
        diagonal_noise=diagonal_noise,
    ):
        assert cache_entry is not None
        return cache_entry
    diagonal_preconditioner = _sample_space_diagonal_preconditioner(
        genotype_matrix=genotype_matrix,
        prior_variances=prior_variances,
        diagonal_noise=diagonal_noise,
        batch_size=batch_size,
    )
    reuse_nystrom = _can_reuse_nystrom_factor(
        cache_entry,
        batch_size=batch_size,
        rank=rank,
        prior_variances=prior_variances,
    )
    if reuse_nystrom and cache_entry is not None and cache_entry.nystrom_factor_cpu is not None:
        preconditioner = _sample_space_cpu_preconditioner_from_factor(
            low_rank_factor=cache_entry.nystrom_factor_cpu,
            diagonal_preconditioner=diagonal_preconditioner,
        )
        nystrom_basis_cpu = cache_entry.nystrom_basis_cpu
        nystrom_factor_cpu = cache_entry.nystrom_factor_cpu
    else:
        preconditioner, nystrom_factor_cpu, nystrom_basis_cpu = _sample_space_cpu_preconditioner_with_factor(
            genotype_matrix=genotype_matrix,
            prior_variances=prior_variances,
            batch_size=batch_size,
            rank=rank,
            random_seed=random_seed,
            diagonal_preconditioner=diagonal_preconditioner,
        )
    cache_entry = _SampleSpacePreconditionerCacheEntry(
        batch_size=int(batch_size),
        rank=int(rank),
        random_seed=int(random_seed),
        prior_variances=np.asarray(prior_variances, dtype=np.float64).copy(),
        diagonal_noise=np.asarray(diagonal_noise, dtype=np.float64).copy(),
        diagonal_preconditioner=np.asarray(diagonal_preconditioner, dtype=np.float64).copy(),
        preconditioner=preconditioner,
        nystrom_basis_cpu=None if nystrom_basis_cpu is None else np.asarray(nystrom_basis_cpu, dtype=np.float64).copy(),
        nystrom_factor_cpu=None if nystrom_factor_cpu is None else np.asarray(nystrom_factor_cpu, dtype=np.float64).copy(),
        global_background=True,
    )
    warm_start.sample_space_background_cpu_preconditioner = cache_entry
    return cache_entry


def _get_or_build_background_sample_space_gpu_preconditioner(
    genotype_matrix: StandardizedGenotypeMatrix,
    *,
    prior_variances: np.ndarray,
    diagonal_noise: np.ndarray,
    batch_size: int,
    rank: int,
    random_seed: int,
    warm_start: _RestrictedPosteriorWarmStart | None,
) -> _SampleSpacePreconditionerCacheEntry | None:
    if warm_start is None:
        return None
    owner_token = _sample_space_background_owner_token(genotype_matrix)
    if (
        warm_start.sample_space_background_owner_token != owner_token
        or warm_start.sample_space_background_variant_count != int(genotype_matrix.shape[1])
        or warm_start.sample_space_background_sample_count != int(genotype_matrix.shape[0])
    ):
        _reset_sample_space_background_preconditioner(warm_start, genotype_matrix)
    cache_entry = warm_start.sample_space_background_gpu_preconditioner
    if _can_reuse_sample_space_preconditioner(
        cache_entry,
        batch_size=batch_size,
        rank=rank,
        random_seed=random_seed,
        prior_variances=prior_variances,
        diagonal_noise=diagonal_noise,
    ):
        assert cache_entry is not None
        return cache_entry
    reuse_nystrom = _can_reuse_nystrom_factor(
        cache_entry,
        batch_size=batch_size,
        rank=rank,
        prior_variances=prior_variances,
    )
    diagonal_preconditioner = _sample_space_diagonal_preconditioner(
        genotype_matrix=genotype_matrix,
        prior_variances=prior_variances,
        diagonal_noise=diagonal_noise,
        batch_size=batch_size,
    )
    if reuse_nystrom:
        assert cache_entry is not None
        assert cache_entry.nystrom_factor_gpu is not None
        preconditioner = _sample_space_gpu_preconditioner_from_factor(
            genotype_matrix=genotype_matrix,
            nystrom_factor_gpu=cache_entry.nystrom_factor_gpu,
            diagonal_preconditioner=diagonal_preconditioner,
        )
        nystrom_basis_gpu = cache_entry.nystrom_basis_gpu
        nystrom_factor_gpu = cache_entry.nystrom_factor_gpu
    else:
        preconditioner, nystrom_factor_gpu, nystrom_basis_gpu = _sample_space_gpu_preconditioner_with_factor(
            genotype_matrix=genotype_matrix,
            prior_variances=prior_variances,
            diagonal_noise=diagonal_noise,
            batch_size=batch_size,
            rank=rank,
            random_seed=random_seed,
            diagonal_preconditioner=diagonal_preconditioner,
        )
    cache_entry = _SampleSpacePreconditionerCacheEntry(
        batch_size=int(batch_size),
        rank=int(rank),
        random_seed=int(random_seed),
        prior_variances=np.asarray(prior_variances, dtype=np.float64).copy(),
        diagonal_noise=np.asarray(diagonal_noise, dtype=np.float64).copy(),
        diagonal_preconditioner=np.asarray(diagonal_preconditioner, dtype=np.float64).copy(),
        preconditioner=preconditioner,
        nystrom_basis_gpu=nystrom_basis_gpu,
        nystrom_factor_gpu=nystrom_factor_gpu,
        global_background=True,
    )
    warm_start.sample_space_background_gpu_preconditioner = cache_entry
    return cache_entry


def _sample_space_cpu_preconditioner_from_factor(
    *,
    low_rank_factor: np.ndarray,
    diagonal_preconditioner: np.ndarray,
) -> Callable[[jnp.ndarray], jnp.ndarray] | np.ndarray:
    diagonal_preconditioner = np.asarray(diagonal_preconditioner, dtype=np.float64)
    if low_rank_factor.size == 0 or low_rank_factor.shape[1] == 0:
        return diagonal_preconditioner
    weighted_selected_genotypes = np.asarray(low_rank_factor, dtype=np.float64)
    effective_rank = int(weighted_selected_genotypes.shape[1])
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


def _sample_space_cpu_preconditioner_with_factor(
    genotype_matrix: StandardizedGenotypeMatrix,
    prior_variances: np.ndarray,
    batch_size: int,
    rank: int,
    random_seed: int = 0,
    diagonal_preconditioner: np.ndarray | None = None,
) -> tuple[Callable[[jnp.ndarray], jnp.ndarray] | np.ndarray, np.ndarray | None, np.ndarray | None]:
    if diagonal_preconditioner is None:
        raise ValueError("diagonal_preconditioner is required for CPU preconditioner construction.")
    if rank <= 0:
        return np.asarray(diagonal_preconditioner, dtype=np.float64), None, None
    selected_rank = min(int(rank), int(genotype_matrix.shape[0]))
    if selected_rank <= 0:
        return np.asarray(diagonal_preconditioner, dtype=np.float64), None, None
    basis_matrix = _sample_space_nystrom_basis_cpu(
        genotype_matrix=genotype_matrix,
        batch_size=batch_size,
        rank=selected_rank,
        random_seed=random_seed,
    )
    low_rank_factor = _sample_space_nystrom_factor_cpu_from_basis(
        genotype_matrix=genotype_matrix,
        prior_variances=prior_variances,
        basis_matrix=basis_matrix,
        batch_size=batch_size,
    )
    if low_rank_factor is None:
        return np.asarray(diagonal_preconditioner, dtype=np.float64), None, basis_matrix
    effective_rank = int(low_rank_factor.shape[1])
    log(f"      sample-space preconditioner: CPU Nyström-Woodbury rank={effective_rank}")
    return (
        _sample_space_cpu_preconditioner_from_factor(
            low_rank_factor=low_rank_factor,
            diagonal_preconditioner=diagonal_preconditioner,
        ),
        np.asarray(low_rank_factor, dtype=np.float64),
        None if basis_matrix is None else np.asarray(basis_matrix, dtype=np.float64),
    )


def _adapt_background_sample_space_preconditioner_for_subset(
    genotype_matrix: StandardizedGenotypeMatrix,
    background_entry: _SampleSpacePreconditionerCacheEntry,
    *,
    prior_variances: np.ndarray,
    diagonal_noise: np.ndarray,
    batch_size: int,
    rank: int,
    use_gpu: bool,
) -> _SampleSpacePreconditionerCacheEntry:
    diagonal_preconditioner = _sample_space_diagonal_preconditioner(
        genotype_matrix=genotype_matrix,
        prior_variances=prior_variances,
        diagonal_noise=diagonal_noise,
        batch_size=batch_size,
    )
    requested_rank = max(int(rank), 0)
    if use_gpu:
        if background_entry.nystrom_basis_gpu is not None and requested_rank > 0:
            available_rank = int(background_entry.nystrom_basis_gpu.shape[1])
            selected_rank = min(requested_rank, available_rank)
            nystrom_basis_gpu = background_entry.nystrom_basis_gpu[:, :selected_rank]
            nystrom_factor_gpu = _sample_space_nystrom_factor_gpu_from_basis(
                genotype_matrix=genotype_matrix,
                prior_variances=prior_variances,
                basis_gpu=nystrom_basis_gpu,
                batch_size=batch_size,
            )
            preconditioner = (
                np.asarray(diagonal_preconditioner, dtype=np.float64)
                if nystrom_factor_gpu is None
                else _sample_space_gpu_preconditioner_from_factor(
                    genotype_matrix=genotype_matrix,
                    nystrom_factor_gpu=nystrom_factor_gpu,
                    diagonal_preconditioner=diagonal_preconditioner,
                )
            )
        else:
            preconditioner = np.asarray(diagonal_preconditioner, dtype=np.float64)
            nystrom_basis_gpu = None
            nystrom_factor_gpu = None
        nystrom_basis_cpu = None
        nystrom_factor_cpu = None
    else:
        if background_entry.nystrom_basis_cpu is not None and requested_rank > 0:
            available_rank = int(background_entry.nystrom_basis_cpu.shape[1])
            selected_rank = min(requested_rank, available_rank)
            nystrom_basis_cpu = np.asarray(
                background_entry.nystrom_basis_cpu[:, :selected_rank],
                dtype=np.float64,
            )
            nystrom_factor_cpu = _sample_space_nystrom_factor_cpu_from_basis(
                genotype_matrix=genotype_matrix,
                prior_variances=prior_variances,
                basis_matrix=nystrom_basis_cpu,
                batch_size=batch_size,
            )
            preconditioner = (
                np.asarray(diagonal_preconditioner, dtype=np.float64)
                if nystrom_factor_cpu is None
                else _sample_space_cpu_preconditioner_from_factor(
                    low_rank_factor=nystrom_factor_cpu,
                    diagonal_preconditioner=diagonal_preconditioner,
                )
            )
        else:
            preconditioner = np.asarray(diagonal_preconditioner, dtype=np.float64)
            nystrom_basis_cpu = None
            nystrom_factor_cpu = None
        nystrom_basis_gpu = None
        nystrom_factor_gpu = None
    return _SampleSpacePreconditionerCacheEntry(
        batch_size=int(batch_size),
        rank=int(requested_rank),
        random_seed=int(background_entry.random_seed),
        prior_variances=np.asarray(prior_variances, dtype=np.float64).copy(),
        diagonal_noise=np.asarray(diagonal_noise, dtype=np.float64).copy(),
        diagonal_preconditioner=np.asarray(diagonal_preconditioner, dtype=np.float64).copy(),
        preconditioner=preconditioner,
        nystrom_basis_cpu=nystrom_basis_cpu,
        nystrom_basis_gpu=nystrom_basis_gpu,
        nystrom_factor_cpu=nystrom_factor_cpu,
        nystrom_factor_gpu=nystrom_factor_gpu,
        global_background=True,
    )


def _cached_sample_probe_projection(
    genotype_matrix: StandardizedGenotypeMatrix,
    sample_probes: np.ndarray,
    *,
    batch_size: int,
    probe_count: int,
    random_seed: int,
    return_gpu: bool = False,
    cupy=None,
) -> np.ndarray:
    cache_key = (int(probe_count), int(random_seed))
    cached_projection = genotype_matrix._sample_space_probe_projection_cache.get(cache_key)
    cached_projection_gpu = genotype_matrix._sample_space_probe_projection_gpu_cache.get(cache_key)
    if return_gpu:
        if cupy is None:
            raise ValueError("cupy is required when return_gpu=True.")
        if cached_projection_gpu is not None:
            return cupy.asarray(cached_projection_gpu, dtype=cupy.float64)
        if cached_projection is not None:
            projection_gpu = cupy.asarray(cached_projection, dtype=cupy.float64)
            genotype_matrix._sample_space_probe_projection_gpu_cache[cache_key] = projection_gpu
            return projection_gpu
    elif cached_projection is not None:
        return np.asarray(cached_projection, dtype=np.float64)
    probe_projection_matrix = np.asarray(
        genotype_matrix.transpose_matmat(sample_probes, batch_size=batch_size),
        dtype=np.float64,
    )
    genotype_matrix._sample_space_probe_projection_cache[cache_key] = probe_projection_matrix
    if return_gpu:
        assert cupy is not None
        projection_gpu = cupy.asarray(probe_projection_matrix, dtype=cupy.float64)
        genotype_matrix._sample_space_probe_projection_gpu_cache[cache_key] = projection_gpu
        return projection_gpu
    return probe_projection_matrix


def _sample_space_preconditioner(
    genotype_matrix: StandardizedGenotypeMatrix,
    prior_variances: np.ndarray,
    diagonal_noise: np.ndarray,
    batch_size: int,
    rank: int,
    random_seed: int = 0,
    diagonal_preconditioner: np.ndarray | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray] | np.ndarray | jnp.ndarray:
    if diagonal_preconditioner is None:
        diagonal_preconditioner = _sample_space_diagonal_preconditioner(
            genotype_matrix=genotype_matrix,
            prior_variances=prior_variances,
            diagonal_noise=diagonal_noise,
            batch_size=batch_size,
        )
    diagonal_preconditioner = np.asarray(diagonal_preconditioner, dtype=np.float64)
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
        cp_solve_triangular = _resolve_gpu_solve_triangular()

        gpu_cache_source = "full" if genotype_matrix._cupy_cache is not None else "streaming"
        effective_rank = int(low_rank_factor_gpu.shape[1])
        log(f"      sample-space preconditioner: GPU Nyström-Woodbury rank={effective_rank} source={gpu_cache_source}")
        compute_cp_dtype = _cupy_compute_dtype(cp)
        diagonal_preconditioner_gpu = cp.asarray(diagonal_preconditioner, dtype=compute_cp_dtype)
        compute_bundle = _build_sample_space_low_rank_bundle_gpu(
            low_rank_factor_gpu,
            diagonal_preconditioner_gpu,
            cp=cp,
            bundle_dtype=compute_cp_dtype,
        )
        float64_bundle = None

        def apply_preconditioner_gpu(right_hand_side: jnp.ndarray) -> jnp.ndarray:
            nonlocal float64_bundle
            right_hand_side_array = np.asarray(right_hand_side)
            use_float64_bundle = compute_cp_dtype != cp.float64 and right_hand_side_array.dtype == np.float64
            if use_float64_bundle:
                if float64_bundle is None:
                    float64_bundle = _build_sample_space_low_rank_bundle_gpu(
                        low_rank_factor_gpu,
                        cp.asarray(diagonal_preconditioner, dtype=cp.float64),
                        cp=cp,
                        bundle_dtype=cp.float64,
                    )
                bundle = float64_bundle
                right_hand_side_gpu = _to_cupy_float64(right_hand_side)
            else:
                bundle = compute_bundle
                right_hand_side_gpu = _to_cupy_compute(right_hand_side)
            return _cupy_to_jax(
                _apply_sample_space_low_rank_preconditioner_gpu(
                    right_hand_side_gpu,
                    bundle,
                    cp=cp,
                    solve_triangular_gpu=cp_solve_triangular,
                )
            )

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
    return _sample_space_cpu_preconditioner_from_factor(
        low_rank_factor=low_rank_factor,
        diagonal_preconditioner=diagonal_preconditioner,
    )


def _sample_space_gpu_preconditioner(
    genotype_matrix: StandardizedGenotypeMatrix,
    prior_variances: np.ndarray,
    diagonal_noise: np.ndarray,
    batch_size: int,
    rank: int,
    random_seed: int = 0,
    diagonal_preconditioner: np.ndarray | None = None,
):
    import cupy as cp
    cp_solve_triangular = _resolve_gpu_solve_triangular()

    compute_cp_dtype = _cupy_compute_dtype(cp)
    if diagonal_preconditioner is None:
        diagonal_preconditioner = _sample_space_diagonal_preconditioner(
            genotype_matrix=genotype_matrix,
            prior_variances=prior_variances,
            diagonal_noise=diagonal_noise,
            batch_size=batch_size,
        )
    diagonal_preconditioner = np.asarray(diagonal_preconditioner, dtype=np.float64)
    diagonal_preconditioner_gpu = cp.asarray(diagonal_preconditioner, dtype=compute_cp_dtype)

    def apply_diagonal(right_hand_side_gpu):
        rhs_dtype = getattr(right_hand_side_gpu, "dtype", compute_cp_dtype)
        resolved_dtype = cp.float64 if compute_cp_dtype != cp.float64 and rhs_dtype == cp.float64 else compute_cp_dtype
        right_hand_side_gpu = cp.asarray(right_hand_side_gpu, dtype=resolved_dtype)
        diagonal_vector = diagonal_preconditioner_gpu.astype(resolved_dtype, copy=False)
        if right_hand_side_gpu.ndim == 2:
            return right_hand_side_gpu / diagonal_vector[:, None]
        return right_hand_side_gpu / diagonal_vector

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
    compute_bundle = _build_sample_space_low_rank_bundle_gpu(
        low_rank_factor_gpu,
        diagonal_preconditioner_gpu,
        cp=cp,
        bundle_dtype=compute_cp_dtype,
    )
    float64_bundle = None

    def apply_low_rank(right_hand_side_gpu):
        nonlocal float64_bundle
        rhs_dtype = getattr(right_hand_side_gpu, "dtype", compute_cp_dtype)
        if compute_cp_dtype != cp.float64 and rhs_dtype == cp.float64:
            if float64_bundle is None:
                float64_bundle = _build_sample_space_low_rank_bundle_gpu(
                    low_rank_factor_gpu,
                    cp.asarray(diagonal_preconditioner, dtype=cp.float64),
                    cp=cp,
                    bundle_dtype=cp.float64,
                )
            bundle = float64_bundle
        else:
            bundle = compute_bundle
        return _apply_sample_space_low_rank_preconditioner_gpu(
            right_hand_side_gpu,
            bundle,
            cp=cp,
            solve_triangular_gpu=cp_solve_triangular,
        )

    return apply_low_rank


def _get_cached_sample_space_cpu_preconditioner(
    genotype_matrix: StandardizedGenotypeMatrix,
    *,
    prior_variances: np.ndarray,
    diagonal_noise: np.ndarray,
    batch_size: int,
    rank: int,
    random_seed: int,
    warm_start: _RestrictedPosteriorWarmStart | None = None,
):
    cache_entry = cast(_SampleSpacePreconditionerCacheEntry | None, genotype_matrix._sample_space_cpu_preconditioner_cache)
    if _can_reuse_sample_space_preconditioner(
        cache_entry,
        batch_size=batch_size,
        rank=rank,
        random_seed=random_seed,
        prior_variances=prior_variances,
        diagonal_noise=diagonal_noise,
    ):
        assert cache_entry is not None
        return cache_entry.preconditioner, cache_entry
    background_entry = _background_sample_space_preconditioner_entry_for_subset(
        warm_start,
        genotype_matrix,
        diagonal_noise,
        use_gpu=False,
    )
    if background_entry is not None:
        adapted_entry = _adapt_background_sample_space_preconditioner_for_subset(
            genotype_matrix,
            background_entry,
            prior_variances=prior_variances,
            diagonal_noise=diagonal_noise,
            batch_size=batch_size,
            rank=rank,
            use_gpu=False,
        )
        return adapted_entry.preconditioner, adapted_entry
    diagonal_preconditioner = _sample_space_diagonal_preconditioner(
        genotype_matrix=genotype_matrix,
        prior_variances=prior_variances,
        diagonal_noise=diagonal_noise,
        batch_size=batch_size,
    )
    reuse_nystrom = _can_reuse_nystrom_factor(
        cache_entry,
        batch_size=batch_size,
        rank=rank,
        prior_variances=prior_variances,
    )
    if reuse_nystrom and cache_entry is not None and cache_entry.nystrom_factor_cpu is not None:
        preconditioner = _sample_space_cpu_preconditioner_from_factor(
            low_rank_factor=cache_entry.nystrom_factor_cpu,
            diagonal_preconditioner=diagonal_preconditioner,
        )
        nystrom_basis_cpu = cache_entry.nystrom_basis_cpu
        nystrom_factor_cpu = cache_entry.nystrom_factor_cpu
    else:
        preconditioner, nystrom_factor_cpu, nystrom_basis_cpu = _sample_space_cpu_preconditioner_with_factor(
            genotype_matrix=genotype_matrix,
            prior_variances=prior_variances,
            batch_size=batch_size,
            rank=rank,
            random_seed=random_seed,
            diagonal_preconditioner=diagonal_preconditioner,
        )
    cache_entry = _SampleSpacePreconditionerCacheEntry(
        batch_size=int(batch_size),
        rank=int(rank),
        random_seed=int(random_seed),
        prior_variances=np.asarray(prior_variances, dtype=np.float64).copy(),
        diagonal_noise=np.asarray(diagonal_noise, dtype=np.float64).copy(),
        diagonal_preconditioner=np.asarray(diagonal_preconditioner, dtype=np.float64).copy(),
        preconditioner=preconditioner,
        nystrom_basis_cpu=None if nystrom_basis_cpu is None else np.asarray(nystrom_basis_cpu, dtype=np.float64).copy(),
        nystrom_factor_cpu=None if nystrom_factor_cpu is None else np.asarray(nystrom_factor_cpu, dtype=np.float64).copy(),
    )
    genotype_matrix._sample_space_cpu_preconditioner_cache = cache_entry
    return preconditioner, cache_entry


def _sample_space_gpu_preconditioner_from_factor(
    genotype_matrix: StandardizedGenotypeMatrix,
    nystrom_factor_gpu,
    diagonal_preconditioner: np.ndarray,
):
    """Build a GPU preconditioner from a pre-computed Nyström factor.

    This is the fast path: skip the expensive Nyström factor computation (~3s)
    and only rebuild the cheap diagonal + Woodbury bundle (~0.5s).
    """
    import cupy as cp
    cp_solve_triangular = _resolve_gpu_solve_triangular()

    compute_cp_dtype = _cupy_compute_dtype(cp)
    diagonal_preconditioner = np.asarray(diagonal_preconditioner, dtype=np.float64)
    diagonal_preconditioner_gpu = cp.asarray(diagonal_preconditioner, dtype=compute_cp_dtype)

    gpu_cache_source = "full" if genotype_matrix._cupy_cache is not None else "streaming"
    effective_rank = int(nystrom_factor_gpu.shape[1])
    log(f"      sample-space preconditioner: GPU Nyström-Woodbury rank={effective_rank} source={gpu_cache_source} (factor reused)")
    compute_bundle = _build_sample_space_low_rank_bundle_gpu(
        nystrom_factor_gpu,
        diagonal_preconditioner_gpu,
        cp=cp,
        bundle_dtype=compute_cp_dtype,
    )
    float64_bundle = None

    def apply_low_rank(right_hand_side_gpu):
        nonlocal float64_bundle
        rhs_dtype = getattr(right_hand_side_gpu, "dtype", compute_cp_dtype)
        if compute_cp_dtype != cp.float64 and rhs_dtype == cp.float64:
            if float64_bundle is None:
                float64_bundle = _build_sample_space_low_rank_bundle_gpu(
                    nystrom_factor_gpu,
                    cp.asarray(diagonal_preconditioner, dtype=cp.float64),
                    cp=cp,
                    bundle_dtype=cp.float64,
                )
            bundle = float64_bundle
        else:
            bundle = compute_bundle
        return _apply_sample_space_low_rank_preconditioner_gpu(
            right_hand_side_gpu,
            bundle,
            cp=cp,
            solve_triangular_gpu=cp_solve_triangular,
        )

    return apply_low_rank


def _sample_space_gpu_preconditioner_with_factor(
    genotype_matrix: StandardizedGenotypeMatrix,
    prior_variances: np.ndarray,
    diagonal_noise: np.ndarray,
    batch_size: int,
    rank: int,
    random_seed: int = 0,
    diagonal_preconditioner: np.ndarray | None = None,
):
    """Build GPU preconditioner and return both the preconditioner and the Nyström factor.

    Returns (preconditioner_callable, nystrom_factor_gpu_or_None, nystrom_basis_gpu_or_None).
    """
    import cupy as cp
    cp_solve_triangular = _resolve_gpu_solve_triangular()

    compute_cp_dtype = _cupy_compute_dtype(cp)
    if diagonal_preconditioner is None:
        diagonal_preconditioner = _sample_space_diagonal_preconditioner(
            genotype_matrix=genotype_matrix,
            prior_variances=prior_variances,
            diagonal_noise=diagonal_noise,
            batch_size=batch_size,
        )
    diagonal_preconditioner = np.asarray(diagonal_preconditioner, dtype=np.float64)
    diagonal_preconditioner_gpu = cp.asarray(diagonal_preconditioner, dtype=compute_cp_dtype)

    def apply_diagonal(right_hand_side_gpu):
        rhs_dtype = getattr(right_hand_side_gpu, "dtype", compute_cp_dtype)
        resolved_dtype = cp.float64 if compute_cp_dtype != cp.float64 and rhs_dtype == cp.float64 else compute_cp_dtype
        right_hand_side_gpu = cp.asarray(right_hand_side_gpu, dtype=resolved_dtype)
        diagonal_vector = diagonal_preconditioner_gpu.astype(resolved_dtype, copy=False)
        if right_hand_side_gpu.ndim == 2:
            return right_hand_side_gpu / diagonal_vector[:, None]
        return right_hand_side_gpu / diagonal_vector

    if rank <= 0:
        return apply_diagonal, None, None
    selected_rank = min(int(rank), int(genotype_matrix.shape[0]))
    if selected_rank <= 0:
        return apply_diagonal, None, None
    basis_gpu = _sample_space_nystrom_basis_gpu(
        genotype_matrix=genotype_matrix,
        batch_size=batch_size,
        rank=selected_rank,
        random_seed=random_seed,
    )
    low_rank_factor_gpu = _sample_space_nystrom_factor_gpu_from_basis(
        genotype_matrix=genotype_matrix,
        prior_variances=prior_variances,
        basis_gpu=basis_gpu,
        batch_size=batch_size,
    )
    if low_rank_factor_gpu is None:
        return apply_diagonal, None, basis_gpu

    gpu_cache_source = "full" if genotype_matrix._cupy_cache is not None else "streaming"
    effective_rank = int(low_rank_factor_gpu.shape[1])
    log(f"      sample-space preconditioner: GPU Nyström-Woodbury rank={effective_rank} source={gpu_cache_source}")
    compute_bundle = _build_sample_space_low_rank_bundle_gpu(
        low_rank_factor_gpu,
        diagonal_preconditioner_gpu,
        cp=cp,
        bundle_dtype=compute_cp_dtype,
    )
    float64_bundle = None

    def apply_low_rank(right_hand_side_gpu):
        nonlocal float64_bundle
        rhs_dtype = getattr(right_hand_side_gpu, "dtype", compute_cp_dtype)
        if compute_cp_dtype != cp.float64 and rhs_dtype == cp.float64:
            if float64_bundle is None:
                float64_bundle = _build_sample_space_low_rank_bundle_gpu(
                    low_rank_factor_gpu,
                    cp.asarray(diagonal_preconditioner, dtype=cp.float64),
                    cp=cp,
                    bundle_dtype=cp.float64,
                )
            bundle = float64_bundle
        else:
            bundle = compute_bundle
        return _apply_sample_space_low_rank_preconditioner_gpu(
            right_hand_side_gpu,
            bundle,
            cp=cp,
            solve_triangular_gpu=cp_solve_triangular,
        )

    return apply_low_rank, low_rank_factor_gpu, basis_gpu


def _get_cached_sample_space_gpu_preconditioner(
    genotype_matrix: StandardizedGenotypeMatrix,
    *,
    prior_variances: np.ndarray,
    diagonal_noise: np.ndarray,
    batch_size: int,
    rank: int,
    random_seed: int,
    warm_start: _RestrictedPosteriorWarmStart | None = None,
):
    cache_entry = cast(_SampleSpacePreconditionerCacheEntry | None, genotype_matrix._sample_space_gpu_preconditioner_cache)
    if _can_reuse_sample_space_preconditioner(
        cache_entry,
        batch_size=batch_size,
        rank=rank,
        random_seed=random_seed,
        prior_variances=prior_variances,
        diagonal_noise=diagonal_noise,
    ):
        assert cache_entry is not None
        return cache_entry.preconditioner, cache_entry
    background_entry = _background_sample_space_preconditioner_entry_for_subset(
        warm_start,
        genotype_matrix,
        diagonal_noise,
        use_gpu=True,
    )
    if background_entry is not None:
        adapted_entry = _adapt_background_sample_space_preconditioner_for_subset(
            genotype_matrix,
            background_entry,
            prior_variances=prior_variances,
            diagonal_noise=diagonal_noise,
            batch_size=batch_size,
            rank=rank,
            use_gpu=True,
        )
        return adapted_entry.preconditioner, adapted_entry

    # Check if we can reuse the expensive Nyström factor (depends on X and
    # prior_variances only) and just rebuild the cheap diagonal + Woodbury bundle
    # (depends on diagonal_noise). This is the common case during Newton iterations
    # where only Polya-Gamma weights change.
    reuse_nystrom = _can_reuse_nystrom_factor(
        cache_entry,
        batch_size=batch_size,
        rank=rank,
        prior_variances=prior_variances,
    )

    diagonal_preconditioner = _sample_space_diagonal_preconditioner(
        genotype_matrix=genotype_matrix,
        prior_variances=prior_variances,
        diagonal_noise=diagonal_noise,
        batch_size=batch_size,
    )

    if reuse_nystrom:
        # Fast path: reuse cached Nyström factor, only rebuild diagonal + Woodbury
        assert cache_entry is not None
        assert cache_entry.nystrom_factor_gpu is not None
        preconditioner = _sample_space_gpu_preconditioner_from_factor(
            genotype_matrix=genotype_matrix,
            nystrom_factor_gpu=cache_entry.nystrom_factor_gpu,
            diagonal_preconditioner=diagonal_preconditioner,
        )
        nystrom_basis_gpu = cache_entry.nystrom_basis_gpu
        nystrom_factor_gpu = cache_entry.nystrom_factor_gpu
    else:
        # Slow path: rebuild everything including the expensive Nyström factor
        preconditioner, nystrom_factor_gpu, nystrom_basis_gpu = _sample_space_gpu_preconditioner_with_factor(
            genotype_matrix=genotype_matrix,
            prior_variances=prior_variances,
            diagonal_noise=diagonal_noise,
            batch_size=batch_size,
            rank=rank,
            random_seed=random_seed,
            diagonal_preconditioner=diagonal_preconditioner,
        )

    cache_entry = _SampleSpacePreconditionerCacheEntry(
        batch_size=int(batch_size),
        rank=int(rank),
        random_seed=int(random_seed),
        prior_variances=np.asarray(prior_variances, dtype=np.float64).copy(),
        diagonal_noise=np.asarray(diagonal_noise, dtype=np.float64).copy(),
        diagonal_preconditioner=np.asarray(diagonal_preconditioner, dtype=np.float64).copy(),
        preconditioner=preconditioner,
        nystrom_basis_gpu=nystrom_basis_gpu,
        nystrom_factor_gpu=nystrom_factor_gpu,
    )
    genotype_matrix._sample_space_gpu_preconditioner_cache = cache_entry
    return preconditioner, cache_entry


def _apply_sample_space_operator_gpu(
    genotype_matrix: StandardizedGenotypeMatrix,
    prior_variances,
    diagonal_noise,
    matrix_gpu,
    *,
    batch_size: int,
    cp,
    dtype,
    _already_on_gpu: bool = False,
):
    input_gpu = cp.asarray(matrix_gpu, dtype=dtype)
    diagonal_noise_gpu = diagonal_noise if _already_on_gpu else cp.asarray(diagonal_noise, dtype=dtype)
    prior_variances_gpu = prior_variances if _already_on_gpu else cp.asarray(prior_variances, dtype=dtype)
    if input_gpu.ndim == 1:
        input_gpu = input_gpu[:, None]
        vector_input = True
    elif input_gpu.ndim == 2:
        vector_input = False
    else:
        raise ValueError("sample-space GPU operator expects a vector or matrix right-hand side.")
    result_gpu = diagonal_noise_gpu[:, None] * input_gpu
    if genotype_matrix._cupy_cache is not None and not _cupy_cache_is_int8_standardized(genotype_matrix._cupy_cache):
        projected_gpu = genotype_matrix.gpu_transpose_matmat(
            input_gpu,
            batch_size=batch_size,
            cupy=cp,
            dtype=dtype,
        )
        scaled_projection_gpu = prior_variances_gpu[:, None] * projected_gpu
        result_gpu += genotype_matrix.gpu_matmat(
            scaled_projection_gpu,
            batch_size=batch_size,
            cupy=cp,
            dtype=dtype,
        )
    elif genotype_matrix._cupy_cache is not None:
        for batch_slice, x_chunk in _iter_cupy_cache_standardized_batches(
            genotype_matrix._cupy_cache,
            sample_count=genotype_matrix.shape[0],
            batch_size=batch_size,
            cupy=cp,
            dtype=dtype,
        ):
            scaled_projection = prior_variances_gpu[batch_slice, None] * (x_chunk.T @ input_gpu)
            result_gpu += x_chunk @ scaled_projection
    else:
        raw_matrix = cast(RawGenotypeMatrix, genotype_matrix.raw)
        for batch_slice, standardized_batch in _iter_standardized_gpu_batches(
            raw_matrix,
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
    initial_guess_gpu,
    tolerance: float,
    max_iterations: int,
    preconditioner,
    batch_size: int,
    cp,
    compute_cp_dtype,
    column_iteration_limits: np.ndarray | None = None,
    required_columns: np.ndarray | None = None,
    lanczos_recorder: _SampleSpaceCGLanczosRecorder | None = None,
) -> tuple[Any, int]:
    rhs_gpu = cp.asarray(right_hand_side_gpu, dtype=compute_cp_dtype)
    vector_input = rhs_gpu.ndim == 1
    if vector_input:
        rhs_gpu = rhs_gpu[:, None]
    elif rhs_gpu.ndim != 2:
        raise ValueError("GPU sample-space solve expects a vector or matrix right-hand side.")
    n_rhs = int(rhs_gpu.shape[1])
    if column_iteration_limits is None:
        resolved_iteration_limits = np.full(n_rhs, int(max_iterations), dtype=np.int32)
    else:
        resolved_iteration_limits = np.asarray(column_iteration_limits, dtype=np.int32).reshape(-1)
        if resolved_iteration_limits.shape != (n_rhs,):
            raise ValueError("column_iteration_limits must match the number of rhs columns.")
    if required_columns is None:
        required_mask = np.ones(n_rhs, dtype=bool)
    else:
        required_mask = np.asarray(required_columns, dtype=bool).reshape(-1)
        if required_mask.shape != (n_rhs,):
            raise ValueError("required_columns must match the number of rhs columns.")

    # Pre-upload constants to GPU — avoids re-uploading every CG iteration
    _diagonal_noise_gpu = cp.asarray(diagonal_noise, dtype=compute_cp_dtype)
    _prior_variances_gpu = cp.asarray(prior_variances, dtype=compute_cp_dtype)

    def _gpu_to_f64(arr):
        """Safe GPU→CPU transfer that works with both real CuPy and mocked numpy."""
        return arr.get().astype(np.float64) if hasattr(arr, "get") else np.asarray(arr, dtype=np.float64)

    def apply_operator(matrix_gpu):
        return _apply_sample_space_operator_gpu(
            genotype_matrix=genotype_matrix,
            prior_variances=_prior_variances_gpu,
            diagonal_noise=_diagonal_noise_gpu,
            matrix_gpu=matrix_gpu,
            batch_size=batch_size,
            cp=cp,
            dtype=compute_cp_dtype,
            _already_on_gpu=True,
        )

    residual_refresh_interval = 32
    tol_sq = float(tolerance) * float(tolerance)
    if initial_guess_gpu is not None:
        solution_gpu = cp.asarray(initial_guess_gpu, dtype=compute_cp_dtype)
        if solution_gpu.ndim == 1:
            solution_gpu = solution_gpu[:, None]
        if solution_gpu.shape != rhs_gpu.shape:
            raise ValueError("GPU sample-space initial_guess must match right_hand_side shape.")
    else:
        solution_gpu = cp.asarray(preconditioner(rhs_gpu), dtype=compute_cp_dtype)
    residual_gpu = rhs_gpu - apply_operator(solution_gpu)
    residual_norm_sq = _gpu_to_f64(cp.sum(residual_gpu * residual_gpu, axis=0, dtype=cp.float64))
    rhs_norm_sq = _gpu_to_f64(cp.sum(rhs_gpu * rhs_gpu, axis=0, dtype=cp.float64))
    convergence_threshold_sq = np.maximum(tol_sq, tol_sq * np.maximum(residual_norm_sq, rhs_norm_sq))
    converged = residual_norm_sq <= convergence_threshold_sq
    done = converged | (resolved_iteration_limits <= 0)
    if np.all(done):
        solution = cp.asarray(solution_gpu, dtype=cp.float64)
        return (solution[:, 0] if vector_input else solution), 0

    preconditioned_residual_gpu = preconditioner(residual_gpu)
    search_direction_gpu = preconditioned_residual_gpu
    residual_dot = _gpu_to_f64(cp.sum(residual_gpu * preconditioned_residual_gpu, axis=0, dtype=cp.float64))
    iterations_used = 0
    _cg_t0 = time.monotonic()
    _cg_log_interval = max(max_iterations // 10, 1)
    for iteration_index in range(max_iterations):
        iterations_used = iteration_index + 1
        active_columns = np.flatnonzero(~done).astype(np.int32, copy=False)
        if active_columns.size == 0:
            break
        if iteration_index % _cg_log_interval == 0 or iteration_index == max_iterations - 1:
            pct_converged = int(100 * (n_rhs - active_columns.size) / max(n_rhs, 1))
            max_residual = float(np.max(residual_norm_sq[active_columns])) if active_columns.size > 0 else 0.0
            log(f"       CG iter {iteration_index+1}/{max_iterations}: {pct_converged}% converged  residual={max_residual:.2e}  ({time.monotonic()-_cg_t0:.1f}s)")
        masked_search_gpu = search_direction_gpu[:, active_columns]
        operator_search_gpu = apply_operator(masked_search_gpu)
        step_denom = _gpu_to_f64(cp.sum(masked_search_gpu * operator_search_gpu, axis=0, dtype=cp.float64))
        if np.any(~np.isfinite(step_denom) | (step_denom <= 0.0)):
            raise RuntimeError("GPU conjugate-gradient operator is not positive definite.")
        step_scale = residual_dot[active_columns] / step_denom
        _record_sample_space_cg_lanczos_alpha(
            lanczos_recorder,
            iteration_index=iteration_index,
            active_columns=active_columns,
            step_scale=step_scale,
        )
        step_scale_gpu = cp.asarray(step_scale, dtype=compute_cp_dtype)
        solution_gpu[:, active_columns] += masked_search_gpu * step_scale_gpu[None, :]
        residual_gpu[:, active_columns] -= operator_search_gpu * step_scale_gpu[None, :]
        if (iteration_index + 1) % residual_refresh_interval == 0:
            residual_gpu[:, active_columns] = rhs_gpu[:, active_columns] - apply_operator(solution_gpu[:, active_columns])
        residual_norm_sq[active_columns] = cp.sum(
            residual_gpu[:, active_columns] * residual_gpu[:, active_columns], axis=0, dtype=cp.float64,
        ).get().astype(np.float64)
        converged = residual_norm_sq <= convergence_threshold_sq
        limit_reached_active = (iteration_index + 1) >= resolved_iteration_limits[active_columns]
        if np.any(limit_reached_active):
            _finalize_sample_space_cg_lanczos_steps(
                lanczos_recorder,
                completed_columns=active_columns[limit_reached_active],
                iteration_count=iteration_index + 1,
            )
        done = converged | ((iteration_index + 1) >= resolved_iteration_limits)
        if np.all(done):
            break
        refreshed_residual_gpu = residual_gpu[:, active_columns]
        refreshed_preconditioned_gpu = preconditioner(refreshed_residual_gpu)
        updated_residual_dot_active = cp.sum(
            refreshed_residual_gpu * refreshed_preconditioned_gpu, axis=0, dtype=cp.float64,
        ).get().astype(np.float64)
        beta_active = updated_residual_dot_active / np.maximum(residual_dot[active_columns], 1e-30)
        _record_sample_space_cg_lanczos_beta(
            lanczos_recorder,
            iteration_index=iteration_index,
            active_columns=active_columns,
            beta_value=beta_active,
        )
        if np.any(~np.isfinite(beta_active) | (beta_active < 0.0)):
            raise RuntimeError("GPU conjugate-gradient preconditioner produced an invalid update.")
        beta_active_gpu = cp.asarray(beta_active, dtype=compute_cp_dtype)
        search_direction_gpu[:, active_columns] = refreshed_preconditioned_gpu + (
            search_direction_gpu[:, active_columns] * beta_active_gpu[None, :]
        )
        residual_dot[active_columns] = updated_residual_dot_active
    _finalize_sample_space_cg_lanczos_steps(
        lanczos_recorder,
        completed_columns=lanczos_recorder.monitored_columns[lanczos_recorder.step_lengths == 0]
        if lanczos_recorder is not None
        else np.empty(0, dtype=np.int32),
        iteration_count=min(iterations_used, max_iterations),
    )
    log(f"       GPU CG done: {iterations_used} iterations in {time.monotonic()-_cg_t0:.1f}s  mem={mem()}")
    required_residual = residual_norm_sq[required_mask]
    required_threshold = convergence_threshold_sq[required_mask]
    final_required_residual = float(np.max(required_residual)) if required_residual.size > 0 else 0.0
    final_required_threshold = float(np.max(required_threshold)) if required_threshold.size > 0 else 0.0
    if final_required_residual > final_required_threshold:
        raise RuntimeError(
            "GPU conjugate-gradient solve failed to converge: "
            + f"residual={final_required_residual:.2e} threshold={final_required_threshold:.2e} "
            + f"iterations={max_iterations}"
        )
    solution = cp.asarray(solution_gpu, dtype=cp.float64)
    return (solution[:, 0] if vector_input else solution), iterations_used


def _resolve_sample_space_solve_result(result, *, fallback_iterations: int) -> tuple[Any, int]:
    resolved_result = result
    total_iterations = 0
    while isinstance(resolved_result, tuple) and len(resolved_result) == 2 and isinstance(resolved_result[1], (int, np.integer)):
        total_iterations += int(resolved_result[1])
        resolved_result = resolved_result[0]
    if total_iterations == 0:
        total_iterations = int(fallback_iterations)
    return resolved_result, total_iterations


def _build_sample_space_cg_lanczos_recorder(
    *,
    total_rhs_count: int,
    monitored_columns: np.ndarray,
    maximum_steps: int,
) -> _SampleSpaceCGLanczosRecorder | None:
    monitored = np.asarray(monitored_columns, dtype=np.int32).reshape(-1)
    if monitored.size == 0 or maximum_steps < 1:
        return None
    if np.any(monitored < 0) or np.any(monitored >= int(total_rhs_count)):
        raise ValueError("monitored_columns must lie within the sample-space rhs block.")
    column_to_slot = np.full(int(total_rhs_count), -1, dtype=np.int32)
    column_to_slot[monitored] = np.arange(monitored.shape[0], dtype=np.int32)
    return _SampleSpaceCGLanczosRecorder(
        monitored_columns=monitored,
        maximum_steps=int(maximum_steps),
        alpha_history=np.full((int(maximum_steps), monitored.shape[0]), np.nan, dtype=np.float64),
        beta_history=np.full((max(int(maximum_steps) - 1, 0), monitored.shape[0]), np.nan, dtype=np.float64),
        step_lengths=np.zeros(monitored.shape[0], dtype=np.int32),
        column_to_slot=column_to_slot,
    )


def _record_sample_space_cg_lanczos_alpha(
    recorder: _SampleSpaceCGLanczosRecorder | None,
    *,
    iteration_index: int,
    active_columns: np.ndarray,
    step_scale: np.ndarray,
) -> None:
    if recorder is None or iteration_index >= recorder.maximum_steps:
        return
    active = np.asarray(active_columns, dtype=np.int32).reshape(-1)
    if active.size == 0:
        return
    slots = recorder.column_to_slot[active]
    tracked_mask = slots >= 0
    if not np.any(tracked_mask):
        return
    recorder.alpha_history[int(iteration_index), slots[tracked_mask]] = np.asarray(step_scale, dtype=np.float64)[tracked_mask]


def _record_sample_space_cg_lanczos_beta(
    recorder: _SampleSpaceCGLanczosRecorder | None,
    *,
    iteration_index: int,
    active_columns: np.ndarray,
    beta_value: np.ndarray,
) -> None:
    if recorder is None or iteration_index >= recorder.maximum_steps - 1:
        return
    active = np.asarray(active_columns, dtype=np.int32).reshape(-1)
    if active.size == 0:
        return
    slots = recorder.column_to_slot[active]
    tracked_mask = slots >= 0
    if not np.any(tracked_mask):
        return
    recorder.beta_history[int(iteration_index), slots[tracked_mask]] = np.asarray(beta_value, dtype=np.float64)[tracked_mask]


def _finalize_sample_space_cg_lanczos_steps(
    recorder: _SampleSpaceCGLanczosRecorder | None,
    *,
    completed_columns: np.ndarray,
    iteration_count: int,
) -> None:
    if recorder is None:
        return
    completed = np.asarray(completed_columns, dtype=np.int32).reshape(-1)
    if completed.size == 0:
        return
    slots = recorder.column_to_slot[completed]
    tracked_mask = slots >= 0
    if not np.any(tracked_mask):
        return
    recorder.step_lengths[slots[tracked_mask]] = np.maximum(
        recorder.step_lengths[slots[tracked_mask]],
        min(int(iteration_count), recorder.maximum_steps),
    )


def _sample_space_logdet_from_cg_lanczos(
    recorder: _SampleSpaceCGLanczosRecorder,
    *,
    dimension: int,
    baseline_logdet: float,
) -> float:
    estimates: list[float] = []
    for slot_index in range(recorder.monitored_columns.shape[0]):
        step_length = int(recorder.step_lengths[slot_index])
        if step_length < 1:
            continue
        alpha = np.asarray(recorder.alpha_history[:step_length, slot_index], dtype=np.float64)
        if np.any(~np.isfinite(alpha)) or np.any(alpha <= 0.0):
            continue
        diagonal = np.empty(step_length, dtype=np.float64)
        diagonal[0] = 1.0 / alpha[0]
        if step_length > 1:
            beta = np.asarray(recorder.beta_history[: step_length - 1, slot_index], dtype=np.float64)
            if np.any(~np.isfinite(beta)) or np.any(beta < 0.0):
                continue
            diagonal[1:] = 1.0 / alpha[1:] + beta[:-1] / np.maximum(alpha[:-1], 1e-30)
            off_diagonal = np.sqrt(beta) / np.maximum(alpha[:-1], 1e-30)
        else:
            off_diagonal = np.empty(0, dtype=np.float64)
        tridiagonal = np.diag(diagonal)
        if off_diagonal.size > 0:
            tridiagonal[np.arange(off_diagonal.size), np.arange(1, off_diagonal.size + 1)] = off_diagonal
            tridiagonal[np.arange(1, off_diagonal.size + 1), np.arange(off_diagonal.size)] = off_diagonal
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (tridiagonal + tridiagonal.T))
        clipped_eigenvalues = np.maximum(eigenvalues, 1e-12)
        estimates.append(float(np.sum((eigenvectors[0, :] ** 2) * np.log(clipped_eigenvalues))))
    if not estimates:
        raise RuntimeError("Sample-space CG logdet recorder did not capture any valid Lanczos estimates.")
    return float(baseline_logdet + int(dimension) * float(np.mean(np.asarray(estimates, dtype=np.float64))))


def _effective_sample_space_preconditioner_rank(
    genotype_matrix: StandardizedGenotypeMatrix,
    sample_count: int,
    variant_count: int,
    requested_rank: int,
) -> int:
    resolved_rank = max(int(requested_rank), 0)
    if resolved_rank == 0:
        return 0
    return min(resolved_rank, sample_count, variant_count)


def _streaming_cupy_backend_available(genotype_matrix: StandardizedGenotypeMatrix) -> bool:
    return (
        genotype_matrix._cupy_cache is None
        and genotype_matrix.raw is not None
        and not genotype_matrix.supports_jax_dense_ops()
        and _try_import_cupy() is not None
    )


def _prefer_iterative_variant_space(
    genotype_matrix: StandardizedGenotypeMatrix,
    sample_count: int,
    variant_count: int,
    *,
    compute_beta_variance: bool,
    compute_logdet: bool,
    initial_beta_guess: np.ndarray | None,
) -> bool:
    if _streaming_cupy_backend_available(genotype_matrix):
        return False
    streaming_matrix = (
        genotype_matrix._cupy_cache is None
        and genotype_matrix._dense_cache is None
        and genotype_matrix._jax_cache is None
        and genotype_matrix.raw is not None
    )
    if streaming_matrix:
        return False
    return (
        not compute_beta_variance
        and
        not compute_logdet
        and initial_beta_guess is not None
        and genotype_matrix._cupy_cache is None
        and variant_count > sample_count
    )


def _use_exact_sample_space_solve(
    *,
    sample_count: int,
    variant_count: int,
    exact_solver_matrix_limit: int,
    use_gpu_exact_variant: bool,
) -> bool:
    sample_space_is_exact = sample_count <= exact_solver_matrix_limit
    variant_space_is_exact = variant_count <= exact_solver_matrix_limit
    if use_gpu_exact_variant:
        return False
    if not sample_space_is_exact:
        return False
    if not variant_space_is_exact:
        return True
    return sample_count <= variant_count


def _weighted_covariate_projection_signature(
    inverse_diagonal_noise: np.ndarray,
    covariate_matrix: np.ndarray,
) -> str:
    hasher = hashlib.sha256()
    hasher.update(np.ascontiguousarray(np.asarray(inverse_diagonal_noise, dtype=np.float64)).view(np.uint8).tobytes())
    covariate_array = np.asarray(covariate_matrix, dtype=np.float64)
    hasher.update(np.asarray(covariate_array.shape, dtype=np.int64).tobytes())
    hasher.update(np.ascontiguousarray(covariate_array).view(np.uint8).tobytes())
    return hasher.hexdigest()


def _cached_weighted_covariate_projection(
    genotype_matrix: StandardizedGenotypeMatrix,
    covariate_matrix: np.ndarray,
    inverse_diagonal_noise: np.ndarray,
    batch_size: int,
    warm_start: _RestrictedPosteriorWarmStart | None,
    *,
    return_gpu: bool = False,
    cupy=None,
):
    variant_count = genotype_matrix.shape[1]
    covariate_count = int(covariate_matrix.shape[1])
    if covariate_count == 0:
        if return_gpu:
            if cupy is None:
                raise ValueError("cupy is required when return_gpu=True.")
            return cupy.asarray(np.zeros((variant_count, 0), dtype=np.float64), dtype=cupy.float64)
        return np.zeros((variant_count, 0), dtype=np.float64)
    matrix_token = id(genotype_matrix)
    cache_signature = _weighted_covariate_projection_signature(inverse_diagonal_noise, covariate_matrix)
    cache_valid = (
        warm_start is not None
        and warm_start.weighted_covariate_projection_matrix_token == matrix_token
        and warm_start.weighted_covariate_projection_signature == cache_signature
    )
    if return_gpu:
        if cupy is None:
            raise ValueError("cupy is required when return_gpu=True.")
        if (
            cache_valid
            and warm_start is not None
            and warm_start.weighted_covariate_projection_gpu is not None
            and getattr(warm_start.weighted_covariate_projection_gpu, "shape", None) == (variant_count, covariate_count)
        ):
            return warm_start.weighted_covariate_projection_gpu
        if (
            cache_valid
            and warm_start is not None
            and warm_start.weighted_covariate_projection is not None
            and warm_start.weighted_covariate_projection.shape == (variant_count, covariate_count)
        ):
            projection_gpu = cupy.asarray(warm_start.weighted_covariate_projection, dtype=cupy.float64)
            warm_start.weighted_covariate_projection_gpu = projection_gpu
            return projection_gpu
    if (
        cache_valid
        and warm_start is not None
        and warm_start.weighted_covariate_projection is not None
        and warm_start.weighted_covariate_projection.shape == (variant_count, covariate_count)
    ):
        return np.asarray(warm_start.weighted_covariate_projection, dtype=np.float64)
    weighted_covariates = np.asarray(inverse_diagonal_noise[:, None] * covariate_matrix, dtype=np.float64)
    if genotype_matrix._cupy_cache is not None:
        cp_module = cupy
        if cp_module is None:
            import cupy as imported_cupy
            cp_module = imported_cupy
        weighted_covariates_gpu = cp_module.asarray(weighted_covariates, dtype=cp_module.float64)
        projection_gpu = cp_module.empty((variant_count, covariate_count), dtype=cp_module.float64)
        for batch_slice, standardized_batch in _iter_cupy_cache_standardized_batches(
            genotype_matrix._cupy_cache,
            sample_count=genotype_matrix.shape[0],
            batch_size=batch_size,
            cupy=cp_module,
            dtype=cp_module.float64,
        ):
            projection_gpu[batch_slice, :] = standardized_batch.T @ weighted_covariates_gpu
        if warm_start is not None:
            warm_start.weighted_covariate_projection_matrix_token = matrix_token
            warm_start.weighted_covariate_projection_signature = cache_signature
            warm_start.weighted_covariate_projection_gpu = projection_gpu
        if return_gpu:
            return projection_gpu
        projection_matrix = _cupy_array_to_numpy(projection_gpu, dtype=np.float64)
    elif _streaming_cupy_backend_available(genotype_matrix):
        cp = cupy if cupy is not None else _try_import_cupy()
        assert cp is not None
        raw_matrix = cast(RawGenotypeMatrix, genotype_matrix.raw)
        weighted_covariates_gpu = cp.asarray(weighted_covariates, dtype=cp.float64)
        projection_gpu = cp.empty((variant_count, covariate_count), dtype=cp.float64)
        for batch_slice, standardized_batch in _iter_standardized_gpu_batches(
            raw_matrix,
            genotype_matrix.variant_indices,
            genotype_matrix.means,
            genotype_matrix.scales,
            batch_size=batch_size,
            cupy=cp,
            dtype=cp.float64,
        ):
            projection_gpu[batch_slice, :] = standardized_batch.T @ weighted_covariates_gpu
        if warm_start is not None:
            warm_start.weighted_covariate_projection_matrix_token = matrix_token
            warm_start.weighted_covariate_projection_signature = cache_signature
            warm_start.weighted_covariate_projection_gpu = projection_gpu
        if return_gpu:
            return projection_gpu
        projection_matrix = _cupy_array_to_numpy(projection_gpu, dtype=np.float64)
    else:
        projection_matrix = np.asarray(
            genotype_matrix.transpose_matmat(weighted_covariates, batch_size=batch_size),
            dtype=np.float64,
        )
    if warm_start is not None:
        warm_start.weighted_covariate_projection_matrix_token = matrix_token
        warm_start.weighted_covariate_projection_signature = cache_signature
        warm_start.weighted_covariate_projection = np.asarray(projection_matrix, dtype=np.float64)
        if not return_gpu:
            warm_start.weighted_covariate_projection_gpu = None
    return np.asarray(projection_matrix, dtype=np.float64)


def _build_restricted_projector_gpu_bundle(
    inverse_diagonal_noise: np.ndarray,
    covariate_matrix: np.ndarray,
    covariate_precision_cholesky: np.ndarray,
    *,
    cp,
    dtype,
):
    inverse_diagonal_noise_gpu = cp.asarray(inverse_diagonal_noise, dtype=dtype)
    covariate_matrix_gpu = cp.asarray(covariate_matrix, dtype=dtype)
    covariate_precision_cholesky_gpu = cp.asarray(covariate_precision_cholesky, dtype=dtype)
    weighted_covariates_gpu = inverse_diagonal_noise_gpu[:, None] * covariate_matrix_gpu
    return (
        inverse_diagonal_noise_gpu,
        covariate_matrix_gpu,
        weighted_covariates_gpu,
        covariate_precision_cholesky_gpu,
    )


def _apply_restricted_projector_gpu(
    right_hand_side_gpu,
    projector_bundle,
    *,
    cp,
    solve_triangular_gpu,
):
    inverse_diagonal_noise_gpu, covariate_matrix_gpu, weighted_covariates_gpu, covariate_precision_cholesky_gpu = (
        projector_bundle
    )
    rhs_gpu = cp.asarray(right_hand_side_gpu, dtype=covariate_precision_cholesky_gpu.dtype)
    if rhs_gpu.ndim == 1:
        weighted_rhs = inverse_diagonal_noise_gpu * rhs_gpu
    elif rhs_gpu.ndim == 2:
        weighted_rhs = inverse_diagonal_noise_gpu[:, None] * rhs_gpu
    else:
        raise ValueError("restricted projector expects a vector or matrix right-hand side.")
    if covariate_matrix_gpu.shape[1] == 0:
        return weighted_rhs
    correction_rhs = covariate_matrix_gpu.T @ weighted_rhs
    correction = _gpu_cholesky_solve(
        correction_rhs,
        covariate_precision_cholesky_gpu,
        solve_triangular_gpu,
    )
    return weighted_rhs - weighted_covariates_gpu @ correction


def _solve_sample_space_rhs_gpu(
    genotype_matrix: StandardizedGenotypeMatrix,
    prior_variances: np.ndarray,
    diagonal_noise: np.ndarray,
    right_hand_side: np.ndarray,
    initial_guess: np.ndarray | None,
    tolerance: float,
    max_iterations: int,
    preconditioner,
    batch_size: int,
    return_iterations: bool = False,
    column_iteration_limits: np.ndarray | None = None,
    required_columns: np.ndarray | None = None,
    lanczos_recorder: _SampleSpaceCGLanczosRecorder | None = None,
) -> np.ndarray | tuple[np.ndarray, int]:
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
    n_rhs = int(right_hand_side_gpu64.shape[1])
    if column_iteration_limits is None:
        resolved_iteration_limits = np.full(n_rhs, int(max_iterations), dtype=np.int32)
    else:
        resolved_iteration_limits = np.asarray(column_iteration_limits, dtype=np.int32).reshape(-1)
        if resolved_iteration_limits.shape != (n_rhs,):
            raise ValueError("column_iteration_limits must match the number of rhs columns.")
    if required_columns is None:
        required_mask = np.ones(n_rhs, dtype=bool)
    else:
        required_mask = np.asarray(required_columns, dtype=bool).reshape(-1)
        if required_mask.shape != (n_rhs,):
            raise ValueError("required_columns must match the number of rhs columns.")
    initial_solution_gpu64 = None
    if initial_guess is not None:
        initial_solution_gpu64 = cp.asarray(initial_guess, dtype=cp.float64)
        if initial_solution_gpu64.ndim == 1:
            initial_solution_gpu64 = initial_solution_gpu64[:, None]
        if initial_solution_gpu64.shape != right_hand_side_gpu64.shape:
            raise ValueError("GPU sample-space initial_guess must match right_hand_side shape.")

    rhs_norm_sq = _cupy_array_to_numpy(cp.sum(right_hand_side_gpu64 * right_hand_side_gpu64, axis=0, dtype=cp.float64), dtype=np.float64)
    convergence_threshold_sq = np.maximum(float(tolerance) * float(tolerance), float(tolerance) * float(tolerance) * rhs_norm_sq)
    mixed_precision_enabled = compute_cp_dtype == cp.float32
    mixed_precision_failure: RuntimeError | None = None
    total_iterations_used = 0

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
        residual_norm_sq = _cupy_array_to_numpy(cp.sum(residual_gpu64 * residual_gpu64, axis=0, dtype=cp.float64), dtype=np.float64)
        return residual_gpu64, residual_norm_sq

    if not mixed_precision_enabled:
        solution_gpu64_result = _solve_sample_space_rhs_gpu_inner(
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
            initial_guess_gpu=initial_solution_gpu64,
            column_iteration_limits=resolved_iteration_limits,
            required_columns=required_mask,
            lanczos_recorder=lanczos_recorder,
        )
        solution_gpu64, iterations_used = _resolve_sample_space_solve_result(
            solution_gpu64_result,
            fallback_iterations=max_iterations,
        )
        total_iterations_used += iterations_used
        if solution_gpu64.ndim == 1:
            solution_gpu64 = solution_gpu64[:, None]
        _, residual_norm_sq = true_residual(solution_gpu64)
        final_residual = float(np.max(residual_norm_sq))
        final_threshold = float(np.max(convergence_threshold_sq))
        required_residual = residual_norm_sq[required_mask]
        required_threshold = convergence_threshold_sq[required_mask]
        final_residual = float(np.max(required_residual)) if required_residual.size > 0 else 0.0
        final_threshold = float(np.max(required_threshold)) if required_threshold.size > 0 else 0.0
        if final_residual > final_threshold:
            raise RuntimeError(
                "GPU conjugate-gradient solve failed to converge: "
                + f"residual={final_residual:.2e} threshold={final_threshold:.2e} "
                + f"iterations={max_iterations}"
            )
        solution = np.asarray(solution_gpu64.get() if hasattr(solution_gpu64, "get") else solution_gpu64, dtype=np.float64)
        resolved_solution = solution[:, 0] if vector_input else solution
        if return_iterations:
            return resolved_solution, total_iterations_used
        return resolved_solution

    solution_gpu64 = (
        cp.asarray(initial_solution_gpu64, dtype=cp.float64)
        if initial_solution_gpu64 is not None
        else cp.zeros(right_hand_side_gpu64.shape, dtype=cp.float64)
    )
    inner_tolerance = max(float(tolerance), 1e-4)
    max_refinement_steps = 4
    for _refinement_index in range(max_refinement_steps):
        residual_gpu64, residual_norm_sq = true_residual(solution_gpu64)
        if np.all(residual_norm_sq <= convergence_threshold_sq):
            solution = np.asarray(solution_gpu64.get() if hasattr(solution_gpu64, "get") else solution_gpu64, dtype=np.float64)
            resolved_solution = solution[:, 0] if vector_input else solution
            if return_iterations:
                return resolved_solution, total_iterations_used
            return resolved_solution
        try:
            correction_gpu64_result = _solve_sample_space_rhs_gpu_inner(
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
                initial_guess_gpu=None,
                column_iteration_limits=resolved_iteration_limits,
                required_columns=required_mask,
                lanczos_recorder=lanczos_recorder if _refinement_index == 0 else None,
            )
            correction_gpu64, iterations_used = _resolve_sample_space_solve_result(
                correction_gpu64_result,
                fallback_iterations=max_iterations,
            )
            total_iterations_used += iterations_used
        except RuntimeError as exc:
            mixed_precision_failure = exc
            break
        if correction_gpu64.ndim == 1:
            correction_gpu64 = correction_gpu64[:, None]
        solution_gpu64 += correction_gpu64
    residual_gpu64, residual_norm_sq = true_residual(solution_gpu64)
    if np.any(residual_norm_sq[required_mask] > convergence_threshold_sq[required_mask]):
        fallback_reason = (
            str(mixed_precision_failure)
            if mixed_precision_failure is not None
            else "mixed-precision iterative refinement did not hit the float64 residual target"
        )
        log(f"      sample-space solve: retrying GPU CG in float64 ({fallback_reason})")
        correction_gpu64_result = _solve_sample_space_rhs_gpu_inner(
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
            initial_guess_gpu=None,
            column_iteration_limits=resolved_iteration_limits,
            required_columns=required_mask,
            lanczos_recorder=None,
        )
        correction_gpu64, iterations_used = _resolve_sample_space_solve_result(
            correction_gpu64_result,
            fallback_iterations=max_iterations,
        )
        total_iterations_used += iterations_used
        if correction_gpu64.ndim == 1:
            correction_gpu64 = correction_gpu64[:, None]
        solution_gpu64 += correction_gpu64
        _, residual_norm_sq = true_residual(solution_gpu64)
    required_residual = residual_norm_sq[required_mask]
    required_threshold = convergence_threshold_sq[required_mask]
    final_residual = float(np.max(required_residual)) if required_residual.size > 0 else 0.0
    final_threshold = float(np.max(required_threshold)) if required_threshold.size > 0 else 0.0
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
    resolved_solution = solution[:, 0] if vector_input else solution
    if return_iterations:
        return resolved_solution, total_iterations_used
    return resolved_solution


def _solve_sample_space_rhs_cpu(
    genotype_matrix: StandardizedGenotypeMatrix,
    prior_variances: np.ndarray,
    diagonal_noise: np.ndarray,
    right_hand_side: np.ndarray,
    initial_guess: np.ndarray | None,
    tolerance: float,
    max_iterations: int,
    preconditioner: Callable[[np.ndarray], np.ndarray | jnp.ndarray] | Callable[[jnp.ndarray], np.ndarray | jnp.ndarray] | np.ndarray | jnp.ndarray,
    batch_size: int,
    return_iterations: bool = False,
    column_iteration_limits: np.ndarray | None = None,
    required_columns: np.ndarray | None = None,
    lanczos_recorder: _SampleSpaceCGLanczosRecorder | None = None,
) -> np.ndarray | tuple[np.ndarray, int]:
    import time

    rhs = np.asarray(right_hand_side, dtype=np.float64)
    vector_input = rhs.ndim == 1
    if vector_input:
        rhs = rhs[:, None]
    elif rhs.ndim != 2:
        raise ValueError("CPU sample-space solve expects a vector or matrix right-hand side.")
    n_rhs = int(rhs.shape[1])
    if column_iteration_limits is None:
        resolved_iteration_limits = np.full(n_rhs, int(max_iterations), dtype=np.int32)
    else:
        resolved_iteration_limits = np.asarray(column_iteration_limits, dtype=np.int32).reshape(-1)
        if resolved_iteration_limits.shape != (n_rhs,):
            raise ValueError("column_iteration_limits must match the number of rhs columns.")
    if required_columns is None:
        required_mask = np.ones(n_rhs, dtype=bool)
    else:
        required_mask = np.asarray(required_columns, dtype=bool).reshape(-1)
        if required_mask.shape != (n_rhs,):
            raise ValueError("required_columns must match the number of rhs columns.")

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
    if initial_guess is not None:
        solution = np.array(initial_guess, dtype=np.float64, copy=True)
        if solution.ndim == 1:
            solution = solution[:, None]
        if solution.shape != rhs.shape:
            raise ValueError("CPU sample-space initial_guess must match right_hand_side shape.")
    else:
        solution = (
            np.array(apply_preconditioner(rhs), dtype=np.float64, copy=True)
            if apply_preconditioner is not None
            else np.array(rhs / np.maximum(np.asarray(preconditioner, dtype=np.float64)[:, None], 1e-12), dtype=np.float64, copy=True)
        )
    residual = rhs - apply_operator(solution)
    residual_norm_sq = np.sum(residual * residual, axis=0, dtype=np.float64)
    initial_residual_norm_sq = residual_norm_sq.copy()
    rhs_norm_sq = np.sum(rhs * rhs, axis=0, dtype=np.float64)
    convergence_threshold_sq = np.maximum(tol_sq, tol_sq * np.maximum(residual_norm_sq, rhs_norm_sq))
    converged = residual_norm_sq <= convergence_threshold_sq
    done = converged | (resolved_iteration_limits <= 0)
    if np.all(done):
        resolved_solution = solution[:, 0] if vector_input else solution
        if return_iterations:
            return resolved_solution, 0
        return resolved_solution

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
        active_columns = np.flatnonzero(~done).astype(np.int32, copy=False)
        if active_columns.size == 0:
            break
        masked_search = search_direction[:, active_columns]
        operator_search = apply_operator(masked_search)
        step_denom = np.sum(masked_search * operator_search, axis=0, dtype=np.float64)
        if np.any(~np.isfinite(step_denom) | (step_denom <= 0.0)):
            raise RuntimeError("CPU conjugate-gradient operator is not positive definite.")
        step_scale = residual_dot[active_columns] / step_denom
        _record_sample_space_cg_lanczos_alpha(
            lanczos_recorder,
            iteration_index=iteration_index,
            active_columns=active_columns,
            step_scale=step_scale,
        )
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
        limit_reached_active = (iteration_index + 1) >= resolved_iteration_limits[active_columns]
        if np.any(limit_reached_active):
            _finalize_sample_space_cg_lanczos_steps(
                lanczos_recorder,
                completed_columns=active_columns[limit_reached_active],
                iteration_count=iteration_index + 1,
            )
        done = converged | ((iteration_index + 1) >= resolved_iteration_limits)
        if np.all(done):
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
        _record_sample_space_cg_lanczos_beta(
            lanczos_recorder,
            iteration_index=iteration_index,
            active_columns=active_columns,
            beta_value=beta_active,
        )
        if np.any(~np.isfinite(beta_active) | (beta_active < 0.0)):
            raise RuntimeError("CPU conjugate-gradient preconditioner produced an invalid update.")
        search_direction[:, active_columns] = refreshed_preconditioned + (
            search_direction[:, active_columns] * beta_active[None, :]
        )
        residual_dot[active_columns] = updated_residual_dot_active

        now = time.monotonic()
        if now - last_log >= 5.0:
            progress = np.zeros_like(residual_norm_sq)
            progress[done] = 100.0
            unconverged = ~done
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
                + f"active={int(np.sum(~done))}/{rhs.shape[1]}  residual={float(np.max(residual_norm_sq)):.2e}  "
                + f"({now - t_start:.1f}s)"
            )
            last_log = now

    _finalize_sample_space_cg_lanczos_steps(
        lanczos_recorder,
        completed_columns=lanczos_recorder.monitored_columns[lanczos_recorder.step_lengths == 0]
        if lanczos_recorder is not None
        else np.empty(0, dtype=np.int32),
        iteration_count=iteration_index + 1 if "iteration_index" in locals() else 0,
    )
    required_residual = residual_norm_sq[required_mask]
    required_threshold = convergence_threshold_sq[required_mask]
    final_residual = float(np.max(required_residual)) if required_residual.size > 0 else 0.0
    final_threshold = float(np.max(required_threshold)) if required_threshold.size > 0 else 0.0
    if final_residual > final_threshold:
        raise RuntimeError(
            "CPU conjugate-gradient solve failed to converge: "
            + f"residual={final_residual:.2e} threshold={final_threshold:.2e} "
            + f"iterations={max_iterations}"
        )
    resolved_solution = solution[:, 0] if vector_input else solution
    iterations_used = iteration_index + 1 if "iteration_index" in locals() else 0
    if return_iterations:
        return resolved_solution, iterations_used
    return resolved_solution


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
    warm_start: _RestrictedPosteriorWarmStart | None = None,
) -> np.ndarray:
    compute_np_dtype = gpu_compute_numpy_dtype()
    weighted_covariate_projection = _cached_weighted_covariate_projection(
        genotype_matrix=genotype_matrix,
        covariate_matrix=covariate_matrix,
        inverse_diagonal_noise=inverse_diagonal_noise,
        batch_size=batch_size,
        warm_start=warm_start,
    )
    cross = np.asarray(weighted_covariate_projection.T, dtype=compute_np_dtype)
    correction = _cholesky_solve(covariate_precision_cholesky, cross)
    diag_correction = np.sum(cross * correction, axis=0)
    if genotype_matrix._cupy_cache is not None:
        import cupy as cp
        inv_d = cp.asarray(inverse_diagonal_noise, dtype=cp.float64)
        # Compute diag(X^T D^{-1} X) on GPU and reuse cached X^T D^{-1} C.
        raw_diag = cp.empty(genotype_matrix.shape[1], dtype=cp.float64)
        for batch_slice, standardized_batch in _iter_cupy_cache_standardized_batches(
            genotype_matrix._cupy_cache,
            sample_count=genotype_matrix.shape[0],
            batch_size=batch_size,
            cupy=cp,
            dtype=cp.float64,
        ):
            weighted_batch = inv_d[:, None] * standardized_batch
            raw_diag[batch_slice] = cp.sum(standardized_batch * weighted_batch, axis=0)
        raw_diag_np = raw_diag.get().astype(compute_np_dtype)
        return np.maximum(prior_precision + raw_diag_np - diag_correction, 1e-8)
    if _streaming_cupy_backend_available(genotype_matrix):
        cupy = _try_import_cupy()
        if cupy is not None:
            raw_matrix = cast(RawGenotypeMatrix, genotype_matrix.raw)
            inv_d_gpu = cupy.asarray(inverse_diagonal_noise, dtype=cupy.float64)
            raw_diag_gpu = cupy.empty(genotype_matrix.shape[1], dtype=cupy.float64)
            for batch_slice, standardized_batch in _iter_standardized_gpu_batches(
                raw_matrix,
                genotype_matrix.variant_indices,
                genotype_matrix.means,
                genotype_matrix.scales,
                batch_size=batch_size,
                cupy=cupy,
                dtype=cupy.float64,
            ):
                weighted_batch = inv_d_gpu[:, None] * standardized_batch
                raw_diag_gpu[batch_slice] = cupy.sum(standardized_batch * weighted_batch, axis=0)
            raw_diag_np = raw_diag_gpu.get().astype(compute_np_dtype)
            return np.maximum(prior_precision + raw_diag_np - diag_correction, 1e-8)
    diagonal = np.asarray(prior_precision, dtype=compute_np_dtype).copy()
    for batch in genotype_matrix.iter_column_batches(batch_size=batch_size):
        genotype_batch = np.asarray(batch.values, dtype=compute_np_dtype)
        weighted_batch = inverse_diagonal_noise[:, None] * genotype_batch
        diagonal[batch.variant_indices] += np.sum(genotype_batch * weighted_batch, axis=0)
    diagonal -= diag_correction
    return np.maximum(diagonal, 1e-8)


def _posterior_variance_low_rank_residual_diagonal(
    solve_variant_rhs: Callable[[np.ndarray], np.ndarray],
    dimension: int,
    probe_count: int,
    random_seed: int,
) -> np.ndarray:
    """Estimate diag(A^{-1}) from one batched solve via low-rank + residual probes."""
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


def _orthogonal_probe_matrix_in_complement(
    *,
    dimension: int,
    probe_count: int,
    random_seed: int,
    orthonormal_basis: np.ndarray | None,
) -> np.ndarray:
    if orthonormal_basis is None or orthonormal_basis.size == 0:
        return _orthogonal_probe_matrix(
            dimension=dimension,
            probe_count=probe_count,
            random_seed=random_seed,
        )
    basis_matrix = np.asarray(orthonormal_basis, dtype=np.float64)
    if basis_matrix.ndim != 2 or basis_matrix.shape[0] != int(dimension):
        raise ValueError("orthonormal_basis must have shape (dimension, rank).")
    if probe_count < 1:
        raise ValueError("probe_count must be positive.")
    basis_matrix, triangular_matrix = np.linalg.qr(basis_matrix, mode="reduced")
    diagonal = np.abs(np.diag(triangular_matrix))
    effective_rank = int(np.sum(diagonal > 1e-10))
    if effective_rank <= 0:
        return _orthogonal_probe_matrix(
            dimension=dimension,
            probe_count=probe_count,
            random_seed=random_seed,
        )
    basis_matrix = np.asarray(basis_matrix[:, :effective_rank], dtype=np.float64)
    residual_dimension = max(int(dimension) - effective_rank, 0)
    if residual_dimension <= 0:
        return np.zeros((int(dimension), 0), dtype=np.float64)
    random_generator = np.random.default_rng(random_seed)
    probe_blocks: list[np.ndarray] = []
    probes_remaining = int(probe_count)
    while probes_remaining > 0:
        block_probe_count = min(probes_remaining, residual_dimension)
        gaussian_block = random_generator.standard_normal((int(dimension), block_probe_count)).astype(np.float64)
        projected_block = gaussian_block - basis_matrix @ (basis_matrix.T @ gaussian_block)
        orthogonal_block, triangular_block = np.linalg.qr(projected_block, mode="reduced")
        diagonal_signs = np.sign(np.diag(triangular_block))
        diagonal_signs[diagonal_signs == 0.0] = 1.0
        orthogonal_block *= diagonal_signs[None, :]
        diagonal = np.abs(np.diag(triangular_block))
        effective_block_rank = int(np.sum(diagonal > 1e-10))
        if effective_block_rank <= 0:
            raise RuntimeError("Failed to build residual probe block in the complement subspace.")
        probe_blocks.append(np.sqrt(float(residual_dimension)) * orthogonal_block[:, :effective_block_rank])
        probes_remaining -= effective_block_rank
    return np.ascontiguousarray(np.column_stack(probe_blocks), dtype=np.float64)


def _sample_space_variance_probe_plan(
    *,
    sample_count: int,
    probe_count: int,
    random_seed: int,
    sample_space_preconditioner_cache_entry: _SampleSpacePreconditionerCacheEntry | None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if probe_count <= 0:
        return None, None, None
    deflation_basis = None
    if probe_count > 1 and sample_space_preconditioner_cache_entry is not None:
        basis_source = getattr(sample_space_preconditioner_cache_entry, "nystrom_basis_cpu", None)
        basis_source_gpu = getattr(sample_space_preconditioner_cache_entry, "nystrom_basis_gpu", None)
        if basis_source is None and basis_source_gpu is not None:
            basis_source = _cupy_array_to_numpy(
                basis_source_gpu,
                dtype=np.float64,
            )
        if basis_source is not None:
            available_rank = min(
                int(np.asarray(basis_source).shape[1]),
                max(int(sample_count) - 1, 0),
                max(int(probe_count) // 2, 0),
            )
            if available_rank > 0:
                deflation_basis = np.asarray(basis_source[:, :available_rank], dtype=np.float64)
    residual_probe_count = int(probe_count) - (0 if deflation_basis is None else int(deflation_basis.shape[1]))
    residual_probes = (
        _orthogonal_probe_matrix_in_complement(
            dimension=sample_count,
            probe_count=residual_probe_count,
            random_seed=random_seed,
            orthonormal_basis=deflation_basis,
        )
        if residual_probe_count > 0
        else None
    )
    if deflation_basis is None:
        assert residual_probes is not None
        return residual_probes, None, residual_probes
    if residual_probes is None or residual_probes.shape[1] == 0:
        return deflation_basis, deflation_basis, None
    return (
        np.ascontiguousarray(np.concatenate([deflation_basis, residual_probes], axis=1), dtype=np.float64),
        deflation_basis,
        residual_probes,
    )


def _use_gpu_exact_variant_solve(
    genotype_matrix: StandardizedGenotypeMatrix,
    variant_count: int,
    exact_solver_matrix_limit: int,
    covariate_count: int,
) -> bool:
    if genotype_matrix._cupy_cache is None:
        return False
    if variant_count <= exact_solver_matrix_limit:
        return False
    cupy = _try_import_cupy()
    if cupy is None:
        return False
    sample_count = genotype_matrix.shape[0]
    cache_is_int8_standardized = _cupy_cache_is_int8_standardized(genotype_matrix._cupy_cache)
    if _gpu_exact_variant_full_matrix_fits(
        cupy,
        sample_count=sample_count,
        variant_count=variant_count,
        covariate_count=covariate_count,
        cache_is_int8_standardized=cache_is_int8_standardized,
    ):
        return True
    return _gpu_exact_variant_tile_size(
        cupy,
        sample_count=sample_count,
        variant_count=variant_count,
        covariate_count=covariate_count,
    ) > 0


def _should_use_posterior_working_set(
    genotype_matrix: StandardizedGenotypeMatrix,
    *,
    variant_count: int,
    compute_logdet: bool,
    compute_beta_variance: bool,
    posterior_working_sets: bool,
    posterior_working_set_min_variants: int,
    exact_solver_matrix_limit: int,
    use_exact_variant: bool,
    use_gpu_exact_variant: bool,
    allow_working_set: bool,
) -> bool:
    promoted_min_variants = int(posterior_working_set_min_variants)
    if (
        genotype_matrix._cupy_cache is not None
        and not use_exact_variant
        and not use_gpu_exact_variant
    ):
        promoted_min_variants = min(
            promoted_min_variants,
            max(int(exact_solver_matrix_limit) * 8, 4_096),
        )
    return (
        allow_working_set
        and posterior_working_sets
        and not compute_logdet
        and not compute_beta_variance
        and variant_count >= max(promoted_min_variants, 0)
        and genotype_matrix.shape[0] > 0
    )


def _working_set_screening_score(
    gradient: np.ndarray,
    beta: np.ndarray,
    prior_variances: np.ndarray,
) -> np.ndarray:
    coefficient_scale_violation = np.abs(np.asarray(prior_variances, dtype=np.float64) * np.asarray(gradient, dtype=np.float64))
    return np.maximum(coefficient_scale_violation, np.abs(np.asarray(beta, dtype=np.float64)))


def _working_set_posterior_update_score(
    gradient: np.ndarray,
    prior_variances: np.ndarray,
) -> np.ndarray:
    return np.abs(np.asarray(prior_variances, dtype=np.float64) * np.asarray(gradient, dtype=np.float64))


def _ordered_unique_indices(
    index_blocks: Sequence[np.ndarray | None],
    variant_count: int,
) -> np.ndarray:
    seen = np.zeros(max(int(variant_count), 0), dtype=bool)
    ordered_blocks: list[np.ndarray] = []
    for block in index_blocks:
        if block is None:
            continue
        block_array = np.asarray(block, dtype=np.int64).reshape(-1)
        if block_array.size == 0:
            continue
        block_array = block_array[(block_array >= 0) & (block_array < variant_count)]
        if block_array.size == 0:
            continue
        unseen_mask = ~seen[block_array]
        if not np.any(unseen_mask):
            continue
        unique_block = np.asarray(block_array[unseen_mask], dtype=np.int32)
        seen[unique_block] = True
        ordered_blocks.append(unique_block)
    if not ordered_blocks:
        return np.empty(0, dtype=np.int32)
    return np.concatenate(ordered_blocks).astype(np.int32, copy=False)


def _active_working_set_indices(
    beta: np.ndarray,
    coefficient_tolerance: float,
) -> np.ndarray:
    active_threshold = max(float(coefficient_tolerance), 1e-10)
    return np.flatnonzero(np.abs(np.asarray(beta, dtype=np.float64)) > active_threshold).astype(np.int32, copy=False)


def _posterior_working_set_indices(
    screening_score: np.ndarray,
    ever_active_indices: np.ndarray,
    target_size: int,
) -> np.ndarray:
    total_variants = int(np.asarray(screening_score).shape[0])
    mandatory = _ordered_unique_indices([ever_active_indices], total_variants)
    if mandatory.shape[0] >= total_variants:
        return np.arange(total_variants, dtype=np.int32)
    if mandatory.shape[0] >= max(int(target_size), 1):
        return mandatory
    candidate_score = np.asarray(screening_score, dtype=np.float64).reshape(-1)
    remaining_size = max(int(target_size) - mandatory.shape[0], 0)
    if remaining_size <= 0:
        return mandatory
    available_mask = np.ones(total_variants, dtype=bool)
    available_mask[mandatory] = False
    available_indices = np.flatnonzero(available_mask).astype(np.int32, copy=False)
    if available_indices.size <= remaining_size:
        return _ordered_unique_indices([mandatory, available_indices], total_variants)
    ranked_remaining = available_indices[
        np.argsort(candidate_score[available_indices])[-remaining_size:][::-1]
    ]
    return _ordered_unique_indices([mandatory, ranked_remaining], total_variants)


def _reset_posterior_working_set_warm_start(
    warm_start: _RestrictedPosteriorWarmStart | None,
    genotype_matrix: StandardizedGenotypeMatrix,
    variant_count: int,
) -> None:
    if warm_start is None:
        return
    matrix_token = id(genotype_matrix)
    if (
        warm_start.posterior_working_set_variant_count != int(variant_count)
        or warm_start.posterior_working_set_matrix_token != matrix_token
    ):
        warm_start.posterior_working_set_variant_count = int(variant_count)
        warm_start.posterior_working_set_matrix_token = matrix_token
        warm_start.posterior_working_set_ever_active = None
        warm_start.posterior_working_set_screening_score = None
        warm_start.posterior_working_set_target_size = None


def _posterior_working_set_seed_score(
    *,
    beta: np.ndarray,
    prior_variances: np.ndarray,
    warm_start: _RestrictedPosteriorWarmStart | None,
    variant_count: int,
) -> np.ndarray:
    beta_score = np.abs(np.asarray(beta, dtype=np.float64))
    if beta_score.shape != (variant_count,):
        raise ValueError("beta must match variant count for posterior working sets.")
    stale_score = None if warm_start is None else warm_start.posterior_working_set_screening_score
    if stale_score is not None:
        stale_score_array = np.asarray(stale_score, dtype=np.float64).reshape(-1)
        if stale_score_array.shape == (variant_count,):
            return np.maximum(beta_score, stale_score_array)
    if np.any(beta_score > 0.0):
        return beta_score
    return np.asarray(prior_variances, dtype=np.float64).reshape(-1)


def _posterior_working_set_target_size(
    *,
    initial_size: int,
    ever_active_count: int,
    warm_start: _RestrictedPosteriorWarmStart | None,
    variant_count: int,
) -> int:
    stale_target_size = (
        0
        if warm_start is None or warm_start.posterior_working_set_target_size is None
        else int(warm_start.posterior_working_set_target_size)
    )
    return min(
        max(int(initial_size), int(ever_active_count), stale_target_size, 1),
        int(variant_count),
    )


def _update_posterior_working_set_warm_start(
    *,
    warm_start: _RestrictedPosteriorWarmStart | None,
    ever_active_indices: np.ndarray,
    screening_score: np.ndarray,
    target_size: int,
    variant_count: int,
) -> None:
    if warm_start is None:
        return
    warm_start.posterior_working_set_ever_active = np.asarray(ever_active_indices, dtype=np.int32)
    warm_start.posterior_working_set_screening_score = np.asarray(screening_score, dtype=np.float64).copy()
    warm_start.posterior_working_set_target_size = min(
        max(int(target_size), warm_start.posterior_working_set_ever_active.shape[0], 1),
        int(variant_count),
    )


def _restricted_posterior_result_without_diagnostics(
    *,
    alpha: np.ndarray,
    beta: np.ndarray,
    prior_variances: np.ndarray,
    projected_targets: np.ndarray,
    linear_predictor: np.ndarray,
    restricted_quadratic: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    return (
        np.asarray(alpha, dtype=np.float64),
        np.asarray(beta, dtype=np.float64),
        np.zeros_like(np.asarray(prior_variances, dtype=np.float64), dtype=np.float64),
        np.asarray(projected_targets, dtype=np.float64),
        np.asarray(linear_predictor, dtype=np.float64),
        float(restricted_quadratic),
        0.0,
        0.0,
    )


def _solve_restricted_mean_only(
    genotype_matrix: StandardizedGenotypeMatrix,
    covariate_matrix: np.ndarray,
    targets: np.ndarray,
    prior_variances: np.ndarray,
    diagonal_noise: np.ndarray,
    solver_tolerance: float,
    maximum_linear_solver_iterations: int,
    exact_solver_matrix_limit: int,
    posterior_variance_batch_size: int,
    random_seed: int,
    initial_beta_guess: np.ndarray | None = None,
    sample_space_preconditioner_rank: int = 256,
    warm_start: _RestrictedPosteriorWarmStart | None = None,
    posterior_working_sets: bool = True,
    posterior_working_set_min_variants: int = 65_536,
    posterior_working_set_initial_size: int = 16_384,
    posterior_working_set_growth: int = 16_384,
    posterior_working_set_max_passes: int = 6,
    posterior_working_set_coefficient_tolerance: float = 1e-4,
    allow_working_set: bool = True,
    allow_gpu_exact_variant: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
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
    use_exact_variant = variant_count <= exact_solver_matrix_limit
    use_gpu_exact_variant = allow_gpu_exact_variant and _use_gpu_exact_variant_solve(
        genotype_matrix=genotype_matrix,
        variant_count=variant_count,
        exact_solver_matrix_limit=exact_solver_matrix_limit,
        covariate_count=covariate_matrix.shape[1],
    )
    use_exact_sample = _use_exact_sample_space_solve(
        sample_count=sample_count,
        variant_count=variant_count,
        exact_solver_matrix_limit=exact_solver_matrix_limit,
        use_gpu_exact_variant=use_gpu_exact_variant,
    )
    gpu_available = (
        genotype_matrix._cupy_cache is not None
        or _streaming_cupy_backend_available(genotype_matrix)
    )
    use_variant_space = (
        use_exact_variant
        or use_gpu_exact_variant
        or (not use_exact_sample and (not gpu_available) and variant_count <= sample_count)
    )

    if use_exact_sample:
        log(f"    restricted mean: exact sample-space Cholesky for n={sample_count}")
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
            prior_variances * np.asarray(genotype_matrix.transpose_matvec_numpy(projected_targets), dtype=np.float64),
            dtype=np.float64,
        )
        linear_predictor = covariate_matrix @ alpha + np.asarray(
            genotype_matrix.matvec_numpy(beta, batch_size=posterior_variance_batch_size),
            dtype=np.float64,
        )
        restricted_quadratic = float(np.dot(targets, projected_targets))
        log(
            "    restricted mean done: "
            + f"max|beta|={float(np.max(np.abs(beta))):.4f}  "
            + f"mean|beta|={float(np.mean(np.abs(beta))):.6f}  mem={mem()}"
        )
        return (
            np.asarray(alpha, dtype=np.float64),
            np.asarray(beta, dtype=np.float64),
            np.asarray(projected_targets, dtype=np.float64),
            np.asarray(linear_predictor, dtype=np.float64),
            restricted_quadratic,
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
    if _should_use_posterior_working_set(
        genotype_matrix=genotype_matrix,
        variant_count=variant_count,
        compute_logdet=False,
        compute_beta_variance=False,
        posterior_working_sets=posterior_working_sets,
        posterior_working_set_min_variants=posterior_working_set_min_variants,
        exact_solver_matrix_limit=exact_solver_matrix_limit,
        use_exact_variant=use_exact_variant,
        use_gpu_exact_variant=use_gpu_exact_variant,
        allow_working_set=allow_working_set,
    ):
        alpha, beta, _beta_variance, projected_targets, linear_predictor, restricted_quadratic, _logdet_covariance, _logdet_gls = (
            _restricted_posterior_state_posterior_working_set(
                genotype_matrix=genotype_matrix,
                covariate_matrix=covariate_matrix,
                targets=targets,
                prior_variances=prior_variances,
                prior_precision=prior_precision,
                inverse_diagonal_noise=inverse_diagonal_noise,
                covariate_precision_cholesky=covariate_precision_cholesky,
                covariate_precision_logdet=covariate_precision_logdet,
                apply_projector=apply_projector,
                solver_tolerance=solver_tolerance,
                maximum_linear_solver_iterations=maximum_linear_solver_iterations,
                exact_solver_matrix_limit=exact_solver_matrix_limit,
                posterior_variance_batch_size=posterior_variance_batch_size,
                random_seed=random_seed,
                sample_space_preconditioner_rank=sample_space_preconditioner_rank,
                initial_beta_guess=initial_beta_guess,
                posterior_working_set_initial_size=posterior_working_set_initial_size,
                posterior_working_set_growth=posterior_working_set_growth,
                posterior_working_set_max_passes=posterior_working_set_max_passes,
                posterior_working_set_coefficient_tolerance=posterior_working_set_coefficient_tolerance,
                warm_start=warm_start,
                allow_gpu_exact_variant=allow_gpu_exact_variant,
            )
        )
        return (
            np.asarray(alpha, dtype=np.float64),
            np.asarray(beta, dtype=np.float64),
            np.asarray(projected_targets, dtype=np.float64),
            np.asarray(linear_predictor, dtype=np.float64),
            float(restricted_quadratic),
        )

    if use_variant_space:
        if not (use_exact_variant or use_gpu_exact_variant):
            log(
                "    restricted mean: iterative variant-space solve "
                + f"(p={variant_count}, n={sample_count})  mem={mem()}"
            )
        if use_exact_variant or use_gpu_exact_variant:
            log(f"    restricted mean: exact variant-space Cholesky (p={variant_count}, n={sample_count})  mem={mem()}")
            if genotype_matrix._cupy_cache is not None:
                import cupy as cp

                cp_solve_triangular = _resolve_gpu_solve_triangular()
                covariate_count = covariate_matrix.shape[1]
                cache_is_int8_standardized = _cupy_cache_is_int8_standardized(genotype_matrix._cupy_cache)
                use_full_gpu_exact = _gpu_exact_variant_full_matrix_fits(
                    cp,
                    sample_count=sample_count,
                    variant_count=variant_count,
                    covariate_count=covariate_count,
                    cache_is_int8_standardized=cache_is_int8_standardized,
                )
                tiled_exact_batch_size = 0 if use_full_gpu_exact else _gpu_exact_variant_tile_size(
                    cp,
                    sample_count=sample_count,
                    variant_count=variant_count,
                    covariate_count=covariate_count,
                )
                if not use_full_gpu_exact and tiled_exact_batch_size <= 0:
                    raise RuntimeError(
                        "GPU exact variant-space solve was selected without enough tile workspace."
                    )
                projector_bundle_gpu = _build_restricted_projector_gpu_bundle(
                    inverse_diagonal_noise=inverse_diagonal_noise,
                    covariate_matrix=covariate_matrix,
                    covariate_precision_cholesky=covariate_precision_cholesky,
                    cp=cp,
                    dtype=cp.float64,
                )
                projected_targets_gpu = _apply_restricted_projector_gpu(
                    cp.asarray(targets, dtype=cp.float64),
                    projector_bundle_gpu,
                    cp=cp,
                    solve_triangular_gpu=cp_solve_triangular,
                )
                diagonal_index = cp.arange(variant_count)
                inverse_diagonal_noise_gpu = cp.asarray(inverse_diagonal_noise, dtype=cp.float64)
                if use_full_gpu_exact:
                    cached_standardized_gpu = _cupy_cache_standardized_columns(
                        genotype_matrix._cupy_cache,
                        slice(None),
                        cupy=cp,
                        dtype=cp.float64,
                    )
                    X_gpu_compute = cp.empty(cached_standardized_gpu.shape, dtype=cp.float64, order="F")
                    X_gpu_compute[...] = cp.asarray(cached_standardized_gpu, dtype=cp.float64)
                    _gram_t0 = time.monotonic()
                    log(f"    building X^T W X (p={variant_count}, n={sample_count}, float64 exact GEMM)...  mem={mem()}")
                    variant_precision_gpu = cp.zeros((variant_count, variant_count), dtype=cp.float64)
                    col_chunk = min(_GPU_EXACT_VARIANT_TILE_MAX_VARIANTS, variant_count)
                    n_col_chunks = (variant_count + col_chunk - 1) // col_chunk
                    for col_idx, col_start in enumerate(range(0, variant_count, col_chunk)):
                        col_end = min(col_start + col_chunk, variant_count)
                        weighted_chunk = inverse_diagonal_noise_gpu[:, None] * X_gpu_compute[:, col_start:col_end]
                        variant_precision_gpu[:, col_start:col_end] = X_gpu_compute.T @ weighted_chunk
                        del weighted_chunk
                        if (col_idx + 1) % max(n_col_chunks // 4, 1) == 0 or col_idx == n_col_chunks - 1:
                            log(f"      Gram matrix: {col_end:,}/{variant_count:,} cols ({100*col_end/variant_count:.0f}%)  {time.monotonic()-_gram_t0:.1f}s")
                    variant_precision_gpu = 0.5 * (variant_precision_gpu + variant_precision_gpu.T)
                    log(f"    X^T W X built in {time.monotonic()-_gram_t0:.1f}s  mem={mem()}")
                    variant_rhs_gpu = genotype_matrix.gpu_transpose_matmat(
                        projected_targets_gpu,
                        batch_size=variant_count,
                        cupy=cp,
                        dtype=cp.float64,
                    )
                    exact_gpu_matmul_batch_size = variant_count
                else:
                    _gram_t0 = time.monotonic()
                    log(
                        "    building X^T W X "
                        + f"(p={variant_count}, n={sample_count}, tiled exact GPU batches={tiled_exact_batch_size})...  mem={mem()}"
                    )
                    variant_precision_gpu = cp.zeros((variant_count, variant_count), dtype=cp.float64)
                    row_starts = range(0, variant_count, tiled_exact_batch_size)
                    n_row_tiles = (variant_count + tiled_exact_batch_size - 1) // tiled_exact_batch_size
                    for row_idx, row_start in enumerate(row_starts):
                        row_end = min(row_start + tiled_exact_batch_size, variant_count)
                        row_batch_gpu = _cupy_cache_standardized_columns(
                            genotype_matrix._cupy_cache,
                            slice(row_start, row_end),
                            cupy=cp,
                            dtype=cp.float64,
                        )
                        for col_start in range(row_start, variant_count, tiled_exact_batch_size):
                            col_end = min(col_start + tiled_exact_batch_size, variant_count)
                            col_batch_gpu = _cupy_cache_standardized_columns(
                                genotype_matrix._cupy_cache,
                                slice(col_start, col_end),
                                cupy=cp,
                                dtype=cp.float64,
                            )
                            col_batch_gpu *= inverse_diagonal_noise_gpu[:, None]
                            block_precision_gpu = row_batch_gpu.T @ col_batch_gpu
                            variant_precision_gpu[row_start:row_end, col_start:col_end] = block_precision_gpu
                            if row_start != col_start:
                                variant_precision_gpu[col_start:col_end, row_start:row_end] = block_precision_gpu.T
                            del col_batch_gpu, block_precision_gpu
                        del row_batch_gpu
                        if (row_idx + 1) % max(n_row_tiles // 4, 1) == 0 or row_idx == n_row_tiles - 1:
                            log(f"      Gram matrix: {row_end:,}/{variant_count:,} rows ({100*row_end/variant_count:.0f}%)  {time.monotonic()-_gram_t0:.1f}s")
                    log(f"    X^T W X built in {time.monotonic()-_gram_t0:.1f}s  mem={mem()}")
                    variant_rhs_gpu = genotype_matrix.gpu_transpose_matmat(
                        projected_targets_gpu,
                        batch_size=tiled_exact_batch_size,
                        cupy=cp,
                        dtype=cp.float64,
                    )
                    exact_gpu_matmul_batch_size = tiled_exact_batch_size
                if covariate_count > 0:
                    CtWX_gpu = _cached_weighted_covariate_projection(
                        genotype_matrix=genotype_matrix,
                        covariate_matrix=covariate_matrix,
                        inverse_diagonal_noise=inverse_diagonal_noise,
                        batch_size=posterior_variance_batch_size,
                        warm_start=warm_start,
                        return_gpu=True,
                        cupy=cp,
                    )
                    correction_coeff_gpu = _gpu_cholesky_solve(
                        CtWX_gpu.T,
                        projector_bundle_gpu[3],
                        cp_solve_triangular,
                    )
                    correction_gpu = cp.asarray(CtWX_gpu @ correction_coeff_gpu, dtype=cp.float64)
                    variant_precision_gpu -= correction_gpu
                variant_precision_gpu[diagonal_index, diagonal_index] += cp.asarray(prior_precision, dtype=cp.float64)
            else:
                dense_genotypes = np.asarray(genotype_matrix.materialize(batch_size=posterior_variance_batch_size), dtype=np.float64)
                projected_genotypes = apply_projector(dense_genotypes)
                XtPX = dense_genotypes.T @ projected_genotypes
                variant_rhs = dense_genotypes.T @ apply_projector(targets)
                variant_precision_matrix = np.diag(prior_precision) + XtPX
                variant_precision_matrix += np.eye(variant_count, dtype=np.float64) * 1e-8
                variant_precision_cholesky = np.linalg.cholesky(variant_precision_matrix)

                def solve_variant_rhs(right_hand_side: np.ndarray) -> np.ndarray:
                    return _cholesky_solve(variant_precision_cholesky, np.asarray(right_hand_side, dtype=np.float64))

                beta = np.asarray(solve_variant_rhs(variant_rhs), dtype=np.float64)
            if genotype_matrix._cupy_cache is not None:
                variant_precision_gpu[diagonal_index, diagonal_index] += 1e-8
                log(f"    Cholesky factorization ({variant_count}×{variant_count} float64)...  mem={mem()}")
                variant_precision_cholesky_gpu = cp.linalg.cholesky(variant_precision_gpu)
                log(f"    Cholesky done, solving...  mem={mem()}")

                def solve_variant_rhs_gpu(right_hand_side):
                    return _gpu_cholesky_solve(
                        right_hand_side,
                        variant_precision_cholesky_gpu,
                        cp_solve_triangular,
                    )

                beta_gpu = solve_variant_rhs_gpu(variant_rhs_gpu)
                if use_full_gpu_exact:
                    genetic_linear_predictor_gpu = X_gpu_compute @ cp.asarray(beta_gpu, dtype=cp.float64)
                else:
                    genetic_linear_predictor_gpu = genotype_matrix.gpu_matmat(
                        cp.asarray(beta_gpu, dtype=cp.float64),
                        batch_size=exact_gpu_matmul_batch_size,
                        cupy=cp,
                        dtype=cp.float64,
                    )
                genetic_linear_predictor = _cupy_array_to_numpy(genetic_linear_predictor_gpu, dtype=np.float64)
                beta = _cupy_array_to_numpy(beta_gpu, dtype=np.float64)
            else:
                genetic_linear_predictor = np.asarray(genotype_matrix.matvec_numpy(beta), dtype=np.float64)
        else:
            log(f"    restricted mean: PCG variant-space solve (p={variant_count}, n={sample_count})  mem={mem()}")
            _t0 = time.monotonic()
            variant_operator = _restricted_variant_space_operator(
                genotype_matrix=genotype_matrix,
                prior_precision=prior_precision,
                inverse_diagonal_noise=inverse_diagonal_noise,
                covariate_matrix=covariate_matrix,
                covariate_precision_cholesky=covariate_precision_cholesky,
                batch_size=posterior_variance_batch_size,
            )
            log(f"      operator setup: {time.monotonic() - _t0:.1f}s  mem={mem()}")
            _t0 = time.monotonic()
            variant_preconditioner = _restricted_variant_space_diagonal_preconditioner(
                genotype_matrix=genotype_matrix,
                covariate_matrix=covariate_matrix,
                inverse_diagonal_noise=inverse_diagonal_noise,
                covariate_precision_cholesky=covariate_precision_cholesky,
                prior_precision=prior_precision,
                batch_size=posterior_variance_batch_size,
                warm_start=warm_start,
            )
            log(f"      preconditioner: {time.monotonic() - _t0:.1f}s  mem={mem()}")
            _t0 = time.monotonic()
            restricted_targets = apply_projector_jax(targets)
            variant_rhs = np.asarray(
                genotype_matrix.transpose_matvec_numpy(
                    restricted_targets,
                    batch_size=posterior_variance_batch_size,
                ),
                dtype=np.float64,
            )
            log(f"      rhs: {time.monotonic() - _t0:.1f}s  mem={mem()}")
            beta = np.asarray(
                solve_spd_system(
                    variant_operator,
                    variant_rhs,
                    tolerance=solver_tolerance,
                    max_iterations=maximum_linear_solver_iterations,
                    initial_guess=initial_beta_guess,
                    preconditioner=variant_preconditioner,
                ),
                dtype=np.float64,
            )
            genetic_linear_predictor = np.asarray(
                genotype_matrix.matvec_numpy(beta, batch_size=posterior_variance_batch_size),
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
        log(
            "    restricted mean done: "
            + f"max|beta|={float(np.max(np.abs(beta))):.4f}  "
            + f"mean|beta|={float(np.mean(np.abs(beta))):.6f}  mem={mem()}"
        )
        return (
            np.asarray(alpha, dtype=np.float64),
            np.asarray(beta, dtype=np.float64),
            np.asarray(projected_targets, dtype=np.float64),
            np.asarray(linear_predictor, dtype=np.float64),
            restricted_quadratic,
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
            "    restricted mean: GPU block-CG sample-space solve "
            + f"(p={variant_count}, n={sample_count}, source={gpu_source})"
        )
        _t0 = time.monotonic()
        log("      building preconditioner...")
        sample_space_preconditioner_gpu, sample_space_preconditioner_cache_entry = _get_cached_sample_space_gpu_preconditioner(
            genotype_matrix=genotype_matrix,
            prior_variances=prior_variances,
            diagonal_noise=diagonal_noise,
            batch_size=posterior_variance_batch_size,
            rank=effective_sample_space_preconditioner_rank,
            random_seed=random_seed,
            warm_start=warm_start,
        )
        log(f"      preconditioner ready ({time.monotonic()-_t0:.1f}s)  mem={mem()}")

        def solve_rhs_iterative(
            right_hand_side: np.ndarray,
            *,
            initial_guess: np.ndarray | None = None,
        ) -> np.ndarray:
            _solve_t0 = time.monotonic()
            log(f"      GPU CG solve starting: rhs_cols={right_hand_side.shape[1] if right_hand_side.ndim > 1 else 1}")
            solve_result = _solve_sample_space_rhs_gpu(
                genotype_matrix=genotype_matrix,
                prior_variances=prior_variances,
                diagonal_noise=diagonal_noise,
                right_hand_side=right_hand_side,
                initial_guess=initial_guess,
                tolerance=solver_tolerance,
                max_iterations=maximum_linear_solver_iterations,
                preconditioner=sample_space_preconditioner_gpu,
                batch_size=posterior_variance_batch_size,
                return_iterations=True,
            )
            solved_rhs, iterations_used = _resolve_sample_space_solve_result(
                solve_result,
                fallback_iterations=maximum_linear_solver_iterations,
            )
            _update_sample_space_preconditioner_iterations(
                sample_space_preconditioner_cache_entry,
                iterations_used,
            )
            log(f"      GPU CG done: {iterations_used} iterations in {time.monotonic()-_solve_t0:.1f}s  mem={mem()}")
            return solved_rhs
    else:
        log(
            "    restricted mean: CPU block-PCG sample-space solve "
            + f"(p={variant_count}, n={sample_count}, preconditioner_rank={effective_sample_space_preconditioner_rank})"
        )
        sample_space_preconditioner, sample_space_preconditioner_cache_entry = _get_cached_sample_space_cpu_preconditioner(
            genotype_matrix=genotype_matrix,
            prior_variances=prior_variances,
            diagonal_noise=diagonal_noise,
            batch_size=posterior_variance_batch_size,
            rank=effective_sample_space_preconditioner_rank,
            random_seed=random_seed,
            warm_start=warm_start,
        )

        def solve_rhs_iterative(
            right_hand_side: np.ndarray,
            *,
            initial_guess: np.ndarray | None = None,
        ) -> np.ndarray:
            solve_result = _solve_sample_space_rhs_cpu(
                genotype_matrix=genotype_matrix,
                prior_variances=prior_variances,
                diagonal_noise=diagonal_noise,
                right_hand_side=right_hand_side,
                initial_guess=initial_guess,
                tolerance=solver_tolerance,
                max_iterations=maximum_linear_solver_iterations,
                preconditioner=sample_space_preconditioner,
                batch_size=posterior_variance_batch_size,
                return_iterations=True,
            )
            solved_rhs, iterations_used = _resolve_sample_space_solve_result(
                solve_result,
                fallback_iterations=maximum_linear_solver_iterations,
            )
            _update_sample_space_preconditioner_iterations(
                sample_space_preconditioner_cache_entry,
                iterations_used,
            )
            return np.asarray(solved_rhs, dtype=np.float64)

    required_rhs_matrix = np.concatenate([targets[:, None], covariate_matrix], axis=1)
    initial_sample_space_guess = None
    if warm_start is not None and warm_start.sample_space_inverse_covariance_rhs is not None:
        cached_guess = warm_start.sample_space_inverse_covariance_rhs
        if (
            warm_start.sample_space_inverse_covariance_rhs_matrix_token == _sample_space_rhs_matrix_token(genotype_matrix)
            and getattr(cached_guess, "shape", None) == required_rhs_matrix.shape
        ):
            initial_sample_space_guess = np.asarray(cached_guess, dtype=np.float64)

    inverse_covariance_rhs = solve_rhs_iterative(
        required_rhs_matrix,
        initial_guess=initial_sample_space_guess,
    )
    cupy_module = _try_import_cupy() if sample_space_gpu_enabled else None
    gpu_postprocess_enabled = (
        sample_space_gpu_enabled
        and cupy_module is not None
        and hasattr(cupy_module, "asarray")
    )
    if warm_start is not None:
        warm_start.sample_space_inverse_covariance_rhs = (
            inverse_covariance_rhs
            if gpu_postprocess_enabled
            else np.asarray(inverse_covariance_rhs, dtype=np.float64)
        )
        warm_start.sample_space_inverse_covariance_rhs_matrix_token = _sample_space_rhs_matrix_token(genotype_matrix)
    if gpu_postprocess_enabled:
        assert cupy_module is not None
        cp = cupy_module
        cp_solve_triangular = _resolve_gpu_solve_triangular()
        compute_cp_dtype = _cupy_compute_dtype(cp)
        inverse_covariance_rhs_gpu = cp.asarray(inverse_covariance_rhs, dtype=cp.float64)
        targets_gpu = cp.asarray(targets, dtype=cp.float64)
        covariate_matrix_gpu = cp.asarray(covariate_matrix, dtype=cp.float64)
        prior_variances_gpu = cp.asarray(prior_variances, dtype=cp.float64)
        inverse_covariance_targets_gpu = inverse_covariance_rhs_gpu[:, 0]
        inverse_covariance_covariates_gpu = inverse_covariance_rhs_gpu[:, 1 : 1 + covariate_matrix.shape[1]]
        gls_normal_matrix_gpu = covariate_matrix_gpu.T @ inverse_covariance_covariates_gpu
        diagonal_index = np.arange(covariate_matrix.shape[1])
        gls_normal_matrix_gpu[diagonal_index, diagonal_index] += 1e-8
        gls_cholesky_gpu = cp.linalg.cholesky(gls_normal_matrix_gpu)
        alpha_gpu = _gpu_cholesky_solve(
            covariate_matrix_gpu.T @ inverse_covariance_targets_gpu,
            gls_cholesky_gpu,
            cp_solve_triangular,
        )
        projected_targets_gpu = inverse_covariance_targets_gpu - inverse_covariance_covariates_gpu @ alpha_gpu
        beta_gpu = prior_variances_gpu * genotype_matrix.gpu_transpose_matmat(
            projected_targets_gpu,
            batch_size=posterior_variance_batch_size,
            cupy=cp,
            dtype=compute_cp_dtype,
        )
        linear_predictor_gpu = covariate_matrix_gpu @ alpha_gpu + genotype_matrix.gpu_matmat(
            beta_gpu,
            batch_size=posterior_variance_batch_size,
            cupy=cp,
            dtype=compute_cp_dtype,
        )
        restricted_quadratic = float(_cupy_array_to_numpy(cp.dot(targets_gpu, projected_targets_gpu), dtype=np.float64))
        alpha = _cupy_array_to_numpy(alpha_gpu, dtype=np.float64)
        beta = _cupy_array_to_numpy(beta_gpu, dtype=np.float64)
        projected_targets = _cupy_array_to_numpy(projected_targets_gpu, dtype=np.float64)
        linear_predictor = _cupy_array_to_numpy(linear_predictor_gpu, dtype=np.float64)
        log(
            "    restricted mean done: "
            + f"max|beta|={float(np.max(np.abs(beta))):.4f}  "
            + f"mean|beta|={float(np.mean(np.abs(beta))):.6f}  mem={mem()}"
        )
        return (
            np.asarray(alpha, dtype=np.float64),
            np.asarray(beta, dtype=np.float64),
            np.asarray(projected_targets, dtype=np.float64),
            np.asarray(linear_predictor, dtype=np.float64),
            restricted_quadratic,
        )

    inverse_covariance_targets = np.asarray(inverse_covariance_rhs[:, 0], dtype=np.float64)
    inverse_covariance_covariates = np.asarray(
        inverse_covariance_rhs[:, 1 : 1 + covariate_matrix.shape[1]],
        dtype=np.float64,
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
    _t0 = time.monotonic()
    log(f"    computing beta: X^T @ projected_targets ({variant_count:,} variants)...")
    beta = np.asarray(
        prior_variances * np.asarray(genotype_matrix.transpose_matvec_numpy(projected_targets), dtype=compute_np_dtype),
        dtype=compute_np_dtype,
    )
    log(f"    beta computed in {time.monotonic()-_t0:.1f}s  mem={mem()}")
    _t0 = time.monotonic()
    log(f"    computing linear predictor: X @ beta ({variant_count:,} variants)...")
    linear_predictor = covariate_matrix @ alpha + np.asarray(
        genotype_matrix.matvec_numpy(beta, batch_size=posterior_variance_batch_size),
        dtype=compute_np_dtype,
    )
    log(f"    linear predictor computed in {time.monotonic()-_t0:.1f}s  mem={mem()}")
    restricted_quadratic = float(np.dot(targets, projected_targets))
    log(
        "    restricted mean done: "
        + f"max|beta|={float(np.max(np.abs(beta))):.4f}  "
        + f"mean|beta|={float(np.mean(np.abs(beta))):.6f}  mem={mem()}"
    )
    return (
        np.asarray(alpha, dtype=np.float64),
        np.asarray(beta, dtype=np.float64),
        np.asarray(projected_targets, dtype=np.float64),
        np.asarray(linear_predictor, dtype=np.float64),
        restricted_quadratic,
    )


def _restricted_posterior_state_posterior_working_set(
    genotype_matrix: StandardizedGenotypeMatrix,
    covariate_matrix: np.ndarray,
    targets: np.ndarray,
    prior_variances: np.ndarray,
    prior_precision: np.ndarray,
    inverse_diagonal_noise: np.ndarray,
    covariate_precision_cholesky: np.ndarray,
    covariate_precision_logdet: float,
    apply_projector: Callable[[np.ndarray], np.ndarray],
    solver_tolerance: float,
    maximum_linear_solver_iterations: int,
    exact_solver_matrix_limit: int,
    posterior_variance_batch_size: int,
    random_seed: int,
    sample_space_preconditioner_rank: int,
    initial_beta_guess: np.ndarray | None,
    posterior_working_set_initial_size: int,
    posterior_working_set_growth: int,
    posterior_working_set_max_passes: int,
    posterior_working_set_coefficient_tolerance: float,
    warm_start: _RestrictedPosteriorWarmStart | None,
    allow_gpu_exact_variant: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    variant_count = genotype_matrix.shape[1]
    _reset_posterior_working_set_warm_start(warm_start, genotype_matrix, variant_count)
    current_beta = (
        np.zeros(variant_count, dtype=np.float64)
        if initial_beta_guess is None
        else np.asarray(initial_beta_guess, dtype=np.float64).copy()
    )
    if current_beta.shape != (variant_count,):
        raise ValueError("initial_beta_guess must match variant count for posterior working sets.")

    ever_active_indices = _ordered_unique_indices(
        [
            warm_start.posterior_working_set_ever_active if warm_start is not None else None,
            _active_working_set_indices(current_beta, posterior_working_set_coefficient_tolerance),
        ],
        variant_count,
    )
    current_screening_score = _posterior_working_set_seed_score(
        beta=current_beta,
        prior_variances=prior_variances,
        warm_start=warm_start,
        variant_count=variant_count,
    )
    working_size = _posterior_working_set_target_size(
        initial_size=posterior_working_set_initial_size,
        ever_active_count=ever_active_indices.shape[0],
        warm_start=warm_start,
        variant_count=variant_count,
    )
    log(
        "    seeding posterior working set without an upfront full gradient "
        + f"(size={working_size}/{variant_count}, ever_active={ever_active_indices.shape[0]})  mem={mem()}"
    )
    working_indices = _posterior_working_set_indices(
        current_screening_score,
        ever_active_indices,
        working_size,
    )

    for working_pass in range(max(int(posterior_working_set_max_passes), 1)):
        _ws_pass_t0 = time.monotonic()
        log(
            "    posterior working-set pass "
            + f"{working_pass + 1}/{int(posterior_working_set_max_passes)} "
            + f"size={working_indices.shape[0]}/{variant_count} "
            + f"ever_active={ever_active_indices.shape[0]}  mem={mem()}"
        )
        working_set_genotypes = genotype_matrix.subset(working_indices)
        # Upload working set to GPU for exact Cholesky solve (if it fits)
        working_set_genotypes.try_materialize_gpu()
        subset_alpha, subset_beta, _subset_projected_targets, subset_fitted, _subset_restricted_quadratic = _solve_restricted_mean_only(
            genotype_matrix=working_set_genotypes,
            covariate_matrix=covariate_matrix,
            targets=targets,
            prior_variances=prior_variances[working_indices],
            diagonal_noise=1.0 / np.maximum(inverse_diagonal_noise, 1e-12),
            solver_tolerance=solver_tolerance,
            maximum_linear_solver_iterations=maximum_linear_solver_iterations,
            exact_solver_matrix_limit=exact_solver_matrix_limit,
            posterior_variance_batch_size=posterior_variance_batch_size,
            random_seed=random_seed + working_pass,
            initial_beta_guess=current_beta[working_indices],
            sample_space_preconditioner_rank=sample_space_preconditioner_rank,
            posterior_working_sets=False,
            allow_working_set=False,
            allow_gpu_exact_variant=allow_gpu_exact_variant,
        )
        # Extract genetic prediction from subset result — avoids one full forward
        # matvec on all variants.  The subset solve returns linear_predictor =
        # covariates @ alpha + working_genotypes @ beta_working.  Since excluded
        # betas are zero, this equals the full genotype_matrix.matvec(candidate_beta).
        subset_alpha = np.asarray(subset_alpha, dtype=np.float64)
        subset_fitted = np.asarray(subset_fitted, dtype=np.float64)
        # Free GPU memory after extracting what we need
        working_set_genotypes._cupy_cache = None
        del working_set_genotypes
        candidate_beta = np.zeros(variant_count, dtype=np.float64)
        candidate_beta[working_indices] = np.asarray(subset_beta, dtype=np.float64)
        ever_active_indices = _ordered_unique_indices(
            [
                ever_active_indices,
                _active_working_set_indices(candidate_beta, posterior_working_set_coefficient_tolerance),
            ],
            variant_count,
        )
        genetic_linear_predictor = np.asarray(
            subset_fitted - covariate_matrix @ subset_alpha,
            dtype=np.float64,
        )
        residual_vector = np.asarray(targets - genetic_linear_predictor, dtype=np.float64)
        projected_targets = np.asarray(apply_projector(residual_vector), dtype=np.float64)
        if working_indices.shape[0] == variant_count:
            alpha = np.asarray(
                _cholesky_solve(
                    covariate_precision_cholesky,
                    covariate_matrix.T @ (inverse_diagonal_noise * (targets - genetic_linear_predictor)),
                ),
                dtype=np.float64,
            )
            linear_predictor = covariate_matrix @ alpha + genetic_linear_predictor
            restricted_quadratic = float(np.dot(targets, projected_targets))
            _update_posterior_working_set_warm_start(
                warm_start=warm_start,
                ever_active_indices=ever_active_indices,
                screening_score=current_screening_score,
                target_size=working_indices.shape[0],
                variant_count=variant_count,
            )
            _ws_pass_seconds = time.monotonic() - _ws_pass_t0
            log(
                "    working set spans all variants; accepting without KKT pass "
                + f"(pass_time={_ws_pass_seconds:.1f}s)"
            )
            return (
                np.asarray(alpha, dtype=np.float64),
                np.asarray(candidate_beta, dtype=np.float64),
                np.zeros(variant_count, dtype=np.float64),
                np.asarray(projected_targets, dtype=np.float64),
                np.asarray(linear_predictor, dtype=np.float64),
                restricted_quadratic,
                0.0,
                covariate_precision_logdet,
            )
        # The first global pass now doubles as KKT certification. That removes
        # the cold-start screening pass and lets later EM iterations reuse the
        # last certified score ordering instead of rebuilding it from scratch.
        _ws_kkt_t0 = time.monotonic()
        log(f"    KKT check: computing gradient on all {variant_count:,} variants...  mem={mem()}")
        candidate_gradient = np.asarray(
            genotype_matrix.transpose_matvec_numpy(
                projected_targets,
                batch_size=posterior_variance_batch_size,
            ),
            dtype=np.float64,
        ) - prior_precision * candidate_beta
        _ws_kkt_seconds = time.monotonic() - _ws_kkt_t0
        log(f"    KKT gradient computed in {_ws_kkt_seconds:.1f}s  mem={mem()}")
        candidate_score = _working_set_screening_score(candidate_gradient, candidate_beta, prior_variances)
        current_screening_score = candidate_score
        candidate_update_score = _working_set_posterior_update_score(candidate_gradient, prior_variances)
        excluded_mask = np.ones(variant_count, dtype=bool)
        excluded_mask[working_indices] = False
        max_excluded_update = (
            float(np.max(candidate_update_score[excluded_mask]))
            if np.any(excluded_mask)
            else 0.0
        )
        if max_excluded_update <= float(posterior_working_set_coefficient_tolerance):
            alpha = np.asarray(
                _cholesky_solve(
                    covariate_precision_cholesky,
                    covariate_matrix.T @ (inverse_diagonal_noise * (targets - genetic_linear_predictor)),
                ),
                dtype=np.float64,
            )
            linear_predictor = covariate_matrix @ alpha + genetic_linear_predictor
            restricted_quadratic = float(np.dot(targets, projected_targets))
            _ws_pass_seconds = time.monotonic() - _ws_pass_t0
            log(
                "    KKT CERTIFIED — working set accepted "
                + f"(size={working_indices.shape[0]}/{variant_count}, ever_active={ever_active_indices.shape[0]}, "
                + f"excluded_update={max_excluded_update:.2e}, pass_time={_ws_pass_seconds:.1f}s)"
            )
            _update_posterior_working_set_warm_start(
                warm_start=warm_start,
                ever_active_indices=ever_active_indices,
                screening_score=candidate_score,
                target_size=working_indices.shape[0],
                variant_count=variant_count,
            )
            return (
                np.asarray(alpha, dtype=np.float64),
                np.asarray(candidate_beta, dtype=np.float64),
                np.zeros(variant_count, dtype=np.float64),
                np.asarray(projected_targets, dtype=np.float64),
                np.asarray(linear_predictor, dtype=np.float64),
                restricted_quadratic,
                0.0,
                covariate_precision_logdet,
            )
        current_beta = candidate_beta
        all_violating = np.flatnonzero(
            excluded_mask & (candidate_update_score > float(posterior_working_set_coefficient_tolerance))
        ).astype(np.int32, copy=False)
        # Cap violations at growth budget: include only the TOP violating
        # indices ranked by KKT violation magnitude.  Without this cap, a cold
        # start (all betas zero → every variant violates KKT) expands the
        # working set from 8K to 222K in one step, triggering a catastrophic
        # fallback to CPU CG on all variants.
        growth_budget = int(posterior_working_set_growth)
        if all_violating.shape[0] > growth_budget:
            top_violation_order = np.argsort(candidate_update_score[all_violating])[-growth_budget:]
            violating_indices = all_violating[top_violation_order]
        else:
            violating_indices = all_violating
        next_size = min(
            max(
                working_indices.shape[0] + growth_budget,
                ever_active_indices.shape[0] + growth_budget,
            ),
            variant_count,
        )
        working_indices = _ordered_unique_indices(
            [
                ever_active_indices,
                violating_indices,
                _posterior_working_set_indices(candidate_score, ever_active_indices, next_size),
            ],
            variant_count,
        )
        log(
            "    working set expanded "
            + f"(violations={all_violating.shape[0]}, included={violating_indices.shape[0]}, "
            + f"next_size={working_indices.shape[0]}/{variant_count})"
        )

    log(
        "    working set exhausted max passes; retrying exact restricted solve "
        + f"(last_size={working_indices.shape[0]}/{variant_count}, "
        + f"max_excluded_update={max_excluded_update:.2e})"
    )
    exact_alpha, exact_beta, exact_projected_targets, exact_linear_predictor, exact_restricted_quadratic = _solve_restricted_mean_only(
        genotype_matrix=genotype_matrix,
        covariate_matrix=covariate_matrix,
        targets=targets,
        prior_variances=prior_variances,
        diagonal_noise=1.0 / np.maximum(inverse_diagonal_noise, 1e-12),
        solver_tolerance=solver_tolerance,
        maximum_linear_solver_iterations=maximum_linear_solver_iterations,
        exact_solver_matrix_limit=exact_solver_matrix_limit,
        posterior_variance_batch_size=posterior_variance_batch_size,
        random_seed=random_seed + max(int(posterior_working_set_max_passes), 1),
        initial_beta_guess=current_beta,
        sample_space_preconditioner_rank=sample_space_preconditioner_rank,
        warm_start=warm_start,
        posterior_working_sets=False,
        allow_working_set=False,
        allow_gpu_exact_variant=allow_gpu_exact_variant,
    )
    if warm_start is not None:
        _update_posterior_working_set_warm_start(
            warm_start=warm_start,
            ever_active_indices=_ordered_unique_indices(
                [
                    ever_active_indices,
                    _active_working_set_indices(
                        np.asarray(exact_beta, dtype=np.float64),
                        posterior_working_set_coefficient_tolerance,
                    ),
                ],
                variant_count,
            ),
            screening_score=current_screening_score,
            target_size=working_indices.shape[0],
            variant_count=variant_count,
        )
    return _restricted_posterior_result_without_diagnostics(
        alpha=exact_alpha,
        beta=exact_beta,
        prior_variances=prior_variances,
        projected_targets=exact_projected_targets,
        linear_predictor=exact_linear_predictor,
        restricted_quadratic=exact_restricted_quadratic,
    )


def _solve_restricted_full(
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
    warm_start: _RestrictedPosteriorWarmStart | None = None,
    posterior_working_sets: bool = True,
    posterior_working_set_min_variants: int = 65_536,
    posterior_working_set_initial_size: int = 16_384,
    posterior_working_set_growth: int = 16_384,
    posterior_working_set_max_passes: int = 6,
    posterior_working_set_coefficient_tolerance: float = 1e-4,
    allow_working_set: bool = True,
    allow_gpu_exact_variant: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    from sv_pgs.progress import log, mem
    if not compute_logdet and not compute_beta_variance:
        raise ValueError(
            "_solve_restricted_full requires compute_logdet or compute_beta_variance. "
            "Use _solve_restricted_mean_only for point-estimate solves."
        )
    compute_jax_dtype = gpu_compute_jax_dtype()
    compute_np_dtype = gpu_compute_numpy_dtype()
    sample_count = genotype_matrix.shape[0]
    diagonal_noise = np.asarray(diagonal_noise, dtype=np.float64)
    if diagonal_noise.shape != (sample_count,):
        raise ValueError("diagonal_noise must have one entry per sample.")

    prior_variances = np.maximum(np.asarray(prior_variances, dtype=np.float64), 1e-8)
    prior_precision = 1.0 / prior_variances
    variant_count = genotype_matrix.shape[1]
    use_exact_variant = variant_count <= exact_solver_matrix_limit
    use_gpu_exact_variant = allow_gpu_exact_variant and _use_gpu_exact_variant_solve(
        genotype_matrix=genotype_matrix,
        variant_count=variant_count,
        exact_solver_matrix_limit=exact_solver_matrix_limit,
        covariate_count=covariate_matrix.shape[1],
    )
    use_exact_sample = _use_exact_sample_space_solve(
        sample_count=sample_count,
        variant_count=variant_count,
        exact_solver_matrix_limit=exact_solver_matrix_limit,
        use_gpu_exact_variant=use_gpu_exact_variant,
    )
    # GPU availability: cached (uploaded to GPU memory) or streaming (mmap→GPU batches)
    gpu_available = (
        genotype_matrix._cupy_cache is not None
        or _streaming_cupy_backend_available(genotype_matrix)
    )

    # Solver hierarchy:
    # 1. Exact Cholesky in the smaller exact space.
    #    On GPU, exact variant-space always wins when it fits comfortably.
    # 2. GPU exact variant-space Cholesky when p fits in GPU VRAM (dynamic limit).
    #    Forms X^T W X via cuBLAS syrk + Cholesky. 10-20x faster than CG and
    #    gives exact solutions.
    # 3. CPU exact variant-space Cholesky when p is the smaller exact system
    # 4. GPU sample-space CG with Nyström preconditioner (for large p)
    # 5. CPU sample-space CG with Nyström
    # 6. Iterative variant-space CG — last resort
    use_variant_space = (
        use_exact_variant
        or use_gpu_exact_variant  # GPU exact Cholesky: fast + exact, always preferred
        or (not use_exact_sample and (not gpu_available) and variant_count <= sample_count)
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
            prior_variances * np.asarray(genotype_matrix.transpose_matvec_numpy(projected_targets), dtype=np.float64),
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
            genotype_matrix.matvec_numpy(beta, batch_size=posterior_variance_batch_size),
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
    if _should_use_posterior_working_set(
        genotype_matrix=genotype_matrix,
        variant_count=variant_count,
        compute_logdet=compute_logdet,
        compute_beta_variance=compute_beta_variance,
        posterior_working_sets=posterior_working_sets,
        posterior_working_set_min_variants=posterior_working_set_min_variants,
        exact_solver_matrix_limit=exact_solver_matrix_limit,
        use_exact_variant=use_exact_variant,
        use_gpu_exact_variant=use_gpu_exact_variant,
        allow_working_set=allow_working_set,
    ):
        return _restricted_posterior_state_posterior_working_set(
            genotype_matrix=genotype_matrix,
            covariate_matrix=covariate_matrix,
            targets=targets,
            prior_variances=prior_variances,
            prior_precision=prior_precision,
            inverse_diagonal_noise=inverse_diagonal_noise,
            covariate_precision_cholesky=covariate_precision_cholesky,
            covariate_precision_logdet=covariate_precision_logdet,
            apply_projector=apply_projector,
            solver_tolerance=solver_tolerance,
            maximum_linear_solver_iterations=maximum_linear_solver_iterations,
            exact_solver_matrix_limit=exact_solver_matrix_limit,
            posterior_variance_batch_size=posterior_variance_batch_size,
            random_seed=random_seed,
            sample_space_preconditioner_rank=sample_space_preconditioner_rank,
            initial_beta_guess=initial_beta_guess,
            posterior_working_set_initial_size=posterior_working_set_initial_size,
            posterior_working_set_growth=posterior_working_set_growth,
            posterior_working_set_max_passes=posterior_working_set_max_passes,
            posterior_working_set_coefficient_tolerance=posterior_working_set_coefficient_tolerance,
            warm_start=warm_start,
            allow_gpu_exact_variant=allow_gpu_exact_variant,
        )

    if use_variant_space:
        if not (use_exact_variant or use_gpu_exact_variant):
            log(
                "    restricted posterior: iterative variant-space solve "
                + f"(p={variant_count}, n={sample_count})  mem={mem()}"
            )
        if use_exact_variant or use_gpu_exact_variant:
            log(f"    restricted posterior: exact variant-space Cholesky (p={variant_count}, n={sample_count})  mem={mem()}")
            if genotype_matrix._cupy_cache is not None:
                import cupy as cp
                cp_solve_triangular = _resolve_gpu_solve_triangular()
                covariate_count = covariate_matrix.shape[1]
                cache_is_int8_standardized = _cupy_cache_is_int8_standardized(genotype_matrix._cupy_cache)
                use_full_gpu_exact = _gpu_exact_variant_full_matrix_fits(
                    cp,
                    sample_count=sample_count,
                    variant_count=variant_count,
                    covariate_count=covariate_count,
                    cache_is_int8_standardized=cache_is_int8_standardized,
                )
                tiled_exact_batch_size = 0 if use_full_gpu_exact else _gpu_exact_variant_tile_size(
                    cp,
                    sample_count=sample_count,
                    variant_count=variant_count,
                    covariate_count=covariate_count,
                )
                if not use_full_gpu_exact and tiled_exact_batch_size <= 0:
                    raise RuntimeError(
                        "GPU exact variant-space solve was selected without enough tile workspace."
                    )
                projector_bundle_gpu = _build_restricted_projector_gpu_bundle(
                    inverse_diagonal_noise=inverse_diagonal_noise,
                    covariate_matrix=covariate_matrix,
                    covariate_precision_cholesky=covariate_precision_cholesky,
                    cp=cp,
                    dtype=cp.float64,
                )
                projected_targets_gpu = _apply_restricted_projector_gpu(
                    cp.asarray(targets, dtype=cp.float64),
                    projector_bundle_gpu,
                    cp=cp,
                    solve_triangular_gpu=cp_solve_triangular,
                )
                diagonal_index = cp.arange(variant_count)
                inverse_diagonal_noise_gpu = cp.asarray(inverse_diagonal_noise, dtype=cp.float64)
                if use_full_gpu_exact:
                    cached_standardized_gpu = _cupy_cache_standardized_columns(
                        genotype_matrix._cupy_cache,
                        slice(None),
                        cupy=cp,
                        dtype=cp.float64,
                    )
                    X_gpu_compute = cp.empty(cached_standardized_gpu.shape, dtype=cp.float64, order="F")
                    X_gpu_compute[...] = cp.asarray(cached_standardized_gpu, dtype=cp.float64)
                    _gram_t0 = time.monotonic()
                    log(f"    building X^T W X (p={variant_count}, n={sample_count}, float64 exact GEMM)...  mem={mem()}")
                    variant_precision_gpu = cp.zeros((variant_count, variant_count), dtype=cp.float64)
                    col_chunk = min(_GPU_EXACT_VARIANT_TILE_MAX_VARIANTS, variant_count)
                    n_col_chunks = (variant_count + col_chunk - 1) // col_chunk
                    for col_idx, col_start in enumerate(range(0, variant_count, col_chunk)):
                        col_end = min(col_start + col_chunk, variant_count)
                        weighted_chunk = inverse_diagonal_noise_gpu[:, None] * X_gpu_compute[:, col_start:col_end]
                        variant_precision_gpu[:, col_start:col_end] = X_gpu_compute.T @ weighted_chunk
                        del weighted_chunk
                        if (col_idx + 1) % max(n_col_chunks // 4, 1) == 0 or col_idx == n_col_chunks - 1:
                            log(f"      Gram matrix: {col_end:,}/{variant_count:,} cols ({100*col_end/variant_count:.0f}%)  {time.monotonic()-_gram_t0:.1f}s")
                    variant_precision_gpu = 0.5 * (variant_precision_gpu + variant_precision_gpu.T)
                    log(f"    X^T W X built in {time.monotonic()-_gram_t0:.1f}s  mem={mem()}")
                    variant_rhs_gpu = genotype_matrix.gpu_transpose_matmat(
                        projected_targets_gpu,
                        batch_size=variant_count,
                        cupy=cp,
                        dtype=cp.float64,
                    )
                    exact_gpu_matmul_batch_size = variant_count
                else:
                    _gram_t0 = time.monotonic()
                    log(
                        "    building X^T W X "
                        + f"(p={variant_count}, n={sample_count}, tiled exact GPU batches={tiled_exact_batch_size})...  mem={mem()}"
                    )
                    variant_precision_gpu = cp.zeros((variant_count, variant_count), dtype=cp.float64)
                    row_starts = range(0, variant_count, tiled_exact_batch_size)
                    n_row_tiles = (variant_count + tiled_exact_batch_size - 1) // tiled_exact_batch_size
                    for row_idx, row_start in enumerate(row_starts):
                        row_end = min(row_start + tiled_exact_batch_size, variant_count)
                        row_batch_gpu = _cupy_cache_standardized_columns(
                            genotype_matrix._cupy_cache,
                            slice(row_start, row_end),
                            cupy=cp,
                            dtype=cp.float64,
                        )
                        for col_start in range(row_start, variant_count, tiled_exact_batch_size):
                            col_end = min(col_start + tiled_exact_batch_size, variant_count)
                            col_batch_gpu = _cupy_cache_standardized_columns(
                                genotype_matrix._cupy_cache,
                                slice(col_start, col_end),
                                cupy=cp,
                                dtype=cp.float64,
                            )
                            col_batch_gpu *= inverse_diagonal_noise_gpu[:, None]
                            block_precision_gpu = row_batch_gpu.T @ col_batch_gpu
                            variant_precision_gpu[row_start:row_end, col_start:col_end] = block_precision_gpu
                            if row_start != col_start:
                                variant_precision_gpu[col_start:col_end, row_start:row_end] = block_precision_gpu.T
                            del col_batch_gpu, block_precision_gpu
                        del row_batch_gpu
                        if (row_idx + 1) % max(n_row_tiles // 4, 1) == 0 or row_idx == n_row_tiles - 1:
                            log(f"      Gram matrix: {row_end:,}/{variant_count:,} rows ({100*row_end/variant_count:.0f}%)  {time.monotonic()-_gram_t0:.1f}s")
                    log(f"    X^T W X built in {time.monotonic()-_gram_t0:.1f}s  mem={mem()}")
                    variant_rhs_gpu = genotype_matrix.gpu_transpose_matmat(
                        projected_targets_gpu,
                        batch_size=tiled_exact_batch_size,
                        cupy=cp,
                        dtype=cp.float64,
                    )
                    exact_gpu_matmul_batch_size = tiled_exact_batch_size
                if covariate_count > 0:
                    CtWX_gpu = _cached_weighted_covariate_projection(
                        genotype_matrix=genotype_matrix,
                        covariate_matrix=covariate_matrix,
                        inverse_diagonal_noise=inverse_diagonal_noise,
                        batch_size=posterior_variance_batch_size,
                        warm_start=warm_start,
                        return_gpu=True,
                        cupy=cp,
                    )
                    correction_coeff_gpu = _gpu_cholesky_solve(
                        CtWX_gpu.T,
                        projector_bundle_gpu[3],
                        cp_solve_triangular,
                    )
                    correction_gpu = cp.asarray(CtWX_gpu @ correction_coeff_gpu, dtype=cp.float64)
                    variant_precision_gpu -= correction_gpu
                variant_precision_gpu[diagonal_index, diagonal_index] += cp.asarray(prior_precision, dtype=cp.float64)
            else:
                dense_genotypes = np.asarray(genotype_matrix.materialize(batch_size=posterior_variance_batch_size), dtype=np.float64)
                projected_genotypes = apply_projector(dense_genotypes)
                XtPX = dense_genotypes.T @ projected_genotypes
                variant_rhs = dense_genotypes.T @ apply_projector(targets)
                variant_precision_matrix = np.diag(prior_precision) + XtPX
                # Standard small regularization for numerical stability.
                variant_precision_matrix += np.eye(variant_count, dtype=np.float64) * 1e-8
                variant_precision_cholesky = np.linalg.cholesky(variant_precision_matrix)

                def solve_variant_rhs(right_hand_side: np.ndarray) -> np.ndarray:
                    return _cholesky_solve(variant_precision_cholesky, np.asarray(right_hand_side, dtype=np.float64))

                beta = np.asarray(solve_variant_rhs(variant_rhs), dtype=np.float64)
            if genotype_matrix._cupy_cache is not None:
                variant_precision_gpu[diagonal_index, diagonal_index] += 1e-8
                log(f"    Cholesky factorization ({variant_count}×{variant_count} float64)...  mem={mem()}")
                variant_precision_cholesky_gpu = cp.linalg.cholesky(variant_precision_gpu)
                log(f"    Cholesky done, solving...  mem={mem()}")

                def solve_variant_rhs_gpu(right_hand_side):
                    return _gpu_cholesky_solve(
                        right_hand_side,
                        variant_precision_cholesky_gpu,
                        cp_solve_triangular,
                    )

                def solve_variant_rhs(right_hand_side: np.ndarray) -> np.ndarray:
                    return _cupy_array_to_numpy(
                        solve_variant_rhs_gpu(right_hand_side),
                        dtype=np.float64,
                    )

                beta_gpu = solve_variant_rhs_gpu(variant_rhs_gpu)
            if genotype_matrix._cupy_cache is not None:
                beta_variance = (
                    np.maximum(
                        _cupy_array_to_numpy(
                            _gpu_exact_variant_inverse_diagonal(
                                variant_precision_cholesky_gpu,
                                solve_triangular_gpu=cp_solve_triangular,
                                cupy=cp,
                            ),
                            dtype=np.float64,
                        ),
                        1e-8,
                    )
                    if compute_beta_variance
                    else np.zeros(variant_count, dtype=np.float64)
                )
            else:
                beta_variance = (
                    np.maximum(np.diag(solve_variant_rhs(np.eye(variant_count, dtype=np.float64))), 1e-8)
                    if compute_beta_variance and variant_count <= exact_solver_matrix_limit
                    else (
                        _posterior_variance_low_rank_residual_diagonal(
                            solve_variant_rhs=solve_variant_rhs,
                            dimension=variant_count,
                            probe_count=posterior_variance_probe_count,
                            random_seed=random_seed,
                        )
                        if compute_beta_variance
                        else np.zeros(variant_count, dtype=np.float64)
                    )
                )
            if genotype_matrix._cupy_cache is not None:
                logdet_A = (
                    2.0
                    * float(
                        np.sum(
                            np.log(
                                np.maximum(
                                    _cupy_array_to_numpy(
                                        variant_precision_cholesky_gpu[
                                            diagonal_index,
                                            diagonal_index,
                                        ],
                                        dtype=np.float64,
                                    ),
                                    1e-12,
                                )
                            )
                        )
                    )
                    if compute_logdet
                    else 0.0
                )
                if use_full_gpu_exact:
                    genetic_linear_predictor_gpu = X_gpu_compute @ cp.asarray(beta_gpu, dtype=cp.float64)
                else:
                    genetic_linear_predictor_gpu = genotype_matrix.gpu_matmat(
                        cp.asarray(beta_gpu, dtype=cp.float64),
                        batch_size=exact_gpu_matmul_batch_size,
                        cupy=cp,
                        dtype=cp.float64,
                    )
                genetic_linear_predictor = _cupy_array_to_numpy(genetic_linear_predictor_gpu, dtype=np.float64)
                beta = _cupy_array_to_numpy(beta_gpu, dtype=np.float64)
            else:
                logdet_A = 2.0 * float(np.sum(np.log(np.diag(variant_precision_cholesky)))) if compute_logdet else 0.0
                genetic_linear_predictor = np.asarray(genotype_matrix.matvec_numpy(beta), dtype=np.float64)
        else:
            log(f"    restricted posterior: PCG variant-space solve (p={variant_count}, n={sample_count})  mem={mem()}")
            _t0 = time.monotonic()
            variant_operator = _restricted_variant_space_operator(
                genotype_matrix=genotype_matrix,
                prior_precision=prior_precision,
                inverse_diagonal_noise=inverse_diagonal_noise,
                covariate_matrix=covariate_matrix,
                covariate_precision_cholesky=covariate_precision_cholesky,
                batch_size=posterior_variance_batch_size,
            )
            log(f"      operator setup: {time.monotonic() - _t0:.1f}s  mem={mem()}")
            _t0 = time.monotonic()
            variant_preconditioner = _restricted_variant_space_diagonal_preconditioner(
                genotype_matrix=genotype_matrix,
                covariate_matrix=covariate_matrix,
                inverse_diagonal_noise=inverse_diagonal_noise,
                covariate_precision_cholesky=covariate_precision_cholesky,
                prior_precision=prior_precision,
                batch_size=posterior_variance_batch_size,
                warm_start=warm_start,
            )
            log(f"      preconditioner: {time.monotonic() - _t0:.1f}s  mem={mem()}")
            _t0 = time.monotonic()
            restricted_targets = apply_projector_jax(targets)
            variant_rhs = np.asarray(
                genotype_matrix.transpose_matvec_numpy(
                    restricted_targets,
                    batch_size=posterior_variance_batch_size,
                ),
                dtype=np.float64,
            )
            log(f"      rhs: {time.monotonic() - _t0:.1f}s  mem={mem()}")

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
                beta_variance = _posterior_variance_low_rank_residual_diagonal(
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
                    control_variate_diagonal=variant_preconditioner,
                )
                if compute_logdet
                else 0.0
            )
            genetic_linear_predictor = np.asarray(
                genotype_matrix.matvec_numpy(beta, batch_size=posterior_variance_batch_size),
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
        _t0 = time.monotonic()
        log("      building preconditioner...")
        sample_space_preconditioner_gpu, sample_space_preconditioner_cache_entry = _get_cached_sample_space_gpu_preconditioner(
            genotype_matrix=genotype_matrix,
            prior_variances=prior_variances,
            diagonal_noise=diagonal_noise,
            batch_size=posterior_variance_batch_size,
            rank=effective_sample_space_preconditioner_rank,
            random_seed=random_seed,
            warm_start=warm_start,
        )
        log(f"      preconditioner ready ({time.monotonic()-_t0:.1f}s)  mem={mem()}")

        def solve_rhs_iterative(
            right_hand_side: np.ndarray,
            *,
            initial_guess: np.ndarray | None = None,
            column_iteration_limits: np.ndarray | None = None,
            required_columns: np.ndarray | None = None,
            lanczos_recorder: _SampleSpaceCGLanczosRecorder | None = None,
            preconditioner_override=None,
        ) -> np.ndarray:
            _solve_t0 = time.monotonic()
            log(f"      GPU CG solve starting: rhs_cols={right_hand_side.shape[1] if right_hand_side.ndim > 1 else 1}")
            solve_result = _solve_sample_space_rhs_gpu(
                genotype_matrix=genotype_matrix,
                prior_variances=prior_variances,
                diagonal_noise=diagonal_noise,
                right_hand_side=right_hand_side,
                initial_guess=initial_guess,
                tolerance=solver_tolerance,
                max_iterations=maximum_linear_solver_iterations,
                preconditioner=sample_space_preconditioner_gpu if preconditioner_override is None else preconditioner_override,
                batch_size=posterior_variance_batch_size,
                return_iterations=True,
                column_iteration_limits=column_iteration_limits,
                required_columns=required_columns,
                lanczos_recorder=lanczos_recorder,
            )
            solved_rhs, iterations_used = _resolve_sample_space_solve_result(
                solve_result,
                fallback_iterations=maximum_linear_solver_iterations,
            )
            _update_sample_space_preconditioner_iterations(
                sample_space_preconditioner_cache_entry,
                iterations_used,
            )
            log(f"      GPU CG done: {iterations_used} iterations in {time.monotonic()-_solve_t0:.1f}s  mem={mem()}")
            return solved_rhs
    else:
        log(
            "    restricted posterior: CPU block-PCG sample-space solve "
            + f"(p={variant_count}, n={sample_count}, preconditioner_rank={effective_sample_space_preconditioner_rank})"
        )
        sample_space_preconditioner, sample_space_preconditioner_cache_entry = _get_cached_sample_space_cpu_preconditioner(
            genotype_matrix=genotype_matrix,
            prior_variances=prior_variances,
            diagonal_noise=diagonal_noise,
            batch_size=posterior_variance_batch_size,
            rank=effective_sample_space_preconditioner_rank,
            random_seed=random_seed,
            warm_start=warm_start,
        )

        def solve_rhs_iterative(
            right_hand_side: np.ndarray,
            *,
            initial_guess: np.ndarray | None = None,
            column_iteration_limits: np.ndarray | None = None,
            required_columns: np.ndarray | None = None,
            lanczos_recorder: _SampleSpaceCGLanczosRecorder | None = None,
            preconditioner_override=None,
        ) -> np.ndarray:
            solve_result = _solve_sample_space_rhs_cpu(
                genotype_matrix=genotype_matrix,
                prior_variances=prior_variances,
                diagonal_noise=diagonal_noise,
                right_hand_side=right_hand_side,
                initial_guess=initial_guess,
                tolerance=solver_tolerance,
                max_iterations=maximum_linear_solver_iterations,
                preconditioner=sample_space_preconditioner if preconditioner_override is None else preconditioner_override,
                batch_size=posterior_variance_batch_size,
                return_iterations=True,
                column_iteration_limits=column_iteration_limits,
                required_columns=required_columns,
                lanczos_recorder=lanczos_recorder,
            )
            solved_rhs, iterations_used = _resolve_sample_space_solve_result(
                solve_result,
                fallback_iterations=maximum_linear_solver_iterations,
            )
            _update_sample_space_preconditioner_iterations(
                sample_space_preconditioner_cache_entry,
                iterations_used,
            )
            return np.asarray(solved_rhs, dtype=np.float64)

    variance_probe_matrix, low_rank_variance_probes, stochastic_variance_probes = _sample_space_variance_probe_plan(
        sample_count=sample_count,
        probe_count=int(posterior_variance_probe_count) if compute_beta_variance else 0,
        random_seed=random_seed,
        sample_space_preconditioner_cache_entry=sample_space_preconditioner_cache_entry,
    )
    low_rank_variance_probe_count = 0 if low_rank_variance_probes is None else int(low_rank_variance_probes.shape[1])
    stochastic_variance_probe_count = (
        0
        if stochastic_variance_probes is None
        else int(stochastic_variance_probes.shape[1])
    )
    logdet_probe_block = (
        _orthogonal_probe_matrix(
            dimension=sample_count,
            probe_count=int(logdet_probe_count),
            random_seed=random_seed,
        )
        if compute_logdet and int(logdet_probe_count) > 0
        else None
    )
    diagonal_control_variate = np.asarray(sample_space_preconditioner_cache_entry.diagonal_preconditioner, dtype=np.float64)
    baseline_logdet = float(np.sum(np.log(np.maximum(diagonal_control_variate, 1e-12))))
    logdet_probe_rhs = (
        np.asarray(np.sqrt(np.maximum(diagonal_control_variate, 1e-12))[:, None] * logdet_probe_block, dtype=np.float64)
        if logdet_probe_block is not None
        else None
    )
    solve_rhs_blocks = [targets[:, None], covariate_matrix]
    if variance_probe_matrix is not None:
        solve_rhs_blocks.append(variance_probe_matrix)
    required_rhs_matrix = np.concatenate(solve_rhs_blocks, axis=1)
    solve_rhs_matrix = (
        np.concatenate([required_rhs_matrix, logdet_probe_rhs], axis=1)
        if logdet_probe_rhs is not None
        else required_rhs_matrix
    )
    initial_sample_space_guess = None
    if warm_start is not None and warm_start.sample_space_inverse_covariance_rhs is not None:
        cached_guess = warm_start.sample_space_inverse_covariance_rhs
        if (
            warm_start.sample_space_inverse_covariance_rhs_matrix_token == _sample_space_rhs_matrix_token(genotype_matrix)
            and getattr(cached_guess, "shape", None) == required_rhs_matrix.shape
        ):
            cached_required_guess = np.asarray(cached_guess, dtype=np.float64)
            if logdet_probe_rhs is None:
                initial_sample_space_guess = cached_required_guess
            else:
                initial_sample_space_guess = np.concatenate(
                    [cached_required_guess, np.zeros_like(logdet_probe_rhs, dtype=np.float64)],
                    axis=1,
                )
    covariate_rhs_stop = 1 + covariate_matrix.shape[1]
    required_rhs_count = required_rhs_matrix.shape[1]
    logdet_column_indices = (
        np.arange(required_rhs_count, solve_rhs_matrix.shape[1], dtype=np.int32)
        if logdet_probe_rhs is not None
        else np.empty(0, dtype=np.int32)
    )
    logdet_recorder = _build_sample_space_cg_lanczos_recorder(
        total_rhs_count=solve_rhs_matrix.shape[1],
        monitored_columns=logdet_column_indices,
        maximum_steps=logdet_lanczos_steps,
    )

    if logdet_probe_rhs is not None:
        if sample_space_gpu_enabled:
            cp_preconditioner = _try_import_cupy()
            assert cp_preconditioner is not None

            def _mixed_preconditioner(matrix):
                matrix_gpu = cp_preconditioner.asarray(matrix)
                matrix_was_vector = matrix_gpu.ndim == 1
                if matrix_was_vector:
                    matrix_gpu = matrix_gpu[:, None]
                if matrix_gpu.ndim != 2:
                    raise ValueError("sample-space preconditioner expects a vector or matrix right-hand side.")
                diagonal_gpu = cp_preconditioner.asarray(diagonal_control_variate, dtype=matrix_gpu.dtype)
                preconditioned_matrix = cp_preconditioner.zeros_like(matrix_gpu)
                if required_rhs_count > 0:
                    preconditioned_matrix[:, :required_rhs_count] = sample_space_preconditioner_gpu(
                        matrix_gpu[:, :required_rhs_count]
                    )
                if logdet_column_indices.size > 0:
                    preconditioned_matrix[:, logdet_column_indices] = (
                        matrix_gpu[:, logdet_column_indices] / diagonal_gpu[:, None]
                    )
                return preconditioned_matrix[:, 0] if matrix_was_vector else preconditioned_matrix
        else:
            def _mixed_preconditioner(matrix: np.ndarray) -> np.ndarray:
                matrix_array = np.asarray(matrix, dtype=np.float64)
                matrix_was_vector = matrix_array.ndim == 1
                if matrix_was_vector:
                    matrix_array = matrix_array[:, None]
                if matrix_array.ndim != 2:
                    raise ValueError("sample-space preconditioner expects a vector or matrix right-hand side.")
                preconditioned_matrix = np.zeros_like(matrix_array, dtype=np.float64)
                if required_rhs_count > 0:
                    preconditioned_matrix[:, :required_rhs_count] = np.asarray(
                        sample_space_preconditioner(matrix_array[:, :required_rhs_count]),
                        dtype=np.float64,
                    )
                if logdet_column_indices.size > 0:
                    preconditioned_matrix[:, logdet_column_indices] = (
                        matrix_array[:, logdet_column_indices]
                        / np.maximum(diagonal_control_variate[:, None], 1e-12)
                    )
                return preconditioned_matrix[:, 0] if matrix_was_vector else preconditioned_matrix
    else:
        _mixed_preconditioner = None

    inverse_covariance_rhs = solve_rhs_iterative(
        solve_rhs_matrix,
        initial_guess=initial_sample_space_guess,
        column_iteration_limits=(
            np.concatenate(
                [
                    np.full(required_rhs_count, int(maximum_linear_solver_iterations), dtype=np.int32),
                    np.full(logdet_column_indices.shape[0], int(logdet_lanczos_steps), dtype=np.int32),
                ]
            )
            if logdet_probe_rhs is not None
            else None
        ),
        required_columns=(
            np.concatenate(
                [
                    np.ones(required_rhs_count, dtype=bool),
                    np.zeros(logdet_column_indices.shape[0], dtype=bool),
                ]
            )
            if logdet_probe_rhs is not None
            else None
        ),
        lanczos_recorder=logdet_recorder,
        preconditioner_override=_mixed_preconditioner,
    )
    cupy_module = _try_import_cupy() if sample_space_gpu_enabled else None
    gpu_postprocess_enabled = (
        sample_space_gpu_enabled
        and cupy_module is not None
        and hasattr(cupy_module, "asarray")
    )
    if warm_start is not None:
        warm_start.sample_space_inverse_covariance_rhs = (
            inverse_covariance_rhs[:, :required_rhs_count]
            if gpu_postprocess_enabled
            else np.asarray(inverse_covariance_rhs[:, :required_rhs_count], dtype=np.float64)
        )
        warm_start.sample_space_inverse_covariance_rhs_matrix_token = _sample_space_rhs_matrix_token(genotype_matrix)
    if gpu_postprocess_enabled:
        assert cupy_module is not None
        cp = cupy_module
        cp_solve_triangular = _resolve_gpu_solve_triangular()

        compute_cp_dtype = _cupy_compute_dtype(cp)
        inverse_covariance_rhs_gpu = cp.asarray(inverse_covariance_rhs, dtype=cp.float64)
        targets_gpu = cp.asarray(targets, dtype=cp.float64)
        covariate_matrix_gpu = cp.asarray(covariate_matrix, dtype=cp.float64)
        prior_variances_gpu = cp.asarray(prior_variances, dtype=cp.float64)
        inverse_covariance_targets_gpu = inverse_covariance_rhs_gpu[:, 0]
        inverse_covariance_covariates_gpu = inverse_covariance_rhs_gpu[:, 1 : 1 + covariate_matrix.shape[1]]
        gls_normal_matrix_gpu = covariate_matrix_gpu.T @ inverse_covariance_covariates_gpu
        diagonal_index = np.arange(covariate_matrix.shape[1])
        gls_normal_matrix_gpu[diagonal_index, diagonal_index] += 1e-8
        gls_cholesky_gpu = cp.linalg.cholesky(gls_normal_matrix_gpu)
        alpha_gpu = _gpu_cholesky_solve(
            covariate_matrix_gpu.T @ inverse_covariance_targets_gpu,
            gls_cholesky_gpu,
            cp_solve_triangular,
        )
        projected_targets_gpu = inverse_covariance_targets_gpu - inverse_covariance_covariates_gpu @ alpha_gpu
        beta_gpu = prior_variances_gpu * genotype_matrix.gpu_transpose_matmat(
            projected_targets_gpu,
            batch_size=posterior_variance_batch_size,
            cupy=cp,
            dtype=compute_cp_dtype,
        )
        if compute_beta_variance:
            probe_rhs_start = covariate_rhs_stop
            low_rank_probe_rhs_stop = probe_rhs_start + low_rank_variance_probe_count
            probe_rhs_stop = low_rank_probe_rhs_stop + stochastic_variance_probe_count
            inverse_covariance_low_rank_probe_matrix_gpu = (
                cp.asarray(
                    inverse_covariance_rhs_gpu[:, probe_rhs_start:low_rank_probe_rhs_stop],
                    dtype=cp.float64,
                )
                if low_rank_variance_probe_count > 0
                else None
            )
            inverse_covariance_probe_matrix_gpu = (
                cp.asarray(
                    inverse_covariance_rhs_gpu[:, low_rank_probe_rhs_stop:probe_rhs_stop],
                    dtype=cp.float64,
                )
                if stochastic_variance_probe_count > 0
                else None
            )
            low_rank_probe_projection_matrix_gpu = (
                genotype_matrix.gpu_transpose_matmat(
                    cp.asarray(low_rank_variance_probes, dtype=compute_cp_dtype),
                    batch_size=posterior_variance_batch_size,
                    cupy=cp,
                    dtype=compute_cp_dtype,
                )
                if low_rank_variance_probe_count > 0 and low_rank_variance_probes is not None
                else None
            )
            probe_projection_matrix_gpu = (
                genotype_matrix.gpu_transpose_matmat(
                    cp.asarray(stochastic_variance_probes, dtype=compute_cp_dtype),
                    batch_size=posterior_variance_batch_size,
                    cupy=cp,
                    dtype=compute_cp_dtype,
                )
                if stochastic_variance_probe_count > 0 and stochastic_variance_probes is not None
                else None
            )
            leverage_diagonal_gpu = _stochastic_restricted_cross_leverage_diagonal_gpu(
                genotype_matrix=genotype_matrix,
                covariate_matrix_gpu=covariate_matrix_gpu,
                inverse_covariance_covariates_gpu=inverse_covariance_covariates_gpu,
                gls_cholesky_gpu=gls_cholesky_gpu,
                inverse_covariance_probe_matrix_gpu=inverse_covariance_probe_matrix_gpu,
                probe_projection_matrix_gpu=probe_projection_matrix_gpu,
                batch_size=posterior_variance_batch_size,
                cp=cp,
                solve_triangular_gpu=cp_solve_triangular,
                low_rank_probe_projection_matrix_gpu=low_rank_probe_projection_matrix_gpu,
                restricted_low_rank_probe_projection_matrix_gpu=(
                    genotype_matrix.gpu_transpose_matmat(
                        inverse_covariance_low_rank_probe_matrix_gpu
                        - inverse_covariance_covariates_gpu @ _gpu_cholesky_solve(
                            covariate_matrix_gpu.T @ inverse_covariance_low_rank_probe_matrix_gpu,
                            gls_cholesky_gpu,
                            cp_solve_triangular,
                        ),
                        batch_size=posterior_variance_batch_size,
                        cupy=cp,
                        dtype=compute_cp_dtype,
                    )
                    if inverse_covariance_low_rank_probe_matrix_gpu is not None
                    else None
                ),
            )
            beta_variance_gpu = cp.maximum(
                prior_variances_gpu - (prior_variances_gpu * prior_variances_gpu) * leverage_diagonal_gpu,
                cp.float64(1e-8),
            )
            beta_variance = _cupy_array_to_numpy(beta_variance_gpu, dtype=np.float64)
        else:
            beta_variance = np.zeros_like(prior_variances, dtype=np.float64)
        linear_predictor_gpu = covariate_matrix_gpu @ alpha_gpu + genotype_matrix.gpu_matmat(
            beta_gpu,
            batch_size=posterior_variance_batch_size,
            cupy=cp,
            dtype=compute_cp_dtype,
        )
        logdet_covariance = (
            _sample_space_logdet_from_cg_lanczos(
                logdet_recorder,
                dimension=sample_count,
                baseline_logdet=baseline_logdet,
            )
            if compute_logdet
            else 0.0
        )
        sign_gls = 1.0
        logdet_gls = (
            2.0
            * float(
                _cupy_array_to_numpy(
                    cp.sum(
                        cp.log(
                            cp.maximum(
                                gls_cholesky_gpu[diagonal_index, diagonal_index],
                                cp.float64(1e-12),
                            )
                        ),
                        dtype=cp.float64,
                    ),
                    dtype=np.float64,
                )
            )
            if compute_logdet
            else 0.0
        )
        restricted_quadratic = float(_cupy_array_to_numpy(cp.dot(targets_gpu, projected_targets_gpu), dtype=np.float64))
        alpha = _cupy_array_to_numpy(alpha_gpu, dtype=np.float64)
        beta = _cupy_array_to_numpy(beta_gpu, dtype=np.float64)
        projected_targets = _cupy_array_to_numpy(projected_targets_gpu, dtype=np.float64)
        linear_predictor = _cupy_array_to_numpy(linear_predictor_gpu, dtype=np.float64)
        if sign_gls <= 0.0:
            raise RuntimeError("Restricted GLS normal matrix is not positive definite.")
        log(f"    posterior done: max|beta|={float(np.max(np.abs(beta))):.4f}  mean|beta|={float(np.mean(np.abs(beta))):.6f}  logdet_cov={float(logdet_covariance):.4f}  mem={mem()}")
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
    inverse_covariance_targets = np.asarray(inverse_covariance_rhs[:, 0], dtype=np.float64)
    inverse_covariance_covariates = np.asarray(
        inverse_covariance_rhs[:, 1 : 1 + covariate_matrix.shape[1]],
        dtype=np.float64,
    )
    low_rank_probe_rhs_stop = covariate_rhs_stop + low_rank_variance_probe_count
    probe_rhs_stop = low_rank_probe_rhs_stop + stochastic_variance_probe_count
    inverse_covariance_low_rank_probe_matrix = (
        np.asarray(
            inverse_covariance_rhs[:, covariate_rhs_stop:low_rank_probe_rhs_stop],
            dtype=np.float64,
        )
        if low_rank_variance_probe_count > 0
        else None
    )
    inverse_covariance_probe_matrix = (
        np.asarray(
            inverse_covariance_rhs[:, low_rank_probe_rhs_stop:probe_rhs_stop],
            dtype=np.float64,
        )
        if stochastic_variance_probe_count > 0
        else None
    )
    logdet_covariance = (
        _sample_space_logdet_from_cg_lanczos(
            logdet_recorder,
            dimension=sample_count,
            baseline_logdet=baseline_logdet,
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
    if compute_beta_variance:
        low_rank_probe_projection_matrix = (
            np.asarray(
                genotype_matrix.transpose_matmat(
                    low_rank_variance_probes,
                    batch_size=posterior_variance_batch_size,
                ),
                dtype=np.float64,
            )
            if low_rank_variance_probe_count > 0 and low_rank_variance_probes is not None
            else None
        )
        probe_projection_matrix = (
            np.asarray(
                genotype_matrix.transpose_matmat(
                    stochastic_variance_probes,
                    batch_size=posterior_variance_batch_size,
                ),
                dtype=np.float64,
            )
            if stochastic_variance_probe_count > 0 and stochastic_variance_probes is not None
            else None
        )
        restricted_low_rank_probe_matrix = (
            np.asarray(
                inverse_covariance_low_rank_probe_matrix - inverse_covariance_covariates @ _cholesky_solve(
                    gls_cholesky,
                    covariate_matrix.T @ inverse_covariance_low_rank_probe_matrix,
                ),
                dtype=np.float64,
            )
            if inverse_covariance_low_rank_probe_matrix is not None
            else None
        )
        restricted_probe_matrix = (
            np.asarray(
                inverse_covariance_probe_matrix - inverse_covariance_covariates @ _cholesky_solve(
                    gls_cholesky,
                    covariate_matrix.T @ inverse_covariance_probe_matrix,
                ),
                dtype=np.float64,
            )
            if inverse_covariance_probe_matrix is not None
            else None
        )
        restricted_projection_blocks = [projected_targets[:, None]]
        if restricted_low_rank_probe_matrix is not None:
            restricted_projection_blocks.append(restricted_low_rank_probe_matrix)
        if restricted_probe_matrix is not None:
            restricted_projection_blocks.append(restricted_probe_matrix)
        combined_projection_matrix = np.asarray(
            genotype_matrix.transpose_matmat(
                np.column_stack(restricted_projection_blocks),
                batch_size=posterior_variance_batch_size,
            ),
            dtype=compute_np_dtype,
        )
        beta_projection = np.asarray(combined_projection_matrix[:, 0], dtype=compute_np_dtype)
        projection_cursor = 1
        restricted_low_rank_probe_projection_matrix = (
            np.asarray(
                combined_projection_matrix[:, projection_cursor : projection_cursor + low_rank_variance_probe_count],
                dtype=np.float64,
            )
            if low_rank_variance_probe_count > 0
            else None
        )
        projection_cursor += low_rank_variance_probe_count
        restricted_probe_projection_matrix = (
            np.asarray(
                combined_projection_matrix[:, projection_cursor : projection_cursor + stochastic_variance_probe_count],
                dtype=np.float64,
            )
            if stochastic_variance_probe_count > 0
            else None
        )
        leverage_mode = (
            f"deflated stochastic leverage diagonal ({low_rank_variance_probe_count} low-rank + {stochastic_variance_probe_count} residual probes)"
            if low_rank_variance_probe_count > 0
            else f"stochastic leverage diagonal ({stochastic_variance_probe_count} probes)"
        )
        log(f"    computing {leverage_mode} for beta variance ({variant_count} variants)...")
        beta = np.asarray(prior_variances * beta_projection, dtype=compute_np_dtype)
        leverage_diagonal = _stochastic_restricted_cross_leverage_diagonal(
            genotype_matrix=genotype_matrix,
            covariate_matrix=covariate_matrix,
            solve_rhs=solve_rhs_iterative,
            inverse_covariance_covariates=inverse_covariance_covariates,
            gls_cholesky=gls_cholesky,
            batch_size=posterior_variance_batch_size,
            probe_count=stochastic_variance_probe_count,
            random_seed=random_seed,
            sample_probes=stochastic_variance_probes,
            inverse_covariance_probe_matrix=inverse_covariance_probe_matrix,
            probe_projection_matrix=probe_projection_matrix,
            restricted_probe_projection_matrix=restricted_probe_projection_matrix,
            low_rank_probe_projection_matrix=low_rank_probe_projection_matrix,
            restricted_low_rank_probe_projection_matrix=restricted_low_rank_probe_projection_matrix,
        )
        beta_variance = np.maximum(prior_variances - (prior_variances * prior_variances) * leverage_diagonal, 1e-8)
    else:
        _t0 = time.monotonic()
        log(f"    computing beta: X^T @ projected_targets ({variant_count:,} variants)...")
        beta = np.asarray(
            prior_variances * np.asarray(genotype_matrix.transpose_matvec_numpy(projected_targets), dtype=compute_np_dtype),
            dtype=compute_np_dtype,
        )
        log(f"    beta computed in {time.monotonic()-_t0:.1f}s  mem={mem()}")
        beta_variance = np.zeros_like(prior_variances, dtype=np.float64)
    _t0 = time.monotonic()
    log(f"    computing linear predictor: X @ beta ({variant_count:,} variants)...")
    linear_predictor = covariate_matrix @ alpha + np.asarray(
        genotype_matrix.matvec_numpy(beta, batch_size=posterior_variance_batch_size),
        dtype=compute_np_dtype,
    )
    log(f"    linear predictor computed in {time.monotonic()-_t0:.1f}s  mem={mem()}")
    sign_gls, logdet_gls = np.linalg.slogdet(gls_normal_matrix)
    if sign_gls <= 0.0:
        raise RuntimeError("Restricted GLS normal matrix is not positive definite.")
    restricted_quadratic = float(np.dot(targets, projected_targets))
    log(f"    posterior done: max|beta|={float(np.max(np.abs(beta))):.4f}  mean|beta|={float(np.mean(np.abs(beta))):.6f}  logdet_cov={float(logdet_covariance):.4f}  mem={mem()}")
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
    probe_projection_matrix: np.ndarray | None = None,
    restricted_probe_projection_matrix: np.ndarray | None = None,
    low_rank_probe_projection_matrix: np.ndarray | None = None,
    restricted_low_rank_probe_projection_matrix: np.ndarray | None = None,
) -> np.ndarray:
    low_rank_diagonal = np.zeros(genotype_matrix.shape[1], dtype=np.float64)
    if low_rank_probe_projection_matrix is not None or restricted_low_rank_probe_projection_matrix is not None:
        if low_rank_probe_projection_matrix is None or restricted_low_rank_probe_projection_matrix is None:
            raise ValueError("low-rank leverage projections must be provided together.")
        low_rank_probe_projection_matrix = np.asarray(low_rank_probe_projection_matrix, dtype=np.float64)
        restricted_low_rank_probe_projection_matrix = np.asarray(
            restricted_low_rank_probe_projection_matrix,
            dtype=np.float64,
        )
        if low_rank_probe_projection_matrix.shape != restricted_low_rank_probe_projection_matrix.shape:
            raise ValueError("low-rank leverage projection matrices must have matching shapes.")
        low_rank_diagonal = np.sum(
            low_rank_probe_projection_matrix * restricted_low_rank_probe_projection_matrix,
            axis=1,
            dtype=np.float64,
        )
    if sample_probes is None:
        if probe_count <= 0:
            return np.maximum(low_rank_diagonal, 1e-8)
        sample_probes = _orthogonal_probe_matrix(
            dimension=genotype_matrix.shape[0],
            probe_count=probe_count,
            random_seed=random_seed,
        )
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
    if probe_projection_matrix is None:
        probe_projection_matrix = _cached_sample_probe_projection(
            genotype_matrix=genotype_matrix,
            sample_probes=sample_probes,
            batch_size=batch_size,
            probe_count=probe_count,
            random_seed=random_seed,
        )
    else:
        probe_projection_matrix = np.asarray(probe_projection_matrix, dtype=np.float64)
        if probe_projection_matrix.shape != (genotype_matrix.shape[1], probe_count):
            raise ValueError("probe_projection_matrix shape does not match variant_count/probe_count.")
    if restricted_probe_projection_matrix is None:
        restricted_probe_projection_matrix = np.asarray(
            genotype_matrix.transpose_matmat(restricted_probe_matrix, batch_size=batch_size),
            dtype=np.float64,
        )
    else:
        restricted_probe_projection_matrix = np.asarray(restricted_probe_projection_matrix, dtype=np.float64)
        if restricted_probe_projection_matrix.shape != (genotype_matrix.shape[1], probe_count):
            raise ValueError("restricted_probe_projection_matrix shape does not match variant_count/probe_count.")
    residual_diagonal = np.mean(probe_projection_matrix * restricted_probe_projection_matrix, axis=1)
    return np.maximum(low_rank_diagonal + residual_diagonal, 1e-8)


def _stochastic_restricted_cross_leverage_diagonal_gpu(
    genotype_matrix: StandardizedGenotypeMatrix,
    *,
    covariate_matrix_gpu,
    inverse_covariance_covariates_gpu,
    gls_cholesky_gpu,
    inverse_covariance_probe_matrix_gpu,
    probe_projection_matrix_gpu,
    batch_size: int,
    cp,
    solve_triangular_gpu,
    low_rank_probe_projection_matrix_gpu=None,
    restricted_low_rank_probe_projection_matrix_gpu=None,
) -> Any:
    leverage_diagonal_gpu = cp.zeros(genotype_matrix.shape[1], dtype=cp.float64)
    if low_rank_probe_projection_matrix_gpu is not None or restricted_low_rank_probe_projection_matrix_gpu is not None:
        if low_rank_probe_projection_matrix_gpu is None or restricted_low_rank_probe_projection_matrix_gpu is None:
            raise ValueError("GPU low-rank leverage projections must be provided together.")
        leverage_diagonal_gpu = cp.sum(
            cp.asarray(low_rank_probe_projection_matrix_gpu, dtype=cp.float64)
            * cp.asarray(restricted_low_rank_probe_projection_matrix_gpu, dtype=cp.float64),
            axis=1,
            dtype=cp.float64,
        )
    if inverse_covariance_probe_matrix_gpu is None or probe_projection_matrix_gpu is None:
        return cp.maximum(leverage_diagonal_gpu, cp.float64(1e-8))
    restricted_probe_matrix_gpu = inverse_covariance_probe_matrix_gpu - inverse_covariance_covariates_gpu @ _gpu_cholesky_solve(
        covariate_matrix_gpu.T @ inverse_covariance_probe_matrix_gpu,
        gls_cholesky_gpu,
        solve_triangular_gpu,
    )
    restricted_probe_projection_matrix_gpu = genotype_matrix.gpu_transpose_matmat(
        restricted_probe_matrix_gpu,
        batch_size=batch_size,
        cupy=cp,
        dtype=probe_projection_matrix_gpu.dtype,
    )
    residual_diagonal_gpu = cp.mean(
        probe_projection_matrix_gpu * restricted_probe_projection_matrix_gpu,
        axis=1,
        dtype=cp.float64,
    )
    return cp.maximum(leverage_diagonal_gpu + residual_diagonal_gpu, cp.float64(1e-8))


# Build the metadata design matrix for the prior scale model.
#
# The scale hypermodel is schema-driven:
#   - class effects
#   - pooled categorical / multi-membership effects
#   - nested categorical node effects
#   - smooth bases for continuous features
#   - class-varying interactions for factor and smooth terms
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
        for prior_class, prior_weight in zip(record.prior_class_members, record.prior_class_membership, strict=True):
            class_membership_matrix[record_index, class_lookup[prior_class]] = prior_weight

    annotation_tables = _prior_annotation_tables(records)
    class_membership_by_class = {
        variant_class: class_membership_matrix[:, class_index]
        for class_index, variant_class in enumerate(unique_classes)
    }
    feature_specs = _compile_prior_feature_specs(
        annotation_tables=annotation_tables,
        class_membership_by_class=class_membership_by_class,
    )
    feature_names = [_feature_name_from_spec(feature_spec) for feature_spec in feature_specs]
    design_matrix = _design_matrix_for_feature_specs(records=records, feature_specs=feature_specs)
    return PriorDesign(
        design_matrix=design_matrix,
        feature_names=feature_names,
        feature_specs=feature_specs,
        class_membership_matrix=class_membership_matrix,
        inverse_class_lookup=inverse_class_lookup,
    )


def _compile_prior_feature_specs(
    annotation_tables: _PriorAnnotationTables,
    class_membership_by_class: dict[VariantClass, np.ndarray],
) -> tuple[ScaleModelFeatureSpec, ...]:
    feature_specs: list[ScaleModelFeatureSpec] = []
    class_totals = {
        variant_class: float(np.sum(class_membership))
        for variant_class, class_membership in class_membership_by_class.items()
    }

    def append_if_nonzero(feature_spec: ScaleModelFeatureSpec) -> None:
        feature_column = _column_for_feature_spec(
            feature_spec=feature_spec,
            annotation_tables=annotation_tables,
            class_membership_by_class=class_membership_by_class,
        )
        if np.max(np.abs(_center_design_column(feature_column))) < 1e-10:
            return
        feature_specs.append(feature_spec)

    for variant_class in sorted(class_membership_by_class, key=lambda class_value: class_value.value):
        append_if_nonzero(ScaleModelFeatureSpec(kind="type_offset", variant_class=variant_class))

    for source_name in sorted(annotation_tables.factor_weights_by_source):
        level_weights_by_name = annotation_tables.factor_weights_by_source[source_name]
        for level_name in _factor_levels_to_encode(level_weights_by_name):
            append_if_nonzero(
                ScaleModelFeatureSpec(
                    kind="factor_level",
                    source_name=source_name,
                    level_name=level_name,
                )
            )
            for variant_class, class_membership in class_membership_by_class.items():
                if class_totals[variant_class] < 3.0 or np.max(class_membership) <= 0.0:
                    continue
                append_if_nonzero(
                    ScaleModelFeatureSpec(
                        kind="factor_interaction",
                        variant_class=variant_class,
                        source_name=source_name,
                        level_name=level_name,
                    )
                )

    for source_name in sorted(annotation_tables.nested_weights_by_source):
        for nested_depth in sorted(annotation_tables.nested_weights_by_source[source_name]):
            for level_name in sorted(annotation_tables.nested_weights_by_source[source_name][nested_depth]):
                append_if_nonzero(
                    ScaleModelFeatureSpec(
                        kind="nested_level",
                        source_name=source_name,
                        level_name=level_name,
                        nested_depth=nested_depth,
                    )
                )
                for variant_class, class_membership in class_membership_by_class.items():
                    if class_totals[variant_class] < 3.0 or np.max(class_membership) <= 0.0:
                        continue
                    append_if_nonzero(
                        ScaleModelFeatureSpec(
                            kind="nested_interaction",
                            variant_class=variant_class,
                            source_name=source_name,
                            level_name=level_name,
                            nested_depth=nested_depth,
                        )
                    )

    for source_name in sorted(annotation_tables.continuous_values_by_source):
        continuous_values = annotation_tables.continuous_values_by_source[source_name]
        for base_feature_spec in _continuous_spline_feature_specs(source_name, continuous_values):
            append_if_nonzero(base_feature_spec)
            for variant_class, class_membership in class_membership_by_class.items():
                if class_totals[variant_class] < 3.0 or np.max(class_membership) <= 0.0:
                    continue
                append_if_nonzero(
                    ScaleModelFeatureSpec(
                        kind="continuous_spline_interaction",
                        variant_class=variant_class,
                        source_name=base_feature_spec.source_name,
                        basis_index=base_feature_spec.basis_index,
                        basis_kind=base_feature_spec.basis_kind,
                        standardize_mean=base_feature_spec.standardize_mean,
                        standardize_scale=base_feature_spec.standardize_scale,
                        knot_values=base_feature_spec.knot_values,
                    )
                )

    return tuple(feature_specs)


def _design_matrix_for_feature_specs(
    records: Sequence[VariantRecord],
    feature_specs: Sequence[ScaleModelFeatureSpec],
) -> np.ndarray:
    if len(feature_specs) == 0:
        return np.zeros((len(records), 0), dtype=np.float64)
    class_membership_by_class = {
        variant_class: np.asarray(
            [
                _class_membership_weight(record, variant_class)
                for record in records
            ],
            dtype=np.float64,
        )
        for variant_class in VariantClass
    }
    annotation_tables = _prior_annotation_tables(records)
    design_columns = [
        _center_design_column(
            _column_for_feature_spec(
                feature_spec=feature_spec,
                annotation_tables=annotation_tables,
                class_membership_by_class=class_membership_by_class,
            )
        )
        for feature_spec in feature_specs
    ]
    return np.column_stack(design_columns).astype(np.float64)


def _column_for_feature_spec(
    feature_spec: ScaleModelFeatureSpec,
    annotation_tables: _PriorAnnotationTables,
    class_membership_by_class: dict[VariantClass, np.ndarray],
) -> np.ndarray:
    if feature_spec.kind == "type_offset":
        if feature_spec.variant_class is None:
            raise ValueError("Scale-model feature spec is missing variant_class.")
        return np.asarray(class_membership_by_class[feature_spec.variant_class], dtype=np.float64)

    if feature_spec.kind in {"factor_level", "factor_interaction"}:
        if feature_spec.source_name is None or feature_spec.level_name is None:
            raise ValueError("Factor scale-model feature spec is missing source_name or level_name.")
        if feature_spec.source_name not in annotation_tables.factor_weights_by_source:
            raise ValueError("Unknown factor scale-model feature: " + str(feature_spec.source_name))
        if feature_spec.level_name not in annotation_tables.factor_weights_by_source[feature_spec.source_name]:
            raise ValueError("Unknown factor scale-model level: " + str(feature_spec.level_name))
        feature_column = np.asarray(
            annotation_tables.factor_weights_by_source[feature_spec.source_name][feature_spec.level_name],
            dtype=np.float64,
        )
        if feature_spec.kind == "factor_level":
            return feature_column
        if feature_spec.variant_class is None:
            raise ValueError("Scale-model interaction feature spec is missing variant_class.")
        return feature_column * class_membership_by_class[feature_spec.variant_class]

    if feature_spec.kind in {"nested_level", "nested_interaction"}:
        if feature_spec.source_name is None or feature_spec.level_name is None or feature_spec.nested_depth is None:
            raise ValueError("Nested scale-model feature spec is missing source_name, level_name, or nested_depth.")
        if feature_spec.source_name not in annotation_tables.nested_weights_by_source:
            raise ValueError("Unknown nested scale-model feature: " + str(feature_spec.source_name))
        if feature_spec.nested_depth not in annotation_tables.nested_weights_by_source[feature_spec.source_name]:
            raise ValueError("Unknown nested depth for scale-model feature: " + str(feature_spec.nested_depth))
        nested_weights = annotation_tables.nested_weights_by_source[feature_spec.source_name][feature_spec.nested_depth]
        if feature_spec.level_name not in nested_weights:
            raise ValueError("Unknown nested scale-model level: " + str(feature_spec.level_name))
        feature_column = np.asarray(nested_weights[feature_spec.level_name], dtype=np.float64)
        if feature_spec.kind == "nested_level":
            return feature_column
        if feature_spec.variant_class is None:
            raise ValueError("Scale-model interaction feature spec is missing variant_class.")
        return feature_column * class_membership_by_class[feature_spec.variant_class]

    if feature_spec.kind in {"continuous_spline", "continuous_spline_interaction"}:
        if feature_spec.source_name is None:
            raise ValueError("Continuous scale-model feature spec is missing source_name.")
        if feature_spec.source_name not in annotation_tables.continuous_values_by_source:
            raise ValueError("Unknown continuous scale-model feature: " + str(feature_spec.source_name))
        feature_column = _continuous_spline_basis_column(
            raw_values=annotation_tables.continuous_values_by_source[feature_spec.source_name],
            feature_spec=feature_spec,
        )
        if feature_spec.kind == "continuous_spline":
            return feature_column
        if feature_spec.variant_class is None:
            raise ValueError("Scale-model interaction feature spec is missing variant_class.")
        return feature_column * class_membership_by_class[feature_spec.variant_class]

    raise ValueError("Unsupported scale-model feature kind: " + feature_spec.kind)


def _feature_name_from_spec(feature_spec: ScaleModelFeatureSpec) -> str:
    if feature_spec.kind == "type_offset":
        if feature_spec.variant_class is None:
            raise ValueError("Class effect scale-model feature spec is missing variant_class.")
        return "type_offset::" + feature_spec.variant_class.value
    if feature_spec.kind == "factor_level":
        return "factor_level::" + str(feature_spec.source_name) + "::" + str(feature_spec.level_name)
    if feature_spec.kind == "factor_interaction":
        return (
            "factor_interaction::"
            + str(feature_spec.source_name)
            + "::"
            + str(feature_spec.level_name)
            + "::"
            + str(feature_spec.variant_class.value if feature_spec.variant_class is not None else "")
        )
    if feature_spec.kind == "nested_level":
        return (
            "nested_level::"
            + str(feature_spec.source_name)
            + "::"
            + str(feature_spec.nested_depth)
            + "::"
            + str(feature_spec.level_name)
        )
    if feature_spec.kind == "nested_interaction":
        return (
            "nested_interaction::"
            + str(feature_spec.source_name)
            + "::"
            + str(feature_spec.nested_depth)
            + "::"
            + str(feature_spec.level_name)
            + "::"
            + str(feature_spec.variant_class.value if feature_spec.variant_class is not None else "")
        )
    if feature_spec.kind == "continuous_spline":
        return "continuous_spline::" + str(feature_spec.source_name) + "::basis_" + str(feature_spec.basis_index)
    if feature_spec.kind == "continuous_spline_interaction":
        return (
            "continuous_spline_interaction::"
            + str(feature_spec.source_name)
            + "::"
            + str(feature_spec.variant_class.value if feature_spec.variant_class is not None else "")
            + "::basis_"
            + str(feature_spec.basis_index)
        )
    raise ValueError("Unsupported scale-model feature kind: " + feature_spec.kind)


def _class_membership_weight(record: VariantRecord, variant_class: VariantClass) -> float:
    for member_class, member_weight in zip(record.prior_class_members, record.prior_class_membership, strict=True):
        if member_class == variant_class:
            return float(member_weight)
    return 0.0


def _prior_annotation_tables(records: Sequence[VariantRecord]) -> _PriorAnnotationTables:
    return _PriorAnnotationTables(
        continuous_values_by_source=_continuous_prior_annotation_values(records),
        factor_weights_by_source=_factor_prior_annotation_weights(records),
        nested_weights_by_source=_nested_prior_annotation_weights(records),
    )


def _continuous_prior_annotation_values(records: Sequence[VariantRecord]) -> dict[str, np.ndarray]:
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
    feature_values_by_name = {
        "log_length": np.log(np.maximum(np.asarray([record.length for record in records], dtype=np.float64), 1.0)),
        "logit_allele_frequency": np.log(allele_frequencies) - np.log1p(-allele_frequencies),
        "quality": np.nan_to_num(
            np.asarray([record.quality for record in records], dtype=np.float64),
            nan=1.0,
            posinf=1.0,
            neginf=0.0,
        ),
        "log_training_support": np.log1p(training_support),
    }
    for feature_name in sorted(
        {
            feature_name
            for record in records
            for feature_name in record.prior_continuous_features
        }
    ):
        feature_values_by_name[feature_name] = np.asarray(
            [
                record.prior_continuous_features.get(feature_name, 0.0)
                for record in records
            ],
            dtype=np.float64,
        )
    return feature_values_by_name


def _factor_prior_annotation_weights(records: Sequence[VariantRecord]) -> dict[str, dict[str, np.ndarray]]:
    factor_weights_by_source: dict[str, dict[str, np.ndarray]] = {}
    repeat_values = np.asarray([float(record.is_repeat) for record in records], dtype=np.float64)
    factor_weights_by_source["repeat_indicator"] = {
        "false": 1.0 - repeat_values,
        "true": repeat_values,
    }
    copy_number_values = np.asarray([float(record.is_copy_number) for record in records], dtype=np.float64)
    factor_weights_by_source["copy_number_indicator"] = {
        "false": 1.0 - copy_number_values,
        "true": copy_number_values,
    }

    minor_allele_frequency = _minor_allele_frequency_values(records)
    factor_weights_by_source["maf_bucket"] = {
        "common": np.asarray(minor_allele_frequency >= 5e-2, dtype=np.float64),
        "low_frequency": np.asarray((minor_allele_frequency >= 1e-2) & (minor_allele_frequency < 5e-2), dtype=np.float64),
        "rare": np.asarray((minor_allele_frequency >= 1e-3) & (minor_allele_frequency < 1e-2), dtype=np.float64),
        "ultra_rare": np.asarray(minor_allele_frequency < 1e-3, dtype=np.float64),
    }

    for feature_name in sorted(
        {
            binary_feature_name
            for record in records
            for binary_feature_name in record.prior_binary_features
        }
    ):
        true_values = np.asarray(
            [float(record.prior_binary_features.get(feature_name, False)) for record in records],
            dtype=np.float64,
        )
        factor_weights_by_source[feature_name] = {
            "false": 1.0 - true_values,
            "true": true_values,
        }

    for feature_name in sorted(
        {
            categorical_feature_name
            for record in records
            for categorical_feature_name in record.prior_categorical_features
        }
    ):
        levels = sorted(
            {
                record.prior_categorical_features[feature_name]
                for record in records
                if feature_name in record.prior_categorical_features
            }
        )
        factor_weights_by_source[feature_name] = {
            level_name: np.asarray(
                [
                    float(record.prior_categorical_features.get(feature_name) == level_name)
                    for record in records
                ],
                dtype=np.float64,
            )
            for level_name in levels
        }

    for feature_name in sorted(
        {
            membership_feature_name
            for record in records
            for membership_feature_name in record.prior_membership_features
        }
    ):
        levels = sorted(
            {
                level_name
                for record in records
                for level_name in record.prior_membership_features.get(feature_name, {})
            }
        )
        factor_weights_by_source[feature_name] = {
            level_name: np.asarray(
                [
                    record.prior_membership_features.get(feature_name, {}).get(level_name, 0.0)
                    for record in records
                ],
                dtype=np.float64,
            )
            for level_name in levels
        }
    return factor_weights_by_source


def _nested_prior_annotation_weights(records: Sequence[VariantRecord]) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    nested_weights_by_source: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    nested_feature_names = sorted(
        {
            feature_name
            for record in records
            for feature_name in record.prior_nested_features
        }
        | {
            feature_name
            for record in records
            for feature_name in record.prior_nested_membership_features
        }
    )
    for feature_name in nested_feature_names:
        source_nested_weights: dict[int, dict[str, np.ndarray]] = {}
        for record_index, record in enumerate(records):
            path_weights: dict[str, float] = {}
            if feature_name in record.prior_nested_features:
                path_weights[NESTED_PATH_DELIMITER.join(record.prior_nested_features[feature_name])] = 1.0
            for path_name, path_weight in record.prior_nested_membership_features.get(feature_name, {}).items():
                path_weights[path_name] = path_weights.get(path_name, 0.0) + float(path_weight)
            for path_name, path_weight in path_weights.items():
                if path_weight <= 0.0:
                    continue
                path_parts = tuple(path_name.split(NESTED_PATH_DELIMITER))
                for nested_depth in range(len(path_parts)):
                    nested_level_name = NESTED_PATH_DELIMITER.join(path_parts[: nested_depth + 1])
                    source_nested_weights.setdefault(nested_depth, {}).setdefault(
                        nested_level_name,
                        np.zeros(len(records), dtype=np.float64),
                    )[record_index] += path_weight
        nested_weights_by_source[feature_name] = source_nested_weights
    return nested_weights_by_source


def _factor_levels_to_encode(level_weights_by_name: dict[str, np.ndarray]) -> tuple[str, ...]:
    if len(level_weights_by_name) <= 1:
        return ()
    reference_level = max(
        sorted(level_weights_by_name),
        key=lambda level_name: float(np.sum(level_weights_by_name[level_name])),
    )
    return tuple(level_name for level_name in sorted(level_weights_by_name) if level_name != reference_level)


def _continuous_spline_feature_specs(
    source_name: str,
    raw_values: np.ndarray,
) -> tuple[ScaleModelFeatureSpec, ...]:
    mean_value = float(np.mean(raw_values))
    scale_value = float(np.std(raw_values))
    if scale_value < 1e-8:
        return ()
    standardized_values = (np.asarray(raw_values, dtype=np.float64) - mean_value) / scale_value
    feature_specs = [
        ScaleModelFeatureSpec(
            kind="continuous_spline",
            source_name=source_name,
            basis_index=0,
            basis_kind="linear",
            standardize_mean=mean_value,
            standardize_scale=scale_value,
        )
    ]
    for basis_index, knot_value in enumerate(_continuous_spline_knots(standardized_values), start=1):
        feature_specs.append(
            ScaleModelFeatureSpec(
                kind="continuous_spline",
                source_name=source_name,
                basis_index=basis_index,
                basis_kind="cubic_hinge",
                standardize_mean=mean_value,
                standardize_scale=scale_value,
                knot_values=(float(knot_value),),
            )
        )
    return tuple(feature_specs)


def _continuous_spline_knots(standardized_values: np.ndarray) -> tuple[float, ...]:
    if np.unique(np.round(standardized_values, 12)).shape[0] < 3:
        return ()
    candidate_knots = np.quantile(standardized_values, [0.25, 0.5, 0.75])
    minimum_value = float(np.min(standardized_values))
    maximum_value = float(np.max(standardized_values))
    knot_values: list[float] = []
    for knot_value in candidate_knots:
        knot_float = float(knot_value)
        if knot_float <= minimum_value + 1e-6 or knot_float >= maximum_value - 1e-6:
            continue
        if knot_values and abs(knot_values[-1] - knot_float) < 1e-6:
            continue
        knot_values.append(knot_float)
    return tuple(knot_values)


def _continuous_spline_basis_column(
    raw_values: np.ndarray,
    feature_spec: ScaleModelFeatureSpec,
) -> np.ndarray:
    if feature_spec.standardize_mean is None or feature_spec.standardize_scale is None:
        raise ValueError("Continuous spline feature spec is missing standardization parameters.")
    standardized_values = (
        np.asarray(raw_values, dtype=np.float64) - float(feature_spec.standardize_mean)
    ) / float(feature_spec.standardize_scale)
    if feature_spec.basis_kind == "linear":
        return standardized_values
    if feature_spec.basis_kind == "cubic_hinge":
        if len(feature_spec.knot_values) != 1:
            raise ValueError("Cubic hinge spline feature spec must store exactly one knot.")
        return np.maximum(standardized_values - float(feature_spec.knot_values[0]), 0.0) ** 3
    raise ValueError("Unsupported continuous spline basis kind: " + str(feature_spec.basis_kind))


def _minor_allele_frequency_values(records: Sequence[VariantRecord]) -> np.ndarray:
    allele_frequencies = np.clip(
        np.nan_to_num(
            np.asarray([record.allele_frequency for record in records], dtype=np.float64),
            nan=0.5,
            posinf=1.0,
            neginf=0.0,
        ),
        0.0,
        1.0,
    )
    return np.minimum(allele_frequencies, 1.0 - allele_frequencies)


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
    predictor_offset: np.ndarray | None = None,
) -> float:
    offset = (
        np.zeros(len(targets), dtype=np.float64)
        if predictor_offset is None
        else np.asarray(predictor_offset, dtype=np.float64).reshape(-1)
    )
    if isinstance(genotype_matrix, StandardizedGenotypeMatrix):
        genotype_component = np.asarray(genotype_matrix.matvec_numpy(beta), dtype=np.float64)
    else:
        genotype_component = np.asarray(genotype_matrix @ beta, dtype=np.float64)
    linear_predictor = offset + genotype_component + covariate_matrix @ alpha
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

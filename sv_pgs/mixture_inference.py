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

from dataclasses import dataclass
from typing import Callable, Sequence

import sv_pgs._jax  # noqa: F401
import jax.numpy as jnp
from jax.scipy.special import digamma as jax_digamma
from jax.scipy.special import gammaln as jax_gammaln
import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import kve as scipy_bessel_kve

from sv_pgs._jax import gpu_compute_jax_dtype, gpu_compute_numpy_dtype
from sv_pgs.config import ModelConfig, TraitType, VariantClass
from sv_pgs.data import TieMap, VariantRecord
from sv_pgs.genotype import DenseRawGenotypeMatrix, StandardizedGenotypeMatrix
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
    class_membership_matrix: np.ndarray # which variant class(es) each variant belongs to
    inverse_class_lookup: dict[int, VariantClass]  # column index -> VariantClass enum


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


def fit_variational_em(
    genotypes: StandardizedGenotypeMatrix | np.ndarray,
    covariates: np.ndarray,
    targets: np.ndarray,
    records: Sequence[VariantRecord],
    config: ModelConfig,
    tie_map: TieMap,
    validation_data: tuple[StandardizedGenotypeMatrix | np.ndarray, np.ndarray, np.ndarray] | None = None,
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
    prior_design = _build_prior_design(reduced_records)
    scale_penalty = _scale_model_penalty(prior_design.feature_names, config)

    global_scale, scale_model_coefficients = _initialize_scale_model(prior_design, config)
    tpb_shape_a_vector = _initialize_tpb_shape_a_vector(prior_design, config)
    tpb_shape_b_vector = _initialize_tpb_shape_b_vector(prior_design, config)
    local_shape_a = prior_design.class_membership_matrix @ tpb_shape_a_vector
    local_shape_b = prior_design.class_membership_matrix @ tpb_shape_b_vector

    local_scale = np.ones(len(reduced_records), dtype=np.float64)
    auxiliary_delta = np.asarray(local_shape_b, dtype=np.float64)
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
    best_local_scale: np.ndarray | None = None
    best_theta: np.ndarray | None = None
    best_sigma_error2: float | None = None
    best_tpb_shape_a_vector: np.ndarray | None = None
    best_tpb_shape_b_vector: np.ndarray | None = None

    log(f"  variational EM: {genotype_matrix.shape[1]} reduced variants, {covariate_matrix.shape[1]} covariates, {target_vector.shape[0]} samples, max_iter={config.max_outer_iterations}")

    for outer_iteration in range(config.max_outer_iterations):
        log(f"  variational EM iteration {outer_iteration + 1}/{config.max_outer_iterations} start  sigma_e2={sigma_error2:.6f}  global_scale={global_scale:.6f}  mem={mem()}")
        posterior_theta = _pack_theta(
            global_scale=float(global_scale),
            scale_model_coefficients=scale_model_coefficients,
        )
        posterior_local_scale = local_scale.copy()
        posterior_tpb_shape_a_vector = tpb_shape_a_vector.copy()
        posterior_tpb_shape_b_vector = tpb_shape_b_vector.copy()
        # Build each variant's prior variance from three pieces:
        #   1. global_scale — one number shared by all variants
        #   2. metadata_baseline — per-variant, based on type/length/repeat
        #   3. local_scale (lambda) — per-variant adaptive shrinkage
        # A variant with large local_scale is "allowed" to have a big effect.
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
        )
        alpha_state = posterior_state.alpha
        beta_state = posterior_state.beta
        sigma_error2 = posterior_state.sigma_error2

        # How "big" is each variant's effect?  We need both the point estimate
        # (beta^2) and the uncertainty (variance) to properly update the scales.
        reduced_second_moment = np.asarray(beta_state * beta_state + posterior_state.beta_variance, dtype=np.float64)
        # Overall model quality = data fit + how well the shrinkage priors fit
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
        # Update the auxiliary rate parameter delta for each variant.
        # delta controls how aggressively the prior pulls lambda (and thus
        # beta) toward zero.  Variants with large local_scale get a smaller
        # delta — the model "loosens" the leash on variants that look real.
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
    )
    if config.trait_type == TraitType.BINARY and config.binary_intercept_calibration:
        calibrated_alpha = final_state.alpha.copy()
        calibrated_alpha[0] += np.float64(
            _calibrate_binary_intercept(
                linear_predictor=final_state.linear_predictor,
                targets=target_vector,
            )
        )
        final_state = PosteriorState(
            alpha=calibrated_alpha,
            beta=final_state.beta,
            beta_variance=final_state.beta_variance,
            linear_predictor=np.asarray(genotype_matrix.matvec(final_state.beta), dtype=np.float64) + covariate_matrix @ calibrated_alpha,
            collapsed_objective=final_state.collapsed_objective,
            sigma_error2=final_state.sigma_error2,
        )

    log(f"  final posterior computed  obj={final_state.collapsed_objective:.6f}  sigma_e2={final_state.sigma_error2:.4f}  mem={mem()}")
    final_member_prior_variances = _member_prior_variances_from_reduced_state(
        member_records=member_records,
        tie_map=tie_map,
        scale_model_coefficients=scale_model_coefficients,
        scale_model_feature_names=prior_design.feature_names,
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
            compute_logdet=True,
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
        + logdet_covariance
        + logdet_gls
        + (sample_count - covariate_matrix.shape[1]) * np.log(2.0 * np.pi)
    )
    return alpha, beta, beta_variance, linear_predictor, collapsed_objective, sigma_error2_new


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    standardized_genotypes = _as_standardized_genotype_matrix(genotype_matrix)
    compute_np_dtype = gpu_compute_numpy_dtype()
    compute_jax_dtype = gpu_compute_jax_dtype()
    prior_precision = (1.0 / np.maximum(prior_variances, 1e-8)).astype(compute_np_dtype, copy=False)
    covariate_count = covariate_matrix.shape[1]
    parameters = np.concatenate([alpha_init, beta_init], axis=0).astype(compute_np_dtype, copy=True)
    damping = float(initial_damping)

    # Pre-convert the covariate matrix to JAX once so that matmuls stay on GPU
    # throughout the Newton loop.  The covariate matrix is small
    # (n_samples x ~5-10 covariates) so this is cheap.
    covariate_matrix_jax = jnp.asarray(covariate_matrix, dtype=compute_jax_dtype)
    targets_jax = jnp.asarray(targets, dtype=compute_jax_dtype)
    prior_precision_jax = jnp.asarray(prior_precision, dtype=compute_jax_dtype)

    penalized_term_calls = 0

    def penalized_terms(current_parameters: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
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
            covariate_matrix_jax @ alpha_jax
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
        return penalized_log_posterior, gradient, weights, linear_predictor

    import time as _time
    stalled_objective_relative_tolerance = 1e-12
    stalled_step_tolerance = max(1e-3, gradient_tolerance * 100.0)
    log(f"      Newton trust-region: {standardized_genotypes.shape[1]} variants, max_iter={max_iterations}, damping={damping:.2e}  mem={mem()}")
    for _iteration_index in range(max_iterations):
        _newton_t0 = _time.monotonic()
        current_objective, gradient, weights, linear_predictor = penalized_terms(parameters)
        _t_penalized = _time.monotonic() - _newton_t0
        gradient_norm = float(np.linalg.norm(gradient))
        if gradient_norm <= gradient_tolerance:
            log(f"      Newton converged at iter {_iteration_index+1}: grad_norm={gradient_norm:.2e} <= tol={gradient_tolerance:.2e}")
            break
        working_response = linear_predictor + (targets.astype(compute_np_dtype, copy=False) - np.asarray(stable_sigmoid(jnp.asarray(linear_predictor, dtype=compute_jax_dtype)), dtype=compute_np_dtype)) / weights
        _t_solve_start = _time.monotonic()
        proposed_alpha, proposed_beta, _, _projected_targets, _fitted_response, _restricted_quadratic, _logdet_covariance, _logdet_gls = (
            _restricted_posterior_state(
                    genotype_matrix=standardized_genotypes,
                covariate_matrix=covariate_matrix,
                targets=working_response,
                prior_variances=prior_variances,
                diagonal_noise=1.0 / weights,
                solver_tolerance=solver_tolerance,
                maximum_linear_solver_iterations=maximum_linear_solver_iterations,
                logdet_probe_count=logdet_probe_count,
                logdet_lanczos_steps=logdet_lanczos_steps,
                exact_solver_matrix_limit=exact_solver_matrix_limit,
                posterior_variance_batch_size=posterior_variance_batch_size,
                posterior_variance_probe_count=posterior_variance_probe_count,
                random_seed=random_seed + _iteration_index,
                compute_logdet=False,
                compute_beta_variance=False,
                initial_beta_guess=parameters[covariate_count:],
            )
        )
        proposed_parameters = np.concatenate([proposed_alpha, proposed_beta], axis=0)
        newton_direction = proposed_parameters - parameters
        step_scale = 1.0 / (1.0 + damping)
        candidate_parameters = parameters + step_scale * newton_direction
        _t_solve = _time.monotonic() - _t_solve_start
        candidate_objective, _candidate_gradient, _candidate_weights, _candidate_linear_predictor = penalized_terms(candidate_parameters)
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
                and relative_step_size <= stalled_step_tolerance
            ):
                log(
                    f"      Newton converged at iter {_iteration_index+1}: stalled gain={actual_gain:.2e} "
                    + f"with step={relative_step_size:.2e}"
                )
                break
            damping *= damping_increase_factor
            log(f"      Newton iter {_iteration_index+1}/{max_iterations}: REJECT  obj={current_objective:.4f}  cand_obj={candidate_objective:.4f}  gain={actual_gain:.2e}  ratio={gain_ratio:.3f}  grad={gradient_norm:.2e}  step={relative_step_size:.2e}  damping={damping:.2e}  [penalized={_t_penalized:.1f}s solve={_t_solve:.1f}s total={_t_total:.1f}s]  mem={mem()}")

    final_objective, _final_gradient, final_weights, linear_predictor = penalized_terms(parameters)
    target_values = targets.astype(compute_np_dtype, copy=False)
    final_working_response = linear_predictor + (target_values - np.asarray(stable_sigmoid(linear_predictor), dtype=compute_np_dtype)) / final_weights
    working_alpha, working_beta, _working_variance, _working_projected_targets, _working_fitted_response, _working_quadratic, _working_logdet_covariance, _working_logdet_gls = (
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
            random_seed=random_seed + 2 * max_iterations,
            compute_logdet=False,
            compute_beta_variance=False,
            initial_beta_guess=parameters[covariate_count:],
        )
    )
    working_parameters = np.concatenate([working_alpha, working_beta], axis=0)
    working_objective, _working_gradient, _working_weights, _working_linear_predictor = penalized_terms(working_parameters)
    if working_objective >= final_objective:
        parameters = working_parameters
        final_objective = working_objective
        final_weights = _working_weights
        linear_predictor = _working_linear_predictor

    final_working_response = linear_predictor + (target_values - np.asarray(stable_sigmoid(linear_predictor), dtype=compute_np_dtype)) / final_weights
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
            compute_logdet=True,
            initial_beta_guess=parameters[covariate_count:],
        )
    )
    laplace_weights = np.asarray(final_weights, dtype=compute_np_dtype)
    final_parameters = np.concatenate([final_alpha, final_beta], axis=0)
    final_objective, _final_gradient, _final_penalty_weights, linear_predictor = penalized_terms(final_parameters)
    logdet_hessian = (
        float(np.sum(np.log(np.maximum(prior_precision, 1e-12))))
        + float(np.sum(np.log(np.maximum(laplace_weights, 1e-12))))
        + logdet_covariance
        + logdet_gls
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
):
    compute_dtype = gpu_compute_jax_dtype()
    diag_noise_jax = jnp.asarray(diagonal_noise, dtype=compute_dtype)
    prior_var_jax = jnp.asarray(prior_variances, dtype=compute_dtype)

    def matvec(vector) -> jnp.ndarray:
        v = jnp.asarray(vector, dtype=compute_dtype)
        projected = genotype_matrix.transpose_matvec(v)
        return diag_noise_jax * v + genotype_matrix.matvec(prior_var_jax * projected)

    def matmat(matrix) -> jnp.ndarray:
        matrix_jax = jnp.asarray(matrix, dtype=compute_dtype)
        # X^T @ M gives (p, k), scale by prior variance, then X @ result gives (n, k)
        projected = genotype_matrix.transpose_matmat(matrix_jax)  # (p, k)
        scaled = prior_var_jax[:, None] * projected  # (p, k)
        genotype_term = genotype_matrix.matmat(scaled)
        return diag_noise_jax[:, None] * matrix_jax + genotype_term

    return build_linear_operator(
        shape=(genotype_matrix.shape[0], genotype_matrix.shape[0]),
        matvec=matvec,
        matmat=matmat,
        dtype=compute_dtype,
    )


def _sample_space_diagonal_preconditioner(
    genotype_matrix: StandardizedGenotypeMatrix,
    prior_variances: np.ndarray,
    diagonal_noise: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    if genotype_matrix._cupy_cache is not None:
        import cupy as cp
        X = genotype_matrix._cupy_cache  # (n, p) float32 on GPU
        pv = cp.asarray(prior_variances, dtype=cp.float32)
        # diag(X @ diag(pv) @ X^T) = sum(X^2 * pv, axis=1)
        # Chunked to avoid allocating a full (n, p) intermediate on GPU.
        diag_gpu = cp.zeros(X.shape[0], dtype=cp.float32)
        chunk = max(1, min(512, X.shape[1]))
        for start in range(0, X.shape[1], chunk):
            end = min(start + chunk, X.shape[1])
            diag_gpu += cp.sum(X[:, start:end] ** 2 * pv[start:end], axis=1)
        result = np.asarray(diagonal_noise, dtype=np.float64) + diag_gpu.get().astype(np.float64)
        return np.maximum(result, 1e-8)
    diagonal = np.asarray(diagonal_noise, dtype=np.float64).copy()
    for batch in genotype_matrix.iter_column_batches(batch_size=batch_size):
        genotype_batch = np.asarray(batch.values, dtype=np.float64)
        diagonal += np.sum(
            genotype_batch * genotype_batch * prior_variances[batch.variant_indices][None, :],
            axis=1,
        )
    return np.maximum(diagonal, 1e-8)


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
    apply_projector: Callable[[np.ndarray], np.ndarray],
    batch_size: int,
):
    compute_dtype = gpu_compute_jax_dtype()
    compute_np_dtype = gpu_compute_numpy_dtype()
    prior_precision_jax = jnp.asarray(prior_precision, dtype=compute_dtype)

    def matvec(vector) -> jnp.ndarray:
        coefficients = jnp.asarray(vector, dtype=compute_dtype)
        genotype_projection = np.asarray(
            genotype_matrix.matvec(coefficients, batch_size=batch_size),
            dtype=compute_np_dtype,
        )
        restricted_projection = apply_projector(genotype_projection)
        return prior_precision_jax * coefficients + genotype_matrix.transpose_matvec(
            restricted_projection,
            batch_size=batch_size,
        )

    def matmat(matrix) -> jnp.ndarray:
        coefficients = jnp.asarray(matrix, dtype=compute_dtype)
        genotype_projection = np.asarray(
            genotype_matrix.matmat(coefficients, batch_size=batch_size),
            dtype=compute_np_dtype,
        )
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
    )


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
        inv_d = cp.asarray(inverse_diagonal_noise, dtype=cp.float32)
        cov_gpu = cp.asarray(covariate_matrix, dtype=cp.float32)  # (n, k) — tiny
        # Compute diag(X^T D^{-1} X) and X^T D^{-1} C entirely on GPU.
        # Only download small results: (p,) diagonal and (p, k) cross-term.
        raw_diag = cp.zeros(X.shape[1], dtype=cp.float32)
        weighted_Xt_gpu = cp.empty((X.shape[1], cov_gpu.shape[1]), dtype=cp.float32)
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
    random_generator = np.random.default_rng(random_seed)
    probes = random_generator.choice((-1.0, 1.0), size=(dimension, probe_count)).astype(np.float64)
    solutions = np.asarray(solve_variant_rhs(probes), dtype=np.float64)
    return np.maximum(np.mean(probes * solutions, axis=1), 1e-8)


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    from sv_pgs.progress import log, mem
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
    use_variant_space = (not use_exact_sample) and (use_exact_variant or use_gpu_exact_variant or variant_count <= sample_count)

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

        inverse_covariance_targets = solve_rhs(targets)
        inverse_covariance_covariates = solve_rhs(covariate_matrix)
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

    if use_variant_space:
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
                inv_d_cp = cp.asarray(inverse_diagonal_noise, dtype=cp.float32)
                cov_cp = cp.asarray(covariate_matrix, dtype=cp.float32)
                chunk = 256
                xtdxc_gpu = cp.empty((variant_count, cov_cp.shape[1]), dtype=cp.float32)
                xtdx_gpu = cp.empty((variant_count, variant_count), dtype=cp.float32)
                for start in range(0, variant_count, chunk):
                    end = min(start + chunk, variant_count)
                    weighted_chunk = inv_d_cp[:, None] * X_gpu[:, start:end]
                    xtdxc_gpu[start:end, :] = weighted_chunk.T @ cov_cp
                    xtdx_gpu[:, start:end] = X_gpu.T @ weighted_chunk
                CtWX = xtdxc_gpu.T.get().astype(np.float64)  # (k, p)
                correction_coeff = _cholesky_solve(covariate_precision_cholesky, CtWX)
                projected_targets_np = apply_projector(targets)
                variant_rhs_cp = cp.asarray(
                    np.asarray(genotype_matrix.transpose_matvec(projected_targets_np), dtype=np.float32),
                    dtype=cp.float32,
                )
                if use_gpu_exact_variant:
                    correction_gpu = xtdxc_gpu @ cp.asarray(correction_coeff, dtype=cp.float32)
                    variant_precision_gpu = xtdx_gpu - correction_gpu
                    variant_precision_gpu += cp.diag(cp.asarray(prior_precision, dtype=cp.float32))
                    variant_precision_gpu = (variant_precision_gpu + variant_precision_gpu.T) * 0.5
                    variant_precision_gpu += cp.eye(variant_count, dtype=cp.float32) * 1e-4
                    variant_precision_cholesky_gpu = cp.linalg.cholesky(variant_precision_gpu)

                    def solve_variant_rhs(right_hand_side: np.ndarray) -> np.ndarray:
                        rhs_cp = cp.asarray(np.asarray(right_hand_side, dtype=np.float32), dtype=cp.float32)
                        y = cp_solve_triangular(variant_precision_cholesky_gpu, rhs_cp, lower=True)
                        x = cp_solve_triangular(variant_precision_cholesky_gpu.T, y, lower=False)
                        return np.asarray(x.get(), dtype=np.float64)

                    beta = np.asarray(solve_variant_rhs(variant_rhs_cp.get()), dtype=np.float64)
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
                    genetic_linear_predictor = np.asarray(genotype_matrix.matvec(beta), dtype=np.float64)
                    del correction_gpu, xtdxc_gpu, xtdx_gpu, variant_precision_gpu, variant_precision_cholesky_gpu, variant_rhs_cp
                else:
                    correction_cpu = CtWX.T @ correction_coeff
                    XtPX = np.asarray(xtdx_gpu.get(), dtype=np.float64) - correction_cpu
                    variant_rhs = np.asarray(variant_rhs_cp.get(), dtype=np.float64)
                    del xtdxc_gpu, xtdx_gpu, variant_rhs_cp
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
                genetic_linear_predictor = np.asarray(genotype_matrix.matvec(beta), dtype=np.float64)
        else:
            import time as _time
            log(f"    restricted posterior: PCG variant-space solve (p={variant_count}, n={sample_count})  mem={mem()}")
            _t0 = _time.monotonic()
            variant_operator = _restricted_variant_space_operator(
                genotype_matrix=genotype_matrix,
                prior_precision=prior_precision,
                apply_projector=apply_projector,
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
            variant_rhs = np.asarray(
                genotype_matrix.transpose_matvec(
                    apply_projector(targets),
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
            beta_variance = (
                _posterior_variance_hutchinson_diagonal(
                    solve_variant_rhs=solve_variant_rhs,
                    dimension=variant_count,
                    probe_count=posterior_variance_probe_count,
                    random_seed=random_seed,
                )
                if compute_beta_variance
                else np.zeros(variant_count, dtype=compute_np_dtype)
            )
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

    log(f"    restricted posterior: PCG sample-space solve (p={variant_count}, n={sample_count})")
    covariance_operator = _sample_space_operator(genotype_matrix, prior_variances, diagonal_noise)
    sample_space_preconditioner = _sample_space_diagonal_preconditioner(
        genotype_matrix=genotype_matrix,
        prior_variances=prior_variances,
        diagonal_noise=diagonal_noise,
        batch_size=posterior_variance_batch_size,
    )

    def solve_rhs(right_hand_side: np.ndarray) -> np.ndarray:
        return solve_spd_system(
            covariance_operator,
            right_hand_side,
            tolerance=solver_tolerance,
            max_iterations=maximum_linear_solver_iterations,
            preconditioner=sample_space_preconditioner,
        )

    inverse_covariance_targets = solve_rhs(targets)
    inverse_covariance_covariates = solve_rhs(covariate_matrix)
    logdet_covariance = (
        stochastic_logdet(
            covariance_operator,
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

    log_length = np.log(np.maximum(np.asarray([record.length for record in records], dtype=np.float64), 1.0))
    repeat_indicator = np.asarray([float(record.is_repeat) for record in records], dtype=np.float64)
    copy_number_indicator = np.asarray([float(record.is_copy_number) for record in records], dtype=np.float64)

    standardized_log_length = _standardize_metadata(log_length)

    design_columns: list[np.ndarray] = []
    feature_names: list[str] = []
    for class_index, variant_class in enumerate(unique_classes):
        class_membership = class_membership_matrix[:, class_index]
        design_columns.append(_center_design_column(class_membership))
        feature_names.append("type_offset::" + variant_class.value)

        if class_membership_totals[class_index] < 3.0:
            continue

        design_columns.append(_center_design_column(class_membership * standardized_log_length))
        feature_names.append("log_length_linear::" + variant_class.value)
        design_columns.append(_center_design_column(class_membership * standardized_log_length * standardized_log_length))
        feature_names.append("log_length_quadratic::" + variant_class.value)
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


def _design_matrix_for_feature_names(
    records: Sequence[VariantRecord],
    feature_names: Sequence[str],
) -> np.ndarray:
    if len(feature_names) == 0:
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

    log_length = np.log(np.maximum(np.asarray([record.length for record in records], dtype=np.float64), 1.0))
    standardized_log_length = _standardize_metadata(log_length)
    repeat_indicator = np.asarray([float(record.is_repeat) for record in records], dtype=np.float64)
    copy_number_indicator = np.asarray([float(record.is_copy_number) for record in records], dtype=np.float64)

    design_columns: list[np.ndarray] = []
    for feature_name in feature_names:
        if feature_name == "repeat_indicator":
            design_columns.append(_center_design_column(repeat_indicator))
            continue
        if feature_name == "copy_number_indicator":
            design_columns.append(_center_design_column(copy_number_indicator))
            continue
        prefix, _, class_name = feature_name.partition("::")
        if not class_name:
            raise ValueError("scale-model feature names must include a class suffix when required.")
        variant_class = VariantClass(class_name)
        class_membership = class_membership_by_class[variant_class]
        if prefix == "type_offset":
            design_columns.append(_center_design_column(class_membership))
            continue
        if prefix == "log_length_linear":
            design_columns.append(_center_design_column(class_membership * standardized_log_length))
            continue
        if prefix == "log_length_quadratic":
            design_columns.append(
                _center_design_column(class_membership * standardized_log_length * standardized_log_length)
            )
            continue
        raise ValueError("Unsupported scale-model feature: " + feature_name)

    return np.column_stack(design_columns).astype(np.float64)


def _class_membership_weight(record: VariantRecord, variant_class: VariantClass) -> float:
    for member_class, member_weight in zip(record.prior_class_members, record.prior_class_membership, strict=True):
        if member_class == variant_class:
            return float(member_weight)
    return 0.0


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
    scale_model_feature_names: Sequence[str],
    global_scale: float,
    local_scale: np.ndarray,
    config: ModelConfig,
) -> np.ndarray:
    member_design_matrix = _design_matrix_for_feature_names(
        records=member_records,
        feature_names=scale_model_feature_names,
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

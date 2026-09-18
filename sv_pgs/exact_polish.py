"""Stage 2: exact individual-level polish of every model to the joint fixed point.

Stage 1 fits the empirical-Bayes hyperparameters in LD space from summary
statistics; that is a warm start only (SPEC). This stage takes it to the fixed
point of the CAVI map of the one joint model on the individual-level data and
certifies the result.

E-step mean. For fixed hyperparameters (prior tau^2, sigma_e^2) the posterior
mean maximizes

    L(alpha, beta) = loglik(offset + C alpha + X beta) - beta' diag(1 / tau^2) beta / 2

with a flat prior on the covariate effects alpha. For a Gaussian trait the
beta part of the maximizer is the restricted (alpha-marginal) posterior mean;
for a binary trait it is the penalized logistic mode, which is the fixed point
of the Polya-Gamma IRLS E-step because kappa - omega(eta) eta = y - sigmoid(eta).

L is maximized by block Gauss-Seidel over LD blocks (sequential; parallel
block-Jacobi diverges). Each block step maximizes a quadratic minorizer of L
over (alpha, beta_b) jointly, tight at the current linear predictor eta:

    Gaussian  the log-likelihood itself, curvature w = 1 / sigma_e^2;
    binary    the Jaakkola-Jordan (Polya-Gamma) bound, curvature
              w = omega(eta) = tanh(eta / 2) / (2 eta).

Profiling alpha out of the minorizer gives the Schur-complement system

    A_bb delta = X_b' u_tilde - P_b beta_b
    A_bb = X_b' P_W X_b + diag(P_b),  P_W = W - W C (C' W C)^-1 C' W
    u_tilde = u - W C (C' W C)^-1 C' u,  u = the log-likelihood score,

solved by preconditioned CG from zero on the block's resident genotypes.
Every CG iterate raises the minorizer, so every block step raises L and the
sweep converges to the unique maximizer. The preconditioner is mean(w) times
the covariate-downdated training Gram plus diag(P_b), inverted through its
Cholesky factor. For a Gaussian model it equals A_bb, so CG stops after one
step. The answer depends only on the exact operator applied through X_b; the
Gram only sets the convergence rate.

E-step variances. q(beta) is the full-covariance Gaussian, so the M-step needs
diag(A^-1) of the whole restricted precision A = X' P_W X + diag(1 / tau^2).
It is estimated with the control-variate Hutchinson identity

    diag(A^-1) = diag(B) + E_z[z * (A^-1 z - B z)],  z Rademacher,

where B is block-diagonal: the inverse of each block's exact restricted
precision (for a binary model, at the start state's weights once its
block-variance fixed point is reached; before that, the preconditioner's
inverse, which keeps the estimator unbiased). A^-1 z_k are solved by the same
Gauss-Seidel sweeps as extra right-hand sides of every block read. With one
block the correction vanishes identically and the variances are exact.

Certificate. In the same read of each block, every pass also evaluates the
exact full-data gradient of L, and the residual of every probe system, at the
state the pass started from. A model is certified at that start state when
    the gradient is below tolerance, both as a relative norm and as a
    per-variant KKT residual in posterior-SD units;
    the covariate gradient vanishes;
    every probe system is solved to tolerance;
    one more CAVI M-step from the start state's moments leaves the E-step
    inputs (prior variances, sigma_e^2, TPB shapes, global scale) unchanged to
    the EM convergence tolerance.
The standard error of the variance estimator is reported with it.

Batching. One read of each block serves every model (traits x folds): the
models, and their probes, are the columns of every right-hand side; a fold is
a sample mask. Sample-side arrays (n x columns) live on the source's compute
device; variant-side arrays live on the host and move per block.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_triangular as scipy_solve_triangular

from sv_pgs.anderson import AndersonState, anderson_step
from sv_pgs.config import ModelConfig, TraitType
from sv_pgs.mixture_inference import (
    PriorDesign,
    VariationalFitCheckpoint,
    _binary_expected_polya_gamma_weights_cupy,
    _cavi_auxiliary_delta,
    _effective_prior_variances,
    _metadata_baseline_scales_from_coefficients,
    _propose_cavi_hyperparameters,
    _quantitative_sigma_error2_update,
    _relative_change,
    _resolve_gpu_solve_triangular,
    _scale_model_penalty,
)
from sv_pgs.progress import log

# Each pass is one read of every genotype block; this bounds a polish that
# cannot certify (it then reports the uncertified models rather than loop).
_MAXIMUM_POLISH_PASSES = 200
# Relative full-data gradient norm, ||X'u - P beta|| / (||X'u|| + ||P beta||).
_GRADIENT_TOLERANCE = 1e-7
# Largest per-variant coordinate step |g_j| sqrt(Sigma_jj), in posterior SDs.
_KKT_TOLERANCE = 1e-5
# Relative residual ||z - A x|| / ||z|| of every variance probe system.
_PROBE_RESIDUAL_TOLERANCE = 1e-6
# Rademacher probes per model for the full-covariance posterior variances.
_PROBE_COUNT = 32
# Inner PCG on one block: relative residual of the Schur-complement system.
_INNER_SOLVER_TOLERANCE = 1e-10
_MAXIMUM_INNER_ITERATIONS = 64
# Anderson(m) over each model's whole polish state. When the weighted
# fixed-point residual grows past the restart factor of its best, the history
# restarts with a plain step; past the divergence factor, the plain step is
# taken from the best state instead.
_ANDERSON_MEMORY_DEPTH = 5
_ANDERSON_RESTART_GROWTH = 2.0
_ANDERSON_DIVERGENCE_GROWTH = 10.0


class GenotypeBlockTile(Protocol):
    """Standardized genotypes of one LD block, resident on the compute device."""

    def matmat(self, right: Any) -> Any:
        """X_b @ right for right of shape (p_b, K); returns (n, K)."""

    def rmatmat(self, left: Any) -> Any:
        """X_b.T @ left for left of shape (n, K); returns (p_b, K)."""

    def weighted_gram(self, weights: Any) -> Any:
        """X_b.T diag(weights) X_b; returns (p_b, p_b)."""

    def weighted_cross(self, weights: Any, covariates: Any) -> Any:
        """X_b.T diag(weights) C for C of shape (n, k); returns (p_b, k)."""


class GenotypeBlockSource(Protocol):
    """Sequential reader of the LD blocks of the standardized genotype matrix."""

    @property
    def sample_count(self) -> int: ...

    @property
    def array_module(self) -> Any:
        """``numpy`` or ``cupy``: the device the tiles, and every sample-side array, live on."""

    @property
    def block_variant_indices(self) -> Sequence[NDArray]:
        """Reduced-variant indices of every block; together a partition of all variants."""

    def iter_tiles(self) -> Iterator[tuple[int, GenotypeBlockTile]]:
        """Yield (block index, tile) for every block, in genome order."""


class DenseGenotypeTile:
    """Tile over a standardized block held in device memory."""

    def __init__(self, values: Any) -> None:
        self._values = values

    def matmat(self, right: Any) -> Any:
        return self._values @ right

    def rmatmat(self, left: Any) -> Any:
        return self._values.T @ left

    def weighted_gram(self, weights: Any) -> Any:
        return self._values.T @ (weights[:, None] * self._values)

    def weighted_cross(self, weights: Any, covariates: Any) -> Any:
        return self._values.T @ (weights[:, None] * covariates)


class DenseGenotypeBlockSource:
    """Block source over an in-memory standardized genotype matrix on one device."""

    def __init__(self, genotypes: Any, block_variant_indices: Sequence[NDArray], array_module: Any = np) -> None:
        self._array_module = array_module
        self._genotypes = array_module.asarray(genotypes, dtype=array_module.float64)
        self._block_variant_indices = [np.asarray(indices, dtype=np.int64) for indices in block_variant_indices]
        covered = np.sort(np.concatenate(self._block_variant_indices))
        if not np.array_equal(covered, np.arange(int(self._genotypes.shape[1]))):
            raise ValueError("LD blocks must partition the variant axis.")

    @property
    def sample_count(self) -> int:
        return int(self._genotypes.shape[0])

    @property
    def array_module(self) -> Any:
        return self._array_module

    @property
    def block_variant_indices(self) -> Sequence[NDArray]:
        return self._block_variant_indices

    def iter_tiles(self) -> Iterator[tuple[int, GenotypeBlockTile]]:
        for block_index, variant_indices in enumerate(self._block_variant_indices):
            yield block_index, DenseGenotypeTile(self._genotypes[:, self._array_module.asarray(variant_indices)])


@dataclass(frozen=True)
class _Device:
    """The array module of a source with its triangular solve and host transfer."""

    array_module: Any
    solve_triangular: Callable[..., Any]
    to_host: Callable[[Any], NDArray]


def _device_for(array_module: Any) -> _Device:
    if array_module is np:
        return _Device(array_module=np, solve_triangular=scipy_solve_triangular, to_host=np.asarray)
    return _Device(
        array_module=array_module,
        solve_triangular=_resolve_gpu_solve_triangular(),
        to_host=array_module.asnumpy,
    )


@dataclass(frozen=True)
class PolishStart:
    """The CAVI state one model enters Stage 2 with (Stage 1's hand-off)."""

    alpha: NDArray
    beta: NDArray
    global_scale: float
    scale_model_coefficients: NDArray
    tpb_shape_a_vector: NDArray
    tpb_shape_b_vector: NDArray
    local_scale: NDArray
    auxiliary_delta: NDArray
    sigma_error2: float

    @classmethod
    def from_checkpoint(cls, checkpoint: VariationalFitCheckpoint) -> PolishStart:
        return cls(
            alpha=np.asarray(checkpoint.alpha_state, dtype=np.float64).copy(),
            beta=np.asarray(checkpoint.beta_state, dtype=np.float64).copy(),
            global_scale=float(checkpoint.global_scale),
            scale_model_coefficients=np.asarray(checkpoint.scale_model_coefficients, dtype=np.float64).copy(),
            tpb_shape_a_vector=np.asarray(checkpoint.tpb_shape_a_vector, dtype=np.float64).copy(),
            tpb_shape_b_vector=np.asarray(checkpoint.tpb_shape_b_vector, dtype=np.float64).copy(),
            local_scale=np.asarray(checkpoint.local_scale, dtype=np.float64).copy(),
            auxiliary_delta=np.asarray(checkpoint.auxiliary_delta, dtype=np.float64).copy(),
            sigma_error2=float(checkpoint.sigma_error2),
        )


@dataclass(frozen=True)
class PolishModel:
    """One model (a trait fitted on one training sample mask)."""

    config: ModelConfig
    targets: NDArray
    sample_mask_index: int
    predictor_offset: NDArray
    start: PolishStart


@dataclass(frozen=True)
class PolishCertificate:
    """Exactness evidence for one model at its returned state.

    ``gradient_relative_norm`` is ||X'u - P beta|| / (||X'u|| + ||P beta||) over
    all variants, with u the exact log-likelihood score on the training samples.
    ``maximum_kkt_residual`` is max_j |g_j| sqrt(Sigma_jj): the largest
    coordinate Newton step still available, in posterior-SD units.
    ``covariate_gradient_relative_norm`` is ||C'u|| / ||C'|u| ||.
    ``maximum_probe_residual`` is the largest ||z - A x|| / ||z|| of the
    variance probe systems. ``variance_relative_standard_error`` is the
    Hutchinson standard error of sum_j Sigma_jj / tau_j^2 relative to its value.
    ``fixed_point_residual`` is the largest relative change of the E-step
    inputs under one more CAVI M-step.
    """

    certified: bool
    passes: int
    probe_count: int
    gradient_relative_norm: float
    maximum_kkt_residual: float
    covariate_gradient_relative_norm: float
    maximum_probe_residual: float
    variance_relative_standard_error: float
    fixed_point_residual: float


@dataclass(frozen=True)
class PolishedFit:
    alpha: NDArray
    beta: NDArray
    beta_variance: NDArray
    linear_predictor: NDArray
    sigma_error2: float
    global_scale: float
    scale_model_coefficients: NDArray
    tpb_shape_a_vector: NDArray
    tpb_shape_b_vector: NDArray
    local_scale: NDArray
    auxiliary_delta: NDArray
    prior_variances: NDArray
    certificate: PolishCertificate


@dataclass
class _ModelHyperparameters:
    """Mutable CAVI hyperparameter state of one model."""

    global_scale: float
    scale_model_coefficients: NDArray
    tpb_shape_a_vector: NDArray
    tpb_shape_b_vector: NDArray
    local_scale: NDArray
    auxiliary_delta: NDArray
    sigma_error2: float


@dataclass(frozen=True)
class _EStepInputs:
    """What the E-step of one model depends on: its prior and its noise."""

    baseline_prior_variances: NDArray
    prior_variances: NDArray
    sigma_error2: float


@dataclass(frozen=True)
class _PassMoments:
    """E-step moments of all models at the start and at the end of one pass (host arrays).

    ``*_block_variance`` is diag(B), the block-diagonal inverse alone;
    ``*_variance`` adds the Hutchinson correction toward diag(A^-1).
    """

    start_block_variance: NDArray
    start_variance: NDArray
    end_block_variance: NDArray
    end_variance: NDArray
    start_residual_sum_squares: NDArray
    end_residual_sum_squares: NDArray
    gradient_relative_norm: NDArray
    maximum_kkt_residual: NDArray
    covariate_gradient_relative_norm: NDArray
    maximum_probe_residual: NDArray
    variance_relative_standard_error: NDArray


def _covariate_normal_matrix(array_module: Any, weighted_covariate_gram: Any) -> Any:
    """C' W C plus the ridge the restricted-posterior projector uses (batched over leading axes)."""
    covariate_count = weighted_covariate_gram.shape[-1]
    trace = array_module.trace(weighted_covariate_gram, axis1=-2, axis2=-1)
    ridge = array_module.maximum(trace / max(covariate_count, 1) * 1e-6, 1e-8)
    return weighted_covariate_gram + ridge[..., None, None] * array_module.eye(covariate_count)


def _e_step_inputs(
    hyperparameters: _ModelHyperparameters,
    prior_design: PriorDesign,
    config: ModelConfig,
) -> _EStepInputs:
    """The plug-in prior N(0, tau^2) and the noise the CAVI E-step uses for this state."""
    metadata_baseline_scales = _metadata_baseline_scales_from_coefficients(
        hyperparameters.scale_model_coefficients,
        prior_design.design_matrix,
        config,
    )
    baseline_prior_variances = (float(hyperparameters.global_scale) * metadata_baseline_scales) ** 2
    prior_variances = np.maximum(
        _effective_prior_variances(
            baseline_prior_variances=baseline_prior_variances,
            local_scale=hyperparameters.local_scale,
            config=config,
        ),
        1e-8,
    )
    sigma_error2 = (
        1.0
        if config.trait_type == TraitType.BINARY
        else max(float(hyperparameters.sigma_error2), config.sigma_error_floor)
    )
    return _EStepInputs(
        baseline_prior_variances=np.asarray(baseline_prior_variances, dtype=np.float64),
        prior_variances=np.asarray(prior_variances, dtype=np.float64),
        sigma_error2=sigma_error2,
    )


def _cavi_step(
    *,
    hyperparameters: _ModelHyperparameters,
    e_step_inputs: _EStepInputs,
    beta: NDArray,
    beta_variance: NDArray,
    residual_sum_squares: float,
    training_sample_count: int,
    covariate_count: int,
    prior_design: PriorDesign,
    config: ModelConfig,
) -> _ModelHyperparameters:
    """One CAVI M-step from the E-step moments, as the individual-level EM loop takes it."""
    proposal = _propose_cavi_hyperparameters(
        reduced_second_moment=np.asarray(beta * beta + beta_variance, dtype=np.float64),
        baseline_reduced_prior_variances=e_step_inputs.baseline_prior_variances,
        local_shape_a=prior_design.class_membership_matrix @ hyperparameters.tpb_shape_a_vector,
        local_shape_b=prior_design.class_membership_matrix @ hyperparameters.tpb_shape_b_vector,
        auxiliary_delta=hyperparameters.auxiliary_delta,
        global_scale=hyperparameters.global_scale,
        scale_model_coefficients=hyperparameters.scale_model_coefficients,
        tpb_shape_a_vector=hyperparameters.tpb_shape_a_vector,
        tpb_shape_b_vector=hyperparameters.tpb_shape_b_vector,
        update_scale_and_shapes=True,
        prior_design=prior_design,
        scale_penalty=_scale_model_penalty(prior_design.feature_names, config),
        config=config,
    )
    sigma_error2 = (
        1.0
        if config.trait_type == TraitType.BINARY
        else _quantitative_sigma_error2_update(
            residual_sum_squares=residual_sum_squares,
            sample_count=training_sample_count,
            covariate_count=covariate_count,
            beta_variance=np.maximum(beta_variance, 1e-8),
            prior_variances=e_step_inputs.prior_variances,
            sigma_error2=e_step_inputs.sigma_error2,
            sigma_error_floor=config.sigma_error_floor,
        )
    )
    return _ModelHyperparameters(
        global_scale=proposal.global_scale,
        scale_model_coefficients=proposal.scale_model_coefficients,
        tpb_shape_a_vector=proposal.tpb_shape_a_vector,
        tpb_shape_b_vector=proposal.tpb_shape_b_vector,
        local_scale=proposal.local_scale,
        auxiliary_delta=_cavi_auxiliary_delta(
            local_shape_a=prior_design.class_membership_matrix @ proposal.tpb_shape_a_vector,
            local_shape_b=prior_design.class_membership_matrix @ proposal.tpb_shape_b_vector,
            local_scale=proposal.local_scale,
            config=config,
        ),
        sigma_error2=sigma_error2,
    )


def _fixed_point_residual(
    *,
    current: _ModelHyperparameters,
    current_inputs: _EStepInputs,
    proposed: _ModelHyperparameters,
    proposed_inputs: _EStepInputs,
) -> float:
    """Largest relative change of the E-step inputs under one CAVI step.

    Prior variances and precisions are compared as scale-free relative norms
    (the variance norm is dominated by the influential variants, the
    precision norm by the most shrunk ones), so tiny prior variances cannot
    hide movement the way an absolute RMS change would.
    """
    return float(
        max(
            _relative_change(proposed_inputs.prior_variances, current_inputs.prior_variances),
            _relative_change(1.0 / proposed_inputs.prior_variances, 1.0 / current_inputs.prior_variances),
            abs(proposed_inputs.sigma_error2 - current_inputs.sigma_error2) / current_inputs.sigma_error2,
            abs(proposed.global_scale - current.global_scale) / current.global_scale,
            _relative_change(proposed.tpb_shape_a_vector, current.tpb_shape_a_vector),
            _relative_change(proposed.tpb_shape_b_vector, current.tpb_shape_b_vector),
        )
    )


class _BatchedModelState:
    """Per-model arrays of the E-step: one mean column and ``probe_count`` probe columns per model.

    Probe columns are model-major: column s of model m is ``m * probe_count + s``.
    Sample-side arrays live on the device, variant-side arrays on the host.
    """

    def __init__(
        self,
        *,
        models: Sequence[PolishModel],
        covariates: NDArray,
        sample_masks: NDArray,
        variant_count: int,
        probe_count: int,
        device: _Device,
    ) -> None:
        array_module = device.array_module
        model_count = len(models)
        self.device = device
        self.probe_count = int(probe_count)
        self.covariates = array_module.asarray(covariates, dtype=array_module.float64)
        self.sample_masks = array_module.asarray(sample_masks, dtype=array_module.float64)
        self.mask_index = np.asarray([model.sample_mask_index for model in models], dtype=np.int64)
        self.training_weight = self.sample_masks[array_module.asarray(self.mask_index)].T
        self.is_binary = np.asarray(
            [model.config.trait_type == TraitType.BINARY for model in models],
            dtype=bool,
        )
        self.minimum_binary_curvature = array_module.asarray(
            [model.config.polya_gamma_minimum_weight for model in models],
            dtype=array_module.float64,
        )
        self.targets = array_module.asarray(
            np.column_stack([np.asarray(model.targets, dtype=np.float64) for model in models])
        )
        self.offsets = array_module.asarray(
            np.column_stack([np.asarray(model.predictor_offset, dtype=np.float64) for model in models])
        )
        self.alpha = array_module.asarray(
            np.column_stack([np.asarray(model.start.alpha, dtype=np.float64) for model in models])
        )
        self.beta = np.column_stack([np.asarray(model.start.beta, dtype=np.float64) for model in models])
        self.prior_precision = np.ones_like(self.beta)
        self.noise_variance = np.ones(model_count, dtype=np.float64)
        self.linear_predictor = array_module.zeros_like(self.targets)
        probe_generator = np.random.default_rng(int(models[0].config.random_seed))
        self.probes = probe_generator.choice(np.array([-1.0, 1.0]), size=(variant_count, self.probe_count))
        self.probe_solution = np.zeros((variant_count, model_count * self.probe_count), dtype=np.float64)
        self.probe_predictor = array_module.zeros(
            (int(self.targets.shape[0]), model_count * self.probe_count),
            dtype=array_module.float64,
        )
        # Model of every PCG column (mean columns, then model-major probe
        # columns) and each model's columns as one row of a gather index.
        self.column_models = np.concatenate([np.arange(model_count), np.repeat(np.arange(model_count), self.probe_count)])
        self.model_column_index = np.concatenate(
            [
                np.arange(model_count)[:, None],
                model_count + np.arange(model_count * self.probe_count).reshape(model_count, self.probe_count),
            ],
            axis=1,
        )
        if self.covariates.shape[0] != self.targets.shape[0]:
            raise ValueError("covariates must have one row per sample.")
        if self.alpha.shape[0] != self.covariates.shape[1]:
            raise ValueError("every model's alpha must match the covariate count.")

    @property
    def model_count(self) -> int:
        return int(self.beta.shape[1])

    def probe_columns(self, model_index: int) -> slice:
        return slice(model_index * self.probe_count, (model_index + 1) * self.probe_count)

    def score_and_curvature(self, linear_predictor: Any) -> tuple[Any, Any]:
        """Log-likelihood score u and minorizer curvature w, masked to training samples."""
        array_module = self.device.array_module
        noise_variance = array_module.asarray(self.noise_variance)[None, :]
        is_binary = array_module.asarray(self.is_binary)[None, :]
        gaussian_residual = (self.targets - linear_predictor) / noise_variance
        gaussian_curvature = array_module.broadcast_to(1.0 / noise_variance, linear_predictor.shape)
        # sigmoid(eta) = (1 + tanh(eta / 2)) / 2 without overflow.
        binary_residual = self.targets - 0.5 * (1.0 + array_module.tanh(0.5 * linear_predictor))
        # The PG-IRLS E-step floors its weights; a larger curvature keeps the
        # Jaakkola-Jordan bound a minorizer, so the floor leaves the MM valid.
        binary_curvature = array_module.maximum(
            _binary_expected_polya_gamma_weights_cupy(
                array_module,
                linear_predictor,
                float(np.finfo(np.float64).tiny),
                dtype=array_module.float64,
            ),
            self.minimum_binary_curvature[None, :],
        )
        training = self.training_weight > 0.0
        score = array_module.where(training, array_module.where(is_binary, binary_residual, gaussian_residual), 0.0)
        curvature = array_module.where(
            training,
            array_module.where(is_binary, binary_curvature, gaussian_curvature),
            0.0,
        )
        return score, curvature

    def covariate_inverses(self, curvature: Any) -> Any:
        """(C' W_m C + ridge)^-1, one per model; shape (K, k, k).

        The ridge bounds the condition number (as in the restricted-posterior
        projector), so the explicit inverse is accurate and every covariate
        solve becomes one batched product.
        """
        array_module = self.device.array_module
        weighted_gram = array_module.einsum("sa,sm,sb->mab", self.covariates, curvature, self.covariates)
        return array_module.linalg.inv(_covariate_normal_matrix(array_module, weighted_gram))

    def solve_covariates(self, inverses: Any, right_hand_side: Any, column_models: NDArray) -> Any:
        """(C' W_m C + ridge)^-1 applied to every column of a k x columns right-hand side."""
        array_module = self.device.array_module
        return array_module.einsum("cab,bc->ac", inverses[array_module.asarray(column_models)], right_hand_side)

    def project(self, inverses: Any, curvature: Any, values: Any, column_models: NDArray) -> Any:
        """P_W of each column's model: W v - W C (C'WC)^-1 C' W v."""
        column_curvature = curvature[:, self.device.array_module.asarray(column_models)]
        weighted_values = column_curvature * values
        coefficients = self.solve_covariates(inverses, self.covariates.T @ weighted_values, column_models)
        return weighted_values - column_curvature * (self.covariates @ coefficients)


class _BlockFactor:
    """Explicit inverses of one SPD block matrix per model, through Jacobi-scaled Cholesky factors."""

    def __init__(self, device: _Device, matrices: Any) -> None:
        array_module = device.array_module
        scale = array_module.sqrt(array_module.diagonal(matrices, axis1=1, axis2=2))
        factors = array_module.linalg.cholesky(matrices / (scale[:, :, None] * scale[:, None, :]))
        identity = array_module.eye(int(matrices.shape[1]))
        inverse_factors = array_module.stack(
            [device.solve_triangular(factor, identity, lower=True, check_finite=False) for factor in factors]
        )
        self.inverse = (
            array_module.matmul(array_module.swapaxes(inverse_factors, 1, 2), inverse_factors)
            / (scale[:, :, None] * scale[:, None, :])
        )
        self.inverse_diagonal = array_module.sum(inverse_factors * inverse_factors, axis=1) / (scale * scale)


def _factor_with_prior(
    device: _Device,
    downdated_grams: Sequence[Any],
    curvature_scale: Any,
    block_prior_precision: Any,
) -> _BlockFactor:
    """Inverses of curvature_scale[m] * downdated_grams[m] + diag(P_b[:, m]) for every model m."""
    array_module = device.array_module
    matrices = curvature_scale[:, None, None] * array_module.stack(downdated_grams)
    diagonal = array_module.arange(int(matrices.shape[1]))
    matrices[:, diagonal, diagonal] += block_prior_precision.T
    return _BlockFactor(device, matrices)


def _downdated_gram(device: _Device, *, gram: Any, covariate_cross: Any, covariate_gram: Any) -> Any:
    """X_b' W X_b - X_b' W C (C' W C + ridge)^-1 C' W X_b from weighted Grams.

    The ridge is proportional to C' W C, so for W = c * mask the result is
    c times the downdated training Gram: one Gram per sample mask serves
    every model trained on it at any mean curvature c.
    """
    array_module = device.array_module
    covariate_factor = array_module.linalg.cholesky(_covariate_normal_matrix(array_module, covariate_gram))
    whitened_cross = device.solve_triangular(covariate_factor, covariate_cross.T, lower=True, check_finite=False)
    return gram - whitened_cross.T @ whitened_cross


def _apply_inverse(state: _BatchedModelState, factor: _BlockFactor, values: Any) -> Any:
    """Each model's block inverse applied to its mean and probe columns."""
    array_module = state.device.array_module
    gather = array_module.asarray(state.model_column_index)
    grouped = array_module.swapaxes(values[:, gather], 0, 1)
    solved = array_module.matmul(factor.inverse, grouped)
    result = array_module.empty_like(values)
    result[:, gather] = array_module.swapaxes(solved, 0, 1)
    return result


def _block_pcg(
    *,
    tile: GenotypeBlockTile,
    state: _BatchedModelState,
    covariate_inverses: Any,
    curvature: Any,
    column_prior_precision: Any,
    right_hand_side: Any,
    preconditioner: _BlockFactor,
) -> tuple[Any, Any]:
    """Solve A_bb delta = rhs for every mean and probe column from delta = 0.

    Columns are the K mean columns, then the model-major probe columns.
    Returns (delta, X_b delta).
    """
    array_module = state.device.array_module
    step = array_module.zeros_like(right_hand_side)
    applied_step = array_module.zeros((int(curvature.shape[0]), int(right_hand_side.shape[1])), dtype=array_module.float64)
    residual = right_hand_side.copy()
    right_hand_side_norm = array_module.linalg.norm(right_hand_side, axis=0)
    active = right_hand_side_norm > 0.0
    preconditioned = _apply_inverse(state, preconditioner, residual)
    direction = preconditioned.copy()
    residual_inner = array_module.sum(residual * preconditioned, axis=0)
    for _iteration_index in range(_MAXIMUM_INNER_ITERATIONS):
        if not bool(array_module.any(active)):
            break
        applied_direction = tile.matmat(direction)
        operator_direction = (
            tile.rmatmat(state.project(covariate_inverses, curvature, applied_direction, state.column_models))
            + column_prior_precision * direction
        )
        curvature_along_direction = array_module.sum(direction * operator_direction, axis=0)
        step_length = array_module.where(
            active,
            residual_inner / array_module.where(active, curvature_along_direction, 1.0),
            0.0,
        )
        step += step_length[None, :] * direction
        applied_step += step_length[None, :] * applied_direction
        residual -= step_length[None, :] * operator_direction
        active &= array_module.linalg.norm(residual, axis=0) > _INNER_SOLVER_TOLERANCE * right_hand_side_norm
        preconditioned = _apply_inverse(state, preconditioner, residual)
        updated_residual_inner = array_module.sum(residual * preconditioned, axis=0)
        conjugacy = array_module.where(
            active,
            updated_residual_inner / array_module.where(active, residual_inner, 1.0),
            0.0,
        )
        direction = array_module.where(active[None, :], preconditioned + conjugacy[None, :] * direction, 0.0)
        residual_inner = updated_residual_inner
    return step, applied_step


def _hutchinson_block_variance(
    *,
    state: _BatchedModelState,
    control_variate: _BlockFactor,
    block_probes: Any,
    block_probe_solution: Any,
    block_prior_variances: Any,
) -> tuple[Any, Any, Any]:
    """diag(B) and diag(B) + mean_s z_s * (x_s - B z_s) on one block, and each probe's leverage term.

    Returns the (p_b, K) block variances, the (p_b, K) Hutchinson estimate,
    and the (K, S) per-probe sums of z_s * (x_s - B z_s) / tau^2 that give
    the estimator's standard error.
    """
    array_module = state.device.array_module
    model_count = state.model_count
    block_variance = control_variate.inverse_diagonal.T
    control_images = array_module.matmul(control_variate.inverse, block_probes[None, :, :])
    solutions = array_module.swapaxes(
        block_probe_solution.reshape(int(block_probe_solution.shape[0]), model_count, state.probe_count),
        0,
        1,
    )
    correction_terms = block_probes[None, :, :] * (solutions - control_images)
    variance = block_variance + array_module.mean(correction_terms, axis=2).T
    probe_leverage = array_module.sum(correction_terms / block_prior_variances.T[:, :, None], axis=1)
    return block_variance, variance, probe_leverage


def _gauss_seidel_pass(
    *,
    source: GenotypeBlockSource,
    state: _BatchedModelState,
    exact_binary_variance: NDArray,
) -> _PassMoments:
    """One sweep over all LD blocks for every model; certifies the start state on the way.

    Every block is read once. On it, the start state's gradient, probe
    residuals and posterior variances are evaluated first (the certificate),
    then the block's mean and probe columns take one Gauss-Seidel step.

    ``exact_binary_variance`` (K,) marks binary models whose control variate B
    is the exact block inverse at the start state's weights (from the weighted
    block Gram); every other model's B is its preconditioner's inverse, which
    for a Gaussian model is exact.
    """
    device = state.device
    array_module = device.array_module
    model_count = state.model_count
    probe_count = state.probe_count
    probe_column_count = model_count * probe_count
    mean_column_models = np.arange(model_count)
    probe_column_models = state.column_models[model_count:]
    start_linear_predictor = state.linear_predictor.copy()
    start_score, start_curvature = state.score_and_curvature(start_linear_predictor)
    start_inverses = state.covariate_inverses(start_curvature)
    start_projected_probe_predictor = state.project(
        start_inverses,
        start_curvature,
        state.probe_predictor,
        probe_column_models,
    )
    start_covariate_gradient = state.covariates.T @ start_score
    covariate_gradient_scale = array_module.abs(state.covariates).T @ array_module.abs(start_score)
    covariate_gram_by_mask = array_module.einsum("sa,ms,sb->mab", state.covariates, state.sample_masks, state.covariates)
    training_sample_count = state.training_weight.sum(axis=0)
    prior_variances = 1.0 / state.prior_precision
    exact_models = np.flatnonzero(exact_binary_variance)
    exact_mask = array_module.asarray(exact_binary_variance)
    used_masks = np.unique(state.mask_index)
    start_block_variance = np.zeros_like(state.beta)
    start_variance = np.zeros_like(state.beta)
    end_block_variance = np.zeros_like(state.beta)
    end_variance = np.zeros_like(state.beta)
    gradient_square_sum = array_module.zeros(model_count, dtype=array_module.float64)
    score_square_sum = array_module.zeros(model_count, dtype=array_module.float64)
    penalty_square_sum = array_module.zeros(model_count, dtype=array_module.float64)
    maximum_kkt_residual = array_module.zeros(model_count, dtype=array_module.float64)
    probe_residual_square_sum = array_module.zeros(probe_column_count, dtype=array_module.float64)
    probe_leverage = array_module.zeros((model_count, probe_count), dtype=array_module.float64)
    for block_index, tile in source.iter_tiles():
        variant_indices = source.block_variant_indices[block_index]
        block_beta = array_module.asarray(state.beta[variant_indices])
        block_prior_precision = array_module.asarray(state.prior_precision[variant_indices])
        block_prior_variances = array_module.asarray(prior_variances[variant_indices])
        block_probes = array_module.asarray(state.probes[variant_indices])
        block_probe_solution = array_module.asarray(state.probe_solution[variant_indices])
        repeated_probes = array_module.tile(block_probes, (1, model_count))
        column_prior_precision = block_prior_precision[:, array_module.asarray(state.column_models)]

        mask_downdated = {
            int(mask_index): _downdated_gram(
                device,
                gram=tile.weighted_gram(state.sample_masks[int(mask_index)]),
                covariate_cross=tile.weighted_cross(state.sample_masks[int(mask_index)], state.covariates),
                covariate_gram=covariate_gram_by_mask[int(mask_index)],
            )
            for mask_index in used_masks
        }
        preconditioner_grams = [mask_downdated[int(state.mask_index[model_index])] for model_index in range(model_count)]

        score, curvature = state.score_and_curvature(state.linear_predictor)
        covariate_inverses = state.covariate_inverses(curvature)
        covariate_score = state.covariates.T @ score
        projected_score = score - curvature * (
            state.covariates @ state.solve_covariates(covariate_inverses, covariate_score, mean_column_models)
        )
        products = tile.rmatmat(
            array_module.concatenate(
                [
                    projected_score,
                    start_score,
                    state.project(covariate_inverses, curvature, state.probe_predictor, probe_column_models),
                    start_projected_probe_predictor,
                ],
                axis=1,
            )
        )
        score_products = products[:, :model_count]
        start_score_products = products[:, model_count : 2 * model_count]
        probe_products = products[:, 2 * model_count : 2 * model_count + probe_column_count]
        start_probe_products = products[:, 2 * model_count + probe_column_count :]
        penalty = block_prior_precision * block_beta
        mean_curvature = curvature.sum(axis=0) / training_sample_count
        preconditioner = _factor_with_prior(device, preconditioner_grams, mean_curvature, block_prior_precision)
        control_variate = preconditioner
        if exact_models.size:
            control_variate_grams = list(preconditioner_grams)
            for model_index in exact_models:
                model_weights = start_curvature[:, int(model_index)]
                control_variate_grams[int(model_index)] = _downdated_gram(
                    device,
                    gram=tile.weighted_gram(model_weights),
                    covariate_cross=tile.weighted_cross(model_weights, state.covariates),
                    covariate_gram=state.covariates.T @ (model_weights[:, None] * state.covariates),
                )
            control_variate = _factor_with_prior(
                device,
                control_variate_grams,
                array_module.where(exact_mask, 1.0, mean_curvature),
                block_prior_precision,
            )

        # Certificate quantities at the start state.
        start_gradient = start_score_products - penalty
        gradient_square_sum += array_module.sum(start_gradient * start_gradient, axis=0)
        score_square_sum += array_module.sum(start_score_products * start_score_products, axis=0)
        penalty_square_sum += array_module.sum(penalty * penalty, axis=0)
        start_probe_residual = (
            repeated_probes - start_probe_products - column_prior_precision[:, model_count:] * block_probe_solution
        )
        probe_residual_square_sum += array_module.sum(start_probe_residual * start_probe_residual, axis=0)
        block_start_block_variance, block_start_variance, block_probe_leverage = _hutchinson_block_variance(
            state=state,
            control_variate=control_variate,
            block_probes=block_probes,
            block_probe_solution=block_probe_solution,
            block_prior_variances=block_prior_variances,
        )
        start_block_variance[variant_indices] = device.to_host(block_start_block_variance)
        start_variance[variant_indices] = device.to_host(block_start_variance)
        probe_leverage += block_probe_leverage
        maximum_kkt_residual = array_module.maximum(
            maximum_kkt_residual,
            array_module.max(array_module.abs(start_gradient) * array_module.sqrt(array_module.maximum(block_start_variance, 0.0)), axis=0),
        )

        # One Gauss-Seidel step of the mean and probe columns.
        step, applied_step = _block_pcg(
            tile=tile,
            state=state,
            covariate_inverses=covariate_inverses,
            curvature=curvature,
            column_prior_precision=column_prior_precision,
            right_hand_side=array_module.concatenate(
                [
                    score_products - penalty,
                    repeated_probes - probe_products - column_prior_precision[:, model_count:] * block_probe_solution,
                ],
                axis=1,
            ),
            preconditioner=preconditioner,
        )
        covariate_step = state.solve_covariates(
            covariate_inverses,
            covariate_score - state.covariates.T @ (curvature * applied_step[:, :model_count]),
            mean_column_models,
        )
        updated_probe_solution = block_probe_solution + step[:, model_count:]
        state.beta[variant_indices] = device.to_host(block_beta + step[:, :model_count])
        state.alpha += covariate_step
        state.linear_predictor += applied_step[:, :model_count] + state.covariates @ covariate_step
        state.probe_solution[variant_indices] = device.to_host(updated_probe_solution)
        state.probe_predictor += applied_step[:, model_count:]
        block_end_block_variance, block_end_variance, _end_probe_leverage = _hutchinson_block_variance(
            state=state,
            control_variate=control_variate,
            block_probes=block_probes,
            block_probe_solution=updated_probe_solution,
            block_prior_variances=block_prior_variances,
        )
        end_block_variance[variant_indices] = device.to_host(block_end_block_variance)
        end_variance[variant_indices] = device.to_host(block_end_variance)

    training = state.training_weight > 0.0
    start_residual = array_module.where(training, state.targets - start_linear_predictor, 0.0)
    end_residual = array_module.where(training, state.targets - state.linear_predictor, 0.0)
    probe_norm = np.sqrt(float(state.probes.shape[0]))
    leverage_total = np.sum(start_variance * state.prior_precision, axis=0)
    host = device.to_host
    return _PassMoments(
        start_block_variance=start_block_variance,
        start_variance=start_variance,
        end_block_variance=end_block_variance,
        end_variance=end_variance,
        start_residual_sum_squares=host(array_module.sum(start_residual * start_residual, axis=0)),
        end_residual_sum_squares=host(array_module.sum(end_residual * end_residual, axis=0)),
        gradient_relative_norm=host(
            array_module.sqrt(gradient_square_sum)
            / array_module.maximum(
                array_module.sqrt(score_square_sum) + array_module.sqrt(penalty_square_sum),
                np.finfo(np.float64).tiny,
            )
        ),
        maximum_kkt_residual=host(maximum_kkt_residual),
        covariate_gradient_relative_norm=host(
            array_module.linalg.norm(start_covariate_gradient, axis=0)
            / array_module.maximum(array_module.linalg.norm(covariate_gradient_scale, axis=0), np.finfo(np.float64).tiny)
        ),
        maximum_probe_residual=host(
            array_module.max(
                array_module.sqrt(probe_residual_square_sum).reshape(model_count, probe_count) / probe_norm,
                axis=1,
            )
        ),
        variance_relative_standard_error=host(
            array_module.std(probe_leverage, axis=1, ddof=1) / np.sqrt(probe_count)
        )
        / np.maximum(leverage_total, np.finfo(np.float64).tiny),
    )


def _initial_linear_predictor(source: GenotypeBlockSource, state: _BatchedModelState) -> Any:
    array_module = state.device.array_module
    linear_predictor = state.offsets + state.covariates @ state.alpha
    for block_index, tile in source.iter_tiles():
        linear_predictor += tile.matmat(array_module.asarray(state.beta[source.block_variant_indices[block_index]]))
    return linear_predictor


class _AndersonCoordinates:
    """Packs one model's polish state into the vector Anderson(m) mixes.

    The state is everything the next pass depends on besides the probe
    solutions (which track A^-1 z as inner solves): beta, alpha and the CAVI
    hyperparameters, positive ones in log space. The linear predictor is
    affine in (alpha, beta), so it is carried along with zero residual weight
    and a mixed state keeps a consistent predictor without a genotype pass.
    Residual weights put beta in prior-SD units and give each group of
    coordinates the same total weight.
    """

    def __init__(
        self,
        *,
        prior_variances: NDArray,
        covariate_scale: NDArray,
        sample_count: int,
        hyperparameters: _ModelHyperparameters,
    ) -> None:
        self._variant_count = int(prior_variances.shape[0])
        self._covariate_count = int(covariate_scale.shape[0])
        self._sample_count = int(sample_count)
        self._coefficient_count = int(hyperparameters.scale_model_coefficients.shape[0])
        self._class_count = int(hyperparameters.tpb_shape_a_vector.shape[0])
        global_count = 2 + self._coefficient_count + 2 * self._class_count
        per_variant = 1.0 / np.sqrt(self._variant_count)
        self.residual_weights = np.concatenate(
            [
                per_variant / np.sqrt(prior_variances),
                covariate_scale / np.sqrt(max(self._covariate_count, 1)),
                np.zeros(self._sample_count),
                np.full(2 * self._variant_count, per_variant),
                np.full(global_count, 1.0 / np.sqrt(global_count)),
            ]
        )

    def pack(
        self,
        *,
        beta: NDArray,
        alpha: NDArray,
        linear_predictor: NDArray,
        hyperparameters: _ModelHyperparameters,
    ) -> NDArray:
        return np.concatenate(
            [
                beta,
                alpha,
                linear_predictor,
                np.log(hyperparameters.local_scale),
                np.log(hyperparameters.auxiliary_delta),
                [np.log(hyperparameters.global_scale), np.log(hyperparameters.sigma_error2)],
                hyperparameters.scale_model_coefficients,
                np.log(hyperparameters.tpb_shape_a_vector),
                np.log(hyperparameters.tpb_shape_b_vector),
            ]
        )

    def unpack(
        self,
        vector: NDArray,
        config: ModelConfig,
    ) -> tuple[NDArray, NDArray, NDArray, _ModelHyperparameters]:
        boundaries = np.cumsum(
            [
                self._variant_count,
                self._covariate_count,
                self._sample_count,
                self._variant_count,
                self._variant_count,
                2,
                self._coefficient_count,
                self._class_count,
            ]
        )
        (
            beta,
            alpha,
            linear_predictor,
            log_local_scale,
            log_auxiliary_delta,
            log_scalars,
            coefficients,
            log_shape_a,
            log_shape_b,
        ) = np.split(vector, boundaries)
        hyperparameters = _ModelHyperparameters(
            global_scale=float(np.clip(np.exp(log_scalars[0]), config.global_scale_floor, config.global_scale_ceiling)),
            scale_model_coefficients=coefficients.copy(),
            tpb_shape_a_vector=np.clip(np.exp(log_shape_a), config.minimum_tpb_shape, config.maximum_tpb_shape),
            tpb_shape_b_vector=np.clip(np.exp(log_shape_b), config.minimum_tpb_shape, config.maximum_tpb_shape),
            local_scale=np.maximum(np.exp(log_local_scale), config.local_scale_floor),
            auxiliary_delta=np.maximum(np.exp(log_auxiliary_delta), config.local_scale_floor),
            sigma_error2=float(np.exp(log_scalars[1])),
        )
        return beta.copy(), alpha.copy(), linear_predictor.copy(), hyperparameters


def _polished_fit(
    *,
    alpha: NDArray,
    beta: NDArray,
    beta_variance: NDArray,
    linear_predictor: NDArray,
    hyperparameters: _ModelHyperparameters,
    e_step_inputs: _EStepInputs,
    certificate: PolishCertificate,
) -> PolishedFit:
    return PolishedFit(
        alpha=alpha.copy(),
        beta=beta.copy(),
        beta_variance=beta_variance.copy(),
        linear_predictor=linear_predictor.copy(),
        sigma_error2=e_step_inputs.sigma_error2,
        global_scale=hyperparameters.global_scale,
        scale_model_coefficients=hyperparameters.scale_model_coefficients.copy(),
        tpb_shape_a_vector=hyperparameters.tpb_shape_a_vector.copy(),
        tpb_shape_b_vector=hyperparameters.tpb_shape_b_vector.copy(),
        local_scale=hyperparameters.local_scale.copy(),
        auxiliary_delta=hyperparameters.auxiliary_delta.copy(),
        prior_variances=e_step_inputs.prior_variances.copy(),
        certificate=certificate,
    )


@dataclass
class _ModelProgress:
    """Where one model is in the polish.

    ``full_variance`` switches the M-step from the block-diagonal variances
    diag(B) to the Hutchinson estimate of diag(A^-1) once the block-variance
    fixed point is reached (for a binary model, B then also becomes the exact
    block inverse at its weights). ``frozen`` holds the hyperparameters fixed
    while the sweeps drive the E-step to the certificate tolerances.
    """

    full_variance: bool = False
    frozen: bool = False


def _mean_is_stationary(moments: _PassMoments, model_index: int) -> bool:
    return bool(
        moments.gradient_relative_norm[model_index] <= _GRADIENT_TOLERANCE
        and moments.maximum_kkt_residual[model_index] <= _KKT_TOLERANCE
        and moments.covariate_gradient_relative_norm[model_index] <= _GRADIENT_TOLERANCE
        and moments.maximum_probe_residual[model_index] <= _PROBE_RESIDUAL_TOLERANCE
    )


def polish_to_fixed_point(
    *,
    source: GenotypeBlockSource,
    covariates: NDArray,
    sample_masks: NDArray,
    models: Sequence[PolishModel],
    prior_design: PriorDesign,
) -> list[PolishedFit]:
    """Polish every model to the certified fixed point of the joint CAVI map.

    ``sample_masks`` is (M, n) with 0/1 entries; model m trains on the samples
    of ``sample_masks[models[m].sample_mask_index]``. All models share one
    read of every block per pass, and every pass certifies the state it
    started from.

    Per model the polish runs in three phases:
      1. the CAVI map with block-diagonal variances diag(B) (a Gauss-Seidel
         sweep, then the M-step), extrapolated by safeguarded Anderson(m),
         until its fixed-point residual is below the EM tolerance; the probe
         systems are solved alongside;
      2. the same with the full-covariance variances (Hutchinson with the
         exact block control variate), Anderson restarted;
      3. once that residual is below tolerance the hyperparameters freeze and
         the sweeps drive the mean and probe solves to the certificate
         tolerances. If the converged moments move the hyperparameters by
         more than the EM tolerance, phase 2 resumes.
    The Anderson history restarts with a plain step whenever the weighted
    fixed-point residual grows, from the best state when it grows tenfold.
    The returned state of a certified model is the start state of its
    certifying pass.
    """
    if not models:
        return []
    device = _device_for(source.array_module)
    variant_count = int(sum(indices.shape[0] for indices in source.block_variant_indices))
    for model in models:
        if model.start.beta.shape != (variant_count,):
            raise ValueError("every model's beta must cover the reduced variants of the source.")
    state = _BatchedModelState(
        models=models,
        covariates=covariates,
        sample_masks=sample_masks,
        variant_count=variant_count,
        probe_count=_PROBE_COUNT,
        device=device,
    )
    covariate_count = int(state.covariates.shape[1])
    hyperparameters = [
        _ModelHyperparameters(
            global_scale=float(model.start.global_scale),
            scale_model_coefficients=np.asarray(model.start.scale_model_coefficients, dtype=np.float64),
            tpb_shape_a_vector=np.asarray(model.start.tpb_shape_a_vector, dtype=np.float64),
            tpb_shape_b_vector=np.asarray(model.start.tpb_shape_b_vector, dtype=np.float64),
            local_scale=np.asarray(model.start.local_scale, dtype=np.float64),
            auxiliary_delta=np.asarray(model.start.auxiliary_delta, dtype=np.float64),
            sigma_error2=float(model.start.sigma_error2),
        )
        for model in models
    ]
    host_sample_masks = np.asarray(sample_masks, dtype=np.float64)
    host_covariates = np.asarray(covariates, dtype=np.float64)
    training_sample_count = [int(round(float(host_sample_masks[model.sample_mask_index].sum()))) for model in models]
    covariate_scale = np.sqrt(
        np.einsum("ms,sa->am", host_sample_masks, host_covariates * host_covariates)
        / np.maximum(host_sample_masks.sum(axis=1), 1.0)
    )
    state.linear_predictor = _initial_linear_predictor(source, state)
    progress = [_ModelProgress() for _model in models]
    anderson_states = [AndersonState(memory_depth=_ANDERSON_MEMORY_DEPTH) for _model in models]
    coordinates: list[_AndersonCoordinates | None] = [None] * state.model_count
    smallest_residual_norm = np.full(state.model_count, np.inf)
    # The map value at each model's smallest-residual state: a diverging
    # extrapolation restarts from this plain step.
    best_mapped_vector: list[NDArray | None] = [None] * state.model_count
    certified_results: list[PolishedFit | None] = [None] * state.model_count
    last_certificates: list[PolishCertificate | None] = [None] * state.model_count

    def restart_anderson(model_index: int) -> None:
        anderson_states[model_index].reset()
        smallest_residual_norm[model_index] = np.inf
        best_mapped_vector[model_index] = None

    for pass_index in range(_MAXIMUM_POLISH_PASSES):
        e_step_inputs = [
            _e_step_inputs(model_hyperparameters, prior_design, model.config)
            for model_hyperparameters, model in zip(hyperparameters, models)
        ]
        for model_index, inputs in enumerate(e_step_inputs):
            state.prior_precision[:, model_index] = 1.0 / inputs.prior_variances
            state.noise_variance[model_index] = inputs.sigma_error2
        exact_binary_variance = np.asarray(
            [model_progress.full_variance for model_progress in progress],
            dtype=bool,
        ) & state.is_binary
        start_alpha = device.to_host(state.alpha).copy()
        start_beta = state.beta.copy()
        start_linear_predictor = device.to_host(state.linear_predictor).copy()
        moments = _gauss_seidel_pass(source=source, state=state, exact_binary_variance=exact_binary_variance)
        swept_alpha = device.to_host(state.alpha)
        swept_linear_predictor = device.to_host(state.linear_predictor)
        for model_index, model in enumerate(models):
            if certified_results[model_index] is not None:
                continue
            model_progress = progress[model_index]
            start_variance = (
                moments.start_variance[:, model_index]
                if model_progress.full_variance
                else moments.start_block_variance[:, model_index]
            )
            start_proposal = _cavi_step(
                hyperparameters=hyperparameters[model_index],
                e_step_inputs=e_step_inputs[model_index],
                beta=start_beta[:, model_index],
                beta_variance=start_variance,
                residual_sum_squares=float(moments.start_residual_sum_squares[model_index]),
                training_sample_count=training_sample_count[model_index],
                covariate_count=covariate_count,
                prior_design=prior_design,
                config=model.config,
            )
            fixed_point_residual = _fixed_point_residual(
                current=hyperparameters[model_index],
                current_inputs=e_step_inputs[model_index],
                proposed=start_proposal,
                proposed_inputs=_e_step_inputs(start_proposal, prior_design, model.config),
            )
            mean_is_stationary = _mean_is_stationary(moments, model_index)
            at_fixed_point = fixed_point_residual <= model.config.convergence_tolerance
            certificate = PolishCertificate(
                certified=bool(model_progress.full_variance and mean_is_stationary and at_fixed_point),
                passes=pass_index + 1,
                probe_count=state.probe_count,
                gradient_relative_norm=float(moments.gradient_relative_norm[model_index]),
                maximum_kkt_residual=float(moments.maximum_kkt_residual[model_index]),
                covariate_gradient_relative_norm=float(moments.covariate_gradient_relative_norm[model_index]),
                maximum_probe_residual=float(moments.maximum_probe_residual[model_index]),
                variance_relative_standard_error=float(moments.variance_relative_standard_error[model_index]),
                fixed_point_residual=fixed_point_residual,
            )
            last_certificates[model_index] = certificate
            if certificate.certified:
                certified_results[model_index] = _polished_fit(
                    alpha=start_alpha[:, model_index],
                    beta=start_beta[:, model_index],
                    beta_variance=start_variance,
                    linear_predictor=start_linear_predictor[:, model_index],
                    hyperparameters=hyperparameters[model_index],
                    e_step_inputs=e_step_inputs[model_index],
                    certificate=certificate,
                )
                continue
            if model_progress.frozen:
                if mean_is_stationary and not at_fixed_point:
                    model_progress.frozen = False
                    restart_anderson(model_index)
                else:
                    continue
            elif at_fixed_point:
                if model_progress.full_variance:
                    model_progress.frozen = True
                    continue
                model_progress.full_variance = True
                restart_anderson(model_index)
            mapped_hyperparameters = _cavi_step(
                hyperparameters=hyperparameters[model_index],
                e_step_inputs=e_step_inputs[model_index],
                beta=state.beta[:, model_index],
                beta_variance=(
                    moments.end_variance[:, model_index]
                    if model_progress.full_variance
                    else moments.end_block_variance[:, model_index]
                ),
                residual_sum_squares=float(moments.end_residual_sum_squares[model_index]),
                training_sample_count=training_sample_count[model_index],
                covariate_count=covariate_count,
                prior_design=prior_design,
                config=model.config,
            )
            model_coordinates = coordinates[model_index]
            if model_coordinates is None or not anderson_states[model_index].residuals:
                model_coordinates = _AndersonCoordinates(
                    prior_variances=e_step_inputs[model_index].prior_variances,
                    covariate_scale=covariate_scale[:, state.mask_index[model_index]],
                    sample_count=int(state.targets.shape[0]),
                    hyperparameters=hyperparameters[model_index],
                )
                coordinates[model_index] = model_coordinates
            start_vector = model_coordinates.pack(
                beta=start_beta[:, model_index],
                alpha=start_alpha[:, model_index],
                linear_predictor=start_linear_predictor[:, model_index],
                hyperparameters=hyperparameters[model_index],
            )
            mapped_vector = model_coordinates.pack(
                beta=state.beta[:, model_index],
                alpha=swept_alpha[:, model_index],
                linear_predictor=swept_linear_predictor[:, model_index],
                hyperparameters=mapped_hyperparameters,
            )
            residual_norm = float(np.linalg.norm(model_coordinates.residual_weights * (mapped_vector - start_vector)))
            restart_vector = best_mapped_vector[model_index]
            if restart_vector is not None and residual_norm > _ANDERSON_DIVERGENCE_GROWTH * smallest_residual_norm[model_index]:
                restart_anderson(model_index)
                next_vector = restart_vector
            elif residual_norm > _ANDERSON_RESTART_GROWTH * smallest_residual_norm[model_index]:
                restart_anderson(model_index)
                next_vector = mapped_vector
            else:
                if residual_norm < smallest_residual_norm[model_index]:
                    smallest_residual_norm[model_index] = residual_norm
                    best_mapped_vector[model_index] = mapped_vector
                next_vector = anderson_step(
                    anderson_states[model_index],
                    x_current=start_vector,
                    map_value=mapped_vector,
                    residual_weights=model_coordinates.residual_weights,
                )
            (
                next_beta,
                next_alpha,
                next_linear_predictor,
                hyperparameters[model_index],
            ) = model_coordinates.unpack(next_vector, model.config)
            state.beta[:, model_index] = next_beta
            state.alpha[:, model_index] = device.array_module.asarray(next_alpha)
            state.linear_predictor[:, model_index] = device.array_module.asarray(next_linear_predictor)
        certified_count = sum(result is not None for result in certified_results)
        uncertified = [
            certificate
            for certificate, result in zip(last_certificates, certified_results)
            if certificate is not None and result is None
        ]
        log(
            f"  exact polish pass {pass_index + 1}: certified {certified_count}/{state.model_count}  "
            f"frozen {sum(model_progress.frozen for model_progress in progress)}  "
            f"max_gradient={max((certificate.gradient_relative_norm for certificate in uncertified), default=0.0):.2e}  "
            f"max_probe={max((certificate.maximum_probe_residual for certificate in uncertified), default=0.0):.2e}  "
            f"max_fixed_point={max((certificate.fixed_point_residual for certificate in uncertified), default=0.0):.2e}"
        )
        if certified_count == state.model_count:
            break
    results: list[PolishedFit] = []
    final_alpha = device.to_host(state.alpha)
    final_linear_predictor = device.to_host(state.linear_predictor)
    for model_index, model in enumerate(models):
        result = certified_results[model_index]
        if result is None:
            certificate = last_certificates[model_index]
            assert certificate is not None
            log(
                f"  ERROR: exact polish did not certify model {model_index} in {_MAXIMUM_POLISH_PASSES} passes "
                f"(gradient={certificate.gradient_relative_norm:.2e}, "
                f"kkt={certificate.maximum_kkt_residual:.2e}, "
                f"probe={certificate.maximum_probe_residual:.2e}, "
                f"fixed_point={certificate.fixed_point_residual:.2e})"
            )
            result = _polished_fit(
                alpha=final_alpha[:, model_index],
                beta=state.beta[:, model_index],
                beta_variance=np.full(variant_count, np.nan, dtype=np.float64),
                linear_predictor=final_linear_predictor[:, model_index],
                hyperparameters=hyperparameters[model_index],
                e_step_inputs=_e_step_inputs(hyperparameters[model_index], prior_design, model.config),
                certificate=certificate,
            )
        results.append(result)
    return results

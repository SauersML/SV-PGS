"""Stage 1 of the fast architecture: the per-trait, per-fold empirical-Bayes fit in LD space.

Stage 0 reduces the genotypes to phenotype-independent LD blocks, and each trait
and fold to one score vector. This module fits SV-PGS's global-local prior to
those sufficient statistics. Its cost depends on the LD block sizes, not on the
number of samples.

Model. Genotype columns are standardized and the covariates W are profiled out
(Frisch-Waugh-Lovell), so ``R_b = X̃_bᵀX̃_b / n`` is block b's projected LD
matrix and ``g = X̃ᵀr`` is the score of the covariate residual r:

    quantitative:  g_b | β_b ~ N(nR_bβ_b, σ²_eff nR_b),  σ²_eff = rᵀr / (n − q)
    binary:        the mean-weight IRLS working model at β₀ (the LD stays shared):
                   ℓ(β) ≈ βᵀX̃ᵀ(y − π + w̄X̃β₀) − ½ w̄ βᵀ nR β,  w̄ = mean π(1 − π)
    β_j | λ_j ~ N(0, u_j λ_j),   log u_j = level + o_j + d_jᵀθ,   θ ~ N(θ₀, diag(1/P))
    λ_j ~ BetaPrime(a, b_c(j))   (the TPB: λ | δ ~ Gamma(a, rate δ), δ ~ Gamma(b, 1))

o_j is a fixed offset. With o_j = log r̂²_j the prior is on the effect per SD of
the true genotype, which is the imputation-reliability factor. d_j is the
annotation hypermodel row; it holds the class offsets. The spike shape a is
pooled and fixed because it is not identified against the level (theory-inference
E1b: corr(log τ, log a) = −0.999). The tail shapes b_c are fitted per class and
partially pooled.

After the covariate and PC projection the LD between blocks is taken to be zero.
That is not exact in a sample: with p ≫ n every block's score also carries the
other blocks' effects through the sampling correlations R_bc ~ O(1/√n), which
add ≈ n h²_outside to each score's variance. So the block likelihood is the
summary-statistics one: its noise is the covariate-residual variance
σ²_eff = σ² + h²_outside ≈ rᵀr / (n − q), held fixed. The residual-based σ²
update (RSS + σ² Σ_j (1 − P_j Σ_jj)) / (n − q) is exact only for exactly
block-orthogonal designs; under the block-diagonal LD it counts every block's
fit of that cross-block signal as explained, and it went negative (−0.76) at
n = 600, p = 1500. The exact σ² belongs to Stage 2 on individual-level data.

Given the hyperparameters every block posterior is exact and independent:
a Cholesky factor of ``κ nR_b + diag(precision)``, with the diagonal of the
inverse from the triangular inverse of that factor. The block loop is the outer
loop, so one pass over the LD store serves every model (trait x fold). On a
CUDA budget the blocks run in turn on the device, and so do the M-step
objectives; with several devices each takes blocks from a shared queue. A
device whose fp64 Cholesky is at least 3x slower than its fp32 one (T4, L4:
1/32 rate) factors a Jacobi-scaled system in fp32 instead; see
_CudaBackend._single_precision_posterior. On the host the blocks run
concurrently on worker threads with
single-threaded BLAS: OpenBLAS's own threading reaches 8-31 GFLOP/s on a
4096-wide potrf at 16-32 threads (EPYC 7763), against 16 GFLOP/s per core
single-threaded, so thousands of independent blocks scale by the core count.
The host factorization uses NumPy, whose linalg releases the GIL; SciPy's f2py
LAPACK wrappers hold it, and eight threads of potrf plus trtri ran no faster
than one.

Schedule. The hyperparameters are a few dozen numbers, so they are fitted on a
phenotype-independent random subset of the blocks (at least 2M variants and 20k
of every class, or everything that exists), where each pass runs every block's
local iterations and then the M-steps. The remaining blocks are then fitted
once with the hyperparameters fixed, each iterated to its own convergence.
When the subset is every block (up to 2M variants) this is the full EB fit.
Stage 2 refits the hyperparameters on all the individual-level data.

Inference. The scheme is chosen in one place, ``_local_scheme``:

- ``expectation_propagation`` (the default): Gaussian sites are fitted by the
  exact tilted moments of the scale-mixture prior on a log-λ grid. The
  hyperparameters are the type-II maximum-likelihood fit of the EP cavity
  marginal Σ_j log Z_j (theory-inference REPORT §8, clipped sites).
- ``coherent_vb``: the q(β)q(λ)q(δ) mean field. GIG moments come from Bessel
  ratios, the level uses E[β²]E[1/λ], and the shapes use E[log δ]
  (Minka's inverse digamma).
- ``plug_in``: coherent_vb with E[λ] plugged in, for the scheme comparison.

Every scheme uses the exact block posterior diagonal, never the prior variance.
"""

from __future__ import annotations

import contextlib
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Protocol, Sequence

import numpy as np
from scipy.special import betaincinv, betaln, digamma, gammaln, kve, polygamma
from threadpoolctl import threadpool_limits

from sv_pgs._typing import F64Array, I64Array
from sv_pgs.anderson import AndersonState, anderson_step
from sv_pgs.compute_budget import ComputeBudget
from sv_pgs.config import ModelConfig
from sv_pgs.genotype import _try_import_cupy
from sv_pgs.genotype_statistics import GenotypeSufficientStatistics
from sv_pgs.mixture_inference import PriorDesign, _gig_moment, _scale_model_penalty
from sv_pgs.progress import log

# EP damping of the site parameters. All sites of a block update in parallel
# from one factorization, and a signal shared by k strongly correlated variants
# makes the damped update stable only for damping below about 2/k: with AR(0.99)
# LD (≈10 variants at r > 0.9) a fixed 0.5 left posterior means flipping between
# neighbours every pass. So each site keeps its own damping, which halves (down
# to the floor) whenever the site's update reverses direction and otherwise
# recovers towards the start value; the EP fixed point is unchanged.
_SITE_DAMPING = 0.5
_MINIMUM_SITE_DAMPING = 1.0 / 64.0
_SITE_DAMPING_RECOVERY = 1.2
# Local (site or mean-field) iterations per block per pass over the LD store.
# The hyperparameters stay fixed within a pass, so a resident block can be
# iterated without re-reading it. theory-inference refit the hyperparameters
# every 5 sweeps.
_LOCAL_ITERATIONS_PER_PASS = 5
_MAXIMUM_PASSES = 60
# The hyperparameter subset: SE(log b) is ≈ 0.02 per 1e5 independent variants at
# biobank per-variant signal (theory-inference E1b), so 2M variants leave ≈ 0.005,
# and 20k variants per class ≈ 0.045. The block order is a fixed permutation, so
# the subset does not depend on the phenotype.
_HYPERPARAMETER_SUBSET_VARIANTS = 2_000_000
_HYPERPARAMETER_SUBSET_CLASS_VARIANTS = 20_000
_HYPERPARAMETER_SUBSET_SEED = 20260918
# With the hyperparameters fixed a block is iterated until its posterior mean
# and variance move less than the convergence tolerance, at most this often.
_MAXIMUM_LOCAL_ITERATIONS = 200
# Convergence: one pass moves every hyperparameter less than this on its log
# scale (level, θ, log b), and the posterior mean less than this relative to its
# norm.
_CONVERGENCE_TOLERANCE = 1e-5
# One pass is an EM-like map of the hyperparameters; plain iteration converges
# linearly at rate ≈ 0.9 (the level of a sparse prior carries most of its
# information in the unobserved local scales). Anderson(5) on the hyperparameter
# vector keeps the fixed point and cut the passes from 64-74 to 12-14 in tests.
_ANDERSON_MEMORY = 5
# Safeguard: every hyperparameter is on a log scale, and an extrapolation that
# moves one more than this beyond the plain map value in a pass (e² ≈ 7x), or a
# non-finite one, is replaced by the plain value and the history restarts.
# Coherent VB, which has no interior fixed point, otherwise ran to 5e56 in the
# level. A bound relative to the plain step would block correct extrapolation of
# a slow EM (rate 0.95 extrapolates ≈ 20 plain steps).
_ANDERSON_MAXIMUM_LOG_EXTRAPOLATION = 2.0
_MAXIMUM_NEWTON_ITERATIONS = 50
_NEWTON_STEP_TOLERANCE = 1e-9
_MINIMUM_LINE_SEARCH_STEP = 1e-6

# The log-λ grid for the EP tilted moments: the BetaPrime density in t = log λ,
# λ^a (1 + λ)^(−a−b) / B(a, b), times the step at each grid point. The integrand
# is analytic in t and vanishes at both ends, so the rule is spectrally accurate
# (1.5e-9 relative in the tilted mean and 1.5e-8 in the variance at step 0.5,
# against adaptive quadrature; step 0.75 gives 1e-6). The lower end leaves the
# spike mass exp(a t_min) / a = 2e-9 / a. The upper end drops λ > e^50, where
# the likelihood decays like λ^(−1/2) and the prior like λ^(−b); at e^40 that
# still cost 4e-7 in log Z for b = 0.1 at |z| = 30. With the exact normalizer
# B(a, b), Σ_j log Z_j is the continuous prior's cavity marginal likelihood, and
# its b-derivatives are closed form.
_GRID_LOWER_TAIL_LOG_MASS = 20.0
_GRID_UPPER_LOG_LOCAL_SCALE = 50.0
_GRID_STEP = 0.5
# A cavity precision 1/Σ_jj − τ̃_j is a difference; values this far below zero
# relative to 1/Σ_jj are rounding of a zero precision (the data do not identify
# β_j given the other sites) and are set to zero.
_CAVITY_ROUNDING_TOLERANCE = 1e-8
# Tail shapes are kept inside the range the grid resolves.
_MINIMUM_SHAPE_B = 0.1
_MAXIMUM_SHAPE_B = 10.0

# The moment initialization of the prior level assumes at least this signal:
# κ Σ_j u_j λ_typical ≥ 1e-3 (the prior genetic variance over the noise variance).
_MINIMUM_INITIAL_SIGNAL_RATIO = 1e-3
# Bound on the temporaries (variants x grid points) of the hyperparameter
# objectives, as a fraction of the working memory: the host's share is split
# between the worker threads; on a device the block loop is idle during M-steps.
_HOST_OBJECTIVE_MEMORY_FRACTION = 0.05
_DEVICE_OBJECTIVE_MEMORY_FRACTION = 0.25
# Float64 temporaries of one (variants x grid) objective evaluation.
_OBJECTIVE_TEMPORARIES = 6
# M-step chunks per worker, so that every worker thread gets work (memory alone
# gave 7 chunks for 1M variants and 64 workers).
_OBJECTIVE_CHUNKS_PER_WORKER = 4
# Float64 (width x width) arrays one host block worker holds: the LD block, the
# system and its factor, and the triangular inverse.
_HOST_BLOCK_WORKER_MATRICES = 4
# Panel width of the host triangular inverse: each panel is one GEMM against the
# inverse so far, so the inverse runs at single-core GEMM speed.
_HOST_INVERSE_PANEL_WIDTH = 256
# The device posterior diagonal takes the triangular inverse's columns in this
# many panels; the trailing-triangle solves then cost ≈ 0.37 p³ against the
# 1/3 p³ of LAPACK trtri, which CuPy does not expose.
_DIAGONAL_PANEL_COUNT = 16
# A device factors in fp32 when its fp64 Cholesky is at least this much slower
# (measured once per fit on a block-sized system): the fp32 path adds a few
# p² passes and two fp64 refinement matrix-vector products per solve.
_SINGLE_PRECISION_MINIMUM_SPEEDUP = 3.0
_SINGLE_PRECISION_REFINEMENT_STEPS = 2
# Half-width of the Richardson-extrapolated central difference in the Bessel
# order for E[log λ]: truncation error O(1e-12), rounding about eps |log K| / 1e-3.
_BESSEL_ORDER_DIFFERENCE_STEP = 1e-3


class LDBlockSource(Protocol):
    """Phenotype-independent LD from Stage 0.

    Blocks are contiguous variant ranges ``block_boundaries[i]:block_boundaries[i + 1]``.
    Each block is the standardized, covariate-projected LD matrix
    ``R_b = X̃_bᵀX̃_b / n`` (symmetric positive definite, float32 or float64).
    """

    @property
    def block_boundaries(self) -> I64Array: ...

    def correlation_block(self, block_index: int) -> NDArrayLike: ...

    def ld_diagonal(self) -> F64Array:
        """R_jj for every variant (at most 1 after the covariate projection)."""
        ...

    def ld_scores(self) -> F64Array:
        """Σ_i R_ij² for every variant j (its within-block LD score)."""
        ...


NDArrayLike = Any


@dataclass(frozen=True, slots=True)
class InMemoryLDBlocks:
    block_boundaries: I64Array
    correlation_blocks: tuple[F64Array, ...]

    def __post_init__(self) -> None:
        boundaries = np.asarray(self.block_boundaries)
        if boundaries.ndim != 1 or boundaries.shape[0] != len(self.correlation_blocks) + 1 or boundaries[0] != 0:
            raise ValueError("block_boundaries must start at 0 and have one more entry than there are blocks.")
        for block_index, block in enumerate(self.correlation_blocks):
            width = int(boundaries[block_index + 1] - boundaries[block_index])
            if width <= 0 or block.shape != (width, width):
                raise ValueError(f"LD block {block_index} has shape {block.shape}; its boundaries imply {width}.")
            if not np.array_equal(block, block.T):
                raise ValueError(f"LD block {block_index} is not exactly symmetric.")

    def correlation_block(self, block_index: int) -> F64Array:
        return self.correlation_blocks[block_index]

    def ld_diagonal(self) -> F64Array:
        return np.concatenate([np.diagonal(block) for block in self.correlation_blocks]).astype(np.float64)

    def ld_scores(self) -> F64Array:
        return np.concatenate(
            [np.einsum("ij,ij->j", block, block) for block in self.correlation_blocks]
        ).astype(np.float64)


@dataclass(frozen=True, slots=True)
class QuantitativeTraitStatistics:
    """One quantitative trait (and fold) reduced to its sufficient statistics.

    ``score`` is X̃ᵀr in standardized-genotype units and ``residual_sum_of_squares``
    is rᵀr, where r is the OLS residual of the phenotype on the covariates.
    ``covariate_count`` counts the columns of W, the intercept included. The
    block likelihood's noise variance is rᵀr / (n − q) (see the module notes).
    """

    sample_count: int
    covariate_count: int
    score: F64Array
    residual_sum_of_squares: float


@dataclass(frozen=True, slots=True)
class BinaryTraitStatistics:
    """One binary trait (and fold) as the mean-weight IRLS working model at β₀.

    With α refit given the offset X̃β₀ and π the fitted probabilities (so
    Wᵀ(y − π) = 0), the working response is z = X̃β₀ + (y − π)/w̄, w̄ the sample
    mean of π(1 − π), and ``score`` is w̄X̃ᵀz = X̃ᵀ(y − π) + w̄X̃ᵀX̃β₀. The first fit
    has β₀ = 0; one weight refresh re-expands at its posterior mean
    (``binary_statistics_at``). The refresh uses the full-data X̃ᵀX̃β₀ from the
    same genotype pass, not the block-diagonal nRβ₀: the exact score already
    subtracts the cross-block part of X̃ᵀX̃β₀, and adding back only the
    block-diagonal part would be a block-Jacobi step through the cross-block
    sampling LD (it cost 0.011 AUC at n = p = 1500).
    """

    sample_count: int
    covariate_count: int
    score: F64Array
    mean_fisher_weight: float


TraitStatistics = QuantitativeTraitStatistics | BinaryTraitStatistics


def quantitative_trait_statistics(statistics: GenotypeSufficientStatistics, trait_index: int) -> QuantitativeTraitStatistics:
    """One quantitative trait from Stage 0, whose target column ``trait_index`` is the phenotype.

    The score is Stage 0's projected score X̃ᵀỹ; rᵀr = ỹᵀỹ = yᵀy − (Wᵀy)ᵀ(WᵀW)⁻¹Wᵀy.
    """
    ld = statistics.ld
    covariate_target = statistics.covariate_target[:, trait_index]
    return QuantitativeTraitStatistics(
        sample_count=statistics.sample_count,
        covariate_count=int(statistics.covariate_gram.shape[0]),
        score=np.concatenate([ld.block(block_index).projected_score[:, trait_index] for block_index in range(ld.block_count)]),
        residual_sum_of_squares=float(
            statistics.target_gram[trait_index, trait_index]
            - covariate_target @ np.linalg.solve(statistics.covariate_gram, covariate_target)
        ),
    )


def binary_statistics_at(
    *,
    residual_score: F64Array,
    gram_times_expansion: F64Array,
    fitted_probability: F64Array,
    covariate_count: int,
) -> BinaryTraitStatistics:
    """The working model at an expansion point β₀.

    ``fitted_probability`` is π = sigmoid(Wα + X̃β₀) with α refit given the
    offset X̃β₀, ``residual_score`` is X̃ᵀ(y − π) and ``gram_times_expansion`` is
    X̃ᵀX̃β₀ (zero for the first fit); all three come from one genotype pass.
    """
    probability = np.asarray(fitted_probability, dtype=np.float64)
    mean_fisher_weight = float(np.mean(probability * (1.0 - probability)))
    return BinaryTraitStatistics(
        sample_count=int(probability.shape[0]),
        covariate_count=covariate_count,
        score=np.asarray(residual_score, dtype=np.float64) + mean_fisher_weight * np.asarray(gram_times_expansion, dtype=np.float64),
        mean_fisher_weight=mean_fisher_weight,
    )


@dataclass(frozen=True, slots=True)
class LDPriorHypermodel:
    """The prior's annotation hypermodel, shared by every trait fit on these variants.

    log u_j = level + log_variance_offset_j + annotation_design_j · θ, with
    θ ~ N(annotation_prior_mean, diag(1 / annotation_prior_precision)); the level
    is fitted with a flat prior. ``variant_class_index`` picks the class of each
    variant's local-scale prior BetaPrime(shape_a, shape_b[class]); the log tail
    shapes are pooled around their mean with variance ``shape_b_pooling_variance``.
    """

    annotation_design: F64Array
    annotation_prior_mean: F64Array
    annotation_prior_precision: F64Array
    log_variance_offset: F64Array
    variant_class_index: I64Array
    class_names: tuple[str, ...]
    shape_a: float
    initial_shape_b: F64Array
    shape_b_pooling_variance: float

    def __post_init__(self) -> None:
        variant_count = self.log_variance_offset.shape[0]
        feature_count = self.annotation_design.shape[1]
        class_count = len(self.class_names)
        if self.annotation_design.shape[0] != variant_count or self.variant_class_index.shape != (variant_count,):
            raise ValueError("The hypermodel's per-variant arrays must all have one row per variant.")
        if self.annotation_prior_mean.shape != (feature_count,) or self.annotation_prior_precision.shape != (feature_count,):
            raise ValueError("The annotation prior needs one mean and one precision per design column.")
        if np.any(self.annotation_prior_precision <= 0.0):
            raise ValueError("Every annotation coefficient needs a proper prior (positive precision).")
        if self.initial_shape_b.shape != (class_count,):
            raise ValueError("initial_shape_b needs one tail shape per class.")
        if np.any(self.variant_class_index < 0) or np.any(self.variant_class_index >= class_count):
            raise ValueError("variant_class_index must index class_names.")
        if not (self.shape_a > 0.0 and self.shape_b_pooling_variance > 0.0):
            raise ValueError("shape_a and shape_b_pooling_variance must be positive.")

    @property
    def variant_count(self) -> int:
        return int(self.log_variance_offset.shape[0])

    def log_prior_variance(self, log_variance_level: float, coefficients: F64Array, variants: slice) -> F64Array:
        return np.asarray(
            log_variance_level + self.log_variance_offset[variants] + self.annotation_design[variants] @ coefficients,
            dtype=np.float64,
        )


def hypermodel_from_prior_design(
    prior_design: PriorDesign,
    config: ModelConfig,
    log_variance_offset: F64Array,
    shape_a: float,
) -> LDPriorHypermodel:
    """The Stage 1 hypermodel on today's prior design (class offsets, factor levels, splines).

    The design's coefficients act on the log prior SD with ridge precisions from
    ``config``; on the log variance they are doubled, so their precisions are
    quartered. Each variant needs exactly one prior class.
    """
    membership = np.asarray(prior_design.class_membership_matrix, dtype=np.float64)
    if not (np.all((membership == 0.0) | (membership == 1.0)) and np.all(membership.sum(axis=1) == 1.0)):
        raise ValueError("Stage 1 needs one prior class per variant; soft class membership is not supported.")
    classes = [prior_design.inverse_class_lookup[class_position] for class_position in range(membership.shape[1])]
    default_shape_b = config.class_tpb_shape_b()
    feature_count = prior_design.design_matrix.shape[1]
    return LDPriorHypermodel(
        annotation_design=np.asarray(prior_design.design_matrix, dtype=np.float64),
        annotation_prior_mean=np.zeros(feature_count),
        annotation_prior_precision=_scale_model_penalty(prior_design.feature_names, config) / 4.0,
        log_variance_offset=np.asarray(log_variance_offset, dtype=np.float64),
        variant_class_index=np.argmax(membership, axis=1).astype(np.int64),
        class_names=tuple(variant_class.value for variant_class in classes),
        shape_a=shape_a,
        initial_shape_b=np.array([default_shape_b[variant_class] for variant_class in classes], dtype=np.float64),
        shape_b_pooling_variance=config.tpb_hierarchical_prior_variance,
    )


@dataclass(slots=True)
class ExpectationPropagationSites:
    """Gaussian sites exp(−½ τ̃_j β_j² + ν̃_j β_j) and the cavities they last produced.

    Cavities are in natural parameters exp(−½ P_j β_j² + h_j β_j), so a cavity
    with P_j = 0 (the data do not identify β_j given the other sites) is exact.
    """

    site_precision: F64Array
    site_shift: F64Array
    cavity_precision: F64Array
    cavity_shift: F64Array
    site_damping: F64Array
    previous_precision_step: F64Array
    previous_shift_step: F64Array


@dataclass(slots=True)
class MeanFieldMoments:
    """q(λ_j) = GIG and q(δ_j) = Gamma moments, and E[β_j²] from the last solve."""

    expected_local_scale: F64Array
    expected_inverse_local_scale: F64Array
    expected_log_local_scale: F64Array
    expected_auxiliary_rate: F64Array
    expected_log_auxiliary_rate: F64Array
    coefficient_second_moment: F64Array


LocalState = ExpectationPropagationSites | MeanFieldMoments


@dataclass(frozen=True, slots=True)
class LDSpaceFit:
    """A converged Stage 1 fit for one trait and fold (standardized-genotype units)."""

    scheme: str
    posterior_mean: F64Array
    posterior_variance: F64Array
    log_variance_level: float
    annotation_coefficients: F64Array
    shape_a: float
    shape_b: F64Array
    likelihood_precision: float
    local_state: LocalState
    passes: int
    converged: bool


@dataclass(slots=True)
class _ModelState:
    statistics: TraitStatistics
    log_variance_level: float
    annotation_coefficients: F64Array
    shape_b: F64Array
    likelihood_precision: float
    local: LocalState
    posterior_mean: F64Array
    posterior_variance: F64Array
    acceleration: AndersonState
    warm_started: bool
    passes: int = 0
    converged: bool = False


# --------------------------------------------------------------------------- array backends


class _ArrayBackend(Protocol):
    """Where the block posteriors and site updates run: host LAPACK or a CUDA device.

    ``map_blocks`` runs independent LD blocks: on worker threads on the host
    (NumPy's linalg releases the GIL), on one thread per device on CUDA.
    ``map_chunks`` runs M-step variant chunks: on the worker threads on the host,
    in turn on the primary device on CUDA. Work items create their own device
    arrays; none is shared between items.
    """

    @property
    def xp(self) -> Any:
        """The array module: numpy or cupy."""
        ...

    objective_bytes: int
    chunk_workers: int

    def map_blocks(self, function: Callable[[Any], Any], items: Sequence[Any]) -> list[Any]: ...

    def map_chunks(self, function: Callable[[Any], Any], items: Sequence[Any]) -> list[Any]: ...

    def to_device(self, values: NDArrayLike) -> NDArrayLike: ...

    def to_host(self, values: NDArrayLike) -> F64Array: ...

    def block_posterior(
        self,
        correlation_block: NDArrayLike,
        likelihood_scale: float,
        diagonal_precision: NDArrayLike,
        linear_term: NDArrayLike,
    ) -> tuple[NDArrayLike, NDArrayLike]:
        """Mean and diag(A⁻¹) for A = likelihood_scale · R_b + diag(precision)."""
        ...


class _HostBackend:
    xp = np

    def __init__(self, executor: ThreadPoolExecutor, worker_count: int, host_bytes: int) -> None:
        self._executor = executor
        self.chunk_workers = worker_count
        self.objective_bytes = int(_HOST_OBJECTIVE_MEMORY_FRACTION * host_bytes / worker_count)

    def map_blocks(self, function: Callable[[Any], Any], items: Sequence[Any]) -> list[Any]:
        return list(self._executor.map(function, items))

    def map_chunks(self, function: Callable[[Any], Any], items: Sequence[Any]) -> list[Any]:
        return list(self._executor.map(function, items))

    def to_device(self, values: NDArrayLike) -> F64Array:
        return np.asarray(values, dtype=np.float64)

    def to_host(self, values: NDArrayLike) -> F64Array:
        return np.asarray(values, dtype=np.float64)

    def block_posterior(
        self,
        correlation_block: F64Array,
        likelihood_scale: float,
        diagonal_precision: F64Array,
        linear_term: F64Array,
    ) -> tuple[F64Array, F64Array]:
        # diag(A⁻¹)_j = Σ_i (L⁻¹)_ij², and the mean is L⁻ᵀ(L⁻¹ h).
        system = likelihood_scale * correlation_block
        system[np.diag_indices_from(system)] += diagonal_precision
        inverse_factor = _lower_triangular_inverse(np.linalg.cholesky(system))
        mean = inverse_factor.T @ (inverse_factor @ linear_term)
        return mean, np.einsum("ij,ij->j", inverse_factor, inverse_factor)


def _lower_triangular_inverse(factor: F64Array) -> F64Array:
    """L⁻¹ by block forward substitution.

    Panel by panel, (L⁻¹)_ss = L_ss⁻¹ and (L⁻¹)_s,<s = −L_ss⁻¹ L_s,<s (L⁻¹)_<s,<s, all
    from NumPy matmul and inv, which release the GIL.
    """
    width = factor.shape[0]
    inverse = np.zeros_like(factor)
    for start in range(0, width, _HOST_INVERSE_PANEL_WIDTH):
        stop = min(start + _HOST_INVERSE_PANEL_WIDTH, width)
        diagonal_inverse = np.tril(np.linalg.inv(factor[start:stop, start:stop]))
        inverse[start:stop, start:stop] = diagonal_inverse
        if start > 0:
            inverse[start:stop, :start] = -diagonal_inverse @ (factor[start:stop, :start] @ inverse[:start, :start])
    return inverse


class _CudaBackend:
    def __init__(
        self, cupy: Any, primary_device: int, device_bytes: int, executor: ThreadPoolExecutor, single_precision: bool
    ) -> None:
        # cupyx ships with CuPy, which _array_backend has already imported.
        from cupyx import errstate
        from cupyx.scipy.linalg import solve_triangular

        self.xp = cupy
        self.objective_bytes = int(_DEVICE_OBJECTIVE_MEMORY_FRACTION * device_bytes)
        self.chunk_workers = 1
        self.single_precision = single_precision
        self._primary_device = primary_device
        self._executor = executor
        # CuPy ignores cuSOLVER failures unless asked, which would hand back a
        # partial factor of an indefinite system.
        self._raise_on_linalg_error = errstate
        self._solve_triangular = solve_triangular

    def map_blocks(self, function: Callable[[Any], Any], items: Sequence[Any]) -> list[Any]:
        return list(self._executor.map(function, items))

    def map_chunks(self, function: Callable[[Any], Any], items: Sequence[Any]) -> list[Any]:
        with self.xp.cuda.Device(self._primary_device):
            return [function(item) for item in items]

    def to_device(self, values: NDArrayLike) -> NDArrayLike:
        return self.xp.asarray(values).astype(self.xp.float64, copy=False)

    def to_host(self, values: NDArrayLike) -> F64Array:
        return np.asarray(self.xp.asnumpy(values), dtype=np.float64)

    def block_posterior(
        self,
        correlation_block: NDArrayLike,
        likelihood_scale: float,
        diagonal_precision: NDArrayLike,
        linear_term: NDArrayLike,
    ) -> tuple[NDArrayLike, NDArrayLike]:
        if self.single_precision:
            return self._single_precision_posterior(correlation_block, likelihood_scale, diagonal_precision, linear_term)
        cupy = self.xp
        width = int(correlation_block.shape[0])
        system = likelihood_scale * correlation_block
        system[cupy.arange(width), cupy.arange(width)] += diagonal_precision
        with self._raise_on_linalg_error(linalg="raise"):
            factor = cupy.linalg.cholesky(system)
        mean = self._solve_triangular(factor, linear_term, lower=True)
        mean = self._solve_triangular(factor, mean, lower=True, trans="T")
        variance = cupy.empty(width, dtype=cupy.float64)
        for start, stop, columns in self._inverse_factor_panels(factor):
            variance[start:stop] = cupy.sum(columns * columns, axis=0)
        return mean, variance

    def _inverse_factor_panels(self, factor: NDArrayLike) -> Iterator[tuple[int, int, NDArrayLike]]:
        """Column panels of L⁻¹ below the diagonal: L⁻¹ e_j vanishes above row j,
        so each panel needs only the trailing triangle of the factor."""
        cupy = self.xp
        width = int(factor.shape[0])
        panel_width = -(-width // _DIAGONAL_PANEL_COUNT)
        for start in range(0, width, panel_width):
            stop = min(start + panel_width, width)
            identity_panel = cupy.zeros((width - start, stop - start), dtype=factor.dtype)
            identity_panel[cupy.arange(stop - start), cupy.arange(stop - start)] = 1.0
            yield start, stop, self._solve_triangular(factor[start:, start:], identity_panel, lower=True)

    def _single_precision_posterior(
        self,
        correlation_block: NDArrayLike,
        likelihood_scale: float,
        diagonal_precision: NDArrayLike,
        linear_term: NDArrayLike,
    ) -> tuple[NDArrayLike, NDArrayLike]:
        """The block posterior from an fp32 factor of the Jacobi-scaled system Ã = SAS, S = diag(A)^(−1/2).

        The EP cavity precision 1/Σ_jj − τ̃_j is a small difference for
        prior-dominated variants, so Σ_jj must be accurate relative to its
        excess over the prior. With a unit diagonal, [Ã⁻¹]_jj = 1 + e_j where
        e_j = Σ_{k<j} L̃_jk² / L̃_jj² + Σ_{i>j} (L̃⁻¹)_ij² is a sum of squares, so
        it keeps fp32 relative accuracy however small it is (against fp64: 9e-7
        in Σ_jj and 7e-6 in the cavity precision on a 3000-wide block with
        r = 0.9995 pairs; the plain 1/Σ_jj − τ̃_j in fp32 would lose a factor
        τ̃_j / (data precision) ≈ 10²-10⁴). The mean gets fp64 residual refinement.
        """
        cupy = self.xp
        width = int(correlation_block.shape[0])
        diagonal_index = cupy.arange(width)
        root = 1.0 / cupy.sqrt(likelihood_scale * cupy.diagonal(correlation_block) + diagonal_precision)
        scaled = (likelihood_scale * correlation_block) * root[:, None] * root[None, :]
        scaled[diagonal_index, diagonal_index] = 1.0
        with self._raise_on_linalg_error(linalg="raise"):
            factor = cupy.linalg.cholesky(scaled.astype(cupy.float32))
        factor_diagonal = cupy.diagonal(factor).astype(cupy.float64)
        below_diagonal = cupy.tril(factor, -1).astype(cupy.float64)
        excess = cupy.sum(below_diagonal * below_diagonal, axis=1) / (factor_diagonal * factor_diagonal)
        for start, stop, columns in self._inverse_factor_panels(factor):
            below = cupy.tril(columns.astype(cupy.float64), -1)
            excess[start:stop] += cupy.sum(below * below, axis=0)

        def scaled_solve(right_hand_side: NDArrayLike) -> NDArrayLike:
            half = self._solve_triangular(factor, (root * right_hand_side).astype(cupy.float32), lower=True)
            return root * self._solve_triangular(factor, half, lower=True, trans="T").astype(cupy.float64)

        mean = scaled_solve(linear_term)
        for _refinement in range(_SINGLE_PRECISION_REFINEMENT_STEPS):
            residual = linear_term - likelihood_scale * (correlation_block @ mean) - diagonal_precision * mean
            mean = mean + scaled_solve(residual)
        return mean, root * root * (1.0 + excess)


def _measured_single_precision_speedup(cupy: Any, width: int) -> float:
    """fp64 over fp32 Cholesky time for one width x width system on the current device."""
    base = cupy.random.default_rng(0).standard_normal((width, width))
    system = base @ base.T / width + cupy.eye(width)
    seconds = []
    for dtype in (cupy.float64, cupy.float32):
        typed = system.astype(dtype)
        cupy.linalg.cholesky(typed)
        cupy.cuda.Device().synchronize()
        start = time.perf_counter()
        cupy.linalg.cholesky(typed)
        cupy.cuda.Device().synchronize()
        seconds.append(time.perf_counter() - start)
    return seconds[0] / seconds[1]


def _host_worker_count(budget: ComputeBudget, widest_block: int) -> int:
    """Concurrent host block workers: one per thread, as many as the memory holds."""
    worker_bytes = _HOST_BLOCK_WORKER_MATRICES * 8 * widest_block * widest_block
    fitting_workers = int(budget.host_bytes // worker_bytes)
    if fitting_workers < 1:
        raise MemoryError(f"One {widest_block}-wide LD block needs {worker_bytes} bytes; the budget has {budget.host_bytes}.")
    return min(budget.cpu_threads, fitting_workers)


@contextlib.contextmanager
def _array_backend(budget: ComputeBudget, widest_block: int) -> Iterator[_ArrayBackend]:
    if budget.device_kind == "cpu":
        worker_count = _host_worker_count(budget, widest_block)
        with ThreadPoolExecutor(max_workers=worker_count) as executor, threadpool_limits(limits=1):
            yield _HostBackend(executor, worker_count, budget.host_bytes)
        return
    cupy = _try_import_cupy()
    if cupy is None:
        raise RuntimeError("The compute budget is CUDA but CuPy cannot be imported.")
    primary_device = budget.device_ids[0]
    with cupy.cuda.Device(primary_device):
        speedup = _measured_single_precision_speedup(cupy, widest_block)
    single_precision = speedup >= _SINGLE_PRECISION_MINIMUM_SPEEDUP
    log(
        f"  Stage 1 on {len(budget.device_ids)} CUDA device(s): fp32 Cholesky {speedup:.1f}x faster than fp64, "
        f"factoring in {'fp32 with fp64 refinement' if single_precision else 'fp64'}"
    )
    unbound_devices: queue.SimpleQueue[int] = queue.SimpleQueue()
    for device_id in budget.device_ids:
        unbound_devices.put(device_id)

    def bind_worker_to_device() -> None:
        cupy.cuda.Device(unbound_devices.get()).use()

    with ThreadPoolExecutor(max_workers=len(budget.device_ids), initializer=bind_worker_to_device) as executor:
        yield _CudaBackend(cupy, primary_device, budget.working_bytes, executor, single_precision)


def _variant_chunks(backend: _ArrayBackend, ranges: Sequence[slice], grid_size: int) -> list[slice]:
    """Pieces of ``ranges`` whose (variants x grid) objective temporaries fit the backend's
    share, and at least a few per worker."""
    variant_count = sum(span.stop - span.start for span in ranges)
    memory_chunk = backend.objective_bytes // (8 * _OBJECTIVE_TEMPORARIES * grid_size)
    balanced_chunk = -(-variant_count // (_OBJECTIVE_CHUNKS_PER_WORKER * backend.chunk_workers))
    chunk_size = max(min(memory_chunk, balanced_chunk), 1)
    return [
        slice(start, min(start + chunk_size, span.stop))
        for span in ranges
        for start in range(span.start, span.stop, chunk_size)
    ]


# --------------------------------------------------------------------------- log-λ grid


@dataclass(frozen=True, slots=True)
class _LogLocalScaleGrid:
    local_scale: F64Array
    log_one_plus_local_scale: F64Array
    class_log_prior_mass: F64Array
    # E_prior[log(1 + λ)] = ψ(a + b) − ψ(b), and its b-derivative ψ'(a + b) − ψ'(b).
    class_prior_mean_log_one_plus: F64Array
    class_prior_mean_log_one_plus_slope: F64Array


def _log_local_scale_grid(shape_a: float, shape_b: F64Array) -> _LogLocalScaleGrid:
    """BetaPrime(a, b_c) as point masses on an even log-λ grid (see _GRID_STEP)."""
    lower = -_GRID_LOWER_TAIL_LOG_MASS / shape_a
    point_count = int(np.ceil((_GRID_UPPER_LOG_LOCAL_SCALE - lower) / _GRID_STEP)) + 1
    log_local_scale = np.linspace(lower, _GRID_UPPER_LOG_LOCAL_SCALE, point_count)
    log_one_plus_local_scale = np.logaddexp(0.0, log_local_scale)
    class_log_prior_mass = (
        shape_a * log_local_scale[None, :]
        - (shape_a + shape_b[:, None]) * log_one_plus_local_scale[None, :]
        - betaln(shape_a, shape_b)[:, None]
        + np.log(log_local_scale[1] - log_local_scale[0])
    )
    return _LogLocalScaleGrid(
        local_scale=np.exp(log_local_scale),
        log_one_plus_local_scale=log_one_plus_local_scale,
        class_log_prior_mass=class_log_prior_mass,
        class_prior_mean_log_one_plus=digamma(shape_a + shape_b) - digamma(shape_b),
        class_prior_mean_log_one_plus_slope=polygamma(1, shape_a + shape_b) - polygamma(1, shape_b),
    )


def _logsumexp_rows(values: NDArrayLike) -> NDArrayLike:
    maximum = values.max(axis=1, keepdims=True)
    return (maximum + np.log(np.exp(values - maximum).sum(axis=1, keepdims=True)))[:, 0]


def _tilted_weights(
    cavity_precision: NDArrayLike,
    cavity_shift: NDArrayLike,
    prior_variance: NDArrayLike,
    log_prior_mass: NDArrayLike,
    local_scale: NDArrayLike,
) -> tuple[NDArrayLike, NDArrayLike, NDArrayLike, NDArrayLike]:
    """Exact tilted distribution of p(β_j) exp(−½ P_j β_j² + h_j β_j) on the log-λ grid.

    Given λ_k the tilted β_j is Gaussian with variance c_k = uλ_k / (1 + q_k),
    q_k = uλ_k P, and mean h c_k. Returns log Z_j up to its hyperparameter-free
    cavity normalizer, the grid weights, c and q. Works on NumPy and CuPy arrays.
    """
    slab_variance = prior_variance[:, None] * local_scale[None, :]
    relative_precision = slab_variance * cavity_precision[:, None]
    component_variance = slab_variance / (1.0 + relative_precision)
    log_weight = (
        log_prior_mass
        - 0.5 * np.log1p(relative_precision)
        + 0.5 * np.square(cavity_shift)[:, None] * component_variance
    )
    log_normalizer = _logsumexp_rows(log_weight)
    weight = np.exp(log_weight - log_normalizer[:, None])
    return log_normalizer, weight, component_variance, relative_precision


def _tilted_mean_and_variance(
    cavity_shift: NDArrayLike,
    weight: NDArrayLike,
    component_variance: NDArrayLike,
) -> tuple[NDArrayLike, NDArrayLike]:
    mean_component_variance = np.sum(weight * component_variance, axis=1)
    component_variance_spread = np.sum(weight * np.square(component_variance - mean_component_variance[:, None]), axis=1)
    return (
        cavity_shift * mean_component_variance,
        mean_component_variance + np.square(cavity_shift) * component_variance_spread,
    )


# --------------------------------------------------------------------------- local schemes


@dataclass(frozen=True, slots=True)
class _DeviceGrid:
    """One model's log-λ grid for one pass, on the backend's device."""

    local_scale: NDArrayLike
    class_log_prior_mass: NDArrayLike


@dataclass(frozen=True, slots=True)
class _BlockPrior:
    """One block's prior for one model and pass, on the backend's device."""

    prior_variance: NDArrayLike
    class_index: I64Array
    log_prior_mass: NDArrayLike
    local_scale: NDArrayLike
    shape_a: float
    shape_b: F64Array


class _LocalScheme(Protocol):
    name: str

    def initial_state(
        self, prior_variance: F64Array, typical_local_scale: F64Array, shape_a: float, shape_b: F64Array
    ) -> LocalState: ...

    def load_block(self, backend: _ArrayBackend, state: LocalState, variants: slice) -> tuple[NDArrayLike, ...]:
        """The block's slice of the local state, where update_block computes."""
        ...

    def reset_range(
        self,
        state: LocalState,
        variants: slice,
        prior_variance: F64Array,
        typical_local_scale: F64Array,
        shape_a: float,
        shape_b: F64Array,
    ) -> None:
        """Restart a variant range from the prior at the typical local scale."""
        ...

    def solve_terms(
        self, backend: _ArrayBackend, block_state: tuple[NDArrayLike, ...], block_prior: _BlockPrior
    ) -> tuple[NDArrayLike, NDArrayLike]:
        """The diagonal precision and linear shift the prior adds to a block solve (fresh arrays)."""
        ...

    def update_block(
        self,
        backend: _ArrayBackend,
        block_state: tuple[NDArrayLike, ...],
        posterior_mean: NDArrayLike,
        posterior_variance: NDArrayLike,
        block_prior: _BlockPrior,
    ) -> tuple[NDArrayLike, ...]: ...

    def store_block(
        self, backend: _ArrayBackend, state: LocalState, variants: slice, block_state: tuple[NDArrayLike, ...]
    ) -> None: ...

    def scale_objective(
        self,
        backend: _ArrayBackend,
        state: LocalState,
        variants: slice,
        log_prior_variance: NDArrayLike,
        class_index: NDArrayLike,
        grid: _DeviceGrid,
    ) -> tuple[float, NDArrayLike, NDArrayLike]:
        """The M-step objective's value and per-variant first and second derivatives in log u_j."""
        ...

    def shape_objective(
        self,
        backend: _ArrayBackend,
        state: LocalState,
        log_prior_variance: F64Array,
        class_index: I64Array,
        shape_a: float,
        shape_b: F64Array,
        chunks: Sequence[slice],
    ) -> tuple[float, F64Array, F64Array]:
        """The M-step objective's value, gradient and Hessian diagonal in the per-class b."""
        ...

    def shape_starting_point(
        self, state: LocalState, shape_b: F64Array, class_index: I64Array, chunks: Sequence[slice]
    ) -> F64Array: ...


def _shape_b_per_variant(shape_b: F64Array, class_index: I64Array) -> F64Array:
    return np.asarray(shape_b[class_index], dtype=np.float64)


class _ExpectationPropagationScheme:
    name = "expectation_propagation"

    def initial_state(
        self, prior_variance: F64Array, typical_local_scale: F64Array, shape_a: float, shape_b: F64Array
    ) -> ExpectationPropagationSites:
        # Start from the Gaussian prior at the typical local scale.
        return ExpectationPropagationSites(
            site_precision=1.0 / (prior_variance * typical_local_scale),
            site_shift=np.zeros_like(prior_variance),
            cavity_precision=np.zeros_like(prior_variance),
            cavity_shift=np.zeros_like(prior_variance),
            site_damping=np.full_like(prior_variance, _SITE_DAMPING),
            previous_precision_step=np.zeros_like(prior_variance),
            previous_shift_step=np.zeros_like(prior_variance),
        )

    def reset_range(
        self,
        state: LocalState,
        variants: slice,
        prior_variance: F64Array,
        typical_local_scale: F64Array,
        shape_a: float,
        shape_b: F64Array,
    ) -> None:
        fresh = self.initial_state(prior_variance, typical_local_scale, shape_a, shape_b)
        sites = _require_sites(state)
        sites.site_precision[variants] = fresh.site_precision
        sites.site_shift[variants] = fresh.site_shift
        sites.site_damping[variants] = fresh.site_damping
        sites.previous_precision_step[variants] = fresh.previous_precision_step
        sites.previous_shift_step[variants] = fresh.previous_shift_step

    def load_block(self, backend: _ArrayBackend, state: LocalState, variants: slice) -> tuple[NDArrayLike, ...]:
        sites = _require_sites(state)
        return tuple(backend.to_device(values[variants]) for values in _site_arrays(sites))

    def solve_terms(
        self, backend: _ArrayBackend, block_state: tuple[NDArrayLike, ...], block_prior: _BlockPrior
    ) -> tuple[NDArrayLike, NDArrayLike]:
        return block_state[0], block_state[1]

    def update_block(
        self,
        backend: _ArrayBackend,
        block_state: tuple[NDArrayLike, ...],
        posterior_mean: NDArrayLike,
        posterior_variance: NDArrayLike,
        block_prior: _BlockPrior,
    ) -> tuple[NDArrayLike, ...]:
        site_precision, site_shift, _cavity_precision, _cavity_shift, damping, previous_precision_step, previous_shift_step = (
            block_state
        )
        marginal_precision = 1.0 / posterior_variance
        cavity_precision = marginal_precision - site_precision
        if bool(np.any(cavity_precision < -_CAVITY_ROUNDING_TOLERANCE * marginal_precision)):
            raise FloatingPointError(
                "An EP cavity precision is negative beyond rounding; the block posterior diagonal is inaccurate."
            )
        cavity_precision = np.maximum(cavity_precision, 0.0)
        cavity_shift = posterior_mean * marginal_precision - site_shift
        _log_normalizer, weight, component_variance, _relative_precision = _tilted_weights(
            cavity_precision, cavity_shift, block_prior.prior_variance, block_prior.log_prior_mass, block_prior.local_scale
        )
        tilted_mean, tilted_variance = _tilted_mean_and_variance(cavity_shift, weight, component_variance)
        # Clipped sites: the precision never goes below zero, so every block system
        # stays positive definite, and the mean is matched exactly through ν̃.
        target_precision = np.maximum(1.0 / tilted_variance - cavity_precision, 0.0)
        target_shift = tilted_mean * (cavity_precision + target_precision) - cavity_shift
        precision_step = target_precision - site_precision
        shift_step = target_shift - site_shift
        reversed_step = (precision_step * previous_precision_step < 0.0) | (shift_step * previous_shift_step < 0.0)
        damping = np.where(
            reversed_step,
            np.maximum(0.5 * damping, _MINIMUM_SITE_DAMPING),
            np.minimum(_SITE_DAMPING_RECOVERY * damping, _SITE_DAMPING),
        )
        return (
            site_precision + damping * precision_step,
            site_shift + damping * shift_step,
            cavity_precision,
            cavity_shift,
            damping,
            precision_step,
            shift_step,
        )

    def store_block(
        self, backend: _ArrayBackend, state: LocalState, variants: slice, block_state: tuple[NDArrayLike, ...]
    ) -> None:
        sites = _require_sites(state)
        for values, block_values in zip(_site_arrays(sites), block_state, strict=True):
            values[variants] = backend.to_host(block_values)

    def scale_objective(
        self,
        backend: _ArrayBackend,
        state: LocalState,
        variants: slice,
        log_prior_variance: NDArrayLike,
        class_index: NDArrayLike,
        grid: _DeviceGrid,
    ) -> tuple[float, NDArrayLike, NDArrayLike]:
        # Σ_j log Z_j with the cavities held fixed. With η = log u, component k has
        # log weight −½ log(1 + q_k) + ½ h² c_k, and with r_k = 1/(1 + q_k):
        # g_k = ∂/∂η = ½ r_k (h² c_k − q_k) and ∂g_k/∂η = ½ r_k² (h² c_k (1 − q_k) − q_k),
        # so ∂ log Z/∂η = E_w[g] and ∂² log Z/∂η² = Var_w(g) + E_w[∂g/∂η].
        sites = _require_sites(state)
        cavity_shift = backend.to_device(sites.cavity_shift[variants])
        log_normalizer, weight, component_variance, relative_precision = _tilted_weights(
            backend.to_device(sites.cavity_precision[variants]),
            cavity_shift,
            np.exp(log_prior_variance),
            grid.class_log_prior_mass[class_index],
            grid.local_scale,
        )
        explained = np.square(cavity_shift)[:, None] * component_variance
        retained = 1.0 / (1.0 + relative_precision)
        component_gradient = 0.5 * retained * (explained - relative_precision)
        first = np.sum(weight * component_gradient, axis=1)
        component_curvature = 0.5 * np.square(retained) * (explained * (1.0 - relative_precision) - relative_precision)
        second = np.sum(weight * (np.square(component_gradient - first[:, None]) + component_curvature), axis=1)
        return float(np.sum(log_normalizer)), first, second

    def shape_objective(
        self,
        backend: _ArrayBackend,
        state: LocalState,
        log_prior_variance: F64Array,
        class_index: I64Array,
        shape_a: float,
        shape_b: F64Array,
        chunks: Sequence[slice],
    ) -> tuple[float, F64Array, F64Array]:
        # With L_k = log(1 + λ_k) and the exact normalizer B(a, b):
        # ∂ log π_ck/∂b_c = −L_k + ψ(a + b_c) − ψ(b_c) and
        # ∂² log π_ck/∂b_c² = ψ'(a + b_c) − ψ'(b_c), so per variant
        # ∂ log Z/∂b = E_prior[L] − E_w[L] and ∂² log Z/∂b² = Var_w(L) + ψ'(a + b) − ψ'(b).
        sites = _require_sites(state)
        grid = _log_local_scale_grid(shape_a, shape_b)
        class_count = shape_b.shape[0]
        xp = backend.xp

        def chunk_sums(variants: slice) -> tuple[float, F64Array, F64Array, F64Array]:
            class_log_prior_mass = backend.to_device(grid.class_log_prior_mass)
            local_scale = backend.to_device(grid.local_scale)
            log_one_plus = backend.to_device(grid.log_one_plus_local_scale)
            chunk_class = xp.asarray(class_index[variants])
            log_normalizer, weight, _component_variance, _relative_precision = _tilted_weights(
                backend.to_device(sites.cavity_precision[variants]),
                backend.to_device(sites.cavity_shift[variants]),
                np.exp(backend.to_device(log_prior_variance[variants])),
                class_log_prior_mass[chunk_class],
                local_scale,
            )
            # Row sums rather than a matrix-vector product keep BLAS threads out of the worker threads.
            posterior_mean = np.sum(weight * log_one_plus[None, :], axis=1)
            posterior_spread = np.sum(weight * np.square(log_one_plus)[None, :], axis=1) - np.square(posterior_mean)
            return (
                float(np.sum(log_normalizer)),
                backend.to_host(xp.bincount(chunk_class, weights=posterior_mean, minlength=class_count)),
                backend.to_host(xp.bincount(chunk_class, weights=posterior_spread, minlength=class_count)),
                np.bincount(class_index[variants], minlength=class_count).astype(np.float64),
            )

        sums = backend.map_chunks(chunk_sums, chunks)
        value = float(sum(chunk[0] for chunk in sums))
        posterior_mean_sum = np.sum([chunk[1] for chunk in sums], axis=0)
        posterior_spread_sum = np.sum([chunk[2] for chunk in sums], axis=0)
        class_counts = np.sum([chunk[3] for chunk in sums], axis=0)
        gradient = class_counts * grid.class_prior_mean_log_one_plus - posterior_mean_sum
        hessian = posterior_spread_sum + class_counts * grid.class_prior_mean_log_one_plus_slope
        return value, gradient, hessian

    def shape_starting_point(
        self, state: LocalState, shape_b: F64Array, class_index: I64Array, chunks: Sequence[slice]
    ) -> F64Array:
        return shape_b


class _MeanFieldScheme:
    """q(β)q(λ)q(δ) with exact GIG and Gamma moments; ``plug_in`` swaps E[1/λ] for 1/E[λ].

    The GIG moments need SciPy's Bessel functions, so the moment update runs on
    the host whatever the backend.
    """

    def __init__(self, *, plug_in: bool) -> None:
        self.plug_in = plug_in
        self.name = "plug_in" if plug_in else "coherent_vb"

    def initial_state(
        self, prior_variance: F64Array, typical_local_scale: F64Array, shape_a: float, shape_b: F64Array
    ) -> MeanFieldMoments:
        auxiliary_rate = (shape_a + shape_b) / (1.0 + typical_local_scale)
        return MeanFieldMoments(
            expected_local_scale=typical_local_scale.copy(),
            expected_inverse_local_scale=1.0 / typical_local_scale,
            expected_log_local_scale=np.log(typical_local_scale),
            expected_auxiliary_rate=auxiliary_rate,
            expected_log_auxiliary_rate=np.log(auxiliary_rate),
            coefficient_second_moment=prior_variance * typical_local_scale,
        )

    def reset_range(
        self,
        state: LocalState,
        variants: slice,
        prior_variance: F64Array,
        typical_local_scale: F64Array,
        shape_a: float,
        shape_b: F64Array,
    ) -> None:
        fresh = self.initial_state(prior_variance, typical_local_scale, shape_a, shape_b)
        moments = _require_moments(state)
        moments.expected_local_scale[variants] = fresh.expected_local_scale
        moments.expected_inverse_local_scale[variants] = fresh.expected_inverse_local_scale
        moments.expected_log_local_scale[variants] = fresh.expected_log_local_scale
        moments.expected_auxiliary_rate[variants] = fresh.expected_auxiliary_rate
        moments.expected_log_auxiliary_rate[variants] = fresh.expected_log_auxiliary_rate
        moments.coefficient_second_moment[variants] = fresh.coefficient_second_moment

    def load_block(self, backend: _ArrayBackend, state: LocalState, variants: slice) -> tuple[NDArrayLike, ...]:
        moments = _require_moments(state)
        return (
            moments.expected_local_scale[variants].copy(),
            moments.expected_inverse_local_scale[variants].copy(),
            moments.expected_log_local_scale[variants].copy(),
            moments.expected_auxiliary_rate[variants].copy(),
            moments.expected_log_auxiliary_rate[variants].copy(),
            moments.coefficient_second_moment[variants].copy(),
        )

    def solve_terms(
        self, backend: _ArrayBackend, block_state: tuple[NDArrayLike, ...], block_prior: _BlockPrior
    ) -> tuple[NDArrayLike, NDArrayLike]:
        expected_local_scale, expected_inverse_local_scale = block_state[0], block_state[1]
        prior_variance = backend.to_host(block_prior.prior_variance)
        if self.plug_in:
            precision = 1.0 / (prior_variance * expected_local_scale)
        else:
            precision = expected_inverse_local_scale / prior_variance
        return backend.to_device(precision), backend.to_device(np.zeros_like(precision))

    def update_block(
        self,
        backend: _ArrayBackend,
        block_state: tuple[NDArrayLike, ...],
        posterior_mean: NDArrayLike,
        posterior_variance: NDArrayLike,
        block_prior: _BlockPrior,
    ) -> tuple[NDArrayLike, ...]:
        # q(λ) = GIG(a − ½, E[β²]/u, 2E[δ]), then q(δ) = Gamma(a + b, 1 + E[λ]).
        shape_a = block_prior.shape_a
        variant_shape_b = _shape_b_per_variant(block_prior.shape_b, block_prior.class_index)
        second_moment = np.square(backend.to_host(posterior_mean)) + backend.to_host(posterior_variance)
        order = np.full(second_moment.shape, shape_a - 0.5)
        chi = second_moment / backend.to_host(block_prior.prior_variance)
        psi = 2.0 * block_state[3]
        expected_local_scale = _gig_moment(order, chi, psi, 1.0)
        return (
            expected_local_scale,
            _gig_moment(order, chi, psi, -1.0),
            gig_expected_log(order, chi, psi),
            (shape_a + variant_shape_b) / (1.0 + expected_local_scale),
            digamma(shape_a + variant_shape_b) - np.log1p(expected_local_scale),
            second_moment,
        )

    def store_block(
        self, backend: _ArrayBackend, state: LocalState, variants: slice, block_state: tuple[NDArrayLike, ...]
    ) -> None:
        moments = _require_moments(state)
        (
            moments.expected_local_scale[variants],
            moments.expected_inverse_local_scale[variants],
            moments.expected_log_local_scale[variants],
            moments.expected_auxiliary_rate[variants],
            moments.expected_log_auxiliary_rate[variants],
            moments.coefficient_second_moment[variants],
        ) = block_state

    def scale_objective(
        self,
        backend: _ArrayBackend,
        state: LocalState,
        variants: slice,
        log_prior_variance: NDArrayLike,
        class_index: NDArrayLike,
        grid: _DeviceGrid,
    ) -> tuple[float, NDArrayLike, NDArrayLike]:
        # E_q log N(β; 0, uλ) = −½ η − ½ S e^(−η) + const with S = E[β²]E[1/λ]
        # (plug-in: E[β²]/E[λ]); concave in η.
        moments = _require_moments(state)
        if self.plug_in:
            statistic = moments.coefficient_second_moment[variants] / moments.expected_local_scale[variants]
        else:
            statistic = moments.coefficient_second_moment[variants] * moments.expected_inverse_local_scale[variants]
        scaled = backend.to_device(statistic) * np.exp(-log_prior_variance)
        value = float(np.sum(-0.5 * log_prior_variance - 0.5 * scaled))
        return value, 0.5 * (scaled - 1.0), -0.5 * scaled

    def shape_objective(
        self,
        backend: _ArrayBackend,
        state: LocalState,
        log_prior_variance: F64Array,
        class_index: I64Array,
        shape_a: float,
        shape_b: F64Array,
        chunks: Sequence[slice],
    ) -> tuple[float, F64Array, F64Array]:
        # The b-dependent part of E_q log p(δ | b): (b − 1) E[log δ] − log Γ(b).
        class_counts, rate_sum = self._class_rate_sums(state, class_index, shape_b.shape[0], chunks)
        value = float(np.sum((shape_b - 1.0) * rate_sum - class_counts * gammaln(shape_b)))
        return value, rate_sum - class_counts * digamma(shape_b), -class_counts * polygamma(1, shape_b)

    def shape_starting_point(
        self, state: LocalState, shape_b: F64Array, class_index: I64Array, chunks: Sequence[slice]
    ) -> F64Array:
        """The closed-form b M-step without pooling, ψ(b_c) = mean_c E[log δ] (Minka)."""
        class_counts, rate_sum = self._class_rate_sums(state, class_index, shape_b.shape[0], chunks)
        unpooled = inverse_digamma(rate_sum / np.maximum(class_counts, 1.0))
        return np.clip(np.where(class_counts > 0.0, unpooled, shape_b), _MINIMUM_SHAPE_B, _MAXIMUM_SHAPE_B)

    def _class_rate_sums(
        self, state: LocalState, class_index: I64Array, class_count: int, chunks: Sequence[slice]
    ) -> tuple[F64Array, F64Array]:
        """Per-class counts and sums of the E[log δ] statistic over the chunks."""
        moments = _require_moments(state)
        counts = np.zeros(class_count)
        sums = np.zeros(class_count)
        for variants in chunks:
            if self.plug_in:
                statistic = np.log(moments.expected_auxiliary_rate[variants])
            else:
                statistic = moments.expected_log_auxiliary_rate[variants]
            counts += np.bincount(class_index[variants], minlength=class_count)
            sums += np.bincount(class_index[variants], weights=statistic, minlength=class_count)
        return counts, sums


def _local_scheme(name: str) -> _LocalScheme:
    """The one switch between inference schemes (the choice is still being validated)."""
    if name == "expectation_propagation":
        return _ExpectationPropagationScheme()
    if name == "coherent_vb":
        return _MeanFieldScheme(plug_in=False)
    if name == "plug_in":
        return _MeanFieldScheme(plug_in=True)
    raise ValueError(f"Unknown Stage 1 inference scheme {name!r}.")


_DEFAULT_SCHEME = "expectation_propagation"


def _site_arrays(sites: ExpectationPropagationSites) -> tuple[F64Array, ...]:
    """Every per-variant EP array, in the order of a block's local state."""
    return (
        sites.site_precision,
        sites.site_shift,
        sites.cavity_precision,
        sites.cavity_shift,
        sites.site_damping,
        sites.previous_precision_step,
        sites.previous_shift_step,
    )


def _require_sites(state: LocalState) -> ExpectationPropagationSites:
    if not isinstance(state, ExpectationPropagationSites):
        raise TypeError("The EP scheme needs EP sites; the warm start came from a mean-field fit.")
    return state


def _require_moments(state: LocalState) -> MeanFieldMoments:
    if not isinstance(state, MeanFieldMoments):
        raise TypeError("The mean-field schemes need mean-field moments; the warm start came from an EP fit.")
    return state


def gig_expected_log(p_parameter: F64Array, chi: F64Array, psi: F64Array) -> F64Array:
    """E[log X] for X ~ GIG(p, chi, psi): 0.5 log(chi / psi) + d/dp log K_p(sqrt(chi psi)).

    The order derivative has no closed form, so it is a Richardson-extrapolated
    central difference of log kve (the exponential scaling is constant in the
    order and cancels). K_p is even in p, so signed orders are passed as they are.
    """
    z_value = np.sqrt(chi) * np.sqrt(psi)

    def central_difference(step: float) -> F64Array:
        return (np.log(kve(p_parameter + step, z_value)) - np.log(kve(p_parameter - step, z_value))) / (2.0 * step)

    order_derivative = (
        4.0 * central_difference(0.5 * _BESSEL_ORDER_DIFFERENCE_STEP) - central_difference(_BESSEL_ORDER_DIFFERENCE_STEP)
    ) / 3.0
    expected_log = 0.5 * (np.log(chi) - np.log(psi)) + order_derivative
    if not np.all(np.isfinite(expected_log)):
        raise FloatingPointError("GIG expected log is non-finite; chi and psi must be positive and finite.")
    return np.asarray(expected_log, dtype=np.float64)


def inverse_digamma(values: F64Array) -> F64Array:
    """ψ⁻¹ by Minka's initialization and five Newton steps (machine precision)."""
    target = np.asarray(values, dtype=np.float64)
    solution = np.where(target >= -2.22, np.exp(target) + 0.5, -1.0 / (target - digamma(1.0)))
    for _newton_step in range(5):
        solution = solution - (digamma(solution) - target) / polygamma(1, solution)
    return np.asarray(solution, dtype=np.float64)


# --------------------------------------------------------------------------- hyperparameter M-steps


def _ascent_step(gradient: F64Array, hessian: F64Array) -> F64Array:
    """Newton ascent direction, Levenberg-Marquardt damped until −H is positive definite."""
    if not (np.all(np.isfinite(gradient)) and np.all(np.isfinite(hessian))):
        raise FloatingPointError("A hyperparameter M-step produced a non-finite gradient or Hessian.")
    negative_hessian = -hessian
    damping = 0.0
    damping_floor = 1e-10 * max(float(np.max(np.abs(np.diag(negative_hessian)))), 1.0)
    identity = np.eye(gradient.shape[0])
    while True:
        try:
            factor = np.linalg.cholesky(negative_hessian + damping * identity)
        except np.linalg.LinAlgError:
            damping = max(10.0 * damping, damping_floor)
            continue
        return np.linalg.solve(factor.T, np.linalg.solve(factor, gradient))


_Objective = Callable[[F64Array], tuple[float, F64Array, F64Array]]


def _maximize(
    evaluate: _Objective,
    initial_parameters: F64Array,
    lower_bound: float,
    upper_bound: float,
) -> F64Array:
    """Damped Newton ascent with backtracking, for smooth objectives of a few parameters.

    ``evaluate(parameters) -> (value, gradient, hessian)``; the box bounds are
    applied by projection.
    """
    parameters = np.clip(np.asarray(initial_parameters, dtype=np.float64), lower_bound, upper_bound)
    value, gradient, hessian = evaluate(parameters)
    for _newton_iteration in range(_MAXIMUM_NEWTON_ITERATIONS):
        direction = _ascent_step(gradient, hessian)
        step_length = 1.0
        while step_length >= _MINIMUM_LINE_SEARCH_STEP:
            candidate = np.clip(parameters + step_length * direction, lower_bound, upper_bound)
            candidate_value, candidate_gradient, candidate_hessian = evaluate(candidate)
            if candidate_value >= value:
                break
            step_length *= 0.5
        if step_length < _MINIMUM_LINE_SEARCH_STEP:
            return parameters
        movement = float(np.max(np.abs(candidate - parameters)))
        parameters, value, gradient, hessian = candidate, candidate_value, candidate_gradient, candidate_hessian
        if movement < _NEWTON_STEP_TOLERANCE:
            return parameters
    return parameters


def _update_scale_model(
    scheme: _LocalScheme,
    backend: _ArrayBackend,
    model: _ModelState,
    hypermodel: LDPriorHypermodel,
    ranges: Sequence[slice],
) -> None:
    """Maximize Σ_j f_j(level + o_j + d_jᵀθ) + log N(θ; θ₀, diag(1/P)) over (level, θ), j in ``ranges``."""
    feature_count = hypermodel.annotation_design.shape[1]
    grid_size = _log_local_scale_grid(hypermodel.shape_a, model.shape_b).local_scale.shape[0]
    chunks = _variant_chunks(backend, ranges, grid_size)
    xp = backend.xp

    def chunk_terms(parameters: F64Array, variants: slice) -> tuple[float, F64Array, F64Array]:
        grid = _device_grid(backend, hypermodel, model)
        log_prior_variance = backend.to_device(hypermodel.log_prior_variance(parameters[0], parameters[1:], variants))
        chunk_value, first, second = scheme.scale_objective(
            backend, model.local, variants, log_prior_variance, xp.asarray(hypermodel.variant_class_index[variants]), grid
        )
        chunk_design = backend.to_device(hypermodel.annotation_design[variants])
        gradient = xp.concatenate([xp.sum(first)[None], chunk_design.T @ first])
        cross = chunk_design.T @ second
        hessian = xp.empty((feature_count + 1, feature_count + 1))
        hessian[0, 0] = xp.sum(second)
        hessian[1:, 0] = cross
        hessian[0, 1:] = cross
        hessian[1:, 1:] = chunk_design.T @ (second[:, None] * chunk_design)
        return chunk_value, backend.to_host(gradient), backend.to_host(hessian)

    def evaluate(parameters: F64Array) -> tuple[float, F64Array, F64Array]:
        terms = backend.map_chunks(lambda variants: chunk_terms(parameters, variants), chunks)
        value = float(sum(term[0] for term in terms))
        gradient = np.sum([term[1] for term in terms], axis=0)
        hessian = np.sum([term[2] for term in terms], axis=0)
        coefficient_deviation = parameters[1:] - hypermodel.annotation_prior_mean
        value -= 0.5 * float(np.sum(hypermodel.annotation_prior_precision * np.square(coefficient_deviation)))
        gradient[1:] -= hypermodel.annotation_prior_precision * coefficient_deviation
        hessian[1:, 1:] -= np.diag(hypermodel.annotation_prior_precision)
        return value, gradient, hessian

    parameters = _maximize(
        evaluate,
        np.concatenate([[model.log_variance_level], model.annotation_coefficients]),
        -np.inf,
        np.inf,
    )
    model.log_variance_level = float(parameters[0])
    model.annotation_coefficients = parameters[1:].copy()


def _update_shapes(
    scheme: _LocalScheme,
    backend: _ArrayBackend,
    model: _ModelState,
    hypermodel: LDPriorHypermodel,
    ranges: Sequence[slice],
) -> None:
    """Maximize the tail-shape objective (variants in ``ranges``) plus the pooling prior over x_c = log b_c.

    Pooling: −½ Σ_c (x_c − x̄)² / s², with the mean x̄ left free.
    """
    class_index = hypermodel.variant_class_index
    class_count = len(hypermodel.class_names)
    log_prior_variance = hypermodel.log_prior_variance(
        model.log_variance_level, model.annotation_coefficients, slice(0, hypermodel.variant_count)
    )
    grid_size = _log_local_scale_grid(hypermodel.shape_a, model.shape_b).local_scale.shape[0]
    chunks = _variant_chunks(backend, ranges, grid_size)
    centering = np.eye(class_count) - np.full((class_count, class_count), 1.0 / class_count)
    pooling_precision = 1.0 / hypermodel.shape_b_pooling_variance

    def evaluate(log_shape_b: F64Array) -> tuple[float, F64Array, F64Array]:
        shape_b = np.exp(log_shape_b)
        value, gradient_b, hessian_b = scheme.shape_objective(
            backend, model.local, log_prior_variance, class_index, hypermodel.shape_a, shape_b, chunks
        )
        centered = centering @ log_shape_b
        value -= 0.5 * pooling_precision * float(centered @ centered)
        gradient = shape_b * gradient_b - pooling_precision * centered
        hessian = np.diag(np.square(shape_b) * hessian_b + shape_b * gradient_b) - pooling_precision * centering
        return value, gradient, hessian

    initial = np.log(scheme.shape_starting_point(model.local, model.shape_b, class_index, chunks))
    model.shape_b = np.exp(_maximize(evaluate, initial, np.log(_MINIMUM_SHAPE_B), np.log(_MAXIMUM_SHAPE_B)))


# --------------------------------------------------------------------------- initialization


def _typical_local_scale(shape_a: float, shape_b: F64Array) -> F64Array:
    """The median of BetaPrime(a, b): λ/(1 + λ) ~ Beta(a, b)."""
    median_fraction = betaincinv(shape_a, shape_b, 0.5)
    return np.asarray(median_fraction / (1.0 - median_fraction), dtype=np.float64)


def _score_scale(statistics: TraitStatistics, likelihood_precision: float) -> float:
    """The Gaussian factor's linear term is κ X̃ᵀr for a quantitative trait and the working score for a binary one."""
    return likelihood_precision if isinstance(statistics, QuantitativeTraitStatistics) else 1.0


def _statistics_likelihood_precision(statistics: TraitStatistics) -> float:
    if isinstance(statistics, QuantitativeTraitStatistics):
        return float((statistics.sample_count - statistics.covariate_count) / statistics.residual_sum_of_squares)
    return float(statistics.mean_fisher_weight)


def _moment_log_variance_level(
    ld_diagonal: F64Array,
    ld_scores: F64Array,
    statistics: TraitStatistics,
    hypermodel: LDPriorHypermodel,
    typical_local_scale: F64Array,
    likelihood_precision: float,
) -> float:
    """Level from E‖ℓ‖² = κ tr(H) + κ² Σ_k Var(β_k) (H²)_kk (Haseman-Elston in LD space).

    ℓ is the Gaussian factor's linear term (κg, or the working score for binary), H = nR,
    and (H²)_kk = n² ld_score_k. The level is floored at a prior genetic variance of
    1e-3 over the noise.
    """
    sample_count = float(statistics.sample_count)
    linear_term = statistics.score * _score_scale(statistics, likelihood_precision)
    relative_variance = np.exp(
        hypermodel.log_prior_variance(0.0, hypermodel.annotation_prior_mean, slice(0, hypermodel.variant_count))
    ) * typical_local_scale
    signal = float(linear_term @ linear_term) - likelihood_precision * sample_count * float(np.sum(ld_diagonal))
    weighted_trace = likelihood_precision**2 * sample_count**2 * float(relative_variance @ ld_scores)
    floor = _MINIMUM_INITIAL_SIGNAL_RATIO / (likelihood_precision * float(np.sum(relative_variance)))
    return float(np.log(max(signal / weighted_trace, floor)))


def _initial_model_state(
    scheme: _LocalScheme,
    ld_diagonal: F64Array,
    ld_scores: F64Array,
    statistics: TraitStatistics,
    hypermodel: LDPriorHypermodel,
) -> _ModelState:
    class_typical = _typical_local_scale(hypermodel.shape_a, hypermodel.initial_shape_b)
    typical_local_scale = class_typical[hypermodel.variant_class_index]
    likelihood_precision = _statistics_likelihood_precision(statistics)
    level = _moment_log_variance_level(
        ld_diagonal, ld_scores, statistics, hypermodel, typical_local_scale, likelihood_precision
    )
    coefficients = hypermodel.annotation_prior_mean.copy()
    prior_variance = np.exp(hypermodel.log_prior_variance(level, coefficients, slice(0, hypermodel.variant_count)))
    variant_count = hypermodel.variant_count
    return _ModelState(
        statistics=statistics,
        log_variance_level=level,
        annotation_coefficients=coefficients,
        shape_b=hypermodel.initial_shape_b.copy(),
        likelihood_precision=likelihood_precision,
        local=scheme.initial_state(
            prior_variance,
            typical_local_scale,
            hypermodel.shape_a,
            _shape_b_per_variant(hypermodel.initial_shape_b, hypermodel.variant_class_index),
        ),
        posterior_mean=np.zeros(variant_count),
        posterior_variance=prior_variance * typical_local_scale,
        acceleration=AndersonState(memory_depth=_ANDERSON_MEMORY),
        warm_started=False,
    )


def _warm_model_state(statistics: TraitStatistics, warm_start: LDSpaceFit) -> _ModelState:
    local = warm_start.local_state
    copied_local: LocalState
    if isinstance(local, ExpectationPropagationSites):
        copied_local = ExpectationPropagationSites(*(values.copy() for values in _site_arrays(local)))
    else:
        copied_local = MeanFieldMoments(
            expected_local_scale=local.expected_local_scale.copy(),
            expected_inverse_local_scale=local.expected_inverse_local_scale.copy(),
            expected_log_local_scale=local.expected_log_local_scale.copy(),
            expected_auxiliary_rate=local.expected_auxiliary_rate.copy(),
            expected_log_auxiliary_rate=local.expected_log_auxiliary_rate.copy(),
            coefficient_second_moment=local.coefficient_second_moment.copy(),
        )
    return _ModelState(
        statistics=statistics,
        log_variance_level=warm_start.log_variance_level,
        annotation_coefficients=warm_start.annotation_coefficients.copy(),
        shape_b=warm_start.shape_b.copy(),
        likelihood_precision=_statistics_likelihood_precision(statistics),
        local=copied_local,
        posterior_mean=warm_start.posterior_mean.copy(),
        posterior_variance=warm_start.posterior_variance.copy(),
        acceleration=AndersonState(memory_depth=_ANDERSON_MEMORY),
        warm_started=True,
    )


# --------------------------------------------------------------------------- the fit


def _device_grid(backend: _ArrayBackend, hypermodel: LDPriorHypermodel, model: _ModelState) -> _DeviceGrid:
    grid = _log_local_scale_grid(hypermodel.shape_a, model.shape_b)
    return _DeviceGrid(
        local_scale=backend.to_device(grid.local_scale),
        class_log_prior_mass=backend.to_device(grid.class_log_prior_mass),
    )


def _block_prior(
    backend: _ArrayBackend, model: _ModelState, hypermodel: LDPriorHypermodel, grid: _DeviceGrid, variants: slice
) -> _BlockPrior:
    class_index = hypermodel.variant_class_index[variants]
    return _BlockPrior(
        prior_variance=backend.to_device(
            np.exp(hypermodel.log_prior_variance(model.log_variance_level, model.annotation_coefficients, variants))
        ),
        class_index=class_index,
        log_prior_mass=grid.class_log_prior_mass[backend.xp.asarray(class_index)],
        local_scale=grid.local_scale,
        shape_a=hypermodel.shape_a,
        shape_b=model.shape_b,
    )


def _update_model_block(
    scheme: _LocalScheme,
    backend: _ArrayBackend,
    model: _ModelState,
    hypermodel: LDPriorHypermodel,
    grid: _DeviceGrid,
    correlation_block: NDArrayLike,
    variants: slice,
    until_converged: bool,
) -> tuple[float, bool, int]:
    """Run one model's local iterations on one block.

    Either exactly _LOCAL_ITERATIONS_PER_PASS of them (a pass of the EM map) or,
    with ``until_converged``, until the block posterior settles. Returns the
    squared change of the posterior mean, whether the block settled, and the
    iterations run.
    """
    statistics = model.statistics
    block_prior = _block_prior(backend, model, hypermodel, grid, variants)
    likelihood_scale = model.likelihood_precision * statistics.sample_count
    base_linear_term = _score_scale(statistics, model.likelihood_precision) * backend.to_device(statistics.score[variants])
    block_state = scheme.load_block(backend, model.local, variants)
    previous_mean = backend.to_device(model.posterior_mean[variants])
    previous_variance = backend.to_device(model.posterior_variance[variants])
    settled = False
    iteration_limit = _MAXIMUM_LOCAL_ITERATIONS if until_converged else _LOCAL_ITERATIONS_PER_PASS
    iterations = 0
    for _local_iteration in range(iteration_limit):
        iterations += 1
        diagonal_precision, linear_shift = scheme.solve_terms(backend, block_state, block_prior)
        mean, variance = backend.block_posterior(
            correlation_block, likelihood_scale, diagonal_precision, base_linear_term + linear_shift
        )
        block_state = scheme.update_block(backend, block_state, mean, variance, block_prior)
        mean_scale = max(float(np.max(np.abs(mean))), 1e-300)
        settled = (
            float(np.max(np.abs(mean - previous_mean))) <= _CONVERGENCE_TOLERANCE * mean_scale
            and float(np.max(np.abs(variance / previous_variance - 1.0))) <= _CONVERGENCE_TOLERANCE
        )
        if until_converged and settled:
            break
        previous_mean, previous_variance = mean, variance
    scheme.store_block(backend, model.local, variants, block_state)
    host_mean = backend.to_host(mean)
    squared_mean_change = float(np.sum(np.square(host_mean - model.posterior_mean[variants])))
    model.posterior_mean[variants] = host_mean
    model.posterior_variance[variants] = backend.to_host(variance)
    return squared_mean_change, settled, iterations


def _hyperparameter_blocks(boundaries: I64Array, class_index: I64Array, class_count: int) -> I64Array:
    """The blocks the hyperparameters are fitted on: a fixed random order, taken until the subset is large enough."""
    order = np.random.default_rng(_HYPERPARAMETER_SUBSET_SEED).permutation(boundaries.shape[0] - 1)
    class_targets = np.minimum(np.bincount(class_index, minlength=class_count), _HYPERPARAMETER_SUBSET_CLASS_VARIANTS)
    class_counts = np.zeros(class_count, dtype=np.int64)
    chosen: list[int] = []
    for block_index in order:
        if int(class_counts.sum()) >= _HYPERPARAMETER_SUBSET_VARIANTS and np.all(class_counts >= class_targets):
            break
        chosen.append(int(block_index))
        block_classes = class_index[int(boundaries[block_index]) : int(boundaries[block_index + 1])]
        class_counts += np.bincount(block_classes, minlength=class_count)
    return np.sort(np.asarray(chosen, dtype=np.int64))


def _block_ranges(boundaries: I64Array, blocks: I64Array) -> list[slice]:
    """Contiguous variant ranges covering ``blocks`` (sorted), adjacent blocks merged."""
    ranges: list[slice] = []
    for block_index in blocks:
        start, stop = int(boundaries[block_index]), int(boundaries[block_index + 1])
        if ranges and ranges[-1].stop == start:
            ranges[-1] = slice(ranges[-1].start, stop)
        else:
            ranges.append(slice(start, stop))
    return ranges


def _hyperparameter_vector(model: _ModelState) -> F64Array:
    return np.concatenate([[model.log_variance_level], model.annotation_coefficients, np.log(model.shape_b)])


def _set_hyperparameters(model: _ModelState, vector: F64Array) -> None:
    feature_count = model.annotation_coefficients.shape[0]
    model.log_variance_level = float(vector[0])
    model.annotation_coefficients = vector[1 : 1 + feature_count].copy()
    model.shape_b = np.exp(np.clip(vector[1 + feature_count :], np.log(_MINIMUM_SHAPE_B), np.log(_MAXIMUM_SHAPE_B)))


def _accelerated_hyperparameters(acceleration: AndersonState, current: F64Array, mapped: F64Array) -> F64Array:
    proposal = anderson_step(acceleration, x_current=current, map_value=mapped)
    extrapolation = float(np.max(np.abs(proposal - mapped)))
    if not (np.all(np.isfinite(proposal)) and extrapolation <= _ANDERSON_MAXIMUM_LOG_EXTRAPOLATION):
        acceleration.reset()
        return mapped
    return proposal


def _validate_inputs(
    ld_blocks: LDBlockSource,
    trait_statistics: Sequence[TraitStatistics],
    hypermodel: LDPriorHypermodel,
    warm_starts: Sequence[LDSpaceFit | None],
    scheme_name: str,
) -> None:
    variant_count = int(ld_blocks.block_boundaries[-1])
    if hypermodel.variant_count != variant_count:
        raise ValueError(f"The hypermodel has {hypermodel.variant_count} variants; the LD blocks have {variant_count}.")
    if len(warm_starts) != len(trait_statistics):
        raise ValueError("Pass one warm start (or None) per trait.")
    widest_block = int(np.max(np.diff(ld_blocks.block_boundaries)))
    for statistics in trait_statistics:
        # A block at least as wide as the projected rank n − q has a singular R_b.
        # EP then finds zero-precision sites that span the data, every cavity goes
        # flat, and Σ log Z_j no longer identifies the level.
        if scheme_name == "expectation_propagation" and widest_block >= statistics.sample_count - statistics.covariate_count:
            raise ValueError(
                f"An LD block has {widest_block} variants but the projected sample rank is "
                f"{statistics.sample_count - statistics.covariate_count}; EP needs every block narrower."
            )
        if statistics.score.shape != (variant_count,):
            raise ValueError("Every trait score needs one entry per variant.")


def fit_ld_space(
    ld_blocks: LDBlockSource,
    trait_statistics: Sequence[TraitStatistics],
    hypermodel: LDPriorHypermodel,
    budget: ComputeBudget,
    warm_starts: Sequence[LDSpaceFit | None] | None = None,
    scheme_name: str = _DEFAULT_SCHEME,
) -> list[LDSpaceFit]:
    """Fit every trait (x fold) to convergence, reading each LD block once per pass.

    One pass: for every block, every unconverged model runs its local iterations
    (block solve, then the scheme's site or moment update). Then, for every model,
    the level and annotation coefficients and the tail shapes are updated, and
    Anderson extrapolates the result.
    """
    starts = [None] * len(trait_statistics) if warm_starts is None else list(warm_starts)
    _validate_inputs(ld_blocks, trait_statistics, hypermodel, starts, scheme_name)
    widest_block = int(np.max(np.diff(ld_blocks.block_boundaries)))
    with _array_backend(budget, widest_block) as backend:
        return _fit_models(ld_blocks, trait_statistics, hypermodel, backend, starts, scheme_name)


def _fit_models(
    ld_blocks: LDBlockSource,
    trait_statistics: Sequence[TraitStatistics],
    hypermodel: LDPriorHypermodel,
    backend: _ArrayBackend,
    starts: Sequence[LDSpaceFit | None],
    scheme_name: str,
) -> list[LDSpaceFit]:
    scheme = _local_scheme(scheme_name)
    ld_diagonal = ld_blocks.ld_diagonal()
    ld_scores = ld_blocks.ld_scores()
    models = [
        _initial_model_state(scheme, ld_diagonal, ld_scores, statistics, hypermodel)
        if warm_start is None
        else _warm_model_state(statistics, warm_start)
        for statistics, warm_start in zip(trait_statistics, starts, strict=True)
    ]
    boundaries = np.asarray(ld_blocks.block_boundaries, dtype=np.int64)
    class_count = len(hypermodel.class_names)
    subset_blocks = _hyperparameter_blocks(boundaries, hypermodel.variant_class_index, class_count)
    subset_ranges = _block_ranges(boundaries, subset_blocks)
    remaining_blocks = np.setdiff1d(np.arange(boundaries.shape[0] - 1), subset_blocks)

    def run_blocks(
        block_indices: I64Array, fitted: list[_ModelState], until_converged: bool
    ) -> list[list[tuple[float, bool, int]]]:
        def block_results(block_index: int) -> list[tuple[float, bool, int]]:
            # Blocks write disjoint variant ranges of every model's state, and
            # each creates its device arrays on the device it runs on.
            correlation_block = backend.to_device(ld_blocks.correlation_block(block_index))
            variants = slice(int(boundaries[block_index]), int(boundaries[block_index + 1]))
            return [
                _update_model_block(
                    scheme,
                    backend,
                    model,
                    hypermodel,
                    _device_grid(backend, hypermodel, model),
                    correlation_block,
                    variants,
                    until_converged,
                )
                for model in fitted
            ]

        return backend.map_blocks(block_results, [int(block_index) for block_index in block_indices])

    for pass_index in range(_MAXIMUM_PASSES):
        active = [model for model in models if not model.converged]
        if not active:
            break
        pass_start = time.perf_counter()
        results = run_blocks(subset_blocks, active, until_converged=False)
        block_seconds = time.perf_counter() - pass_start
        largest_change = 0.0
        for position, model in enumerate(active):
            squared_mean_change = float(sum(block[position][0] for block in results))
            subset_mean_norm = float(np.sqrt(sum(float(np.sum(np.square(model.posterior_mean[span]))) for span in subset_ranges)))
            current = _hyperparameter_vector(model)
            _update_scale_model(scheme, backend, model, hypermodel, subset_ranges)
            _update_shapes(scheme, backend, model, hypermodel, subset_ranges)
            mapped = _hyperparameter_vector(model)
            model.passes += 1
            hyperparameter_change = float(np.max(np.abs(mapped - current)))
            mean_change = float(np.sqrt(squared_mean_change) / max(subset_mean_norm, 1e-300))
            largest_change = max(largest_change, hyperparameter_change, mean_change)
            model.converged = max(hyperparameter_change, mean_change) < _CONVERGENCE_TOLERANCE
            if not model.converged:
                _set_hyperparameters(model, _accelerated_hyperparameters(model.acceleration, current, mapped))
        log(
            f"  Stage 1 pass {pass_index + 1} over {subset_blocks.shape[0]} blocks: {len(active)} models, "
            f"blocks {block_seconds:.1f}s, M-steps {time.perf_counter() - pass_start - block_seconds:.1f}s, "
            f"largest change {largest_change:.2e}"
        )
    if remaining_blocks.shape[0] > 0:
        pass_start = time.perf_counter()
        for model in models:
            if model.warm_started:
                continue
            typical_local_scale = _typical_local_scale(hypermodel.shape_a, model.shape_b)[hypermodel.variant_class_index]
            for span in _block_ranges(boundaries, remaining_blocks):
                scheme.reset_range(
                    model.local,
                    span,
                    np.exp(hypermodel.log_prior_variance(model.log_variance_level, model.annotation_coefficients, span)),
                    typical_local_scale[span],
                    hypermodel.shape_a,
                    _shape_b_per_variant(model.shape_b, hypermodel.variant_class_index[span]),
                )
        results = run_blocks(remaining_blocks, models, until_converged=True)
        for position, model in enumerate(models):
            model.converged = model.converged and all(block[position][1] for block in results)
        iterations = np.array([result[2] for block in results for result in block])
        log(
            f"  Stage 1 fixed-hyperparameter pass over {remaining_blocks.shape[0]} blocks: {len(models)} models, "
            f"{time.perf_counter() - pass_start:.1f}s, local iterations median {np.median(iterations):.0f} max {iterations.max()}"
        )
    return [
        LDSpaceFit(
            scheme=scheme.name,
            posterior_mean=model.posterior_mean,
            posterior_variance=model.posterior_variance,
            log_variance_level=model.log_variance_level,
            annotation_coefficients=model.annotation_coefficients,
            shape_a=hypermodel.shape_a,
            shape_b=model.shape_b,
            likelihood_precision=model.likelihood_precision,
            local_state=model.local,
            passes=model.passes,
            converged=model.converged,
        )
        for model in models
    ]

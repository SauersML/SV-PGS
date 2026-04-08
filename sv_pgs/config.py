from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class TraitType(str, Enum):
    QUANTITATIVE = "quantitative"
    BINARY = "binary"


class VariantClass(str, Enum):
    SNV = "snv"
    SMALL_INDEL = "small_indel"
    DELETION_SHORT = "deletion_short"
    DELETION_LONG = "deletion_long"
    DUPLICATION_SHORT = "duplication_short"
    DUPLICATION_LONG = "duplication_long"
    INSERTION_MEI = "insertion_mei"
    INVERSION_BND_COMPLEX = "inversion_bnd_complex"
    STR_VNTR_REPEAT = "str_vntr_repeat"
    OTHER_COMPLEX_SV = "other_complex_sv"


# Default log-scale for each variant class's prior effect size.
# More negative = smaller expected effects.  SNVs (-4.5) are expected to
# have the smallest individual effects; complex SVs (-3.1) the largest.
# These are starting points — the model updates them during fitting.
# Values are in log-space: exp(-4.5) ≈ 0.011, exp(-3.1) ≈ 0.045.
DEFAULT_CLASS_LOG_BASELINE_SCALE = {
    VariantClass.SNV: -4.5,
    VariantClass.SMALL_INDEL: -4.2,
    VariantClass.DELETION_SHORT: -3.8,
    VariantClass.DELETION_LONG: -3.3,
    VariantClass.DUPLICATION_SHORT: -3.7,
    VariantClass.DUPLICATION_LONG: -3.3,
    VariantClass.INSERTION_MEI: -3.6,
    VariantClass.INVERSION_BND_COMPLEX: -3.1,
    VariantClass.STR_VNTR_REPEAT: -3.5,
    VariantClass.OTHER_COMPLEX_SV: -3.3,
}

# TPB shape parameters control the "tail weight" of the shrinkage prior.
# Shape_a (below) and shape_b control how tolerant the prior is of large effects:
#   - shape_a=1.0 (SNVs): moderate tails — most effects shrunk to near zero
#   - shape_a=0.55 (inversions): heavy tails — more tolerance for large effects
# SVs get heavier tails because they disrupt more DNA and are more likely
# to have individually detectable phenotypic effects.
DEFAULT_CLASS_TPB_SHAPE_A: dict[VariantClass, float] = {
    VariantClass.SNV: 1.0,
    VariantClass.SMALL_INDEL: 0.9,
    VariantClass.DELETION_SHORT: 0.7,
    VariantClass.DELETION_LONG: 0.6,
    VariantClass.DUPLICATION_SHORT: 0.7,
    VariantClass.DUPLICATION_LONG: 0.6,
    VariantClass.INSERTION_MEI: 0.65,
    VariantClass.INVERSION_BND_COMPLEX: 0.55,
    VariantClass.STR_VNTR_REPEAT: 0.6,
    VariantClass.OTHER_COMPLEX_SV: 0.6,
}

# Shape_b controls the auxiliary rate distribution.  Together with shape_a,
# it determines the marginal distribution of the local shrinkage factor.
# Smaller values = heavier tails = more large effects allowed.
DEFAULT_CLASS_TPB_SHAPE_B: dict[VariantClass, float] = {
    VariantClass.SNV: 0.5,
    VariantClass.SMALL_INDEL: 0.5,
    VariantClass.DELETION_SHORT: 0.45,
    VariantClass.DELETION_LONG: 0.4,
    VariantClass.DUPLICATION_SHORT: 0.45,
    VariantClass.DUPLICATION_LONG: 0.4,
    VariantClass.INSERTION_MEI: 0.42,
    VariantClass.INVERSION_BND_COMPLEX: 0.38,
    VariantClass.STR_VNTR_REPEAT: 0.4,
    VariantClass.OTHER_COMPLEX_SV: 0.4,
}

STRUCTURAL_VARIANT_CLASSES = (
    VariantClass.DELETION_SHORT,
    VariantClass.DELETION_LONG,
    VariantClass.DUPLICATION_SHORT,
    VariantClass.DUPLICATION_LONG,
    VariantClass.INSERTION_MEI,
    VariantClass.INVERSION_BND_COMPLEX,
    VariantClass.STR_VNTR_REPEAT,
    VariantClass.OTHER_COMPLEX_SV,
)


@dataclass(slots=True)
class ModelConfig:
    """All tunable parameters for SV-PGS inference.

    Most users won't need to change these. Key groups:

    Variational Bayes: prior scales, TPB shapes, linear algebra, working sets
    """
    trait_type: TraitType = TraitType.BINARY       # binary (case/control) or quantitative
    max_outer_iterations: int = 20                 # EM iterations (usually converges in 10-15, step size negligible beyond 20)
    convergence_tolerance: float = 1e-5            # stop when parameters change < this
    minimum_scale: float = 1e-6                    # variants with std < this are treated as monomorphic
    polya_gamma_minimum_weight: float = 1e-4       # floor on IRLS weights to prevent division by ~zero
    sigma_error_floor: float = 1e-3                # noise variance can't go below this
    prior_scale_floor: float = 1e-6
    prior_scale_ceiling: float = 10.0
    global_scale_floor: float = 1e-4
    global_scale_ceiling: float = 10.0
    local_scale_floor: float = 1e-8
    scale_model_ridge_penalty: float = 1.0
    type_offset_penalty: float = 2.0
    maximum_scale_model_iterations: int = 8

    tpb_hierarchical_prior_variance: float = 1.0
    maximum_tpb_shape_iterations: int = 8
    tpb_shape_learning_rate: float = 0.5
    minimum_tpb_shape: float = 0.1
    maximum_tpb_shape: float = 10.0

    max_inner_newton_iterations: int = 20
    newton_gradient_tolerance: float = 1e-6

    linear_solver_tolerance: float = 1e-6
    maximum_linear_solver_iterations: int = 1024
    logdet_probe_count: int = 12
    logdet_lanczos_steps: int = 20
    exact_solver_matrix_limit: int = 2048  # below this: direct solve; above: Woodbury or CG
    posterior_variance_batch_size: int = 1024
    posterior_variance_probe_count: int = 24
    beta_variance_update_interval: int = 4
    minimum_minor_allele_frequency: float = 1e-3

    sample_space_preconditioner_rank: int = 256
    validation_interval: int = 2
    pipeline_validation_fraction: float = 0.0
    pipeline_validation_min_samples: int = 0
    binary_intercept_calibration: bool = True
    stochastic_variational_updates: bool = True
    stochastic_min_variant_count: int = 4096
    stochastic_variant_batch_size: int = 8192
    stochastic_step_offset: float = 8.0
    stochastic_step_exponent: float = 0.6
    final_posterior_refinement: bool = True
    posterior_working_sets: bool = True
    posterior_working_set_min_variants: int = 65_536
    posterior_working_set_initial_size: int = 8_192
    posterior_working_set_growth: int = 8_192
    posterior_working_set_max_passes: int = 6
    posterior_working_set_coefficient_tolerance: float = 1e-4

    update_hyperparameters: bool = True
    random_seed: int = 0

    def __post_init__(self) -> None:
        if self.max_outer_iterations < 1:
            raise ValueError("max_outer_iterations must be positive.")
        if self.minimum_scale <= 0.0:
            raise ValueError("minimum_scale must be positive.")
        if self.polya_gamma_minimum_weight <= 0.0:
            raise ValueError("polya_gamma_minimum_weight must be positive.")
        if self.prior_scale_floor <= 0.0:
            raise ValueError("prior_scale_floor must be positive.")
        if self.prior_scale_ceiling <= self.prior_scale_floor:
            raise ValueError("prior_scale_ceiling must exceed prior_scale_floor.")
        if self.global_scale_floor <= 0.0:
            raise ValueError("global_scale_floor must be positive.")
        if self.global_scale_ceiling <= self.global_scale_floor:
            raise ValueError("global_scale_ceiling must exceed global_scale_floor.")
        if self.local_scale_floor <= 0.0:
            raise ValueError("local_scale_floor must be positive.")
        if self.maximum_scale_model_iterations < 1:
            raise ValueError("maximum_scale_model_iterations must be positive.")
        if self.minimum_tpb_shape <= 0.0:
            raise ValueError("minimum_tpb_shape must be positive.")
        if self.maximum_tpb_shape <= self.minimum_tpb_shape:
            raise ValueError("maximum_tpb_shape must exceed minimum_tpb_shape.")
        if self.tpb_hierarchical_prior_variance <= 0.0:
            raise ValueError("tpb_hierarchical_prior_variance must be positive.")
        if self.maximum_tpb_shape_iterations < 1:
            raise ValueError("maximum_tpb_shape_iterations must be positive.")
        if self.tpb_shape_learning_rate <= 0.0:
            raise ValueError("tpb_shape_learning_rate must be positive.")
        if self.max_inner_newton_iterations < 1:
            raise ValueError("max_inner_newton_iterations must be positive.")
        if self.newton_gradient_tolerance <= 0.0:
            raise ValueError("newton_gradient_tolerance must be positive.")
        if self.linear_solver_tolerance <= 0.0:
            raise ValueError("linear_solver_tolerance must be positive.")
        if self.maximum_linear_solver_iterations < 1:
            raise ValueError("maximum_linear_solver_iterations must be positive.")
        if self.logdet_probe_count < 1:
            raise ValueError("logdet_probe_count must be positive.")
        if self.logdet_lanczos_steps < 2:
            raise ValueError("logdet_lanczos_steps must be at least 2.")
        if self.exact_solver_matrix_limit < 1:
            raise ValueError("exact_solver_matrix_limit must be positive.")
        if self.posterior_variance_batch_size < 1:
            raise ValueError("posterior_variance_batch_size must be positive.")
        if self.posterior_variance_probe_count < 1:
            raise ValueError("posterior_variance_probe_count must be positive.")
        if self.beta_variance_update_interval < 1:
            raise ValueError("beta_variance_update_interval must be positive.")
        if not 0.0 <= self.minimum_minor_allele_frequency < 0.5:
            raise ValueError("minimum_minor_allele_frequency must lie in [0.0, 0.5).")
        if self.sample_space_preconditioner_rank < 0:
            raise ValueError("sample_space_preconditioner_rank must be non-negative.")
        if self.validation_interval < 1:
            raise ValueError("validation_interval must be positive.")
        if not 0.0 <= self.pipeline_validation_fraction < 1.0:
            raise ValueError("pipeline_validation_fraction must lie in [0.0, 1.0).")
        if self.pipeline_validation_min_samples < 0:
            raise ValueError("pipeline_validation_min_samples must be non-negative.")
        if self.stochastic_min_variant_count < 0:
            raise ValueError("stochastic_min_variant_count must be non-negative.")
        if self.stochastic_variant_batch_size < 1:
            raise ValueError("stochastic_variant_batch_size must be positive.")
        if self.stochastic_step_offset < 0.0:
            raise ValueError("stochastic_step_offset must be non-negative.")
        if not 0.0 < self.stochastic_step_exponent <= 1.0:
            raise ValueError("stochastic_step_exponent must lie in (0.0, 1.0].")
        if self.posterior_working_set_min_variants < 0:
            raise ValueError("posterior_working_set_min_variants must be non-negative.")
        if self.posterior_working_set_initial_size < 1:
            raise ValueError("posterior_working_set_initial_size must be positive.")
        if self.posterior_working_set_growth < 1:
            raise ValueError("posterior_working_set_growth must be positive.")
        if self.posterior_working_set_max_passes < 1:
            raise ValueError("posterior_working_set_max_passes must be positive.")
        if self.posterior_working_set_coefficient_tolerance < 0.0:
            raise ValueError("posterior_working_set_coefficient_tolerance must be non-negative.")

    def class_log_baseline_scales(self) -> Mapping[VariantClass, float]:
        return dict(DEFAULT_CLASS_LOG_BASELINE_SCALE)

    def class_tpb_shape_a(self) -> Mapping[VariantClass, float]:
        return dict(DEFAULT_CLASS_TPB_SHAPE_A)

    def class_tpb_shape_b(self) -> Mapping[VariantClass, float]:
        return dict(DEFAULT_CLASS_TPB_SHAPE_B)

    @staticmethod
    def structural_variant_classes() -> tuple[VariantClass, ...]:
        return STRUCTURAL_VARIANT_CLASSES


@dataclass(slots=True)
class BenchmarkConfig:
    shared_config: ModelConfig
    snv_classes: tuple[VariantClass, ...] = (VariantClass.SNV,)
    top_tail_fraction: float = 0.05

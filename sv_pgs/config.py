from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class TraitType(str, Enum):
    QUANTITATIVE = "quantitative"
    BINARY = "binary"


class JaxDevicePreference(str, Enum):
    DEFAULT = "default"
    GPU = "gpu"
    CPU = "cpu"
    TPU = "tpu"


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
    trait_type: TraitType = TraitType.BINARY
    jax_device_preference: JaxDevicePreference = JaxDevicePreference.GPU
    jax_device_index: int = 0
    require_jax_device: bool = False
    max_outer_iterations: int = 30
    convergence_tolerance: float = 1e-4
    minimum_scale: float = 1e-6
    polya_gamma_minimum_weight: float = 1e-4
    sigma_error_floor: float = 1e-3
    minimum_structural_variant_carriers: int = 5
    ld_block_max_variants: int = 512
    ld_block_window_bp: int = 3_000_000
    discarded_spectrum_tolerance: float = 0.005
    block_jitter_floor: float = 1e-6
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

    regularized_horseshoe_slab_scale: float = 2.0
    enable_horseshoe_slab: bool = True

    max_inner_newton_iterations: int = 20
    newton_gradient_tolerance: float = 1e-5
    trust_region_initial_damping: float = 1.0
    trust_region_damping_increase_factor: float = 10.0
    trust_region_damping_decrease_factor: float = 0.1
    trust_region_success_threshold: float = 0.25
    trust_region_minimum_damping: float = 1e-8

    update_hyperparameters: bool = True
    prior_version: str = "collapsed-laplace-em-tpb-gamma-gamma-v1"
    transform_version: str = "numeric-impute-standardize-v2"
    random_seed: int = 0

    def __post_init__(self) -> None:
        if self.max_outer_iterations < 1:
            raise ValueError("max_outer_iterations must be positive.")
        if self.jax_device_index < 0:
            raise ValueError("jax_device_index must be non-negative.")
        if self.minimum_scale <= 0.0:
            raise ValueError("minimum_scale must be positive.")
        if self.polya_gamma_minimum_weight <= 0.0:
            raise ValueError("polya_gamma_minimum_weight must be positive.")
        if self.minimum_structural_variant_carriers < 1:
            raise ValueError("minimum_structural_variant_carriers must be positive.")
        if self.discarded_spectrum_tolerance <= 0.0:
            raise ValueError("discarded_spectrum_tolerance must be positive.")
        if self.discarded_spectrum_tolerance >= 1.0:
            raise ValueError("discarded_spectrum_tolerance must be less than 1.")
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
        if self.regularized_horseshoe_slab_scale <= 0.0:
            raise ValueError("regularized_horseshoe_slab_scale must be positive.")
        if self.max_inner_newton_iterations < 1:
            raise ValueError("max_inner_newton_iterations must be positive.")
        if self.newton_gradient_tolerance <= 0.0:
            raise ValueError("newton_gradient_tolerance must be positive.")
        if self.trust_region_initial_damping <= 0.0:
            raise ValueError("trust_region_initial_damping must be positive.")
        if self.trust_region_damping_increase_factor <= 1.0:
            raise ValueError("trust_region_damping_increase_factor must exceed 1.")
        if not 0.0 < self.trust_region_damping_decrease_factor < 1.0:
            raise ValueError("trust_region_damping_decrease_factor must lie in (0, 1).")
        if not 0.0 < self.trust_region_success_threshold < 1.0:
            raise ValueError("trust_region_success_threshold must lie in (0, 1).")
        if self.trust_region_minimum_damping <= 0.0:
            raise ValueError("trust_region_minimum_damping must be positive.")

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
    snv_classes: tuple[VariantClass, ...] = (VariantClass.SNV,)
    top_tail_fraction: float = 0.05
    prevalence: float | None = None
    shared_config: ModelConfig = field(default_factory=ModelConfig)

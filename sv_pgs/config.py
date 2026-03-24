from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

import numpy as np


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


DEFAULT_CLASS_LOG_PRIOR_SCALE = {
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

DEFAULT_CLASS_TAIL_SHAPES = {
    VariantClass.SNV: 6.0,
    VariantClass.SMALL_INDEL: 5.0,
    VariantClass.DELETION_SHORT: 3.8,
    VariantClass.DELETION_LONG: 3.2,
    VariantClass.DUPLICATION_SHORT: 3.6,
    VariantClass.DUPLICATION_LONG: 3.0,
    VariantClass.INSERTION_MEI: 3.4,
    VariantClass.INVERSION_BND_COMPLEX: 2.8,
    VariantClass.STR_VNTR_REPEAT: 3.1,
    VariantClass.OTHER_COMPLEX_SV: 3.0,
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
    max_outer_iterations: int = 30
    max_pcg_iterations: int = 200
    pcg_tolerance: float = 1e-5
    operator_tile_size: int = 256
    convergence_tolerance: float = 1e-4
    minimum_scale: float = 1e-6
    polya_gamma_minimum_weight: float = 1e-4
    sigma_error_floor: float = 1e-3
    minimum_structural_variant_carriers: int = 2
    duplicate_signature_decimals: int = 6
    ld_block_max_variants: int = 512
    ld_block_window_bp: int = 3_000_000
    discarded_spectrum_tolerance: float = 0.005
    block_jitter_floor: float = 1e-6
    prior_scale_floor: float = 1e-6
    prior_scale_ceiling: float = 10.0
    minimum_tail_shape: float = 0.25
    maximum_tail_shape: float = 20.0
    scale_model_ridge_penalty: float = 1.0
    type_offset_penalty: float = 2.0
    maximum_scale_model_iterations: int = 8
    maximum_tail_shape_iterations: int = 8
    update_hyperparameters: bool = True
    prior_version: str = "metadata-conditioned-continuous-shrinkage-v1"
    transform_version: str = "numeric-impute-standardize-v2"
    random_seed: int = 0

    def __post_init__(self) -> None:
        if self.max_outer_iterations < 1:
            raise ValueError("max_outer_iterations must be positive.")
        if self.max_pcg_iterations < 1:
            raise ValueError("max_pcg_iterations must be positive.")
        if self.pcg_tolerance <= 0.0:
            raise ValueError("pcg_tolerance must be positive.")
        if self.operator_tile_size < 1:
            raise ValueError("operator_tile_size must be positive.")
        if self.minimum_scale <= 0.0:
            raise ValueError("minimum_scale must be positive.")
        if self.polya_gamma_minimum_weight <= 0.0:
            raise ValueError("polya_gamma_minimum_weight must be positive.")
        if self.discarded_spectrum_tolerance <= 0.0:
            raise ValueError("discarded_spectrum_tolerance must be positive.")
        if self.discarded_spectrum_tolerance >= 1.0:
            raise ValueError("discarded_spectrum_tolerance must be less than 1.")
        if self.prior_scale_floor <= 0.0:
            raise ValueError("prior_scale_floor must be positive.")
        if self.prior_scale_ceiling <= self.prior_scale_floor:
            raise ValueError("prior_scale_ceiling must exceed prior_scale_floor.")
        if self.minimum_tail_shape <= 0.0:
            raise ValueError("minimum_tail_shape must be positive.")
        if self.maximum_tail_shape <= self.minimum_tail_shape:
            raise ValueError("maximum_tail_shape must exceed minimum_tail_shape.")
        if self.maximum_tail_shape_iterations < 1:
            raise ValueError("maximum_tail_shape_iterations must be positive.")

    def class_log_prior_scales(self) -> Mapping[VariantClass, float]:
        return dict(DEFAULT_CLASS_LOG_PRIOR_SCALE)

    def class_tail_shapes(self) -> Mapping[VariantClass, float]:
        return dict(DEFAULT_CLASS_TAIL_SHAPES)

    @staticmethod
    def structural_variant_classes() -> tuple[VariantClass, ...]:
        return STRUCTURAL_VARIANT_CLASSES


@dataclass(slots=True)
class BenchmarkConfig:
    snv_classes: tuple[VariantClass, ...] = (VariantClass.SNV,)
    top_tail_fraction: float = 0.05
    prevalence: float | None = None
    shared_config: ModelConfig = field(default_factory=ModelConfig)

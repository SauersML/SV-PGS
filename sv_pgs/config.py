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

DEFAULT_CLASS_MIXTURE_WEIGHTS = {
    VariantClass.SNV: np.array([0.68, 0.20, 0.08, 0.03, 0.01], dtype=np.float32),
    VariantClass.SMALL_INDEL: np.array([0.64, 0.22, 0.09, 0.04, 0.01], dtype=np.float32),
    VariantClass.DELETION_SHORT: np.array([0.56, 0.24, 0.12, 0.06, 0.02], dtype=np.float32),
    VariantClass.DELETION_LONG: np.array([0.50, 0.25, 0.14, 0.08, 0.03], dtype=np.float32),
    VariantClass.DUPLICATION_SHORT: np.array([0.54, 0.24, 0.12, 0.07, 0.03], dtype=np.float32),
    VariantClass.DUPLICATION_LONG: np.array([0.48, 0.25, 0.15, 0.09, 0.03], dtype=np.float32),
    VariantClass.INSERTION_MEI: np.array([0.53, 0.24, 0.13, 0.07, 0.03], dtype=np.float32),
    VariantClass.INVERSION_BND_COMPLEX: np.array([0.45, 0.26, 0.16, 0.10, 0.03], dtype=np.float32),
    VariantClass.STR_VNTR_REPEAT: np.array([0.50, 0.25, 0.14, 0.08, 0.03], dtype=np.float32),
    VariantClass.OTHER_COMPLEX_SV: np.array([0.48, 0.25, 0.15, 0.09, 0.03], dtype=np.float32),
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
    convergence_tolerance: float = 1e-4
    tile_size: int = 256
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
    mixture_variance_multipliers: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
    dirichlet_strength: float = 24.0
    scale_model_ridge_penalty: float = 1.0
    type_offset_penalty: float = 2.0
    maximum_scale_model_iterations: int = 8
    update_hyperparameters: bool = True
    prior_version: str = "metadata-conditioned-bayesr-mixture-v1"
    transform_version: str = "numeric-impute-standardize-v2"
    random_seed: int = 0

    def __post_init__(self) -> None:
        if self.max_outer_iterations < 1:
            raise ValueError("max_outer_iterations must be positive.")
        if self.tile_size < 1:
            raise ValueError("tile_size must be positive.")
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
        if len(self.mixture_variance_multipliers) < 2:
            raise ValueError("mixture_variance_multipliers must have at least two components.")
        if any(component <= 0.0 for component in self.mixture_variance_multipliers):
            raise ValueError("mixture_variance_multipliers must be positive.")
        if tuple(sorted(self.mixture_variance_multipliers)) != self.mixture_variance_multipliers:
            raise ValueError("mixture_variance_multipliers must be sorted ascending.")
        if self.dirichlet_strength <= 0.0:
            raise ValueError("dirichlet_strength must be positive.")

    def class_log_prior_scales(self) -> Mapping[VariantClass, float]:
        return dict(DEFAULT_CLASS_LOG_PRIOR_SCALE)

    def class_mixture_weights(self) -> Mapping[VariantClass, np.ndarray]:
        normalized_weights: dict[VariantClass, np.ndarray] = {}
        component_count = len(self.mixture_variance_multipliers)
        for variant_class, mixture_weights in DEFAULT_CLASS_MIXTURE_WEIGHTS.items():
            if mixture_weights.shape[0] != component_count:
                raise ValueError(
                    f"Class prior for {variant_class.value} does not match mixture component count."
                )
            normalized_weights[variant_class] = mixture_weights / np.sum(mixture_weights)
        return normalized_weights

    @staticmethod
    def structural_variant_classes() -> tuple[VariantClass, ...]:
        return STRUCTURAL_VARIANT_CLASSES


@dataclass(slots=True)
class BenchmarkConfig:
    snv_classes: tuple[VariantClass, ...] = (VariantClass.SNV,)
    top_tail_fraction: float = 0.05
    prevalence: float | None = None
    shared_config: ModelConfig = field(default_factory=ModelConfig)

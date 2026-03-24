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

MIXTURE_COMPONENT_COUNT = 5

DEFAULT_COMPONENT_VARIANCE_FRACTIONS: tuple[float, ...] = (
    1e-6,
    1e-4,
    1e-3,
    1e-2,
    1e-1,
)

DEFAULT_CLASS_MIXTURE_WEIGHTS: dict[VariantClass, tuple[float, ...]] = {
    VariantClass.SNV:                   (0.40, 0.30, 0.15, 0.10, 0.05),
    VariantClass.SMALL_INDEL:           (0.38, 0.28, 0.17, 0.11, 0.06),
    VariantClass.DELETION_SHORT:        (0.32, 0.26, 0.20, 0.14, 0.08),
    VariantClass.DELETION_LONG:         (0.28, 0.24, 0.22, 0.16, 0.10),
    VariantClass.DUPLICATION_SHORT:     (0.30, 0.25, 0.21, 0.15, 0.09),
    VariantClass.DUPLICATION_LONG:      (0.28, 0.24, 0.22, 0.16, 0.10),
    VariantClass.INSERTION_MEI:         (0.30, 0.25, 0.21, 0.15, 0.09),
    VariantClass.INVERSION_BND_COMPLEX: (0.26, 0.23, 0.23, 0.17, 0.11),
    VariantClass.STR_VNTR_REPEAT:       (0.30, 0.25, 0.21, 0.15, 0.09),
    VariantClass.OTHER_COMPLEX_SV:      (0.28, 0.24, 0.22, 0.16, 0.10),
}


@dataclass(slots=True)
class ModelConfig:
    trait_type: TraitType = TraitType.BINARY

    # BayesR mixture prior
    component_variance_fractions: tuple[float, ...] = DEFAULT_COMPONENT_VARIANCE_FRACTIONS
    dirichlet_concentration: float = 10.0
    base_genetic_variance: float = 0.01

    # Smooth metadata prior adjustments
    metadata_ridge_penalty: float = 1.0
    update_hyperparameters: bool = True

    # LD blocks
    ld_block_max_variants: int = 512
    ld_block_window_bp: int = 3_000_000
    eigenvalue_tolerance: float = 0.005
    block_jitter_floor: float = 1e-6

    # Solver
    max_outer_iterations: int = 30
    convergence_tolerance: float = 1e-4
    tile_size: int = 256

    # Data processing
    minimum_scale: float = 1e-6
    minimum_structural_variant_carriers: int = 2
    duplicate_signature_decimals: int = 6

    # Likelihood
    polya_gamma_minimum_weight: float = 1e-4
    sigma_error_floor: float = 1e-3

    # Misc
    random_seed: int = 0
    transform_version: str = "blockwise-bayesr-v1"

    def __post_init__(self) -> None:
        if len(self.component_variance_fractions) != MIXTURE_COMPONENT_COUNT:
            raise ValueError(
                "component_variance_fractions must have exactly "
                + str(MIXTURE_COMPONENT_COUNT) + " entries."
            )
        if self.eigenvalue_tolerance <= 0.0 or self.eigenvalue_tolerance >= 1.0:
            raise ValueError("eigenvalue_tolerance must be in (0, 1).")
        if self.max_outer_iterations < 1:
            raise ValueError("max_outer_iterations must be positive.")

    def component_variances(self) -> np.ndarray:
        return self.base_genetic_variance * np.asarray(
            self.component_variance_fractions, dtype=np.float64,
        )

    def class_mixture_weights(self) -> Mapping[VariantClass, np.ndarray]:
        return {
            variant_class: np.asarray(weights, dtype=np.float64)
            for variant_class, weights in DEFAULT_CLASS_MIXTURE_WEIGHTS.items()
        }

    @staticmethod
    def structural_variant_classes() -> tuple[VariantClass, ...]:
        return STRUCTURAL_VARIANT_CLASSES


@dataclass(slots=True)
class BenchmarkConfig:
    snv_classes: tuple[VariantClass, ...] = (VariantClass.SNV,)
    top_tail_fraction: float = 0.05
    prevalence: float | None = None
    shared_config: ModelConfig = field(default_factory=ModelConfig)

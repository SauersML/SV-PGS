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


DEFAULT_CLASS_PRIOR = {
    VariantClass.SNV: np.array([0.70, 0.20, 0.08, 0.02], dtype=np.float32),
    VariantClass.SMALL_INDEL: np.array([0.68, 0.22, 0.08, 0.02], dtype=np.float32),
    VariantClass.DELETION_SHORT: np.array([0.62, 0.24, 0.10, 0.04], dtype=np.float32),
    VariantClass.DELETION_LONG: np.array([0.58, 0.25, 0.12, 0.05], dtype=np.float32),
    VariantClass.DUPLICATION_SHORT: np.array([0.60, 0.24, 0.11, 0.05], dtype=np.float32),
    VariantClass.DUPLICATION_LONG: np.array([0.56, 0.25, 0.13, 0.06], dtype=np.float32),
    VariantClass.INSERTION_MEI: np.array([0.60, 0.24, 0.11, 0.05], dtype=np.float32),
    VariantClass.INVERSION_BND_COMPLEX: np.array([0.55, 0.25, 0.14, 0.06], dtype=np.float32),
    VariantClass.STR_VNTR_REPEAT: np.array([0.60, 0.24, 0.11, 0.05], dtype=np.float32),
    VariantClass.OTHER_COMPLEX_SV: np.array([0.58, 0.25, 0.12, 0.05], dtype=np.float32),
}


@dataclass(slots=True)
class ModelConfig:
    trait_type: TraitType = TraitType.BINARY
    mixture_components: int = 4
    max_outer_iters: int = 30
    max_inner_pcg_iters: int = 200
    pcg_tolerance: float = 1e-5
    convergence_tolerance: float = 1e-4
    covariance_max_block_exact: int = 32
    covariance_max_block_dense: int = 96
    covariance_low_rank: int = 8
    tile_size: int = 256
    prior_floor_variance: float = 1e-4
    prior_variance_growth: tuple[float, ...] = (1.0, 10.0, 100.0, 1000.0)
    dirichlet_strength: float = 25.0
    variance_shrinkage: float = 0.35
    variance_min_gap_log: float = 0.4
    same_class_edge_strength: float = 0.6
    cross_class_edge_strength: float = 0.25
    correlation_threshold: float = 0.98
    graph_window_bp: int = 2_000_000
    max_cluster_span: int = 512
    min_scale: float = 1e-6
    pg_min_weight: float = 1e-4
    sigma_e_prior: float = 1e-3
    graph_metadata_version: str = "v1"
    transform_version: str = "numeric-impute-standardize-v1"
    random_seed: int = 0

    def __post_init__(self) -> None:
        if self.mixture_components != 4:
            raise ValueError("This implementation requires exactly 4 mixture components.")
        if len(self.prior_variance_growth) != self.mixture_components:
            raise ValueError("prior_variance_growth must match mixture_components.")
        if self.covariance_low_rank < 0:
            raise ValueError("covariance_low_rank must be non-negative.")

    def base_component_variances(self) -> np.ndarray:
        growth = np.asarray(self.prior_variance_growth, dtype=np.float32)
        return self.prior_floor_variance * growth

    def class_prior_weights(self) -> Mapping[VariantClass, np.ndarray]:
        priors: dict[VariantClass, np.ndarray] = {}
        for variant_class, weights in DEFAULT_CLASS_PRIOR.items():
            weights = np.asarray(weights, dtype=np.float32)
            if weights.shape[0] != self.mixture_components:
                raise ValueError(
                    f"Class prior for {variant_class.value} does not match mixture components."
                )
            priors[variant_class] = weights / weights.sum()
        return priors


@dataclass(slots=True)
class BenchmarkConfig:
    snv_classes: tuple[VariantClass, ...] = (VariantClass.SNV,)
    top_tail_fraction: float = 0.05
    prevalence: float | None = None
    metric_eps: float = 1e-8
    shared_config: ModelConfig = field(default_factory=ModelConfig)

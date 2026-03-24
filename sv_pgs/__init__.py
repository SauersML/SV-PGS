"""SV-PGS: Class-adaptive, graph-coupled Bayesian GLM for polygenic scoring."""

from sv_pgs.benchmark import run_benchmark_suite
from sv_pgs.config import BenchmarkConfig, ModelConfig, TraitType, VariantClass
from sv_pgs.data import VariantRecord
from sv_pgs.model import BayesianPGS

__all__ = [
    "BayesianPGS",
    "BenchmarkConfig",
    "ModelConfig",
    "TraitType",
    "VariantClass",
    "VariantRecord",
    "run_benchmark_suite",
]

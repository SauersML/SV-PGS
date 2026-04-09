"""SV-PGS: single-device joint empirical-Bayes GLM for polygenic scoring with structural variants."""

from __future__ import annotations

from sv_pgs.benchmark import run_benchmark_suite
from sv_pgs.config import BenchmarkConfig, ModelConfig, TraitType, VariantClass
from sv_pgs.data import VariantRecord
from sv_pgs.io import load_dataset_from_files
from sv_pgs.model import BayesianPGS
from sv_pgs.pipeline import run_training_pipeline

__all__ = [
    "BayesianPGS",
    "BenchmarkConfig",
    "ModelConfig",
    "TraitType",
    "VariantClass",
    "VariantRecord",
    "load_dataset_from_files",
    "run_benchmark_suite",
    "run_training_pipeline",
]

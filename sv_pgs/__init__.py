"""SV-PGS: single-device joint empirical-Bayes GLM for polygenic scoring with structural variants."""

from __future__ import annotations

from sv_pgs.benchmark import run_benchmark_suite
from sv_pgs.config import BenchmarkConfig, InferenceBackend, ModelConfig, TraitType, VariantClass
from sv_pgs.data import VariantRecord
from sv_pgs.io import load_dataset_from_files, run_training_pipeline
from sv_pgs.model import BayesianPGS

_NON_EXPORTS = frozenset(
    {
        "AllOfUsDiseaseRequest",
        "available_disease_names",
        "prepare_all_of_us_disease_sample_table",
    }
)


def __getattr__(name: str):
    if name in _NON_EXPORTS:
        raise AttributeError(
            f"module {__name__!r} does not export {name!r}; import it from 'sv_pgs.all_of_us' instead."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BayesianPGS",
    "BenchmarkConfig",
    "InferenceBackend",
    "ModelConfig",
    "TraitType",
    "VariantClass",
    "VariantRecord",
    "load_dataset_from_files",
    "run_benchmark_suite",
    "run_training_pipeline",
]

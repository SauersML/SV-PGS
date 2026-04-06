"""SV-PGS: single-device joint empirical-Bayes GLM for polygenic scoring with structural variants."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from sv_pgs.benchmark import run_benchmark_suite
from sv_pgs.config import BenchmarkConfig, ModelConfig, TraitType, VariantClass
from sv_pgs.data import VariantRecord
from sv_pgs.io import load_dataset_from_files, run_training_pipeline
from sv_pgs.model import BayesianPGS

if TYPE_CHECKING:
    from sv_pgs.all_of_us import AllOfUsDiseaseRequest, available_disease_names, prepare_all_of_us_disease_sample_table

_LAZY_ALL_OF_US_EXPORTS = frozenset(
    {
        "AllOfUsDiseaseRequest",
        "available_disease_names",
        "prepare_all_of_us_disease_sample_table",
    }
)


def __getattr__(name: str) -> Any:
    if name not in _LAZY_ALL_OF_US_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        all_of_us_module = import_module("sv_pgs.all_of_us")
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("google"):
            raise ModuleNotFoundError(
                "All of Us helpers require google-cloud-bigquery. "
                "Install that dependency before importing All of Us exports from sv_pgs."
            ) from exc
        raise
    value = getattr(all_of_us_module, name)
    globals()[name] = value
    return value

__all__ = [
    "AllOfUsDiseaseRequest",
    "BayesianPGS",
    "BenchmarkConfig",
    "ModelConfig",
    "TraitType",
    "VariantClass",
    "VariantRecord",
    "available_disease_names",
    "load_dataset_from_files",
    "prepare_all_of_us_disease_sample_table",
    "run_benchmark_suite",
    "run_training_pipeline",
]

"""SV-PGS: single-device joint empirical-Bayes GLM for polygenic scoring with structural variants."""

from __future__ import annotations

from importlib import import_module

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

_EXPORTS = {
    "AllOfUsDiseaseRequest": ("sv_pgs.all_of_us", "AllOfUsDiseaseRequest"),
    "BayesianPGS": ("sv_pgs.model", "BayesianPGS"),
    "BenchmarkConfig": ("sv_pgs.config", "BenchmarkConfig"),
    "ModelConfig": ("sv_pgs.config", "ModelConfig"),
    "TraitType": ("sv_pgs.config", "TraitType"),
    "VariantClass": ("sv_pgs.config", "VariantClass"),
    "VariantRecord": ("sv_pgs.data", "VariantRecord"),
    "available_disease_names": ("sv_pgs.all_of_us", "available_disease_names"),
    "load_dataset_from_files": ("sv_pgs.io", "load_dataset_from_files"),
    "prepare_all_of_us_disease_sample_table": ("sv_pgs.all_of_us", "prepare_all_of_us_disease_sample_table"),
    "run_benchmark_suite": ("sv_pgs.benchmark", "run_benchmark_suite"),
    "run_training_pipeline": ("sv_pgs.io", "run_training_pipeline"),
}


def __getattr__(name: str) -> object:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _EXPORTS[name]
    return getattr(import_module(module_name), attribute_name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

"""Pin: ``_compute_metrics`` handles single-class targets without crashing.

If the held-out cohort happens to be entirely cases (or entirely controls),
``roc_auc_score``/``log_loss`` from sklearn raise ``ValueError`` ("Only one
class present in y_true"). The benchmark must catch that condition and return
``auc=None, pr_auc=None, log_loss=None`` so the upstream evaluation logs the
gap instead of crashing the whole run.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from sv_pgs import benchmark as benchmark_module
from sv_pgs.config import BenchmarkConfig, ModelConfig, TraitType


class _FakeBinaryModel:
    def __init__(self) -> None:
        self.config = ModelConfig(trait_type=TraitType.BINARY)

    def decision_function(self, genotypes, covariates):
        # Return deterministic logit scores so we exercise the metric paths.
        return np.linspace(-1.0, 1.0, genotypes.shape[0], dtype=np.float32)


class _FakeQuantitativeModel:
    def __init__(self) -> None:
        self.config = ModelConfig(trait_type=TraitType.QUANTITATIVE)

    def decision_function(self, genotypes, covariates):
        return np.linspace(0.0, 1.0, genotypes.shape[0], dtype=np.float32)


def _bench_config() -> BenchmarkConfig:
    return BenchmarkConfig(shared_config=ModelConfig(trait_type=TraitType.BINARY))


@pytest.mark.parametrize("constant_label", [0, 1])
def test_binary_single_class_targets_return_none_metrics(constant_label):
    """All-zero or all-one targets → metrics None, no exception."""
    n = 64
    model = _FakeBinaryModel()
    metrics = benchmark_module._compute_metrics(
        model=model,
        genotypes=np.zeros((n, 4), dtype=np.float32),
        covariates=np.zeros((n, 1), dtype=np.float32),
        targets=np.full(n, constant_label, dtype=np.float32),
        benchmark_config=_bench_config(),
    )
    assert metrics.auc is None
    assert metrics.pr_auc is None
    assert metrics.log_loss is None
    # top_tail_enrichment stays finite (returns 0.0 in the controls-only case).
    assert metrics.top_tail_enrichment == 0.0 or np.isfinite(metrics.top_tail_enrichment)


def test_binary_both_classes_produces_real_metrics():
    """Sanity: with both classes present the metrics are floats, not None."""
    n = 100
    model = _FakeBinaryModel()
    targets = np.zeros(n, dtype=np.float32)
    targets[n // 2 :] = 1.0
    metrics = benchmark_module._compute_metrics(
        model=model,
        genotypes=np.zeros((n, 4), dtype=np.float32),
        covariates=np.zeros((n, 1), dtype=np.float32),
        targets=targets,
        benchmark_config=_bench_config(),
    )
    assert metrics.auc is not None and 0.0 <= metrics.auc <= 1.0
    assert metrics.pr_auc is not None
    assert metrics.log_loss is not None and metrics.log_loss > 0.0


def test_quantitative_trait_uses_r2_not_auc():
    """Quantitative trait path returns r2, leaves AUC None."""
    n = 64
    model = _FakeQuantitativeModel()
    targets = np.linspace(0.0, 1.0, n, dtype=np.float32) + 1e-3
    metrics = benchmark_module._compute_metrics(
        model=model,
        genotypes=np.zeros((n, 4), dtype=np.float32),
        covariates=np.zeros((n, 1), dtype=np.float32),
        targets=targets,
        benchmark_config=_bench_config(),
    )
    assert metrics.auc is None
    assert metrics.r2 is not None

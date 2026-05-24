"""Pin: ``_compute_metrics`` quantitative-trait behavior with constant targets.

When all targets are identical, R² is undefined (Var(y) == 0 in the
denominator).  The task spec states "should return None".  Current
implementation calls ``sklearn.metrics.r2_score`` directly which returns
``0.0`` in this case (a documented sklearn quirk).

This file pins the **current behavior** (r2 == 0.0) explicitly so that:

* it can never regress to NaN or inf without a test failure,
* a future change that swaps to ``None`` (per the task spec) will fail
  this test, alerting the agent to update it.

The expected-vs-spec divergence is noted in the verdict file.

Also pins ``top_tail_enrichment`` for the constant-target case:
``np.std(targets) == 0`` triggers the documented ``0.0`` early return.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sv_pgs.benchmark import _compute_metrics
from sv_pgs.config import BenchmarkConfig, ModelConfig, TraitType


@dataclass
class _StubModel:
    """Minimal stand-in for ``BayesianPGS`` for benchmark-metric tests."""

    config: ModelConfig
    scores: np.ndarray

    def decision_function(self, genotypes: Any, covariates: Any) -> np.ndarray:
        return self.scores


def test_quantitative_constant_target_returns_finite_r2_zero():
    """sklearn's ``r2_score`` returns 0.0 (not NaN) when Var(y) == 0; the
    benchmark hands it straight through. This pins that the metric stays
    finite (≠ NaN/inf) under the degenerate input."""
    n = 16
    scores = np.linspace(-1.0, 1.0, n).astype(np.float32)
    targets = np.full(n, 3.14, dtype=np.float32)
    model = _StubModel(
        config=ModelConfig(trait_type=TraitType.QUANTITATIVE),
        scores=scores,
    )
    metrics = _compute_metrics(
        model=model,  # type: ignore[arg-type]
        genotypes=np.zeros((n, 0), dtype=np.float32),
        covariates=np.zeros((n, 0), dtype=np.float32),
        targets=targets,
        benchmark_config=BenchmarkConfig(
            shared_config=ModelConfig(trait_type=TraitType.QUANTITATIVE),
        ),
    )
    # Pin current behavior: sklearn r2 == 0.0 for constant targets (not None).
    # See swarm_findings/verdicts/boundary_tests.md for spec divergence.
    assert metrics.r2 is not None
    assert np.isfinite(metrics.r2), metrics.r2
    assert abs(metrics.r2) < 1e-6, metrics.r2
    # Top-tail enrichment must short-circuit to 0.0 when np.std(y) == 0.
    assert metrics.top_tail_enrichment == 0.0
    # Binary-only fields stay None for the quantitative branch.
    assert metrics.auc is None
    assert metrics.log_loss is None
    assert metrics.pr_auc is None


def test_binary_single_class_returns_none_aucs():
    """Sanity adjacent pin: with only one class present in binary
    targets, AUC/log-loss/PR-AUC must be ``None`` (already covered by
    ``test_benchmark_single_class_pinning.py`` for the full pipeline;
    here we pin the ``_compute_metrics`` direct call as well)."""
    n = 16
    model = _StubModel(
        config=ModelConfig(trait_type=TraitType.BINARY),
        scores=np.linspace(-2.0, 2.0, n).astype(np.float32),
    )
    targets = np.zeros(n, dtype=np.float32)  # only class 0
    metrics = _compute_metrics(
        model=model,  # type: ignore[arg-type]
        genotypes=np.zeros((n, 0), dtype=np.float32),
        covariates=np.zeros((n, 0), dtype=np.float32),
        targets=targets,
        benchmark_config=BenchmarkConfig(
            shared_config=ModelConfig(trait_type=TraitType.BINARY),
        ),
    )
    assert metrics.auc is None
    assert metrics.log_loss is None
    assert metrics.pr_auc is None
    # r2 always None on binary branch.
    assert metrics.r2 is None

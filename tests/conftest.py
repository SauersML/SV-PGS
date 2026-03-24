"""Shared test fixtures for SV-PGS."""

from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.config import ModelConfig, TraitType, VariantClass
from sv_pgs.data import VariantRecord


def _make_records(p: int, classes: list[VariantClass] | None = None) -> list[VariantRecord]:
    """Create synthetic VariantRecords."""
    if classes is None:
        all_classes = list(VariantClass)
        classes = [all_classes[i % len(all_classes)] for i in range(p)]
    records = []
    for j in range(p):
        records.append(VariantRecord(
            variant_id=f"var_{j}",
            variant_class=classes[j],
            length_bin="short" if j % 2 == 0 else "long",
            chromosome=f"chr{1 + j % 22}",
            position=1000 * j,
            quality=max(0.5, 1.0 - 0.01 * (j % 20)),
            cluster_id=f"cluster_{j // 5}" if j % 3 == 0 else None,
        ))
    return records


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def small_dataset(rng):
    """Small synthetic dataset: 100 samples, 20 variants, 3 covariates."""
    n, p, d = 100, 20, 3
    X = rng.standard_normal((n, p)).astype(np.float32)
    C = rng.standard_normal((n, d)).astype(np.float32)
    true_beta = np.zeros(p, dtype=np.float32)
    true_beta[:5] = rng.standard_normal(5).astype(np.float32) * 0.5
    true_alpha = rng.standard_normal(d + 1).astype(np.float32) * 0.3  # +1 for intercept
    C_with_intercept = np.column_stack([np.ones(n), C]).astype(np.float32)
    eta = X @ true_beta + C_with_intercept @ true_alpha
    y_cont = eta + rng.standard_normal(n).astype(np.float32) * 0.5
    y_binary = (rng.random(n) < 1.0 / (1.0 + np.exp(-eta))).astype(np.float32)
    records = _make_records(p)
    return {
        "X": X, "C": C, "y_cont": y_cont, "y_binary": y_binary,
        "records": records, "true_beta": true_beta, "n": n, "p": p, "d": d,
    }


@pytest.fixture
def binary_config() -> ModelConfig:
    return ModelConfig(trait_type=TraitType.BINARY, max_outer_iters=5)


@pytest.fixture
def quant_config() -> ModelConfig:
    return ModelConfig(trait_type=TraitType.QUANTITATIVE, max_outer_iters=5)

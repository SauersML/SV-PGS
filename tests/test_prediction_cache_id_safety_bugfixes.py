"""Regression test for prediction-cache sample-ID safety.

Bug: ``_write_predictions_and_summary`` formerly considered the cached
training decision components reusable whenever the *shape* matched —
which silently borrowed training-cohort scores for a held-out cohort
with the same N. The fix tightens the guard: the cache fast-path is
only taken when

  (a) the caller explicitly flags ``is_training_dataset=True``, AND
  (b) the FittedState's ``training_sample_ids`` list matches
      ``dataset.sample_ids`` exactly (length AND order).

Anything else must fall through to ``model.decision_components(...)``.
"""
from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest

from sv_pgs import pipeline as pipeline_module
from sv_pgs.config import ModelConfig, TraitType


class _FakeFittedState:
    """Minimal stand-in for FittedState carrying the training_sample_ids field."""

    def __init__(self, sample_ids: list[str]) -> None:
        self.training_sample_ids = list(sample_ids)


class _FakeModel:
    def __init__(
        self,
        config: ModelConfig,
        training_sample_ids: list[str],
        cached_components: tuple[np.ndarray, np.ndarray],
    ) -> None:
        self.config = config
        self.state = _FakeFittedState(training_sample_ids)
        self._cached_components = cached_components
        self.decision_components_calls = 0

    def training_decision_components(self):
        return self._cached_components

    def decision_components(self, genotypes, covariates):
        # Record fall-through and return distinct values so any test assertion
        # on the written predictions can distinguish "used cache" vs "recomputed".
        self.decision_components_calls += 1
        n = genotypes.shape[0]
        return np.full(n, 0.111, dtype=np.float32), np.full(n, 0.222, dtype=np.float32)


class _FakeRawGenotypes:
    def __init__(self, n_samples: int, n_variants: int) -> None:
        self.shape = (n_samples, n_variants)


def _make_dataset(sample_ids: list[str]) -> types.SimpleNamespace:
    n = len(sample_ids)
    return types.SimpleNamespace(
        sample_ids=list(sample_ids),
        genotypes=_FakeRawGenotypes(n, 5),
        covariates=np.zeros((n, 1), dtype=np.float32),
        targets=np.zeros(n, dtype=np.float32),
    )


def test_cache_fallthrough_when_sample_ids_differ(tmp_path: Path):
    """Same shape, different IDs → MUST recompute, not reuse the cache."""
    training_ids = [f"train_{i}" for i in range(8)]
    holdout_ids = [f"holdout_{i}" for i in range(8)]
    cached = (
        np.full(8, 9.999, dtype=np.float32),
        np.full(8, 9.999, dtype=np.float32),
    )
    model = _FakeModel(
        config=ModelConfig(trait_type=TraitType.QUANTITATIVE),
        training_sample_ids=training_ids,
        cached_components=cached,
    )
    dataset = _make_dataset(holdout_ids)

    pipeline_module._write_predictions_and_summary(
        tmp_path / "predictions.tsv",
        dataset=dataset,
        model=model,
        is_training_dataset=True,  # caller asserts "training" but IDs disagree
    )
    assert model.decision_components_calls == 1, (
        "decision_components must be called when sample IDs differ — "
        "the cache fast-path is unsafe in that case."
    )

    # The written predictions should reflect the recomputed values
    # (0.111 + 0.222 = 0.333), not the bogus 9.999 cached scores.
    rows = (tmp_path / "predictions.tsv").read_text(encoding="utf-8").splitlines()
    # Header + 8 sample rows.
    assert len(rows) == 9
    for row in rows[1:]:
        cells = row.split("\t")
        genetic_score = float(cells[2])
        covariate_score = float(cells[3])
        assert genetic_score == pytest.approx(0.111)
        assert covariate_score == pytest.approx(0.222)


def test_cache_fallthrough_when_not_training_dataset(tmp_path: Path):
    """is_training_dataset=False (held-out) → never use the cache, even
    if every other invariant happens to line up."""
    ids = [f"s_{i}" for i in range(5)]
    cached = (
        np.full(5, 7.7, dtype=np.float32),
        np.full(5, 8.8, dtype=np.float32),
    )
    model = _FakeModel(
        config=ModelConfig(trait_type=TraitType.QUANTITATIVE),
        training_sample_ids=ids,
        cached_components=cached,
    )
    dataset = _make_dataset(ids)

    pipeline_module._write_predictions_and_summary(
        tmp_path / "predictions.tsv",
        dataset=dataset,
        model=model,
        is_training_dataset=False,
    )
    assert model.decision_components_calls == 1


def test_cache_fallthrough_when_lengths_differ(tmp_path: Path):
    """Different sample counts → cache is structurally inapplicable."""
    training_ids = [f"s_{i}" for i in range(10)]
    holdout_ids = [f"s_{i}" for i in range(7)]
    cached = (
        np.zeros(10, dtype=np.float32),
        np.zeros(10, dtype=np.float32),
    )
    model = _FakeModel(
        config=ModelConfig(trait_type=TraitType.QUANTITATIVE),
        training_sample_ids=training_ids,
        cached_components=cached,
    )
    dataset = _make_dataset(holdout_ids)

    pipeline_module._write_predictions_and_summary(
        tmp_path / "predictions.tsv",
        dataset=dataset,
        model=model,
        is_training_dataset=True,
    )
    assert model.decision_components_calls == 1

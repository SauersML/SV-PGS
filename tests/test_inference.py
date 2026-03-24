"""Tests for variational EM inference."""

from __future__ import annotations

import numpy as np

from sv_pgs.config import ModelConfig, TraitType, VariantClass
from sv_pgs.data import GraphEdges, TieMap, TieGroup, VariantRecord
from sv_pgs.inference import fit_variational_em


def _identity_tie_map(p: int) -> TieMap:
    """No ties: every variant maps to itself."""
    groups = [
        TieGroup(
            representative_index=i,
            member_indices=np.array([i], dtype=np.int32),
            signs=np.array([1.0], dtype=np.float32),
        )
        for i in range(p)
    ]
    return TieMap(
        kept_indices=np.arange(p, dtype=np.int32),
        original_to_reduced=np.arange(p, dtype=np.int32),
        reduced_to_group=groups,
    )


def _empty_graph(p: int) -> GraphEdges:
    return GraphEdges(
        src=np.array([], dtype=np.int32),
        dst=np.array([], dtype=np.int32),
        sign=np.array([], dtype=np.float32),
        weight=np.array([], dtype=np.float32),
        block_ids=np.arange(p, dtype=np.int32),
    )


def _make_records(p: int) -> list[VariantRecord]:
    return [
        VariantRecord(f"v{i}", VariantClass.SNV, "short", "chr1", i * 1000)
        for i in range(p)
    ]


class TestFitVariationalEM:
    def test_quantitative_runs(self):
        rng = np.random.default_rng(200)
        n, p, d = 80, 10, 2
        X = rng.standard_normal((n, p)).astype(np.float32)
        C = np.column_stack([np.ones(n), rng.standard_normal((n, d))]).astype(np.float32)
        beta_true = np.zeros(p, dtype=np.float32)
        beta_true[0] = 1.0
        y = X @ beta_true + rng.standard_normal(n).astype(np.float32) * 0.3

        config = ModelConfig(trait_type=TraitType.QUANTITATIVE, max_outer_iters=5, tile_size=16)
        records = _make_records(p)
        tie_map = _identity_tie_map(p)
        graph = _empty_graph(p)

        result = fit_variational_em(
            genotypes=X, covariates=C, targets=y,
            records=records, tie_map=tie_map, graph=graph, config=config,
        )

        assert result.beta_reduced.shape == (p,)
        assert result.alpha.shape == (C.shape[1],)
        assert len(result.objective_history) > 0

    def test_binary_runs(self):
        rng = np.random.default_rng(201)
        n, p = 100, 8
        X = rng.standard_normal((n, p)).astype(np.float32)
        C = np.ones((n, 1), dtype=np.float32)
        beta_true = np.zeros(p, dtype=np.float32)
        beta_true[0] = 1.5
        eta = X @ beta_true
        y = (rng.random(n) < 1.0 / (1.0 + np.exp(-eta))).astype(np.float32)

        config = ModelConfig(trait_type=TraitType.BINARY, max_outer_iters=5, tile_size=16)
        records = _make_records(p)
        tie_map = _identity_tie_map(p)
        graph = _empty_graph(p)

        result = fit_variational_em(
            genotypes=X, covariates=C, targets=y,
            records=records, tie_map=tie_map, graph=graph, config=config,
        )

        assert result.beta_reduced.shape == (p,)
        assert result.sigma_e2 == 1.0  # Binary trait always 1.0

    def test_signal_recovery(self):
        """The largest coefficient should correspond to the true signal variant."""
        rng = np.random.default_rng(202)
        n, p = 200, 10
        X = rng.standard_normal((n, p)).astype(np.float32)
        # Standardize.
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        C = np.ones((n, 1), dtype=np.float32)
        beta_true = np.zeros(p, dtype=np.float32)
        beta_true[3] = 2.0
        y = X @ beta_true + rng.standard_normal(n).astype(np.float32) * 0.5

        config = ModelConfig(trait_type=TraitType.QUANTITATIVE, max_outer_iters=15, tile_size=16)
        records = _make_records(p)
        tie_map = _identity_tie_map(p)
        graph = _empty_graph(p)

        result = fit_variational_em(
            genotypes=X, covariates=C, targets=y,
            records=records, tie_map=tie_map, graph=graph, config=config,
        )

        # Variant 3 should have the largest absolute coefficient.
        assert np.argmax(np.abs(result.beta_reduced)) == 3

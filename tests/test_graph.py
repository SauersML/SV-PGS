"""Tests for correlation graph construction."""

from __future__ import annotations

import numpy as np

from sv_pgs.config import ModelConfig, VariantClass
from sv_pgs.data import VariantRecord
from sv_pgs.graph import build_correlation_graph, correlation_blocks
from sv_pgs.preprocessing import build_tie_map


def _make_records(p: int, cls: VariantClass = VariantClass.SNV) -> list[VariantRecord]:
    return [
        VariantRecord(f"v{i}", cls, "short", "chr1", i * 100, quality=1.0)
        for i in range(p)
    ]


class TestBuildCorrelationGraph:
    def test_uncorrelated_no_edges(self):
        rng = np.random.default_rng(10)
        X = rng.standard_normal((200, 5)).astype(np.float32)
        records = _make_records(5)
        tie_map = build_tie_map(X, records)
        config = ModelConfig(correlation_threshold=0.98)
        graph = build_correlation_graph(X, records, tie_map, config)
        assert graph.src.shape[0] == 0

    def test_correlated_pair_creates_edge(self):
        rng = np.random.default_rng(11)
        X = rng.standard_normal((200, 4)).astype(np.float32)
        # Make v2 nearly identical to v0.
        X[:, 2] = X[:, 0] + rng.standard_normal(200).astype(np.float32) * 0.01
        records = [
            VariantRecord("v0", VariantClass.SNV, "short", "chr1", 0),
            VariantRecord("v1", VariantClass.SNV, "short", "chr1", 100),
            VariantRecord("v2", VariantClass.DELETION_SHORT, "short", "chr1", 200),
            VariantRecord("v3", VariantClass.SNV, "short", "chr1", 300),
        ]
        tie_map = build_tie_map(X, records)
        config = ModelConfig(correlation_threshold=0.98)
        graph = build_correlation_graph(X, records, tie_map, config)
        assert graph.src.shape[0] >= 1

    def test_different_chromosomes_no_edge(self):
        rng = np.random.default_rng(12)
        X = rng.standard_normal((200, 3)).astype(np.float32)
        X[:, 1] = X[:, 0] + rng.standard_normal(200).astype(np.float32) * 0.01
        records = [
            VariantRecord("v0", VariantClass.SNV, "short", "chr1", 0),
            VariantRecord("v1", VariantClass.SNV, "short", "chr2", 0),  # Different chrom
            VariantRecord("v2", VariantClass.SNV, "short", "chr1", 100),
        ]
        tie_map = build_tie_map(X, records)
        config = ModelConfig(correlation_threshold=0.98)
        graph = build_correlation_graph(X, records, tie_map, config)
        # v0 and v1 on different chromosomes, should not be paired.
        for s, d in zip(graph.src, graph.dst):
            edge_set = {int(tie_map.kept_indices[s]), int(tie_map.kept_indices[d])}
            assert edge_set != {0, 1}

    def test_cluster_based_edges(self):
        rng = np.random.default_rng(13)
        X = rng.standard_normal((200, 4)).astype(np.float32)
        X[:, 3] = X[:, 0] + rng.standard_normal(200).astype(np.float32) * 0.005
        records = [
            VariantRecord("v0", VariantClass.SNV, "short", "chr1", 0, cluster_id="c1"),
            VariantRecord("v1", VariantClass.SNV, "short", "chr1", 100),
            VariantRecord("v2", VariantClass.SNV, "short", "chr1", 200),
            # v3 on different chrom but same cluster as v0.
            VariantRecord("v3", VariantClass.SNV, "short", "chr2", 5_000_000, cluster_id="c1"),
        ]
        tie_map = build_tie_map(X, records)
        config = ModelConfig(correlation_threshold=0.98)
        graph = build_correlation_graph(X, records, tie_map, config)
        assert graph.src.shape[0] >= 1


class TestCorrelationBlocks:
    def test_singleton_blocks(self):
        rng = np.random.default_rng(20)
        X = rng.standard_normal((100, 5)).astype(np.float32)
        records = _make_records(5)
        tie_map = build_tie_map(X, records)
        config = ModelConfig()
        graph = build_correlation_graph(X, records, tie_map, config)
        blocks = correlation_blocks(graph)
        # Uncorrelated: every variant in its own block.
        assert len(blocks) == tie_map.kept_indices.shape[0]

from __future__ import annotations

import numpy as np

from sv_pgs.config import ModelConfig, VariantClass
from sv_pgs.data import VariantRecord
from sv_pgs.graph import build_correlation_graph, correlation_blocks
from sv_pgs.preprocessing import build_tie_map

from tests.conftest import make_variant_records


def test_uncorrelated_variants_create_no_edges(random_generator):
    genotype_matrix = random_generator.standard_normal((200, 5)).astype(np.float32)
    variant_records = make_variant_records(5)
    tie_map = build_tie_map(genotype_matrix, variant_records)
    correlation_graph = build_correlation_graph(genotype_matrix, variant_records, tie_map, ModelConfig())
    assert correlation_graph.src.shape[0] == 0


def test_nearly_identical_cross_class_variants_create_edge(random_generator):
    genotype_matrix = random_generator.standard_normal((200, 4)).astype(np.float32)
    genotype_matrix[:, 2] = genotype_matrix[:, 0] + random_generator.standard_normal(200).astype(np.float32) * 0.01
    variant_records = [
        VariantRecord("variant_0", VariantClass.SNV, "short", "chr1", 0),
        VariantRecord("variant_1", VariantClass.SNV, "short", "chr1", 100),
        VariantRecord("variant_2", VariantClass.DELETION_SHORT, "short", "chr1", 200),
        VariantRecord("variant_3", VariantClass.SNV, "short", "chr1", 300),
    ]
    tie_map = build_tie_map(genotype_matrix, variant_records)
    correlation_graph = build_correlation_graph(
        genotype_matrix,
        variant_records,
        tie_map,
        ModelConfig(correlation_threshold=0.98),
    )
    assert correlation_graph.src.shape[0] >= 1


def test_cluster_match_allows_cross_chromosome_edges(random_generator):
    genotype_matrix = random_generator.standard_normal((200, 4)).astype(np.float32)
    genotype_matrix[:, 3] = genotype_matrix[:, 0] + random_generator.standard_normal(200).astype(np.float32) * 0.005
    variant_records = [
        VariantRecord("variant_0", VariantClass.SNV, "short", "chr1", 0, cluster_id="cluster_a"),
        VariantRecord("variant_1", VariantClass.SNV, "short", "chr1", 100),
        VariantRecord("variant_2", VariantClass.SNV, "short", "chr1", 200),
        VariantRecord("variant_3", VariantClass.SNV, "short", "chr2", 5_000_000, cluster_id="cluster_a"),
    ]
    tie_map = build_tie_map(genotype_matrix, variant_records)
    correlation_graph = build_correlation_graph(
        genotype_matrix,
        variant_records,
        tie_map,
        ModelConfig(correlation_threshold=0.98),
    )
    assert correlation_graph.src.shape[0] >= 1


def test_uncorrelated_variants_form_singleton_blocks(random_generator):
    genotype_matrix = random_generator.standard_normal((100, 5)).astype(np.float32)
    variant_records = make_variant_records(5)
    tie_map = build_tie_map(genotype_matrix, variant_records)
    correlation_graph = build_correlation_graph(genotype_matrix, variant_records, tie_map, ModelConfig())
    blocks = correlation_blocks(correlation_graph)
    assert len(blocks) == tie_map.kept_indices.shape[0]

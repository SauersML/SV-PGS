from __future__ import annotations

import numpy as np

from sv_pgs.blocks import build_block_decomposition, compute_block_posterior
from sv_pgs.config import ModelConfig, VariantClass
from sv_pgs.data import VariantRecord


def _make_records_two_chromosomes() -> list[VariantRecord]:
    return [
        VariantRecord("variant_0", VariantClass.SNV, "chr1", 100),
        VariantRecord("variant_1", VariantClass.SNV, "chr1", 200),
        VariantRecord("variant_2", VariantClass.SNV, "chr2", 100),
        VariantRecord("variant_3", VariantClass.SNV, "chr2", 200),
    ]


def test_blocks_separate_by_chromosome(random_generator):
    genotypes = random_generator.standard_normal((50, 4)).astype(np.float64)
    records = _make_records_two_chromosomes()
    decomposition = build_block_decomposition(genotypes, records, ModelConfig())
    chr1_block_ids = set(decomposition.variant_to_block[:2].tolist())
    chr2_block_ids = set(decomposition.variant_to_block[2:].tolist())
    assert chr1_block_ids.isdisjoint(chr2_block_ids)


def test_eigendecomposition_captures_target_variance(random_generator):
    variant_count = 20
    sample_count = 100
    genotypes = random_generator.standard_normal((sample_count, variant_count)).astype(np.float64)
    records = [
        VariantRecord("variant_" + str(variant_index), VariantClass.SNV, "chr1", variant_index * 100)
        for variant_index in range(variant_count)
    ]
    config = ModelConfig(discarded_spectrum_tolerance=0.01)
    decomposition = build_block_decomposition(genotypes, records, config)
    for ld_block in decomposition.blocks:
        total_retained = float(np.sum(ld_block.eigenvalues))
        block_geno = genotypes[:, ld_block.variant_indices]
        full_eigenvalues = np.linalg.eigvalsh((block_geno.T @ block_geno) / sample_count)
        full_total = float(np.sum(np.maximum(full_eigenvalues, 0.0)))
        omitted_total = full_total - total_retained
        assert omitted_total <= config.discarded_spectrum_tolerance * full_total * 1.05


def test_block_posterior_matches_exact_for_small_block(random_generator):
    variant_count = 6
    sample_count = 40
    genotypes = random_generator.standard_normal((sample_count, variant_count)).astype(np.float64)
    records = [
        VariantRecord("variant_" + str(variant_index), VariantClass.SNV, "chr1", variant_index * 100)
        for variant_index in range(variant_count)
    ]
    config = ModelConfig(discarded_spectrum_tolerance=0.001)
    decomposition = build_block_decomposition(genotypes, records, config)
    ld_block = decomposition.blocks[0]
    sample_weights = np.ones(sample_count, dtype=np.float64)
    prior_precision = np.full(variant_count, 2.0, dtype=np.float64)
    true_beta = random_generator.standard_normal(variant_count).astype(np.float64)
    residual = genotypes @ true_beta + random_generator.standard_normal(sample_count) * 0.1
    weighted_residual = sample_weights * residual
    block_mean, block_var = compute_block_posterior(ld_block, genotypes, sample_weights, weighted_residual, prior_precision)
    exact_precision = genotypes.T @ np.diag(sample_weights) @ genotypes + np.diag(prior_precision)
    exact_covariance = np.linalg.inv(exact_precision)
    exact_mean = exact_covariance @ (genotypes.T @ weighted_residual)
    np.testing.assert_allclose(block_mean, exact_mean.astype(np.float32), atol=0.15)
    np.testing.assert_allclose(block_var, np.diag(exact_covariance).astype(np.float32), rtol=0.4)


def test_block_jitter_increases_with_condition_number(random_generator):
    sample_count = 80
    easy_genotypes = random_generator.standard_normal((sample_count, 5)).astype(np.float64)
    hard_genotypes = random_generator.standard_normal((sample_count, 5)).astype(np.float64)
    hard_genotypes[:, 1] = hard_genotypes[:, 0] + random_generator.standard_normal(sample_count) * 0.001
    easy_records = [
        VariantRecord("easy_variant_" + str(variant_index), VariantClass.SNV, "chr1", variant_index * 100)
        for variant_index in range(5)
    ]
    hard_records = [
        VariantRecord("hard_variant_" + str(variant_index), VariantClass.SNV, "chr2", variant_index * 100)
        for variant_index in range(5)
    ]
    combined = np.column_stack([easy_genotypes, hard_genotypes])
    records = easy_records + hard_records
    decomposition = build_block_decomposition(combined, records, ModelConfig())
    easy_block = decomposition.blocks[0]
    hard_block = decomposition.blocks[1]
    assert hard_block.condition_number > easy_block.condition_number

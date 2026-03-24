from __future__ import annotations

import numpy as np

from sv_pgs.config import ModelConfig, TraitType
from sv_pgs.data import TieGroup, TieMap
from sv_pgs.inference import fit_variational_em

from tests.conftest import empty_graph, make_variant_records


def _identity_tie_map(variant_count: int) -> TieMap:
    tie_groups = [
        TieGroup(
            representative_index=variant_index,
            member_indices=np.array([variant_index], dtype=np.int32),
            signs=np.array([1.0], dtype=np.float32),
        )
        for variant_index in range(variant_count)
    ]
    return TieMap(
        kept_indices=np.arange(variant_count, dtype=np.int32),
        original_to_reduced=np.arange(variant_count, dtype=np.int32),
        reduced_to_group=tie_groups,
    )


def test_quantitative_inference_runs(random_generator):
    sample_count, variant_count, covariate_count = 80, 10, 2
    genotype_matrix = random_generator.standard_normal((sample_count, variant_count)).astype(np.float32)
    covariate_matrix = np.column_stack(
        [np.ones(sample_count), random_generator.standard_normal((sample_count, covariate_count))]
    ).astype(np.float32)
    true_coefficients = np.zeros(variant_count, dtype=np.float32)
    true_coefficients[0] = 1.0
    target_vector = genotype_matrix @ true_coefficients + random_generator.standard_normal(sample_count).astype(np.float32) * 0.3

    fit_result = fit_variational_em(
        genotypes=genotype_matrix,
        covariates=covariate_matrix,
        targets=target_vector,
        records=make_variant_records(variant_count),
        tie_map=_identity_tie_map(variant_count),
        graph=empty_graph(variant_count),
        config=ModelConfig(trait_type=TraitType.QUANTITATIVE, max_outer_iters=5, tile_size=16),
    )

    assert fit_result.beta_reduced.shape == (variant_count,)
    assert fit_result.alpha.shape == (covariate_matrix.shape[1],)
    assert fit_result.objective_history


def test_binary_inference_runs(random_generator):
    sample_count, variant_count = 100, 8
    genotype_matrix = random_generator.standard_normal((sample_count, variant_count)).astype(np.float32)
    covariate_matrix = np.ones((sample_count, 1), dtype=np.float32)
    true_coefficients = np.zeros(variant_count, dtype=np.float32)
    true_coefficients[0] = 1.5
    linear_predictor = genotype_matrix @ true_coefficients
    target_vector = (
        random_generator.random(sample_count) < 1.0 / (1.0 + np.exp(-linear_predictor))
    ).astype(np.float32)

    fit_result = fit_variational_em(
        genotypes=genotype_matrix,
        covariates=covariate_matrix,
        targets=target_vector,
        records=make_variant_records(variant_count),
        tie_map=_identity_tie_map(variant_count),
        graph=empty_graph(variant_count),
        config=ModelConfig(trait_type=TraitType.BINARY, max_outer_iters=5, tile_size=16),
    )

    assert fit_result.beta_reduced.shape == (variant_count,)
    assert fit_result.sigma_e2 == 1.0


def test_signal_variant_receives_largest_effect(random_generator):
    sample_count, variant_count = 200, 10
    genotype_matrix = random_generator.standard_normal((sample_count, variant_count)).astype(np.float32)
    genotype_matrix = (genotype_matrix - genotype_matrix.mean(axis=0)) / (genotype_matrix.std(axis=0) + 1e-6)
    covariate_matrix = np.ones((sample_count, 1), dtype=np.float32)
    true_coefficients = np.zeros(variant_count, dtype=np.float32)
    true_coefficients[3] = 2.0
    target_vector = genotype_matrix @ true_coefficients + random_generator.standard_normal(sample_count).astype(np.float32) * 0.5

    fit_result = fit_variational_em(
        genotypes=genotype_matrix,
        covariates=covariate_matrix,
        targets=target_vector,
        records=make_variant_records(variant_count),
        tie_map=_identity_tie_map(variant_count),
        graph=empty_graph(variant_count),
        config=ModelConfig(trait_type=TraitType.QUANTITATIVE, max_outer_iters=15, tile_size=16),
    )

    assert np.argmax(np.abs(fit_result.beta_reduced)) == 3

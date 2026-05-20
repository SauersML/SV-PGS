"""Tests for randomized cyclic block-coordinate-descent block shuffling in SVI.

The SVI path in ``fit_variational_em`` permutes the within-epoch variant block
order using a deterministic per-epoch RNG seeded by
``random_seed + outer_iteration``. These tests verify:

  1. With shuffle enabled and a fixed ``random_seed``, repeated fits are
     bit-identical (deterministic given the seed).
  2. Different ``random_seed`` values produce different fit results (the seed
     actually changes the within-epoch order, which changes intermediate
     iterates even though the fixed point is the same in expectation).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sv_pgs.config import ModelConfig, TraitType
from sv_pgs.inference import fit_variational_em
from sv_pgs.preprocessing import build_tie_map

from tests.conftest import make_variant_records


def _make_data(seed: int = 0):
    sample_count, variant_count = 64, 12
    rng = np.random.default_rng(seed)
    genotype_matrix = rng.standard_normal((sample_count, variant_count)).astype(np.float32)
    covariate_matrix = np.column_stack(
        [
            np.ones(sample_count, dtype=np.float32),
            rng.standard_normal(sample_count).astype(np.float32),
        ]
    )
    true_coefficients = np.zeros(variant_count, dtype=np.float32)
    true_coefficients[:3] = np.array([1.1, -0.8, 0.5], dtype=np.float32)
    target_vector = (
        genotype_matrix @ true_coefficients
        + 0.2 * rng.standard_normal(sample_count).astype(np.float32)
    )
    records = make_variant_records(variant_count)
    return genotype_matrix, covariate_matrix, target_vector, records


def _base_config(**overrides: Any) -> ModelConfig:
    kwargs: dict[str, Any] = dict(
        trait_type=TraitType.QUANTITATIVE,
        max_outer_iterations=3,
        stochastic_variational_updates=True,
        stochastic_min_variant_count=1,
        stochastic_variant_batch_size=4,
        random_seed=0,
    )
    kwargs.update(overrides)
    return ModelConfig(**kwargs)


def _fit(config, data):
    genotype_matrix, covariate_matrix, target_vector, records = data
    tie_map = build_tie_map(genotype_matrix, records, config)
    return fit_variational_em(
        genotypes=genotype_matrix,
        covariates=covariate_matrix,
        targets=target_vector,
        records=records,
        config=config,
        tie_map=tie_map,
    )


def test_same_seed_same_shuffle_is_bit_identical():
    data = _make_data()
    config_a = _base_config(random_seed=7)
    config_b = _base_config(random_seed=7)
    result_a = _fit(config_a, data)
    result_b = _fit(config_b, data)
    np.testing.assert_array_equal(result_a.beta_reduced, result_b.beta_reduced)
    np.testing.assert_array_equal(result_a.alpha, result_b.alpha)


def test_different_seeds_produce_different_results():
    data = _make_data()
    config_a = _base_config(random_seed=1)
    config_b = _base_config(random_seed=2)
    result_a = _fit(config_a, data)
    result_b = _fit(config_b, data)
    # Different shuffle permutations make the intermediate iterates differ;
    # after only ``max_outer_iterations=3`` epochs we have not converged and
    # the iterates should not be bit-identical.
    assert not np.array_equal(result_a.beta_reduced, result_b.beta_reduced)

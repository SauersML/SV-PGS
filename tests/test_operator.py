from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from sv_pgs.config import ModelConfig
from sv_pgs.operator import GenotypeOperator, apply_hessian, matvec, pcg_solve, rmatvec, weighted_column_norms

from tests.conftest import empty_graph


def test_matvec_matches_numpy(random_generator):
    sample_count, variant_count = 50, 30
    genotype_matrix = random_generator.standard_normal((sample_count, variant_count)).astype(np.float32)
    coefficient_vector = random_generator.standard_normal(variant_count).astype(np.float32)
    operator = GenotypeOperator.from_numpy(genotype_matrix, empty_graph(variant_count), ModelConfig(tile_size=8))
    np.testing.assert_allclose(
        np.asarray(matvec(operator, jnp.asarray(coefficient_vector))),
        genotype_matrix @ coefficient_vector,
        atol=1e-4,
    )


def test_rmatvec_matches_numpy(random_generator):
    sample_count, variant_count = 50, 30
    genotype_matrix = random_generator.standard_normal((sample_count, variant_count)).astype(np.float32)
    residual_vector = random_generator.standard_normal(sample_count).astype(np.float32)
    operator = GenotypeOperator.from_numpy(genotype_matrix, empty_graph(variant_count), ModelConfig(tile_size=8))
    np.testing.assert_allclose(
        np.asarray(rmatvec(operator, jnp.asarray(residual_vector))),
        np.transpose(genotype_matrix) @ residual_vector,
        atol=1e-4,
    )


def test_weighted_column_norms_match_numpy(random_generator):
    sample_count, variant_count = 50, 20
    genotype_matrix = random_generator.standard_normal((sample_count, variant_count)).astype(np.float32)
    sample_weights = np.abs(random_generator.standard_normal(sample_count)).astype(np.float32)
    operator = GenotypeOperator.from_numpy(genotype_matrix, empty_graph(variant_count), ModelConfig(tile_size=8))
    np.testing.assert_allclose(
        np.asarray(weighted_column_norms(operator, jnp.asarray(sample_weights))),
        np.sum(genotype_matrix ** 2 * sample_weights[:, None], axis=0),
        atol=1e-3,
    )


def test_pcg_recovers_known_solution(random_generator):
    sample_count, variant_count = 80, 15
    genotype_matrix = random_generator.standard_normal((sample_count, variant_count)).astype(np.float32)
    sample_weights = np.ones(sample_count, dtype=np.float32)
    prior_precision = np.full(variant_count, 1.0, dtype=np.float32)
    operator = GenotypeOperator.from_numpy(genotype_matrix, empty_graph(variant_count), ModelConfig(tile_size=8))
    hessian = np.transpose(genotype_matrix) @ genotype_matrix + np.diag(prior_precision)
    true_coefficients = random_generator.standard_normal(variant_count).astype(np.float32)
    right_hand_side = hessian @ true_coefficients
    np.testing.assert_allclose(
        np.asarray(
            pcg_solve(
                operator=operator,
                right_hand_side=jnp.asarray(right_hand_side),
                sample_weights=jnp.asarray(sample_weights),
                prior_precision=jnp.asarray(prior_precision),
                preconditioner_diagonal=jnp.asarray(np.diag(hessian).astype(np.float32)),
                initial_coefficients=jnp.zeros(variant_count, dtype=jnp.float32),
                tolerance=1e-6,
                maximum_iterations=200,
            )
        ),
        true_coefficients,
        atol=1e-3,
    )


def test_apply_hessian_matches_closed_form(random_generator):
    sample_count, variant_count = 40, 10
    genotype_matrix = random_generator.standard_normal((sample_count, variant_count)).astype(np.float32)
    sample_weights = np.ones(sample_count, dtype=np.float32)
    prior_precision = np.full(variant_count, 0.5, dtype=np.float32)
    coefficient_vector = random_generator.standard_normal(variant_count).astype(np.float32)
    operator = GenotypeOperator.from_numpy(genotype_matrix, empty_graph(variant_count), ModelConfig(tile_size=8))
    expected_hessian_product = (
        np.transpose(genotype_matrix) @ genotype_matrix + np.diag(prior_precision)
    ) @ coefficient_vector
    np.testing.assert_allclose(
        np.asarray(
            apply_hessian(
                operator,
                jnp.asarray(coefficient_vector),
                jnp.asarray(sample_weights),
                jnp.asarray(prior_precision),
            )
        ),
        expected_hessian_product,
        atol=1e-4,
    )

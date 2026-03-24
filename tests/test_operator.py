from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from sv_pgs.operator import GenotypeOperator, apply_sample_system, matvec, pcg_solve_sample_space, rmatvec, sample_space_diagonal


def test_matvec_matches_numpy(random_generator):
    sample_count, variant_count = 50, 30
    genotype_matrix = random_generator.standard_normal((sample_count, variant_count)).astype(np.float32)
    coefficient_vector = random_generator.standard_normal(variant_count).astype(np.float32)
    operator = GenotypeOperator.from_numpy(genotype_matrix, tile_size=8)
    np.testing.assert_allclose(
        np.asarray(matvec(operator, jnp.asarray(coefficient_vector))),
        genotype_matrix @ coefficient_vector,
        atol=1e-4,
    )


def test_rmatvec_matches_numpy(random_generator):
    sample_count, variant_count = 50, 30
    genotype_matrix = random_generator.standard_normal((sample_count, variant_count)).astype(np.float32)
    residual_vector = random_generator.standard_normal(sample_count).astype(np.float32)
    operator = GenotypeOperator.from_numpy(genotype_matrix, tile_size=8)
    np.testing.assert_allclose(
        np.asarray(rmatvec(operator, jnp.asarray(residual_vector))),
        np.transpose(genotype_matrix) @ residual_vector,
        atol=1e-4,
    )


def test_sample_space_operator_matches_closed_form(random_generator):
    sample_count, variant_count = 40, 12
    genotype_matrix = random_generator.standard_normal((sample_count, variant_count)).astype(np.float32)
    sample_inverse_weights = np.abs(random_generator.standard_normal(sample_count)).astype(np.float32) + 0.5
    prior_covariance = np.abs(random_generator.standard_normal(variant_count)).astype(np.float32) + 0.25
    sample_vector = random_generator.standard_normal(sample_count).astype(np.float32)
    operator = GenotypeOperator.from_numpy(genotype_matrix, tile_size=8)

    expected_system_product = (
        np.diag(sample_inverse_weights)
        + genotype_matrix @ np.diag(prior_covariance) @ np.transpose(genotype_matrix)
    ) @ sample_vector
    np.testing.assert_allclose(
        np.asarray(
            apply_sample_system(
                operator,
                jnp.asarray(sample_vector),
                jnp.asarray(sample_inverse_weights),
                jnp.asarray(prior_covariance),
            )
        ),
        expected_system_product,
        atol=1e-4,
    )


def test_sample_space_diagonal_matches_closed_form(random_generator):
    sample_count, variant_count = 40, 10
    genotype_matrix = random_generator.standard_normal((sample_count, variant_count)).astype(np.float32)
    sample_inverse_weights = np.abs(random_generator.standard_normal(sample_count)).astype(np.float32) + 0.5
    prior_covariance = np.abs(random_generator.standard_normal(variant_count)).astype(np.float32) + 0.25
    operator = GenotypeOperator.from_numpy(genotype_matrix, tile_size=8)
    expected_diagonal = sample_inverse_weights + np.sum(genotype_matrix * genotype_matrix * prior_covariance[None, :], axis=1)
    np.testing.assert_allclose(
        np.asarray(
            sample_space_diagonal(
                operator,
                jnp.asarray(sample_inverse_weights),
                jnp.asarray(prior_covariance),
            )
        ),
        expected_diagonal,
        atol=1e-4,
    )


def test_sample_space_pcg_recovers_known_solution(random_generator):
    sample_count, variant_count = 60, 15
    genotype_matrix = random_generator.standard_normal((sample_count, variant_count)).astype(np.float32)
    sample_inverse_weights = np.abs(random_generator.standard_normal(sample_count)).astype(np.float32) + 1.0
    prior_covariance = np.abs(random_generator.standard_normal(variant_count)).astype(np.float32) + 0.25
    operator = GenotypeOperator.from_numpy(genotype_matrix, tile_size=8)
    sample_system = np.diag(sample_inverse_weights) + genotype_matrix @ np.diag(prior_covariance) @ np.transpose(genotype_matrix)
    true_solution = random_generator.standard_normal(sample_count).astype(np.float32)
    right_hand_side = sample_system @ true_solution
    preconditioner_diagonal = np.diag(sample_system).astype(np.float32)
    recovered_solution = pcg_solve_sample_space(
        operator=operator,
        right_hand_side=jnp.asarray(right_hand_side),
        sample_inverse_weights=jnp.asarray(sample_inverse_weights),
        prior_covariance=jnp.asarray(prior_covariance),
        preconditioner_diagonal=jnp.asarray(preconditioner_diagonal),
        initial_solution=jnp.zeros(sample_count, dtype=jnp.float32),
        tolerance=1e-6,
        maximum_iterations=300,
    )
    np.testing.assert_allclose(np.asarray(recovered_solution), true_solution, atol=1e-3)

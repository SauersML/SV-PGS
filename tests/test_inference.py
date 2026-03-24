from __future__ import annotations

import numpy as np
from jax.scipy.special import polygamma

from sv_pgs.config import ModelConfig, TraitType
from sv_pgs.data import VariantRecord
from sv_pgs.inference import fit_variational_em
from sv_pgs.mixture_inference import _trigamma, _update_local_scales

from tests.conftest import make_variant_records


def test_quantitative_inference_runs(random_generator):
    sample_count, variant_count = 80, 10
    genotype_matrix = random_generator.standard_normal((sample_count, variant_count)).astype(np.float32)
    covariate_matrix = np.column_stack(
        [np.ones(sample_count), random_generator.standard_normal((sample_count, 2))]
    ).astype(np.float32)
    true_coefficients = np.zeros(variant_count, dtype=np.float32)
    true_coefficients[0] = 1.0
    target_vector = genotype_matrix @ true_coefficients + random_generator.standard_normal(sample_count).astype(np.float32) * 0.3

    result = fit_variational_em(
        genotypes=genotype_matrix,
        covariates=covariate_matrix,
        targets=target_vector,
        records=make_variant_records(variant_count),
        config=ModelConfig(trait_type=TraitType.QUANTITATIVE, max_outer_iterations=5),
    )
    assert result.beta_reduced.shape == (variant_count,)
    assert result.alpha.shape == (covariate_matrix.shape[1],)
    assert result.objective_history
    assert result.prior_scales.shape[0] == variant_count


def test_binary_inference_runs(random_generator):
    sample_count, variant_count = 100, 8
    genotype_matrix = random_generator.standard_normal((sample_count, variant_count)).astype(np.float32)
    covariate_matrix = np.ones((sample_count, 1), dtype=np.float32)
    true_coefficients = np.zeros(variant_count, dtype=np.float32)
    true_coefficients[0] = 1.5
    linear_predictor = genotype_matrix @ true_coefficients
    target_vector = (random_generator.random(sample_count) < 1.0 / (1.0 + np.exp(-linear_predictor))).astype(np.float32)

    result = fit_variational_em(
        genotypes=genotype_matrix,
        covariates=covariate_matrix,
        targets=target_vector,
        records=make_variant_records(variant_count),
        config=ModelConfig(trait_type=TraitType.BINARY, max_outer_iterations=5),
    )
    assert result.beta_reduced.shape == (variant_count,)
    assert result.sigma_error2 == 1.0
    assert len(result.class_tpb_shape_a) == 1
    assert len(result.class_tpb_shape_b) == 1


def test_signal_variant_receives_largest_effect(random_generator):
    sample_count, variant_count = 200, 10
    genotype_matrix = random_generator.standard_normal((sample_count, variant_count)).astype(np.float32)
    genotype_matrix = (genotype_matrix - genotype_matrix.mean(axis=0)) / (genotype_matrix.std(axis=0) + 1e-6)
    covariate_matrix = np.ones((sample_count, 1), dtype=np.float32)
    true_coefficients = np.zeros(variant_count, dtype=np.float32)
    true_coefficients[3] = 2.0
    target_vector = genotype_matrix @ true_coefficients + random_generator.standard_normal(sample_count).astype(np.float32) * 0.5
    records = make_variant_records(variant_count)
    records[3] = VariantRecord(
        variant_id=records[3].variant_id,
        variant_class=records[3].variant_class,
        chromosome=records[3].chromosome,
        position=records[3].position,
        length=3_000.0,
        allele_frequency=0.02,
        quality=1.0,
        is_repeat=False,
        is_copy_number=False,
    )

    result = fit_variational_em(
        genotypes=genotype_matrix,
        covariates=covariate_matrix,
        targets=target_vector,
        records=records,
        config=ModelConfig(trait_type=TraitType.QUANTITATIVE, max_outer_iterations=12),
    )
    assert np.argmax(np.abs(result.beta_reduced)) == 3


def test_trigamma_matches_jax_polygamma_for_small_shapes():
    shape_values = np.array([0.1, 0.2, 0.5, 1.0, 2.0], dtype=np.float32)
    for shape_value in shape_values:
        expected_value = float(polygamma(1, shape_value))
        actual_value = float(_trigamma(float(shape_value)))
        assert np.isclose(actual_value, expected_value, rtol=1e-6, atol=1e-6)


def test_local_scale_update_uses_unslabbed_baseline_variance():
    config = ModelConfig()
    coefficient_second_moment = np.array([9.0], dtype=np.float64)
    baseline_prior_variances = np.array([4.0], dtype=np.float64)
    local_shape_a = np.array([2.0], dtype=np.float64)
    local_shape_b = np.array([0.5], dtype=np.float64)
    auxiliary_delta = np.array([0.75], dtype=np.float64)

    updated_local_scale, updated_auxiliary_delta = _update_local_scales(
        coefficient_second_moment=coefficient_second_moment,
        baseline_prior_variances=baseline_prior_variances,
        local_shape_a=local_shape_a,
        local_shape_b=local_shape_b,
        auxiliary_delta=auxiliary_delta,
        config=config,
    )

    normalized_second_moment = coefficient_second_moment / baseline_prior_variances
    shape_offset = local_shape_a - 1.5
    expected_local_scale = (
        shape_offset
        + np.sqrt(shape_offset * shape_offset + 2.0 * auxiliary_delta * normalized_second_moment)
    ) / (2.0 * auxiliary_delta)
    expected_auxiliary_delta = (local_shape_a + local_shape_b) / (1.0 + expected_local_scale)

    np.testing.assert_allclose(updated_local_scale, expected_local_scale)
    np.testing.assert_allclose(updated_auxiliary_delta, expected_auxiliary_delta)

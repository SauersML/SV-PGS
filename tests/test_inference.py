from __future__ import annotations

import numpy as np
import pytest
from scipy.special import kve
from scipy.special import polygamma

from sv_pgs.config import ModelConfig, TraitType, VariantClass
from sv_pgs.data import VariantRecord
from sv_pgs.inference import fit_variational_em
from sv_pgs.mixture_inference import (
    PosteriorState,
    _quantitative_posterior_state,
    _trigamma,
    _update_local_scales,
    _update_tpb_shape_vectors,
)
import sv_pgs.mixture_inference as mixture_inference
from sv_pgs.preprocessing import build_tie_map

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
    records = make_variant_records(variant_count)
    config = ModelConfig(trait_type=TraitType.QUANTITATIVE, max_outer_iterations=5)

    result = fit_variational_em(
        genotypes=genotype_matrix,
        covariates=covariate_matrix,
        targets=target_vector,
        records=records,
        config=config,
        tie_map=build_tie_map(genotype_matrix, records, config),
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
    records = make_variant_records(variant_count)
    config = ModelConfig(trait_type=TraitType.BINARY, max_outer_iterations=5)

    result = fit_variational_em(
        genotypes=genotype_matrix,
        covariates=covariate_matrix,
        targets=target_vector,
        records=records,
        config=config,
        tie_map=build_tie_map(genotype_matrix, records, config),
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
    config = ModelConfig(trait_type=TraitType.QUANTITATIVE, max_outer_iterations=12)

    result = fit_variational_em(
        genotypes=genotype_matrix,
        covariates=covariate_matrix,
        targets=target_vector,
        records=records,
        config=config,
        tie_map=build_tie_map(genotype_matrix, records, config),
    )
    assert np.argmax(np.abs(result.beta_reduced)) == 3


def test_trigamma_matches_scipy_polygamma_for_small_shapes():
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

    p_parameter = local_shape_a - 0.5
    chi = coefficient_second_moment / baseline_prior_variances
    expected_auxiliary_delta = auxiliary_delta.copy()
    expected_local_scale = np.ones_like(expected_auxiliary_delta)
    for _iteration_index in range(6):
        psi = 2.0 * expected_auxiliary_delta
        z_value = np.sqrt(chi * psi)
        expected_local_scale = np.sqrt(chi / psi) * (kve(p_parameter + 1.0, z_value) / kve(p_parameter, z_value))
        expected_auxiliary_delta = (local_shape_a + local_shape_b) / (1.0 + expected_local_scale)

    np.testing.assert_allclose(updated_local_scale, expected_local_scale)
    np.testing.assert_allclose(updated_auxiliary_delta, expected_auxiliary_delta)


def test_quantitative_noise_update_includes_posterior_uncertainty(random_generator):
    sample_count, variant_count = 30, 4
    genotype_matrix = random_generator.standard_normal((sample_count, variant_count)).astype(np.float32)
    covariate_matrix = np.ones((sample_count, 1), dtype=np.float32)
    target_vector = random_generator.standard_normal(sample_count).astype(np.float32)
    prior_variances = np.array([3.0, 2.0, 1.5, 0.5], dtype=np.float32)
    sigma_error2 = 0.7

    alpha, beta, _beta_variance, _linear_predictor, _objective, updated_sigma_error2 = _quantitative_posterior_state(
        genotype_matrix=genotype_matrix,
        covariate_matrix=covariate_matrix,
        targets=target_vector,
        prior_variances=prior_variances,
        sigma_error2=sigma_error2,
        sigma_error_floor=1e-6,
    )

    genotype_matrix64 = genotype_matrix.astype(np.float64)
    covariate_matrix64 = covariate_matrix.astype(np.float64)
    target_vector64 = target_vector.astype(np.float64)
    prior_variances64 = prior_variances.astype(np.float64)
    covariance_matrix = sigma_error2 * np.eye(sample_count, dtype=np.float64) + (
        genotype_matrix64 * prior_variances64[None, :]
    ) @ genotype_matrix64.T
    covariance_inverse = np.linalg.inv(covariance_matrix)
    gls_normal_matrix = covariate_matrix64.T @ covariance_inverse @ covariate_matrix64
    alpha_exact = np.linalg.solve(
        gls_normal_matrix,
        covariate_matrix64.T @ covariance_inverse @ target_vector64,
    )
    beta_array = np.asarray(beta, dtype=np.float32).astype(np.float64)
    residual_vector = target_vector64 - covariate_matrix64 @ alpha_exact - genotype_matrix64 @ beta_array
    inverse_covariance_genotypes = covariance_inverse @ genotype_matrix64
    restricted_projected_genotypes = inverse_covariance_genotypes - (covariance_inverse @ covariate_matrix64) @ np.linalg.solve(
        gls_normal_matrix,
        covariate_matrix64.T @ inverse_covariance_genotypes,
    )
    leverage_diagonal = np.sum(genotype_matrix64 * restricted_projected_genotypes, axis=0)
    expected_sigma_error2 = (
        np.sum(residual_vector * residual_vector)
        + sigma_error2 * np.sum(prior_variances64 * leverage_diagonal)
    ) / sample_count

    np.testing.assert_allclose(alpha, alpha_exact.astype(np.float32), rtol=1e-5, atol=1e-5)
    assert np.isclose(float(updated_sigma_error2), expected_sigma_error2, rtol=1e-5, atol=1e-5)


def test_validation_restores_best_iterate(monkeypatch: pytest.MonkeyPatch):
    call_counter = {"count": 0}

    def fake_fit_collapsed_posterior(
        genotype_matrix,
        covariate_matrix,
        targets,
        reduced_prior_variances,
        sigma_error2,
        alpha_init,
        beta_init,
        trait_type,
        config,
    ):
        call_counter["count"] += 1
        if call_counter["count"] <= config.max_outer_iterations:
            parameter_value = np.float32(call_counter["count"])
            return PosteriorState(
                alpha=np.array([parameter_value], dtype=np.float32),
                beta=np.array([parameter_value], dtype=np.float32),
                beta_variance=np.ones(1, dtype=np.float32),
                linear_predictor=np.zeros(targets.shape[0], dtype=np.float32),
                collapsed_objective=0.0,
                sigma_error2=1.0,
            )
        return PosteriorState(
            alpha=np.asarray(alpha_init, dtype=np.float32),
            beta=np.asarray(beta_init, dtype=np.float32),
            beta_variance=np.ones(1, dtype=np.float32),
            linear_predictor=np.zeros(targets.shape[0], dtype=np.float32),
            collapsed_objective=0.0,
            sigma_error2=1.0,
        )

    def fake_validation_metric(trait_type, genotype_matrix, covariate_matrix, targets, alpha, beta):
        return float(beta[0])

    monkeypatch.setattr(mixture_inference, "_fit_collapsed_posterior", fake_fit_collapsed_posterior)
    monkeypatch.setattr(mixture_inference, "_validation_metric", fake_validation_metric)

    config = ModelConfig(
        trait_type=TraitType.QUANTITATIVE,
        max_outer_iterations=3,
        update_hyperparameters=False,
    )

    result = fit_variational_em(
        genotypes=np.zeros((8, 1), dtype=np.float32),
        covariates=np.ones((8, 1), dtype=np.float32),
        targets=np.zeros(8, dtype=np.float32),
        records=[VariantRecord("variant_0", VariantClass.SNV, "1", 100)],
        config=config,
        tie_map=build_tie_map(
            np.zeros((8, 1), dtype=np.float32),
            [VariantRecord("variant_0", VariantClass.SNV, "1", 100)],
            config,
        ),
        validation_data=(
            np.zeros((4, 1), dtype=np.float32),
            np.ones((4, 1), dtype=np.float32),
            np.zeros(4, dtype=np.float32),
        ),
    )

    assert np.isclose(float(result.beta_reduced[0]), 1.0)


def test_tpb_shape_vectors_are_learned_from_local_scale_state():
    updated_shape_a, updated_shape_b = _update_tpb_shape_vectors(
        class_membership_matrix=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        current_shape_a_vector=np.array([1.0, 1.0], dtype=np.float64),
        current_shape_b_vector=np.array([0.5, 0.5], dtype=np.float64),
        local_scale=np.array([0.1, 5.0], dtype=np.float64),
        auxiliary_delta=np.array([2.0, 0.2], dtype=np.float64),
        config=ModelConfig(maximum_tpb_shape_iterations=6, tpb_shape_learning_rate=0.1),
    )

    assert not np.allclose(updated_shape_a, [1.0, 1.0])
    assert not np.allclose(updated_shape_b, [0.5, 0.5])
    assert not np.isclose(updated_shape_a[0], updated_shape_a[1])

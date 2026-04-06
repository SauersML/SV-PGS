from __future__ import annotations

import numpy as np
import pytest
import jax.numpy as jnp
from scipy.special import kve

from sv_pgs.config import ModelConfig, TraitType, VariantClass
from sv_pgs.data import TieGroup, TieMap, VariantRecord
from sv_pgs.genotype import as_raw_genotype_matrix
from sv_pgs.inference import fit_variational_em
from sv_pgs.mixture_inference import (
    _build_restricted_projector_jax,
    _calibrate_binary_intercept,
    PosteriorState,
    _binary_posterior_state,
    _initialize_alpha_state,
    _member_prior_variances_from_reduced_state,
    _parse_scale_model_feature_names,
    _quantitative_posterior_state,
    _restricted_precision_projector,
    _restricted_variant_space_operator,
    _sample_space_operator,
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


def test_initialize_alpha_state_uses_target_prevalence_for_binary():
    covariate_matrix = np.ones((6, 2), dtype=np.float64)
    targets = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    alpha_state = _initialize_alpha_state(
        covariate_matrix=covariate_matrix,
        targets=targets,
        trait_type=TraitType.BINARY,
    )

    assert alpha_state.shape == (2,)
    assert np.isclose(alpha_state[0], 0.0, atol=1e-8)
    assert np.isclose(alpha_state[1], 0.0, atol=1e-8)


def test_initialize_alpha_state_fits_covariates_for_quantitative():
    covariate_matrix = np.array(
        [
            [1.0, -1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
        ],
        dtype=np.float64,
    )
    targets = np.array([-1.0, 1.0, 3.0, 5.0], dtype=np.float64)

    alpha_state = _initialize_alpha_state(
        covariate_matrix=covariate_matrix,
        targets=targets,
        trait_type=TraitType.QUANTITATIVE,
    )

    np.testing.assert_allclose(alpha_state, np.array([1.0, 2.0], dtype=np.float64), atol=1e-6)


def test_binary_posterior_stops_after_stalled_trust_region_step(random_generator, monkeypatch):
    sample_count, variant_count = 64, 6
    genotype_matrix = random_generator.standard_normal((sample_count, variant_count)).astype(np.float32)
    standardized = as_raw_genotype_matrix(genotype_matrix).standardized(
        means=np.zeros(variant_count, dtype=np.float32),
        scales=np.ones(variant_count, dtype=np.float32),
    )
    standardized._dense_cache = standardized.materialize()
    covariate_matrix = np.ones((sample_count, 1), dtype=np.float32)
    targets = np.concatenate(
        [np.ones(sample_count // 2, dtype=np.float32), np.zeros(sample_count - sample_count // 2, dtype=np.float32)]
    )
    prior_variances = np.ones(variant_count, dtype=np.float32)
    restricted_call_count = 0

    def fake_restricted_posterior_state(
        genotype_matrix,
        covariate_matrix,
        targets,
        prior_variances,
        diagonal_noise,
        solver_tolerance,
        maximum_linear_solver_iterations,
        logdet_probe_count,
        logdet_lanczos_steps,
        exact_solver_matrix_limit,
        posterior_variance_batch_size,
        posterior_variance_probe_count,
        random_seed,
        compute_logdet,
        compute_beta_variance=True,
        initial_beta_guess=None,
    ):
        nonlocal restricted_call_count
        restricted_call_count += 1
        beta = np.zeros(prior_variances.shape[0], dtype=np.float64) if initial_beta_guess is None else np.asarray(initial_beta_guess, dtype=np.float64)
        alpha = np.zeros(covariate_matrix.shape[1], dtype=np.float64)
        sample_dim = targets.shape[0]
        return (
            alpha,
            beta,
            np.zeros_like(beta),
            np.zeros(sample_dim, dtype=np.float64),
            np.zeros(sample_dim, dtype=np.float64),
            0.0,
            0.0,
            0.0,
        )

    monkeypatch.setattr(mixture_inference, "_restricted_posterior_state", fake_restricted_posterior_state)

    alpha, beta, beta_variance, linear_predictor, objective = _binary_posterior_state(
        genotype_matrix=standardized,
        covariate_matrix=covariate_matrix,
        targets=targets,
        prior_variances=prior_variances,
        alpha_init=np.zeros(1, dtype=np.float32),
        beta_init=np.zeros(variant_count, dtype=np.float32),
        minimum_weight=1e-4,
        max_iterations=20,
        gradient_tolerance=1e-5,
        initial_damping=1.0,
        damping_increase_factor=10.0,
        damping_decrease_factor=0.1,
        success_threshold=0.25,
        minimum_damping=1e-8,
    )

    assert restricted_call_count == 3
    assert alpha.shape == (1,)
    assert beta.shape == (variant_count,)
    assert beta_variance.shape == (variant_count,)
    assert linear_predictor.shape == (sample_count,)
    assert np.isfinite(objective)


def test_binary_posterior_stops_immediately_after_tiny_reject_gain(random_generator, monkeypatch):
    sample_count, variant_count = 64, 6
    genotype_matrix = random_generator.standard_normal((sample_count, variant_count)).astype(np.float32)
    standardized = as_raw_genotype_matrix(genotype_matrix).standardized(
        means=np.zeros(variant_count, dtype=np.float32),
        scales=np.ones(variant_count, dtype=np.float32),
    )
    standardized._dense_cache = standardized.materialize()
    covariate_matrix = np.ones((sample_count, 1), dtype=np.float32)
    targets = np.concatenate(
        [np.ones(sample_count // 2, dtype=np.float32), np.zeros(sample_count - sample_count // 2, dtype=np.float32)]
    )
    prior_variances = np.ones(variant_count, dtype=np.float32)
    restricted_call_count = 0

    def fake_restricted_posterior_state(
        genotype_matrix,
        covariate_matrix,
        targets,
        prior_variances,
        diagonal_noise,
        solver_tolerance,
        maximum_linear_solver_iterations,
        logdet_probe_count,
        logdet_lanczos_steps,
        exact_solver_matrix_limit,
        posterior_variance_batch_size,
        posterior_variance_probe_count,
        random_seed,
        compute_logdet,
        compute_beta_variance=True,
        initial_beta_guess=None,
    ):
        nonlocal restricted_call_count
        restricted_call_count += 1
        alpha = np.zeros(covariate_matrix.shape[1], dtype=np.float64)
        beta = np.full(prior_variances.shape[0], 10.0, dtype=np.float64)
        sample_dim = targets.shape[0]
        return (
            alpha,
            beta,
            np.zeros_like(beta),
            np.zeros(sample_dim, dtype=np.float64),
            np.zeros(sample_dim, dtype=np.float64),
            0.0,
            0.0,
            0.0,
        )

    monkeypatch.setattr(mixture_inference, "_restricted_posterior_state", fake_restricted_posterior_state)

    alpha, beta, beta_variance, linear_predictor, objective = _binary_posterior_state(
        genotype_matrix=standardized,
        covariate_matrix=covariate_matrix,
        targets=targets,
        prior_variances=prior_variances,
        alpha_init=np.zeros(1, dtype=np.float32),
        beta_init=np.zeros(variant_count, dtype=np.float32),
        minimum_weight=1e-4,
        max_iterations=20,
        gradient_tolerance=1e-8,
        initial_damping=1.0,
        damping_increase_factor=10.0,
        damping_decrease_factor=0.1,
        success_threshold=0.25,
        minimum_damping=1e-8,
    )

    assert restricted_call_count == 3
    assert alpha.shape == (1,)
    assert beta.shape == (variant_count,)
    assert beta_variance.shape == (variant_count,)
    assert linear_predictor.shape == (sample_count,)
    assert np.isfinite(objective)


def test_binary_posterior_reuses_proposal_across_rejects(random_generator, monkeypatch):
    sample_count, variant_count = 64, 6
    genotype_matrix = random_generator.standard_normal((sample_count, variant_count)).astype(np.float32)
    standardized = as_raw_genotype_matrix(genotype_matrix).standardized(
        means=np.zeros(variant_count, dtype=np.float32),
        scales=np.ones(variant_count, dtype=np.float32),
    )
    standardized._dense_cache = standardized.materialize()
    covariate_matrix = np.ones((sample_count, 1), dtype=np.float32)
    targets = np.concatenate(
        [np.ones(sample_count // 2, dtype=np.float32), np.zeros(sample_count - sample_count // 2, dtype=np.float32)]
    )
    prior_variances = np.full(variant_count, 1e-3, dtype=np.float32)
    restricted_call_count = 0

    def fake_restricted_posterior_state(
        genotype_matrix,
        covariate_matrix,
        targets,
        prior_variances,
        diagonal_noise,
        solver_tolerance,
        maximum_linear_solver_iterations,
        logdet_probe_count,
        logdet_lanczos_steps,
        exact_solver_matrix_limit,
        posterior_variance_batch_size,
        posterior_variance_probe_count,
        random_seed,
        compute_logdet,
        compute_beta_variance=True,
        initial_beta_guess=None,
    ):
        nonlocal restricted_call_count
        restricted_call_count += 1
        alpha = np.zeros(covariate_matrix.shape[1], dtype=np.float64)
        beta = np.full(prior_variances.shape[0], 100.0, dtype=np.float64)
        sample_dim = targets.shape[0]
        return (
            alpha,
            beta,
            np.zeros_like(beta),
            np.zeros(sample_dim, dtype=np.float64),
            np.zeros(sample_dim, dtype=np.float64),
            0.0,
            0.0,
            0.0,
        )

    monkeypatch.setattr(mixture_inference, "_restricted_posterior_state", fake_restricted_posterior_state)

    alpha, beta, beta_variance, linear_predictor, objective = _binary_posterior_state(
        genotype_matrix=standardized,
        covariate_matrix=covariate_matrix,
        targets=targets,
        prior_variances=prior_variances,
        alpha_init=np.zeros(1, dtype=np.float32),
        beta_init=np.zeros(variant_count, dtype=np.float32),
        minimum_weight=1e-4,
        max_iterations=5,
        gradient_tolerance=1e-8,
        initial_damping=1.0,
        damping_increase_factor=10.0,
        damping_decrease_factor=0.1,
        success_threshold=0.25,
        minimum_damping=1e-8,
    )

    assert restricted_call_count == 3
    assert alpha.shape == (1,)
    assert beta.shape == (variant_count,)
    assert beta_variance.shape == (variant_count,)
    assert linear_predictor.shape == (sample_count,)
    assert np.isfinite(objective)


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


def test_gpu_sample_space_operator_matmat_matches_dense_reference(random_generator):
    sample_count, variant_count = 24, 96
    genotype_values = random_generator.normal(size=(sample_count, variant_count)).astype(np.float32)
    prior_variances = random_generator.uniform(0.2, 1.2, size=variant_count).astype(np.float64)
    diagonal_noise = random_generator.uniform(0.5, 1.5, size=sample_count).astype(np.float64)
    rhs_matrix = random_generator.normal(size=(sample_count, 11)).astype(np.float64)

    standardized = as_raw_genotype_matrix(genotype_values).standardized(
        means=np.zeros(variant_count, dtype=np.float32),
        scales=np.ones(variant_count, dtype=np.float32),
    )
    standardized._dense_cache = standardized.materialize()
    operator = _sample_space_operator(standardized, prior_variances, diagonal_noise)

    dense_matrix = genotype_values.astype(np.float64)
    expected = diagonal_noise[:, None] * rhs_matrix + dense_matrix @ (
        prior_variances[:, None] * (dense_matrix.T @ rhs_matrix)
    )

    np.testing.assert_allclose(
        np.asarray(operator.matmat(rhs_matrix), dtype=np.float64),
        expected,
        rtol=1e-5,
        atol=1e-5,
    )


def test_restricted_variant_space_operator_matmat_matches_columnwise(random_generator):
    sample_count, variant_count = 32, 7
    genotype_values = random_generator.normal(size=(sample_count, variant_count)).astype(np.float32)
    covariate_matrix = np.column_stack(
        [np.ones(sample_count), random_generator.normal(size=(sample_count, 2))]
    ).astype(np.float32)
    diagonal_noise = random_generator.uniform(0.5, 1.5, size=sample_count).astype(np.float64)
    prior_precision = random_generator.uniform(0.8, 2.0, size=variant_count).astype(np.float64)
    rhs_matrix = random_generator.normal(size=(variant_count, 5)).astype(np.float64)

    standardized = as_raw_genotype_matrix(genotype_values).standardized(
        means=np.zeros(variant_count, dtype=np.float32),
        scales=np.ones(variant_count, dtype=np.float32),
    )
    standardized._dense_cache = standardized.materialize()
    _inverse_diagonal_noise, covariate_precision_cholesky, _covariate_precision_logdet, _apply_projector = (
        _restricted_precision_projector(covariate_matrix.astype(np.float64), diagonal_noise)
    )
    operator = _restricted_variant_space_operator(
        genotype_matrix=standardized,
        prior_precision=prior_precision,
        inverse_diagonal_noise=_inverse_diagonal_noise,
        covariate_matrix=covariate_matrix.astype(np.float64),
        covariate_precision_cholesky=covariate_precision_cholesky,
        batch_size=4,
    )

    expected = np.column_stack(
        [
            np.asarray(operator.matvec(rhs_matrix[:, column_index]), dtype=np.float64)
            for column_index in range(rhs_matrix.shape[1])
        ]
    )

    np.testing.assert_allclose(
        np.asarray(operator.matmat(rhs_matrix), dtype=np.float64),
        expected,
        rtol=1e-5,
        atol=1e-5,
    )


def test_restricted_projector_jax_matches_numpy_projector(random_generator):
    sample_count = 24
    covariate_matrix = np.column_stack(
        [np.ones(sample_count), random_generator.normal(size=(sample_count, 2))]
    ).astype(np.float64)
    diagonal_noise = random_generator.uniform(0.5, 1.5, size=sample_count).astype(np.float64)
    rhs_matrix = random_generator.normal(size=(sample_count, 4)).astype(np.float64)

    inverse_diagonal_noise, covariate_precision_cholesky, _covariate_precision_logdet, apply_projector = (
        _restricted_precision_projector(covariate_matrix, diagonal_noise)
    )

    expected = apply_projector(rhs_matrix)
    apply_projector_jax = _build_restricted_projector_jax(
        inverse_diagonal_noise=inverse_diagonal_noise,
        covariate_matrix=covariate_matrix,
        covariate_precision_cholesky=covariate_precision_cholesky,
        compute_dtype=jnp.float64,
    )
    actual = np.asarray(apply_projector_jax(rhs_matrix), dtype=np.float64)

    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)


def test_materialized_woodbury_posterior_matches_dense_reference(random_generator):
    sample_count, variant_count = 40, 6
    genotype_values = random_generator.normal(size=(sample_count, variant_count)).astype(np.float32)
    covariate_matrix = np.column_stack(
        [np.ones(sample_count), random_generator.normal(size=(sample_count, 2))]
    ).astype(np.float32)
    target_vector = random_generator.normal(size=sample_count).astype(np.float32)
    prior_variances = random_generator.uniform(0.2, 1.0, size=variant_count).astype(np.float32)

    standardized = as_raw_genotype_matrix(genotype_values).standardized(
        means=np.zeros(variant_count, dtype=np.float32),
        scales=np.ones(variant_count, dtype=np.float32),
    )
    standardized._dense_cache = standardized.materialize()

    gpu_result = _quantitative_posterior_state(
        genotype_matrix=standardized,
        covariate_matrix=covariate_matrix,
        targets=target_vector,
        prior_variances=prior_variances,
        sigma_error2=1.0,
        sigma_error_floor=1e-6,
        exact_solver_matrix_limit=8,
        posterior_variance_batch_size=3,
    )
    dense_result = _quantitative_posterior_state(
        genotype_matrix=genotype_values,
        covariate_matrix=covariate_matrix,
        targets=target_vector,
        prior_variances=prior_variances,
        sigma_error2=1.0,
        sigma_error_floor=1e-6,
        exact_solver_matrix_limit=8,
        posterior_variance_batch_size=3,
    )

    for gpu_value, dense_value in zip(gpu_result, dense_result):
        np.testing.assert_allclose(
            np.asarray(gpu_value, dtype=np.float64),
            np.asarray(dense_value, dtype=np.float64),
            rtol=1e-5,
            atol=1e-5,
        )


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


def test_binary_validation_uses_calibrated_intercept(monkeypatch: pytest.MonkeyPatch):
    validation_alphas: list[np.ndarray] = []
    expected_shift = _calibrate_binary_intercept(
        linear_predictor=np.zeros(8, dtype=np.float64),
        targets=np.array([1.0] * 6 + [0.0] * 2, dtype=np.float64),
    )

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
        return PosteriorState(
            alpha=np.zeros(covariate_matrix.shape[1], dtype=np.float64),
            beta=np.zeros(1, dtype=np.float64),
            beta_variance=np.ones(1, dtype=np.float64),
            linear_predictor=np.zeros(targets.shape[0], dtype=np.float64),
            collapsed_objective=0.0,
            sigma_error2=1.0,
        )

    def fake_validation_metric(trait_type, genotype_matrix, covariate_matrix, targets, alpha, beta):
        validation_alphas.append(np.asarray(alpha, dtype=np.float64).copy())
        return 0.0

    monkeypatch.setattr(mixture_inference, "_fit_collapsed_posterior", fake_fit_collapsed_posterior)
    monkeypatch.setattr(mixture_inference, "_validation_metric", fake_validation_metric)

    config = ModelConfig(
        trait_type=TraitType.BINARY,
        max_outer_iterations=1,
        update_hyperparameters=False,
        binary_intercept_calibration=True,
    )
    targets = np.array([1.0] * 6 + [0.0] * 2, dtype=np.float32)

    result = fit_variational_em(
        genotypes=np.zeros((8, 1), dtype=np.float32),
        covariates=np.ones((8, 1), dtype=np.float32),
        targets=targets,
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
            np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32),
        ),
    )

    assert validation_alphas
    assert np.isclose(validation_alphas[0][0], expected_shift, atol=1e-6)
    assert np.isclose(float(result.alpha[0]), expected_shift, atol=1e-6)


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


def test_member_prior_variances_preserve_member_metadata_with_ties():
    member_records = [
        VariantRecord("variant_0", VariantClass.SNV, "1", 100),
        VariantRecord("variant_1", VariantClass.DELETION_SHORT, "1", 101, is_copy_number=True),
    ]
    tie_map = TieMap(
        kept_indices=np.array([0], dtype=np.int32),
        original_to_reduced=np.array([0, 0], dtype=np.int32),
        reduced_to_group=[
            TieGroup(
                representative_index=0,
                member_indices=np.array([0, 1], dtype=np.int32),
                signs=np.array([1.0, 1.0], dtype=np.float32),
            )
        ],
    )

    member_prior_variances = _member_prior_variances_from_reduced_state(
        member_records=member_records,
        tie_map=tie_map,
        scale_model_coefficients=np.array([1.0, -1.0], dtype=np.float64),
        scale_model_feature_specs=_parse_scale_model_feature_names([
            "type_offset::snv",
            "type_offset::deletion_short",
        ]),
        global_scale=1.0,
        local_scale=np.array([2.0], dtype=np.float64),
        config=ModelConfig(),
    )

    assert member_prior_variances.shape == (2,)
    assert member_prior_variances[0] > member_prior_variances[1]
    assert not np.isclose(member_prior_variances[0], member_prior_variances[1])

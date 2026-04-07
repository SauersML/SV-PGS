from __future__ import annotations

import sys
import types
from typing import Any, cast

import numpy as np
import pytest
import jax.numpy as jnp
from scipy.linalg import solve_triangular as scipy_solve_triangular
from scipy.special import kve

from sv_pgs.config import ModelConfig, TraitType, VariantClass
from sv_pgs.data import TieGroup, TieMap, VariantRecord
from sv_pgs.genotype import as_raw_genotype_matrix
from sv_pgs.inference import fit_variational_em
from sv_pgs.mixture_inference import (
    _binary_newton_solver_controls,
    _basil_working_set_indices,
    _build_restricted_projector_jax,
    _calibrate_binary_intercept,
    PosteriorState,
    VariationalFitCheckpoint,
    _binary_posterior_state,
    _initialize_alpha_state,
    _member_prior_variances_from_reduced_state,
    _orthogonal_probe_matrix,
    _parse_scale_model_feature_names,
    _prefer_iterative_variant_space,
    _quantitative_posterior_state,
    _restricted_posterior_state,
    _restricted_precision_projector,
    _sample_space_preconditioner,
    _solve_sample_space_rhs_cpu,
    _stochastic_restricted_cross_leverage_diagonal,
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


def test_quantitative_inference_runs_with_stochastic_variant_updates(random_generator):
    sample_count, variant_count = 64, 12
    genotype_matrix = random_generator.standard_normal((sample_count, variant_count)).astype(np.float32)
    covariate_matrix = np.column_stack(
        [np.ones(sample_count, dtype=np.float32), random_generator.standard_normal(sample_count).astype(np.float32)]
    )
    true_coefficients = np.zeros(variant_count, dtype=np.float32)
    true_coefficients[:2] = np.array([1.1, -0.8], dtype=np.float32)
    target_vector = genotype_matrix @ true_coefficients + 0.2 * random_generator.standard_normal(sample_count).astype(np.float32)
    records = make_variant_records(variant_count)
    config = ModelConfig(
        trait_type=TraitType.QUANTITATIVE,
        max_outer_iterations=3,
        update_hyperparameters=False,
        stochastic_variational_updates=True,
        stochastic_min_variant_count=1,
        stochastic_variant_batch_size=4,
    )

    result = fit_variational_em(
        genotypes=genotype_matrix,
        covariates=covariate_matrix,
        targets=target_vector,
        records=records,
        config=config,
        tie_map=build_tie_map(genotype_matrix, records, config),
    )

    assert result.objective_history


def test_temporary_working_set_restricted_posterior_matches_full_solution(random_generator):
    sample_count, variant_count = 32, 12
    genotype_values = random_generator.standard_normal((sample_count, variant_count)).astype(np.float32)
    covariate_matrix = np.column_stack(
        [np.ones(sample_count, dtype=np.float32), random_generator.standard_normal(sample_count).astype(np.float32)]
    )
    targets = random_generator.standard_normal(sample_count).astype(np.float32)
    prior_variances = random_generator.uniform(0.2, 1.5, size=variant_count).astype(np.float64)
    diagonal_noise = np.full(sample_count, 1.0, dtype=np.float64)

    standardized = as_raw_genotype_matrix(genotype_values).standardized(
        means=np.zeros(variant_count, dtype=np.float32),
        scales=np.ones(variant_count, dtype=np.float32),
    )
    standardized._dense_cache = standardized.materialize()
    initial_beta = np.zeros(variant_count, dtype=np.float64)

    full_result = _restricted_posterior_state(
        genotype_matrix=standardized,
        covariate_matrix=np.asarray(covariate_matrix, dtype=np.float64),
        targets=np.asarray(targets, dtype=np.float64),
        prior_variances=prior_variances,
        diagonal_noise=diagonal_noise,
        solver_tolerance=1e-7,
        maximum_linear_solver_iterations=256,
        logdet_probe_count=4,
        logdet_lanczos_steps=6,
        exact_solver_matrix_limit=2,
        posterior_variance_batch_size=4,
        posterior_variance_probe_count=4,
        random_seed=0,
        compute_logdet=False,
        compute_beta_variance=False,
        initial_beta_guess=initial_beta,
        temporary_working_sets=False,
    )
    working_result = _restricted_posterior_state(
        genotype_matrix=standardized,
        covariate_matrix=np.asarray(covariate_matrix, dtype=np.float64),
        targets=np.asarray(targets, dtype=np.float64),
        prior_variances=prior_variances,
        diagonal_noise=diagonal_noise,
        solver_tolerance=1e-7,
        maximum_linear_solver_iterations=256,
        logdet_probe_count=4,
        logdet_lanczos_steps=6,
        exact_solver_matrix_limit=2,
        posterior_variance_batch_size=4,
        posterior_variance_probe_count=4,
        random_seed=0,
        compute_logdet=False,
        compute_beta_variance=False,
        initial_beta_guess=initial_beta,
        temporary_working_sets=True,
        temporary_working_set_min_variants=1,
        temporary_working_set_initial_size=3,
        temporary_working_set_growth=3,
        temporary_working_set_max_passes=8,
        temporary_working_set_coefficient_tolerance=0.0,
    )

    for working_value, full_value in zip(working_result[:6], full_result[:6]):
        np.testing.assert_allclose(
            np.asarray(working_value, dtype=np.float64),
            np.asarray(full_value, dtype=np.float64),
            rtol=1e-5,
            atol=1e-5,
        )


def test_basil_working_set_keeps_ever_active_variants():
    screening_score = np.array([0.9, 0.1, 0.8, 0.7, 0.6], dtype=np.float64)
    ever_active_indices = np.array([1, 4], dtype=np.int32)

    working_indices = _basil_working_set_indices(
        screening_score=screening_score,
        ever_active_indices=ever_active_indices,
        target_size=3,
    )

    assert set(working_indices.tolist()) >= {1, 4}
    assert working_indices.shape == (3,)


def test_temporary_working_set_warm_start_tracks_ever_active_support(random_generator):
    sample_count, variant_count = 36, 14
    genotype_values = random_generator.standard_normal((sample_count, variant_count)).astype(np.float32)
    targets = (
        1.4 * genotype_values[:, 0]
        - 0.9 * genotype_values[:, 5]
        + 0.2 * random_generator.standard_normal(sample_count).astype(np.float32)
    )
    standardized = as_raw_genotype_matrix(genotype_values).standardized(
        means=np.zeros(variant_count, dtype=np.float32),
        scales=np.ones(variant_count, dtype=np.float32),
    )
    standardized._dense_cache = standardized.materialize()
    warm_start = mixture_inference._RestrictedPosteriorWarmStart()

    result = _restricted_posterior_state(
        genotype_matrix=standardized,
        covariate_matrix=np.ones((sample_count, 1), dtype=np.float64),
        targets=np.asarray(targets, dtype=np.float64),
        prior_variances=np.ones(variant_count, dtype=np.float64),
        diagonal_noise=np.ones(sample_count, dtype=np.float64),
        solver_tolerance=1e-7,
        maximum_linear_solver_iterations=256,
        logdet_probe_count=2,
        logdet_lanczos_steps=4,
        exact_solver_matrix_limit=2,
        posterior_variance_batch_size=4,
        posterior_variance_probe_count=2,
        random_seed=0,
        compute_logdet=False,
        compute_beta_variance=False,
        initial_beta_guess=np.zeros(variant_count, dtype=np.float64),
        warm_start=warm_start,
        temporary_working_sets=True,
        temporary_working_set_min_variants=1,
        temporary_working_set_initial_size=2,
        temporary_working_set_growth=2,
        temporary_working_set_max_passes=8,
        temporary_working_set_coefficient_tolerance=0.0,
    )

    assert warm_start.temporary_working_set_variant_count == variant_count
    assert warm_start.temporary_working_set_ever_active is not None
    assert warm_start.temporary_working_set_ever_active.size > 0
    active_support = np.flatnonzero(np.abs(np.asarray(result[1], dtype=np.float64)) > 1e-10)
    assert set(active_support.tolist()).issubset(set(warm_start.temporary_working_set_ever_active.tolist()))


def test_temporary_working_set_warm_start_resets_for_new_matrix_with_same_width(random_generator):
    sample_count, variant_count = 20, 8
    first_genotype_values = random_generator.standard_normal((sample_count, variant_count)).astype(np.float32)
    second_genotype_values = random_generator.standard_normal((sample_count, variant_count)).astype(np.float32)
    first_standardized = as_raw_genotype_matrix(first_genotype_values).standardized(
        means=np.zeros(variant_count, dtype=np.float32),
        scales=np.ones(variant_count, dtype=np.float32),
    )
    second_standardized = as_raw_genotype_matrix(second_genotype_values).standardized(
        means=np.zeros(variant_count, dtype=np.float32),
        scales=np.ones(variant_count, dtype=np.float32),
    )
    warm_start = mixture_inference._RestrictedPosteriorWarmStart(
        temporary_working_set_variant_count=variant_count,
        temporary_working_set_matrix_token=id(first_standardized),
        temporary_working_set_ever_active=np.array([1, 3, 5], dtype=np.int32),
    )

    mixture_inference._reset_temporary_working_set_warm_start(
        warm_start=warm_start,
        genotype_matrix=second_standardized,
        variant_count=variant_count,
    )

    assert warm_start.temporary_working_set_variant_count == variant_count
    assert warm_start.temporary_working_set_matrix_token == id(second_standardized)
    assert warm_start.temporary_working_set_ever_active is None


def test_binary_inference_runs_with_stochastic_variant_updates(random_generator):
    sample_count, variant_count = 72, 10
    genotype_matrix = random_generator.standard_normal((sample_count, variant_count)).astype(np.float32)
    covariate_matrix = np.column_stack(
        [np.ones(sample_count, dtype=np.float32), random_generator.standard_normal(sample_count).astype(np.float32)]
    )
    true_coefficients = np.zeros(variant_count, dtype=np.float32)
    true_coefficients[0] = 1.25
    linear_predictor = genotype_matrix @ true_coefficients + 0.3 * covariate_matrix[:, 1]
    target_vector = (random_generator.random(sample_count) < (1.0 / (1.0 + np.exp(-linear_predictor)))).astype(np.float32)
    records = make_variant_records(variant_count)
    config = ModelConfig(
        trait_type=TraitType.BINARY,
        max_outer_iterations=3,
        update_hyperparameters=False,
        stochastic_variational_updates=True,
        stochastic_min_variant_count=1,
        stochastic_variant_batch_size=3,
    )

    result = fit_variational_em(
        genotypes=genotype_matrix,
        covariates=covariate_matrix,
        targets=target_vector,
        records=records,
        config=config,
        tie_map=build_tie_map(genotype_matrix, records, config),
    )

    assert result.objective_history
    assert np.all(np.isfinite(result.alpha))
    assert np.all(np.isfinite(result.beta_reduced))
    assert result.sigma_error2 == 1.0


def test_stochastic_updates_can_skip_final_posterior_refinement(random_generator, monkeypatch):
    sample_count, variant_count = 48, 9
    genotype_matrix = random_generator.standard_normal((sample_count, variant_count)).astype(np.float32)
    covariate_matrix = np.column_stack(
        [np.ones(sample_count, dtype=np.float32), random_generator.standard_normal(sample_count).astype(np.float32)]
    )
    target_vector = (
        genotype_matrix[:, 0] * 0.8
        - genotype_matrix[:, 1] * 0.4
        + 0.1 * random_generator.standard_normal(sample_count).astype(np.float32)
    )
    records = make_variant_records(variant_count)
    config = ModelConfig(
        trait_type=TraitType.QUANTITATIVE,
        max_outer_iterations=2,
        update_hyperparameters=False,
        stochastic_variational_updates=True,
        stochastic_min_variant_count=1,
        stochastic_variant_batch_size=3,
        final_posterior_refinement=False,
    )
    tie_map = build_tie_map(genotype_matrix, records, config)
    original_fit_collapsed_posterior = mixture_inference._fit_collapsed_posterior

    def guarded_fit_collapsed_posterior(*args, **kwargs):
        genotype_arg = kwargs["genotype_matrix"]
        compute_logdet = kwargs.get("compute_logdet", True)
        if compute_logdet and genotype_arg.shape[1] == tie_map.kept_indices.shape[0]:
            raise AssertionError("unexpected final posterior refinement")
        return original_fit_collapsed_posterior(*args, **kwargs)

    monkeypatch.setattr(mixture_inference, "_fit_collapsed_posterior", guarded_fit_collapsed_posterior)

    result = fit_variational_em(
        genotypes=genotype_matrix,
        covariates=covariate_matrix,
        targets=target_vector,
        records=records,
        config=config,
        tie_map=tie_map,
    )

    assert result.objective_history
    assert np.all(np.isfinite(result.beta_variance))


def test_fit_variational_em_ignores_incompatible_resume_checkpoint(random_generator):
    sample_count, variant_count = 24, 5
    genotype_matrix = random_generator.normal(size=(sample_count, variant_count)).astype(np.float32)
    covariate_matrix = np.column_stack(
        [np.ones(sample_count, dtype=np.float32), random_generator.normal(size=sample_count).astype(np.float32)]
    )
    target_vector = random_generator.normal(size=sample_count).astype(np.float32)
    records = make_variant_records(variant_count)
    config = ModelConfig(
        trait_type=TraitType.QUANTITATIVE,
        max_outer_iterations=2,
        exact_solver_matrix_limit=128,
        minimum_minor_allele_frequency=0.0,
    )
    tie_map = build_tie_map(genotype_matrix, records, config)
    reduced_count = len(tie_map.reduced_to_group)
    checkpoint = VariationalFitCheckpoint(
        config_signature="stale-config",
        prior_design_signature="stale-design",
        validation_enabled=False,
        completed_iterations=1,
        alpha_state=np.full(covariate_matrix.shape[1], np.nan, dtype=np.float64),
        beta_state=np.full(reduced_count, np.nan, dtype=np.float64),
        local_scale=np.full(reduced_count, np.nan, dtype=np.float64),
        auxiliary_delta=np.full(reduced_count, np.nan, dtype=np.float64),
        sigma_error2=np.nan,
        global_scale=np.nan,
        scale_model_coefficients=np.full(1, np.nan, dtype=np.float64),
        tpb_shape_a_vector=np.full(1, np.nan, dtype=np.float64),
        tpb_shape_b_vector=np.full(1, np.nan, dtype=np.float64),
        objective_history=[np.nan],
        validation_history=[],
        previous_alpha=None,
        previous_beta=None,
        previous_local_scale=None,
        previous_theta=None,
        previous_tpb_shape_a_vector=None,
        previous_tpb_shape_b_vector=None,
        best_validation_metric=None,
        best_alpha=None,
        best_beta=None,
        best_beta_variance=None,
        best_local_scale=None,
        best_theta=None,
        best_sigma_error2=None,
        best_tpb_shape_a_vector=None,
        best_tpb_shape_b_vector=None,
    )

    result = fit_variational_em(
        genotypes=genotype_matrix,
        covariates=covariate_matrix,
        targets=target_vector,
        records=records,
        config=config,
        tie_map=tie_map,
        resume_checkpoint=checkpoint,
    )

    assert result.objective_history
    assert np.all(np.isfinite(result.alpha))
    assert np.all(np.isfinite(result.beta_reduced))


def test_fit_variational_em_ignores_resume_checkpoint_when_validation_is_present(random_generator):
    sample_count, variant_count = 24, 5
    genotype_matrix = random_generator.normal(size=(sample_count, variant_count)).astype(np.float32)
    covariate_matrix = np.column_stack(
        [np.ones(sample_count, dtype=np.float32), random_generator.normal(size=sample_count).astype(np.float32)]
    )
    target_vector = random_generator.normal(size=sample_count).astype(np.float32)
    validation_genotypes = random_generator.normal(size=(8, variant_count)).astype(np.float32)
    validation_covariates = np.column_stack(
        [np.ones(8, dtype=np.float32), random_generator.normal(size=8).astype(np.float32)]
    )
    validation_targets = random_generator.normal(size=8).astype(np.float32)
    records = make_variant_records(variant_count)
    config = ModelConfig(
        trait_type=TraitType.QUANTITATIVE,
        max_outer_iterations=2,
        exact_solver_matrix_limit=128,
        minimum_minor_allele_frequency=0.0,
    )
    tie_map = build_tie_map(genotype_matrix, records, config)
    reduced_records = mixture_inference.collapse_tie_groups(list(records), tie_map)
    prior_design = mixture_inference._build_prior_design(reduced_records)
    reduced_count = len(tie_map.reduced_to_group)
    checkpoint = VariationalFitCheckpoint(
        config_signature=mixture_inference._checkpoint_config_signature(config),
        prior_design_signature=mixture_inference._checkpoint_prior_design_signature(prior_design),
        validation_enabled=True,
        completed_iterations=1,
        alpha_state=np.full(covariate_matrix.shape[1], np.nan, dtype=np.float64),
        beta_state=np.full(reduced_count, np.nan, dtype=np.float64),
        local_scale=np.full(reduced_count, np.nan, dtype=np.float64),
        auxiliary_delta=np.full(reduced_count, np.nan, dtype=np.float64),
        sigma_error2=np.nan,
        global_scale=np.nan,
        scale_model_coefficients=np.full(prior_design.design_matrix.shape[1], np.nan, dtype=np.float64),
        tpb_shape_a_vector=np.full(prior_design.class_membership_matrix.shape[1], np.nan, dtype=np.float64),
        tpb_shape_b_vector=np.full(prior_design.class_membership_matrix.shape[1], np.nan, dtype=np.float64),
        objective_history=[np.nan],
        validation_history=[np.nan],
        previous_alpha=None,
        previous_beta=None,
        previous_local_scale=None,
        previous_theta=None,
        previous_tpb_shape_a_vector=None,
        previous_tpb_shape_b_vector=None,
        best_validation_metric=None,
        best_alpha=None,
        best_beta=None,
        best_beta_variance=None,
        best_local_scale=None,
        best_theta=None,
        best_sigma_error2=None,
        best_tpb_shape_a_vector=None,
        best_tpb_shape_b_vector=None,
    )

    result = fit_variational_em(
        genotypes=genotype_matrix,
        covariates=covariate_matrix,
        targets=target_vector,
        records=records,
        config=config,
        tie_map=tie_map,
        validation_data=(validation_genotypes, validation_covariates, validation_targets),
        resume_checkpoint=checkpoint,
        checkpoint_callback=lambda _checkpoint: (_ for _ in ()).throw(AssertionError("checkpoint callback should be disabled")),
    )

    assert result.objective_history
    assert result.validation_history
    assert np.all(np.isfinite(result.alpha))
    assert np.all(np.isfinite(result.beta_reduced))


def test_variational_em_supports_covariates_only_mode():
    covariate_matrix = np.array(
        [
            [1.0, -1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
        ],
        dtype=np.float32,
    )
    target_vector = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    validation_covariates = np.array(
        [
            [1.0, -0.5],
            [1.0, 1.5],
        ],
        dtype=np.float32,
    )
    validation_targets = np.array([-0.5, 1.5], dtype=np.float32)

    result = fit_variational_em(
        genotypes=np.empty((covariate_matrix.shape[0], 0), dtype=np.float32),
        covariates=covariate_matrix,
        targets=target_vector,
        records=[],
        config=ModelConfig(trait_type=TraitType.QUANTITATIVE, max_outer_iterations=5),
        tie_map=TieMap(
            kept_indices=np.zeros(0, dtype=np.int32),
            original_to_reduced=np.zeros(0, dtype=np.int32),
            reduced_to_group=[],
        ),
        validation_data=(
            np.empty((validation_covariates.shape[0], 0), dtype=np.float32),
            validation_covariates,
            validation_targets,
        ),
    )

    np.testing.assert_allclose(
        result.alpha,
        _initialize_alpha_state(covariate_matrix, target_vector, TraitType.QUANTITATIVE).astype(np.float32),
    )
    assert result.beta_reduced.shape == (0,)
    assert result.beta_variance.shape == (0,)
    assert result.prior_scales.shape == (0,)
    assert result.member_prior_variances.shape == (0,)
    assert len(result.objective_history) == 1
    assert len(result.validation_history) == 1


def test_variational_em_supports_binary_covariates_only_mode():
    covariate_matrix = np.array(
        [
            [1.0, -2.0],
            [1.0, -1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
        ],
        dtype=np.float32,
    )
    target_vector = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32)

    result = fit_variational_em(
        genotypes=np.empty((covariate_matrix.shape[0], 0), dtype=np.float32),
        covariates=covariate_matrix,
        targets=target_vector,
        records=[],
        config=ModelConfig(trait_type=TraitType.BINARY, max_outer_iterations=5),
        tie_map=TieMap(
            kept_indices=np.zeros(0, dtype=np.int32),
            original_to_reduced=np.zeros(0, dtype=np.int32),
            reduced_to_group=[],
        ),
    )

    probabilities = 1.0 / (1.0 + np.exp(-(covariate_matrix @ result.alpha)))
    assert result.beta_reduced.shape == (0,)
    assert result.beta_variance.shape == (0,)
    assert probabilities[0] < probabilities[-1]
    assert float(np.mean(probabilities[target_vector == 1.0])) > float(np.mean(probabilities[target_vector == 0.0]))


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


def test_binary_posterior_converges_after_single_latent_gaussian_update(random_generator, monkeypatch):
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
        sample_space_preconditioner_rank=256,
        warm_start=None,
        **_kwargs,
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

    assert restricted_call_count == 2
    assert alpha.shape == (1,)
    assert beta.shape == (variant_count,)
    assert beta_variance.shape == (variant_count,)
    assert linear_predictor.shape == (sample_count,)
    assert np.isfinite(objective)


def test_cpu_sample_space_solver_uses_single_streaming_operator_pass(monkeypatch):
    genotype_matrix = np.array(
        [
            [0.0, 1.0, 2.0, 0.0],
            [1.0, 0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 1.0],
            [1.0, 2.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 2.0],
            [2.0, 0.0, 2.0, 1.0],
        ],
        dtype=np.float32,
    )
    standardized = as_raw_genotype_matrix(genotype_matrix).standardized(
        means=np.mean(genotype_matrix, axis=0, dtype=np.float32),
        scales=np.std(genotype_matrix, axis=0, dtype=np.float32) + 1e-3,
    )
    dense_standardized = np.asarray(standardized.materialize(), dtype=np.float64)
    prior_variances = np.array([0.6, 1.2, 0.8, 1.1], dtype=np.float64)
    diagonal_noise = np.full(genotype_matrix.shape[0], 0.7, dtype=np.float64)
    right_hand_side = np.array(
        [
            [1.0, 0.5],
            [0.2, -0.1],
            [-0.4, 0.7],
            [0.6, 0.3],
            [-0.2, 0.4],
            [0.1, -0.5],
        ],
        dtype=np.float64,
    )
    exact_operator = (
        np.diag(diagonal_noise)
        + dense_standardized @ np.diag(prior_variances) @ dense_standardized.T
    )
    expected_solution = np.linalg.solve(exact_operator, right_hand_side)

    def _forbid_two_pass(*args, **kwargs):
        raise AssertionError("CPU sample-space solve regressed to the two-pass matmat path.")

    monkeypatch.setattr(type(standardized), "transpose_matmat", _forbid_two_pass)
    monkeypatch.setattr(type(standardized), "matmat", _forbid_two_pass)

    solved = _solve_sample_space_rhs_cpu(
        genotype_matrix=standardized,
        prior_variances=prior_variances,
        diagonal_noise=diagonal_noise,
        right_hand_side=right_hand_side,
        initial_guess=None,
        tolerance=1e-10,
        max_iterations=64,
        preconditioner=np.diag(exact_operator).astype(np.float64),
        batch_size=2,
    )

    np.testing.assert_allclose(solved, expected_solution, atol=1e-7, rtol=1e-7)


def test_restricted_posterior_sample_space_merges_probe_rhs(monkeypatch: pytest.MonkeyPatch):
    sample_count, variant_count = 6, 7
    genotype_values = np.array(
        [
            [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0],
            [1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0],
            [1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0],
            [0.0, 1.0, 1.0, 2.0, 1.0, 0.0, 2.0],
            [2.0, 0.0, 2.0, 1.0, 0.0, 2.0, 1.0],
        ],
        dtype=np.float32,
    )
    standardized = as_raw_genotype_matrix(genotype_values).standardized(
        means=np.mean(genotype_values, axis=0, dtype=np.float32),
        scales=np.std(genotype_values, axis=0, dtype=np.float32) + 1e-3,
    )
    covariate_matrix = np.column_stack(
        [
            np.ones(sample_count, dtype=np.float64),
            np.linspace(-1.0, 1.0, sample_count, dtype=np.float64),
        ]
    )
    targets = np.linspace(-0.5, 0.75, sample_count, dtype=np.float64)
    prior_variances = np.linspace(0.5, 1.1, variant_count, dtype=np.float64)
    diagonal_noise = np.linspace(0.8, 1.3, sample_count, dtype=np.float64)
    solve_rhs_shapes: list[tuple[int, int]] = []

    monkeypatch.setattr(
        mixture_inference,
        "_sample_space_preconditioner",
        lambda **kwargs: np.ones(sample_count, dtype=np.float64),
    )
    monkeypatch.setattr(mixture_inference, "stochastic_logdet", lambda *args, **kwargs: 0.0)

    def fake_solve_sample_space_rhs_cpu(
        genotype_matrix,
        prior_variances,
        diagonal_noise,
        right_hand_side,
        initial_guess,
        tolerance,
        max_iterations,
        preconditioner,
        batch_size,
        return_iterations=False,
    ):
        rhs = np.asarray(right_hand_side, dtype=np.float64)
        if rhs.ndim == 1:
            rhs = rhs[:, None]
        solve_rhs_shapes.append(rhs.shape)
        return (rhs, 3) if return_iterations else rhs

    monkeypatch.setattr(mixture_inference, "_solve_sample_space_rhs_cpu", fake_solve_sample_space_rhs_cpu)

    alpha, beta, beta_variance, projected_targets, linear_predictor, *_ = mixture_inference._restricted_posterior_state(
        genotype_matrix=standardized,
        covariate_matrix=covariate_matrix,
        targets=targets,
        prior_variances=prior_variances,
        diagonal_noise=diagonal_noise,
        solver_tolerance=1e-6,
        maximum_linear_solver_iterations=16,
        logdet_probe_count=2,
        logdet_lanczos_steps=4,
        exact_solver_matrix_limit=2,
        posterior_variance_batch_size=3,
        posterior_variance_probe_count=5,
        random_seed=7,
        compute_logdet=False,
        compute_beta_variance=True,
        sample_space_preconditioner_rank=0,
    )

    assert solve_rhs_shapes == [(sample_count, 1 + covariate_matrix.shape[1] + 5)]
    assert alpha.shape == (covariate_matrix.shape[1],)
    assert beta.shape == (variant_count,)
    assert beta_variance.shape == (variant_count,)
    assert projected_targets.shape == (sample_count,)
    assert linear_predictor.shape == (sample_count,)


def test_restricted_posterior_sample_space_reuses_matching_warm_start(monkeypatch: pytest.MonkeyPatch):
    sample_count, variant_count = 6, 7
    genotype_values = np.array(
        [
            [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0],
            [1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0],
            [1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0],
            [0.0, 1.0, 1.0, 2.0, 1.0, 0.0, 2.0],
            [2.0, 0.0, 2.0, 1.0, 0.0, 2.0, 1.0],
        ],
        dtype=np.float32,
    )
    standardized = as_raw_genotype_matrix(genotype_values).standardized(
        means=np.mean(genotype_values, axis=0, dtype=np.float32),
        scales=np.std(genotype_values, axis=0, dtype=np.float32) + 1e-3,
    )
    covariate_matrix = np.column_stack(
        [
            np.ones(sample_count, dtype=np.float64),
            np.linspace(-1.0, 1.0, sample_count, dtype=np.float64),
        ]
    )
    targets = np.linspace(-0.5, 0.75, sample_count, dtype=np.float64)
    prior_variances = np.linspace(0.5, 1.1, variant_count, dtype=np.float64)
    diagonal_noise = np.linspace(0.8, 1.3, sample_count, dtype=np.float64)
    initial_guess_shapes: list[tuple[int, int] | None] = []
    warm_start = mixture_inference._RestrictedPosteriorWarmStart(
        sample_space_inverse_covariance_rhs=np.ones((sample_count, 1 + covariate_matrix.shape[1] + 5), dtype=np.float64)
    )

    monkeypatch.setattr(
        mixture_inference,
        "_sample_space_preconditioner",
        lambda **kwargs: np.ones(sample_count, dtype=np.float64),
    )
    monkeypatch.setattr(mixture_inference, "stochastic_logdet", lambda *args, **kwargs: 0.0)

    def fake_solve_sample_space_rhs_cpu(
        genotype_matrix,
        prior_variances,
        diagonal_noise,
        right_hand_side,
        initial_guess,
        tolerance,
        max_iterations,
        preconditioner,
        batch_size,
        return_iterations=False,
    ):
        rhs = np.asarray(right_hand_side, dtype=np.float64)
        if rhs.ndim == 1:
            rhs = rhs[:, None]
        initial_guess_shapes.append(None if initial_guess is None else tuple(np.asarray(initial_guess).shape))
        return (rhs, 3) if return_iterations else rhs

    monkeypatch.setattr(mixture_inference, "_solve_sample_space_rhs_cpu", fake_solve_sample_space_rhs_cpu)

    mixture_inference._restricted_posterior_state(
        genotype_matrix=standardized,
        covariate_matrix=covariate_matrix,
        targets=targets,
        prior_variances=prior_variances,
        diagonal_noise=diagonal_noise,
        solver_tolerance=1e-6,
        maximum_linear_solver_iterations=16,
        logdet_probe_count=2,
        logdet_lanczos_steps=4,
        exact_solver_matrix_limit=2,
        posterior_variance_batch_size=3,
        posterior_variance_probe_count=5,
        random_seed=7,
        compute_logdet=False,
        compute_beta_variance=True,
        sample_space_preconditioner_rank=0,
        warm_start=warm_start,
    )

    assert initial_guess_shapes == [(sample_count, 1 + covariate_matrix.shape[1] + 5)]


def test_stochastic_restricted_cross_leverage_diagonal_caches_fixed_probe_projection(
    monkeypatch: pytest.MonkeyPatch,
):
    sample_count = 6
    genotype_values = np.array(
        [
            [0.0, 1.0, 2.0, 0.0],
            [1.0, 0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 1.0],
            [1.0, 2.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 2.0],
            [2.0, 0.0, 2.0, 1.0],
        ],
        dtype=np.float32,
    )
    standardized = as_raw_genotype_matrix(genotype_values).standardized(
        means=np.mean(genotype_values, axis=0, dtype=np.float32),
        scales=np.std(genotype_values, axis=0, dtype=np.float32) + 1e-3,
    )
    covariate_matrix = np.ones((sample_count, 1), dtype=np.float64)
    sample_probes = _orthogonal_probe_matrix(
        dimension=sample_count,
        probe_count=3,
        random_seed=11,
    )
    inverse_covariance_probe_matrix = sample_probes * 0.5
    inverse_covariance_covariates = np.ones((sample_count, 1), dtype=np.float64) * 0.1
    gls_cholesky = np.linalg.cholesky(covariate_matrix.T @ covariate_matrix + np.eye(1, dtype=np.float64) * 1e-8)

    transpose_probe_counts = {"fixed": 0, "restricted": 0}

    class _CountingGenotypeMatrix:
        def __init__(self, base_matrix):
            self._base_matrix = base_matrix

        def __getattr__(self, name):
            return getattr(self._base_matrix, name)

        @property
        def shape(self):
            return self._base_matrix.shape

        def transpose_matmat(self, matrix, batch_size):
            matrix_array = np.asarray(matrix, dtype=np.float64)
            if np.allclose(matrix_array, sample_probes):
                transpose_probe_counts["fixed"] += 1
            else:
                transpose_probe_counts["restricted"] += 1
            return self._base_matrix.transpose_matmat(matrix, batch_size=batch_size)

    counting_standardized = _CountingGenotypeMatrix(standardized)

    expected = mixture_inference._stochastic_restricted_cross_leverage_diagonal(
        genotype_matrix=counting_standardized,
        covariate_matrix=covariate_matrix,
        solve_rhs=lambda rhs: rhs,
        inverse_covariance_covariates=inverse_covariance_covariates,
        gls_cholesky=gls_cholesky,
        batch_size=2,
        probe_count=sample_probes.shape[1],
        random_seed=11,
        sample_probes=sample_probes,
        inverse_covariance_probe_matrix=inverse_covariance_probe_matrix,
    )
    actual = mixture_inference._stochastic_restricted_cross_leverage_diagonal(
        genotype_matrix=counting_standardized,
        covariate_matrix=covariate_matrix,
        solve_rhs=lambda rhs: rhs,
        inverse_covariance_covariates=inverse_covariance_covariates,
        gls_cholesky=gls_cholesky,
        batch_size=2,
        probe_count=sample_probes.shape[1],
        random_seed=11,
        sample_probes=sample_probes,
        inverse_covariance_probe_matrix=inverse_covariance_probe_matrix,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)
    assert transpose_probe_counts["fixed"] == 1
    assert transpose_probe_counts["restricted"] == 2


def test_restricted_posterior_sample_space_reuses_preconditioner_until_iterations_blow_up(
    monkeypatch: pytest.MonkeyPatch,
):
    sample_count, variant_count = 6, 7
    genotype_values = np.array(
        [
            [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0],
            [1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0],
            [1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0],
            [0.0, 1.0, 1.0, 2.0, 1.0, 0.0, 2.0],
            [2.0, 0.0, 2.0, 1.0, 0.0, 2.0, 1.0],
        ],
        dtype=np.float32,
    )
    standardized = as_raw_genotype_matrix(genotype_values).standardized(
        means=np.mean(genotype_values, axis=0, dtype=np.float32),
        scales=np.std(genotype_values, axis=0, dtype=np.float32) + 1e-3,
    )
    covariate_matrix = np.column_stack(
        [
            np.ones(sample_count, dtype=np.float64),
            np.linspace(-1.0, 1.0, sample_count, dtype=np.float64),
        ]
    )
    targets = np.linspace(-0.5, 0.75, sample_count, dtype=np.float64)
    prior_variances = np.linspace(0.5, 1.1, variant_count, dtype=np.float64)
    diagonal_noise = np.linspace(0.8, 1.3, sample_count, dtype=np.float64)
    built_preconditioners: list[object] = []
    iteration_sequence = iter([4, 20, 5])

    def fake_sample_space_preconditioner(**kwargs):
        preconditioner = object()
        built_preconditioners.append(preconditioner)
        return preconditioner

    def fake_solve_sample_space_rhs_cpu(
        genotype_matrix,
        prior_variances,
        diagonal_noise,
        right_hand_side,
        initial_guess,
        tolerance,
        max_iterations,
        preconditioner,
        batch_size,
        return_iterations=False,
    ):
        rhs = np.asarray(right_hand_side, dtype=np.float64)
        if rhs.ndim == 1:
            rhs = rhs[:, None]
        iterations_used = next(iteration_sequence)
        return (rhs, iterations_used) if return_iterations else rhs

    monkeypatch.setattr(mixture_inference, "_sample_space_preconditioner", fake_sample_space_preconditioner)
    monkeypatch.setattr(mixture_inference, "_solve_sample_space_rhs_cpu", fake_solve_sample_space_rhs_cpu)

    for _ in range(3):
        mixture_inference._restricted_posterior_state(
            genotype_matrix=standardized,
            covariate_matrix=covariate_matrix,
            targets=targets,
            prior_variances=prior_variances,
            diagonal_noise=diagonal_noise,
            solver_tolerance=1e-6,
            maximum_linear_solver_iterations=32,
            logdet_probe_count=2,
            logdet_lanczos_steps=4,
            exact_solver_matrix_limit=2,
            posterior_variance_batch_size=3,
            posterior_variance_probe_count=5,
            random_seed=7,
            compute_logdet=False,
            compute_beta_variance=False,
            sample_space_preconditioner_rank=0,
        )

    assert len(built_preconditioners) == 2


def test_restricted_posterior_sample_space_fuses_beta_and_restricted_probe_transpose_passes(
    monkeypatch: pytest.MonkeyPatch,
):
    sample_count, variant_count = 6, 7
    genotype_values = np.array(
        [
            [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0],
            [1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0],
            [1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0],
            [0.0, 1.0, 1.0, 2.0, 1.0, 0.0, 2.0],
            [2.0, 0.0, 2.0, 1.0, 0.0, 2.0, 1.0],
        ],
        dtype=np.float32,
    )
    standardized = as_raw_genotype_matrix(genotype_values).standardized(
        means=np.mean(genotype_values, axis=0, dtype=np.float32),
        scales=np.std(genotype_values, axis=0, dtype=np.float32) + 1e-3,
    )
    covariate_matrix = np.column_stack(
        [
            np.ones(sample_count, dtype=np.float64),
            np.linspace(-1.0, 1.0, sample_count, dtype=np.float64),
        ]
    )
    targets = np.linspace(-0.5, 0.75, sample_count, dtype=np.float64)
    prior_variances = np.linspace(0.5, 1.1, variant_count, dtype=np.float64)
    diagonal_noise = np.linspace(0.8, 1.3, sample_count, dtype=np.float64)
    transpose_matvec_calls = 0
    transpose_matmat_widths: list[int] = []

    class _CountingGenotypeMatrix:
        def __init__(self, base_matrix):
            self._base_matrix = base_matrix

        def __getattr__(self, name):
            return getattr(self._base_matrix, name)

        @property
        def shape(self):
            return self._base_matrix.shape

        def transpose_matvec(self, vector, batch_size=None):
            nonlocal transpose_matvec_calls
            transpose_matvec_calls += 1
            return self._base_matrix.transpose_matvec(vector, batch_size=batch_size)

        def transpose_matmat(self, matrix, batch_size):
            transpose_matmat_widths.append(np.asarray(matrix).shape[1])
            return self._base_matrix.transpose_matmat(matrix, batch_size=batch_size)

        def matvec(self, vector, batch_size=None):
            return self._base_matrix.matvec(vector, batch_size=batch_size)

        def iter_column_batches(self, batch_size):
            return self._base_matrix.iter_column_batches(batch_size=batch_size)

    counting_standardized = _CountingGenotypeMatrix(standardized)

    def fake_solve_sample_space_rhs_cpu(
        genotype_matrix,
        prior_variances,
        diagonal_noise,
        right_hand_side,
        initial_guess,
        tolerance,
        max_iterations,
        preconditioner,
        batch_size,
        return_iterations=False,
    ):
        rhs = np.asarray(right_hand_side, dtype=np.float64)
        if rhs.ndim == 1:
            rhs = rhs[:, None]
        return (rhs, 2) if return_iterations else rhs

    monkeypatch.setattr(
        mixture_inference,
        "_sample_space_preconditioner",
        lambda **kwargs: np.ones(sample_count, dtype=np.float64),
    )
    monkeypatch.setattr(mixture_inference, "_solve_sample_space_rhs_cpu", fake_solve_sample_space_rhs_cpu)
    monkeypatch.setattr(mixture_inference, "stochastic_logdet", lambda *args, **kwargs: 0.0)

    alpha, beta, beta_variance, projected_targets, linear_predictor, *_ = mixture_inference._restricted_posterior_state(
        genotype_matrix=counting_standardized,
        covariate_matrix=covariate_matrix,
        targets=targets,
        prior_variances=prior_variances,
        diagonal_noise=diagonal_noise,
        solver_tolerance=1e-6,
        maximum_linear_solver_iterations=16,
        logdet_probe_count=2,
        logdet_lanczos_steps=4,
        exact_solver_matrix_limit=2,
        posterior_variance_batch_size=3,
        posterior_variance_probe_count=5,
        random_seed=7,
        compute_logdet=False,
        compute_beta_variance=True,
        sample_space_preconditioner_rank=0,
        allow_working_set=False,
    )

    assert alpha.shape == (covariate_matrix.shape[1],)
    assert beta.shape == (variant_count,)
    assert beta_variance.shape == (variant_count,)
    assert projected_targets.shape == (sample_count,)
    assert linear_predictor.shape == (sample_count,)
    assert transpose_matvec_calls == 0
    assert sorted(transpose_matmat_widths) == [5, 6]


def test_restricted_posterior_sample_space_logdet_uses_cached_diagonal_control_variate(
    monkeypatch: pytest.MonkeyPatch,
):
    sample_count, variant_count = 6, 7
    genotype_values = np.array(
        [
            [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0],
            [1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0],
            [1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0],
            [0.0, 1.0, 1.0, 2.0, 1.0, 0.0, 2.0],
            [2.0, 0.0, 2.0, 1.0, 0.0, 2.0, 1.0],
        ],
        dtype=np.float32,
    )
    standardized = as_raw_genotype_matrix(genotype_values).standardized(
        means=np.mean(genotype_values, axis=0, dtype=np.float32),
        scales=np.std(genotype_values, axis=0, dtype=np.float32) + 1e-3,
    )
    covariate_matrix = np.column_stack(
        [
            np.ones(sample_count, dtype=np.float64),
            np.linspace(-1.0, 1.0, sample_count, dtype=np.float64),
        ]
    )
    targets = np.linspace(-0.5, 0.75, sample_count, dtype=np.float64)
    prior_variances = np.linspace(0.5, 1.1, variant_count, dtype=np.float64)
    diagonal_noise = np.linspace(0.8, 1.3, sample_count, dtype=np.float64)
    expected_diagonal = np.linspace(2.0, 3.0, sample_count, dtype=np.float64)
    captured_control_variate: list[np.ndarray] = []

    def fake_diagonal_preconditioner(**kwargs):
        return expected_diagonal

    def fake_solve_sample_space_rhs_cpu(
        genotype_matrix,
        prior_variances,
        diagonal_noise,
        right_hand_side,
        initial_guess,
        tolerance,
        max_iterations,
        preconditioner,
        batch_size,
        return_iterations=False,
    ):
        rhs = np.asarray(right_hand_side, dtype=np.float64)
        if rhs.ndim == 1:
            rhs = rhs[:, None]
        return (rhs, 2) if return_iterations else rhs

    def fake_stochastic_logdet(operator, dimension, probe_count, lanczos_steps, random_seed, **kwargs):
        captured_control_variate.append(np.asarray(kwargs["control_variate_diagonal"], dtype=np.float64))
        return 0.0

    monkeypatch.setattr(mixture_inference, "_sample_space_diagonal_preconditioner", fake_diagonal_preconditioner)
    monkeypatch.setattr(mixture_inference, "_solve_sample_space_rhs_cpu", fake_solve_sample_space_rhs_cpu)
    monkeypatch.setattr(mixture_inference, "stochastic_logdet", fake_stochastic_logdet)

    mixture_inference._restricted_posterior_state(
        genotype_matrix=standardized,
        covariate_matrix=covariate_matrix,
        targets=targets,
        prior_variances=prior_variances,
        diagonal_noise=diagonal_noise,
        solver_tolerance=1e-6,
        maximum_linear_solver_iterations=16,
        logdet_probe_count=8,
        logdet_lanczos_steps=4,
        exact_solver_matrix_limit=2,
        posterior_variance_batch_size=3,
        posterior_variance_probe_count=5,
        random_seed=7,
        compute_logdet=True,
        compute_beta_variance=False,
        sample_space_preconditioner_rank=0,
        allow_working_set=False,
    )

    assert len(captured_control_variate) == 1
    np.testing.assert_allclose(captured_control_variate[0], expected_diagonal, atol=0.0)


def test_binary_posterior_stops_when_predictor_update_is_stalled(random_generator, monkeypatch):
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
        sample_space_preconditioner_rank=256,
        warm_start=None,
        **_kwargs,
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

    assert restricted_call_count == 2
    assert alpha.shape == (1,)
    assert beta.shape == (variant_count,)
    assert beta_variance.shape == (variant_count,)
    assert linear_predictor.shape == (sample_count,)
    assert np.isfinite(objective)


def test_binary_newton_solver_controls_relax_large_problem_settings():
    standardized = as_raw_genotype_matrix(np.zeros((1, 1), dtype=np.float32)).standardized(
        means=np.zeros(1, dtype=np.float32),
        scales=np.ones(1, dtype=np.float32),
    )
    standardized._n_samples = 20_000
    standardized.variant_indices = np.arange(40_000, dtype=np.int32)
    standardized.means = np.zeros(40_000, dtype=np.float32)
    standardized.scales = np.ones(40_000, dtype=np.float32)

    tolerance, max_iterations = _binary_newton_solver_controls(
        standardized,
        solver_tolerance=1e-6,
        maximum_linear_solver_iterations=1024,
    )

    assert tolerance == 5e-3
    assert max_iterations == 96


def test_binary_posterior_uses_inexact_solver_controls_before_final_solve(monkeypatch):
    sample_count, variant_count = 8, 6
    genotype_values = np.array(
        [
            [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0, 2.0, 0.0, 1.0],
            [2.0, 1.0, 0.0, 1.0, 2.0, 0.0],
            [0.0, 2.0, 1.0, 0.0, 2.0, 1.0],
            [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            [2.0, 0.0, 2.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0, 1.0, 2.0],
            [1.0, 2.0, 1.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    standardized = as_raw_genotype_matrix(genotype_values).standardized(
        means=np.ones(variant_count, dtype=np.float32),
        scales=np.ones(variant_count, dtype=np.float32),
    )
    standardized._dense_cache = standardized.materialize()
    covariate_matrix = np.column_stack(
        [
            np.ones(sample_count, dtype=np.float32),
            np.linspace(-1.0, 1.0, sample_count, dtype=np.float32),
        ]
    )
    targets = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    solver_settings: list[tuple[float, int, bool, bool]] = []

    monkeypatch.setattr(
        mixture_inference,
        "_binary_newton_solver_controls",
        lambda *args, **kwargs: (7e-4, 33),
    )

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
        sample_space_preconditioner_rank=256,
        warm_start=None,
        **_kwargs,
    ):
        solver_settings.append(
            (
                float(solver_tolerance),
                int(maximum_linear_solver_iterations),
                bool(compute_logdet),
                bool(compute_beta_variance),
            )
        )
        beta = np.zeros(prior_variances.shape[0], dtype=np.float64)
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

    _binary_posterior_state(
        genotype_matrix=standardized,
        covariate_matrix=covariate_matrix,
        targets=targets,
        prior_variances=np.ones(variant_count, dtype=np.float32),
        alpha_init=np.zeros(covariate_matrix.shape[1], dtype=np.float32),
        beta_init=np.zeros(variant_count, dtype=np.float32),
        minimum_weight=1e-4,
        max_iterations=1,
        gradient_tolerance=1e-8,
        initial_damping=1.0,
        damping_increase_factor=10.0,
        damping_decrease_factor=0.1,
        success_threshold=0.25,
        minimum_damping=1e-8,
        solver_tolerance=1e-6,
        maximum_linear_solver_iterations=1024,
    )

    assert solver_settings[0] == (7e-4, 33, False, False)
    assert solver_settings[1] == (1e-6, 1024, True, True)


def test_sample_space_preconditioner_matches_exact_covariance_inverse_at_full_rank():
    genotype_matrix = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    standardized = as_raw_genotype_matrix(genotype_matrix).standardized(
        means=np.zeros(genotype_matrix.shape[1], dtype=np.float32),
        scales=np.ones(genotype_matrix.shape[1], dtype=np.float32),
    )
    standardized._dense_cache = standardized.materialize()
    prior_variances = np.array([2.0, 0.5], dtype=np.float64)
    diagonal_noise = np.array([1.5, 1.0, 2.0], dtype=np.float64)
    right_hand_side = np.array([0.5, -1.0, 2.0], dtype=np.float64)

    apply_preconditioner = _sample_space_preconditioner(
        genotype_matrix=standardized,
        prior_variances=prior_variances,
        diagonal_noise=diagonal_noise,
        batch_size=2,
        rank=genotype_matrix.shape[1],
    )

    covariance_matrix = np.diag(diagonal_noise) + genotype_matrix @ np.diag(prior_variances) @ genotype_matrix.T
    expected = np.linalg.solve(covariance_matrix, right_hand_side)
    actual = np.asarray(apply_preconditioner(right_hand_side), dtype=np.float64)

    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)


def test_sample_space_preconditioner_gpu_path_matches_exact_covariance_inverse_at_full_rank(monkeypatch: pytest.MonkeyPatch):
    genotype_matrix = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    standardized = as_raw_genotype_matrix(genotype_matrix).standardized(
        means=np.zeros(genotype_matrix.shape[1], dtype=np.float32),
        scales=np.ones(genotype_matrix.shape[1], dtype=np.float32),
    )
    standardized._cupy_cache = standardized.materialize().astype(np.float32, copy=False)
    standardized._dense_cache = None
    prior_variances = np.array([2.0, 0.5], dtype=np.float64)
    diagonal_noise = np.array([1.5, 1.0, 2.0], dtype=np.float64)
    right_hand_side = np.array([0.5, -1.0, 2.0], dtype=np.float64)

    fake_cupy: Any = types.ModuleType("cupy")
    fake_cupy.float32 = np.float32
    fake_cupy.float64 = np.float64
    fake_cupy.asarray = lambda array, dtype=None: np.asarray(array, dtype=dtype)
    fake_cupy.sum = np.sum
    fake_cupy.sqrt = np.sqrt
    fake_cupy.abs = np.abs
    fake_cupy.diag = np.diag
    fake_cupy.maximum = np.maximum
    fake_cupy.eye = np.eye
    fake_cupy.linalg = types.SimpleNamespace(cholesky=np.linalg.cholesky, qr=np.linalg.qr)
    fake_cupyx: Any = types.ModuleType("cupyx")
    fake_cupyx_scipy: Any = types.ModuleType("cupyx.scipy")
    fake_cupyx_scipy_linalg: Any = types.ModuleType("cupyx.scipy.linalg")
    fake_cupyx_scipy_linalg.solve_triangular = scipy_solve_triangular
    fake_cupyx_scipy.linalg = fake_cupyx_scipy_linalg

    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    monkeypatch.setitem(sys.modules, "cupyx", fake_cupyx)
    monkeypatch.setitem(sys.modules, "cupyx.scipy", fake_cupyx_scipy)
    monkeypatch.setitem(sys.modules, "cupyx.scipy.linalg", fake_cupyx_scipy_linalg)
    monkeypatch.setattr(
        mixture_inference,
        "_sample_space_diagonal_preconditioner",
        lambda **kwargs: np.diag(
            np.diag(diagonal_noise) + genotype_matrix @ np.diag(prior_variances) @ genotype_matrix.T
        ).astype(np.float64, copy=False),
    )

    apply_preconditioner = _sample_space_preconditioner(
        genotype_matrix=standardized,
        prior_variances=prior_variances,
        diagonal_noise=diagonal_noise,
        batch_size=2,
        rank=genotype_matrix.shape[1],
    )

    covariance_matrix = np.diag(diagonal_noise) + genotype_matrix @ np.diag(prior_variances) @ genotype_matrix.T
    expected = np.linalg.solve(covariance_matrix, right_hand_side)
    actual = np.asarray(cast(Any, apply_preconditioner)(right_hand_side), dtype=np.float64)

    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=2e-5)


def test_sample_space_preconditioner_uses_operator_sketch_without_subset_materialization(monkeypatch: pytest.MonkeyPatch):
    genotype_matrix = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    standardized = as_raw_genotype_matrix(genotype_matrix).standardized(
        means=np.zeros(genotype_matrix.shape[1], dtype=np.float32),
        scales=np.ones(genotype_matrix.shape[1], dtype=np.float32),
    )
    prior_variances = np.array([2.0, 0.5], dtype=np.float64)
    diagonal_noise = np.array([1.5, 1.0, 2.0], dtype=np.float64)
    right_hand_side = np.array([0.5, -1.0, 2.0], dtype=np.float64)
    fake_cupy: Any = types.ModuleType("cupy")
    fake_cupy.float32 = np.float32
    fake_cupy.float64 = np.float64
    fake_cupy.asarray = lambda array, dtype=None: np.asarray(array, dtype=dtype)
    fake_cupy.sum = np.sum
    fake_cupy.sqrt = np.sqrt
    fake_cupy.abs = np.abs
    fake_cupy.diag = np.diag
    fake_cupy.maximum = np.maximum
    fake_cupy.eye = np.eye
    fake_cupy.linalg = types.SimpleNamespace(cholesky=np.linalg.cholesky, qr=np.linalg.qr)
    fake_cupyx: Any = types.ModuleType("cupyx")
    fake_cupyx_scipy: Any = types.ModuleType("cupyx.scipy")
    fake_cupyx_scipy_linalg: Any = types.ModuleType("cupyx.scipy.linalg")
    fake_cupyx_scipy_linalg.solve_triangular = scipy_solve_triangular
    fake_cupyx_scipy.linalg = fake_cupyx_scipy_linalg

    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    monkeypatch.setitem(sys.modules, "cupyx", fake_cupyx)
    monkeypatch.setitem(sys.modules, "cupyx.scipy", fake_cupyx_scipy)
    monkeypatch.setitem(sys.modules, "cupyx.scipy.linalg", fake_cupyx_scipy_linalg)
    monkeypatch.setattr(
        type(standardized),
        "try_materialize_gpu_subset",
        lambda self, indices: (_ for _ in ()).throw(AssertionError("subset materialization should not be used")),
    )
    monkeypatch.setattr(
        mixture_inference,
        "_sample_space_diagonal_preconditioner",
        lambda **kwargs: np.diag(
            np.diag(diagonal_noise) + genotype_matrix @ np.diag(prior_variances) @ genotype_matrix.T
        ).astype(np.float64, copy=False),
    )

    apply_preconditioner = _sample_space_preconditioner(
        genotype_matrix=standardized,
        prior_variances=prior_variances,
        diagonal_noise=diagonal_noise,
        batch_size=2,
        rank=genotype_matrix.shape[1],
    )

    covariance_matrix = np.diag(diagonal_noise) + genotype_matrix @ np.diag(prior_variances) @ genotype_matrix.T
    expected = np.linalg.solve(covariance_matrix, right_hand_side)
    actual = np.asarray(cast(Any, apply_preconditioner)(right_hand_side), dtype=np.float64)

    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=2e-5)


def test_sample_space_preconditioner_handles_semidefinite_sketch_exactly():
    genotype_matrix = np.array(
        [
            [1.0, 1.0],
            [0.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    standardized = as_raw_genotype_matrix(genotype_matrix).standardized(
        means=np.zeros(genotype_matrix.shape[1], dtype=np.float32),
        scales=np.ones(genotype_matrix.shape[1], dtype=np.float32),
    )
    standardized._dense_cache = standardized.materialize()
    prior_variances = np.array([1.5, 0.5], dtype=np.float64)
    diagonal_noise = np.array([1.0, 1.25, 0.8], dtype=np.float64)
    right_hand_side = np.array([0.5, -1.0, 1.25], dtype=np.float64)

    apply_preconditioner = _sample_space_preconditioner(
        genotype_matrix=standardized,
        prior_variances=prior_variances,
        diagonal_noise=diagonal_noise,
        batch_size=2,
        rank=genotype_matrix.shape[1],
    )

    covariance_matrix = np.diag(diagonal_noise) + genotype_matrix @ np.diag(prior_variances) @ genotype_matrix.T
    expected = np.linalg.solve(covariance_matrix, right_hand_side)
    actual = np.asarray(apply_preconditioner(right_hand_side), dtype=np.float64)

    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)


def test_sample_space_preconditioner_reuses_cached_genotype_sketch(monkeypatch: pytest.MonkeyPatch):
    genotype_matrix = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    standardized = as_raw_genotype_matrix(genotype_matrix).standardized(
        means=np.zeros(genotype_matrix.shape[1], dtype=np.float32),
        scales=np.ones(genotype_matrix.shape[1], dtype=np.float32),
    )
    standardized._dense_cache = standardized.materialize()
    gram_call_count = 0
    weighted_kernel_call_count = 0
    original_gram = mixture_inference._sample_space_genotype_gram_matmat_cpu
    original_weighted_kernel = mixture_inference._sample_space_kernel_matmat_cpu

    def counted_gram(*args, **kwargs):
        nonlocal gram_call_count
        gram_call_count += 1
        return original_gram(*args, **kwargs)

    def counted_weighted_kernel(*args, **kwargs):
        nonlocal weighted_kernel_call_count
        weighted_kernel_call_count += 1
        return original_weighted_kernel(*args, **kwargs)

    monkeypatch.setattr(mixture_inference, "_sample_space_genotype_gram_matmat_cpu", counted_gram)
    monkeypatch.setattr(mixture_inference, "_sample_space_kernel_matmat_cpu", counted_weighted_kernel)

    _sample_space_preconditioner(
        genotype_matrix=standardized,
        prior_variances=np.array([1.0, 0.5, 0.75], dtype=np.float64),
        diagonal_noise=np.ones(genotype_matrix.shape[0], dtype=np.float64),
        batch_size=2,
        rank=2,
        random_seed=11,
    )
    _sample_space_preconditioner(
        genotype_matrix=standardized,
        prior_variances=np.array([0.75, 1.25, 0.4], dtype=np.float64),
        diagonal_noise=np.full(genotype_matrix.shape[0], 1.5, dtype=np.float64),
        batch_size=2,
        rank=2,
        random_seed=11,
    )

    assert gram_call_count == 1
    assert weighted_kernel_call_count == 2


def test_sample_space_preconditioner_gpu_path_handles_semidefinite_sketch_exactly(monkeypatch: pytest.MonkeyPatch):
    genotype_matrix = np.array(
        [
            [1.0, 1.0],
            [0.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    standardized = as_raw_genotype_matrix(genotype_matrix).standardized(
        means=np.zeros(genotype_matrix.shape[1], dtype=np.float32),
        scales=np.ones(genotype_matrix.shape[1], dtype=np.float32),
    )
    standardized._cupy_cache = standardized.materialize().astype(np.float32, copy=False)
    standardized._dense_cache = None
    prior_variances = np.array([1.5, 0.5], dtype=np.float64)
    diagonal_noise = np.array([1.0, 1.25, 0.8], dtype=np.float64)
    right_hand_side = np.array([0.5, -1.0, 1.25], dtype=np.float64)

    fake_cupy: Any = types.ModuleType("cupy")
    fake_cupy.float32 = np.float32
    fake_cupy.float64 = np.float64
    fake_cupy.asarray = lambda array, dtype=None: np.asarray(array, dtype=dtype)
    fake_cupy.sum = np.sum
    fake_cupy.sqrt = np.sqrt
    fake_cupy.abs = np.abs
    fake_cupy.diag = np.diag
    fake_cupy.maximum = np.maximum
    fake_cupy.eye = np.eye
    fake_cupy.linalg = types.SimpleNamespace(cholesky=np.linalg.cholesky, qr=np.linalg.qr)
    fake_cupyx: Any = types.ModuleType("cupyx")
    fake_cupyx_scipy: Any = types.ModuleType("cupyx.scipy")
    fake_cupyx_scipy_linalg: Any = types.ModuleType("cupyx.scipy.linalg")
    fake_cupyx_scipy_linalg.solve_triangular = scipy_solve_triangular
    fake_cupyx_scipy.linalg = fake_cupyx_scipy_linalg

    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    monkeypatch.setitem(sys.modules, "cupyx", fake_cupyx)
    monkeypatch.setitem(sys.modules, "cupyx.scipy", fake_cupyx_scipy)
    monkeypatch.setitem(sys.modules, "cupyx.scipy.linalg", fake_cupyx_scipy_linalg)
    monkeypatch.setattr(
        mixture_inference,
        "_sample_space_diagonal_preconditioner",
        lambda **kwargs: np.diag(
            np.diag(diagonal_noise) + genotype_matrix @ np.diag(prior_variances) @ genotype_matrix.T
        ).astype(np.float64, copy=False),
    )

    apply_preconditioner = _sample_space_preconditioner(
        genotype_matrix=standardized,
        prior_variances=prior_variances,
        diagonal_noise=diagonal_noise,
        batch_size=2,
        rank=genotype_matrix.shape[1],
    )

    covariance_matrix = np.diag(diagonal_noise) + genotype_matrix @ np.diag(prior_variances) @ genotype_matrix.T
    expected = np.linalg.solve(covariance_matrix, right_hand_side)
    actual = np.asarray(cast(Any, apply_preconditioner)(right_hand_side), dtype=np.float64)

    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)


def test_gpu_sample_space_block_cg_matches_dense_solution(monkeypatch: pytest.MonkeyPatch):
    genotype_matrix = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    standardized = as_raw_genotype_matrix(genotype_matrix).standardized(
        means=np.zeros(genotype_matrix.shape[1], dtype=np.float32),
        scales=np.ones(genotype_matrix.shape[1], dtype=np.float32),
    )
    standardized._cupy_cache = standardized.materialize().astype(np.float32, copy=False)
    standardized._dense_cache = None
    prior_variances = np.array([1.5, 0.75, 0.5], dtype=np.float64)
    diagonal_noise = np.array([1.0, 1.25, 0.8, 1.1], dtype=np.float64)
    right_hand_side = np.column_stack(
        [
            np.array([0.5, -1.0, 0.2, 1.5], dtype=np.float64),
            np.array([-0.25, 0.75, 1.0, -0.5], dtype=np.float64),
        ]
    )

    fake_cupy: Any = types.ModuleType("cupy")
    fake_cupy.float32 = np.float32
    fake_cupy.float64 = np.float64
    fake_cupy.asarray = lambda array, dtype=None: np.asarray(array, dtype=dtype)
    fake_cupy.sum = np.sum
    fake_cupy.sqrt = np.sqrt
    fake_cupy.abs = np.abs
    fake_cupy.diag = np.diag
    fake_cupy.maximum = np.maximum
    fake_cupy.eye = np.eye
    fake_cupy.zeros = np.zeros
    fake_cupy.linalg = types.SimpleNamespace(cholesky=np.linalg.cholesky, qr=np.linalg.qr)
    fake_cupyx: Any = types.ModuleType("cupyx")
    fake_cupyx_scipy: Any = types.ModuleType("cupyx.scipy")
    fake_cupyx_scipy_linalg: Any = types.ModuleType("cupyx.scipy.linalg")
    fake_cupyx_scipy_linalg.solve_triangular = scipy_solve_triangular
    fake_cupyx_scipy.linalg = fake_cupyx_scipy_linalg

    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    monkeypatch.setitem(sys.modules, "cupyx", fake_cupyx)
    monkeypatch.setitem(sys.modules, "cupyx.scipy", fake_cupyx_scipy)
    monkeypatch.setitem(sys.modules, "cupyx.scipy.linalg", fake_cupyx_scipy_linalg)
    monkeypatch.setattr(mixture_inference, "_try_import_cupy", lambda: fake_cupy)
    monkeypatch.setattr(
        mixture_inference,
        "_sample_space_diagonal_preconditioner",
        lambda **kwargs: np.diag(
            np.diag(diagonal_noise) + genotype_matrix @ np.diag(prior_variances) @ genotype_matrix.T
        ).astype(np.float64, copy=False),
    )

    preconditioner = mixture_inference._sample_space_gpu_preconditioner(
        genotype_matrix=standardized,
        prior_variances=prior_variances,
        diagonal_noise=diagonal_noise,
        batch_size=2,
        rank=genotype_matrix.shape[1],
    )
    actual = mixture_inference._solve_sample_space_rhs_gpu(
        genotype_matrix=standardized,
        prior_variances=prior_variances,
        diagonal_noise=diagonal_noise,
        right_hand_side=right_hand_side,
        initial_guess=None,
        tolerance=1e-7,
        max_iterations=64,
        preconditioner=preconditioner,
        batch_size=2,
    )

    covariance_matrix = np.diag(diagonal_noise) + genotype_matrix @ np.diag(prior_variances) @ genotype_matrix.T
    expected = np.linalg.solve(covariance_matrix, right_hand_side)

    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=2e-5)


def test_gpu_sample_space_block_cg_mixed_precision_refinement_matches_dense_solution(
    monkeypatch: pytest.MonkeyPatch,
):
    genotype_matrix = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    standardized = as_raw_genotype_matrix(genotype_matrix).standardized(
        means=np.zeros(genotype_matrix.shape[1], dtype=np.float32),
        scales=np.ones(genotype_matrix.shape[1], dtype=np.float32),
    )
    standardized._cupy_cache = standardized.materialize().astype(np.float32, copy=False)
    standardized._dense_cache = None
    prior_variances = np.array([1.5, 0.75, 0.5], dtype=np.float64)
    diagonal_noise = np.array([1.0, 1.25, 0.8, 1.1], dtype=np.float64)
    right_hand_side = np.column_stack(
        [
            np.array([0.5, -1.0, 0.2, 1.5], dtype=np.float64),
            np.array([-0.25, 0.75, 1.0, -0.5], dtype=np.float64),
        ]
    )

    fake_cupy: Any = types.ModuleType("cupy")
    fake_cupy.float32 = np.float32
    fake_cupy.float64 = np.float64
    fake_cupy.asarray = lambda array, dtype=None: np.asarray(array, dtype=dtype)
    fake_cupy.sum = np.sum
    fake_cupy.sqrt = np.sqrt
    fake_cupy.abs = np.abs
    fake_cupy.diag = np.diag
    fake_cupy.maximum = np.maximum
    fake_cupy.eye = np.eye
    fake_cupy.zeros = np.zeros
    fake_cupy.linalg = types.SimpleNamespace(cholesky=np.linalg.cholesky, qr=np.linalg.qr)
    fake_cupyx: Any = types.ModuleType("cupyx")
    fake_cupyx_scipy: Any = types.ModuleType("cupyx.scipy")
    fake_cupyx_scipy_linalg: Any = types.ModuleType("cupyx.scipy.linalg")
    fake_cupyx_scipy_linalg.solve_triangular = scipy_solve_triangular
    fake_cupyx_scipy.linalg = fake_cupyx_scipy_linalg

    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    monkeypatch.setitem(sys.modules, "cupyx", fake_cupyx)
    monkeypatch.setitem(sys.modules, "cupyx.scipy", fake_cupyx_scipy)
    monkeypatch.setitem(sys.modules, "cupyx.scipy.linalg", fake_cupyx_scipy_linalg)
    monkeypatch.setattr(mixture_inference, "_try_import_cupy", lambda: fake_cupy)
    monkeypatch.setattr(mixture_inference, "_cupy_compute_dtype", lambda cupy_module: cupy_module.float32)
    monkeypatch.setattr(
        mixture_inference,
        "_sample_space_diagonal_preconditioner",
        lambda **kwargs: np.diag(
            np.diag(diagonal_noise) + genotype_matrix @ np.diag(prior_variances) @ genotype_matrix.T
        ).astype(np.float64, copy=False),
    )

    preconditioner = mixture_inference._sample_space_gpu_preconditioner(
        genotype_matrix=standardized,
        prior_variances=prior_variances,
        diagonal_noise=diagonal_noise,
        batch_size=2,
        rank=genotype_matrix.shape[1],
    )
    actual = mixture_inference._solve_sample_space_rhs_gpu(
        genotype_matrix=standardized,
        prior_variances=prior_variances,
        diagonal_noise=diagonal_noise,
        right_hand_side=right_hand_side,
        initial_guess=None,
        tolerance=1e-7,
        max_iterations=64,
        preconditioner=preconditioner,
        batch_size=2,
    )

    covariance_matrix = np.diag(diagonal_noise) + genotype_matrix @ np.diag(prior_variances) @ genotype_matrix.T
    expected = np.linalg.solve(covariance_matrix, right_hand_side)

    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=5e-5)


def test_gpu_sample_space_solver_retries_in_float64_after_mixed_precision_stalls(
    monkeypatch: pytest.MonkeyPatch,
):
    genotype_matrix = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    standardized = as_raw_genotype_matrix(genotype_matrix).standardized(
        means=np.zeros(genotype_matrix.shape[1], dtype=np.float32),
        scales=np.ones(genotype_matrix.shape[1], dtype=np.float32),
    )
    standardized._cupy_cache = standardized.materialize().astype(np.float32, copy=False)
    standardized._dense_cache = None
    prior_variances = np.array([1.5, 0.75, 0.5], dtype=np.float64)
    diagonal_noise = np.array([1.0, 1.25, 0.8, 1.1], dtype=np.float64)
    right_hand_side = np.array([0.5, -1.0, 0.2, 1.5], dtype=np.float64)
    covariance_matrix = np.diag(diagonal_noise) + genotype_matrix @ np.diag(prior_variances) @ genotype_matrix.T
    expected = np.linalg.solve(covariance_matrix, right_hand_side)

    fake_cupy: Any = types.ModuleType("cupy")
    fake_cupy.float32 = np.float32
    fake_cupy.float64 = np.float64
    fake_cupy.asarray = lambda array, dtype=None: np.asarray(array, dtype=dtype)
    fake_cupy.sum = np.sum
    fake_cupy.zeros = np.zeros

    inner_call_dtypes: list[type[np.floating[Any]]] = []

    def fake_inner(**kwargs):
        compute_cp_dtype = kwargs["compute_cp_dtype"]
        rhs = np.asarray(kwargs["right_hand_side_gpu"], dtype=np.float64)
        inner_call_dtypes.append(compute_cp_dtype)
        if compute_cp_dtype == np.float32:
            return np.zeros_like(rhs, dtype=np.float64)
        return np.linalg.solve(covariance_matrix, rhs)

    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    monkeypatch.setattr(mixture_inference, "_try_import_cupy", lambda: fake_cupy)
    monkeypatch.setattr(mixture_inference, "_cupy_compute_dtype", lambda cp: cp.float32)
    monkeypatch.setattr(mixture_inference, "_solve_sample_space_rhs_gpu_inner", fake_inner)

    actual = mixture_inference._solve_sample_space_rhs_gpu(
        genotype_matrix=standardized,
        prior_variances=prior_variances,
        diagonal_noise=diagonal_noise,
        right_hand_side=right_hand_side,
        initial_guess=None,
        tolerance=1e-7,
        max_iterations=64,
        preconditioner=lambda rhs: rhs,
        batch_size=2,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)
    assert inner_call_dtypes[-1] == np.float64
    assert any(dtype == np.float32 for dtype in inner_call_dtypes[:-1])


def test_binary_posterior_state_streaming_gpu_avoids_jax_genotype_matvec(monkeypatch: pytest.MonkeyPatch):
    genotype_matrix = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 2.0],
            [2.0, 1.0, 1.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=np.float32,
    )
    standardized = as_raw_genotype_matrix(genotype_matrix).standardized(
        means=np.array([0.75, 1.0, 0.75], dtype=np.float32),
        scales=np.array([0.8291562, 0.70710677, 0.8291562], dtype=np.float32),
    )
    covariate_matrix = np.column_stack(
        [
            np.ones(genotype_matrix.shape[0], dtype=np.float32),
            np.array([0.5, -1.0, 1.5, 0.25], dtype=np.float32),
        ]
    )
    targets = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)
    prior_variances = np.array([1.0, 0.5, 0.75], dtype=np.float64)

    class _FakeCupy:
        float32 = np.float32
        float64 = np.float64

        @staticmethod
        def asarray(array, dtype=None):
            return np.asarray(array, dtype=dtype)

        @staticmethod
        def zeros(shape, dtype=None):
            return np.zeros(shape, dtype=dtype)

        @staticmethod
        def empty(shape, dtype=None, order=None):
            return np.empty(shape, dtype=dtype, order="C" if order is None else order)

        @staticmethod
        def isnan(array):
            return np.isnan(array)

        @staticmethod
        def exp(array):
            return np.exp(array)

        @staticmethod
        def where(condition, x, y):
            return np.where(condition, x, y)

        @staticmethod
        def log1p(array):
            return np.log1p(array)

        @staticmethod
        def abs(array):
            return np.abs(array)

        @staticmethod
        def tanh(array):
            return np.tanh(array)

        @staticmethod
        def maximum(x, y):
            return np.maximum(x, y)

        @staticmethod
        def sum(array, axis=None, dtype=None):
            return np.sum(array, axis=axis, dtype=dtype)

    def _unexpected_matvec(*args, **kwargs):
        raise AssertionError("streaming GPU binary path should not call StandardizedGenotypeMatrix.matvec")

    def _unexpected_transpose_matvec(*args, **kwargs):
        raise AssertionError("streaming GPU binary path should not call StandardizedGenotypeMatrix.transpose_matvec")

    def _fake_restricted_posterior_state(**kwargs):
        covariates = np.asarray(kwargs["covariate_matrix"], dtype=np.float64)
        target_vector = np.asarray(kwargs["targets"], dtype=np.float64)
        variant_count = kwargs["genotype_matrix"].shape[1]
        return (
            np.zeros(covariates.shape[1], dtype=np.float64),
            np.zeros(variant_count, dtype=np.float64),
            np.ones(variant_count, dtype=np.float64),
            np.zeros_like(target_vector),
            np.zeros_like(target_vector),
            0.0,
            0.0,
            0.0,
        )

    monkeypatch.setattr(mixture_inference, "_try_import_cupy", lambda: _FakeCupy())
    monkeypatch.setattr(mixture_inference, "_cupy_compute_dtype", lambda cp: cp.float32)
    monkeypatch.setattr(type(standardized), "matvec", _unexpected_matvec)
    monkeypatch.setattr(type(standardized), "transpose_matvec", _unexpected_transpose_matvec)
    monkeypatch.setattr(mixture_inference, "_restricted_posterior_state", _fake_restricted_posterior_state)

    alpha, beta, beta_variance, linear_predictor, collapsed_objective = mixture_inference._binary_posterior_state(
        genotype_matrix=standardized,
        covariate_matrix=covariate_matrix,
        targets=targets,
        prior_variances=prior_variances,
        alpha_init=np.array([0.1, -0.2], dtype=np.float64),
        beta_init=np.array([0.3, -0.1, 0.2], dtype=np.float64),
        minimum_weight=1e-4,
        max_iterations=1,
        gradient_tolerance=1e-6,
        initial_damping=1.0,
        damping_increase_factor=2.0,
        damping_decrease_factor=0.5,
        success_threshold=0.1,
        minimum_damping=1e-4,
        compute_logdet=False,
        compute_beta_variance=False,
        posterior_variance_batch_size=2,
    )

    assert alpha.shape == (covariate_matrix.shape[1],)
    assert beta.shape == (genotype_matrix.shape[1],)
    assert beta_variance.shape == (genotype_matrix.shape[1],)
    assert linear_predictor.shape == (genotype_matrix.shape[0],)
    assert np.isfinite(collapsed_objective)


def test_cpu_sample_space_block_cg_matches_dense_solution(monkeypatch: pytest.MonkeyPatch):
    genotype_matrix = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    standardized = as_raw_genotype_matrix(genotype_matrix).standardized(
        means=np.zeros(genotype_matrix.shape[1], dtype=np.float32),
        scales=np.ones(genotype_matrix.shape[1], dtype=np.float32),
    )
    prior_variances = np.array([1.5, 0.75, 0.5], dtype=np.float64)
    diagonal_noise = np.array([1.0, 1.25, 0.8, 1.1], dtype=np.float64)
    right_hand_side = np.column_stack(
        [
            np.array([0.5, -1.0, 0.2, 1.5], dtype=np.float64),
            np.array([-0.25, 0.75, 1.0, -0.5], dtype=np.float64),
        ]
    )

    monkeypatch.setattr(
        mixture_inference,
        "_sample_space_diagonal_preconditioner",
        lambda **kwargs: np.diag(
            np.diag(diagonal_noise) + genotype_matrix @ np.diag(prior_variances) @ genotype_matrix.T
        ).astype(np.float64, copy=False),
    )

    preconditioner = _sample_space_preconditioner(
        genotype_matrix=standardized,
        prior_variances=prior_variances,
        diagonal_noise=diagonal_noise,
        batch_size=2,
        rank=genotype_matrix.shape[1],
    )
    actual = mixture_inference._solve_sample_space_rhs_cpu(
        genotype_matrix=standardized,
        prior_variances=prior_variances,
        diagonal_noise=diagonal_noise,
        right_hand_side=right_hand_side,
        initial_guess=None,
        tolerance=1e-7,
        max_iterations=64,
        preconditioner=preconditioner,
        batch_size=2,
    )

    covariance_matrix = np.diag(diagonal_noise) + genotype_matrix @ np.diag(prior_variances) @ genotype_matrix.T
    expected = np.linalg.solve(covariance_matrix, right_hand_side)

    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=2e-5)


def test_orthogonal_probe_matrix_has_expected_column_norms_and_shape():
    probes = _orthogonal_probe_matrix(
        dimension=8,
        probe_count=13,
        random_seed=0,
    )

    assert probes.shape == (8, 13)
    np.testing.assert_allclose(
        np.sum(probes * probes, axis=0),
        np.full(13, 8.0, dtype=np.float64),
        rtol=1e-7,
        atol=1e-7,
    )


def test_posterior_variance_diagonal_uses_low_rank_residual_estimator():
    low_rank_factor = np.array(
        [
            [1.0, 0.0],
            [0.5, 1.0],
            [0.0, 1.5],
            [1.0, -0.5],
            [0.25, 0.75],
            [0.75, 0.25],
        ],
        dtype=np.float64,
    )
    inverse_precision = low_rank_factor @ low_rank_factor.T
    solve_calls: list[tuple[int, int]] = []

    def solve_variant_rhs(right_hand_side: np.ndarray) -> np.ndarray:
        rhs = np.asarray(right_hand_side, dtype=np.float64)
        if rhs.ndim == 1:
            rhs = rhs[:, None]
        solve_calls.append(rhs.shape)
        return inverse_precision @ rhs

    estimated_diagonal = mixture_inference._posterior_variance_low_rank_residual_diagonal(
        solve_variant_rhs=solve_variant_rhs,
        dimension=inverse_precision.shape[0],
        probe_count=4,
        random_seed=0,
    )

    np.testing.assert_allclose(estimated_diagonal, np.diag(inverse_precision), rtol=3e-5, atol=3e-5)
    assert solve_calls == [(inverse_precision.shape[0], 4)]


def test_stochastic_restricted_cross_leverage_diagonal_tracks_exact_leverage():
    genotype_matrix = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    standardized = as_raw_genotype_matrix(genotype_matrix).standardized(
        means=np.zeros(genotype_matrix.shape[1], dtype=np.float32),
        scales=np.ones(genotype_matrix.shape[1], dtype=np.float32),
    )
    standardized._dense_cache = standardized.materialize()
    covariate_matrix = np.column_stack(
        [np.ones(genotype_matrix.shape[0], dtype=np.float64), np.array([0.0, 1.0, -1.0, 0.5], dtype=np.float64)]
    )
    diagonal_noise = np.array([1.0, 1.5, 0.8, 1.2], dtype=np.float64)
    covariance_matrix = np.diag(diagonal_noise) + genotype_matrix.astype(np.float64) @ genotype_matrix.astype(np.float64).T
    covariance_inverse = np.linalg.inv(covariance_matrix)
    inverse_covariance_covariates = covariance_inverse @ covariate_matrix
    gls_normal_matrix = covariate_matrix.T @ inverse_covariance_covariates
    gls_cholesky = np.linalg.cholesky(gls_normal_matrix + np.eye(gls_normal_matrix.shape[0], dtype=np.float64) * 1e-8)

    def solve_rhs(right_hand_side: np.ndarray) -> np.ndarray:
        return covariance_inverse @ np.asarray(right_hand_side, dtype=np.float64)

    exact_inverse_covariance_genotypes = covariance_inverse @ genotype_matrix.astype(np.float64)
    exact_restricted_genotypes = exact_inverse_covariance_genotypes - inverse_covariance_covariates @ np.linalg.solve(
        gls_normal_matrix + np.eye(gls_normal_matrix.shape[0], dtype=np.float64) * 1e-8,
        covariate_matrix.T @ exact_inverse_covariance_genotypes,
    )
    exact_leverage = np.sum(genotype_matrix.astype(np.float64) * exact_restricted_genotypes, axis=0)

    estimated_leverage = _stochastic_restricted_cross_leverage_diagonal(
        genotype_matrix=standardized,
        covariate_matrix=covariate_matrix,
        solve_rhs=solve_rhs,
        inverse_covariance_covariates=inverse_covariance_covariates,
        gls_cholesky=gls_cholesky,
        batch_size=2,
        probe_count=1024,
        random_seed=0,
    )

    np.testing.assert_allclose(estimated_leverage, exact_leverage, rtol=0.15, atol=0.05)


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


def test_streaming_sample_space_operator_matmat_matches_dense_reference(random_generator):
    sample_count, variant_count = 24, 96
    genotype_values = random_generator.normal(size=(sample_count, variant_count)).astype(np.float32)
    prior_variances = random_generator.uniform(0.2, 1.2, size=variant_count).astype(np.float64)
    diagonal_noise = random_generator.uniform(0.5, 1.5, size=sample_count).astype(np.float64)
    rhs_matrix = random_generator.normal(size=(sample_count, 7)).astype(np.float64)

    standardized = as_raw_genotype_matrix(genotype_values).standardized(
        means=np.zeros(variant_count, dtype=np.float32),
        scales=np.ones(variant_count, dtype=np.float32),
    )
    operator = _sample_space_operator(standardized, prior_variances, diagonal_noise, batch_size=11)

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


def test_prefer_iterative_variant_space_targets_fast_warm_started_point_estimate_updates():
    sample_count, variant_count = 4, 6
    genotype_values = np.array(
        [
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    standardized = as_raw_genotype_matrix(genotype_values).standardized(
        means=np.zeros(variant_count, dtype=np.float32),
        scales=np.ones(variant_count, dtype=np.float32),
    )
    standardized._dense_cache = standardized.materialize()
    assert not _prefer_iterative_variant_space(
        genotype_matrix=standardized,
        sample_count=sample_count,
        variant_count=variant_count,
        compute_beta_variance=True,
        compute_logdet=False,
        initial_beta_guess=np.zeros(variant_count, dtype=np.float64),
    )
    assert _prefer_iterative_variant_space(
        genotype_matrix=standardized,
        sample_count=sample_count,
        variant_count=variant_count,
        compute_beta_variance=False,
        compute_logdet=False,
        initial_beta_guess=np.zeros(variant_count, dtype=np.float64),
    )
    assert not _prefer_iterative_variant_space(
        genotype_matrix=standardized,
        sample_count=sample_count,
        variant_count=variant_count,
        compute_beta_variance=True,
        compute_logdet=True,
        initial_beta_guess=np.zeros(variant_count, dtype=np.float64),
    )
    assert not _prefer_iterative_variant_space(
        genotype_matrix=standardized,
        sample_count=sample_count,
        variant_count=sample_count,
        compute_beta_variance=False,
        compute_logdet=False,
        initial_beta_guess=np.zeros(sample_count, dtype=np.float64),
    )
    streamed = as_raw_genotype_matrix(genotype_values).standardized(
        means=np.zeros(variant_count, dtype=np.float32),
        scales=np.ones(variant_count, dtype=np.float32),
    )
    assert not _prefer_iterative_variant_space(
        genotype_matrix=streamed,
        sample_count=sample_count,
        variant_count=variant_count,
        compute_beta_variance=False,
        compute_logdet=False,
        initial_beta_guess=np.zeros(variant_count, dtype=np.float64),
    )


def test_restricted_posterior_state_streaming_warm_start_keeps_sample_space_dispatch(monkeypatch: pytest.MonkeyPatch):
    sample_count, variant_count = 4, 6
    genotype_values = np.array(
        [
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    standardized = as_raw_genotype_matrix(genotype_values).standardized(
        means=np.zeros(variant_count, dtype=np.float32),
        scales=np.ones(variant_count, dtype=np.float32),
    )
    covariate_matrix = np.ones((sample_count, 1), dtype=np.float32)
    targets = np.array([0.5, -1.0, 0.25, 1.5], dtype=np.float32)
    prior_variances = np.ones(variant_count, dtype=np.float64)
    diagonal_noise = np.ones(sample_count, dtype=np.float64)

    def _unexpected_variant_solver(*args, **kwargs):
        raise AssertionError("streamed warm-started solve should not force iterative variant-space dispatch")

    monkeypatch.setattr(mixture_inference, "_solve_spd_system_variant_space", _unexpected_variant_solver, raising=False)
    monkeypatch.setattr(mixture_inference, "_try_import_cupy", lambda: None)

    alpha, beta, beta_variance, projected_targets, linear_predictor, restricted_quadratic, logdet_covariance, logdet_gls = (
        mixture_inference._restricted_posterior_state(
            genotype_matrix=standardized,
            covariate_matrix=covariate_matrix,
            targets=targets,
            prior_variances=prior_variances,
            diagonal_noise=diagonal_noise,
            solver_tolerance=1e-6,
            maximum_linear_solver_iterations=32,
            logdet_probe_count=4,
            logdet_lanczos_steps=8,
            exact_solver_matrix_limit=2,
            posterior_variance_batch_size=2,
            posterior_variance_probe_count=4,
            random_seed=0,
            compute_logdet=False,
            compute_beta_variance=False,
            initial_beta_guess=np.zeros(variant_count, dtype=np.float64),
            sample_space_preconditioner_rank=4,
        )
    )

    assert alpha.shape == (covariate_matrix.shape[1],)
    assert beta.shape == (variant_count,)
    assert beta_variance.shape == (variant_count,)
    assert projected_targets.shape == (sample_count,)
    assert linear_predictor.shape == (sample_count,)
    assert np.isfinite(restricted_quadratic)
    assert np.isfinite(logdet_covariance)
    assert np.isfinite(logdet_gls)


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


def test_exact_variant_gpu_summary_path_matches_dense_reference(monkeypatch: pytest.MonkeyPatch, random_generator):
    sample_count, variant_count = 12, 5
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
    standardized._cupy_cache = standardized.materialize().astype(np.float32, copy=False)
    standardized._dense_cache = None

    empty_dtypes: list[object] = []

    fake_cupy: Any = types.ModuleType("cupy")
    fake_cupy.float32 = np.float32
    fake_cupy.float64 = np.float64
    fake_cupy.asarray = lambda array, dtype=None: np.asarray(array, dtype=dtype)
    fake_cupy.empty = lambda shape, dtype=None, order=None: (
        empty_dtypes.append(dtype),
        np.empty(shape, dtype=dtype, order="C" if order is None else order),
    )[1]
    fake_cupy.eye = np.eye

    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    monkeypatch.setattr(mixture_inference, "_cupy_compute_dtype", lambda cp: cp.float32)

    gpu_result = _quantitative_posterior_state(
        genotype_matrix=standardized,
        covariate_matrix=covariate_matrix,
        targets=target_vector,
        prior_variances=prior_variances,
        sigma_error2=1.0,
        sigma_error_floor=1e-6,
        exact_solver_matrix_limit=6,
        posterior_variance_batch_size=3,
    )
    dense_result = _quantitative_posterior_state(
        genotype_matrix=genotype_values,
        covariate_matrix=covariate_matrix,
        targets=target_vector,
        prior_variances=prior_variances,
        sigma_error2=1.0,
        sigma_error_floor=1e-6,
        exact_solver_matrix_limit=6,
        posterior_variance_batch_size=3,
    )

    assert np.float32 in empty_dtypes
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
        compute_logdet,
        compute_beta_variance=True,
        predictor_offset=None,
        stale_beta_variance=None,
        restricted_posterior_warm_start=None,
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

    def fake_validation_metric(trait_type, genotype_matrix, covariate_matrix, targets, alpha, beta, predictor_offset=None):
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
        compute_logdet,
        compute_beta_variance=True,
        predictor_offset=None,
        stale_beta_variance=None,
        restricted_posterior_warm_start=None,
    ):
        return PosteriorState(
            alpha=np.zeros(covariate_matrix.shape[1], dtype=np.float64),
            beta=np.zeros(1, dtype=np.float64),
            beta_variance=np.ones(1, dtype=np.float64),
            linear_predictor=np.zeros(targets.shape[0], dtype=np.float64),
            collapsed_objective=0.0,
            sigma_error2=1.0,
        )

    def fake_validation_metric(trait_type, genotype_matrix, covariate_matrix, targets, alpha, beta, predictor_offset=None):
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


def test_variational_em_reuses_beta_variance_between_refreshes(monkeypatch: pytest.MonkeyPatch):
    compute_beta_variance_calls: list[bool] = []
    stale_beta_variance_flags: list[bool] = []

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
        compute_logdet,
        compute_beta_variance=True,
        predictor_offset=None,
        stale_beta_variance=None,
        restricted_posterior_warm_start=None,
    ):
        compute_beta_variance_calls.append(bool(compute_beta_variance))
        stale_beta_variance_flags.append(stale_beta_variance is not None)
        beta_variance = (
            np.full(1, float(len(compute_beta_variance_calls)), dtype=np.float64)
            if compute_beta_variance
            else np.asarray(stale_beta_variance, dtype=np.float64)
        )
        return PosteriorState(
            alpha=np.asarray(alpha_init, dtype=np.float64),
            beta=np.asarray(beta_init, dtype=np.float64),
            beta_variance=beta_variance,
            linear_predictor=np.zeros(targets.shape[0], dtype=np.float64),
            collapsed_objective=0.0,
            sigma_error2=1.0,
        )

    monkeypatch.setattr(mixture_inference, "_fit_collapsed_posterior", fake_fit_collapsed_posterior)

    config = ModelConfig(
        trait_type=TraitType.QUANTITATIVE,
        max_outer_iterations=5,
        convergence_tolerance=0.0,
        update_hyperparameters=False,
        beta_variance_update_interval=4,
        stochastic_variational_updates=False,
    )

    fit_variational_em(
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
    )

    assert compute_beta_variance_calls == [False, False, False, True, False, True]
    assert stale_beta_variance_flags == [True, True, True, True, True, False]


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

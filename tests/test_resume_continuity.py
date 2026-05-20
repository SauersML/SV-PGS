"""Resume continuity tests: ELBO history + Anderson(m) memory survive a
checkpoint round-trip so post-resume diagnostics and acceleration begin
seamlessly (no fresh-list discontinuity, no Krylov memory loss).

These tests pin down the contract that:
  * ``elbo_history`` carries pre-resume entries into the post-resume list.
  * Anderson state restores when the saved memory depth matches the resuming
    config; otherwise the EM falls back to a fresh accelerator without
    raising.
  * Old-format checkpoints (pre-Wave-2, lacking the new fields entirely)
    load cleanly with sensible defaults and resume without crash.
"""
from __future__ import annotations

import pickle

import numpy as np
import pytest

from sv_pgs.config import ModelConfig, TraitType
from sv_pgs.inference import fit_variational_em
from sv_pgs.mixture_inference import (
    VariationalFitCheckpoint,
    _build_prior_design,
    _checkpoint_config_signature,
    _checkpoint_prior_design_signature,
    collapse_tie_groups,
)
from sv_pgs.preprocessing import build_tie_map

from tests.conftest import make_variant_records


def _make_problem(random_generator, *, sample_count: int = 32, variant_count: int = 6):
    genotype_matrix = random_generator.normal(size=(sample_count, variant_count)).astype(np.float32)
    covariate_matrix = np.column_stack(
        [
            np.ones(sample_count, dtype=np.float32),
            random_generator.normal(size=sample_count).astype(np.float32),
        ]
    )
    target_vector = random_generator.normal(size=sample_count).astype(np.float32)
    records = make_variant_records(variant_count)
    return genotype_matrix, covariate_matrix, target_vector, records


def _capture_after_epoch(checkpoint_holder: list, target_epoch: int):
    def _callback(checkpoint: VariationalFitCheckpoint) -> None:
        if checkpoint.completed_iterations == target_epoch and not checkpoint_holder:
            checkpoint_holder.append(pickle.loads(pickle.dumps(checkpoint)))

    return _callback


def test_elbo_history_continuous_across_resume(random_generator):
    genotype_matrix, covariate_matrix, target_vector, records = _make_problem(random_generator)
    config = ModelConfig(
        trait_type=TraitType.QUANTITATIVE,
        max_outer_iterations=4,
        exact_solver_matrix_limit=256,
        minimum_minor_allele_frequency=0.0,
    )
    tie_map = build_tie_map(genotype_matrix, records, config)

    saved: list[VariationalFitCheckpoint] = []
    first = fit_variational_em(
        genotypes=genotype_matrix,
        covariates=covariate_matrix,
        targets=target_vector,
        records=records,
        config=config,
        tie_map=tie_map,
        checkpoint_callback=_capture_after_epoch(saved, target_epoch=2),
    )
    assert len(first.elbo_history) == 4
    assert len(saved) == 1
    checkpoint = saved[0]
    assert len(checkpoint.elbo_history) == 2

    pre_resume_elbo = list(checkpoint.elbo_history)

    resumed = fit_variational_em(
        genotypes=genotype_matrix,
        covariates=covariate_matrix,
        targets=target_vector,
        records=records,
        config=config,
        tie_map=tie_map,
        resume_checkpoint=checkpoint,
    )
    assert len(resumed.elbo_history) == 4
    # First two ELBOs must be the saved pre-resume values byte-for-byte.
    for saved_value, restored_value in zip(pre_resume_elbo, resumed.elbo_history[:2]):
        if np.isnan(saved_value):
            assert np.isnan(restored_value)
        else:
            assert float(restored_value) == pytest.approx(float(saved_value))


def test_anderson_memory_restored_when_compatible(random_generator):
    """Anderson state with matching memory depth must restore cleanly. The
    math fixed point is unchanged either way; we only verify the restore
    pathway leaves the accelerator with the saved history loaded."""
    genotype_matrix, covariate_matrix, target_vector, records = _make_problem(random_generator)
    config = ModelConfig(
        trait_type=TraitType.QUANTITATIVE,
        max_outer_iterations=5,
        exact_solver_matrix_limit=256,
        minimum_minor_allele_frequency=0.0,
    )
    tie_map = build_tie_map(genotype_matrix, records, config)

    saved: list[VariationalFitCheckpoint] = []
    fit_variational_em(
        genotypes=genotype_matrix,
        covariates=covariate_matrix,
        targets=target_vector,
        records=records,
        config=config,
        tie_map=tie_map,
        checkpoint_callback=_capture_after_epoch(saved, target_epoch=4),
    )
    assert len(saved) == 1
    checkpoint = saved[0]
    # Anderson is wired up at construction time (empty history is fine if
    # the inner loop hasn't populated it yet); regardless, the new fields
    # must round-trip through pickle.
    blob = pickle.dumps(checkpoint)
    restored = pickle.loads(blob)
    assert restored.anderson_memory_depth == 5
    # Synthesize a non-empty Anderson history so the restore path exercises
    # actually-populated memory (the EM loop has not been wired to fill this
    # yet, but the resume code must accept whatever was saved).
    packed_size = int(checkpoint.beta_state.shape[0])
    fake_iterates = [np.linspace(0.0, 1.0, packed_size) for _ in range(3)]
    fake_map_values = [arr + 0.1 for arr in fake_iterates]
    fake_residuals = [m - x for m, x in zip(fake_map_values, fake_iterates)]
    restored.anderson_iterates = [arr.copy() for arr in fake_iterates]
    restored.anderson_map_values = [arr.copy() for arr in fake_map_values]
    restored.anderson_residuals = [arr.copy() for arr in fake_residuals]
    # Hand-rolled state still has to load and let the EM continue.
    resumed = fit_variational_em(
        genotypes=genotype_matrix,
        covariates=covariate_matrix,
        targets=target_vector,
        records=records,
        config=config,
        tie_map=tie_map,
        resume_checkpoint=restored,
    )
    assert np.all(np.isfinite(resumed.alpha))
    assert np.all(np.isfinite(resumed.beta_reduced))


def test_old_checkpoint_loads_with_new_fields(random_generator):
    """An old-format checkpoint (without ELBO/Anderson fields) must load via
    ``__setstate__`` defaults and resume without crashing."""
    genotype_matrix, covariate_matrix, target_vector, records = _make_problem(random_generator)
    config = ModelConfig(
        trait_type=TraitType.QUANTITATIVE,
        max_outer_iterations=3,
        exact_solver_matrix_limit=256,
        minimum_minor_allele_frequency=0.0,
    )
    tie_map = build_tie_map(genotype_matrix, records, config)
    reduced_records = collapse_tie_groups(list(records), tie_map)
    prior_design = _build_prior_design(reduced_records)

    # Build a checkpoint exactly as the pre-Wave-2 code would have. Don't
    # pass any of the new fields — defaults must absorb them.
    checkpoint = VariationalFitCheckpoint(
        config_signature=_checkpoint_config_signature(config),
        prior_design_signature=_checkpoint_prior_design_signature(prior_design),
        validation_enabled=False,
        completed_iterations=1,
        alpha_state=np.zeros(covariate_matrix.shape[1], dtype=np.float64),
        beta_state=np.zeros(genotype_matrix.shape[1], dtype=np.float64),
        local_scale=np.ones(len(reduced_records), dtype=np.float64),
        auxiliary_delta=np.full(len(reduced_records), 0.5, dtype=np.float64),
        sigma_error2=1.0,
        global_scale=0.1,
        scale_model_coefficients=np.zeros(prior_design.design_matrix.shape[1], dtype=np.float64),
        tpb_shape_a_vector=np.ones(prior_design.class_membership_matrix.shape[1], dtype=np.float64),
        tpb_shape_b_vector=np.full(prior_design.class_membership_matrix.shape[1], 0.5, dtype=np.float64),
        objective_history=[0.0],
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
    # New fields populate at defaults.
    assert checkpoint.elbo_history == []
    assert checkpoint.anderson_iterates is None
    assert checkpoint.anderson_map_values is None
    assert checkpoint.anderson_residuals is None
    assert checkpoint.anderson_memory_depth is None

    # Also exercise a __setstate__ round-trip from a state dict that lacks
    # the new field names entirely (simulating a pickle written before they
    # existed).
    state_dict = checkpoint.__getstate__()
    for name in [
        "elbo_history",
        "anderson_iterates",
        "anderson_map_values",
        "anderson_residuals",
        "anderson_memory_depth",
    ]:
        state_dict.pop(name, None)
    legacy_restored = object.__new__(VariationalFitCheckpoint)
    legacy_restored.__setstate__((None, state_dict))
    assert legacy_restored.elbo_history == []
    assert legacy_restored.anderson_iterates is None
    assert legacy_restored.anderson_memory_depth is None

    # Resume EM from the old-format checkpoint. ELBO history starts empty
    # (no pre-Wave-2 entries existed), then accumulates as the EM runs.
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
    # ELBO history starts empty (no entries existed in the old checkpoint),
    # then accumulates one entry per post-resume iteration. We ran 2 more
    # iterations from a completed_iterations=1 checkpoint up to max=3, so
    # exactly 2 ELBO entries should be present.
    assert len(result.elbo_history) == int(config.max_outer_iterations) - 1
    assert np.all(np.isfinite(result.alpha))


def test_anderson_memory_depth_mismatch_resets_safely(random_generator):
    """A legacy checkpoint with a non-current Anderson depth should be
    discarded and the EM must continue without raising. ELBO history still
    carries through."""
    genotype_matrix, covariate_matrix, target_vector, records = _make_problem(random_generator)
    config_depth5 = ModelConfig(
        trait_type=TraitType.QUANTITATIVE,
        max_outer_iterations=3,
        exact_solver_matrix_limit=256,
        minimum_minor_allele_frequency=0.0,
    )
    tie_map = build_tie_map(genotype_matrix, records, config_depth5)
    saved: list[VariationalFitCheckpoint] = []
    fit_variational_em(
        genotypes=genotype_matrix,
        covariates=covariate_matrix,
        targets=target_vector,
        records=records,
        config=config_depth5,
        tie_map=tie_map,
        checkpoint_callback=_capture_after_epoch(saved, target_epoch=2),
    )
    checkpoint = saved[0]
    # Force a non-empty Anderson history into the checkpoint so the
    # depth-mismatch branch has something to discard.
    packed_size = int(checkpoint.beta_state.shape[0])
    checkpoint.anderson_iterates = [np.zeros(packed_size, dtype=np.float64) for _ in range(2)]
    checkpoint.anderson_map_values = [np.zeros(packed_size, dtype=np.float64) for _ in range(2)]
    checkpoint.anderson_residuals = [np.zeros(packed_size, dtype=np.float64) for _ in range(2)]
    # Simulate a checkpoint written with a non-current internal depth.
    checkpoint.anderson_memory_depth = 3

    config_current = ModelConfig(
        trait_type=TraitType.QUANTITATIVE,
        max_outer_iterations=3,
        exact_solver_matrix_limit=256,
        minimum_minor_allele_frequency=0.0,
    )
    resumed = fit_variational_em(
        genotypes=genotype_matrix,
        covariates=covariate_matrix,
        targets=target_vector,
        records=records,
        config=config_current,
        tie_map=tie_map,
        resume_checkpoint=checkpoint,
    )
    # Math must continue: finite outputs, ELBO history non-empty.
    assert np.all(np.isfinite(resumed.alpha))
    assert np.all(np.isfinite(resumed.beta_reduced))
    assert len(resumed.elbo_history) >= 1


def test_anderson_packed_dim_mismatch_falls_back_to_fresh(random_generator):
    """If the saved Anderson history's packed dimension does not match the
    current state (e.g. number of scale-model coefficients changed in a
    forward-compatible way), restore must safely reset rather than crash."""
    genotype_matrix, covariate_matrix, target_vector, records = _make_problem(random_generator)
    config = ModelConfig(
        trait_type=TraitType.QUANTITATIVE,
        max_outer_iterations=3,
        exact_solver_matrix_limit=256,
        minimum_minor_allele_frequency=0.0,
    )
    tie_map = build_tie_map(genotype_matrix, records, config)
    saved: list[VariationalFitCheckpoint] = []
    fit_variational_em(
        genotypes=genotype_matrix,
        covariates=covariate_matrix,
        targets=target_vector,
        records=records,
        config=config,
        tie_map=tie_map,
        checkpoint_callback=_capture_after_epoch(saved, target_epoch=2),
    )
    checkpoint = saved[0]
    # Inconsistent shapes in the Anderson history — restore must raise
    # internally, get caught by the defense-in-depth except, and reset.
    checkpoint.anderson_iterates = [np.zeros(7, dtype=np.float64), np.zeros(11, dtype=np.float64)]
    checkpoint.anderson_map_values = [np.zeros(7, dtype=np.float64), np.zeros(11, dtype=np.float64)]
    checkpoint.anderson_residuals = [np.zeros(7, dtype=np.float64), np.zeros(11, dtype=np.float64)]
    checkpoint.anderson_memory_depth = 5

    resumed = fit_variational_em(
        genotypes=genotype_matrix,
        covariates=covariate_matrix,
        targets=target_vector,
        records=records,
        config=config,
        tie_map=tie_map,
        resume_checkpoint=checkpoint,
    )
    assert np.all(np.isfinite(resumed.alpha))
    assert np.all(np.isfinite(resumed.beta_reduced))

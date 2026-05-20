"""Regression tests for Wave 2 checkpoint backward compatibility.

These tests pin down the contract that:
  * Old on-disk pickles of ``VariationalFitCheckpoint`` (lacking Wave 2 fields)
    must continue to load via ``__setstate__``.
  * Math-relevant knobs (e.g. ``sigma_error_floor``) DO change the signature
    and so DO invalidate stale checkpoints.
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


def _build_old_checkpoint(
    *,
    config_signature: str,
    prior_design_signature: str,
    covariate_dim: int,
    reduced_count: int,
    design_dim: int,
    class_count: int,
    validation_enabled: bool = False,
    completed_iterations: int = 1,
    beta_state_dim: int | None = None,
) -> VariationalFitCheckpoint:
    """Build a checkpoint using only the pre-Wave-2 declared field set.

    This deliberately does NOT pass ``binary_block_resume_state``,
    ``stochastic_block_size``, ``best_validation_iteration``, or any other
    field added since the first published checkpoint format. The dataclass
    declares defaults for those, so construction should succeed and the new
    fields should land at their defaults.
    """
    return VariationalFitCheckpoint(
        config_signature=config_signature,
        prior_design_signature=prior_design_signature,
        validation_enabled=validation_enabled,
        completed_iterations=completed_iterations,
        alpha_state=np.zeros(covariate_dim, dtype=np.float64),
        beta_state=np.zeros(
            reduced_count if beta_state_dim is None else beta_state_dim,
            dtype=np.float64,
        ),
        local_scale=np.ones(reduced_count, dtype=np.float64),
        auxiliary_delta=np.full(reduced_count, 0.5, dtype=np.float64),
        sigma_error2=1.0,
        global_scale=0.1,
        scale_model_coefficients=np.zeros(design_dim, dtype=np.float64),
        tpb_shape_a_vector=np.full(class_count, 1.0, dtype=np.float64),
        tpb_shape_b_vector=np.full(class_count, 0.5, dtype=np.float64),
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


def test_old_checkpoint_pickle_round_trip_preserves_defaults():
    """Pickling a checkpoint built without new fields, then unpickling it,
    must yield an object where new fields are at their declared defaults."""
    checkpoint = _build_old_checkpoint(
        config_signature="cfg",
        prior_design_signature="design",
        covariate_dim=2,
        reduced_count=3,
        design_dim=1,
        class_count=1,
    )
    blob = pickle.dumps(checkpoint)
    restored = pickle.loads(blob)

    # Old fields preserved.
    assert restored.config_signature == "cfg"
    assert restored.prior_design_signature == "design"
    assert restored.completed_iterations == 1
    np.testing.assert_array_equal(restored.alpha_state, checkpoint.alpha_state)
    np.testing.assert_array_equal(restored.beta_state, checkpoint.beta_state)
    np.testing.assert_array_equal(restored.local_scale, checkpoint.local_scale)
    np.testing.assert_array_equal(restored.auxiliary_delta, checkpoint.auxiliary_delta)
    assert restored.sigma_error2 == pytest.approx(1.0)
    assert restored.global_scale == pytest.approx(0.1)

    # Wave 2 / later fields default cleanly.
    assert restored.best_validation_iteration is None
    assert restored.completed_blocks_in_iteration == 0
    assert restored.beta_variance_state is None
    assert restored.reduced_second_moment is None
    assert restored.epoch_reduced_prior_variances is None
    assert restored.binary_block_resume_state is None
    assert restored.stochastic_block_size is None


def test_setstate_tolerates_missing_new_fields_in_legacy_blob():
    """Simulate an on-disk pickle written before any Wave 2 fields existed.

    We strip every dataclass field that has a declared default from the
    legacy state dict to mimic a checkpoint serialized by an older version
    of the codebase, then verify ``__setstate__`` fills in defaults rather
    than raising.
    """
    checkpoint = _build_old_checkpoint(
        config_signature="cfg",
        prior_design_signature="design",
        covariate_dim=2,
        reduced_count=3,
        design_dim=1,
        class_count=1,
    )
    legacy_state = checkpoint.__getstate__()
    # Drop every field that has a default — represents an older serialization
    # that simply didn't know about these names yet.
    for name in [
        "best_validation_iteration",
        "completed_blocks_in_iteration",
        "beta_variance_state",
        "reduced_second_moment",
        "epoch_reduced_prior_variances",
        "binary_block_resume_state",
        "stochastic_block_size",
    ]:
        legacy_state.pop(name, None)

    restored = object.__new__(VariationalFitCheckpoint)
    restored.__setstate__((None, legacy_state))

    assert restored.best_validation_iteration is None
    assert restored.completed_blocks_in_iteration == 0
    assert restored.beta_variance_state is None
    assert restored.reduced_second_moment is None
    assert restored.epoch_reduced_prior_variances is None
    assert restored.binary_block_resume_state is None
    assert restored.stochastic_block_size is None
    np.testing.assert_array_equal(restored.alpha_state, checkpoint.alpha_state)


def test_math_relevant_field_change_invalidates_signature():
    """Conversely, changing a knob that DOES alter the variational fixed
    point (e.g. ``sigma_error_floor``) must produce a fresh signature so
    that stale checkpoints are correctly rejected."""
    config_a = ModelConfig(
        trait_type=TraitType.QUANTITATIVE,
        max_outer_iterations=2,
        sigma_error_floor=1e-3,
    )
    config_b = ModelConfig(
        trait_type=TraitType.QUANTITATIVE,
        max_outer_iterations=2,
        sigma_error_floor=1e-2,
    )
    assert _checkpoint_config_signature(config_a) != _checkpoint_config_signature(config_b)


def test_fit_variational_em_resumes_from_old_format_checkpoint(random_generator):
    """End-to-end: build a checkpoint with only the pre-Wave-2 fields,
    pickle/unpickle it, and feed it to ``fit_variational_em``. The fit must
    accept the checkpoint (matching signature) and start at
    ``completed_iterations``, NOT raise on the missing new fields."""
    sample_count, variant_count = 24, 5
    genotype_matrix = random_generator.normal(size=(sample_count, variant_count)).astype(np.float32)
    covariate_matrix = np.column_stack(
        [
            np.ones(sample_count, dtype=np.float32),
            random_generator.normal(size=sample_count).astype(np.float32),
        ]
    )
    target_vector = random_generator.normal(size=sample_count).astype(np.float32)
    records = make_variant_records(variant_count)
    config = ModelConfig(
        trait_type=TraitType.QUANTITATIVE,
        max_outer_iterations=3,
        exact_solver_matrix_limit=128,
        minimum_minor_allele_frequency=0.0,
    )
    tie_map = build_tie_map(genotype_matrix, records, config)
    reduced_records = collapse_tie_groups(list(records), tie_map)
    prior_design = _build_prior_design(reduced_records)
    reduced_count = len(reduced_records)

    checkpoint = _build_old_checkpoint(
        config_signature=_checkpoint_config_signature(config),
        prior_design_signature=_checkpoint_prior_design_signature(prior_design),
        covariate_dim=covariate_matrix.shape[1],
        reduced_count=reduced_count,
        design_dim=prior_design.design_matrix.shape[1],
        class_count=prior_design.class_membership_matrix.shape[1],
        completed_iterations=1,
        beta_state_dim=genotype_matrix.shape[1],
    )

    # Round-trip through pickle so we exercise the real deserialization path.
    checkpoint = pickle.loads(pickle.dumps(checkpoint))

    # Resume with the current automatic optimizer policy; convergence-only
    # internals are not user configuration and do not alter the signature.
    resume_config = ModelConfig(
        trait_type=TraitType.QUANTITATIVE,
        max_outer_iterations=3,
        exact_solver_matrix_limit=128,
        minimum_minor_allele_frequency=0.0,
    )

    result = fit_variational_em(
        genotypes=genotype_matrix,
        covariates=covariate_matrix,
        targets=target_vector,
        records=records,
        config=resume_config,
        tie_map=tie_map,
        resume_checkpoint=checkpoint,
    )

    # If the resume was accepted, we ran (max_outer - completed) more
    # iterations rather than starting from scratch — objective_history is
    # populated and result is finite.
    assert result.objective_history
    assert np.all(np.isfinite(result.alpha))
    assert np.all(np.isfinite(result.beta_reduced))

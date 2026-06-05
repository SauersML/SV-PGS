"""Warm-start checkpoints survive a change to the variant-category *set*.

When the set of variant categories changes between runs (a class added,
removed, or reordered) the per-category TPB shape vectors no longer line up
column-for-column with the old checkpoint. Rather than throw the whole warm
start away, ``_reconcile_checkpoint_categories`` remaps each per-category
hyperparameter by NAME: matching classes keep their learned values, added
classes are seeded from the config default, removed classes are dropped.
"""
from __future__ import annotations

import numpy as np

from sv_pgs.config import ModelConfig, VariantClass
from sv_pgs import mixture_inference as mi


def _make_prior(classes: list[VariantClass]) -> mi.PriorDesign:
    n_classes = len(classes)
    membership = np.zeros((3, n_classes), dtype=np.float64)
    membership[:, 0] = 1.0  # arbitrary; reconciliation only reads the columns
    return mi.PriorDesign(
        design_matrix=np.ones((3, 1), dtype=np.float64),
        feature_names=["baseline"],
        feature_specs=(),
        class_membership_matrix=membership,
        inverse_class_lookup={index: cls for index, cls in enumerate(classes)},
    )


def _make_checkpoint(class_names: list[str] | None) -> mi.VariationalFitCheckpoint:
    return mi.VariationalFitCheckpoint(
        config_signature="cfg",
        prior_design_signature="old",
        validation_enabled=False,
        completed_iterations=1,
        alpha_state=np.zeros(1),
        beta_state=np.zeros(3),
        local_scale=np.ones(3),
        auxiliary_delta=np.ones(3),
        sigma_error2=1.0,
        global_scale=1.0,
        scale_model_coefficients=np.zeros(1),
        tpb_shape_a_vector=np.array([11.0, 22.0]),
        tpb_shape_b_vector=np.array([110.0, 220.0]),
        objective_history=[0.0],
        validation_history=[],
        previous_alpha=None,
        previous_beta=None,
        previous_local_scale=None,
        previous_theta=None,
        previous_tpb_shape_a_vector=np.array([1.0, 2.0]),
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
        prior_class_names=class_names,
    )


def test_prior_class_names_are_column_ordered():
    prior = _make_prior([VariantClass.SNV, VariantClass.DELETION_SHORT])
    assert mi._checkpoint_prior_class_names(prior) == ["snv", "deletion_short"]


def test_reconcile_keeps_matching_adds_new_drops_removed():
    old = [VariantClass.SNV, VariantClass.DELETION_SHORT]
    checkpoint = _make_checkpoint(mi._checkpoint_prior_class_names(_make_prior(old)))

    # SNV survives; DELETION_SHORT removed; DUPLICATION_SHORT added.
    new_prior = _make_prior([VariantClass.SNV, VariantClass.DUPLICATION_SHORT])
    config = ModelConfig()
    result = mi._reconcile_checkpoint_categories(checkpoint, new_prior, config)
    assert result is not None
    reconciled, tally = result

    assert tally == {"kept": 1, "added": 1, "dropped": 1}
    # SNV (column 0) keeps its learned shape-a value.
    assert reconciled.tpb_shape_a_vector[0] == 11.0
    # New class is seeded from the config default, not zero.
    default_a = config.class_tpb_shape_a()
    assert reconciled.tpb_shape_a_vector[1] == float(default_a.get(VariantClass.DUPLICATION_SHORT, 1.0))
    # previous_* vectors are remapped too (the present one), absent ones stay None.
    assert reconciled.previous_tpb_shape_a_vector[0] == 1.0
    assert reconciled.previous_tpb_shape_b_vector is None
    # Vectors now match the new column count so the downstream shape gate passes.
    assert reconciled.tpb_shape_a_vector.shape == (2,)
    # Accepted under the new prior design's signature.
    assert reconciled.prior_design_signature == mi._checkpoint_prior_design_signature(new_prior)


def test_reconcile_reorder_only_preserves_values_by_name():
    old = [VariantClass.SNV, VariantClass.DELETION_SHORT]
    checkpoint = _make_checkpoint(mi._checkpoint_prior_class_names(_make_prior(old)))
    # Same classes, swapped order: values must follow their names, not columns.
    new_prior = _make_prior([VariantClass.DELETION_SHORT, VariantClass.SNV])
    result = mi._reconcile_checkpoint_categories(checkpoint, new_prior, ModelConfig())
    assert result is not None
    reconciled, tally = result
    assert tally == {"kept": 2, "added": 0, "dropped": 0}
    assert reconciled.tpb_shape_a_vector[0] == 22.0  # deletion_short was 22.0
    assert reconciled.tpb_shape_a_vector[1] == 11.0  # snv was 11.0


def test_reconcile_returns_none_for_pre_category_aware_checkpoint():
    # An old pickle without recorded category names cannot be safely remapped.
    checkpoint = _make_checkpoint(None)
    new_prior = _make_prior([VariantClass.SNV, VariantClass.DUPLICATION_SHORT])
    assert mi._reconcile_checkpoint_categories(checkpoint, new_prior, ModelConfig()) is None

"""Pin: ``save_artifact`` + ``load_artifact`` round-trip with empty
histories, zero variants, and zero non-zero coefficients.

Edge cases:

* ``objective_history=[]`` and ``validation_history=[]`` must serialize
  as empty JSON arrays and round-trip back to empty lists.
* Zero variants (``records=[]``) must produce zero-length arrays in the
  artifact and round-trip without raising.
* ``beta_reduced``, ``beta_full``, ``alpha`` all zeros must compare
  exactly equal after the round-trip (float32, no quantization loss).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from sv_pgs.artifact import ModelArtifact, load_artifact, save_artifact
from sv_pgs.config import ModelConfig, TraitType, VariantClass
from sv_pgs.data import TieMap


def _make_artifact(variant_count: int) -> ModelArtifact:
    from sv_pgs.data import VariantRecord

    records = [
        VariantRecord(
            variant_id=f"v{i}",
            variant_class=VariantClass.SNV,
            chromosome="1",
            position=1000 + i,
        )
        for i in range(variant_count)
    ]
    tie_map = TieMap(
        kept_indices=np.arange(variant_count, dtype=np.int32),
        original_to_reduced=np.arange(variant_count, dtype=np.int32),
        reduced_to_group=[],
    )
    return ModelArtifact(
        config=ModelConfig(trait_type=TraitType.QUANTITATIVE),
        records=records,
        means=np.zeros(variant_count, dtype=np.float32),
        scales=np.ones(variant_count, dtype=np.float32),
        alpha=np.zeros(0, dtype=np.float32),
        beta_reduced=np.zeros(variant_count, dtype=np.float32),
        beta_full=np.zeros(variant_count, dtype=np.float32),
        beta_variance=np.zeros(variant_count, dtype=np.float32),
        tie_map=tie_map,
        sigma_e2=1.0,
        prior_scales=np.ones(variant_count, dtype=np.float32),
        global_scale=1.0,
        class_tpb_shape_a={VariantClass.SNV: 1.0},
        class_tpb_shape_b={VariantClass.SNV: 1.0},
        scale_model_coefficients=np.zeros(0, dtype=np.float32),
        scale_model_feature_names=[],
        objective_history=[],
        validation_history=[],
    )


def test_empty_histories_roundtrip(tmp_path: Path):
    artifact = _make_artifact(variant_count=2)
    save_artifact(tmp_path, artifact)
    restored = load_artifact(tmp_path)
    assert restored.objective_history == []
    assert restored.validation_history == []


def test_zero_variants_roundtrip(tmp_path: Path):
    artifact = _make_artifact(variant_count=0)
    save_artifact(tmp_path, artifact)
    restored = load_artifact(tmp_path)
    assert len(restored.records) == 0
    assert restored.means.shape == (0,)
    assert restored.beta_full.shape == (0,)
    assert restored.tie_map.kept_indices.shape == (0,)
    assert restored.tie_map.original_to_reduced.shape == (0,)


def test_all_zero_coefficients_roundtrip_exactly(tmp_path: Path):
    artifact = _make_artifact(variant_count=4)
    save_artifact(tmp_path, artifact)
    restored = load_artifact(tmp_path)
    np.testing.assert_array_equal(restored.beta_full, np.zeros(4, dtype=np.float32))
    np.testing.assert_array_equal(restored.beta_reduced, np.zeros(4, dtype=np.float32))
    np.testing.assert_array_equal(restored.alpha, np.zeros(0, dtype=np.float32))
    np.testing.assert_array_equal(restored.beta_variance, np.zeros(4, dtype=np.float32))


def test_diagnostics_defaults_roundtrip(tmp_path: Path):
    """``converged`` defaults to False, and the four ``final_*_change``
    fields default to None.  The round-trip must preserve these defaults
    bit-exactly (False, not None; None, not 0.0)."""
    artifact = _make_artifact(variant_count=1)
    save_artifact(tmp_path, artifact)
    restored = load_artifact(tmp_path)
    assert restored.converged is False
    assert restored.selected_iteration_count == 0
    assert restored.final_parameter_change is None
    assert restored.final_predictor_change is None
    assert restored.final_objective_change is None
    assert restored.final_hyperparameter_change is None

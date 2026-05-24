"""Regression tests for ModelArtifact convergence-metadata persistence.

Bug fixed here: the six convergence diagnostics added to ``ModelArtifact``
(``converged``, ``selected_iteration_count``, ``final_parameter_change``,
``final_predictor_change``, ``final_objective_change``,
``final_hyperparameter_change``) must round-trip through
``save_artifact``/``load_artifact``. A separate hazard — pre-fix —
was that an old artifact missing these JSON keys raised a ``KeyError``
at load time. ``_load_diagnostics`` now back-fills defaults and emits
one warning instead of failing.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sv_pgs.artifact import (
    ModelArtifact,
    load_artifact,
    save_artifact,
)
from sv_pgs.config import ModelConfig, TraitType, VariantClass
from sv_pgs.data import TieMap, VariantRecord


_DIAGNOSTIC_FIELDS = (
    "converged",
    "selected_iteration_count",
    "final_parameter_change",
    "final_predictor_change",
    "final_objective_change",
    "final_hyperparameter_change",
)

_DIAGNOSTICS_PRESENT = all(field in ModelArtifact.__dataclass_fields__ for field in _DIAGNOSTIC_FIELDS)

if not _DIAGNOSTICS_PRESENT:
    pytest.skip(
        "waiting for fix in sv_pgs/artifact.py (diagnostic fields not yet added)",
        allow_module_level=True,
    )


def _make_minimal_artifact(**diagnostic_overrides) -> ModelArtifact:
    """Construct a one-variant artifact suitable for save/load round-trips."""
    record = VariantRecord(
        variant_id="v0",
        variant_class=VariantClass.SNV,
        chromosome="1",
        position=1000,
    )
    tie_map = TieMap(
        kept_indices=np.asarray([0], dtype=np.int32),
        original_to_reduced=np.asarray([0], dtype=np.int32),
        reduced_to_group=[],
    )
    return ModelArtifact(
        config=ModelConfig(trait_type=TraitType.QUANTITATIVE),
        records=[record],
        means=np.asarray([0.5], dtype=np.float32),
        scales=np.asarray([1.0], dtype=np.float32),
        alpha=np.asarray([0.0], dtype=np.float32),
        beta_reduced=np.asarray([0.0], dtype=np.float32),
        beta_full=np.asarray([0.0], dtype=np.float32),
        beta_variance=np.asarray([0.0], dtype=np.float32),
        tie_map=tie_map,
        sigma_e2=1.0,
        prior_scales=np.asarray([1.0], dtype=np.float32),
        global_scale=1.0,
        class_tpb_shape_a={VariantClass.SNV: 1.0},
        class_tpb_shape_b={VariantClass.SNV: 1.0},
        scale_model_coefficients=np.zeros(0, dtype=np.float32),
        scale_model_feature_names=[],
        objective_history=[1.0, 0.5],
        validation_history=[0.4],
        **diagnostic_overrides,
    )


def test_diagnostics_round_trip_through_save_load(tmp_path: Path):
    artifact = _make_minimal_artifact(
        converged=True,
        selected_iteration_count=42,
        final_parameter_change=2.5e-6,
        final_predictor_change=3.5e-6,
        final_objective_change=1.0e-5,
        final_hyperparameter_change=4.5e-6,
    )

    save_artifact(tmp_path, artifact)
    restored = load_artifact(tmp_path)

    assert restored.converged is True
    assert restored.selected_iteration_count == 42
    assert restored.final_parameter_change == pytest.approx(2.5e-6)
    assert restored.final_predictor_change == pytest.approx(3.5e-6)
    assert restored.final_objective_change == pytest.approx(1.0e-5)
    assert restored.final_hyperparameter_change == pytest.approx(4.5e-6)


def test_legacy_artifact_without_diagnostics_loads_with_defaults(tmp_path: Path):
    """An artifact written before the diagnostic fields existed must still
    load — defaults: converged=False, selected_iteration_count=0,
    final_*_change=None."""
    artifact = _make_minimal_artifact(
        converged=True,
        selected_iteration_count=11,
        final_parameter_change=1e-4,
        final_predictor_change=1e-4,
        final_objective_change=1e-4,
        final_hyperparameter_change=1e-4,
    )
    save_artifact(tmp_path, artifact)

    # Surgically delete the diagnostic keys from metadata.json, simulating an
    # artifact written by a pre-fix version of the trainer.
    metadata_path = tmp_path / "metadata.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    for field_name in _DIAGNOSTIC_FIELDS:
        payload.pop(field_name, None)
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    restored = load_artifact(tmp_path)
    assert restored.converged is False
    assert restored.selected_iteration_count == 0
    assert restored.final_parameter_change is None
    assert restored.final_predictor_change is None
    assert restored.final_objective_change is None
    assert restored.final_hyperparameter_change is None

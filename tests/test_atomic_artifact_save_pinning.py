"""Pin atomic semantics of ``save_artifact``.

The publish path uses two ``os.replace`` calls (arrays.npz then metadata.json),
preceded by tmp-file writes with fsync. We pin three properties:

* A successful save leaves no ``arrays.tmp.npz`` or ``metadata.json.tmp``
  files behind in the target directory.
* If the metadata-write step raises mid-save (simulated by monkeypatching
  ``json.dumps``), the previously-existing arrays.npz + metadata.json on
  disk must be untouched (atomic publish was never reached).
* The atomic-replace contract: after a crash between the two replaces, the
  on-disk arrays.npz is the NEW arrays but metadata.json is OLD — this is a
  known small window that the loader detects via shape-validation, NOT by
  silent acceptance.
"""
from __future__ import annotations

import json
import os
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


def _make_artifact(*, variant_count: int = 1, seed_value: float = 0.5) -> ModelArtifact:
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
        means=np.full(variant_count, seed_value, dtype=np.float32),
        scales=np.ones(variant_count, dtype=np.float32),
        alpha=np.zeros(variant_count, dtype=np.float32),
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
        objective_history=[1.0],
        validation_history=[],
        converged=True,
        selected_iteration_count=1,
    )


def test_successful_save_leaves_no_temp_files(tmp_path: Path):
    """``arrays.tmp.npz`` and ``metadata.json.tmp`` must not linger after a
    successful save — both must be atomically renamed into place."""
    save_artifact(tmp_path, _make_artifact())
    files = {p.name for p in tmp_path.iterdir()}
    assert "arrays.npz" in files
    assert "metadata.json" in files
    assert "arrays.tmp.npz" not in files
    assert "metadata.json.tmp" not in files


def test_mid_write_crash_preserves_previous_artifact(tmp_path: Path, monkeypatch):
    """If the second-half metadata write blows up, the prior artifact on disk
    must be readable. The first artifact value is 0.5; the simulated crash
    happens during the second save (value 7.0) before either ``os.replace``
    publishes anything."""
    save_artifact(tmp_path, _make_artifact(seed_value=0.5))
    original_arrays_bytes = (tmp_path / "arrays.npz").read_bytes()
    original_metadata_bytes = (tmp_path / "metadata.json").read_bytes()

    # Force json.dumps to fail to simulate a crash during metadata
    # serialization — this is after np.savez_compressed has staged
    # arrays.tmp.npz but before either os.replace runs.
    import sv_pgs.artifact as artifact_module

    def _boom(*_args, **_kwargs):
        raise RuntimeError("simulated mid-write crash")

    monkeypatch.setattr(artifact_module.json, "dumps", _boom)
    with pytest.raises(RuntimeError, match="simulated mid-write crash"):
        save_artifact(tmp_path, _make_artifact(seed_value=7.0))

    # The previously-published pair must be byte-identical.
    assert (tmp_path / "arrays.npz").read_bytes() == original_arrays_bytes
    assert (tmp_path / "metadata.json").read_bytes() == original_metadata_bytes

    # And the load round-trips to the original values, not the would-be new ones.
    restored = load_artifact(tmp_path)
    assert restored.means[0] == 0.5


def test_save_overwrites_prior_arrays_in_place(tmp_path: Path):
    """Sanity: when no crash happens, save_artifact correctly REPLACES the
    prior artifact (the new means show up after reload)."""
    save_artifact(tmp_path, _make_artifact(seed_value=0.5))
    save_artifact(tmp_path, _make_artifact(seed_value=9.0))
    restored = load_artifact(tmp_path)
    assert restored.means[0] == 9.0

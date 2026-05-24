"""Adversarial regression tests for artifact load with malformed inputs.

Extends ``test_artifact_metadata_bugfixes.py`` to pin behaviour against:

* corrupted ``metadata.json`` (invalid JSON, missing required keys)
* shape mismatches between ``arrays.npz`` and ``metadata.json``
* fingerprint matching: whitespace and case sensitivity contracts on
  ``try_load_artifact_if_fingerprint_matches``.

These guards protect the auto-reuse path: a partial-match must NEVER silently
reuse stale coefficients.
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
    try_load_artifact_if_fingerprint_matches,
)
from sv_pgs.config import ModelConfig, TraitType, VariantClass
from sv_pgs.data import TieMap, VariantRecord


def _make_minimal_artifact(*, fingerprint: str = "") -> ModelArtifact:
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
        fit_fingerprint=fingerprint,
        converged=True,
        selected_iteration_count=3,
    )


def test_corrupted_metadata_json_raises_clear_error(tmp_path: Path):
    """A metadata.json with invalid JSON syntax must raise (not silently load
    an empty/partial artifact)."""
    save_artifact(tmp_path, _make_minimal_artifact())
    (tmp_path / "metadata.json").write_text("{not valid json", encoding="utf-8")
    with pytest.raises((json.JSONDecodeError, ValueError)):
        load_artifact(tmp_path)


def test_metadata_missing_required_key_raises(tmp_path: Path):
    """Stripping a required key (sigma_e2) must raise — not silently default it."""
    save_artifact(tmp_path, _make_minimal_artifact())
    metadata_path = tmp_path / "metadata.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload.pop("sigma_e2", None)
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((KeyError, ValueError, TypeError)):
        load_artifact(tmp_path)


def test_mismatched_arrays_shape_vs_metadata_records_raises(tmp_path: Path):
    """If arrays.npz means/beta_full have length 3 but metadata lists 1 record,
    ModelArtifact.__post_init__ must reject the inconsistency."""
    save_artifact(tmp_path, _make_minimal_artifact())
    # Overwrite arrays.npz with three-element arrays while leaving metadata
    # at one record. ModelArtifact validates record-count alignment.
    arrays_path = tmp_path / "arrays.npz"
    original = dict(np.load(arrays_path, allow_pickle=False))
    bad = dict(original)
    bad["means"] = np.zeros(3, dtype=np.float32)
    bad["scales"] = np.ones(3, dtype=np.float32)
    bad["beta_full"] = np.zeros(3, dtype=np.float32)
    bad["beta_reduced"] = np.zeros(3, dtype=np.float32)
    bad["beta_variance"] = np.zeros(3, dtype=np.float32)
    bad["alpha"] = np.zeros(3, dtype=np.float32)
    bad["prior_scales"] = np.ones(3, dtype=np.float32)
    bad["tie_kept_indices"] = np.arange(3, dtype=np.int32)
    bad["tie_original_to_reduced"] = np.arange(3, dtype=np.int32)
    np.savez(arrays_path, **bad)
    with pytest.raises(ValueError):
        load_artifact(tmp_path)


def test_fingerprint_whitespace_is_not_trimmed(tmp_path: Path):
    """Fingerprint comparison is exact string equality — leading/trailing
    whitespace must NOT match. Loosening this would allow a corrupted or
    accidentally-padded fingerprint to silently reuse a stale fit."""
    fp = "abc123def"
    save_artifact(tmp_path, _make_minimal_artifact(fingerprint=fp))
    assert try_load_artifact_if_fingerprint_matches(tmp_path, fp) is not None
    assert try_load_artifact_if_fingerprint_matches(tmp_path, " " + fp) is None
    assert try_load_artifact_if_fingerprint_matches(tmp_path, fp + "\n") is None
    assert try_load_artifact_if_fingerprint_matches(tmp_path, "\t" + fp + " ") is None


def test_fingerprint_is_case_sensitive(tmp_path: Path):
    """SHA-256 hex digests are conventionally lowercase but the comparison
    must be case-sensitive to avoid collisions with hypothetical uppercase
    fingerprints from a different hashing convention."""
    fp = "abc123def"
    save_artifact(tmp_path, _make_minimal_artifact(fingerprint=fp))
    assert try_load_artifact_if_fingerprint_matches(tmp_path, "ABC123DEF") is None
    assert try_load_artifact_if_fingerprint_matches(tmp_path, "Abc123def") is None


def test_empty_fingerprint_never_matches(tmp_path: Path):
    """Sentinel: an unfingerprinted (legacy) artifact must never auto-reuse,
    even when the caller passes an empty string."""
    save_artifact(tmp_path, _make_minimal_artifact(fingerprint=""))
    assert try_load_artifact_if_fingerprint_matches(tmp_path, "") is None


def test_missing_arrays_file_returns_none(tmp_path: Path):
    """Loader-level safety: a directory missing arrays.npz must not raise
    from the auto-reuse helper — it returns None."""
    save_artifact(tmp_path, _make_minimal_artifact(fingerprint="fp"))
    (tmp_path / "arrays.npz").unlink()
    assert try_load_artifact_if_fingerprint_matches(tmp_path, "fp") is None

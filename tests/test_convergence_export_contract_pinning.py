"""Pin the convergence-export gating contract.

Recent change: ``_guard_nonconverged_export`` is now a SOFT guard — it logs a
WARNING with all four delta diagnostics and continues, instead of raising.
Pre-fix behaviour raised RuntimeError which broke legitimate short/small-N
runs. We pin the current behaviour so a future swarm cannot silently revert.

What we pin:

1. The guard does NOT raise when converged=False (soft guard).
2. The emitted log line contains all four delta field names.
3. The guard short-circuits cleanly when converged=True.
4. The guard short-circuits when ``allow_nonconverged_export=True`` is set
   AND logs an override marker.
5. A legacy artifact (loaded via ``BayesianPGS.load`` with the load_artifact
   back-fill of converged=False) does NOT block load — the model is usable.
"""
from __future__ import annotations

import types
from pathlib import Path

import numpy as np

from sv_pgs import pipeline as pipeline_module
from sv_pgs.artifact import save_artifact
from sv_pgs.config import ModelConfig, TraitType, VariantClass
from sv_pgs.data import TieMap, VariantRecord
from sv_pgs.model import BayesianPGS
from sv_pgs.artifact import ModelArtifact


_DELTA_FIELD_NAMES = (
    "final_parameter_change",
    "final_predictor_change",
    "final_objective_change",
    "final_hyperparameter_change",
)


def _make_stub_model(
    *,
    converged: bool,
    deltas: tuple[float | None, ...] = (1e-3, 2e-3, 3e-3, 4e-3),
) -> object:
    fit_result = types.SimpleNamespace(
        converged=converged,
        selected_iteration_count=5,
        final_parameter_change=deltas[0],
        final_predictor_change=deltas[1],
        final_objective_change=deltas[2],
        final_hyperparameter_change=deltas[3],
    )
    state = types.SimpleNamespace(fit_result=fit_result)
    return types.SimpleNamespace(state=state)


def test_nonconverged_export_warns_but_does_not_raise(capsys):
    """SOFT guard: non-converged + allow_nonconverged_export=False must warn,
    NOT raise. This pins the post-swarm relaxation. The warning text must
    actually be emitted (otherwise this guard could silently disappear)."""
    config = ModelConfig(trait_type=TraitType.QUANTITATIVE)
    assert config.allow_nonconverged_export is False
    model = _make_stub_model(converged=False)
    pipeline_module._guard_nonconverged_export(model, config)
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "WARNING" in combined, (
        f"non-converged guard must emit a WARNING line; got: {combined!r}"
    )
    assert "non-converged" in combined.lower() or "did not converge" in combined.lower(), (
        f"non-converged guard message must mention non-convergence; got: {combined!r}"
    )
    for field_name in _DELTA_FIELD_NAMES:
        assert field_name in combined, (
            f"non-converged guard must include {field_name} in its log line; "
            f"got: {combined!r}"
        )


def test_nonconverged_guard_emits_all_four_delta_fields(capsys):
    """The warning must surface all 4 delta diagnostic field names so the
    operator can read the convergence story without re-running."""
    config = ModelConfig(trait_type=TraitType.QUANTITATIVE)
    model = _make_stub_model(
        converged=False, deltas=(1.1e-3, 2.2e-3, 3.3e-3, 4.4e-3),
    )
    pipeline_module._guard_nonconverged_export(model, config)
    captured = capsys.readouterr().out + capsys.readouterr().err
    # All four delta keys must appear in the diagnostic line.
    for field_name in _DELTA_FIELD_NAMES:
        assert field_name in captured, (
            f"non-converged guard must include {field_name} in its log line; "
            f"got: {captured!r}"
        )


def test_converged_fit_short_circuits_without_warning(capsys):
    """When converged=True, the guard returns immediately with no warning."""
    config = ModelConfig(trait_type=TraitType.QUANTITATIVE)
    model = _make_stub_model(converged=True)
    pipeline_module._guard_nonconverged_export(model, config)
    captured = capsys.readouterr().out
    # Either nothing printed, or at most no "did not converge" message.
    assert "did not converge" not in captured.lower()


def test_allow_nonconverged_export_override_logs_distinct_marker(capsys):
    """``allow_nonconverged_export=True`` must take a different code path —
    it must NOT raise AND it should mark the export as user-overridden."""
    config = ModelConfig(
        trait_type=TraitType.QUANTITATIVE, allow_nonconverged_export=True,
    )
    model = _make_stub_model(converged=False)
    pipeline_module._guard_nonconverged_export(model, config)
    captured = capsys.readouterr().out
    # The override branch references the flag name explicitly.
    assert "allow_nonconverged_export" in captured


def test_guard_with_none_state_returns_without_raising():
    """Edge case: model.state is None (no fit yet). Guard must no-op."""
    config = ModelConfig(trait_type=TraitType.QUANTITATIVE)
    model = types.SimpleNamespace(state=None)
    pipeline_module._guard_nonconverged_export(model, config)


def _make_minimal_artifact_with_converged_flag(converged: bool) -> ModelArtifact:
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
        converged=converged,
        selected_iteration_count=2,
    )


def test_legacy_artifact_with_converged_false_loads_into_bayesian_pgs(tmp_path: Path):
    """A legacy artifact with converged=False (back-filled default from
    metadata.json without diagnostic fields) must STILL load via
    ``BayesianPGS.load`` — the load path does not consult the guard."""
    artifact = _make_minimal_artifact_with_converged_flag(converged=False)
    save_artifact(tmp_path, artifact)
    # Must not raise.
    loaded = BayesianPGS.load(tmp_path)
    assert loaded.state is not None
    assert bool(loaded.state.fit_result.converged) is False

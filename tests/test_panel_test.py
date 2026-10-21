"""The pre-registered panel test: per-trait one-sided tests on a frozen primary fold,
Hommel-adjusted, combined by w'z / sqrt(w'Rw) with frozen power weights."""
from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import stats

from sv_pgs import held_out_comparison as comparison, workspace_pipeline as pipeline
from sv_pgs.config import TraitType
from sv_pgs.probability_calibration import frozen_gain_influence, paired_frozen_gain


def test_hommel_matches_its_closed_form_and_never_falls_below_the_raw_p_value():
    # Two hypotheses: Hommel equals Hochberg, max(p) for the larger and min(2 p_min, max p) for the smaller.
    np.testing.assert_allclose(comparison.hommel_adjusted(np.array([0.01, 0.04])), [0.02, 0.04])
    np.testing.assert_allclose(comparison.hommel_adjusted(np.array([0.03, 0.04])), [0.04, 0.04])
    p = np.random.default_rng(1).uniform(size=12)
    adjusted = comparison.hommel_adjusted(p)
    assert np.all(adjusted >= p) and np.all(adjusted <= np.minimum(1.0, p.size * p) + 1e-15)


def test_the_panel_z_is_stouffers_for_independent_traits_and_refuses_negative_weights():
    z = np.array([1.0, 2.0, 3.0])
    assert comparison.panel_z(z, np.ones(3), np.eye(3)) == pytest.approx(6 / np.sqrt(3))
    # Perfectly correlated traits with equal z carry one trait's evidence.
    assert comparison.panel_z(np.full(3, 2.0), np.ones(3), np.ones((3, 3))) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="non-negative"):
        comparison.panel_z(z, np.array([1.0, -1.0, 1.0]), np.eye(3))
    np.testing.assert_allclose(comparison.power_weights(np.array([2.0, -1.0, 2.0])), [0.5, 0.0, 0.5])


def test_the_influences_give_each_statistics_standard_error():
    rng = np.random.default_rng(5)
    labels = (rng.uniform(size=400) < 0.3).astype(float)
    baseline, candidate = rng.normal(size=400) + labels, rng.normal(size=400) + 1.5 * labels
    families = np.arange(400) // 2
    gain = paired_frozen_gain(labels, baseline - 1, candidate - 1, families)
    _estimate, influence = frozen_gain_influence(labels, baseline - 1, candidate - 1)
    totals = comparison.cluster_totals(influence, families)
    assert gain.standard_error == pytest.approx(np.sqrt(totals @ totals))
    auc = comparison.paired_delta_auc(labels, baseline, candidate, families)
    estimate, auc_influence = comparison.delta_auc_influence(labels, baseline, candidate)
    totals = comparison.cluster_totals(auc_influence, families)
    assert estimate == pytest.approx(auc.estimate)
    assert auc.standard_error == pytest.approx(np.sqrt(totals @ totals * 200 / 199))


def _panel_run(tmp_path, monkeypatch, weights, primary_fold=1):
    rng = np.random.default_rng(7)
    n = 3000
    folds = np.arange(n) % 3
    families = np.arange(n) // 2
    liability = rng.normal(size=(n, 3))
    targets = np.column_stack([
        (liability[:, 0] + rng.normal(size=n) > 1.0).astype(float),
        (liability[:, 1] + rng.normal(size=n) > 0.5).astype(float),
        liability[:, 2] + rng.normal(size=n),
    ])
    # The with_sv arm sees more of each liability than the snv_indel arm.
    snv = liability * 0.5 + rng.normal(size=(n, 3))
    sv = liability * 0.9 + rng.normal(size=(n, 3))
    scores_by_arm = np.stack([snv, sv])
    cohort = SimpleNamespace(
        trait_names=("asthma", "gout", "height"), trait_types=(TraitType.BINARY, TraitType.BINARY, TraitType.QUANTITATIVE),
        targets=targets, held_out=np.ones((n, 3), dtype=bool), trait_models=lambda trait: np.array([trait]), covariates=np.ones((n, 1)), covariate_columns=np.ones((3, 1), bool), folds=folds, families=families,
        ancestry=["group"] * n, half=["imputed"] * n, evaluated_folds=lambda trait, fold_count: tuple(range(fold_count)),
    )
    scores = SimpleNamespace(
        held_out_linear_predictor=scores_by_arm, held_out_genetic=scores_by_arm, held_out_calibrated_logits=scores_by_arm - 1.0, held_out_structural=None,
    )
    monkeypatch.setattr(pipeline, "_fit_cohort", lambda run: cohort)
    monkeypatch.setattr(pipeline, "_read_scores", lambda run: scores)
    panel = pipeline.PanelInputs.from_mapping({"primary_fold": primary_fold, "weights": weights}, cohort.trait_names, 3)
    run = SimpleNamespace(
        summary=lambda step: {"quarantined": {}}, budget=SimpleNamespace(host_bytes=1 << 28),
        config=SimpleNamespace(fold_count=3, calibration_fraction=0.1, panel=panel, seed=7, test_fold=None, baselines=lambda: tmp_path / "run" / "baselines"),
    )
    summary = pipeline._report_step(run, tmp_path)
    with (tmp_path / "panel.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return summary, rows, cohort, scores


def test_the_report_writes_the_panel_on_the_primary_fold_and_it_is_exportable(tmp_path, monkeypatch):
    weights = {"asthma": 0.5, "gout": 0.3, "height": 0.2}
    summary, rows, cohort, scores = _panel_run(tmp_path, monkeypatch, weights)
    pipeline.check_exportable(tmp_path / "panel.tsv")
    assert "panel.tsv" in summary["files"] and json.loads((tmp_path / pipeline.EXPORTABLE_FILE).read_text())[-1] == "panel.tsv"
    by_trait = {row["trait"]: row for row in rows}
    assert [row["metric"] for row in rows[:3]] == ["log_likelihood_gain", "log_likelihood_gain", "delta_pgs_partial_r2"]
    # Each trait's z is its paired test on the primary fold alone.
    fold = cohort.folds == 1
    expected = paired_frozen_gain(
        cohort.targets[fold, 0], scores.held_out_calibrated_logits[0, fold, 0], scores.held_out_calibrated_logits[1, fold, 0], cohort.families[fold]
    )
    assert float(by_trait["asthma"]["z"]) == pytest.approx(expected.z_score, rel=1e-5)
    assert by_trait["asthma"]["n"] == "1000"
    z = np.array([float(by_trait[name]["z"]) for name in cohort.trait_names])
    assert np.all(z > 0) and summary["panel"]["panel_z"] > 3
    assert float(by_trait["panel"]["z"]) == pytest.approx(summary["panel"]["panel_z"], rel=1e-5)
    # Unrelated people across traits: R is near the identity, so the panel is near the weighted Stouffer.
    w = np.array([0.5, 0.3, 0.2])
    assert summary["panel"]["panel_z"] == pytest.approx(w @ z / np.sqrt(w @ w), rel=0.1)
    adjusted = comparison.hommel_adjusted(stats.norm.sf(z))
    np.testing.assert_allclose([float(by_trait[name]["p_hommel"]) for name in cohort.trait_names], adjusted, rtol=1e-3)


def test_a_zero_weight_trait_is_tested_but_does_not_move_the_panel(tmp_path, monkeypatch):
    summary, rows, _cohort, _scores = _panel_run(tmp_path, monkeypatch, {"asthma": 1.0, "gout": 0.0, "height": 0.0})
    asthma = next(row for row in rows if row["trait"] == "asthma")
    assert summary["panel"]["panel_z"] == pytest.approx(float(asthma["z"]), rel=1e-5)
    assert all(row["z"] for row in rows if row["trait"] != "panel")


@pytest.mark.parametrize("panel,message", [
    ({"primary_fold": 3, "weights": {"asthma": 1.0}}, "primary_fold"),
    ({"primary_fold": 0, "weights": {"asthma": 1.0}}, "every disease"),
    ({"primary_fold": 0, "weights": {"asthma": -1.0, "gout": 1.0}}, "non-negative"),
    ({"primary_fold": 0, "weights": {"asthma": 0.0, "gout": 0.0}}, "all be 0"),
    ({"primary_fold": 0}, "null or"),
])
def test_the_panel_config_is_refused_unless_frozen_and_complete(panel, message):
    with pytest.raises(ValueError, match=message):
        pipeline.PanelInputs.from_mapping(panel, ("asthma", "gout"), 3)


def test_the_template_freezes_a_panel_over_every_disease():
    from tests.test_workspace_pipeline import LAUNCHER, _filled
    config = pipeline.WorkspaceConfig.from_mapping(json.loads(_filled((LAUNCHER / "run_config.template.json").read_text())))
    assert config.panel is not None and config.panel.primary_fold == 0
    assert [name for name, _weight in config.panel.weights] == list(config.diseases + config.traits)
    assert sum(weight for _name, weight in config.panel.weights) == pytest.approx(1.0, abs=1e-5)

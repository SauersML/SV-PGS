"""Degenerate predictions remain reportable without invented significance."""
from __future__ import annotations

import csv
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import expit

from sv_pgs import held_out_comparison as comparison, workspace_pipeline as pipeline
from sv_pgs.config import TraitType


@pytest.mark.parametrize("metric", [comparison.paired_delta_auc, comparison.paired_delta_r2, comparison.paired_delta_log_loss])
def test_identical_nonconstant_scores_have_no_evidence_of_improvement(metric):
    labels = np.array([0., 1., 0., 1., 0., 1.])
    score = np.array([0., 2., 1., 3., 4., 5.])
    result = metric(labels, score, score.copy())
    assert result == comparison.PairedComparison(0., 0., 0.)


def test_constant_binary_score_fits_prevalence_and_compares_without_crashing():
    labels = np.array([0., 0., 0., 1.])
    score = np.full(4, 7.)
    intercept, slope = comparison.logistic_recalibration_coefficients(score, labels)
    assert slope == 0.
    assert expit(intercept) == pytest.approx(labels.mean())
    assert comparison.paired_delta_log_loss(labels, score, score).z_score == 0.
    assert comparison.paired_delta_auc(labels, score, score).estimate == 0.


def test_zero_auc_variance_with_nonzero_difference_is_not_infinite_evidence():
    labels = np.array([0., 0., 1., 1.])
    result = comparison.paired_delta_auc(labels, -labels, labels)
    assert result.estimate == 1.
    assert result.standard_error == 0.
    assert np.isnan(result.z_score)


def test_auc_with_only_one_case_has_an_estimate_but_no_variance():
    result = comparison.paired_delta_auc(np.array([0., 0., 1.]), np.zeros(3), np.arange(3.))
    assert result.estimate == .5
    assert np.isnan(result.standard_error)
    assert np.isnan(result.z_score)


@pytest.mark.parametrize("labels,score", [
    ([0., 2.], [0., 1.]), ([0., np.nan], [0., 1.]),
    ([0., 1.], [0., np.inf]), ([0., 1.], [[0.], [1.]]),
    ([0., 1.], [0.]), ([], []),
])
@pytest.mark.parametrize("function", [comparison.auc_components, lambda y, s: comparison.logistic_recalibration_coefficients(s, y)])
def test_binary_metrics_reject_misaligned_or_invalid_data(function, labels, score):
    with pytest.raises(ValueError):
        function(np.asarray(labels), np.asarray(score))


def test_family_auc_variance_is_invariant_to_duplicating_each_person_within_family():
    rng = np.random.default_rng(41)
    labels = np.tile([0., 1.], 100)
    baseline = rng.normal(size=200) + labels
    candidate = rng.normal(size=200) + labels
    families = np.arange(200)
    original = comparison.paired_delta_auc(labels, baseline, candidate, families)
    duplicated = comparison.paired_delta_auc(
        np.repeat(labels, 2), np.repeat(baseline, 2), np.repeat(candidate, 2), np.repeat(families, 2),
    )
    independent = comparison.paired_delta_auc(np.repeat(labels, 2), np.repeat(baseline, 2), np.repeat(candidate, 2))
    assert duplicated.estimate == pytest.approx(original.estimate)
    assert duplicated.standard_error == pytest.approx(original.standard_error)
    assert duplicated.standard_error > independent.standard_error


def test_family_auc_includes_cross_case_control_covariance():
    labels = np.array([0., 1., 0., 1., 0., 1.])
    baseline = np.array([0., 2., 1., 3., 4., 5.])
    candidate = np.array([3., 1., 5., 2., 0., 4.])
    families = np.repeat([100, 700, 900], 2)
    # Independent reference: pairwise wins, including half credit for ties.
    def wins(score):
        difference = score[labels == 1, None] - score[None, labels == 0]
        return (difference > 0).astype(float) + .5 * (difference == 0)
    delta = wins(candidate) - wins(baseline)
    influence = np.empty(6)
    influence[labels == 1] = (delta.mean(axis=1) - delta.mean()) / 3
    influence[labels == 0] = (delta.mean(axis=0) - delta.mean()) / 3
    totals = np.array([influence[families == family].sum() for family in np.unique(families)])
    result = comparison.paired_delta_auc(labels, baseline, candidate, families)
    assert result.estimate == pytest.approx(delta.mean())
    assert result.standard_error**2 == pytest.approx(totals @ totals * 3 / 2)


@pytest.mark.parametrize("fold_count", [1, 2])
@pytest.mark.parametrize("separated", [False, True])
def test_report_keeps_degenerate_scores_and_labels_cross_fit_inference(tmp_path, monkeypatch, fold_count, separated):
    n = 200
    labels = np.tile([0., 1.], n // 2)
    score = labels if separated else np.zeros(n)
    cohort = SimpleNamespace(
        trait_names=("disease",), trait_types=(TraitType.BINARY,),
        targets=labels[:, None], covariates=np.ones((n, 1)), covariate_columns=np.ones((1, 1), dtype=bool),
        folds=np.arange(n) // 2 % fold_count, families=np.arange(n) // 2,
        ancestry=["group"] * n, half=["imputed"] * n,
    )
    both = np.stack([score, score])[:, :, None]
    scores = SimpleNamespace(held_out_linear_predictor=both, held_out_genetic=both, held_out_predictive_mean=expit(both))
    monkeypatch.setattr(pipeline, "_read_cohort", lambda run: cohort)
    monkeypatch.setattr(pipeline, "_read_scores", lambda run: scores)
    summary = pipeline._report_step(SimpleNamespace(config=SimpleNamespace(fold_count=fold_count)), tmp_path)
    with (tmp_path / "comparison.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 6
    assert "shared-training covariance" in summary["comparison_inference"]
    for row in rows:
        if fold_count > 1:
            assert row["inference"] == "descriptive_cross_fit"
            assert row["z"] == row["p_value"] == ""
        elif separated and row["metric"] == "log_likelihood_gain":
            assert row["inference"] == "undefined"
            assert row["z"] == row["p_value"] == ""
        else:
            assert float(row["estimate"]) == 0.
            assert float(row["z"]) == 0.
            assert float(row["p_value"]) == 1.

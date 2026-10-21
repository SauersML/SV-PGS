"""Degenerate predictions remain reportable without invented significance."""
from __future__ import annotations

import csv
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import expit

from sv_pgs import held_out_comparison as comparison, workspace_pipeline as pipeline
from sv_pgs.config import TraitType
from sv_pgs.probability_calibration import FrozenCalibration, paired_frozen_gain, proper_scores


@pytest.mark.parametrize("metric", [comparison.paired_delta_auc, comparison.paired_delta_r2, paired_frozen_gain])
def test_identical_nonconstant_scores_have_no_evidence_of_improvement(metric):
    labels = np.array([0., 1., 0., 1., 0., 1.])
    score = np.array([0., 2., 1., 3., 4., 5.])
    result = metric(labels, score, score.copy())
    assert result == comparison.PairedComparison(0., 0., 0.)


def _no_quarantine(step: str) -> dict:
    """The score step's summary as the report step reads it: no trait quarantined."""
    assert step == "score"
    return {"quarantined": {}}


def _report_run(tmp_path, fold_count: int) -> SimpleNamespace:
    """The run as the report step reads it: no trait quarantined, no baseline in its run directory."""
    config = SimpleNamespace(fold_count=fold_count, calibration_fraction=.1, panel=None, seed=7, test_fold=None, baselines=lambda: tmp_path / "run" / "baselines")
    return SimpleNamespace(summary=_no_quarantine, config=config, budget=SimpleNamespace(host_bytes=1 << 28))


def test_constant_binary_score_fits_prevalence_and_compares_without_crashing():
    labels = np.array([0., 0., 0., 1.])
    score = np.full(4, 7.)
    intercept, slope = comparison.logistic_recalibration_coefficients(score, labels)
    assert slope == 0.
    assert expit(intercept) == pytest.approx(labels.mean())
    calibration = FrozenCalibration.fit(score, labels)
    assert calibration.slope == 0. and expit(calibration.intercept) == pytest.approx(labels.mean())
    assert paired_frozen_gain(labels, calibration.logits(score), calibration.logits(score)).z_score == 0.
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
        targets=labels[:, None], held_out=np.ones((n, 1), dtype=bool), trait_models=lambda trait: np.array([0]), covariates=np.ones((n, 1)), covariate_columns=np.ones((1, 1), dtype=bool),
        folds=np.arange(n) // 2 % fold_count, families=np.arange(n) // 2,
        ancestry=["group"] * n, half=["imputed"] * n, evaluated_folds=lambda trait, fold_count: tuple(range(fold_count)),
    )
    both = np.stack([score, score])[:, :, None]
    scores = SimpleNamespace(held_out_linear_predictor=both, held_out_genetic=both, held_out_predictive_mean=expit(both), held_out_calibrated_logits=both, held_out_structural=None)
    monkeypatch.setattr(pipeline, "_read_cohort", lambda run: cohort)
    monkeypatch.setattr(pipeline, "_read_scores", lambda run: scores)
    summary = pipeline._report_step(_report_run(tmp_path, fold_count), tmp_path)
    with (tmp_path / "comparison.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 8
    assert "shared-training covariance" in summary["comparison_inference"]
    for row in rows:
        if fold_count > 1:
            assert row["inference"] == "descriptive_cross_fit"
            assert row["z"] == row["p_value"] == ""
        else:
            assert float(row["estimate"]) == 0.
            assert float(row["z"]) == 0.
            assert float(row["p_value"]) == 1.


def test_frozen_probabilities_include_single_class_folds_without_refitting_calibration(tmp_path, monkeypatch):
    n = 300
    # Fold 0 has no cases: its frozen probabilities are still scored (log loss, Brier), while its AUC is not.
    labels = np.r_[np.zeros(100), np.tile([0., 1.], 100)]
    fold = np.repeat(np.arange(3), 100)
    linear = labels + np.random.default_rng(3).normal(size=n)
    cohort = SimpleNamespace(trait_names=('disease',), trait_types=(TraitType.BINARY,), targets=labels[:, None], held_out=np.ones((n, 1), dtype=bool), trait_models=lambda trait: np.array([0]),
        covariates=np.ones((n, 1)), covariate_columns=np.ones((1, 1), bool), folds=fold, families=np.arange(n),
        ancestry=['group'] * n, half=['imputed'] * n, evaluated_folds=lambda trait, fold_count: tuple(range(fold_count)))
    both = np.stack([linear, linear])[:, :, None]
    scores = SimpleNamespace(held_out_linear_predictor=both, held_out_genetic=both, held_out_calibrated_logits=both.copy(), held_out_structural=None)
    monkeypatch.setattr(pipeline, '_read_cohort', lambda run: cohort)
    monkeypatch.setattr(pipeline, '_read_scores', lambda run: scores)
    pipeline._report_step(_report_run(tmp_path, 3), tmp_path)
    with (tmp_path / 'accuracy.tsv').open() as handle:
        rows = [row for row in csv.DictReader(handle, delimiter='\t') if row['cell'] == 'all' and row['arm'] == 'with_sv']
    metrics = {row['metric']: row for row in rows}
    assert metrics['log_loss']['n'] == str(n) and metrics['auc']['n'] == '200'
    frozen = linear.copy()
    assert np.isfinite(frozen).all()
    for metric in ('log_loss', 'brier'):
        assert float(metrics[metric]['value']) == pytest.approx(proper_scores(labels, frozen)[metric].mean(), rel=1e-5)
    assert float(metrics['auc']['value']) == pytest.approx(
        np.mean([comparison.auc_components(labels[fold == index], linear[fold == index]).auc for index in (1, 2)]), rel=1e-5
    )
    # the genetic score's partial r^2 on the trait's covariates, the LDpred2 baseline's measure (aou_ldpred2.py)
    assert float(metrics['pgs_partial_r2']['value']) == pytest.approx(
        np.mean([np.corrcoef(labels[fold == index], linear[fold == index])[0, 1] ** 2 for index in (1, 2)]), rel=1e-5
    )
    slopes = [comparison.logistic_recalibration_coefficients(frozen[fold == index], labels[fold == index])[1] for index in (1, 2)]
    assert float(metrics['calibration_slope']['value']) == pytest.approx(np.mean(slopes), rel=1e-5)


def test_a_cells_rows_show_no_small_count_by_their_difference_and_stay_exportable(tmp_path, monkeypatch):
    def group(fold_cases, fold_size):
        labels = np.concatenate([np.r_[np.ones(cases), np.zeros(fold_size - cases)] for cases in fold_cases])
        return labels, np.repeat(np.arange(len(fold_cases)), fold_size)
    big_labels = np.tile([0., 1.], 222)
    # "small": its fold 0 has one case, so AUC leaves it out; frozen probabilities reading it would show one more
    # case than AUC in the same cell. "thin": the folds AUC reads hold 20 cases, too few to show.
    small_labels, small_folds = group((1, 10, 12), 30)
    thin_labels, thin_folds = group((1, 9, 11), 22)
    labels = np.r_[big_labels, small_labels, thin_labels]
    fold = np.r_[np.arange(444) % 3, small_folds, thin_folds]
    ancestry = ['big'] * 444 + ['small'] * 90 + ['thin'] * 66
    n = labels.shape[0]
    linear = labels + np.random.default_rng(8).normal(size=n)
    cohort = SimpleNamespace(trait_names=('disease',), trait_types=(TraitType.BINARY,), targets=labels[:, None], held_out=np.ones((n, 1), dtype=bool), trait_models=lambda trait: np.array([0]),
        covariates=np.ones((n, 1)), covariate_columns=np.ones((1, 1), bool), folds=fold, families=np.arange(n),
        ancestry=ancestry, half=['imputed'] * n, evaluated_folds=lambda trait, fold_count: tuple(range(fold_count)))
    both = np.stack([linear, linear])[:, :, None]
    scores = SimpleNamespace(held_out_linear_predictor=both, held_out_genetic=both, held_out_calibrated_logits=both.copy(), held_out_structural=None)
    monkeypatch.setattr(pipeline, '_read_cohort', lambda run: cohort)
    monkeypatch.setattr(pipeline, '_read_scores', lambda run: scores)
    pipeline._report_step(_report_run(tmp_path, 3), tmp_path)
    for name in ('accuracy.tsv', 'comparison.tsv'):
        pipeline.check_exportable(tmp_path / name)
        with (tmp_path / name).open() as handle:
            rows = list(csv.DictReader(handle, delimiter='\t'))
        for level in ('small', 'thin'):
            cell = [row for row in rows if row['level'] == level]
            assert cell and len({(row['n'], row['cases']) for row in cell}) == 1
        assert {row['n'] for row in rows if row['level'] == 'small'} == {'60'}
        assert not any(row.get('value') or row.get('estimate') for row in rows if row['level'] == 'thin')

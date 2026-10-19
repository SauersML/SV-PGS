"""The power study's planning calculator (benchmarks/power/expected.py) and its simulation grid's controlled scenarios
(benchmarks/power/grid.py, on bench-sim's truth generator)."""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from benchmarks.bench_sim import truth
from benchmarks.power import expected, grid


def _architecture(**changes) -> expected.Architecture:
    base = expected.Architecture(
        name="disease", prevalence=0.1, heritability=0.2, sv_share=0.0, tagged_by_small=0.0, sv_imputation_r2=0.8,
        causal_variants=10_000, effective_segments=60_000, training_people=80_000, test_people=20_000,
    )
    return replace(base, **changes)


def test_the_auc_rises_with_the_variance_explained_from_one_half() -> None:
    assert expected.auc_of(0.0, 0.1) == 0.5
    values = [expected.auc_of(variance, 0.1) for variance in (0.01, 0.05, 0.1, 0.2)]
    assert all(0.5 < value < 1.0 for value in values) and values == sorted(values)


def test_the_sv_share_follows_the_causal_fraction_and_effect_fold() -> None:
    assert expected.sv_share_of(0.0, 3.0) == 0.0 and expected.sv_share_of(1.0, 1.0) == 1.0
    # 10% of causal variants at three times the effect SD: 0.9 / (0.9 + 0.9) of the heritability.
    assert expected.sv_share_of(0.1, 3.0) == pytest.approx(0.5)


def test_without_sv_heritability_the_arms_capture_the_same_and_the_test_holds_its_size() -> None:
    architecture = _architecture()
    small, with_sv = expected.captured_heritability(architecture)
    assert small == with_sv == pytest.approx(0.2)
    result = expected.expectation(architecture, replicates=40, seed=1, significance=0.05, tests=15)
    assert result.small_auc == result.sv_auc and abs(result.mean_delta_auc_simulated) < 1e-3 and result.power <= 0.2


def test_untagged_sv_heritability_gives_the_sv_arm_more_accuracy_and_power() -> None:
    architecture = _architecture(sv_share=0.3, tagged_by_small=0.2, test_people=50_000)
    small, with_sv = expected.captured_heritability(architecture)
    assert with_sv - small == pytest.approx(0.2 * 0.3 * (0.8 - 0.2))
    result = expected.expectation(architecture, replicates=20, seed=2, significance=0.05, tests=15)
    assert result.sv_auc > result.small_auc and result.mean_delta_auc_simulated > 0.0 and result.power > 0.5
    # SVs the small variants already tag better than they are imputed add nothing.
    assert expected.captured_heritability(replace(architecture, tagged_by_small=0.9))[0] == expected.captured_heritability(replace(architecture, tagged_by_small=0.9))[1]


def _cohort(directory, variant_count=400, size=300, seed=0):
    """A tiny bench-sim cohort chromosome: a quarter of the records SVs (half deletions), the rest SNVs."""
    generator = np.random.default_rng(seed)
    classes = np.where(np.arange(variant_count) % 4 == 0, truth.SV, truth.SNV)
    frequency = generator.uniform(0.05, 0.5, variant_count)
    genotypes = generator.binomial(2, frequency[:, None], size=(variant_count, size)).astype(np.int8)
    length_change = np.where(classes == truth.SV, np.where(np.arange(variant_count) % 8 == 0, -1, 1) * generator.integers(50, 50_000, variant_count), 0)
    np.savez(directory / "variants.npz", cls=classes, len_change=length_change, superpop_af=np.tile(frequency, (5, 1)), donor_an=np.int64(1000))
    np.savez(
        directory / "annotations.npz", in_gene=np.zeros(variant_count, bool), in_exon=np.zeros(variant_count, bool),
        in_repeat=np.zeros(variant_count, bool), log_tss_distance=np.zeros(variant_count),
        log_sv_length=np.where(classes == truth.SV, np.log(np.maximum(np.abs(length_change), 1)), 0.0), ld_score=np.ones(variant_count),
        repeat_locus=np.full(variant_count, -1),
    )
    np.save(directory / "truth_G.npy", genotypes)
    np.savez(directory / "samples.npz", age=generator.uniform(18, 80, size), sex=generator.integers(0, 2, size), batch=generator.integers(0, 2, size),
             realized_proportions=np.tile([1.0, 0, 0, 0, 0], (size, 1)))
    return classes, length_change


def test_a_point_sets_the_expected_causal_count_and_sv_fraction(tmp_path) -> None:
    classes, _change = _cohort(tmp_path)
    parameters = grid.scenario_parameters("sv_20pct", 0, tmp_path)
    small_probability = parameters["pi"]
    odds = small_probability / (1 - small_probability) * parameters["sv_probability_fold"]
    structural_probability = odds / (1 + odds)
    expected_structural = structural_probability * np.sum(classes == truth.SV)
    expected_small = small_probability * np.sum(classes != truth.SV)
    assert expected_structural + expected_small == pytest.approx(grid.CAUSAL_VARIANTS)
    assert expected_structural / (expected_structural + expected_small) == pytest.approx(0.2)
    null = grid.scenario_parameters("null", 0, tmp_path)
    assert null["sv_probability_fold"] == 0.0


def test_the_null_point_makes_no_sv_causal_and_the_controls_leave_pre_registered_scenarios_unchanged(tmp_path) -> None:
    classes, _change = _cohort(tmp_path)
    parameters = grid.scenario_parameters("null", 3, tmp_path)
    with np.errstate(divide="ignore"):
        _value, result = truth.genetic_value(parameters, tmp_path)
    assert not np.any(classes[result["effects"]["causal"]] == truth.SV)
    # The power study's knobs at their defaults are the pre-registered generator exactly.
    drawn = truth.draw_parameters(5)
    original, _ = truth.genetic_value(drawn, tmp_path)
    with_defaults, _ = truth.genetic_value({**drawn, "sv_log_length_slope": 0.0, "sv_deletion_scale_fold": 1.0}, tmp_path)
    np.testing.assert_array_equal(original, with_defaults)


def test_the_positive_controls_move_sv_effects_with_length_and_class(tmp_path) -> None:
    classes, change = _cohort(tmp_path, variant_count=4_000, size=200)
    parameters = {**grid.scenario_parameters("sv_20pct", 0, tmp_path), "pi": 0.5, "sv_probability_fold": 1.0}
    lengths = np.log(np.maximum(np.abs(change), 1))

    def structural_effects(settings):
        _value, result = truth.genetic_value(settings, tmp_path)
        causal = result["effects"]["causal"]
        rows = classes[causal] == truth.SV
        return causal[rows], np.abs(result["effects"]["per_allele"][rows])

    rows, magnitude = structural_effects({**parameters, "sv_log_length_slope": 2.0})
    flat_rows, flat_magnitude = structural_effects(parameters)
    assert np.corrcoef(lengths[rows], np.log(magnitude))[0, 1] > np.corrcoef(lengths[flat_rows], np.log(flat_magnitude))[0, 1] + 0.3
    rows, magnitude = structural_effects({**parameters, "sv_deletion_scale_fold": 4.0})
    deletion = change[rows] < 0
    assert np.median(magnitude[deletion]) > 2.0 * np.median(magnitude[~deletion])


@pytest.mark.parametrize('changes', [
    {'training_people': 0}, {'test_people': 2.5}, {'causal_variants': 0},
    {'effective_segments': float('inf')}, {'sv_share': float('nan')},
])
def test_planning_rejects_invalid_architectures(changes):
    with pytest.raises(ValueError):
        _architecture(**changes)


def test_planning_handles_zero_captured_baseline_and_counts_unusable_replicates():
    result = expected.expectation(_architecture(sv_share=1.0, tagged_by_small=0.0), 3, 11, .05, 15)
    assert result.small_auc == .5 and result.sv_auc > .5
    assert result.valid_replicates + result.invalid_replicates == 3
    result = expected.expectation(_architecture(prevalence=1e-12, test_people=2), 4, 1, .05, 15)
    assert result.invalid_replicates == 4 and np.isnan(result.power)


def test_monte_carlo_interval_is_informative_at_zero_successes():
    from benchmarks.power.uncertainty import rejection_summary
    result = rejection_summary(0, 20)
    assert result['rate'] == 0 and result['lower'] == pytest.approx(0)
    assert .1 < result['upper'] < .2
    result = rejection_summary(20, 20)
    assert .8 < result['lower'] < .9 and result['upper'] == pytest.approx(1)


def test_summary_reports_invalid_comparisons_separately(tmp_path):
    import csv
    import json
    for replicate, z in enumerate([2.0, 0.0, float('nan')]):
        path = tmp_path / 'tasks' / 'null' / 'n5000' / f'replicate{replicate}'
        path.mkdir(parents=True)
        (path / 'result.json').write_text(json.dumps(dict(
            point='null', train_count=5000, replicate=replicate, z=z, delta_auc=.01,
            snv_indel_auc=.6, with_sv_auc=.61, length_rho=0., deletion_over_insertion=1.)))
    with grid.summarize(tmp_path, tmp_path / 'cohort').open() as handle:
        row, = list(csv.DictReader(handle, delimiter='\t'))
    assert row['valid_comparisons'] == '2' and row['invalid_comparisons'] == '1'
    assert float(row['share_favouring_sv']) == .5
    assert float(row['rate_lower_95']) < .5 < float(row['rate_upper_95'])


def test_the_comparison_waits_for_the_core_arms_and_adds_the_all_variant_ldpred2_arm_when_it_completes(tmp_path):
    import json
    rng = np.random.default_rng(0)
    cohort, scenario = tmp_path / 'cohort', tmp_path / 'scenarios' / 'null' / 'replicate0'
    cohort.mkdir()
    scenario.mkdir(parents=True)
    is_test = np.arange(400) >= 100
    labels = rng.random(400) < 0.3
    np.savez(cohort / 'samples.npz', is_test=is_test)
    np.savez(scenario / 'truth.npz', phenotype=labels)
    (scenario / 'scenario.json').write_text(json.dumps(dict(summary=dict(structural_share=0.0))))
    task = grid._task(tmp_path, 'null', 5000, 0)

    def finish(arm, signal, **certification):
        folder = task / arm
        folder.mkdir(parents=True)
        np.savez(folder / 'prediction.npz', total=signal * labels[is_test] + rng.normal(size=is_test.sum()))
        (folder / 'meta.json').write_text(json.dumps(dict(fit_seconds=1.0, **certification)))
        (folder / 'attempt.json').write_text(json.dumps(dict(point='null', train_count=5000, replicate=0, arm=arm, status='completed')))
        if arm == 'with_sv':
            (folder / 'sv_readout.json').write_text(json.dumps(dict(length_rho=0.0)))

    finish('snv_indel', 1.0, certified=False, certification_issues=['outer_criterion_met'])
    finish('with_sv', 1.5, certified=True, certification_issues=[])
    assert grid.compare(cohort, tmp_path, 'null', 5000, 0) is None
    finish('ldpred2_auto', 0.5)
    first = grid.compare(cohort, tmp_path, 'null', 5000, 0)
    assert 'ldpred2_allvar_auc' not in first and first['with_sv_auc'] > first['ldpred2_auc']
    # an uncertified fit is scored and named; LDpred2 records no certification
    assert first['uncertified_arms'] == ['snv_indel']
    finish('ldpred2_allvar', 1.0)
    second = grid.compare(cohort, tmp_path, 'null', 5000, 0)
    assert second['with_sv_auc'] == first['with_sv_auc'] and np.isfinite(second['z_vs_ldpred2_allvar'])
    assert json.loads((task / 'result.json').read_text()) == second


def test_a_group_fit_records_each_point_on_its_own_and_leaves_out_completed_arms(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    class Fitted:
        def score(self, test):
            return {"total": np.zeros(3)}

    fitted_columns = []

    def fit_phenotypes(train, phenotypes):
        fitted_columns.append(phenotypes.shape[1])
        return [Fitted(), ValueError("Model cannot be scored: fit_failed")]

    views = {"null": 0.0, "sv_5pct": 1.0, "sv_20pct": 2.0}
    monkeypatch.setattr(grid, "_scenario", lambda cohort, out, point, replicate: tmp_path / "scenarios" / point)
    monkeypatch.setattr(grid.harness, "views", lambda cohort, scenario, *rest: (
        SimpleNamespace(phenotype=np.full(4, views[scenario.name]), n_samples=4), SimpleNamespace(n_samples=3)))
    monkeypatch.setattr(grid.harness, "load_method", lambda method: SimpleNamespace(fit_phenotypes=fit_phenotypes))
    monkeypatch.setattr(grid, "compare", lambda *args: None)
    done = grid._task(tmp_path, "null", 5000, 0) / "snv_indel"
    done.mkdir(parents=True)
    (done / "attempt.json").write_text(json.dumps(dict(status="completed")))
    grid.run_group(tmp_path / "method.py", tmp_path / "cohort", tmp_path, ["null", "sv_5pct", "sv_20pct"], 5000, 0, "snv_indel", 1)
    assert fitted_columns == [2]
    accepted, quarantined = (json.loads((grid._task(tmp_path, point, 5000, 0) / "snv_indel" / "attempt.json").read_text()) for point in ("sv_5pct", "sv_20pct"))
    assert accepted["status"] == "completed" and accepted["fitted_with"] == ["sv_5pct", "sv_20pct"]
    assert quarantined["status"] == "failed" and "cannot be scored" in quarantined["error"]
    meta = json.loads((grid._task(tmp_path, "sv_5pct", 5000, 0) / "snv_indel" / "meta.json").read_text())
    assert meta["models_fitted_together"] == 2 and meta["train_count"] == 4
    assert json.loads((done / "attempt.json").read_text()) == dict(status="completed")
    with pytest.raises(ValueError, match="submitted method"):
        grid.run_group(tmp_path / "method.py", tmp_path / "cohort", tmp_path, ["null"], 5000, 0, "ldpred2_auto", 1)


@pytest.mark.parametrize("failure", ["views", "missing_outcome", "extra_outcome", "score", "readout"])
def test_group_failures_are_recorded_without_discarding_other_models(tmp_path, monkeypatch, failure):
    import json
    from types import SimpleNamespace

    class Fitted:
        def __init__(self, index):
            self.index = index

        def score(self, test):
            if failure == "score" and self.index == 0:
                raise RuntimeError("scoring failed")
            return {"total": np.arange(test.n_samples, dtype=float)}

    def views(*args):
        if failure == "views":
            raise RuntimeError("view failed")
        return SimpleNamespace(phenotype=np.zeros(4), n_samples=4), SimpleNamespace(n_samples=3)

    def readout(model, *args):
        if failure == "readout" and model.index == 0:
            raise RuntimeError("effect readout failed")
        return {}

    count = {"missing_outcome": 1, "extra_outcome": 3}.get(failure, 2)
    monkeypatch.setattr(grid, "_scenario", lambda *args: tmp_path / "scenario")
    monkeypatch.setattr(grid.harness, "views", views)
    monkeypatch.setattr(grid.harness, "load_method", lambda *args: SimpleNamespace(
        fit_phenotypes=lambda *args: [Fitted(index) for index in range(count)]))
    monkeypatch.setattr(grid, "structural_effect_readout", readout)
    compared = []
    monkeypatch.setattr(grid, "compare", lambda cohort, out, point, *args: compared.append(point))
    with pytest.raises((ValueError, RuntimeError, ExceptionGroup)):
        grid.run_group(tmp_path / "method.py", tmp_path / "cohort", tmp_path, ["null", "sv_20pct"], 5000, 0, "with_sv", 1)
    attempts = [json.loads((grid._task(tmp_path, point, 5000, 0) / "with_sv" / "attempt.json").read_text())
                for point in ("null", "sv_20pct")]
    assert attempts[0]["status"] == "failed"
    if failure in ("score", "readout"):
        assert attempts[1]["status"] == "completed"
        assert compared == ["null", "sv_20pct"]
    else:
        assert attempts[1]["status"] == "failed"


@pytest.mark.parametrize("prediction", [{}, {"total": [0.0]}, {"total": [0.0, np.nan, 1.0]},
                                        {"total": [0.0, 0.5, 1.0], "structural": [np.inf, 0.0, 1.0]}])
def test_invalid_predictions_are_not_written(tmp_path, prediction):
    from types import SimpleNamespace
    with pytest.raises(ValueError, match="prediction|Prediction"):
        grid.harness.write_prediction(SimpleNamespace(score=lambda test: prediction), SimpleNamespace(n_samples=3), tmp_path, 1.0)
    assert not (tmp_path / "prediction.npz").exists()
    assert not (tmp_path / "meta.json").exists()


def test_failed_tasks_are_counted_even_if_no_fit_completes(tmp_path, monkeypatch):
    import csv
    def failed(*args, **kwargs):
        raise ValueError('synthetic rejected fit')
    monkeypatch.setattr(grid, '_scenario', lambda *args: tmp_path / 'scenario')
    monkeypatch.setattr(grid.harness, 'run', failed)
    with pytest.raises(ValueError, match='rejected fit'):
        grid.run(tmp_path / 'method.py', tmp_path / 'cohort', tmp_path, 'null', 5000, 0, 'with_sv', 1)
    with grid.summarize(tmp_path, tmp_path / 'cohort').open() as handle:
        row, = list(csv.DictReader(handle, delimiter='\t'))
    assert row['failed_tasks'] == '1' and row['replicates'] == '0'
    assert row['valid_comparisons'] == '0' and row['share_favouring_sv'] == 'nan'


def test_unshared_arm_error_lowers_the_paired_power_and_leaves_the_expected_aucs():
    architecture = _architecture(sv_share=0.05, tagged_by_small=0.3)
    shared = expected.expectation(architecture, 40, 3, .05, 15)
    assert expected.expectation(architecture, 40, 3, .05, 15, arm_error_correlation=1.0) == shared
    unshared = expected.expectation(architecture, 40, 3, .05, 15, arm_error_correlation=0.5)
    assert (unshared.small_auc, unshared.sv_auc) == (shared.small_auc, shared.sv_auc)
    assert unshared.power <= shared.power
    with pytest.raises(ValueError, match="correlation"):
        expected.expectation(architecture, 4, 3, .05, 15, arm_error_correlation=1.5)


def test_grid_varies_polygenicity_and_includes_joint_and_off_distribution_controls(tmp_path):
    classes, _ = _cohort(tmp_path, variant_count=4000, size=20)
    for point, expected_count in (('sparse_sv', 50), ('polygenic_sv', 2000)):
        params = grid.scenario_parameters(point, 0, tmp_path)
        odds = grid._odds(params['pi']) * params['sv_probability_fold']
        count = params['pi'] * (classes != truth.SV).sum() + odds / (1 + odds) * (classes == truth.SV).sum()
        assert count == pytest.approx(expected_count)
    joint = grid.scenario_parameters('multi_feature_positive', 0, tmp_path)
    assert joint['sv_log_length_slope'] > 0 and joint['sv_deletion_scale_fold'] > 1 and joint['annotation_mode'] == 'scale'
    heavy = grid.scenario_parameters('sv_heavy_tail', 0, tmp_path)
    assert heavy['sv_shape'] == 'student_t' and heavy['sv_own_shape']
    assert grid.scenario_parameters('rare_disease', 0, tmp_path)['prevalence'] == .01


def test_a_task_observes_its_records_through_the_chosen_measurement_arm(tmp_path, monkeypatch):
    seen = []

    def stopped(method, cohort, scenario, folder, cores, measurement, *rest, **kwargs):
        seen.append(measurement)
        raise ValueError('synthetic stop')
    monkeypatch.setattr(grid, '_scenario', lambda *args: tmp_path / 'scenario')
    monkeypatch.setattr(grid.harness, 'run', stopped)
    with pytest.raises(ValueError, match='synthetic stop'):
        grid.run(tmp_path / 'method.py', tmp_path / 'cohort', tmp_path, 'null', 5000, 0, 'with_sv', 1, measurement='truth')
    attempt = json.loads((tmp_path / 'tasks' / 'null' / 'n5000' / 'replicate0' / 'with_sv' / 'attempt.json').read_text())
    assert seen == ['truth'] and attempt['measurement'] == 'truth' and attempt['status'] == 'failed'


def test_a_fit_s_certification_reaches_its_meta_and_a_method_without_one_records_nothing():
    from types import SimpleNamespace
    from benchmarks.bench_sim import harness
    verdict = {"accepted": False, "issues": [{"code": "outer_criterion_met", "severity": "fatal"},
                                             {"code": "approximate_inference", "severity": "info"}]}
    assert harness.certification(SimpleNamespace(profile={"acceptance": verdict})) == {
        "certified": False, "certification_issues": ["outer_criterion_met"]}
    assert harness.certification(SimpleNamespace()) == {}

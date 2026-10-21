"""The All of Us LDpred2 baseline reads the run's store, rows and covariates as the fit and the report do."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.bench_real import aou_compare, aou_ldpred2
from sv_pgs.config import TraitType
from sv_pgs.held_out_comparison import auc_components, paired_delta_auc, paired_delta_r2
from sv_pgs.workspace_pipeline import _partial_r2, _residualized
from tests.test_full_data_fit import _SAMPLES, _store


def _cohort(generator: np.random.Generator, held_out_count: int) -> SimpleNamespace:
    """A one-trait cohort over every store column in shuffled order, two structure covariates and one of the trait's own
    beside another trait's, and one model whose last ``held_out_count`` rows are held out."""
    columns = generator.permutation(_SAMPLES)
    held_out = np.zeros((_SAMPLES, 1), dtype=bool)
    held_out[-held_out_count:] = True
    return SimpleNamespace(
        store_columns=columns, covariates=generator.normal(size=(_SAMPLES, 4)), covariate_columns=np.array([[True, True, True, False]]),
        targets=generator.normal(size=(_SAMPLES, 1)), training=~held_out, held_out=held_out, trait_names=("trait",),
        all_rows_model=lambda trait: 0,
    )


def test_the_view_reads_its_records_over_its_columns_and_offsets_each_chromosome(tmp_path: Path) -> None:
    store = _store(tmp_path / "store", 3)[0]
    records = np.r_[np.arange(5, 40), np.arange(41, 60, 3), np.arange(175, 200)]
    columns = np.sort(np.random.default_rng(1).choice(_SAMPLES, size=120, replace=False))
    view = aou_ldpred2.StoreView(store, records, columns, np.zeros((120, 1)), np.zeros(120))
    rows = np.r_[np.arange(30, 50), 2, 3]
    np.testing.assert_array_equal(view.codes(rows), store.read_codes(0, store.n_variants)[records[rows]][:, columns])
    table = store.variant_table
    cm = view.variants["cm"]
    assert np.all(np.diff(cm) > 0.0)
    on_second = np.asarray(table.chromosome)[records] != np.asarray(table.chromosome)[records[0]]
    assert cm[on_second].min() - cm[~on_second].max() > aou_ldpred2.CHROMOSOME_SPACING_CM / 2


def test_a_trait_view_pairs_each_store_column_with_its_row(tmp_path: Path) -> None:
    store = _store(tmp_path / "store", 3)[0]
    cohort = _cohort(np.random.default_rng(2), 60)
    view = aou_ldpred2.cohort_view(store, np.arange(10), cohort, cohort.held_out[:, 0], 0)
    assert np.all(np.diff(view.columns) > 0)
    rows = np.flatnonzero(cohort.held_out[:, 0])
    order = np.argsort(cohort.store_columns[rows])
    np.testing.assert_array_equal(view.columns, cohort.store_columns[rows][order])
    np.testing.assert_array_equal(view.phenotype, cohort.targets[rows, 0][order])
    np.testing.assert_array_equal(view.covariates, cohort.covariates[rows][order][:, :3])


def test_the_held_out_score_and_its_partial_r2_are_the_dense_ones(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _store(tmp_path / "store", 3)[0]
    generator = np.random.default_rng(4)
    cohort = _cohort(generator, 100)
    records = np.r_[np.arange(0, 50), np.arange(150, 260)]
    effects = generator.normal(size=records.shape[0])
    monkeypatch.setattr(aou_ldpred2.ldpred2_auto_allvar, "CHUNK", 64)
    monkeypatch.setattr(
        aou_ldpred2.ldpred2_auto_allvar, "posterior_effects",
        lambda train, rows: (train.codes(rows).astype(np.float64).mean(axis=1) / 127.0, effects),
    )
    result, rows, scores = aou_ldpred2.ldpred2_trait(store, records, cohort, 0)
    dosage = store.read_codes(0, store.n_variants)[records].astype(np.float64).T / 127.0
    training, held_out = np.flatnonzero(cohort.training[:, 0]), np.flatnonzero(cohort.held_out[:, 0])
    score = (dosage[cohort.store_columns[held_out]] - dosage[cohort.store_columns[training]].mean(axis=0)) @ effects
    expected = _partial_r2(cohort.targets[held_out, 0], score, cohort.covariates[held_out][:, :3])
    assert result == {"trait": "trait", "held_out_rows": 100, "variants": records.shape[0], "pgs_partial_r2": pytest.approx(expected, rel=1e-9)}
    # Each held-out person's own score, by cohort row.
    assert sorted(rows.tolist()) == held_out.tolist()
    by_row = dict(zip(rows.tolist(), scores))
    np.testing.assert_allclose([by_row[row] for row in held_out.tolist()], score, rtol=1e-9)


def test_a_held_out_count_under_the_workspace_threshold_is_not_shown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _store(tmp_path / "store", 3)[0]
    cohort = _cohort(np.random.default_rng(5), aou_ldpred2.SMALLEST_SHOWN_COUNT - 1)
    monkeypatch.setattr(aou_ldpred2.ldpred2_auto_allvar, "posterior_effects", lambda train, rows: (np.zeros(rows.shape[0]), np.ones(rows.shape[0])))
    result, rows, _scores = aou_ldpred2.ldpred2_trait(store, np.arange(30), cohort, 0)
    assert result["held_out_rows"] == f"<{aou_ldpred2.SMALLEST_SHOWN_COUNT}" and result["pgs_partial_r2"] is None and rows.shape == (0,)


@pytest.mark.parametrize("count", [0, 20])
def test_empty_or_suppressed_evaluation_does_not_fit_ldpred2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int) -> None:
    store = _store(tmp_path / "store", 3)[0]
    cohort = _cohort(np.random.default_rng(5), max(count, 1))
    if count == 0:
        cohort.held_out[:] = False
        cohort.training[:] = True

    def unexpected_fit(*args):
        pytest.fail("an empty or suppressed held-out cell must not launch LDpred2")

    monkeypatch.setattr(aou_ldpred2.ldpred2_auto_allvar, "posterior_effects", unexpected_fit)
    result, rows, scores = aou_ldpred2.ldpred2_trait(store, np.arange(30), cohort, 0)
    assert result["held_out_rows"] == (0 if count == 0 else "<21") and result["pgs_partial_r2"] is None
    assert rows.shape == scores.shape == (0,)


def test_the_binary_ldpred2_row_uses_the_workspace_case_suppression(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from sv_pgs.config import TraitType

    store = _store(tmp_path / "store", 3)[0]
    cohort = _cohort(np.random.default_rng(5), 100)
    cohort.trait_types = (TraitType.BINARY,)
    cohort.targets[:] = 0.0
    cohort.targets[-20:] = 1.0
    monkeypatch.setattr(aou_ldpred2.ldpred2_auto_allvar, "posterior_effects", lambda *args: pytest.fail("suppressed cell"))
    result, rows, _scores = aou_ldpred2.ldpred2_trait(store, np.arange(30), cohort, 0)
    assert result["held_out_rows"] == 100 and result["cases"] == "suppressed" and result["pgs_partial_r2"] is None and rows.shape == (0,)
    # 40 cases and 60 controls are both shown.
    cohort.targets[-40:] = 1.0
    monkeypatch.setattr(aou_ldpred2.ldpred2_auto_allvar, "posterior_effects", lambda train, rows: (np.zeros(rows.shape[0]), np.ones(rows.shape[0])))
    result, rows, _scores = aou_ldpred2.ldpred2_trait(store, np.arange(30), cohort, 0)
    assert result["cases"] == "40" and result["pgs_partial_r2"] is not None and sorted(rows.tolist()) == np.flatnonzero(cohort.held_out[:, 0]).tolist()


def test_cross_fit_baseline_uses_each_held_out_models_training_rows_and_fold_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _store(tmp_path / "store", 3)[0]
    cohort = _cohort(np.random.default_rng(6), 100)
    cohort.held_out = np.zeros((_SAMPLES, 3), dtype=bool)
    cohort.held_out[:_SAMPLES // 2, 0] = True
    cohort.held_out[_SAMPLES // 2:, 1] = True
    cohort.training = ~cohort.held_out
    cohort.trait_models = lambda trait: np.arange(3)
    cohort.all_rows_model = lambda trait: 2
    records = np.arange(30)
    effects = np.linspace(-1.0, 1.0, records.size)
    trained = []

    def fit(train, rows):
        trained.append(train.columns.copy())
        return train.codes(rows).astype(np.float64).mean(axis=1) / 127.0, effects

    monkeypatch.setattr(aou_ldpred2.ldpred2_auto_allvar, "posterior_effects", fit)
    result, rows, scores = aou_ldpred2.ldpred2_trait(store, records, cohort, 0)
    by_row = dict(zip(rows.tolist(), scores))
    values = []
    for model in (0, 1):
        np.testing.assert_array_equal(trained[model], np.sort(cohort.store_columns[cohort.training[:, model]]))
        train = aou_ldpred2.cohort_view(store, records, cohort, cohort.training[:, model], 0)
        test = aou_ldpred2.cohort_view(store, records, cohort, cohort.held_out[:, model], 0)
        means = train.codes(records).astype(np.float64).mean(axis=1) / 127.0
        score = (test.codes(records).astype(np.float64).T / 127.0 - means) @ effects
        values.append((test.n_samples, _partial_r2(test.phenotype, score, test.covariates)))
        # Each held-out person's score is the one of the model that left them out.
        np.testing.assert_allclose([by_row[row] for row in aou_ldpred2.cohort_rows(cohort, cohort.held_out[:, model]).tolist()], score, rtol=1e-9)
    expected = sum(count * value for count, value in values) / sum(count for count, _ in values)
    assert len(trained) == 2 and result["held_out_rows"] == _SAMPLES and sorted(by_row) == list(range(_SAMPLES))
    assert result["pgs_partial_r2"] == pytest.approx(expected, rel=1e-9)


def test_a_fold_fit_kept_as_it_ends_is_not_fitted_again_after_a_restart(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A task stopped after one fold's fit, started again, fits only the other fold and writes the line and held-out
    scores an uninterrupted pass writes; a kept fold of other code or steps, or one damaged on the disk, is fitted
    again."""
    store = _store(tmp_path / "store", 3)[0]
    cohort = _cohort(np.random.default_rng(7), 100)
    cohort.held_out = np.zeros((_SAMPLES, 3), dtype=bool)
    cohort.held_out[:_SAMPLES // 2, 0] = True
    cohort.held_out[_SAMPLES // 2:, 1] = True
    cohort.training = ~cohort.held_out
    cohort.trait_models = lambda trait: np.arange(3)
    cohort.all_rows_model = lambda trait: 2
    records = np.arange(30)
    fitted = []

    def fit(train, rows, stop=None):
        if len(fitted) == stop:
            raise RuntimeError("preempted fit")
        fitted.append(train.columns.copy())
        return train.codes(rows).astype(np.float64).mean(axis=1) / 127.0, np.linspace(-1.0, 1.0, rows.size) * (1 + len(fitted) % 2)

    monkeypatch.setattr(aou_ldpred2.ldpred2_auto_allvar, "posterior_effects", fit)
    expected = aou_ldpred2.ldpred2_trait(store, records, cohort, 0)
    units = aou_ldpred2.FoldUnits(tmp_path / "units", {"format": "test", "steps": {"cohort": "a"}, "code": "b"})
    fitted.clear()
    monkeypatch.setattr(aou_ldpred2.ldpred2_auto_allvar, "posterior_effects", lambda train, rows: fit(train, rows, stop=1))
    with pytest.raises(RuntimeError, match="preempted fit"):
        aou_ldpred2.ldpred2_trait(store, records, cohort, 0, units)
    assert len(list((tmp_path / "units").glob("*.npz"))) == 1 and not list((tmp_path / "units").glob(".*"))
    fitted.clear()
    monkeypatch.setattr(aou_ldpred2.ldpred2_auto_allvar, "posterior_effects", fit)
    # The restart fits the second fold only, as the uninterrupted pass's second fit (the stub's effects say which).
    fitted.append(None)
    resumed = aou_ldpred2.ldpred2_trait(store, records, cohort, 0, units)
    assert len(fitted) == 2 and np.array_equal(fitted[1], np.sort(cohort.store_columns[cohort.training[:, 1]]))
    assert resumed[0] == expected[0]
    np.testing.assert_array_equal(resumed[1], expected[1])
    np.testing.assert_array_equal(resumed[2], expected[2])
    # Every fold is kept now: nothing is fitted again.
    monkeypatch.setattr(aou_ldpred2.ldpred2_auto_allvar, "posterior_effects", lambda *args: pytest.fail("a kept fold was fitted again"))
    again = aou_ldpred2.ldpred2_trait(store, records, cohort, 0, units)
    assert again[0] == expected[0] and np.array_equal(again[2], expected[2])
    # Another step key (or code) is another fit.
    other = aou_ldpred2.FoldUnits(tmp_path / "units", {"format": "test", "steps": {"cohort": "c"}, "code": "b"})
    with pytest.raises(pytest.fail.Exception, match="fitted again"):
        aou_ldpred2.ldpred2_trait(store, records, cohort, 0, other)
    # A kept fold cut short on the disk (no ZIP directory left) is fitted again, not fatal.
    damaged = sorted((tmp_path / "units").glob("*.npz"))[0]
    damaged.write_bytes(damaged.read_bytes()[: damaged.stat().st_size // 2])
    fitted.clear()
    monkeypatch.setattr(aou_ldpred2.ldpred2_auto_allvar, "posterior_effects", fit)
    aou_ldpred2.ldpred2_trait(store, records, cohort, 0, units)
    assert len(fitted) == 1


def _binary_cohort(generator: np.random.Generator, rows: int) -> tuple[SimpleNamespace, np.ndarray]:
    """A one-disease cohort of ``rows`` people in families of two, its covariates an intercept and one more, and three
    held-out scores [snv_indel, with_sv, ldpred2] that each carry some of the liability."""
    liability = generator.normal(size=rows)
    targets = (liability + generator.normal(size=rows) > 1.0).astype(np.float64)
    covariates = np.column_stack([np.ones(rows), generator.normal(size=rows)])
    genetic = np.vstack([liability * weight + generator.normal(size=rows) for weight in (0.4, 0.5, 0.3)])
    cohort = SimpleNamespace(
        trait_names=("disease",), trait_types=(TraitType.BINARY,), targets=targets[:, None], families=np.arange(rows) // 2,
        covariates=covariates, covariate_columns=np.array([[True, True]]), folds=np.zeros(rows, dtype=np.int64),
    )
    return cohort, genetic


def test_the_comparison_reads_each_score_as_the_report_does_on_the_same_people() -> None:
    cohort, genetic = _binary_cohort(np.random.default_rng(6), 3000)
    cohort.targets[:40, 0] = np.nan  # no outcome: in no score's rows
    rows = {(row[1], row[4]): row for row in aou_compare.trait_rows(cohort, genetic, 0)}
    kept = np.arange(40, 3000)
    outcome, covariates, families = cohort.targets[kept, 0], cohort.covariates[kept], cohort.families[kept]
    assert {row[2] for row in rows.values()} == {str(kept.shape[0])} and {row[3] for row in rows.values()} == {str(int(outcome.sum()))}
    for index, score in enumerate(aou_compare.SCORES):
        r2, auc = rows[(score, "pgs_partial_r2")], rows[(score, "pgs_auc")]
        assert float(r2[5]) == pytest.approx(_partial_r2(outcome, genetic[index, kept], covariates), rel=1e-5)
        assert float(auc[5]) == pytest.approx(auc_components(outcome, genetic[index, kept]).auc, rel=1e-5)
        for row in (r2, auc):
            estimate, error, lower, upper = (float(value) for value in row[5:])
            assert lower == pytest.approx(estimate - aou_compare.INTERVAL_Z * error, rel=1e-4, abs=1e-9)
            assert upper == pytest.approx(estimate + aou_compare.INTERVAL_Z * error, rel=1e-4, abs=1e-9)
        # The AUC's clustered DeLong error is close to the unclustered one in families of two unrelated scores' noise.
        components = auc_components(outcome, genetic[index, kept])
        delong = np.sqrt(np.var(components.positive, ddof=1) / components.positive.size + np.var(components.negative, ddof=1) / components.negative.size)
        assert float(auc[6]) == pytest.approx(delong, rel=0.2)
    for first, second in aou_compare.PAIRS:
        candidate, baseline = genetic[aou_compare.SCORES.index(first), kept], genetic[aou_compare.SCORES.index(second), kept]
        r2 = paired_delta_r2(*(_residualized(series, covariates) for series in (outcome, baseline, candidate)), families)
        auc = paired_delta_auc(outcome, baseline, candidate, families)
        assert float(rows[(f"{first} - {second}", "delta_pgs_partial_r2")][5]) == pytest.approx(r2.estimate, rel=1e-5)
        assert float(rows[(f"{first} - {second}", "delta_pgs_auc")][6]) == pytest.approx(auc.standard_error, rel=1e-5)


def test_the_partial_r2_error_is_the_delta_method_one() -> None:
    """For a bivariate normal pair, Var(r^2) is about 4 rho^2 (1 - rho^2)^2 / n."""
    generator = np.random.default_rng(8)
    count, rho = 20000, 0.3
    outcome = generator.normal(size=count)
    score = rho * outcome + np.sqrt(1.0 - rho**2) * generator.normal(size=count)
    estimate, error = aou_compare.partial_r2(outcome, score, np.ones((count, 1)), np.arange(count))
    assert estimate == pytest.approx(rho**2, abs=0.02)
    assert error == pytest.approx(2.0 * rho * (1.0 - rho**2) / np.sqrt(count), rel=0.1)


def test_a_comparison_with_few_cases_shows_no_value() -> None:
    cohort, genetic = _binary_cohort(np.random.default_rng(7), 400)
    cohort.targets[:, 0] = 0.0
    cohort.targets[:15, 0] = 1.0
    rows = aou_compare.trait_rows(cohort, genetic, 0)
    assert len(rows) == 2 * (len(aou_compare.SCORES) + len(aou_compare.PAIRS))
    assert all(row[2] == "400" and row[3] == "suppressed" and row[5:] == ("", "", "", "") for row in rows)


def test_each_fold_is_read_on_its_own_people_and_the_folds_are_weighted_by_their_rows() -> None:
    cohort, genetic = _binary_cohort(np.random.default_rng(9), 3000)
    cohort.folds = np.arange(3000) % 3
    cohort.folds[cohort.folds == 2] = 1  # two folds of 1,000 and 2,000 people
    rows = {(row[1], row[4]): row for row in aou_compare.trait_rows(cohort, genetic, 0)}
    per_fold = [np.flatnonzero(cohort.folds == fold) for fold in (0, 1)]
    for index, score in enumerate(aou_compare.SCORES):
        values = [aou_compare.partial_r2(cohort.targets[fold, 0], genetic[index, fold], cohort.covariates[fold], cohort.families[fold]) for fold in per_fold]
        estimate = (1000 * values[0][0] + 2000 * values[1][0]) / 3000
        error = np.sqrt((values[0][1] / 3) ** 2 + (2 * values[1][1] / 3) ** 2)
        assert float(rows[(score, "pgs_partial_r2")][5]) == pytest.approx(estimate, rel=1e-5)
        assert float(rows[(score, "pgs_partial_r2")][6]) == pytest.approx(error, rel=1e-5)
        # Not the one pooled correlation of both folds' scores.
        assert estimate != pytest.approx(_partial_r2(cohort.targets[:, 0], genetic[index], cohort.covariates), rel=1e-6)


def test_scores_of_different_people_are_refused() -> None:
    cohort, genetic = _binary_cohort(np.random.default_rng(10), 3000)
    genetic[2, :5] = np.nan  # LDpred2 left out five people the arms scored
    with pytest.raises(ValueError, match="different held-out people"):
        aou_compare.trait_rows(cohort, genetic, 0)

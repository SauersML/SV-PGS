"""The All of Us LDpred2 baseline reads the run's store, rows and covariates as the fit and the report do."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.bench_real import aou_ldpred2
from sv_pgs.workspace_pipeline import _partial_r2
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
    result = aou_ldpred2.ldpred2_trait(store, records, cohort, 0)
    dosage = store.read_codes(0, store.n_variants)[records].astype(np.float64).T / 127.0
    training, held_out = np.flatnonzero(cohort.training[:, 0]), np.flatnonzero(cohort.held_out[:, 0])
    score = (dosage[cohort.store_columns[held_out]] - dosage[cohort.store_columns[training]].mean(axis=0)) @ effects
    expected = _partial_r2(cohort.targets[held_out, 0], score, cohort.covariates[held_out][:, :3])
    assert result == {"trait": "trait", "held_out_rows": 100, "variants": records.shape[0], "pgs_partial_r2": pytest.approx(expected, rel=1e-9)}


def test_a_held_out_count_under_twenty_is_not_shown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _store(tmp_path / "store", 3)[0]
    cohort = _cohort(np.random.default_rng(5), aou_ldpred2.SMALLEST_SHOWN_COUNT - 1)
    monkeypatch.setattr(aou_ldpred2.ldpred2_auto_allvar, "posterior_effects", lambda train, rows: (np.zeros(rows.shape[0]), np.ones(rows.shape[0])))
    result = aou_ldpred2.ldpred2_trait(store, np.arange(30), cohort, 0)
    assert result["held_out_rows"] == f"<{aou_ldpred2.SMALLEST_SHOWN_COUNT}" and result["pgs_partial_r2"] is None

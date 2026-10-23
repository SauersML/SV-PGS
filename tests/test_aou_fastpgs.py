"""The fast SNV-vs-SV baselines fit both arms on one pass: the snv_indel arm reads exactly what a pass of its own would."""
from pathlib import Path

import numpy as np
import pytest

from benchmarks.bench_real import aou_fastpgs
from benchmarks.bench_sim.submissions import ldpred2_auto_allvar
from sv_pgs.workspace_pipeline import _partial_r2
from tests.test_full_data_fit import _SAMPLES, _store


def _view(generator: np.random.Generator, records: int, people: int) -> aou_fastpgs.MemoryView:
    """Correlated dosages (each record a noisy copy of a neighbour's) over ``people`` people, two covariates and a target
    on 10 of the records, positions 0.1 cM apart in three chromosome blocks."""
    dosages = np.empty((records, people))
    dosages[0] = generator.uniform(0, 2, people)
    for row in range(1, records):
        dosages[row] = np.clip(dosages[row - 1] + generator.normal(0, 0.5, people), 0, 2) if generator.random() < 0.8 else generator.uniform(0, 2, people)
    codes = np.rint(dosages * 127).astype(np.uint8)
    covariates = generator.normal(size=(people, 2))
    phenotype = codes[generator.choice(records, 10, replace=False)].sum(axis=0) / 127.0 + covariates[:, 0] + generator.normal(0, 2, people)
    view = aou_fastpgs.MemoryView(codes, np.arange(people), covariates, phenotype)
    view.variants = {"cm": np.sort(np.repeat([0.0, 1e4, 2e4], -(-records // 3))[:records] + np.arange(records) * 0.1)}
    return view


def _read_band(folder: Path, lo: np.ndarray, hi: np.ndarray) -> list[np.ndarray]:
    values = np.fromfile(folder / "ld_values.bin", dtype="<f4")
    offsets = np.r_[0, np.cumsum(hi - lo)]
    assert values.shape[0] == offsets[-1]
    return [values[offsets[j] : offsets[j + 1]] for j in range(lo.shape[0])]


def test_the_snv_band_and_gwas_are_those_of_a_pass_over_its_records_alone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(aou_fastpgs, "CHUNK", 32)
    monkeypatch.setattr(ldpred2_auto_allvar, "CHUNK", 32)
    generator = np.random.default_rng(1)
    view = _view(generator, 300, 160)
    cm = view.variants["cm"]
    short = generator.random(300) < 0.85
    part = aou_fastpgs.MemoryView(view.all_codes, np.arange(0, 160, 2), view.covariates[::2], view.phenotype[::2])
    folders = {arm: tmp_path / arm for arm in aou_fastpgs.ARMS}
    for folder in folders.values():
        folder.mkdir()
    arms = aou_fastpgs.gwas_and_bands(view, part, cm, short, folders)

    alone = aou_fastpgs.MemoryView(view.all_codes[short], np.arange(160), view.covariates, view.phenotype)
    alone.variants = {"cm": cm[short]}
    (tmp_path / "alone").mkdir()
    means, beta, se, lo, hi, scores = ldpred2_auto_allvar.write_band(alone, np.arange(int(short.sum())), tmp_path / "alone")
    snv = arms["snv_indel"]
    np.testing.assert_array_equal(snv["lo"], lo)
    np.testing.assert_array_equal(snv["hi"], hi)
    for found, expected in zip(_read_band(folders["snv_indel"], snv["lo"], snv["hi"]), _read_band(tmp_path / "alone", lo, hi)):
        np.testing.assert_allclose(found, expected, atol=2e-6)
    for name, expected in (("means", means), ("beta", beta), ("se", se), ("ld_scores", scores)):
        np.testing.assert_allclose(snv[name], expected, rtol=1e-5, atol=1e-6)
    part_alone = aou_fastpgs.Adjuster(aou_fastpgs.MemoryView(view.all_codes[short], np.arange(0, 160, 2), view.covariates[::2], view.phenotype[::2]))
    _, beta_part, se_part, _, _ = part_alone.block(np.arange(int(short.sum())))
    np.testing.assert_allclose(snv["beta_part"], beta_part, rtol=1e-5)
    np.testing.assert_allclose(snv["se_part"], se_part, rtol=1e-5)

    wide = arms["with_sv"]
    (tmp_path / "wide").mkdir()
    view_all = aou_fastpgs.MemoryView(view.all_codes, np.arange(160), view.covariates, view.phenotype)
    view_all.variants = {"cm": cm}
    *_, lo_all, hi_all, _ = ldpred2_auto_allvar.write_band(view_all, np.arange(300), tmp_path / "wide")
    for found, expected in zip(_read_band(folders["with_sv"], wide["lo"], wide["hi"]), _read_band(tmp_path / "wide", lo_all, hi_all)):
        np.testing.assert_allclose(found, expected, atol=2e-6)


def test_without_svs_the_arms_are_one_fit(tmp_path: Path) -> None:
    generator = np.random.default_rng(2)
    view = _view(generator, 120, 100)
    folders = {arm: tmp_path / arm for arm in aou_fastpgs.ARMS}
    for folder in folders.values():
        folder.mkdir()
    arms = aou_fastpgs.gwas_and_bands(view, view, view.variants["cm"], np.ones(120, dtype=bool), folders)
    assert (folders["with_sv"] / "ld_values.bin").read_bytes() == (folders["snv_indel"] / "ld_values.bin").read_bytes()
    for name, values in arms["with_sv"].items():
        np.testing.assert_allclose(arms["snv_indel"][name], values, rtol=1e-12)


def test_clumping_keeps_no_correlated_pair_and_removes_each_other_record_by_a_better_one(tmp_path: Path) -> None:
    generator = np.random.default_rng(3)
    view = _view(generator, 200, 150)
    folders = {arm: tmp_path / arm for arm in aou_fastpgs.ARMS}
    for folder in folders.values():
        folder.mkdir()
    arm = aou_fastpgs.gwas_and_bands(view, view, view.variants["cm"], np.ones(200, dtype=bool), folders)["with_sv"]
    p = aou_fastpgs.p_values(arm["beta"], arm["se"])
    values = np.fromfile(folders["with_sv"] / "ld_values.bin", dtype="<f4")
    index = aou_fastpgs.clump(p, arm["lo"], arm["hi"], values, 0.1)
    columns = _read_band(folders["with_sv"], arm["lo"], arm["hi"])
    rank = np.argsort(np.argsort(p, kind="stable"))
    assert index[np.argmin(p)]
    for j in range(200):
        window = np.arange(arm["lo"][j], arm["hi"][j])
        strong = window[(columns[j].astype(np.float64) ** 2 > 0.1) & (window != j)]
        if index[j]:
            assert not np.any(index[strong])
        else:
            assert np.any(index[strong] & (rank[strong] < rank[j]))
    effects = aou_fastpgs.ct_effects(arm["beta"], p, index)
    assert effects.shape == (200, len(aou_fastpgs.THRESHOLDS))
    np.testing.assert_array_equal(effects[:, -1], np.where(index, arm["beta"], 0.0))
    assert np.all(np.count_nonzero(effects, axis=0)[:-1] <= np.count_nonzero(effects, axis=0)[1:])


def test_the_tuning_slice_takes_whole_families_and_is_fixed_by_its_seed() -> None:
    families = np.repeat(np.arange(5000), np.random.default_rng(4).integers(1, 4, 5000))
    tuning = aou_fastpgs.tuning_slice(families, [1, 7])
    for family in range(0, 5000, 97):
        assert np.unique(tuning[families == family]).shape[0] == 1
    assert 0.18 < np.mean([tuning[families == family][0] for family in range(5000)]) < 0.22
    np.testing.assert_array_equal(aou_fastpgs.tuning_slice(families, [1, 7]), tuning)
    assert not np.array_equal(aou_fastpgs.tuning_slice(families, [2, 7]), tuning)


def test_the_codes_are_the_store_rows_in_the_order_asked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _store(tmp_path / "store", 3)[0]
    dense = store.read_codes(0, store.n_variants)
    generator = np.random.default_rng(5)
    records = generator.permutation(np.r_[np.arange(3, 40), np.arange(41, 120, 7), np.arange(150, 300, 2)])
    columns = np.sort(generator.choice(_SAMPLES, 90, replace=False))
    for gap, span in ((1, 4096), (64, 4096), (64, 5)):
        monkeypatch.setattr(aou_fastpgs, "MERGE_GAP", gap)
        monkeypatch.setattr(aou_fastpgs, "MAX_SPAN", span)
        np.testing.assert_array_equal(aou_fastpgs.read_codes(store, records, columns, threads=3), dense[records][:, columns])
    ascending = np.sort(records)
    np.testing.assert_array_equal(aou_fastpgs.read_codes(store, ascending, columns, threads=2), dense[ascending][:, columns])


def test_scores_and_the_tuning_choice() -> None:
    generator = np.random.default_rng(6)
    codes = generator.integers(0, 255, size=(70, 40), dtype=np.uint8)
    means = codes.mean(axis=1) / 127.0
    effects = generator.normal(size=(70, 4)).astype(np.float32)
    effects[:, 2] = np.nan
    rows = np.array([3, 1, 30, 7])
    found = aou_fastpgs.candidate_scores(codes, rows, means, effects)
    expected = (codes[:, rows].T / 127.0 - means) @ effects[:, [0, 1, 3]].astype(np.float64)
    np.testing.assert_allclose(found[:, [0, 1, 3]], expected, rtol=1e-4, atol=1e-4)
    assert np.all(np.isnan(found[:, 2]))

    people = 300
    target = generator.normal(size=people)
    covariates = generator.normal(size=(people, 2))
    tuning = np.column_stack([generator.normal(size=people) + weight * target for weight in (0.1, 0.5, 0.0, 0.3, 0.2)])
    tuning[:, 2] = np.nan
    labels = np.array(["snv_indel|ct|a", "snv_indel|ct|b", "snv_indel|ct|c", "with_sv|ct|a", "with_sv|ldpred2_auto|auto"])
    chosen, r2 = aou_fastpgs.choose(labels, tuning, target, covariates)
    assert chosen == {("snv_indel", "ct"): 1, ("with_sv", "ct"): 3, ("with_sv", "ldpred2_auto"): 4}
    assert np.isnan(r2[2])
    assert r2[1] == pytest.approx(_partial_r2(target, tuning[:, 1], covariates))


def test_a_record_with_no_training_variance_correlates_with_nothing(tmp_path: Path) -> None:
    generator = np.random.default_rng(7)
    view = _view(generator, 90, 80)
    view.all_codes[40] = 0
    folders = {arm: tmp_path / arm for arm in aou_fastpgs.ARMS}
    for folder in folders.values():
        folder.mkdir()
    short = np.ones(90, dtype=bool)
    short[40] = False
    arms = aou_fastpgs.gwas_and_bands(view, view, view.variants["cm"], short, folders)
    wide = arms["with_sv"]
    band = _read_band(folders["with_sv"], wide["lo"], wide["hi"])
    assert all(np.all(np.isfinite(column)) for column in band)
    assert np.all(np.isfinite(wide["ld_scores"]))
    np.testing.assert_array_equal(band[40], (np.arange(wide["lo"][40], wide["hi"][40]) == 40).astype(np.float32))
    aou_fastpgs.write_inputs(folders["with_sv"], wide, 80, 80, 1)
    assert np.all(np.isfinite(np.fromfile(folders["with_sv"] / "sumstats.bin", dtype="<f8")))


def test_ldpred2_auto_is_kept_without_a_usable_tuning_target() -> None:
    generator = np.random.default_rng(8)
    tuning = generator.normal(size=(50, 3))
    labels = np.array(["snv_indel|ldpred2_auto|auto", "snv_indel|ct|p<=1", "with_sv|ldpred2_auto|auto"])
    tuning[:, 2] = np.nan
    chosen, _ = aou_fastpgs.choose(labels, tuning, np.zeros(50), np.ones((50, 1)))
    assert chosen == {("snv_indel", "ldpred2_auto"): 0}


def test_ldpred2_auto_alone_skips_lassosum2_and_clumping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    generator = np.random.default_rng(9)
    view = _view(generator, 90, 80)
    short = generator.random(90) < 0.8
    calls = []

    def popen(command, env):
        folder = Path(command[2])
        calls.append(folder.name)
        assert (folder / "meta.txt").read_text().split()[5] == "0"
        count = int((folder / "meta.txt").read_text().split()[0])
        np.full(count, 0.5).astype("<f8").tofile(folder / "beta_auto.bin")
        return type("Done", (), {"wait": staticmethod(lambda: 0)})()

    monkeypatch.setattr(aou_fastpgs.subprocess, "Popen", popen)
    monkeypatch.setattr(aou_fastpgs, "clump", lambda *args: pytest.fail("clumping ran"))
    fitted = aou_fastpgs.fit_effects(view.variants["cm"], short, view, view, 2, ("ldpred2_auto",))
    assert sorted(calls) == ["snv_indel", "with_sv"]
    assert fitted["labels"].tolist() == ["snv_indel|ldpred2_auto|auto", "with_sv|ldpred2_auto|auto"]
    np.testing.assert_array_equal(fitted["effects"][:, 0], np.where(short, 0.5, 0.0))
    np.testing.assert_array_equal(fitted["effects"][:, 1], np.full(90, 0.5))
    tuning = generator.normal(size=(30, 2))
    chosen, _ = aou_fastpgs.choose(fitted["labels"], tuning, generator.normal(size=30), np.ones((30, 1)), ("ldpred2_auto",))
    assert chosen == {("snv_indel", "ldpred2_auto"): 0, ("with_sv", "ldpred2_auto"): 1}


def test_the_r_sessions_run_until_the_fit_is_finished(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # start_effects leaves both arms' R sessions running, so the next trait's GWAS and bands overlap them; the finish
    # waits for them, reads their outputs and removes the trait's folder.
    generator = np.random.default_rng(10)
    view = _view(generator, 60, 70)
    short = generator.random(60) < 0.8
    folders, waited = [], []

    def popen(command, env):
        folder = Path(command[2])
        folders.append(folder)

        def wait() -> int:
            waited.append(folder.name)
            count = int((folder / "meta.txt").read_text().split()[0])
            np.full(count, 0.25).astype("<f8").tofile(folder / "beta_auto.bin")
            return 0

        return type("Running", (), {"wait": staticmethod(wait)})()

    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(aou_fastpgs.subprocess, "Popen", popen)
    finish = aou_fastpgs.start_effects(view.variants["cm"], short, view, view, 2, ("ldpred2_auto",))
    assert len(folders) == 2 and waited == [] and all(folder.exists() for folder in folders)
    fitted = finish()
    assert sorted(waited) == ["snv_indel", "with_sv"] and not any(folder.exists() for folder in folders)
    np.testing.assert_array_equal(fitted["effects"][:, 1], np.full(60, 0.25))
    np.testing.assert_array_equal(fitted["effects"][:, 0], np.where(short, 0.25, 0.0))

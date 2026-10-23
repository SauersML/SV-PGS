"""The fast baselines' paired tests read each run's held-out scores and compare the arms on the same people."""
from pathlib import Path

import numpy as np
import pytest

from benchmarks.bench_real import aou_fastpgs, fastpgs_meta
from sv_pgs.workspace_pipeline import _partial_r2


def _write(root: Path, run: str, trait: str, binary: bool, ids: np.ndarray, generator: np.random.Generator, failed: bool = False) -> None:
    people = ids.shape[0]
    covariates = np.column_stack([np.ones(people), generator.normal(size=(people, 2))])  # the cohort's covariates hold the intercept
    genetic = generator.normal(size=people)
    target = (genetic + covariates[:, 0] + generator.normal(size=people) > 1.0).astype(float) if binary else genetic + generator.normal(size=people)
    scores = np.stack([np.stack([genetic + generator.normal(0, noise, people) for noise in (1.0, 0.8)]) for _ in aou_fastpgs.METHODS])
    if failed:
        scores[1, 0] = np.nan
    folder = root / run / "fastpgs"
    folder.mkdir(parents=True)
    with open(folder / f"{trait}.npz", "wb") as handle:
        np.savez(handle, trait=np.array(trait), binary=np.array(binary), methods=np.array(aou_fastpgs.METHODS), arms=np.array(aou_fastpgs.ARMS),
                 research_ids=ids, families=np.arange(people) // 2, ancestry=np.where(np.arange(people) % 3 == 0, "AFR", "EUR"),
                 target=target, covariates=covariates, scores=scores, seed=np.array(1))


def test_every_method_is_tested_on_both_traits_and_a_failed_arm_is_named(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    generator = np.random.default_rng(1)
    _write(tmp_path, "run-a", "ldl", False, np.array([f"p{index}" for index in range(600)]), generator)
    _write(tmp_path, "run-b", "gout", True, np.array([f"p{index}" for index in range(300, 900)]), generator, failed=True)
    assert fastpgs_meta.main(["run-a", "run-b", "--root", str(tmp_path), "--resamples", "60", "--permutations", "3"]) == 0
    printed = capsys.readouterr().out
    assert "union of held-out people 900" in printed
    for method in aou_fastpgs.METHODS:
        assert f"=== {method}" in printed
    assert "=== lassosum2: 1 traits; no score in an arm for gout" in printed
    assert "AUC" in printed and "PRIMARY common relative gain" in printed
    assert "\tAFR\t" in printed and "\tEUR\t" in printed


def test_a_trait_whose_arms_chose_the_same_model_is_listed_and_not_combined(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    generator = np.random.default_rng(3)
    _write(tmp_path, "run-a", "ldl", False, np.array([f"p{index}" for index in range(500)]), generator)
    _write(tmp_path, "run-b", "gout", True, np.array([f"p{index}" for index in range(500)]), generator)
    path = tmp_path / "run-b" / "fastpgs" / "gout.npz"
    with np.load(path, allow_pickle=False) as arrays:
        saved = {name: np.array(arrays[name]) for name in arrays.files}
    saved["scores"][2, 1] = saved["scores"][2, 0]
    with open(path, "wb") as handle:
        np.savez(handle, **saved)
    assert fastpgs_meta.main(["run-a", "run-b", "--root", str(tmp_path), "--resamples", "40", "--permutations", "2"]) == 0
    printed = capsys.readouterr().out
    ct = printed[printed.index("=== ct"):]
    assert "the same model in both arms (gain 0, not combined) for gout" in ct
    assert "Z = nan" not in ct and "PRIMARY common relative gain" in ct


def test_a_cell_reads_the_arms_partial_r2_on_its_rows() -> None:
    generator = np.random.default_rng(2)
    people = 400
    covariates = np.column_stack([np.ones(people), generator.normal(size=(people, 2))])
    target = generator.normal(size=people)
    baseline, candidate = target + generator.normal(0, 2, people), target + generator.normal(0, 1.5, people)
    trait = fastpgs_meta.PairedTrait("t", False, np.arange(people), target, covariates, baseline, candidate)
    clusters = np.arange(people)
    weights = np.ones((5, people))
    rows = np.arange(0, people, 2)
    found = fastpgs_meta.cell(trait, rows, weights, clusters)
    assert found["snv_indel"] == pytest.approx(_partial_r2(target[rows], baseline[rows], covariates[rows]))
    assert found["with_sv"] == pytest.approx(_partial_r2(target[rows], candidate[rows], covariates[rows]))
    assert found["n"] == rows.shape[0]


def test_benjamini_hochberg_q_values() -> None:
    q = fastpgs_meta.benjamini_hochberg([0.01, 0.04, 0.03, 0.2])
    np.testing.assert_allclose(q, [0.04, 0.16 / 3, 0.16 / 3, 0.2])
    assert fastpgs_meta.benjamini_hochberg([0.9, 0.95]).max() <= 1.0


def test_the_screen_tests_each_admitted_trait_one_sided_and_names_the_rest(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    generator = np.random.default_rng(4)
    ids = np.array([f"p{index}" for index in range(1200)])
    _write(tmp_path, "run-a", "ldl", False, ids, generator)
    _write(tmp_path, "run-b", "gout", True, ids, generator)
    _write(tmp_path, "run-c", "hdl", False, ids[:500], generator)
    (tmp_path / "lr.tsv").write_text("research_id\tother\np0\tx\np1\tx\n")
    assert fastpgs_meta.main(["run-a", "run-b", "run-c", "--root", str(tmp_path), "--resamples", "60", "--screen", "ldpred2_auto",
                              "--exclude", str(tmp_path / "lr.tsv")]) == 0
    printed = capsys.readouterr().out
    assert "ldl: <40 held-out people in" in printed
    assert "below the power threshold: hdl" in printed
    lines = [line for line in printed.splitlines() if line.startswith("  ") and "one-sided p=" in line]
    assert sorted(line.split("\t")[0].strip() for line in lines) == ["gout", "ldl"]
    assert "n=1198" in printed and "traits with a with_sv gain at BH q < 0.05" in printed

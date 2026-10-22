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
    assert "AUC" in printed and "PRIMARY Stouffer" in printed
    assert "\tAFR\t" in printed and "\tEUR\t" in printed


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

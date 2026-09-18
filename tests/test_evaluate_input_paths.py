"""evaluate_all_of_us must read the files run_all_of_us wrote for the requested disease.

run_all_of_us writes ``predictions.tsv.gz`` and ``<canonical>.samples.tsv`` into the
disease's work_dir and keeps the ancestry predictions in the shared cache beside it
(``local_ancestry_predictions_path``). The evaluator must never substitute another
disease's predictions, and must find the ancestry file where the runner put it.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from sv_pgs.aou_runner import local_ancestry_predictions_path
from sv_pgs.evaluate import evaluate_all_of_us

GROUP_SIZE = 12


def _write_predictions(path: Path, scores: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as handle:
        handle.write("sample_id\ttarget\tgenetic_score\tprobability\n")
        for sample_id, score in scores.items():
            handle.write(f"{sample_id}\t0\t{score}\t0.5\n")


def _write_sample_table(path: Path, occurrence_counts: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("sample_id\tphenotype_occurrence_count\n")
        for sample_id, count in occurrence_counts.items():
            handle.write(f"{sample_id}\t{count}\n")


def _cohort() -> tuple[dict[str, float], dict[str, int]]:
    # 0-code controls score low and 1-code controls score high, so the ICD
    # stratification AUC is 1.0 whenever the scores are read.
    scores: dict[str, float] = {}
    occurrence_counts: dict[str, int] = {}
    for sample_number in range(GROUP_SIZE):
        scores[f"zero{sample_number}"] = -1.0 - sample_number
        occurrence_counts[f"zero{sample_number}"] = 0
        scores[f"one{sample_number}"] = 1.0 + sample_number
        occurrence_counts[f"one{sample_number}"] = 1
    return scores, occurrence_counts


def test_missing_predictions_never_fall_back_to_another_disease(tmp_path):
    scores, occurrence_counts = _cohort()
    work_dir = tmp_path / "type2_diabetes_results"
    _write_sample_table(work_dir / "type2_diabetes.samples.tsv", occurrence_counts)
    _write_predictions(tmp_path / "hypertension_result" / "predictions.tsv.gz", scores)

    with pytest.raises(FileNotFoundError, match="predictions.tsv.gz"):
        evaluate_all_of_us(work_dir, "type2_diabetes")


def test_ancestry_and_alias_resolve_to_the_runner_layout(tmp_path):
    scores, occurrence_counts = _cohort()
    work_dir = tmp_path / "type2_diabetes_results"
    _write_predictions(work_dir / "predictions.tsv.gz", scores)
    _write_sample_table(work_dir / "type2_diabetes.samples.tsv", occurrence_counts)
    ancestry_path = local_ancestry_predictions_path(work_dir)
    ancestry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ancestry_path, "w", encoding="utf-8") as handle:
        handle.write("research_id\tancestry_pred\n")
        for sample_id in scores:
            handle.write(f"{sample_id}\teur\n")

    results = evaluate_all_of_us(work_dir, "t2d")

    assert results["disease"] == "type2_diabetes"
    assert results["test1_all_auc"] == pytest.approx(1.0)
    assert results["test1_eur_auc"] == pytest.approx(1.0)
    assert results["test1_eur_n_neg"] == GROUP_SIZE
    assert (work_dir / "type2_diabetes.evaluation.json").exists()

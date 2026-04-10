"""Quasi-holdout evaluation for AoU binary trait models.

Two tests that use signal the model never saw during training:

1. ICD stratification: among controls (0-1 codes), can the genetic score
   distinguish 0-code from 1-code people? The model was trained with both
   as target=0, so any separation is out-of-distribution signal.

2. Survey self-report: among controls/missing, can the genetic score
   distinguish people who self-reported the condition in the AoU survey
   from those who did not? Survey data was never used in training.
"""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from sv_pgs.progress import log


def _build_survey_hypertension_sql() -> str:
    """BigQuery SQL to extract self-reported hypertension from AoU surveys."""
    import os
    dataset = os.environ.get("WORKSPACE_CDR", "")
    if not dataset:
        raise RuntimeError("WORKSPACE_CDR environment variable required for survey query")
    return f"""
SELECT
  CAST(person.person_id AS STRING) AS person_id,
  1 AS self_reported_hypertension
FROM `{dataset}.ds_survey` AS survey
JOIN `{dataset}.person` AS person
  ON person.person_id = survey.person_id
WHERE survey.question LIKE '%blood pressure%'
  AND survey.question LIKE '%hypertension%'
  AND survey.answer = 'Self'
GROUP BY person.person_id
""".strip()


def _fetch_survey_self_report(disease: str) -> set[str]:
    """Query BigQuery for survey self-reported cases. Returns set of person_ids."""
    if disease != "hypertension":
        log(f"  survey validation not yet implemented for disease: {disease}")
        return set()
    try:
        from google.cloud import bigquery
    except ImportError:
        log("  google-cloud-bigquery not available — skipping survey validation")
        return set()
    sql = _build_survey_hypertension_sql()
    log("  querying BigQuery for survey self-reported hypertension...")
    try:
        client = bigquery.Client()
        rows = client.query(sql).result()
        positive_ids = {str(row.person_id) for row in rows}
        log(f"  {len(positive_ids):,} people self-reported hypertension in survey")
        return positive_ids
    except Exception as exc:
        log(f"  BigQuery survey query failed: {exc}")
        return set()


def evaluate_all_of_us(
    output_dir: Path,
    disease: str,
) -> dict[str, object]:
    """Run quasi-holdout evaluation on a completed AoU run."""
    work_dir = Path(output_dir)

    # Find predictions and sample table
    predictions_candidates = [
        work_dir / "predictions.tsv.gz",
        work_dir.parent / "hypertension_result" / "predictions.tsv.gz",
    ]
    predictions_path = None
    for candidate in predictions_candidates:
        if candidate.exists():
            predictions_path = candidate
            break
    if predictions_path is None:
        raise FileNotFoundError(f"No predictions.tsv.gz found in {work_dir}")

    sample_table_path = work_dir / f"{disease}.samples.tsv"
    if not sample_table_path.exists():
        # Try parent
        for parent in [work_dir, work_dir.parent / "hypertension_results"]:
            candidate = parent / f"{disease}.samples.tsv"
            if candidate.exists():
                sample_table_path = candidate
                break
    if not sample_table_path.exists():
        raise FileNotFoundError(f"No {disease}.samples.tsv found")

    log("=== QUASI-HOLDOUT EVALUATION ===")
    log(f"  disease: {disease}")
    log(f"  predictions: {predictions_path}")
    log(f"  sample table: {sample_table_path}")

    # Load predictions
    scores: dict[str, float] = {}
    with gzip.open(predictions_path, "rt") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            scores[row["sample_id"]] = float(row["predicted_probability"])
    log(f"  loaded {len(scores):,} predicted scores")

    # Load sample table
    observation_counts: dict[str, int] = {}
    with open(sample_table_path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            sid = row["sample_id"]
            cnt = int(row["phenotype_occurrence_count"])
            observation_counts[sid] = cnt

    # Group by observation count
    groups: dict[int, list[float]] = {}
    missing_from_predictions: list[str] = []
    for sid, cnt in observation_counts.items():
        if sid in scores:
            groups.setdefault(cnt, []).append(scores[sid])
        else:
            missing_from_predictions.append(sid)

    if missing_from_predictions:
        log(f"  {len(missing_from_predictions):,} sample table entries missing from predictions (dropped during training)")

    results: dict[str, object] = {"disease": disease}

    # === Test 1: ICD code stratification ===
    log("")
    log("=== TEST 1: ICD Code Stratification (0-code vs 1-code) ===")
    log("  The model trained both groups as target=0.")
    log("  If it learned real biology, 1-code people should score higher.")

    zero_scores = np.array(groups.get(0, []))
    one_scores = np.array(groups.get(1, []))

    if zero_scores.size > 10 and one_scores.size > 10:
        labels_01 = np.concatenate([np.zeros(len(zero_scores)), np.ones(len(one_scores))])
        preds_01 = np.concatenate([zero_scores, one_scores])
        auc_01 = float(roc_auc_score(labels_01, preds_01))

        log(f"  0-code: n={len(zero_scores):,}  mean_score={zero_scores.mean():.6f}  std={zero_scores.std():.6f}")
        log(f"  1-code: n={len(one_scores):,}  mean_score={one_scores.mean():.6f}  std={one_scores.std():.6f}")
        log(f"  difference: {one_scores.mean() - zero_scores.mean():.6f}")
        log(f"  >>> AUC (0 vs 1): {auc_01:.4f} <<<")
        if auc_01 > 0.52:
            log("  interpretation: model captures hypertension risk beyond training labels")
        elif auc_01 > 0.50:
            log("  interpretation: weak signal, possibly noise")
        else:
            log("  interpretation: no signal — model may be overfitting training labels")

        results["test1_auc"] = auc_01
        results["test1_n_zero"] = len(zero_scores)
        results["test1_n_one"] = len(one_scores)
        results["test1_mean_zero"] = float(zero_scores.mean())
        results["test1_mean_one"] = float(one_scores.mean())
    else:
        log("  insufficient data for 0-vs-1 test")

    # === Test 2: Survey self-report ===
    log("")
    log("=== TEST 2: Survey Self-Report Validation ===")
    log("  Among controls (0-1 ICD codes) and people missing from EHR,")
    log("  compare genetic scores for survey-positive vs survey-negative.")

    survey_positive = _fetch_survey_self_report(disease)

    if survey_positive:
        # Controls + missing people
        control_or_missing = {
            sid for sid, cnt in observation_counts.items()
            if cnt <= 1 and sid in scores
        }
        # Also include people in predictions but not in observation_counts
        # (people in VCF without EHR — if any)
        all_scored = set(scores.keys())
        missing_ehr = all_scored - set(observation_counts.keys())
        eval_pool = control_or_missing | missing_ehr

        survey_pos = eval_pool & survey_positive
        survey_neg = eval_pool - survey_positive

        pos_scores = np.array([scores[sid] for sid in survey_pos])
        neg_scores = np.array([scores[sid] for sid in survey_neg])

        if pos_scores.size > 10 and neg_scores.size > 10:
            labels_survey = np.concatenate([np.zeros(len(neg_scores)), np.ones(len(pos_scores))])
            preds_survey = np.concatenate([neg_scores, pos_scores])
            auc_survey = float(roc_auc_score(labels_survey, preds_survey))

            log(f"  survey-negative controls: n={len(neg_scores):,}  mean_score={neg_scores.mean():.6f}")
            log(f"  survey-positive controls: n={len(pos_scores):,}  mean_score={pos_scores.mean():.6f}")
            log(f"  difference: {pos_scores.mean() - neg_scores.mean():.6f}")
            log(f"  >>> AUC (survey-neg vs survey-pos): {auc_survey:.4f} <<<")
            if auc_survey > 0.55:
                log("  interpretation: strong evidence the model learned real hypertension biology")
            elif auc_survey > 0.52:
                log("  interpretation: moderate evidence of real signal")
            else:
                log("  interpretation: weak or no signal from survey validation")

            results["test2_auc"] = auc_survey
            results["test2_n_pos"] = len(pos_scores)
            results["test2_n_neg"] = len(neg_scores)
            results["test2_mean_pos"] = float(pos_scores.mean())
            results["test2_mean_neg"] = float(neg_scores.mean())
        else:
            log(f"  insufficient survey overlap: {pos_scores.size} positive, {neg_scores.size} negative")
    else:
        log("  no survey data available")

    # === Dose-response ===
    log("")
    log("=== DOSE-RESPONSE (score vs observation count) ===")
    for cnt in sorted(groups.keys()):
        g = np.array(groups[cnt])
        if cnt <= 5 or cnt in (10, 20, 50, 100) or cnt == max(groups.keys()):
            log(f"  {cnt:>3} codes: n={len(g):>6,}  mean_score={g.mean():.6f}")

    # Save results
    eval_output = work_dir / f"{disease}.evaluation.json"
    with open(eval_output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    log(f"\n  results saved to {eval_output}")
    log("=== EVALUATION COMPLETE ===")
    return results

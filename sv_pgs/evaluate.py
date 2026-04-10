"""Quasi-holdout evaluation for AoU binary trait models.

Two tests that use signal the model never saw during training:

1. ICD stratification: among controls (0-1 codes), can the genetic score
   distinguish 0-code from 1-code people? The model was trained with both
   as target=0, so any separation is out-of-distribution signal.

2. Survey self-report: among controls/missing, can the genetic score
   distinguish people who self-reported the condition in the AoU survey
   from those who did not? Survey data was never used in training.

Both tests are run on the full cohort and also restricted to EUR ancestry.
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
    import os
    dataset = os.environ.get("WORKSPACE_CDR", "")
    if not dataset:
        raise RuntimeError("WORKSPACE_CDR environment variable required for survey query")
    # AoU Personal Medical History survey asks "Heart and blood conditions:
    # Has a doctor or health care provider ever told you that you have...?"
    # Hypertension is an ANSWER option, not the question itself.
    return f"""
SELECT DISTINCT
  CAST(person_id AS STRING) AS person_id
FROM `{dataset}.ds_survey`
WHERE LOWER(answer) LIKE '%hypertension%'
   OR LOWER(answer) LIKE '%high blood pressure%'
""".strip()


def _fetch_survey_self_report(disease: str) -> set[str]:
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
    except (OSError, RuntimeError, ValueError) as exc:
        log(f"  BigQuery survey query failed: {exc}")
        return set()


def _load_ancestry_labels(ancestry_path: Path) -> dict[str, str]:
    """Load ancestry predictions. Returns {person_id: ancestry_label}."""
    labels: dict[str, str] = {}
    if not ancestry_path.exists():
        return labels
    with open(ancestry_path, encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        id_col = None
        for candidate in ("research_id", "person_id", "sample_id"):
            if candidate in (reader.fieldnames or []):
                id_col = candidate
                break
        if id_col is None or "ancestry_pred" not in (reader.fieldnames or []):
            return labels
        for row in reader:
            sid = row.get(id_col, "").strip()
            ancestry = row.get("ancestry_pred", "").strip()
            if sid and ancestry:
                labels[sid] = ancestry
    return labels


def _compute_auc_safe(labels: np.ndarray, preds: np.ndarray) -> float | None:
    if len(labels) < 20 or len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, preds))


def _run_auc_test(
    name: str,
    neg_scores: np.ndarray,
    pos_scores: np.ndarray,
    results: dict[str, object],
    key_prefix: str,
) -> None:
    if neg_scores.size < 10 or pos_scores.size < 10:
        log(f"  {name}: insufficient data (neg={neg_scores.size}, pos={pos_scores.size})")
        return
    labels = np.concatenate([np.zeros(len(neg_scores)), np.ones(len(pos_scores))])
    preds = np.concatenate([neg_scores, pos_scores])
    auc = _compute_auc_safe(labels, preds)
    if auc is None:
        log(f"  {name}: could not compute AUC")
        return
    log(f"  {name}:")
    log(f"    negative: n={len(neg_scores):,}  mean={neg_scores.mean():.6f}")
    log(f"    positive: n={len(pos_scores):,}  mean={pos_scores.mean():.6f}")
    log(f"    diff={pos_scores.mean() - neg_scores.mean():.6f}  >>> AUC={auc:.4f} <<<")
    results[f"{key_prefix}_auc"] = auc
    results[f"{key_prefix}_n_neg"] = len(neg_scores)
    results[f"{key_prefix}_n_pos"] = len(pos_scores)
    results[f"{key_prefix}_mean_neg"] = float(neg_scores.mean())
    results[f"{key_prefix}_mean_pos"] = float(pos_scores.mean())


def evaluate_all_of_us(
    output_dir: Path,
    disease: str,
) -> dict[str, object]:
    work_dir = Path(output_dir)

    # Find predictions
    predictions_path = None
    for candidate in [work_dir / "predictions.tsv.gz", work_dir.parent / "hypertension_result" / "predictions.tsv.gz"]:
        if candidate.exists():
            predictions_path = candidate
            break
    if predictions_path is None:
        raise FileNotFoundError(f"No predictions.tsv.gz found in {work_dir}")

    # Find sample table
    sample_table_path = None
    for parent in [work_dir, work_dir.parent / "hypertension_results"]:
        candidate = parent / f"{disease}.samples.tsv"
        if candidate.exists():
            sample_table_path = candidate
            break
    if sample_table_path is None:
        raise FileNotFoundError(f"No {disease}.samples.tsv found")

    # Find ancestry file
    ancestry_path = None
    for parent in [work_dir, work_dir.parent / "hypertension_results"]:
        candidate = parent / "ancestry_preds.tsv"
        if candidate.exists():
            ancestry_path = candidate
            break

    log("=== QUASI-HOLDOUT EVALUATION ===")
    log(f"  disease: {disease}")
    log(f"  predictions: {predictions_path}")
    log(f"  sample table: {sample_table_path}")
    log(f"  ancestry: {ancestry_path or 'not found'}")

    # Load predictions — auto-detect score column
    scores: dict[str, float] = {}
    with gzip.open(predictions_path, "rt") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        columns = reader.fieldnames or []
        score_col = None
        for candidate in ("probability", "predicted_probability", "genetic_score", "linear_predictor"):
            if candidate in columns:
                score_col = candidate
                break
        if score_col is None:
            raise ValueError(f"No score column found in predictions. Columns: {columns}")
        log(f"  using score column: {score_col}")
        for row in reader:
            scores[row["sample_id"]] = float(row[score_col])
    log(f"  loaded {len(scores):,} predicted scores")

    # Load sample table
    observation_counts: dict[str, int] = {}
    with open(sample_table_path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            observation_counts[row["sample_id"]] = int(row["phenotype_occurrence_count"])

    # Load ancestry
    ancestry_labels: dict[str, str] = {}
    if ancestry_path is not None:
        ancestry_labels = _load_ancestry_labels(ancestry_path)
        if ancestry_labels:
            eur_count = sum(1 for v in ancestry_labels.values() if v.lower() in ("eur", "european"))
            log(f"  ancestry labels: {len(ancestry_labels):,} total, {eur_count:,} EUR")

    eur_ids = {sid for sid, anc in ancestry_labels.items() if anc.lower() in ("eur", "european")} if ancestry_labels else set()

    # Group by observation count
    groups: dict[int, list[tuple[str, float]]] = {}
    for sid, cnt in observation_counts.items():
        if sid in scores:
            groups.setdefault(cnt, []).append((sid, scores[sid]))

    results: dict[str, object] = {"disease": disease}

    # === Test 1: ICD code stratification ===
    log("")
    log("=== TEST 1: ICD Code Stratification (0-code vs 1-code) ===")
    log("  Both groups were target=0 during training.")

    zero_items = groups.get(0, [])
    one_items = groups.get(1, [])
    zero_scores_arr = np.array([s for _, s in zero_items])
    one_scores_arr = np.array([s for _, s in one_items])

    _run_auc_test("ALL", zero_scores_arr, one_scores_arr, results, "test1_all")

    if eur_ids:
        zero_eur = np.array([s for sid, s in zero_items if sid in eur_ids])
        one_eur = np.array([s for sid, s in one_items if sid in eur_ids])
        _run_auc_test("EUR only", zero_eur, one_eur, results, "test1_eur")

    # === Test 2: Survey self-report ===
    log("")
    log("=== TEST 2: Survey Self-Report Validation ===")
    log("  Among controls (0-1 codes) + missing: survey-positive vs survey-negative.")

    survey_positive = _fetch_survey_self_report(disease)

    if survey_positive:
        control_or_missing = {sid for sid, cnt in observation_counts.items() if cnt <= 1 and sid in scores}
        missing_ehr = set(scores.keys()) - set(observation_counts.keys())
        eval_pool = control_or_missing | missing_ehr

        survey_pos_ids = eval_pool & survey_positive
        survey_neg_ids = eval_pool - survey_positive

        pos_arr = np.array([scores[sid] for sid in survey_pos_ids])
        neg_arr = np.array([scores[sid] for sid in survey_neg_ids])

        _run_auc_test("ALL", neg_arr, pos_arr, results, "test2_all")

        if eur_ids:
            pos_eur = np.array([scores[sid] for sid in survey_pos_ids if sid in eur_ids])
            neg_eur = np.array([scores[sid] for sid in survey_neg_ids if sid in eur_ids])
            _run_auc_test("EUR only", neg_eur, pos_eur, results, "test2_eur")
    else:
        log("  no survey data available")

    # === Dose-response ===
    log("")
    log("=== DOSE-RESPONSE (score vs observation count) ===")
    for cnt in sorted(groups.keys()):
        items = groups[cnt]
        g = np.array([s for _, s in items])
        if cnt <= 5 or cnt in (10, 20, 50, 100) or cnt == max(groups.keys()):
            log(f"  {cnt:>3} codes: n={len(g):>6,}  mean_score={g.mean():.6f}")

    # Save
    eval_output = work_dir / f"{disease}.evaluation.json"
    with open(eval_output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    log(f"\n  results saved to {eval_output}")
    log("=== EVALUATION COMPLETE ===")
    return results

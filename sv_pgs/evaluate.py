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
from typing import Literal

import numpy as np

from sv_pgs._typing import NDArray
from sv_pgs.progress import log

EvaluationPurpose = Literal["genetic_only", "full_model"]


def _get_cdr_dataset() -> str:
    import os
    dataset = os.environ.get("WORKSPACE_CDR", "")
    if not dataset:
        raise RuntimeError("WORKSPACE_CDR environment variable required for survey query")
    return dataset


def _build_survey_hypertension_sql(dataset: str) -> str:
    # AoU Personal and Family Health History (PFHH) survey structure:
    #   question: "Including yourself, who in your family has had ...high blood pressure..."
    #   answer: "... - Self" means the participant themselves reported it.
    # Pattern discovered from SauersML/ferromic phewas/extra/family.py
    return f"""
SELECT DISTINCT
  CAST(person_id AS STRING) AS person_id
FROM `{dataset}.ds_survey`
WHERE (LOWER(question) LIKE '%blood pressure%' OR LOWER(question) LIKE '%hypertension%')
  AND LOWER(answer) LIKE '%self%'
""".strip()


def _build_survey_diagnostic_sql(dataset: str) -> str:
    """Discovery query to find blood-pressure-related rows in ds_survey."""
    return f"""
SELECT
  survey,
  question_concept_id,
  SUBSTR(question, 1, 120) AS question_prefix,
  answer_concept_id,
  answer,
  COUNT(DISTINCT person_id) AS n_people
FROM `{dataset}.ds_survey`
WHERE LOWER(question) LIKE '%blood%'
   OR LOWER(question) LIKE '%hypertension%'
   OR LOWER(answer) LIKE '%hypertension%'
   OR LOWER(answer) LIKE '%blood pressure%'
GROUP BY 1, 2, 3, 4, 5
ORDER BY n_people DESC
LIMIT 30
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
    try:
        dataset = _get_cdr_dataset()
    except RuntimeError as exc:
        log(f"  {exc}")
        return set()

    client = bigquery.Client()

    # First try the main query
    sql = _build_survey_hypertension_sql(dataset)
    log("  querying BigQuery for survey self-reported hypertension...")
    try:
        rows = client.query(sql).result()
        positive_ids = {str(row.person_id) for row in rows}
        if positive_ids:
            log(f"  {len(positive_ids):,} people self-reported hypertension in survey")
            return positive_ids
        log("  main query returned 0 — running diagnostic query to discover table schema...")
    except Exception as exc:
        log(f"  main query failed: {exc} — running diagnostic...")

    # Diagnostic: show what blood-pressure-related data actually exists
    try:
        diag_sql = _build_survey_diagnostic_sql(dataset)
        diag_rows = list(client.query(diag_sql).result())
        if diag_rows:
            log(f"  found {len(diag_rows)} blood-pressure-related row patterns in ds_survey:")
            for row in diag_rows[:15]:
                log(f"    survey={row.survey}  q_id={row.question_concept_id}  "
                    f"q={row.question_prefix!r}  a_id={row.answer_concept_id}  "
                    f"a={row.answer!r}  n={row.n_people:,}")
        else:
            log("  diagnostic also returned 0 — ds_survey may not contain blood pressure data")
    except Exception as exc:
        log(f"  diagnostic query failed: {exc}")

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


def _compute_auc_safe(labels: NDArray, preds: NDArray) -> float | None:
    labels = np.asarray(labels, dtype=np.int8).reshape(-1)
    preds = np.asarray(preds, dtype=np.float64).reshape(-1)
    finite = np.isfinite(preds)
    labels = labels[finite]
    preds = preds[finite]
    if len(labels) < 20 or len(np.unique(labels)) < 2:
        return None
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    if n_pos == 0 or n_neg == 0:
        return None

    order = np.argsort(preds, kind="mergesort")
    sorted_preds = preds[order]
    ranks = np.empty_like(preds, dtype=np.float64)
    start = 0
    while start < sorted_preds.size:
        end = start + 1
        while end < sorted_preds.size and sorted_preds[end] == sorted_preds[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end

    positive_rank_sum = float(np.sum(ranks[labels == 1]))
    auc = (positive_rank_sum - n_pos * (n_pos + 1) * 0.5) / (n_pos * n_neg)
    return float(np.clip(auc, 0.0, 1.0))


def _score_column_priority(evaluation_purpose: EvaluationPurpose) -> tuple[str, ...]:
    if evaluation_purpose == "genetic_only":
        return ("genetic_score", "linear_predictor", "probability", "predicted_probability")
    if evaluation_purpose == "full_model":
        return ("probability", "predicted_probability", "genetic_score", "linear_predictor")
    raise ValueError(f"Unknown evaluation_purpose: {evaluation_purpose!r}")


def _select_score_column(
    columns: list[str],
    evaluation_purpose: EvaluationPurpose,
    context: str,
) -> str:
    score_col = next((c for c in _score_column_priority(evaluation_purpose) if c in columns), None)
    if score_col is None:
        raise ValueError(f"No score column found for {context}. Columns: {columns}")
    if evaluation_purpose == "genetic_only" and score_col in ("probability", "predicted_probability"):
        log(
            f"  WARNING: genetic_only requested for {context}, but no genetic_score or "
            f"linear_predictor column is available; falling back to {score_col}"
        )
    return score_col


def _run_auc_test(
    name: str,
    neg_scores: NDArray,
    pos_scores: NDArray,
    results: dict[str, object],
    key_prefix: str,
    *,
    evaluation_purpose: EvaluationPurpose = "full_model",
    score_column: str | None = None,
) -> None:
    if score_column is not None:
        results[f"{key_prefix}_score_column"] = score_column
    results[f"{key_prefix}_evaluation_purpose"] = evaluation_purpose

    neg_scores = np.asarray(neg_scores, dtype=np.float64).reshape(-1)
    pos_scores = np.asarray(pos_scores, dtype=np.float64).reshape(-1)
    original_neg_size = int(neg_scores.size)
    original_pos_size = int(pos_scores.size)
    neg_scores = neg_scores[np.isfinite(neg_scores)]
    pos_scores = pos_scores[np.isfinite(pos_scores)]
    dropped = (original_neg_size - int(neg_scores.size)) + (original_pos_size - int(pos_scores.size))
    if dropped > 0:
        log(f"  {name}: dropped {dropped:,} non-finite scores before AUC")
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
    evaluation_purpose: EvaluationPurpose = "full_model",
) -> dict[str, object]:
    work_dir = Path(output_dir)
    if evaluation_purpose not in ("genetic_only", "full_model"):
        raise ValueError(f"Unknown evaluation_purpose: {evaluation_purpose!r}")
    quasi_holdout_purpose: EvaluationPurpose = "genetic_only"

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

    results: dict[str, object] = {
        "disease": disease,
        "evaluation_purpose": evaluation_purpose,
        "quasi_holdout_evaluation_purpose": quasi_holdout_purpose,
    }

    # Load predictions for quasi-holdout genetic validation.
    scores: dict[str, float] = {}
    with gzip.open(predictions_path, "rt") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        columns = reader.fieldnames or []
        score_col = _select_score_column(columns, quasi_holdout_purpose, "quasi-holdout evaluation")
        results["quasi_holdout_score_column"] = score_col
        log(f"  using score column for quasi-holdout evaluation: {score_col}")
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

    # === Test 1: ICD code stratification ===
    log("")
    log("=== TEST 1: ICD Code Stratification (0-code vs 1-code) ===")
    log("  Both groups were target=0 during training.")

    zero_items = groups.get(0, [])
    one_items = groups.get(1, [])
    zero_scores_arr = np.array([s for _, s in zero_items])
    one_scores_arr = np.array([s for _, s in one_items])

    _run_auc_test(
        "ALL",
        zero_scores_arr,
        one_scores_arr,
        results,
        "test1_all",
        evaluation_purpose=quasi_holdout_purpose,
        score_column=score_col,
    )

    if eur_ids:
        zero_eur = np.array([s for sid, s in zero_items if sid in eur_ids])
        one_eur = np.array([s for sid, s in one_items if sid in eur_ids])
        _run_auc_test(
            "EUR only",
            zero_eur,
            one_eur,
            results,
            "test1_eur",
            evaluation_purpose=quasi_holdout_purpose,
            score_column=score_col,
        )

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

        _run_auc_test(
            "ALL",
            neg_arr,
            pos_arr,
            results,
            "test2_all",
            evaluation_purpose=quasi_holdout_purpose,
            score_column=score_col,
        )

        if eur_ids:
            pos_eur = np.array([scores[sid] for sid in survey_pos_ids if sid in eur_ids])
            neg_eur = np.array([scores[sid] for sid in survey_neg_ids if sid in eur_ids])
            _run_auc_test(
                "EUR only",
                neg_eur,
                pos_eur,
                results,
                "test2_eur",
                evaluation_purpose=quasi_holdout_purpose,
                score_column=score_col,
            )
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

    # === TRUE 20% held-out test set ===
    # The fit pipeline holds out a deterministic 20% of samples (SHA-256 of
    # sample_id) and writes their predictions to test_predictions.tsv.gz. These
    # samples were *never* shown to the EM. Print the full battery on them.
    test_predictions_path = work_dir / "test_predictions.tsv.gz"
    if test_predictions_path.exists():
        log("")
        log("=== TRUE 20% HELD-OUT TEST SET ===")
        log(f"  predictions: {test_predictions_path}")

        test_rows: list[tuple[str, int, float, float, float, float]] = []
        with gzip.open(test_predictions_path, "rt") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            cols = reader.fieldnames or []
            try:
                test_score_col = _select_score_column(cols, evaluation_purpose, "held-out target evaluation")
                test_quasi_score_col = _select_score_column(
                    cols,
                    quasi_holdout_purpose,
                    "held-out ICD/survey validation",
                )
            except ValueError as exc:
                log(f"  {exc}")
            else:
                results["test_holdout_evaluation_purpose"] = evaluation_purpose
                results["test_holdout_score_column"] = test_score_col
                results["test_holdout_quasi_holdout_evaluation_purpose"] = quasi_holdout_purpose
                results["test_holdout_quasi_holdout_score_column"] = test_quasi_score_col
                log(f"  using score column for held-out target evaluation: {test_score_col}")
                log(f"  using score column for held-out ICD/survey validation: {test_quasi_score_col}")
                for row in reader:
                    sid = row["sample_id"]
                    target = int(float(row["target"])) if "target" in row else -1
                    score = float(row[test_score_col])
                    quasi_score = float(row[test_quasi_score_col])
                    genetic = float(row.get("genetic_score", "nan"))
                    covariate = float(row.get("covariate_score", "nan"))
                    test_rows.append((sid, target, score, quasi_score, genetic, covariate))

        if test_rows:
            test_ids = np.array([r[0] for r in test_rows])
            y = np.array([r[1] for r in test_rows], dtype=np.int8)
            p = np.array([r[2] for r in test_rows], dtype=np.float64)
            g = np.array([r[4] for r in test_rows], dtype=np.float64)
            c = np.array([r[5] for r in test_rows], dtype=np.float64)

            n_total = int(y.size)
            n_pos = int((y == 1).sum())
            n_neg = int((y == 0).sum())
            log(f"  test samples: {n_total:,}  cases={n_pos:,}  controls={n_neg:,}")

            mean_pos = float(p[y == 1].mean()) if n_pos else float("nan")
            mean_neg = float(p[y == 0].mean()) if n_neg else float("nan")
            log(f"  mean score: cases={mean_pos:.6f}  controls={mean_neg:.6f}  diff={mean_pos - mean_neg:.6f}")
            if np.isfinite(g).all():
                log(f"  mean genetic_score:   cases={float(g[y==1].mean()):.6f}  controls={float(g[y==0].mean()):.6f}")
            if np.isfinite(c).all():
                log(f"  mean covariate_score: cases={float(c[y==1].mean()):.6f}  controls={float(c[y==0].mean()):.6f}")

            auc_overall = _compute_auc_safe(y, p)
            log(f"  ALL:           AUC={auc_overall if auc_overall is None else f'{auc_overall:.4f}'}  n={n_total:,}")
            results["test_holdout_auc_all"] = auc_overall
            results["test_holdout_n"] = n_total
            results["test_holdout_n_pos"] = n_pos
            results["test_holdout_n_neg"] = n_neg
            results["test_holdout_mean_score_cases"] = mean_pos
            results["test_holdout_mean_score_controls"] = mean_neg

            if np.isfinite(g).all():
                auc_genetic_only = _compute_auc_safe(y, g)
                log(f"  ALL (genetic-only):   AUC={auc_genetic_only if auc_genetic_only is None else f'{auc_genetic_only:.4f}'}")
                results["test_holdout_auc_genetic_only"] = auc_genetic_only
            if np.isfinite(c).all():
                auc_covariate_only = _compute_auc_safe(y, c)
                log(f"  ALL (covariate-only): AUC={auc_covariate_only if auc_covariate_only is None else f'{auc_covariate_only:.4f}'}")
                results["test_holdout_auc_covariate_only"] = auc_covariate_only

            if eur_ids:
                eur_mask = np.array([sid in eur_ids for sid in test_ids])
                if eur_mask.any():
                    auc_eur = _compute_auc_safe(y[eur_mask], p[eur_mask])
                    log(
                        f"  EUR only:      AUC={auc_eur if auc_eur is None else f'{auc_eur:.4f}'}  "
                        f"n={int(eur_mask.sum()):,}  cases={int((y[eur_mask]==1).sum()):,}  "
                        f"controls={int((y[eur_mask]==0).sum()):,}"
                    )
                    results["test_holdout_auc_eur"] = auc_eur
                    results["test_holdout_n_eur"] = int(eur_mask.sum())

            # Dose-response on the held-out set
            test_scores_by_id = {sid: quasi_score for sid, _, _, quasi_score, _, _ in test_rows}
            test_groups: dict[int, list[float]] = {}
            for sid, score in test_scores_by_id.items():
                sid_cnt = observation_counts.get(sid)
                if sid_cnt is not None:
                    test_groups.setdefault(sid_cnt, []).append(score)
            if test_groups:
                log("  dose-response on held-out (genetic-only score vs observation count):")
                max_cnt_test = max(test_groups.keys())
                for cnt in sorted(test_groups.keys()):
                    arr = np.asarray(test_groups[cnt])
                    if cnt <= 5 or cnt in (10, 20, 50, 100) or cnt == max_cnt_test:
                        log(f"    {cnt:>3} codes: n={len(arr):>6,}  mean_score={arr.mean():.6f}")

            # ICD stratification on held-out: 0-code vs 1-code (both target=0)
            zero_h = np.array([test_scores_by_id[sid] for sid in test_ids if observation_counts.get(sid) == 0 and sid in test_scores_by_id])
            one_h = np.array([test_scores_by_id[sid] for sid in test_ids if observation_counts.get(sid) == 1 and sid in test_scores_by_id])
            if zero_h.size >= 10 and one_h.size >= 10:
                log("  ICD 0-code vs 1-code (held-out):")
                _run_auc_test(
                    "ALL",
                    zero_h,
                    one_h,
                    results,
                    "test_holdout_icd01_all",
                    evaluation_purpose=quasi_holdout_purpose,
                    score_column=test_quasi_score_col,
                )
                if eur_ids:
                    zero_h_eur = np.array([test_scores_by_id[sid] for sid in test_ids if observation_counts.get(sid) == 0 and sid in eur_ids and sid in test_scores_by_id])
                    one_h_eur = np.array([test_scores_by_id[sid] for sid in test_ids if observation_counts.get(sid) == 1 and sid in eur_ids and sid in test_scores_by_id])
                    _run_auc_test(
                        "EUR only",
                        zero_h_eur,
                        one_h_eur,
                        results,
                        "test_holdout_icd01_eur",
                        evaluation_purpose=quasi_holdout_purpose,
                        score_column=test_quasi_score_col,
                    )

            # Survey self-report on held-out
            if survey_positive:
                test_id_set = set(test_ids.tolist())
                controls_or_missing_h = {sid for sid in test_id_set if observation_counts.get(sid, 0) <= 1}
                missing_ehr_h = test_id_set - set(observation_counts.keys())
                pool_h = controls_or_missing_h | missing_ehr_h
                pos_ids_h = pool_h & survey_positive
                neg_ids_h = pool_h - survey_positive
                pos_arr_h = np.array([test_scores_by_id[sid] for sid in pos_ids_h if sid in test_scores_by_id])
                neg_arr_h = np.array([test_scores_by_id[sid] for sid in neg_ids_h if sid in test_scores_by_id])
                if pos_arr_h.size >= 10 and neg_arr_h.size >= 10:
                    log("  Survey self-report (held-out):")
                    _run_auc_test(
                        "ALL",
                        neg_arr_h,
                        pos_arr_h,
                        results,
                        "test_holdout_survey_all",
                        evaluation_purpose=quasi_holdout_purpose,
                        score_column=test_quasi_score_col,
                    )
                    if eur_ids:
                        pos_eur_h = np.array([test_scores_by_id[sid] for sid in pos_ids_h if sid in eur_ids and sid in test_scores_by_id])
                        neg_eur_h = np.array([test_scores_by_id[sid] for sid in neg_ids_h if sid in eur_ids and sid in test_scores_by_id])
                        _run_auc_test(
                            "EUR only",
                            neg_eur_h,
                            pos_eur_h,
                            results,
                            "test_holdout_survey_eur",
                            evaluation_purpose=quasi_holdout_purpose,
                            score_column=test_quasi_score_col,
                        )
    else:
        log("")
        log(f"  (no test_predictions.tsv.gz found at {test_predictions_path} — held-out battery skipped)")

    # Save
    eval_output = work_dir / f"{disease}.evaluation.json"
    with open(eval_output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    log(f"\n  results saved to {eval_output}")
    log("=== EVALUATION COMPLETE ===")
    return results

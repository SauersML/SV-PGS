"""Pin: quasi-holdout evaluation prefers ``genetic_score`` over ``probability``.

The quasi-holdout AUC is supposed to measure the genetic component alone, NOT
the full-model probability (which incorporates covariates and would inflate
the apparent PGS AUC by reading covariate effects). The ``genetic_only``
score-column priority must put ``genetic_score`` first.

Adversarial: when both columns are present in the predictions table but the
probability column would yield AUC=0.5 and the genetic_score column would
yield AUC=0.9, the selector must pick ``genetic_score``.
"""
from __future__ import annotations

import pytest

from sv_pgs import evaluate as evaluate_module


def test_genetic_only_priority_picks_genetic_score():
    """``genetic_only`` purpose: priority must place ``genetic_score`` first."""
    priority = evaluate_module._score_column_priority("genetic_only")
    assert priority[0] == "genetic_score", (
        f"genetic_only must prefer genetic_score; got {priority!r}"
    )
    # linear_predictor is the secondary genetic-only alias.
    assert "linear_predictor" in priority[:2]


def test_full_model_priority_picks_probability_first():
    """``full_model`` purpose prefers probability (full posterior outcome)."""
    priority = evaluate_module._score_column_priority("full_model")
    assert priority[0] == "probability"


def test_select_score_column_returns_genetic_when_both_present(capsys):
    """When both ``genetic_score`` and ``probability`` exist in the columns,
    a ``genetic_only`` request must return ``genetic_score`` — NOT fall back
    to probability (which would inflate the quasi-holdout AUC)."""
    columns = ["sample_id", "target", "genetic_score", "covariate_score", "probability"]
    selected = evaluate_module._select_score_column(
        columns, "genetic_only", "quasi-holdout test"
    )
    assert selected == "genetic_score"


def test_select_score_column_falls_back_to_probability_with_warning(capsys):
    """If genetic_score / linear_predictor are absent, the selector falls
    back to probability and emits a warning that the operator can spot in
    logs (so a missing column doesn't silently corrupt the AUC report)."""
    columns = ["sample_id", "target", "probability"]
    selected = evaluate_module._select_score_column(
        columns, "genetic_only", "quasi-holdout test"
    )
    assert selected == "probability"
    captured = capsys.readouterr().out
    assert "WARNING" in captured or "warning" in captured.lower()


def test_select_score_column_raises_when_no_score_available():
    """No usable column → ValueError. Pin this so a future swarm cannot
    silently default to a non-score column."""
    columns = ["sample_id", "target"]
    with pytest.raises(ValueError, match="No score column"):
        evaluate_module._select_score_column(columns, "genetic_only", "context")


def test_unknown_evaluation_purpose_raises():
    """Typo in evaluation_purpose must raise immediately."""
    with pytest.raises(ValueError):
        evaluate_module._score_column_priority("not_a_purpose")  # type: ignore[arg-type]

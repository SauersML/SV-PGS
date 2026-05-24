"""Pin: ``_select_score_column`` raises on an empty columns list.

Silent ``return None`` would propagate as a downstream ``KeyError`` far
from the source.  Explicit ``ValueError`` keeps the failure local and
includes the columns and context in the message.
"""
from __future__ import annotations

import pytest

from sv_pgs.evaluate import _select_score_column


def test_empty_columns_raises_value_error():
    with pytest.raises(ValueError, match="No score column found"):
        _select_score_column([], "genetic_only", context="unit-test")


def test_empty_columns_full_model_purpose_also_raises():
    with pytest.raises(ValueError, match="No score column found"):
        _select_score_column([], "full_model", context="unit-test")


def test_unknown_purpose_raises_value_error():
    with pytest.raises(ValueError, match="Unknown evaluation_purpose"):
        _select_score_column(["genetic_score"], "made_up_purpose", context="unit-test")  # type: ignore[arg-type]


def test_columns_with_no_relevant_entry_raises():
    """Columns present but none in the priority list → ValueError."""
    with pytest.raises(ValueError, match="No score column found"):
        _select_score_column(
            ["unrelated_a", "unrelated_b"],
            "genetic_only",
            context="unit-test",
        )

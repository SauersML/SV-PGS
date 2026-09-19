"""The survey self-report test fails loudly instead of silently finding no one."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from google.cloud import bigquery

from sv_pgs.evaluate import _fetch_survey_self_report


class _FakeQueryJob:
    def __init__(self, rows: list[SimpleNamespace]) -> None:
        self._rows = rows

    def result(self) -> list[SimpleNamespace]:
        return self._rows


class _FakeClient:
    """Answers the self-report query with ``self_report_rows`` and any other with ``diagnostic_rows``."""

    self_report_rows: list[SimpleNamespace] = []
    diagnostic_rows: list[SimpleNamespace] = []

    def query(self, sql: str) -> _FakeQueryJob:
        if "SELECT DISTINCT" in sql:
            return _FakeQueryJob(self.self_report_rows)
        return _FakeQueryJob(self.diagnostic_rows)


def test_other_diseases_have_no_survey_mapping() -> None:
    assert _fetch_survey_self_report("type2_diabetes") == set()


def test_missing_cdr_dataset_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WORKSPACE_CDR", raising=False)
    with pytest.raises(ValueError, match="WORKSPACE_CDR"):
        _fetch_survey_self_report("hypertension")


def test_self_reports_are_returned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORKSPACE_CDR", "aou_workspace.cdr_dataset")
    monkeypatch.setattr(_FakeClient, "self_report_rows", [SimpleNamespace(person_id=7), SimpleNamespace(person_id="9")])
    monkeypatch.setattr(bigquery, "Client", _FakeClient)
    assert _fetch_survey_self_report("hypertension") == {"7", "9"}


def test_query_matching_nobody_raises_after_logging_the_diagnostic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORKSPACE_CDR", "aou_workspace.cdr_dataset")
    diagnostic_row = SimpleNamespace(
        survey="Personal and Family Health History",
        question_concept_id=1,
        question_prefix="blood pressure",
        answer_concept_id=2,
        answer="Self",
        n_people=100,
    )
    monkeypatch.setattr(_FakeClient, "self_report_rows", [])
    monkeypatch.setattr(_FakeClient, "diagnostic_rows", [diagnostic_row])
    monkeypatch.setattr(bigquery, "Client", _FakeClient)
    with pytest.raises(RuntimeError, match="matched no participants"):
        _fetch_survey_self_report("hypertension")


def test_query_errors_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FailingClient:
        def query(self, sql: str) -> _FakeQueryJob:
            raise RuntimeError("bigquery unavailable")

    monkeypatch.setenv("WORKSPACE_CDR", "aou_workspace.cdr_dataset")
    monkeypatch.setattr(bigquery, "Client", _FailingClient)
    with pytest.raises(RuntimeError, match="bigquery unavailable"):
        _fetch_survey_self_report("hypertension")

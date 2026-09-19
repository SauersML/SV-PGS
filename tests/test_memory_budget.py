"""The address-space budget helper: untouched reservations past the plan fail, the plan itself passes."""
from __future__ import annotations

import sys

import numpy as np
import pytest

from tests.memory_budget import run_within_budget

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc/self/status")

_BUDGET = 64 * 1024 * 1024
_FLOAT64_BYTES = 8


def touched_sum(count: int) -> float:
    """Allocate and write ``count`` float64 values, then sum them."""
    return float(np.ones(count).sum())


def untouched_reservation(count: int) -> int:
    """Reserve ``count`` float64 values without writing them, as a preallocated work basis would."""
    return int(np.empty(count).shape[0])


def column_sums(values: np.ndarray) -> np.ndarray:
    return values.sum(axis=0)


def refuse(message: str) -> None:
    raise ValueError(message)


def test_a_computation_within_its_plan_returns_its_result() -> None:
    count = _BUDGET // (2 * _FLOAT64_BYTES)

    assert run_within_budget(touched_sum, count, working_bytes=_BUDGET) == float(count)


def test_an_untouched_reservation_past_the_plan_fails() -> None:
    # Ten budgets of address space, never written: overcommit would admit it, the budget does not.
    count = 10 * _BUDGET // _FLOAT64_BYTES

    with pytest.raises(AssertionError, match="past its working_bytes budget"):
        run_within_budget(untouched_reservation, count, working_bytes=_BUDGET)


def test_the_arguments_are_not_charged_to_the_budget() -> None:
    # An input four budgets large, loaded before the limit, and a reduction that needs almost nothing.
    values = np.ones((4 * _BUDGET // (_FLOAT64_BYTES * 1024), 1024))

    np.testing.assert_array_equal(run_within_budget(column_sums, values, working_bytes=_BUDGET), values.sum(axis=0))


def test_other_exceptions_reach_the_caller() -> None:
    with pytest.raises(ValueError, match="not a budget"):
        run_within_budget(refuse, "not a budget", working_bytes=_BUDGET)

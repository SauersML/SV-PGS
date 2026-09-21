"""The address-space budget helper: untouched reservations past the plan fail, the plan itself passes."""
from __future__ import annotations

import sys

import numpy as np
import pytest

from sv_pgs.code_products import CodeBlockTile
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


def tile_rmatmat(aligned: np.ndarray, means: np.ndarray, scales: np.ndarray, left: np.ndarray, workspace_bytes: int) -> np.ndarray:
    """X_b.T @ left from a CPU CodeBlockTile over codes the caller already holds aligned (no copy)."""
    variant_count, sample_count = aligned.shape
    tile = CodeBlockTile.from_aligned(
        aligned, variant_count, sample_count, means, scales, float(scales.max() / scales.min()), np, workspace_bytes
    )
    return tile.rmatmat(left)


def test_a_code_tile_product_stays_within_its_workspace() -> None:
    # CodeBlockTile's CPU plan (code_products._codes_times): fixed, the padded operand and two (p, K) float64
    # arrays; per sample chunk, its codes converted to float64. A workspace holding an eighth of the samples
    # per chunk is far below converting every sample at once.
    rng = np.random.default_rng(11)
    variants, samples, columns = 8, 200_000, 2
    aligned = rng.integers(-127, 128, size=(variants, samples)).astype(np.int8)
    means = aligned.mean(axis=1)
    scales = aligned.std(axis=1)
    left = rng.standard_normal((samples, columns))
    fixed = _FLOAT64_BYTES * (samples * columns + 2 * variants * columns)
    workspace_bytes = fixed + (samples // 8) * _FLOAT64_BYTES * variants
    # rmatmat's own (p, K) results outside the tile's workspace: the centring term, the centred products, the scaled result.
    outputs = 3 * _FLOAT64_BYTES * variants * columns

    product = run_within_budget(tile_rmatmat, aligned, means, scales, left, workspace_bytes, working_bytes=workspace_bytes + outputs)

    # The product's accuracy is test_code_products' concern; here it only has to complete within the plan.
    assert product.shape == (variants, columns) and np.all(np.isfinite(product))
    # The same product given a quarter of that budget cannot hold even its padded operand copy.
    with pytest.raises(AssertionError, match="past its working_bytes budget"):
        run_within_budget(tile_rmatmat, aligned, means, scales, left, workspace_bytes, working_bytes=workspace_bytes // 4)

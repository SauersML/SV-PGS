"""The online LD-block partitioner equals a brute-force dynamic program."""

from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.ld_partition import (
    OnlineBlockPartitioner,
    cut_allowed_from_groups,
    fixed_point_pair_weights,
    PAIR_WEIGHT_SCALE,
)
from tests.stage0_support import reference_cuts


def _partitioner_with_costs(costs: np.ndarray, allowed: np.ndarray, block_cap: int) -> OnlineBlockPartitioner:
    partitioner = OnlineBlockPartitioner(allowed, block_cap)
    partitioner.add_pair_weights(0, np.diff(costs), 0, np.zeros(0, dtype=np.int64))
    return partitioner


def _random_problem(rng: np.random.Generator, positions: int, block_cap: int) -> tuple[np.ndarray, np.ndarray]:
    costs = rng.integers(0, 40, size=positions + 1).astype(np.int64)
    costs[rng.random(positions + 1) < 0.3] = rng.integers(0, 3)
    costs[0] = 0
    costs[-1] = 0
    groups = np.repeat(np.arange(positions), rng.integers(1, max(2, block_cap // 3), size=positions))[:positions]
    return costs, cut_allowed_from_groups(groups)


@pytest.mark.parametrize("seed", range(12))
def test_online_partition_matches_brute_force(seed: int) -> None:
    rng = np.random.default_rng(seed)
    block_cap = int(rng.integers(3, 40))
    positions = int(rng.integers(block_cap, 400))
    costs, allowed = _random_problem(rng, positions, block_cap)
    expected = reference_cuts(costs, allowed, block_cap)

    partitioner = _partitioner_with_costs(costs, allowed, block_cap)
    released = [0]
    frontier = 0
    while frontier < positions:
        frontier = min(positions, frontier + int(rng.integers(1, 2 * block_cap)))
        released += partitioner.advance(frontier)
    released += partitioner.finish()
    assert released == expected


def test_released_blocks_are_certain_before_the_end() -> None:
    rng = np.random.default_rng(3)
    block_cap = 16
    positions = 600
    costs = rng.integers(5, 50, size=positions + 1).astype(np.int64)
    costs[::11] = 0
    costs[0] = costs[-1] = 0
    allowed = np.ones(positions + 1, dtype=np.bool_)
    partitioner = _partitioner_with_costs(costs, allowed, block_cap)
    early = partitioner.advance(positions // 2)
    assert len(early) > 5
    assert early == reference_cuts(costs, allowed, block_cap)[1 : len(early) + 1]


@pytest.mark.parametrize("seed", range(6))
def test_forced_cuts_give_a_valid_partition(seed: int) -> None:
    rng = np.random.default_rng(100 + seed)
    block_cap = int(rng.integers(4, 24))
    positions = int(rng.integers(4 * block_cap, 500))
    costs, allowed = _random_problem(rng, positions, block_cap)
    partitioner = _partitioner_with_costs(costs, allowed, block_cap)
    released = [0]
    for frontier in range(block_cap, positions, block_cap):
        released += partitioner.advance(frontier)
        while frontier - partitioner.committed > 2 * block_cap:
            released += partitioner.force_cut()
    released += partitioner.finish()
    assert released[-1] == positions
    assert all(allowed[cut] for cut in released)
    assert np.all(np.diff(released) >= 1)
    assert np.all(np.diff(released) <= block_cap)


def test_an_unsplittable_run_longer_than_the_cap_is_rejected() -> None:
    groups = np.array([0, 1, 1, 1, 1, 1, 2], dtype=np.int64)
    with pytest.raises(ValueError, match="exceed the LD block cap 4"):
        OnlineBlockPartitioner(cut_allowed_from_groups(groups), 4)


def test_pair_weights_are_bias_corrected_squared_correlations() -> None:
    rng = np.random.default_rng(0)
    signed = rng.integers(-127, 128, size=(6, 300)).astype(np.int64)
    signed[3] = signed[1] // 2 + rng.integers(-20, 20, size=300)
    signed[5] = 7
    band = signed @ signed.T
    sums = signed.sum(axis=1)
    squares = (signed * signed).sum(axis=1)
    weights = fixed_point_pair_weights(band, sums, squares, sums, squares, 300, 0, 3)
    correlation = np.corrcoef(signed[:5].astype(np.float64))
    for row in range(6):
        for column in range(6):
            distance = column - row
            if not 1 <= distance <= 3 or 5 in (row, column):
                assert weights[row, column] == 0
                continue
            expected = (correlation[row, column] ** 2 - 1.0 / 299) * PAIR_WEIGHT_SCALE
            assert abs(weights[row, column] - expected) <= 1.0

"""LDetect-style LD-block boundaries, chosen by an exact dynamic program during the genotype pass.

Cut positions ``k = 0 .. n_variants`` separate variant ``k - 1`` from variant ``k``; the
chromosome ends are always cuts. Every pair of variants ``i < j`` with ``j - i <= cap``
(the block cap; farther pairs are split by every admissible partition) carries a weight
``w_ij``: the squared correlation minus its null expectation ``1 / (n - 1)``, in fixed
point. A cut at ``k`` costs ``C(k) = sum over pairs i < k <= j of w_ij``, clipped at zero:
the LD it separates, the LDetect boundary criterion (Berisa & Pickrell 2016). The
partition minimizes the total cost of its interior cuts subject to

* consecutive cuts at most ``cap`` apart (every block has at most ``cap`` variants), and
* cuts only where ``cut_allowed`` (never inside a bubble or tandem-repeat locus).

The recursion ``B(k) = C(k) + min_{k - cap <= j < k} B(j)`` is solved ``cap`` positions at
a time without a Python loop: with ``P(k)`` the minimum over the already-solved part of
the window and ``Q(k) = C(k) + P(k)``, ``C >= 0`` gives
``min_{x <= j < k} B(j) = min_{x <= j < k} Q(j)`` for the chunk ``[x, x + cap)``, a cumulative
minimum. Ties go to the leftmost predecessor, so the solution is unique.

Blocks are released as soon as they are certain: once every chain that a future
position can extend passes through a common cut, that cut and the chain below it are
optimal whatever follows. If the retained genotype buffer fills first, ``force_cut``
commits the next cut of the currently best chain and re-solves from it, so the result
depends only on the data, the cap and the buffer rule (never on the hardware).
Costs are integers (``PAIR_WEIGHT_SCALE`` fixed point), so the choice is bit-identical
on every backend.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np
from numpy.typing import NDArray

PAIR_WEIGHT_SCALE = float(2**20)
"""Fixed-point scale of a pair weight: ``round(w_ij * 2^20)`` as int64."""

_UNREACHABLE = np.int64(2**61)


def fixed_point_pair_weights(
    array_module: ModuleType,
    band: NDArray[np.integer],
    row_sums: NDArray[np.int64],
    row_squares: NDArray[np.int64],
    column_sums: NDArray[np.int64],
    column_squares: NDArray[np.int64],
    profile_count: int,
    first_distance: int,
    maximum_distance: int,
) -> NDArray[np.int64]:
    """Fixed-point pair weights ``round((r_ij^2 - 1/(n-1)) * 2^20)`` of a band of profile products.

    ``band[a, b] = sum_profile s_i s_j`` for row variant ``i`` and column variant ``j`` with
    ``j - i = first_distance + b - a``; pairs with ``j - i`` outside ``[1, maximum_distance]``
    weigh zero, as do pairs with a constant variant. Every step is one correctly rounded fp64
    operation on exact integers, and ``array_module`` (NumPy or CuPy) runs each as its own
    kernel, so both give the same bits.
    """
    xp = array_module
    numerator = profile_count * band.astype(xp.int64) - xp.outer(row_sums, column_sums)
    row_variance = (profile_count * row_squares - row_sums * row_sums).astype(xp.float64)
    column_variance = (profile_count * column_squares - column_sums * column_sums).astype(xp.float64)
    row_varies = row_variance > 0
    column_varies = column_variance > 0
    denominator = xp.where(row_varies, row_variance, 1.0)[:, None] * xp.where(column_varies, column_variance, 1.0)[None, :]
    numerator_float = numerator.astype(xp.float64)
    squared = (numerator_float * numerator_float) / denominator
    weight = xp.rint((squared - 1.0 / (profile_count - 1)) * PAIR_WEIGHT_SCALE).astype(xp.int64)
    distance = first_distance + xp.arange(band.shape[1], dtype=xp.int64)[None, :] - xp.arange(band.shape[0], dtype=xp.int64)[:, None]
    keep = (distance >= 1) & (distance <= maximum_distance) & row_varies[:, None] & column_varies[None, :]
    return xp.where(keep, weight, 0)


def validate_cut_allowed(cut_allowed: NDArray[np.bool_], block_cap: int) -> None:
    """Raise if some run of variants that cannot be cut is longer than ``block_cap``."""
    allowed_positions = np.flatnonzero(cut_allowed)
    if allowed_positions.shape[0] < 2 or allowed_positions[0] != 0 or allowed_positions[-1] != cut_allowed.shape[0] - 1:
        raise ValueError("both chromosome ends must be admissible cuts")
    widest = int(np.max(np.diff(allowed_positions)))
    if widest > block_cap:
        start = int(allowed_positions[np.argmax(np.diff(allowed_positions))])
        raise ValueError(
            f"{widest} consecutive variants starting at variant {start} share one unsplittable "
            f"group (bubble or repeat locus) and exceed the LD block cap {block_cap}"
        )


def cut_allowed_from_groups(unsplittable_group: NDArray[np.int64]) -> NDArray[np.bool_]:
    """Cuts are admissible at the chromosome ends and between variants of different groups."""
    groups = np.asarray(unsplittable_group, dtype=np.int64)
    allowed = np.ones(groups.shape[0] + 1, dtype=np.bool_)
    allowed[1:-1] = groups[1:] != groups[:-1]
    return allowed


class OnlineBlockPartitioner:
    """The exact minimum-cost partition of one chromosome, released block by block."""

    def __init__(self, cut_allowed: NDArray[np.bool_], block_cap: int) -> None:
        validate_cut_allowed(cut_allowed, block_cap)
        self.block_cap = int(block_cap)
        self.variant_count = int(cut_allowed.shape[0] - 1)
        self._allowed = np.asarray(cut_allowed, dtype=np.bool_)
        self._difference = np.zeros(self.variant_count + 2, dtype=np.int64)
        self._cost = np.zeros(self.variant_count + 1, dtype=np.int64)
        self._best = np.full(self.variant_count + 1, _UNREACHABLE, dtype=np.int64)
        self._predecessor = np.full(self.variant_count + 1, -1, dtype=np.int64)
        self._best[0] = 0
        self._running_cost = np.int64(0)
        self._solved = 0
        self.committed = 0
        """The last released cut: every retained chain passes through it."""

    @property
    def cut_costs(self) -> NDArray[np.int64]:
        """``C(k)`` for every cut position solved so far (fixed point, clipped at zero)."""
        return self._cost[: self._solved + 1]

    def add_pair_weights(
        self,
        row_start: int,
        row_weights: NDArray[np.int64],
        column_start: int,
        column_weights: NDArray[np.int64],
    ) -> None:
        """Add pairs ``(i, j)``: ``row_weights[a]`` sums ``w_ij`` over the new pairs of
        ``i = row_start + a`` and ``column_weights[b]`` those of ``j = column_start + b``."""
        self._difference[row_start + 1 : row_start + 1 + row_weights.shape[0]] += row_weights
        self._difference[column_start + 1 : column_start + 1 + column_weights.shape[0]] -= column_weights

    def advance(self, costs_final_through: int) -> list[int]:
        """Solve through cut ``costs_final_through`` (every pair touching it is added) and
        return the cuts that became certain, ascending."""
        target = min(int(costs_final_through), self.variant_count - 1)
        if target <= self._solved:
            return []
        self._finalize_costs(target)
        self._solve(self._solved + 1, target + 1)
        self._solved = target
        return self._release(self._common_ancestor(self._live_positions()))

    def finish(self) -> list[int]:
        """Solve through the chromosome end and return the remaining cuts, the end included."""
        self._finalize_costs(self.variant_count)
        self._solve(self._solved + 1, self.variant_count + 1)
        self._solved = self.variant_count
        return self._release(self.variant_count)

    def force_cut(self) -> list[int]:
        """Commit the next cut of the best current chain (the buffer is full), re-solve from it,
        and return the cuts now certain, ascending."""
        live = self._live_positions()
        live = live[live > self.committed]
        if live.shape[0] == 0:
            raise AssertionError("no cut beyond the committed one is solved yet")
        leader = int(live[np.argmin(self._best[live])])
        cut = self._chain_above_committed(leader)[0]
        self.committed = cut
        self._solve(cut + 1, self._solved + 1)
        return [cut] + self._release(self._common_ancestor(self._live_positions()))

    def _finalize_costs(self, target: int) -> None:
        start = self._solved + 1 if self._solved > 0 else 0
        if start == 0:
            self._running_cost = np.int64(0)
        cumulative = self._running_cost + np.cumsum(self._difference[start : target + 1])
        self._running_cost = cumulative[-1]
        self._cost[start : target + 1] = np.maximum(cumulative, 0)

    def _solve(self, start: int, stop: int) -> None:
        for chunk_start in range(start, stop, self.block_cap):
            self._solve_chunk(chunk_start, min(chunk_start + self.block_cap, stop))

    def _solve_chunk(self, start: int, stop: int) -> None:
        window_low = max(self.committed, start - self.block_cap)
        known = self._best[window_low:start]
        suffix_minimum = np.minimum.accumulate(known[::-1])[::-1]
        known_positions = np.arange(window_low, start, dtype=np.int64)
        attains = np.where(known == suffix_minimum, known_positions, np.int64(self.variant_count + 1))
        suffix_argmin = np.minimum.accumulate(attains[::-1])[::-1]
        positions = np.arange(start, stop, dtype=np.int64)
        window_offset = np.maximum(positions - self.block_cap, window_low) - window_low
        prior_best = suffix_minimum[window_offset]
        prior_argmin = suffix_argmin[window_offset]
        cost = np.where(self._allowed[start:stop], self._cost[start:stop], _UNREACHABLE)
        through_prior = np.minimum(cost + prior_best, _UNREACHABLE)
        running = np.minimum.accumulate(through_prior)
        within_best = np.concatenate(([_UNREACHABLE], running[:-1]))
        new_minimum = np.concatenate(([True], through_prior[1:] < running[:-1]))
        within_argmin = np.concatenate(([-1], np.maximum.accumulate(np.where(new_minimum, positions, -1))[:-1]))
        use_prior = prior_best <= within_best
        window_best = np.where(use_prior, prior_best, within_best)
        best = np.minimum(cost + window_best, _UNREACHABLE)
        if np.any((best < _UNREACHABLE) & (best >= _UNREACHABLE // 2)):
            raise OverflowError("LD block cut costs exceed the fixed-point range")
        self._best[start:stop] = best
        self._predecessor[start:stop] = np.where(use_prior, prior_argmin, within_argmin)

    def _live_positions(self) -> NDArray[np.int64]:
        low = max(self.committed, self._solved - self.block_cap + 1)
        candidates = np.arange(low, self._solved + 1, dtype=np.int64)
        return candidates[self._best[candidates] < _UNREACHABLE]

    def _common_ancestor(self, live: NDArray[np.int64]) -> int:
        """The highest position on every chain from ``live`` (chains never cross ``committed``)."""
        frontier = np.unique(live)
        while frontier.shape[0] > 1:
            frontier = np.unique(np.concatenate((frontier[:1], self._predecessor[frontier[1:]])))
        return int(frontier[0])

    def _chain_above_committed(self, position: int) -> list[int]:
        chain = []
        while position > self.committed:
            chain.append(position)
            position = int(self._predecessor[position])
        if position != self.committed:
            raise AssertionError("an optimal chain skipped the committed cut")
        return chain[::-1]

    def _release(self, ancestor: int) -> list[int]:
        if ancestor <= self.committed:
            return []
        released = self._chain_above_committed(ancestor)
        self.committed = ancestor
        return released

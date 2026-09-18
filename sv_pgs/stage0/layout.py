"""Sample-axis layout of the Stage 0 genotype buffer.

The buffer holds ``s = code - 127`` for one variant per row. Its columns are the
included samples, grouped: group ``g`` occupies ``[offset_g, offset_g + count_g)``, then
zero columns up to the next multiple of ``LAYOUT_ALIGNMENT``. A zero column adds
nothing to any sum or cross-product, so every group is a contiguous, aligned
reduction range and padding is exact.

Inside each group the cut-cost profile samples come first. The profile is every
``stride``-th included sample in store order, a fixed subsample that does not depend
on the group labels. It only chooses block boundaries: the relative standard error of a
cut cost ``C`` (a sum of squared correlations) is about ``2 / sqrt(C * n)``, below 1.6%
for ``C >= 1`` at 16,384 samples, while the band products it needs cost
``n * p * cap`` operations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

LAYOUT_ALIGNMENT = 64
"""Column alignment of every group range (int8 tensor-core GEMM operands need 16-byte rows)."""

PROFILE_SAMPLE_TARGET = 16384
"""Cut-cost profile size (see the module docstring for the precision it buys)."""


@dataclass(frozen=True, slots=True)
class SampleLayout:
    """Mapping from buffer columns to store columns (``-1`` marks a zero pad column)."""

    source_columns: NDArray[np.int64]
    group_offsets: NDArray[np.int64]
    group_widths: NDArray[np.int64]
    group_counts: NDArray[np.int64]
    profile_counts: NDArray[np.int64]
    store_width: int

    @property
    def width(self) -> int:
        return int(self.source_columns.shape[0])

    @property
    def group_count(self) -> int:
        return int(self.group_counts.shape[0])

    @property
    def profile_count(self) -> int:
        return int(self.profile_counts.sum())

    def group_range(self, group: int) -> tuple[int, int]:
        """Aligned buffer columns of ``group``; its pad columns are zero."""
        start = int(self.group_offsets[group])
        return start, start + int(self.group_widths[group])

    def profile_range(self, group: int) -> tuple[int, int]:
        """Aligned buffer columns holding ``group``'s profile samples (and zero pads)."""
        start = int(self.group_offsets[group])
        return start, start + _aligned(int(self.profile_counts[group]))


def _aligned(count: int) -> int:
    return -(-count // LAYOUT_ALIGNMENT) * LAYOUT_ALIGNMENT


def build_sample_layout(sample_groups: NDArray[np.int64], profile_target: int = PROFILE_SAMPLE_TARGET) -> SampleLayout:
    """Lay out the store columns with ``sample_groups[column]`` in ``0..G-1`` (``-1`` excludes).

    Every group must be non-empty. Profile samples sit at the front of their group's range
    and are themselves padded to the alignment, so each group's profile is an aligned
    sub-range of its range.
    """
    groups = np.asarray(sample_groups, dtype=np.int64)
    if groups.ndim != 1:
        raise ValueError("sample_groups must be 1-D")
    if np.any(groups < -1):
        raise ValueError("sample group labels must be -1 (excluded) or 0..G-1")
    included = np.flatnonzero(groups >= 0)
    if included.shape[0] < 2:
        raise ValueError("Stage 0 needs at least two included samples")
    group_count = int(groups.max()) + 1
    stride = -(-included.shape[0] // profile_target)
    is_profile = np.zeros(groups.shape[0], dtype=np.bool_)
    is_profile[included[::stride]] = True
    columns: list[NDArray[np.int64]] = []
    offsets = np.zeros(group_count, dtype=np.int64)
    widths = np.zeros(group_count, dtype=np.int64)
    counts = np.zeros(group_count, dtype=np.int64)
    profile_counts = np.zeros(group_count, dtype=np.int64)
    cursor = 0
    for group in range(group_count):
        members = np.flatnonzero(groups == group)
        if members.shape[0] == 0:
            raise ValueError(f"sample group {group} is empty")
        profile_members = members[is_profile[members]]
        other_members = members[~is_profile[members]]
        profile_padding = _aligned(profile_members.shape[0]) - profile_members.shape[0]
        ordered = np.concatenate((profile_members, np.full(profile_padding, -1, dtype=np.int64), other_members))
        padded = np.concatenate((ordered, np.full(_aligned(ordered.shape[0]) - ordered.shape[0], -1, dtype=np.int64)))
        offsets[group] = cursor
        widths[group] = padded.shape[0]
        counts[group] = members.shape[0]
        profile_counts[group] = profile_members.shape[0]
        columns.append(padded)
        cursor += padded.shape[0]
    return SampleLayout(
        source_columns=np.concatenate(columns),
        group_offsets=offsets,
        group_widths=widths,
        group_counts=counts,
        profile_counts=profile_counts,
        store_width=int(groups.shape[0]),
    )

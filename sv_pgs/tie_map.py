"""Construction of TieMaps: which variants share one coefficient, and with which sign."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from sv_pgs.data import TieGroup, TieMap


def _compact_identity_tie_map(variant_count: int) -> TieMap:
    identity_indices = np.arange(int(variant_count), dtype=np.int32)
    return TieMap(
        kept_indices=identity_indices,
        original_to_reduced=identity_indices.copy(),
        reduced_to_group=[],
    )


def tie_map_from_groups(variant_count: int, groups: Mapping[int, Sequence[tuple[int, float]]]) -> TieMap:
    """The TieMap of ``variant_count`` variants whose every variant sits in one group.

    ``groups`` maps each representative to its ``(member, sign)`` pairs (the representative
    included, sign +1 for a copy and -1 for a negated copy); reduced order follows the
    representatives.
    """
    kept_indices: list[int] = []
    original_to_reduced = np.full(variant_count, -1, dtype=np.int32)
    reduced_to_group: list[TieGroup] = []
    for reduced_idx, (root, members) in enumerate(sorted(groups.items())):
        kept_indices.append(root)
        member_indices = np.array([member[0] for member in members], dtype=np.int32)
        signs = np.array([member[1] for member in members], dtype=np.float32)
        for member_index, _sign in members:
            original_to_reduced[member_index] = reduced_idx
        reduced_to_group.append(TieGroup(
            representative_index=root,
            member_indices=member_indices,
            signs=signs,
        ))
    return TieMap(
        kept_indices=np.asarray(kept_indices, dtype=np.int32),
        original_to_reduced=original_to_reduced,
        reduced_to_group=reduced_to_group,
    )

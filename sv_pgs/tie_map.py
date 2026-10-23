"""Construction of TieMaps: which variants share one coefficient, and with which sign."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from sv_pgs.data import TieMap


def _compact_identity_tie_map(variant_count: int) -> TieMap:
    identity_indices = np.arange(int(variant_count), dtype=np.int32)
    return TieMap(identity_indices, identity_indices.copy(), np.ones(identity_indices.shape[0], dtype=np.float32))


def tie_map_from_representatives(representative: NDArray[np.int64], sign: NDArray[np.float32]) -> TieMap:
    """The TieMap of variants that each name their group's representative (``representative[r] = r`` for a
    representative) and their sign to it (+1 for a copy, -1 for a negated copy); reduced order follows the
    representatives. Linear, in NumPy: a genome has one group per active variant."""
    representatives = np.asarray(representative, dtype=np.int64)
    is_representative = representatives == np.arange(representatives.shape[0])
    if np.any(representatives < 0) or not np.all(is_representative[representatives]):
        raise ValueError("every variant needs a representative that represents itself (its group's own member)")
    reduced_index = np.cumsum(is_representative, dtype=np.int64) - 1
    return TieMap(
        np.flatnonzero(is_representative).astype(np.int32),
        reduced_index[representatives].astype(np.int32),
        np.asarray(sign, dtype=np.float32),
    )


def tie_map_from_groups(variant_count: int, groups: Mapping[int, Sequence[tuple[int, float]]]) -> TieMap:
    """The TieMap of ``variant_count`` variants whose every variant sits in one group.

    ``groups`` maps each representative to its ``(member, sign)`` pairs (the representative
    included, sign +1 for a copy and -1 for a negated copy); reduced order follows the
    representatives.
    """
    representative = np.full(int(variant_count), -1, dtype=np.int64)
    sign = np.ones(int(variant_count), dtype=np.float32)
    for root, members in groups.items():
        for member, member_sign in members:
            representative[member] = root
            sign[member] = member_sign
    return tie_map_from_representatives(representative, sign)

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from sv_pgs._typing import F32Array, I32Array, NDArray


@dataclass(slots=True)
class TieGroup:
    """A group of variants with identical (or exactly negated) genotype columns.

    representative_index: the variant we keep in the reduced model
    member_indices: all variants in the group (including the representative)
    signs: +1 if a member's column matches the representative, -1 if negated
    """
    representative_index: int
    member_indices: I32Array
    signs: F32Array


@dataclass(frozen=True, slots=True)
class TieGroupTable:
    """Every reduced variant's ``TieGroup``, made on demand from a ``TieMap``'s arrays (``signs[j]`` is member j's).

    A genome's map has a group per active variant (about 20M) and exact ties among them, and a Python object per group
    held gigabytes and took minutes to build and to walk: the map holds arrays, and a group is an object only when one
    is asked for (members in ascending order).
    """

    kept_indices: I32Array
    original_to_reduced: I32Array
    signs: F32Array

    def __len__(self) -> int:
        return int(self.kept_indices.shape[0])

    def __getitem__(self, reduced_index: int) -> TieGroup:
        index = range(len(self))[reduced_index]
        return self._group(index, np.flatnonzero(self.original_to_reduced == index))

    def __iter__(self) -> Iterator[TieGroup]:
        order = np.argsort(self.original_to_reduced, kind="stable")
        bounds = np.searchsorted(self.original_to_reduced[order], np.arange(len(self) + 1))
        for index in range(len(self)):
            yield self._group(index, order[bounds[index] : bounds[index + 1]])

    def _group(self, index: int, members: NDArray) -> TieGroup:
        return TieGroup(
            representative_index=int(self.kept_indices[index]),
            member_indices=members.astype(np.int32),
            signs=self.signs[members],
        )


@dataclass(slots=True)
class TieMap:
    """Maps between the full variant set and the de-duplicated (reduced) set.

    Many variants can have identical genotype patterns across all samples
    (especially in LD or when multiple callers detect the same event).
    The tie map lets us fit the model on unique columns only, then expand
    the results back to all variants.

    kept_indices: which original variants were kept as representatives
    original_to_reduced: for each original variant, its reduced-space index
    reduced_to_group: every reduced variant's group of tied members, as the ``TieGroupTable`` of these arrays and each
        member's sign; a list of every group's ``TieGroup`` is taken into the table of the same groups
    """
    kept_indices: I32Array
    original_to_reduced: I32Array
    reduced_to_group: TieGroupTable

    def __post_init__(self) -> None:
        if not isinstance(self.reduced_to_group, TieGroupTable):
            signs = np.ones(self.original_to_reduced.shape[0], dtype=np.float32)
            for tie_group in self.reduced_to_group:
                signs[tie_group.member_indices] = tie_group.signs
            self.reduced_to_group = TieGroupTable(self.kept_indices, self.original_to_reduced, signs)

    @property
    def signs(self) -> F32Array:
        """Each member's sign: +1 for a copy of its group's representative, -1 for a negated copy."""
        return self.reduced_to_group.signs

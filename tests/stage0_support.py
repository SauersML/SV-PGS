"""Shared helpers for the Stage 0 tests: synthetic LD dosage codes and brute-force references."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from benchmarks.gpu_smoke_fixtures import mosaic_codes as mosaic_codes  # shared with the pytest-free GPU command
from sv_pgs.genotype_buffers import SampleLayout
from sv_pgs.ld_partition import fixed_point_pair_weights


def bubble_groups(rng: np.random.Generator, variants: int, maximum_run: int = 4) -> NDArray[np.int64]:
    """Unsplittable group ids: runs of 1..``maximum_run`` adjacent variants share an id."""
    runs = rng.integers(1, maximum_run + 1, size=variants)
    return np.repeat(np.arange(variants), runs)[:variants].astype(np.int64)


@dataclass
class InMemoryTileSource:
    codes: dict[str, NDArray[np.uint8]]
    groups: dict[str, NDArray[np.int64]]

    @property
    def sample_count(self) -> int:
        return next(iter(self.codes.values())).shape[1]

    def chromosomes(self) -> list[str]:
        return list(self.codes)

    def variant_count(self, chromosome: str) -> int:
        return self.codes[chromosome].shape[0]

    def unsplittable_groups(self, chromosome: str) -> NDArray[np.int64]:
        return self.groups[chromosome]

    def read_rows(self, chromosome: str, start: int, stop: int, out: NDArray[np.uint8]) -> None:
        out[...] = self.codes[chromosome][start:stop]

    def store_rows(self, chromosome: str) -> NDArray[np.int64]:
        offset = 0
        for name, codes in self.codes.items():
            if name == chromosome:
                return offset + np.arange(codes.shape[0], dtype=np.int64)
            offset += codes.shape[0]
        raise KeyError(chromosome)


def reference_cut_costs(codes: NDArray[np.uint8], layout: SampleLayout, block_cap: int) -> NDArray[np.int64]:
    """``C(k)`` from every profile pair at once, with exact int64 products."""
    profile_store_columns = np.concatenate(
        [
            layout.source_columns[layout.group_offsets[group] : layout.group_offsets[group] + layout.profile_counts[group]]
            for group in range(layout.group_count)
        ]
    )
    signed = codes[:, profile_store_columns].astype(np.int64) - 127
    band = signed @ signed.T
    sums = signed.sum(axis=1)
    squares = (signed * signed).sum(axis=1)
    variants = codes.shape[0]
    weights = fixed_point_pair_weights(band, sums, squares, sums, squares, layout.profile_count, 0, block_cap - 1)
    costs = np.zeros(variants + 1, dtype=np.int64)
    for cut in range(1, variants):
        costs[cut] = weights[:cut, cut:].sum()
    return np.maximum(costs, 0)


def reference_cuts(costs: NDArray[np.int64], allowed: NDArray[np.bool_], block_cap: int) -> list[int]:
    """Brute-force minimum-cost cuts with leftmost-predecessor ties (the partitioner's rule)."""
    positions = costs.shape[0] - 1
    best = [None] * (positions + 1)
    predecessor = [-1] * (positions + 1)
    best[0] = 0
    for cut in range(1, positions + 1):
        if not allowed[cut]:
            continue
        for previous in range(max(0, cut - block_cap), cut):
            if best[previous] is None:
                continue
            candidate = best[previous] + int(costs[cut])
            if best[cut] is None or candidate < best[cut]:
                best[cut] = candidate
                predecessor[cut] = previous
    cuts = []
    position = positions
    while position > 0:
        cuts.append(position)
        position = predecessor[position]
    return [0] + cuts[::-1]

"""LdBlockPartition wiring object for the LD-block / N-GPU rewrite (Phase 4).

Bundles the per-variant block-id array produced by
:func:`sv_pgs.ld_blocks.assign_ld_blocks` with the inverted
``{block_id: variant_indices}`` partition and a content-derived signature
suitable for hashing into a fit-stage cache key.

This module is deliberately tiny and pure-numpy so it can be imported from
both the model-building path (``sv_pgs.model``) and the matvec hot path
(``sv_pgs.mixture_inference``) without pulling JAX, cupy, or the rest of the
package.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from sv_pgs.ld_blocks import assign_ld_blocks, block_partition

__all__ = ["LdBlockPartition", "build_ld_block_partition"]


@dataclass(slots=True)
class LdBlockPartition:
    """Per-variant LD-block assignment + inverted partition.

    Attributes
    ----------
    block_ids
        Length ``n_variants`` int64 array; entry ``j`` is the block id assigned
        to variant ``j`` (see :func:`sv_pgs.ld_blocks.assign_ld_blocks`).
    partition
        ``{block_id: variant_indices (int64, sorted ascending)}`` mapping
        produced by :func:`sv_pgs.ld_blocks.block_partition`.
    population
        Ancestry label used to load the Berisa-Pickrell table.
    build
        Genome build label used to load the Berisa-Pickrell table.
    block_count
        Number of distinct blocks (including singleton blocks for unmapped
        variants).
    """

    block_ids: NDArray[np.int64]
    partition: Mapping[int, NDArray[np.int64]]
    population: str = "EUR"
    build: str = "hg38"
    block_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.block_ids = np.asarray(self.block_ids, dtype=np.int64)
        if self.block_ids.ndim != 1:
            raise ValueError("block_ids must be 1-D")
        self.block_count = int(len(self.partition))

    # ------------------------------------------------------------------
    # hashing helpers
    # ------------------------------------------------------------------

    def signature_sha256(self) -> str:
        """SHA-256 hex digest of (population, build, block_ids).

        Used as part of the fit-stage cache key so a partition change
        invalidates downstream caches automatically.
        """
        h = hashlib.sha256()
        h.update(self.population.encode("utf-8"))
        h.update(self.build.encode("utf-8"))
        h.update(np.ascontiguousarray(self.block_ids, dtype=np.int64).tobytes())
        return h.hexdigest()

    # ------------------------------------------------------------------
    # iteration
    # ------------------------------------------------------------------

    def iter_blocks(self):
        """Yield ``(block_id, variant_indices)`` in ascending block-id order."""
        for bid in sorted(self.partition.keys()):
            yield bid, self.partition[bid]


def _extract_chrom_pos(
    variant_records: Sequence[object],
) -> tuple[NDArray, NDArray]:
    chroms: list[str] = []
    positions: list[int] = []
    for rec in variant_records:
        chrom = getattr(rec, "chromosome", None)
        if chrom is None and isinstance(rec, dict):
            chrom = rec.get("chromosome") or rec.get("chrom")
        pos = getattr(rec, "position", None)
        if pos is None and isinstance(rec, dict):
            pos = rec.get("position") or rec.get("pos")
        if chrom is None or pos is None:
            raise ValueError(
                "variant_records must expose .chromosome and .position attributes"
            )
        chroms.append(str(chrom))
        positions.append(int(pos))
    return np.asarray(chroms), np.asarray(positions, dtype=np.int64)


def build_ld_block_partition(
    variant_records: Sequence[object],
    *,
    population: str = "EUR",
    build: str = "hg38",
) -> LdBlockPartition:
    """Build an :class:`LdBlockPartition` from a sequence of variant records.

    Each record must have ``chromosome`` and ``position`` attributes (or dict
    keys). Variants outside any block become singletons.
    """
    chroms, positions = _extract_chrom_pos(variant_records)
    block_ids = assign_ld_blocks(chroms, positions, build=build, ancestry=population)
    partition = block_partition(block_ids)
    return LdBlockPartition(
        block_ids=block_ids,
        partition=partition,
        population=population,
        build=build,
    )

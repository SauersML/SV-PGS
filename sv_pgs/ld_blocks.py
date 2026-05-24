"""LD-block partitioning for variant matrices.

Embeds the Berisa-Pickrell EUR LD blocks (1703 LDetect regions) as package
data and provides utilities to assign variants to blocks for downstream
block-wise variational EM matvecs.

The canonical TSV lives at ``sv_pgs/_data/EUR_hg38.tsv`` (3 columns,
1-based-inclusive style is *not* used: BED-style half-open ``[start, end)``
intervals, matching the upstream LDetect convention).

Public API
----------
- :func:`load_ld_blocks` -- return the embedded block table.
- :func:`assign_ld_blocks` -- map per-variant ``(chrom, pos)`` to block IDs.
- :func:`block_partition` -- invert block IDs into ``{block_id: indices}``.
- :func:`normalize_chromosome` -- canonical chrom-string-to-int map.
"""

from __future__ import annotations

from importlib import resources
from typing import Dict

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "load_ld_blocks",
    "assign_ld_blocks",
    "block_partition",
    "normalize_chromosome",
]


# Chromosome integer encoding used throughout sv_pgs.
_CHROM_MAP: Dict[str, int] = {
    **{str(i): i for i in range(1, 23)},
    "X": 23,
    "Y": 24,
    "MT": 25,
    "M": 25,
}
_MIN_CHROM = 1
_MAX_CHROM = 25


def normalize_chromosome(chrom: str) -> int:
    """Map a chromosome label to its canonical integer ID.

    Accepts the common variants ``"chr1"``, ``"1"``, ``"CHR1"``, ``"X"``,
    ``"chrX"``, ``"MT"``, ``"chrM"`` etc. Raises ``ValueError`` for
    anything unrecognised.
    """
    if chrom is None:
        raise ValueError("chromosome is None")
    s = str(chrom).strip().upper()
    if s.startswith("CHR"):
        s = s[3:]
    if s not in _CHROM_MAP:
        raise ValueError(f"unrecognised chromosome label: {chrom!r}")
    return _CHROM_MAP[s]


def _normalize_chromosome_array(chroms: NDArray) -> NDArray[np.int32]:
    out = np.empty(len(chroms), dtype=np.int32)
    for i, c in enumerate(chroms):
        out[i] = normalize_chromosome(c)
    return out


def _resource_path(build: str, ancestry: str) -> str:
    if ancestry.upper() != "EUR":
        raise NotImplementedError(
            f"ancestry={ancestry!r} not bundled; only 'EUR' is currently shipped"
        )
    if build.lower() not in ("hg38", "grch38"):
        raise NotImplementedError(
            f"build={build!r} not bundled; only 'hg38' is currently shipped"
        )
    return "EUR_hg38.tsv"


def _validate_blocks(blocks: NDArray[np.int64]) -> None:
    if blocks.ndim != 2 or blocks.shape[1] != 3:
        raise ValueError(f"blocks must have shape (N, 3); got {blocks.shape}")
    chroms = blocks[:, 0]
    starts = blocks[:, 1]
    ends = blocks[:, 2]
    if not np.all((chroms >= _MIN_CHROM) & (chroms <= _MAX_CHROM)):
        bad = np.unique(chroms[(chroms < _MIN_CHROM) | (chroms > _MAX_CHROM)])
        raise ValueError(f"block chrom IDs out of range [1,25]: {bad.tolist()}")
    if not np.all(ends > starts):
        raise ValueError("block end must be strictly greater than start")
    # Non-overlap check, per-chromosome.
    order = np.lexsort((starts, chroms))
    sc = chroms[order]
    ss = starts[order]
    se = ends[order]
    # within same chrom, next start must be >= previous end (half-open BED)
    same = sc[1:] == sc[:-1]
    overlap = same & (ss[1:] < se[:-1])
    if np.any(overlap):
        i = int(np.argmax(overlap))
        raise ValueError(
            "overlapping blocks detected at chrom="
            f"{int(sc[i])} pos=[{int(ss[i])},{int(se[i])}) and "
            f"[{int(ss[i + 1])},{int(se[i + 1])})"
        )


_BLOCKS_CACHE: Dict[tuple, NDArray[np.int64]] = {}


def load_ld_blocks(
    build: str = "hg38", ancestry: str = "EUR"
) -> NDArray[np.int64]:
    """Load the embedded LD-block table.

    Returns
    -------
    blocks : np.ndarray of shape ``(N_blocks, 3)``
        Columns are ``(chrom_int, start_pos, end_pos)`` with BED-style
        half-open ``[start, end)`` intervals. Sorted by ``(chrom, start)``.
    """
    key = (build.lower(), ancestry.upper())
    if key in _BLOCKS_CACHE:
        return _BLOCKS_CACHE[key]

    rel = _resource_path(build, ancestry)
    pkg = resources.files(__package__).joinpath("_data").joinpath(rel)
    with pkg.open("r", encoding="ascii") as fh:
        rows = []
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                # tolerate whitespace-separated rows
                parts = line.split()
            if len(parts) != 3:
                raise ValueError(
                    f"malformed block file at line {lineno}: {raw!r}"
                )
            chrom = normalize_chromosome(parts[0])
            start = int(parts[1])
            end = int(parts[2])
            rows.append((chrom, start, end))

    arr = np.asarray(rows, dtype=np.int64)
    # canonical sort by (chrom, start)
    order = np.lexsort((arr[:, 1], arr[:, 0]))
    arr = arr[order]
    _validate_blocks(arr)
    arr.setflags(write=False)
    _BLOCKS_CACHE[key] = arr
    return arr


def assign_ld_blocks(
    chromosomes: NDArray,
    positions: NDArray,
    build: str = "hg38",
    ancestry: str = "EUR",
) -> NDArray[np.int64]:
    """Assign each variant to an LD-block ID.

    Parameters
    ----------
    chromosomes : array-like of str
        Per-variant chromosome labels. Accepts mixed ``"chr1"`` / ``"1"``.
    positions : array-like of int
        Per-variant 1-based positions.
    build, ancestry : optional
        Forwarded to :func:`load_ld_blocks`.

    Returns
    -------
    block_ids : np.ndarray of shape ``(n_variants,)`` and dtype int64
        IDs in ``[0, N_blocks)`` for variants falling in a known LD block;
        variants outside any block are assigned a unique singleton block ID
        ``>= N_blocks`` (so each unmapped variant ends up alone).
    """
    chroms_in = np.asarray(chromosomes)
    pos_in = np.asarray(positions, dtype=np.int64)
    if chroms_in.shape != pos_in.shape:
        raise ValueError(
            f"chromosomes shape {chroms_in.shape} != positions shape {pos_in.shape}"
        )
    if chroms_in.ndim != 1:
        raise ValueError("chromosomes/positions must be 1-D")

    blocks = load_ld_blocks(build=build, ancestry=ancestry)
    n_blocks = blocks.shape[0]
    chrom_int = _normalize_chromosome_array(chroms_in)

    block_ids = np.full(chroms_in.shape[0], -1, dtype=np.int64)

    # Group by chromosome for efficient bisect.
    unique_chroms = np.unique(blocks[:, 0])
    chrom_to_slice: Dict[int, tuple] = {}
    for c in unique_chroms:
        mask = blocks[:, 0] == c
        idx = np.flatnonzero(mask)
        chrom_to_slice[int(c)] = (int(idx[0]), int(idx[-1]) + 1)

    for c in np.unique(chrom_int):
        c_int = int(c)
        if c_int not in chrom_to_slice:
            continue
        lo, hi = chrom_to_slice[c_int]
        starts = blocks[lo:hi, 1]
        ends = blocks[lo:hi, 2]
        var_mask = chrom_int == c_int
        var_pos = pos_in[var_mask]
        # Find candidate block: rightmost block with start <= pos.
        idx = np.searchsorted(starts, var_pos, side="right") - 1
        valid = idx >= 0
        # In-range check: pos < end of the candidate block (half-open).
        within = np.zeros_like(valid)
        if np.any(valid):
            cand_ends = ends[np.where(valid, idx, 0)]
            within = valid & (var_pos < cand_ends)
        assigned = np.where(within, lo + idx, -1)
        block_ids[var_mask] = assigned

    # Singleton IDs for unmapped variants: unique ID per variant >= n_blocks.
    unmapped = block_ids < 0
    if np.any(unmapped):
        # deterministic singleton IDs in variant order
        unmapped_idx = np.flatnonzero(unmapped)
        block_ids[unmapped_idx] = n_blocks + np.arange(
            unmapped_idx.shape[0], dtype=np.int64
        )

    return block_ids


def block_partition(block_ids: NDArray) -> Dict[int, NDArray[np.int64]]:
    """Invert per-variant block IDs into ``{block_id: variant_indices}``.

    The returned arrays are sorted by variant index (which, for input
    arrays already sorted by genomic position within each block, also
    yields position-sorted indices).
    """
    ids = np.asarray(block_ids, dtype=np.int64)
    if ids.ndim != 1:
        raise ValueError("block_ids must be 1-D")
    # Stable sort by id, then group via boundaries.
    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    if sorted_ids.size == 0:
        return {}
    # Locate boundaries where the id changes.
    change = np.concatenate(([True], sorted_ids[1:] != sorted_ids[:-1]))
    starts = np.flatnonzero(change)
    ends = np.concatenate((starts[1:], [sorted_ids.size]))
    out: Dict[int, NDArray[np.int64]] = {}
    for s, e in zip(starts, ends):
        bid = int(sorted_ids[s])
        members = np.sort(order[s:e])
        out[bid] = members.astype(np.int64, copy=False)
    return out

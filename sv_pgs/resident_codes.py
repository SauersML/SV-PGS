"""Stage 2's reduced genotype codes held on the compute device for a whole fit, read by deep GEMMs.

A streamed read (``store_block_source``) moves every block's store span to the device once per read
and multiplies it block by block: an int8 GEMM of depth p_b (at most the LD block cap) for X_b R, and
one operand split per block. Both costs are removable when the reduced codes fit on the device:

* **Residency.** ``ResidentCodeSource`` gathers every block's aligned codes once into one variant-major
  int8 matrix (block b in rows g_b .. g_b + aligned(p_b), its alignment rows zero), so no read touches
  the store again. When they fit as well, the same codes are held sample-major, so the products that
  reduce over variants (X R) read them directly instead of transposing every sample chunk per read.
* **Depth.** S V's read is X D X' L, a sum over the variants that any partition of them gives:
  X D X' L = sum_t X_t D_t X_t' L. ``map_reduce`` therefore runs a read over *read tiles*, as few
  consecutive row ranges of the resident matrix as the tile workspace and int32 exactness allow
  (``read_tile_rows``), not over LD blocks: X' L is one GEMM of depth n per read tile, X R one GEMM of
  depth up to ``INT32_EXACT_DIGIT_ROWS`` with one operand split per read tile.

The products are ``code_products.CodeBlockTile``'s, so their accuracy contracts are unchanged: over a read
tile, ``accumulate_matmat``'s split moves column k of R by at most relative_error ||R_k||_2, and a split per
block met the same bound on each block, hence on R. The resident tiles measure that split's digit count on
each call's R (``CodeBlockTile.accumulate_digits``): a read waits on the device once per read tile, and a deep
tile costs no more digits than its data need, where the a-priori bound grows with sqrt(depth). X' L's integer
products are exact either way.
``blocks()`` still yields the LD blocks, as zero-copy tiles of the resident codes, for every per-block
consumer (column squares, block Grams).

The model's variants must be the blocks' variants in block order (``dual_solve.StreamedDualSource``'s
condition). CuPy is passed in through the tiles' array module; this module never imports it.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator, Sequence

import numpy as np

from sv_pgs.code_products import (
    _FLOAT64_BYTES,
    _INT32_BYTES,
    INT32_EXACT_DIGIT_ROWS,
    INT8_GEMM_ALIGNMENT,
    OPERAND_DIGITS,
    CodeBlockTile,
    _transpose_codes,
)


def _aligned(count: int) -> int:
    return -(-count // INT8_GEMM_ALIGNMENT) * INT8_GEMM_ALIGNMENT


def read_tile_rows(sample_count: int, columns: int, workspace_bytes: int) -> int:
    """The most resident rows one read tile may span for a read of ``columns`` columns: a multiple of
    ``INT8_GEMM_ALIGNMENT``, at most the int32-exact GEMM depth, and small enough that the largest of the
    tile's product workspaces fits ``workspace_bytes``.

    The largest is exact ``rmatmat``'s (``CodeBlockTile._codes_times``): the fp64 operand [n, K], and per row
    the fp64 total and two recombination terms and ``OPERAND_DIGITS`` int32 products per column. The others
    hold less per row (``matmat``: the operand, its digits and the [n, K] output; ``accumulate_matmat``: the
    operand and its digits), and their sample chunks are sized from what remains.
    """
    padded_samples = _aligned(sample_count)
    per_row = columns * (3 * _FLOAT64_BYTES + OPERAND_DIGITS * _INT32_BYTES)
    # matmat's fixed buffers also hold the [n, K] output, and each product needs one aligned sample chunk
    sample_side = 2 * _FLOAT64_BYTES * padded_samples * columns
    chunk_reserve = INT8_GEMM_ALIGNMENT * (INT32_EXACT_DIGIT_ROWS + (OPERAND_DIGITS * _INT32_BYTES + 2 * _FLOAT64_BYTES) * columns)
    fitting = (int(workspace_bytes) - sample_side - chunk_reserve) // per_row
    exact = INT32_EXACT_DIGIT_ROWS // INT8_GEMM_ALIGNMENT * INT8_GEMM_ALIGNMENT
    rows = min(fitting, exact) // INT8_GEMM_ALIGNMENT * INT8_GEMM_ALIGNMENT
    if rows < INT8_GEMM_ALIGNMENT:
        raise MemoryError(f"a tile workspace of {workspace_bytes} bytes cannot hold a {columns}-column read of {sample_count} samples")
    return rows


class ResidentCodeSource:
    """A block source's reduced codes, gathered once onto its device: a ``dual_solve.DualTileSource``.

    ``source`` yields ``(block_index, CodeBlockTile)`` from ``iter_tiles()`` in block order (a
    ``store_block_source.StoreGenotypeBlockSource``) and has ``block_variant_indices``, ``sample_count`` and
    ``array_module``; it is read once. ``device_bytes`` is what the caller's memory plan gives the resident
    codes: they need ``variant_major_bytes`` of it, and are also held sample-major when twice that fits.
    ``workspace_bytes`` is each tile's product workspace.
    """

    def __init__(self, source: Any, device_bytes: int, workspace_bytes: int) -> None:
        self.array_module = source.array_module
        self.sample_count = int(source.sample_count)
        indices = [np.asarray(block, dtype=np.int64) for block in source.block_variant_indices]
        if any(block.shape[0] == 0 or not np.array_equal(block, np.arange(block[0], block[-1] + 1)) for block in indices):
            raise ValueError("every block must be a nonempty contiguous run of variants.")
        self.block_bounds = [(int(block[0]), int(block[-1]) + 1) for block in indices]
        if self.block_bounds[0][0] != 0 or any(previous[1] != following[0] for previous, following in zip(self.block_bounds, self.block_bounds[1:])):
            raise ValueError("blocks must cover the variants once, in order.")
        self.variant_count = self.block_bounds[-1][1]
        sizes = np.array([stop - start for start, stop in self.block_bounds], dtype=np.int64)
        self._offsets = np.concatenate([[0], np.cumsum([_aligned(int(size)) for size in sizes])]).astype(np.int64)
        self.resident_rows = int(self._offsets[-1])
        self._padded_samples = _aligned(self.sample_count)
        self._workspace_bytes = int(workspace_bytes)
        if self.variant_major_bytes > device_bytes:
            raise MemoryError(f"the resident codes need {self.variant_major_bytes} bytes; the plan gives {device_bytes}")
        xp = self.array_module
        codes = xp.zeros((self.resident_rows, self._padded_samples), dtype=xp.int8)
        # Alignment rows carry no variant: mean 0 and unit scale, so their products are zero (their codes are).
        means = xp.zeros(self.resident_rows, dtype=xp.float64)
        scales = xp.ones(self.resident_rows, dtype=xp.float64)
        seen = 0
        for block_index, tile in source.iter_tiles():
            start, stop = int(self._offsets[block_index]), int(self._offsets[block_index + 1])
            size = int(sizes[block_index])
            codes[start:stop] = tile.aligned_codes
            means[start : start + size] = tile.means
            scales[start : start + size] = tile.scales
            seen += 1
        if seen != len(self.block_bounds):
            raise ValueError("the source yielded a different number of blocks than it declares")
        self._codes, self._means, self._scales = codes, means, scales
        self._real = np.zeros(self.resident_rows, dtype=bool)
        for block_index, size in enumerate(sizes):
            self._real[int(self._offsets[block_index]) : int(self._offsets[block_index]) + int(size)] = True
        # the resident row of every model variant, for laying a read's variant-side arrays out on the resident rows
        rows = np.flatnonzero(self._real).astype(np.int64)
        self._variant_rows = xp.asarray(rows)
        self._sample_major = None
        if xp is not np and 2 * self.variant_major_bytes <= device_bytes:
            self._sample_major = _transpose_codes(xp, codes, 0, self._padded_samples)
        self.resident_bytes = int(codes.nbytes) + int(means.nbytes) + int(scales.nbytes) + int(self._variant_rows.nbytes) + (
            0 if self._sample_major is None else int(self._sample_major.nbytes)
        )
        self._read_tiles: dict[int, list[tuple[int, int, CodeBlockTile]]] = {}

    @property
    def variant_major_bytes(self) -> int:
        return self.resident_rows * self._padded_samples

    @property
    def sample_major(self) -> bool:
        """Whether the codes are also held sample-major."""
        return self._sample_major is not None

    def _tile(self, start: int, stop: int, variant_count: int) -> CodeBlockTile:
        major = None if self._sample_major is None else (self._sample_major, start)
        return CodeBlockTile.from_aligned(
            self._codes[start:stop], variant_count, self.sample_count, self._means[start : start + variant_count],
            self._scales[start : start + variant_count], None, self.array_module, self._workspace_bytes, major,
        )

    def blocks(self) -> Iterator[tuple[int, int, CodeBlockTile]]:
        """(start, stop, tile) for every LD block, in variant order; the tiles view the resident codes."""
        for block_index, (start, stop) in enumerate(self.block_bounds):
            yield start, stop, self._tile(int(self._offsets[block_index]), int(self._offsets[block_index + 1]), stop - start)

    def read_tiles(self, columns: int) -> list[tuple[int, int, CodeBlockTile]]:
        """(first, last + 1, tile) of resident rows for a read of ``columns`` columns: consecutive ranges of at most
        ``read_tile_rows`` rows covering every resident row, each a tile whose alignment rows are zero."""
        if columns not in self._read_tiles:
            limit = read_tile_rows(self.sample_count, columns, self._workspace_bytes)
            tiles = []
            for start in range(0, self.resident_rows, limit):
                stop = min(start + limit, self.resident_rows)
                tiles.append((start, stop, self._tile(start, stop, stop - start)))
            self._read_tiles[columns] = tiles
        return self._read_tiles[columns]

    def map_reduce(self, work: Callable[..., None], shared: dict, rows: dict, image_shape: tuple[int, ...]) -> Any:
        """sum over read tiles of what work(first, last, tile, shared, rows_t, image) adds into a zero image.

        The read tiles partition the variants differently from the LD blocks, so ``work`` must add what is
        additive over any partition of them, as S V's read (``dual_solve.apply_operator``) is. ``rows``' arrays
        are laid out on the resident rows first (an alignment row gets zeros), and each tile gets its rows.
        """
        xp = self.array_module
        laid_out = {}
        for name, values in rows.items():
            resident = xp.zeros((self.resident_rows,) + tuple(values.shape[1:]), dtype=values.dtype)
            resident[self._variant_rows] = values
            laid_out[name] = resident
        image = xp.zeros(image_shape)
        device_shared = dict(shared)
        for start, stop, tile in self.read_tiles(int(image_shape[1])):
            work(start, stop, tile, device_shared, {name: values[start:stop] for name, values in laid_out.items()}, image)
        return image


def resident_bytes_needed(block_sizes: Sequence[int], sample_count: int) -> int:
    """Bytes the variant-major resident codes of blocks of these sizes take (twice that holds them sample-major too)."""
    return sum(_aligned(int(size)) for size in block_sizes) * _aligned(int(sample_count))

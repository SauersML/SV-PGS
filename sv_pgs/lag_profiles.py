"""Each LD block's lag profile: the excess r^2 of its pairs (r^2 less its null expectation 1/n) summed by how far apart
they are, which sets the block's LD extent (``marginal_variances.extent_from_profile``) for the summary band
(``gram_space.summary_band``) and the EP windows (``marginal_variances.ld_extent``). Stage 0 sums it as it projects each
block (``genotype_statistics._project_block``), and ``marginal_variances.ld_lag_profiles`` sums it from a stored Gram
by the same steps, to the bit.

A module of its own: Stage 0's restart unit is keyed by the source of Stage 0's modules and every module they import
(``restart_store.module_sources``), so this arithmetic keys it, and the window algebra and certificates beside
``ld_lag_profiles`` in ``marginal_variances`` do not."""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from sv_pgs.genotype_buffers import _HOST_PANEL

LAG_PROFILE_ROWS = _HOST_PANEL
"""The most rows of a block summed in one step of its lag profile (``lag_excess_rows``): Stage 0's projection panels
(``genotype_buffers._HOST_PANEL``), so the profile Stage 0 sums for each block as it projects it
(``genotype_statistics._project_block``) is the one ``ld_lag_profiles`` sums from the stored Gram, to the bit, wherever
the working set holds a whole panel's arrays."""


def lag_excess_rows(stored: Any, diagonal: Any, first: int, last: int, sample_count: int, array_module: Any = np) -> Any:
    """Rows ``first`` to ``last`` of one block's stored float32 Gram (``stored``, on ``array_module``) as their part of
    the block's lag profile: the excess r^2 of their pairs, r^2 less its null expectation 1/n, summed by lag (one entry
    per lag, the block's width of them). ``diagonal`` is the Gram's diagonal in float64. In place, with no masks, in
    two arrays of the rows' shape: their excess r^2 in float64, and the diagonals' products and then, over the same
    bytes, the pairs' lags."""
    xp = array_module
    width = int(stored.shape[1])
    squared = xp.asarray(stored[first:last], dtype=xp.float64)
    squared *= squared
    products = xp.empty(squared.shape, dtype=xp.float64)
    xp.multiply(diagonal[first:last, None], diagonal[None, :], out=products)
    squared /= products
    squared -= 1.0 / sample_count
    lags = products.view(xp.int64)
    xp.subtract(xp.arange(width)[None, :], xp.arange(first, last)[:, None], out=lags)
    xp.abs(lags, out=lags)
    return xp.bincount(lags.ravel(), weights=squared.ravel(), minlength=width)[:width]


def lag_profile(parts: Any, width: int, array_module: Any = np) -> NDArray[np.float64]:
    """One block's lag excess from its row chunks' parts (``lag_excess_rows``), summed in row order, with lag 0, which
    is no pair (its terms are the diagonal's own r^2 = 1), at zero; on the host."""
    excess = array_module.zeros(width)
    for part in parts:
        excess += part
    excess = excess.get() if hasattr(excess, "get") else np.asarray(excess)
    if width:
        excess[0] = 0.0
    return excess


def lag_pairs(width: int, length: int) -> NDArray[np.float64]:
    """A block of ``width`` columns' ordered pairs by lag, ``length`` lags of them: 2 (w - d) at lag d, none at lag 0."""
    pairs = np.maximum(2.0 * (width - np.arange(length)), 0.0)
    pairs[:1] = 0.0
    return pairs

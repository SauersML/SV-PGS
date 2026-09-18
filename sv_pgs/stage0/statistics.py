"""Exact integer sufficient statistics of one LD block, and their fp64 standardization.

Stage 0 works on the signed code ``s = code - 127`` (``code`` in [0, 254] is
``round(DS * 127)``), so every genotype cross-product is an integer. Per block
and per sample group it keeps

    gram[g]  = sum_{i in g} s_i s_i^T      (exact int32 / int64)
    sums[g]  = sum_{i in g} s_i            (exact int64)
    counts[g] = |g|

These add across groups, so the statistics of any union of groups (for example
every fold but one) are exact sums. Standardization happens once, in fp64,
from the integers: with ``n``, ``S`` and ``u`` for the chosen groups,

    N = n S - u u^T                         (exact int64; |N| <= 127^2 n^2, so its fp64
                                             conversion is exact below n = 747,000)
    X^T X / n = N / sqrt(diag N diag N^T) = R

where ``X`` is the column-standardized dosage (population standard deviation,
the model's convention), so ``X^T X = n R``. A shift or scale of the dosage
(``DS = (s + 127) / 127``) cancels in ``R``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

SIGNED_CODE_OFFSET = 127
"""``s = code - SIGNED_CODE_OFFSET`` lies in [-127, 127] for every stored code."""

MAXIMUM_STORED_CODE = 254
"""Codes 0..254 are dosages; 255 (missing) never reaches Stage 0."""

INT32_EXACT_ROWS = (2**31 - 1) // (SIGNED_CODE_OFFSET * SIGNED_CODE_OFFSET)
"""Largest row count whose int32 accumulation of ``s_i s_j`` cannot overflow (133,144)."""

FLOAT32_EXACT_ROWS = (2**24) // (SIGNED_CODE_OFFSET * SIGNED_CODE_OFFSET)
"""Largest row count whose float32 accumulation of ``s_i s_j`` stays an exact integer (1,040):
every partial sum, in any order, has magnitude below 2^24."""


@dataclass(frozen=True, slots=True)
class BlockStatistics:
    """Exact Stage 0 statistics of the contiguous variant range ``[start, stop)`` of one chromosome.

    ``start``/``stop`` index the chromosome's variants as the tile source streams them.
    ``grams`` is ``(groups, width, width)`` int32 when every group has at most
    ``INT32_EXACT_ROWS`` samples and int64 otherwise; ``sums`` is ``(groups, width)``
    int64; ``cross_products`` is ``(groups, width, columns)`` float64 holding
    ``sum_{i in g} s_i y_i^T`` for the pass's cross-product columns, or ``None``.
    """

    chromosome: str
    start: int
    stop: int
    group_counts: NDArray[np.int64]
    sums: NDArray[np.int64]
    grams: NDArray[np.integer]
    cross_products: NDArray[np.float64] | None

    @property
    def width(self) -> int:
        return self.stop - self.start


def pooled_integer_statistics(
    block: BlockStatistics, groups: Sequence[int]
) -> tuple[int, NDArray[np.int64], NDArray[np.int64]]:
    """Return ``(n, S, u)`` summed over ``groups``: exact int64."""
    selected = np.asarray(groups, dtype=np.int64)
    count = int(block.group_counts[selected].sum())
    gram = block.grams[selected].astype(np.int64).sum(axis=0)
    sums = block.sums[selected].sum(axis=0)
    return count, gram, sums


def centered_numerator(count: int, gram: NDArray[np.int64], sums: NDArray[np.int64]) -> NDArray[np.int64]:
    """``N = n S - u u^T`` in exact int64 (``n^2 * 127^2 < 2^63`` for ``n`` below 2.4e7)."""
    return count * gram - np.outer(sums, sums)


def block_correlation(
    block: BlockStatistics, groups: Sequence[int]
) -> tuple[int, NDArray[np.float64], NDArray[np.bool_]]:
    """Return ``(n, R, constant)`` for the samples in ``groups``.

    ``R`` is the fp64 correlation matrix, computed from exact integers, so ``X^T X = n R``
    for population-sd standardized columns. ``constant`` flags zero-variance columns,
    whose rows and columns of ``R`` are zero (their diagonal included), since the
    model drops them.
    """
    count, gram, sums = pooled_integer_statistics(block, groups)
    numerator = centered_numerator(count, gram, sums)
    diagonal = np.diagonal(numerator).copy()
    constant = diagonal == 0
    inverse_root = np.zeros(diagonal.shape[0], dtype=np.float64)
    inverse_root[~constant] = 1.0 / np.sqrt(diagonal[~constant].astype(np.float64))
    correlation = numerator.astype(np.float64) * inverse_root[:, None] * inverse_root[None, :]
    return count, correlation, constant


def column_moments(
    block: BlockStatistics, groups: Sequence[int]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Mean and population standard deviation of the dosage ``DS = code / 127`` over ``groups``."""
    count, gram, sums = pooled_integer_statistics(block, groups)
    numerator_diagonal = count * np.diagonal(gram) - sums * sums
    mean_signed = sums.astype(np.float64) / count
    mean = (mean_signed + SIGNED_CODE_OFFSET) / SIGNED_CODE_OFFSET
    standard_deviation = np.sqrt(numerator_diagonal.astype(np.float64)) / (count * SIGNED_CODE_OFFSET)
    return mean, standard_deviation


def centered_cross_products(
    block: BlockStatistics,
    groups: Sequence[int],
    column_sums: NDArray[np.float64],
) -> NDArray[np.float64]:
    """``X^T (Y - 1 ybar^T)`` for population-sd standardized ``X`` over ``groups``.

    ``column_sums[g]`` is ``sum_{i in g} y_i`` for each cross-product column; the result
    is ``(width, columns)``.
    """
    if block.cross_products is None:
        raise ValueError("block was computed without cross-product columns")
    selected = np.asarray(groups, dtype=np.int64)
    count, gram, sums = pooled_integer_statistics(block, selected)
    raw = block.cross_products[selected].sum(axis=0)
    y_sums = np.asarray(column_sums, dtype=np.float64)[selected].sum(axis=0)
    centered = raw - np.outer(sums.astype(np.float64), y_sums) / count
    numerator_diagonal = (count * np.diagonal(gram) - sums * sums).astype(np.float64)
    scale = np.zeros(numerator_diagonal.shape[0], dtype=np.float64)
    varying = numerator_diagonal > 0
    scale[varying] = count / np.sqrt(numerator_diagonal[varying])
    return centered * scale[:, None]

"""Structural-variant context features for the prior, beyond the store's sites fields.

SPEC: every variant's prior variance, SNVs and indels included, depends on its
structural-variant context. The store converter writes the sites-level fields
(``store_converter.sv_context``: distance to the nearest common SV, SV-bearing
bubbles nearby; ``store_converter.tr_loci``: repeat loci). This module adds the
pieces those fields need or cannot supply; the prior design standardizes them
and empirical Bayes learns their weights, so an uninformative feature costs
nothing.

- ``locus_common_flags``: which SV records are common. Commonness belongs to the
  locus, not to one record: the panel splits a common locus into many rare
  allele records, so a record is common when its bubble's non-reference
  haplotype frequency is. With alleles that exclude one another on a haplotype
  that frequency is the sum of the bubble's record frequencies (capped at 1).
- ``ewens_theta``: a tandem-repeat locus's scaled mutation rate from its number
  of distinct alleles. SNV tags carry less of a fast-mutating repeat's signal.
- ``block_tagging``: from one Stage 0 correlation block, each column's largest
  and summed squared correlation with the block's SV columns, and for each SV
  column rho^2, its R^2 on the block's non-SV columns (the part of the SV that
  the SNVs already carry; an imputed SV column adds r^2 (1 - rho^2) beyond them).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import digamma

from sv_pgs._typing import BoolArray, F64Array, NDArray


def locus_common_flags(
    bubble_indices: NDArray,
    is_sv: NDArray,
    alternate_frequency: NDArray,
    common_frequency: float,
) -> BoolArray:
    """SV records whose bubble's non-reference haplotype frequency reaches ``common_frequency``."""
    bubbles = np.asarray(bubble_indices, dtype=np.int64)
    sv = np.asarray(is_sv, dtype=bool)
    frequency = np.asarray(alternate_frequency, dtype=np.float64)
    if not (bubbles.shape == sv.shape == frequency.shape):
        raise ValueError("locus_common_flags needs one bubble, SV flag and frequency per record.")
    sv_bubbles, slot = np.unique(bubbles[sv], return_inverse=True)
    bubble_frequency = np.minimum(np.bincount(slot, weights=frequency[sv], minlength=sv_bubbles.shape[0]), 1.0)
    flags = np.zeros(bubbles.shape[0], dtype=bool)
    flags[sv] = bubble_frequency[slot] >= common_frequency
    return flags


def ewens_theta(allele_count: NDArray, haplotype_count: int) -> F64Array:
    """Ewens estimate of the scaled mutation rate from K distinct alleles among n haplotypes.

    Solves E[K | theta] = 1 + theta (psi(theta + n) - psi(theta + 1)) = K, the
    Ewens sampling formula's expected allele count, which rises monotonically
    from 1 at theta = 0 to n. K = 1 gives 0.
    """
    alleles = np.asarray(allele_count, dtype=np.float64)
    if np.any(alleles < 1) or np.any(alleles >= haplotype_count):
        raise ValueError("allele counts must lie in [1, haplotype_count).")
    log_low = np.full(alleles.shape, -40.0)
    log_high = np.full(alleles.shape, np.log(float(haplotype_count)) + 40.0)
    for _ in range(100):
        log_middle = 0.5 * (log_low + log_high)
        theta = np.exp(log_middle)
        expected = 1.0 + theta * (digamma(theta + haplotype_count) - digamma(theta + 1.0))
        too_many = expected > alleles
        log_high = np.where(too_many, log_middle, log_high)
        log_low = np.where(too_many, log_low, log_middle)
    return np.where(alleles <= 1, 0.0, np.exp(0.5 * (log_low + log_high)))


@dataclass(frozen=True)
class BlockTagging:
    """Per column of one block: LD with the block's SV columns."""

    largest_squared_correlation: F64Array
    summed_squared_correlation: F64Array
    structural_predictability: F64Array


def block_tagging(correlation: NDArray, is_structural: NDArray, relative_tolerance: float = 1e-10) -> BlockTagging:
    """LD-level SV context from one correlation block of standardized columns.

    For every column, the largest and summed squared correlation with the
    block's SV columns other than itself. For every SV column, rho^2, its R^2
    on the block's non-SV columns (NaN for non-SV columns): r' R^+ r with R the
    non-SV correlation block, exact under collinearity because both come from
    one Gram. Eigenvalues below ``relative_tolerance`` times the largest are
    treated as zero.
    """
    correlation = np.asarray(correlation, dtype=np.float64)
    structural_flags = np.asarray(is_structural, dtype=bool)
    if correlation.shape != (structural_flags.shape[0], structural_flags.shape[0]):
        raise ValueError("block_tagging needs a square correlation block and one SV flag per column.")
    structural = np.flatnonzero(structural_flags)
    other = np.flatnonzero(~structural_flags)
    width = structural_flags.shape[0]
    squared = np.square(correlation[:, structural])
    squared[structural, np.arange(structural.shape[0])] = 0.0
    largest = squared.max(axis=1, initial=0.0)
    summed = squared.sum(axis=1)
    predictability = np.full(width, np.nan)
    predictability[structural] = 0.0
    if structural.shape[0] and other.shape[0]:
        eigenvalues, eigenvectors = np.linalg.eigh(correlation[np.ix_(other, other)])
        kept = eigenvalues > relative_tolerance * max(float(eigenvalues[-1]), 0.0)
        projected = eigenvectors[:, kept].T @ correlation[np.ix_(other, structural)]
        predictability[structural] = np.clip(np.sum(np.square(projected) / eigenvalues[kept][:, None], axis=0), 0.0, 1.0)
    return BlockTagging(
        largest_squared_correlation=largest,
        summed_squared_correlation=summed,
        structural_predictability=predictability,
    )

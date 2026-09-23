"""Structural-variant context features for the prior, beyond the store's sites fields.

SPEC: every variant's prior variance, SNVs and indels included, depends on its
structural-variant context. The store converter writes the sites-level SV
context and repeat-locus fields; this module adds the pieces those fields need
or cannot supply. The prior design standardizes them and empirical Bayes learns
their weights, so an uninformative feature costs nothing.

- ``locus_frequency``: each SV record's locus frequency. Frequency belongs to the
  locus, not to one record: the panel splits a common locus into many rare
  allele records, so a record carries its bubble's non-reference haplotype
  frequency. With alleles that exclude one another on a haplotype that
  frequency is the sum of the bubble's record frequencies. A sum above 1 proves
  the records co-occur on haplotypes (they are not exclusive alleles), and the
  sum is then not the locus frequency: that bubble's is NaN (unidentified from
  frequencies alone), never a sum capped at 1. It is a continuous annotation;
  nothing thresholds it into common and rare.
- ``ewens_theta``: a tandem-repeat locus's scaled mutation rate from its number
  of distinct alleles, under the infinite-alleles model (every mutation a new
  allele). A stepwise repeat mutation can return to an existing length, so under
  that model the estimate is a lower bound on the rate; it is a measured feature
  whose weight empirical Bayes learns, not a calibrated rate. SNV tags carry
  less of a fast-mutating repeat's signal.
- ``block_tagging``: from one Stage 0 correlation block, each column's largest
  and summed squared correlation with the block's SV columns, and for each SV
  column rho^2, its R^2 on the block's non-SV columns (the part of the SV that
  the SNVs already carry; an imputed SV column adds r^2 (1 - rho^2) beyond them),
  as the adjusted estimate of the population R^2 from the block's sample count
  (in-sample R^2 on k columns from n people is 1 by interpolation once k reaches
  n - 1, whatever the population's).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import digamma

from sv_pgs._typing import F64Array, NDArray


def locus_frequency(bubble_indices: NDArray, is_sv: NDArray, alternate_frequency: NDArray) -> F64Array:
    """Each SV record's bubble non-reference haplotype frequency; NaN for non-SV records and for a bubble whose record
    frequencies sum above 1 (co-occurring records, whose locus frequency frequencies alone do not identify)."""
    bubbles = np.asarray(bubble_indices, dtype=np.int64)
    sv = np.asarray(is_sv, dtype=bool)
    frequency = np.asarray(alternate_frequency, dtype=np.float64)
    if not (bubbles.shape == sv.shape == frequency.shape):
        raise ValueError("locus_frequency needs one bubble, SV flag and frequency per record.")
    sv_bubbles, slot = np.unique(bubbles[sv], return_inverse=True)
    total = np.bincount(slot, weights=frequency[sv], minlength=sv_bubbles.shape[0])
    bubble_frequency = np.where(total <= 1.0, total, np.nan)
    locus = np.full(bubbles.shape[0], np.nan)
    locus[sv] = bubble_frequency[slot]
    return locus


def ewens_theta(allele_count: NDArray, haplotype_count: int) -> F64Array:
    """Ewens estimate of the scaled mutation rate from K distinct alleles among n haplotypes.

    Solves E[K | theta] = 1 + theta (psi(theta + n) - psi(theta + 1)) = K, the
    Ewens sampling formula's expected allele count, which rises monotonically
    from 1 at theta = 0 to n. Each term theta / (theta + i), i = 1 .. n - 1, lies
    between theta / (theta + n - 1) and theta / i, so the root lies in
    [(K - 1) / H_(n-1), (K - 1)(n - 1) / (n - K)]; bisection runs until the
    bracket holds adjacent floats. K = 1 gives 0.
    """
    alleles = np.asarray(allele_count, dtype=np.float64)
    if np.any(alleles < 1) or np.any(alleles >= haplotype_count):
        raise ValueError("allele counts must lie in [1, haplotype_count).")
    harmonic = float(digamma(float(haplotype_count)) - digamma(1.0))
    low = (alleles - 1.0) / harmonic
    high = (alleles - 1.0) * (haplotype_count - 1.0) / (haplotype_count - alleles)
    while True:
        middle = 0.5 * (low + high)
        unresolved = (middle > low) & (middle < high)
        if not np.any(unresolved):
            return middle
        expected = 1.0 + middle * (digamma(middle + haplotype_count) - digamma(middle + 1.0))
        too_many = expected > alleles
        high = np.where(unresolved & too_many, middle, high)
        low = np.where(unresolved & ~too_many, middle, low)


@dataclass(frozen=True)
class BlockTagging:
    """Per column of one block: LD with the block's SV columns."""

    largest_squared_correlation: F64Array
    summed_squared_correlation: F64Array
    structural_predictability: F64Array


def block_tagging(correlation: NDArray, is_structural: NDArray, sample_count: int) -> BlockTagging:
    """LD-level SV context from one correlation block of standardized columns.

    For every column, the largest and summed squared correlation with the
    block's SV columns other than itself. For every SV column, rho^2, its R^2
    on the block's non-SV columns (NaN for non-SV columns): r' R^+ r with R the
    non-SV correlation block, exact under collinearity because both come from
    one Gram. Eigenvalues at or below the numerical-rank threshold (the largest
    times the block width times float64 epsilon) are treated as zero.

    ``correlation`` is the block's sample correlation over ``sample_count`` people (centred columns), and rho^2 is
    reported as the adjusted R^2, 1 - (1 - R^2)(n - 1) / (n - k - 1) with k the non-SV block's rank: the in-sample
    R^2 of a least-squares fit on k columns overstates the population R^2 by about k / (n - 1) under no association,
    and reaches 1 by interpolation at k = n - 1. It is NaN where n - k - 1 <= 0 (the sample does not identify it)
    and is not clipped (a negative value is an estimate below zero, which a clipped one would bias upward).
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
        kept = eigenvalues > max(float(eigenvalues[-1]), 0.0) * other.shape[0] * np.finfo(np.float64).eps
        projected = eigenvectors[:, kept].T @ correlation[np.ix_(other, structural)]
        in_sample = np.clip(np.sum(np.square(projected) / eigenvalues[kept][:, None], axis=0), 0.0, 1.0)
        residual_freedom = int(sample_count) - int(np.count_nonzero(kept)) - 1
        predictability[structural] = (
            1.0 - (1.0 - in_sample) * (int(sample_count) - 1) / residual_freedom if residual_freedom > 0 else np.nan
        )
    return BlockTagging(
        largest_squared_correlation=largest,
        summed_squared_correlation=summed,
        structural_predictability=predictability,
    )

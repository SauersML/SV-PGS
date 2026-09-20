"""Closed-form power for the cytotoxicity benchmark.

expected_polygenic_r2  Daetwyler et al. 2008 (PLoS ONE 3:e3395): held-out r² of a genome-wide predictor trained on n
                       lines, for heritability h² spread over M_e effectively independent segments.
null_r2_moments        exact mean and SD of a held-out r² with no signal: r² ~ Beta(1/2, (m-2)/2) for m test lines.
locus_power            power of a 1-df test of one variant explaining a share q of trait variance in n lines,
                       at level alpha: the noncentral chi-square with noncentrality n·q/(1 - q).
"""
import numpy as np
from scipy import stats


def expected_polygenic_r2(heritability, training_lines, effective_segments):
    x = training_lines * heritability / effective_segments
    return heritability * x / (1.0 + x)


def null_r2_moments(test_lines):
    a, b = 0.5, (test_lines - 2) / 2.0
    mean = a / (a + b)
    variance = a * b / ((a + b) ** 2 * (a + b + 1.0))
    return mean, np.sqrt(variance)


def locus_power(lines, variance_share, alpha):
    critical = stats.chi2.isf(alpha, 1)
    noncentrality = lines * variance_share / (1.0 - variance_share)
    return stats.ncx2.sf(critical, 1, noncentrality)


def detectable_share(lines, alpha, power):
    """Smallest variance share with at least the given power (bisection on the monotone power curve)."""
    low, high = 0.0, 1.0
    while high - low > np.finfo(float).eps * max(high, 1.0):
        middle = 0.5 * (low + high)
        if locus_power(lines, middle, alpha) >= power:
            high = middle
        else:
            low = middle
    return high

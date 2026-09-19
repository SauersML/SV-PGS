"""Derived tolerances for the phenotype tests: float64 rounding bounds and sampling bounds."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

# A seeded simulation check fails with this probability when the estimator is correct.
FALSE_FAILURE_PROBABILITY = 1e-6


def rounding_gamma(operations: int) -> float:
    """gamma_n = n u / (1 - n u): the relative error bound after n rounded float64
    operations (Higham 2002, Lemma 3.1), u the unit roundoff. A libm or special
    function call counts as two operations (it is within one ulp)."""
    unit_roundoff = np.finfo(np.float64).eps / 2
    return operations * unit_roundoff / (1.0 - operations * unit_roundoff)


def within_rounding(expected: float, operations: int, condition: float = 1.0):
    """``expected`` to the forward error bound when both sides of the comparison
    reach it through ``operations`` rounded operations: 2 gamma_n times the
    result's relative condition number."""
    return pytest.approx(expected, rel=2.0 * rounding_gamma(operations) * condition)


def variance_condition(values) -> float:
    """Relative condition number of the population variance, |d var| / var over
    max |d x_i| / |x_i|: 2 mean(|x_i - m| |x_i|) / var."""
    values_array = np.asarray(values, dtype=np.float64)
    deviations = values_array - values_array.mean()
    return float(2.0 * np.mean(np.abs(deviations * values_array)) / values_array.var())


def sampling_bound(standard_error: float) -> float:
    """Two-sided bound on an estimate's error that a correct estimator exceeds
    with probability FALSE_FAILURE_PROBABILITY (normal sampling distribution)."""
    return float(norm.isf(FALSE_FAILURE_PROBABILITY / 2) * standard_error)


def variance_component_standard_errors(
    occasion_counts, between_variance: float, within_variance: float
) -> tuple[float, float]:
    """Gaussian sampling standard errors of the Henderson III moment estimators
    of (sigma_b^2, sigma_e^2) in the random-intercept model.

    SSW / sigma_e^2 is chi-square on N - m degrees of freedom. The residual sum
    of squares of the m person means, independent of SSW, has variance
    2 sum V_i^2 with V_i = sigma_b^2 + sigma_e^2 / k_i, to first order in the
    mean model's leverages; sigma_b^2 subtracts sigma_e^2 sum 1 / k_i from it
    and divides by m.
    """
    counts = np.asarray(occasion_counts, dtype=np.float64)
    within_standard_error = within_variance * np.sqrt(2.0 / (counts.sum() - len(counts)))
    mean_variances = between_variance + within_variance / counts
    between_standard_error = np.sqrt(
        2.0 * np.sum(mean_variances**2) + np.sum(1.0 / counts) ** 2 * within_standard_error**2
    ) / len(counts)
    return float(between_standard_error), float(within_standard_error)

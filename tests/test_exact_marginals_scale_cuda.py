"""The exact marginals on CUDA (in-place SYRK and cuSOLVER potrf): within the certificate of the extended-precision
inverse, and the same as the host route to within both bounds."""
from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.compute_budget import _try_import_cupy
from sv_pgs.exact_marginals_scale import NotCertified, exact_marginals
from tests.test_exact_marginals_scale import _blocks, _problem, _reference

cupy = _try_import_cupy()
pytestmark = pytest.mark.skipif(cupy is None, reason="needs a CUDA device")


def test_cuda_variances_and_bulk_diagonal_are_within_the_certificate() -> None:
    design, precision = _problem(1, 45, 70, negative=3, zero=2)
    result = exact_marginals(_blocks(design, [30, 25, 15]), precision, design.shape[0], bulk_diagonal=True,
                             identity_block=17, array_module=cupy)
    variances, diagonal = _reference(design, precision)
    assert np.all(np.abs(result.variances - variances.astype(np.float64)) <= result.variance_bound)
    assert np.all(np.abs(result.bulk_diagonal - diagonal.astype(np.float64)) <= result.bulk_diagonal_bound)


def test_cuda_and_host_routes_agree_within_both_bounds() -> None:
    design, precision = _problem(3, 50, 90, negative=2)
    host = exact_marginals(_blocks(design, [7, 40, 1, 42]), precision, design.shape[0], bulk_diagonal=True)
    device = exact_marginals(_blocks(design, [7, 40, 1, 42]), precision, design.shape[0], bulk_diagonal=True,
                             array_module=cupy)
    assert np.all(np.abs(host.variances - device.variances) <= host.variance_bound + device.variance_bound)
    assert np.all(np.abs(host.bulk_diagonal - device.bulk_diagonal) <= host.bulk_diagonal_bound + device.bulk_diagonal_bound)


def test_cuda_kept_kernel_is_lower_triangular_and_factors_the_bulk_kernel() -> None:
    design, precision = _problem(7, 35, 50, negative=2, zero=1)
    result = exact_marginals(_blocks(design, [30, 20]), precision, design.shape[0], keep_kernel=True, array_module=cupy)
    lower = cupy.asnumpy(result.kernel.lower)
    bulk = np.ones(precision.size, dtype=bool)
    bulk[result.resolved] = False
    kernel = np.eye(design.shape[0]) + (design[:, bulk] / precision[bulk]) @ design[:, bulk].T
    assert not np.triu(lower, 1).any()
    np.testing.assert_allclose(lower @ lower.T, kernel, rtol=0, atol=1e-12 * np.abs(kernel).max())


def test_cuda_core_that_is_not_positive_definite_is_refused() -> None:
    design, precision = _problem(4, 30, 40)
    precision[5] = -10 * (design[:, 5] ** 2).sum() - 1e3
    with pytest.raises(NotCertified):
        exact_marginals(_blocks(design, [40]), precision, design.shape[0], array_module=cupy)

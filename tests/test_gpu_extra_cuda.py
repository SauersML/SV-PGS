"""Every CUDA library the gpu extra installs loads and computes (pyproject.toml, the gpu extra)."""
from __future__ import annotations

import numpy as np
import pytest

from sv_pgs.compute_budget import _try_import_cupy
from tests.phenotype_bounds import rounding_gamma, sampling_bound

cupy = _try_import_cupy()
pytestmark = pytest.mark.skipif(cupy is None, reason="needs a CUDA device")


def test_curand_draws_through_both_random_interfaces() -> None:
    # The Generator's bit generator and the legacy RandomState each call cuRAND.
    draws = cupy.random.default_rng(0).standard_normal(4096)
    legacy = cupy.random.RandomState(0).standard_normal(4096)
    for sample in (cupy.asnumpy(draws), cupy.asnumpy(legacy)):
        assert np.all(np.isfinite(sample))
        # Continuous draws are distinct, and their mean is within sampling error of 0.
        assert np.unique(sample).shape[0] == sample.shape[0]
        assert abs(float(sample.mean())) < sampling_bound(1.0 / np.sqrt(sample.shape[0]))


def test_cublas_and_cusolver_match_the_host() -> None:
    size = 64
    factor = np.random.default_rng(1).standard_normal((size, size))
    square = factor @ factor.T + size * np.eye(size)
    device = cupy.asarray(square)
    # A product of n-term inner products errs by at most gamma_n |A| |B| on each side (Higham 2002, eq. 3.13).
    magnitude = np.abs(square) @ np.abs(square)
    assert np.all(np.abs(cupy.asnumpy(device @ device) - square @ square) <= 2.0 * rounding_gamma(size) * magnitude)
    # Cholesky is backward stable: L L' = A + E with |E| <= gamma_(n+1) |L| |L'| (Higham 2002, Theorem 10.3),
    # and the host's product of the factors adds gamma_n |L| |L'|.
    cholesky = cupy.asnumpy(cupy.linalg.cholesky(device))
    assert np.all(np.abs(cholesky @ cholesky.T - square) <= 2.0 * rounding_gamma(size + 1) * (np.abs(cholesky) @ np.abs(cholesky).T))


def test_cusparse_multiplies_a_sparse_matrix() -> None:
    from cupyx.scipy import sparse

    dense = np.diag(np.arange(1.0, 9.0)) + np.diag(np.ones(7), 1)
    vector = np.arange(8.0)
    product = sparse.csr_matrix(cupy.asarray(dense)) @ cupy.asarray(vector)
    np.testing.assert_array_equal(cupy.asnumpy(product), dense @ vector)


def test_nvrtc_compiles_a_raw_kernel() -> None:
    kernel = cupy.RawKernel(
        'extern "C" __global__ void twice(const double* x, double* y, int n) '
        "{ int i = blockDim.x * blockIdx.x + threadIdx.x; if (i < n) y[i] = 2.0 * x[i]; }",
        "twice",
    )
    values = cupy.arange(100, dtype=cupy.float64)
    output = cupy.empty_like(values)
    kernel((1,), (128,), (values, output, cupy.int32(100)))
    np.testing.assert_array_equal(cupy.asnumpy(output), 2.0 * np.arange(100.0))

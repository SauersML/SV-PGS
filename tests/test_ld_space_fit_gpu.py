"""Stage 1 on a CUDA device against the host LAPACK path (run on an MSI GPU node)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from sv_pgs import ld_space_fit
from sv_pgs.compute_budget import detect_compute_budget
from tests.test_ld_space_fit import CPU_BUDGET, HOST, _ld_blocks, _quantitative_problem

cp = pytest.importorskip("cupy")
if cp.cuda.runtime.getDeviceCount() == 0:
    pytest.skip("CUDA device not available", allow_module_level=True)

EXECUTOR = ThreadPoolExecutor(max_workers=1)


def _cuda_backend(single_precision: bool) -> ld_space_fit._CudaBackend:
    return ld_space_fit._CudaBackend(cp, 0, detect_compute_budget().working_bytes, EXECUTOR, single_precision)


def _hard_block(rng: np.random.Generator):
    """AR(0.97) LD with r = 0.9995 pairs, biobank-scale data precision, prior-dominated and zero sites."""
    width = 1500
    lags = np.abs(np.arange(width)[:, None] - np.arange(width)[None, :])
    correlation = 0.97**lags
    for anchor in rng.choice(width - 1, 20, replace=False):
        correlation[anchor, anchor + 1] = correlation[anchor + 1, anchor] = 0.9995
    eigenvalues, vectors = np.linalg.eigh(correlation)
    correlation = (vectors * np.maximum(eigenvalues, 1e-6)) @ vectors.T
    correlation = 0.5 * (correlation + correlation.T)
    precision = rng.uniform(3e6, 3e7, size=width)
    precision[rng.choice(width, 20, replace=False)] = 0.0
    return correlation, 1.0e5, precision, rng.standard_normal(width) * 300.0


def test_single_precision_block_posterior_keeps_the_cavity_precision() -> None:
    correlation, scale, precision, linear = _hard_block(np.random.default_rng(1))
    host_mean, host_variance = HOST.block_posterior(correlation, scale, precision, linear)
    cuda = _cuda_backend(single_precision=True)
    device_mean, device_variance = cuda.block_posterior(
        cuda.to_device(correlation), scale, cuda.to_device(precision), cuda.to_device(linear)
    )
    mean, variance = cuda.to_host(device_mean), cuda.to_host(device_variance)
    np.testing.assert_allclose(mean, host_mean, rtol=1e-8, atol=1e-12 * np.max(np.abs(host_mean)))
    np.testing.assert_allclose(variance, host_variance, rtol=1e-5)
    np.testing.assert_allclose(1.0 / variance - precision, 1.0 / host_variance - precision, rtol=1e-4)


def test_cuda_block_posterior_matches_lapack() -> None:
    rng = np.random.default_rng(0)
    width = 700
    samples = rng.standard_normal((2000, width)) @ np.triu(rng.uniform(0.0, 0.2, size=(width, width)))
    correlation = samples.T @ samples / 2000.0
    correlation = 0.5 * (correlation + correlation.T)
    precision = rng.uniform(1.0, 1e5, size=width)
    linear = rng.standard_normal(width)
    host_mean, host_variance = HOST.block_posterior(correlation, 1500.0, precision, linear)
    cuda = _cuda_backend(single_precision=False)
    device_mean, device_variance = cuda.block_posterior(
        cuda.to_device(correlation), 1500.0, cuda.to_device(precision), cuda.to_device(linear)
    )
    np.testing.assert_allclose(cuda.to_host(device_mean), host_mean, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(cuda.to_host(device_variance), host_variance, rtol=1e-10)


def test_cuda_block_posterior_raises_on_an_indefinite_system() -> None:
    cuda = _cuda_backend(single_precision=False)
    indefinite = np.array([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(np.linalg.LinAlgError):
        cuda.block_posterior(cuda.to_device(indefinite), 1.0, cuda.to_device(np.zeros(2)), cuda.to_device(np.ones(2)))


def test_cuda_fit_matches_host_fit() -> None:
    genotypes, _residual, _covariates, boundaries, statistics, hypermodel, _genetic = _quantitative_problem(
        seed=3, sample_count=800, block_widths=[50, 50, 50, 50]
    )
    ld_blocks = _ld_blocks(genotypes, boundaries)
    budget = detect_compute_budget()
    assert budget.device_kind == "cuda"
    (device_fit,) = ld_space_fit.fit_ld_space(ld_blocks, [statistics], hypermodel, budget)
    (host_fit,) = ld_space_fit.fit_ld_space(ld_blocks, [statistics], hypermodel, CPU_BUDGET)
    # Rounding can move convergence by a pass, so the fits agree to the convergence tolerance.
    assert device_fit.converged and host_fit.converged
    assert abs(device_fit.passes - host_fit.passes) <= 1
    np.testing.assert_allclose(device_fit.log_variance_level, host_fit.log_variance_level, rtol=0.0, atol=1e-4)
    np.testing.assert_allclose(device_fit.shape_b, host_fit.shape_b, rtol=1e-4)
    scale = np.max(np.abs(host_fit.posterior_mean))
    np.testing.assert_allclose(device_fit.posterior_mean, host_fit.posterior_mean, rtol=0.0, atol=1e-4 * scale)

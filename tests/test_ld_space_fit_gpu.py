"""Stage 1 on a CUDA device against the host LAPACK path (run on an MSI GPU node)."""

from __future__ import annotations

import numpy as np
import pytest

from sv_pgs import ld_space_fit
from sv_pgs.compute_budget import detect_compute_budget
from tests.test_ld_space_fit import CPU_BUDGET, HOST, _ld_blocks, _quantitative_problem

cp = pytest.importorskip("cupy")
if cp.cuda.runtime.getDeviceCount() == 0:
    pytest.skip("CUDA device not available", allow_module_level=True)


def test_cuda_block_posterior_matches_lapack() -> None:
    rng = np.random.default_rng(0)
    width = 700
    samples = rng.standard_normal((2000, width)) @ np.triu(rng.uniform(0.0, 0.2, size=(width, width)))
    correlation = samples.T @ samples / 2000.0
    correlation = 0.5 * (correlation + correlation.T)
    precision = rng.uniform(1.0, 1e5, size=width)
    linear = rng.standard_normal(width)
    host_mean, host_variance = HOST.block_posterior(correlation, 1500.0, precision, linear)
    cuda = ld_space_fit._CudaBackend(cp, detect_compute_budget().working_bytes)
    device_mean, device_variance = cuda.block_posterior(
        cuda.to_device(correlation), 1500.0, cuda.to_device(precision), cuda.to_device(linear)
    )
    np.testing.assert_allclose(cuda.to_host(device_mean), host_mean, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(cuda.to_host(device_variance), host_variance, rtol=1e-10)


def test_cuda_block_posterior_raises_on_an_indefinite_system() -> None:
    cuda = ld_space_fit._CudaBackend(cp, detect_compute_budget().working_bytes)
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

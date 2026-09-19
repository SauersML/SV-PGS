"""The dual E-step on a CUDA device gives the host's answer."""
from __future__ import annotations

import numpy as np
import pytest

from sv_pgs import dual_solve
from sv_pgs.compute_budget import _try_import_cupy
from tests.test_dual_solve import EPS, MODEL_COUNT, _CodeTileSource, _coded_problem, _problem, _solve_bound

cupy = _try_import_cupy()
pytestmark = pytest.mark.skipif(cupy is None, reason="needs a CUDA device")


def test_cuda_certified_solve_with_deflation_matches_the_host() -> None:
    genotypes, bounds, covariates, weights, variances, prior_mean, response = _problem(31)
    results = []
    for array_module in (np, cupy):
        source = dual_solve.DenseDualSource(array_module.asarray(genotypes), bounds, array_module)
        models = dual_solve.DualModels(array_module.asarray(weights), array_module.asarray(variances), array_module.asarray(covariates), array_module)
        right = dual_solve.mean_right_hand_side(models, array_module.asarray(response), array_module.asarray(genotypes @ prior_mean))
        count = dual_solve.PassCount()
        deflation, _resolved = dual_solve.spike_deflation(source, models, count)
        bound = array_module.asarray(_solve_bound(np.asarray(right.get() if hasattr(right, "get") else right)))
        result = dual_solve.certified_block_cg(source, models, right, array_module.zeros_like(right), array_module.arange(MODEL_COUNT), bound, count, deflation=deflation)
        solution = result.solution.get() if hasattr(result.solution, "get") else result.solution
        results.append((solution, np.asarray(bound.get() if hasattr(bound, "get") else bound)))
    (host, host_bound), (device, _device_bound) = results
    # Both are certified to their bound, so they agree within twice the bound per column.
    assert np.all(np.linalg.norm(host - device, axis=0) <= 2.0 * host_bound * (1.0 + genotypes.shape[0] * EPS))


def test_cuda_code_tiles_with_relaxed_digits_keep_the_certificate() -> None:
    standardized, codes, means, scales, bounds, covariates, weights, variances, prior_mean, response = _coded_problem(43)
    source = dual_solve.StreamedDualSource(_CodeTileSource(codes, means, scales, bounds, cupy, 1 << 28))
    models = dual_solve.DualModels(cupy.asarray(weights), cupy.asarray(variances), cupy.asarray(covariates), cupy)
    right = dual_solve.mean_right_hand_side(models, cupy.asarray(response), cupy.asarray(standardized @ prior_mean))
    count = dual_solve.PassCount()
    deflation, _resolved = dual_solve.spike_deflation(source, models, count)
    host_right = cupy.asnumpy(right)
    bound = cupy.asarray(_solve_bound(host_right))
    result = dual_solve.certified_block_cg(source, models, right, cupy.zeros_like(right), cupy.arange(MODEL_COUNT), bound, count, deflation=deflation)
    assert bool(cupy.all(result.residual_norm <= bound))
    assert any(relative_error > 0.0 for relative_error in result.relative_errors)
    dense_models = dual_solve.DualModels(weights, variances, covariates)
    exact = host_right - np.column_stack([
        _dense_operator(standardized, covariates, weights, variances, model) @ cupy.asnumpy(result.solution[:, model]) for model in range(MODEL_COUNT)
    ])
    assert np.all(np.linalg.norm(exact, axis=0) <= cupy.asnumpy(bound) * (1.0 + standardized.shape[0] * EPS) + standardized.shape[0] * EPS * np.linalg.norm(host_right, axis=0))
    del dense_models


def _dense_operator(genotypes, covariates, weights, variances, model):
    from tests.test_dual_solve import _dense

    return _dense(genotypes, covariates, weights, variances, model)[3]

"""The dual E-step on a CUDA device gives the host's answer."""
from __future__ import annotations

import numpy as np
import pytest

from sv_pgs import dual_solve
from sv_pgs.compute_budget import _try_import_cupy
from tests.test_dual_solve import EPS, MODEL_COUNT, _problem, _solve_bound

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

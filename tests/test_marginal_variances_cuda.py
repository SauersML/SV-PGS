"""The leave-block-out maps on a CUDA device give the host's answer."""
from __future__ import annotations

import numpy as np
import pytest

from sv_pgs import marginal_variances
from sv_pgs.compute_budget import _try_import_cupy
from tests.test_marginal_variances import _grams, _strong_case

cupy = _try_import_cupy()
pytestmark = pytest.mark.skipif(cupy is None, reason="needs a CUDA device")


def test_cuda_marginals_and_jvp_match_the_host() -> None:
    generator, columns, precision, blocks, solve = _strong_case(26)
    grams = _grams(columns, blocks)
    direction = generator.uniform(0.0, 1.0, size=(columns.shape[1], 2)) * precision[:, None]
    host = marginal_variances.marginal_variances(solve, grams)
    device = marginal_variances.marginal_variances(solve, grams, cupy)
    host_jvp = marginal_variances.variance_jvp(solve, grams, direction).values
    device_jvp = marginal_variances.variance_jvp(solve, grams, direction, cupy).values
    # The same float64 algebra in a different summation order: agreement to the conditioning of the window
    # Cholesky (I + omega_F B has eigenvalues >= 1, so its condition number is at most 1 + omega_F lambda_max),
    # times the unit roundoff and the window size.
    window = max(sum(grams.blocks[member].shape[0] for member in marginal_variances._window_blocks(grams, block)) for block in range(len(blocks)))
    eigenvalues = np.linalg.eigvalsh(columns.T @ columns * np.max(1.0 / precision))
    bound = (1.0 + solve.bulk_trace * float(eigenvalues[-1])) * window * np.finfo(np.float64).eps
    assert np.all(np.abs(device - host) <= bound * np.abs(host))
    assert np.all(np.abs(device_jvp - host_jvp) <= bound * np.abs(host_jvp))

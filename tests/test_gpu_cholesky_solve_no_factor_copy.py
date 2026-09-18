"""The GPU Cholesky solve must not duplicate the p x p factor.

``_gpu_cholesky_solve`` used to copy the factor to F order before the two
triangular solves (and cupyx's solve_triangular copies any non-F-order
factor itself). In the working-set exact solve that is a second fp64 p x p
buffer: 13.4 GB at p = 40960, which ran a 40 GB A100 out of memory right
after the Cholesky succeeded. The solve now runs on the F-contiguous
transpose view of the C-contiguous factor cp.linalg.cholesky returns.
"""
from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
cupyx_linalg = pytest.importorskip("cupyx.scipy.linalg")

from sv_pgs.mixture_inference import _gpu_cholesky_solve


def test_gpu_cholesky_solve_uses_the_factor_memory_and_solves_exactly() -> None:
    rng = np.random.default_rng(4)
    dimension = 257
    base = rng.standard_normal((dimension, dimension))
    precision = base @ base.T / dimension + np.eye(dimension)
    right_hand_side = rng.standard_normal((dimension, 3))
    factor_gpu = cp.linalg.cholesky(cp.asarray(precision))
    assert factor_gpu.flags.c_contiguous
    operand_pointers: list[tuple[int, bool]] = []

    def recording_solve_triangular(matrix, rhs, **kwargs):
        operand_pointers.append((int(matrix.data.ptr), bool(matrix.flags.f_contiguous)))
        return cupyx_linalg.solve_triangular(matrix, rhs, **kwargs)

    solution = _gpu_cholesky_solve(cp.asarray(right_hand_side), factor_gpu, recording_solve_triangular)

    assert operand_pointers == [(int(factor_gpu.data.ptr), True)] * 2
    np.testing.assert_allclose(cp.asnumpy(solution), np.linalg.solve(precision, right_hand_side), rtol=1e-10, atol=1e-10)
    vector_solution = _gpu_cholesky_solve(cp.asarray(right_hand_side[:, 0]), factor_gpu, recording_solve_triangular)
    np.testing.assert_allclose(
        cp.asnumpy(vector_solution), np.linalg.solve(precision, right_hand_side[:, 0]), rtol=1e-10, atol=1e-10
    )

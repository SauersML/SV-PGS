"""A failed CUDA-graph capture must leave the sample-space CG solve untouched.

``_solve_sample_space_rhs_gpu_inner`` tries to capture one CG iteration into a
CUDA graph. With real CuPy the capture fails on every eligible (fp16/fp32
resident) cache because the genotype matmul does a host sync
(``bool(cupy.any(...))``). The warm-up iteration that precedes the capture
used to run in place, so a failed capture left solution/residual/search one
CG step ahead of the host ``residual_dot`` the legacy loop continued from:
the next step size and direction update mixed two iterations. A failed
capture must now be indistinguishable from a device without graph support.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from sv_pgs.genotype import as_raw_genotype_matrix
from sv_pgs.mixture_inference import _solve_sample_space_rhs_gpu_inner


class _CaptureFailsStream:
    def __init__(self, non_blocking: bool = False) -> None:
        self.non_blocking = non_blocking

    def __enter__(self) -> "_CaptureFailsStream":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        return None

    def synchronize(self) -> None:
        return None

    def begin_capture(self) -> None:
        raise RuntimeError("operation not permitted when stream is capturing")


class _NoGraphStream:
    """A stream type without capture support: the graph path is never tried."""

    def __init__(self, non_blocking: bool = False) -> None:
        self.non_blocking = non_blocking


class _NumpyCupy:
    def __init__(self, stream_class: type) -> None:
        self.cuda = SimpleNamespace(Stream=stream_class)

    def __getattr__(self, name: str) -> Any:
        return getattr(np, name)


def _solve(stream_class: type) -> tuple[np.ndarray, int, np.ndarray]:
    rng = np.random.default_rng(5)
    raw_matrix = rng.integers(0, 3, size=(40, 25)).astype(np.float32)
    standardized = as_raw_genotype_matrix(raw_matrix).standardized(
        raw_matrix.mean(axis=0),
        np.maximum(raw_matrix.std(axis=0), 0.5).astype(np.float32),
    )
    dense_matrix = standardized.materialize().astype(np.float64)
    standardized._cupy_cache = dense_matrix
    standardized._dense_cache = None
    prior_variances = np.full(25, 0.05)
    diagonal_noise = np.full(40, 0.8)
    right_hand_side = rng.standard_normal((40, 3))
    solution, iterations = _solve_sample_space_rhs_gpu_inner(
        genotype_matrix=standardized,
        prior_variances=prior_variances,
        diagonal_noise=diagonal_noise,
        right_hand_side_gpu=right_hand_side,
        initial_guess_gpu=None,
        tolerance=1e-10,
        max_iterations=200,
        preconditioner=lambda matrix: matrix / 2.0,
        batch_size=8,
        cp=_NumpyCupy(stream_class),
        compute_cp_dtype=np.float64,
    )
    operator = np.diag(diagonal_noise) + dense_matrix @ (prior_variances[:, None] * dense_matrix.T)
    return np.asarray(solution), int(iterations), np.linalg.solve(operator, right_hand_side)


def test_failed_graph_capture_does_not_advance_the_cg_recurrence() -> None:
    failed_solution, failed_iterations, reference = _solve(_CaptureFailsStream)
    plain_solution, plain_iterations, _ = _solve(_NoGraphStream)

    assert failed_iterations == plain_iterations
    np.testing.assert_array_equal(failed_solution, plain_solution)
    np.testing.assert_allclose(failed_solution, reference, rtol=1e-7, atol=1e-9)

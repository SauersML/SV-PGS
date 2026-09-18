"""The GPU sample-space CG must never attempt a CUDA-graph stream capture.

``_solve_sample_space_rhs_gpu_inner`` used to capture one CG iteration into a
CUDA graph for fp16/fp32-resident caches. The captured body applies the
genotype operator, which is cuBLAS, and CuPy refuses cuBLAS during capture
("calling cuBLAS API during stream capture is currently unsupported"), so the
capture failed on every eligible solve. Measured on A100, V100 and H100, the
solve that attempted it then diverged: the residual grew from 6.7e4 to 1e30
in 60 iterations and the run aborted, while the same system without the
capture attempt converged in 13 iterations. The loop now runs the plain
masked PCG recurrence only.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from sv_pgs.genotype import as_raw_genotype_matrix
from sv_pgs.mixture_inference import _solve_sample_space_rhs_gpu_inner


class _RecordingCaptureStream:
    """A capture-capable stream that records every capture attempt."""

    capture_attempts = 0

    def __init__(self, non_blocking: bool = False) -> None:
        self.non_blocking = non_blocking

    def __enter__(self) -> "_RecordingCaptureStream":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        return None

    def synchronize(self) -> None:
        return None

    def begin_capture(self) -> None:
        type(self).capture_attempts += 1
        raise NotImplementedError("calling cuBLAS API during stream capture is currently unsupported")


class _NoGraphStream:
    """A stream type without capture support."""

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


def test_gpu_cg_never_attempts_stream_capture() -> None:
    capture_solution, capture_iterations, reference = _solve(_RecordingCaptureStream)
    plain_solution, plain_iterations, _ = _solve(_NoGraphStream)

    assert _RecordingCaptureStream.capture_attempts == 0
    assert capture_iterations == plain_iterations
    np.testing.assert_array_equal(capture_solution, plain_solution)
    np.testing.assert_allclose(capture_solution, reference, rtol=1e-7, atol=1e-9)

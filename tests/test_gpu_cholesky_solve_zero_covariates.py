"""Regression test for the zero-covariate GPU Cholesky solve path.

The stochastic binary block path in ``fit_variational_em`` passes
``covariate_matrix = np.zeros((n, 0))`` per block. The downstream
``_solve_restricted_mean_only`` builds a 0x0 GLS Cholesky factor and
calls ``_gpu_cholesky_solve``. cuBLAS dtrsm rejects 0x0 inputs with
"parameter 9 (LDA) illegal" because ``max(1, m=0) = 1 > lda=0``.

The fix in ``mixture_inference._gpu_cholesky_solve`` short-circuits on
``factor.shape[0] == 0`` and returns a shape-correct empty solution
without touching cuBLAS. This test pins that contract.

CuPy import is optional - when unavailable we monkey-patch a minimal
namespace so the function under test exercises the empty-shape branch
purely via numpy. That keeps the test runnable on CI machines without
a GPU while still verifying the zero-covariate logic.
"""
from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest


def _ensure_cupy_stub() -> Any:
    """Provide a numpy-backed ``cupy`` namespace if real CuPy is missing.

    The function under test does ``import cupy as cp`` inside its body, so
    we install a stub into ``sys.modules`` BEFORE importing the function.
    Real GPU runtimes are not required to exercise the 0x0 short-circuit.
    """
    try:
        import cupy as real_cp  # type: ignore[import-not-found]
        return real_cp
    except ImportError:
        pass

    stub = types.ModuleType("cupy")
    stub.asarray = lambda x, dtype=None: np.asarray(x, dtype=dtype)  # type: ignore[attr-defined]
    stub.empty = lambda shape, dtype=None: np.empty(shape, dtype=dtype)  # type: ignore[attr-defined]
    stub.ndarray = np.ndarray  # type: ignore[attr-defined]
    sys.modules.setdefault("cupy", stub)
    return stub


def test_gpu_cholesky_solve_zero_covariate_factor_returns_empty_solution():
    """0x0 factor + (0,) rhs => empty solution; never reaches cuBLAS dtrsm."""
    cp = _ensure_cupy_stub()
    from sv_pgs.mixture_inference import _gpu_cholesky_solve  # noqa: E402

    # Build inputs that look like the stochastic-block call site:
    #   factor : 0x0 from cp.linalg.cholesky on (n, 0).T @ (n, 0) = (0, 0)
    #   rhs    : (0,) from (0, n) @ (n,) = (0,)
    factor_gpu = cp.asarray(np.empty((0, 0), dtype=np.float64), dtype=np.float64)
    rhs_gpu = cp.asarray(np.empty((0,), dtype=np.float64), dtype=np.float64)

    sentinel_called = {"n": 0}

    def must_not_be_called(*_args, **_kwargs):
        sentinel_called["n"] += 1
        raise AssertionError(
            "_gpu_cholesky_solve should NOT delegate to cuBLAS dtrsm "
            "when the factor is 0x0; the empty-factor short-circuit "
            "must intercept first."
        )

    result = _gpu_cholesky_solve(rhs_gpu, factor_gpu, must_not_be_called)
    assert sentinel_called["n"] == 0
    assert isinstance(result, np.ndarray)
    assert result.shape == (0,)
    assert result.dtype == np.float64


def test_gpu_cholesky_solve_zero_covariate_factor_matrix_rhs():
    """0x0 factor with (0, k) matrix rhs returns (0, k) empty solution."""
    cp = _ensure_cupy_stub()
    from sv_pgs.mixture_inference import _gpu_cholesky_solve

    factor_gpu = cp.asarray(np.empty((0, 0), dtype=np.float64), dtype=np.float64)
    rhs_gpu = cp.asarray(np.empty((0, 5), dtype=np.float64), dtype=np.float64)

    def must_not_be_called(*_args, **_kwargs):
        raise AssertionError("dtrsm must not run on 0x0 input.")

    result = _gpu_cholesky_solve(rhs_gpu, factor_gpu, must_not_be_called)
    assert isinstance(result, np.ndarray)
    assert result.shape == (0, 5)
    assert result.dtype == np.float64


def test_gpu_cholesky_solve_nonempty_path_still_dispatches():
    """Non-empty inputs must still call the solve_triangular dispatch.

    Sanity check that the empty-factor short-circuit does NOT swallow the
    normal path.
    """
    cp = _ensure_cupy_stub()
    from sv_pgs.mixture_inference import _gpu_cholesky_solve

    # Symmetric positive-definite 3x3 => valid Cholesky factor (use the
    # lower triangular L directly to keep the test independent of cupy
    # cholesky implementation).
    factor_host = np.array(
        [[2.0, 0.0, 0.0], [1.0, 2.0, 0.0], [0.5, 0.5, 2.0]],
        dtype=np.float64,
    )
    rhs_host = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    factor_gpu = cp.asarray(factor_host, dtype=np.float64)
    rhs_gpu = cp.asarray(rhs_host, dtype=np.float64)

    call_log: list[dict[str, Any]] = []

    def fake_solve_triangular(factor, rhs, *, lower, check_finite, trans=None):
        call_log.append(
            {"factor_shape": np.asarray(factor).shape, "rhs_shape": np.asarray(rhs).shape, "trans": trans}
        )
        from scipy.linalg import solve_triangular  # type: ignore[import-untyped]
        return solve_triangular(
            np.asarray(factor),
            np.asarray(rhs),
            lower=lower,
            check_finite=check_finite,
            trans=("T" if trans == "T" else "N"),
        )

    result = _gpu_cholesky_solve(rhs_gpu, factor_gpu, fake_solve_triangular)
    # Two dispatches: L @ y = b, then L.T @ x = y.
    assert len(call_log) == 2
    assert call_log[0]["trans"] is None
    assert call_log[1]["trans"] == "T"
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    # Verify the result matches scipy reference (L L.T x = b).
    from scipy.linalg import cho_solve  # type: ignore[import-untyped]
    expected = cho_solve((factor_host, True), rhs_host)
    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_gpu_cholesky_solve_rejects_nonsquare_factor():
    cp = _ensure_cupy_stub()
    from sv_pgs.mixture_inference import _gpu_cholesky_solve

    factor_gpu = cp.asarray(np.zeros((3, 4), dtype=np.float64), dtype=np.float64)
    rhs_gpu = cp.asarray(np.zeros((3,), dtype=np.float64), dtype=np.float64)

    with pytest.raises(ValueError, match="square factor"):
        _gpu_cholesky_solve(rhs_gpu, factor_gpu, lambda *a, **k: None)


def test_gpu_cholesky_solve_rejects_rhs_shape_mismatch():
    cp = _ensure_cupy_stub()
    from sv_pgs.mixture_inference import _gpu_cholesky_solve

    factor_gpu = cp.asarray(np.eye(3, dtype=np.float64), dtype=np.float64)
    rhs_gpu = cp.asarray(np.zeros((5,), dtype=np.float64), dtype=np.float64)

    with pytest.raises(ValueError, match="incompatible shape"):
        _gpu_cholesky_solve(rhs_gpu, factor_gpu, lambda *a, **k: None)

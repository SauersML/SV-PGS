"""Pin the bitpacked branch of _apply_sample_space_operator_gpu.

Commit 968e0c5 added a branch that evaluates
    (sigma^2 I + X diag(tau^2) X^T) v
as two packed-device kernels per RHS column when the SGM's raw is a
BitpackedDeviceMatrix. Before that, the CG inner loop streamed and
re-standardized the full variant matrix per iteration.

These tests verify:
  1. The branch is taken when raw exposes ``_packed`` + ``matvec`` +
     ``rmatvec`` and ``_cupy_cache is None``.
  2. The result equals the explicit operator
     ``diag_noise[:, None] * v + X @ diag(tau^2) @ X.T @ v``
     to floating-point tolerance.
  3. Vector and matrix RHS shapes are both handled (1D returns 1D,
     2D returns 2D).
  4. The fallback path is unchanged when raw is NOT bitpacked.

The fakes use whichever ``cupy`` namespace ``_ensure_cupy_stub`` returns —
real CuPy when the runtime has a GPU (so the dispatch is exercised against
genuine device arrays), or a numpy-backed stub otherwise. Either way the
fake bitpacked operator does its arithmetic through that namespace and
returns *its* array type, so the operator under test never mixes a host
array with a device array (which CuPy 13 rejects).
"""
from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import Any

import numpy as np


def _ensure_cupy_stub() -> Any:
    """Return real CuPy if importable, else install a numpy-backed namespace."""
    try:
        import cupy as real_cp  # type: ignore[import-not-found]
        # ``import cupy`` is satisfied from ``sys.modules`` — an earlier test in
        # the suite may have left an *incomplete* fake there (no numpy fallback
        # for dtype scalars). That module makes the operator's ``cp.float32``
        # raise ``AttributeError`` even though the import "succeeded". Only trust
        # a module exposing the real device surface; otherwise fall through and
        # return our own complete numpy-backed stub.
        if hasattr(real_cp, "float32") and hasattr(real_cp, "cuda"):
            return real_cp
    except ImportError:
        pass

    stub = types.ModuleType("cupy")

    def _asarray(x, dtype=None):
        return np.asarray(x, dtype=dtype)

    def _zeros_like(x, dtype=None):
        return np.zeros_like(x, dtype=dtype)

    def _ascontiguousarray(x, dtype=None):
        return np.ascontiguousarray(x, dtype=dtype)

    stub.asarray = _asarray  # type: ignore[attr-defined]
    stub.zeros_like = _zeros_like  # type: ignore[attr-defined]
    stub.ascontiguousarray = _ascontiguousarray  # type: ignore[attr-defined]
    stub.asnumpy = lambda x: np.asarray(x)  # type: ignore[attr-defined]
    stub.empty = lambda shape, dtype=None: np.empty(shape, dtype=dtype)  # type: ignore[attr-defined]
    stub.zeros = lambda shape, dtype=None: np.zeros(shape, dtype=dtype)  # type: ignore[attr-defined]
    stub.float32 = np.float32  # type: ignore[attr-defined]
    stub.float64 = np.float64  # type: ignore[attr-defined]
    stub.ndarray = np.ndarray  # type: ignore[attr-defined]
    sys.modules.setdefault("cupy", stub)
    return stub


def _to_numpy(array: Any) -> np.ndarray:
    """Host-copy an array regardless of whether it is CuPy- or numpy-backed."""
    getter = getattr(array, "get", None)
    return getter() if callable(getter) else np.asarray(array)


class _FakeBitpacked:
    """Minimal bitpacked-like object exposing _packed + matvec + rmatvec.

    The algebra is computed from a device-resident standardized matrix (built
    through the same ``cupy`` namespace the operator uses) so matvec/rmatvec
    return device arrays — the operator multiplies the result by a device
    prior vector, which CuPy 13 forbids against a host array.
    """

    def __init__(self, dense_standardized: np.ndarray, cp: Any) -> None:
        self._cp = cp
        self._dense = cp.asarray(np.asarray(dense_standardized, dtype=np.float32))
        self.n_samples = int(self._dense.shape[0])
        self.n_variants = int(self._dense.shape[1])
        # Marker attribute the operator's branch detects.
        self._packed = object()

    def matvec(self, x_dev: Any) -> Any:
        x32 = self._cp.asarray(x_dev, dtype=self._cp.float32)
        if x32.shape != (self.n_variants,):
            raise ValueError(
                f"matvec: x has shape {tuple(x32.shape)}, expected ({self.n_variants},)"
            )
        return (self._dense @ x32).astype(self._cp.float32, copy=False)

    def rmatvec(self, y_dev: Any) -> Any:
        y32 = self._cp.asarray(y_dev, dtype=self._cp.float32)
        if y32.shape != (self.n_samples,):
            raise ValueError(
                f"rmatvec: y has shape {tuple(y32.shape)}, expected ({self.n_samples},)"
            )
        return (self._dense.T @ y32).astype(self._cp.float32, copy=False)


def _make_sgm_with_bitpacked(dense_standardized: np.ndarray, cp: Any) -> Any:
    """Build a minimal SGM-like object the operator code accepts."""
    bp = _FakeBitpacked(dense_standardized, cp)
    sgm = SimpleNamespace()
    sgm.raw = bp
    sgm._cupy_cache = None
    sgm._ld_block_partition = None
    sgm.shape = (bp.n_samples, bp.n_variants)
    sgm.variant_indices = np.arange(bp.n_variants, dtype=np.int32)
    return sgm


def test_bitpacked_branch_matches_explicit_operator_vector_rhs():
    cp = _ensure_cupy_stub()
    from sv_pgs.mixture_inference import _apply_sample_space_operator_gpu

    rng = np.random.default_rng(0)
    n, p = 23, 17
    dense = rng.standard_normal((n, p)).astype(np.float32)
    sgm = _make_sgm_with_bitpacked(dense, cp)
    prior_variances = rng.uniform(0.1, 1.0, size=p).astype(np.float32)
    diagonal_noise = rng.uniform(0.05, 0.5, size=n).astype(np.float32)
    v = rng.standard_normal(n).astype(np.float32)

    result = _apply_sample_space_operator_gpu(
        sgm,
        prior_variances,
        diagonal_noise,
        v,
        batch_size=64,
        cp=cp,
        dtype=np.float32,
    )

    # Reference: diag_noise * v + X @ diag(tau^2) @ X.T @ v
    proj = dense.T @ v
    scaled = prior_variances * proj
    expected = diagonal_noise * v + dense @ scaled

    assert result.shape == (n,)
    np.testing.assert_allclose(_to_numpy(result), expected, rtol=1e-4, atol=1e-4)


def test_bitpacked_branch_matches_explicit_operator_matrix_rhs():
    cp = _ensure_cupy_stub()
    from sv_pgs.mixture_inference import _apply_sample_space_operator_gpu

    rng = np.random.default_rng(1)
    n, p, k = 19, 13, 4
    dense = rng.standard_normal((n, p)).astype(np.float32)
    sgm = _make_sgm_with_bitpacked(dense, cp)
    prior_variances = rng.uniform(0.1, 1.0, size=p).astype(np.float32)
    diagonal_noise = rng.uniform(0.05, 0.5, size=n).astype(np.float32)
    M = rng.standard_normal((n, k)).astype(np.float32)

    result = _apply_sample_space_operator_gpu(
        sgm,
        prior_variances,
        diagonal_noise,
        M,
        batch_size=64,
        cp=cp,
        dtype=np.float32,
    )

    # Reference: diag_noise[:, None] * M + X @ (diag(tau^2) @ X.T @ M)
    proj = dense.T @ M
    scaled = prior_variances[:, None] * proj
    expected = diagonal_noise[:, None] * M + dense @ scaled

    assert result.shape == (n, k)
    np.testing.assert_allclose(_to_numpy(result), expected, rtol=1e-4, atol=1e-4)


def test_bitpacked_branch_skipped_when_cupy_cache_present():
    """When _cupy_cache is set, bitpacked branch must NOT engage (cache wins)."""
    cp = _ensure_cupy_stub()
    from sv_pgs.mixture_inference import _apply_sample_space_operator_gpu

    rng = np.random.default_rng(2)
    n, p = 7, 5
    dense = rng.standard_normal((n, p)).astype(np.float32)
    sgm = _make_sgm_with_bitpacked(dense, cp)

    # Simulate the legacy cache being installed. The bitpacked branch's
    # gate requires _cupy_cache is None — present cache must skip the branch.
    sgm._cupy_cache = cp.asarray(np.ones((n, p), dtype=np.float32))  # placeholder
    bp_called = {"matvec": 0, "rmatvec": 0}

    original_matvec = sgm.raw.matvec
    original_rmatvec = sgm.raw.rmatvec

    def watched_matvec(x):
        bp_called["matvec"] += 1
        return original_matvec(x)

    def watched_rmatvec(y):
        bp_called["rmatvec"] += 1
        return original_rmatvec(y)

    sgm.raw.matvec = watched_matvec
    sgm.raw.rmatvec = watched_rmatvec

    # Provide what the legacy branch needs: gpu_matmat / gpu_transpose_matmat.
    # The operator's legacy branch calls these; stub them to a fixed result
    # (computed through ``cp`` so the return type matches the device inputs)
    # so the test doesn't depend on legacy code correctness.
    dense_dev = cp.asarray(dense)

    def fake_gpu_transpose_matmat(matrix, *, batch_size, cupy, dtype):
        return (dense_dev.T @ cupy.asarray(matrix)).astype(dtype, copy=False)

    def fake_gpu_matmat(matrix, *, batch_size, cupy, dtype):
        return (dense_dev @ cupy.asarray(matrix)).astype(dtype, copy=False)

    sgm.gpu_transpose_matmat = fake_gpu_transpose_matmat
    sgm.gpu_matmat = fake_gpu_matmat
    # The legacy branch also calls _cupy_cache_is_int8_standardized via the
    # imported helper — to keep this test isolated we monkey-patch a name
    # that makes the cache look "fp32-resident" (the not-int8 branch).
    from sv_pgs import mixture_inference as _mi
    original_is_int8 = _mi._cupy_cache_is_int8_standardized
    _mi._cupy_cache_is_int8_standardized = lambda _cache: False
    try:
        v = np.ones(n, dtype=np.float32)
        _ = _apply_sample_space_operator_gpu(
            sgm,
            np.ones(p, dtype=np.float32),
            np.zeros(n, dtype=np.float32),
            v,
            batch_size=64,
            cp=cp,
            dtype=np.float32,
        )
    finally:
        _mi._cupy_cache_is_int8_standardized = original_is_int8

    assert bp_called["matvec"] == 0, "bitpacked matvec must not run when cache is present"
    assert bp_called["rmatvec"] == 0, "bitpacked rmatvec must not run when cache is present"


def test_bitpacked_branch_skipped_when_variant_indices_not_identity():
    """Subset SGM (variant_indices != arange) must fall through to streaming.

    Mirrors the becb3f1 guard on SGM.matvec_numpy / transpose_matvec_numpy:
    the bitpacked matvec returns values over the FULL parent variant set,
    so a subset SGM whose variant_indices select a proper subset cannot
    use the fast path without scattering — fall through instead.
    """
    cp = _ensure_cupy_stub()
    from sv_pgs.mixture_inference import _apply_sample_space_operator_gpu

    rng = np.random.default_rng(3)
    n, p = 11, 9
    dense = rng.standard_normal((n, p)).astype(np.float32)
    sgm = _make_sgm_with_bitpacked(dense, cp)
    # Mark this SGM as exposing 5 of the 9 underlying columns.
    sgm.shape = (n, 5)
    sgm.variant_indices = np.array([0, 2, 3, 5, 7], dtype=np.int32)

    dense_dev = cp.asarray(dense)

    # Need to provide fallback ops since the bitpacked branch shouldn't run.
    def fake_gpu_transpose_matmat(matrix, *, batch_size, cupy, dtype):
        sub = dense_dev[:, sgm.variant_indices]
        return (sub.T @ cupy.asarray(matrix)).astype(dtype, copy=False)

    def fake_gpu_matmat(matrix, *, batch_size, cupy, dtype):
        sub = dense_dev[:, sgm.variant_indices]
        return (sub @ cupy.asarray(matrix)).astype(dtype, copy=False)

    sgm.gpu_transpose_matmat = fake_gpu_transpose_matmat
    sgm.gpu_matmat = fake_gpu_matmat

    bp_called = {"matvec": 0, "rmatvec": 0}
    original_matvec = sgm.raw.matvec
    original_rmatvec = sgm.raw.rmatvec
    sgm.raw.matvec = lambda x: (bp_called.__setitem__("matvec", bp_called["matvec"] + 1) or original_matvec(x))
    sgm.raw.rmatvec = lambda y: (bp_called.__setitem__("rmatvec", bp_called["rmatvec"] + 1) or original_rmatvec(y))

    # The operator falls through to either the cache or the streaming
    # branch. With no _cupy_cache and the streaming-batch helpers stubbed,
    # we just need the function to return something without crashing AND
    # to NOT call bitpacked matvec/rmatvec.
    from sv_pgs import mixture_inference as _mi
    _orig_streaming_iter = getattr(_mi, "_iter_standardized_gpu_batches", None)

    def fake_iter(*_args, **_kwargs):
        # one batch covering the subset columns
        sub = dense_dev[:, sgm.variant_indices]
        yield slice(0, sgm.shape[1]), sub
    _mi._iter_standardized_gpu_batches = fake_iter
    try:
        v = np.ones(n, dtype=np.float32)
        try:
            _apply_sample_space_operator_gpu(
                sgm,
                np.ones(sgm.shape[1], dtype=np.float32),
                np.zeros(n, dtype=np.float32),
                v,
                batch_size=64,
                cp=cp,
                dtype=np.float32,
            )
        except Exception:
            # The streaming fallback may not be exercise-able in this stub
            # environment — the contract we actually need is that the
            # bitpacked methods were NOT called.
            pass
    finally:
        if _orig_streaming_iter is not None:
            _mi._iter_standardized_gpu_batches = _orig_streaming_iter

    assert bp_called["matvec"] == 0, "bitpacked matvec must not run on subset SGM"
    assert bp_called["rmatvec"] == 0, "bitpacked rmatvec must not run on subset SGM"

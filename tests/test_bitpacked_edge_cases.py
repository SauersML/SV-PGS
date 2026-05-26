"""Edge-case sweep for the bitpacked CPU reference kernels.

Pure-CPU tests — GPU asserts are skipped cleanly when CuPy is unavailable.
Each test covers a boundary condition (tiny n_samples, warp boundary, singleton
variant, all-missing column, constant y, singleton active set, zero variants).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sv_pgs.bitpacked.cpu_reference")

from sv_pgs.bitpacked.cpu_reference import (  # noqa: E402
    compute_mean_scale,
    cpu_gemm_gram,
    cpu_gemv_nt,
    cpu_gemv_tn,
    cpu_screen,
)
from sv_pgs.bitpacked.lut import make_decode_lut  # noqa: E402,F401


# PLINK 1.9 packed-byte codes under count_a1=True: 0->2, 1->missing, 2->1, 3->0.
_ENCODE_A1 = {0: 0b11, 1: 0b10, 2: 0b00, -1: 0b01}  # -1 sentinel = missing


def _bpv(n_samples: int) -> int:
    return (int(n_samples) + 3) // 4


def _encode(dosages: list[list[int]], n_samples: int) -> np.ndarray:
    """Encode list-of-lists ``dosages`` (per variant, value in {0,1,2,-1}) to packed bytes."""
    n_v = len(dosages)
    pk = np.zeros((n_v, _bpv(n_samples)), dtype=np.uint8)
    for v, row in enumerate(dosages):
        assert len(row) == n_samples
        for i, d in enumerate(row):
            pk[v, i // 4] |= (_ENCODE_A1[int(d)] & 0b11) << ((i % 4) * 2)
    return pk


def test_n_samples_3():
    n_s, n_v = 3, 10
    rng = np.random.default_rng(1)
    dosages = [list(rng.integers(0, 3, size=n_s).tolist()) for _ in range(n_v)]
    packed = _encode(dosages, n_s)
    assert packed.shape == (n_v, 1)
    mean, scale = compute_mean_scale(packed, n_s)
    x = np.ones((n_v,), dtype=np.float32)
    y = np.ones((n_s,), dtype=np.float32)
    yo = cpu_gemv_nt(packed, n_s, x, mean, scale)
    go = cpu_gemv_tn(packed, n_s, y, mean, scale)
    gram = cpu_gemm_gram(packed, n_s, mean, scale)
    scr = cpu_screen(packed, n_s, rhs=y)
    assert yo.shape == (n_s,) and np.all(np.isfinite(yo))
    assert go.shape == (n_v,) and np.all(np.isfinite(go))
    assert gram.shape == (n_v, n_v) and np.all(np.isfinite(gram))
    assert scr["count"].shape == (n_v,) and scr["dosage_rhs"].shape == (n_v,)


def test_n_samples_32():
    n_s, n_v = 32, 5
    rng = np.random.default_rng(2)
    dosages = [list(rng.integers(0, 3, size=n_s).tolist()) for _ in range(n_v)]
    packed = _encode(dosages, n_s)
    assert packed.shape == (n_v, 8)
    mean, scale = compute_mean_scale(packed, n_s)
    x = np.arange(n_v, dtype=np.float32) / float(n_v)
    y = np.linspace(-1.0, 1.0, n_s, dtype=np.float32)
    yo = cpu_gemv_nt(packed, n_s, x, mean, scale)
    go = cpu_gemv_tn(packed, n_s, y, mean, scale)
    gram = cpu_gemm_gram(packed, n_s, mean, scale)
    scr = cpu_screen(packed, n_s, rhs=y)
    assert yo.shape == (n_s,) and np.all(np.isfinite(yo))
    assert go.shape == (n_v,) and np.all(np.isfinite(go))
    assert gram.shape == (n_v, n_v) and np.all(np.isfinite(gram))
    assert scr["sumsq"].shape == (n_v,)


def test_n_variants_1():
    n_s, n_v = 50, 1
    rng = np.random.default_rng(3)
    dosages = [list(rng.integers(0, 3, size=n_s).tolist())]
    packed = _encode(dosages, n_s)
    assert packed.shape == (1, _bpv(n_s))
    mean, scale = compute_mean_scale(packed, n_s)
    x = np.array([2.0], dtype=np.float32)
    y = np.ones((n_s,), dtype=np.float32)
    yo = cpu_gemv_nt(packed, n_s, x, mean, scale)
    go = cpu_gemv_tn(packed, n_s, y, mean, scale)
    gram = cpu_gemm_gram(packed, n_s, mean, scale)
    scr = cpu_screen(packed, n_s, rhs=y)
    assert yo.shape == (n_s,) and np.all(np.isfinite(yo))
    assert go.shape == (1,) and np.all(np.isfinite(go))
    assert gram.shape == (1, 1) and np.all(np.isfinite(gram))
    assert scr["count"].shape == (1,)


def test_all_missing_variant():
    n_s = 4  # exact byte boundary so 0x55 = all 4 slots are missing
    packed = np.full((1, _bpv(n_s)), 0x55, dtype=np.uint8)
    mean, scale = compute_mean_scale(packed, n_s)
    scr = cpu_screen(packed, n_s)
    assert int(scr["count"][0]) == 0
    assert float(mean[0]) == 0.0
    assert float(scale[0]) == 1.0
    # z-row must be all zero -> gemv_nt with x=[1] returns all zeros.
    x = np.ones((1,), dtype=np.float32)
    yo = cpu_gemv_nt(packed, n_s, x, mean, scale)
    np.testing.assert_array_equal(yo, np.zeros(n_s, dtype=np.float32))


def test_all_equal_y():
    n_s, n_v = 16, 4
    rng = np.random.default_rng(7)
    dosages = [list(rng.integers(0, 3, size=n_s).tolist()) for _ in range(n_v)]
    packed = _encode(dosages, n_s)
    y = np.ones((n_s,), dtype=np.float32)
    scr = cpu_screen(packed, n_s, rhs=y)
    # With constant y=1, dosage_rhs[v] = sum of observed dosages = scr['sum'][v].
    # observed_rhs[v] = non-missing count = scr['count'][v].
    np.testing.assert_allclose(scr["dosage_rhs"], scr["sum"])
    np.testing.assert_allclose(scr["observed_rhs"], scr["count"].astype(np.float64))


def test_active_set_size_1():
    n_s = 20
    rng = np.random.default_rng(11)
    dosages = [list(rng.integers(0, 3, size=n_s).tolist())]
    packed = _encode(dosages, n_s)
    mean, scale = compute_mean_scale(packed, n_s)
    gram = cpu_gemm_gram(packed, n_s, mean, scale)
    assert gram.shape == (1, 1)
    assert np.isfinite(gram[0, 0])
    assert float(gram[0, 0]) >= 0.0


def test_zero_variants_packed_shape():
    n_s = 12
    packed = np.zeros((0, _bpv(n_s)), dtype=np.uint8)
    mean = np.zeros((0,), dtype=np.float32)
    scale = np.ones((0,), dtype=np.float32)
    x = np.zeros((0,), dtype=np.float32)
    y = np.ones((n_s,), dtype=np.float32)
    yo = cpu_gemv_nt(packed, n_s, x, mean, scale)
    go = cpu_gemv_tn(packed, n_s, y, mean, scale)
    gram = cpu_gemm_gram(packed, n_s, mean, scale)
    assert yo.shape == (n_s,)
    np.testing.assert_array_equal(yo, np.zeros(n_s, dtype=yo.dtype))
    assert go.shape == (0,)
    assert gram.shape == (0, 0)

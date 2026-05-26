from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
gemv_tn_mod = pytest.importorskip("sv_pgs.bitpacked.gemv_tn")
cpu_ref_mod = pytest.importorskip("sv_pgs.bitpacked.cpu_reference")

gemv_tn = gemv_tn_mod.gemv_tn
cpu_gemv_tn = cpu_ref_mod.cpu_gemv_tn

RTOL = 1e-4
ATOL = 1e-3


def _bytes_per_variant(n_samples: int) -> int:
    return (n_samples + 3) // 4


def _make_random_packed(n_samples: int, n_variants: int, seed: int = 0,
                        include_missing: bool = True) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bpv = _bytes_per_variant(n_samples)
    if include_missing:
        codes = rng.integers(0, 4, size=(n_variants, bpv * 4), dtype=np.uint8)
    else:
        # Avoid the missing code 0b01
        choices = np.array([0b00, 0b10, 0b11], dtype=np.uint8)
        idx = rng.integers(0, 3, size=(n_variants, bpv * 4))
        codes = choices[idx]
    # Zero out padding slots beyond n_samples
    if n_samples < bpv * 4:
        codes[:, n_samples:] = 0b11  # arbitrary non-missing code; CPU ref ignores padding
    packed = np.zeros((n_variants, bpv), dtype=np.uint8)
    for j in range(4):
        packed |= (codes[:, j::4] & 0b11) << (2 * j)
    return packed


def _make_mean_std(packed: np.ndarray, n_samples: int, count_a1: bool, seed: int = 1):
    rng = np.random.default_rng(seed)
    n_variants = packed.shape[0]
    mean = rng.uniform(0.1, 1.9, size=n_variants).astype(np.float32)
    std = rng.uniform(0.3, 1.5, size=n_variants).astype(np.float32)
    return mean, std


def test_gemv_tn_matches_cpu_reference_small():
    n_samples, n_variants = 100, 500
    packed = _make_random_packed(n_samples, n_variants, seed=42)
    mean, std = _make_mean_std(packed, n_samples, True, seed=2)
    rng = np.random.default_rng(7)
    y = rng.standard_normal(n_samples).astype(np.float32)

    ref = cpu_gemv_tn(packed, n_samples, y, mean, std, count_a1=True).astype(np.float32)

    packed_d = cp.asarray(packed)
    y_d = cp.asarray(y)
    mean_d = cp.asarray(mean)
    std_d = cp.asarray(std)
    out_d = cp.zeros(n_variants, dtype=cp.float32)

    gemv_tn(packed_d, n_samples, y_d, mean_d, std_d, out_d, count_a1=True)
    got = cp.asnumpy(out_d)

    np.testing.assert_allclose(got, ref, rtol=RTOL, atol=ATOL)


def test_gemv_tn_handles_missing():
    n_samples, n_variants = 100, 500
    rng = np.random.default_rng(123)
    bpv = _bytes_per_variant(n_samples)
    # Force many 0b01 missing codes
    codes = rng.integers(0, 4, size=(n_variants, bpv * 4), dtype=np.uint8)
    # Inject explicit missing slots
    miss_mask = rng.random(size=codes.shape) < 0.3
    codes[miss_mask] = 0b01
    packed = np.zeros((n_variants, bpv), dtype=np.uint8)
    for j in range(4):
        packed |= (codes[:, j::4] & 0b11) << (2 * j)

    mean, std = _make_mean_std(packed, n_samples, True, seed=9)
    y = rng.standard_normal(n_samples).astype(np.float32)

    ref = cpu_gemv_tn(packed, n_samples, y, mean, std, count_a1=True).astype(np.float32)

    packed_d = cp.asarray(packed)
    y_d = cp.asarray(y)
    mean_d = cp.asarray(mean)
    std_d = cp.asarray(std)
    out_d = cp.zeros(n_variants, dtype=cp.float32)

    gemv_tn(packed_d, n_samples, y_d, mean_d, std_d, out_d, count_a1=True)
    got = cp.asnumpy(out_d)

    np.testing.assert_allclose(got, ref, rtol=RTOL, atol=ATOL)


def test_gemv_tn_padding():
    n_samples, n_variants = 97, 250
    packed = _make_random_packed(n_samples, n_variants, seed=11)
    mean, std = _make_mean_std(packed, n_samples, True, seed=3)
    rng = np.random.default_rng(17)
    y = rng.standard_normal(n_samples).astype(np.float32)

    ref = cpu_gemv_tn(packed, n_samples, y, mean, std, count_a1=True).astype(np.float32)

    packed_d = cp.asarray(packed)
    y_d = cp.asarray(y)
    mean_d = cp.asarray(mean)
    std_d = cp.asarray(std)
    out_d = cp.zeros(n_variants, dtype=cp.float32)

    gemv_tn(packed_d, n_samples, y_d, mean_d, std_d, out_d, count_a1=True)
    got = cp.asnumpy(out_d)

    np.testing.assert_allclose(got, ref, rtol=RTOL, atol=ATOL)


def test_gemv_tn_accumulates_into_out():
    n_samples, n_variants = 100, 300
    packed = _make_random_packed(n_samples, n_variants, seed=4)
    mean, std = _make_mean_std(packed, n_samples, True, seed=5)
    rng = np.random.default_rng(21)
    y = rng.standard_normal(n_samples).astype(np.float32)

    ref = cpu_gemv_tn(packed, n_samples, y, mean, std, count_a1=True).astype(np.float32)
    pre = rng.standard_normal(n_variants).astype(np.float32)
    expected = pre + ref

    packed_d = cp.asarray(packed)
    y_d = cp.asarray(y)
    mean_d = cp.asarray(mean)
    std_d = cp.asarray(std)
    out_d = cp.asarray(pre.copy())

    gemv_tn(packed_d, n_samples, y_d, mean_d, std_d, out_d, count_a1=True)
    got = cp.asnumpy(out_d)

    np.testing.assert_allclose(got, expected, rtol=RTOL, atol=ATOL)


def test_gemv_tn_count_a1_false():
    n_samples, n_variants = 100, 400
    packed = _make_random_packed(n_samples, n_variants, seed=33)
    mean, std = _make_mean_std(packed, n_samples, False, seed=6)
    rng = np.random.default_rng(77)
    y = rng.standard_normal(n_samples).astype(np.float32)

    ref = cpu_gemv_tn(packed, n_samples, y, mean, std, count_a1=False).astype(np.float32)

    packed_d = cp.asarray(packed)
    y_d = cp.asarray(y)
    mean_d = cp.asarray(mean)
    std_d = cp.asarray(std)
    out_d = cp.zeros(n_variants, dtype=cp.float32)

    gemv_tn(packed_d, n_samples, y_d, mean_d, std_d, out_d, count_a1=False)
    got = cp.asnumpy(out_d)

    np.testing.assert_allclose(got, ref, rtol=RTOL, atol=ATOL)


def test_gemv_tn_zero_std_column():
    n_samples, n_variants = 100, 50
    packed = _make_random_packed(n_samples, n_variants, seed=99)
    mean, std = _make_mean_std(packed, n_samples, True, seed=8)
    # Per spec, zero-std columns are imputed (mean=0, std=1.0)
    zero_cols = [3, 17, 42]
    for c in zero_cols:
        mean[c] = 0.0
        std[c] = 1.0
    rng = np.random.default_rng(55)
    y = rng.standard_normal(n_samples).astype(np.float32)

    ref = cpu_gemv_tn(packed, n_samples, y, mean, std, count_a1=True).astype(np.float32)

    packed_d = cp.asarray(packed)
    y_d = cp.asarray(y)
    mean_d = cp.asarray(mean)
    std_d = cp.asarray(std)
    out_d = cp.zeros(n_variants, dtype=cp.float32)

    gemv_tn(packed_d, n_samples, y_d, mean_d, std_d, out_d, count_a1=True)
    got = cp.asnumpy(out_d)

    np.testing.assert_allclose(got, ref, rtol=RTOL, atol=ATOL)
    # And the zero-std columns must be finite
    assert np.all(np.isfinite(got[zero_cols]))


def test_gemv_tn_large_n_samples():
    n_samples, n_variants = 10000, 50
    packed = _make_random_packed(n_samples, n_variants, seed=2024)
    mean, std = _make_mean_std(packed, n_samples, True, seed=2025)
    rng = np.random.default_rng(2026)
    y = rng.standard_normal(n_samples).astype(np.float32)

    ref = cpu_gemv_tn(packed, n_samples, y, mean, std, count_a1=True).astype(np.float32)

    packed_d = cp.asarray(packed)
    y_d = cp.asarray(y)
    mean_d = cp.asarray(mean)
    std_d = cp.asarray(std)
    out_d = cp.zeros(n_variants, dtype=cp.float32)

    gemv_tn(packed_d, n_samples, y_d, mean_d, std_d, out_d, count_a1=True)
    got = cp.asnumpy(out_d)

    np.testing.assert_allclose(got, ref, rtol=RTOL, atol=ATOL)

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
gemv_nt_mod = pytest.importorskip("sv_pgs.bitpacked.gemv_nt")
cpu_ref_mod = pytest.importorskip("sv_pgs.bitpacked.cpu_reference")

gemv_nt = gemv_nt_mod.gemv_nt
cpu_gemv_nt = cpu_ref_mod.cpu_gemv_nt


def _bytes_per_variant(n_samples: int) -> int:
    return (n_samples + 3) // 4


def _random_packed(rng: np.random.Generator, n_variants: int, n_samples: int) -> np.ndarray:
    bpv = _bytes_per_variant(n_samples)
    return rng.integers(0, 256, size=(n_variants, bpv), dtype=np.uint8)


def _random_mean_std(rng: np.random.Generator, n_variants: int) -> tuple[np.ndarray, np.ndarray]:
    mean = rng.uniform(0.0, 2.0, size=n_variants).astype(np.float32)
    std = rng.uniform(0.5, 1.5, size=n_variants).astype(np.float32)
    return mean, std


def test_gemv_nt_matches_cpu_reference_small():
    rng = np.random.default_rng(0)
    n_samples, n_variants = 100, 500
    packed = _random_packed(rng, n_variants, n_samples)
    x = rng.standard_normal(n_variants).astype(np.float32)
    mean, std = _random_mean_std(rng, n_variants)

    y_cpu = cpu_gemv_nt(packed, n_samples, x, mean, std, count_a1=True)

    d_packed = cp.asarray(packed)
    d_x = cp.asarray(x)
    d_mean = cp.asarray(mean)
    d_std = cp.asarray(std)
    d_out = cp.zeros(n_samples, dtype=cp.float32)

    gemv_nt(d_packed, n_samples, d_x, d_mean, d_std, d_out, count_a1=True)
    y_gpu = d_out.get()

    np.testing.assert_allclose(y_gpu, y_cpu, rtol=1e-4, atol=1e-3)


def test_gemv_nt_handles_missing():
    rng = np.random.default_rng(1)
    n_samples, n_variants = 100, 500
    packed = _random_packed(rng, n_variants, n_samples)
    # Force several bytes to contain 0b01 codes (missing) explicitly.
    packed[0, 0] = 0b01010101  # all four slots missing in this byte
    packed[10, 5] = 0b01000001
    x = rng.standard_normal(n_variants).astype(np.float32)
    mean, std = _random_mean_std(rng, n_variants)

    y_cpu = cpu_gemv_nt(packed, n_samples, x, mean, std, count_a1=True)

    d_out = cp.zeros(n_samples, dtype=cp.float32)
    gemv_nt(
        cp.asarray(packed),
        n_samples,
        cp.asarray(x),
        cp.asarray(mean),
        cp.asarray(std),
        d_out,
        count_a1=True,
    )
    y_gpu = d_out.get()
    np.testing.assert_allclose(y_gpu, y_cpu, rtol=1e-4, atol=1e-3)


def test_gemv_nt_padding():
    rng = np.random.default_rng(2)
    n_samples, n_variants = 97, 500  # 97 % 4 == 1 → 3 padding slots
    packed = _random_packed(rng, n_variants, n_samples)
    x = rng.standard_normal(n_variants).astype(np.float32)
    mean, std = _random_mean_std(rng, n_variants)

    y_cpu = cpu_gemv_nt(packed, n_samples, x, mean, std, count_a1=True)

    d_out = cp.zeros(n_samples, dtype=cp.float32)
    gemv_nt(
        cp.asarray(packed),
        n_samples,
        cp.asarray(x),
        cp.asarray(mean),
        cp.asarray(std),
        d_out,
        count_a1=True,
    )
    y_gpu = d_out.get()
    assert y_gpu.shape == (n_samples,)
    np.testing.assert_allclose(y_gpu, y_cpu, rtol=1e-4, atol=1e-3)


def test_gemv_nt_accumulates_into_out():
    rng = np.random.default_rng(3)
    n_samples, n_variants = 100, 500
    packed = _random_packed(rng, n_variants, n_samples)
    x = rng.standard_normal(n_variants).astype(np.float32)
    mean, std = _random_mean_std(rng, n_variants)

    y_cpu = cpu_gemv_nt(packed, n_samples, x, mean, std, count_a1=True)

    preinit = rng.standard_normal(n_samples).astype(np.float32)
    d_out = cp.asarray(preinit.copy())
    gemv_nt(
        cp.asarray(packed),
        n_samples,
        cp.asarray(x),
        cp.asarray(mean),
        cp.asarray(std),
        d_out,
        count_a1=True,
    )
    y_gpu = d_out.get()
    np.testing.assert_allclose(y_gpu, preinit + y_cpu, rtol=1e-4, atol=1e-3)


def test_gemv_nt_count_a1_false():
    rng = np.random.default_rng(4)
    n_samples, n_variants = 100, 500
    packed = _random_packed(rng, n_variants, n_samples)
    x = rng.standard_normal(n_variants).astype(np.float32)
    mean, std = _random_mean_std(rng, n_variants)

    y_cpu = cpu_gemv_nt(packed, n_samples, x, mean, std, count_a1=False)

    d_out = cp.zeros(n_samples, dtype=cp.float32)
    gemv_nt(
        cp.asarray(packed),
        n_samples,
        cp.asarray(x),
        cp.asarray(mean),
        cp.asarray(std),
        d_out,
        count_a1=False,
    )
    y_gpu = d_out.get()
    np.testing.assert_allclose(y_gpu, y_cpu, rtol=1e-4, atol=1e-3)


def test_gemv_nt_zero_std_column():
    rng = np.random.default_rng(5)
    n_samples, n_variants = 100, 500
    packed = _random_packed(rng, n_variants, n_samples)
    x = rng.standard_normal(n_variants).astype(np.float32)
    mean, std = _random_mean_std(rng, n_variants)

    zero_col = 42
    std[zero_col] = 0.0
    mean[zero_col] = 0.0

    y_cpu_full = cpu_gemv_nt(packed, n_samples, x, mean, std, count_a1=True)

    # Zero out the contribution of the zero-std column on the input side and
    # compare: removing it from x should not change the output.
    x_without = x.copy()
    x_without[zero_col] = 0.0
    y_cpu_without = cpu_gemv_nt(packed, n_samples, x_without, mean, std, count_a1=True)
    np.testing.assert_allclose(y_cpu_full, y_cpu_without, rtol=1e-4, atol=1e-3)

    d_out = cp.zeros(n_samples, dtype=cp.float32)
    gemv_nt(
        cp.asarray(packed),
        n_samples,
        cp.asarray(x),
        cp.asarray(mean),
        cp.asarray(std),
        d_out,
        count_a1=True,
    )
    y_gpu = d_out.get()
    np.testing.assert_allclose(y_gpu, y_cpu_full, rtol=1e-4, atol=1e-3)

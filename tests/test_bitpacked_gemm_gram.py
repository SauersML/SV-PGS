from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
gemm_gram_mod = pytest.importorskip("sv_pgs.bitpacked.gemm_gram")
cpu_ref = pytest.importorskip("sv_pgs.bitpacked.cpu_reference")

try:
    cp.cuda.runtime.getDeviceCount()
except Exception:  # pragma: no cover
    pytest.skip("CUDA device not available", allow_module_level=True)


GEMM_RTOL = 1e-3
GEMM_ATOL = 1e-2


def _bytes_per_variant(n_samples: int) -> int:
    return (n_samples + 3) // 4


def _pack_dosages(dosages: np.ndarray, missing_mask: np.ndarray, count_a1: bool = True) -> np.ndarray:
    """Pack a (n_variants, n_samples) int array of dosages {0,1,2} with a missing mask
    into PLINK1 bitpacked layout (n_variants, bytes_per_variant) uint8."""
    n_variants, n_samples = dosages.shape
    bpv = _bytes_per_variant(n_samples)
    packed = np.zeros((n_variants, bpv), dtype=np.uint8)
    # Map dosage -> 2-bit code under count_a1=True:
    # 0 -> 0b11, 1 -> 0b10, 2 -> 0b00, missing -> 0b01
    if count_a1:
        dose_to_code = {0: 0b11, 1: 0b10, 2: 0b00}
    else:
        dose_to_code = {0: 0b00, 1: 0b10, 2: 0b11}
    for v in range(n_variants):
        for i in range(n_samples):
            if missing_mask[v, i]:
                code = 0b01
            else:
                code = dose_to_code[int(dosages[v, i])]
            byte_idx = i // 4
            slot = i % 4
            packed[v, byte_idx] |= np.uint8(code << (2 * slot))
    return packed


def _compute_mean_std(dosages: np.ndarray, missing_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_variants, _ = dosages.shape
    mean = np.zeros(n_variants, dtype=np.float32)
    std = np.zeros(n_variants, dtype=np.float32)
    for v in range(n_variants):
        valid = ~missing_mask[v]
        if valid.sum() == 0:
            mean[v] = 0.0
            std[v] = 1.0
            continue
        vals = dosages[v, valid].astype(np.float64)
        m = vals.mean()
        s = vals.std()
        mean[v] = np.float32(m)
        std[v] = np.float32(s) if s > 0 else np.float32(1.0)
    return mean, std


def _make_random_case(n_samples: int, n_variants: int, missing_rate: float = 0.0, seed: int = 0,
                     force_zero_std: list[int] | None = None) -> dict:
    rng = np.random.default_rng(seed)
    dosages = rng.integers(0, 3, size=(n_variants, n_samples), dtype=np.int8)
    missing_mask = rng.random((n_variants, n_samples)) < missing_rate
    if force_zero_std is not None:
        for v in force_zero_std:
            dosages[v, :] = 1
            missing_mask[v, :] = False
    mean, std = _compute_mean_std(dosages.astype(np.int64), missing_mask)
    packed = _pack_dosages(dosages.astype(np.int64), missing_mask, count_a1=True)
    return {
        "dosages": dosages,
        "missing_mask": missing_mask,
        "packed": packed,
        "mean": mean,
        "std": std,
        "n_samples": n_samples,
        "n_variants": n_variants,
    }


def _run_kernel(case: dict, out_init: cp.ndarray | None = None) -> cp.ndarray:
    packed_d = cp.asarray(case["packed"])
    mean_d = cp.asarray(case["mean"])
    std_d = cp.asarray(case["std"])
    n_variants = case["n_variants"]
    if out_init is None:
        out_d = cp.zeros((n_variants, n_variants), dtype=cp.float32)
    else:
        out_d = out_init
    gemm_gram_mod.gemm_gram(
        packed_d, case["n_samples"], mean_d, std_d, out_d, count_a1=True, stream=None
    )
    return out_d


def test_gram_matches_cpu_reference_small():
    case = _make_random_case(n_samples=100, n_variants=50, missing_rate=0.05, seed=1)
    out_d = _run_kernel(case)
    out = cp.asnumpy(out_d)
    ref = cpu_ref.cpu_gemm_gram(
        case["packed"], case["n_samples"], case["mean"], case["std"], count_a1=True
    )
    np.testing.assert_allclose(out, ref, rtol=GEMM_RTOL, atol=GEMM_ATOL)


def test_gram_symmetric():
    case = _make_random_case(n_samples=128, n_variants=40, missing_rate=0.03, seed=2)
    out = cp.asnumpy(_run_kernel(case))
    np.testing.assert_allclose(out, out.T, rtol=1e-5, atol=1e-4)


def test_gram_with_missing():
    case = _make_random_case(n_samples=96, n_variants=32, missing_rate=0.15, seed=3)
    out = cp.asnumpy(_run_kernel(case))

    # Build Z manually with missing-rows zeroed.
    n_samples = case["n_samples"]
    n_variants = case["n_variants"]
    Z = np.zeros((n_samples, n_variants), dtype=np.float64)
    for v in range(n_variants):
        for i in range(n_samples):
            if case["missing_mask"][v, i]:
                Z[i, v] = 0.0
            else:
                Z[i, v] = (float(case["dosages"][v, i]) - case["mean"][v]) / case["std"][v]
    ref = Z.T @ Z
    np.testing.assert_allclose(out, ref, rtol=GEMM_RTOL, atol=GEMM_ATOL)


def test_gram_accumulates_into_out():
    case = _make_random_case(n_samples=100, n_variants=24, missing_rate=0.0, seed=4)
    n_variants = case["n_variants"]
    initial = cp.ones((n_variants, n_variants), dtype=cp.float32) * 3.0
    out_d = initial.copy()
    _run_kernel(case, out_init=out_d)
    fresh = cp.asnumpy(_run_kernel(case))
    got = cp.asnumpy(out_d)
    np.testing.assert_allclose(got, fresh + 3.0, rtol=GEMM_RTOL, atol=GEMM_ATOL)


def test_gram_zero_std_column():
    n_variants = 20
    zero_idx = 7
    case = _make_random_case(
        n_samples=80, n_variants=n_variants, missing_rate=0.02, seed=5,
        force_zero_std=[zero_idx],
    )
    # In _compute_mean_std, when all dosages identical we set std=1, mean=1 -> z=0 per sample.
    # Force std to 0 to exercise the std==0 branch in the kernel.
    case["std"][zero_idx] = 0.0
    out = cp.asnumpy(_run_kernel(case))
    assert np.allclose(out[zero_idx, :], 0.0, atol=1e-5)
    assert np.allclose(out[:, zero_idx], 0.0, atol=1e-5)


def test_gram_n_variants_64():
    case = _make_random_case(n_samples=200, n_variants=64, missing_rate=0.04, seed=6)
    out = cp.asnumpy(_run_kernel(case))
    ref = cpu_ref.cpu_gemm_gram(
        case["packed"], case["n_samples"], case["mean"], case["std"], count_a1=True
    )
    np.testing.assert_allclose(out, ref, rtol=GEMM_RTOL, atol=GEMM_ATOL)


def test_gram_n_variants_127():
    case = _make_random_case(n_samples=150, n_variants=127, missing_rate=0.04, seed=7)
    out = cp.asnumpy(_run_kernel(case))
    ref = cpu_ref.cpu_gemm_gram(
        case["packed"], case["n_samples"], case["mean"], case["std"], count_a1=True
    )
    np.testing.assert_allclose(out, ref, rtol=GEMM_RTOL, atol=GEMM_ATOL)

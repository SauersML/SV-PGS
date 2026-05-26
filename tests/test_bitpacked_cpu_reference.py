"""Tests pinning the CPU reference bitpacked helpers against the canonical
``sv_pgs.plink._decode_payload`` LUT and explicit numpy ground truth.

See ``BITPACKED_SPEC.md`` for the contract these functions must satisfy.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("sv_pgs.bitpacked.cpu_reference")

from sv_pgs.bitpacked import cpu_reference as cpu_ref  # noqa: E402
from sv_pgs.plink import (  # noqa: E402
    PLINK_MISSING_INT8,
    _bytes_per_variant,
    _decode_payload,
)

FP32_RTOL = 1e-5
FP32_ATOL = 1e-5
FP64_RTOL = 1e-12
FP64_ATOL = 1e-12


def _random_packed(
    n_samples: int,
    n_variants: int,
    seed: int,
    zero_padding: bool = True,
) -> np.ndarray:
    """Build a uint8 (n_variants, bytes_per_variant) packed buffer.

    When ``zero_padding`` is True, trailing pad slots in the last byte are
    explicitly cleared so a roundtrip through ``_decode_payload`` (which
    trims to ``n_samples``) is well-defined.
    """
    rng = np.random.default_rng(seed)
    bpv = _bytes_per_variant(n_samples)
    packed = rng.integers(0, 256, size=(n_variants, bpv), dtype=np.uint8)
    if zero_padding:
        leftover = bpv * 4 - n_samples
        if leftover:
            # Clear the top 2*leftover bits of the final byte for each variant.
            keep_bits = (1 << (2 * (4 - leftover))) - 1
            packed[:, -1] = packed[:, -1] & np.uint8(keep_bits)
    return packed


def _mean_scale_from_decoded(
    decoded_v_n: np.ndarray,
    minimum_scale: float = 1e-6,
    dtype: np.dtype = np.float64,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-variant mean/scale matching the BITPACKED_SPEC contract.

    - Denominator for scale is total N (not non-missing count).
    - All-missing column => mean=0, scale=1.
    - Low-variance (scale_raw < minimum_scale) => mean kept (actual mean), scale=1.
    """
    n_variants, N = decoded_v_n.shape
    miss = decoded_v_n == PLINK_MISSING_INT8
    raw = decoded_v_n.astype(np.float64)
    raw_nm = np.where(miss, 0.0, raw)
    count = (~miss).sum(axis=1).astype(np.float64)
    sums = raw_nm.sum(axis=1)
    sumsq = (raw_nm * raw_nm).sum(axis=1)

    mean = np.zeros(n_variants, dtype=np.float64)
    scale = np.ones(n_variants, dtype=np.float64)
    has_obs = count > 0
    safe_count = np.where(has_obs, count, 1.0)
    actual_mean = sums / safe_count
    css = sumsq - (sums * sums) / safe_count
    css = np.where(css < 0, 0.0, css)  # numerical safety
    scale_raw = np.sqrt(css / max(N, 1))

    # all-missing: mean=0, scale=1 (defaults already set)
    # observed & not low-variance: mean=actual, scale=scale_raw
    # observed & low-variance: mean=actual (kept), scale=1
    low_var = scale_raw < minimum_scale
    use_actual_mean = has_obs
    mean = np.where(use_actual_mean, actual_mean, 0.0)
    scale = np.where(has_obs & (~low_var), scale_raw, 1.0)
    return mean.astype(dtype), scale.astype(dtype)


def _standardize(decoded_v_n: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Build Z of shape (n_samples, n_variants) per spec; missing -> 0."""
    miss = decoded_v_n == PLINK_MISSING_INT8
    raw = decoded_v_n.astype(np.float64)
    z_v_n = (raw - mean[:, None]) / scale[:, None]
    z_v_n = np.where(miss, 0.0, z_v_n)
    return z_v_n.T  # (n_samples, n_variants)


# Backward-compat alias for any old call sites that still expect the old name.
_mean_std_from_decoded = _mean_scale_from_decoded


# ---------------------------------------------------------------------------
# Test 1: decode matches plink LUT
# ---------------------------------------------------------------------------


def test_cpu_decode_matches_plink_lut() -> None:
    n_samples = 50
    n_variants = 200
    packed = _random_packed(n_samples, n_variants, seed=0xC0FFEE)
    bpv = _bytes_per_variant(n_samples)

    # Canonical reference: returns (n_samples, n_variants) int8.
    payload = packed.tobytes()
    ref = _decode_payload(
        payload,
        iid_count=n_samples,
        variant_count=n_variants,
        bytes_per_variant=bpv,
        count_a1=True,
    )
    assert ref.shape == (n_samples, n_variants)

    # cpu_reference helper: returns (n_variants, n_samples) int8.
    got = cpu_ref._decode_packed(packed, n_samples=n_samples, count_a1=True)
    assert got.shape == (n_variants, n_samples)
    np.testing.assert_array_equal(got, ref.T)

    # Also under count_a1=False.
    ref2 = _decode_payload(
        payload,
        iid_count=n_samples,
        variant_count=n_variants,
        bytes_per_variant=bpv,
        count_a1=False,
    )
    got2 = cpu_ref._decode_packed(packed, n_samples=n_samples, count_a1=False)
    np.testing.assert_array_equal(got2, ref2.T)


# ---------------------------------------------------------------------------
# Test 2: forward GEMV
# ---------------------------------------------------------------------------


def test_cpu_gemv_nt_matches_explicit_loop() -> None:
    n_samples = 10
    n_variants = 30
    # Don't zero pad so we explicitly verify pad-handling.
    packed = _random_packed(n_samples, n_variants, seed=1234, zero_padding=False)

    decoded = cpu_ref._decode_packed(packed, n_samples=n_samples, count_a1=True)
    # decoded is (V, N) int8.
    missing_mask = decoded == PLINK_MISSING_INT8
    # Force at least one missing per row so the path is exercised.
    decoded_force = decoded.copy()
    decoded_force[:, 0] = PLINK_MISSING_INT8  # not actually mutating packed; checked via mean/std synthesis only

    mean, scale = _mean_scale_from_decoded(decoded)
    rng = np.random.default_rng(7)
    x = rng.standard_normal(n_variants).astype(np.float32)

    # Explicit numpy loop over the (V, N) decoded matrix.
    y_ref = np.zeros(n_samples, dtype=np.float64)
    for v in range(n_variants):
        m = float(mean[v])
        s = float(scale[v])
        for i in range(n_samples):
            code = decoded[v, i]
            if code == PLINK_MISSING_INT8:
                continue
            z = (float(code) - m) / s
            y_ref[i] += z * float(x[v])

    # Ensure at least one missing exists (the random buffer should already
    # contain 0b01 codes); make this asserted.
    assert missing_mask.any(), "expected synthetic data to contain missing slots"

    y_got = cpu_ref.cpu_gemv_nt(
        packed, n_samples=n_samples, x=x, mean=mean, scale=scale, count_a1=True
    )
    assert y_got.shape == (n_samples,)
    np.testing.assert_allclose(y_got, y_ref.astype(np.float32), rtol=FP32_RTOL, atol=FP32_ATOL)

    # fp64 path tolerance.
    y_got64 = cpu_ref.cpu_gemv_nt(
        packed,
        n_samples=n_samples,
        x=x.astype(np.float64),
        mean=mean.astype(np.float64),
        scale=scale.astype(np.float64),
        count_a1=True,
    )
    np.testing.assert_allclose(y_got64, y_ref, rtol=FP64_RTOL, atol=FP64_ATOL)


# ---------------------------------------------------------------------------
# Test 3: transpose GEMV
# ---------------------------------------------------------------------------


def test_cpu_gemv_tn_matches_explicit_loop() -> None:
    n_samples = 10
    n_variants = 30
    packed = _random_packed(n_samples, n_variants, seed=4321, zero_padding=False)

    decoded = cpu_ref._decode_packed(packed, n_samples=n_samples, count_a1=True)
    missing_mask = decoded == PLINK_MISSING_INT8
    assert missing_mask.any()

    mean, scale = _mean_scale_from_decoded(decoded)
    rng = np.random.default_rng(11)
    y = rng.standard_normal(n_samples).astype(np.float32)

    g_ref = np.zeros(n_variants, dtype=np.float64)
    for v in range(n_variants):
        m = float(mean[v])
        s = float(scale[v])
        acc = 0.0
        for i in range(n_samples):
            code = decoded[v, i]
            if code == PLINK_MISSING_INT8:
                continue
            z = (float(code) - m) / s
            acc += z * float(y[i])
        g_ref[v] = acc

    g_got = cpu_ref.cpu_gemv_tn(
        packed, n_samples=n_samples, y=y, mean=mean, scale=scale, count_a1=True
    )
    assert g_got.shape == (n_variants,)
    np.testing.assert_allclose(g_got, g_ref.astype(np.float32), rtol=FP32_RTOL, atol=FP32_ATOL)

    g_got64 = cpu_ref.cpu_gemv_tn(
        packed,
        n_samples=n_samples,
        y=y.astype(np.float64),
        mean=mean.astype(np.float64),
        scale=scale.astype(np.float64),
        count_a1=True,
    )
    np.testing.assert_allclose(g_got64, g_ref, rtol=FP64_RTOL, atol=FP64_ATOL)


# ---------------------------------------------------------------------------
# Test 4: gram == Z.T @ Z
# ---------------------------------------------------------------------------


def test_cpu_gemm_gram_matches_zTz() -> None:
    n_samples = 16
    n_variants = 24
    packed = _random_packed(n_samples, n_variants, seed=99, zero_padding=False)

    decoded = cpu_ref._decode_packed(packed, n_samples=n_samples, count_a1=True)
    assert (decoded == PLINK_MISSING_INT8).any()

    mean, scale = _mean_scale_from_decoded(decoded)
    # Build explicit Z of shape (n_samples, n_variants) with missing -> 0.
    Z = np.zeros((n_samples, n_variants), dtype=np.float64)
    for v in range(n_variants):
        m = float(mean[v])
        s = float(scale[v])
        for i in range(n_samples):
            code = decoded[v, i]
            if code == PLINK_MISSING_INT8:
                Z[i, v] = 0.0
            else:
                Z[i, v] = (float(code) - m) / s
    B_ref = Z.T @ Z

    B_got = cpu_ref.cpu_gemm_gram(
        packed,
        n_samples=n_samples,
        mean=mean.astype(np.float64),
        scale=scale.astype(np.float64),
        count_a1=True,
    )
    assert B_got.shape == (n_variants, n_variants)
    np.testing.assert_allclose(B_got, B_ref, rtol=FP64_RTOL, atol=FP64_ATOL)


# ---------------------------------------------------------------------------
# Test 5: screen reductions
# ---------------------------------------------------------------------------


def test_cpu_screen_count_sum_sumsq() -> None:
    n_samples = 20
    n_variants = 40
    packed = _random_packed(n_samples, n_variants, seed=2025, zero_padding=False)

    decoded = cpu_ref._decode_packed(packed, n_samples=n_samples, count_a1=True)
    missing_mask = decoded == PLINK_MISSING_INT8
    raw = decoded.astype(np.float64)
    raw_nm = np.where(missing_mask, 0.0, raw)
    observed_mask = (~missing_mask).astype(np.float64)

    count_ref = (~missing_mask).sum(axis=1).astype(np.int32)
    sum_ref = raw_nm.sum(axis=1).astype(np.float64)
    sumsq_ref = (raw_nm * raw_nm).sum(axis=1).astype(np.float64)

    rng = np.random.default_rng(3)
    rhs = rng.standard_normal(n_samples).astype(np.float64)
    # New contract: screen returns two per-variant reductions against `rhs`:
    #   dosage_rhs[v]   = Σ_{observed i} raw_dosage[i,v] * rhs[i]
    #   observed_rhs[v] = Σ_{observed i} rhs[i]      (i.e. observed-mask dot rhs)
    dosage_rhs_ref = (raw_nm * rhs[None, :]).sum(axis=1).astype(np.float64)
    observed_rhs_ref = (observed_mask * rhs[None, :]).sum(axis=1).astype(np.float64)

    out = cpu_ref.cpu_screen(
        packed, n_samples=n_samples, rhs=rhs, count_a1=True
    )
    np.testing.assert_array_equal(out["count"], count_ref)
    np.testing.assert_allclose(out["sum"], sum_ref, rtol=FP64_RTOL, atol=FP64_ATOL)
    np.testing.assert_allclose(out["sumsq"], sumsq_ref, rtol=FP64_RTOL, atol=FP64_ATOL)
    np.testing.assert_allclose(
        out["dosage_rhs"], dosage_rhs_ref, rtol=FP64_RTOL, atol=FP64_ATOL
    )
    np.testing.assert_allclose(
        out["observed_rhs"], observed_rhs_ref, rtol=FP64_RTOL, atol=FP64_ATOL
    )

    # Without rhs, those keys must be absent.
    out_no_rhs = cpu_ref.cpu_screen(packed, n_samples=n_samples, count_a1=True)
    assert "dosage_rhs" not in out_no_rhs
    assert "observed_rhs" not in out_no_rhs


def test_cpu_screen_y_resid_alias_emits_deprecation() -> None:
    """Legacy ``y_resid=`` kwarg should still work but emit DeprecationWarning."""
    n_samples = 16
    n_variants = 8
    packed = _random_packed(n_samples, n_variants, seed=7, zero_padding=False)
    rhs = np.random.default_rng(0).standard_normal(n_samples).astype(np.float64)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out_legacy = cpu_ref.cpu_screen(
            packed, n_samples=n_samples, y_resid=rhs, count_a1=True
        )
        assert any(
            issubclass(w.category, DeprecationWarning) for w in caught
        ), "expected DeprecationWarning when using y_resid= alias"

    out_new = cpu_ref.cpu_screen(
        packed, n_samples=n_samples, rhs=rhs, count_a1=True
    )
    np.testing.assert_allclose(out_legacy["dosage_rhs"], out_new["dosage_rhs"])
    np.testing.assert_allclose(out_legacy["observed_rhs"], out_new["observed_rhs"])


# ---------------------------------------------------------------------------
# Test 6: trailing pad samples must not contribute
# ---------------------------------------------------------------------------


def test_padding_samples_ignored() -> None:
    n_samples = 97  # 97 % 4 == 1 -> 3 garbage trailing slots in the last byte.
    n_variants = 8
    bpv = _bytes_per_variant(n_samples)
    assert bpv * 4 - n_samples == 3

    rng = np.random.default_rng(2024)
    packed = rng.integers(0, 256, size=(n_variants, bpv), dtype=np.uint8)
    # Make the LAST byte of every variant 0xFF. Under count_a1=True, 0xFF
    # decodes to four 0b11 codes -> raw dosage 0 for the kept slot, 0 for
    # the 3 trailing pad slots. To meaningfully verify garbage gets dropped,
    # use 0x00 instead, which decodes to four 2's. Then the in-range slot
    # contributes +2 while the 3 pad slots, if leaked, would each add +2.
    packed[:, -1] = 0x00  # this is the "garbage" pattern in our test

    decoded = cpu_ref._decode_packed(packed, n_samples=n_samples, count_a1=True)
    assert decoded.shape == (n_variants, n_samples)

    # Build a reference WITHOUT pad slots: re-decode with full padded width
    # via the canonical LUT, then trim to n_samples — this is exactly what
    # _decode_payload does.
    ref = _decode_payload(
        packed.tobytes(),
        iid_count=n_samples,
        variant_count=n_variants,
        bytes_per_variant=bpv,
        count_a1=True,
    ).T  # (V, N)
    np.testing.assert_array_equal(decoded, ref)

    # Now verify the reductions: counts/sums/sumsq must NOT include the 3
    # trailing 0b00 -> dosage 2 garbage slots.
    out = cpu_ref.cpu_screen(packed, n_samples=n_samples, count_a1=True)
    miss = decoded == PLINK_MISSING_INT8
    raw_nm = np.where(miss, 0.0, decoded.astype(np.float64))
    np.testing.assert_array_equal(out["count"], (~miss).sum(axis=1).astype(np.int32))
    np.testing.assert_allclose(out["sum"], raw_nm.sum(axis=1))
    np.testing.assert_allclose(out["sumsq"], (raw_nm * raw_nm).sum(axis=1))

    # Counterfactual check: a faulty implementation that included the 3
    # trailing slots (each = dosage 2) would report sum increased by 6.
    full_decoded = np.empty((n_variants, bpv * 4), dtype=np.int8)
    # Manual unpack including pad.
    for v in range(n_variants):
        for b in range(bpv):
            byte = int(packed[v, b])
            for k in range(4):
                code = (byte >> (2 * k)) & 0b11
                full_decoded[v, b * 4 + k] = [2, PLINK_MISSING_INT8, 1, 0][code]
    full_miss = full_decoded == PLINK_MISSING_INT8
    full_raw_nm = np.where(full_miss, 0.0, full_decoded.astype(np.float64))
    leaked_sum = full_raw_nm.sum(axis=1)
    # The padded version must be strictly larger (or equal) per variant; here
    # the 3 trailing slots each add 2 -> +6 per variant.
    assert np.all(leaked_sum - out["sum"] == 6.0)


# ---------------------------------------------------------------------------
# Test 7: count_a1=False inverts homozygous codes
# ---------------------------------------------------------------------------


def test_count_a1_false_inverts_codes() -> None:
    # Hand-built packed buffer: a single variant, 4 samples, codes
    #   sample 0 = 0b00, sample 1 = 0b11, sample 2 = 0b10, sample 3 = 0b01.
    # Byte layout: (s3<<6) | (s2<<4) | (s1<<2) | s0
    #            = (0b01<<6) | (0b10<<4) | (0b11<<2) | 0b00
    #            = 0x40 | 0x20 | 0x0C | 0x00 = 0x6C
    byte_val = (0b01 << 6) | (0b10 << 4) | (0b11 << 2) | 0b00
    assert byte_val == 0x6C
    packed = np.array([[byte_val]], dtype=np.uint8)
    n_samples = 4

    # count_a1=True: 0b00->2, 0b11->0, 0b10->1, 0b01->missing.
    got_a1 = cpu_ref._decode_packed(packed, n_samples=n_samples, count_a1=True)
    expected_a1 = np.array([[2, 0, 1, PLINK_MISSING_INT8]], dtype=np.int8)
    np.testing.assert_array_equal(got_a1, expected_a1)

    # count_a1=False: 0b00->0, 0b11->2, 0b10->1, 0b01->missing.
    got_a2 = cpu_ref._decode_packed(packed, n_samples=n_samples, count_a1=False)
    expected_a2 = np.array([[0, 2, 1, PLINK_MISSING_INT8]], dtype=np.int8)
    np.testing.assert_array_equal(got_a2, expected_a2)

    # Cross-check against canonical _decode_payload too.
    bpv = _bytes_per_variant(n_samples)
    ref_a1 = _decode_payload(
        packed.tobytes(),
        iid_count=n_samples,
        variant_count=1,
        bytes_per_variant=bpv,
        count_a1=True,
    ).T
    ref_a2 = _decode_payload(
        packed.tobytes(),
        iid_count=n_samples,
        variant_count=1,
        bytes_per_variant=bpv,
        count_a1=False,
    ).T
    np.testing.assert_array_equal(got_a1, ref_a1)
    np.testing.assert_array_equal(got_a2, ref_a2)


# ---------------------------------------------------------------------------
# Helper: pack arbitrary integer dosages (with optional missing flag) into
# the canonical PLINK 1.9 bitpacked layout under count_a1=True.
# ---------------------------------------------------------------------------


def _pack_dosages_a1(dosages: np.ndarray, missing_mask: np.ndarray) -> np.ndarray:
    """Pack (n_variants, n_samples) int dosages {0,1,2} + missing mask into
    (n_variants, bytes_per_variant) uint8. Trailing pad slots set to 0b00.
    count_a1=True: 0->0b11, 1->0b10, 2->0b00, missing->0b01."""
    n_variants, n_samples = dosages.shape
    bpv = _bytes_per_variant(n_samples)
    packed = np.zeros((n_variants, bpv), dtype=np.uint8)
    dose_to_code = {0: 0b11, 1: 0b10, 2: 0b00}
    for v in range(n_variants):
        for i in range(n_samples):
            code = 0b01 if missing_mask[v, i] else dose_to_code[int(dosages[v, i])]
            packed[v, i // 4] |= np.uint8(code << (2 * (i % 4)))
    return packed


# ---------------------------------------------------------------------------
# Test 8: diag(Z.T @ Z) == N under NO missing and NO low-variance floor
# (this is the load-bearing spec invariant — Σ_i z[i,v]^2 == N exactly).
# ---------------------------------------------------------------------------


def test_cpu_gemm_gram_diag_equals_N_no_missing_no_floor() -> None:
    n_samples = 24
    n_variants = 12
    rng = np.random.default_rng(0xABCDEF)
    # No missing; no constant columns. Force every column to contain at least
    # one 0 and one 2 so scale_raw is comfortably above any floor.
    dosages = rng.integers(0, 3, size=(n_variants, n_samples), dtype=np.int8)
    dosages[:, 0] = 0
    dosages[:, 1] = 2
    missing_mask = np.zeros((n_variants, n_samples), dtype=bool)
    packed = _pack_dosages_a1(dosages.astype(np.int64), missing_mask)

    decoded = cpu_ref._decode_packed(packed, n_samples=n_samples, count_a1=True)
    assert not (decoded == PLINK_MISSING_INT8).any()

    mean, scale = _mean_scale_from_decoded(decoded)
    # Sanity: no scale was floored (the floor is 1e-6; scales here are O(1)).
    assert np.all(scale > 1e-3)

    B = cpu_ref.cpu_gemm_gram(
        packed,
        n_samples=n_samples,
        mean=mean.astype(np.float64),
        scale=scale.astype(np.float64),
        count_a1=True,
    )
    diag = np.diag(B)
    # Spec invariant: diag == N exactly (up to fp64 round-off).
    np.testing.assert_allclose(diag, np.full(n_variants, n_samples, dtype=np.float64),
                               rtol=1e-12, atol=1e-9)


# ---------------------------------------------------------------------------
# Test 9: low-variance floor keeps the actual sample mean (mean != 0),
# and scale==1.0.
# ---------------------------------------------------------------------------


def test_cpu_low_variance_floor_keeps_actual_mean() -> None:
    n_samples = 32
    n_variants = 1
    # Column is almost constant: 31 zeros and 1 single dosage=1.
    # css = sumsq - sum^2/count = 1 - 1/32 ≈ 0.969;
    # scale_raw = sqrt(css/N) ≈ sqrt(0.969/32) ≈ 0.174  → NOT floored.
    # We need a *truly* near-constant column to trigger the floor. Use
    # a single 2 among 31 zeros at a much larger N to push scale_raw below
    # the minimum_scale=1e-6 threshold? Easier: use all-zero except a single
    # 2 with a HUGE n_samples is impractical here.
    #
    # Instead, build a strictly constant column (all zeros, no missing).
    # Then sum=0, sumsq=0, css=0 -> scale_raw=0 < 1e-6 → floored.
    # Actual sample mean = 0.0 (which is also the trivial mean), so we can't
    # distinguish "kept actual mean" from "zeroed mean". Use all-ones instead:
    # constant column of dosage=1. sum=N, sumsq=N, css=N - N^2/N = 0,
    # scale_raw = 0 < 1e-6 → floored; actual mean = 1.0 (NOT 0).
    dosages = np.ones((n_variants, n_samples), dtype=np.int64)
    missing_mask = np.zeros((n_variants, n_samples), dtype=bool)
    packed = _pack_dosages_a1(dosages, missing_mask)

    decoded = cpu_ref._decode_packed(packed, n_samples=n_samples, count_a1=True)
    mean, scale = _mean_scale_from_decoded(decoded)
    # Floored: scale should be exactly 1.0, mean should be the actual sample
    # mean (1.0) — NOT zero.
    assert scale.shape == (1,)
    np.testing.assert_allclose(scale, [1.0], rtol=0, atol=0)
    np.testing.assert_allclose(mean, [1.0], rtol=0, atol=0)
    # Sanity: every decoded value is dosage 1.
    assert np.all(decoded == 1)


# ---------------------------------------------------------------------------
# Test 10: legacy ``std=`` kwarg alias still works (with DeprecationWarning)
# for the standardization-consuming CPU helpers.
# ---------------------------------------------------------------------------


def test_cpu_std_alias_emits_deprecation() -> None:
    n_samples = 12
    n_variants = 6
    packed = _random_packed(n_samples, n_variants, seed=314, zero_padding=False)
    decoded = cpu_ref._decode_packed(packed, n_samples=n_samples, count_a1=True)
    mean, scale = _mean_scale_from_decoded(decoded)
    x = np.random.default_rng(0).standard_normal(n_variants).astype(np.float32)
    y = np.random.default_rng(1).standard_normal(n_samples).astype(np.float32)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        y_legacy = cpu_ref.cpu_gemv_nt(
            packed, n_samples=n_samples, x=x, mean=mean, std=scale, count_a1=True
        )
        assert any(
            issubclass(w.category, DeprecationWarning) for w in caught
        ), "expected DeprecationWarning for cpu_gemv_nt(std=...)"

    y_new = cpu_ref.cpu_gemv_nt(
        packed, n_samples=n_samples, x=x, mean=mean, scale=scale, count_a1=True
    )
    np.testing.assert_allclose(y_legacy, y_new, rtol=1e-6, atol=1e-6)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g_legacy = cpu_ref.cpu_gemv_tn(
            packed, n_samples=n_samples, y=y, mean=mean, std=scale, count_a1=True
        )
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)

    g_new = cpu_ref.cpu_gemv_tn(
        packed, n_samples=n_samples, y=y, mean=mean, scale=scale, count_a1=True
    )
    np.testing.assert_allclose(g_legacy, g_new, rtol=1e-6, atol=1e-6)

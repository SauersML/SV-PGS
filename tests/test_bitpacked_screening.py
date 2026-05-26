from __future__ import annotations

import warnings

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
screening = pytest.importorskip("sv_pgs.bitpacked.screening")

from sv_pgs.bitpacked.cpu_reference import cpu_screen  # noqa: E402


def _bytes_per_variant(n_samples: int) -> int:
    return (n_samples + 3) // 4


def _pack(dosages_or_missing: np.ndarray, count_a1: bool = True) -> np.ndarray:
    """Pack a (n_variants, n_samples) array containing values in {0, 1, 2, -1=missing}
    into PLINK 1.9 bitpacked (n_variants, bytes_per_variant) uint8.
    """
    n_variants, n_samples = dosages_or_missing.shape
    bpv = _bytes_per_variant(n_samples)
    packed = np.zeros((n_variants, bpv), dtype=np.uint8)
    # Map dosage -> 2-bit code (count_a1=True): 0b00=2,0b01=missing,0b10=1,0b11=0
    if count_a1:
        code_map = {2: 0b00, -1: 0b01, 1: 0b10, 0: 0b11}
    else:
        code_map = {0: 0b00, -1: 0b01, 1: 0b10, 2: 0b11}
    for v in range(n_variants):
        for i in range(n_samples):
            d = int(dosages_or_missing[v, i])
            code = code_map[d]
            byte_idx = i // 4
            slot = i % 4
            packed[v, byte_idx] |= code << (2 * slot)
        # Trailing padding slots in last byte are left as 0b00 by default — but
        # PLINK convention pads with 0b00 (homozygous A1) which DOES contribute
        # to counts. We must pad with something the loader would have. PLINK
        # 1.9 in fact pads with 0b00 and the loader masks via n_samples. The
        # kernel under test must respect n_samples and ignore trailing slots.
    return packed


def _rand_packed(rng: np.random.Generator, n_variants: int, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (packed, raw_dosage_with_missing) where missing is -1."""
    # Sample codes in {0,1,2,3} -> map to dosage {2, missing, 1, 0}
    codes = rng.integers(0, 4, size=(n_variants, n_samples), dtype=np.int8)
    dosage = np.where(
        codes == 0, 2, np.where(codes == 1, -1, np.where(codes == 2, 1, 0))
    ).astype(np.int8)
    packed = _pack(dosage, count_a1=True)
    return packed, dosage


def _allocate_outputs(n_variants: int, with_rhs: bool = False):
    out_count = cp.zeros(n_variants, dtype=cp.int32)
    out_sum = cp.zeros(n_variants, dtype=cp.float64)
    out_sumsq = cp.zeros(n_variants, dtype=cp.float64)
    if with_rhs:
        out_dosage_rhs = cp.zeros(n_variants, dtype=cp.float64)
        out_observed_rhs = cp.zeros(n_variants, dtype=cp.float64)
    else:
        out_dosage_rhs = None
        out_observed_rhs = None
    return out_count, out_sum, out_sumsq, out_dosage_rhs, out_observed_rhs


def test_screen_count_matches_cpu_reference():
    rng = np.random.default_rng(0)
    n_samples, n_variants = 100, 500
    packed, _ = _rand_packed(rng, n_variants, n_samples)
    ref = cpu_screen(packed, n_samples, rhs=None, count_a1=True)

    packed_dev = cp.asarray(packed)
    out_count, out_sum, out_sumsq, _, _ = _allocate_outputs(n_variants)
    screening.screen(
        packed_dev,
        n_samples,
        out_count,
        out_sum,
        out_sumsq,
        rhs=None,
        out_dosage_rhs=None,
        out_observed_rhs=None,
        count_a1=True,
    )
    np.testing.assert_array_equal(cp.asnumpy(out_count), ref["count"].astype(np.int32))


def test_screen_sum_sumsq_matches_cpu_reference():
    rng = np.random.default_rng(1)
    n_samples, n_variants = 100, 500
    packed, _ = _rand_packed(rng, n_variants, n_samples)
    ref = cpu_screen(packed, n_samples, rhs=None, count_a1=True)

    packed_dev = cp.asarray(packed)
    out_count, out_sum, out_sumsq, _, _ = _allocate_outputs(n_variants)
    screening.screen(
        packed_dev,
        n_samples,
        out_count,
        out_sum,
        out_sumsq,
        rhs=None,
        out_dosage_rhs=None,
        out_observed_rhs=None,
        count_a1=True,
    )
    np.testing.assert_allclose(cp.asnumpy(out_sum), ref["sum"], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(cp.asnumpy(out_sumsq), ref["sumsq"], rtol=1e-9, atol=1e-9)


def test_screen_rhs_reductions_match_cpu_reference():
    rng = np.random.default_rng(2)
    n_samples, n_variants = 100, 500
    packed, _ = _rand_packed(rng, n_variants, n_samples)
    rhs = rng.standard_normal(n_samples).astype(np.float64)
    ref = cpu_screen(packed, n_samples, rhs=rhs, count_a1=True)

    packed_dev = cp.asarray(packed)
    rhs_dev = cp.asarray(rhs)
    (
        out_count,
        out_sum,
        out_sumsq,
        out_dosage_rhs,
        out_observed_rhs,
    ) = _allocate_outputs(n_variants, with_rhs=True)
    screening.screen(
        packed_dev,
        n_samples,
        out_count,
        out_sum,
        out_sumsq,
        rhs=rhs_dev,
        out_dosage_rhs=out_dosage_rhs,
        out_observed_rhs=out_observed_rhs,
        count_a1=True,
    )
    np.testing.assert_allclose(
        cp.asnumpy(out_dosage_rhs), ref["dosage_rhs"], rtol=1e-9, atol=1e-9
    )
    np.testing.assert_allclose(
        cp.asnumpy(out_observed_rhs), ref["observed_rhs"], rtol=1e-9, atol=1e-9
    )


def test_screen_y_resid_alias_deprecated():
    """Legacy ``y_resid=`` / ``out_y_dot=`` kwargs must still work with DeprecationWarning."""
    rng = np.random.default_rng(202)
    n_samples, n_variants = 64, 128
    packed, _ = _rand_packed(rng, n_variants, n_samples)
    rhs = rng.standard_normal(n_samples).astype(np.float64)

    packed_dev = cp.asarray(packed)
    rhs_dev = cp.asarray(rhs)
    out_count = cp.zeros(n_variants, dtype=cp.int32)
    out_sum = cp.zeros(n_variants, dtype=cp.float64)
    out_sumsq = cp.zeros(n_variants, dtype=cp.float64)
    out_y_dot = cp.zeros(n_variants, dtype=cp.float64)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        screening.screen(
            packed_dev,
            n_samples,
            out_count,
            out_sum,
            out_sumsq,
            y_resid=rhs_dev,
            out_y_dot=out_y_dot,
            count_a1=True,
        )
        assert any(issubclass(w.category, DeprecationWarning) for w in caught), (
            "expected DeprecationWarning for screening.screen(y_resid=..., out_y_dot=...)"
        )

    # Compare against the new-name path: out_y_dot under the legacy alias
    # corresponds to dosage_rhs in the new contract.
    out_dosage_rhs_new = cp.zeros(n_variants, dtype=cp.float64)
    out_observed_rhs_new = cp.zeros(n_variants, dtype=cp.float64)
    out_count_new = cp.zeros(n_variants, dtype=cp.int32)
    out_sum_new = cp.zeros(n_variants, dtype=cp.float64)
    out_sumsq_new = cp.zeros(n_variants, dtype=cp.float64)
    screening.screen(
        packed_dev,
        n_samples,
        out_count_new,
        out_sum_new,
        out_sumsq_new,
        rhs=rhs_dev,
        out_dosage_rhs=out_dosage_rhs_new,
        out_observed_rhs=out_observed_rhs_new,
        count_a1=True,
    )
    np.testing.assert_allclose(
        cp.asnumpy(out_y_dot), cp.asnumpy(out_dosage_rhs_new), rtol=1e-12, atol=1e-12
    )


def test_screen_padding_samples_excluded():
    rng = np.random.default_rng(3)
    n_samples, n_variants = 97, 64  # 97 % 4 == 1 -> 3 padding slots in last byte
    packed, _ = _rand_packed(rng, n_variants, n_samples)

    # Manually corrupt trailing padding slots in the last byte to non-zero codes.
    bpv = _bytes_per_variant(n_samples)
    last_byte_used_slots = n_samples - 4 * (bpv - 1)  # = 1
    # Fill slots [last_byte_used_slots .. 4) with 0b00 (which would otherwise add
    # dosage 2 if NOT masked). Then also try 0b10 (dosage 1).
    for v in range(n_variants):
        b = packed[v, bpv - 1]
        # Clear high (4 - last_byte_used_slots) slots
        keep_mask = (1 << (2 * last_byte_used_slots)) - 1
        b &= keep_mask
        # Inject non-zero codes into trailing slots that MUST be ignored
        for slot in range(last_byte_used_slots, 4):
            b |= (0b10 << (2 * slot))  # dosage=1 if not masked
        packed[v, bpv - 1] = b

    ref = cpu_screen(packed, n_samples, rhs=None, count_a1=True)

    packed_dev = cp.asarray(packed)
    out_count, out_sum, out_sumsq, _, _ = _allocate_outputs(n_variants)
    screening.screen(
        packed_dev,
        n_samples,
        out_count,
        out_sum,
        out_sumsq,
        rhs=None,
        out_dosage_rhs=None,
        out_observed_rhs=None,
        count_a1=True,
    )
    np.testing.assert_array_equal(cp.asnumpy(out_count), ref["count"].astype(np.int32))
    np.testing.assert_allclose(cp.asnumpy(out_sum), ref["sum"], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(cp.asnumpy(out_sumsq), ref["sumsq"], rtol=1e-9, atol=1e-9)
    # Sanity: counts cannot exceed n_samples
    assert int(cp.asnumpy(out_count).max()) <= n_samples


def test_screen_accumulates_into_out():
    rng = np.random.default_rng(4)
    n_samples, n_variants = 100, 200
    packed, _ = _rand_packed(rng, n_variants, n_samples)
    rhs = rng.standard_normal(n_samples).astype(np.float64)

    packed_dev = cp.asarray(packed)
    rhs_dev = cp.asarray(rhs)
    (
        out_count,
        out_sum,
        out_sumsq,
        out_dosage_rhs,
        out_observed_rhs,
    ) = _allocate_outputs(n_variants, with_rhs=True)

    screening.screen(
        packed_dev, n_samples, out_count, out_sum, out_sumsq,
        rhs=rhs_dev, out_dosage_rhs=out_dosage_rhs,
        out_observed_rhs=out_observed_rhs, count_a1=True,
    )
    c1 = cp.asnumpy(out_count).copy()
    s1 = cp.asnumpy(out_sum).copy()
    ss1 = cp.asnumpy(out_sumsq).copy()
    drhs1 = cp.asnumpy(out_dosage_rhs).copy()
    orhs1 = cp.asnumpy(out_observed_rhs).copy()

    screening.screen(
        packed_dev, n_samples, out_count, out_sum, out_sumsq,
        rhs=rhs_dev, out_dosage_rhs=out_dosage_rhs,
        out_observed_rhs=out_observed_rhs, count_a1=True,
    )
    np.testing.assert_array_equal(cp.asnumpy(out_count), 2 * c1)
    np.testing.assert_allclose(cp.asnumpy(out_sum), 2 * s1, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(cp.asnumpy(out_sumsq), 2 * ss1, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(cp.asnumpy(out_dosage_rhs), 2 * drhs1, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(cp.asnumpy(out_observed_rhs), 2 * orhs1, rtol=1e-9, atol=1e-9)


def test_screen_count_a1_false():
    rng = np.random.default_rng(5)
    n_samples, n_variants = 100, 300
    packed, _ = _rand_packed(rng, n_variants, n_samples)

    ref = cpu_screen(packed, n_samples, rhs=None, count_a1=False)

    packed_dev = cp.asarray(packed)
    out_count, out_sum, out_sumsq, _, _ = _allocate_outputs(n_variants)
    screening.screen(
        packed_dev,
        n_samples,
        out_count,
        out_sum,
        out_sumsq,
        rhs=None,
        out_dosage_rhs=None,
        out_observed_rhs=None,
        count_a1=False,
    )
    np.testing.assert_array_equal(cp.asnumpy(out_count), ref["count"].astype(np.int32))
    np.testing.assert_allclose(cp.asnumpy(out_sum), ref["sum"], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(cp.asnumpy(out_sumsq), ref["sumsq"], rtol=1e-9, atol=1e-9)


def test_screen_all_missing_column():
    n_samples, n_variants = 100, 4
    bpv = _bytes_per_variant(n_samples)
    # Every sample slot = 0b01 (missing) => each byte = 0b01010101 = 0x55
    packed = np.full((n_variants, bpv), 0x55, dtype=np.uint8)
    rhs = np.random.default_rng(6).standard_normal(n_samples).astype(np.float64)

    packed_dev = cp.asarray(packed)
    rhs_dev = cp.asarray(rhs)
    (
        out_count,
        out_sum,
        out_sumsq,
        out_dosage_rhs,
        out_observed_rhs,
    ) = _allocate_outputs(n_variants, with_rhs=True)
    screening.screen(
        packed_dev, n_samples, out_count, out_sum, out_sumsq,
        rhs=rhs_dev, out_dosage_rhs=out_dosage_rhs,
        out_observed_rhs=out_observed_rhs, count_a1=True,
    )
    assert int(cp.asnumpy(out_count).sum()) == 0
    np.testing.assert_array_equal(cp.asnumpy(out_count), np.zeros(n_variants, dtype=np.int32))
    np.testing.assert_allclose(cp.asnumpy(out_sum), np.zeros(n_variants), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(cp.asnumpy(out_sumsq), np.zeros(n_variants), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(cp.asnumpy(out_dosage_rhs), np.zeros(n_variants), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(cp.asnumpy(out_observed_rhs), np.zeros(n_variants), rtol=1e-9, atol=1e-9)


def test_finalize_standardized_rhs_matches_numpy():
    """`finalize_standardized_rhs(...)` combines per-variant reductions into the
    final standardized G_v . r vector. Compare against an explicit numpy
    reconstruction built from the CPU reference's per-variant means/scales.
    """
    rng = np.random.default_rng(404)
    n_samples, n_variants = 80, 64
    packed, _ = _rand_packed(rng, n_variants, n_samples)
    rhs = rng.standard_normal(n_samples).astype(np.float64)

    # Build per-variant mean/scale per spec on CPU.
    from sv_pgs.bitpacked.cpu_reference import _decode_packed
    decoded = _decode_packed(packed, n_samples=n_samples, count_a1=True)
    PLINK_MISSING_INT8 = -127
    miss = decoded == PLINK_MISSING_INT8
    count = (~miss).sum(axis=1).astype(np.float64)
    raw_nm = np.where(miss, 0.0, decoded.astype(np.float64))
    sums = raw_nm.sum(axis=1)
    sumsq = (raw_nm * raw_nm).sum(axis=1)
    safe_count = np.where(count > 0, count, 1.0)
    mean = np.where(count > 0, sums / safe_count, 0.0)
    css = np.maximum(sumsq - sums * sums / safe_count, 0.0)
    scale_raw = np.sqrt(css / max(n_samples, 1))
    minimum_scale = 1e-6
    low_var = scale_raw < minimum_scale
    scale = np.where(count > 0, np.where(low_var, 1.0, scale_raw), 1.0)

    # Per-variant raw reductions.
    ref = cpu_screen(packed, n_samples, rhs=rhs, count_a1=True)
    dosage_rhs = ref["dosage_rhs"]
    observed_rhs = ref["observed_rhs"]

    # Final standardized inner product:
    #   sum_{observed i} ((dosage[i,v] - mean[v]) / scale[v]) * rhs[i]
    # = (dosage_rhs[v] - mean[v] * observed_rhs[v]) / scale[v]
    final_ref = (dosage_rhs - mean * observed_rhs) / scale

    out_dev = screening.finalize_standardized_rhs(
        dosage_rhs=cp.asarray(dosage_rhs),
        observed_rhs=cp.asarray(observed_rhs),
        mean=cp.asarray(mean.astype(np.float32)),
        scale=cp.asarray(scale.astype(np.float32)),
    )
    np.testing.assert_allclose(cp.asnumpy(out_dev), final_ref, rtol=1e-5, atol=1e-5)

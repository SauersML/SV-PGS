from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sv_pgs.bitpacked_matrix")

from sv_pgs.bitpacked_matrix import BitpackedDeviceMatrix  # noqa: E402


# ---------- helpers ----------------------------------------------------------


def _bytes_per_variant(n_samples: int) -> int:
    return (n_samples + 3) // 4


def _pack_dosages(dosages: np.ndarray, count_a1: bool = True) -> np.ndarray:
    """Pack a (n_variants, n_samples) int array of dosages (0,1,2 or -1 for missing)
    into PLINK 1.9 BED two-bit codes (count_a1 semantics)."""
    n_variants, n_samples = dosages.shape
    bpv = _bytes_per_variant(n_samples)
    packed = np.zeros((n_variants, bpv), dtype=np.uint8)
    for v in range(n_variants):
        for i in range(n_samples):
            d = int(dosages[v, i])
            if count_a1:
                if d == 2:
                    code = 0b00
                elif d == -1:
                    code = 0b01
                elif d == 1:
                    code = 0b10
                elif d == 0:
                    code = 0b11
                else:
                    raise ValueError(d)
            else:
                if d == 0:
                    code = 0b00
                elif d == -1:
                    code = 0b01
                elif d == 1:
                    code = 0b10
                elif d == 2:
                    code = 0b11
                else:
                    raise ValueError(d)
            byte_index = i // 4
            slot = i % 4
            packed[v, byte_index] |= code << (2 * slot)
    return packed


def _make_synthetic(n_samples: int = 37, n_variants: int = 11, seed: int = 0):
    """Build a small synthetic dataset including missing codes and trailing padding."""
    rng = np.random.default_rng(seed)
    dosages = rng.integers(0, 3, size=(n_variants, n_samples)).astype(np.int64)
    # sprinkle missing
    mask = rng.random((n_variants, n_samples)) < 0.1
    dosages[mask] = -1
    packed_np = _pack_dosages(dosages, count_a1=True)

    # Compute per-variant mean/std over non-missing
    raw = dosages.astype(np.float64)
    raw_missing = dosages == -1
    raw_for_stats = np.where(raw_missing, np.nan, raw)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(raw_for_stats, axis=1)
        std = np.nanstd(raw_for_stats, axis=1)
    mean = np.where(np.isnan(mean), 0.0, mean).astype(np.float32)
    std = np.where((std == 0) | np.isnan(std), 1.0, std).astype(np.float32)
    return dosages, packed_np, mean, std


def _build_matrix(n_samples: int = 37, n_variants: int = 11, seed: int = 0):
    cp = pytest.importorskip("cupy")
    dosages, packed_np, mean, std = _make_synthetic(n_samples, n_variants, seed)
    packed = cp.asarray(packed_np)
    mean_d = cp.asarray(mean)
    std_d = cp.asarray(std)
    mat = BitpackedDeviceMatrix(
        packed=packed,
        mean=mean_d,
        std=std_d,
        n_samples=n_samples,
        count_a1=True,
    )
    return mat, dosages, packed_np, mean, std


# ---------- tests ------------------------------------------------------------


def test_shape_and_dtype():
    cp = pytest.importorskip("cupy")
    n_samples, n_variants = 37, 11
    _, packed_np, mean, std = _make_synthetic(n_samples, n_variants)
    packed = cp.asarray(packed_np)
    mat = BitpackedDeviceMatrix(
        packed=packed,
        mean=cp.asarray(mean),
        std=cp.asarray(std),
        n_samples=n_samples,
        count_a1=True,
    )
    assert mat.shape == (n_samples, n_variants)
    assert mat.dtype == cp.dtype("float32")


def test_matvec_matches_cpu_reference():
    pytest.importorskip("cupy")
    from sv_pgs.bitpacked.cpu_reference import cpu_gemv_nt

    mat, _, packed_np, mean, std = _build_matrix()
    rng = np.random.default_rng(1)
    x = rng.standard_normal(mat.shape[1]).astype(np.float32)
    y_dev = mat.matvec_numpy(x)
    y_ref = cpu_gemv_nt(packed_np, mat.n_samples, x, mean, std, count_a1=True)
    np.testing.assert_allclose(y_dev, y_ref, rtol=1e-4, atol=1e-3)


def test_rmatvec_matches_cpu_reference():
    pytest.importorskip("cupy")
    from sv_pgs.bitpacked.cpu_reference import cpu_gemv_tn

    mat, _, packed_np, mean, std = _build_matrix()
    rng = np.random.default_rng(2)
    y = rng.standard_normal(mat.shape[0]).astype(np.float32)
    g_dev = mat.rmatvec_numpy(y)
    g_ref = cpu_gemv_tn(packed_np, mat.n_samples, y, mean, std, count_a1=True)
    np.testing.assert_allclose(g_dev, g_ref, rtol=1e-4, atol=1e-3)


def test_subset_preserves_alignment():
    cp = pytest.importorskip("cupy")
    mat, _, _, _, _ = _build_matrix()
    indices = np.array([0, 3, 7, 10], dtype=np.int64)
    sub = mat.subset(indices)
    orig_means = mat.column_means()
    sub_means = sub.column_means()
    np.testing.assert_allclose(sub_means, orig_means[indices], rtol=0, atol=0)
    # also confirm shape
    assert sub.shape == (mat.shape[0], indices.size)
    del cp


def test_gram_block_matches_subset_then_full_gram():
    cp = pytest.importorskip("cupy")
    from sv_pgs.bitpacked.cpu_reference import cpu_gemm_gram

    mat, _, packed_np, mean, std = _build_matrix()
    indices = np.array([1, 4, 5, 9], dtype=np.int64)
    block_dev = mat.gram_block(cp.asarray(indices))
    block_host = cp.asnumpy(block_dev) if hasattr(block_dev, "device") else np.asarray(block_dev)

    packed_sub = packed_np[indices]
    mean_sub = mean[indices]
    std_sub = std[indices]
    block_ref = cpu_gemm_gram(packed_sub, mat.n_samples, mean_sub, std_sub, count_a1=True)
    np.testing.assert_allclose(block_host, block_ref, rtol=1e-4, atol=1e-3)


def test_to_host_int8_matches_decode():
    pytest.importorskip("cupy")
    mat, dosages, _, _, _ = _build_matrix()
    host = mat.to_host_int8()
    # matrix is (n_samples, n_variants); dosages is (n_variants, n_samples)
    assert host.shape == (mat.shape[0], mat.shape[1])
    expected = dosages.T.astype(np.int8)
    # missing sentinel
    from sv_pgs.plink import PLINK_MISSING_INT8

    expected = np.where(expected == -1, PLINK_MISSING_INT8, expected).astype(np.int8)
    np.testing.assert_array_equal(host, expected)


def test_close_releases_arrays():
    pytest.importorskip("cupy")
    mat, _, _, _, _ = _build_matrix()
    mat.close()
    # After close, _packed should be None or accessing it should raise.
    packed_attr = getattr(mat, "_packed", "SENTINEL")
    if packed_attr == "SENTINEL":
        # No private attr; try the public one — it should raise or be None.
        with pytest.raises(Exception):
            _ = mat.packed  # noqa: B018
    else:
        assert packed_attr is None or isinstance(packed_attr, type(None))

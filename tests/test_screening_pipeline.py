from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
screening_pipeline = pytest.importorskip("sv_pgs.screening_pipeline")

from sv_pgs.bitpacked.cpu_reference import cpu_screen  # noqa: E402
from sv_pgs.mmap_reader import BedMmapReader  # noqa: E402
from sv_pgs.plink import to_bed  # noqa: E402


def _make_synthetic_bed(
    bed_path: Path,
    n_samples: int,
    n_variants: int,
    seed: int = 0,
) -> np.ndarray:
    """Write a synthetic .bed via sv_pgs.plink.to_bed; return packed bytes."""
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 4, size=(n_samples, n_variants), dtype=np.int8)
    dosage = np.where(
        codes == 0, 2.0, np.where(codes == 1, np.nan, np.where(codes == 2, 1.0, 0.0))
    ).astype(np.float32)
    properties = {
        "iid": [f"s{i}" for i in range(n_samples)],
        "sid": [f"v{j}" for j in range(n_variants)],
    }
    to_bed(bed_path, dosage, properties)
    with BedMmapReader(bed_path, n_samples=n_samples, n_variants=n_variants) as reader:
        return np.array(reader.read_all_packed(), copy=True)


def _to_numpy(arr) -> np.ndarray:
    """Convert a CuPy or NumPy array to NumPy."""
    if hasattr(arr, "get"):
        return arr.get()
    return np.asarray(arr)


def test_screening_pipeline_single_bed():
    with tempfile.TemporaryDirectory() as tmp:
        bed_path = Path(tmp) / "single.bed"
        # n_samples % 4 != 0 to exercise trailing-padding handling.
        n_samples, n_variants = 17, 23
        packed = _make_synthetic_bed(bed_path, n_samples, n_variants, seed=11)

        out = screening_pipeline.run_screening_pass(
            [bed_path],
            [n_samples],
            [n_variants],
        )
        ref = cpu_screen(packed, n_samples, rhs=None, count_a1=True)

        np.testing.assert_array_equal(
            _to_numpy(out["count"]).astype(np.int32), ref["count"].astype(np.int32)
        )
        np.testing.assert_allclose(_to_numpy(out["sum"]), ref["sum"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(_to_numpy(out["sumsq"]), ref["sumsq"], rtol=1e-9, atol=1e-9)


def test_screening_pipeline_two_beds_concat():
    with tempfile.TemporaryDirectory() as tmp:
        bed_a = Path(tmp) / "a.bed"
        bed_b = Path(tmp) / "b.bed"
        n_samples = 13  # % 4 != 0
        n_variants_a, n_variants_b = 19, 7
        packed_a = _make_synthetic_bed(bed_a, n_samples, n_variants_a, seed=21)
        packed_b = _make_synthetic_bed(bed_b, n_samples, n_variants_b, seed=22)

        out = screening_pipeline.run_screening_pass(
            [bed_a, bed_b],
            [n_samples, n_samples],
            [n_variants_a, n_variants_b],
        )
        ref_a = cpu_screen(packed_a, n_samples, rhs=None, count_a1=True)
        ref_b = cpu_screen(packed_b, n_samples, rhs=None, count_a1=True)
        ref_count = np.concatenate([ref_a["count"], ref_b["count"]]).astype(np.int32)
        ref_sum = np.concatenate([ref_a["sum"], ref_b["sum"]])
        ref_sumsq = np.concatenate([ref_a["sumsq"], ref_b["sumsq"]])

        assert _to_numpy(out["count"]).shape == (n_variants_a + n_variants_b,)
        np.testing.assert_array_equal(_to_numpy(out["count"]).astype(np.int32), ref_count)
        np.testing.assert_allclose(_to_numpy(out["sum"]), ref_sum, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(_to_numpy(out["sumsq"]), ref_sumsq, rtol=1e-9, atol=1e-9)


def test_screening_pipeline_with_rhs():
    with tempfile.TemporaryDirectory() as tmp:
        bed_path = Path(tmp) / "with_rhs.bed"
        n_samples, n_variants = 21, 29  # n_samples % 4 != 0
        packed = _make_synthetic_bed(bed_path, n_samples, n_variants, seed=33)

        rng = np.random.default_rng(99)
        rhs = rng.standard_normal(n_samples).astype(np.float64)

        out = screening_pipeline.run_screening_pass(
            [bed_path],
            [n_samples],
            [n_variants],
            rhs=rhs,
        )
        ref = cpu_screen(packed, n_samples, rhs=rhs, count_a1=True)

        assert "dosage_rhs" in out
        assert "observed_rhs" in out
        np.testing.assert_allclose(
            _to_numpy(out["dosage_rhs"]), ref["dosage_rhs"], rtol=1e-9, atol=1e-9
        )
        np.testing.assert_allclose(
            _to_numpy(out["observed_rhs"]), ref["observed_rhs"], rtol=1e-9, atol=1e-9
        )

        # Reconstruct the standardized inner product G_v . r and verify against
        # an explicit numpy build of (Z[:, v] . rhs) using the spec mean/scale.
        from sv_pgs.bitpacked.cpu_reference import _decode_packed
        PLINK_MISSING_INT8 = -127
        decoded = _decode_packed(packed, n_samples=n_samples, count_a1=True)
        miss = decoded == PLINK_MISSING_INT8
        raw_nm = np.where(miss, 0.0, decoded.astype(np.float64))
        count = (~miss).sum(axis=1).astype(np.float64)
        sums = raw_nm.sum(axis=1)
        sumsq = (raw_nm * raw_nm).sum(axis=1)
        safe_count = np.where(count > 0, count, 1.0)
        mean = np.where(count > 0, sums / safe_count, 0.0)
        css = np.maximum(sumsq - sums * sums / safe_count, 0.0)
        scale_raw = np.sqrt(css / max(n_samples, 1))
        minimum_scale = 1e-6
        low_var = scale_raw < minimum_scale
        scale = np.where(count > 0, np.where(low_var, 1.0, scale_raw), 1.0)

        dosage_rhs = _to_numpy(out["dosage_rhs"])
        observed_rhs = _to_numpy(out["observed_rhs"])
        standardized_inner_product = (dosage_rhs - mean * observed_rhs) / scale

        # Explicit Z[:, v] . rhs reconstruction.
        z_dot_rhs = np.zeros(n_variants, dtype=np.float64)
        for v in range(n_variants):
            for i in range(n_samples):
                if miss[v, i]:
                    continue
                z_dot_rhs[v] += (
                    (float(decoded[v, i]) - mean[v]) / scale[v]
                ) * float(rhs[i])
        np.testing.assert_allclose(
            standardized_inner_product, z_dot_rhs, rtol=1e-9, atol=1e-9
        )

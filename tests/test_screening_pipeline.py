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
        ref = cpu_screen(packed, n_samples, y_resid=None, count_a1=True)

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
        ref_a = cpu_screen(packed_a, n_samples, y_resid=None, count_a1=True)
        ref_b = cpu_screen(packed_b, n_samples, y_resid=None, count_a1=True)
        ref_count = np.concatenate([ref_a["count"], ref_b["count"]]).astype(np.int32)
        ref_sum = np.concatenate([ref_a["sum"], ref_b["sum"]])
        ref_sumsq = np.concatenate([ref_a["sumsq"], ref_b["sumsq"]])

        assert _to_numpy(out["count"]).shape == (n_variants_a + n_variants_b,)
        np.testing.assert_array_equal(_to_numpy(out["count"]).astype(np.int32), ref_count)
        np.testing.assert_allclose(_to_numpy(out["sum"]), ref_sum, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(_to_numpy(out["sumsq"]), ref_sumsq, rtol=1e-9, atol=1e-9)


def test_screening_pipeline_with_y_resid():
    with tempfile.TemporaryDirectory() as tmp:
        bed_path = Path(tmp) / "with_y.bed"
        n_samples, n_variants = 21, 29  # n_samples % 4 != 0
        packed = _make_synthetic_bed(bed_path, n_samples, n_variants, seed=33)

        rng = np.random.default_rng(99)
        y_resid = rng.standard_normal(n_samples).astype(np.float64)

        out = screening_pipeline.run_screening_pass(
            [bed_path],
            [n_samples],
            [n_variants],
            y_resid=y_resid,
        )
        ref = cpu_screen(packed, n_samples, y_resid=y_resid, count_a1=True)

        assert "y_dot" in out
        np.testing.assert_allclose(
            _to_numpy(out["y_dot"]), ref["y_dot"], rtol=1e-9, atol=1e-9
        )

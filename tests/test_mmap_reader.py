from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sv_pgs.mmap_reader")

from sv_pgs.mmap_reader import BedMmapReader  # noqa: E402
from sv_pgs.plink import open_bed, to_bed  # noqa: E402


def _make_synthetic_bed(
    bed_path: Path,
    n_samples: int,
    n_variants: int,
    seed: int = 0,
) -> np.ndarray:
    """Write a synthetic BED via sv_pgs.plink.to_bed; return the dosage matrix.

    The dosage matrix has shape (n_samples, n_variants) with values in
    {0, 1, 2, NaN}. NaN exercises the missing-code (0b01) path and the
    chosen n_samples ensures trailing-padding slots when n_samples % 4 != 0.
    """
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
    return dosage


def _decoded_int8_matrix(bed_path: Path, n_samples: int, n_variants: int) -> np.ndarray:
    """Return (n_samples, n_variants) int8 with PLINK_MISSING_INT8 for missing."""
    with open_bed(bed_path, iid_count=n_samples, sid_count=n_variants) as reader:
        return reader.read(dtype="int8", order="C")


def _packed_to_decoded(packed: np.ndarray, n_samples: int) -> np.ndarray:
    """Decode (n_variants, bpv) packed bytes to (n_variants, n_samples) int8 under
    count_a1=True. Missing slot encoded as -127 (matching sv_pgs.plink)."""
    from sv_pgs.plink import _DECODE_LOOKUP_A1  # noqa: PLC0415

    n_variants, bpv = packed.shape
    # Expand each byte into 4 two-bit codes
    bits = np.empty((n_variants, bpv * 4), dtype=np.uint8)
    bits[:, 0::4] = packed & 0b11
    bits[:, 1::4] = (packed >> 2) & 0b11
    bits[:, 2::4] = (packed >> 4) & 0b11
    bits[:, 3::4] = (packed >> 6) & 0b11
    decoded = _DECODE_LOOKUP_A1[bits[:, :n_samples]]
    return decoded


def test_mmap_reader_header_validation():
    with tempfile.TemporaryDirectory() as tmp:
        bad_path = Path(tmp) / "bad.bed"
        # Build a file with the right size but wrong magic.
        n_samples, n_variants = 7, 3
        bpv = (n_samples + 3) // 4
        payload = bytes(n_variants * bpv)
        bad_path.write_bytes(b"\x00\x00\x00" + payload)
        with pytest.raises(ValueError, match="Invalid PLINK 1 .bed header"):
            BedMmapReader(bad_path, n_samples=n_samples, n_variants=n_variants)


def test_mmap_reader_read_all_packed():
    with tempfile.TemporaryDirectory() as tmp:
        bed_path = Path(tmp) / "good.bed"
        # 7 % 4 != 0 exercises trailing padding.
        n_samples, n_variants = 7, 11
        _make_synthetic_bed(bed_path, n_samples, n_variants, seed=42)

        with BedMmapReader(bed_path, n_samples=n_samples, n_variants=n_variants) as reader:
            packed = np.array(reader.read_all_packed(), copy=True)

        assert packed.shape == (n_variants, (n_samples + 3) // 4)
        assert packed.dtype == np.uint8

        # Decode via our local decoder and compare to open_bed.read(dtype='int8').
        decoded_packed = _packed_to_decoded(packed, n_samples)  # (V, N)
        decoded_ref = _decoded_int8_matrix(bed_path, n_samples, n_variants)  # (N, V)
        np.testing.assert_array_equal(decoded_packed.T, decoded_ref)


def test_mmap_reader_read_packed_range():
    with tempfile.TemporaryDirectory() as tmp:
        bed_path = Path(tmp) / "good.bed"
        n_samples, n_variants = 9, 17
        _make_synthetic_bed(bed_path, n_samples, n_variants, seed=7)

        with BedMmapReader(bed_path, n_samples=n_samples, n_variants=n_variants) as reader:
            full = np.array(reader.read_all_packed(), copy=True)
            sub = np.array(reader.read_packed_range(3, 12), copy=True)

        np.testing.assert_array_equal(sub, full[3:12])

        # Bounds: empty slice
        with BedMmapReader(bed_path, n_samples=n_samples, n_variants=n_variants) as reader:
            empty = reader.read_packed_range(5, 5)
        assert empty.shape == (0, (n_samples + 3) // 4)


def test_mmap_reader_read_packed_indexed():
    with tempfile.TemporaryDirectory() as tmp:
        bed_path = Path(tmp) / "good.bed"
        n_samples, n_variants = 10, 20
        _make_synthetic_bed(bed_path, n_samples, n_variants, seed=123)

        indices = np.array([0, 5, 19, 3, 7, 7], dtype=np.int64)
        with BedMmapReader(bed_path, n_samples=n_samples, n_variants=n_variants) as reader:
            full = np.array(reader.read_all_packed(), copy=True)
            with pytest.warns(RuntimeWarning):
                gathered = np.array(reader.read_packed_indexed(indices), copy=True)

        np.testing.assert_array_equal(gathered, full[indices])


def test_mmap_reader_context_manager():
    with tempfile.TemporaryDirectory() as tmp:
        bed_path = Path(tmp) / "good.bed"
        n_samples, n_variants = 5, 4
        _make_synthetic_bed(bed_path, n_samples, n_variants, seed=1)

        reader = BedMmapReader(bed_path, n_samples=n_samples, n_variants=n_variants)
        with reader as r:
            assert r is reader
            # While open, reads succeed.
            _ = r.read_all_packed()
        # After __exit__, the reader is closed: subsequent reads must raise.
        with pytest.raises(RuntimeError, match="closed"):
            reader.read_all_packed()
        # close() is idempotent.
        reader.close()

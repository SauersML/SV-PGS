"""Parallel-vs-single-stream parity for the bitpacked BED range reader.

The reader at module load time picks one of two paths:

  - single-stream sequential ``open_bed._pread_payload`` (the old code path).
  - N-worker ``os.pread``-based range reads dispatched across a thread pool
    (the new ``_read_packed_parallel`` helper).

This test writes a synthetic uint8 blob to a tempfile, runs both paths
against it, and asserts byte-for-byte equality of the resulting buffer.

Pure CPU: no cupy / GPU required.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from sv_pgs.bitpacked_loader import _read_packed_parallel


def _make_blob(tmp_path: Path, *, payload_offset: int, payload_size: int, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    header = b"\x6c\x1b\x01"  # PLINK1 magic; payload offset = 3
    if payload_offset != len(header):
        header = b"\x00" * payload_offset
    payload = rng.integers(0, 256, size=payload_size, dtype=np.uint8).tobytes()
    p = tmp_path / "blob.bed"
    with open(p, "wb") as fh:
        fh.write(header)
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    return p


@pytest.mark.parametrize("n_workers", [1, 2, 3, 4, 8])
def test_parallel_read_matches_sequential_read(tmp_path: Path, n_workers: int) -> None:
    payload_offset = 3
    payload_size = 1024 * 1024 + 17  # not a multiple of n_workers; exercises uneven split
    blob = _make_blob(tmp_path, payload_offset=payload_offset, payload_size=payload_size, seed=42)

    # Ground-truth: read the entire payload via a single fd.
    with open(blob, "rb") as fh:
        fh.seek(payload_offset)
        ground_truth = np.frombuffer(fh.read(payload_size), dtype=np.uint8)

    out = np.zeros((payload_size,), dtype=np.uint8)
    _read_packed_parallel(
        blob,
        payload_offset=payload_offset,
        total_bytes=payload_size,
        pinned_view=out,
        n_workers=n_workers,
    )
    np.testing.assert_array_equal(out, ground_truth)


def test_parallel_read_zero_bytes(tmp_path: Path) -> None:
    blob = _make_blob(tmp_path, payload_offset=3, payload_size=10, seed=0)
    out = np.zeros((0,), dtype=np.uint8)
    # Zero-length read must be a no-op even with N>1 workers.
    _read_packed_parallel(
        blob,
        payload_offset=3,
        total_bytes=0,
        pinned_view=out,
        n_workers=4,
    )
    assert out.shape == (0,)


def test_parallel_read_rejects_small_buffer(tmp_path: Path) -> None:
    blob = _make_blob(tmp_path, payload_offset=3, payload_size=128, seed=0)
    out = np.zeros((64,), dtype=np.uint8)
    with pytest.raises(ValueError, match="pinned_view too small"):
        _read_packed_parallel(
            blob,
            payload_offset=3,
            total_bytes=128,
            pinned_view=out,
            n_workers=2,
        )

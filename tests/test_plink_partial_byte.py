"""Regression guard for partial-byte sample-window decoding in PLINK 1 .bed.

A .bed byte packs 4 samples (low-bit-first) and the trailing bits in the last
byte are zero-padded. When a caller requests a sample window like [5:8] from a
file with 11 samples, the decoder must read whole bytes (samples 4..7) and then
drop the leading sample (4) before truncating to the requested count. This test
exercises non-byte-aligned start/stop boundaries to ensure the byte_start /
byte_stop formula plus leading_sample_offset slicing line up exactly.
"""

from __future__ import annotations

import numpy as np

from sv_pgs.plink import open_bed, to_bed


def _build_synthetic_bed(tmp_path):
    sample_count = 11
    # Genotypes chosen to span all valid PLINK codes (0, 1, 2) and exercise
    # every offset within a byte; one variant suffices to isolate the
    # sample-window slicing logic from any variant-axis bugs.
    genotypes = np.array(
        [[0], [1], [2], [0], [1], [2], [0], [1], [2], [0], [1]],
        dtype=np.float32,
    )
    assert genotypes.shape == (sample_count, 1)
    bed_path = tmp_path / "partial.bed"
    properties = {
        "iid": [f"s{i}" for i in range(sample_count)],
        "sid": ["v0"],
        "chromosome": ["1"],
        "bp_position": [1],
        "allele_1": ["A"],
        "allele_2": ["C"],
    }
    to_bed(bed_path, genotypes, properties)
    return bed_path, genotypes


def test_partial_byte_sample_windows(tmp_path):
    bed_path, genotypes = _build_synthetic_bed(tmp_path)
    reader = open_bed(bed_path, iid_count=genotypes.shape[0], sid_count=1)

    for start, stop in [(0, 4), (4, 8), (5, 7), (8, 11), (3, 10)]:
        window = reader.read(index=(slice(start, stop), slice(0, 1)), dtype="float32")
        expected = genotypes[start:stop]
        assert window.shape == expected.shape, (
            f"window shape mismatch for [{start}:{stop}]: got {window.shape}, "
            f"expected {expected.shape}"
        )
        np.testing.assert_array_equal(
            window,
            expected,
            err_msg=f"partial-byte window [{start}:{stop}] mismatch",
        )

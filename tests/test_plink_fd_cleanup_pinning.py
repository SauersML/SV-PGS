"""Pin ``open_bed`` file-descriptor lifetime.

Bug class: forgetting to close the OS-level fd opened in __post_init__ would
leak file descriptors across long-running runs (the AoU pipeline opens many
.bed files). Pin both the ``with``-context and explicit-close paths.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from sv_pgs.plink import open_bed, to_bed


def _build_tiny_bed(tmp_path: Path):
    sample_count = 4
    genotypes = np.array([[0], [1], [2], [0]], dtype=np.float32)
    bed_path = tmp_path / "fd.bed"
    to_bed(
        bed_path,
        genotypes,
        properties={
            "iid": [f"s{i}" for i in range(sample_count)],
            "sid": ["v0"],
            "chromosome": ["1"],
            "bp_position": [1],
            "allele_1": ["A"],
            "allele_2": ["C"],
        },
    )
    return bed_path, genotypes


def test_context_manager_exit_closes_fd(tmp_path: Path):
    """After exiting the ``with`` block, ``_bed_fd`` must be None and the OS
    fd must be closed (operations on it raise OSError)."""
    bed_path, genotypes = _build_tiny_bed(tmp_path)
    with open_bed(bed_path, iid_count=genotypes.shape[0], sid_count=1) as reader:
        assert reader._bed_fd is not None
        captured_fd = reader._bed_fd
        # The fd is usable inside the context.
        os.fstat(captured_fd)
    # After exit: attribute reset.
    assert reader._bed_fd is None
    # And the OS-level fd is closed.
    with pytest.raises(OSError):
        os.fstat(captured_fd)


def test_explicit_close_is_idempotent(tmp_path: Path):
    """Calling close() twice must not raise — close-after-close is a no-op."""
    bed_path, genotypes = _build_tiny_bed(tmp_path)
    reader = open_bed(bed_path, iid_count=genotypes.shape[0], sid_count=1)
    reader.close()
    assert reader._bed_fd is None
    # Second close must be safe.
    reader.close()
    assert reader._bed_fd is None


def test_read_after_close_raises(tmp_path: Path):
    """After close(), attempts to read via the reader must raise rather than
    silently succeed against a closed/recycled fd."""
    bed_path, genotypes = _build_tiny_bed(tmp_path)
    reader = open_bed(bed_path, iid_count=genotypes.shape[0], sid_count=1)
    reader.close()
    with pytest.raises((RuntimeError, OSError, ValueError)):
        reader._pread_payload(0, 1)

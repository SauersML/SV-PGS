"""Unit tests for the per-iteration bitpacked profiler.

CPU-only: no GPU sync; just the accumulator + render contracts.
"""

from __future__ import annotations

import time

from sv_pgs import bitpacked_profile as bp


def setup_function(_):
    bp.reset_cumulative()
    # Drop any prior thread-local bucket so each test starts clean.
    bp.snapshot_and_reset()


def test_record_and_reset_round_trip():
    with bp.record("matvec"):
        time.sleep(0.001)
    with bp.record("rmatvec"):
        time.sleep(0.001)
    per_iter, cumul = bp.snapshot_and_reset()
    assert "matvec" in per_iter
    assert "rmatvec" in per_iter
    assert per_iter["matvec"] > 0.0
    assert per_iter["rmatvec"] > 0.0
    assert cumul["matvec"] == per_iter["matvec"]
    # Next reset returns empty per-iter; cumulative keeps the prior matvec.
    per_iter_2, cumul_2 = bp.snapshot_and_reset()
    assert per_iter_2 == {}
    assert cumul_2["matvec"] == per_iter["matvec"]


def test_add_imperative_path():
    bp.add("gram", 1.23)
    per_iter, _ = bp.snapshot_and_reset()
    assert per_iter["gram"] == 1.23


def test_summary_line_canonical_order_then_unknown():
    per_iter = {
        "loss": 0.01,
        "matvec": 0.1,
        "rmatvec": 0.2,
        "custom_op": 0.05,
    }
    line = bp.summary_line(per_iter, iter_index=3, n_iter_total=10)
    assert line.startswith("iter 3/10:")
    assert "matvec=0.100s" in line
    assert "rmatvec=0.200s" in line
    assert "loss=0.010s" in line
    assert "custom_op=0.050s" in line
    assert "total=0.360s" in line
    # Canonical ordering: matvec must appear before rmatvec, both before custom_op.
    assert line.index("matvec=") < line.index("rmatvec=")
    assert line.index("rmatvec=") < line.index("custom_op=")


def test_summary_line_empty_returns_empty():
    assert bp.summary_line({}) == ""


def test_enable_disable_cuda_sync():
    # No-cupy environment: enable+disable should not raise.
    bp.enable_cuda_sync(True)
    assert bp.cuda_sync_enabled() is True
    with bp.record("test_op"):
        pass
    bp.enable_cuda_sync(False)
    assert bp.cuda_sync_enabled() is False
    # Reset state for other tests.
    bp.snapshot_and_reset()

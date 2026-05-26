"""Tests for the named-region in-flight diagnostics tracker."""

from __future__ import annotations

import time

from sv_pgs.diagnostics import (
    _active_regions,
    format_region_line,
    region,
    snapshot,
    update_bytes,
)


def test_region_progress_and_rates() -> None:
    """A region with bytes_total + update_bytes yields positive rate + ETA."""
    assert _active_regions == [], "diagnostics tracker not clean at test start"
    with region("unit.read", bytes_total=1000, source="fake.bed") as r:
        # Sleep enough so elapsed > 0 even on the fastest CI hardware.
        time.sleep(0.01)
        update_bytes("unit.read", 250)
        snap = snapshot()
        assert len(snap) == 1
        entry = snap[0]
        assert entry["name"] == "unit.read"
        assert entry["bytes_done"] == 250
        assert entry["bytes_total"] == 1000
        assert entry["elapsed"] > 0
        assert entry["mb_per_sec"] is not None
        assert entry["mb_per_sec"] > 0
        assert entry["eta_sec"] is not None
        assert entry["eta_sec"] > 0
        # extra captured verbatim
        assert entry["extra"]["source"] == "fake.bed"
        # The region object itself reflects the latest bytes_done.
        assert r.bytes_done == 250
        # Formatter must not crash and must include the region name + elapsed.
        line = format_region_line(entry)
        assert "unit.read" in line
        assert "elapsed=" in line


def test_nested_regions_stack_and_unwind() -> None:
    """Nested regions both appear in snapshot; exiting inner removes only inner."""
    assert _active_regions == [], "diagnostics tracker not clean at test start"
    with region("outer", bytes_total=500):
        update_bytes("outer", 100)
        with region("inner", bytes_total=200):
            update_bytes("inner", 50)
            snap = snapshot()
            assert [e["name"] for e in snap] == ["outer", "inner"]
            assert snap[0]["bytes_done"] == 100
            assert snap[1]["bytes_done"] == 50
        # Inner exited; outer remains.
        snap_after_inner = snapshot()
        assert [e["name"] for e in snap_after_inner] == ["outer"]
        assert snap_after_inner[0]["bytes_done"] == 100
    # All exited.
    assert snapshot() == []


def test_update_bytes_innermost_wins_for_duplicate_names() -> None:
    """When two regions share a name, update_bytes targets the innermost."""
    assert _active_regions == [], "diagnostics tracker not clean at test start"
    with region("same", bytes_total=1000):
        with region("same", bytes_total=2000):
            update_bytes("same", 42)
            snap = snapshot()
            assert snap[0]["bytes_done"] == 0
            assert snap[1]["bytes_done"] == 42


def test_snapshot_with_no_open_regions_is_empty() -> None:
    assert _active_regions == [], "diagnostics tracker not clean at test start"
    assert snapshot() == []


def test_update_bytes_unknown_region_is_noop() -> None:
    # No region open: must not raise.
    update_bytes("nope", 999)
    with region("present"):
        update_bytes("absent", 999)
        snap = snapshot()
        assert snap[0]["bytes_done"] == 0

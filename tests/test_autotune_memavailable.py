"""Tests for `sv_pgs.compute_budget._detect_available_host_ram_bytes`.

Regression: previously this function preferred `os.sysconf("SC_AVPHYS_PAGES")`,
which is `MemFree`-equivalent on Linux. On a box with hundreds of GB of
page cache reclaim headroom the kernel reports tiny MemFree, which would
clamp `bed_batch_bytes` to the 128 MB floor and `prefetch_depth` to 1.
The fix prefers `/proc/meminfo:MemAvailable`.
"""
from __future__ import annotations

import builtins
import io
from typing import Any

import pytest

from sv_pgs import compute_budget


def _patch_proc_meminfo(monkeypatch: pytest.MonkeyPatch, contents: str | None) -> None:
    """Patch `open()` so reads of `/proc/meminfo` return `contents`.

    If `contents` is None, simulate an unreadable `/proc/meminfo`
    (raises OSError, as on Darwin / containers without /proc).
    """
    real_open = builtins.open

    def fake_open(path: Any, *args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        if str(path) == "/proc/meminfo":
            if contents is None:
                raise OSError("simulated: /proc/meminfo unavailable")
            return io.StringIO(contents)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)


def test_memavailable_primary(monkeypatch: pytest.MonkeyPatch) -> None:
    """Primary path: read MemAvailable directly."""
    # 250_000_000 kB = 256_000_000_000 bytes ≈ 238.4 GiB
    _patch_proc_meminfo(
        monkeypatch,
        "MemTotal:       263000000 kB\n"
        "MemFree:           243000 kB\n"
        "MemAvailable:   250000000 kB\n"
        "Cached:         200000000 kB\n"
        "SReclaimable:    10000000 kB\n",
    )
    result = compute_budget._detect_available_host_ram_bytes()
    assert result == 250_000_000 * 1024  # ≈ 256 GB
    # Sanity: vastly larger than the historical 128 MB floor.
    assert result > 100 * 1024 * 1024 * 1024


def test_memfree_plus_cached_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fallback 1: MemAvailable absent (pre-3.14 kernel) -> MemFree+Cached+SReclaimable."""
    _patch_proc_meminfo(
        monkeypatch,
        "MemTotal:       263000000 kB\n"
        "MemFree:           500000 kB\n"
        "Cached:         200000000 kB\n"
        "SReclaimable:    10000000 kB\n",
    )
    result = compute_budget._detect_available_host_ram_bytes()
    expected = (500_000 + 200_000_000 + 10_000_000) * 1024
    assert result == expected


def test_hardcoded_4gb_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Last resort: no /proc/meminfo and no sysconf -> 4 GB hardcoded."""
    _patch_proc_meminfo(monkeypatch, None)
    monkeypatch.setattr(
        compute_budget.os,
        "sysconf",
        lambda _name: (_ for _ in ()).throw(OSError("disabled for test")),
    )
    result = compute_budget._detect_available_host_ram_bytes()
    assert result == compute_budget._AUTO_TUNE_HOST_RAM_FALLBACK_BYTES
    assert result == 4 * 1024 * 1024 * 1024

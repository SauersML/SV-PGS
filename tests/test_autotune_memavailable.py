"""Tests for `sv_pgs.compute_budget._detect_available_host_ram_bytes`.

The budget reads `/proc/meminfo:MemAvailable`, which counts reclaimable page
cache (MemFree does not), and refuses to guess when it is missing.
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
    assert result == 250_000_000 * 1024


def test_missing_memavailable_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A kernel without MemAvailable (pre-3.14) is refused instead of guessed."""
    _patch_proc_meminfo(
        monkeypatch,
        "MemTotal:       263000000 kB\n"
        "MemFree:           500000 kB\n",
    )
    with pytest.raises(RuntimeError, match="MemAvailable"):
        compute_budget._detect_available_host_ram_bytes()


def test_unreadable_meminfo_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """No /proc/meminfo: the error propagates; there is no hand-set fallback size."""
    _patch_proc_meminfo(monkeypatch, None)
    with pytest.raises(OSError):
        compute_budget._detect_available_host_ram_bytes()

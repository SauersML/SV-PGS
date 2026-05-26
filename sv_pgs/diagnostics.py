"""Named-region in-flight diagnostics for stall investigation.

The pipeline's hot loops can stall for minutes inside a single function call
(e.g. a slow rebitpack or a wedged NVMe read), making the heartbeat-stack
dump the only signal of what's stuck. This module adds a process-wide
"region tracker" that the heartbeat reads at log time: each hot loop wraps
itself in :func:`region` (optionally with a known ``bytes_total``), and
calls :func:`update_bytes` as it makes progress. The heartbeat then emits a
one-line summary per open region with elapsed time, MB/s, and ETA.

Designed for surgical wrapping of hot loops. Cheap (lock-protected list
append/remove), thread-safe, never imports heavy dependencies at top level.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class _Region:
    name: str
    started_at: float
    bytes_done: int = 0
    bytes_total: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


_active_regions: list[_Region] = []
_lock = threading.Lock()


@contextmanager
def region(
    name: str,
    *,
    bytes_total: int | None = None,
    **extra: Any,
) -> Iterator[_Region]:
    """Open a named region; nested regions are stacked.

    The heartbeat reads the stack and emits all open regions with elapsed
    time and progress. ``extra`` is captured verbatim into the snapshot so
    callers can attach loop-specific context (chunk index, worker count,
    source path, etc.) without inventing new region-tracker APIs.
    """
    r = _Region(
        name=name,
        started_at=time.perf_counter(),
        bytes_total=bytes_total,
        extra=dict(extra),
    )
    with _lock:
        _active_regions.append(r)
    try:
        yield r
    finally:
        with _lock:
            try:
                _active_regions.remove(r)
            except ValueError:
                # Defensive: caller mutated the tracker externally.
                pass


def update_bytes(name: str, bytes_done: int) -> None:
    """Bump ``bytes_done`` on the innermost matching region.

    No-op if no region with ``name`` is currently open. The innermost
    (most recently entered) match wins, so nested regions of the same
    name behave intuitively.
    """
    with _lock:
        for r in reversed(_active_regions):
            if r.name == name:
                r.bytes_done = int(bytes_done)
                return


def update_extra(name: str, **extra: Any) -> None:
    """Merge ``extra`` into the innermost matching region's extra dict."""
    with _lock:
        for r in reversed(_active_regions):
            if r.name == name:
                r.extra.update(extra)
                return


def snapshot() -> list[dict[str, Any]]:
    """Return one dict per open region (outer-first).

    Each dict has keys: ``name``, ``elapsed``, ``bytes_done``,
    ``bytes_total``, ``mb_per_sec``, ``eta_sec``, ``extra``. ``mb_per_sec``
    and ``eta_sec`` are ``None`` when not computable (no bytes yet, no
    total known, or zero elapsed). The caller must not hold the
    diagnostics lock while logging — this function copies everything out
    before returning so heartbeat I/O does not block region open/close.
    """
    now = time.perf_counter()
    with _lock:
        snapshot_list = []
        for r in _active_regions:
            elapsed = now - r.started_at
            if r.bytes_done > 0 and elapsed > 0:
                rate_bytes_per_sec = r.bytes_done / elapsed
                mb_per_sec: float | None = r.bytes_done / 1e6 / elapsed
            else:
                rate_bytes_per_sec = 0.0
                mb_per_sec = None
            if (
                r.bytes_total is not None
                and r.bytes_done > 0
                and rate_bytes_per_sec > 0
            ):
                remaining = max(r.bytes_total - r.bytes_done, 0)
                eta_sec: float | None = remaining / rate_bytes_per_sec
            else:
                eta_sec = None
            snapshot_list.append(
                {
                    "name": r.name,
                    "elapsed": elapsed,
                    "bytes_done": int(r.bytes_done),
                    "bytes_total": (
                        int(r.bytes_total) if r.bytes_total is not None else None
                    ),
                    "mb_per_sec": mb_per_sec,
                    "eta_sec": eta_sec,
                    "extra": dict(r.extra),
                }
            )
        return snapshot_list


def _format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds - 60 * minutes
    if minutes < 60:
        return f"{minutes}m{secs:.0f}s"
    hours = minutes // 60
    minutes = minutes % 60
    return f"{hours}h{minutes}m{secs:.0f}s"


def _format_bytes_gb(value: int | None) -> str:
    if value is None:
        return "?"
    gb = value / 1e9
    if gb >= 1.0:
        return f"{gb:.1f}GB"
    mb = value / 1e6
    if mb >= 1.0:
        return f"{mb:.1f}MB"
    return f"{value}B"


def format_region_line(entry: dict[str, Any]) -> str:
    """Format one snapshot entry as a single human-readable log line."""
    name = entry["name"]
    elapsed = _format_elapsed(float(entry["elapsed"]))
    parts = [f"region '{name}' elapsed={elapsed}"]
    bytes_done = entry.get("bytes_done") or 0
    bytes_total = entry.get("bytes_total")
    if bytes_done or bytes_total:
        parts.append(
            f"bytes={_format_bytes_gb(bytes_done)}/{_format_bytes_gb(bytes_total)}"
        )
    rate = entry.get("mb_per_sec")
    eta = entry.get("eta_sec")
    rate_eta_parts: list[str] = []
    if rate is not None:
        rate_eta_parts.append(f"{rate:.1f} MB/s")
    if eta is not None:
        rate_eta_parts.append(f"ETA {_format_elapsed(float(eta))}")
    if rate_eta_parts:
        parts.append("(" + ", ".join(rate_eta_parts) + ")")
    extra = entry.get("extra") or {}
    if extra:
        kv = ", ".join(f"{k}={v}" for k, v in extra.items())
        parts.append("extra={" + kv + "}")
    return " ".join(parts)

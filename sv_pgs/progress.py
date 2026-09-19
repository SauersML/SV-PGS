"""Timestamped progress lines on stderr, with wall time since import and resident memory."""

from __future__ import annotations

import sys
import time

_start_time: float = time.monotonic()


def mem() -> str:
    """Current resident set size, from /proc/self/status."""
    with open("/proc/self/status", encoding="ascii") as status:
        for line in status:
            if line.startswith("VmRSS:"):
                return f"{int(line.split()[1]) // 1024} MB"
    raise RuntimeError("/proc/self/status has no VmRSS line")


def elapsed() -> str:
    """Wall-clock time since this module was imported."""
    seconds = time.monotonic() - _start_time
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{seconds:.0f}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h{minutes}m{seconds:.0f}s"


def log(message: str) -> None:
    """Print a timestamped progress line to stderr."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp} | {elapsed()} | {mem()}] {message}", file=sys.stderr, flush=True)

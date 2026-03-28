"""Lightweight progress logging for pipeline diagnostics."""

from __future__ import annotations

import os
import resource
import sys
import time

_start_time: float = time.monotonic()


def mem() -> str:
    """Return current RSS in MB (Linux: from /proc, macOS: peak RSS)."""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return line.split()[1] + " kB->~" + str(int(line.split()[1]) // 1024) + " MB"
    except (OSError, IndexError):
        pass
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        rss //= 1024 * 1024
    else:
        rss //= 1024
    return f"{rss} MB"


def peak_mem() -> str:
    """Return peak RSS in MB."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        rss //= 1024 * 1024
    else:
        rss //= 1024
    return f"{rss} MB"


def elapsed() -> str:
    """Return wall-clock time since module import."""
    dt = time.monotonic() - _start_time
    if dt < 60:
        return f"{dt:.1f}s"
    minutes = int(dt // 60)
    seconds = dt % 60
    if minutes < 60:
        return f"{minutes}m{seconds:.0f}s"
    hours = minutes // 60
    minutes = minutes % 60
    return f"{hours}h{minutes}m{seconds:.0f}s"


def log(message: str) -> None:
    """Print a timestamped progress message to stderr."""
    print(f"[{elapsed()} | {mem()}] {message}", file=sys.stderr, flush=True)

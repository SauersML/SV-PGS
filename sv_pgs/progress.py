"""Timestamped progress lines on stderr, with wall time since import and resident memory."""

from __future__ import annotations

import os
import sys
import threading
import time
import traceback
from pathlib import Path

QUIET_STACK_SECONDS = "SVPGS_QUIET_STACK_SECONDS"
"""The environment variable that, set, has the driver report where it is after that many seconds without a progress
line (``sample_quiet_stacks``)."""
_PACKAGE = Path(__file__).resolve().parent
_start_time: float = time.monotonic()
_last_line: float = _start_time


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
    global _last_line
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp} | {elapsed()} | {mem()}] {message}", file=sys.stderr, flush=True)
    _last_line = time.monotonic()


def sample_quiet_stacks(seconds: float) -> threading.Event:
    """Whenever ``seconds`` pass with no progress line, log where the main thread is: its stack's frames in this
    package, innermost first, and the innermost frame wherever it is. A silent stretch then shows what ran, a sampling
    profile at that period: the AoU disease fits went twenty minutes between lines, and a whole-genome measurement step
    an hour, with no sign of what they did. Setting the returned event stops the sampling."""
    main, stop = threading.main_thread().ident, threading.Event()

    def watch() -> None:
        while not stop.wait(seconds):
            quiet = time.monotonic() - _last_line
            frame = sys._current_frames().get(main)
            if quiet < seconds or frame is None:
                continue
            stack = traceback.extract_stack(frame)
            shown = [entry for entry in stack if Path(entry.filename).resolve().is_relative_to(_PACKAGE)]
            if not shown or shown[-1] is not stack[-1]:
                shown.append(stack[-1])
            log(f"quiet {quiet:.0f} s; main thread at " + " < ".join(f"{Path(entry.filename).name}:{entry.lineno} {entry.name}" for entry in reversed(shown)))

    threading.Thread(target=watch, name="quiet-stacks", daemon=True).start()
    return stop

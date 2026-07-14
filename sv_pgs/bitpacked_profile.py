"""Lightweight per-iteration timing accumulators for the bitpacked hot path.

Designed for production logs: each call to ``record(op)`` is a context
manager that adds the elapsed wall-clock to a thread-local accumulator
keyed by ``op`` (one of "matvec", "rmatvec", "gram", "posterior", "loss",
or any arbitrary string the caller chooses). The bitpacked path can call
``cuda_sync()`` from this module before stopping the timer so the recorded
elapsed includes the kernel's device runtime, not just the launch overhead.

After each EM iteration the inner loop calls :func:`snapshot_and_reset`
to obtain a ``(per_iter, cumulative)`` pair of ``{op: seconds}`` dicts and
:func:`summary_line` to render the production log row::

    iter K/N: matvec=0.04s rmatvec=0.04s gram=0.6s posterior=0.02s loss=0.01s total=0.71s

The module is intentionally self-contained: nothing in this file imports
``cupy`` at import time, so it stays importable in CPU-only test
environments. Synchronization is gated behind a runtime probe.
"""

from __future__ import annotations

import contextlib
import threading
import time
from typing import Iterator


_TLS = threading.local()
# Process-wide cumulative counters: simple dict, mutated only from the EM thread
# (the EM loop). We don't try to be thread-safe across worker threads — the
# hot loop is single-threaded by design.
_CUMULATIVE: dict[str, float] = {}
_SYNC_ENABLED = False


def enable_cuda_sync(enabled: bool = True) -> None:
    """Turn cupy device synchronization on or off around timer stops.

    The bitpacked hot path is fully GPU-resident; without
    ``cupy.cuda.runtime.deviceSynchronize()`` the recorded times would
    represent only kernel-launch overhead. CPU-only callers should leave
    this disabled (the default) to avoid an unconditional cupy import.
    """
    global _SYNC_ENABLED
    _SYNC_ENABLED = bool(enabled)


def cuda_sync_enabled() -> bool:
    return _SYNC_ENABLED


def _maybe_sync() -> None:
    if not _SYNC_ENABLED:
        return
    try:  # noqa: SIM105 - explicit branches kept for readability
        import cupy as cp  # type: ignore[import-not-found]
        cp.cuda.runtime.deviceSynchronize()
    except Exception:  # pragma: no cover - sync failures must not block run
        pass


def _per_iter() -> dict[str, float]:
    """Return the per-iteration accumulator dict, creating one if needed."""
    bucket = getattr(_TLS, "per_iter", None)
    if bucket is None:
        bucket = {}
        _TLS.per_iter = bucket
    return bucket


@contextlib.contextmanager
def record(op: str) -> Iterator[None]:
    """Accumulate wall-clock for ``op`` in the current iteration bucket.

    Safe to nest, but nested calls double-count time (matvec inside posterior
    appears in both); callers should avoid that.
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _maybe_sync()
        elapsed = time.perf_counter() - t0
        bucket = _per_iter()
        bucket[op] = bucket.get(op, 0.0) + elapsed


def add(op: str, seconds: float) -> None:
    """Imperative variant of :func:`record` — add ``seconds`` to ``op``."""
    bucket = _per_iter()
    bucket[op] = bucket.get(op, 0.0) + float(seconds)


def snapshot_and_reset() -> tuple[dict[str, float], dict[str, float]]:
    """Return ``(per_iter, cumulative)`` snapshots and reset the per-iter dict.

    ``cumulative`` keeps growing across iterations so the heartbeat can show
    where the run has been spending time overall.
    """
    bucket = _per_iter()
    per_iter_copy = dict(bucket)
    # Fold the per-iteration values into the process-wide cumulative.
    for key, value in per_iter_copy.items():
        _CUMULATIVE[key] = _CUMULATIVE.get(key, 0.0) + float(value)
    bucket.clear()
    return per_iter_copy, dict(_CUMULATIVE)


def cumulative_snapshot() -> dict[str, float]:
    """Return a copy of the cumulative-since-process-start counters."""
    return dict(_CUMULATIVE)


def reset_cumulative() -> None:
    """Drop the process-wide cumulative counters. Tests use this for isolation."""
    _CUMULATIVE.clear()


_CANONICAL_ORDER: tuple[str, ...] = (
    "matvec",
    "rmatvec",
    "gram",
    "posterior",
    "loss",
)


def summary_line(
    per_iter: dict[str, float],
    *,
    iter_index: int | None = None,
    n_iter_total: int | None = None,
) -> str:
    """Render a single ``iter K/N: <op>=<sec>s ... total=<sec>s`` line.

    Unknown ops (anything not in :data:`_CANONICAL_ORDER`) are appended at
    the end in insertion order so a caller adding a new ``record("foo")``
    site sees ``foo`` in the log without any other change.
    """
    if not per_iter:
        return ""
    total = sum(float(v) for v in per_iter.values())
    parts: list[str] = []
    if iter_index is not None and n_iter_total is not None:
        parts.append(f"iter {int(iter_index)}/{int(n_iter_total)}:")
    elif iter_index is not None:
        parts.append(f"iter {int(iter_index)}:")
    seen: set[str] = set()
    for key in _CANONICAL_ORDER:
        if key in per_iter:
            parts.append(f"{key}={per_iter[key]:.3f}s")
            seen.add(key)
    for key, value in per_iter.items():
        if key in seen:
            continue
        parts.append(f"{key}={float(value):.3f}s")
    parts.append(f"total={total:.3f}s")
    return " ".join(parts)

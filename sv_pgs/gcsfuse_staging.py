"""gcsfuse detection and local-staging helpers.

AoU mounts ``~/workspace/vwb-aou-datasets-controlled/v8/`` via gcsfuse.
Every read against a gcsfuse path is an HTTP GET to Google Cloud Storage.
mmap-ing such a file turns random page faults into network round-trips,
which is orders of magnitude slower than local NVMe. The helpers here
detect gcsfuse-backed paths and force a copy to local disk before any
hot-path read is performed.

Staging is concurrent-worker safe: a per-destination ``<dest>.lock`` file
is acquired with ``fcntl.lockf`` to serialize workers, copies stream into
a unique ``<dest>.partial.<pid>.<uuid4>`` path, and the publish step is
``fsync(file)`` + ``os.replace`` + ``fsync(parent_dir)``. Each successful
publish writes a ``<dest>.manifest.json`` sidecar; ``verify_local_cache``
uses the manifest (not just file existence) for cache-hit detection.

The module is pure stdlib and must import cleanly on darwin and other
non-Linux platforms; detection is a no-op there.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from sv_pgs.diagnostics import region, update_bytes

try:  # fcntl is POSIX-only; we degrade to no-op locking on platforms without it.
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - non-POSIX
    _fcntl = None  # type: ignore[assignment]

__all__ = [
    "is_gcsfuse_path",
    "gcsfuse_mounts",
    "stage_to_local",
    "stage_to_local_parallel",
    "stage_bed_trio_to_local",
    "open_for_sequential_read",
    "verify_local_cache",
    "read_manifest",
    "manifest_path_for",
    "MANIFEST_SCHEMA_VERSION",
]

_LOG = logging.getLogger(__name__)

# 64 MB read/write buffer — gcsfuse benefits from large sequential I/O.
_COPY_BUFFER_BYTES: int = 64 * 1024 * 1024
# Threshold (bytes) above which we surface copy progress logs to the user.
_LARGE_COPY_LOG_THRESHOLD: int = 1 * 1024 * 1024 * 1024
# Sidecar suffixes.
_MANIFEST_SUFFIX: str = ".manifest.json"
_LOCK_SUFFIX: str = ".lock"
_PARTIAL_INFIX: str = ".partial"

MANIFEST_SCHEMA_VERSION: int = 1
_PRODUCER: str = "sv_pgs.gcsfuse_staging.stage_to_local"


def _is_linux() -> bool:
    return sys.platform.startswith("linux")


def _parse_proc_mounts() -> list[tuple[Path, str, str, str]]:
    """Parse /proc/mounts. Returns list of (mount_point, source, fstype, options).

    Returns an empty list on any failure or on non-Linux platforms.
    """
    if not _is_linux():
        return []
    try:
        with open("/proc/mounts", "r", encoding="utf-8", errors="replace") as fh:
            raw_lines = fh.readlines()
    except OSError:
        return []

    parsed: list[tuple[Path, str, str, str]] = []
    for line in raw_lines:
        parts = line.split()
        if len(parts) < 4:
            continue
        source, mount_point, fstype, options = parts[0], parts[1], parts[2], parts[3]
        try:
            mount_point_decoded = (
                mount_point.encode("utf-8").decode("unicode_escape")
            )
        except UnicodeDecodeError:
            mount_point_decoded = mount_point
        parsed.append((Path(mount_point_decoded), source, fstype, options))
    return parsed


@lru_cache(maxsize=1)
def gcsfuse_mounts() -> list[Path]:
    """Return all detected gcsfuse mount points.

    Detection strategy:
      * Parse ``/proc/mounts`` on Linux.
      * A mount qualifies as gcsfuse when its fstype begins with ``fuse`` AND
        either its source string contains ``gcsfuse`` OR its mount options
        contain ``gcsfuse`` / ``fuse.gcsfuse`` / ``fsname=gcsfuse``.
      * Returns an empty list on non-Linux platforms.

    Cached for the lifetime of the process — gcsfuse mounts do not change
    while a run is in progress.
    """
    if not _is_linux():
        return []

    mounts: list[Path] = []
    for mount_point, source, fstype, options in _parse_proc_mounts():
        if not fstype.startswith("fuse"):
            continue
        source_lc = source.lower()
        options_lc = options.lower()
        fstype_lc = fstype.lower()
        is_gcsfuse = (
            "gcsfuse" in source_lc
            or "gcsfuse" in options_lc
            or "fuse.gcsfuse" in options_lc
            or "fsname=gcsfuse" in options_lc
            or fstype_lc == "fuse.gcsfuse"
        )
        if is_gcsfuse:
            mounts.append(mount_point)
    return mounts


def _resolve_safely(path: Path) -> Path:
    """Resolve symlinks; on failure, return the original path."""
    try:
        return path.resolve()
    except (OSError, RuntimeError):
        return path


@lru_cache(maxsize=1024)
def _is_gcsfuse_path_cached(resolved_str: str) -> bool:
    resolved = Path(resolved_str)
    mounts = gcsfuse_mounts()
    if mounts:
        for mount in mounts:
            try:
                resolved.relative_to(mount)
                return True
            except ValueError:
                continue
        return False

    if not _is_linux():
        return False
    try:
        os.statvfs(str(resolved))
    except OSError:
        return False
    return False


def is_gcsfuse_path(path: Path) -> bool:
    """Return True if ``path`` lives on a gcsfuse-mounted filesystem.

    The check resolves symlinks first (AoU's runner symlinks files INTO the
    gcsfuse mount, so the link target is what matters). Result is cached
    per resolved path string. Always returns False on non-Linux platforms.
    """
    if not _is_linux():
        return False
    resolved = _resolve_safely(Path(path))
    return _is_gcsfuse_path_cached(str(resolved))


# ---------------------------------------------------------------------------
# Sidecar paths
# ---------------------------------------------------------------------------


def manifest_path_for(local_path: Path) -> Path:
    """Return the manifest sidecar path for ``local_path``."""
    local_path = Path(local_path)
    return local_path.with_name(local_path.name + _MANIFEST_SUFFIX)


def _lock_path_for(local_path: Path) -> Path:
    return local_path.with_name(local_path.name + _LOCK_SUFFIX)


def _unique_partial_path(local_path: Path) -> Path:
    suffix = f"{_PARTIAL_INFIX}.{os.getpid()}.{uuid.uuid4().hex}"
    return local_path.with_name(local_path.name + suffix)


def _resumable_partial_path(local_path: Path) -> Path:
    """Return a deterministic per-destination partial path for resumable copies.

    Unlike ``_unique_partial_path``, this path is stable across runs so that
    ``gcloud storage cp`` can detect its prior ``<dest>_.gstmp`` sibling and
    resume an interrupted download from the existing byte offset. Safe because
    ``_DestinationLock`` already serializes concurrent stagers per-dest.
    """
    return local_path.with_suffix(local_path.suffix + ".resume.partial")


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


class _DestinationLock:
    """Exclusive lock on ``<dest>.lock`` via ``fcntl.lockf``.

    Degrades to a no-op on platforms without ``fcntl`` (Windows, etc.) — but
    the production target here is Linux, where fcntl is always available.
    """

    def __init__(self, local_path: Path) -> None:
        self._lock_path = _lock_path_for(local_path)
        self._fd: int | None = None

    def __enter__(self) -> "_DestinationLock":
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(
            str(self._lock_path),
            os.O_RDWR | os.O_CREAT,
            0o644,
        )
        if _fcntl is not None:
            _fcntl.lockf(self._fd, _fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fd is None:
            return
        try:
            if _fcntl is not None:
                try:
                    _fcntl.lockf(self._fd, _fcntl.LOCK_UN)
                except OSError:
                    pass
        finally:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None


# ---------------------------------------------------------------------------
# fsync helpers
# ---------------------------------------------------------------------------


def _fsync_dir(directory: Path) -> None:
    """Best-effort ``fsync`` of a directory so the rename is durable."""
    try:
        dir_fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        try:
            os.fsync(dir_fd)
        except OSError:
            pass
    finally:
        os.close(dir_fd)


def _posix_fadvise_sequential(fd: int, size: int) -> None:
    """Hint sequential read-ahead on the source file. Best-effort."""
    if not hasattr(os, "posix_fadvise"):
        return
    try:
        os.posix_fadvise(fd, 0, size, os.POSIX_FADV_SEQUENTIAL)  # type: ignore[attr-defined]
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _utcnow_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _write_manifest(
    *,
    local_path: Path,
    source_path: Path,
    source_size_bytes: int,
    source_mtime_ns: int,
    source_was_gcsfuse: bool,
    local_size_bytes: int,
    crc32c: str | None,
    md5: str | None,
) -> None:
    """Atomically write a manifest sidecar next to ``local_path``."""
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_utc": _utcnow_iso(),
        "producer": _PRODUCER,
        "source_path": str(source_path),
        "source_size_bytes": int(source_size_bytes),
        "source_mtime": int(source_mtime_ns),
        "source_was_gcsfuse": bool(source_was_gcsfuse),
        "local_path": str(local_path),
        "local_size_bytes": int(local_size_bytes),
        "crc32c": crc32c,
        "md5": md5,
        "complete": True,
    }
    manifest_path = manifest_path_for(local_path)
    tmp = manifest_path.with_name(
        manifest_path.name + f".partial.{os.getpid()}.{uuid.uuid4().hex}"
    )
    tmp.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(fd, payload)
        try:
            os.fsync(fd)
        except OSError:
            pass
    finally:
        os.close(fd)
    os.replace(str(tmp), str(manifest_path))
    _fsync_dir(manifest_path.parent)


def read_manifest(local_path: Path) -> dict[str, Any] | None:
    """Read and parse the manifest sidecar for ``local_path``.

    Returns ``None`` if the manifest is absent or unparseable.
    """
    manifest_path = manifest_path_for(local_path)
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def verify_local_cache(
    local_path: Path,
    *,
    expected_size: int | None = None,
    expected_md5: str | None = None,
    expected_crc32c: str | None = None,
) -> bool:
    """Return True if ``local_path`` has a valid manifest and matches expectations.

    Checks:
      * manifest exists and is parseable
      * manifest schema version matches
      * manifest ``complete`` is True
      * ``local_path`` exists and ``stat().st_size`` matches the manifest's
        ``local_size_bytes``
      * if ``expected_size`` is provided, manifest size matches
      * if ``expected_md5`` / ``expected_crc32c`` provided, the manifest's
        recorded hash matches
    """
    local_path = Path(local_path)
    manifest = read_manifest(local_path)
    if manifest is None:
        return False
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        return False
    if not manifest.get("complete"):
        return False
    if not local_path.exists():
        return False
    try:
        actual_size = local_path.stat().st_size
    except OSError:
        return False
    manifest_local_size = manifest.get("local_size_bytes")
    if not isinstance(manifest_local_size, int) or manifest_local_size != actual_size:
        return False
    if expected_size is not None and manifest_local_size != expected_size:
        return False
    if expected_md5 is not None and manifest.get("md5") != expected_md5:
        return False
    if expected_crc32c is not None and manifest.get("crc32c") != expected_crc32c:
        return False
    return True


# ---------------------------------------------------------------------------
# Copy primitive
# ---------------------------------------------------------------------------


def _copy_and_publish(
    source: Path,
    destination: Path,
    *,
    buffer_bytes: int,
    expected_size: int | None,
    expected_md5: str | None,
    compute_md5: bool,
    source_was_gcsfuse: bool,
) -> tuple[int, str | None]:
    """Stream-copy ``source`` to a unique partial then atomically publish.

    Returns ``(bytes_written, md5_hex_or_None)``.

    Verification steps:
      * If ``expected_size`` is provided, the source's pre-copy size must match.
      * If the source is NOT on gcsfuse (regular file), pre- vs. post-copy
        ``st_size`` must agree (catch concurrent writers).
      * If ``expected_md5`` is provided, the post-copy hash must match.
      * ``fsync`` the dest fd before close, then ``fsync`` the parent dir
        after the rename, so the publish survives a crash.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    pre_stat = source.stat()
    src_size_pre = pre_stat.st_size
    if expected_size is not None and src_size_pre != expected_size:
        raise ValueError(
            f"stage_to_local: source size {src_size_pre} != expected {expected_size} "
            f"for {source}"
        )

    partial = _unique_partial_path(destination)

    log_progress = src_size_pre >= _LARGE_COPY_LOG_THRESHOLD
    if log_progress:
        _LOG.info(
            "gcsfuse staging: copying %s -> %s (%.1f GB)",
            source,
            destination,
            src_size_pre / 1e9,
        )

    start_time = time.monotonic()
    bytes_copied = 0
    next_log_at = 1 * 1024 * 1024 * 1024
    md5_hasher = hashlib.md5() if (compute_md5 or expected_md5 is not None) else None

    try:
        with open(source, "rb", buffering=0) as src_fh:
            _posix_fadvise_sequential(src_fh.fileno(), src_size_pre)
            dst_fd = os.open(
                str(partial),
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o644,
            )
            try:
                while True:
                    chunk = src_fh.read(buffer_bytes)
                    if not chunk:
                        break
                    os.write(dst_fd, chunk)
                    bytes_copied += len(chunk)
                    if md5_hasher is not None:
                        md5_hasher.update(chunk)
                    if log_progress and bytes_copied >= next_log_at:
                        elapsed = max(time.monotonic() - start_time, 1e-6)
                        rate_mbps = (bytes_copied / 1e6) / elapsed
                        _LOG.info(
                            "gcsfuse staging: %.1f / %.1f GB (%.0f MB/s)",
                            bytes_copied / 1e9,
                            src_size_pre / 1e9,
                            rate_mbps,
                        )
                        next_log_at += 1 * 1024 * 1024 * 1024
                try:
                    os.fsync(dst_fd)
                except OSError:
                    pass
            finally:
                os.close(dst_fd)

        # Post-copy verification.
        if not source_was_gcsfuse:
            try:
                post_size = source.stat().st_size
            except OSError:
                post_size = src_size_pre
            if post_size != src_size_pre:
                raise RuntimeError(
                    f"stage_to_local: source size changed during copy "
                    f"({src_size_pre} -> {post_size}) for {source}"
                )

        md5_hex: str | None = md5_hasher.hexdigest() if md5_hasher is not None else None
        if expected_md5 is not None and md5_hex != expected_md5:
            raise RuntimeError(
                f"stage_to_local: md5 mismatch (expected {expected_md5}, got {md5_hex}) "
                f"for {source}"
            )
        if bytes_copied != src_size_pre:
            raise RuntimeError(
                f"stage_to_local: short copy ({bytes_copied} != {src_size_pre}) "
                f"for {source}"
            )

        # Preserve mtime for backward-compatible legacy idempotency checks.
        try:
            os.utime(partial, ns=(pre_stat.st_atime_ns, pre_stat.st_mtime_ns))
        except OSError:
            pass

        os.replace(str(partial), str(destination))
        _fsync_dir(destination.parent)

        if log_progress:
            elapsed = max(time.monotonic() - start_time, 1e-6)
            rate_mbps = (src_size_pre / 1e6) / elapsed
            _LOG.info(
                "gcsfuse staging: finished %s in %.1fs (%.0f MB/s)",
                destination.name,
                elapsed,
                rate_mbps,
            )

        return bytes_copied, md5_hex
    finally:
        # Always clean up our own partial. If we successfully renamed, the
        # path no longer exists and unlink will silently fail.
        try:
            partial.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Public staging API
# ---------------------------------------------------------------------------


def _legacy_already_staged(source: Path, local_path: Path) -> bool:
    """Legacy size+mtime check for caches written before manifest support."""
    if not local_path.exists():
        return False
    try:
        src_stat = source.stat()
        dst_stat = local_path.stat()
    except OSError:
        return False
    if src_stat.st_size != dst_stat.st_size:
        return False
    return dst_stat.st_mtime_ns >= src_stat.st_mtime_ns


def stage_to_local(
    source: Path,
    local_path: Path,
    *,
    force: bool = False,
    copy_threshold_bytes: int = 0,
    expected_size: int | None = None,
    expected_md5: str | None = None,
    expected_crc32c: str | None = None,
    compute_md5: bool = False,
) -> Path:
    """Stage ``source`` to ``local_path`` if it lives on gcsfuse.

    Behaviour:
      * If ``force`` is True, always copy.
      * Else, copy only when ``source`` is gcsfuse-backed.
      * If neither force nor gcsfuse, returns ``source`` unchanged (no copy).
      * Files smaller than ``copy_threshold_bytes`` are returned as-is even
        when gcsfuse-backed.

    Cache-hit detection:
      * A valid manifest sidecar with ``complete=True`` and matching size is
        treated as a cache hit (no re-copy). Optional ``expected_size`` /
        ``expected_md5`` / ``expected_crc32c`` are checked against the
        manifest.
      * If no manifest is present but a legacy size+mtime match exists, a
        synthetic manifest is written and the existing file is reused.

    Concurrency:
      * Workers serialize on a per-destination ``fcntl.lockf`` lock so only
        one copy runs at a time; the rest discover the cache hit on entry.

    Atomicity:
      * The copy streams into ``<dest>.partial.<pid>.<uuid4>``, then
        ``fsync(file)`` + ``os.replace`` + ``fsync(parent_dir)`` publishes.
      * On failure or if a peer wins the rename, the loser's partial is
        unlinked.

    Verification:
      * If ``expected_size`` provided, source pre-copy size must match.
      * If source is not gcsfuse, pre/post source size must agree.
      * If ``expected_md5`` provided, post-copy hash must match.

    Returns the path callers should read from.
    """
    source = Path(source)
    local_path = Path(local_path)

    if not source.exists():
        raise FileNotFoundError(f"stage_to_local: source does not exist: {source}")

    source_was_gcsfuse = is_gcsfuse_path(source)
    needs_copy = force or source_was_gcsfuse
    if not needs_copy:
        return source

    if copy_threshold_bytes > 0 and not force:
        try:
            if source.stat().st_size < copy_threshold_bytes:
                return source
        except OSError:
            pass

    local_path.parent.mkdir(parents=True, exist_ok=True)

    with _DestinationLock(local_path):
        # Cache-hit check inside the lock so concurrent workers converge.
        if not force and verify_local_cache(
            local_path,
            expected_size=expected_size,
            expected_md5=expected_md5,
            expected_crc32c=expected_crc32c,
        ):
            return local_path

        # Legacy cache (no manifest): adopt it by writing a manifest.
        if not force and _legacy_already_staged(source, local_path):
            src_stat = source.stat()
            _write_manifest(
                local_path=local_path,
                source_path=source,
                source_size_bytes=src_stat.st_size,
                source_mtime_ns=src_stat.st_mtime_ns,
                source_was_gcsfuse=source_was_gcsfuse,
                local_size_bytes=local_path.stat().st_size,
                crc32c=expected_crc32c,
                md5=expected_md5,
            )
            return local_path

        # If a stale manifest exists from a prior incomplete attempt, drop it
        # so we don't end up with a manifest pointing at out-of-date bytes.
        manifest_path = manifest_path_for(local_path)
        try:
            manifest_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass

        local_size, md5_hex = _copy_and_publish(
            source,
            local_path,
            buffer_bytes=_COPY_BUFFER_BYTES,
            expected_size=expected_size,
            expected_md5=expected_md5,
            compute_md5=compute_md5,
            source_was_gcsfuse=source_was_gcsfuse,
        )

        src_stat = source.stat()
        _write_manifest(
            local_path=local_path,
            source_path=source,
            source_size_bytes=src_stat.st_size,
            source_mtime_ns=src_stat.st_mtime_ns,
            source_was_gcsfuse=source_was_gcsfuse,
            local_size_bytes=local_size,
            crc32c=expected_crc32c,
            md5=md5_hex if md5_hex is not None else expected_md5,
        )

    return local_path


# ---------------------------------------------------------------------------
# Parallel staging (N-worker pread copy)
# ---------------------------------------------------------------------------


def _default_parallel_workers() -> int:
    """Return ``min(os.cpu_count(), 8)`` clamped to >= 1."""
    try:
        cpu = int(os.cpu_count() or 1)
    except Exception:  # noqa: BLE001
        cpu = 1
    return max(1, min(8, cpu))


def _copy_and_publish_parallel(
    source: Path,
    destination: Path,
    *,
    n_workers: int,
    source_was_gcsfuse: bool,
    log_label: str | None = None,
) -> int:
    """Parallel copy of ``source`` into ``destination`` via N independent fds.

    Each worker owns a contiguous byte range and ``os.pread`` / ``os.pwrite``
    its slice from the gcsfuse source into the local partial file. The dest
    file is pre-allocated with ``os.ftruncate(size)`` so workers can write
    their non-overlapping ranges concurrently without lseek collisions.

    Returns total bytes written. Performs ``fsync(dest)`` + ``os.replace`` +
    ``fsync(parent_dir)`` for atomic publish.
    """
    import threading as _threading
    import time as _time
    from concurrent.futures import ThreadPoolExecutor

    destination.parent.mkdir(parents=True, exist_ok=True)

    src_size = source.stat().st_size
    if src_size == 0:
        # Edge case: empty source. Just touch the destination.
        partial = _unique_partial_path(destination)
        open(partial, "wb").close()
        os.replace(str(partial), str(destination))
        _fsync_dir(destination.parent)
        return 0

    partial = _unique_partial_path(destination)
    label = log_label or source.name
    total_gb = src_size / 1e9

    _LOG.info(
        "gcsfuse staging: %s -> %s (%.2f GB, %d parallel writers)",
        source,
        destination,
        total_gb,
        n_workers,
    )

    # Pre-size the destination so each worker pwrites into a fixed region.
    dst_fd = os.open(
        str(partial),
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        os.ftruncate(dst_fd, src_size)
    except OSError:
        os.close(dst_fd)
        try:
            partial.unlink()
        except OSError:
            pass
        raise

    chunk = (src_size + n_workers - 1) // n_workers
    bytes_done = [0]
    bytes_done_lock = _threading.Lock()
    stop_event = _threading.Event()
    t_start = _time.monotonic()
    # 5 GB progress cadence per spec.
    progress_step = 5 * 1024 * 1024 * 1024

    def _emitter() -> None:
        last_emitted_gb_bucket = 0
        while not stop_event.wait(2.0):
            with bytes_done_lock:
                cur = bytes_done[0]
            bucket = cur // progress_step
            if bucket > last_emitted_gb_bucket and cur > 0:
                elapsed = max(_time.monotonic() - t_start, 1e-6)
                mb_per_sec = (cur / 1e6) / elapsed
                remaining = max(src_size - cur, 0)
                eta_min = (remaining / max(cur / elapsed, 1.0)) / 60.0
                _LOG.info(
                    "staging %s: %.1f/%.1f GB (%.0f MB/s, ETA %.1f min) "
                    "[parallel x%d]",
                    label,
                    cur / 1e9,
                    total_gb,
                    mb_per_sec,
                    eta_min,
                    n_workers,
                )
                last_emitted_gb_bucket = bucket

    src_fds: list[int] = []
    emitter_thread = _threading.Thread(
        target=_emitter, name="gcsfuse-stage-progress", daemon=True
    )
    bytes_copied_total = 0
    _stage_region = region(
        "gcsfuse.stage",
        bytes_total=src_size,
        source=source.name,
        dest=destination.name,
        n_workers=n_workers,
    )
    _stage_region.__enter__()
    try:
        for _ in range(n_workers):
            sfd = os.open(str(source), os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
            _posix_fadvise_sequential(sfd, src_size)
            src_fds.append(sfd)

        def _worker(worker_idx: int) -> int:
            start = worker_idx * chunk
            stop = min(src_size, start + chunk)
            if start >= stop:
                return 0
            sfd = src_fds[worker_idx]
            sub = 64 * 1024 * 1024  # 64 MB sub-reads
            cursor = start
            worker_total = 0
            while cursor < stop:
                want = min(sub, stop - cursor)
                buf = os.pread(sfd, want, cursor)
                got = len(buf)
                if got == 0:
                    raise RuntimeError(
                        f"pread returned 0 bytes at offset {cursor} "
                        f"(worker={worker_idx}); source truncated?"
                    )
                # pwrite is thread-safe at the OS level on Linux; each worker
                # writes a disjoint byte range, so there is no contention.
                written = 0
                while written < got:
                    n = os.pwrite(dst_fd, buf[written:], cursor + written)
                    if n <= 0:
                        raise RuntimeError(
                            f"pwrite returned {n} bytes at offset {cursor + written} "
                            f"(worker={worker_idx})"
                        )
                    written += n
                cursor += got
                worker_total += got
                with bytes_done_lock:
                    bytes_done[0] += got
                    update_bytes("gcsfuse.stage", bytes_done[0])
            return worker_total

        emitter_thread.start()
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_worker, i) for i in range(n_workers)]
            for f in futures:
                bytes_copied_total += f.result()

        try:
            os.fsync(dst_fd)
        except OSError:
            pass
    finally:
        stop_event.set()
        if emitter_thread.is_alive():
            try:
                emitter_thread.join(timeout=2.0)
            except RuntimeError:
                pass
        try:
            os.close(dst_fd)
        except OSError:
            pass
        for sfd in src_fds:
            try:
                os.close(sfd)
            except OSError:
                pass
        _stage_region.__exit__(None, None, None)

    if bytes_copied_total != src_size:
        # Clean up partial then raise.
        try:
            partial.unlink()
        except OSError:
            pass
        raise RuntimeError(
            f"stage_to_local_parallel: short copy "
            f"({bytes_copied_total} != {src_size}) for {source}"
        )

    # Preserve mtime so legacy idempotency checks still work.
    try:
        pre_stat = source.stat()
        os.utime(partial, ns=(pre_stat.st_atime_ns, pre_stat.st_mtime_ns))
    except OSError:
        pass

    try:
        os.replace(str(partial), str(destination))
        _fsync_dir(destination.parent)
    except Exception:
        try:
            partial.unlink()
        except OSError:
            pass
        raise

    elapsed = max(_time.monotonic() - t_start, 1e-6)
    mb_per_sec = (src_size / 1e6) / elapsed
    _LOG.info(
        "gcsfuse staging: finished %s in %.1fs (%.0f MB/s) [parallel x%d]",
        destination.name,
        elapsed,
        mb_per_sec,
        n_workers,
    )
    return bytes_copied_total


# ---------------------------------------------------------------------------
# Fast path: gcloud storage cp / gsutil cp -m (bypass gcsfuse network cap)
# ---------------------------------------------------------------------------


def _gcs_uri_for_gcsfuse_path(path: Path) -> str | None:
    """Map a gcsfuse-mounted local path to its underlying ``gs://`` URI.

    Resolves symlinks first (AoU runners often symlink placeholders INTO the
    gcsfuse mount). Returns None if the resolved path is not under any
    gcsfuse mount, or if we cannot recover the source bucket from
    ``/proc/mounts``.
    """
    if not _is_linux():
        return None
    resolved = _resolve_safely(Path(path))
    try:
        resolved_str = str(resolved)
    except Exception:  # noqa: BLE001
        return None

    # Scan /proc/mounts for gcsfuse entries; longest matching mount_point wins.
    best: tuple[Path, str] | None = None
    for mount_point, source, fstype, options in _parse_proc_mounts():
        if not fstype.startswith("fuse"):
            continue
        source_lc = source.lower()
        options_lc = options.lower()
        fstype_lc = fstype.lower()
        is_gcsfuse = (
            "gcsfuse" in source_lc
            or "gcsfuse" in options_lc
            or "fuse.gcsfuse" in options_lc
            or "fsname=gcsfuse" in options_lc
            or fstype_lc == "fuse.gcsfuse"
        )
        if not is_gcsfuse:
            continue
        try:
            resolved.relative_to(mount_point)
        except ValueError:
            continue
        if best is None or len(str(mount_point)) > len(str(best[0])):
            best = (mount_point, source)

    if best is None:
        return None

    mount_point, source = best
    # The bucket name is the source string. gcsfuse may render it as just
    # "<bucket>" or sometimes prefixed; strip any leading scheme just in case.
    bucket = source
    for prefix in ("gcsfuse#", "gs://"):
        if bucket.startswith(prefix):
            bucket = bucket[len(prefix):]
    # Bucket may include a subdir mount (gcsfuse --only-dir). We can't easily
    # disambiguate from /proc/mounts options without parsing them, so trust
    # the source-as-bucket convention used by the AoU mounts.
    try:
        rel = resolved.relative_to(mount_point).as_posix()
    except ValueError:
        return None
    if not bucket:
        return None
    return f"gs://{bucket}/{rel}"


_GCLOUD_PROGRESS_RE = re.compile(
    r"([\d.]+)\s*([KMGT]?i?B)\s*/\s*([\d.]+)\s*([KMGT]?i?B)",
    re.IGNORECASE,
)
_UNIT_FACTORS = {
    "B": 1,
    "KB": 10**3, "KIB": 2**10,
    "MB": 10**6, "MIB": 2**20,
    "GB": 10**9, "GIB": 2**30,
    "TB": 10**12, "TIB": 2**40,
}


def _parse_bytes(value: str, unit: str) -> int | None:
    try:
        v = float(value)
    except ValueError:
        return None
    factor = _UNIT_FACTORS.get(unit.upper())
    if factor is None:
        return None
    return int(v * factor)


def _stage_via_gcloud_storage(
    src_gs_uri: str,
    local_dest: Path,
    *,
    billing_project: str | None,
    expected_size: int | None,
    log_label: str | None,
) -> bool:
    """Stream ``src_gs_uri`` to a ``local_dest`` partial via gcloud / gsutil.

    Streams progress lines from stderr to ``update_bytes("gcsfuse.stage", ...)``
    so the diagnostics region tracks live bytes. Atomic publish (fsync +
    os.replace + parent fsync) is the caller's job after this returns True.

    Returns True on success (partial fully written + renamed in place), False
    if neither gcloud nor gsutil is available or the subprocess failed.
    """
    gcloud = shutil.which("gcloud")
    gsutil = shutil.which("gsutil")
    if gcloud is None and gsutil is None:
        return False

    local_dest.parent.mkdir(parents=True, exist_ok=True)
    partial = _resumable_partial_path(local_dest)
    label = log_label or local_dest.name

    # Detect stale partial from a different source (e.g. AoU re-released the
    # file). gcloud's HTTP range check protects us when sizes are consistent,
    # but a partial larger than the expected source size cannot possibly be
    # the prefix of the current object — drop it before we start.
    if expected_size is not None:
        try:
            cur_partial_size = partial.stat().st_size
        except OSError:
            cur_partial_size = None
        if cur_partial_size is not None and cur_partial_size > expected_size:
            _LOG.warning(
                "staging %s: discarding stale resumable partial "
                "(%.1f GB > expected %.1f GB)",
                label, cur_partial_size / 1e9, expected_size / 1e9,
            )
            try:
                partial.unlink()
            except OSError:
                pass
            gstmp_sibling = partial.with_suffix(partial.suffix + "_.gstmp")
            # gcloud actually writes to "<dest>_.gstmp" (underscore appended to
            # the full name, not a real suffix). Cover both shapes defensively.
            for sib in (
                gstmp_sibling,
                partial.with_name(partial.name + "_.gstmp"),
            ):
                try:
                    sib.unlink()
                except OSError:
                    pass

    # Log resume case so a user grepping logs sees we kept prior progress.
    try:
        if partial.exists():
            sz = partial.stat().st_size
            _LOG.info(
                "staging %s: resuming from existing partial (%s, %.1f GB)",
                label, partial, sz / 1e9,
            )
        else:
            gstmp = partial.with_name(partial.name + "_.gstmp")
            if gstmp.exists():
                sz = gstmp.stat().st_size
                _LOG.info(
                    "staging %s: resuming from existing .gstmp (%s, %.1f GB)",
                    label, gstmp, sz / 1e9,
                )
    except OSError:
        pass

    if gcloud is not None:
        cmd = [gcloud, "storage", "cp"]
        if billing_project:
            cmd.append(f"--billing-project={billing_project}")
        cmd.extend([src_gs_uri, str(partial)])
        tool = "gcloud storage cp"
    else:
        cmd = [
            gsutil,
            "-m",
            "-o", "GSUtil:parallel_thread_count=16",
            "-o", "GSUtil:parallel_process_count=4",
        ]
        if billing_project:
            cmd.extend(["-u", billing_project])
        cmd.extend(["cp", src_gs_uri, str(partial)])
        tool = "gsutil -m cp"

    _LOG.info(
        "staging %s: using %s (%s -> %s)",
        label, tool, src_gs_uri, local_dest,
    )

    t_start = time.monotonic()
    _stage_region = region(
        "gcsfuse.stage",
        bytes_total=expected_size or 0,
        source=src_gs_uri,
        dest=local_dest.name,
        n_workers=1,
        tool=tool,
    )
    _stage_region.__enter__()

    proc: subprocess.Popen[str] | None = None
    last_bytes = 0
    stop_poller = threading.Event()
    poller_thread: threading.Thread | None = None

    def _poll_partial_size() -> None:
        nonlocal last_bytes
        while not stop_poller.is_set():
            if proc is not None and proc.poll() is not None:
                break
            try:
                cur = partial.stat().st_size
            except (FileNotFoundError, OSError):
                if stop_poller.wait(2.0):
                    break
                continue
            if cur >= last_bytes:
                last_bytes = cur
                try:
                    update_bytes("gcsfuse.stage", cur)
                except Exception:
                    pass
            if stop_poller.wait(2.0):
                break

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        poller_thread = threading.Thread(
            target=_poll_partial_size, daemon=True
        )
        poller_thread.start()
        for line in proc.stdout:
            m = _GCLOUD_PROGRESS_RE.search(line)
            if m is not None:
                cur = _parse_bytes(m.group(1), m.group(2))
                if cur is not None and cur >= last_bytes:
                    last_bytes = cur
                    update_bytes("gcsfuse.stage", cur)
        rc = proc.wait()
        if rc != 0:
            _LOG.warning(
                "staging %s: %s exited with rc=%d; falling back",
                label, tool, rc,
            )
            return False

        # Verify size if known.
        try:
            actual = partial.stat().st_size
        except OSError:
            _LOG.warning("staging %s: partial missing after %s", label, tool)
            return False
        if expected_size is not None and actual != expected_size:
            _LOG.warning(
                "staging %s: size mismatch after %s (%d != %d); falling back",
                label, tool, actual, expected_size,
            )
            return False

        # fsync the file before rename so the publish survives a crash.
        try:
            fd = os.open(str(partial), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass

        os.replace(str(partial), str(local_dest))
        _fsync_dir(local_dest.parent)

        elapsed = max(time.monotonic() - t_start, 1e-6)
        mb_per_sec = (actual / 1e6) / elapsed
        _LOG.info(
            "staging %s: finished via %s in %.1fs (%.0f MB/s)",
            label, tool, elapsed, mb_per_sec,
        )
        update_bytes("gcsfuse.stage", actual)
        return True
    except (OSError, subprocess.SubprocessError) as e:
        _LOG.warning("staging %s: %s failed: %s; falling back", label, tool, e)
        return False
    finally:
        stop_poller.set()
        if poller_thread is not None:
            poller_thread.join(timeout=5.0)
        _stage_region.__exit__(None, None, None)
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass
        # Intentionally DO NOT unlink the resumable partial (or its _.gstmp
        # sibling) on failure: leaving them lets the next run resume from the
        # offset already on disk. On success, os.replace above renamed the
        # partial away, so there is nothing left to clean.


def stage_to_local_parallel(
    source: Path,
    local_path: Path,
    *,
    n_workers: int | None = None,
    force: bool = False,
    expected_size: int | None = None,
    log_label: str | None = None,
) -> Path:
    """Stage ``source`` to ``local_path`` using N parallel pread/pwrite workers.

    Like :func:`stage_to_local` but uses a multi-worker copy primitive that
    issues independent ``os.pread`` calls on per-worker file descriptors. On
    gcsfuse this saturates the aggregate per-connection bandwidth (typically
    ~1 GB/s with 8 workers vs ~150 MB/s single-stream).

    Behaviour matches :func:`stage_to_local`:
      * Symlinks are resolved (we stat/read the underlying gcsfuse target).
      * Cache hit (via manifest) returns early; otherwise the file is copied
        into ``<local>.partial.<pid>.<uuid4>`` and atomically renamed after
        ``fsync``.
      * If ``source`` is not gcsfuse-backed and ``force`` is False, returns
        ``source`` unchanged (no copy).

    Progress is logged at ~5 GB granularity with MB/s + ETA.

    Returns the path the caller should read from.
    """
    source = Path(source)
    local_path = Path(local_path)

    if not source.exists():
        raise FileNotFoundError(
            f"stage_to_local_parallel: source does not exist: {source}"
        )

    # Resolve the symlink: AoU runners often symlink local placeholders INTO
    # the gcsfuse mount, so the link target is what we need to read from and
    # what gcsfuse detection must inspect.
    resolved_source = _resolve_safely(source)
    source_was_gcsfuse = is_gcsfuse_path(resolved_source)
    if not force and not source_was_gcsfuse:
        return source

    if n_workers is None:
        n_workers = _default_parallel_workers()
    n_workers = max(1, int(n_workers))

    local_path.parent.mkdir(parents=True, exist_ok=True)

    with _DestinationLock(local_path):
        if not force and verify_local_cache(
            local_path, expected_size=expected_size
        ):
            return local_path

        if not force and _legacy_already_staged(resolved_source, local_path):
            src_stat = resolved_source.stat()
            _write_manifest(
                local_path=local_path,
                source_path=resolved_source,
                source_size_bytes=src_stat.st_size,
                source_mtime_ns=src_stat.st_mtime_ns,
                source_was_gcsfuse=source_was_gcsfuse,
                local_size_bytes=local_path.stat().st_size,
                crc32c=None,
                md5=None,
            )
            return local_path

        # Drop stale manifest so a crashed prior attempt cannot be mistaken
        # for a cache hit.
        manifest_path = manifest_path_for(local_path)
        try:
            manifest_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass

        # --- FAST PATH: gcloud storage cp against gs:// (bypasses gcsfuse) ---
        # gcsfuse caps single-mount throughput at ~100-200 MB/s. Going direct
        # to the GCS XML API via `gcloud storage cp` parallel multipart uses
        # the full NIC, typically 500-1000 MB/s on AoU.
        local_size: int | None = None
        if source_was_gcsfuse:
            gs_uri = _gcs_uri_for_gcsfuse_path(resolved_source)
            billing_project = (
                os.environ.get("GOOGLE_PROJECT")
                or os.environ.get("GOOGLE_CLOUD_PROJECT")
            )
            try:
                src_size_hint = resolved_source.stat().st_size
            except OSError:
                src_size_hint = expected_size
            if gs_uri is not None and billing_project:
                ok = _stage_via_gcloud_storage(
                    gs_uri,
                    local_path,
                    billing_project=billing_project,
                    expected_size=src_size_hint,
                    log_label=log_label or resolved_source.name,
                )
                if ok:
                    try:
                        local_size = local_path.stat().st_size
                    except OSError:
                        local_size = None
            elif gs_uri is None:
                _LOG.info(
                    "staging %s: could not resolve gs:// URI; "
                    "falling back to parallel gcsfuse pread (slower)",
                    log_label or resolved_source.name,
                )
            else:
                _LOG.info(
                    "staging %s: no GOOGLE_PROJECT/GOOGLE_CLOUD_PROJECT set; "
                    "falling back to parallel gcsfuse pread (slower)",
                    log_label or resolved_source.name,
                )

        if local_size is None:
            if (
                source_was_gcsfuse
                and shutil.which("gcloud") is None
                and shutil.which("gsutil") is None
            ):
                _LOG.info(
                    "staging %s: gcloud not available; "
                    "falling back to parallel gcsfuse pread (slower)",
                    log_label or resolved_source.name,
                )
            local_size = _copy_and_publish_parallel(
                resolved_source,
                local_path,
                n_workers=n_workers,
                source_was_gcsfuse=source_was_gcsfuse,
                log_label=log_label,
            )
        src_stat = resolved_source.stat()
        _write_manifest(
            local_path=local_path,
            source_path=resolved_source,
            source_size_bytes=src_stat.st_size,
            source_mtime_ns=src_stat.st_mtime_ns,
            source_was_gcsfuse=source_was_gcsfuse,
            local_size_bytes=local_size,
            crc32c=None,
            md5=None,
        )
    return local_path


def stage_bed_trio_to_local(bed_path: Path, local_bed_path: Path) -> Path:
    """Stage a PLINK ``.bed`` plus sibling ``.bim`` and ``.fam`` to local disk.

    The trio is staged atomically: either all three files end up locally, or
    the function raises and leaves any partial state cleaned up. If the
    source bed is not gcsfuse-backed, the original path is returned
    unchanged (and no copies are made).
    """
    bed_path = Path(bed_path)
    local_bed_path = Path(local_bed_path)

    if bed_path.suffix != ".bed":
        raise ValueError(
            f"stage_bed_trio_to_local: expected .bed path, got {bed_path}"
        )

    bim_source = bed_path.with_suffix(".bim")
    fam_source = bed_path.with_suffix(".fam")
    for sibling in (bim_source, fam_source):
        if not sibling.exists():
            raise FileNotFoundError(
                f"stage_bed_trio_to_local: missing trio sibling: {sibling}"
            )

    if not is_gcsfuse_path(bed_path):
        return bed_path

    local_bim_path = local_bed_path.with_suffix(".bim")
    local_fam_path = local_bed_path.with_suffix(".fam")

    try:
        staged_bed = stage_to_local(bed_path, local_bed_path)
        stage_to_local(bim_source, local_bim_path)
        stage_to_local(fam_source, local_fam_path)
    except Exception:
        # Best-effort: drop any manifest sidecars whose underlying file is
        # mismatched, and clean up our partials. We deliberately do NOT
        # delete a file that already cache-hit cleanly.
        for src, dst in (
            (bed_path, local_bed_path),
            (bim_source, local_bim_path),
            (fam_source, local_fam_path),
        ):
            try:
                if dst.exists() and dst.stat().st_size != src.stat().st_size:
                    dst.unlink()
                    manifest_path_for(dst).unlink(missing_ok=True)
            except OSError:
                pass
            for stray in dst.parent.glob(dst.name + _PARTIAL_INFIX + ".*"):
                try:
                    stray.unlink()
                except OSError:
                    pass
        raise

    return staged_bed


def open_for_sequential_read(path: Path) -> tuple[Path, bool]:
    """Return ``(path_to_read, is_local)`` for ``path``.

    Callers like ``mmap_reader`` use this to decide whether mmap is safe:
    mmap should only be used when ``is_local`` is True, because mmap over
    gcsfuse converts random page faults into individual HTTP GETs.

    This function does NOT copy. If the path is gcsfuse-backed the caller
    should fall back to a non-mmap read path or call ``stage_to_local``
    first.
    """
    path = Path(path)
    resolved = _resolve_safely(path)
    is_local = not is_gcsfuse_path(resolved)
    return resolved, is_local

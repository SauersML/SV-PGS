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
import sys
import time
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

try:  # fcntl is POSIX-only; we degrade to no-op locking on platforms without it.
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - non-POSIX
    _fcntl = None  # type: ignore[assignment]

__all__ = [
    "is_gcsfuse_path",
    "gcsfuse_mounts",
    "stage_to_local",
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

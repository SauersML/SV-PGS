"""Storage classification policy for hot genotype I/O paths.

Every hot-path genotype reader in sv_pgs must read from local disk; reading
genotype bytes off a gcsfuse mount or a ``gs://`` URI turns each random page
access into an HTTP round-trip and destroys throughput. This module formalises
the classification used by those entry points so they can fail loudly when a
caller hands them a non-local path.

The module is pure stdlib and may be imported on any platform. On darwin the
network-FS probe is skipped (we only check for gcsfuse and gs://); on Linux we
consult ``/proc/mounts`` to identify NFS/SMB/CIFS/SSHFS mounts.
"""

from __future__ import annotations

import enum
import os
import sys
from functools import lru_cache
from pathlib import Path

__all__ = [
    "StorageClass",
    "classify_path",
    "assert_hot_local_path",
    "assert_safe_for_purpose",
    "is_local_hot",
    "is_gcsfuse_path",
    "gcsfuse_mounts",
]


_NETWORK_FSTYPES: frozenset[str] = frozenset(
    {"nfs", "nfs4", "smbfs", "cifs", "sshfs", "fuse.sshfs"}
)


class StorageClass(enum.Enum):
    LOCAL_HOT = "local_hot"
    GCSFUSE_MOUNT = "gcsfuse_mount"
    GS_URI = "gs_uri"
    UNKNOWN_REMOTE_OR_NETWORK = "unknown_remote_or_network"
    NONEXISTENT = "nonexistent"


def _is_linux() -> bool:
    return sys.platform.startswith("linux")


def _resolve_safely(path: Path) -> Path:
    try:
        return path.resolve()
    except (OSError, RuntimeError):
        return path


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


@lru_cache(maxsize=1)
def _mount_table() -> tuple[tuple[str, str], ...]:
    """Return ((mount_point, fstype), ...) parsed from /proc/mounts on Linux."""
    if not _is_linux():
        return ()
    try:
        with open("/proc/mounts", "r", encoding="utf-8", errors="replace") as fh:
            raw_lines = fh.readlines()
    except OSError:
        return ()

    entries: list[tuple[str, str]] = []
    for line in raw_lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        mount_point, fstype = parts[1], parts[2]
        try:
            mount_point_decoded = mount_point.encode("utf-8").decode("unicode_escape")
        except UnicodeDecodeError:
            mount_point_decoded = mount_point
        entries.append((mount_point_decoded, fstype))
    return tuple(entries)


def _fstype_for(resolved: Path) -> str | None:
    """Return the filesystem type for ``resolved`` via /proc/mounts (Linux)."""
    if not _is_linux():
        return None
    best_match: tuple[int, str] | None = None
    resolved_str = str(resolved)
    for mount_point, fstype in _mount_table():
        if mount_point == "/" or resolved_str == mount_point or resolved_str.startswith(
            mount_point.rstrip("/") + "/"
        ):
            length = len(mount_point)
            if best_match is None or length > best_match[0]:
                best_match = (length, fstype)
    return best_match[1] if best_match is not None else None


def classify_path(path: str | Path) -> StorageClass:
    """Classify ``path`` into a :class:`StorageClass`.

    Detection order:
      1. ``gs://`` prefix  -> ``GS_URI``.
      2. resolve symlinks; ask :func:`is_gcsfuse_path` -> ``GCSFUSE_MOUNT``.
      3. parent of resolved path missing -> ``NONEXISTENT``.
      4. Linux fstype in {nfs, smbfs, cifs, sshfs} -> ``UNKNOWN_REMOTE_OR_NETWORK``.
      5. otherwise -> ``LOCAL_HOT``.

    Darwin and other non-Linux platforms only check for ``gs://`` and gcsfuse;
    everything else that exists is treated as ``LOCAL_HOT``.
    """
    raw = os.fspath(path)
    if raw.startswith("gs://"):
        return StorageClass.GS_URI

    p = Path(raw)
    resolved = _resolve_safely(p)

    try:
        if is_gcsfuse_path(resolved):
            return StorageClass.GCSFUSE_MOUNT
    except Exception:
        # Detection helpers must never propagate -- treat as unknown remote.
        return StorageClass.UNKNOWN_REMOTE_OR_NETWORK

    parent = resolved.parent
    if not parent.exists():
        return StorageClass.NONEXISTENT

    if _is_linux():
        fstype = _fstype_for(resolved)
        if fstype is not None and fstype.lower() in _NETWORK_FSTYPES:
            return StorageClass.UNKNOWN_REMOTE_OR_NETWORK
        # Confirm the path is on a filesystem we can statvfs (best-effort).
        try:
            os.statvfs(str(resolved if resolved.exists() else parent))
        except OSError:
            return StorageClass.UNKNOWN_REMOTE_OR_NETWORK

    return StorageClass.LOCAL_HOT


def is_local_hot(path: str | Path) -> bool:
    """Return True iff ``path`` is safe for repeated random-access I/O."""
    return classify_path(path) is StorageClass.LOCAL_HOT


def assert_hot_local_path(path: str | Path, *, purpose: str) -> None:
    """Raise ``RuntimeError`` if ``path`` is not safe for hot genotype I/O.

    "Safe" means :attr:`StorageClass.LOCAL_HOT`. Any other classification
    raises with a message that names the ``purpose``, the detected storage
    class, and a remediation hint pointing at the staging helpers.
    """
    cls = classify_path(path)
    if cls is StorageClass.LOCAL_HOT:
        return

    remediation_by_class: dict[StorageClass, str] = {
        StorageClass.GCSFUSE_MOUNT: (
            "stage to local via sv_pgs.gcsfuse_staging.stage_to_local "
            "(or stage_bed_trio_to_local for PLINK trios) before opening "
            "for random access"
        ),
        StorageClass.GS_URI: (
            "stage to local via sv_pgs.aou_storage.stage_gcs_object first; "
            "raw gs:// URIs cannot be opened as regular files"
        ),
        StorageClass.UNKNOWN_REMOTE_OR_NETWORK: (
            "copy the file to a local NVMe path before hot I/O; network "
            "filesystems (nfs/smbfs/cifs/sshfs) are not supported for "
            "random-access genotype reads"
        ),
        StorageClass.NONEXISTENT: (
            "ensure the parent directory exists and the file has been "
            "staged before opening it"
        ),
    }
    hint = remediation_by_class.get(
        cls, "stage to local via sv_pgs.aou_storage.stage_gcs_object first"
    )

    raise RuntimeError(
        f"path_policy: refusing hot I/O for purpose={purpose!r}: "
        f"path={os.fspath(path)!r} classified as {cls.value}. "
        f"Remediation: {hint}."
    )


def assert_safe_for_purpose(
    path: str | Path,
    *,
    purpose: str,
    allow_sequential_gcsfuse: bool = False,
) -> None:
    """Like :func:`assert_hot_local_path` but with an opt-in escape hatch for
    one-shot sequential reads from a gcsfuse-backed path.

    gcsfuse handles purely sequential reads acceptably (~200 MB/s with a large
    chunked-read buffer); what destroys throughput is repeated random access
    and mmap, where every page fault becomes an HTTP GET.

    When ``allow_sequential_gcsfuse=True`` the function returns silently for a
    gcsfuse-backed path. The caller is asserting it will read the file once,
    sequentially, ideally with ``posix_fadvise(SEQUENTIAL)`` and a large
    buffer. All other non-local classifications (gs:// URIs, nonexistent
    paths, NFS/SSHFS/CIFS network mounts) are still rejected.

    When ``allow_sequential_gcsfuse=False`` this is exactly equivalent to
    :func:`assert_hot_local_path`.
    """
    cls = classify_path(path)
    if cls is StorageClass.LOCAL_HOT:
        return
    if allow_sequential_gcsfuse and cls is StorageClass.GCSFUSE_MOUNT:
        return
    # Defer to the strict guard so callers get the same error/remediation text
    # for every other case.
    assert_hot_local_path(path, purpose=purpose)

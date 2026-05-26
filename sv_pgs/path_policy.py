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

from sv_pgs.gcsfuse_staging import is_gcsfuse_path

__all__ = [
    "StorageClass",
    "classify_path",
    "assert_hot_local_path",
    "is_local_hot",
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

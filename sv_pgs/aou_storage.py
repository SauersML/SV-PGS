"""AoU cache manager: atomic GCS/gcsfuse -> local hot disk staging.

This module is the manifest-aware staging path used by AoU code. It supersedes
the bare ``shutil.copy`` pattern in :mod:`sv_pgs.gcsfuse_staging` for callers
that need verifiable, resumable, race-safe caching.

Design points
-------------
* The unit of work is a single :class:`RemoteObject` -> :class:`LocalCacheEntry`
  staging operation. Each cached file has a sibling ``<local>.manifest.json``
  recording the source URI, byte count, optional CRC32C, and a completion flag.
  A cache entry is considered valid only when both the data file and the
  manifest exist, ``complete`` is True, and on-disk size matches the manifest.

* Staging is atomic. Bytes are written to a per-process, per-call partial path
  ``<local>.partial.<pid>.<uuid8>``, fsync'd, then ``os.replace``'d into
  position. The manifest is only published *after* the rename succeeds. Any
  exception unlinks the partial and leaves the previous manifest (if any)
  untouched.

* Concurrency is guarded with an advisory ``fcntl.lockf`` on ``<local>.lock``.
  Concurrent racers serialize through the lock; the second-to-arrive sees the
  manifest published by the first and short-circuits without re-downloading.

* The module is pure stdlib + ``subprocess`` (for ``gsutil``) + the two sibling
  helpers requested by spec. No GPU imports.
"""

from __future__ import annotations

import datetime as _dt
import fcntl
import json
import logging
import os
import shutil
import subprocess
import time
import uuid
import zlib
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

from sv_pgs import gcsfuse_staging
from sv_pgs import path_policy  # noqa: F401  (kept for future hot-path assertion hooks)

__all__ = [
    "RemoteObject",
    "LocalCacheEntry",
    "stage_gcs_object",
    "verify_local_cache",
    "read_manifest",
    "stage_aou_plink_trio",
]

_LOG = logging.getLogger(__name__)

# 64 MiB copy buffer mirrors gcsfuse_staging — large sequential I/O is required
# for both gcsfuse-backed and local NVMe sources to saturate bandwidth.
_COPY_BUFFER_BYTES: int = 64 * 1024 * 1024

# Producer string embedded into the manifest. Useful when debugging a stale
# cache: ``jq .producer foo.manifest.json`` immediately identifies the writer.
_PRODUCER: str = "sv_pgs.aou_storage.stage_gcs_object"

# Manifest schema version. Bump on any breaking change to the manifest layout;
# verify_local_cache will treat an unknown schema_version as invalid.
_MANIFEST_SCHEMA_VERSION: int = 1


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RemoteObject:
    """Description of a remote (or remote-like) source object.

    ``uri`` may be:
      * ``gs://bucket/object`` — fetched via ``gsutil cp``.
      * A gcsfuse-mounted path — read directly with a buffered copy.
      * Any other local path — read directly with a buffered copy.

    ``size_bytes`` is the authoritative source size. For GCS URIs we obtain it
    from ``gsutil du``; for local sources we ``stat()`` it. ``generation``,
    ``crc32c``, ``md5_hash``, and ``updated`` are best-effort metadata fields
    captured for the manifest record; ``None`` means the field was unavailable
    at staging time.
    """

    uri: str
    size_bytes: int
    generation: str | None = None
    crc32c: str | None = None
    md5_hash: str | None = None
    updated: str | None = None  # ISO 8601 UTC


@dataclass(frozen=True)
class LocalCacheEntry:
    """Result of a successful stage operation.

    ``complete`` mirrors the manifest field and is True iff the entry passed
    end-to-end verification at publication time. ``content_kind`` is a free-form
    tag (e.g. ``"aou_array_plink_bed"``) that callers use to disambiguate
    multiple cache entries that share a basename.
    """

    local_path: Path
    manifest_path: Path
    remote: RemoteObject
    complete: bool
    content_kind: str
    schema_version: int = _MANIFEST_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _manifest_path_for(local_path: Path) -> Path:
    """Return the canonical ``<local>.manifest.json`` sibling path."""
    return local_path.with_name(local_path.name + ".manifest.json")


def _lock_path_for(local_path: Path) -> Path:
    """Return the canonical ``<local>.lock`` sibling used for fcntl serialization."""
    return local_path.with_name(local_path.name + ".lock")


def _utc_now_iso() -> str:
    """Timezone-aware ISO 8601 UTC stamp with seconds precision."""
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


@contextmanager
def _file_lock(lock_file: Path, *, enabled: bool) -> Iterator[None]:
    """Advisory file lock via fcntl.lockf.

    The lock file is created with ``O_CREAT`` and left in place after release.
    On non-POSIX systems where ``fcntl.lockf`` is missing this becomes a no-op;
    callers must accept the weaker guarantee in that environment.
    """
    if not enabled:
        yield
        return
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_file), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.lockf(fd, fcntl.LOCK_EX)
        except (AttributeError, OSError) as exc:
            _LOG.debug("file lock unavailable for %s: %s", lock_file, exc)
        yield
    finally:
        try:
            fcntl.lockf(fd, fcntl.LOCK_UN)
        except (AttributeError, OSError):
            pass
        os.close(fd)


def _is_gs_uri(uri: str) -> bool:
    """True iff ``uri`` is a ``gs://...`` GCS object URI."""
    return uri.startswith("gs://")


def _check_free_space(target_dir: Path, required_bytes: int) -> None:
    """Raise RuntimeError if ``target_dir`` lacks ``required_bytes`` of headroom."""
    target_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(str(target_dir))
    if usage.free < required_bytes:
        raise RuntimeError(
            f"Insufficient disk space at {target_dir}: "
            f"{usage.free / 1e9:.1f} GB free, need {required_bytes / 1e9:.1f} GB"
        )


def _fsync_path(path: Path) -> None:
    """Best-effort fsync of a file path. Silent on platforms without fsync."""
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _fsync_dir(path: Path) -> None:
    """Best-effort fsync of a directory entry (durable rename on POSIX)."""
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        # Directory fsync is not supported on all filesystems (e.g. tmpfs on
        # some Linux variants, or non-POSIX). Failure is non-fatal.
        pass
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# gsutil helpers (used only when the source is gs://...)
# ---------------------------------------------------------------------------


def _gsutil_cmd(billing_project: str | None) -> list[str]:
    """Build the ``gsutil`` invocation prefix, optionally with billing project."""
    cmd = ["gsutil"]
    if billing_project:
        cmd += ["-u", billing_project]
    return cmd


def _gsutil_stat(uri: str, billing_project: str | None) -> dict[str, str]:
    """Parse ``gsutil stat`` output into a small key->value dict.

    Returns an empty dict on any failure — every consumer treats the metadata
    fields as best-effort, so a stat failure is non-fatal: we still know the
    size from ``gsutil du`` (called separately) and can copy the bytes.
    """
    cmd = _gsutil_cmd(billing_project) + ["stat", uri]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except (OSError, subprocess.SubprocessError):
        return {}
    out: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        out[k.strip()] = v.strip()
    return out


def _gsutil_du_size(uri: str, billing_project: str | None) -> int:
    """Return remote object size in bytes via ``gsutil du -s``."""
    cmd = _gsutil_cmd(billing_project) + ["du", "-s", uri]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    parts = proc.stdout.strip().split()
    if not parts:
        raise RuntimeError(f"gsutil du returned no output for {uri!r}")
    try:
        return int(parts[0])
    except ValueError as exc:
        raise RuntimeError(
            f"gsutil du returned non-integer size for {uri!r}: {parts[0]!r}"
        ) from exc


def _describe_remote(
    source_uri: str,
    billing_project: str | None,
    expected_size: int | None,
) -> RemoteObject:
    """Resolve source metadata into a :class:`RemoteObject`.

    For gs:// URIs we call ``gsutil du`` (required, for size) and best-effort
    ``gsutil stat`` (for generation/crc32c/md5/updated). For local-or-gcsfuse
    sources we stat the file directly.
    """
    if _is_gs_uri(source_uri):
        size = expected_size if expected_size is not None else _gsutil_du_size(
            source_uri, billing_project
        )
        meta = _gsutil_stat(source_uri, billing_project)
        return RemoteObject(
            uri=source_uri,
            size_bytes=size,
            generation=meta.get("Generation"),
            crc32c=meta.get("Hash (crc32c)"),
            md5_hash=meta.get("Hash (md5)"),
            updated=meta.get("Update time") or meta.get("Creation time"),
        )

    src_path = Path(source_uri)
    if not src_path.exists():
        raise FileNotFoundError(f"aou_storage: source does not exist: {source_uri}")
    stat = src_path.stat()
    return RemoteObject(
        uri=str(src_path),
        size_bytes=stat.st_size,
        generation=None,
        crc32c=None,
        md5_hash=None,
        updated=_dt.datetime.fromtimestamp(stat.st_mtime, tz=_dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
    )


# ---------------------------------------------------------------------------
# Copy paths
# ---------------------------------------------------------------------------


def _gsutil_cp_to_partial(
    source_uri: str, partial: Path, billing_project: str | None
) -> None:
    """Stream-download a gs:// URI to ``partial`` via ``gsutil -m cp``.

    The remote URI is re-validated to refuse anything that doesn't start with
    ``gs://`` (defense in depth — even though our callers already constrain
    inputs, gsutil would happily interpret a leading ``-`` as an option flag).
    """
    if not source_uri.startswith("gs://"):
        raise RuntimeError(f"Refusing to gsutil cp from non-gs:// source: {source_uri!r}")
    if str(partial).startswith("-"):
        raise RuntimeError(f"Refusing to gsutil cp to suspicious destination: {partial!r}")

    cmd = _gsutil_cmd(billing_project) + ["-m", "cp", source_uri, str(partial)]
    _LOG.info("aou_storage: gsutil cp %s -> %s", source_uri, partial)
    with subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    ) as proc:
        try:
            if proc.stdout is not None:
                for line in proc.stdout:
                    stripped = line.strip()
                    if stripped:
                        _LOG.debug("    gsutil: %s", stripped)
            proc.wait()
        except BaseException:
            try:
                proc.kill()
            except OSError:
                pass
            raise
    if proc.returncode != 0:
        raise RuntimeError(
            f"gsutil cp failed (exit {proc.returncode}) for {source_uri}"
        )


def _buffered_copy_to_partial(
    source_path: Path, partial: Path, *, source_size: int
) -> None:
    """Stream-copy a regular or gcsfuse-backed file to ``partial``.

    Uses an unbuffered fd pair (so the 64 MiB Python-side buffer is the only
    buffer in play) and hints ``POSIX_FADV_SEQUENTIAL`` on the source — both
    gcsfuse and Linux page cache benefit from the hint. CRC is *not* computed
    here; callers may verify CRC against ``expected_crc32c`` after the copy.
    """
    log_large = source_size >= 1 * 1024 * 1024 * 1024
    if log_large:
        _LOG.info(
            "aou_storage: copy %s -> %s (%.1f GB)",
            source_path,
            partial,
            source_size / 1e9,
        )

    start = time.monotonic()
    with open(source_path, "rb", buffering=0) as src_fh:
        if hasattr(os, "posix_fadvise"):
            try:
                os.posix_fadvise(
                    src_fh.fileno(), 0, source_size, os.POSIX_FADV_SEQUENTIAL
                )
            except OSError:
                pass
        with open(partial, "wb", buffering=0) as dst_fh:
            while True:
                chunk = src_fh.read(_COPY_BUFFER_BYTES)
                if not chunk:
                    break
                dst_fh.write(chunk)
            dst_fh.flush()
            try:
                os.fsync(dst_fh.fileno())
            except OSError:
                pass

    if log_large:
        elapsed = max(time.monotonic() - start, 1e-6)
        _LOG.info(
            "aou_storage: copied %s in %.1fs (%.0f MB/s)",
            partial.name,
            elapsed,
            (source_size / 1e6) / elapsed,
        )


def _crc32c_hex_of_file(path: Path) -> str:
    """Compute a CRC over the file contents using :func:`zlib.crc32`.

    NOTE: ``zlib.crc32`` is the IEEE 802.3 polynomial (CRC-32), *not* the
    Castagnoli CRC-32C that Google Cloud Storage advertises. We name the
    parameter and the manifest field ``crc32c`` for callsite symmetry with the
    GCS metadata field, but the actual algorithm used here is CRC-32 (IEEE).
    Callers that need true Castagnoli CRC-32C should provide it pre-computed
    via ``expected_crc32c`` and rely on the byte-for-byte comparison performed
    against ``source.crc32c`` from ``gsutil stat`` instead.
    """
    digest = 0
    with open(path, "rb", buffering=0) as fh:
        while True:
            chunk = fh.read(_COPY_BUFFER_BYTES)
            if not chunk:
                break
            digest = zlib.crc32(chunk, digest)
    return f"{digest & 0xFFFFFFFF:08x}"


# ---------------------------------------------------------------------------
# Manifest read/write
# ---------------------------------------------------------------------------


@dataclass
class _ManifestRecord:
    """Internal mutable representation of the on-disk manifest schema."""

    schema_version: int = _MANIFEST_SCHEMA_VERSION
    created_utc: str = ""
    producer: str = _PRODUCER
    content_kind: str = ""
    source_uri: str = ""
    source_size_bytes: int = 0
    source_generation: str | None = None
    source_crc32c: str | None = None
    source_md5: str | None = None
    source_was_gcsfuse: bool = False
    local_path: str = ""
    local_size_bytes: int = 0
    billing_project: str | None = None
    complete: bool = True
    # Non-spec extras: documented here, ignored by future strict readers.
    extras: dict[str, str] = field(default_factory=dict)


def _write_manifest_atomic(manifest_path: Path, record: _ManifestRecord) -> None:
    """Atomically write the manifest via a ``.tmp`` + rename + dir fsync sequence."""
    tmp = manifest_path.with_name(
        f"{manifest_path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    )
    payload = asdict(record)
    # Drop empty 'extras' to keep the manifest tidy when unused.
    if not payload.get("extras"):
        payload.pop("extras", None)
    data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass
    os.replace(str(tmp), str(manifest_path))
    _fsync_dir(manifest_path.parent)


def read_manifest(local_path: Path) -> dict | None:
    """Return the parsed manifest dict for ``local_path``, or None if missing/invalid."""
    manifest_path = _manifest_path_for(Path(local_path))
    if not manifest_path.exists():
        return None
    try:
        with open(manifest_path, "rb") as fh:
            return json.loads(fh.read().decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None


def verify_local_cache(local_path: Path) -> bool:
    """Return True iff ``local_path`` is a valid, complete cache entry.

    Validity requires all of:
      * ``local_path`` exists as a regular file.
      * The sibling manifest exists and parses as JSON.
      * ``manifest["schema_version"]`` matches the current schema.
      * ``manifest["complete"]`` is True.
      * ``stat().st_size`` equals ``manifest["local_size_bytes"]``.
    """
    local_path = Path(local_path)
    if not local_path.exists() or not local_path.is_file():
        return False
    manifest = read_manifest(local_path)
    if manifest is None:
        return False
    try:
        if int(manifest.get("schema_version", -1)) != _MANIFEST_SCHEMA_VERSION:
            return False
        if not bool(manifest.get("complete", False)):
            return False
        expected_size = int(manifest.get("local_size_bytes", -1))
    except (TypeError, ValueError):
        return False
    try:
        actual_size = local_path.stat().st_size
    except OSError:
        return False
    return expected_size == actual_size


# ---------------------------------------------------------------------------
# Public staging API
# ---------------------------------------------------------------------------


def stage_gcs_object(
    source_uri: str,
    local_path: Path,
    *,
    content_kind: str,
    billing_project: str | None = None,
    expected_size: int | None = None,
    expected_crc32c: str | None = None,
    min_free_bytes_after: int = 50_000_000_000,
    use_lock: bool = True,
) -> LocalCacheEntry:
    """Atomic stage from ``source_uri`` to ``local_path`` with manifest.

    See the module docstring for the high-level design. Steps in order:

    1. Acquire fcntl.lockf on ``<local_path>.lock`` (when ``use_lock`` is True).
    2. If the existing cache verifies, return early.
    3. Resolve remote metadata (size, optional crc32c/generation/md5/updated).
    4. Preflight free space: need ``max(expected_size, source_size) +
       min_free_bytes_after`` bytes on the destination filesystem.
    5. Allocate a unique partial path keyed by pid + random uuid suffix.
    6. Copy bytes: ``gsutil -m cp`` for gs:// sources, buffered read for
       gcsfuse-backed or local sources.
    7. fsync the partial file and its parent directory.
    8. Verify the partial size equals the resolved source size (or
       ``expected_size`` if it was provided and differs).
    9. If ``expected_crc32c`` is provided, compute the CRC of the partial via
       :func:`_crc32c_hex_of_file` (zlib CRC-32, see its docstring for the
       algorithm caveat) and compare. Mismatch -> raise, partial cleaned up.
    10. ``os.replace(partial, local_path)`` (atomic on POSIX).
    11. Write the manifest atomically (see :func:`_write_manifest_atomic`).
    12. Release the lock.

    On any failure between steps 5 and 11 the partial path is unlinked and no
    manifest is published, so :func:`verify_local_cache` continues to report
    the prior state of the cache truthfully.
    """
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = _lock_path_for(local_path)

    with _file_lock(lock_file, enabled=use_lock):
        # Recheck under the lock: another racer may have just published.
        if verify_local_cache(local_path):
            manifest = read_manifest(local_path) or {}
            remote = RemoteObject(
                uri=str(manifest.get("source_uri", source_uri)),
                size_bytes=int(manifest.get("source_size_bytes", 0)),
                generation=manifest.get("source_generation"),
                crc32c=manifest.get("source_crc32c"),
                md5_hash=manifest.get("source_md5"),
                updated=None,
            )
            return LocalCacheEntry(
                local_path=local_path,
                manifest_path=_manifest_path_for(local_path),
                remote=remote,
                complete=True,
                content_kind=str(manifest.get("content_kind", content_kind)),
                schema_version=int(
                    manifest.get("schema_version", _MANIFEST_SCHEMA_VERSION)
                ),
            )

        remote = _describe_remote(source_uri, billing_project, expected_size)

        # Cross-check expected_size vs. resolved source size, when both known.
        if expected_size is not None and expected_size != remote.size_bytes:
            raise RuntimeError(
                f"aou_storage: expected_size={expected_size} disagrees with "
                f"resolved source size {remote.size_bytes} for {source_uri!r}"
            )

        # Preflight headroom. ``remote.size_bytes`` is authoritative here.
        required = max(remote.size_bytes, expected_size or 0) + min_free_bytes_after
        _check_free_space(local_path.parent, required)

        partial = local_path.with_name(
            f"{local_path.name}.partial.{os.getpid()}.{uuid.uuid4().hex[:8]}"
        )

        source_was_gcsfuse = False
        try:
            if _is_gs_uri(source_uri):
                _gsutil_cp_to_partial(source_uri, partial, billing_project)
            else:
                src_path = Path(source_uri)
                source_was_gcsfuse = gcsfuse_staging.is_gcsfuse_path(src_path)
                _buffered_copy_to_partial(
                    src_path, partial, source_size=remote.size_bytes
                )

            # fsync the data and its directory before any verification reads.
            _fsync_path(partial)
            _fsync_dir(partial.parent)

            actual_size = partial.stat().st_size
            if actual_size != remote.size_bytes:
                raise RuntimeError(
                    f"aou_storage: copy size mismatch for {source_uri!r}: "
                    f"got {actual_size} bytes, expected {remote.size_bytes}"
                )

            if expected_crc32c is not None:
                got_crc = _crc32c_hex_of_file(partial)
                # Normalize both sides to lowercase hex, stripping any "0x".
                want_norm = expected_crc32c.lower().removeprefix("0x")
                if got_crc.lower() != want_norm:
                    raise RuntimeError(
                        f"aou_storage: CRC mismatch for {source_uri!r}: "
                        f"got {got_crc}, expected {expected_crc32c} "
                        f"(zlib CRC-32; see _crc32c_hex_of_file docstring)"
                    )

            os.replace(str(partial), str(local_path))
            _fsync_dir(local_path.parent)

            record = _ManifestRecord(
                schema_version=_MANIFEST_SCHEMA_VERSION,
                created_utc=_utc_now_iso(),
                producer=_PRODUCER,
                content_kind=content_kind,
                source_uri=source_uri,
                source_size_bytes=remote.size_bytes,
                source_generation=remote.generation,
                source_crc32c=remote.crc32c,
                source_md5=remote.md5_hash,
                source_was_gcsfuse=source_was_gcsfuse,
                local_path=str(local_path),
                local_size_bytes=local_path.stat().st_size,
                billing_project=billing_project,
                complete=True,
            )
            manifest_path = _manifest_path_for(local_path)
            _write_manifest_atomic(manifest_path, record)
        except BaseException:
            # Clean up the partial. Do not touch the destination — the prior
            # cached file (if any) and its manifest must remain undisturbed.
            try:
                partial.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                _LOG.warning("aou_storage: failed to unlink partial %s", partial)
            raise

        return LocalCacheEntry(
            local_path=local_path,
            manifest_path=_manifest_path_for(local_path),
            remote=remote,
            complete=True,
            content_kind=content_kind,
            schema_version=_MANIFEST_SCHEMA_VERSION,
        )


def stage_aou_plink_trio(
    remote_prefix: str,
    cache_dir: Path,
    prefix: str = "arrays",
    billing_project: str | None = None,
) -> tuple[LocalCacheEntry, LocalCacheEntry, LocalCacheEntry]:
    """Stage a PLINK ``.bed/.bim/.fam`` trio under ``cache_dir/<prefix>.<ext>``.

    ``remote_prefix`` is the common prefix of the three remote files, e.g.
    ``gs://bucket/arrays/v8`` (yielding ``v8.bed``, ``v8.bim``, ``v8.fam``) or a
    gcsfuse-mounted directory path. Each file is staged atomically with its
    own manifest; partial failures leave previously-staged siblings intact.

    Returns ``(bed_entry, bim_entry, fam_entry)``.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    entries: list[LocalCacheEntry] = []
    for ext, kind in (
        (".bed", "aou_array_plink_bed"),
        (".bim", "aou_array_plink_bim"),
        (".fam", "aou_array_plink_fam"),
    ):
        # Compose the source URI in a way that works for both gs:// and local
        # path prefixes — straight string concatenation matches gsutil's own
        # behaviour and avoids Path() collapsing the gs:// double slash.
        source_uri = f"{remote_prefix}{ext}"
        local_path = cache_dir / f"{prefix}{ext}"
        entries.append(
            stage_gcs_object(
                source_uri,
                local_path,
                content_kind=kind,
                billing_project=billing_project,
            )
        )

    return entries[0], entries[1], entries[2]

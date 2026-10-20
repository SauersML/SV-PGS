"""Complete local-file identities; mutable file metadata is not a content certificate."""

from __future__ import annotations

import functools
import hashlib
import json
import os
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sv_pgs.phase_timer import GB, timing

GCS_MOUNTS = "SVPGS_GCS_MOUNTS"
"""The environment variable naming the Cloud Storage buckets mounted on this machine: a JSON object from mount path to
the ``gs://bucket[/prefix]`` it shows (launcher/workspace sets it for its gcsfuse mounts). A file under one is identified
by its object's generation, an immutable generation contract the storage provider supplies: GCS gives every write of an
object a new generation and never reuses one for other content."""


def _stamp(path: Path) -> tuple[int, ...]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _hashed(path: str, stamp: tuple[int, ...]) -> str:
    with open(path, "rb") as handle:
        digest = hashlib.file_digest(handle, "sha256")
    if _stamp(Path(path)) != stamp:
        raise RuntimeError(f"{path} changed while its content was hashed")
    return digest.hexdigest()


def file_digest(path: str | Path) -> str:
    """Hash every byte on each call. Size, mtime and ctime alone cannot certify contents.

    Immutable object generations may replace this pass when explicitly supplied by the
    storage provider; mutable local files cannot safely use that optimization. A file on a
    mounted bucket (GCS_MOUNTS) is identified by its object's generation and checksum.
    """
    path = Path(path).resolve()
    located = _mounted_object(path)
    if located is not None:
        return _generation_identity(*located)
    return _hashed(str(path), _stamp(path))


def _mounted_object(path: Path) -> tuple[str, str] | None:
    """The (bucket, object name) of a file under one of GCS_MOUNTS, or None."""
    for mount, url in json.loads(os.environ.get(GCS_MOUNTS) or "{}").items():
        if path.is_relative_to(Path(mount)):
            bucket, _, prefix = url.removeprefix("gs://").partition("/")
            return bucket, "/".join(part for part in (prefix.strip("/"), path.relative_to(mount).as_posix()) if part)
    return None


@functools.cache
def _storage_client():
    from google.cloud import storage

    return storage.Client(project=os.environ.get("GOOGLE_PROJECT"))


def _generation_identity(bucket: str, name: str) -> str:
    """The object's generation, size and CRC32C, read from its metadata (billed to GOOGLE_PROJECT, as a requester-pays
    bucket such as the CDR's requires)."""
    blob = _storage_client().bucket(bucket, user_project=os.environ.get("GOOGLE_PROJECT")).get_blob(name)
    if blob is None:
        raise FileNotFoundError(f"gs://{bucket}/{name} does not exist")
    return f"gs://{bucket}/{name}#{blob.generation}:{blob.size}:{blob.crc32c}"


def tree_digest(root: str | Path) -> str:
    """Bind paths and complete contents, including samples, annotations and genotype shards.

    Hash files concurrently but do not reuse mutable metadata as a content identity.
    In particular, a guest kernel clock does not certify host/remote filesystem ctime
    granularity or clock alignment. Reuse requires an immutable generation contract.

    Each pass ends with a ``timing: tree_digest`` line (``phase_timer``): its files, their bytes and the root's last two
    parts, as ``store/`` under a step (``artifact.store_digest``: the All of Us whole-genome store, 547 GB, took about
    12 minutes a start [real]).
    """
    started = time.perf_counter()
    root = Path(root).resolve()
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "CACHE_COMPLETE.json")
    stamps = [_stamp(path) for path in paths]
    with ThreadPoolExecutor(thread_name_prefix="tree-digest") as pool:
        contents = list(pool.map(lambda pair: _hashed(str(pair[0]), pair[1]), zip(paths, stamps)))
    digest = hashlib.sha256()
    for path, content in zip(paths, contents):
        name = path.relative_to(root).as_posix().encode()
        digest.update(struct.pack("<Q", len(name)))
        digest.update(name)
        digest.update(bytes.fromhex(content))
    timing("tree_digest", time.perf_counter() - started, files=len(paths), gb=sum(stamp[2] for stamp in stamps) / GB, root="/".join(root.parts[-2:]))
    return digest.hexdigest()

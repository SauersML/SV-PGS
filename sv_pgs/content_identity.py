"""Complete local-file identities; mutable file metadata is not a content certificate."""

from __future__ import annotations

import functools
import hashlib
import json
import os
import struct
import time
import uuid
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sv_pgs.phase_timer import GB, timing

GCS_MOUNTS = "SVPGS_GCS_MOUNTS"
"""The environment variable naming the Cloud Storage buckets mounted on this machine: a JSON object from mount path to
the ``gs://bucket[/prefix]`` it shows (launcher/workspace sets it for its gcsfuse mounts). A file under one is identified
by its object's generation, an immutable generation contract the storage provider supplies: GCS gives every write of an
object a new generation and never reuses one for other content."""
RESTORED_DIGESTS = "SVPGS_RESTORED_DIGESTS"
"""The environment variable naming a directory on this machine's own disk for restored digests (launcher/workspace sets
it beside each job's new disk): the SHA-256 of every byte the run mirror's restore wrote to a file, hashed as the bytes
reached the disk (``run_mirror.restore``: no second read), each beside the file's stamp (``_stamp``) when its download
closed it. Each byte is still hashed once on this machine: by the restore that wrote it rather than by a pass
that reads it back, which for the All of Us whole-genome store was 547 GB and 12 minutes at every job start [real].
Unset, every digest reads every byte, as before."""
DIGESTS_FORMAT = "svpgs-restored-digests-v1"


def _stamp(path: Path) -> tuple[int, ...]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _hashed(path: str, stamp: tuple[int, ...]) -> str:
    with open(path, "rb") as handle:
        digest = hashlib.file_digest(handle, "sha256")
    if _stamp(Path(path)) != stamp:
        raise RuntimeError(f"{path} changed while its content was hashed")
    return digest.hexdigest()


def record_restored(digests: Iterable[tuple[Path, str]]) -> None:
    """Keep each (file, SHA-256 of the bytes the restore wrote to it) in RESTORED_DIGESTS, beside the file's stamp now,
    as its download has just closed it; nothing where RESTORED_DIGESTS is unset. Each call writes its own record, aside
    and then renamed, so a reader sees a whole one or none."""
    directory = os.environ.get(RESTORED_DIGESTS)
    if not directory:
        return
    files = [[*_stamp(Path(path)), digest] for path, digest in digests]
    if not files:
        return
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    name = uuid.uuid4().hex
    temporary = target / f".{name}.json"
    temporary.write_text(json.dumps({"format": DIGESTS_FORMAT, "files": files}), encoding="utf-8")
    os.replace(temporary, target / f"{name}.json")


def _restored_digests() -> dict[tuple[int, ...], str]:
    """Every restored digest by the stamp its file had when the restore closed it. A stamp two records give different
    digests is in neither: such a file is hashed."""
    directory = os.environ.get(RESTORED_DIGESTS)
    if not directory or not Path(directory).is_dir():
        return {}
    found: dict[tuple[int, ...], str] = {}
    disputed: set[tuple[int, ...]] = set()
    for path in sorted(Path(directory).glob("[!.]*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(record, dict) or record.get("format") != DIGESTS_FORMAT:
            continue
        for *stamp, digest in record["files"]:
            key = tuple(int(value) for value in stamp)
            if found.setdefault(key, digest) != digest:
                disputed.add(key)
    for key in disputed:
        del found[key]
    return found


def _content(path: Path, stamp: tuple[int, ...], restored: dict[tuple[int, ...], str]) -> str:
    """The SHA-256 of a local file: its restored digest while the file keeps the stamp the restore left, else every byte
    read now. A restored digest certifies content because the restore hashed each byte it wrote; the stamp only ties it
    to this file. The device and inode name the file on this disk, and the kernel moves its size, modification
    and change times on any write or truncation after the restore closed it (a time no call can set back is the change
    time), so a rewritten file is read again. The driver never writes a completed step's files: it replaces a step by
    renaming it aside and making a new one, whose files are new inodes. The fit trusts those files to stay unchanged
    for the hours it reads them after any digest, as before."""
    known = restored.get(stamp)
    return known if known is not None else _hashed(str(path), stamp)


def file_digest(path: str | Path) -> str:
    """Hash every byte on each call. Size, mtime and ctime alone cannot certify contents.

    Immutable object generations may replace this pass when explicitly supplied by the
    storage provider; mutable local files cannot safely use that optimization. A file on a
    mounted bucket (GCS_MOUNTS) is identified by its object's generation and checksum. A file
    the run mirror restored on this machine is identified by the digest of the bytes the restore
    wrote to it, while it is unchanged since (RESTORED_DIGESTS, ``_content``).
    """
    path = Path(path).resolve()
    located = _mounted_object(path)
    if located is not None:
        return _generation_identity(*located)
    return _content(path, _stamp(path), _restored_digests())


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
    granularity or clock alignment. Reuse requires an immutable generation contract, or a
    restored digest: the digest of the bytes this machine's restore wrote (``_content``).

    Each pass ends with a ``timing: tree_digest`` line (``phase_timer``): its files, their bytes and the root's last two
    parts, as ``store/`` under a step (``artifact.store_digest``: the All of Us whole-genome store, 547 GB, took about
    12 minutes a start [real]).
    """
    started = time.perf_counter()
    root = Path(root).resolve()
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "CACHE_COMPLETE.json")
    stamps = [_stamp(path) for path in paths]
    restored = _restored_digests()
    with ThreadPoolExecutor(thread_name_prefix="tree-digest") as pool:
        contents = list(pool.map(lambda pair: _content(pair[0], pair[1], restored), zip(paths, stamps)))
    digest = hashlib.sha256()
    for path, content in zip(paths, contents):
        name = path.relative_to(root).as_posix().encode()
        digest.update(struct.pack("<Q", len(name)))
        digest.update(name)
        digest.update(bytes.fromhex(content))
    timing("tree_digest", time.perf_counter() - started, files=len(paths), gb=sum(stamp[2] for stamp in stamps) / GB, root="/".join(root.parts[-2:]))
    return digest.hexdigest()

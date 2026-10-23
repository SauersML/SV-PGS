"""The run directory's copy in the workspace bucket, which carries a run from one Google Batch job to the next.

Batch attaches an existing persistent disk without write access ("All options except writing are supported", Batch's
storage guide, https://docs.cloud.google.com/batch/docs/create-run-job-storage), so no disk can carry a run between
jobs. Instead, each job works on a new disk. ``restore`` fills that disk from the mirror before the driver starts.
``push`` copies the run's durable state back while the driver runs and once more after it exits (launcher/workspace).

The durable state is what the driver resumes from:
- **A completed step's directory** (``<step>/`` with COMPLETE_FILE). The driver never changes one once it is renamed
  into place. It is pushed whole with COMPLETE_FILE last, so a mirrored step without one is an interrupted push, and
  is never restored.
- **The fit step's saved models in ``fit.partial``:** each saved (arm, trait) model with its record, the record last,
  as the fit step saves them, and the partial's KEY_FILE. (Engine checkpoints written there before the restart store
  are keyed by the whole package's source, which no current fit can match: they are neither pushed nor restored.)
- **The fit's restart units in ``restart/<arm>/``** (``restart_store``; RESTART), outside every step, so an edit that
  sets the fit step aside keeps them: a sealed unit (Stage 0 with its LD store, the far-field scale, the band, a
  block's codes) is pushed payload first and its manifest last, chunked in parallel, and a whole remote copy is never
  replaced, so a push cut short leaves the last whole copy or none; a mutable unit's self-checking files (an EB start,
  a completed model's checkpoint, a mode search's sources) are each replaced whole, sent again when their size or
  modification time changes (RECEIPT), a mode search's index (``restart_store.MODE_INDEX``) read first and sent last.
  A unit the fit released is removed from the mirror, but only by a disk that restored that build of it or
  saw it whole there while it held it (RECEIPT), so a job never removes a unit it never held: another job's, another
  build of the same key, one a failed restore left behind, or a region's it never restored.
- **The partial steps' chromosome units:** each converted chromosome's files (STORE_CHROMOSOME_FILES) with its record
  chromosomes/<chromosome>.json, the record last, as the store and direct_sv steps write them; and the KEY_FILE. A
  fan-out of store shards (``workspace_pipeline.run_pipeline``'s ``store_shard``), one job's tasks on separate disks,
  meets here: each pushes its own chromosomes under the one key, and the next job restores them all. A direct_sv
  chromosome is whole only with its rows/<chromosome>.npz, which mirrors before this one did not carry: one without
  it is made again, never restored half.
- **The score and effects steps' completed batches:** each batch directory with its content-digest record last,
  and the partial's KEY_FILE. Hidden batches are still being written. A completed step carries its final outputs
  alone; its batch copies are needed only while it is partial.
- **Each region's run directory** (``WorkspaceConfig.fit_regions``), REGIONS_DIRECTORY/<name>: its own steps from the
  fit on, mirrored as a run's under the same name. Its steps before the fit are links to the run's own, which the run
  mirrors once.

Any other partial work is redone, as after a preemption on one disk. GCS creates an object only when its upload
completes, so a mirrored file is whole or absent.

Two runs that fit one arm each (``workspace_pipeline.run_pipeline``'s ``fit_arms``) each leave their arm's parts in
their partials: its saved models in the fit's, its score batches and their record in the score step's, its effects
batches and, last, its handoff (``workspace_pipeline.arm_handoff``) in the effects step's. ``adopt`` brings the other
run's into this one from that run's mirror, so either run finishes the steps from the fit on (``run_pipeline``'s
``finish``) with no store; a finishing job elsewhere first restores only what that reads (``restore``'s ``finishing``):
the summaries of the steps before the fit, the cohort step whole, and the parts, a few GB where the whole run is a TB.

The store, measurement and direct_sv steps each write a full copy of the genotype codes (``<step>/store``, 1.7 TB at
the design workload), and only the next of them reads it: the fit, score, effects and report steps read the last one
(``workspace_pipeline._final_store``). Where the job mounts the mirror's bucket (content_identity.GCS_MOUNTS, as
launcher/workspace does), a restore links each store a later completed step's replaces to its mirrored copy on the
mount instead of downloading it, so a job start fetches one copy, not three. A step made again that reads a linked
store reads it through the mount.

A run can name base mirrors (``Base``): other runs' mirrors that carry its completed steps before the fit (CARRIED; the
samples, the store and the steps that recalibrate it, which runs of other traits or other fit code share by key). A
restore takes such a step from the first base that holds it whole where this run's mirror holds none of it, and a push
does not send a completed step a base holds whole under the same COMPLETE_FILE; it removes this mirror's own stale copy
and names the base instead (BASES), which later jobs of the run read as a base, as they do the bases a base names
(``bases_of``). So a new run's mirror needs no copy of the 0.5-1.5 TB store chain. A base is only read: a run never
writes or deletes there, and it must be kept while the runs it carries are.

A restore hashes every file's bytes as it writes them and keeps the digests on the job's own disk
(``content_identity.RESTORED_DIGESTS``), so a digest of a restored store (the fit's ``artifact.store_digest``) reads none
of it again.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import multiprocessing
import os
import shutil
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Protocol

import google_crc32c

from sv_pgs.content_identity import GCS_MOUNTS, record_restored
from sv_pgs.postfit_checkpoint import RECORD
from sv_pgs.restart_store import FORMAT, KINDS, MANIFEST, MODE_INDEX, RestartStore
from sv_pgs.workspace_pipeline import (
    ARM_PARTS, ARMS, COMPLETE_FILE, FINISH_INPUTS, HANDOFF, KEY_FILE, REGIONS_DIRECTORY, RESTART_DIRECTORY, STEP_NAMES, arm_handoff,
)

LOGGER = logging.getLogger(__name__)
PARTIAL = ".partial"
RESTORING = ".restoring"
RESTART = RESTART_DIRECTORY
"""The run's restart store, restart/<arm>/<kind>/<key>/ (``restart_store``)."""
RECEIPT = ".pushed.json"
"""In the run's restart directory: each mutable unit file's (size, modification time) when it was last sent or
restored, by object name; a file whose stamp differs is sent again. And by unit (its prefix), the build id of the
mirrored copy the disk restored or saw whole there while it held the unit: once the disk no longer holds the unit, a
push removes the mirrored copy only where it is still that build, and forgets the unit."""
COMPOSE_SOURCES = 32
"""Cloud Storage's compose: at most 32 source objects into one (cloud.google.com/storage/docs/composite-objects)."""
COMPOSING = ".composing"
"""Where a parallel composite upload's parts wait for their compose, under the mirror's root and outside every step's
and the restart store's names, so a part a killed push leaves behind is never read as a file of either."""
DIRECTORIES = ".directories.json"
"""Beside a mirrored step, its directories, which GCS keeps only as the prefixes of objects: an empty one is
restored from this list."""
RECORDS = "models/*/*.json"
"""The fit step's saved-model records in its partial: models/<arm>/<trait>.json, written after models/<arm>/<trait>."""
STORE = "store"
"""A step's store payload, <step>/store: a full copy of the genotype codes."""
STORE_RECORDS = "chromosomes/*.json"
"""The store step's record of each chromosome it converted, in its partial: chromosomes/<chromosome>.json, written
after the chromosome's files."""
BATCH_RECORDS = "batches/*/complete.json"
"""The score/effects partial's committed batches (``postfit_checkpoint``); each record binds its files' content to
the step key and ordered batch identity. The generation in a repaired batch's record makes the mirror send it again."""
STORE_CHROMOSOME_FILES = ("fill", "reported", "calibration", "rows")
"""The store and direct_sv steps' per-chromosome files beside their store, <directory>/<chromosome>.<suffix>
(direct_sv's direct rows, rows/<chromosome>.npz, among them); inside store/, a
chromosome's files are those under a directory named for it (dosage/half<i>/<chromosome>, stats/half<i>/<chromosome>,
variants/<chromosome>, loci/<chromosome>)."""
CARRIED = STEP_NAMES[: STEP_NAMES.index("fit")]
"""The steps a base mirror can carry for a run: those before the fit. The fit and the steps after it are each run's own
results, which a run of other traits or code would only download to set aside."""
BASES = ".bases"
"""In a run's mirror, an empty object .bases/<bucket>/<prefix> for each base gs://<bucket>/<prefix> that carries one of
its completed steps, so a listing of the bucket shows which runs a base must outlive
(launcher/workspace/mirror_sizes.py)."""


def _of_chromosome(relative: str, chromosome: str) -> bool:
    """Whether the store partial's file ``relative`` ('/'-separated) is one of ``chromosome``'s."""
    parts = relative.split("/")
    if parts[0] == STORE:
        return chromosome in parts[1:-1]
    return len(parts) == 2 and parts[0] in STORE_CHROMOSOME_FILES and parts[1].split(".", 1)[0] == chromosome


class Remote(Protocol):
    """The mirror: objects named by their path relative to the run directory, '/'-separated. A download returns the
    SHA-256 of the bytes it wrote (``_HashingFile``), or None where it cannot say."""

    def sizes(self, prefix: str) -> dict[str, int]: ...

    def read(self, name: str) -> bytes | None: ...

    def upload(self, path: Path, name: str) -> None: ...

    def upload_parts(self, path: Path, name: str) -> None: ...

    def write(self, name: str, content: bytes) -> None: ...

    def download(self, name: str, path: Path) -> str | None: ...

    def download_parts(self, name: str, path: Path) -> None: ...

    def download_many(self, pairs: list[tuple[Path, str]]) -> list[str | None] | None: ...

    def version(self, name: str) -> Version: ...

    def download_range(self, name: str, path: Path, start: int, stop: int, generation: int | None) -> None: ...

    def delete(self, names: Iterable[str]) -> None: ...


@dataclass(frozen=True)
class Version:
    """An object as a ranged read takes it: its size, the generation every range is read from (None where the mirror
    keeps none) and its CRC32C, big-endian, which the bytes written are checked against (``_download_pieces``)."""

    size: int
    generation: int | None
    crc32c: bytes


@dataclass(frozen=True)
class Base:
    """Another run's mirror at ``url`` that carries completed steps (CARRIED) for this run, which only reads it;
    ``mount`` is where it shows on this machine (``mounted``)."""

    remote: Remote
    url: str
    mount: Path | None = None

    @property
    def reference(self) -> str:
        """Its name in a mirror it carries steps for (BASES)."""
        return f"{BASES}/{self.url.split('://', 1)[-1].strip('/')}"


class GcsRemote:
    """A ``gs://bucket/prefix`` mirror, billed to GOOGLE_PROJECT when it is set (as in the workspace's jobs)."""

    def __init__(self, url: str) -> None:
        from google.cloud import storage

        if not url.startswith("gs://"):
            raise ValueError(f"the mirror is a gs:// URL, not {url!r}.")
        bucket, _, prefix = url.removeprefix("gs://").partition("/")
        project = os.environ.get("GOOGLE_PROJECT")
        self._client = storage.Client(project=project)
        self._bucket = self._client.bucket(bucket, user_project=project)
        self._root = prefix.strip("/") + "/" if prefix.strip("/") else ""
        self._url = url

    def sizes(self, prefix: str) -> dict[str, int]:
        return {
            blob.name.removeprefix(self._root): int(blob.size)
            for blob in self._client.list_blobs(self._bucket, prefix=self._root + prefix)
        }

    def read(self, name: str) -> bytes | None:
        from google.api_core.exceptions import NotFound

        try:
            return self._bucket.blob(self._root + name).download_as_bytes()
        except NotFound:
            return None

    def upload(self, path: Path, name: str) -> None:
        self._bucket.blob(self._root + name).upload_from_filename(str(path))

    def upload_parts(self, path: Path, name: str) -> None:
        """``upload`` in parts sent at once, then composed into one object (a parallel composite upload): one stream
        sent a restart unit's 150 GB LD store at one connection's rate. The file's bytes are split into up to
        COMPOSE_SOURCES ranges of at least the library's default chunk, each sent to its own object under COMPOSING
        from its own process with its own client, not a thread, whose per-chunk Python work shares one interpreter lock
        (``download_many``); compose writes the object from them, whole or absent like any other, and the parts are
        deleted. Cloud Storage's XML multipart upload, which this replaced, needs storage.multipartUploads.create,
        which the All of Us workspace's pet service account does not have: every restart push of rounds m1 and m2 was
        refused with 403 (AccessDenied), so no restart unit reached the mirror. Compose needs only the object
        permissions every other push uses. A composed object carries a CRC32C and no MD5, and ``download_parts``
        checks the CRC32C."""
        from google.cloud.storage import transfer_manager

        size = path.stat().st_size
        if size <= transfer_manager.TM_DEFAULT_CHUNK_SIZE:
            self.upload(path, name)
            return
        count = min(COMPOSE_SOURCES, -(-size // transfer_manager.TM_DEFAULT_CHUNK_SIZE))
        edges = [size * index // count for index in range(count + 1)]
        parts = [f"{COMPOSING}/{name}.part{index}" for index in range(count)]
        # spawn, not fork: a forked child would share the parent client's connections
        with ProcessPoolExecutor(max_workers=min(count, len(os.sched_getaffinity(0))), mp_context=multiprocessing.get_context("spawn")) as pool:
            list(pool.map(_upload_range_in_process, repeat(self._url), repeat(path), edges[:-1], edges[1:], parts))
        sources = [self._bucket.blob(self._root + part) for part in parts]
        self._bucket.blob(self._root + name).compose(sources)
        self._bucket.delete_blobs(sources)

    def upload_range(self, path: Path, start: int, stop: int, name: str) -> None:
        """Bytes ``start`` to ``stop`` of the file at ``path`` as the object ``name``."""
        from google.cloud.storage.retry import DEFAULT_RETRY

        with path.open("rb") as handle:
            self._bucket.blob(self._root + name).upload_from_file(_FileRange(handle, start, stop), size=stop - start, retry=DEFAULT_RETRY)

    def write(self, name: str, content: bytes) -> None:
        self._bucket.blob(self._root + name).upload_from_string(content)

    def download(self, name: str, path: Path) -> str | None:
        from google.cloud.storage.retry import DEFAULT_RETRY

        path.parent.mkdir(parents=True, exist_ok=True)
        # A broken read resumes from its last byte (a range request) for as long as it keeps failing transiently: under
        # the client's default 120 s deadline, one stalled stream among a restore's parallel downloads ended the AoU
        # whole-genome restore an hour and 547 GB in ("Connection broken: IncompleteRead"). The client writes the bytes
        # in order, a resumed read after the last one written, so they are hashed as they are written.
        with open(path, "wb") as handle:
            written = _HashingFile(handle)
            self._bucket.blob(self._root + name).download_to_file(written, retry=DEFAULT_RETRY.with_timeout(None))
        return written.hexdigest()

    def download_parts(self, name: str, path: Path) -> None:
        """``download`` in ranges read at once, one process per CPU this process may use, each with its own client, the
        whole file's CRC32C checked against the object's from the ranges' own: one stream read a restart unit's 150 GB
        LD store at one connection's rate."""
        from google.cloud.storage import transfer_manager
        from google.cloud.storage.retry import DEFAULT_RETRY

        blob = self._bucket.blob(self._root + name)
        blob.reload()
        if blob.size <= transfer_manager.TM_DEFAULT_CHUNK_SIZE:
            self.download(name, path)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        transfer_manager.download_chunks_concurrently(
            blob, str(path), download_kwargs={"retry": DEFAULT_RETRY.with_timeout(None)}, worker_type=transfer_manager.PROCESS,
            max_workers=len(os.sched_getaffinity(0)), crc32c_checksum=True,
        )

    def download_many(self, pairs: list[tuple[Path, str]]) -> list[str | None]:
        """``download`` each (local path, object name) pair, in one process per CPU this process may use, each with its
        own client: in one process the client's per-chunk Python work shares one interpreter lock, and a thread per CPU
        restored the AoU whole-genome run's 547 GB at 140 MB/s, 65 minutes of every job start."""
        if not pairs:
            return []
        # spawn, not fork: a forked child would share the parent client's connections
        with ProcessPoolExecutor(max_workers=len(os.sched_getaffinity(0)), mp_context=multiprocessing.get_context("spawn")) as pool:
            return list(pool.map(_download_in_process, repeat(self._url), [name for _, name in pairs], [path for path, _ in pairs]))

    def version(self, name: str) -> Version:
        blob = self._bucket.get_blob(self._root + name)
        if blob is None:
            raise FileNotFoundError(f"{self._url}/{name} left the mirror after it was listed; restore again.")
        return Version(int(blob.size), int(blob.generation), base64.b64decode(blob.crc32c))

    def download_range(self, name: str, path: Path, start: int, stop: int, generation: int | None) -> None:
        """Bytes ``start`` to ``stop`` of the object's ``generation`` written at the same offsets of the file at
        ``path``, which holds the object's size already. A broken read resumes from its last byte, as ``download``'s
        does; no checksum covers a range, so the caller checks the whole file's CRC32C (``_download_pieces``)."""
        from google.cloud.storage.retry import DEFAULT_RETRY

        with open(path, "r+b") as handle:
            written = _RangeWriter(handle, start, stop)
            self._bucket.blob(self._root + name, generation=generation).download_to_file(
                written, start=start, end=stop - 1, checksum=None, retry=DEFAULT_RETRY.with_timeout(None),
            )
            written.close()

    def delete(self, names: Iterable[str]) -> None:
        blobs = [self._bucket.blob(self._root + name) for name in names]
        if blobs:
            self._bucket.delete_blobs(blobs)


class _Within:
    """``remote``'s objects under ``prefix``, named relative to it: a region's run directory's mirror inside its run's."""

    def __init__(self, remote: Remote, prefix: str) -> None:
        self._remote, self._prefix = remote, prefix

    def sizes(self, prefix: str) -> dict[str, int]:
        return {name.removeprefix(self._prefix): size for name, size in self._remote.sizes(self._prefix + prefix).items()}

    def read(self, name: str) -> bytes | None:
        return self._remote.read(self._prefix + name)

    def upload(self, path: Path, name: str) -> None:
        self._remote.upload(path, self._prefix + name)

    def upload_parts(self, path: Path, name: str) -> None:
        self._remote.upload_parts(path, self._prefix + name)

    def write(self, name: str, content: bytes) -> None:
        self._remote.write(self._prefix + name, content)

    def download(self, name: str, path: Path) -> str | None:
        return self._remote.download(self._prefix + name, path)

    def download_parts(self, name: str, path: Path) -> None:
        self._remote.download_parts(self._prefix + name, path)

    def download_many(self, pairs: list[tuple[Path, str]]) -> list[str | None] | None:
        return self._remote.download_many([(path, self._prefix + name) for path, name in pairs])

    def version(self, name: str) -> Version:
        return self._remote.version(self._prefix + name)

    def download_range(self, name: str, path: Path, start: int, stop: int, generation: int | None) -> None:
        self._remote.download_range(self._prefix + name, path, start, stop, generation)

    def delete(self, names: Iterable[str]) -> None:
        self._remote.delete([self._prefix + name for name in names])


_PROCESS_REMOTES: dict[str, GcsRemote] = {}
"""A GcsRemote.download_many worker process's own client, one per mirror URL."""


def _process_remote(url: str) -> GcsRemote:
    if url not in _PROCESS_REMOTES:
        _PROCESS_REMOTES[url] = GcsRemote(url)
    return _PROCESS_REMOTES[url]


def _download_in_process(url: str, name: str, path: Path) -> str | None:
    return _process_remote(url).download(name, path)


def _download_range_in_process(url: str, name: str, path: Path, start: int, stop: int, generation: int | None) -> None:
    _process_remote(url).download_range(name, path, start, stop, generation)


def _upload_range_in_process(url: str, path: Path, start: int, stop: int, name: str) -> None:
    _process_remote(url).upload_range(path, start, stop, name)


class _FileRange:
    """Bytes ``start`` to ``stop`` of an open file as a stream of their own, its positions from 0: a resumable upload
    starts only from a stream's beginning ("Stream must be at beginning.") and seeks back within it to send a chunk
    again."""

    def __init__(self, handle, start: int, stop: int) -> None:
        self._handle, self._start, self._stop = handle, start, stop
        handle.seek(start)

    def read(self, size: int | None = -1) -> bytes:
        left = self._stop - self._handle.tell()
        return self._handle.read(max(0, left if size is None or size < 0 else min(size, left)))

    def tell(self) -> int:
        return self._handle.tell() - self._start

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        base = {os.SEEK_SET: self._start, os.SEEK_CUR: self._handle.tell(), os.SEEK_END: self._stop}[whence]
        self._handle.seek(base + offset)
        return self.tell()


class _RangeWriter:
    """Bytes ``start`` to ``stop`` of an open file, written front to back by one ranged read (``download_range``). The
    client seeks only to start a transcoded object again from its first byte (``_HashingFile``), which is the range's
    here. ``close`` checks that the range was filled exactly."""

    def __init__(self, handle, start: int, stop: int) -> None:
        self._handle, self._start, self._stop = handle, start, stop
        handle.seek(start)

    def write(self, data) -> int:
        if self._handle.tell() + len(data) > self._stop:
            raise OSError(f"a ranged read returned bytes past {self._stop}; restore again.")
        return self._handle.write(data)

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if offset != 0 or whence != os.SEEK_SET:
            raise io.UnsupportedOperation("a ranged read starts again only from its first byte")
        self._handle.seek(self._start)
        return 0

    def tell(self) -> int:
        return self._handle.tell() - self._start

    def flush(self) -> None:
        self._handle.flush()

    def close(self) -> None:
        self._handle.flush()
        if self._handle.tell() != self._stop:
            raise OSError(f"a ranged read wrote {self._handle.tell() - self._start} of {self._stop - self._start} bytes; restore again.")


class _HashingFile:
    """A file a download writes front to back, hashed as it is written: the SHA-256 of the bytes on the disk, with no
    second read (``content_identity.RESTORED_DIGESTS``). The client seeks only to start a transcoded object again from
    its first byte, which starts the digest again; after any other seek the digest is unknown (None)."""

    def __init__(self, handle) -> None:
        self._handle = handle
        self._digest = hashlib.sha256()

    def write(self, data) -> int:
        written = self._handle.write(data)
        if self._digest is not None:
            self._digest.update(data)
        return written

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        position = self._handle.seek(offset, whence)
        if position == 0:
            self._handle.truncate()
            self._digest = hashlib.sha256()
        else:
            self._digest = None
        return position

    def tell(self) -> int:
        return self._handle.tell()

    def flush(self) -> None:
        self._handle.flush()

    def hexdigest(self) -> str | None:
        return None if self._digest is None else self._digest.hexdigest()


def _transfer(pairs: list[tuple[Path, str]], move) -> list:
    """``move`` every (local path, object name) pair, on one thread per CPU this process may use; each one's result."""
    if not pairs:
        return []
    with ThreadPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as pool:
        return list(pool.map(lambda pair: move(*pair), pairs))


def _files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*") if path.is_file())


def _key_of(record: bytes | None) -> str | None:
    """The step key a COMPLETE_FILE's content names, or None."""
    try:
        return json.loads(record)["key"] if record is not None else None
    except (ValueError, KeyError, TypeError):
        return None


def _push_step(remote: Remote, final: Path, name: str, bases: Sequence[Base] = ()) -> Base | None:
    """A completed step, whole, its COMPLETE_FILE last; or, where a base holds it whole under the same COMPLETE_FILE,
    nothing, and that base is returned. A mirrored copy under the same key is the same step (made again from the same
    inputs and code): it is kept, never removed to be sent again, so the mirror always holds one."""
    marker = final / COMPLETE_FILE
    content = marker.read_bytes()
    batches = final / "batches" if name in ("score", "effects") else None
    files = [path for path in _files(final) if path != marker and (batches is None or not path.is_relative_to(batches))]
    held = remote.sizes(f"{name}/")
    mirrored = remote.read(f"{name}/{COMPLETE_FILE}")
    for base in bases if mirrored != content and name in CARRIED else ():
        if base.remote.read(f"{name}/{COMPLETE_FILE}") == content:
            carried = base.remote.sizes(f"{name}/")
            if all(carried.get(f"{name}/{path.relative_to(final).as_posix()}") == path.stat().st_size for path in files):
                if held:
                    LOGGER.info("run mirror: %s is carried by the base %s; this mirror's stale copy is removed", name, base.url)
                remote.delete(held)
                remote.delete(remote.sizes(f"{name}{PARTIAL}/"))
                return base
    if mirrored is not None and mirrored != content and _key_of(mirrored) == _key_of(content) and all(
        f"{name}/{path.relative_to(final).as_posix()}" in held for path in files
    ):
        # The same step made again (a disk whose restore did not reach it), whole on the mirror: kept, never removed to
        # be sent again, so the mirror always holds one.
        return None
    if mirrored == content:
        files = [path for path in files if held.get(f"{name}/{path.relative_to(final).as_posix()}") != path.stat().st_size]
        if not files:
            return None
        # A completion marker alone is not a whole step after an interrupted manual copy. Repair only its missing
        # companions, but withdraw that marker while doing so, as for a partial's saved-model unit.
        remote.delete([f"{name}/{COMPLETE_FILE}"])
    else:
        remote.delete(held)
    directories = sorted(path.relative_to(final).as_posix() for path in final.rglob("*") if path.is_dir() and (batches is None or not path.is_relative_to(batches)))
    remote.write(f"{name}{DIRECTORIES}", json.dumps(directories).encode())
    _transfer([(path, f"{name}/{path.relative_to(final).as_posix()}") for path in files], remote.upload)
    if marker.read_bytes() != content:
        raise FileNotFoundError(f"{final} was replaced while it was pushed.")
    remote.upload(marker, f"{name}/{COMPLETE_FILE}")
    remote.delete(remote.sizes(f"{name}{PARTIAL}/"))
    LOGGER.info("run mirror: pushed %s, %d files, %.1f GB", name, len(files) + 1, sum(path.stat().st_size for path in files) / 1e9)
    return None


def _partial_units(partial: Path) -> list[tuple[list[Path], Path]]:
    """The durable units in a step's ``partial``, each as (its files, the file that marks it whole, sent last): the fit
    step's every saved model directory models/<arm>/<trait> beside its record <trait>.json; the store and direct_sv
    steps' every converted chromosome's files beside its record chromosomes/<chromosome>.json; and the score/effects
    steps' committed batch directories beside their content-digest records."""
    units: list[tuple[list[Path], Path]] = []
    if (partial / "models").is_dir():
        for record in sorted(partial.glob(RECORDS)):
            model = record.with_suffix("")
            if model.is_dir():
                units.append((_files(model), record))
    records = sorted(partial.glob(STORE_RECORDS))
    if records:
        files = [(path, path.relative_to(partial).as_posix()) for path in _files(partial)]
        for record in records:
            units.append(([path for path, relative in files if _of_chromosome(relative, record.stem)], record))
    if partial.name in ("score.partial", "effects.partial"):
        for record in sorted(partial.glob(BATCH_RECORDS)):
            if not record.parent.name.startswith("."):
                units.append(([path for path in _files(record.parent) if path != record], record))
    return units


def _push_partial(remote: Remote, partial: Path, step: str) -> None:
    name = f"{step}{PARTIAL}"
    key = (partial / KEY_FILE).read_bytes()
    # The driver set the step's completed directory aside to make it again under another key, so the mirrored one is
    # stale: restored, it would take the partial's place. One under this partial's key is not (a disk whose restore
    # did not reach it): it is kept.
    if _key_of(remote.read(f"{step}/{COMPLETE_FILE}")) != key.decode():
        remote.delete(remote.sizes(f"{step}/"))
    if remote.read(f"{name}/{KEY_FILE}") != key:
        remote.delete(remote.sizes(f"{name}/"))
        remote.upload(partial / KEY_FILE, f"{name}/{KEY_FILE}")
    held = remote.sizes(f"{name}/")
    pushed = 0
    for files, marker in _partial_units(partial):
        marker_name = f"{name}/{marker.relative_to(partial).as_posix()}"
        if step in ("score", "effects") and _matches(marker.relative_to(partial).as_posix(), BATCH_RECORDS):
            # Capture the record before the payload. A repair may replace the directory during upload; never send
            # its new record after old files. A reader also checks each digest, rejecting any mixed generation.
            content = marker.read_bytes()
            if remote.read(marker_name) == content and all(
                held.get(f"{name}/{path.relative_to(partial).as_posix()}") == path.stat().st_size for path in files
            ):
                continue
            if marker_name in held:
                remote.delete([marker_name])
            _transfer([(path, f"{name}/{path.relative_to(partial).as_posix()}") for path in files], remote.upload)
            if marker.read_bytes() != content:
                raise FileNotFoundError(f"{marker} was replaced while it was pushed.")
            remote.write(marker_name, content)
            pushed += 1
            continue
        model_record = _matches(marker.relative_to(partial).as_posix(), RECORDS)
        content = marker.read_bytes() if files or model_record else None
        if model_record:
            record = json.loads(content)
            if record.get("fit_key", key.decode()) != key.decode():
                raise ValueError(f"{marker} belongs to a different fit key; it cannot be merged into {name}.")
            # The source key must travel with an arm, rather than being inferred from the destination's KEY_FILE.
            # Sizes make an interrupted manual copy distinguishable from a whole model before it is loaded.
            record.update(fit_key=key.decode(), model_files={path.relative_to(marker.with_suffix("")).as_posix(): path.stat().st_size for path in files})
            content = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
        complete = all(held.get(f"{name}/{path.relative_to(partial).as_posix()}") == path.stat().st_size for path in files)
        if marker_name in held and complete and (remote.read(marker_name) == content if files or model_record else held[marker_name] == marker.stat().st_size):
            continue
        # A previous marker cannot certify a replacement that has only been partly uploaded. Remove it first,
        # then publish this version's marker only after every companion is present (also repairs missing companions).
        if marker_name in held:
            remote.delete([marker_name])
        _transfer([(path, f"{name}/{path.relative_to(partial).as_posix()}") for path in files], remote.upload)
        if model_record:
            remote.write(marker_name, content)
        else:
            remote.upload(marker, marker_name)
        pushed += 1
    if pushed:
        LOGGER.info("run mirror: pushed %d durable units of %s", pushed, name)


def push(run_directory: Path, remote: Remote, bases: Sequence[Base] = (), arms: Sequence[str] | None = None) -> None:
    """Copy the run's durable state to the mirror, skipping what it already holds and the completed steps a base
    carries (``Base``), which the mirror names instead (BASES); the restart units first (an EB start is the newest
    progress), of ``arms`` only where given (a job that fits those arms). A file the driver moves away during the push
    (a partial renamed to its step, a step set aside, a released unit) ends that directory's push for this time; the
    next push sees where it went. A restart unit or step whose push fails does not stop the others': the push sends
    the rest and then raises ``PushIncomplete`` naming what it did not send."""
    try:
        failed = _push_restart(run_directory, remote, arms)
    except Exception as error:  # noqa: BLE001 - the restart units unlisted or their receipt unwritten: the steps go on
        LOGGER.warning("run mirror: the restart units were not pushed (%s: %s); the push goes on", type(error).__name__, error)
        failed = [(RESTART, error)]
    references: set[str] = set()
    for step in STEP_NAMES:
        final, partial = run_directory / step, run_directory / f"{step}{PARTIAL}"
        if final.is_symlink():
            # A region's link to its run's step, which the run's own push sends.
            continue
        try:
            if (final / COMPLETE_FILE).is_file():
                carrier = _push_step(remote, final, step, bases)
                if carrier is not None:
                    references.add(carrier.reference)
            elif (partial / KEY_FILE).is_file():
                _push_partial(remote, partial, step)
        except FileNotFoundError as moved:
            LOGGER.info("run mirror: %s moved during the push (%s); it is pushed next time", step, moved)
        except Exception as error:  # noqa: BLE001 - one step's failed upload must not stop the others'
            LOGGER.warning("run mirror: %s was not pushed (%s: %s); the push goes on", step, type(error).__name__, error)
            failed.append((step, error))
    try:
        if references:
            named = remote.sizes(f"{BASES}/")
            for reference in sorted(references - named.keys()):
                remote.write(reference, b"")
    except Exception as error:  # noqa: BLE001 - the regions' pushes go on
        LOGGER.warning("run mirror: the run's bases were not named (%s: %s); the push goes on", type(error).__name__, error)
        failed.append((BASES, error))
    regions = run_directory / REGIONS_DIRECTORY
    for region in sorted(regions.iterdir()) if regions.is_dir() else ():
        try:
            push(region, _Within(remote, f"{REGIONS_DIRECTORY}/{region.name}/"), arms=arms)
        except PushIncomplete as incomplete:
            failed.extend((f"{REGIONS_DIRECTORY}/{region.name}/{name}", error) for name, error in incomplete.failures)
    if failed:
        raise PushIncomplete(failed) from failed[0][1]


class PushIncomplete(RuntimeError):
    """A push that sent all it could but ``failures``' (each restart unit's prefix or step, with its error), whose
    uploads failed: the job's pusher (launcher/workspace/run_step.sh.template) counts it a failed push, and the next
    push sends them again."""

    def __init__(self, failures: list[tuple[str, BaseException]]) -> None:
        shown = "; ".join(f"{name} ({type(error).__name__}: {error})" for name, error in failures[:3])
        super().__init__(f"{len(failures)} not pushed: {shown}{' ...' if len(failures) > 3 else ''}")
        self.failures = failures
        self.names = [name for name, _error in failures]


def _restorable_partial(names: dict[str, int], name: str) -> list[str]:
    """The partial's KEY_FILE, every model directory, store chromosome and score/effects batch whose record is
    mirrored, and those records. Payloads without their record are interrupted uploads, never batches; a direct_sv
    chromosome without its rows/<chromosome>.npz (a mirror before it carried them) is made again."""
    prefix = f"{name}/"
    inside = [entry.removeprefix(prefix) for entry in names if entry.startswith(prefix)]
    records = {
        relative for relative in inside
        if _matches(relative, RECORDS) or (_matches(relative, STORE_RECORDS) and (
            name != "direct_sv.partial" or f"rows/{relative.removeprefix('chromosomes/').removesuffix('.json')}.npz" in inside
        ))
    }
    models = tuple(record.removesuffix(".json") + "/" for record in records if _matches(record, RECORDS))
    chromosomes = [record.removeprefix("chromosomes/").removesuffix(".json") for record in records if _matches(record, STORE_RECORDS)]
    batches = tuple(
        relative.removesuffix("complete.json") for relative in inside
        if name in ("score.partial", "effects.partial") and _matches(relative, BATCH_RECORDS) and not relative.split("/")[1].startswith(".")
    )
    return sorted(
        prefix + relative for relative in inside
        if relative == KEY_FILE
        or relative in records
        or relative.startswith(models)
        or relative.startswith(batches)
        or any(_of_chromosome(relative, chromosome) for chromosome in chromosomes)
    )


def _matches(relative: str, pattern: str) -> bool:
    """Whether a '/'-separated name matches a glob pattern component by component, as ``Path.glob`` would."""
    parts, wanted = relative.split("/"), pattern.split("/")
    return len(parts) == len(wanted) and all(fnmatchcase(part, want) for part, want in zip(parts, wanted))


def bases_of(url: str, remote: Remote, urls: Sequence[str], opened: Callable[[str], Base]) -> list[Base]:
    """The bases the run whose mirror is ``remote`` at ``url`` reads, each once: ``urls`` first, then those its mirror
    names (BASES), then those each base names in turn, whose steps that base does not hold itself. So a run need be
    given its base only until its first push names it, and a run based on another based run reads both."""
    found: dict[str, Base | None] = {url: None}
    pending = [*urls, *("gs://" + name.removeprefix(f"{BASES}/") for name in sorted(remote.sizes(f"{BASES}/")))]
    while pending:
        named = pending.pop(0)
        if named not in found:
            base = found[named] = opened(named)
            pending.extend("gs://" + name.removeprefix(f"{BASES}/") for name in sorted(base.remote.sizes(f"{BASES}/")))
    return [base for base in found.values() if base is not None]


def mounted(url: str) -> Path | None:
    """Where the mirror ``url`` (gs://bucket/prefix) shows on this machine's bucket mounts (GCS_MOUNTS), or None."""
    bucket, _, prefix = url.removeprefix("gs://").partition("/")
    for mount, shown in json.loads(os.environ.get(GCS_MOUNTS) or "{}").items():
        shown_bucket, _, shown_prefix = shown.removeprefix("gs://").partition("/")
        inside = Path(prefix.strip("/"))
        if shown_bucket == bucket and inside.is_relative_to(shown_prefix.strip("/")):
            return Path(mount) / inside.relative_to(shown_prefix.strip("/"))
    return None


def _replaced_stores(names: dict[str, int]) -> set[str]:
    """The completed steps whose mirrored store payload a later completed step's replaces: each but the last."""
    complete = {step for step in STEP_NAMES if f"{step}/{COMPLETE_FILE}" in names}
    holding = {name.split("/", 1)[0] for name in names if name.split("/", 2)[1:2] == [STORE] and name.count("/") >= 2}
    stores = [step for step in STEP_NAMES if step in complete and step in holding]
    return set(stores[:-1])


def _piece_bytes(total: int, workers: int) -> int:
    """The range a restore's large files are read in (``_download_pieces``): its ``total`` bytes over ``workers``
    squared, so each worker reads about ``workers`` pieces and the last pieces, read at a worker's share of the rate,
    end within 1/``workers`` of the restore's time after its last byte could; at least the client library's chunk for
    ranged reads at once (``transfer_manager.TM_DEFAULT_CHUNK_SIZE``). The AoU store's codes are 22 files of 10-43 GB
    (dosage/half0/<chromosome>, 525 of the 568 GB a fit job restores [real]), so with one stream per file a restore's
    last minutes read its largest files on a few streams."""
    from google.cloud.storage import transfer_manager

    return max(transfer_manager.TM_DEFAULT_CHUNK_SIZE, total // max(workers, 1) ** 2)


def _hash_pieces(path: Path, version: Version, pieces: list[tuple[int, int, Future]]) -> str:
    """The SHA-256 of the file at ``path`` (``RESTORED_DIGESTS``), read back a piece at a time as each piece's download
    ends, in file order, and its CRC32C checked against ``version``'s."""
    digest, checksum = hashlib.sha256(), google_crc32c.Checksum()
    with open(path, "rb", buffering=0) as handle:
        for start, stop, written in pieces:
            written.result()
            handle.seek(start)
            while start < stop:
                chunk = handle.read(min(stop - start, shutil.COPY_BUFSIZE))
                if not chunk:
                    raise OSError(f"{path} ended at {start} of its {version.size} bytes; restore again.")
                digest.update(chunk)
                checksum.update(chunk)
                start += len(chunk)
    if checksum.digest() != version.crc32c:
        raise OSError(f"{path}'s restored bytes do not match its object's CRC32C; restore again.")
    return digest.hexdigest()


def _download_pieces(objects: list[tuple[Remote, Path, str, int]], piece: int) -> list[str]:
    """Each (mirror, local path, object name, listed size) object read in ranges of ``piece`` bytes, in one pool (spawned
    processes, each with its own client, for GCS objects: ``_download_range_in_process``), with each file's SHA-256 in
    the same order. Every range is read from the generation the object had when the restore began, and each file's
    CRC32C is checked against that generation's. The pieces are queued by their place in their file, the first of every
    file first, so every file grows at a rate in proportion to its size and they all end together: no file is left
    to a few streams at the end, and each is hashed (``_hash_pieces``) as its pieces arrive, at a few hundred MB/s,
    beside the downloads."""
    versions = [source.version(name) for source, _path, name, _size in objects]
    for (_source, path, name, size), version in zip(objects, versions):
        if version.size != size:
            raise RuntimeError(f"{name} changed on its mirror after it was listed; restore again.")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            handle.truncate(size)
    queued = sorted(
        (start / version.size, -version.size, index, start)
        for index, version in enumerate(versions) for start in range(0, version.size, piece)
    )
    spawned = all(isinstance(source, GcsRemote) for source, _path, _name, _size in objects)
    workers = len(os.sched_getaffinity(0))
    # spawn, not fork: a forked child would share the parent client's connections
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) if spawned else ThreadPoolExecutor(max_workers=workers)
    pieces: list[list[tuple[int, int, Future]]] = [[] for _object in objects]
    with pool:
        try:
            for _place, _size, index, start in queued:
                source, path, name, size = objects[index]
                stop = min(start + piece, size)
                if spawned:
                    written = pool.submit(_download_range_in_process, source._url, name, path, start, stop, versions[index].generation)
                else:
                    written = pool.submit(source.download_range, name, path, start, stop, versions[index].generation)
                pieces[index].append((start, stop, written))
            with ThreadPoolExecutor(max_workers=len(objects)) as hashers:
                hashed = [
                    hashers.submit(_hash_pieces, path, version, own)
                    for (_source, path, _name, _size), version, own in zip(objects, versions, pieces)
                ]
                return [future.result() for future in hashed]
        except BaseException:
            for own in pieces:
                for _start, _stop, written in own:
                    written.cancel()
            raise


def _restore_downloads(objects: list[tuple[Remote, Path, str]], sizes: Sequence[int]) -> list[str | None] | None:
    """The planned restore's objects in one pool, with each file's digest in the same order. A carried step may read
    another mirror; GCS objects still use spawned processes with each process's own clients (``_download_in_process``).
    A file of more than one piece (``_piece_bytes``) is read in pieces (``_download_pieces``), in a pool of its own
    beside this one."""
    if not objects:
        return []
    unwrapped = []
    for source, path, name in objects:
        while isinstance(source, _Within):
            name = source._prefix + name
            source = source._remote
        unwrapped.append((source, path, name))
    piece = _piece_bytes(sum(sizes), len(os.sched_getaffinity(0)))
    large = [index for index, size in enumerate(sizes) if size > piece]
    if large:
        whole = [index for index in range(len(unwrapped)) if index not in set(large)]
        with ThreadPoolExecutor(max_workers=1) as beside:
            pieced = beside.submit(_download_pieces, [(*unwrapped[index], sizes[index]) for index in large], piece)
            downloaded = _restore_downloads([unwrapped[index] for index in whole], [sizes[index] for index in whole])
            hashed = pieced.result()
        LOGGER.info(
            "run mirror: %d files of %.1f GB read in pieces of %.3f GB", len(large), sum(sizes[index] for index in large) / 1e9, piece / 1e9,
        )
        digests: list[str | None] = [None] * len(unwrapped)
        for index, digest in zip(whole, downloaded or [None] * len(whole)):
            digests[index] = digest
        for index, digest in zip(large, hashed):
            digests[index] = digest
        return digests
    first = unwrapped[0][0]
    if all(source is first for source, _path, _name in unwrapped):
        return first.download_many([(path, name) for _source, path, name in unwrapped])
    if all(isinstance(source, GcsRemote) for source, _path, _name in unwrapped):
        with ProcessPoolExecutor(max_workers=len(os.sched_getaffinity(0)), mp_context=multiprocessing.get_context("spawn")) as pool:
            return list(pool.map(
                _download_in_process, [source._url for source, _path, _name in unwrapped],
                [name for _source, _path, name in unwrapped], [path for _source, path, _name in unwrapped],
            ))
    with ThreadPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as pool:
        return list(pool.map(lambda item: item[0].download(item[2], item[1]), unwrapped))


def restore(
    run_directory: Path, remote: Remote, mount: Path | None = None, bases: Sequence[Base] = (), arms: Sequence[str] | None = None,
    finishing: bool = False,
) -> None:
    """Fill a run directory from the mirror: every completed step, and each partial step's durable units, for every step
    this run directory does not hold already, and the restart units of ``arms`` (every arm's where None). Each is
    downloaded aside and renamed into place, so an interrupted restore leaves no step directory the driver would take
    for its own and no unit. ``mount`` is where the mirror shows on this machine (``mounted``): a completed step's store
    payload that a later completed step's replaces is linked there, not downloaded. A step before the fit (CARRIED)
    that the mirror holds neither completed nor partial comes from the first of ``bases`` that holds it completed, its
    replaced store linked on that base's mount. A completed step whose COMPLETE_FILE is gone or changed after its
    download (a push or removal where it lies) stops the restore: what was downloaded could mix two steps.

    With ``finishing``, for a run that only finishes (``workspace_pipeline.run_pipeline``'s ``finish``), a completed
    step before the fit is restored as its COMPLETE_FILE alone, but those it reads (FINISH_INPUTS), and no restart unit:
    no store and no fit state, only the models, the scores and the effects.

    Every step's objects share one pool, largest first, before any step is renamed into place. The AoU whole-genome
    restore waited another 2m15s after the store, measurement's 21.3 GB on one or two streams [real]; those
    files can download beside the store's. Each completed step's marker is checked again before its rename."""
    names = remote.sizes("")
    run_directory.mkdir(parents=True, exist_ok=True)
    # The mirror's objects, a carried step's those of the base that carries it.
    objects = dict(names)
    carriers: dict[str, Base] = {}
    for base in bases:
        held = base.remote.sizes("")
        for step in CARRIED:
            if step in carriers or f"{step}/{COMPLETE_FILE}" in names or f"{step}{PARTIAL}/{KEY_FILE}" in names:
                continue
            if f"{step}/{COMPLETE_FILE}" in held:
                carriers[step] = base
                objects = {entry: size for entry, size in objects.items() if not entry.startswith(f"{step}/")}
                objects.update((entry, size) for entry, size in held.items() if entry.startswith(f"{step}/"))
    linked = _replaced_stores(objects)
    planned, downloads = [], []
    for step in STEP_NAMES:
        final, partial = run_directory / step, run_directory / f"{step}{PARTIAL}"
        if final.exists() or partial.exists():
            continue
        carrier = carriers.get(step)
        source, shown = (remote, mount) if carrier is None else (carrier.remote, carrier.mount)
        summary_only = finishing and step in CARRIED and step not in FINISH_INPUTS
        if f"{step}/{COMPLETE_FILE}" in objects:
            name, target = step, final
            chosen = [f"{step}/{COMPLETE_FILE}"] if summary_only else sorted(entry for entry in objects if entry.startswith(f"{step}/"))
        elif f"{step}{PARTIAL}/{KEY_FILE}" in names:
            name, target = f"{step}{PARTIAL}", partial
            chosen = _restorable_partial(names, name)
        else:
            continue
        origin = "this run's mirror" if carrier is None else f"the base {carrier.url}"
        marker = source.read(f"{step}/{COMPLETE_FILE}") if target == final else None
        if target == final and marker is None:
            raise RuntimeError(f"{step}'s {COMPLETE_FILE} left {origin} after it was listed; restore again.")
        aside = run_directory / f"{name}{RESTORING}"
        if aside.exists():
            shutil.rmtree(aside)
        aside.mkdir()
        listed = json.loads(source.read(f"{name}{DIRECTORIES}") or "[]") if target == final and not summary_only else []
        payload = shown / step / STORE if shown is not None and target == final and step in linked and not summary_only else None
        if payload is not None and payload.is_dir():
            chosen = [entry for entry in chosen if not entry.startswith(f"{step}/{STORE}/")]
            listed = [directory for directory in listed if Path(directory).parts[0] != STORE]
            (aside / STORE).symlink_to(payload, target_is_directory=True)
            LOGGER.info("run mirror: %s/%s, which a later step's store replaces, is read from %s", step, STORE, payload)
        for directory in listed:
            (aside / directory).mkdir(parents=True, exist_ok=True)
        pairs = [(aside / entry.removeprefix(f"{name}/"), entry) for entry in chosen]
        planned.append((step, name, aside, target, source, origin, marker, pairs))
        downloads.extend((source, path, entry) for path, entry in pairs)
    downloads.sort(key=lambda item: (-objects[item[2]], item[2]))
    digests = _restore_downloads(downloads, [objects[entry] for _source, _path, entry in downloads])
    restored = {path: digest for (_source, path, _entry), digest in zip(downloads, digests or ())}
    for step, name, aside, target, source, origin, marker, pairs in planned:
        if marker is not None and source.read(f"{step}/{COMPLETE_FILE}") != marker:
            raise RuntimeError(f"{step} changed on {origin} while it was restored; restore again.")
        # Each file's digest as its bytes were written, so a digest of this step's store reads none of them again.
        record_restored((path, restored[path]) for path, _entry in pairs if restored.get(path) is not None)
        os.rename(aside, target)
        LOGGER.info("run mirror: restored %s from %s, %d files, %.1f GB", name, origin, len(pairs), sum(objects[entry] for _path, entry in pairs) / 1e9)
    if not finishing:
        _restore_restart(run_directory, remote, {name: size for name, size in names.items() if name.startswith(f"{RESTART}/")}, arms)
    # Each region's own steps; the driver links the run's steps before the fit into its directory again.
    for region in sorted({entry.split("/")[1] for entry in names if entry.startswith(f"{REGIONS_DIRECTORY}/") and entry.count("/") >= 2}):
        restore(run_directory / REGIONS_DIRECTORY / region, _Within(remote, f"{REGIONS_DIRECTORY}/{region}/"), arms=arms, finishing=finishing)


# ---------------------------------------------------------------------------
# The restart store
# ---------------------------------------------------------------------------


def _remote_units(names: dict[str, int]) -> dict[str, dict[str, int]]:
    """The mirror's restart objects by unit, restart/<arm>/<kind>/<key>/: each unit's object names (relative to it)
    and sizes, interrupted uploads included."""
    units: dict[str, dict[str, int]] = {}
    for name, size in names.items():
        root, _, rest = name.partition("/")
        arm, _, rest = rest.partition("/")
        kind, _, rest = rest.partition("/")
        key, _, inner = rest.partition("/")
        if root == RESTART and kind in KINDS and key and inner:
            units.setdefault(f"{RESTART}/{arm}/{kind}/{key}/", {})[inner] = size
    return units


def _remote_record(remote: Remote, prefix: str, held: dict[str, int]) -> tuple[dict, bytes] | None:
    """The mirrored unit at ``prefix``'s manifest, parsed and as read, where the unit is whole (a sealed unit's every
    file at its recorded size; a mutable unit's files check themselves), else None."""
    if MANIFEST not in held:
        return None
    try:
        content = remote.read(prefix + MANIFEST) or b""
        record = json.loads(content)
        _, _arm, kind, key = prefix.rstrip("/").split("/")
        if record["format"] != FORMAT or record["kind"] != kind or record["key"] != key:
            return None
        if record["files"] is not None and any(
            held.get(name) != size or Path(name).is_absolute() or ".." in Path(name).parts for name, size in record["files"].items()
        ):
            return None
        return record, content
    except (ValueError, KeyError, TypeError, AttributeError):
        return None


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _push_restart(run_directory: Path, remote: Remote, arms: Sequence[str] | None) -> list[tuple[str, BaseException]]:
    """The restart units of ``arms`` (every arm's where None) to the mirror (the module docstring): a sealed unit the
    mirror holds whole is kept as it is; one it does not is sent payload first, in parts, and its manifest last; a
    mutable unit's changed files are each replaced whole, its manifest after them; and a mirrored unit this disk
    restored or saw whole there (its build id in RECEIPT) but no longer holds (released, superseded) is removed,
    manifest first, where the mirror's copy is still that build. Returns the units whose push failed, each with its
    error; the others are pushed all the same."""
    root = run_directory / RESTART
    if not root.is_dir():
        return []
    held = _remote_units(remote.sizes(f"{RESTART}/"))
    receipt = _read_json(root / RECEIPT, {})
    local: set[str] = set()
    pushed = 0
    failed: list[tuple[str, BaseException]] = []
    units = [
        (f"{RESTART}/{directory.name}/{kind}/{path.name}/", path, record)
        for directory in sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith("."))
        if arms is None or directory.name in arms
        for kind in KINDS for path, record in RestartStore(directory).units(kind)
    ]
    # The mutable units first: an EB start or a mode search's progress is a few MB and the fit's newest state, and a
    # sealed unit's push (Stage 0's LD store, minutes) must not hold it back.
    units.sort(key=lambda unit: unit[2]["files"] is not None)
    for prefix, path, record in units:
        local.add(prefix)
        mirrored = held.get(prefix, {})
        try:
            if record["files"] is not None:
                found = _remote_record(remote, prefix, mirrored)
                if found is None:
                    content = (path / MANIFEST).read_bytes()
                    if MANIFEST in mirrored:
                        remote.delete([prefix + MANIFEST])
                    for name in record["files"]:
                        remote.upload_parts(path / name, prefix + name)
                    remote.write(prefix + MANIFEST, content)
                    pushed += 1
                receipt[prefix] = record["id"] if found is None else found[0]["id"]
                continue
            # The mode search's index is read before the files are listed and sent after them: the fit writes a source
            # before the index that names it, so every source a sent index names is sent before it.
            index = None
            try:
                with (path / MODE_INDEX).open("rb") as handle:
                    stat = os.fstat(handle.fileno())
                    index = (handle.read(), [stat.st_size, stat.st_mtime_ns])
            except FileNotFoundError:
                pass
            files = sorted(
                (file for file in path.iterdir() if file.is_file() and not file.name.startswith(".") and file.name != MODE_INDEX),
                key=lambda file: file.name == MANIFEST,
            )
            for file in files:
                stat = file.stat()
                stamp = [stat.st_size, stat.st_mtime_ns]
                if mirrored.get(file.name) == stat.st_size and receipt.get(prefix + file.name) == stamp:
                    continue
                remote.upload(file, prefix + file.name)
                receipt[prefix + file.name] = stamp
                pushed += 1
            if index is not None and not (mirrored.get(MODE_INDEX) == index[1][0] and receipt.get(prefix + MODE_INDEX) == index[1]):
                remote.write(prefix + MODE_INDEX, index[0])
                receipt[prefix + MODE_INDEX] = index[1]
                pushed += 1
            receipt[prefix] = record["id"]
        except FileNotFoundError:
            # Released while it was sent: its removal follows below or at the next push.
            local.discard(prefix)
        except Exception as error:  # noqa: BLE001 - one unit's refused or broken upload must not stop the others'
            # The 403 on every sealed unit of rounds m1 and m2 also kept every EB start behind it off the mirror. The
            # unit stays this disk's (in ``local``), so nothing of it is removed, and the next push sends it again.
            LOGGER.warning("run mirror: restart unit %s was not pushed (%s: %s); the push goes on", prefix, type(error).__name__, error)
            failed.append((prefix, error))
    for prefix, objects in held.items():
        if prefix in local or prefix not in receipt or (arms is not None and prefix.split("/")[1] not in arms):
            continue
        if failed:
            # A unit is released once what replaces it is saved (a completed model's checkpoint for its group's EB
            # starts and Stage 0): while a unit of this push was not sent, the released ones stay on the mirror.
            LOGGER.info("run mirror: restart unit %s is kept on the mirror while %d units are not pushed", prefix, len(failed))
            local.add(prefix)
            continue
        if (root / prefix[len(RESTART) + 1:]).exists():
            # On this disk but not whole there (``RestartStore.units`` lists whole units only): not released.
            local.add(prefix)
            continue
        try:
            build = json.loads(remote.read(prefix + MANIFEST) or b"")["id"] if MANIFEST in objects else None
        except (ValueError, KeyError, TypeError):
            build = None
        except Exception as error:  # noqa: BLE001 - its manifest unread: kept, and read again at the next push
            LOGGER.warning("run mirror: released restart unit %s was not read (%s: %s)", prefix, type(error).__name__, error)
            local.add(prefix)
            continue
        if build != receipt[prefix]:
            LOGGER.info("run mirror: restart unit %s on the mirror is not the build this disk held; it is kept", prefix)
            continue
        try:
            remote.delete([prefix + MANIFEST])
            remote.delete([prefix + name for name in objects if name != MANIFEST])
        except Exception as error:  # noqa: BLE001 - a removal that fails is tried again at the next push; the rest go on
            LOGGER.warning("run mirror: released restart unit %s was not removed (%s: %s)", prefix, type(error).__name__, error)
            local.add(prefix)
            continue
        LOGGER.info("run mirror: removed restart unit %s, which the fit released", prefix)
    # A unit the disk no longer holds is done with here, its mirrored copy removed above or not this disk's to remove.
    for name in list(receipt):
        unit = "/".join(name.split("/")[:4]) + "/"
        if unit not in local and (arms is None or unit.split("/")[1] in arms):
            del receipt[name]
    temporary = root / f"{RECEIPT}.tmp"
    temporary.write_text(json.dumps(receipt), encoding="utf-8")
    os.replace(temporary, root / RECEIPT)
    if pushed:
        LOGGER.info("run mirror: pushed %d restart files", pushed)
    return failed


def _restore_restart(run_directory: Path, remote: Remote, names: dict[str, int], arms: Sequence[str] | None) -> None:
    """Every whole mirrored restart unit of ``arms`` (every arm's where None) the disk does not hold, each downloaded
    aside, its manifest written last and renamed into place, and entered in RECEIPT with its build id, which lets later
    pushes remove it once the fit releases it. A unit whose manifest names a missing or short file, with no manifest,
    or whose files changed while they were downloaded, is left behind."""
    root = run_directory / RESTART
    receipt = _read_json(root / RECEIPT, {})
    pending: list[tuple[Path, Path, dict, bytes, str]] = []
    pairs: list[tuple[Path, str]] = []
    parts: list[tuple[Path, str]] = []
    for prefix, objects in sorted(_remote_units(names).items()):
        _, arm, kind, key = prefix.rstrip("/").split("/")
        if arms is not None and arm not in arms or RestartStore(root / arm).unit(kind, key) is not None:
            continue
        found = _remote_record(remote, prefix, objects)
        if found is None:
            LOGGER.info("run mirror: restart unit %s is not whole on the mirror; it is made again", prefix)
            continue
        record, content = found
        files = list(record["files"]) if record["files"] is not None else [
            name for name in objects if name != MANIFEST and "/" not in name and not name.startswith(".")
        ]
        aside = root / arm / kind / f".restoring.{key}"
        if aside.exists():
            shutil.rmtree(aside)
        aside.mkdir(parents=True)
        pending.append((aside, root / arm / kind / key, record, content, prefix))
        (parts if record["files"] is not None else pairs).extend((aside / name, prefix + name) for name in files)
    # A sealed unit's files are few and large (Stage 0's LD store is one file, ``genotype_statistics``): each is read
    # in ranges at once, the largest first. A mutable unit's are small, one stream each.
    for path, name in sorted(parts, key=lambda pair: -names[pair[1]]):
        remote.download_parts(name, path)
    remote.download_many(pairs)
    for aside, target, record, content, prefix in pending:
        if record["files"] is not None and any((aside / name).stat().st_size != size for name, size in record["files"].items()):
            # The mirrored unit changed during the download (another job's build of its key): it is made again.
            LOGGER.info("run mirror: restart unit %s changed on the mirror while it was restored; it is made again", prefix)
            shutil.rmtree(aside)
            continue
        (aside / MANIFEST).write_bytes(content)
        if target.exists():
            # A unit the disk holds but not whole (``RestartStore.unit``): the mirror's whole one takes its place.
            shutil.rmtree(target)
        os.rename(aside, target)
        for file in target.iterdir() if record["files"] is None else []:
            stat = file.stat()
            receipt[prefix + file.name] = [stat.st_size, stat.st_mtime_ns]
        receipt[prefix] = record["id"]
    if pending:
        (root / RECEIPT).write_text(json.dumps(receipt), encoding="utf-8")
    if pending:
        LOGGER.info("run mirror: restored %d restart units, %.1f GB", len(pending), sum(names[name] for _path, name in pairs + parts) / 1e9)


def _step_key(run_directory: Path, step: str) -> str | None:
    """The key ``step`` holds here: its completed directory's, else its partial's, else None."""
    complete = run_directory / step / COMPLETE_FILE
    if complete.is_file():
        return str(json.loads(complete.read_text(encoding="utf-8"))["key"])
    key = run_directory / f"{step}{PARTIAL}" / KEY_FILE
    return key.read_text(encoding="utf-8") if key.is_file() else None


def _whole_batch(directory: Path) -> bool:
    """Whether a batch directory (``postfit_checkpoint``) holds every file its record lists, at its size and digest."""
    try:
        files = json.loads((directory / RECORD).read_text(encoding="utf-8"))["files"]
        for name, expected in files.items():
            digest = hashlib.sha256()
            with (directory / name).open("rb") as handle:
                for block in iter(lambda: handle.read(io.DEFAULT_BUFFER_SIZE), b""):
                    digest.update(block)
            if expected != {"bytes": (directory / name).stat().st_size, "sha256": digest.hexdigest()}:
                return False
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False
    return True


def _adopt_unit(partner: Remote, held: dict[str, int], prefix: str, target: Path) -> bool:
    """Download the partner's objects under ``prefix`` (a directory, '/'-ended) aside, then rename them into ``target``;
    False where the partner lists none. A ``target`` here already is left as it is, unless it is a batch its record does
    not certify (``_whole_batch``), which is replaced."""
    if target.exists() and not ((target / RECORD).exists() and not _whole_batch(target)):
        return True
    entries = sorted(name for name in held if name.startswith(prefix))
    if not entries:
        return False
    aside = target.with_name(f".{target.name}.adopting")
    if aside.exists():
        shutil.rmtree(aside)
    partner.download_many([(aside / entry.removeprefix(prefix), entry) for entry in entries])
    if target.exists():
        shutil.rmtree(target)
    os.rename(aside, target)
    return True


def _adopt_arm(run_directory: Path, partner: Remote, arm: str, listed: dict, unit: str, handoff: Path) -> bool:
    """The units the partner's handoff of ``arm`` lists (``adopt``), then the handoff itself at ``unit``; False where one
    is not in the partner's mirror."""
    held = {name: size for step in ("fit", *ARM_PARTS) for name, size in partner.sizes(f"{step}{PARTIAL}/").items()}
    models = run_directory / f"fit{PARTIAL}" / "models" / arm
    for trait in [] if (run_directory / "fit" / COMPLETE_FILE).is_file() else listed["models"]:
        if (models / f"{trait}.json").is_file():
            continue
        record = f"fit{PARTIAL}/models/{arm}/{trait}.json"
        if record not in held:
            return False
        # A model directory without its record is an adoption cut short, made again whole.
        if (models / trait).exists():
            shutil.rmtree(models / trait)
        if not _adopt_unit(partner, held, f"fit{PARTIAL}/models/{arm}/{trait}/", models / trait):
            return False
        aside = models / f".{trait}.json.adopting"
        partner.download(record, aside)
        os.replace(aside, models / f"{trait}.json")
    for step in ARM_PARTS:
        if (run_directory / step / COMPLETE_FILE).is_file():
            continue
        for name in listed[step]:
            prefix = f"{step}{PARTIAL}/batches/{name}/"
            if f"{prefix}{RECORD}" not in held or not _adopt_unit(partner, held, prefix, run_directory / f"{step}{PARTIAL}" / "batches" / name):
                return False
    return _adopt_unit(partner, partner.sizes(f"{unit}/"), f"{unit}/", handoff)


def adopt(run_directory: Path, partner: Remote) -> bool:
    """Bring the other arm's parts from ``partner``, the mirror of the run that fitted it (``workspace_pipeline.
    run_pipeline``'s ``fit_arms``), into this run's partials, so the run can finish (``run_pipeline``'s ``finish``) with no
    store; whether every arm's parts are then here. For each arm whose handoff (``workspace_pipeline.arm_handoff``) this
    run lacks, the partner's handoff, checked against its record, names the units to bring: the arm's saved models with
    their records in the fit's partial, and its batches in the score and effects steps' partials. Each is downloaded aside
    and renamed into place, a model's record after it and the handoff last, so an interrupted adoption leaves nothing the
    driver takes for whole and is resumed by the next. A step this run has completed takes nothing.

    False where the partner's mirror holds no handoff of the arm, or loses it during the adoption: its job has not
    finished its parts, or has finished the run itself (its parts are then no longer partial). The partner's partials
    must be keyed as this run's (one fit, one score and one effects key), else this stops; each adopted model is checked
    again when the fit step ends (``_check_fit_records``: its source fit key, its files' sizes and its arrays' digest,
    its arm's record offsets and its training problem). A run of regions adopts each region's parts in its own."""
    regions = run_directory / REGIONS_DIRECTORY
    if regions.is_dir():
        return all([adopt(region, _Within(partner, f"{REGIONS_DIRECTORY}/{region.name}/")) for region in sorted(regions.iterdir()) if region.is_dir()])
    if (run_directory / "effects" / COMPLETE_FILE).is_file():
        return True
    keys = {step: _step_key(run_directory, step) for step in ("fit", *ARM_PARTS)}
    for step, key in keys.items():
        if (run_directory / step / COMPLETE_FILE).is_file():
            continue
        if key is None:
            raise ValueError(f"{run_directory} holds no {step} part to finish from: its own arm's job has not done its parts.")
        mirrored = partner.read(f"{step}{PARTIAL}/{KEY_FILE}")
        if mirrored is None:
            LOGGER.info("run mirror: the partner holds no %s part: its job has not done its parts, or it finished the run", step)
            return False
        if mirrored.decode() != key:
            raise ValueError(f"the partner's {step} part is keyed {mirrored.decode()[:16]}, not {key[:16]} as this run's: the runs differ in config or code.")
    effects = run_directory / f"effects{PARTIAL}"
    for arm in ARMS:
        handoff = arm_handoff(effects, str(keys["effects"]), arm)
        if handoff.read() is not None:
            continue
        unit = handoff.path.relative_to(run_directory).as_posix()
        record, content = partner.read(f"{unit}/{RECORD}"), partner.read(f"{unit}/{HANDOFF}")
        if record is None or content is None:
            LOGGER.info("run mirror: the partner's mirror holds no %s handoff yet", arm)
            return False
        written = json.loads(record)
        if written.get("identity") != handoff.identity or written.get("files", {}).get(HANDOFF) != {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}:
            raise ValueError(f"the partner's {arm} handoff does not match its record or this run's effects key.")
        listed = json.loads(content)
        if listed.get("arm") != arm:
            raise ValueError(f"the partner's {arm} handoff names the arm {listed.get('arm')!r}.")
        try:
            whole = _adopt_arm(run_directory, partner, arm, listed, unit, handoff.path)
        except Exception:
            # Objects that vanish with the handoff were the partner's parts of a run it has since finished.
            if partner.read(f"{unit}/{RECORD}") is not None:
                raise
            whole = False
        if not whole or handoff.read() is None:
            LOGGER.info("run mirror: the partner's %s parts are not all mirrored; nothing is finished from them yet", arm)
            return False
        LOGGER.info("run mirror: adopted the %s parts: %d models, %d score and %d effects batches", arm, len(listed["models"]), len(listed["score"]), len(listed["effects"]))
    return True

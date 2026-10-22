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
  a completed model's checkpoint) are each replaced whole, sent again when their size or modification time changes
  (RECEIPT). A unit the fit released is removed from the mirror, but only by a disk that restored that build of it or
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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Protocol

from sv_pgs.content_identity import GCS_MOUNTS, record_restored
from sv_pgs.postfit_checkpoint import RECORD
from sv_pgs.restart_store import FORMAT, KINDS, MANIFEST, RestartStore
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

    def delete(self, names: Iterable[str]) -> None: ...


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
            handle.seek(start)
            self._bucket.blob(self._root + name).upload_from_file(handle, size=stop - start, retry=DEFAULT_RETRY)

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

    def delete(self, names: Iterable[str]) -> None:
        blobs = [self._bucket.blob(self._root + name) for name in names]
        if blobs:
            self._bucket.delete_blobs(blobs)


class DirectoryRemote:
    """A mirror in a local directory, with GCS's whole-or-absent objects (each file written aside, then renamed)."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    def sizes(self, prefix: str) -> dict[str, int]:
        if not self._root.is_dir():
            return {}
        found = {path.relative_to(self._root).as_posix(): path.stat().st_size for path in self._root.rglob("*") if path.is_file()}
        return {name: size for name, size in found.items() if name.startswith(prefix)}

    def read(self, name: str) -> bytes | None:
        path = self._root / name
        return path.read_bytes() if path.is_file() else None

    def upload(self, path: Path, name: str) -> None:
        target = self._root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        aside = target.with_name(f".{target.name}.upload")
        shutil.copyfile(path, aside)
        os.replace(aside, target)

    def upload_parts(self, path: Path, name: str) -> None:
        self.upload(path, name)

    def write(self, name: str, content: bytes) -> None:
        target = self._root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        aside = target.with_name(f".{target.name}.upload")
        aside.write_bytes(content)
        os.replace(aside, target)

    def download(self, name: str, path: Path) -> str | None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._root / name, "rb") as source, open(path, "wb") as handle:
            written = _HashingFile(handle)
            shutil.copyfileobj(source, written)
        return written.hexdigest()

    def download_parts(self, name: str, path: Path) -> None:
        self.download(name, path)

    def download_many(self, pairs: list[tuple[Path, str]]) -> list[str | None]:
        return _transfer(pairs, lambda path, name: self.download(name, path))

    def delete(self, names: Iterable[str]) -> None:
        for name in names:
            path = self._root / name
            path.unlink(missing_ok=True)
            # A GCS prefix with no object left is gone; so is a directory here.
            while path.parent != self._root and not any(path.parent.iterdir()):
                path = path.parent
                path.rmdir()


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

    def delete(self, names: Iterable[str]) -> None:
        self._remote.delete([self._prefix + name for name in names])


_PROCESS_REMOTES: dict[str, GcsRemote] = {}
"""A GcsRemote.download_many worker process's own client, one per mirror URL."""


def _download_in_process(url: str, name: str, path: Path) -> str | None:
    if url not in _PROCESS_REMOTES:
        _PROCESS_REMOTES[url] = GcsRemote(url)
    return _PROCESS_REMOTES[url].download(name, path)


def _upload_range_in_process(url: str, path: Path, start: int, stop: int, name: str) -> None:
    if url not in _PROCESS_REMOTES:
        _PROCESS_REMOTES[url] = GcsRemote(url)
    _PROCESS_REMOTES[url].upload_range(path, start, stop, name)


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
    next push sees where it went."""
    _push_restart(run_directory, remote, arms)
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
    if references:
        named = remote.sizes(f"{BASES}/")
        for reference in sorted(references - named.keys()):
            remote.write(reference, b"")
    regions = run_directory / REGIONS_DIRECTORY
    for region in sorted(regions.iterdir()) if regions.is_dir() else ():
        push(region, _Within(remote, f"{REGIONS_DIRECTORY}/{region.name}/"), arms=arms)


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

def _restore_downloads(objects: list[tuple[Remote, Path, str]]) -> list[str | None] | None:
    """The planned restore's objects in one pool, with each file's digest in the same order. A carried step may read
    another mirror; GCS objects still use spawned processes with each process's own clients (``_download_in_process``)."""
    if not objects:
        return []
    unwrapped = []
    for source, path, name in objects:
        while isinstance(source, _Within):
            name = source._prefix + name
            source = source._remote
        unwrapped.append((source, path, name))
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
    restore waited another 2m15s after the store, measurement's 21.3 GB on one or two streams [real, aou-wg6]; those
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
    digests = _restore_downloads(downloads)
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


def _push_restart(run_directory: Path, remote: Remote, arms: Sequence[str] | None) -> None:
    """The restart units of ``arms`` (every arm's where None) to the mirror (the module docstring): a sealed unit the
    mirror holds whole is kept as it is; one it does not is sent payload first, in parts, and its manifest last; a
    mutable unit's changed files are each replaced whole, its manifest after them; and a mirrored unit this disk
    restored or saw whole there (its build id in RECEIPT) but no longer holds (released, superseded) is removed,
    manifest first, where the mirror's copy is still that build."""
    root = run_directory / RESTART
    if not root.is_dir():
        return
    held = _remote_units(remote.sizes(f"{RESTART}/"))
    receipt = _read_json(root / RECEIPT, {})
    local: set[str] = set()
    pushed = 0
    for directory in sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")):
        if arms is not None and directory.name not in arms:
            continue
        store = RestartStore(directory)
        for kind in KINDS:
            for path, record in store.units(kind):
                prefix = f"{RESTART}/{directory.name}/{kind}/{path.name}/"
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
                    files = sorted((file for file in path.iterdir() if file.is_file() and not file.name.startswith(".")), key=lambda file: file.name == MANIFEST)
                    for file in files:
                        stat = file.stat()
                        stamp = [stat.st_size, stat.st_mtime_ns]
                        if mirrored.get(file.name) == stat.st_size and receipt.get(prefix + file.name) == stamp:
                            continue
                        remote.upload(file, prefix + file.name)
                        receipt[prefix + file.name] = stamp
                        pushed += 1
                    receipt[prefix] = record["id"]
                except FileNotFoundError:
                    # Released while it was sent: its removal follows below or at the next push.
                    local.discard(prefix)
    for prefix, objects in held.items():
        if prefix in local or prefix not in receipt or (arms is not None and prefix.split("/")[1] not in arms):
            continue
        try:
            build = json.loads(remote.read(prefix + MANIFEST) or b"")["id"] if MANIFEST in objects else None
        except (ValueError, KeyError, TypeError):
            build = None
        if build != receipt[prefix]:
            LOGGER.info("run mirror: restart unit %s on the mirror is not the build this disk held; it is kept", prefix)
            continue
        remote.delete([prefix + MANIFEST])
        remote.delete([prefix + name for name in objects if name != MANIFEST])
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

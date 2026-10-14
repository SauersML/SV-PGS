"""The run directory's copy in the workspace bucket, which carries a run from one Google Batch job to the next.

Batch attaches an existing persistent disk without write access ("All options except writing are supported", Batch's
storage guide, https://docs.cloud.google.com/batch/docs/create-run-job-storage), so no disk can carry a run between
jobs. Instead, each job works on a new disk. ``restore`` fills that disk from the mirror before the driver starts.
``push`` copies the run's durable state back while the driver runs and once more after it exits (launcher/workspace).

The durable state is what the driver resumes from:
- **A completed step's directory** (``<step>/`` with COMPLETE_FILE). The driver never changes one once it is renamed
  into place. It is pushed whole with COMPLETE_FILE last, so a mirrored step without one is an interrupted push, and
  is never restored.
- **The fit step's progress in ``fit.partial``:**
  - each engine checkpoint, which ``fit_checkpoint`` writes whole;
  - each saved (arm, trait) model with its record, the record last, as the fit step saves them;
  - the partial's KEY_FILE.

Any other partial work is redone, as after a preemption on one disk. GCS creates an object only when its upload
completes, so a mirrored file is whole or absent.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from collections.abc import Iterable
from fnmatch import fnmatchcase
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol

from sv_pgs.workspace_pipeline import COMPLETE_FILE, KEY_FILE, STEP_NAMES

LOGGER = logging.getLogger(__name__)
PARTIAL = ".partial"
RESTORING = ".restoring"
CHECKPOINTS = "work/*/checkpoints/*/*.npz"
"""The engine's checkpoints in the fit step's partial: <arm>'s work directory, checkpoints/<key>/model<i>.npz
(``stage2_wiring.fit_models``)."""
DIRECTORIES = ".directories.json"
"""Beside a mirrored step, its directories, which GCS keeps only as the prefixes of objects: an empty one is
restored from this list."""
RECORDS = "models/*/*.json"
"""The fit step's saved-model records in its partial: models/<arm>/<trait>.json, written after models/<arm>/<trait>."""


class Remote(Protocol):
    """The mirror: objects named by their path relative to the run directory, '/'-separated."""

    def sizes(self, prefix: str) -> dict[str, int]: ...

    def read(self, name: str) -> bytes | None: ...

    def upload(self, path: Path, name: str) -> None: ...

    def write(self, name: str, content: bytes) -> None: ...

    def download(self, name: str, path: Path) -> None: ...

    def delete(self, names: Iterable[str]) -> None: ...


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

    def write(self, name: str, content: bytes) -> None:
        self._bucket.blob(self._root + name).upload_from_string(content)

    def download(self, name: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._bucket.blob(self._root + name).download_to_filename(str(path))

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

    def write(self, name: str, content: bytes) -> None:
        target = self._root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        aside = target.with_name(f".{target.name}.upload")
        aside.write_bytes(content)
        os.replace(aside, target)

    def download(self, name: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._root / name, path)

    def delete(self, names: Iterable[str]) -> None:
        for name in names:
            path = self._root / name
            path.unlink(missing_ok=True)
            # A GCS prefix with no object left is gone; so is a directory here.
            while path.parent != self._root and not any(path.parent.iterdir()):
                path = path.parent
                path.rmdir()


def _transfer(pairs: list[tuple[Path, str]], move) -> None:
    """``move`` every (local path, object name) pair, on one thread per CPU this process may use."""
    if not pairs:
        return
    with ThreadPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as pool:
        for _ in pool.map(lambda pair: move(*pair), pairs):
            pass


def _files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*") if path.is_file())


def _push_step(remote: Remote, final: Path, name: str) -> None:
    marker = final / COMPLETE_FILE
    content = marker.read_bytes()
    if remote.read(f"{name}/{COMPLETE_FILE}") == content:
        return
    remote.delete(remote.sizes(f"{name}/"))
    files = [path for path in _files(final) if path != marker]
    directories = sorted(path.relative_to(final).as_posix() for path in final.rglob("*") if path.is_dir())
    remote.write(f"{name}{DIRECTORIES}", json.dumps(directories).encode())
    _transfer([(path, f"{name}/{path.relative_to(final).as_posix()}") for path in files], remote.upload)
    if marker.read_bytes() != content:
        raise FileNotFoundError(f"{final} was replaced while it was pushed.")
    remote.upload(marker, f"{name}/{COMPLETE_FILE}")
    remote.delete(remote.sizes(f"{name}{PARTIAL}/"))
    LOGGER.info("run mirror: pushed %s, %d files, %.1f GB", name, len(files) + 1, sum(path.stat().st_size for path in files) / 1e9)


def _partial_units(partial: Path) -> list[tuple[list[Path], Path]]:
    """The fit step's durable units in ``partial``, each as (its files, the file that marks it whole, sent last): every
    engine checkpoint (CHECKPOINTS; a hidden name is one ``fit_checkpoint`` is still writing), and every saved model
    directory models/<arm>/<trait> beside its record <trait>.json."""
    units: list[tuple[list[Path], Path]] = [([], path) for path in sorted(partial.glob(CHECKPOINTS)) if not path.name.startswith(".")]
    if (partial / "models").is_dir():
        for record in sorted(partial.glob(RECORDS)):
            model = record.with_suffix("")
            if model.is_dir():
                units.append((_files(model), record))
    return units


def _push_partial(remote: Remote, partial: Path, step: str) -> None:
    # The driver set the step's completed directory aside to make it again under another key, so the mirrored one is
    # stale: restored, it would take the partial's place.
    remote.delete(remote.sizes(f"{step}/"))
    name = f"{step}{PARTIAL}"
    key = (partial / KEY_FILE).read_bytes()
    if remote.read(f"{name}/{KEY_FILE}") != key:
        remote.delete(remote.sizes(f"{name}/"))
        remote.upload(partial / KEY_FILE, f"{name}/{KEY_FILE}")
    held = remote.sizes(f"{name}/")
    pushed = 0
    for files, marker in _partial_units(partial):
        marker_name = f"{name}/{marker.relative_to(partial).as_posix()}"
        if marker_name in held and (remote.read(marker_name) == marker.read_bytes() if files else held[marker_name] == marker.stat().st_size):
            continue
        _transfer([(path, f"{name}/{path.relative_to(partial).as_posix()}") for path in files], remote.upload)
        remote.upload(marker, marker_name)
        pushed += 1
    if pushed:
        LOGGER.info("run mirror: pushed %d checkpoints and saved models of %s", pushed, name)


def push(run_directory: Path, remote: Remote) -> None:
    """Copy the run's durable state to the mirror, skipping what it already holds. A file the driver moves away during
    the push (a partial renamed to its step, a step set aside) ends that directory's push for this time; the next push
    sees where it went."""
    for step in STEP_NAMES:
        final, partial = run_directory / step, run_directory / f"{step}{PARTIAL}"
        try:
            if (final / COMPLETE_FILE).is_file():
                _push_step(remote, final, step)
            elif (partial / KEY_FILE).is_file():
                _push_partial(remote, partial, step)
        except FileNotFoundError as moved:
            LOGGER.info("run mirror: %s moved during the push (%s); it is pushed next time", step, moved)


def _restorable_partial(names: dict[str, int], name: str) -> list[str]:
    """The partial's KEY_FILE, checkpoints and every model directory whose record is mirrored, records last."""
    prefix = f"{name}/"
    inside = [entry.removeprefix(prefix) for entry in names if entry.startswith(prefix)]
    records = {relative for relative in inside if _matches(relative, RECORDS)}
    models = tuple(record.removesuffix(".json") + "/" for record in records)
    return sorted(
        prefix + relative for relative in inside
        if relative == KEY_FILE
        or relative in records
        or relative.startswith(models)
        or (_matches(relative, CHECKPOINTS) and not relative.rsplit("/", 1)[-1].startswith("."))
    )


def _matches(relative: str, pattern: str) -> bool:
    """Whether a '/'-separated name matches a glob pattern component by component, as ``Path.glob`` would."""
    parts, wanted = relative.split("/"), pattern.split("/")
    return len(parts) == len(wanted) and all(fnmatchcase(part, want) for part, want in zip(parts, wanted))


def restore(run_directory: Path, remote: Remote) -> None:
    """Fill a run directory from the mirror: every completed step, and each partial step's durable units, for every step
    this run directory does not hold already. Each is downloaded aside and renamed into place, so an interrupted restore
    leaves no step directory the driver would take for its own."""
    names = remote.sizes("")
    run_directory.mkdir(parents=True, exist_ok=True)
    for step in STEP_NAMES:
        final, partial = run_directory / step, run_directory / f"{step}{PARTIAL}"
        if final.exists() or partial.exists():
            continue
        if f"{step}/{COMPLETE_FILE}" in names:
            name, target = step, final
            chosen = sorted(entry for entry in names if entry.startswith(f"{step}/"))
        elif f"{step}{PARTIAL}/{KEY_FILE}" in names:
            name, target = f"{step}{PARTIAL}", partial
            chosen = _restorable_partial(names, name)
        else:
            continue
        aside = run_directory / f"{name}{RESTORING}"
        if aside.exists():
            shutil.rmtree(aside)
        aside.mkdir()
        listed = remote.read(f"{name}{DIRECTORIES}") if target == final else None
        for directory in json.loads(listed) if listed is not None else []:
            (aside / directory).mkdir(parents=True, exist_ok=True)
        _transfer([(aside / entry.removeprefix(f"{name}/"), entry) for entry in chosen], lambda path, entry: remote.download(entry, path))
        os.rename(aside, target)
        LOGGER.info("run mirror: restored %s, %d files, %.1f GB", name, len(chosen), sum(names[entry] for entry in chosen) / 1e9)

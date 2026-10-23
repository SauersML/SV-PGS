"""Whole batches of the score and effects steps, below their keyed partial directories.

The step key covers the completed inputs and the code that reads them; the batch identity adds the ordered models
and their output columns. A payload is reusable only beside its record, with that identity and every file's size and
content digest intact. A half-written payload, an interrupted mirror upload or a same-size damaged file is work to
do again, never a result to assemble. These are workspace results, with the same access rules as the final scores
and per-record effects, and are not exportable tables.

A score batch's store read is itself kept part way through (``BatchReads``, ``fast_scoring.ReadCheckpoint``): one
read of every scored row, tens of minutes to over an hour at the All of Us whole-genome workload [real], is the one
piece of the post-fit steps a preemption could otherwise lose whole. Its kept reads are batches of their own beside
it, published and mirrored the same way, and removed once the batch is committed.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)
BATCHES = "batches"
RECORD = "complete.json"
FORMAT = "score-effects-batches-v1"
READ_SECONDS = "SVPGS_SCORE_READ_CHECKPOINT_SECONDS"
"""The environment variable giving the seconds between a score batch's kept reads (launcher/workspace sets it). Unset,
a batch's read is not kept part way through, as before."""
READS = ".read-"
"""Between a batch's directory name and its kept read's key: <batch>.read-<key>."""


def _digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class BatchCheckpoint:
    """One immutable batch, published by renaming a directory after all its files are closed. Hidden directories are
    work in progress and the mirror ignores them. A repaired batch gets a fresh generation in its record even when
    its results have the same bytes: the mirror must replace a damaged older payload, not skip it by its old record.
    ``step_key`` is the partial's KEY_FILE, not a fit key inferred from filenames or modification times.
    """

    def __init__(self, directory: Path, step_key: str, identity: Any, files: Sequence[str], prefix: str = "") -> None:
        self.identity = json.loads(json.dumps({"format": FORMAT, "step_key": step_key, "batch": identity}, sort_keys=True))
        key = hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode()).hexdigest()
        self.path = directory / BATCHES / f"{prefix}{key}"
        self.files = tuple(files)

    def read(self) -> Path | None:
        """The whole, matching batch, or None. The record is checked before any of its values reach the assembled
        outputs; the file list comes from the caller, so a damaged record cannot redirect reads outside the batch."""
        try:
            record = json.loads((self.path / RECORD).read_text(encoding="utf-8"))
            if record["identity"] != self.identity or set(record["files"]) != set(self.files):
                raise ValueError("batch identity or files differ")
            for name in self.files:
                path = self.path / name
                if record["files"][name] != {"bytes": path.stat().st_size, "sha256": _digest(path)}:
                    raise ValueError(f"batch file {name} differs")
        except (OSError, ValueError, KeyError, TypeError) as error:
            if self.path.exists():
                LOGGER.warning("post-fit checkpoint: %s is incomplete or mismatched; doing it again (%s)", self.path.name, error)
            return None
        return self.path

    @contextmanager
    def write(self) -> Iterator[Path]:
        """Write the caller's files aside, then record and publish them together. The previous generation, if any,
        is hidden before it is removed; a mirror racing the replacement either loses its path or carries a payload
        whose digest the next reader rejects. No partially written generation is accepted by ``read``."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        generation = uuid.uuid4().hex
        temporary = self.path.with_name(f".{self.path.name}-{generation}")
        temporary.mkdir()
        try:
            yield temporary
            record = {
                "identity": self.identity, "generation": generation,
                "files": {name: {"bytes": (temporary / name).stat().st_size, "sha256": _digest(temporary / name)} for name in self.files},
            }
            (temporary / RECORD).write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
            if self.path.exists():
                previous = self.path.with_name(f".{self.path.name}-replaced-{generation}")
                os.rename(self.path, previous)
                shutil.rmtree(previous)
            os.rename(temporary, self.path)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


class BatchReads:
    """The kept reads of one score batch (``fast_scoring.ReadCheckpoint``), each a batch of its own named
    <batch>.read-<key> beside it, its identity the batch's and the read's. ``seconds`` apart at the least: each keeps
    every lane's partial sums (its columns x people x 8 bytes a lane: ~0.1 GB for one trait's models at the whole-genome
    workload, an estimate), which the mirror sends again on each one."""

    def __init__(self, batch: BatchCheckpoint, seconds: float) -> None:
        self._batch, self._seconds = batch, seconds
        self._kept = time.monotonic()

    def _checkpoint(self, identity: Any, files: Sequence[str]) -> BatchCheckpoint:
        return BatchCheckpoint(
            self._batch.path.parent.parent, self._batch.identity["step_key"], {"batch": self._batch.identity["batch"], "read": identity}, files,
            prefix=f"{self._batch.path.name}{READS}",
        )

    def due(self) -> bool:
        return time.monotonic() - self._kept >= self._seconds

    def read(self, identity: Any, files: Sequence[str]) -> Path | None:
        return self._checkpoint(identity, files).read()

    @contextmanager
    def write(self, identity: Any, files: Sequence[str]) -> Iterator[Path]:
        with self._checkpoint(identity, files).write() as temporary:
            yield temporary
        self._kept = time.monotonic()


def batch_reads(batch: BatchCheckpoint) -> BatchReads | None:
    """The kept reads of ``batch`` at the interval READ_SECONDS gives, or None where it is unset."""
    seconds = os.environ.get(READ_SECONDS)
    return None if not seconds else BatchReads(batch, float(seconds))


def discard_reads(directory: Path, batch: BatchCheckpoint | None = None) -> None:
    """Remove ``batch``'s kept reads, or every batch's, with any a stopped write left hidden: a committed batch holds
    its scores, and a step completes without the reads it no longer needs."""
    for pattern in ("*", ".*") if batch is None else (batch.path.name, f".{batch.path.name}"):
        for path in (directory / BATCHES).glob(f"{pattern}{READS}*"):
            shutil.rmtree(path)

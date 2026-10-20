"""Whole batches of the score and effects steps, below their keyed partial directories.

The step key covers the completed inputs and the code that reads them; the batch identity adds the ordered models
and their output columns. A payload is reusable only beside its record, with that identity and every file's size and
content digest intact. A half-written payload, an interrupted mirror upload or a same-size damaged file is work to
do again, never a result to assemble. These are workspace results, with the same access rules as the final scores
and per-record effects, and are not exportable tables.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import shutil
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)
BATCHES = "batches"
RECORD = "complete.json"
FORMAT = "score-effects-batches-v1"


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(io.DEFAULT_BUFFER_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


class BatchCheckpoint:
    """One immutable batch, published by renaming a directory after all its files are closed. Hidden directories are
    work in progress and the mirror ignores them. A repaired batch gets a fresh generation in its record even when
    its results have the same bytes: the mirror must replace a damaged older payload, not skip it by its old record.
    ``step_key`` is the partial's KEY_FILE, not a fit key inferred from filenames or modification times.
    """

    def __init__(self, directory: Path, step_key: str, identity: Any, files: Sequence[str]) -> None:
        self.identity = json.loads(json.dumps({"format": FORMAT, "step_key": step_key, "batch": identity}, sort_keys=True))
        key = hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode()).hexdigest()
        self.path = directory / BATCHES / key
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

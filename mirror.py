"""The resume units in the run mirror: ``run_mirror.push`` and ``restore`` (wrapped) also carry ``live_restart/``.

The mirror's own code lists, writes and deletes only its steps' names, so the units live beside them under their own
prefix and are moved here, after the mirror's own work, which this never changes or fails:

- **push** uploads each unit file whose size or modification time differs from what this disk last pushed or restored
  (``.mirrored.json`` beside the units; an archive rewritten in place keeps its name and often its size), and deletes
  the remote copy of each file this disk pushed or restored that is gone here (a released unit). A file another job
  wrote and this disk never held is never deleted, so per-arm jobs pushing to one mirror leave each other's units.
- **restore** downloads every remote unit file this disk does not hold, each aside, and renames it into place only
  where it is whole (a saved state's checksum holds; ``units.whole``).

An archive is whole on either side: the fit renames each into place, and GCS creates an object only when its upload
completes."""
from __future__ import annotations

import functools
import json
import logging
import os
from pathlib import Path
from typing import Any

from benchmarks.live_resume.units import ROOT, whole

LOGGER = logging.getLogger("sv_pgs.live_resume")
LEDGER = ".mirrored.json"
"""What this disk last pushed or restored of each unit file: its relative name, size and modification time."""


def _hidden(relative: str) -> bool:
    return any(part.startswith(".") for part in relative.split("/"))


def _ledger(root: Path) -> dict[str, list[int]]:
    try:
        return {str(name): [int(value) for value in entry] for name, entry in json.loads((root / LEDGER).read_text(encoding="utf-8")).items()}
    except (OSError, ValueError, TypeError, AttributeError):
        return {}


def _write_ledger(root: Path, ledger: dict[str, list[int]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    aside = root / f"{LEDGER}.writing"
    aside.write_text(json.dumps(ledger, sort_keys=True), encoding="utf-8")
    os.replace(aside, root / LEDGER)


def _stamp(path: Path) -> list[int]:
    status = path.stat()
    return [int(status.st_size), int(status.st_mtime_ns)]


def push_units(run_directory: Path, remote: Any) -> None:
    """Send the run's units that changed since this disk last sent or received them; delete the remote copies of the
    ones this disk released."""
    from sv_pgs.run_mirror import _transfer

    root = Path(run_directory) / ROOT
    if not root.is_dir():
        return
    ledger = _ledger(root)
    local = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_file() and not _hidden(relative):
            try:
                local[relative] = _stamp(path)
            except FileNotFoundError:
                continue
    changed = [relative for relative, stamp in local.items() if ledger.get(relative) != stamp]
    sent: list[str] = []

    def send(path: Path, name: str) -> None:
        try:
            remote.upload(path, name)
        except FileNotFoundError:
            # Released while this push ran: the next push deletes its remote copy, if it has one.
            return
        sent.append(name.removeprefix(f"{ROOT}/"))

    _transfer([(root / relative, f"{ROOT}/{relative}") for relative in changed], send)
    for relative in sent:
        ledger[relative] = local[relative]
    gone = sorted(relative for relative in ledger if relative not in local and not (root / relative).is_file())
    if gone:
        remote.delete([f"{ROOT}/{relative}" for relative in gone])
        for relative in gone:
            del ledger[relative]
    _write_ledger(root, ledger)
    if sent or gone:
        LOGGER.info("run mirror: pushed %d resume unit files, removed %d released ones", len(sent), len(gone))


def restore_units(run_directory: Path, remote: Any) -> None:
    """Fetch every mirrored unit file this disk does not hold."""
    root = Path(run_directory) / ROOT
    names = remote.sizes(f"{ROOT}/")
    wanted = []
    for name, size in sorted(names.items()):
        relative = name.removeprefix(f"{ROOT}/")
        target = root / relative
        if _hidden(relative) or (target.is_file() and target.stat().st_size == size):
            continue
        wanted.append((target.with_name(f".{target.name}.restoring"), name, target))
    if not wanted:
        return
    remote.download_many([(aside, name) for aside, name, _target in wanted])
    ledger = _ledger(root)
    restored = 0
    for aside, name, target in wanted:
        if not whole(aside, target.name):
            # Never a resume start: the fit makes this one again.
            aside.unlink(missing_ok=True)
            LOGGER.warning("run mirror: the resume unit file %s is not whole; it is not restored", name)
            continue
        os.replace(aside, target)
        ledger[name.removeprefix(f"{ROOT}/")] = _stamp(target)
        restored += 1
    _write_ledger(root, ledger)
    LOGGER.info("run mirror: restored %d resume unit files, %.2f GB", restored, sum(names[name] for _aside, name, _target in wanted) / 1e9)


def install() -> None:
    """Wrap ``run_mirror.push`` and ``restore`` so each also moves the units, after its own work; a failure of the units'
    part is logged and never fails the mirror's."""
    from sv_pgs import run_mirror

    if getattr(run_mirror.push, "__wrapped__", None) is not None:
        return
    original_push, original_restore = run_mirror.push, run_mirror.restore

    @functools.wraps(original_push)
    def push(run_directory: Path, remote: Any) -> None:
        original_push(run_directory, remote)
        try:
            push_units(run_directory, remote)
        except Exception as error:
            LOGGER.warning("run mirror: the resume units were not pushed (%s: %s)", type(error).__name__, error)

    @functools.wraps(original_restore)
    def restore(run_directory: Path, remote: Any, mount: Path | None = None) -> None:
        original_restore(run_directory, remote, mount)
        try:
            restore_units(run_directory, remote)
        except Exception as error:
            LOGGER.warning("run mirror: the resume units were not restored (%s: %s)", type(error).__name__, error)

    run_mirror.push, run_mirror.restore = push, restore

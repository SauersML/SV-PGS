from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sv_pgs.run_mirror import DirectoryRemote, push, restore
from sv_pgs.workspace_pipeline import COMPLETE_FILE, KEY_FILE


def _write(path: Path, content: str | bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, str):
        path.write_text(content)
    else:
        path.write_bytes(content)
    return path


def _tree(root: Path) -> dict[str, bytes]:
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()} if root.is_dir() else {}


def _run(root: Path) -> Path:
    """A run directory with two completed steps, a superseded one, and the fit step's partial: a checkpoint, a checkpoint
    still being written, scratch, a saved model with its record and a saved model without one yet."""
    _write(root / "samples" / "samples.json", "{}")
    _write(root / "samples" / COMPLETE_FILE, '{"key": "a"}')
    _write(root / "store" / "store" / "chunk" / "0", np.arange(64, dtype=np.uint8).tobytes())
    _write(root / "store" / COMPLETE_FILE, '{"key": "b"}')
    _write(root / "store.superseded-0123" / COMPLETE_FILE, '{"key": "old"}')
    partial = root / "fit.partial"
    _write(partial / KEY_FILE, "fitkey")
    _write(partial / "work" / "with_sv" / "checkpoints" / "k" / "model0.npz", b"checkpoint")
    _write(partial / "work" / "with_sv" / "checkpoints" / "k" / ".model1.npz.tmp.npz", b"half")
    _write(partial / "work" / "with_sv" / "scratch" / "state.npz", b"scratch")
    _write(partial / "models" / "snv_indel" / "asthma" / "model.pickle", b"model")
    _write(partial / "models" / "snv_indel" / "asthma.json", "{}")
    _write(partial / "models" / "snv_indel" / "gout" / "model.pickle", b"unrecorded")
    return root


def test_a_restore_brings_back_completed_steps_and_the_fit_s_durable_progress_only(tmp_path: Path) -> None:
    run, remote = _run(tmp_path / "run"), DirectoryRemote(tmp_path / "mirror")
    push(run, remote)
    restored = tmp_path / "restored"
    restore(restored, remote)
    expected = {
        name: content for name, content in _tree(run).items()
        if not name.startswith("store.superseded")
        and "scratch" not in name and "/." not in name and "/gout/" not in name
    }
    assert _tree(restored) == expected


def test_a_push_sends_only_what_changed_and_a_finished_step_drops_its_partial(tmp_path: Path) -> None:
    run, mirror = _run(tmp_path / "run"), tmp_path / "mirror"
    remote = DirectoryRemote(mirror)
    push(run, remote)
    pushed = {path: path.stat().st_mtime_ns for path in mirror.rglob("*") if path.is_file()}
    push(run, remote)
    assert {path: path.stat().st_mtime_ns for path in mirror.rglob("*") if path.is_file()} == pushed
    # The fit finishes: the driver renames its partial into place, and the mirror keeps the step instead.
    (run / "fit.partial").rename(run / "fit")
    _write(run / "fit" / COMPLETE_FILE, '{"key": "c"}')
    push(run, remote)
    assert not (mirror / "fit.partial").exists() and (mirror / "fit" / COMPLETE_FILE).is_file()


def test_a_partial_under_another_key_replaces_the_mirrored_one(tmp_path: Path) -> None:
    run, mirror = _run(tmp_path / "run"), tmp_path / "mirror"
    remote = DirectoryRemote(mirror)
    push(run, remote)
    _write(run / "fit.partial" / KEY_FILE, "newkey")
    (run / "fit.partial" / "models" / "snv_indel" / "asthma.json").unlink()
    push(run, remote)
    assert (mirror / "fit.partial" / KEY_FILE).read_text() == "newkey"
    assert not (mirror / "fit.partial" / "models" / "snv_indel" / "asthma.json").exists()


def test_an_interrupted_push_is_never_restored_and_a_restore_keeps_what_the_disk_holds(tmp_path: Path) -> None:
    run, mirror = _run(tmp_path / "run"), tmp_path / "mirror"
    remote = DirectoryRemote(mirror)
    push(run, remote)
    # A push cut off before its marker: the step's files without COMPLETE_FILE.
    (mirror / "store" / COMPLETE_FILE).unlink()
    restored = tmp_path / "restored"
    _write(restored / "samples" / COMPLETE_FILE, "local")
    restore(restored, remote)
    assert not (restored / "store").exists()
    assert (restored / "samples" / COMPLETE_FILE).read_text() == "local"
    assert not list(restored.glob("*.restoring"))


class _FailingRemote(DirectoryRemote):
    def __init__(self, root: Path, failing: str) -> None:
        super().__init__(root)
        self.failing = failing

    def download(self, name: str, path: Path) -> None:
        if name.endswith(self.failing):
            raise ConnectionError("preempted")
        super().download(name, path)


def test_an_interrupted_restore_leaves_no_step_directory(tmp_path: Path) -> None:
    run, mirror = _run(tmp_path / "run"), tmp_path / "mirror"
    push(run, DirectoryRemote(mirror))
    restored = tmp_path / "restored"
    with pytest.raises(ConnectionError):
        restore(restored, _FailingRemote(mirror, "chunk/0"))
    assert not (restored / "store").exists()
    restore(restored, DirectoryRemote(mirror))
    assert (restored / "store" / COMPLETE_FILE).is_file() and not list(restored.glob("*.restoring"))

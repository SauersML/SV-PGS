"""Post-fit batches survive only as whole, keyed results, on disk and through the run mirror."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from sv_pgs.postfit_checkpoint import BatchCheckpoint, RECORD
from sv_pgs.run_mirror import PushIncomplete, push, restore
from sv_pgs.workspace_pipeline import COMPLETE_FILE, KEY_FILE
from tests.directory_remote import DirectoryRemote


def _batch(partial: Path, key: str = "step", identity=None) -> BatchCheckpoint:
    partial.mkdir(parents=True, exist_ok=True)
    (partial / KEY_FILE).write_text(key)
    return BatchCheckpoint(partial, key, {"models": [0, 1]} if identity is None else identity, ("values", "summary.json"))


def _commit(batch: BatchCheckpoint) -> None:
    with batch.write() as directory:
        (directory / "values").write_bytes(b"whole values")
        (directory / "summary.json").write_text('{"rows": []}')


@pytest.mark.parametrize("damage", ["missing", "truncated", "same_size", "record", "identity", "files"])
def test_a_damaged_batch_is_redone_even_when_its_size_has_not_changed(tmp_path: Path, damage: str) -> None:
    batch = _batch(tmp_path / "score.partial")
    _commit(batch)
    assert batch.read() == batch.path
    old_record = (batch.path / RECORD).read_bytes()
    if damage == "missing":
        (batch.path / "values").unlink()
    elif damage == "truncated":
        (batch.path / "values").write_bytes(b"whole")
    elif damage == "same_size":
        (batch.path / "values").write_bytes(b"wrong values")
    elif damage == "record":
        (batch.path / RECORD).write_text("{")
    else:
        record = json.loads(old_record)
        if damage == "identity":
            record["identity"]["step_key"] = "another step"
        else:
            record["files"] = {"../outside": {"bytes": 0, "sha256": ""}}
        (batch.path / RECORD).write_text(json.dumps(record))
    assert batch.read() is None
    _commit(batch)
    assert batch.read() == batch.path
    assert (batch.path / RECORD).read_bytes() != old_record


def test_batch_keys_cover_the_step_and_ordered_membership(tmp_path: Path) -> None:
    partial = tmp_path / "score.partial"
    batch = _batch(partial)
    _commit(batch)
    assert _batch(partial, key="changed code or upstream").read() is None
    assert _batch(partial, identity={"models": [1, 0]}).read() is None
    assert _batch(partial, identity={"models": [0]}).read() is None
    assert batch.read() == batch.path


@pytest.mark.parametrize("step", ["score", "effects"])
def test_only_committed_batches_are_mirrored_and_completion_drops_the_batch_copies(tmp_path: Path, step: str) -> None:
    run, remote = tmp_path / "run", DirectoryRemote(tmp_path / "mirror")
    partial = run / f"{step}.partial"
    batch = _batch(partial)
    with pytest.raises(RuntimeError, match="preempted"):
        with batch.write() as directory:
            (directory / "values").write_bytes(b"half")
            push(run, remote)
            assert not any(name.endswith(RECORD) for name in remote.sizes(""))
            raise RuntimeError("preempted")
    assert batch.read() is None
    _commit(batch)
    push(run, remote)
    mirrored = remote.sizes("")
    push(run, remote)
    assert remote.sizes("") == mirrored
    restored = tmp_path / "restored"
    restore(restored, remote)
    assert _batch(restored / partial.name).read() is not None
    # The final output is separate from the internal batch protocol, and is all a completed step needs to restore.
    (partial / "output").write_bytes(b"assembled")
    (partial / COMPLETE_FILE).write_text('{"key": "step"}')
    partial.rename(run / step)
    push(run, remote)
    assert not remote.sizes(f"{step}.partial/")
    assert not remote.sizes(f"{step}/batches/")
    restore(tmp_path / "completed", remote)
    assert (tmp_path / "completed" / step / "output").read_bytes() == b"assembled"


class _InterruptedUpload(DirectoryRemote):
    def upload(self, path: Path, name: str) -> None:
        if name.endswith("/values"):
            raise ConnectionError("preempted upload")
        super().upload(path, name)


@pytest.mark.parametrize("replacement", [False, True])
def test_an_interrupted_upload_never_leaves_a_completion_record(tmp_path: Path, replacement: bool) -> None:
    run, mirror = tmp_path / "run", tmp_path / "mirror"
    batch = _batch(run / "effects.partial")
    _commit(batch)
    if replacement:
        push(run, DirectoryRemote(mirror))
        _commit(batch)
    with pytest.raises(PushIncomplete, match="preempted upload") as failed:
        push(run, _InterruptedUpload(mirror))
    assert len(failed.value.failures) == 1
    step, error = failed.value.failures[0]
    assert step == "effects" and isinstance(error, ConnectionError)
    restored = tmp_path / "restored"
    restore(restored, DirectoryRemote(mirror))
    assert _batch(restored / "effects.partial").read() is None
    push(run, DirectoryRemote(mirror))
    shutil.rmtree(restored)
    restore(restored, DirectoryRemote(mirror))
    assert _batch(restored / "effects.partial").read() is not None


def test_a_replacement_during_upload_cannot_commit_a_mixed_generation(tmp_path: Path) -> None:
    run, mirror = tmp_path / "run", tmp_path / "mirror"
    batch = _batch(run / "score.partial")
    _commit(batch)

    class ReplacingRemote(DirectoryRemote):
        def upload(self, path: Path, name: str) -> None:
            super().upload(path, name)
            if name.endswith("/values"):
                _commit(batch)

    # push treats the moved generation like a step renamed by the driver, and tries again next time.
    push(run, ReplacingRemote(mirror))
    restored = tmp_path / "restored"
    restore(restored, DirectoryRemote(mirror))
    assert _batch(restored / "score.partial").read() is None
    push(run, DirectoryRemote(mirror))
    shutil.rmtree(restored)
    restore(restored, DirectoryRemote(mirror))
    assert _batch(restored / "score.partial").read() is not None


def test_a_new_step_key_replaces_every_mirrored_batch_of_the_previous_step(tmp_path: Path) -> None:
    run, remote = tmp_path / "run", DirectoryRemote(tmp_path / "mirror")
    partial = run / "effects.partial"
    batch = _batch(partial)
    _commit(batch)
    push(run, remote)
    shutil.rmtree(partial)
    current = _batch(partial, key="new inputs")
    _commit(current)
    push(run, remote)
    restored = tmp_path / "restored"
    restore(restored, remote)
    assert not (restored / partial.name / "batches" / batch.path.name).exists()
    assert _batch(restored / partial.name, key="new inputs").read() is not None

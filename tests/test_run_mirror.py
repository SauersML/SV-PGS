from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from sv_pgs.content_identity import GCS_MOUNTS
from sv_pgs.run_mirror import DirectoryRemote, GcsRemote, mounted, push, restore
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
    (root / "store" / "calibration").mkdir()
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
    assert (restored / "store" / "calibration").is_dir()


def test_store_shards_push_their_converted_chromosomes_into_one_partial_and_a_restore_takes_each_whole_one(tmp_path: Path) -> None:
    """Each converted chromosome is a unit: its store arrays and its fill, reported and calibration files, its record
    chromosomes/<chromosome>.json last. chr1's files are not chr10's, and a chromosome without its record, or a batch's
    decoded codes, stays behind."""
    remote = DirectoryRemote(tmp_path / "mirror")
    shards = []
    for shard, chromosomes in enumerate((["chr1"], ["chr10", "chr2"])):
        partial = tmp_path / f"shard{shard}" / "store.partial"
        _write(partial / KEY_FILE, "storekey")
        for chromosome in chromosomes:
            _write(partial / "store" / "dosage" / "half0" / chromosome / "shard0", chromosome.encode())
            _write(partial / "store" / "stats" / "half0" / chromosome / "sum_code" / "values", b"sums")
            _write(partial / "store" / "variants" / chromosome / "position" / "values", b"positions")
            _write(partial / "store" / "loci" / chromosome / "tr_start" / "values", b"loci")
            for directory in ("fill", "reported"):
                _write(partial / directory / f"{chromosome}.npy", chromosome.encode())
            _write(partial / "calibration" / f"{chromosome}.npz", b"pairs")
            _write(partial / "batches" / chromosome / "batch0.codes.npy", b"decoded")
        _write(partial / "chromosomes" / f"{chromosomes[0]}.json", "{}")
        shards.append(partial.parent)
    for shard in shards:
        push(shard, remote)
    push(shards[0], remote)
    restored = tmp_path / "restored"
    restore(restored, remote)
    expected = {
        name: content for shard in shards for name, content in _tree(shard).items()
        if "/batches/" not in name and "chr2" not in name
    }
    assert _tree(restored) == expected
    assert {"store.partial/store/dosage/half0/chr1/shard0", "store.partial/store/dosage/half0/chr10/shard0"} <= set(expected)


def test_a_step_made_again_under_another_key_drops_its_stale_mirrored_copy(tmp_path: Path) -> None:
    run, mirror = _run(tmp_path / "run"), tmp_path / "mirror"
    remote = DirectoryRemote(mirror)
    push(run, remote)
    # The driver sets the completed store aside and makes it again: the mirror must not restore the old one over it.
    (run / "store").rename(run / "store.superseded-b")
    _write(run / "store.partial" / KEY_FILE, "newstore")
    push(run, remote)
    restored = tmp_path / "restored"
    restore(restored, remote)
    assert not (restored / "store").exists() and (restored / "store.partial" / KEY_FILE).read_text() == "newstore"


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


def _stores(root: Path, steps: tuple[str, ...]) -> Path:
    """A run directory whose completed steps ``steps`` each hold a store payload, as the store, measurement and
    direct_sv steps do."""
    for position, step in enumerate(steps):
        _write(root / step / "store" / "chunk" / "0", bytes([position]) * 16)
        (root / step / "store" / "loci").mkdir()
        _write(root / step / "summary.json", "{}")
        _write(root / step / COMPLETE_FILE, f'{{"key": "{step}"}}')
    return root


def test_a_store_a_later_step_s_replaces_is_linked_to_the_mount_and_the_last_is_downloaded(tmp_path: Path) -> None:
    run, mirror = _stores(tmp_path / "run", ("store", "measurement", "direct_sv")), tmp_path / "mirror"
    remote = DirectoryRemote(mirror)
    push(run, remote)
    restored = tmp_path / "restored"
    # The mirror's directory is where a DirectoryRemote's objects show, as the bucket's mount shows a GcsRemote's.
    restore(restored, remote, mirror)
    for step in ("store", "measurement"):
        assert (restored / step / "store").is_symlink() and (restored / step / "store").resolve() == (mirror / step / "store").resolve()
        assert (restored / step / "summary.json").is_file() and not (restored / step / "summary.json").is_symlink()
    assert not (restored / "direct_sv" / "store").is_symlink() and (restored / "direct_sv" / "store" / "loci").is_dir()
    # Read through the links, the restored run holds what the pushed one did.
    followed = {
        (Path(directory) / name).relative_to(restored).as_posix(): (Path(directory) / name).read_bytes()
        for directory, _directories, names in os.walk(restored, followlinks=True) for name in names
    }
    assert followed == _tree(run)


def test_the_store_the_next_step_reads_is_downloaded_and_without_a_mount_every_store_is(tmp_path: Path) -> None:
    run, mirror = _stores(tmp_path / "run", ("store", "measurement")), tmp_path / "mirror"
    remote = DirectoryRemote(mirror)
    push(run, remote)
    # direct_sv has not run: it reads the measurement step's store, which is downloaded.
    restore(tmp_path / "linked", remote, mirror)
    assert (tmp_path / "linked" / "store" / "store").is_symlink() and not (tmp_path / "linked" / "measurement" / "store").is_symlink()
    restore(tmp_path / "copied", remote)
    assert not any(path.is_symlink() for path in (tmp_path / "copied").rglob("*"))
    assert _tree(tmp_path / "copied") == _tree(run)


def test_the_mirror_is_found_on_the_bucket_mount_that_shows_it(monkeypatch) -> None:
    monkeypatch.setenv(GCS_MOUNTS, '{"/mnt/disks/ops": "gs://ops", "/mnt/disks/cdr": "gs://cdr/v8"}')
    assert mounted("gs://ops/svpgs/run1/run") == Path("/mnt/disks/ops/svpgs/run1/run")
    assert mounted("gs://cdr/v8/svpgs") == Path("/mnt/disks/cdr/svpgs")
    assert mounted("gs://cdr/v7/svpgs") is None and mounted("gs://other/run") is None


def test_a_gcs_download_retries_a_broken_read_without_a_deadline(tmp_path: Path) -> None:
    calls = []

    class Blob:
        def __init__(self, name: str) -> None:
            self.name = name

        def download_to_filename(self, filename: str, **options) -> None:
            calls.append((self.name, filename, options))

    remote = GcsRemote.__new__(GcsRemote)
    remote._bucket, remote._root = type("Bucket", (), {"blob": lambda self, name: Blob(name)})(), "svpgs/run/"
    remote.download("store/chr1.codes", tmp_path / "a" / "chr1.codes")
    (name, filename, options), = calls
    assert name == "svpgs/run/store/chr1.codes" and filename == str(tmp_path / "a" / "chr1.codes") and (tmp_path / "a").is_dir()
    assert options["retry"].timeout is None

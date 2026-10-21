from __future__ import annotations

import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from sv_pgs import run_mirror
from sv_pgs.content_identity import GCS_MOUNTS
from sv_pgs.restart_store import MANIFEST, RestartStore, facts, unit_key
from sv_pgs.run_mirror import Base, DirectoryRemote, GcsRemote, bases_of, mounted, push, restore
from sv_pgs.workspace_pipeline import COMPLETE_FILE, KEY_FILE


def _write(path: Path, content: str | bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, str):
        path.write_text(content)
    else:
        path.write_bytes(content)
    return path


_BOOKKEEPING = {f"restart/{run_mirror.RECEIPT}"}
"""The mirror's own record of a disk's restart store, which no run pushes."""


def _tree(root: Path) -> dict[str, bytes]:
    """Every file under ``root`` but the mirror's own records (_BOOKKEEPING)."""
    return {
        path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() not in _BOOKKEEPING
    } if root.is_dir() else {}


def _unit(run: Path, arm: str, inputs: str, payload: bytes = b"gram", group: str = "model0") -> Path:
    """A sealed Stage 0 restart unit of ``arm`` (``restart_store``) made from ``inputs``: its LD file and statistics."""
    store = RestartStore(run / "restart" / arm)
    named = facts({"inputs": inputs})
    with store.build("stage0", unit_key(named), group=group, named=named) as directory:
        _write(directory / "ld" / "gram.f32", payload)
        _write(directory / "statistics.npz", b"statistics")
    return run / "restart" / arm / "stage0" / unit_key(named)


def _followed(root: Path) -> dict[str, bytes]:
    """``_tree``, read through the links a restore leaves to the mount."""
    return {
        (Path(directory) / name).relative_to(root).as_posix(): (Path(directory) / name).read_bytes()
        for directory, _directories, names in os.walk(root, followlinks=True) for name in names
    }


def _run(root: Path) -> Path:
    """A run directory with two completed steps, a superseded one, the fit step's partial (an engine checkpoint of the
    old layout and one still being written, scratch, a saved model with its record and a saved model without one yet)
    and the fit's restart store: a sealed unit, an EB start's mutable unit and a unit still being built."""
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
    _unit(root, "with_sv", "stage0")
    start = RestartStore(root / "restart" / "with_sv").mutable("eb", "start", group="model0", named=facts({"inputs": "eb"}))
    _write(start / "model0.npz", b"warm start")
    _write(root / "restart" / "with_sv" / "band" / ".unfinished" / "within.f32", b"half")
    return root


def test_a_restore_brings_back_completed_steps_and_the_fit_s_durable_progress_only(tmp_path: Path) -> None:
    run, remote = _run(tmp_path / "run"), DirectoryRemote(tmp_path / "mirror")
    push(run, remote)
    restored = tmp_path / "restored"
    restore(restored, remote)
    # The old layout's engine checkpoints are keyed by the whole package's source, which no current fit matches.
    expected = {
        name: content for name, content in _tree(run).items()
        if not name.startswith("store.superseded")
        and "scratch" not in name and "/." not in name and "/gout/" not in name and "/checkpoints/" not in name
    }
    assert {name for name in expected if name.startswith("restart/")} == {
        f"restart/with_sv/stage0/{_unit(tmp_path / 'other', 'with_sv', 'stage0').name}/{name}" for name in (MANIFEST, "ld/gram.f32", "statistics.npz")
    } | {f"restart/with_sv/eb/start/{name}" for name in (MANIFEST, "model0.npz")}
    record = "fit.partial/models/snv_indel/asthma.json"
    expected[record] = (json.dumps({"fit_key": "fitkey", "model_files": {"model.pickle": len(b"model")}}, indent=2, sort_keys=True) + "\n").encode()
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
        if (Path(directory) / name).relative_to(restored).as_posix() not in _BOOKKEEPING
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


def test_a_restore_downloads_a_step_s_largest_objects_first(tmp_path: Path) -> None:
    """In name order chr1's codes go before chr10's and chr2's last; by size, chr2's (the largest here) go first."""
    run = tmp_path / "run"
    for chromosome, size in (("chr1", 240), ("chr10", 80), ("chr2", 400)):
        _write(run / "store" / "store" / "dosage" / "half0" / chromosome / "c" / "0" / "0", bytes(size))
    _write(run / "store" / COMPLETE_FILE, '{"key": "b"}')
    push(run, DirectoryRemote(tmp_path / "mirror"))
    requested = []

    class Recording(DirectoryRemote):
        def download_many(self, pairs: list[tuple[Path, str]]) -> None:
            requested.extend(name for _path, name in pairs)
            super().download_many(pairs)

    restore(tmp_path / "restored", Recording(tmp_path / "mirror"))
    assert [name.split("/")[4] for name in requested[:3]] == ["chr2", "chr1", "chr10"] and _tree(tmp_path / "restored") == _tree(run)


def test_the_mirror_is_found_on_the_bucket_mount_that_shows_it(monkeypatch) -> None:
    monkeypatch.setenv(GCS_MOUNTS, '{"/mnt/disks/ops": "gs://ops", "/mnt/disks/cdr": "gs://cdr/v8"}')
    assert mounted("gs://ops/svpgs/run1/run") == Path("/mnt/disks/ops/svpgs/run1/run")
    assert mounted("gs://cdr/v8/svpgs") == Path("/mnt/disks/cdr/svpgs")
    assert mounted("gs://cdr/v7/svpgs") is None and mounted("gs://other/run") is None


def test_a_gcs_download_retries_a_broken_read_without_a_deadline_and_hashes_the_bytes_it_writes(tmp_path: Path) -> None:
    calls = []

    class Blob:
        def __init__(self, name: str) -> None:
            self.name = name

        def download_to_file(self, handle, **options) -> None:
            calls.append((self.name, options))
            # A resumed read writes on from the last byte written.
            handle.write(b"chr1 ")
            handle.write(b"codes")

    remote = GcsRemote.__new__(GcsRemote)
    remote._bucket, remote._root = type("Bucket", (), {"blob": lambda self, name: Blob(name)})(), "svpgs/run/"
    digest = remote.download("store/chr1.codes", tmp_path / "a" / "chr1.codes")
    (name, options), = calls
    assert name == "svpgs/run/store/chr1.codes" and (tmp_path / "a" / "chr1.codes").read_bytes() == b"chr1 codes"
    assert options["retry"].timeout is None and digest == hashlib.sha256(b"chr1 codes").hexdigest()


def test_a_download_started_again_from_its_first_byte_is_hashed_again_and_any_other_seek_is_not_hashed(tmp_path: Path) -> None:
    with open(tmp_path / "codes", "wb") as handle:
        written = run_mirror._HashingFile(handle)
        written.write(b"transcoded once")
        written.seek(0)
        written.write(b"codes")
    assert (tmp_path / "codes").read_bytes() == b"codes" and written.hexdigest() == hashlib.sha256(b"codes").hexdigest()
    with open(tmp_path / "codes", "wb") as handle:
        written = run_mirror._HashingFile(handle)
        written.write(b"codes")
        written.seek(2)
    assert written.hexdigest() is None


def test_a_restore_keeps_the_digest_of_every_byte_it_wrote_so_the_store_digest_reads_none_of_them(tmp_path: Path, monkeypatch) -> None:
    from sv_pgs import content_identity

    run, remote = _stores(tmp_path / "run", ("store", "measurement", "direct_sv")), DirectoryRemote(tmp_path / "mirror")
    monkeypatch.delenv(content_identity.RESTORED_DIGESTS, raising=False)
    expected = {step: content_identity.tree_digest(run / step / "store") for step in ("store", "measurement", "direct_sv")}
    push(run, remote)
    monkeypatch.setenv(content_identity.RESTORED_DIGESTS, str(tmp_path / "restored-digests"))
    restored = tmp_path / "restored"
    restore(restored, remote, tmp_path / "mirror")

    def unread(path, stamp):
        pytest.fail(f"{path} was read again after its restore")

    monkeypatch.setattr(content_identity, "_hashed", unread)
    # The store the fit reads (direct_sv's) was downloaded and hashed as it was written.
    assert content_identity.tree_digest(restored / "direct_sv" / "store") == expected["direct_sv"]
    # A store linked to the mount was not downloaded, so no restored digest claims it.
    with pytest.raises(pytest.fail.Exception):
        content_identity.tree_digest(restored / "store" / "store" / "chunk")


def test_a_gcs_restore_downloads_each_object_in_a_spawned_process_through_that_process_s_client(tmp_path: Path, monkeypatch) -> None:
    calls, contexts = [], []

    class Client:
        def download(self, name: str, path: Path) -> None:
            calls.append((name, path))

    class Pool(ThreadPoolExecutor):
        def __init__(self, max_workers: int, mp_context) -> None:
            contexts.append(mp_context.get_start_method())
            super().__init__(max_workers)

    monkeypatch.setitem(run_mirror._PROCESS_REMOTES, "gs://ops/svpgs/run", Client())
    monkeypatch.setattr(run_mirror, "ProcessPoolExecutor", Pool)
    remote = GcsRemote.__new__(GcsRemote)
    remote._url = "gs://ops/svpgs/run"
    pairs = [(tmp_path / f"chunk{index}", f"store/chunk{index}") for index in range(7)]
    remote.download_many(pairs)
    assert sorted(calls) == sorted((name, path) for path, name in pairs) and contexts == ["spawn"]



def test_a_saved_model_marker_does_not_hide_a_missing_companion(tmp_path: Path) -> None:
    run, mirror = _run(tmp_path / "run"), tmp_path / "mirror"
    remote = DirectoryRemote(mirror)
    push(run, remote)
    payload = mirror / "fit.partial" / "models" / "snv_indel" / "asthma" / "model.pickle"
    payload.unlink()
    push(run, remote)
    assert payload.read_bytes() == b"model"
    marker = json.loads((payload.parent.with_suffix(".json")).read_text())
    assert marker["fit_key"] == "fitkey" and marker["model_files"] == {"model.pickle": len(b"model")}


def test_a_replacement_model_loses_its_old_marker_until_its_upload_finishes(tmp_path: Path) -> None:
    run, mirror = _run(tmp_path / "run"), tmp_path / "mirror"
    remote = DirectoryRemote(mirror)
    push(run, remote)
    model = run / "fit.partial" / "models" / "snv_indel" / "asthma"
    _write(model / "model.pickle", b"replacement model")

    class InterruptedUpload(DirectoryRemote):
        def upload(self, path, name):
            if name.endswith("model.pickle"):
                raise ConnectionError("interrupted replacement")
            super().upload(path, name)

    with pytest.raises(ConnectionError, match="interrupted replacement"):
        push(run, InterruptedUpload(mirror))
    restored = tmp_path / "restored"
    restore(restored, remote)
    assert not (restored / "fit.partial" / "models" / "snv_indel" / "asthma").exists()
    push(run, remote)
    assert (mirror / "fit.partial" / "models" / "snv_indel" / "asthma.json").is_file()


def test_an_arm_record_from_another_fit_key_is_not_pushed(tmp_path: Path) -> None:
    run = _run(tmp_path / "run")
    record = run / "fit.partial" / "models" / "snv_indel" / "asthma.json"
    record.write_text(json.dumps({"fit_key": "another fit"}))
    with pytest.raises(ValueError, match="different fit key"):
        push(run, DirectoryRemote(tmp_path / "mirror"))


def test_a_completed_step_marker_does_not_hide_a_missing_payload(tmp_path: Path) -> None:
    run, mirror = _run(tmp_path / "run"), tmp_path / "mirror"
    remote = DirectoryRemote(mirror)
    push(run, remote)
    payload = mirror / "store" / "store" / "chunk" / "0"
    payload.unlink()
    push(run, remote)
    assert payload.read_bytes() == (run / "store" / "store" / "chunk" / "0").read_bytes()
    assert (mirror / "store" / COMPLETE_FILE).is_file()


def _base(tmp_path: Path) -> tuple[Path, Base]:
    """A base run whose samples, store, measurement and fit steps are complete (the store and measurement each with a
    store payload), pushed to its mirror; and the base as another run names it."""
    run = _stores(tmp_path / "base", ("store", "measurement"))
    _write(run / "samples" / "samples.json", "{}")
    _write(run / "samples" / COMPLETE_FILE, '{"key": "samples"}')
    _write(run / "fit" / "models.json", "{}")
    _write(run / "fit" / COMPLETE_FILE, '{"key": "basefit"}')
    mirror = tmp_path / "base-mirror"
    push(run, DirectoryRemote(mirror))
    return run, Base(DirectoryRemote(mirror), "gs://ops/svpgs/base/run", mirror)


def test_a_base_carries_the_steps_before_the_fit_that_a_run_s_mirror_lacks_and_is_never_written(tmp_path: Path) -> None:
    base_run, base = _base(tmp_path)
    written = {path: path.stat().st_mtime_ns for path in (tmp_path / "base-mirror").rglob("*")}
    remote, run = DirectoryRemote(tmp_path / "mirror"), tmp_path / "run"
    restore(run, remote, tmp_path / "mirror", bases=[base])
    # The store a later step's replaces is linked on the base's mount and the measurement's downloaded; the base's fit
    # is its own run's result, never restored.
    assert (run / "store" / "store").resolve() == (tmp_path / "base-mirror" / "store" / "store").resolve()
    assert not (run / "measurement" / "store").is_symlink() and (run / "measurement" / "store" / "loci").is_dir()
    assert _followed(run) == {name: content for name, content in _tree(base_run).items() if not name.startswith("fit/")}
    # The run's own work goes to its own mirror; the steps the base carries do not, and the mirror names the base.
    _write(run / "fit.partial" / KEY_FILE, "fitkey")
    push(run, remote, bases=[base])
    push(run, remote, bases=[base])
    assert set(_tree(tmp_path / "mirror")) == {f"fit.partial/{KEY_FILE}", ".bases/ops/svpgs/base/run"}
    assert {path: path.stat().st_mtime_ns for path in (tmp_path / "base-mirror").rglob("*")} == written
    # The next job's new disk is filled the same way.
    again = tmp_path / "again"
    restore(again, remote, tmp_path / "mirror", bases=[base])
    assert _tree(again) == _tree(run) and (again / "store" / "store").resolve() == (run / "store" / "store").resolve()


def test_a_run_s_own_steps_come_before_a_base_s_and_a_step_made_as_the_base_holds_it_is_the_base_s(tmp_path: Path) -> None:
    _base_run, base = _base(tmp_path)
    remote, mirror = DirectoryRemote(tmp_path / "mirror"), tmp_path / "mirror"
    # This run's measurement was made under another key (another cohort), and its store is partway (a fan-out).
    run = _stores(tmp_path / "run", ("measurement",))
    _write(run / "measurement" / COMPLETE_FILE, '{"key": "other cohort"}')
    _write(run / "store.partial" / KEY_FILE, "storekey")
    push(run, remote, bases=[base])
    assert (mirror / "measurement" / COMPLETE_FILE).read_text() == '{"key": "other cohort"}'
    restored = tmp_path / "restored"
    restore(restored, remote, None, bases=[base])
    assert (restored / "measurement" / COMPLETE_FILE).read_text() == '{"key": "other cohort"}'
    assert (restored / "store.partial" / KEY_FILE).read_text() == "storekey" and not (restored / "store").exists()
    assert (restored / "samples" / COMPLETE_FILE).read_text() == '{"key": "samples"}'
    # Made again as the base holds it, the step is the base's: this mirror's stale copy is removed, not sent again.
    shutil.rmtree(run / "measurement")
    shutil.copytree(tmp_path / "base" / "measurement", run / "measurement")
    push(run, remote, bases=[base])
    assert not (mirror / "measurement").exists() and (mirror / ".bases" / "ops" / "svpgs" / "base" / "run").is_file()
    # Without the base, the same step is pushed whole, as before.
    push(run, remote)
    assert _tree(mirror / "measurement") == _tree(run / "measurement")


def test_a_restore_stops_when_a_step_changes_where_it_lies_while_it_is_downloaded(tmp_path: Path) -> None:
    _base_run, base = _base(tmp_path)

    class Pushed(DirectoryRemote):
        """The base's run pushes its measurement again, under another key, while this restore downloads it."""

        def download_many(self, pairs: list[tuple[Path, str]]) -> None:
            super().download_many(pairs)
            if any(name == f"measurement/{COMPLETE_FILE}" for _path, name in pairs):
                self.write(f"measurement/{COMPLETE_FILE}", b'{"key": "made again"}')

    run = tmp_path / "run"
    with pytest.raises(RuntimeError, match="measurement changed on the base gs://ops/svpgs/base/run"):
        restore(run, DirectoryRemote(tmp_path / "mirror"), None, bases=[Base(Pushed(tmp_path / "base-mirror"), base.url, base.mount)])
    assert (run / "store" / COMPLETE_FILE).is_file() and not (run / "measurement").exists()


def test_a_run_reads_the_bases_its_mirror_names_and_those_each_base_names(tmp_path: Path) -> None:
    """A run whose samples and store a base carries makes its measurement, and is itself named as a third run's base:
    the third run, given only that one, reads both, and once it has pushed it needs neither given."""
    opened = {name: Base(DirectoryRemote(tmp_path / name), f"gs://ops/svpgs/{name}/run", tmp_path / name) for name in ("s", "a", "r")}
    open_base = lambda url: opened[url.split("/")[-2]]
    store = _stores(tmp_path / "store-run", ("store",))
    _write(store / "samples" / COMPLETE_FILE, '{"key": "samples"}')
    push(store, opened["s"].remote)
    first = tmp_path / "first-run"
    restore(first, opened["a"].remote, None, bases=bases_of(opened["a"].url, opened["a"].remote, [opened["s"].url], open_base))
    _stores(first, ("measurement",))
    push(first, opened["a"].remote, bases=bases_of(opened["a"].url, opened["a"].remote, [opened["s"].url], open_base))
    assert set(_tree(tmp_path / "a")) == {".bases/ops/svpgs/s/run", "measurement.directories.json"} | {
        f"measurement/{name}" for name in _tree(first / "measurement")
    }
    # A base that names the run reading it is not read twice.
    opened["s"].remote.write(opened["r"].reference, b"")
    bases = bases_of(opened["r"].url, opened["r"].remote, [opened["a"].url], open_base)
    assert bases == [opened["a"], opened["s"]]
    third = tmp_path / "third-run"
    restore(third, opened["r"].remote, None, bases=bases)
    assert _followed(third) == _tree(first) and (third / "store" / "store").is_symlink()
    push(third, opened["r"].remote, bases=bases)
    assert set(_tree(tmp_path / "r")) == {".bases/ops/svpgs/a/run", ".bases/ops/svpgs/s/run"}
    assert bases_of(opened["r"].url, opened["r"].remote, [], open_base) == [opened["a"], opened["s"]]


def test_a_restart_unit_is_restored_whole_and_a_whole_mirrored_unit_is_never_replaced(tmp_path: Path) -> None:
    remote = DirectoryRemote(tmp_path / "mirror")
    unit = _unit(tmp_path / "first", "snv_indel", "stage0")
    push(tmp_path / "first", remote)
    manifest = tmp_path / "mirror" / "restart" / "snv_indel" / "stage0" / unit.name / MANIFEST
    pushed = manifest.read_bytes()
    # Another disk made the same unit again, its own build: the mirrored one stays as it is, never removed to be sent again.
    _unit(tmp_path / "second", "snv_indel", "stage0", b"same key, another build")
    push(tmp_path / "second", remote)
    assert manifest.read_bytes() == pushed and (manifest.parent / "ld" / "gram.f32").read_bytes() == b"gram"
    restore(tmp_path / "third", remote)
    found = RestartStore(tmp_path / "third" / "restart" / "snv_indel").unit("stage0", unit.name)
    assert found is not None and (found[0] / "ld" / "gram.f32").read_bytes() == b"gram" and found[1]["id"] == json.loads(pushed)["id"]
    # A disk that holds the unit already keeps its own.
    restore(tmp_path / "second", remote)
    assert (tmp_path / "second" / "restart" / "snv_indel" / "stage0" / unit.name / "ld" / "gram.f32").read_bytes() == b"same key, another build"


def test_an_interrupted_or_damaged_mirrored_unit_is_never_restored_and_the_next_push_completes_it(tmp_path: Path) -> None:
    run, mirror = tmp_path / "run", tmp_path / "mirror"
    unit = _unit(run, "snv_indel", "stage0")

    class Interrupted(DirectoryRemote):
        def upload_parts(self, path: Path, name: str) -> None:
            if name.endswith("statistics.npz"):
                raise ConnectionError("preempted")
            super().upload_parts(path, name)

    with pytest.raises(ConnectionError, match="preempted"):
        push(run, Interrupted(mirror))
    assert (mirror / "restart" / "snv_indel" / "stage0" / unit.name / "ld" / "gram.f32").is_file()
    restore(tmp_path / "first", DirectoryRemote(mirror))
    assert RestartStore(tmp_path / "first" / "restart" / "snv_indel").units("stage0") == []
    assert not list((tmp_path / "first" / "restart").rglob("gram.f32"))
    push(run, DirectoryRemote(mirror))
    # Damaged after its manifest: a file at another size is no unit, and the next push sends it again.
    (mirror / "restart" / "snv_indel" / "stage0" / unit.name / "ld" / "gram.f32").write_bytes(b"short")
    restore(tmp_path / "second", DirectoryRemote(mirror))
    assert RestartStore(tmp_path / "second" / "restart" / "snv_indel").units("stage0") == []
    push(run, DirectoryRemote(mirror))
    restore(tmp_path / "third", DirectoryRemote(mirror))
    assert RestartStore(tmp_path / "third" / "restart" / "snv_indel").unit("stage0", unit.name) is not None


def test_a_released_unit_leaves_the_mirror_only_from_a_disk_that_held_it_whole_there(tmp_path: Path) -> None:
    remote = DirectoryRemote(tmp_path / "mirror")
    snv = _unit(tmp_path / "origin", "snv_indel", "a")
    sv = _unit(tmp_path / "origin", "with_sv", "b")
    push(tmp_path / "origin", remote)

    def mirrored(arm: str) -> dict[str, int]:
        return remote.sizes(f"restart/{arm}/")

    # A job fitting snv_indel restores that arm alone, and its fit's release removes that arm's units alone.
    job = tmp_path / "job"
    restore(job, remote, arms=["snv_indel"])
    assert RestartStore(job / "restart" / "snv_indel").unit("stage0", snv.name) is not None and not (job / "restart" / "with_sv").exists()
    _unit(job, "with_sv", "c")
    RestartStore(job / "restart" / "snv_indel").release()
    push(job, remote, arms=["snv_indel"])
    assert mirrored("snv_indel") == {}
    assert set(mirrored("with_sv")) == {f"restart/with_sv/stage0/{sv.name}/{name}" for name in (MANIFEST, "ld/gram.f32", "statistics.npz")}
    # Another job's unit, which this disk never restored nor saw whole while it held it, stays; so does an interrupted
    # upload's, which no disk saw whole.
    stray = tmp_path / "stray"
    restore(stray, remote, arms=["snv_indel"])
    other = _unit(tmp_path / "other", "with_sv", "d")
    push(tmp_path / "other", remote)
    _write(tmp_path / "mirror" / "restart" / "with_sv" / "band" / "cut" / "within.f32", b"half")
    own = _unit(stray, "with_sv", "e")
    push(stray, remote)
    RestartStore(stray / "restart" / "with_sv").release()
    push(stray, remote)
    names = set(mirrored("with_sv"))
    assert not any(name.startswith(f"restart/with_sv/stage0/{own.name}/") for name in names)
    assert {f"restart/with_sv/stage0/{unit}/{MANIFEST}" for unit in (sv.name, other.name)} | {"restart/with_sv/band/cut/within.f32"} <= names
    # The disk that made and pushed a unit removes it once its fit releases it.
    RestartStore(tmp_path / "origin" / "restart" / "with_sv").release()
    push(tmp_path / "origin", remote)
    assert not any(name.startswith(f"restart/with_sv/stage0/{sv.name}/") for name in mirrored("with_sv"))


def test_a_push_removes_only_the_build_of_a_unit_its_disk_held(tmp_path: Path) -> None:
    remote = DirectoryRemote(tmp_path / "mirror")
    unit = _unit(tmp_path / "origin", "with_sv", "u")
    push(tmp_path / "origin", remote)
    first, second = tmp_path / "first", tmp_path / "second"
    restore(first, remote)
    restore(second, remote)
    manifest = f"restart/with_sv/stage0/{unit.name}/{MANIFEST}"
    RestartStore(first / "restart" / "with_sv").release()
    push(first, remote)
    assert manifest not in remote.sizes("restart/")
    # Another disk makes the key again and pushes its build: neither a disk that held the earlier build and releases
    # it now, nor one that released it already and forgot it, removes this one.
    _unit(tmp_path / "rebuilt", "with_sv", "u", b"another build")
    push(tmp_path / "rebuilt", remote)
    RestartStore(second / "restart" / "with_sv").release()
    push(second, remote)
    push(first, remote)
    assert (tmp_path / "mirror" / "restart" / "with_sv" / "stage0" / unit.name / "ld" / "gram.f32").read_bytes() == b"another build"


def test_a_unit_changed_on_the_mirror_during_its_restore_is_left_behind_and_kept_there(tmp_path: Path) -> None:
    mirror = tmp_path / "mirror"
    unit = _unit(tmp_path / "origin", "with_sv", "u")
    push(tmp_path / "origin", DirectoryRemote(mirror))
    _unit(tmp_path / "rebuilt", "with_sv", "u", b"another, longer build")

    class Changing(DirectoryRemote):
        def download_parts(self, name: str, path: Path) -> None:
            if name.endswith("gram.f32"):
                # Between the listing and this read, another job removed the unit and pushed its own build of the key.
                shutil.rmtree(mirror / "restart" / "with_sv" / "stage0" / unit.name)
                push(tmp_path / "rebuilt", DirectoryRemote(mirror))
            super().download_parts(name, path)

    job = tmp_path / "job"
    restore(job, Changing(mirror))
    assert RestartStore(job / "restart" / "with_sv").units("stage0") == [] and not list((job / "restart").rglob("gram.f32"))
    push(job, DirectoryRemote(mirror))
    assert (mirror / "restart" / "with_sv" / "stage0" / unit.name / "ld" / "gram.f32").read_bytes() == b"another, longer build"


def test_a_region_s_restart_units_mirror_in_its_directory_and_leave_it_only_from_a_disk_that_held_them(tmp_path: Path) -> None:
    remote = DirectoryRemote(tmp_path / "mirror")
    job = tmp_path / "job"
    restore(job, remote)
    unit = _unit(job / "regions" / "apoe", "with_sv", "window")
    push(job, remote)
    mirrored = f"regions/apoe/restart/with_sv/stage0/{unit.name}/{MANIFEST}"
    assert mirrored in remote.sizes("regions/apoe/restart/")
    later = tmp_path / "later"
    restore(later, remote)
    assert RestartStore(later / "regions" / "apoe" / "restart" / "with_sv").unit("stage0", unit.name) is not None
    # A disk whose restore found nothing of the region, pushing a unit of its own there, removes nothing it never held.
    stray = tmp_path / "stray"
    restore(stray, DirectoryRemote(tmp_path / "empty"))
    _unit(stray / "regions" / "apoe", "with_sv", "other")
    push(stray, remote)
    assert mirrored in remote.sizes("regions/apoe/restart/")
    # The disks that held it, the one that pushed it and the one that restored it, remove it once their fit releases it.
    RestartStore(later / "regions" / "apoe" / "restart" / "with_sv").release()
    push(later, remote)
    assert mirrored not in remote.sizes("regions/apoe/restart/")


def test_a_changed_warm_start_is_sent_again_and_an_unchanged_one_is_not(tmp_path: Path) -> None:
    run, mirror = tmp_path / "run", tmp_path / "mirror"
    start = RestartStore(run / "restart" / "snv_indel").mutable("eb", "start", group="model0", named=facts({"inputs": "eb"}))
    _write(start / "model0.npz", b"first start")
    remote = DirectoryRemote(mirror)
    push(run, remote)
    copy = mirror / "restart" / "snv_indel" / "eb" / "start" / "model0.npz"
    sent = copy.stat().st_mtime_ns
    push(run, remote)
    assert copy.stat().st_mtime_ns == sent
    # A better start of the same size, written later (``fit_checkpoint.save_archive`` replaces the file whole).
    _write(start / "model0.npz", b"later start")
    stat = (start / "model0.npz").stat()
    os.utime(start / "model0.npz", ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    push(run, remote)
    assert copy.read_bytes() == b"later start"
    restore(tmp_path / "restored", remote)
    assert (tmp_path / "restored" / "restart" / "snv_indel" / "eb" / "start" / "model0.npz").read_bytes() == b"later start"
    # The restored disk's receipt knows the start it holds: nothing is sent again.
    sent = copy.stat().st_mtime_ns
    push(tmp_path / "restored", remote)
    assert copy.stat().st_mtime_ns == sent


def test_a_direct_sv_chromosome_is_restored_only_with_its_rows(tmp_path: Path) -> None:
    partial = tmp_path / "run" / "direct_sv.partial"
    _write(partial / KEY_FILE, "directkey")
    for chromosome in ("chr1", "chr2"):
        _write(partial / "store" / "dosage" / "half0" / chromosome / "shard0", chromosome.encode())
        _write(partial / "chromosomes" / f"{chromosome}.json", "{}")
    _write(partial / "rows" / "chr1.npz", b"rows")
    remote = DirectoryRemote(tmp_path / "mirror")
    push(tmp_path / "run", remote)
    # chr2's rows did not reach the mirror (a mirror before it carried them): chr2 is made again, never restored half.
    restore(tmp_path / "restored", remote)
    restored = tmp_path / "restored" / "direct_sv.partial"
    assert (restored / "rows" / "chr1.npz").read_bytes() == b"rows" and (restored / "chromosomes" / "chr1.json").is_file()
    assert not (restored / "chromosomes" / "chr2.json").exists() and not (restored / "store" / "dosage" / "half0" / "chr2").exists()


def test_a_step_made_again_under_its_own_key_keeps_its_mirrored_copy(tmp_path: Path) -> None:
    run, mirror = _run(tmp_path / "run"), tmp_path / "mirror"
    remote = DirectoryRemote(mirror)
    push(run, remote)
    pushed = (mirror / "samples" / COMPLETE_FILE).read_bytes()
    # A disk the restore did not reach made the step again: the same key, another finishing time.
    _write(run / "samples" / COMPLETE_FILE, '{"key": "a", "finished": "later"}')
    push(run, remote)
    assert (mirror / "samples" / COMPLETE_FILE).read_bytes() == pushed
    # A partial under the completed step's own key leaves the mirrored step; one under another key does not.
    other = tmp_path / "other"
    _write(other / "store.partial" / KEY_FILE, "b")
    push(other, remote)
    assert (mirror / "store" / COMPLETE_FILE).is_file()
    _write(other / "store.partial" / KEY_FILE, "newstore")
    push(other, remote)
    assert not (mirror / "store").exists()


def test_a_gcs_restart_file_is_sent_in_parallel_parts_within_the_part_limit(tmp_path: Path, monkeypatch) -> None:
    transfer_manager = pytest.importorskip("google.cloud.storage.transfer_manager")
    calls, whole = [], []
    monkeypatch.setattr(transfer_manager, "upload_chunks_concurrently", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(transfer_manager, "TM_DEFAULT_CHUNK_SIZE", 2)
    blob = object()
    remote = GcsRemote.__new__(GcsRemote)
    remote._bucket, remote._root = type("Bucket", (), {"blob": lambda self, name: blob})(), "svpgs/run/"
    monkeypatch.setattr(remote, "upload", lambda path, name: whole.append(name))
    tiny = _write(tmp_path / "tiny", bytes(2))
    small = _write(tmp_path / "small", bytes(16))
    large = _write(tmp_path / "large", bytes(2 * run_mirror.MULTIPART_PARTS + 1))
    for path in (tiny, small, large):
        remote.upload_parts(path, f"restart/snv_indel/stage0/key/{path.name}")
    # A file within one part is sent whole.
    assert whole == ["restart/snv_indel/stage0/key/tiny"]
    (small_args, small_options), (large_args, large_options) = calls
    assert small_args == (str(small), blob) and small_options["worker_type"] == transfer_manager.PROCESS and small_options["chunk_size"] == 2
    # Past the default chunk's 10,000 parts, each part grows so the object stays within Cloud Storage's limit.
    assert large_options["chunk_size"] == 3 and -(-large.stat().st_size // large_options["chunk_size"]) <= run_mirror.MULTIPART_PARTS


def test_a_gcs_restart_file_is_restored_in_parallel_ranges_with_its_checksum(tmp_path: Path, monkeypatch) -> None:
    transfer_manager = pytest.importorskip("google.cloud.storage.transfer_manager")
    calls, whole = [], []
    monkeypatch.setattr(transfer_manager, "download_chunks_concurrently", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(transfer_manager, "TM_DEFAULT_CHUNK_SIZE", 2)

    class Blob:
        def __init__(self, name: str) -> None:
            self.name, self.size = name, None

        def reload(self) -> None:
            self.size = 2 if self.name.endswith("tiny") else 64

    remote = GcsRemote.__new__(GcsRemote)
    remote._bucket, remote._root = type("Bucket", (), {"blob": lambda self, name: Blob(name)})(), "svpgs/run/"
    monkeypatch.setattr(remote, "download", lambda name, path: whole.append(name))
    remote.download_parts("restart/snv_indel/stage0/key/tiny", tmp_path / "tiny")
    remote.download_parts("restart/snv_indel/stage0/key/ld/ld_grams.f32", tmp_path / "ld" / "ld_grams.f32")
    # A file within one range is read whole; a larger one in ranges, one process per CPU, its CRC32C checked.
    assert whole == ["restart/snv_indel/stage0/key/tiny"]
    ((blob, filename), options), = calls
    assert blob.name == "svpgs/run/restart/snv_indel/stage0/key/ld/ld_grams.f32" and filename == str(tmp_path / "ld" / "ld_grams.f32")
    assert (tmp_path / "ld").is_dir() and options["worker_type"] == transfer_manager.PROCESS and options["crc32c_checksum"] is True
    assert options["max_workers"] == len(os.sched_getaffinity(0)) and "retry" in options["download_kwargs"]


def test_a_restore_reads_a_sealed_unit_s_files_in_parts_largest_first_and_a_mutable_unit_s_in_one_pass(tmp_path: Path) -> None:
    run, mirror = _run(tmp_path / "run"), tmp_path / "mirror"
    push(run, DirectoryRemote(mirror))
    parts, many = [], []

    class Recording(DirectoryRemote):
        def download_parts(self, name: str, path: Path) -> None:
            parts.append(name)
            super().download_parts(name, path)

        def download_many(self, pairs: list[tuple[Path, str]]) -> None:
            many.extend(name for _path, name in pairs if name.startswith("restart/"))
            super().download_many(pairs)

    restore(tmp_path / "job", Recording(mirror))
    sizes = DirectoryRemote(mirror).sizes("restart/")
    assert parts and all("/stage0/" in name for name in parts) and [sizes[name] for name in parts] == sorted((sizes[name] for name in parts), reverse=True)
    assert many and all("/eb/" in name for name in many)

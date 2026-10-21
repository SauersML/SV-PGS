from __future__ import annotations

import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from sv_pgs import run_mirror
from sv_pgs.content_identity import GCS_MOUNTS
from sv_pgs.run_mirror import Base, DirectoryRemote, GcsRemote, bases_of, mounted, push, restore
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


def _followed(root: Path) -> dict[str, bytes]:
    """``_tree``, read through the links a restore leaves to the mount."""
    return {
        (Path(directory) / name).relative_to(root).as_posix(): (Path(directory) / name).read_bytes()
        for directory, _directories, names in os.walk(root, followlinks=True) for name in names
    }


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

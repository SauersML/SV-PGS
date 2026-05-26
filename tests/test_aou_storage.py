from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import pytest

pytest.importorskip("sv_pgs.aou_storage")

from sv_pgs import aou_storage
from sv_pgs.aou_storage import (
    read_manifest,
    stage_aou_plink_trio,
    stage_gcs_object,
    verify_local_cache,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _disable_gcsfuse_detection(monkeypatch):
    monkeypatch.setattr(
        aou_storage.gcsfuse_staging, "is_gcsfuse_path", lambda _p: False
    )


def _disable_disk_check(monkeypatch):
    # Pretend we have a comfortable amount of free space everywhere.
    import shutil as _shutil

    fake_usage = _shutil._ntuple_diskusage(
        total=2 * 10**12, used=10**11, free=10**12
    )
    monkeypatch.setattr(aou_storage.shutil, "disk_usage", lambda _p: fake_usage)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_stage_local_file_to_local_creates_manifest(tmp_path, monkeypatch):
    _disable_gcsfuse_detection(monkeypatch)
    _disable_disk_check(monkeypatch)

    src = tmp_path / "src.bed"
    payload = b"\x6c\x1b\x01" + os.urandom(1024)
    src.write_bytes(payload)

    dst = tmp_path / "cache" / "out.bed"
    entry = stage_gcs_object(
        str(src),
        dst,
        content_kind="test_bed",
        min_free_bytes_after=0,
    )
    assert entry.local_path == dst
    assert entry.complete is True
    assert dst.exists()
    assert dst.read_bytes() == payload

    manifest_path = dst.with_name(dst.name + ".manifest.json")
    assert manifest_path.exists()
    assert manifest_path == entry.manifest_path


def test_stage_emits_complete_manifest_on_success(tmp_path, monkeypatch):
    _disable_gcsfuse_detection(monkeypatch)
    _disable_disk_check(monkeypatch)

    src = tmp_path / "src.bin"
    src.write_bytes(b"abcdefg" * 100)
    dst = tmp_path / "cache" / "out.bin"

    stage_gcs_object(
        str(src), dst, content_kind="generic", min_free_bytes_after=0
    )

    manifest = read_manifest(dst)
    assert manifest is not None
    assert manifest["complete"] is True
    assert manifest["content_kind"] == "generic"
    assert manifest["source_uri"] == str(src)
    assert manifest["local_size_bytes"] == src.stat().st_size
    assert manifest["source_size_bytes"] == src.stat().st_size
    assert manifest["schema_version"] == 1
    assert verify_local_cache(dst) is True


def test_verify_local_cache_returns_false_when_manifest_missing(tmp_path):
    f = tmp_path / "lonely.bed"
    f.write_bytes(b"\x00\x01\x02")
    # No manifest written.
    assert verify_local_cache(f) is False


def test_verify_local_cache_returns_false_when_size_mismatch(tmp_path, monkeypatch):
    _disable_gcsfuse_detection(monkeypatch)
    _disable_disk_check(monkeypatch)

    src = tmp_path / "src.bin"
    src.write_bytes(b"x" * 256)
    dst = tmp_path / "cache" / "out.bin"
    stage_gcs_object(str(src), dst, content_kind="x", min_free_bytes_after=0)
    assert verify_local_cache(dst) is True

    # Corrupt the data file (truncate it) — manifest still claims the old size.
    with open(dst, "r+b") as fh:
        fh.truncate(10)
    assert verify_local_cache(dst) is False


# --- multiprocess race test ------------------------------------------------


def _race_worker(src_path: str, dst_path: str, ready_barrier_path: str):
    """Worker that waits for a sentinel then races to stage."""
    # Crude barrier: poll for the sentinel file.
    while not Path(ready_barrier_path).exists():
        time.sleep(0.01)
    # Re-import inside the child process.
    from sv_pgs import aou_storage as _aou

    # Stub the disk check + gcsfuse detection inside the child too.
    import shutil as _shutil

    fake_usage = _shutil._ntuple_diskusage(
        total=2 * 10**12, used=10**11, free=10**12
    )
    _aou.shutil.disk_usage = lambda _p: fake_usage  # type: ignore[assignment]
    _aou.gcsfuse_staging.is_gcsfuse_path = lambda _p: False  # type: ignore[assignment]

    _aou.stage_gcs_object(
        src_path,
        Path(dst_path),
        content_kind="race",
        min_free_bytes_after=0,
    )


def test_partial_file_collision_resolved_by_lock(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(os.urandom(64 * 1024))
    dst = tmp_path / "cache" / "out.bin"
    dst.parent.mkdir(parents=True, exist_ok=True)
    sentinel = tmp_path / "GO"

    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(
            target=_race_worker, args=(str(src), str(dst), str(sentinel))
        )
        for _ in range(2)
    ]
    for p in procs:
        p.start()
    # Let both processes reach the barrier-poll loop.
    time.sleep(0.5)
    sentinel.write_bytes(b"go")

    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0, f"worker exit={p.exitcode}"

    # Both observe a single final file with a single manifest.
    assert dst.exists()
    assert dst.read_bytes() == src.read_bytes()
    manifest_path = dst.with_name(dst.name + ".manifest.json")
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["complete"] is True
    assert manifest["local_size_bytes"] == src.stat().st_size

    # No partial files should be left behind.
    leftovers = sorted(
        p.name for p in dst.parent.iterdir() if ".partial." in p.name
    )
    assert leftovers == [], f"leftover partials: {leftovers}"


def test_stage_interrupted_leaves_no_manifest(tmp_path, monkeypatch):
    _disable_gcsfuse_detection(monkeypatch)
    _disable_disk_check(monkeypatch)

    src = tmp_path / "src.bin"
    src.write_bytes(b"y" * 4096)
    dst = tmp_path / "cache" / "out.bin"

    # Inject a failure mid-copy by patching the buffered-copy helper.
    def boom(*_a, **_kw):
        raise RuntimeError("simulated mid-copy failure")

    monkeypatch.setattr(aou_storage, "_buffered_copy_to_partial", boom)

    with pytest.raises(RuntimeError, match="simulated mid-copy failure"):
        stage_gcs_object(
            str(src), dst, content_kind="x", min_free_bytes_after=0
        )

    # No final data file, no manifest.
    assert not dst.exists()
    manifest_path = dst.with_name(dst.name + ".manifest.json")
    assert not manifest_path.exists()

    # No leftover partials either.
    if dst.parent.exists():
        leftovers = [p.name for p in dst.parent.iterdir() if ".partial." in p.name]
        assert leftovers == []


def test_stage_aou_plink_trio_atomic_per_file(tmp_path, monkeypatch):
    _disable_gcsfuse_detection(monkeypatch)
    _disable_disk_check(monkeypatch)

    # Build three small synthetic remote-prefix files.
    remote_dir = tmp_path / "remote"
    remote_dir.mkdir()
    remote_prefix_path = remote_dir / "v8"
    payloads: dict[str, bytes] = {
        ".bed": b"\x6c\x1b\x01" + b"\x00" * 200,
        ".bim": b"chr1\trs1\t0\t1\tA\tG\n" * 50,
        ".fam": b"FID IID 0 0 0 -9\n" * 50,
    }
    for ext, data in payloads.items():
        (remote_dir / f"v8{ext}").write_bytes(data)

    cache_dir = tmp_path / "cache"
    bed_entry, bim_entry, fam_entry = stage_aou_plink_trio(
        str(remote_prefix_path), cache_dir, prefix="arrays"
    )

    for entry, ext in (
        (bed_entry, ".bed"),
        (bim_entry, ".bim"),
        (fam_entry, ".fam"),
    ):
        assert entry.complete is True
        assert entry.local_path.exists()
        assert entry.local_path.read_bytes() == payloads[ext]
        manifest_path = entry.local_path.with_name(
            entry.local_path.name + ".manifest.json"
        )
        assert manifest_path.exists()
        manifest = read_manifest(entry.local_path)
        assert manifest is not None
        assert manifest["complete"] is True
        assert verify_local_cache(entry.local_path) is True

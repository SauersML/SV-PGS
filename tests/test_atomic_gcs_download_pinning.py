"""Pin atomic semantics of ``_download_gcs_object_if_missing``.

The downloader stages into a unique ``.partial.<pid>.<uuid>`` filename then
atomically renames to ``local_path``. Two failure modes are easy to break:

* On a mid-write failure (``_gsutil_cp`` raises), the partial file must be
  removed AND ``local_path`` must NOT exist — callers rely on
  ``local_path.exists()`` as the cache-hit oracle.
* If a concurrent fetcher published the final file while we were downloading,
  we must discard our staging file rather than overwrite the published one.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from sv_pgs import aou_runner


def test_failed_download_cleans_partial_and_leaves_no_local_path(
    tmp_path: Path, monkeypatch
):
    """``_gsutil_cp`` raises → no ``.partial.*`` files survive AND
    ``local_path`` was never created."""

    def _failing_cp(src: str, dst: str) -> None:
        # Simulate gsutil exiting non-zero AFTER writing a partial file.
        Path(dst).write_bytes(b"partial garbage")
        raise RuntimeError("gsutil cp failed (exit 1)")

    monkeypatch.setattr(aou_runner, "_gsutil_cp", _failing_cp)
    local_path = tmp_path / "subdir" / "object.bin"
    with pytest.raises(RuntimeError, match="gsutil cp failed"):
        aou_runner._download_gcs_object_if_missing("gs://bucket/object.bin", local_path)

    assert not local_path.exists(), (
        "local_path must NOT exist on failed download — callers "
        "use local_path.exists() as the cache-hit oracle."
    )
    partial_files = list(local_path.parent.glob(f"{local_path.name}.partial.*"))
    assert partial_files == [], (
        f"partial files must be cleaned on failure; survivors: {partial_files}"
    )


def test_short_circuit_when_local_path_already_exists(tmp_path: Path, monkeypatch):
    """If ``local_path`` already exists, gsutil must NOT be invoked."""
    local_path = tmp_path / "cached.bin"
    local_path.write_bytes(b"cached content")

    invocations: list[tuple[str, str]] = []

    def _spy_cp(src: str, dst: str) -> None:
        invocations.append((src, dst))

    monkeypatch.setattr(aou_runner, "_gsutil_cp", _spy_cp)
    aou_runner._download_gcs_object_if_missing("gs://bucket/cached.bin", local_path)
    assert invocations == [], "gsutil must not run when local_path already exists"
    assert local_path.read_bytes() == b"cached content"


def test_successful_download_publishes_and_cleans_partial(tmp_path: Path, monkeypatch):
    """Happy path: partial file is renamed in place; no ``.partial.*`` leftovers."""

    def _writing_cp(src: str, dst: str) -> None:
        Path(dst).write_bytes(b"hello")

    monkeypatch.setattr(aou_runner, "_gsutil_cp", _writing_cp)
    local_path = tmp_path / "out.bin"
    aou_runner._download_gcs_object_if_missing("gs://bucket/out.bin", local_path)
    assert local_path.exists()
    assert local_path.read_bytes() == b"hello"
    assert list(local_path.parent.glob(f"{local_path.name}.partial.*")) == []


def test_concurrent_publish_discards_local_staging(tmp_path: Path, monkeypatch):
    """If a concurrent racer publishes ``local_path`` while ``_gsutil_cp`` is
    running, the loser must discard its partial — NOT overwrite the winner."""
    local_path = tmp_path / "shared.bin"

    def _race_cp(src: str, dst: str) -> None:
        # Write our staging contents...
        Path(dst).write_bytes(b"loser")
        # ...while a concurrent racer (simulated here) publishes the canonical
        # file in the meantime.
        local_path.write_bytes(b"winner")

    monkeypatch.setattr(aou_runner, "_gsutil_cp", _race_cp)
    aou_runner._download_gcs_object_if_missing("gs://bucket/shared.bin", local_path)
    # The winner's bytes must remain — our partial must be cleaned.
    assert local_path.read_bytes() == b"winner"
    assert list(local_path.parent.glob(f"{local_path.name}.partial.*")) == []

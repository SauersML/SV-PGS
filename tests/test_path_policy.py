from __future__ import annotations

import pytest

pytest.importorskip("sv_pgs.path_policy")

from sv_pgs import path_policy
from sv_pgs.path_policy import (
    StorageClass,
    assert_hot_local_path,
    classify_path,
    is_local_hot,
)


def test_local_path_is_local_hot(tmp_path, monkeypatch):
    # Force is_gcsfuse_path to False so a tmpdir on any platform is classified
    # purely on its real properties.
    monkeypatch.setattr(path_policy, "is_gcsfuse_path", lambda _p: False)
    f = tmp_path / "data.bin"
    f.write_bytes(b"hello")
    assert classify_path(f) is StorageClass.LOCAL_HOT
    assert is_local_hot(f) is True


def test_gcsfuse_mounted_path_classified_correctly(tmp_path, monkeypatch):
    f = tmp_path / "fake_gcsfuse_file"
    f.write_bytes(b"x")
    monkeypatch.setattr(path_policy, "is_gcsfuse_path", lambda _p: True)
    assert classify_path(f) is StorageClass.GCSFUSE_MOUNT
    assert is_local_hot(f) is False


def test_gs_uri_classified(monkeypatch):
    # Even if is_gcsfuse_path is called, the gs:// short-circuit fires first.
    monkeypatch.setattr(path_policy, "is_gcsfuse_path", lambda _p: False)
    assert classify_path("gs://my-bucket/path/object.bed") is StorageClass.GS_URI
    assert is_local_hot("gs://my-bucket/path/object.bed") is False


def test_nonexistent_path_classified(tmp_path, monkeypatch):
    monkeypatch.setattr(path_policy, "is_gcsfuse_path", lambda _p: False)
    missing = tmp_path / "no_such_dir" / "no_such_file.bed"
    cls = classify_path(missing)
    assert cls is StorageClass.NONEXISTENT
    assert is_local_hot(missing) is False


def test_assert_hot_local_path_raises_on_gcsfuse_with_purpose_in_message(
    tmp_path, monkeypatch
):
    f = tmp_path / "pretend_gcsfuse.bed"
    f.write_bytes(b"x")
    monkeypatch.setattr(path_policy, "is_gcsfuse_path", lambda _p: True)
    with pytest.raises(RuntimeError) as excinfo:
        assert_hot_local_path(f, purpose="hot_genotype_read")
    msg = str(excinfo.value)
    assert "hot_genotype_read" in msg
    assert "gcsfuse_mount" in msg


def test_assert_hot_local_path_succeeds_on_local(tmp_path, monkeypatch):
    monkeypatch.setattr(path_policy, "is_gcsfuse_path", lambda _p: False)
    f = tmp_path / "ok.bed"
    f.write_bytes(b"x")
    # Should NOT raise.
    assert_hot_local_path(f, purpose="hot_genotype_read")

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sv_pgs import content_identity
from sv_pgs.content_identity import GCS_MOUNTS, file_digest


class _Bucket:
    def __init__(self, objects: dict[tuple[str, str], SimpleNamespace], bucket: str, user_project: str | None, asked: list) -> None:
        self.objects, self.name, self.user_project, self.asked = objects, bucket, user_project, asked

    def get_blob(self, name: str):
        self.asked.append((self.name, name, self.user_project))
        return self.objects.get((self.name, name))


@pytest.fixture
def bucket_objects(monkeypatch):
    objects: dict[tuple[str, str], SimpleNamespace] = {}
    asked: list = []
    client = SimpleNamespace(bucket=lambda bucket, user_project=None: _Bucket(objects, bucket, user_project, asked))
    monkeypatch.setattr(content_identity, "_storage_client", lambda: client)
    monkeypatch.setenv("GOOGLE_PROJECT", "workspace-project")
    monkeypatch.setenv(GCS_MOUNTS, json.dumps({"/mnt/disks/cdr": "gs://cdr-bucket/v8", "/mnt/disks/ops": "gs://ops-bucket"}))
    return objects, asked


def test_a_file_on_a_mounted_bucket_is_its_object_generation_without_reading_a_byte(bucket_objects) -> None:
    objects, asked = bucket_objects
    objects[("cdr-bucket", "v8/wgs/sv/chr1.vcf.gz")] = SimpleNamespace(generation=17, size=1 << 40, crc32c="AAAAAA==")
    # The path is not even readable here: the identity comes from the object's metadata, billed to the workspace.
    identity = file_digest("/mnt/disks/cdr/wgs/sv/chr1.vcf.gz")
    assert identity == "gs://cdr-bucket/v8/wgs/sv/chr1.vcf.gz#17:1099511627776:AAAAAA=="
    assert asked == [("cdr-bucket", "v8/wgs/sv/chr1.vcf.gz", "workspace-project")]
    # A rewrite gets a new generation, so a new identity.
    objects[("cdr-bucket", "v8/wgs/sv/chr1.vcf.gz")] = SimpleNamespace(generation=18, size=1 << 40, crc32c="AAAAAA==")
    assert file_digest("/mnt/disks/cdr/wgs/sv/chr1.vcf.gz") != identity
    objects[("ops-bucket", "imputed/chr2.bcf")] = SimpleNamespace(generation=3, size=9, crc32c="BBBBBB==")
    assert file_digest("/mnt/disks/ops/imputed/chr2.bcf") == "gs://ops-bucket/imputed/chr2.bcf#3:9:BBBBBB=="


def test_a_missing_object_is_refused(bucket_objects) -> None:
    with pytest.raises(FileNotFoundError, match="gs://cdr-bucket/v8/absent.vcf.gz"):
        file_digest("/mnt/disks/cdr/absent.vcf.gz")


def test_a_file_off_the_mounts_is_hashed_byte_by_byte(bucket_objects, tmp_path: Path) -> None:
    objects, asked = bucket_objects
    path = tmp_path / "local.bin"
    path.write_bytes(b"local bytes")
    assert file_digest(path) == hashlib.sha256(b"local bytes").hexdigest() and asked == []


def _serial_tree_digest(root: Path) -> str:
    import struct

    expected = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        name = path.relative_to(root).as_posix().encode()
        expected.update(struct.pack("<Q", len(name)) + name + hashlib.sha256(path.read_bytes()).digest())
    return expected.hexdigest()


def test_a_restored_file_is_known_by_its_restored_digest_until_it_changes(tmp_path: Path, monkeypatch) -> None:
    """The restore hashed each byte as it wrote it, so a digest reads none of them again; a file written since is read."""
    store = tmp_path / "store"
    for name, data in (("MANIFEST.json", b"{}"), ("dosage/half0/chr1/codes", b"abcd" * 1000), ("variants/chr1/position", b"1")):
        (store / name).parent.mkdir(parents=True, exist_ok=True)
        (store / name).write_bytes(data)
    expected = _serial_tree_digest(store)
    monkeypatch.setenv(content_identity.RESTORED_DIGESTS, str(tmp_path / "restored-digests"))
    content_identity.record_restored((path, hashlib.sha256(path.read_bytes()).hexdigest()) for path in store.rglob("*") if path.is_file())
    hashed = []
    original = content_identity._hashed

    def counted(path: str, stamp):
        hashed.append(Path(path).relative_to(store).as_posix())
        return original(path, stamp)

    monkeypatch.setattr(content_identity, "_hashed", counted)
    assert content_identity.tree_digest(store) == expected and hashed == []
    assert file_digest(store / "MANIFEST.json") == hashlib.sha256(b"{}").hexdigest() and hashed == []
    # A rewrite moves the file's stamp (a new inode here, its change time in place), so its restored digest no longer applies.
    (store / "variants" / "chr1" / "position").write_bytes(b"2")
    assert content_identity.tree_digest(store) == _serial_tree_digest(store) != expected
    assert hashed == ["variants/chr1/position"]


def test_restored_digests_are_kept_only_where_the_launcher_names_their_directory_and_a_disputed_one_is_ignored(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "codes"
    path.write_bytes(b"codes")
    monkeypatch.delenv(content_identity.RESTORED_DIGESTS, raising=False)
    content_identity.record_restored([(path, "0" * 64)])
    assert file_digest(path) == hashlib.sha256(b"codes").hexdigest() and not list(tmp_path.glob("*.json"))
    monkeypatch.setenv(content_identity.RESTORED_DIGESTS, str(tmp_path / "restored-digests"))
    content_identity.record_restored([(path, hashlib.sha256(b"codes").hexdigest())])
    content_identity.record_restored([(path, "1" * 64)])
    # Two records that disagree on one file certify neither: the file is read.
    assert file_digest(path) == hashlib.sha256(b"codes").hexdigest()
    assert len(list((tmp_path / "restored-digests").glob("*.json"))) == 2 and not list((tmp_path / "restored-digests").glob(".*"))

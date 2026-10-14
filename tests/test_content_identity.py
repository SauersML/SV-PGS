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

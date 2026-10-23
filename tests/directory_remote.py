"""A run mirror in a local directory for the tests, in place of GCS (``run_mirror.GcsRemote``)."""
from __future__ import annotations

import os
import shutil
from collections.abc import Iterable
from pathlib import Path

import google_crc32c

from sv_pgs.run_mirror import Version, _HashingFile, _RangeWriter, _transfer


class DirectoryRemote:
    """A mirror in a local directory, with GCS's whole-or-absent objects (each file written aside, then renamed)."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    def sizes(self, prefix: str) -> dict[str, int]:
        if not self._root.is_dir():
            return {}
        found = {path.relative_to(self._root).as_posix(): path.stat().st_size for path in self._root.rglob("*") if path.is_file()}
        return {name: size for name, size in found.items() if name.startswith(prefix)}

    def read(self, name: str) -> bytes | None:
        path = self._root / name
        return path.read_bytes() if path.is_file() else None

    def upload(self, path: Path, name: str) -> None:
        target = self._root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        aside = target.with_name(f".{target.name}.upload")
        shutil.copyfile(path, aside)
        os.replace(aside, target)

    def upload_parts(self, path: Path, name: str) -> None:
        self.upload(path, name)

    def write(self, name: str, content: bytes) -> None:
        target = self._root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        aside = target.with_name(f".{target.name}.upload")
        aside.write_bytes(content)
        os.replace(aside, target)

    def download(self, name: str, path: Path) -> str | None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._root / name, "rb") as source, open(path, "wb") as handle:
            written = _HashingFile(handle)
            shutil.copyfileobj(source, written)
        return written.hexdigest()

    def download_parts(self, name: str, path: Path) -> None:
        self.download(name, path)

    def download_many(self, pairs: list[tuple[Path, str]]) -> list[str | None]:
        return _transfer(pairs, lambda path, name: self.download(name, path))

    def version(self, name: str) -> Version:
        checksum = google_crc32c.Checksum()
        with open(self._root / name, "rb") as source:
            for chunk in iter(lambda: source.read(shutil.COPY_BUFSIZE), b""):
                checksum.update(chunk)
        return Version((self._root / name).stat().st_size, None, checksum.digest())

    def download_range(self, name: str, path: Path, start: int, stop: int, generation: int | None) -> None:
        with open(self._root / name, "rb") as source, open(path, "r+b") as handle:
            source.seek(start)
            written = _RangeWriter(handle, start, stop)
            written.write(source.read(stop - start))
            written.close()

    def delete(self, names: Iterable[str]) -> None:
        for name in names:
            path = self._root / name
            path.unlink(missing_ok=True)
            # A GCS prefix with no object left is gone; so is a directory here.
            while path.parent != self._root and not any(path.parent.iterdir()):
                path = path.parent
                path.rmdir()

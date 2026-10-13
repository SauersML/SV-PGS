"""Complete local-file identities; mutable file metadata is not a content certificate."""

from __future__ import annotations

import hashlib
from pathlib import Path


def _stamp(path: Path) -> tuple[int, ...]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _hashed(path: str, stamp: tuple[int, ...]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    if _stamp(Path(path)) != stamp:
        raise RuntimeError(f"{path} changed while its content was hashed")
    return digest.hexdigest()


def file_digest(path: str | Path) -> str:
    """Hash every byte on each call. Size, mtime and ctime alone cannot certify contents.

    Immutable object generations may replace this pass when explicitly supplied by the
    storage provider; mutable local files cannot safely use that optimization.
    """
    path = Path(path).resolve()
    return _hashed(str(path), _stamp(path))


def tree_digest(root: str | Path) -> str:
    """Bind paths and complete contents, including samples, annotations and genotype shards."""
    root = Path(root)
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "CACHE_COMPLETE.json"):
        name = path.relative_to(root).as_posix().encode()
        digest.update(len(name).to_bytes(8, "little"))
        digest.update(name)
        digest.update(bytes.fromhex(file_digest(path)))
    return digest.hexdigest()

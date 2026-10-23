"""A fit job's restore, old against new, and its saved mode components read one at a time against several at once, on
synthetic data on the fit jobs' machine (a2-highgpu-2g, 24 vCPUs, four local SSDs):

    python -m benchmarks.restore_bench make <gs://mirror> <sizes file>
    python -m benchmarks.restore_bench restore <gs://mirror> <run directory>
    python -m benchmarks.restore_bench seeds [--members 1061205] [--nodes 64] [--components 8]
    python -m benchmarks.restore_bench remove <gs://mirror>

- make: a run mirror of synthetic objects at ``<gs://mirror>``, one per line of the sizes file ("<bytes> <name>", the
  names and sizes of a base mirror's objects: a listing, no content), each one random block repeated, a large one sent
  in parts and composed (a CRC32C and no MD5, as the mirror's restart files are). A ``.json`` object holds a JSON value
  instead (a step's marker, its directories' list), as a restore reads those.
- restore: ``run_mirror.restore`` of that mirror into ``<run directory>``, first with every file read whole (the old
  route: ``_piece_bytes`` past every file) and then with its large files in pieces (the new), each into an empty
  directory, with the CPU's busy share and the bytes the run disk wrote every 10 s. Each file's SHA-256 the two
  restores recorded (``content_identity.RESTORED_DIGESTS``) must agree: FAIL where one differs.
- seeds: ``warm_seeds.component_means`` of a synthetic product-mixture law of the AoU arms' size (1,061,205 members
  [real, aou-m3f-k2]) within this machine's budget, one component at a time and then several at once as the fit step
  reads them (half the host's usable bytes, every CPU): FAIL where a bit differs.
- remove: the mirror's objects deleted.

RESULT lines give each restore's seconds and the seeds' seconds both ways. The objects are random bytes: no participant
data.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

_BLOCK = 1 << 26
"""Bytes of the random block every synthetic object repeats: past gzip's 32 KiB window, so no transfer compresses it."""
_PARTS = 32
"""Cloud Storage's compose: at most 32 source objects into one (``run_mirror.COMPOSE_SOURCES``)."""


class _Synthetic(io.RawIOBase):
    """``size`` bytes of ``block`` repeated, as a seekable stream an upload reads (and rereads on a retry)."""

    def __init__(self, block: bytes, size: int) -> None:
        self._block, self._size, self._position = block, size, 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        base = {os.SEEK_SET: 0, os.SEEK_CUR: self._position, os.SEEK_END: self._size}[whence]
        self._position = max(0, min(self._size, base + offset))
        return self._position

    def read(self, count: int | None = -1) -> bytes:
        count = self._size - self._position if count is None or count < 0 else min(count, self._size - self._position)
        pieces = []
        while count > 0:
            offset = self._position % len(self._block)
            take = min(count, len(self._block) - offset)
            pieces.append(self._block[offset:offset + take])
            self._position += take
            count -= take
        return b"".join(pieces)


_BLOCKS: dict[int, bytes] = {}


def _bucket(url: str):
    from google.cloud import storage

    bucket, _, prefix = url.removeprefix("gs://").partition("/")
    project = os.environ.get("GOOGLE_PROJECT")
    return storage.Client(project=project).bucket(bucket, user_project=project), prefix.strip("/") + "/"


def _upload(url: str, name: str, size: int, seed: int) -> None:
    """One object (or a part to compose) of ``size`` synthetic bytes, or a JSON object's value."""
    from google.cloud.storage.retry import DEFAULT_RETRY

    bucket, root = _bucket(url)
    content = b"[]" if name.endswith(".directories.json") else json.dumps({"key": name}).encode() if name.endswith(".json") else None
    if content is not None:
        bucket.blob(root + name).upload_from_string(content)
        return
    if seed not in _BLOCKS:
        _BLOCKS[seed] = np.random.default_rng(seed).integers(0, 256, _BLOCK, dtype=np.uint8).tobytes()
    bucket.blob(root + name).upload_from_file(_Synthetic(_BLOCKS[seed], size), size=size, retry=DEFAULT_RETRY)


def make(url: str, sizes: Path) -> None:
    rows = [line.split(None, 1) for line in sizes.read_text().splitlines() if line.strip()]
    objects = sorted(((int(size), name.strip()) for size, name in rows), reverse=True)
    started = time.perf_counter()
    # An object past _BLOCK goes up in _PARTS parts at once, composed into it; every part and small object in one pool.
    uploads, composed = [], []
    for seed, (size, name) in enumerate(objects):
        if size <= _BLOCK:
            uploads.append((name, size, seed))
            continue
        edges = [size * index // _PARTS for index in range(_PARTS + 1)]
        parts = [f".composing/{name}.part{index}" for index in range(_PARTS)]
        uploads.extend((part, edges[index + 1] - edges[index], seed) for index, part in enumerate(parts))
        composed.append((name, parts))
    with ProcessPoolExecutor(max_workers=2 * len(os.sched_getaffinity(0))) as pool:
        list(pool.map(_upload, *zip(*[(url, name, size, seed) for name, size, seed in uploads]), chunksize=4))
    bucket, root = _bucket(url)
    for name, parts in composed:
        sources = [bucket.blob(root + part) for part in parts]
        bucket.blob(root + name).compose(sources)
        bucket.delete_blobs(sources)
    print(f"made {len(objects)} objects, {sum(size for size, _name in objects) / 1e9:.1f} GB, in {time.perf_counter() - started:.0f} s", flush=True)


def _cpu() -> tuple[int, int]:
    fields = [int(value) for value in open("/proc/stat").readline().split()[1:]]
    return sum(fields) - fields[3] - fields[4], sum(fields)


def _written(device: str) -> int:
    for line in open("/proc/diskstats"):
        parts = line.split()
        if parts[2] == device:
            return int(parts[9]) * 512
    return 0


def _device_of(path: Path) -> str:
    """The block device (as /proc/diskstats names it) under ``path``'s mount."""
    target = os.stat(path).st_dev
    for line in open("/proc/self/mountinfo"):
        parts = line.split()
        major, minor = (int(value) for value in parts[2].split(":"))
        if os.makedev(major, minor) == target:
            return os.path.basename(parts[parts.index("-") + 2])
    return ""


def _timed_restore(label: str, url: str, run: Path, digests: Path) -> dict:
    from sv_pgs import content_identity, run_mirror

    if run.exists():
        shutil.rmtree(run)
    run.mkdir(parents=True)
    os.environ[content_identity.RESTORED_DIGESTS] = str(digests)
    device = _device_of(run)
    profile: list[str] = []
    stop = threading.Event()

    def sample() -> None:
        busy, total = _cpu()
        written = _written(device)
        while not stop.wait(10):
            now_busy, now_total = _cpu()
            now_written = _written(device)
            profile.append(f"{100 * (now_busy - busy) / max(now_total - total, 1):.0f}%/{(now_written - written) / 10e6:.0f}")
            busy, total, written = now_busy, now_total, now_written

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    started = time.perf_counter()
    run_mirror.restore(run, run_mirror.GcsRemote(url))
    seconds = time.perf_counter() - started
    stop.set()
    sampler.join()
    print(f"restore {label}: {seconds:.0f} s; every 10 s, CPU busy / MB/s written to {device}: {' '.join(profile)}", flush=True)
    files = {}
    for record in sorted(digests.glob("[!.]*.json")):
        for *_stamp, digest in json.loads(record.read_text())["files"]:
            files.setdefault(digest, 0)
            files[digest] += 1
    return {"seconds": seconds, "digests": files}


def restore(url: str, run: Path) -> int:
    from sv_pgs import run_mirror

    piece = run_mirror._piece_bytes
    run_mirror._piece_bytes = lambda total, workers: 1 << 62
    old = _timed_restore("old (each file on one stream)", url, run, run.parent / "digests-old")
    run_mirror._piece_bytes = piece
    new = _timed_restore("new (large files in pieces)", url, run, run.parent / "digests-new")
    same = old["digests"] == new["digests"]
    print(f"{'PASS' if same else 'FAIL'} the two restores recorded the same SHA-256 for every file ({sum(new['digests'].values())} files)", flush=True)
    print(f"RESULT restore old {old['seconds']:.0f} s new {new['seconds']:.0f} s saved {old['seconds'] - new['seconds']:.0f} s", flush=True)
    return 0 if same else 1


def seeds(members: int, nodes: int, components: int) -> int:
    from sv_pgs.compute_budget import detect_compute_budget
    from sv_pgs.draw_laws import ProductMixtureDraws
    from sv_pgs.warm_seeds import component_means

    budget = detect_compute_budget()
    generator = np.random.default_rng(3)
    log_density = generator.standard_normal((4, nodes))
    law = ProductMixtureDraws(
        class_index=generator.integers(0, 4, members), log_scale=generator.normal(-11.0, 1.5, members),
        log_density=log_density - np.log(np.exp(log_density).sum(axis=1, keepdims=True)), log_variance_grid=np.linspace(-8.0, 6.0, nodes),
        shift=generator.standard_normal((components, members)) * 3.0, omega=generator.uniform(1e3, 3e4, (components, members)),
        weights=np.full(components, 1.0 / components), key=(1, 2), draw_count=64,
    )
    started = time.perf_counter()
    alone = component_means(law, budget.working_bytes, components)
    serial = time.perf_counter() - started
    started = time.perf_counter()
    together = component_means(law, budget.working_bytes, components, budget.host_bytes // 2, budget.cpu_threads)
    parallel = time.perf_counter() - started
    same = alone.tobytes() == together.tobytes()
    print(f"{'PASS' if same else 'FAIL'} {components} components read at once are the bits read one at a time ({budget.describe()})", flush=True)
    print(f"RESULT seeds {members} members x {nodes} nodes x {components} components: one at a time {serial:.1f} s, at once {parallel:.1f} s", flush=True)
    return 0 if same else 1


def remove(url: str) -> None:
    bucket, root = _bucket(url)
    blobs = list(bucket.client.list_blobs(bucket, prefix=root))
    for start in range(0, len(blobs), 100):
        bucket.delete_blobs(blobs[start:start + 100])
    print(f"removed {len(blobs)} objects under {url}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    made = commands.add_parser("make")
    made.add_argument("url")
    made.add_argument("sizes", type=Path)
    restored = commands.add_parser("restore")
    restored.add_argument("url")
    restored.add_argument("run", type=Path)
    seeded = commands.add_parser("seeds")
    seeded.add_argument("--members", type=int, default=1_061_205)
    seeded.add_argument("--nodes", type=int, default=64)
    seeded.add_argument("--components", type=int, default=8)
    removed = commands.add_parser("remove")
    removed.add_argument("url")
    arguments = parser.parse_args()
    if arguments.command == "make":
        make(arguments.url, arguments.sizes)
        return 0
    if arguments.command == "restore":
        return restore(arguments.url, arguments.run)
    if arguments.command == "seeds":
        return seeds(arguments.members, arguments.nodes, arguments.components)
    remove(arguments.url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
